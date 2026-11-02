#!/usr/bin/env python3
"""
ソルバー別の破綻モード分析

Phase 3: 各ソルバーの破綻条件を体系的に検証

破綻モード:
1. パターン誤認: NONをONOと判定（またはその逆）
2. 膜厚収縮: 隣接層が融合
3. 膜厚発散: 非物理的な値（負や>100Å）
4. Bimodal検出失敗: 2つのNピークを1つと判定
5. 非収束: 最適化が失敗
"""

import pytest
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# =============================================================================
# 物性パラメータ（test_non_comprehensive.pyから流用）
# =============================================================================

def imfp_tpp2m(KE: float, Nv: int, density: float, Aw: float, Eg: float) -> float:
    """TPP-2M formula for IMFP calculation"""
    if KE <= 50:
        return 5.0
    gamma = 0.191 * density ** (-0.50)
    Ep = 28.8 * (density * Nv / Aw) ** 0.5
    beta = -0.10 + 0.944 / (Ep**2 + Eg**2) ** 0.5 + 0.069 * density ** 0.1
    U = Ep**2 / 829.4
    C = 1.97 - 0.91 * U
    D = 53.4 - 20.8 * U
    return KE / (Ep**2 * (beta * np.log(gamma * KE) - C / KE + D / KE**2))


MATERIALS = {
    'SiO2':  (16, 2.2, 60.0843, 8.9),
    # Eg 5.6 eV: S. Toyoda et al., Appl. Phys. Lett. 87, 102901 (2005), Table I,
    # measured on CVD SiN/Si films (5.6-5.7 eV). CompoundTable.csv (aSi3N4) and
    # toyomacro carry 5.3 eV (Robertson, calculated). Different provenance, not a
    # discrepancy to "fix"; effect on lambda is ~0.65%. Mw 140.28 was always right here.
    'Si3N4': (32, 3.2, 140.28346, 5.6),
    'Si':    (4, 2.33, 28.09, 1.12),
}

XRAY_SOURCES = {
    'Al-Kα': 1486.6,
    'Cr-Kα': 5414.7,
}

BE_Si2p = 103
BE_O1s = 532
BE_N1s = 398


def get_imfp_mixed(KE: float, x_O: float) -> float:
    """Get IMFP for SiO2(x)/Si3N4(1-x) mixture"""
    imfp_sio2 = imfp_tpp2m(KE, *MATERIALS['SiO2'])
    imfp_si3n4 = imfp_tpp2m(KE, *MATERIALS['Si3N4'])
    return imfp_sio2 * x_O + imfp_si3n4 * (1 - x_O)


# =============================================================================
# 破綻モード定義
# =============================================================================

@dataclass
class FailureMode:
    """破綻モードの検出結果"""
    pattern_misidentification: bool = False  # パターン誤認
    thickness_collapse: bool = False          # 膜厚収縮
    thickness_divergence: bool = False        # 膜厚発散
    bimodal_failure: bool = False             # Bimodal検出失敗
    non_convergence: bool = False             # 非収束

    @property
    def any_failure(self) -> bool:
        return any([
            self.pattern_misidentification,
            self.thickness_collapse,
            self.thickness_divergence,
            self.bimodal_failure,
            self.non_convergence,
        ])

    def to_dict(self) -> Dict:
        return {
            'pattern_misidentification': self.pattern_misidentification,
            'thickness_collapse': self.thickness_collapse,
            'thickness_divergence': self.thickness_divergence,
            'bimodal_failure': self.bimodal_failure,
            'non_convergence': self.non_convergence,
            'any_failure': self.any_failure,
        }


# =============================================================================
# シミュレーション関数
# =============================================================================

def simulate_layer_model(params: np.ndarray, theta: np.ndarray, hv: float) -> np.ndarray:
    """
    3層モデルのARXPSシミュレーション

    params: [x1, x2, x3, d1, d2, d3] where x=O/(O+N), d=thickness
    """
    x1, x2, x3 = params[0], params[1], params[2]
    d1, d2, d3 = params[3], params[4], params[5]

    KE_Si = hv - BE_Si2p
    KE_O = hv - BE_O1s
    KE_N = hv - BE_N1s

    if KE_O <= 0 or KE_N <= 0:
        return None

    # IMFP for each layer
    lambda1_Si = get_imfp_mixed(KE_Si, x1)
    lambda1_O = get_imfp_mixed(KE_O, x1)
    lambda1_N = get_imfp_mixed(KE_N, x1)

    lambda2_Si = get_imfp_mixed(KE_Si, x2)
    lambda2_O = get_imfp_mixed(KE_O, x2)
    lambda2_N = get_imfp_mixed(KE_N, x2)

    lambda3_Si = get_imfp_mixed(KE_Si, x3)
    lambda3_O = get_imfp_mixed(KE_O, x3)
    lambda3_N = get_imfp_mixed(KE_N, x3)

    lambda4_Si = imfp_tpp2m(KE_Si, *MATERIALS['Si'])

    cos_theta = np.cos(theta * np.pi / 180)
    n_angles = len(theta)

    # Attenuation factors
    att1_Si = np.exp(-d1 / lambda1_Si / cos_theta)
    att1_O = np.exp(-d1 / lambda1_O / cos_theta)
    att1_N = np.exp(-d1 / lambda1_N / cos_theta)

    att2_Si = np.exp(-d2 / lambda2_Si / cos_theta)
    att2_O = np.exp(-d2 / lambda2_O / cos_theta)
    att2_N = np.exp(-d2 / lambda2_N / cos_theta)

    att3_Si = np.exp(-d3 / lambda3_Si / cos_theta)
    att3_O = np.exp(-d3 / lambda3_O / cos_theta)
    att3_N = np.exp(-d3 / lambda3_N / cos_theta)

    # Si2p from oxide/nitride layers
    Int_Si_ON = (
        (3 - 2*x1) * lambda1_Si * (1 - att1_Si) +
        (3 - 2*x2) * lambda2_Si * (1 - att2_Si) * att1_Si +
        (3 - 2*x3) * lambda3_Si * (1 - att3_Si) * att1_Si * att2_Si
    )

    # Si2p from bulk
    Int_Si_B = lambda4_Si * att1_Si * att2_Si * att3_Si

    # O1s
    Int_O = 2 * (
        x1 * lambda1_O * (1 - att1_O) +
        x2 * lambda2_O * (1 - att2_O) * att1_O +
        x3 * lambda3_O * (1 - att3_O) * att1_O * att2_O
    )

    # N1s
    Int_N = 4 * (
        (1 - x1) * lambda1_N * (1 - att1_N) +
        (1 - x2) * lambda2_N * (1 - att2_N) * att1_N +
        (1 - x3) * lambda3_N * (1 - att3_N) * att1_N * att2_N
    )

    return np.column_stack([Int_Si_ON, Int_Si_B, Int_O, Int_N])


# =============================================================================
# Test 1: GenericLayerSolver - パターン誤認テスト
# =============================================================================

@pytest.mark.slow  # 実測 90s (2026-07-29)
def test_pattern_misidentification():
    """
    NON vs ONOのパターン誤認率を測定

    真の構造がNONの場合に、ONOモデルでフィットした場合のChi²を比較
    Chi²(ONO) / Chi²(NON) < 2.0 の場合、パターン誤認のリスクあり
    """
    print("="*80)
    print("TEST 1: Pattern Misidentification (GenericLayerSolver)")
    print("="*80)

    theta = np.array([0, 20, 35, 45, 55, 65, 75])
    hv = XRAY_SOURCES['Al-Kα']

    # 真の構造: NON (N=5Å / O=15Å / N=5Å)
    true_NON = np.array([0.0, 1.0, 0.0, 5.0, 15.0, 5.0])  # x1=N, x2=O, x3=N

    contrasts = [100, 60, 40, 25]  # %
    n_trials = 10
    noise_level = 0.02

    results = {}

    for contrast in contrasts:
        # Adjust composition for reduced contrast
        x_mix = (100 - contrast) / 200
        true_params = np.array([
            x_mix,        # Layer 1: N-rich
            1.0 - x_mix,  # Layer 2: O-rich
            x_mix,        # Layer 3: N-rich
            5.0, 15.0, 5.0
        ])

        misidentification_count = 0
        chi2_ratios = []

        for trial in range(n_trials):
            # Generate synthetic data
            I_true = simulate_layer_model(true_params, theta, hv)
            if I_true is None:
                continue

            np.random.seed(trial)
            I_obs = I_true * (1 + noise_level * np.random.randn(*I_true.shape))

            # Fit with NON model (correct)
            def objective_NON(params):
                I_calc = simulate_layer_model(params, theta, hv)
                if I_calc is None:
                    return 1e10
                ratio_obs = I_obs[:, 3] / (I_obs[:, 2] + 1e-10)
                ratio_calc = I_calc[:, 3] / (I_calc[:, 2] + 1e-10)
                return np.sum((ratio_obs - ratio_calc)**2)

            # NON bounds: x1~0, x2~1, x3~0
            bounds_NON = [(0, 0.3), (0.7, 1), (0, 0.3), (1, 50), (1, 50), (1, 50)]
            result_NON = differential_evolution(objective_NON, bounds_NON, seed=trial, maxiter=200, tol=1e-6)

            # Fit with ONO model (incorrect)
            # ONO: x1~1, x2~0, x3~1
            bounds_ONO = [(0.7, 1), (0, 0.3), (0.7, 1), (1, 50), (1, 50), (1, 50)]
            result_ONO = differential_evolution(objective_NON, bounds_ONO, seed=trial, maxiter=200, tol=1e-6)

            chi2_NON = result_NON.fun
            chi2_ONO = result_ONO.fun

            ratio = chi2_ONO / chi2_NON if chi2_NON > 0 else 1e10
            chi2_ratios.append(ratio)

            # パターン誤認: Chi²比が小さい（ONOでも良くフィット）
            if ratio < 2.0:
                misidentification_count += 1

        misidentification_rate = misidentification_count / n_trials
        results[contrast] = {
            'misidentification_rate': misidentification_rate,
            'chi2_ratio_mean': np.mean(chi2_ratios),
            'chi2_ratio_std': np.std(chi2_ratios),
        }

        print(f"Contrast {contrast:3d}%: Misidentification rate = {misidentification_rate*100:5.1f}%, "
              f"Chi²(ONO)/Chi²(NON) = {np.mean(chi2_ratios):.2f}±{np.std(chi2_ratios):.2f}")

    return results


# =============================================================================
# Test 2: 090317 Method - Bimodal検出テスト
# =============================================================================

def test_bimodal_detection():
    """
    Bimodal N分布の検出成功率を測定

    NON構造（N/O/N）の場合、N濃度プロファイルは2つのピークを持つべき
    """
    print("\n" + "="*80)
    print("TEST 2: Bimodal Detection (090317 Method)")
    print("="*80)

    step_size = 1.0
    depth = np.arange(0, 50, step_size)
    theta = np.array([0, 20, 35, 45, 55, 65, 75])
    hv = XRAY_SOURCES['Al-Kα']

    # True NON structure
    n1_thick, o_thick, n2_thick = 5.0, 15.0, 5.0

    noise_levels = [0.02, 0.05, 0.10]
    n_trials = 10

    results = {}

    for noise in noise_levels:
        detection_count = 0

        for trial in range(n_trials):
            # Create true profile
            D_true = np.zeros((len(depth), 4))  # Si-B, Si-O, O, N
            for j, z in enumerate(depth):
                if z < n1_thick:
                    D_true[j] = [0, 3/7, 0, 4/7]
                elif z < n1_thick + o_thick:
                    D_true[j] = [0, 1/3, 2/3, 0]
                elif z < n1_thick + o_thick + n2_thick:
                    D_true[j] = [0, 3/7, 0, 4/7]
                else:
                    D_true[j] = [1, 0, 0, 0]

            # Simulate and add noise
            KE_O = hv - BE_O1s
            KE_N = hv - BE_N1s
            imfp_O = get_imfp_mixed(KE_O, 0.5)
            imfp_N = get_imfp_mixed(KE_N, 0.5)

            # Build transfer matrix (simplified)
            T_O = np.zeros((len(theta), len(depth)))
            T_N = np.zeros((len(theta), len(depth)))
            for i, ang in enumerate(theta):
                cos_ang = np.cos(np.radians(ang))
                for j in range(len(depth)):
                    T_O[i, j] = np.exp(-j * step_size / imfp_O / cos_ang)
                    T_N[i, j] = np.exp(-j * step_size / imfp_N / cos_ang)

            I_O = T_O @ D_true[:, 2]
            I_N = T_N @ D_true[:, 3]

            np.random.seed(trial)
            I_O_obs = I_O * (1 + noise * np.random.randn(*I_O.shape))
            I_N_obs = I_N * (1 + noise * np.random.randn(*I_N.shape))

            # Simple inversion: D ≈ (T^T T)^-1 T^T I (Tikhonov with small alpha)
            alpha = 0.001
            D_N_opt = np.linalg.solve(T_N.T @ T_N + alpha * np.eye(len(depth)), T_N.T @ I_N_obs)
            D_N_opt = np.clip(D_N_opt, 0, 1)

            # Detect bimodal distribution
            peaks, _ = find_peaks(D_N_opt, height=0.1 * np.max(D_N_opt), distance=3)

            is_bimodal = len(peaks) >= 2
            if is_bimodal:
                detection_count += 1

        detection_rate = detection_count / n_trials
        results[noise] = {
            'detection_rate': detection_rate,
            'n_trials': n_trials,
        }

        print(f"Noise {noise*100:5.1f}%: Bimodal detection rate = {detection_rate*100:5.1f}%")

    return results


# =============================================================================
# Test 3: FreeSolver - 事前分布依存性テスト
# =============================================================================

def test_prior_dependence():
    """
    事前分布の選択が結果に与える影響を測定

    - Exponential Prior: 深さとともに指数減衰
    - Bimodal Prior: 表面と界面に2つのピーク
    - Uniform Prior: 均一分布
    """
    print("\n" + "="*80)
    print("TEST 3: Prior Dependence (FreeSolver)")
    print("="*80)

    step_size = 1.0
    depth = np.arange(0, 50, step_size)
    theta = np.array([0, 20, 35, 45, 55, 65, 75])
    hv = XRAY_SOURCES['Al-Kα']

    # True NON structure
    n1_thick, o_thick, n2_thick = 5.0, 15.0, 5.0
    total_thick = n1_thick + o_thick + n2_thick

    noise = 0.02
    n_trials = 10

    # Create true profile
    D_true_N = np.zeros(len(depth))
    for j, z in enumerate(depth):
        if z < n1_thick:
            D_true_N[j] = 4/7
        elif z < n1_thick + o_thick:
            D_true_N[j] = 0
        elif z < n1_thick + o_thick + n2_thick:
            D_true_N[j] = 4/7
        else:
            D_true_N[j] = 0

    priors = {
        'Exponential': np.exp(-depth / 20) * 0.5,
        'Bimodal': 0.3 * (np.exp(-((depth - 2.5)**2) / 10) + np.exp(-((depth - 22.5)**2) / 10)),
        'Uniform': np.ones(len(depth)) * 0.3,
    }

    KE_N = hv - BE_N1s
    imfp_N = get_imfp_mixed(KE_N, 0.5)

    # Build transfer matrix
    T_N = np.zeros((len(theta), len(depth)))
    for i, ang in enumerate(theta):
        cos_ang = np.cos(np.radians(ang))
        for j in range(len(depth)):
            T_N[i, j] = np.exp(-j * step_size / imfp_N / cos_ang)

    results = {}

    for prior_name, D_prior in priors.items():
        rmse_values = []
        bimodal_detected = 0

        for trial in range(n_trials):
            # Simulate observation
            I_N = T_N @ D_true_N
            np.random.seed(trial)
            I_N_obs = I_N * (1 + noise * np.random.randn(*I_N.shape))

            # MEM-like optimization with prior
            def objective(x):
                D_N = D_prior * np.exp(np.clip(x, -10, 10))
                I_calc = T_N @ D_N
                chi2 = np.sum((I_N_obs - I_calc)**2)
                # Entropy term
                S = -np.sum(D_N * np.log(D_N / D_prior + 1e-10))
                return chi2 - 0.001 * S

            x0 = np.zeros(len(depth))
            result = minimize(objective, x0, method='L-BFGS-B',
                            bounds=[(-5, 5)] * len(depth), options={'maxiter': 200})

            D_N_opt = D_prior * np.exp(np.clip(result.x, -10, 10))

            # Calculate RMSE
            rmse = np.sqrt(np.mean((D_N_opt - D_true_N)**2))
            rmse_values.append(rmse)

            # Check bimodal
            peaks, _ = find_peaks(D_N_opt, height=0.1 * np.max(D_N_opt), distance=3)
            if len(peaks) >= 2:
                bimodal_detected += 1

        results[prior_name] = {
            'rmse_mean': np.mean(rmse_values),
            'rmse_std': np.std(rmse_values),
            'bimodal_rate': bimodal_detected / n_trials,
        }

        print(f"{prior_name:12s}: RMSE = {np.mean(rmse_values):.4f}±{np.std(rmse_values):.4f}, "
              f"Bimodal detection = {bimodal_detected/n_trials*100:.0f}%")

    return results


# =============================================================================
# Test 4: 破綻モード統合検出
# =============================================================================

def detect_failure_modes(params_true: np.ndarray, params_fit: np.ndarray,
                         optimizer_success: bool, N_profile: np.ndarray = None) -> FailureMode:
    """
    フィット結果から破綻モードを検出
    """
    failure = FailureMode()

    x1_t, x2_t, x3_t = params_true[0], params_true[1], params_true[2]
    d1_t, d2_t, d3_t = params_true[3], params_true[4], params_true[5]

    x1_f, x2_f, x3_f = params_fit[0], params_fit[1], params_fit[2]
    d1_f, d2_f, d3_f = params_fit[3], params_fit[4], params_fit[5]

    # 1. パターン誤認: 層の順序が逆転
    true_pattern = 'N' if x1_t < 0.5 else 'O'
    fit_pattern = 'N' if x1_f < 0.5 else 'O'
    if true_pattern != fit_pattern:
        failure.pattern_misidentification = True

    # 2. 膜厚収縮: 隣接層が融合
    if d1_f < 0.2 * d1_t or d2_f < 0.2 * d2_t or d3_f < 0.2 * d3_t:
        failure.thickness_collapse = True

    # 3. 膜厚発散
    if d1_f < 0 or d2_f < 0 or d3_f < 0 or d1_f > 100 or d2_f > 100 or d3_f > 100:
        failure.thickness_divergence = True

    # 4. Bimodal検出失敗
    if N_profile is not None:
        peaks, _ = find_peaks(N_profile, height=0.1 * np.max(N_profile), distance=3)
        if len(peaks) < 2:
            failure.bimodal_failure = True

    # 5. 非収束
    if not optimizer_success:
        failure.non_convergence = True

    return failure


@pytest.mark.slow  # 実測 44s (2026-07-29)
def test_integrated_failure_detection():
    """
    統合的な破綻モード検出テスト
    """
    print("\n" + "="*80)
    print("TEST 4: Integrated Failure Mode Detection")
    print("="*80)

    theta = np.array([0, 20, 35, 45, 55, 65, 75])
    hv = XRAY_SOURCES['Al-Kα']

    # True NON structure
    true_params = np.array([0.0, 1.0, 0.0, 5.0, 15.0, 5.0])

    conditions = [
        {'contrast': 100, 'noise': 0.02},
        {'contrast': 60, 'noise': 0.02},
        {'contrast': 40, 'noise': 0.02},
        {'contrast': 40, 'noise': 0.10},
    ]

    n_trials = 10
    results = {}

    for cond in conditions:
        contrast = cond['contrast']
        noise = cond['noise']

        x_mix = (100 - contrast) / 200
        params = np.array([x_mix, 1.0 - x_mix, x_mix, 5.0, 15.0, 5.0])

        failure_counts = {
            'pattern_misidentification': 0,
            'thickness_collapse': 0,
            'thickness_divergence': 0,
            'bimodal_failure': 0,
            'non_convergence': 0,
            'any_failure': 0,
        }

        for trial in range(n_trials):
            I_true = simulate_layer_model(params, theta, hv)
            if I_true is None:
                continue

            np.random.seed(trial)
            I_obs = I_true * (1 + noise * np.random.randn(*I_true.shape))

            def objective(p):
                I_calc = simulate_layer_model(p, theta, hv)
                if I_calc is None:
                    return 1e10
                ratio_obs = I_obs[:, 3] / (I_obs[:, 2] + 1e-10)
                ratio_calc = I_calc[:, 3] / (I_calc[:, 2] + 1e-10)
                return np.sum((ratio_obs - ratio_calc)**2)

            bounds = [(0, 1), (0, 1), (0, 1), (0.1, 50), (0.1, 50), (0.1, 50)]
            result = differential_evolution(objective, bounds, seed=trial, maxiter=200, tol=1e-6)

            failure = detect_failure_modes(params, result.x, result.success)

            for key in failure_counts:
                if key == 'any_failure':
                    if failure.any_failure:
                        failure_counts[key] += 1
                elif getattr(failure, key):
                    failure_counts[key] += 1

        key = f"C{contrast}_N{int(noise*100)}"
        results[key] = {k: v / n_trials for k, v in failure_counts.items()}

        print(f"\nContrast={contrast}%, Noise={noise*100:.0f}%:")
        print(f"  Pattern misidentification: {failure_counts['pattern_misidentification']/n_trials*100:.0f}%")
        print(f"  Thickness collapse:        {failure_counts['thickness_collapse']/n_trials*100:.0f}%")
        print(f"  Thickness divergence:      {failure_counts['thickness_divergence']/n_trials*100:.0f}%")
        print(f"  Any failure:               {failure_counts['any_failure']/n_trials*100:.0f}%")

    return results


# =============================================================================
# サマリーテーブル生成
# =============================================================================

def create_summary_table(results: Dict) -> str:
    """Markdown形式のサマリーテーブルを生成"""

    lines = []
    lines.append("## ソルバー別破綻モード サマリー")
    lines.append("")
    lines.append("### Table: 破綻条件")
    lines.append("")
    lines.append("| ソルバー | 最小コントラスト | 破綻モード | 備考 |")
    lines.append("|---------|----------------|----------|------|")

    if 'pattern_misidentification' in results:
        pm = results['pattern_misidentification']
        min_contrast = 100
        for c, v in pm.items():
            if v['misidentification_rate'] > 0.3:
                min_contrast = min(min_contrast, c)
        lines.append(f"| GenericLayerSolver | {min_contrast}% | パターン誤認 | Chi²比で検出 |")

    if 'bimodal_detection' in results:
        bd = results['bimodal_detection']
        max_noise = 2
        for n, v in bd.items():
            if v['detection_rate'] >= 0.7:
                max_noise = max(max_noise, int(n * 100))
        lines.append(f"| 090317 Method | - | Bimodal失敗 | ノイズ>{max_noise}%で失敗 |")

    if 'prior_dependence' in results:
        pd = results['prior_dependence']
        best_prior = min(pd.items(), key=lambda x: x[1]['rmse_mean'])[0]
        lines.append(f"| FreeSolver | - | 事前分布依存 | {best_prior}が最適 |")

    return "\n".join(lines)


# =============================================================================
# メイン実行
# =============================================================================

def main():
    print("="*80)
    print("SOLVER FAILURE MODE ANALYSIS")
    print("="*80)

    results = {}

    # Test 1: Pattern misidentification
    results['pattern_misidentification'] = test_pattern_misidentification()

    # Test 2: Bimodal detection
    results['bimodal_detection'] = test_bimodal_detection()

    # Test 3: Prior dependence
    results['prior_dependence'] = test_prior_dependence()

    # Test 4: Integrated failure detection
    results['integrated'] = test_integrated_failure_detection()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(create_summary_table(results))

    # Save plot
    from deppro.paths import ensure_outputs_dir
    output_dir = str(ensure_outputs_dir())
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Pattern misidentification rate
    ax = axes[0, 0]
    pm = results['pattern_misidentification']
    contrasts = list(pm.keys())
    rates = [pm[c]['misidentification_rate'] * 100 for c in contrasts]
    ax.bar(range(len(contrasts)), rates, color='coral')
    ax.set_xticks(range(len(contrasts)))
    ax.set_xticklabels([f'{c}%' for c in contrasts])
    ax.set_xlabel('Contrast')
    ax.set_ylabel('Misidentification Rate (%)')
    ax.set_title('Pattern Misidentification (NON→ONO)')
    ax.axhline(30, color='red', linestyle='--', label='Threshold')
    ax.legend()

    # Plot 2: Bimodal detection rate
    ax = axes[0, 1]
    bd = results['bimodal_detection']
    noises = list(bd.keys())
    rates = [bd[n]['detection_rate'] * 100 for n in noises]
    ax.bar(range(len(noises)), rates, color='steelblue')
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([f'{n*100:.0f}%' for n in noises])
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Bimodal N Detection (090317)')
    ax.axhline(70, color='green', linestyle='--', label='Success threshold')
    ax.legend()

    # Plot 3: Prior dependence RMSE
    ax = axes[1, 0]
    pd = results['prior_dependence']
    priors = list(pd.keys())
    rmses = [pd[p]['rmse_mean'] for p in priors]
    ax.bar(range(len(priors)), rmses, color='forestgreen')
    ax.set_xticks(range(len(priors)))
    ax.set_xticklabels(priors)
    ax.set_xlabel('Prior Type')
    ax.set_ylabel('RMSE')
    ax.set_title('Prior Dependence (FreeSolver)')

    # Plot 4: Integrated failure rates
    ax = axes[1, 1]
    ig = results['integrated']
    conds = list(ig.keys())
    failure_rates = [ig[c]['any_failure'] * 100 for c in conds]
    ax.bar(range(len(conds)), failure_rates, color='purple')
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace('_', '\n') for c in conds], fontsize=8)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Any Failure Rate (%)')
    ax.set_title('Integrated Failure Detection')
    ax.axhline(50, color='red', linestyle='--', label='50% threshold')
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'solver_failure_modes.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close()

    return results


if __name__ == '__main__':
    results = main()
