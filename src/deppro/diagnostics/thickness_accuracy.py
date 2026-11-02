#!/usr/bin/env python
"""
膜厚決定精度の系統的検証

膜厚（小・中・大）× ノイズレベル × 初期値条件

Note: LayerStructureSolverは4層膜(C/Al2O3/TiO2/SiO2/Si)専用設計のため、
      単層SiO2/Si系のテストからは除外。
      代わりに、SemiFreeSolverの初期値ロバスト性を2パターンでテスト。
"""

import numpy as np
from scipy.special import erf
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/src

from deppro import (
    TransferMatrix, SolverConfig, ElementConfig,
    SemiFreeSolver, SemiFreeSolverConfig,
)


def create_sio2_si_profile(depth, oxide_thickness, interface_width=5.0):
    """SiO2/Si 2層モデルを生成"""
    n_depth = len(depth)
    D = np.zeros((n_depth, 3))
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    erf_profile = 0.5 * (1 + erf((depth - oxide_thickness) / sigma))
    D[:, 2] = erf_profile  # Si_bulk
    film_fraction = 1.0 - erf_profile
    D[:, 0] = film_fraction / 3.0  # Si_ox
    D[:, 1] = 2.0 * D[:, 0]        # O
    return D


def run_single_test(tm, elements, depth, true_thickness, interface_width,
                    I_obs, thickness_init, alpha=0.001, method='MEM'):
    """単一の推定を実行"""
    semi_config = SemiFreeSolverConfig(
        thickness_init=thickness_init,
        abruptness_init=interface_width,
        free_indices=[0],
        linked_indices=[1],
        substrate_indices=[2],
        optimize_thickness=True,
        optimize_abruptness=False,
        thickness_bounds=(1.0, 200.0),
    )
    D_prior = create_sio2_si_profile(depth, thickness_init, interface_width)
    solver = SemiFreeSolver(
        tm, SolverConfig(method=method, alpha=alpha, max_iter=500),
        semi_config, D_prior, elements
    )
    result = solver.solve(I_obs, verbose=False)
    return result


def run_thickness_accuracy_test():
    """膜厚決定精度の系統的検証"""

    print("=" * 80)
    print("膜厚決定精度の系統的検証 (SemiFreeSolver)")
    print("=" * 80)
    print("")
    print("条件:")
    print("  - 界面幅: 5.0 Å (固定、既知として与える)")
    print("  - 角度: [15, 30, 45, 60, 75, 85]°")
    print("  - 試行回数: 30回/条件")
    print("  - ソルバー: SemiFreeSolver (MEM, α=0.001)")
    print("")

    # 共通設定
    depth = np.linspace(0, 200, 201)
    interface_width = 5.0
    angles = np.array([15, 30, 45, 60, 75, 85])

    elements = [
        ElementConfig(symbol='Si', orbital='2p', compound='SiO2',
                     binding_energy=103.5, oxidation_state=4),
        ElementConfig(symbol='O', orbital='1s', compound='SiO2',
                     binding_energy=531.0, oxidation_state=-2),
        ElementConfig(symbol='Si', orbital='2p', compound='Si',
                     binding_energy=99.3, oxidation_state=0),
    ]

    # テスト条件
    true_thicknesses = [10.0, 30.0, 80.0]  # 小・中・大
    noise_levels = [0.0, 0.01, 0.02, 0.05]  # 0%, 1%, 2%, 5%
    n_trials = 30

    # 結果格納
    results_good_init = {}   # 良い初期値 (真値 ± 5 Å)
    results_poor_init = {}   # 悪い初期値 (真値 ± 20 Å)

    for true_thickness in true_thicknesses:
        print(f"\n{'='*80}")
        print(f"真の膜厚: {true_thickness:.0f} Å")
        print(f"{'='*80}")

        # 真のプロファイル生成
        D_true = create_sio2_si_profile(depth, true_thickness, interface_width)

        # 転送行列
        tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
        tm.compute(D_true, elements)
        I_obs_clean = tm.forward(D_true, normalize=True)

        for noise_level in noise_levels:
            print(f"\n--- ノイズ: {noise_level*100:.0f}% ---")

            errors_good = []
            errors_poor = []

            for trial in range(n_trials):
                np.random.seed(42 + trial)

                # ノイズ追加
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level, I_obs_clean.shape)
                    I_obs = I_obs_clean * (1 + noise)
                    I_obs = np.clip(I_obs, 0, None)
                    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
                else:
                    I_obs = I_obs_clean.copy()

                # ===== 良い初期値 (真値 ± 5 Å) =====
                thickness_init = true_thickness + np.random.uniform(-5, 5)
                thickness_init = max(3.0, thickness_init)
                try:
                    result = run_single_test(
                        tm, elements, depth, true_thickness, interface_width,
                        I_obs, thickness_init
                    )
                    error = result.thickness - true_thickness
                    errors_good.append(error)
                except Exception:
                    pass

                # ===== 悪い初期値 (真値 ± 20 Å) =====
                thickness_init = true_thickness + np.random.uniform(-20, 20)
                thickness_init = max(3.0, thickness_init)
                try:
                    result = run_single_test(
                        tm, elements, depth, true_thickness, interface_width,
                        I_obs, thickness_init
                    )
                    error = result.thickness - true_thickness
                    errors_poor.append(error)
                except Exception:
                    pass

            # 統計計算
            def calc_stats(errors):
                if errors:
                    # 外れ値除去（3σ）
                    arr = np.array(errors)
                    mean = np.mean(arr)
                    std = np.std(arr)
                    mask = np.abs(arr - mean) < 3 * std
                    arr_clean = arr[mask]
                    if len(arr_clean) > 0:
                        return {
                            'mean': np.mean(arr_clean),
                            'std': np.std(arr_clean),
                            'abs_mean': np.mean(np.abs(arr_clean)),
                            'n_success': len(arr_clean),
                            'n_total': len(errors),
                        }
                return {'mean': np.nan, 'std': np.nan, 'abs_mean': np.nan,
                        'n_success': 0, 'n_total': n_trials}

            stats_good = calc_stats(errors_good)
            stats_poor = calc_stats(errors_poor)

            print(f"  良い初期値(±5Å):  誤差 = {stats_good['mean']:+.2f} ± {stats_good['std']:.2f} Å, "
                  f"|誤差| = {stats_good['abs_mean']:.2f} Å ({stats_good['n_success']}/{stats_good['n_total']})")
            print(f"  悪い初期値(±20Å): 誤差 = {stats_poor['mean']:+.2f} ± {stats_poor['std']:.2f} Å, "
                  f"|誤差| = {stats_poor['abs_mean']:.2f} Å ({stats_poor['n_success']}/{stats_poor['n_total']})")

            key = (true_thickness, noise_level)
            results_good_init[key] = stats_good
            results_poor_init[key] = stats_poor

    # まとめ表
    print("\n")
    print("=" * 80)
    print("結果まとめ: 良い初期値 (真値 ± 5 Å)")
    print("=" * 80)
    print(f"{'膜厚(Å)':<10} {'ノイズ':<8} {'誤差(Å)':<18} {'|誤差|(Å)':<12} {'有効/全':<10}")
    print("-" * 65)
    for true_thickness in true_thicknesses:
        for noise_level in noise_levels:
            key = (true_thickness, noise_level)
            r = results_good_init[key]
            print(f"{true_thickness:<10.0f} {noise_level*100:<8.0f}% "
                  f"{r['mean']:+.2f} ± {r['std']:.2f}      {r['abs_mean']:<12.2f} "
                  f"{r['n_success']}/{r['n_total']}")

    print("\n")
    print("=" * 80)
    print("結果まとめ: 悪い初期値 (真値 ± 20 Å)")
    print("=" * 80)
    print(f"{'膜厚(Å)':<10} {'ノイズ':<8} {'誤差(Å)':<18} {'|誤差|(Å)':<12} {'有効/全':<10}")
    print("-" * 65)
    for true_thickness in true_thicknesses:
        for noise_level in noise_levels:
            key = (true_thickness, noise_level)
            r = results_poor_init[key]
            print(f"{true_thickness:<10.0f} {noise_level*100:<8.0f}% "
                  f"{r['mean']:+.2f} ± {r['std']:.2f}      {r['abs_mean']:<12.2f} "
                  f"{r['n_success']}/{r['n_total']}")

    # 膜厚依存性まとめ
    print("\n")
    print("=" * 80)
    print("膜厚依存性まとめ (2%ノイズ, 良い初期値)")
    print("=" * 80)
    print(f"{'膜厚(Å)':<12} {'誤差(Å)':<18} {'相対誤差(%)':<15}")
    print("-" * 45)
    noise_level = 0.02
    for true_thickness in true_thicknesses:
        key = (true_thickness, noise_level)
        r = results_good_init[key]
        rel_error = abs(r['mean']) / true_thickness * 100 if not np.isnan(r['mean']) else np.nan
        print(f"{true_thickness:<12.0f} {r['mean']:+.2f} ± {r['std']:.2f}      {rel_error:.1f}%")

    return results_good_init, results_poor_init


if __name__ == '__main__':
    run_thickness_accuracy_test()
