"""
初期値ロバスト性テスト
========================

様々な初期値から始めたときの収束の割合と誤差を調べる（1 Å 以内を収束とみなす）。

Usage:
    python -m deppro.test_initial_value_robustness
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from scipy.optimize import minimize, Bounds
from scipy.special import erf
from pathlib import Path
import time

from deppro.depth_profile import create_multilayer_profile


@dataclass
class RobustnessTestConfig:
    """ロバスト性テスト設定"""
    # 真値
    true_thicknesses: List[float] = None  # 累積界面位置 [5, 15, 35, 65]
    true_widths: List[float] = None  # [2, 3, 4, 5]

    # 測定条件
    angles_deg: np.ndarray = None
    noise_level: float = 0.02

    # 最適化設定
    n_trials: int = 50  # 試行回数
    max_iter: int = 500
    thickness_bounds: Tuple[float, float] = (1.0, 150.0)

    def __post_init__(self):
        if self.true_thicknesses is None:
            self.true_thicknesses = [5.0, 15.0, 35.0, 65.0]  # 累積位置
        if self.true_widths is None:
            self.true_widths = [2.0, 3.0, 4.0, 5.0]
        if self.angles_deg is None:
            self.angles_deg = np.array([0, 30, 45, 55, 65, 75])


def build_multilayer_profile(
    depth: np.ndarray,
    thicknesses: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    """
    Build C/Al₂O₃/TiO₂/SiO₂/Si profile from parameters.

    Args:
        depth: Depth array
        thicknesses: Cumulative interface positions [4]
        widths: Interface widths FWHM [4]

    Returns:
        Concentration array [n_depth, 6]
    """
    n_depth = len(depth)
    z = depth

    # Convert FWHM to sigma
    sigmas = widths / (2 * np.sqrt(np.log(2)))

    def erf_layer(z_start, z_end, sigma_start, sigma_end):
        rising = 0.5 * (1 + erf((z - z_start) / sigma_start))
        falling = 0.5 * (1 + erf((z - z_end) / sigma_end))
        return np.maximum(rising - falling, 0)

    D = np.zeros((n_depth, 6))

    # Carbon
    t_C = thicknesses[0]
    D_C = 0.3 * np.exp(-z / 3.0) * (1 - 0.5 * (1 + erf((z - t_C) / sigmas[0])))
    D[:, 0] = np.maximum(D_C, 0)

    # Al₂O₃
    D[:, 1] = 0.4 * erf_layer(thicknesses[0], thicknesses[1], sigmas[0], sigmas[1])

    # TiO₂
    D[:, 2] = 0.333 * erf_layer(thicknesses[1], thicknesses[2], sigmas[1], sigmas[2])

    # SiO₂
    D[:, 3] = 0.333 * erf_layer(thicknesses[2], thicknesses[3], sigmas[2], sigmas[3])

    # Si substrate
    D[:, 5] = 0.5 * (1 + erf((z - thicknesses[3]) / sigmas[3]))

    # Oxygen from stoichiometry
    D[:, 4] = 1.5 * D[:, 1] + 2.0 * D[:, 2] + 2.0 * D[:, 3]

    # Normalize
    D = np.clip(D, 1e-6, 1.0)
    total = D.sum(axis=1, keepdims=True)
    D = D / np.maximum(total, 1e-10)

    return D


def build_transfer_matrices(depth: np.ndarray, angles_deg: np.ndarray) -> List[np.ndarray]:
    """角度分解用の転送行列を構築"""
    n_depth = len(depth)
    n_angles = len(angles_deg)
    dz = depth[1] - depth[0] if len(depth) > 1 else 1.0

    # 元素ごとのIMFP (Å) - C, Al, Ti, Si(ox), O, Si(sub)
    imfp_values = np.array([25.0, 20.0, 22.0, 20.0, 24.0, 28.0])

    T_matrices = []
    for elem_idx in range(6):
        T = np.zeros((n_angles, n_depth))
        for a, angle in enumerate(angles_deg):
            theta = np.radians(angle)
            lambda_eff = imfp_values[elem_idx] * np.cos(theta)
            for j in range(n_depth):
                T[a, j] = np.exp(-depth[j] / lambda_eff) * dz / lambda_eff
        T_matrices.append(T)

    return T_matrices


def generate_random_initial_values(
    true_positions: np.ndarray,
    perturbation_range: Tuple[float, float],
    n_samples: int,
    seed: int = None
) -> np.ndarray:
    """ランダムな初期値を生成

    Args:
        true_positions: 真の界面位置
        perturbation_range: (min_perturb, max_perturb) 摂動範囲
        n_samples: サンプル数
        seed: 乱数シード

    Returns:
        shape (n_samples, n_interfaces) の初期値配列
    """
    if seed is not None:
        np.random.seed(seed)

    n_interfaces = len(true_positions)
    min_p, max_p = perturbation_range

    initial_values = np.zeros((n_samples, n_interfaces))

    for i in range(n_samples):
        perturbations = np.random.uniform(min_p, max_p, n_interfaces)
        # ランダムに符号を決める
        signs = np.random.choice([-1, 1], n_interfaces)
        initial_values[i] = true_positions + perturbations * signs

        # 位置が負にならないようクリップ
        initial_values[i] = np.maximum(initial_values[i], 1.0)
        # 順序を保つ（小さい順にソート）
        initial_values[i] = np.sort(initial_values[i])

    return initial_values


def solve_with_initial(
    I_obs: np.ndarray,
    T_matrices: List[np.ndarray],
    depth: np.ndarray,
    fixed_widths: np.ndarray,
    initial_positions: np.ndarray,
    config: RobustnessTestConfig,
) -> Tuple[np.ndarray, float, int]:
    """指定した初期値から最適化

    Returns:
        (optimized_positions, final_rmse, n_iterations)
    """
    n_angles, n_elements = I_obs.shape
    n_interfaces = len(initial_positions)

    I_obs_norm = I_obs / np.maximum(I_obs.sum(axis=1, keepdims=True), 1e-10)
    n_iter = [0]

    def objective(thicknesses):
        D = build_multilayer_profile(depth, thicknesses, fixed_widths)

        I_pred = np.zeros_like(I_obs)
        for e in range(6):
            I_pred[:, e] = T_matrices[e] @ D[:, e]
        I_pred_norm = I_pred / np.maximum(I_pred.sum(axis=1, keepdims=True), 1e-10)

        chi2 = np.sum((I_obs_norm - I_pred_norm) ** 2)
        n_iter[0] += 1
        return chi2

    # 単調性制約
    constraints = []
    for i in range(1, n_interfaces):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i] - x[i-1] - 1.0,
        })

    result = minimize(
        objective,
        initial_positions,
        method='SLSQP',
        bounds=Bounds(
            lb=np.full(n_interfaces, config.thickness_bounds[0]),
            ub=np.full(n_interfaces, config.thickness_bounds[1]),
        ),
        constraints=constraints,
        options={'maxiter': config.max_iter, 'ftol': 1e-10},
    )

    final_positions = result.x
    final_rmse = np.sqrt(result.fun / (n_angles * n_elements))

    return final_positions, final_rmse, n_iter[0]


def run_robustness_test(config: RobustnessTestConfig, verbose: bool = True):
    """ロバスト性テストを実行"""

    print("=" * 70)
    print("初期値ロバスト性テスト")
    print("=" * 70)

    # 真のプロファイル生成
    true_positions = np.array(config.true_thicknesses)
    layer_thicknesses = [true_positions[0]]
    for i in range(1, len(true_positions)):
        layer_thicknesses.append(true_positions[i] - true_positions[i-1])

    print(f"\n真の界面位置: {true_positions} Å")
    print(f"真の界面幅: {config.true_widths}")
    print(f"ノイズレベル: {config.noise_level * 100:.0f}%")
    print(f"試行回数: {config.n_trials}")

    # 深さ軸
    depth = np.linspace(0, 100, 101)
    n_depth = len(depth)

    # 真のプロファイルを生成
    D_true = build_multilayer_profile(
        depth,
        true_positions,
        np.array(config.true_widths),
    )

    # 転送行列
    T_matrices = build_transfer_matrices(depth, config.angles_deg)

    # 観測強度生成
    n_angles = len(config.angles_deg)
    n_elements = 6
    I_true = np.zeros((n_angles, n_elements))
    for e in range(n_elements):
        I_true[:, e] = T_matrices[e] @ D_true[:, e]

    # ノイズ追加
    np.random.seed(42)
    noise = np.random.normal(0, config.noise_level, I_true.shape)
    I_obs = I_true * (1 + noise)
    I_obs = np.maximum(I_obs, 1e-10)

    # 固定する界面幅
    fixed_widths = np.array(config.true_widths)

    # テストケース定義
    test_cases = [
        ("小摂動 (±3 Å)", (1, 3)),
        ("中摂動 (±5 Å)", (3, 5)),
        ("大摂動 (±10 Å)", (5, 10)),
        ("極大摂動 (±15 Å)", (10, 15)),
        ("ランダム (±20 Å)", (10, 20)),
    ]

    results = {}

    for case_name, perturb_range in test_cases:
        print(f"\n{'-' * 60}")
        print(f"テストケース: {case_name}")
        print(f"{'-' * 60}")

        # 初期値生成
        initial_values = generate_random_initial_values(
            true_positions,
            perturb_range,
            config.n_trials,
            seed=123
        )

        # 各初期値から最適化
        errors_list = []
        converged_count = 0
        start_time = time.time()

        for trial in range(config.n_trials):
            init_pos = initial_values[trial]

            final_pos, rmse, n_iter = solve_with_initial(
                I_obs, T_matrices, depth,
                fixed_widths, init_pos, config
            )

            # 誤差計算
            errors = final_pos - true_positions
            mean_abs_error = np.mean(np.abs(errors))
            errors_list.append(errors)

            # 1Å以内に収束したかチェック
            if mean_abs_error < 1.0:
                converged_count += 1

            if verbose and trial < 5:
                print(f"  Trial {trial+1}: init={init_pos.round(1)}, "
                      f"final={final_pos.round(1)}, err={errors.round(2)}")

        elapsed = time.time() - start_time

        errors_array = np.array(errors_list)
        mean_errors = np.mean(np.abs(errors_array), axis=0)
        std_errors = np.std(errors_array, axis=0)

        convergence_rate = converged_count / config.n_trials * 100

        print(f"\n  結果:")
        print(f"    平均絶対誤差: {mean_errors.round(2)} Å")
        print(f"    誤差標準偏差: {std_errors.round(2)} Å")
        print(f"    全体平均誤差: {np.mean(mean_errors):.2f} Å")
        print(f"    1Å以内収束率: {convergence_rate:.0f}% ({converged_count}/{config.n_trials})")
        print(f"    計算時間: {elapsed:.1f}秒 ({elapsed/config.n_trials*1000:.0f}ms/試行)")

        results[case_name] = {
            'mean_errors': mean_errors,
            'std_errors': std_errors,
            'convergence_rate': convergence_rate,
            'all_errors': errors_array,
        }

    # 結果サマリー
    print("\n" + "=" * 70)
    print("結果サマリー")
    print("=" * 70)
    print(f"\n{'ケース':<20} | {'平均誤差':>10} | {'収束率':>10}")
    print("-" * 50)
    for case_name, res in results.items():
        mean = np.mean(res['mean_errors'])
        rate = res['convergence_rate']
        print(f"{case_name:<20} | {mean:>8.2f} Å | {rate:>8.0f}%")

    # 可視化
    create_robustness_visualization(results, true_positions, config)

    return results


def create_robustness_visualization(results, true_positions, config):
    """ロバスト性テスト結果の可視化"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 収束率バーチャート
    ax1 = axes[0, 0]
    case_names = list(results.keys())
    convergence_rates = [results[c]['convergence_rate'] for c in case_names]

    colors = plt.cm.RdYlGn(np.array(convergence_rates) / 100)
    bars = ax1.bar(range(len(case_names)), convergence_rates, color=colors)
    ax1.set_xticks(range(len(case_names)))
    ax1.set_xticklabels([c.split('(')[0].strip() for c in case_names], rotation=15)
    ax1.set_ylabel('Convergence Rate (%)')
    ax1.set_title('Convergence Rate by Initial Value Perturbation')
    ax1.set_ylim(0, 105)

    for bar, rate in zip(bars, convergence_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', fontsize=10)

    # 2. 界面ごとの誤差分布
    ax2 = axes[0, 1]
    interface_names = ['C/Al₂O₃', 'Al₂O₃/TiO₂', 'TiO₂/SiO₂', 'SiO₂/Si']

    # 大摂動ケースの誤差分布を表示
    large_perturb_case = "大摂動 (±10 Å)"
    if large_perturb_case in results:
        errors = results[large_perturb_case]['all_errors']
        bp = ax2.boxplot([errors[:, i] for i in range(4)],
                         labels=interface_names, patch_artist=True)

        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(1, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(-1, color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Position Error (Å)')
        ax2.set_title(f'Error Distribution per Interface\n({large_perturb_case})')
        ax2.grid(True, alpha=0.3)

    # 3. 平均誤差 vs 初期値摂動
    ax3 = axes[1, 0]
    perturb_sizes = [3, 5, 10, 15, 20]  # 各ケースの最大摂動
    mean_errors = [np.mean(results[c]['mean_errors']) for c in case_names]

    ax3.plot(perturb_sizes, mean_errors, 'bo-', markersize=10, linewidth=2)
    ax3.fill_between(perturb_sizes, 0, 1, alpha=0.2, color='green', label='|error| < 1 Å')
    ax3.set_xlabel('Initial Value Perturbation (Å)')
    ax3.set_ylabel('Mean Absolute Error (Å)')
    ax3.set_title('Mean Error vs Initial Value Perturbation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(mean_errors) * 1.2)

    # 4. サマリーテキスト
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = f"""
    Robustness Test Summary
    =======================

    Test Configuration:
      True interface positions: {true_positions.tolist()} Å
      Fixed interface widths: {config.true_widths}
      Noise level: {config.noise_level * 100:.0f}%
      Number of trials: {config.n_trials}

    Results by Perturbation:
    """

    for case_name, res in results.items():
        mean = np.mean(res['mean_errors'])
        rate = res['convergence_rate']
        summary += f"\n      {case_name}:\n"
        summary += f"        Mean error: {mean:.2f} Å\n"
        summary += f"        Convergence (<1Å): {rate:.0f}%"

    summary += """

    (values from this run; convergence = |error| < 1 Å)
    """

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.tight_layout()
    output_path = Path('/tmp/initial_value_robustness.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to {output_path}")


def run_all_tests():
    """全てのロバスト性テストを実行"""
    config = RobustnessTestConfig(
        n_trials=50,
        noise_level=0.02,
    )

    results = run_robustness_test(config, verbose=True)

    print("\n" + "=" * 70)
    print("テスト完了！")
    print("=" * 70)

    # 最終評価
    large_perturb = "大摂動 (±10 Å)"
    if large_perturb in results:
        rate = results[large_perturb]['convergence_rate']
        mean_err = np.mean(results[large_perturb]['mean_errors'])

        print(f"\n【±10 Å 摂動】収束の割合 {rate:.0f}%、平均誤差 {mean_err:.2f} Å（この実行）")


if __name__ == "__main__":
    run_all_tests()
