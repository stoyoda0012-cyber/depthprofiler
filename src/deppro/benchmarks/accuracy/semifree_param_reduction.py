"""
SemiFreeSolver パラメータ数削減テスト
=====================================

パラメータ数を600→200→100に減らして、
LLS/MEM/L1/L2の4手法での膜厚推定精度を検証。

パラメータ数の削減方法:
- 600: depth=0-150, step=1.0 (150点 × 4 free elements)
- 200: depth=0-100, step=2.0 (50点 × 4 free elements)
- 100: depth=0-100, step=4.0 (25点 × 4 free elements)

膜厚推定: SemiFreeSolverのthicknessパラメータ（erf中心位置）

Date: 2026-01-24
"""

import numpy as np
from scipy.special import erf
from typing import List, Dict, Tuple
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig


def create_4layer_profile(
    depth: np.ndarray,
    thicknesses: List[float] = [5.0, 15.0, 35.0, 30.0],
    interface_widths: List[float] = [2.0, 3.0, 4.0, 5.0],
) -> np.ndarray:
    """
    4層膜プロファイル生成: C / Al2O3 / TiO2 / SiO2 / Si

    Elements: [C, Al, Ti, Si_ox, O, Si_bulk]
    """
    sigma = [w / (2 * np.sqrt(np.log(2))) for w in interface_widths]

    # 界面位置（累積）
    z = [0] + list(np.cumsum(thicknesses))

    D = np.zeros((len(depth), 6))

    # C層 (0 ~ z[1])
    c_profile = 0.5 * (1 - erf((depth - z[1]) / sigma[0]))
    D[:, 0] = c_profile

    # Al2O3層 (z[1] ~ z[2])
    al2o3_rise = 0.5 * (1 + erf((depth - z[1]) / sigma[0]))
    al2o3_fall = 0.5 * (1 - erf((depth - z[2]) / sigma[1]))
    al2o3_profile = al2o3_rise * al2o3_fall
    D[:, 1] = (2/5) * al2o3_profile  # Al
    D[:, 4] += (3/5) * al2o3_profile  # O from Al2O3

    # TiO2層 (z[2] ~ z[3])
    tio2_rise = 0.5 * (1 + erf((depth - z[2]) / sigma[1]))
    tio2_fall = 0.5 * (1 - erf((depth - z[3]) / sigma[2]))
    tio2_profile = tio2_rise * tio2_fall
    D[:, 2] = (1/3) * tio2_profile  # Ti
    D[:, 4] += (2/3) * tio2_profile  # O from TiO2

    # SiO2層 (z[3] ~ z[4])
    sio2_rise = 0.5 * (1 + erf((depth - z[3]) / sigma[2]))
    sio2_fall = 0.5 * (1 - erf((depth - z[4]) / sigma[3]))
    sio2_profile = sio2_rise * sio2_fall
    D[:, 3] = (1/3) * sio2_profile  # Si_ox
    D[:, 4] += (2/3) * sio2_profile  # O from SiO2

    # Si基板 (z[4] ~)
    si_profile = 0.5 * (1 + erf((depth - z[4]) / sigma[3]))
    D[:, 5] = si_profile

    # 正規化
    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    return D


def create_test_data_with_resolution(
    depth_end: float = 150.0,
    step_size: float = 1.0,
    thicknesses: List[float] = [5.0, 15.0, 35.0, 30.0],
    interface_widths: List[float] = [2.0, 3.0, 4.0, 5.0],
    angles: List[float] = None,
    noise_level: float = 0.02,
    seed: int = 42,
) -> Tuple[np.ndarray, TransferMatrix, List[ElementConfig], np.ndarray, np.ndarray]:
    """
    指定した深度分解能でテストデータを生成

    Returns:
        I_obs, tm, elements, D_true, depth
    """
    if angles is None:
        angles = [0, 10, 20, 30, 40, 50, 60]  # 0-60°範囲

    np.random.seed(seed)
    depth = np.arange(0, depth_end + step_size, step_size)

    elements = [
        ElementConfig('C', '1s', 'C', oxidation_state=0),
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_4layer_profile(depth, thicknesses, interface_widths)

    tm = TransferMatrix(depth=depth, step_size=step_size, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    if noise_level > 0:
        noise = noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true + noise * I_true
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    else:
        I_obs = I_true.copy()

    return I_obs, tm, elements, D_true, depth


def run_semifree_solver(
    I_obs: np.ndarray,
    tm: TransferMatrix,
    elements: List[ElementConfig],
    D_true: np.ndarray,
    method: str = 'LLS',
    alpha: float = 0.0,
    thickness_init: float = 70.0,
    max_iter: int = 500,
    verbose: bool = False,
) -> Dict:
    """SemiFreeSolverを実行"""

    n_depth = len(tm.depth)
    n_free = 4  # C, Al, Ti, Si_ox

    # Elements: [C, Al, Ti, Si_ox, O, Si_bulk]
    # Indices:   0   1    2    3     4    5
    semi_config = SemiFreeSolverConfig(
        thickness_init=thickness_init,
        abruptness_init=5.0,
        thickness_bounds=(10.0, 150.0),
        free_indices=[0, 1, 2, 3],      # C, Al, Ti, Si_ox
        linked_indices=[4],              # O
        substrate_indices=[5],           # Si_bulk
    )

    solver_config = SolverConfig(
        method=method,
        alpha=alpha,
        scipy_method='SLSQP',
        max_iter=max_iter,
    )

    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)

    start_time = time.time()
    try:
        result = solver.solve(I_obs, verbose=verbose)
        elapsed = time.time() - start_time

        if result.success:
            D_opt = result.D_opt
            rmse = np.sqrt(np.mean((D_opt - D_true)**2))

            # Si_bulkの50%点から推定（補間使用）
            si_bulk = D_opt[:, 5]
            depth = tm.depth

            # 線形補間で正確な50%点を求める
            for i in range(len(depth) - 1):
                if si_bulk[i] < 0.5 <= si_bulk[i+1]:
                    t = (0.5 - si_bulk[i]) / (si_bulk[i+1] - si_bulk[i])
                    estimated_total = depth[i] + t * (depth[i+1] - depth[i])
                    break
            else:
                # フォールバック: 最近点
                interface_idx = np.argmin(np.abs(si_bulk - 0.5))
                estimated_total = depth[interface_idx]

            return {
                'success': True,
                'thickness': result.thickness,  # erfパラメータ（直接最適化）
                'estimated_total': estimated_total,  # 50%点から推定
                'chi2': result.chi2,
                'rmse': rmse,
                'elapsed': elapsed,
                'n_iterations': result.n_iterations,
                'D_opt': D_opt,
                'n_params': n_depth * n_free + 2 + 1,  # free_conc + thickness + abruptness + substrate_ratio
            }
        else:
            return {
                'success': False,
                'error': 'Optimization failed',
                'elapsed': elapsed,
                'n_params': n_depth * n_free + 2 + 1,
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'error': str(e),
            'elapsed': elapsed,
            'n_params': n_depth * n_free + 2 + 1,
        }


def test_parameter_reduction():
    """パラメータ数削減テスト: 600→200→100"""
    print("=" * 80)
    print("SemiFreeSolver パラメータ数削減テスト")
    print("=" * 80)

    thicknesses = [5.0, 15.0, 35.0, 30.0]
    interface_widths = [2.0, 3.0, 4.0, 5.0]
    total_thickness = sum(thicknesses)

    print(f"\n真の膜厚: C={thicknesses[0]}, Al2O3={thicknesses[1]}, "
          f"TiO2={thicknesses[2]}, SiO2={thicknesses[3]} Å")
    print(f"真の合計膜厚: {total_thickness} Å")

    # パラメータ数設定
    # n_params = n_depth * n_free + 3 (thickness, abruptness, substrate_ratio)
    # n_free = 4 (C, Al, Ti, Si_ox)
    param_configs = [
        {'label': '~600', 'depth_end': 150.0, 'step_size': 1.0},   # 151 * 4 + 3 = 607
        {'label': '~200', 'depth_end': 100.0, 'step_size': 2.0},   # 51 * 4 + 3 = 207
        {'label': '~100', 'depth_end': 100.0, 'step_size': 4.0},   # 26 * 4 + 3 = 107
    ]

    methods = [
        ('LLS', 0.0),
        ('L2', 0.01),
        ('L1', 0.01),
        ('MEM', 0.01),
    ]

    n_trials = 10
    noise_level = 0.02

    # 結果格納
    all_results = {}

    for config in param_configs:
        label = config['label']
        print(f"\n{'='*60}")
        print(f"パラメータ数: {label}")
        print(f"  depth_end={config['depth_end']}, step_size={config['step_size']}")
        print(f"{'='*60}")

        all_results[label] = {}

        for method, alpha in methods:
            method_name = f"{method}"
            all_results[label][method_name] = {
                'thickness_errors': [],
                'estimated_errors': [],
                'rmses': [],
                'times': [],
                'successes': 0,
                'n_params': None,
            }

            for seed in range(n_trials):
                I_obs, tm, elements, D_true, depth = create_test_data_with_resolution(
                    depth_end=config['depth_end'],
                    step_size=config['step_size'],
                    thicknesses=thicknesses,
                    interface_widths=interface_widths,
                    noise_level=noise_level,
                    seed=42 + seed,
                )

                result = run_semifree_solver(
                    I_obs, tm, elements, D_true,
                    method=method,
                    alpha=alpha,
                    thickness_init=70.0,
                    max_iter=500,
                )

                if all_results[label][method_name]['n_params'] is None:
                    all_results[label][method_name]['n_params'] = result.get('n_params', 0)

                if result['success']:
                    all_results[label][method_name]['successes'] += 1
                    # erfパラメータからの誤差
                    thickness_error = result['thickness'] - total_thickness
                    all_results[label][method_name]['thickness_errors'].append(thickness_error)
                    # 50%点からの誤差
                    estimated_error = result['estimated_total'] - total_thickness
                    all_results[label][method_name]['estimated_errors'].append(estimated_error)
                    all_results[label][method_name]['rmses'].append(result['rmse'])
                    all_results[label][method_name]['times'].append(result['elapsed'])

            # 結果表示
            data = all_results[label][method_name]
            success_rate = data['successes'] / n_trials * 100
            if data['thickness_errors']:
                t_mean = np.mean(data['thickness_errors'])
                t_std = np.std(data['thickness_errors'])
                e_mean = np.mean(data['estimated_errors'])
                e_std = np.std(data['estimated_errors'])
                mean_time = np.mean(data['times']) * 1000
                print(f"  {method_name:<5}: 成功率={success_rate:3.0f}%, "
                      f"erf誤差={t_mean:+6.2f}±{t_std:5.2f}Å, "
                      f"50%誤差={e_mean:+6.2f}±{e_std:5.2f}Å, "
                      f"時間={mean_time:6.0f}ms")
            else:
                print(f"  {method_name:<5}: 成功率={success_rate:3.0f}%")

    # サマリーテーブル
    print("\n" + "=" * 80)
    print("サマリー: erfパラメータからの膜厚誤差 (Å)")
    print("=" * 80)
    print(f"{'パラメータ':<12} {'LLS':<20} {'L2':<20} {'L1':<20} {'MEM':<20}")
    print("-" * 92)

    for config in param_configs:
        label = config['label']
        row = f"{label:<12}"
        for method, _ in methods:
            method_name = f"{method}"
            data = all_results[label][method_name]
            if data['thickness_errors']:
                mean = np.mean(data['thickness_errors'])
                std = np.std(data['thickness_errors'])
                row += f"{mean:+6.2f}±{std:5.2f}       "
            else:
                row += f"{'FAIL':<20}"
        print(row)

    print("\n" + "=" * 80)
    print("サマリー: 50%点からの膜厚誤差 (Å)")
    print("=" * 80)
    print(f"{'パラメータ':<12} {'LLS':<20} {'L2':<20} {'L1':<20} {'MEM':<20}")
    print("-" * 92)

    for config in param_configs:
        label = config['label']
        row = f"{label:<12}"
        for method, _ in methods:
            method_name = f"{method}"
            data = all_results[label][method_name]
            if data['estimated_errors']:
                mean = np.mean(data['estimated_errors'])
                std = np.std(data['estimated_errors'])
                row += f"{mean:+6.2f}±{std:5.2f}       "
            else:
                row += f"{'FAIL':<20}"
        print(row)

    # プロット作成
    create_summary_plot(all_results, param_configs, methods, total_thickness)

    return all_results


def create_summary_plot(all_results, param_configs, methods, total_thickness):
    """サマリープロット作成"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    param_labels = [c['label'] for c in param_configs]
    method_names = [m[0] for m in methods]
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']

    # (0,0) erfパラメータからの膜厚誤差
    ax = axes[0, 0]
    x = np.arange(len(param_labels))
    width = 0.2

    for i, method in enumerate(method_names):
        means = []
        stds = []
        for label in param_labels:
            data = all_results[label][method]
            if data['thickness_errors']:
                means.append(np.mean(data['thickness_errors']))
                stds.append(np.std(data['thickness_errors']))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=method,
                      color=colors[i], capsize=3)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('Thickness Error (Å)')
    ax.set_title('erf Parameter Error')
    ax.set_xticks(x)
    ax.set_xticklabels(param_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # (0,1) 50%点からの膜厚誤差
    ax = axes[0, 1]

    for i, method in enumerate(method_names):
        means = []
        stds = []
        for label in param_labels:
            data = all_results[label][method]
            if data['estimated_errors']:
                means.append(np.mean(data['estimated_errors']))
                stds.append(np.std(data['estimated_errors']))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=method,
                      color=colors[i], capsize=3)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('Thickness Error (Å)')
    ax.set_title('50% Point Error')
    ax.set_xticks(x)
    ax.set_xticklabels(param_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # (1,0) 成功率
    ax = axes[1, 0]

    for i, method in enumerate(method_names):
        success_rates = []
        for label in param_labels:
            data = all_results[label][method]
            success_rates.append(data['successes'] / 10 * 100)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, success_rates, width, label=method,
                      color=colors[i])

    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate (10 trials)')
    ax.set_xticks(x)
    ax.set_xticklabels(param_labels)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # (1,1) 計算時間
    ax = axes[1, 1]

    for i, method in enumerate(method_names):
        mean_times = []
        for label in param_labels:
            data = all_results[label][method]
            if data['times']:
                mean_times.append(np.mean(data['times']) * 1000)
            else:
                mean_times.append(np.nan)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, mean_times, width, label=method,
                      color=colors[i])

    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('Mean Time (ms)')
    ax.set_title('Computation Time')
    ax.set_xticks(x)
    ax.set_xticklabels(param_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'SemiFreeSolver Parameter Reduction Test\n'
                 f'4-Layer: C/Al2O3/TiO2/SiO2/Si, True thickness: {total_thickness}Å, Noise: 2%',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('semifree_param_reduction.png', dpi=150)
    print("\n→ semifree_param_reduction.png を保存しました")


def test_profile_comparison_by_params():
    """パラメータ数による深度プロファイル比較"""
    print("\n" + "=" * 80)
    print("パラメータ数による深度プロファイル比較")
    print("=" * 80)

    thicknesses = [5.0, 15.0, 35.0, 30.0]
    interface_widths = [2.0, 3.0, 4.0, 5.0]
    total_thickness = sum(thicknesses)

    param_configs = [
        {'label': '~600', 'depth_end': 150.0, 'step_size': 1.0},
        {'label': '~200', 'depth_end': 100.0, 'step_size': 2.0},
        {'label': '~100', 'depth_end': 100.0, 'step_size': 4.0},
    ]

    element_names = ['C', 'Al', 'Ti', 'Si(ox)', 'O', 'Si(bulk)']
    colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']

    # MEM法で比較
    method = 'MEM'
    alpha = 0.01

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, config in enumerate(param_configs):
        ax = axes[idx]
        label = config['label']

        I_obs, tm, elements, D_true, depth = create_test_data_with_resolution(
            depth_end=config['depth_end'],
            step_size=config['step_size'],
            thicknesses=thicknesses,
            interface_widths=interface_widths,
            noise_level=0.02,
            seed=42,
        )

        result = run_semifree_solver(
            I_obs, tm, elements, D_true,
            method=method,
            alpha=alpha,
            max_iter=500,
        )

        # 真のプロファイル（点線）- 高解像度で生成
        depth_hires = np.arange(0, 120, 0.5)
        D_true_hires = create_4layer_profile(depth_hires, thicknesses, interface_widths)

        for j, (name, color) in enumerate(zip(element_names, colors)):
            ax.plot(depth_hires, D_true_hires[:, j], '--', color=color, alpha=0.5, linewidth=1)

        if result['success']:
            D_opt = result['D_opt']
            # 推定プロファイル（実線）
            for j, (name, color) in enumerate(zip(element_names, colors)):
                ax.plot(depth, D_opt[:, j], 'o-', color=color, linewidth=1.5,
                       markersize=3, label=name)

            t_error = result['thickness'] - total_thickness
            e_error = result['estimated_total'] - total_thickness
            n_params = result['n_params']
            title = (f"{label} params (n={n_params})\n"
                    f"erf誤差: {t_error:+.2f}Å, 50%誤差: {e_error:+.2f}Å")
        else:
            title = f"{label} params\nFAILED"

        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Depth (Å)')
        ax.set_ylabel('Composition')
        ax.set_title(title)
        ax.axvline(x=total_thickness, color='gray', linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(f'Profile Comparison by Parameter Count (MEM, Noise: 2%)\n'
                 f'True thickness: {total_thickness}Å', fontsize=12)
    plt.tight_layout()
    plt.savefig('semifree_profile_by_params.png', dpi=150)
    print("\n→ semifree_profile_by_params.png を保存しました")


def main():
    """メイン"""
    # パラメータ数削減テスト
    all_results = test_parameter_reduction()

    # プロファイル比較
    test_profile_comparison_by_params()

    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    main()
