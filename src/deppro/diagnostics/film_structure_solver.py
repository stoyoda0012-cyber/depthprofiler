#!/usr/bin/env python
"""
FilmStructureSolver (MATLAB-style semi-free model) テスト

MATLABのRun_DepthProfiling.mと同じ設計:
- 基板: erf関数でモデル化 (thickness, abruptness パラメータ)
- free元素: 各深さで自由に最適化
- linked元素: 電荷保存則から自動計算
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/src

from deppro import (
    TransferMatrix, SolverConfig, ElementConfig,
    InverseSolver
)
from deppro.inverse_solver import FilmStructureSolver, FilmStructureConfig


def create_sio2_si_profile(depth, oxide_thickness=30.0, interface_width=5.0):
    """
    SiO2/Si 2層モデルを生成

    Elements:
    - 0: Si_ox (Si4+ in SiO2)
    - 1: O (O2- in SiO2)
    - 2: Si_bulk (Si0 substrate)
    """
    n_depth = len(depth)
    D = np.zeros((n_depth, 3))

    # Substrate profile (erf function)
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    erf_profile = 0.5 * (1 + erf((depth - oxide_thickness) / sigma))

    # Si_bulk (substrate)
    D[:, 2] = erf_profile

    # Si_ox (film) - fills remaining space
    # In SiO2: Si4+ stoichiometry
    film_fraction = 1.0 - erf_profile
    D[:, 0] = film_fraction / 3.0  # Si:O = 1:2 in SiO2

    # O (linked by charge balance)
    D[:, 1] = 2.0 * D[:, 0]  # O = 2 * Si_ox

    return D


def run_film_structure_test(
    noise_level: float = 0.0,
    alpha: float = 0.01,
    method: str = 'L2',
    show_plot: bool = True,
):
    """FilmStructureSolverのラウンドトリップテスト"""

    print("=" * 70)
    print("FilmStructureSolver (MATLAB-style) テスト")
    print("=" * 70)

    # 1. 真のプロファイルを生成
    print("\n1. 真のプロファイルを生成...")

    depth = np.linspace(0, 100, 101)
    n_depth = len(depth)
    step_size = depth[1] - depth[0]

    true_thickness = 30.0
    true_abruptness = 5.0

    D_true = create_sio2_si_profile(depth, true_thickness, true_abruptness)

    elements = [
        ElementConfig(symbol='Si', orbital='2p', compound='SiO2',
                     binding_energy=103.5, oxidation_state=4),
        ElementConfig(symbol='O', orbital='1s', compound='SiO2',
                     binding_energy=531.0, oxidation_state=-2),
        ElementConfig(symbol='Si', orbital='2p', compound='Si',
                     binding_energy=99.3, oxidation_state=0),
    ]
    n_elements = len(elements)

    print(f"  元素: Si_ox(+4), O(-2), Si_bulk(0)")
    print(f"  真の厚さ: {true_thickness:.1f} Å")
    print(f"  真の界面幅: {true_abruptness:.1f} Å")

    # 2. 順問題: 角度分解強度を計算
    print("\n2. 順問題: 角度分解強度を計算...")

    angles = np.array([15, 30, 45, 60, 75, 85])

    tm = TransferMatrix(
        depth=depth,
        step_size=step_size,
        angles=angles,
        photon_energy=1486.6,
    )
    tm.compute(D_true, elements)
    I_obs = tm.forward(D_true, normalize=True)

    # ノイズを加える
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, I_obs.shape)
        I_obs = I_obs * (1 + noise)
        I_obs = np.clip(I_obs, 0, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    print(f"  角度: {angles}")
    print(f"  ノイズ: {noise_level*100:.1f}%")

    # 3. 逆問題: FilmStructureSolver
    print("\n3. 逆問題: FilmStructureSolver...")

    # Film structure configuration
    film_config = FilmStructureConfig(
        thickness_init=25.0,       # 初期推定 (真値から少しずらす)
        abruptness_init=8.0,       # 初期推定
        thickness_bounds=(5.0, 80.0),
        abruptness_bounds=(1.0, 20.0),
        free_indices=[0],          # Si_ox (自由に最適化)
        linked_indices=[1],        # O (電荷保存から計算)
        substrate_indices=[2],     # Si_bulk (erf関数)
        optimize_thickness=True,
        optimize_abruptness=True,
    )

    solver_config = SolverConfig(
        method=method,
        alpha=alpha,
        max_iter=200,
        tol=1e-8,
    )

    # 初期推定
    D_prior = create_sio2_si_profile(depth, film_config.thickness_init, film_config.abruptness_init)

    solver = FilmStructureSolver(
        tm, solver_config, film_config, D_prior, elements
    )

    start_time = time.time()
    result = solver.solve(I_obs, verbose=True)
    elapsed = time.time() - start_time

    D_recon = result.D_opt

    print(f"\n  収束: {result.success}")
    print(f"  推定厚さ: {result.thickness:.2f} Å (真値: {true_thickness:.1f})")
    print(f"  推定界面幅: {result.abruptness:.2f} Å (真値: {true_abruptness:.1f})")
    print(f"  厚さ誤差: {abs(result.thickness - true_thickness):.2f} Å")
    print(f"  時間: {elapsed:.3f}秒")

    # 4. 精度評価
    print("\n4. 精度評価:")

    mse = np.mean((D_recon - D_true)**2)
    rmse = np.sqrt(mse)

    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.4f}")

    # 元素ごとのRMSE
    labels = ['Si_ox', 'O', 'Si_bulk']
    print("\n  元素別RMSE:")
    for j, label in enumerate(labels):
        rmse_elem = np.sqrt(np.mean((D_recon[:, j] - D_true[:, j])**2))
        print(f"    {label:>10}: {rmse_elem:.4f}")

    # 5. 比較: 純粋なフリーモデル (InverseSolver)
    print("\n5. 比較: 純粋なフリーモデル (InverseSolver)...")

    free_solver = InverseSolver(tm, solver_config, D_prior, elements)
    free_result = free_solver.solve(I_obs)
    D_free = free_result.D_opt

    free_rmse = np.sqrt(np.mean((D_free - D_true)**2))
    print(f"  フリーモデル RMSE: {free_rmse:.4f}")
    print(f"  セミフリー  RMSE: {rmse:.4f}")
    print(f"  改善率: {(free_rmse - rmse) / free_rmse * 100:.1f}%")

    # 6. 可視化
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        colors = ['tab:blue', 'tab:orange', 'tab:green']

        # Depth profiles comparison
        ax = axes[0, 0]
        for j, label in enumerate(labels):
            ax.plot(depth, D_true[:, j], '-', color=colors[j],
                   label=f'{label} (true)', linewidth=2)
            ax.plot(depth, D_recon[:, j], '--', color=colors[j],
                   label=f'{label} (recon)', linewidth=1.5, alpha=0.8)
        ax.axvline(x=true_thickness, color='gray', linestyle=':', alpha=0.5, label='true thickness')
        ax.axvline(x=result.thickness, color='red', linestyle=':', alpha=0.5, label='recon thickness')
        ax.set_xlabel('Depth (Å)')
        ax.set_ylabel('Concentration')
        ax.set_title('FilmStructureSolver: Depth Profile')
        ax.legend(fontsize=8)
        ax.set_xlim([0, depth[-1]])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        # Free model comparison
        ax = axes[0, 1]
        for j, label in enumerate(labels):
            ax.plot(depth, D_true[:, j], '-', color=colors[j],
                   label=f'{label} (true)', linewidth=2)
            ax.plot(depth, D_free[:, j], '--', color=colors[j],
                   label=f'{label} (free)', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Depth (Å)')
        ax.set_ylabel('Concentration')
        ax.set_title(f'InverseSolver (Free Model): RMSE={free_rmse:.4f}')
        ax.legend(fontsize=8)
        ax.set_xlim([0, depth[-1]])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        # Error comparison
        ax = axes[1, 0]
        err_semi = np.mean((D_recon - D_true)**2, axis=1)
        err_free = np.mean((D_free - D_true)**2, axis=1)
        ax.fill_between(depth, 0, err_semi, alpha=0.5, label=f'Semi-free (RMSE={rmse:.4f})')
        ax.fill_between(depth, 0, err_free, alpha=0.3, label=f'Free (RMSE={free_rmse:.4f})')
        ax.set_xlabel('Depth (Å)')
        ax.set_ylabel('MSE')
        ax.set_title('MSE vs Depth')
        ax.legend()
        ax.set_xlim([0, depth[-1]])
        ax.grid(True, alpha=0.3)

        # Intensity comparison
        ax = axes[1, 1]
        width = 0.25
        x = np.arange(len(angles))

        # Observed
        for j in range(n_elements):
            offset = (j - 1) * width
            ax.bar(x + offset, I_obs[:, j], width, color=colors[j], alpha=0.7, label=labels[j] if j == 0 else None)

        # Reconstructed (semi-free)
        tm.compute(D_recon, elements)
        I_recon = tm.forward(D_recon, normalize=True)
        for j in range(n_elements):
            offset = (j - 1) * width
            ax.scatter(x + offset, I_recon[:, j], marker='x', color='black', s=50, zorder=5)

        ax.set_xlabel('Angle')
        ax.set_ylabel('Intensity')
        ax.set_title('Angle-resolved Intensity (bars=obs, x=recon)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{a}°' for a in angles])
        ax.legend(labels, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'SiO2/Si Model: Semi-free vs Free ({method}, α={alpha})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('film_structure_solver_result.png', dpi=150, bbox_inches='tight')
        print("\n図を film_structure_solver_result.png に保存しました")
        plt.show()

    return {
        'rmse': rmse,
        'free_rmse': free_rmse,
        'thickness_error': abs(result.thickness - true_thickness),
        'abruptness_error': abs(result.abruptness - true_abruptness),
    }


def parameter_comparison():
    """異なるパラメータでの比較"""
    print("\n" + "=" * 70)
    print("パラメータ比較")
    print("=" * 70)

    methods = ['L2', 'MEM', 'L1']
    alphas = [0.001, 0.01, 0.1]

    results = []

    for method in methods:
        for alpha in alphas:
            print(f"\n--- {method}, α={alpha} ---")
            try:
                r = run_film_structure_test(
                    noise_level=0.0,
                    method=method,
                    alpha=alpha,
                    show_plot=False,
                )
                results.append({
                    'method': method,
                    'alpha': alpha,
                    **r
                })
            except Exception as e:
                print(f"  エラー: {e}")

    print("\n" + "=" * 70)
    print("結果まとめ")
    print("=" * 70)
    print(f"{'Method':<8} {'Alpha':>8} {'Semi-free RMSE':>15} {'Free RMSE':>12} {'厚さ誤差(Å)':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['method']:<8} {r['alpha']:>8.3f} {r['rmse']:>15.4f} {r['free_rmse']:>12.4f} {r['thickness_error']:>12.2f}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        parameter_comparison()
    else:
        run_film_structure_test(
            noise_level=0.0,
            method='L2',
            alpha=0.01,
            show_plot=True,
        )
