#!/usr/bin/env python
"""
5層モデル (C/Al2O3/TiO2/SiO2/Si) フリーモデル ラウンドトリップ検証

MultiLayerFilmSolver（構造パラメータ最適化）ではなく、
通常のInverseSolver（各深さで濃度を自由に最適化）を使用
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

from deppro import (
    TransferMatrix, InverseSolver, SolverConfig, ElementConfig,
    IMFP, CompoundDatabase
)
from deppro.depth_profile import create_multilayer_profile, DepthProfile


def run_5layer_free_roundtrip(
    noise_level: float = 0.0,
    alpha: float = 0.001,
    max_iter: int = 200,
    use_good_prior: bool = True,
    show_plot: bool = True,
):
    """
    5層モデルのフリーモデル ラウンドトリップテスト

    Args:
        noise_level: ノイズレベル (0-1)
        alpha: 正則化パラメータ
        max_iter: 最大反復回数
        use_good_prior: 良い初期推定を使うか
        show_plot: プロットを表示するか
    """
    print("=" * 70)
    print("5層モデル (C/Al2O3/TiO2/SiO2/Si) フリーモデル ラウンドトリップ")
    print("=" * 70)

    # 1. 真のプロファイルを生成
    print("\n1. 真のプロファイルを生成...")

    # 層厚: C(5Å), Al2O3(10Å), TiO2(20Å), SiO2(30Å), Si(基板)
    # 累積: 5, 15, 35, 65
    true_thicknesses = [5.0, 15.0, 35.0, 65.0]
    true_widths = [2.0, 3.0, 4.0, 5.0]

    profile = create_multilayer_profile(
        layer_thicknesses=true_thicknesses,
        interface_widths=true_widths,
        depth_end=100.0,
        step_size=1.0,
    )

    depth = profile.depth
    n_depth = profile.n_depth
    elements = profile.elements
    n_elements = profile.n_elements
    D_true = profile.concentrations

    print(f"  元素: {profile.element_labels}")
    print(f"  深さ: 0-{depth[-1]:.0f} Å ({n_depth}点)")
    print(f"  層構造: C(0-5) / Al2O3(5-15) / TiO2(15-35) / SiO2(35-65) / Si(65+)")

    # 2. 順問題: 角度分解強度を計算
    print("\n2. 順問題: 角度分解強度を計算...")

    angles = np.array([0, 15, 30, 45, 60, 75])

    tm = TransferMatrix(
        depth=depth,
        step_size=profile.step_size,
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

    # 強度表示
    print(f"\n  {'Angle':>6} | " + " | ".join(f"{l:>7}" for l in profile.element_labels))
    print("  " + "-" * (10 + 10 * n_elements))
    for i, theta in enumerate(angles):
        vals = " | ".join(f"{v:>7.4f}" for v in I_obs[i])
        print(f"  {theta:>6.0f} | {vals}")

    # 3. 逆問題: フリーモデルで再構成
    print("\n3. 逆問題: フリーモデルで再構成...")

    solver_config = SolverConfig(
        method='L2',
        alpha=alpha,
        max_iter=max_iter,
        tol=1e-8,
    )

    # 初期推定
    if use_good_prior:
        # 5層構造を仮定した初期推定
        D_prior = np.ones((n_depth, n_elements)) * 0.01

        for i, z in enumerate(depth):
            if z < 8:        # C layer
                D_prior[i] = [0.6, 0.05, 0.05, 0.05, 0.2, 0.05]
            elif z < 20:     # Al2O3 layer
                D_prior[i] = [0.05, 0.35, 0.05, 0.05, 0.45, 0.05]
            elif z < 40:     # TiO2 layer
                D_prior[i] = [0.02, 0.05, 0.30, 0.05, 0.53, 0.05]
            elif z < 70:     # SiO2 layer
                D_prior[i] = [0.01, 0.02, 0.05, 0.25, 0.55, 0.12]
            else:            # Si substrate
                D_prior[i] = [0.01, 0.01, 0.01, 0.01, 0.05, 0.91]

        D_prior = D_prior / D_prior.sum(axis=1, keepdims=True)
    else:
        # 一様分布
        D_prior = np.ones((n_depth, n_elements)) / n_elements

    solver = InverseSolver(tm, solver_config, D_prior, elements)

    start_time = time.time()
    result = solver.solve(I_obs)
    elapsed = time.time() - start_time

    D_recon = result.D_opt

    print(f"  収束: {result.success}, 反復: {result.n_iterations}")
    print(f"  Chi²: {result.chi2:.6f}")
    print(f"  時間: {elapsed:.3f}秒")

    # 4. 精度評価
    print("\n4. 精度評価:")

    mse = np.mean((D_recon - D_true)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(D_recon - D_true))

    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")

    # 元素ごとのRMSE
    print("\n  元素別RMSE:")
    for j, label in enumerate(profile.element_labels):
        rmse_elem = np.sqrt(np.mean((D_recon[:, j] - D_true[:, j])**2))
        print(f"    {label:>6}: {rmse_elem:.4f}")

    # 再構成強度との比較
    tm.compute(D_recon, elements)
    I_recon = tm.forward(D_recon, normalize=True)
    I_residual = np.sqrt(np.mean((I_recon - I_obs)**2))
    print(f"\n  強度残差 RMSE: {I_residual:.6f}")

    # 5. 可視化
    if show_plot:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig)

        colors = plt.cm.tab10(np.arange(n_elements))
        elem_labels = profile.element_labels

        # 深さプロファイル比較
        ax1 = fig.add_subplot(gs[0, 0])
        for j in range(n_elements):
            ax1.plot(depth, D_true[:, j], '-', color=colors[j],
                    label=f'{elem_labels[j]} (true)', linewidth=2)
            ax1.plot(depth, D_recon[:, j], '--', color=colors[j],
                    label=f'{elem_labels[j]} (recon)', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Depth (Å)')
        ax1.set_ylabel('Concentration')
        ax1.set_title('Depth Profile: True vs Reconstructed')
        ax1.legend(fontsize=7, ncol=2, loc='right')
        ax1.set_xlim([0, depth[-1]])
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)

        # 主要元素のみ表示
        ax2 = fig.add_subplot(gs[0, 1])
        main_idx = [0, 1, 2, 3, 5]  # C, Al, Ti, Si-ox, Si-bulk
        for j in main_idx:
            ax2.plot(depth, D_true[:, j], '-', color=colors[j],
                    label=f'{elem_labels[j]} (true)', linewidth=2)
            ax2.plot(depth, D_recon[:, j], '--', color=colors[j],
                    label=f'{elem_labels[j]} (recon)', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Depth (Å)')
        ax2.set_ylabel('Concentration')
        ax2.set_title('Cation Profiles')
        ax2.legend(fontsize=8)
        ax2.set_xlim([0, depth[-1]])
        ax2.set_ylim([0, 0.7])
        ax2.grid(True, alpha=0.3)

        # 誤差プロット
        ax3 = fig.add_subplot(gs[0, 2])
        diff = D_recon - D_true
        mse_per_depth = np.mean(diff**2, axis=1)
        ax3.fill_between(depth, 0, mse_per_depth, alpha=0.5)
        ax3.plot(depth, mse_per_depth, 'b-', linewidth=1.5)
        ax3.set_xlabel('Depth (Å)')
        ax3.set_ylabel('MSE')
        ax3.set_title(f'MSE vs Depth (Total RMSE: {rmse:.4f})')
        ax3.set_xlim([0, depth[-1]])
        ax3.grid(True, alpha=0.3)

        # 角度分解強度比較
        ax4 = fig.add_subplot(gs[1, 0])
        width = 0.12
        x = np.arange(len(angles))
        for j in range(n_elements):
            offset = (j - n_elements/2 + 0.5) * width
            ax4.bar(x + offset, I_obs[:, j], width, color=colors[j],
                   label=elem_labels[j], alpha=0.7)
            ax4.scatter(x + offset, I_recon[:, j], marker='x', color='k', s=30, zorder=5)
        ax4.set_xlabel('Angle')
        ax4.set_ylabel('Intensity')
        ax4.set_title('Angle-resolved Intensity (bars=obs, x=recon)')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{a}°' for a in angles])
        ax4.legend(fontsize=7, ncol=3)
        ax4.grid(True, alpha=0.3, axis='y')

        # 強度残差
        ax5 = fig.add_subplot(gs[1, 1])
        I_diff = I_recon - I_obs
        for j in range(n_elements):
            ax5.plot(angles, I_diff[:, j], 'o-', color=colors[j], label=elem_labels[j])
        ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax5.set_xlabel('Angle (°)')
        ax5.set_ylabel('Intensity Difference')
        ax5.set_title('Intensity Residual')
        ax5.legend(fontsize=7, ncol=2)
        ax5.grid(True, alpha=0.3)

        # メトリクス
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        metrics_text = f"""
    === 5-Layer Free Model Results ===

    Settings:
      α = {alpha}
      noise = {noise_level*100:.1f}%
      good prior = {use_good_prior}

    Overall Accuracy:
      MSE:  {mse:.6f}
      RMSE: {rmse:.4f}
      MAE:  {mae:.4f}

    Per Element RMSE:
"""
        for j, label in enumerate(elem_labels):
            rmse_j = np.sqrt(np.mean((D_recon[:, j] - D_true[:, j])**2))
            metrics_text += f"      {label:>6}: {rmse_j:.4f}\n"

        metrics_text += f"""
    Solver:
      Chi²: {result.chi2:.6f}
      Iter: {result.n_iterations}
      Time: {elapsed:.3f}s
        """

        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        title = f"5-Layer Free Model: α={alpha}, noise={noise_level*100:.0f}%"
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('5layer_free_result.png', dpi=150, bbox_inches='tight')
        print("\n図を 5layer_free_result.png に保存しました")
        plt.show()

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'chi2': result.chi2,
        'n_iter': result.n_iterations,
        'I_residual': I_residual,
        'D_true': D_true,
        'D_recon': D_recon,
    }


def compare_prior_effect():
    """初期推定の効果を比較"""
    print("\n" + "=" * 70)
    print("初期推定の効果比較")
    print("=" * 70)

    results = []

    for use_good_prior in [False, True]:
        label = "良い初期推定" if use_good_prior else "一様分布"
        print(f"\n--- {label} ---")
        r = run_5layer_free_roundtrip(
            noise_level=0.0,
            alpha=0.001,
            use_good_prior=use_good_prior,
            show_plot=False,
        )
        results.append({
            'prior': label,
            'rmse': r['rmse'],
            'chi2': r['chi2'],
        })

    print("\n" + "=" * 50)
    print("比較結果")
    print("=" * 50)
    print(f"{'初期推定':<15} {'RMSE':>10} {'Chi²':>12}")
    print("-" * 40)
    for r in results:
        print(f"{r['prior']:<15} {r['rmse']:>10.4f} {r['chi2']:>12.6f}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare_prior_effect()
    else:
        # デフォルト: 良い初期推定でテスト
        run_5layer_free_roundtrip(
            noise_level=0.0,
            alpha=0.001,
            use_good_prior=True,
            show_plot=True,
        )
