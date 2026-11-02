#!/usr/bin/env python
"""
4層モデル ラウンドトリップ検証
depth profile → spectra → depth profile

目標: 元のプロファイルを高精度で再構成

MATLAB互換H5出力:
    save_to_h5()でMATLABのDepthProfilerが読めるH5ファイルを生成
    - /Drlt: 結果プロファイル (depth, elements, stack)
    - /Dstk: 初期プロファイル
    - /Dtrn: モデルプロファイル
    - /Qrlt: 目的関数値 (stack, 3) [chi2, reg, objective]
    - /depth, /element, /angle: 軸情報
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import h5py

from deppro import (
    TransferMatrix, InverseSolver, SolverConfig, ElementConfig,
    BatchInverseSolver, IMFP
)


def create_4layer_profile(depth: np.ndarray, params: dict = None) -> np.ndarray:
    """
    4層モデルのdepth profileを生成

    Layer 1: C contamination (表面)
    Layer 2: SiO2 (酸化膜)
    Layer 3: SiOx (遷移層)
    Layer 4: Si (基板)

    Args:
        depth: 深さ配列 (Å)
        params: 層パラメータ
            - d_C: C層厚さ (Å)
            - d_SiO2: SiO2層終端深さ (Å)
            - d_SiOx: SiOx層終端深さ (Å)
            - w_*: 界面幅 (Å)

    Returns:
        profile: (n_depth, 3) 配列 [C, O, Si]
    """
    if params is None:
        params = {
            'd_C': 15.0,      # C層: 0-15Å (~1.5nm)
            'd_SiO2': 45.0,   # SiO2層: 15-45Å (~3nm)
            'd_SiOx': 70.0,   # SiOx層: 45-70Å (~2.5nm)
            'w_C': 5.0,       # C/SiO2界面幅
            'w_SiO2': 8.0,    # SiO2/SiOx界面幅
            'w_SiOx': 10.0,   # SiOx/Si界面幅
        }

    n_depth = len(depth)
    profile = np.zeros((n_depth, 3))  # [C, O, Si]

    d_C = params['d_C']
    d_SiO2 = params['d_SiO2']
    d_SiOx = params['d_SiOx']
    w_C = params['w_C']
    w_SiO2 = params['w_SiO2']
    w_SiOx = params['w_SiOx']

    for i, z in enumerate(depth):
        # シグモイド関数で滑らかな遷移を表現
        def sigmoid(x, x0, w):
            return 1 / (1 + np.exp(-(x - x0) / max(w, 0.1)))

        # 各層の寄与を計算
        # C層: 表面から減衰
        f_C = 1 - sigmoid(z, d_C, w_C)

        # SiO2層: d_C から d_SiO2 まで
        f_SiO2 = sigmoid(z, d_C, w_C) * (1 - sigmoid(z, d_SiO2, w_SiO2))

        # SiOx遷移層: d_SiO2 から d_SiOx まで
        f_SiOx = sigmoid(z, d_SiO2, w_SiO2) * (1 - sigmoid(z, d_SiOx, w_SiOx))

        # Si基板: d_SiOx 以深
        f_Si = sigmoid(z, d_SiOx, w_SiOx)

        # 組成を設定
        # C層: 主にC
        c_C = 0.85 * f_C
        c_O = 0.05 * f_C
        c_Si = 0.10 * f_C

        # SiO2層: SiO2化学量論
        c_C += 0.02 * f_SiO2
        c_O += 0.66 * f_SiO2   # O: 66.7%
        c_Si += 0.32 * f_SiO2  # Si: 33.3%

        # SiOx遷移層: 酸素減少
        c_C += 0.01 * f_SiOx
        c_O += 0.30 * f_SiOx   # O減少
        c_Si += 0.69 * f_SiOx  # Si増加

        # Si基板
        c_C += 0.00 * f_Si
        c_O += 0.00 * f_Si
        c_Si += 1.00 * f_Si

        # 正規化
        total = c_C + c_O + c_Si
        profile[i] = [c_C / total, c_O / total, c_Si / total]

    return profile


def forward_model(profile: np.ndarray, tm: TransferMatrix,
                  elements: list, noise_level: float = 0.0) -> np.ndarray:
    """
    順問題: depth profile → 角度分解強度

    Args:
        profile: (n_depth, n_elements) 深さプロファイル
        tm: TransferMatrix
        elements: ElementConfig リスト
        noise_level: ノイズレベル (0-1)

    Returns:
        I_obs: (n_angles, n_elements) 角度分解強度
    """
    tm.compute(profile, elements)
    I_obs = tm.forward(profile, normalize=True)

    if noise_level > 0:
        # ポアソンノイズを模擬
        noise = np.random.normal(0, noise_level, I_obs.shape)
        I_obs = I_obs * (1 + noise)
        I_obs = np.clip(I_obs, 0, None)
        # 再正規化
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    return I_obs


def inverse_model(I_obs: np.ndarray, tm: TransferMatrix,
                  elements: list, config: SolverConfig,
                  D_prior: np.ndarray = None) -> np.ndarray:
    """
    逆問題: 角度分解強度 → depth profile

    Args:
        I_obs: (n_angles, n_elements) 観測強度
        tm: TransferMatrix
        elements: ElementConfig リスト
        config: SolverConfig
        D_prior: 事前分布

    Returns:
        D_opt: (n_depth, n_elements) 再構成プロファイル
    """
    n_depth = len(tm.depth)
    n_elements = len(elements)

    if D_prior is None:
        D_prior = np.ones((n_depth, n_elements)) / n_elements

    solver = InverseSolver(tm, config, D_prior, elements)
    result = solver.solve(I_obs)

    return result.D_opt, result


def plot_roundtrip(depth, profile_true, profile_recon, I_obs, I_recon,
                   angles, elements, title="Roundtrip Verification"):
    """ラウンドトリップ結果を可視化"""

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # C, O, Si
    elem_names = ['C', 'O', 'Si']

    # 1. 深さプロファイル比較 (上段左)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (name, color) in enumerate(zip(elem_names, colors)):
        ax1.plot(depth, profile_true[:, i], '-', color=color,
                label=f'{name} (true)', linewidth=2)
        ax1.plot(depth, profile_recon[:, i], '--', color=color,
                label=f'{name} (recon)', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Depth (Å)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('Depth Profile: True vs Reconstructed')
    ax1.legend(loc='right', fontsize=8)
    ax1.set_xlim([0, depth[-1]])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # 2. 差分プロット (上段中央)
    ax2 = fig.add_subplot(gs[0, 1])
    diff = profile_recon - profile_true
    for i, (name, color) in enumerate(zip(elem_names, colors)):
        ax2.plot(depth, diff[:, i], '-', color=color, label=name, linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Depth (Å)')
    ax2.set_ylabel('Difference (recon - true)')
    ax2.set_title('Reconstruction Error')
    ax2.legend()
    ax2.set_xlim([0, depth[-1]])
    ax2.grid(True, alpha=0.3)

    # 3. MSE vs Depth (上段右)
    ax3 = fig.add_subplot(gs[0, 2])
    mse_per_depth = np.mean(diff**2, axis=1)
    ax3.fill_between(depth, 0, mse_per_depth, alpha=0.5)
    ax3.plot(depth, mse_per_depth, 'b-', linewidth=1.5)
    ax3.set_xlabel('Depth (Å)')
    ax3.set_ylabel('MSE')
    ax3.set_title(f'MSE vs Depth (Total: {np.mean(mse_per_depth):.4f})')
    ax3.set_xlim([0, depth[-1]])
    ax3.set_ylim([0, max(mse_per_depth) * 1.2 + 0.001])
    ax3.grid(True, alpha=0.3)

    # 4. 角度分解強度比較 (下段左)
    ax4 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(angles))
    width = 0.25
    for i, (name, color) in enumerate(zip(elem_names, colors)):
        ax4.bar(x - width + i*width, I_obs[:, i], width,
               label=f'{name} (obs)', color=color, alpha=0.7)
        ax4.scatter(x - width + i*width, I_recon[:, i],
                   marker='x', color='k', s=50, zorder=5)
    ax4.set_xlabel('Angle')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Angle-resolved Intensity')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{a}°' for a in angles])
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. 強度差分 (下段中央)
    ax5 = fig.add_subplot(gs[1, 1])
    I_diff = I_recon - I_obs
    for i, (name, color) in enumerate(zip(elem_names, colors)):
        ax5.plot(angles, I_diff[:, i], 'o-', color=color, label=name)
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax5.set_xlabel('Angle (°)')
    ax5.set_ylabel('Intensity Difference')
    ax5.set_title('Intensity Residual')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 精度メトリクス (下段右)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # メトリクス計算
    mse_total = np.mean((profile_recon - profile_true)**2)
    rmse_total = np.sqrt(mse_total)
    mae_total = np.mean(np.abs(profile_recon - profile_true))
    max_error = np.max(np.abs(profile_recon - profile_true))

    # 元素ごとのRMSE
    rmse_per_elem = [np.sqrt(np.mean((profile_recon[:, i] - profile_true[:, i])**2))
                    for i in range(3)]

    # 強度の残差
    I_residual = np.sqrt(np.mean((I_recon - I_obs)**2))

    metrics_text = f"""
    === Accuracy Metrics ===

    Overall:
      MSE:        {mse_total:.6f}
      RMSE:       {rmse_total:.4f}
      MAE:        {mae_total:.4f}
      Max Error:  {max_error:.4f}

    Per Element RMSE:
      C:   {rmse_per_elem[0]:.4f}
      O:   {rmse_per_elem[1]:.4f}
      Si:  {rmse_per_elem[2]:.4f}

    Intensity Residual:
      RMSE: {I_residual:.6f}
    """
    ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def run_roundtrip_test(noise_level: float = 0.0,
                       solver_method: str = 'L2',
                       alpha: float = 1.0,
                       max_iter: int = 100,
                       show_plot: bool = True):
    """
    ラウンドトリップテストを実行

    Args:
        noise_level: ノイズレベル (0-1)
        solver_method: ソルバー手法 ('L2', 'L1', 'MEM')
        alpha: 正則化パラメータ
        max_iter: 最大反復回数
        show_plot: プロットを表示するか

    Returns:
        dict: 結果メトリクス
    """
    print("=" * 60)
    print("4層モデル ラウンドトリップ検証")
    print("=" * 60)

    # セットアップ
    angles = np.array([15, 30, 45, 60, 75, 85])
    depth = np.linspace(0, 150, 76)  # 0-150Å, 76点 (2Å刻み)
    n_depth = len(depth)
    step_size = depth[1] - depth[0]

    elements = [
        ElementConfig(symbol='C', orbital='1s', compound='C', binding_energy=285.0),
        ElementConfig(symbol='O', orbital='1s', compound='SiO2', binding_energy=532.0),
        ElementConfig(symbol='Si', orbital='2p', compound='Si', binding_energy=99.0),
    ]
    n_elements = len(elements)

    print(f"\n設定:")
    print(f"  深さ範囲: 0-{depth[-1]:.0f} Å ({n_depth}点)")
    print(f"  角度: {angles}")
    print(f"  元素: C, O, Si")
    print(f"  ノイズ: {noise_level*100:.1f}%")
    print(f"  ソルバー: {solver_method}, α={alpha}")

    # Transfer Matrix
    tm = TransferMatrix(depth=depth, step_size=step_size, angles=angles)

    # 1. 真のプロファイルを生成
    print("\n1. 真のプロファイルを生成...")
    profile_true = create_4layer_profile(depth)

    # 2. 順問題: スペクトル生成
    print("2. 順問題: 角度分解強度を計算...")
    I_obs = forward_model(profile_true, tm, elements, noise_level=noise_level)

    print(f"   観測強度 (shape: {I_obs.shape}):")
    for i, angle in enumerate(angles):
        print(f"     {angle:2d}°: C={I_obs[i,0]:.3f}, O={I_obs[i,1]:.3f}, Si={I_obs[i,2]:.3f}")

    # 3. 逆問題: プロファイル再構成
    print("3. 逆問題: プロファイルを再構成...")

    solver_config = SolverConfig(
        method=solver_method,
        alpha=alpha,
        max_iter=max_iter,
        tol=1e-8,
    )

    # 初期推定: 真値に近い推定を使う（実際は事前知識として）
    D_prior = np.ones((n_depth, n_elements)) / n_elements

    # より現実的な初期推定: 4層構造を仮定
    for i, z in enumerate(depth):
        # シンプルな4層モデル
        if z < 20:    # C layer
            D_prior[i] = [0.7, 0.15, 0.15]
        elif z < 50:  # SiO2 layer
            D_prior[i] = [0.05, 0.60, 0.35]
        elif z < 80:  # SiOx transition
            D_prior[i] = [0.02, 0.35, 0.63]
        else:         # Si substrate
            D_prior[i] = [0.01, 0.02, 0.97]

    D_prior = D_prior / D_prior.sum(axis=1, keepdims=True)

    profile_recon, result = inverse_model(I_obs, tm, elements, solver_config, D_prior)

    print(f"   収束: {result.success}, 反復: {result.n_iterations}")
    print(f"   Chi²: {result.chi2:.6f}")

    # 4. 再構成プロファイルから強度を計算
    tm.compute(profile_recon, elements)
    I_recon = tm.forward(profile_recon, normalize=True)

    # 5. 精度評価
    print("\n4. 精度評価:")
    mse = np.mean((profile_recon - profile_true)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(profile_recon - profile_true))

    print(f"   MSE:  {mse:.6f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")

    # 元素ごと
    for i, name in enumerate(['C', 'O', 'Si']):
        rmse_elem = np.sqrt(np.mean((profile_recon[:, i] - profile_true[:, i])**2))
        print(f"   RMSE({name}): {rmse_elem:.4f}")

    # 6. 可視化
    if show_plot:
        title = f"Roundtrip: {solver_method}, α={alpha}, noise={noise_level*100:.0f}%"
        fig = plot_roundtrip(depth, profile_true, profile_recon,
                            I_obs, I_recon, angles, elements, title)
        plt.savefig('roundtrip_result.png', dpi=150, bbox_inches='tight')
        print("\n図を roundtrip_result.png に保存しました")
        plt.show()

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'chi2': result.chi2,
        'regularization': result.regularization,
        'objective': result.objective,
        'n_iter': result.n_iterations,
        'profile_true': profile_true,
        'profile_recon': profile_recon,
        'D_prior': D_prior,
        'I_obs': I_obs,
        'I_recon': I_recon,
        'depth': depth,
        'angles': angles,
        'elements': ['C', 'O', 'Si'],
    }


def save_to_h5(
    filepath: str,
    results: dict,
    config: dict = None,
):
    """
    ラウンドトリップ結果をMATLAB互換H5形式で保存

    Args:
        filepath: 出力H5ファイルパス
        results: run_roundtrip_test()の戻り値
        config: 追加設定情報 (optional)

    MATLAB読み込み例:
        D = h5read('roundtrip.h5', '/Drlt');
        depth = h5read('roundtrip.h5', '/depth');
        Q = h5read('roundtrip.h5', '/Qrlt');
    """
    filepath = Path(filepath)

    # データ取得
    profile_true = results['profile_true']      # (depth, elements)
    profile_recon = results['profile_recon']
    D_prior = results['D_prior']
    depth = results['depth']
    elements = results['elements']

    n_depth, n_elements = profile_true.shape
    n_stack = 1  # 単一スペクトルの場合

    with h5py.File(filepath, 'w') as f:
        # メイン深さプロファイル
        # Python (depth, elements) → H5 (stack, elements, depth) for MATLAB (depth, elements, stack)
        # MATLABはcolumn-major、H5はrow-majorなので転置が必要
        # Python: (76, 3) → (1, 3, 76) → MATLAB読込時 (76, 3, 1)
        Drlt_h5 = profile_recon.T[np.newaxis, :, :].astype(np.float32)  # (1, 3, 76)
        Dstk_h5 = D_prior.T[np.newaxis, :, :].astype(np.float32)
        Dtrn_h5 = profile_true.T[np.newaxis, :, :].astype(np.float32)

        f.create_dataset('/Drlt', data=Drlt_h5)
        f.create_dataset('/Dstk', data=Dstk_h5)
        f.create_dataset('/Dtrn', data=Dtrn_h5)

        # 目的関数値: Python (1, 3) → H5 (3, 1) → MATLAB (1, 3)
        Q = np.array([
            results['chi2'],
            results['regularization'],
            results['objective'],
        ], dtype=np.float64).reshape(3, 1)
        f.create_dataset('/Qrlt', data=Q)
        f.create_dataset('/Qstk', data=np.zeros_like(Q))
        f.create_dataset('/Qtrn', data=np.zeros_like(Q))

        # 軸情報
        f.create_dataset('/depth', data=depth.astype(np.float64))
        f.create_dataset('/angle', data=np.array(results['angles'], dtype=np.float64))

        # 元素名 (MATLAB互換の固定長文字列)
        dt = h5py.string_dtype('ascii', 16)
        f.create_dataset('/element', data=np.array(elements, dtype='S16'), dtype=dt)

        # 観測強度 (angles, elements)
        f.create_dataset('/I_obs', data=results['I_obs'].astype(np.float32))
        f.create_dataset('/I_recon', data=results['I_recon'].astype(np.float32))

        # メトリクス
        f.attrs['mse'] = results['mse']
        f.attrs['rmse'] = results['rmse']
        f.attrs['mae'] = results['mae']
        f.attrs['n_iterations'] = results['n_iter']

        # 設定情報
        if config:
            grp = f.create_group('/config')
            for key, val in config.items():
                if isinstance(val, (int, float, str)):
                    grp.attrs[key] = val

    print(f"H5ファイルを保存: {filepath}")
    print(f"  Drlt: {profile_recon.shape} → ({n_depth}, {n_elements}, {n_stack})")
    print(f"  Qrlt: {Q.shape}")

    return filepath


def save_batch_to_h5(
    filepath: str,
    results_list: list,
    depth: np.ndarray,
    elements: list,
    angles: np.ndarray,
):
    """
    複数のラウンドトリップ結果をバッチでH5保存

    Args:
        filepath: 出力H5ファイルパス
        results_list: run_roundtrip_test()の戻り値のリスト
        depth: 深さ軸 (共通)
        elements: 元素リスト (共通)
        angles: 角度リスト (共通)

    MATLAB読み込み例:
        D = h5read('batch.h5', '/Drlt');  % (depth, elements, n_stack)
        Q = h5read('batch.h5', '/Qrlt');  % (n_stack, 3)
    """
    filepath = Path(filepath)
    n_stack = len(results_list)
    n_depth = len(depth)
    n_elements = len(elements)

    # バッチデータを集約 (Python形式で一旦構築)
    Drlt_py = np.zeros((n_stack, n_depth, n_elements), dtype=np.float32)
    Dstk_py = np.zeros((n_stack, n_depth, n_elements), dtype=np.float32)
    Dtrn_py = np.zeros((n_stack, n_depth, n_elements), dtype=np.float32)
    Qrlt_py = np.zeros((n_stack, 3), dtype=np.float64)

    for i, r in enumerate(results_list):
        Drlt_py[i, :, :] = r['profile_recon']
        Dstk_py[i, :, :] = r['D_prior']
        Dtrn_py[i, :, :] = r['profile_true']
        Qrlt_py[i, :] = [r['chi2'], r['regularization'], r['objective']]

    # MATLAB互換形式に変換
    # Python (n_stack, n_depth, n_elements) → H5 (n_elements, n_depth, n_stack)
    # → MATLAB読込時 (n_stack, n_depth, n_elements) → permute で (n_depth, n_elements, n_stack)
    # 実際にはtranspose(2,1,0)で (n_elements, n_depth, n_stack) として保存
    Drlt_h5 = np.transpose(Drlt_py, (2, 1, 0))  # (n_elements, n_depth, n_stack)
    Dstk_h5 = np.transpose(Dstk_py, (2, 1, 0))
    Dtrn_h5 = np.transpose(Dtrn_py, (2, 1, 0))
    Qrlt_h5 = Qrlt_py.T  # (3, n_stack) → MATLAB (n_stack, 3)

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('/Drlt', data=Drlt_h5)
        f.create_dataset('/Dstk', data=Dstk_h5)
        f.create_dataset('/Dtrn', data=Dtrn_h5)
        f.create_dataset('/Qrlt', data=Qrlt_h5)
        f.create_dataset('/Qstk', data=np.zeros_like(Qrlt_h5))
        f.create_dataset('/Qtrn', data=np.zeros_like(Qrlt_h5))

        f.create_dataset('/depth', data=depth.astype(np.float64))
        f.create_dataset('/angle', data=angles.astype(np.float64))

        dt = h5py.string_dtype('ascii', 16)
        f.create_dataset('/element', data=np.array(elements, dtype='S16'), dtype=dt)

        f.attrs['n_stack'] = n_stack

    print(f"バッチH5ファイルを保存: {filepath}")
    print(f"  Drlt: Python {Drlt_py.shape} → MATLAB ({n_depth}, {n_elements}, {n_stack})")
    print(f"  Qrlt: Python {Qrlt_py.shape} → MATLAB ({n_stack}, 3)")

    return filepath


def parameter_sweep():
    """パラメータスイープで最適設定を探索"""
    print("\n" + "=" * 60)
    print("パラメータスイープ")
    print("=" * 60)

    alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
    methods = ['L2', 'MEM']

    results = []

    for method in methods:
        for alpha in alphas:
            print(f"\n--- {method}, α={alpha} ---")
            try:
                r = run_roundtrip_test(
                    noise_level=0.0,
                    solver_method=method,
                    alpha=alpha,
                    max_iter=200,
                    show_plot=False
                )
                results.append({
                    'method': method,
                    'alpha': alpha,
                    'rmse': r['rmse'],
                    'chi2': r['chi2'],
                })
                print(f"    RMSE: {r['rmse']:.4f}, Chi²: {r['chi2']:.6f}")
            except Exception as e:
                print(f"    Error: {e}")
                results.append({
                    'method': method,
                    'alpha': alpha,
                    'rmse': np.nan,
                    'chi2': np.nan,
                })

    # 結果表示
    print("\n" + "=" * 60)
    print("スイープ結果")
    print("=" * 60)
    print(f"{'Method':<8} {'Alpha':>8} {'RMSE':>10} {'Chi²':>12}")
    print("-" * 40)
    for r in results:
        print(f"{r['method']:<8} {r['alpha']:>8.3f} {r['rmse']:>10.4f} {r['chi2']:>12.6f}")

    # 最良の結果
    valid = [r for r in results if not np.isnan(r['rmse'])]
    if valid:
        best = min(valid, key=lambda x: x['rmse'])
        print(f"\n最良: {best['method']}, α={best['alpha']}, RMSE={best['rmse']:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='4層モデル ラウンドトリップ検証')
    parser.add_argument('--sweep', action='store_true', help='パラメータスイープを実行')
    parser.add_argument('--save-h5', type=str, metavar='PATH',
                        help='結果をMATLAB互換H5形式で保存')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='ノイズレベル (0-1, default: 0.0)')
    parser.add_argument('--method', type=str, default='L2',
                        choices=['L2', 'L1', 'MEM'],
                        help='ソルバー手法 (default: L2)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='正則化パラメータ (default: 1.0)')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='最大反復回数 (default: 100)')
    parser.add_argument('--no-plot', action='store_true',
                        help='プロットを表示しない')

    args = parser.parse_args()

    if args.sweep:
        parameter_sweep()
    else:
        # 単一テスト
        results = run_roundtrip_test(
            noise_level=args.noise,
            solver_method=args.method,
            alpha=args.alpha,
            max_iter=args.max_iter,
            show_plot=not args.no_plot,
        )

        # H5保存
        if args.save_h5:
            config = {
                'noise_level': args.noise,
                'method': args.method,
                'alpha': args.alpha,
                'max_iter': args.max_iter,
            }
            save_to_h5(args.save_h5, results, config)
            print(f"\nMATLABで読み込み:")
            print(f"  D = h5read('{args.save_h5}', '/Drlt');")
            print(f"  depth = h5read('{args.save_h5}', '/depth');")
