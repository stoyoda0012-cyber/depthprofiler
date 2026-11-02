#!/usr/bin/env python
"""
HXPS (8keV) 多層膜サンプルのH5ファイルをゼロから生成する例

MATLAB DepthProfiler が書く 6 元素 H5 と同様の構造を持つH5ファイルを生成します。
- 6元素: Si(基板), SiO2, TiO2, Al2O3, C(汚染), O(酸化物)
- 8keV HXPS測定を想定
- 4層構造: C表面汚染 / Al2O3 / TiO2 / Si基板

新しい initializer モジュールを使用して、MATLABのSetボタンと
同等の初期化を行います。
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deppro.io import H5Writer
from deppro.io.initializer import DepthProfileInitializer, create_initialized_h5


def create_multilayer_profile(depth: np.ndarray) -> np.ndarray:
    """
    4層構造の深さプロファイルを生成

    構造（上から）:
    - C汚染層: 0-20Å
    - Al2O3層: 20-80Å
    - TiO2層: 80-160Å
    - Si基板: 160Å以深

    元素: [Si(0), Si(+4), Ti(+4), Al(+3), C(0), O(-2)]

    Returns:
        profile: (n_depth, 6) 深さプロファイル
    """
    from scipy.special import erf

    n_depth = len(depth)
    profile = np.zeros((n_depth, 6))

    # FWHM→σ変換係数
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(np.log(2)))

    # 界面位置と急峻度（FWHM）
    z1 = 20.0   # C/Al2O3 界面
    z2 = 80.0   # Al2O3/TiO2 界面
    z3 = 160.0  # TiO2/Si 界面
    w = 20.0    # 界面幅（FWHM）
    sigma = w * fwhm_to_sigma

    for i, z in enumerate(depth):
        # 各層の存在関数（シグモイド遷移）
        f_C = 0.5 * (1 - erf((z - z1) / sigma))
        f_Al2O3 = 0.5 * (1 + erf((z - z1) / sigma)) * 0.5 * (1 - erf((z - z2) / sigma))
        f_TiO2 = 0.5 * (1 + erf((z - z2) / sigma)) * 0.5 * (1 - erf((z - z3) / sigma))
        f_Si = 0.5 * (1 + erf((z - z3) / sigma))

        # 元素濃度
        # [Si(0), Si(+4), Ti(+4), Al(+3), C(0), O(-2)]
        profile[i, 0] = f_Si * 1.0                          # Si(0): 基板
        profile[i, 1] = 0.0                                  # Si(+4): なし
        profile[i, 2] = f_TiO2 * 0.33                        # Ti(+4): TiO2層
        profile[i, 3] = f_Al2O3 * 0.40                       # Al(+3): Al2O3層
        profile[i, 4] = f_C * 0.90                           # C(0): 表面汚染のみ
        profile[i, 5] = (f_TiO2 * 0.67 + f_Al2O3 * 0.60 +    # O(-2): 酸化物
                        f_C * 0.08)

    # 正規化
    total = profile.sum(axis=1, keepdims=True)
    profile = np.where(total > 0, profile / total, profile)

    return profile


def main():
    print("=" * 70)
    print("HXPS多層膜H5ファイル生成サンプル")
    print("=" * 70)

    # =========================================================================
    # 1. 基本パラメータ設定
    # =========================================================================
    photon_energy = 8000.0  # 8keV HXPS
    total_depth = 600.0     # 解析深さ [Å]
    n_depth = 20            # 深さ分割数
    slice_size = 20.0       # 膜分割サイズ [Å]

    # 深さ軸
    depth = np.linspace(0, total_depth, n_depth)
    step_size = depth[1] - depth[0]

    # 測定角度（14点等間隔 0-60°）
    thetaobs = np.linspace(0, 60, 14)

    # 最適化/表示用角度（標準16点）
    thetaopt = np.array([-10, 0, 10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
                       dtype=np.float32)

    # =========================================================================
    # 2. 元素設定
    # =========================================================================
    # 元素記号（化学種ごとに区別）
    elements = ['Si', 'Si', 'Ti', 'Al', 'C', 'O']

    # オービタル（HXPS 8keVでは1sを使用）
    orbitals = ['1s', '1s', '1s', '1s', '1s', '1s']

    # 化合物名（物性値の参照用）
    compounds = ['Si', 'SiO2', 'TiO2', 'Al2O3', 'C', 'Oxide']

    # 電荷状態
    charges = [0.0, 4.0, 4.0, 3.0, 0.0, -2.0]

    # 基板・自由・リンク設定（1-based index）
    subst = [1]           # Si(0) が基板
    free = [2, 3, 4, 5]   # Si(+4), Ti, Al, C が自由
    linked = [6]          # O がリンク（酸化物として他元素に従属）

    # プロパティ文字列
    prpties = ['Subst.', 'Free', 'Free', 'Free', 'Free', 'Linked']

    # =========================================================================
    # 3. 深さプロファイル生成
    # =========================================================================
    print("\n深さプロファイル生成中...")
    profile_true = create_multilayer_profile(depth)

    # 初期推定 = 真のプロファイルのコピー（MATLABのSetで再計算される）
    profile_init = profile_true.copy()

    # 結果（少しノイズを加えた真値）
    noise_level = 0.02
    profile_result = profile_true + np.random.normal(0, noise_level, profile_true.shape)
    profile_result = np.clip(profile_result, 0, 1)
    profile_result = profile_result / profile_result.sum(axis=1, keepdims=True)

    print(f"  深さ点数: {n_depth}")
    print(f"  元素数: {len(elements)}")

    # =========================================================================
    # 4. 初期化計算（MATLABのSetボタンと同等）
    # =========================================================================
    print("\n初期化計算中（MATLAB Set相当）...")

    initializer = DepthProfileInitializer(
        depth=depth,
        angles_obs=thetaobs,
        elements=elements,
        compounds=compounds,
        angles_opt=thetaopt,
        photon_energy=photon_energy,
        charges=charges,
    )

    # 真のプロファイルから転送行列・角度プロファイルを計算
    init_results = initializer.initialize_from_profile(profile_true)

    print(f"  lambdabasic (対角): {np.diag(init_results['lambdabasic'])}")
    print(f"  Xobs[0] (angle=0): {init_results['Xobs'][0]}")

    # =========================================================================
    # 5. H5Writer でファイル生成
    # =========================================================================
    from deppro.paths import ensure_outputs_dir
    output_dir = ensure_outputs_dir()
    output_path = output_dir / 'example_hxps_multilayer.h5'

    print(f"\nH5ファイル生成中...")
    print(f"  出力: {output_path}")

    writer = H5Writer(output_path)

    # 基本パラメータ設定
    writer.set_basic_params(
        depth=depth,
        angles=thetaobs,
        angles_opt=thetaopt,
        elements=elements,
        orbitals=orbitals,
        compounds=compounds,
        photon_energy=photon_energy,
        simulation=True,
        slice_size=slice_size,
        subst=subst,
        free=free,
        linked=linked,
        prpties=prpties,
        charges=charges,
    )

    # 初期化結果をWriterに適用（転送行列、IMFP等）
    initializer.apply_to_writer(writer)

    # プロファイル設定
    writer.set_profiles(
        Drlt=profile_result,
        Dtrn=profile_true,
        Dstk=profile_init,
        Xobs=init_results['Xobs'],
        Xopt=init_results['Xopt'],
    )

    # ★ set_profiles後に初期化結果を再適用（上書き防止）
    initializer.apply_to_writer(writer)

    # 目的関数（ダミー値）
    writer.set_objectives(
        Qrlt=np.array([[1000.0, 0.0, 1000.0]]),
    )

    # ezdeprof の追加設定
    ez = writer._ezdeprof

    # 膜厚パラメータ（4層構造）
    thck_init = 160.0  # Si基板との界面位置
    abrp_init = 20.0   # 界面急峻度
    ez['thckini'] = np.array([thck_init, 0.0, total_depth], dtype=np.float32)
    ez['abrpini'] = np.array([abrp_init, abrp_init, abrp_init], dtype=np.float32)
    ez['thckmod'] = ez['thckini'].copy()
    ez['abrpmod'] = ez['abrpini'].copy()

    # FeatureVector_Source（freeconst=4要素）
    # [thck, abrp, pos1, width1, height1, pos2, width2, height2, ...]
    n_freeconst = 4
    fv = np.zeros(2 + 3 * n_freeconst, dtype=np.float32)
    fv[0] = thck_init
    fv[1] = abrp_init
    # 各free元素のパラメータ（参考値）
    free_params = [
        (thck_init, 60.0, 33.3),   # Si(+4): 界面付近
        (120.0, 80.0, 33.3),       # Ti: TiO2層
        (50.0, 80.0, 33.3),        # Al: Al2O3層
        (10.0, 40.0, 80.0),        # C: 表面汚染
    ]
    for i, (pos, width, height) in enumerate(free_params):
        fv[2 + 3*i + 0] = pos
        fv[2 + 3*i + 1] = width
        fv[2 + 3*i + 2] = height
    ez['FeatureVector_Source'] = fv
    ez['FeatureVector_Set'] = fv.copy()
    ez['FeatureVector_Calc'] = fv.copy()

    # SupportVector（基板+リンク）
    # [pos_Si, width_Si, height_Si, pos_O, width_O, height_O]
    sv = np.array([
        total_depth / 2, total_depth / 2, 100.0,  # Si基板
        thck_init / 2, thck_init, 60.0,           # O（酸化物）
    ], dtype=np.float32)
    ez['SupportVector_Set'] = sv
    ez['SupportVector_Calc'] = sv.copy()

    # 書き込み
    writer.write()

    # =========================================================================
    # 6. 検証
    # =========================================================================
    print("\n" + "=" * 70)
    print("検証")
    print("=" * 70)

    import h5py
    with h5py.File(output_path, 'r') as f:
        print("\n【ルートデータセット】")
        for key in ['Drlt', 'Dtrn', 'Dstk', 'Xrlt', 'Xstk', 'Qrlt']:
            if key in f:
                print(f"  {key}: {f[key].shape}")

        print("\n【ezdeprof 主要フィールド】")
        ez = f['ezdeprof']
        for key in ['depth', 'thetaobs', 'elementsymbol', 'Dini', 'lambdabasic']:
            if key in ez:
                print(f"  {key}: {ez[key].shape}")

        print("\n【物性パラメータ】")
        print(f"  photon: {ez['photon'][()][0]} eV")
        print(f"  bindingenergy: {ez['bindingenergy'][:].flatten()}")
        print(f"  FeatureVector_Source: {ez['FeatureVector_Source'][:].flatten()}")

    print("\n" + "=" * 70)
    print("完了")
    print("=" * 70)
    print(f"""
MATLABでの読み込み:
  cd '{output_dir}'

  % GUIで読み込み
  DepthProfiler

  % File > Read Result で example_hxps_multilayer.h5 を選択
  % Distribution function で 'Gaussian' を選択
  % Setボタンを押して初期プロファイルを確認

  % または直接データ読み込み
  store = deppro.io.DataStore('example_hxps_multilayer.h5');
  ez = store.getEzdeprof('full', true);
  D = store.getDrlt(1);
  plot(ez.depth, D');
  legend(ez.element);
""")


if __name__ == '__main__':
    main()
