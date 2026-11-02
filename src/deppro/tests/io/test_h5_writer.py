#!/usr/bin/env python
"""
H5Writer テスト - MATLAB GUI互換性確認

2026-09 に ``deppro/io/`` から移動。配布パッケージの中に test_*.py があると wheel に入り、
pytest の testpaths からも外れて誰も走らせない状態だった（テストは tests/ に置く）。
出力先は pytest の tmp_path（``__main__`` では一時ディレクトリ）。
"""

import tempfile
from pathlib import Path

import numpy as np

from deppro._compat import trapezoid
from deppro.io import H5Writer, create_gui_compatible_h5


def create_test_profile(depth: np.ndarray) -> np.ndarray:
    """4層モデルのテストプロファイル（シグモイドベース、FeatureVectorとは非整合）"""
    n_depth = len(depth)
    profile = np.zeros((n_depth, 3))  # [C, O, Si]

    for i, z in enumerate(depth):
        # シグモイド関数
        def sigmoid(x, x0, w):
            return 1 / (1 + np.exp(-(x - x0) / max(w, 0.1)))

        # 各層の寄与
        f_C = 1 - sigmoid(z, 15, 5)
        f_SiO2 = sigmoid(z, 15, 5) * (1 - sigmoid(z, 45, 8))
        f_SiOx = sigmoid(z, 45, 8) * (1 - sigmoid(z, 70, 10))
        f_Si = sigmoid(z, 70, 10)

        # 組成
        c_C = 0.85 * f_C + 0.02 * f_SiO2 + 0.01 * f_SiOx
        c_O = 0.05 * f_C + 0.66 * f_SiO2 + 0.30 * f_SiOx
        c_Si = 0.10 * f_C + 0.32 * f_SiO2 + 0.69 * f_SiOx + 1.00 * f_Si

        total = c_C + c_O + c_Si
        profile[i] = [c_C / total, c_O / total, c_Si / total]

    return profile


def create_gaussian_profile(
    depth: np.ndarray,
    thck: float,
    abrp: float,
    free_params: list,
    subs_ratio: float = 1.0,
) -> tuple:
    """
    MATLABのCalculateDistribution.m（Gaussianモード）と整合するプロファイルを生成

    CalculateDistribution.mの計算式（Gaussian）:
        S(k) = width(k) / (2*sqrt(log(2)))  # FWHM→σ変換
        D0(:,k) = height(k)/100 * exp(-(depth-pos(k))^2 / S(k)^2) * |1-D_subst|
        D(:,subst) = subs * (1/2) * |1 + erf((depth-thck)/abrp_sigma)|
        abrp_sigma = abrp / (2*sqrt(log(2)))

    Args:
        depth: 深さ軸 [Å]
        thck: 基板との界面位置 [Å]
        abrp: 界面の急峻度（FWHM形式） [Å]
        free_params: [(pos, width, height), ...] 各自由元素のパラメータ
            pos: ピーク位置 [Å]
            width: FWHM [Å]
            height: 最大濃度 [%] (0-100)
        subs_ratio: 基板元素の比率（1元素の場合は1.0）

    Returns:
        profile: (n_depth, n_elements) 深さプロファイル
        fv_source: FeatureVector_Source 配列
    """
    from scipy.special import erf

    n_depth = len(depth)
    n_free = len(free_params)
    n_elements = n_free + 1  # 自由元素 + 基板

    # FWHM→σ変換係数
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(np.log(2)))

    # 基板プロファイル（最後の元素）
    abrp_sigma = abrp * fwhm_to_sigma
    D_subst = subs_ratio * 0.5 * np.abs(1 + erf((depth - thck) / abrp_sigma))

    # 自由元素のプロファイル
    profile = np.zeros((n_depth, n_elements))
    profile[:, -1] = D_subst  # 基板は最後

    for k, (pos, width, height) in enumerate(free_params):
        sigma = width * fwhm_to_sigma
        D0_k = (height / 100.0) * np.exp(-((depth - pos) ** 2) / (sigma ** 2)) * np.abs(1 - D_subst)
        profile[:, k] = D0_k

    # 正規化（合計が1を超えないように）
    total = profile.sum(axis=1, keepdims=True)
    profile = np.where(total > 1, profile / total, profile)

    # FeatureVector_Source: [thck, abrp, pos1, width1, height1, pos2, width2, height2, ...]
    fv_source = [thck, abrp]
    for pos, width, height in free_params:
        fv_source.extend([pos, width, height])
    fv_source = np.array(fv_source, dtype=np.float32)

    return profile, fv_source


def compute_transfer_matrix(depth: np.ndarray, angles: np.ndarray, imfp: np.ndarray) -> np.ndarray:
    """
    簡易転送行列の計算

    Args:
        depth: 深さ軸 [Å]
        angles: 放出角度 [°]
        imfp: 各元素のIMFP [Å] (n_elements,)

    Returns:
        TransMat: (n_angles, n_depth, n_elements)
    """
    n_depth = len(depth)
    n_angles = len(angles)
    n_elements = len(imfp)

    step = depth[1] - depth[0] if len(depth) > 1 else 2.0
    TransMat = np.zeros((n_angles, n_depth, n_elements))

    for i, theta in enumerate(angles):
        cos_theta = np.cos(np.radians(theta))
        for j, z in enumerate(depth):
            for k, lam in enumerate(imfp):
                # 減衰項: exp(-z / (λ * cos(θ)))
                atten = np.exp(-z / (lam * cos_theta + 1e-10))
                TransMat[i, j, k] = atten * step

    return TransMat


def compute_intensity(profile: np.ndarray, TransMat: np.ndarray) -> np.ndarray:
    """
    深さプロファイルから角度依存強度を計算

    Args:
        profile: (n_depth, n_elements)
        TransMat: (n_angles, n_depth, n_elements)

    Returns:
        X: (n_angles, n_elements)
    """
    n_angles = TransMat.shape[0]
    n_elements = TransMat.shape[2]
    X = np.zeros((n_angles, n_elements))

    for k in range(n_elements):
        # X[i, k] = sum_j TransMat[i, j, k] * profile[j, k]
        X[:, k] = np.einsum('ij,j->i', TransMat[:, :, k], profile[:, k])

    # 正規化
    X = X / X.sum(axis=1, keepdims=True)

    return X


def test_basic(tmp_path):
    """基本テスト（旧シグモイドプロファイル）- Setボタンとは非整合"""
    print("=" * 60)
    print("H5Writer 基本テスト（シグモイドプロファイル）")
    print("=" * 60)

    # パラメータ設定
    depth = np.linspace(0, 150, 76)
    angles = np.array([15, 30, 45, 60, 75, 85], dtype=np.float64)
    elements = ['C', 'O', 'Si']

    # IMFPの推定値 [Å] - C 1s, O 1s, Si 2p
    imfp = np.array([25.0, 22.0, 30.0])

    # プロファイル生成
    profile_true = create_test_profile(depth)
    profile_recon = profile_true + np.random.normal(0, 0.02, profile_true.shape)
    profile_recon = np.clip(profile_recon, 0, 1)
    profile_recon = profile_recon / profile_recon.sum(axis=1, keepdims=True)

    # 転送行列の計算
    TransMat = compute_transfer_matrix(depth, angles, imfp)

    # λ行列の作成
    n_elements = len(elements)
    n_angles = len(angles)
    n_depth = len(depth)

    # lambdabasic: (n_elements, n_elements, 1) - 対角要素にIMFP
    # 参照ファイルでは (1, n_elements, n_elements) の完全な行列
    # 簡易版: 対角要素のみ
    lambda_basic = np.zeros((n_elements, n_elements, 1), dtype=np.float32)
    for i in range(n_elements):
        lambda_basic[i, i, 0] = imfp[i]

    # lambdaobs: (n_elements, n_depth) - 深さ方向の実効IMFP
    # 参照ファイルでは (1, n_elements, n_depth) 形式
    # 各元素のIMFPを深さ方向に並べる（一様）
    lambda_obs = np.zeros((n_elements, n_depth), dtype=np.float32)
    for i in range(n_elements):
        lambda_obs[i, :] = imfp[i]

    # lambdacosinethetaobs: (n_angles, n_elements) - λ*cos(θ)
    lambda_cos_theta = np.zeros((n_angles, n_elements), dtype=np.float32)
    for i, theta in enumerate(angles):
        cos_theta = np.cos(np.radians(theta))
        lambda_cos_theta[i, :] = imfp * cos_theta

    # 観測強度の計算（真のプロファイルから）
    Xobs = compute_intensity(profile_true, TransMat)
    # ノイズ追加
    Xobs = Xobs + np.random.normal(0, 0.01, Xobs.shape)
    Xobs = np.clip(Xobs, 0.001, None)
    Xobs = Xobs / Xobs.sum(axis=1, keepdims=True)

    # 目的関数
    Qrlt = np.array([[0.015, 0.0003, 0.008]])

    # 出力パス
    output_dir = Path(tmp_path)
    output_path = output_dir / 'test_sigmoid_profile.h5'

    # 詳細版APIを使用（TransMatとXobsを設定するため）
    writer = H5Writer(output_path)
    writer.set_basic_params(
        depth=depth,
        angles=angles,
        elements=elements,
    )
    writer.set_transfer_matrix(
        TransMat,
        lambda_basic=lambda_basic,
        lambda_obs=lambda_obs,
        lambda_cos_theta_obs=lambda_cos_theta,
    )
    writer.set_profiles(
        Drlt=profile_recon,
        Dtrn=profile_true,
        Dstk=np.ones_like(profile_true) / 3,
        Xobs=Xobs,
    )
    writer.set_objectives(Qrlt=Qrlt)

    writer.write()

    print(f"\n出力: {output_path}")
    print("注意: このファイルはSetボタンの初期プロファイル再計算と整合しません")

    return output_path


def test_gaussian_compatible(tmp_path):
    """
    Gaussianプロファイルテスト - MATLABのSetボタンと完全整合

    FeatureVector_Sourceから生成したプロファイルが、MATLABの
    CalculateDistribution.m（Gaussianモード）で再計算した結果と一致する。
    """
    print("=" * 60)
    print("H5Writer Gaussianプロファイルテスト（Set互換）")
    print("=" * 60)

    # パラメータ設定
    depth = np.linspace(0, 150, 76)
    angles = np.array([15, 30, 45, 60, 75, 85], dtype=np.float64)
    elements = ['C', 'O', 'Si']
    n_elements = len(elements)
    n_depth = len(depth)
    n_angles = len(angles)

    # IMFPの推定値 [Å] - C 1s, O 1s, Si 2p
    imfp = np.array([25.0, 22.0, 30.0])

    # === Gaussianプロファイルのパラメータ ===
    # 物理モデル: C表面汚染 + SiO2層 + Si基板
    thck = 50.0     # 基板界面位置 [Å]
    abrp = 10.0     # 界面急峻度（FWHM） [Å]

    # 自由元素（C, O）のGaussianパラメータ
    # [(position, width_FWHM, height_%)]
    free_params = [
        (5.0, 15.0, 80.0),   # C: 表面近くにピーク、FWHM=15Å、最大80%
        (30.0, 25.0, 60.0),  # O: SiO2層にピーク、FWHM=25Å、最大60%
    ]

    # プロファイル生成（MATLABと同じアルゴリズム）
    profile_true, fv_source = create_gaussian_profile(depth, thck, abrp, free_params)

    print(f"\nFeatureVector_Source: {fv_source}")
    print(f"  thck={fv_source[0]:.1f}Å, abrp={fv_source[1]:.1f}Å")
    print(f"  C: pos={fv_source[2]:.1f}Å, width={fv_source[3]:.1f}Å, height={fv_source[4]:.1f}%")
    print(f"  O: pos={fv_source[5]:.1f}Å, width={fv_source[6]:.1f}Å, height={fv_source[7]:.1f}%")

    # プロファイル特性の出力
    for i, elem in enumerate(elements):
        max_val = profile_true[:, i].max() * 100
        max_idx = profile_true[:, i].argmax()
        print(f"  {elem}: max={max_val:.1f}% at depth={depth[max_idx]:.1f}Å")

    # ノイズを加えた再構成プロファイル
    profile_recon = profile_true + np.random.normal(0, 0.02, profile_true.shape)
    profile_recon = np.clip(profile_recon, 0, 1)
    profile_recon = profile_recon / profile_recon.sum(axis=1, keepdims=True)

    # 転送行列の計算
    TransMat = compute_transfer_matrix(depth, angles, imfp)

    # λ行列の作成
    lambda_basic = np.zeros((n_elements, n_elements, 1), dtype=np.float32)
    for i in range(n_elements):
        lambda_basic[i, i, 0] = imfp[i]

    lambda_obs = np.zeros((n_elements, n_depth), dtype=np.float32)
    for i in range(n_elements):
        lambda_obs[i, :] = imfp[i]

    lambda_cos_theta = np.zeros((n_angles, n_elements), dtype=np.float32)
    for i, theta in enumerate(angles):
        cos_theta = np.cos(np.radians(theta))
        lambda_cos_theta[i, :] = imfp * cos_theta

    # 観測強度の計算
    Xobs = compute_intensity(profile_true, TransMat)
    Xobs = Xobs + np.random.normal(0, 0.01, Xobs.shape)
    Xobs = np.clip(Xobs, 0.001, None)
    Xobs = Xobs / Xobs.sum(axis=1, keepdims=True)

    # 目的関数
    Qrlt = np.array([[0.015, 0.0003, 0.008]])

    # thetaopt: 標準16点（-10〜90度）- 滑らかな曲線表示用
    angles_opt = np.array([-10, 0, 10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90], dtype=np.float32)
    n_angles_opt = len(angles_opt)

    # thetaopt 用の転送行列と λcosθ
    TransMatOpt = compute_transfer_matrix(depth, angles_opt, imfp)
    lambda_cos_theta_opt = np.zeros((n_angles_opt, n_elements), dtype=np.float32)
    for i, theta in enumerate(angles_opt):
        cos_theta = np.cos(np.radians(theta))
        lambda_cos_theta_opt[i, :] = imfp * cos_theta

    # 出力パス
    output_dir = Path(tmp_path)
    output_path = output_dir / 'test_gui_compatible.h5'

    # H5Writer
    writer = H5Writer(output_path)
    writer.set_basic_params(
        depth=depth,
        angles=angles,
        angles_opt=angles_opt,
        elements=elements,
    )
    writer.set_transfer_matrix(
        TransMat,
        TransMatOpt=TransMatOpt,
        lambda_basic=lambda_basic,
        lambda_obs=lambda_obs,
        lambda_cos_theta_obs=lambda_cos_theta,
        lambda_cos_theta_opt=lambda_cos_theta_opt,
    )
    # Xobs: thetaobs用、Xopt: thetaopt用
    Xobs_data = compute_intensity(profile_true, TransMat)  # (n_angles_obs, n_elements)
    Xopt_data = compute_intensity(profile_true, TransMatOpt)  # (n_angles_opt, n_elements)

    writer.set_profiles(
        Drlt=profile_recon,
        Dtrn=profile_true,
        Dstk=np.ones_like(profile_true) / 3,
        Xobs=Xobs_data,
        Xopt=Xopt_data,
    )
    writer.set_objectives(Qrlt=Qrlt)

    # === 初期パラメータを設定（Setボタン対応） ===
    writer._ezdeprof['NP_Set'] = 2  # 1層 + 基板（2の倍数）
    writer._ezdeprof['NP_Calc'] = 2
    writer._ezdeprof['NP_Max'] = 4

    writer._ezdeprof['deltaphoton'] = 100.0
    # datapoints は set_basic_params で正しく設定されるので削除
    # writer._ezdeprof['datapoints'] = np.array([n_angles, 1, n_angles_opt], dtype=np.uint16)

    # 元素属性
    writer._ezdeprof['prpties'] = ['Free', 'Free', 'Subst.']
    writer._ezdeprof['subst'] = np.array([3], dtype=np.uint8)
    writer._ezdeprof['free'] = np.array([1, 2], dtype=np.uint8)
    writer._ezdeprof['antisubst'] = np.array([1, 2], dtype=np.uint8)

    # 膜厚・急峻度
    writer._ezdeprof['thckini'] = np.array([thck, 0.0, 0.0], dtype=np.float32)
    writer._ezdeprof['thckmod'] = np.array([thck, 0.0, 0.0], dtype=np.float32)
    writer._ezdeprof['abrpini'] = np.array([abrp, 0.0, 0.0], dtype=np.float32)
    writer._ezdeprof['abrpmod'] = np.array([abrp, 0.0, 0.0], dtype=np.float32)

    # 基板組成
    # subsini/subsmod: 参照ファイル形式 (3, n_subst)
    # 各列が基板元素の [初期値; 下限; 上限]
    writer._ezdeprof['subsini'] = np.array([[1.0], [0.0], [1.0]], dtype=np.float32)
    writer._ezdeprof['subsmod'] = np.array([[1.0], [0.0], [1.0]], dtype=np.float32)

    # FeatureVector（プロファイルと整合）
    writer._ezdeprof['FeatureVector_Source'] = fv_source
    writer._ezdeprof['FeatureVector_Set'] = fv_source.copy()
    writer._ezdeprof['FeatureVector_Calc'] = fv_source.copy()

    # SupportVector（基板+リンク元素のみ）
    # 参照: DepthProfiler.m の Simulation コード
    #   handles.SupportVector_Set = [reshape([p(handles.subst)' w(handles.subst)' h(handles.subst)']', ...)
    #                                reshape([p(handles.linked)' w(handles.linked)' h(handles.linked)']', ...)]
    #
    # つまり SupportVector には free 元素(C, O)は含まれない！
    # 基板(Si)のみ: [pos_Si, width_Si, height_Si]
    #
    # Dobsから自動計算される値:
    #   p = 重心位置 = sum(D * depth) / sum(D)
    #   h = 最大濃度 (%) = 100 * max(D)
    #   w = 実効幅 = 2*sqrt(log(2)/pi) * integral(D) / max(D)

    # Dtrnから基板のプロファイル特性を計算
    D_Si = profile_true[:, -1]  # 最後の元素が基板
    # 重心位置
    pos_Si = np.sum(D_Si * depth) / np.sum(D_Si) if np.sum(D_Si) > 0 else 0
    # 最大濃度
    height_Si = 100.0 * np.max(D_Si)
    # 実効幅
    width_Si = 2 * np.sqrt(np.log(2) / np.pi) * trapezoid(100 * D_Si, depth) / height_Si if height_Si > 0 else 0

    print(f"  Si(基板): pos={pos_Si:.1f}Å, width={width_Si:.1f}Å, height={height_Si:.1f}%")

    sv_set = np.array([pos_Si, width_Si, height_Si], dtype=np.float32)
    writer._ezdeprof['SupportVector_Set'] = sv_set
    writer._ezdeprof['SupportVector_Calc'] = sv_set.copy()

    # 平均濃度
    mean_conc = profile_true.mean(axis=0) * 100
    writer._ezdeprof['Conc_Set'] = mean_conc.astype(np.float32)
    writer._ezdeprof['Conc_Calc'] = mean_conc.astype(np.float32)

    # スパースネス
    writer._ezdeprof['Sprs_Set'] = np.array([1.0, 1.0], dtype=np.float32)
    writer._ezdeprof['Sprs_Calc'] = np.array([1.0, 1.0], dtype=np.float32)

    writer.write()

    print(f"\n出力: {output_path}")
    print("\nMATLABで検証:")
    print("  1. DepthProfilerを起動")
    print("  2. File > Read Result で test_gui_compatible.h5 を読み込み")
    print("  3. 'Gaussian'を選択してSetボタンを押す")
    print("  4. 初期プロファイル（青線）が読み込んだプロファイルと一致するか確認")

    return output_path


def test_detailed(tmp_path):
    """詳細APIテスト"""
    print("\n" + "=" * 60)
    print("H5Writer 詳細APIテスト")
    print("=" * 60)

    depth = np.linspace(0, 100, 51)
    angles = np.array([0, 30, 45, 60])
    elements = ['O', 'Si']

    output_dir = Path(tmp_path)
    output_path = output_dir / 'test_gui_detailed.h5'

    writer = H5Writer(output_path)

    writer.set_basic_params(
        depth=depth,
        angles=angles,
        elements=elements,
        orbitals=['1s', '2p'],
        compounds=['SiO2', 'Si'],
        binding_energies=np.array([532.0, 99.0]),
        photon_energy=1486.6,
        simulation=True,
    )

    # 簡単な SiO2/Si 2層
    n_depth = len(depth)
    profile = np.zeros((n_depth, 2))
    sio2_thickness = 30  # Å

    for i, z in enumerate(depth):
        if z < sio2_thickness:
            profile[i] = [0.667, 0.333]  # SiO2
        else:
            profile[i] = [0.0, 1.0]  # Si

    writer.set_profiles(
        Drlt=profile,
        Dtrn=profile,
        Dstk=np.ones_like(profile) / 2,
    )

    writer.set_objectives(
        Qrlt=np.array([0.01, 0.001, 0.005]),
    )

    writer.write()

    print(f"\n出力: {output_path}")

    return output_path


def verify_h5_structure(filepath: Path):
    """H5ファイル構造を検証"""
    import h5py

    print(f"\n=== H5構造検証: {filepath.name} ===")

    with h5py.File(filepath, 'r') as f:
        print("\nルートデータセット:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"  /{key}: {f[key].shape} {f[key].dtype}")
            else:
                print(f"  /{key}/ (group)")

        print("\n/ezdeprof/ フィールド:")
        if 'ezdeprof' in f:
            ezdeprof = f['ezdeprof']
            for key in sorted(ezdeprof.keys())[:20]:  # 最初の20個
                ds = ezdeprof[key]
                print(f"  {key}: {ds.shape} {ds.dtype}")
            if len(ezdeprof.keys()) > 20:
                print(f"  ... ({len(ezdeprof.keys())} fields total)")

        # 重要なフィールドの値を確認
        print("\n重要なフィールド:")
        print(f"  depth: {f['/ezdeprof/depth'].shape}")
        elem_data = f['/ezdeprof/elementsymbol'][:]
        if elem_data.ndim == 2:
            elems = [s.decode() if isinstance(s, bytes) else str(s) for s in elem_data[0]]
        else:
            elems = [s.decode() if isinstance(s, bytes) else str(s) for s in elem_data]
        print(f"  elementsymbol: {elems}")
        print(f"  Drlt: {f['/Drlt'].shape}")
        print(f"  Qrlt: {f['/Qrlt'].shape}")


if __name__ == '__main__':
    _tmp = Path(tempfile.mkdtemp(prefix="h5writer_"))
    # メインテスト: Gaussianプロファイル（Set互換）
    path1 = test_gaussian_compatible(_tmp)
    verify_h5_structure(path1)

    # サブテスト: シグモイドプロファイル（参考用）
    path2 = test_basic(_tmp)

    # サブテスト: 詳細API
    path3 = test_detailed(_tmp)
    verify_h5_structure(path3)

    print("\n" + "=" * 60)
    print("MATLABでの読み込みテスト用コマンド:")
    print("=" * 60)
    print(f"""
cd '{path1.parent}'

% Gaussian互換テスト（Setボタン検証用）
store = deppro.io.DataStore('test_gui_compatible.h5');
ez = store.getEzdeprof('full', true);
D = store.getDrlt(1);
fprintf('depth: %d, elements: %d\\n', size(D, 1), size(D, 2));
disp('FeatureVector_Source:'); disp(ez.FeatureVector_Source');

% GUIで検証:
%   1. DepthProfiler を起動
%   2. File > Read Result で test_gui_compatible.h5 を読み込み
%   3. Distribution function で 'Gaussian' を選択
%   4. Setボタンを押す
%   5. 初期プロファイル（青線）がDtrnと一致するか確認
""")
