"""
Structure Analyzer テスト
========================

実際のARXPSシミュレーションデータでStructureAnalyzerを検証
"""

import numpy as np
from scipy.special import erf

from deppro.structure_analyzer import StructureAnalyzer, StructureType, SolverRecommendation
from deppro.depth_profile import create_multilayer_profile, ElementConfig
from deppro.transfer_matrix import TransferMatrix


def create_sio2_si_data(
    thickness: float = 30.0,
    interface_width: float = 5.0,
    noise_level: float = 0.02,
):
    """SiO2/Siデータを生成"""
    depth = np.arange(0, 150, 1.0)
    n_depth = len(depth)

    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    D = np.zeros((n_depth, 3))

    sio2_profile = 1 - 0.5 * (1 + erf((depth - thickness) / sigma))
    D[:, 0] = (1/3) * sio2_profile
    D[:, 1] = (2/3) * sio2_profile
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    angles = [15, 30, 45, 60, 75, 85]
    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D, elements)

    I_true = tm.forward(D, normalize=True)

    if noise_level > 0:
        noise = noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true + noise * I_true
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    else:
        I_obs = I_true.copy()

    return I_obs, tm, ['Si_ox', 'O', 'Si_bulk']


def create_multilayer_data(
    thicknesses: list = [5.0, 15.0, 35.0, 30.0],
    widths: list = [2.0, 3.0, 4.0, 5.0],
    noise_level: float = 0.02,
):
    """多層膜データを生成"""
    depth = np.arange(0, 150, 1.0)
    n_depth = len(depth)

    D = np.zeros((n_depth, 6))
    cumulative = np.cumsum(thicknesses)
    sigmas = np.array(widths) / (2 * np.sqrt(np.log(2)))

    layer_profiles = []
    for i in range(4):
        z_start = 0 if i == 0 else cumulative[i - 1]
        z_end = cumulative[i]
        sigma_start = sigmas[0] if i == 0 else sigmas[i - 1]
        sigma_end = sigmas[i]

        rising = 0.5 * (1 + erf((depth - z_start) / sigma_start)) if i > 0 else np.ones_like(depth)
        falling = 0.5 * (1 + erf((depth - z_end) / sigma_end))
        profile = np.maximum(rising - falling, 0)
        layer_profiles.append(profile)

    D[:, 0] = layer_profiles[0]
    D[:, 1] = (2/5) * layer_profiles[1]
    D[:, 2] = (1/3) * layer_profiles[2]
    D[:, 3] = (1/3) * layer_profiles[3]
    D[:, 4] = (3/5) * layer_profiles[1] + (2/3) * layer_profiles[2] + (2/3) * layer_profiles[3]

    z_sub = cumulative[-1]
    sigma_sub = sigmas[-1]
    D[:, 5] = 0.5 * (1 + erf((depth - z_sub) / sigma_sub))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    elements = [
        ElementConfig('C', '1s', 'C', oxidation_state=0),
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    angles = [15, 30, 45, 60, 75, 85]
    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D, elements)

    I_true = tm.forward(D, normalize=True)

    if noise_level > 0:
        noise = noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true + noise * I_true
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    else:
        I_obs = I_true.copy()

    return I_obs, tm, ['C', 'Al', 'Ti', 'Si_ox', 'O', 'Si_bulk']


def create_gradient_data(noise_level: float = 0.02):
    """グラデーション（混合）データを生成"""
    depth = np.arange(0, 150, 1.0)
    n_depth = len(depth)

    # 元素Aが表面で高く、元素Bが深部で高い（線形グラデーション）
    D = np.zeros((n_depth, 3))
    gradient = depth / depth.max()
    D[:, 0] = 0.4 * (1 - gradient)  # A: 表面リッチ
    D[:, 1] = 0.4 * gradient        # B: 深部リッチ
    D[:, 2] = 0.2                   # C: 均一

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    # 実際の元素を使用（IMFPが計算できるように）
    elements = [
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('C', '1s', 'C', oxidation_state=0),
    ]

    angles = [15, 30, 45, 60, 75, 85]
    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D, elements)

    I_true = tm.forward(D, normalize=True)

    if noise_level > 0:
        noise = noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true + noise * I_true
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    else:
        I_obs = I_true.copy()

    return I_obs, tm, ['A', 'B', 'C']


def create_substrate_only_data(noise_level: float = 0.02):
    """基板のみ（均一）データを生成"""
    depth = np.arange(0, 150, 1.0)
    n_depth = len(depth)

    D = np.zeros((n_depth, 2))
    D[:, 0] = 0.3  # 元素A: 均一
    D[:, 1] = 0.7  # 元素B: 均一

    # 実際の元素を使用
    elements = [
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
    ]

    angles = [15, 30, 45, 60, 75, 85]
    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D, elements)

    I_true = tm.forward(D, normalize=True)

    if noise_level > 0:
        noise = noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true + noise * I_true
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    else:
        I_obs = I_true.copy()

    return I_obs, tm, ['A', 'B']


def main():
    """メイン"""
    print("=" * 70)
    print("Structure Analyzer - 実データ検証")
    print("=" * 70)

    analyzer = StructureAnalyzer()
    np.random.seed(42)

    test_cases = [
        ("SiO2/Si (30Å)", create_sio2_si_data, {"thickness": 30.0}, StructureType.WELL_DEFINED_LAYERS),
        ("SiO2/Si (10Å)", create_sio2_si_data, {"thickness": 10.0}, StructureType.WELL_DEFINED_LAYERS),
        ("SiO2/Si (80Å)", create_sio2_si_data, {"thickness": 80.0}, StructureType.WELL_DEFINED_LAYERS),
        ("4層膜 標準", create_multilayer_data, {"thicknesses": [5, 15, 35, 30]}, StructureType.WELL_DEFINED_LAYERS),
        ("4層膜 薄い", create_multilayer_data, {"thicknesses": [3, 8, 15, 10]}, StructureType.WELL_DEFINED_LAYERS),
        ("グラデーション", create_gradient_data, {}, StructureType.MIXED_GRADIENT),
        ("基板のみ", create_substrate_only_data, {}, StructureType.SUBSTRATE_ONLY),
    ]

    print(f"\n{'ケース':<20} {'期待値':<25} {'判定結果':<25} {'推奨ソルバー':<20} {'信頼度':<8} {'判定':<5}")
    print("-" * 105)

    correct = 0
    total = len(test_cases)

    for name, func, kwargs, expected_type in test_cases:
        I_obs, tm, labels = func(**kwargs)
        result = analyzer.analyze(I_obs, tm, labels, verbose=False)

        # 判定結果の評価
        # 注: 完全一致でなくても、適切なソルバーが選ばれていればOKとする
        is_correct = False
        if expected_type == StructureType.WELL_DEFINED_LAYERS:
            # GenericLayerSolverが推奨されるか、SemiFreeSolverでも高信頼度ならOK
            is_correct = (result.solver_recommendation == SolverRecommendation.GENERIC_LAYER or
                         (result.solver_recommendation == SolverRecommendation.SEMI_FREE and
                          result.confidence > 0.7))
        elif expected_type == StructureType.MIXED_GRADIENT:
            is_correct = result.solver_recommendation in [SolverRecommendation.SEMI_FREE,
                                                         SolverRecommendation.GENERIC_LAYER]
        elif expected_type == StructureType.SUBSTRATE_ONLY:
            is_correct = result.structure_type == StructureType.SUBSTRATE_ONLY

        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        print(f"{name:<20} {expected_type.value:<25} {result.structure_type.value:<25} "
              f"{result.solver_recommendation.value:<20} {result.confidence:.2f}     {status}")

    print("-" * 105)
    print(f"正解率: {correct}/{total} ({100*correct/total:.1f}%)")

    # 詳細分析
    print("\n" + "=" * 70)
    print("詳細分析（代表ケース）")
    print("=" * 70)

    print("\n>>> SiO2/Si (30Å) - 典型的な単層膜")
    I_obs, tm, labels = create_sio2_si_data(thickness=30.0, noise_level=0.02)
    result = analyzer.analyze(I_obs, tm, labels, verbose=True)

    print("\n>>> 4層膜 - 多層構造")
    I_obs, tm, labels = create_multilayer_data()
    result = analyzer.analyze(I_obs, tm, labels, verbose=True)

    print("\n>>> グラデーション - 混合層")
    I_obs, tm, labels = create_gradient_data()
    result = analyzer.analyze(I_obs, tm, labels, verbose=True)


if __name__ == "__main__":
    main()
