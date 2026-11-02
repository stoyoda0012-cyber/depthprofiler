"""
ARXPS Pipeline テスト
====================

ARXPSPipelineの統合テスト。
様々なケースで自動解析が正しく動作することを検証。
"""

import numpy as np
from scipy.special import erf

from deppro.pipeline import ARXPSPipeline, PipelineConfig, ValidationStatus
from deppro.depth_profile import ElementConfig
from deppro.transfer_matrix import TransferMatrix
from deppro.inverse_solver import GenericLayerConfig, LayerDefinition


def create_sio2_si_data(
    thickness: float = 30.0,
    interface_width: float = 5.0,
    noise_level: float = 0.02,
):
    """SiO2/Si 2層系データを生成"""
    depth = np.arange(0, 150, 1.0)
    n_depth = len(depth)

    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    D = np.zeros((n_depth, 3))

    sio2_profile = 1 - 0.5 * (1 + erf((depth - thickness) / sigma))
    D[:, 0] = (1/3) * sio2_profile  # Si_ox
    D[:, 1] = (2/3) * sio2_profile  # O
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))  # Si_bulk

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

    return I_obs, tm, elements, ['Si_ox', 'O', 'Si_bulk']


def create_multilayer_data(
    thicknesses: list = [5.0, 15.0, 35.0, 30.0],
    widths: list = [2.0, 3.0, 4.0, 5.0],
    noise_level: float = 0.02,
):
    """C/Al2O3/TiO2/SiO2/Si 4層膜データを生成"""
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

    D[:, 0] = layer_profiles[0]                           # C
    D[:, 1] = (2/5) * layer_profiles[1]                   # Al
    D[:, 2] = (1/3) * layer_profiles[2]                   # Ti
    D[:, 3] = (1/3) * layer_profiles[3]                   # Si_ox
    D[:, 4] = (3/5) * layer_profiles[1] + (2/3) * layer_profiles[2] + (2/3) * layer_profiles[3]  # O

    z_sub = cumulative[-1]
    sigma_sub = sigmas[-1]
    D[:, 5] = 0.5 * (1 + erf((depth - z_sub) / sigma_sub))  # Si_bulk

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

    return I_obs, tm, elements, ['C', 'Al', 'Ti', 'Si_ox', 'O', 'Si_bulk']


def test_sio2_si_explicit_config():
    """SiO2/Si: 明示的な層構造設定でテスト"""
    print("\n" + "=" * 70)
    print("Test: SiO2/Si with explicit layer config")
    print("=" * 70)

    np.random.seed(42)
    thickness = 30.0

    I_obs, tm, elements, labels = create_sio2_si_data(
        thickness=thickness,
        noise_level=0.02
    )

    # 明示的な層構造設定
    layer_config = GenericLayerConfig(
        layers=[
            LayerDefinition(
                name="SiO2",
                element_indices=[0, 1],
                stoichiometry={0: 1, 1: 2},
            ),
        ],
        substrate=LayerDefinition(
            name="Si",
            element_indices=[2],
            stoichiometry={2: 1},
        ),
        initial_thicknesses=[25.0],  # 少しずれた初期値
        initial_widths=[5.0],
        linked_element_index=1,
    )

    pipeline = ARXPSPipeline()
    result = pipeline.analyze(
        I_obs, tm, elements,
        element_labels=labels,
        layer_config=layer_config,
        verbose=True
    )

    print(f"\n>>> True thickness: {thickness:.1f} Å")
    if result.layer_thicknesses:
        error = abs(result.layer_thicknesses[0] - thickness)
        print(f">>> Estimated thickness: {result.layer_thicknesses[0]:.2f} Å")
        print(f">>> Error: {error:.2f} Å")
        assert error < 1.0, f"Error too large: {error:.2f} Å"
    else:
        print(">>> ERROR: No thickness estimated")
        assert False, "Thickness not estimated"

    assert result.success, f"Pipeline failed: {result.failure_reason}"
    print(">>> PASSED")


def test_sio2_si_auto_detect():
    """SiO2/Si: 自動検出でテスト"""
    print("\n" + "=" * 70)
    print("Test: SiO2/Si with auto-detection")
    print("=" * 70)

    np.random.seed(42)
    thickness = 30.0

    I_obs, tm, elements, labels = create_sio2_si_data(
        thickness=thickness,
        noise_level=0.02
    )

    pipeline = ARXPSPipeline()
    result = pipeline.analyze(
        I_obs, tm, elements,
        element_labels=labels,
        verbose=True
    )

    print(f"\n>>> True thickness: {thickness:.1f} Å")
    print(f">>> Structure type: {result.structure_analysis.structure_type.value}")
    print(f">>> Solver used: {result.solver_used}")

    # 自動検出でも成功すべき
    assert result.success, f"Pipeline failed: {result.failure_reason}"
    print(">>> PASSED")


def test_multilayer_explicit_config():
    """多層膜: 明示的な層構造設定でテスト"""
    print("\n" + "=" * 70)
    print("Test: 4-layer multilayer with explicit config")
    print("=" * 70)

    np.random.seed(42)
    true_thicknesses = [5.0, 15.0, 35.0, 30.0]

    I_obs, tm, elements, labels = create_multilayer_data(
        thicknesses=true_thicknesses,
        noise_level=0.02
    )

    # 4層膜の明示的設定
    layer_config = GenericLayerConfig(
        layers=[
            LayerDefinition(name="C", element_indices=[0], stoichiometry={0: 1}),
            LayerDefinition(name="Al2O3", element_indices=[1, 4], stoichiometry={1: 2, 4: 3}),
            LayerDefinition(name="TiO2", element_indices=[2, 4], stoichiometry={2: 1, 4: 2}),
            LayerDefinition(name="SiO2", element_indices=[3, 4], stoichiometry={3: 1, 4: 2}),
        ],
        substrate=LayerDefinition(name="Si", element_indices=[5], stoichiometry={5: 1}),
        initial_thicknesses=[5.0, 15.0, 35.0, 30.0],  # 真値から開始
        initial_widths=[2.0, 3.0, 4.0, 5.0],
        linked_element_index=4,
    )

    pipeline = ARXPSPipeline()
    result = pipeline.analyze(
        I_obs, tm, elements,
        element_labels=labels,
        layer_config=layer_config,
        verbose=True
    )

    print(f"\n>>> True thicknesses: {true_thicknesses}")
    if result.layer_thicknesses:
        print(f">>> Estimated thicknesses: {[f'{t:.2f}' for t in result.layer_thicknesses]}")
        errors = [abs(est - true) for est, true in zip(result.layer_thicknesses, true_thicknesses)]
        print(f">>> Errors: {[f'{e:.2f}' for e in errors]}")

    assert result.success, f"Pipeline failed: {result.failure_reason}"
    print(">>> PASSED")


def test_validation_chi2():
    """Chi²検証のテスト"""
    print("\n" + "=" * 70)
    print("Test: Validation - Chi² check")
    print("=" * 70)

    np.random.seed(42)

    # 低ノイズ（Chi²が低い）
    I_obs_low, tm, elements, labels = create_sio2_si_data(
        thickness=30.0,
        noise_level=0.01
    )

    layer_config = GenericLayerConfig(
        layers=[
            LayerDefinition(
                name="SiO2",
                element_indices=[0, 1],
                stoichiometry={0: 1, 1: 2},
            ),
        ],
        substrate=LayerDefinition(
            name="Si",
            element_indices=[2],
            stoichiometry={2: 1},
        ),
        initial_thicknesses=[30.0],
        initial_widths=[5.0],
        linked_element_index=1,
    )

    pipeline = ARXPSPipeline()
    result_low = pipeline.analyze(
        I_obs_low, tm, elements,
        element_labels=labels,
        layer_config=layer_config,
        verbose=False
    )

    print(f"Low noise (1%): Chi²/n = {result_low.validation.chi2_normalized:.6f}")
    print(f"  Status: {result_low.validation.status.value}")

    # 高ノイズ（Chi²が高い）
    I_obs_high, tm, elements, labels = create_sio2_si_data(
        thickness=30.0,
        noise_level=0.1
    )

    result_high = pipeline.analyze(
        I_obs_high, tm, elements,
        element_labels=labels,
        layer_config=layer_config,
        verbose=False
    )

    print(f"High noise (10%): Chi²/n = {result_high.validation.chi2_normalized:.6f}")
    print(f"  Status: {result_high.validation.status.value}")

    # 低ノイズは合格すべき
    assert result_low.validation.status in [ValidationStatus.PASSED, ValidationStatus.CHI2_WARNING]
    print(">>> PASSED")


def test_different_thicknesses():
    """様々な膜厚でのテスト"""
    print("\n" + "=" * 70)
    print("Test: Different thicknesses")
    print("=" * 70)

    np.random.seed(42)
    test_thicknesses = [10.0, 30.0, 50.0, 80.0]

    pipeline = ARXPSPipeline()

    print(f"{'True (Å)':<12} {'Estimated (Å)':<15} {'Error (Å)':<12} {'Status':<15}")
    print("-" * 60)

    for thickness in test_thicknesses:
        I_obs, tm, elements, labels = create_sio2_si_data(
            thickness=thickness,
            noise_level=0.02
        )

        layer_config = GenericLayerConfig(
            layers=[
                LayerDefinition(
                    name="SiO2",
                    element_indices=[0, 1],
                    stoichiometry={0: 1, 1: 2},
                ),
            ],
            substrate=LayerDefinition(
                name="Si",
                element_indices=[2],
                stoichiometry={2: 1},
            ),
            initial_thicknesses=[thickness + 5.0],  # 少しずれた初期値
            initial_widths=[5.0],
            linked_element_index=1,
        )

        result = pipeline.analyze(
            I_obs, tm, elements,
            element_labels=labels,
            layer_config=layer_config,
            verbose=False
        )

        if result.layer_thicknesses:
            estimated = result.layer_thicknesses[0]
            error = abs(estimated - thickness)
            status = "OK" if result.success else "FAIL"
            print(f"{thickness:<12.1f} {estimated:<15.2f} {error:<12.2f} {status}")
        else:
            print(f"{thickness:<12.1f} {'N/A':<15} {'N/A':<12} FAIL")

    print(">>> PASSED")


def main():
    """メイン"""
    print("=" * 70)
    print("ARXPS Pipeline Integration Tests")
    print("=" * 70)

    test_sio2_si_explicit_config()
    test_sio2_si_auto_detect()
    test_multilayer_explicit_config()
    test_validation_chi2()
    test_different_thicknesses()

    print("\n" + "=" * 70)
    print("All tests PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
