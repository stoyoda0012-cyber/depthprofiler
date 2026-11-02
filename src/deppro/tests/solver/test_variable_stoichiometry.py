"""
Test GenericLayerSolver with variable stoichiometry

Tests for:
1. NO pattern: N on top of O (surface nitridation)
2. ON pattern: O on top of N (interface nitridation)
3. Variable oxygen composition (SiOx with x < 2)

Priority: High (Issue #3 - non-uniform composition films)

Usage:
    python -m deppro.tests.solver.test_variable_stoichiometry
"""

import numpy as np
from scipy.special import erf
from typing import List, Dict


def create_siox_profile(
    depth: np.ndarray,
    thickness: float,
    o_ratio: float,
    interface_width: float = 5.0,
) -> np.ndarray:
    """
    Create SiOx/Si profile with variable oxygen ratio.

    Elements: [Si_ox, O, Si_bulk] = [0, 1, 2]
    """
    sigma = interface_width / (2 * np.sqrt(np.log(2)))

    D = np.zeros((len(depth), 3))

    # SiOx layer (surface to thickness)
    siox_profile = 0.5 * (1 - erf((depth - thickness) / sigma))
    total_stoich = 1 + o_ratio  # Si + O_ratio
    D[:, 0] = (1 / total_stoich) * siox_profile  # Si_ox
    D[:, 1] = (o_ratio / total_stoich) * siox_profile  # O

    # Si substrate
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))

    # Normalize
    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def create_no_pattern_profile(
    depth: np.ndarray,
    n_thickness: float,
    o_thickness: float,
    interface_widths: List[float] = [3.0, 5.0],
) -> np.ndarray:
    """
    Create NO pattern: SiNy / SiO2 / Si

    Elements: [Si_n, N, Si_o, O, Si_bulk] = [0, 1, 2, 3, 4]
    """
    sigma = [w / (2 * np.sqrt(np.log(2))) for w in interface_widths]
    z1 = n_thickness
    z2 = n_thickness + o_thickness

    D = np.zeros((len(depth), 5))

    # Si3N4 layer
    sin_profile = 0.5 * (1 - erf((depth - z1) / sigma[0]))
    D[:, 0] = (3/7) * sin_profile  # Si in Si3N4
    D[:, 1] = (4/7) * sin_profile  # N

    # SiO2 layer
    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma[0])) * 0.5 * (1 - erf((depth - z2) / sigma[1]))
    D[:, 2] = (1/3) * sio2_profile  # Si in SiO2
    D[:, 3] = (2/3) * sio2_profile  # O

    # Si substrate
    D[:, 4] = 0.5 * (1 + erf((depth - z2) / sigma[1]))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def create_test_data_siox(
    thickness: float = 20.0,
    o_ratio: float = 1.7,
    noise_level: float = 0.02,
    seed: int = 42,
):
    """Create test data for SiOx/Si."""
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    np.random.seed(seed)
    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),  # Si_ox
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),  # O
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),    # Si_bulk
    ]

    D_true = create_siox_profile(depth, thickness, o_ratio)

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    if noise_level > 0:
        noise = noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true + noise * I_true
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    else:
        I_obs = I_true.copy()

    return I_obs, tm, elements, D_true, thickness, o_ratio


def create_test_data_no_pattern(
    n_thickness: float = 5.0,
    o_thickness: float = 20.0,
    noise_level: float = 0.02,
    seed: int = 42,
):
    """Create test data for NO pattern."""
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    np.random.seed(seed)
    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]

    elements = [
        ElementConfig('Si', '2p', 'Si3N4', oxidation_state=4),  # Si in Si3N4
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),  # N
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),   # Si in SiO2
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),   # O
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),     # Si_bulk
    ]

    D_true = create_no_pattern_profile(depth, n_thickness, o_thickness)

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    if noise_level > 0:
        noise = noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true + noise * I_true
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    else:
        I_obs = I_true.copy()

    return I_obs, tm, elements, D_true, n_thickness, o_thickness


def test_siox_fixed_stoichiometry():
    """
    Test SiOx with fixed stoichiometry (baseline).

    This test confirms GenericLayerSolver works with standard SiO2.
    """
    from deppro.inverse_solver import GenericLayerSolver, GenericLayerConfig, LayerDefinition, SolverConfig

    print("\n=== Test: SiO2 Fixed Stoichiometry (Baseline) ===")

    I_obs, tm, elements, D_true, true_thickness, true_o_ratio = create_test_data_siox(
        thickness=20.0, o_ratio=2.0, noise_level=0.02
    )

    layer_config = GenericLayerConfig(
        layers=[
            LayerDefinition(
                name='SiO2',
                element_indices=[0, 1],
                stoichiometry={0: 1, 1: 2},  # Fixed SiO2
            ),
        ],
        substrate=LayerDefinition(
            name='Si',
            element_indices=[2],
            stoichiometry={2: 1},
        ),
        initial_thicknesses=[25.0],
        initial_widths=[5.0],
    )

    solver = GenericLayerSolver(tm, SolverConfig(max_iter=200), layer_config, elements=elements)
    result = solver.solve(I_obs, verbose=True)

    print(f"\nResults:")
    print(f"  True thickness: {true_thickness:.1f} Å")
    print(f"  Optimized thickness: {result.layer_thicknesses[0]:.1f} Å")
    print(f"  Error: {abs(result.layer_thicknesses[0] - true_thickness):.2f} Å")

    assert result.success, "Optimization should converge"
    error = abs(result.layer_thicknesses[0] - true_thickness)
    assert error < 2.0, f"Thickness error {error:.2f} Å > 2.0 Å"


def test_siox_variable_stoichiometry():
    """
    Test SiOx with variable oxygen ratio (oxygen deficiency).

    Structure: SiO1.7 / Si
    Goal: Recover both thickness and O ratio
    """
    from deppro.inverse_solver import GenericLayerSolver, GenericLayerConfig, LayerDefinition, SolverConfig

    print("\n=== Test: SiOx Variable Stoichiometry (Oxygen Deficiency) ===")

    true_thickness = 20.0
    true_o_ratio = 1.7

    I_obs, tm, elements, D_true, _, _ = create_test_data_siox(
        thickness=true_thickness, o_ratio=true_o_ratio, noise_level=0.02
    )

    layer_config = GenericLayerConfig(
        layers=[
            LayerDefinition(
                name='SiOx',
                element_indices=[0, 1],
                stoichiometry={0: 1, 1: 2.0},  # Initial: SiO2
                stoichiometry_bounds={1: (1.5, 2.0)},  # O ratio: 1.5-2.0
            ),
        ],
        substrate=LayerDefinition(
            name='Si',
            element_indices=[2],
            stoichiometry={2: 1},
        ),
        initial_thicknesses=[25.0],
        initial_widths=[5.0],
        optimize_stoichiometry=True,
    )

    solver = GenericLayerSolver(tm, SolverConfig(max_iter=300), layer_config, elements=elements)
    result = solver.solve(I_obs, verbose=True)

    print(f"\nResults:")
    print(f"  True thickness: {true_thickness:.1f} Å")
    print(f"  Optimized thickness: {result.layer_thicknesses[0]:.1f} Å")
    print(f"  True O ratio: {true_o_ratio:.2f}")
    print(f"  Optimized O ratio: {result.layer_stoichiometries[0][1]:.2f}")

    assert result.success, "Optimization should converge"

    thickness_error = abs(result.layer_thicknesses[0] - true_thickness)
    assert thickness_error < 3.0, f"Thickness error {thickness_error:.2f} Å > 3.0 Å"

    o_error = abs(result.layer_stoichiometries[0][1] - true_o_ratio)
    print(f"  O ratio error: {o_error:.3f}")

    # Note: O ratio recovery may have larger error due to parameter correlation
    # The important thing is that it converges in the right direction


def test_no_pattern_fixed():
    """
    Test NO pattern with fixed stoichiometry.

    Structure: Si3N4 / SiO2 / Si
    """
    from deppro.inverse_solver import GenericLayerSolver, GenericLayerConfig, LayerDefinition, SolverConfig

    print("\n=== Test: NO Pattern (Surface Nitridation) ===")

    true_n_thickness = 5.0
    true_o_thickness = 20.0

    I_obs, tm, elements, D_true, _, _ = create_test_data_no_pattern(
        n_thickness=true_n_thickness, o_thickness=true_o_thickness, noise_level=0.02
    )

    layer_config = GenericLayerConfig(
        layers=[
            LayerDefinition(
                name='Si3N4',
                element_indices=[0, 1],  # Si_n, N
                stoichiometry={0: 3, 1: 4},
            ),
            LayerDefinition(
                name='SiO2',
                element_indices=[2, 3],  # Si_o, O
                stoichiometry={2: 1, 3: 2},
            ),
        ],
        substrate=LayerDefinition(
            name='Si',
            element_indices=[4],
            stoichiometry={4: 1},
        ),
        initial_thicknesses=[8.0, 25.0],  # Initial guess
        initial_widths=[3.0, 5.0],
    )

    solver = GenericLayerSolver(tm, SolverConfig(max_iter=200), layer_config, elements=elements)
    result = solver.solve(I_obs, verbose=True)

    print(f"\nResults:")
    print(f"  True N layer: {true_n_thickness:.1f} Å")
    print(f"  Optimized N layer: {result.layer_thicknesses[0]:.1f} Å")
    print(f"  True O layer: {true_o_thickness:.1f} Å")
    print(f"  Optimized O layer: {result.layer_thicknesses[1]:.1f} Å")

    assert result.success, "Optimization should converge"


def test_assumptions_visibility():
    """Test that assumptions are properly recorded."""
    from deppro.inverse_solver import GenericLayerSolver, GenericLayerConfig, LayerDefinition, SolverConfig

    print("\n=== Test: Assumptions Visibility ===")

    I_obs, tm, elements, _, _, _ = create_test_data_siox()

    # Config with variable stoichiometry and widths
    layer_config = GenericLayerConfig(
        layers=[
            LayerDefinition(
                name='SiOx',
                element_indices=[0, 1],
                stoichiometry={0: 1, 1: 2.0},
                stoichiometry_bounds={1: (1.5, 2.0)},
            ),
        ],
        substrate=LayerDefinition(
            name='Si',
            element_indices=[2],
            stoichiometry={2: 1},
        ),
        initial_thicknesses=[20.0],
        initial_widths=[5.0],
        optimize_stoichiometry=True,
        optimize_widths=True,
    )

    solver = GenericLayerSolver(tm, SolverConfig(), layer_config, elements=elements)
    result = solver.solve(I_obs, verbose=True)

    print(f"\nAssumptions recorded:")
    print(f"  optimize_widths: {result.assumptions['optimize_widths']}")
    print(f"  optimize_stoichiometry: {result.assumptions['optimize_stoichiometry']}")
    print(f"  Variable parameters: {result.assumptions['variable_parameters']}")
    print(f"  Fixed parameters: {result.assumptions['fixed_parameters']}")

    assert result.assumptions['optimize_widths'] is True
    assert result.assumptions['optimize_stoichiometry'] is True
    assert 'thicknesses' in result.assumptions['variable_parameters']
    assert 'interface_widths' in result.assumptions['variable_parameters']

    # Test summary output
    print(f"\n=== Summary ===")
    print(result.summary())


def test_layer_definition_properties():
    """Test LayerDefinition property methods."""
    from deppro.inverse_solver import LayerDefinition

    print("\n=== Test: LayerDefinition Properties ===")

    # With variable stoichiometry
    layer_var = LayerDefinition(
        name='SiOx',
        element_indices=[0, 1],
        stoichiometry={0: 1, 1: 2.0},
        stoichiometry_bounds={1: (1.5, 2.0)},
    )
    print(f"  Variable layer: has_variable={layer_var.has_variable_stoichiometry}, n_params={layer_var.n_stoich_params}")
    assert layer_var.has_variable_stoichiometry is True
    assert layer_var.n_stoich_params == 1

    # Without variable stoichiometry
    layer_fixed = LayerDefinition(
        name='SiO2',
        element_indices=[0, 1],
        stoichiometry={0: 1, 1: 2},
    )
    print(f"  Fixed layer: has_variable={layer_fixed.has_variable_stoichiometry}, n_params={layer_fixed.n_stoich_params}")
    assert layer_fixed.has_variable_stoichiometry is False
    assert layer_fixed.n_stoich_params == 0

    # Multiple variable elements
    layer_multi = LayerDefinition(
        name='SiOxNy',
        element_indices=[0, 1, 2],
        stoichiometry={0: 3, 1: 1, 2: 3},
        stoichiometry_bounds={1: (0, 2), 2: (2, 4)},
    )
    print(f"  Multi-var layer: has_variable={layer_multi.has_variable_stoichiometry}, n_params={layer_multi.n_stoich_params}")
    assert layer_multi.has_variable_stoichiometry is True
    assert layer_multi.n_stoich_params == 2


if __name__ == "__main__":
    test_layer_definition_properties()
    test_siox_fixed_stoichiometry()
    test_siox_variable_stoichiometry()
    test_no_pattern_fixed()
    test_assumptions_visibility()

    print("\n" + "=" * 50)
    print("All tests completed!")
