"""
GenericLayerSolver Test Suite
=============================

Tests for GenericLayerSolver (parametric multilayer film analysis).

Usage:
    pytest tests/solver/test_generic_layer_solver.py -v
    python -m deppro.tests.solver.test_generic_layer_solver
"""

import numpy as np
from scipy.special import erf
from typing import List


def create_4layer_profile(
    depth: np.ndarray,
    thicknesses: List[float] = [5.0, 15.0, 35.0, 30.0],
    interface_widths: List[float] = [2.0, 3.0, 4.0, 5.0],
) -> np.ndarray:
    """Create 4-layer profile: C / Al2O3 / TiO2 / SiO2 / Si"""
    sigma = [w / (2 * np.sqrt(np.log(2))) for w in interface_widths]
    z = [0] + list(np.cumsum(thicknesses))

    D = np.zeros((len(depth), 6))

    D[:, 0] = 0.5 * (1 - erf((depth - z[1]) / sigma[0]))

    al2o3 = 0.5 * (1 + erf((depth - z[1]) / sigma[0])) * 0.5 * (1 - erf((depth - z[2]) / sigma[1]))
    D[:, 1] = (2/5) * al2o3
    D[:, 4] = (3/5) * al2o3

    tio2 = 0.5 * (1 + erf((depth - z[2]) / sigma[1])) * 0.5 * (1 - erf((depth - z[3]) / sigma[2]))
    D[:, 2] = (1/3) * tio2
    D[:, 4] += (2/3) * tio2

    sio2 = 0.5 * (1 + erf((depth - z[3]) / sigma[2])) * 0.5 * (1 - erf((depth - z[4]) / sigma[3]))
    D[:, 3] = (1/3) * sio2
    D[:, 4] += (2/3) * sio2

    D[:, 5] = 0.5 * (1 + erf((depth - z[4]) / sigma[3]))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def create_layer_config(thicknesses_init: List[float] = None):
    """Create GenericLayerConfig for 4-layer model."""
    from deppro.inverse_solver import GenericLayerConfig, LayerDefinition

    if thicknesses_init is None:
        thicknesses_init = [10.0, 20.0, 40.0, 35.0]

    # Elements: [C, Al, Ti, Si_ox, O, Si_bulk] = [0, 1, 2, 3, 4, 5]
    layers = [
        LayerDefinition(
            name='C',
            element_indices=[0],
            stoichiometry={0: 1.0},
        ),
        LayerDefinition(
            name='Al2O3',
            element_indices=[1, 4],
            stoichiometry={1: 2.0, 4: 3.0},  # Al2O3
        ),
        LayerDefinition(
            name='TiO2',
            element_indices=[2, 4],
            stoichiometry={2: 1.0, 4: 2.0},  # TiO2
        ),
        LayerDefinition(
            name='SiO2',
            element_indices=[3, 4],
            stoichiometry={3: 1.0, 4: 2.0},  # SiO2
        ),
    ]

    substrate = LayerDefinition(
        name='Si',
        element_indices=[5],
        stoichiometry={5: 1.0},
    )

    return GenericLayerConfig(
        layers=layers,
        substrate=substrate,
        initial_thicknesses=thicknesses_init,
        initial_widths=[2.0, 3.0, 4.0, 5.0],
    )


def create_test_data(
    thicknesses: List[float] = [5.0, 15.0, 35.0, 30.0],
    interface_widths: List[float] = [2.0, 3.0, 4.0, 5.0],
    noise_level: float = 0.02,
    seed: int = 42,
):
    """Create test data."""
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    np.random.seed(seed)
    depth = np.arange(0, 100, 1.0)
    angles = [0, 10, 20, 30, 40, 50, 60]

    elements = [
        ElementConfig('C', '1s', 'C', oxidation_state=0),
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_4layer_profile(depth, thicknesses, interface_widths)

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

    return I_obs, tm, elements, D_true, thicknesses


def test_generic_layer_lls():
    """Test GenericLayerSolver with LLS method."""
    from deppro.inverse_solver import GenericLayerSolver, SolverConfig

    I_obs, tm, elements, D_true, true_thicknesses = create_test_data()
    layer_config = create_layer_config()

    solver_config = SolverConfig(method='LLS', alpha=0.0)
    solver = GenericLayerSolver(tm, solver_config, layer_config, elements=elements)

    result = solver.solve(I_obs, verbose=False)

    # Check layer thicknesses
    for i, (est, true) in enumerate(zip(result.layer_thicknesses, true_thicknesses)):
        error = abs(est - true)
        assert error < 2.0, f"Layer {i} error {error:.2f} Å > 2.0 Å"


def test_generic_layer_initial_robustness():
    """Test robustness to initial value perturbations."""
    from deppro.inverse_solver import GenericLayerSolver, SolverConfig

    I_obs, tm, elements, D_true, true_thicknesses = create_test_data()

    # Test with 5 Å offset
    init_with_offset = [t + 5.0 for t in true_thicknesses]
    layer_config = create_layer_config(init_with_offset)

    solver_config = SolverConfig(method='LLS', alpha=0.0)
    solver = GenericLayerSolver(tm, solver_config, layer_config, elements=elements)

    result = solver.solve(I_obs, verbose=False)

    total_true = sum(true_thicknesses)
    total_est = sum(result.layer_thicknesses)
    total_error = abs(total_est - total_true)

    assert total_error < 1.0, f"Total thickness error {total_error:.2f} Å > 1.0 Å"


def test_generic_layer_method_comparison():
    """Compare LLS, L2, L1, MEM methods."""
    from deppro.inverse_solver import GenericLayerSolver, SolverConfig

    I_obs, tm, elements, D_true, true_thicknesses = create_test_data()

    methods = [
        ('LLS', 0.0),
        ('L2', 0.01),
        ('L1', 0.01),
        ('MEM', 0.01),
    ]

    for method, alpha in methods:
        layer_config = create_layer_config()
        solver_config = SolverConfig(method=method, alpha=alpha)
        solver = GenericLayerSolver(tm, solver_config, layer_config, elements=elements)

        result = solver.solve(I_obs, verbose=False)

        total_true = sum(true_thicknesses)
        total_est = sum(result.layer_thicknesses)
        total_error = abs(total_est - total_true)

        assert total_error < 3.0, f"{method}: Total error {total_error:.2f} Å > 3.0 Å"


def test_generic_layer_noise_robustness():
    """Test noise robustness."""
    from deppro.inverse_solver import GenericLayerSolver, SolverConfig

    noise_levels = [0.0, 0.02, 0.05]

    for noise in noise_levels:
        I_obs, tm, elements, D_true, true_thicknesses = create_test_data(noise_level=noise)
        layer_config = create_layer_config()

        solver_config = SolverConfig(method='LLS', alpha=0.0)
        solver = GenericLayerSolver(tm, solver_config, layer_config, elements=elements)

        result = solver.solve(I_obs, verbose=False)

        total_true = sum(true_thicknesses)
        total_est = sum(result.layer_thicknesses)
        total_error = abs(total_est - total_true)

        max_error = 1.0 + noise * 20
        assert total_error < max_error, f"Noise {noise*100:.0f}%: error {total_error:.2f} > {max_error:.1f}"


def run_all_tests():
    """Run all tests (standalone mode)."""
    print("=" * 60)
    print("GenericLayerSolver Test Suite")
    print("=" * 60)
    print()

    test_generic_layer_lls()
    print("Test 1: LLS ... PASSED")

    test_generic_layer_initial_robustness()
    print("Test 2: Initial value robustness ... PASSED")

    test_generic_layer_method_comparison()
    print("Test 3: Method comparison ... PASSED")

    test_generic_layer_noise_robustness()
    print("Test 4: Noise robustness ... PASSED")

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
