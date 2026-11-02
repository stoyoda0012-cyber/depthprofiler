"""
SemiFreeSolver Test Suite
=========================

Tests for SemiFreeSolver (semi-parametric depth profile reconstruction).

Usage:
    pytest tests/solver/test_semifree_solver.py -v
    python -m deppro.tests.solver.test_semifree_solver
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

    # C layer
    D[:, 0] = 0.5 * (1 - erf((depth - z[1]) / sigma[0]))

    # Al2O3 layer
    al2o3 = 0.5 * (1 + erf((depth - z[1]) / sigma[0])) * 0.5 * (1 - erf((depth - z[2]) / sigma[1]))
    D[:, 1] = (2/5) * al2o3
    D[:, 4] = (3/5) * al2o3

    # TiO2 layer
    tio2 = 0.5 * (1 + erf((depth - z[2]) / sigma[1])) * 0.5 * (1 - erf((depth - z[3]) / sigma[2]))
    D[:, 2] = (1/3) * tio2
    D[:, 4] += (2/3) * tio2

    # SiO2 layer
    sio2 = 0.5 * (1 + erf((depth - z[3]) / sigma[2])) * 0.5 * (1 - erf((depth - z[4]) / sigma[3]))
    D[:, 3] = (1/3) * sio2
    D[:, 4] += (2/3) * sio2

    # Si substrate
    D[:, 5] = 0.5 * (1 + erf((depth - z[4]) / sigma[3]))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


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
    depth = np.arange(0, 100, 4.0)  # ~100 parameters
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

    tm = TransferMatrix(depth=depth, step_size=4.0, angles=angles, photon_energy=1486.6)
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


def test_semifree_lls():
    """Test SemiFreeSolver with LLS method."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    I_obs, tm, elements, D_true, thicknesses = create_test_data()
    true_total = sum(thicknesses)

    semi_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
    )

    solver_config = SolverConfig(method='LLS', alpha=0.0, scipy_method='SLSQP')
    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)

    result = solver.solve(I_obs, verbose=False)

    thickness_error = abs(result.thickness - true_total)
    assert thickness_error < 15.0, f"Thickness error should be < 15.0 Å, got {thickness_error:.2f}"


def test_semifree_l2():
    """Test SemiFreeSolver with L2 regularization."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    I_obs, tm, elements, D_true, thicknesses = create_test_data()
    true_total = sum(thicknesses)

    semi_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
    )

    solver_config = SolverConfig(method='L2', alpha=0.001, scipy_method='SLSQP')
    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)

    result = solver.solve(I_obs, verbose=False)

    thickness_error = abs(result.thickness - true_total)
    assert thickness_error < 15.0, f"Thickness error should be < 15.0 Å, got {thickness_error:.2f}"


def test_semifree_method_comparison():
    """Compare LLS, L2, L1, MEM methods."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    I_obs, tm, elements, D_true, thicknesses = create_test_data()
    true_total = sum(thicknesses)

    methods = [
        ('LLS', 0.0),
        ('L2', 0.001),
        ('L1', 0.001),
        ('MEM', 0.001),
    ]

    for method, alpha in methods:
        semi_config = SemiFreeSolverConfig(
            thickness_init=70.0,
            abruptness_init=5.0,
            free_indices=[0, 1, 2, 3],
            linked_indices=[4],
            substrate_indices=[5],
        )

        solver_config = SolverConfig(method=method, alpha=alpha, scipy_method='SLSQP')
        solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)

        result = solver.solve(I_obs, verbose=False)
        thickness_error = abs(result.thickness - true_total)

        assert thickness_error < 15.0, f"{method}: Thickness error should be < 15.0 Å, got {thickness_error:.2f}"


def test_semifree_noise_robustness():
    """Test noise robustness."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    noise_levels = [0.0, 0.02, 0.05]

    for noise in noise_levels:
        I_obs, tm, elements, D_true, thicknesses = create_test_data(noise_level=noise)
        true_total = sum(thicknesses)

        semi_config = SemiFreeSolverConfig(
            thickness_init=70.0,
            abruptness_init=5.0,
            free_indices=[0, 1, 2, 3],
            linked_indices=[4],
            substrate_indices=[5],
        )

        solver_config = SolverConfig(method='LLS', alpha=0.0, scipy_method='SLSQP')
        solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)

        result = solver.solve(I_obs, verbose=False)
        thickness_error = abs(result.thickness - true_total)

        # Allow larger error for higher noise
        max_error = 15.0 + noise * 200  # e.g., 15.0 for 0%, 19.0 for 2%, 25.0 for 5%
        assert thickness_error < max_error, f"Noise {noise*100:.0f}%: error {thickness_error:.2f} > {max_error:.1f}"


def run_all_tests():
    """Run all tests (standalone mode)."""
    print("=" * 60)
    print("SemiFreeSolver Test Suite")
    print("=" * 60)
    print()

    test_semifree_lls()
    print("Test 1: LLS ... PASSED")

    test_semifree_l2()
    print("Test 2: L2 ... PASSED")

    test_semifree_method_comparison()
    print("Test 3: Method comparison ... PASSED")

    test_semifree_noise_robustness()
    print("Test 4: Noise robustness ... PASSED")

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
