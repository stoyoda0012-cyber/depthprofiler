"""
Narrow Angle Range ARXPS Analysis Benchmark
============================================

Tests for ARXPS with limited angle range (0-20 deg, e.g., near-ambient XPS).
Validates L1 regularization effectiveness.

Usage:
    pytest tests/benchmark/test_narrow_angle.py -v
    python -m deppro.tests.benchmark.test_narrow_angle
"""

import numpy as np
from scipy.special import erf
from typing import List


def build_multilayer_profile(
    depth: np.ndarray,
    layer_thicknesses: List[float],
    interface_widths: List[float],
) -> np.ndarray:
    """Build 4-layer profile: C / Al2O3 / TiO2 / SiO2 / Si"""
    z = depth
    n_depth = len(z)
    D = np.zeros((n_depth, 6))

    cumulative = np.cumsum(layer_thicknesses)
    sigmas = np.array(interface_widths) / (2 * np.sqrt(np.log(2)))

    layer_profiles = []

    for i in range(4):
        z_start = 0 if i == 0 else cumulative[i - 1]
        z_end = cumulative[i]
        sigma_start = sigmas[0] if i == 0 else sigmas[i - 1]
        sigma_end = sigmas[i]

        rising = 0.5 * (1 + erf((z - z_start) / sigma_start)) if i > 0 else np.ones_like(z)
        falling = 0.5 * (1 + erf((z - z_end) / sigma_end))
        profile = np.maximum(rising - falling, 0)
        layer_profiles.append(profile)

    D[:, 0] = layer_profiles[0]
    D[:, 1] = (2/5) * layer_profiles[1]
    D[:, 2] = (1/3) * layer_profiles[2]
    D[:, 3] = (1/3) * layer_profiles[3]
    D[:, 4] = (3/5) * layer_profiles[1] + (2/3) * layer_profiles[2] + (2/3) * layer_profiles[3]

    z_sub = cumulative[-1]
    sigma_sub = sigmas[-1]
    D[:, 5] = 0.5 * (1 + erf((z - z_sub) / sigma_sub))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    return D


def create_test_data(
    layer_thicknesses: List[float] = [5.0, 15.0, 35.0, 30.0],
    interface_widths: List[float] = [2.0, 3.0, 4.0, 5.0],
    angles: List[float] = None,
    noise_level: float = 0.02,
    seed: int = 42,
):
    """Create test data."""
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    if angles is None:
        angles = [0, 5, 10, 15, 20]  # Narrow angle range

    np.random.seed(seed)
    depth = np.arange(0, 100, 4.0)

    elements = [
        ElementConfig('C', '1s', 'C', oxidation_state=0),
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = build_multilayer_profile(depth, layer_thicknesses, interface_widths)

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

    return I_obs, tm, elements, D_true, layer_thicknesses


def test_narrow_angle_generic_layer():
    """Test GenericLayerSolver with narrow angles."""
    from deppro.inverse_solver import GenericLayerSolver, GenericLayerConfig, LayerDefinition, SolverConfig

    I_obs, tm, elements, D_true, true_thicknesses = create_test_data()

    layers = [
        LayerDefinition(name='C', element_indices=[0], stoichiometry={0: 1.0}),
        LayerDefinition(name='Al2O3', element_indices=[1, 4], stoichiometry={1: 2.0, 4: 3.0}),
        LayerDefinition(name='TiO2', element_indices=[2, 4], stoichiometry={2: 1.0, 4: 2.0}),
        LayerDefinition(name='SiO2', element_indices=[3, 4], stoichiometry={3: 1.0, 4: 2.0}),
    ]
    substrate = LayerDefinition(name='Si', element_indices=[5], stoichiometry={5: 1.0})
    layer_config = GenericLayerConfig(
        layers=layers,
        substrate=substrate,
        initial_thicknesses=[10.0, 20.0, 40.0, 35.0],
        initial_widths=[2.0, 3.0, 4.0, 5.0],
    )

    solver_config = SolverConfig(method='LLS', alpha=0.0)
    solver = GenericLayerSolver(tm, solver_config, layer_config, elements=elements)

    result = solver.solve(I_obs, verbose=False)

    total_true = sum(true_thicknesses)
    total_est = sum(result.layer_thicknesses)
    total_error = abs(total_est - total_true)

    # Narrow angle range may have larger errors
    assert total_error < 5.0, f"Total error {total_error:.2f} Å > 5.0 Å"


def test_narrow_angle_semifree():
    """Test SemiFreeSolver with narrow angles."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    I_obs, tm, elements, D_true, true_thicknesses = create_test_data()
    true_total = sum(true_thicknesses)

    semi_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
    )

    # scipy_method must be set explicitly: SemiFreeSolver passes the configured
    # method straight to scipy.optimize.minimize together with its equality
    # constraints, and the SolverConfig default 'L-BFGS-B' cannot take
    # constraints (SciPy warns "Method L-BFGS-B cannot handle constraints." and
    # drops them). With the default this gate solved an unconstrained problem:
    # max |sum_e D - 1| = 0.85 and a 10.35 Å thickness error on this data
    # (2026-10-18). With SLSQP the same data, seed, initial values and
    # threshold give 0.14 Å with the constraints satisfied to 1e-15 in the
    # optimisation variables (the returned D_opt is clipped at min_conc, so
    # its row sums carry a residual of ~5e-6; that is what the assertion below
    # sees).
    solver_config = SolverConfig(method='LLS', alpha=0.0, scipy_method='SLSQP')
    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)

    result = solver.solve(I_obs, verbose=False)

    # The gate judges a *constrained* solution: the concentration sum must hold
    # at the returned profile (the solver's eq. constraints). An unconstrained
    # solve cannot pass this, whatever thickness it reports.
    row_sum_residual = float(np.max(np.abs(result.D_opt.sum(axis=1) - 1.0)))
    assert row_sum_residual < 1e-4, (
        f"concentration-sum constraint not satisfied: max |sum - 1| = {row_sum_residual:.2e}"
    )

    # A solve that stopped at max_iter (success=False) is not a converged
    # constrained solution either; the gate requires convergence.
    assert result.success, f"solver did not converge: status {result.opt_status} ({result.opt_message})"

    thickness_error = abs(result.thickness - true_total)
    assert thickness_error < 5.0, f"Thickness error {thickness_error:.2f} Å > 5.0 Å"


def test_narrow_vs_wide_angle():
    """Compare narrow angle vs wide angle performance."""
    from deppro.inverse_solver import GenericLayerSolver, GenericLayerConfig, LayerDefinition, SolverConfig

    true_thicknesses = [5.0, 15.0, 35.0, 30.0]
    true_total = sum(true_thicknesses)

    # Narrow angle
    I_narrow, tm_narrow, elements, _, _ = create_test_data(angles=[0, 5, 10, 15, 20])

    # Wide angle
    I_wide, tm_wide, _, _, _ = create_test_data(angles=[0, 15, 30, 45, 60, 75])

    layers = [
        LayerDefinition(name='C', element_indices=[0], stoichiometry={0: 1.0}),
        LayerDefinition(name='Al2O3', element_indices=[1, 4], stoichiometry={1: 2.0, 4: 3.0}),
        LayerDefinition(name='TiO2', element_indices=[2, 4], stoichiometry={2: 1.0, 4: 2.0}),
        LayerDefinition(name='SiO2', element_indices=[3, 4], stoichiometry={3: 1.0, 4: 2.0}),
    ]
    substrate = LayerDefinition(name='Si', element_indices=[5], stoichiometry={5: 1.0})

    solver_config = SolverConfig(method='LLS', alpha=0.0)

    # Narrow
    layer_config_narrow = GenericLayerConfig(
        layers=layers,
        substrate=substrate,
        initial_thicknesses=[10.0, 20.0, 40.0, 35.0],
        initial_widths=[2.0, 3.0, 4.0, 5.0],
    )
    solver_narrow = GenericLayerSolver(tm_narrow, solver_config, layer_config_narrow, elements=elements)
    result_narrow = solver_narrow.solve(I_narrow, verbose=False)
    error_narrow = abs(sum(result_narrow.layer_thicknesses) - true_total)

    # Wide
    layer_config_wide = GenericLayerConfig(
        layers=layers,
        substrate=substrate,
        initial_thicknesses=[10.0, 20.0, 40.0, 35.0],
        initial_widths=[2.0, 3.0, 4.0, 5.0],
    )
    solver_wide = GenericLayerSolver(tm_wide, solver_config, layer_config_wide, elements=elements)
    result_wide = solver_wide.solve(I_wide, verbose=False)
    error_wide = abs(sum(result_wide.layer_thicknesses) - true_total)

    # Wide angle should generally be better
    assert error_wide <= error_narrow or error_wide < 1.0, "Wide angle should perform at least as well"


def run_all_tests():
    """Run all tests (standalone mode)."""
    print("=" * 60)
    print("Narrow Angle Benchmark Test Suite")
    print("=" * 60)
    print()

    test_narrow_angle_generic_layer()
    print("Test 1: GenericLayerSolver narrow angle ... PASSED")

    test_narrow_angle_semifree()
    print("Test 2: SemiFreeSolver narrow angle ... PASSED")

    test_narrow_vs_wide_angle()
    print("Test 3: Narrow vs wide angle comparison ... PASSED")

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
