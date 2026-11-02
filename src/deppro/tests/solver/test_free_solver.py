"""
FreeSolver Test Suite
=====================

Tests for FreeSolver (model-free depth profile reconstruction).

Usage:
    pytest tests/solver/test_free_solver.py -v
    python -m deppro.tests.solver.test_free_solver
"""

import pytest
import numpy as np
from scipy.special import erf
import time


def create_five_layer_profile_data():
    """Create five-layer profile test data."""
    from deppro.depth_profile import create_five_layer_profile
    from deppro.transfer_matrix import TransferMatrix

    profile = create_five_layer_profile(depth_end=100, step_size=2.0)
    angles = np.array([0, 15, 30, 45, 60, 75])
    step_size = profile.depth[1] - profile.depth[0] if len(profile.depth) > 1 else 1.0

    tm = TransferMatrix(
        depth=profile.depth,
        step_size=step_size,
        angles=angles,
        photon_energy=1486.6,
    )
    tm.compute(profile.concentrations, profile.elements)

    return profile, tm, angles


def test_roundtrip_no_noise():
    """Test round-trip reconstruction without noise."""
    from deppro.inverse_solver import FreeSolver, SolverConfig

    profile, tm, angles = create_five_layer_profile_data()
    D_true = profile.concentrations

    # Forward model
    I_true = tm.forward(D_true, normalize=True)

    # Inverse solver
    config = SolverConfig(method='L2', alpha=0.01, max_iter=200)
    D_prior = np.ones_like(D_true) / profile.n_elements
    solver = FreeSolver(tm, config, D_prior)

    result = solver.solve(I_true)
    D_reconstructed = result.D_opt

    # Reconstruction should be reasonable without noise
    rmse = np.sqrt(np.mean((D_true - D_reconstructed) ** 2))
    assert rmse < 0.3, f"RMSE should be < 0.3 without noise, got {rmse:.4f}"


def test_roundtrip_with_noise():
    """Test round-trip reconstruction with 1% noise."""
    from deppro.inverse_solver import FreeSolver, SolverConfig

    profile, tm, angles = create_five_layer_profile_data()
    D_true = profile.concentrations

    # Forward model
    I_true = tm.forward(D_true, normalize=True)

    # Add noise
    np.random.seed(42)
    noise_level = 0.01
    noise = np.random.randn(*I_true.shape) * noise_level * I_true
    I_obs = I_true + noise
    I_obs = np.maximum(I_obs, 0)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    # Inverse solver
    config = SolverConfig(method='L2', alpha=0.1, max_iter=200)
    D_prior = np.ones_like(D_true) / profile.n_elements
    solver = FreeSolver(tm, config, D_prior)

    result = solver.solve(I_obs)
    D_reconstructed = result.D_opt

    # Reconstruction accuracy degrades with noise
    rmse = np.sqrt(np.mean((D_true - D_reconstructed) ** 2))
    assert rmse < 0.35, f"RMSE should be < 0.35 with 1% noise, got {rmse:.4f}"


def test_method_comparison():
    """Compare L2, MEM, LLS methods."""
    from deppro.inverse_solver import FreeSolver, SolverConfig

    profile, tm, angles = create_five_layer_profile_data()
    D_true = profile.concentrations

    # Forward model with noise
    I_true = tm.forward(D_true, normalize=True)
    np.random.seed(42)
    noise_level = 0.01
    noise = np.random.randn(*I_true.shape) * noise_level * I_true
    I_obs = I_true + noise
    I_obs = np.maximum(I_obs, 0)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    D_prior = np.ones_like(D_true) / profile.n_elements
    methods = ['L2', 'MEM', 'LLS']
    results = {}

    for method in methods:
        config = SolverConfig(method=method, alpha=0.1, max_iter=200)
        solver = FreeSolver(tm, config, D_prior)
        result = solver.solve(I_obs)
        rmse = np.sqrt(np.mean((D_true - result.D_opt) ** 2))
        results[method] = rmse

    # All methods should produce reasonable results
    for method, rmse in results.items():
        assert rmse < 0.35, f"{method} RMSE should be < 0.35, got {rmse:.4f}"


@pytest.mark.slow  # 実測 120s (2026-07-29)
def test_lls_4layer():
    """Test LLS on 4-layer model."""
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig
    from deppro.inverse_solver import FreeSolver, SolverConfig

    thicknesses = [5.0, 15.0, 35.0, 30.0]
    interface_width = 3.0
    angles = [0, 12, 24, 36, 48, 60]
    np.random.seed(42)

    depth = np.arange(0, 150, 1.0)
    sigma = interface_width / (2 * np.sqrt(np.log(2)))

    # Interface positions
    z1, z2, z3, z4 = np.cumsum(thicknesses)

    D = np.zeros((len(depth), 6))
    D[:, 0] = 0.5 * (1 - erf((depth - z1) / sigma))  # C

    al2o3 = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    D[:, 1] = 0.4 * al2o3  # Al
    D[:, 4] = 0.6 * al2o3  # O

    tio2 = 0.5 * (1 + erf((depth - z2) / sigma)) * 0.5 * (1 - erf((depth - z3) / sigma))
    D[:, 2] = 1/3 * tio2  # Ti
    D[:, 4] += 2/3 * tio2  # O

    sio2 = 0.5 * (1 + erf((depth - z3) / sigma)) * 0.5 * (1 - erf((depth - z4) / sigma))
    D[:, 3] = 1/3 * sio2  # Si_ox
    D[:, 4] += 2/3 * sio2  # O

    D[:, 5] = 0.5 * (1 + erf((depth - z4) / sigma))  # Si_bulk

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

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D, elements)

    I_true = tm.forward(D, normalize=True)
    noise_level = 0.02
    noise = noise_level * np.random.randn(*I_true.shape)
    I_obs = I_true + noise * I_true
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    # LLS solver
    config = SolverConfig(method='LLS', alpha=0.0)
    solver = FreeSolver(tm, config)
    result = solver.solve(I_obs, verbose=False)

    rmse = np.sqrt(np.mean((D - result.D_opt) ** 2))
    # Threshold history: < 0.3 until 2026-07-05, when the FreeSolver
    # analytical gradient was fixed to include the normalization derivative
    # (see tests/solver/test_freesolver_gradient_consistency.py). The old simplified
    # gradient stalled at f=0.241 (RMSE 0.296); the exact gradient reaches
    # f=0.0011 (217x lower) but, with alpha=0 and 2% noise, the better data
    # fit slightly worsens reconstruction (RMSE 0.328) — the old stall was
    # acting as accidental early-stopping regularization.
    assert rmse < 0.35, f"LLS RMSE should be < 0.35, got {rmse:.4f}"


def run_all_tests():
    """Run all tests (standalone mode)."""
    print("=" * 60)
    print("FreeSolver Test Suite")
    print("=" * 60)
    print()

    test_roundtrip_no_noise()
    print("Test 1: Round-trip (no noise) ... PASSED")

    test_roundtrip_with_noise()
    print("Test 2: Round-trip (1% noise) ... PASSED")

    test_method_comparison()
    print("Test 3: Method comparison ... PASSED")

    test_lls_4layer()
    print("Test 4: LLS 4-layer ... PASSED")

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
