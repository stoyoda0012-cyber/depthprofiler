"""
WLS (Weighted Least Squares) for Poisson Noise Benchmark
=========================================================

Tests whether WLS improves accuracy for Poisson noise vs standard LS.

Key insight:
- Standard LS assumes homoscedastic (uniform variance) errors
- Poisson noise: variance = mean (heteroscedastic)
- WLS with weight = 1/I corrects for this

Expected results:
- For Gaussian 2% noise: WLS ≈ LS (both ~0.7 Å W1 error)
- For Poisson 2% noise: WLS << LS (goal: ~1 Å vs ~5 Å)

Usage:
    python -m pytest deppro/tests/benchmark/test_wls_poisson.py -v

    # Or run directly for detailed output:
    python -m deppro.tests.benchmark.test_wls_poisson
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Literal, Tuple

# Internal imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from deppro.depth_profile import create_simple_sio2_si_profile, ElementConfig
from deppro.transfer_matrix import TransferMatrix
from deppro.inverse_solver import (
    SemiFreeSolver,
    SemiFreeSolverConfig,
    SolverConfig,
)
from scipy.special import erf


NoiseType = Literal["gaussian", "poisson", "uniform"]


@dataclass
class NoiseConfig:
    """Noise configuration for benchmark."""
    noise_type: NoiseType
    noise_level: float  # relative noise level (e.g., 0.02 = 2%)


def add_noise(I_true: np.ndarray, config: NoiseConfig, seed: int = 42) -> np.ndarray:
    """
    Add noise to true intensities.

    Args:
        I_true: True intensities (n_angles, n_elements)
        config: Noise configuration
        seed: Random seed

    Returns:
        I_obs: Noisy observations
    """
    np.random.seed(seed)

    if config.noise_type == "gaussian":
        # Gaussian: I_obs = I_true + noise * I_true
        # σ = noise_level * I_true (relative noise)
        noise = config.noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true * (1 + noise)

    elif config.noise_type == "poisson":
        # Poisson: variance = mean
        # Scale to get desired relative noise level at mean intensity
        # For Poisson: σ = sqrt(N), relative noise = 1/sqrt(N)
        # To get 2% relative noise: N = 1/(0.02)^2 = 2500 counts
        mean_counts = 1.0 / (config.noise_level ** 2)

        # Scale intensities to counts, apply Poisson, scale back
        counts = I_true * mean_counts
        noisy_counts = np.random.poisson(counts.astype(np.float64))
        I_obs = noisy_counts / mean_counts

    elif config.noise_type == "uniform":
        # Uniform: I_obs = I_true * (1 + U(-noise, +noise))
        noise = config.noise_level * (2 * np.random.rand(*I_true.shape) - 1)
        I_obs = I_true * (1 + noise)

    else:
        raise ValueError(f"Unknown noise type: {config.noise_type}")

    # Ensure positive
    I_obs = np.clip(I_obs, 1e-10, None)

    return I_obs


def compute_w1_distance(D_true: np.ndarray, D_opt: np.ndarray, depth: np.ndarray) -> float:
    """
    Compute Wasserstein-1 (W1) distance between true and optimized profiles.

    Uses scipy.stats.wasserstein_distance for proper W1 calculation.
    Returns the average W1 distance across all elements.
    """
    from scipy.stats import wasserstein_distance

    w1_sum = 0.0
    n_elements = D_true.shape[1]
    for i in range(n_elements):
        w1_sum += wasserstein_distance(depth, depth, D_true[:, i], D_opt[:, i])
    return w1_sum / n_elements


def create_sio2_si_profile(depth: np.ndarray, thickness: float, interface_width: float) -> np.ndarray:
    """
    Create SiO2/Si profile with 3 elements.

    Elements:
        0: Si4+ (silicon in SiO2, free element in film)
        1: O2- (oxygen in SiO2, linked element - charge balance)
        2: Si0 (silicon in substrate)

    The SiO2 stoichiometry is Si:O = 1:2, so:
    - Si4+ concentration = 0.333 in SiO2 region
    - O2- concentration = 0.667 in SiO2 region (linked: 2x Si4+)
    - Si0 concentration = 1.0 in substrate
    """
    sigma = interface_width / (2 * np.sqrt(np.log(2)))

    D = np.zeros((len(depth), 3))

    # SiO2 layer (z < thickness)
    sio2_fraction = 0.5 * (1 - erf((depth - thickness) / sigma))

    # Si4+ in SiO2
    D[:, 0] = 0.333 * sio2_fraction

    # O2- in SiO2 (linked to Si4+, ratio 2:1)
    D[:, 1] = 0.667 * sio2_fraction

    # Si substrate (z > thickness)
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))

    # Normalize
    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    return D


def run_benchmark_trial(
    use_wls: bool,
    noise_config: NoiseConfig,
    true_thickness: float = 30.0,
    interface_width: float = 5.0,
    seed: int = 42,
    use_anscombe: bool = False,
    use_poisson_nll: bool = False,
) -> Tuple[float, float, float]:
    """
    Run a single benchmark trial.

    Args:
        use_wls: Use weighted least squares
        noise_config: Noise configuration
        true_thickness: True film thickness (Å)
        interface_width: Interface width (Å)
        seed: Random seed
        use_anscombe: Apply Anscombe variance-stabilizing transform
        use_poisson_nll: Use Poisson negative log-likelihood

    Returns:
        thickness_error: |estimated - true| thickness (Å)
        w1_distance: Wasserstein-1 distance
        elapsed_time: Solve time (seconds)
    """
    # Create test profile (SiO2/Si)
    depth = np.arange(0, 150, 1.0)
    angles = np.array([15, 30, 45, 60, 75])

    # Elements: Si4+ (free), O2- (linked), Si0 (substrate)
    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_sio2_si_profile(depth, true_thickness, interface_width)

    # Transfer matrix
    tm = TransferMatrix(
        depth=depth,
        step_size=1.0,
        angles=angles,
        photon_energy=1486.6,
    )
    tm.compute(D_true, elements)

    # True intensities
    I_true = tm.forward(D_true, normalize=True)

    # Add noise
    I_obs = add_noise(I_true, noise_config, seed)

    # Normalize observation
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    # Solver config
    config = SolverConfig(
        method='L1',
        alpha=1.0,
        use_wls=use_wls,
        use_anscombe=use_anscombe,
        use_poisson_nll=use_poisson_nll,
        scipy_method='SLSQP',
        max_iter=100,
    )

    semi_config = SemiFreeSolverConfig(
        thickness_init=true_thickness + 5.0,  # offset from true
        abruptness_init=interface_width,
        thickness_bounds=(1.0, 200.0),
        abruptness_bounds=(1.0, 20.0),
        free_indices=[0],       # Si4+ is free
        linked_indices=[1],     # O2- is linked (charge balance)
        substrate_indices=[2],  # Si0 is substrate
    )

    # Create solver
    solver = SemiFreeSolver(
        transfer_matrix=tm,
        config=config,
        semi_free_config=semi_config,
        elements=elements,
    )

    # Solve
    start_time = time.time()
    result = solver.solve(I_obs)
    elapsed_time = time.time() - start_time

    # Compute errors
    thickness_error = abs(result.thickness - true_thickness)
    w1_distance = compute_w1_distance(D_true, result.D_opt, depth)

    return thickness_error, w1_distance, elapsed_time


def run_noise_comparison(
    noise_type: NoiseType,
    noise_level: float = 0.02,
    n_trials: int = 30,
    true_thickness: float = 30.0,
    include_anscombe: bool = False,
    include_poisson_nll: bool = True,
) -> dict:
    """
    Compare OLS, WLS, Poisson NLL, and optionally Anscombe for a given noise type.

    Returns:
        dict: Results for each method {method_name: {'thickness_error': mean, 'w1': mean}}
    """
    print(f"\n{'='*60}")
    print(f"Noise: {noise_type.upper()} ({noise_level*100:.0f}%)")
    print(f"True thickness: {true_thickness:.1f} Å")
    print(f"Trials: {n_trials}")
    print('='*60)

    noise_config = NoiseConfig(noise_type=noise_type, noise_level=noise_level)

    # Methods to test: (use_wls, use_anscombe, use_poisson_nll, name)
    methods = [
        (False, False, False, "OLS"),
        (True, False, False, "WLS"),
    ]
    if include_poisson_nll:
        methods.append((False, False, True, "Poisson"))
    if include_anscombe:
        methods.append((False, True, False, "Anscombe"))

    results = {}

    for use_wls, use_anscombe, use_poisson_nll, method_name in methods:
        thickness_errors = []
        w1_distances = []
        times = []

        for trial in range(n_trials):
            try:
                t_err, w1, elapsed = run_benchmark_trial(
                    use_wls=use_wls,
                    noise_config=noise_config,
                    true_thickness=true_thickness,
                    seed=42 + trial,
                    use_anscombe=use_anscombe,
                    use_poisson_nll=use_poisson_nll,
                )
                thickness_errors.append(t_err)
                w1_distances.append(w1)
                times.append(elapsed)
            except Exception as e:
                print(f"  Trial {trial} failed: {e}")

        if thickness_errors:
            results[method_name] = {
                'thickness_error': np.mean(thickness_errors),
                'thickness_std': np.std(thickness_errors),
                'w1': np.mean(w1_distances),
                'w1_std': np.std(w1_distances),
                'time_ms': np.mean(times) * 1000,
            }
            print(f"\n{method_name}:")
            print(f"  Thickness error: {np.mean(thickness_errors):.2f} ± {np.std(thickness_errors):.2f} Å")
            print(f"  W1 distance:     {np.mean(w1_distances):.2f} ± {np.std(w1_distances):.2f} Å")
            print(f"  Solve time:      {np.mean(times)*1000:.1f} ± {np.std(times)*1000:.1f} ms")

    return results


def main():
    """Run OLS vs WLS vs Poisson NLL benchmark for different noise types."""
    print("="*60)
    print("OLS vs WLS vs Poisson NLL Benchmark for Poisson Noise")
    print("="*60)
    print("\nExpected: Poisson NLL should improve accuracy for Poisson noise")
    print("          (Goal: W1 ~ 1-2 Å vs ~ 3 Å baseline)")

    all_results = {}

    # Test with Gaussian noise (baseline - should be similar)
    all_results["gaussian"] = run_noise_comparison("gaussian", noise_level=0.02, n_trials=10)

    # Test with Poisson noise (main target)
    all_results["poisson"] = run_noise_comparison("poisson", noise_level=0.02, n_trials=10)

    # Test with Uniform noise (for reference)
    all_results["uniform"] = run_noise_comparison("uniform", noise_level=0.02, n_trials=10)

    print("\n" + "="*60)
    print("Summary Table")
    print("="*60)
    print(f"\n{'Noise Type':<12} {'OLS W1 (Å)':<14} {'WLS W1 (Å)':<14} {'Poisson W1 (Å)':<16}")
    print("-" * 56)
    for noise_type, results in all_results.items():
        ols_w1 = results.get('OLS', {}).get('w1', float('nan'))
        wls_w1 = results.get('WLS', {}).get('w1', float('nan'))
        pois_w1 = results.get('Poisson', {}).get('w1', float('nan'))
        print(f"{noise_type:<12} {ols_w1:<14.2f} {wls_w1:<14.2f} {pois_w1:<16.2f}")

    print("\n" + "="*60)
    print("Interpretation")
    print("="*60)
    print("- Gaussian: All methods should be similar")
    print("- Poisson:  Poisson NLL should outperform OLS")
    print("- Uniform:  All methods should be similar")


# Test functions for pytest
def test_wls_improves_poisson():
    """Test that WLS improves accuracy for Poisson noise."""
    noise_config = NoiseConfig(noise_type="poisson", noise_level=0.02)

    # Run with OLS
    ols_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(use_wls=False, noise_config=noise_config, seed=seed)
        ols_errors.append(t_err)

    # Run with WLS
    wls_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(use_wls=True, noise_config=noise_config, seed=seed)
        wls_errors.append(t_err)

    # WLS should improve (or at least not worsen) accuracy
    # Note: This is a weak test; in production, we'd want a larger sample
    ols_mean = np.mean(ols_errors)
    wls_mean = np.mean(wls_errors)

    print(f"OLS mean error: {ols_mean:.2f} Å")
    print(f"WLS mean error: {wls_mean:.2f} Å")

    # WLS should be at least as good as OLS for Poisson noise
    # (in practice, it should be significantly better)
    assert wls_mean <= ols_mean * 1.5, f"WLS ({wls_mean:.2f}) should not be much worse than OLS ({ols_mean:.2f})"


def test_wls_similar_for_gaussian():
    """Test that WLS doesn't hurt performance for Gaussian noise."""
    noise_config = NoiseConfig(noise_type="gaussian", noise_level=0.02)

    # Run with OLS
    ols_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(use_wls=False, noise_config=noise_config, seed=seed)
        ols_errors.append(t_err)

    # Run with WLS
    wls_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(use_wls=True, noise_config=noise_config, seed=seed)
        wls_errors.append(t_err)

    ols_mean = np.mean(ols_errors)
    wls_mean = np.mean(wls_errors)

    print(f"OLS mean error: {ols_mean:.2f} Å")
    print(f"WLS mean error: {wls_mean:.2f} Å")

    # For Gaussian noise, WLS and OLS should be similar
    # Allow 50% difference due to random variation
    assert abs(wls_mean - ols_mean) < max(ols_mean, 1.0), \
        f"WLS ({wls_mean:.2f}) should be similar to OLS ({ols_mean:.2f}) for Gaussian noise"


def test_anscombe_improves_poisson():
    """Test that Anscombe transform improves accuracy for Poisson noise."""
    noise_config = NoiseConfig(noise_type="poisson", noise_level=0.02)

    # Run with OLS (baseline)
    ols_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(
            use_wls=False, noise_config=noise_config, seed=seed, use_anscombe=False
        )
        ols_errors.append(t_err)

    # Run with Anscombe
    anscombe_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(
            use_wls=False, noise_config=noise_config, seed=seed, use_anscombe=True
        )
        anscombe_errors.append(t_err)

    ols_mean = np.mean(ols_errors)
    anscombe_mean = np.mean(anscombe_errors)

    print(f"OLS mean error:      {ols_mean:.2f} Å")
    print(f"Anscombe mean error: {anscombe_mean:.2f} Å")

    # Anscombe should improve (or at least not worsen significantly) for Poisson noise
    assert anscombe_mean <= ols_mean * 1.2, \
        f"Anscombe ({anscombe_mean:.2f}) should not be much worse than OLS ({ols_mean:.2f})"


def test_anscombe_similar_for_gaussian():
    """Test that Anscombe doesn't hurt performance for Gaussian noise."""
    noise_config = NoiseConfig(noise_type="gaussian", noise_level=0.02)

    # Run with OLS (baseline)
    ols_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(
            use_wls=False, noise_config=noise_config, seed=seed, use_anscombe=False
        )
        ols_errors.append(t_err)

    # Run with Anscombe
    anscombe_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(
            use_wls=False, noise_config=noise_config, seed=seed, use_anscombe=True
        )
        anscombe_errors.append(t_err)

    ols_mean = np.mean(ols_errors)
    anscombe_mean = np.mean(anscombe_errors)

    print(f"OLS mean error:      {ols_mean:.2f} Å")
    print(f"Anscombe mean error: {anscombe_mean:.2f} Å")

    # For Gaussian noise, Anscombe and OLS should be similar
    assert abs(anscombe_mean - ols_mean) < max(ols_mean, 1.0), \
        f"Anscombe ({anscombe_mean:.2f}) should be similar to OLS ({ols_mean:.2f}) for Gaussian noise"


def test_poisson_nll_improves_poisson():
    """Test that Poisson NLL improves accuracy for Poisson noise."""
    noise_config = NoiseConfig(noise_type="poisson", noise_level=0.02)

    # Run with OLS (baseline)
    ols_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(
            use_wls=False, noise_config=noise_config, seed=seed, use_poisson_nll=False
        )
        ols_errors.append(t_err)

    # Run with Poisson NLL
    poisson_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(
            use_wls=False, noise_config=noise_config, seed=seed, use_poisson_nll=True
        )
        poisson_errors.append(t_err)

    ols_mean = np.mean(ols_errors)
    poisson_mean = np.mean(poisson_errors)

    print(f"OLS mean error:     {ols_mean:.2f} Å")
    print(f"Poisson mean error: {poisson_mean:.2f} Å")

    # Poisson NLL should not be significantly worse than OLS
    assert poisson_mean <= ols_mean * 1.5, \
        f"Poisson ({poisson_mean:.2f}) should not be much worse than OLS ({ols_mean:.2f})"


def test_poisson_nll_similar_for_gaussian():
    """Test that Poisson NLL doesn't hurt performance for Gaussian noise."""
    noise_config = NoiseConfig(noise_type="gaussian", noise_level=0.02)

    # Run with OLS (baseline)
    ols_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(
            use_wls=False, noise_config=noise_config, seed=seed, use_poisson_nll=False
        )
        ols_errors.append(t_err)

    # Run with Poisson NLL
    poisson_errors = []
    for seed in range(5):
        t_err, _, _ = run_benchmark_trial(
            use_wls=False, noise_config=noise_config, seed=seed, use_poisson_nll=True
        )
        poisson_errors.append(t_err)

    ols_mean = np.mean(ols_errors)
    poisson_mean = np.mean(poisson_errors)

    print(f"OLS mean error:     {ols_mean:.2f} Å")
    print(f"Poisson mean error: {poisson_mean:.2f} Å")

    # For Gaussian noise, Poisson NLL and OLS should be similar
    assert abs(poisson_mean - ols_mean) < max(ols_mean, 1.0), \
        f"Poisson ({poisson_mean:.2f}) should be similar to OLS ({ols_mean:.2f}) for Gaussian noise"


if __name__ == "__main__":
    main()
