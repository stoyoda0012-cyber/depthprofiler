"""
Heteroscedastic Noise Handling Benchmark
=========================================

Compare OLS, WLS, and Poisson NLL for handling Poisson noise (heteroscedastic).

Key comparison:
- Gaussian 2% (homoscedastic baseline)
- Poisson 0.5% (RMS-equivalent to Gaussian 2%)

Methods:
- OLS: Ordinary Least Squares (use_wls=False, use_poisson_nll=False)
- WLS: Weighted Least Squares (use_wls=True, weight=I)
- Poisson NLL: Poisson deviance (use_poisson_nll=True)

Date: 2026-01-28
"""

import numpy as np
from scipy.special import erf
from typing import List, Tuple, Dict
import time


def create_sio2_si_profile(depth: np.ndarray, thickness: float = 30.0, interface_width: float = 3.0) -> np.ndarray:
    """Create SiO2/Si bilayer profile.

    Args:
        depth: Depth array (Å)
        thickness: SiO2 thickness (Å)
        interface_width: Interface width parameter (Å)

    Returns:
        D: Concentration profile (n_depth, 3) for [Si_ox, O, Si_bulk]
    """
    sigma = interface_width / (2 * np.sqrt(np.log(2)))

    D = np.zeros((len(depth), 3))

    # SiO2 layer
    sio2 = 0.5 * (1 - erf((depth - thickness) / sigma))
    D[:, 0] = (1/3) * sio2  # Si in SiO2
    D[:, 1] = (2/3) * sio2  # O in SiO2

    # Si substrate
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))

    # Ensure sum = 1
    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    return D


def add_gaussian_noise(I_true: np.ndarray, noise_level: float, seed: int = None) -> np.ndarray:
    """Add Gaussian noise (homoscedastic).

    noise_level: relative noise σ/I (same for all intensities)
    """
    if seed is not None:
        np.random.seed(seed)

    noise = noise_level * np.random.randn(*I_true.shape) * I_true
    I_obs = I_true + noise
    I_obs = np.clip(I_obs, 1e-10, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    return I_obs


def add_poisson_noise(I_true: np.ndarray, noise_level: float, seed: int = None) -> np.ndarray:
    """Add Poisson noise (heteroscedastic).

    noise_level: relative noise σ/I at maximum intensity (I=1.0)
                 Total counts = 1/(noise_level)^2

    At lower intensities, relative noise increases: σ_rel = noise_level / sqrt(I)
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert noise_level to total counts
    # At I=1.0: σ_rel = 1/sqrt(N) = noise_level → N = 1/noise_level^2
    total_counts = 1.0 / (noise_level ** 2)

    # Scale intensities to counts
    counts = I_true * total_counts

    # Generate Poisson samples
    noisy_counts = np.random.poisson(counts)

    # Convert back to normalized intensities
    I_obs = noisy_counts / total_counts
    I_obs = np.clip(I_obs, 1e-10, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    return I_obs


def compute_wasserstein(depth: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Wasserstein-1 distance between two 1D distributions."""
    p1_norm = p1 / (np.sum(p1) + 1e-10)
    p2_norm = p2 / (np.sum(p2) + 1e-10)

    cdf1 = np.cumsum(p1_norm)
    cdf2 = np.cumsum(p2_norm)

    step = depth[1] - depth[0] if len(depth) > 1 else 1.0
    w1 = np.sum(np.abs(cdf1 - cdf2)) * step
    return w1


def run_solver(
    I_obs: np.ndarray,
    tm,
    elements: List,
    method: str = 'L1',
    alpha: float = 1.0,
    use_wls: bool = False,
    use_poisson_nll: bool = False,
) -> Tuple[np.ndarray, float, bool]:
    """Run SemiFreeSolver with specified loss function.

    Returns:
        D_opt: Reconstructed depth profile
        thickness: Estimated film thickness
        success: Convergence status
    """
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    semi_config = SemiFreeSolverConfig(
        thickness_init=30.0,
        abruptness_init=3.0,
        free_indices=[0],     # Si in SiO2
        linked_indices=[1],   # O linked to Si (2:1)
        substrate_indices=[2], # Si bulk
    )

    solver_config = SolverConfig(
        method=method,
        alpha=alpha,
        max_iter=100,
        use_wls=use_wls,
        use_poisson_nll=use_poisson_nll,
    )

    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)
    result = solver.solve(I_obs, verbose=False)

    return result.D_opt, result.thickness, result.success


def run_benchmark(
    n_trials: int = 30,
    noise_type: str = 'gaussian',
    noise_level: float = 0.02,
    loss_type: str = 'OLS',
    method: str = 'L1',
    alpha: float = 1.0,
    film_thickness: float = 30.0,
) -> Dict:
    """Run benchmark for a specific noise/loss combination.

    Args:
        n_trials: Number of trials
        noise_type: 'gaussian' or 'poisson'
        noise_level: Noise level (σ_rel for Gaussian, σ_rel at I=1 for Poisson)
        loss_type: 'OLS', 'WLS', or 'Poisson_NLL'
        method: Regularization method ('L1', 'MEM', 'LLS')
        alpha: Regularization strength
        film_thickness: True SiO2 thickness (Å)

    Returns:
        Dictionary with statistics
    """
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    # Setup
    depth = np.arange(0, 80, 2.0)
    angles = [0, 15, 30, 45, 60]

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_sio2_si_profile(depth, thickness=film_thickness, interface_width=3.0)

    tm = TransferMatrix(depth=depth, step_size=2.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    # Determine loss function settings
    use_wls = (loss_type == 'WLS')
    use_poisson_nll = (loss_type == 'Poisson_NLL')

    # Run trials
    w1_list = []
    thickness_errors = []
    success_count = 0

    for trial in range(n_trials):
        # Add noise
        if noise_type == 'gaussian':
            I_obs = add_gaussian_noise(I_true, noise_level, seed=42 + trial)
        else:  # poisson
            I_obs = add_poisson_noise(I_true, noise_level, seed=42 + trial)

        # Run solver
        D_opt, thickness_est, success = run_solver(
            I_obs, tm, elements,
            method=method,
            alpha=alpha,
            use_wls=use_wls,
            use_poisson_nll=use_poisson_nll,
        )

        if success:
            success_count += 1

            # Compute W1 for each element and take mean
            w1_elements = []
            for i in range(3):
                w1 = compute_wasserstein(depth, D_true[:, i], D_opt[:, i])
                w1_elements.append(w1)

            w1_mean = np.mean(w1_elements)
            w1_list.append(w1_mean)
            thickness_errors.append(abs(thickness_est - film_thickness))
        else:
            # Count as failure, don't include in statistics
            w1_list.append(np.nan)
            thickness_errors.append(np.nan)

    # Compute statistics (excluding failures)
    w1_valid = [w for w in w1_list if not np.isnan(w)]
    thick_valid = [t for t in thickness_errors if not np.isnan(t)]

    return {
        'noise_type': noise_type,
        'noise_level': noise_level,
        'loss_type': loss_type,
        'method': method,
        'alpha': alpha,
        'n_trials': n_trials,
        'n_success': success_count,
        'success_rate': success_count / n_trials,
        'w1_median': np.median(w1_valid) if w1_valid else np.nan,
        'w1_mean': np.mean(w1_valid) if w1_valid else np.nan,
        'w1_std': np.std(w1_valid) if w1_valid else np.nan,
        'thickness_error_median': np.median(thick_valid) if thick_valid else np.nan,
        'thickness_error_mean': np.mean(thick_valid) if thick_valid else np.nan,
    }


def main():
    """Main benchmark function."""

    print("=" * 80)
    print("Heteroscedastic Noise Handling Benchmark")
    print("SemiFreeSolver, method='L1', α=1.0, D_prior=None (→1e-6)")
    print("=" * 80)

    n_trials = 30

    # Define test conditions
    # RMS-equivalent: Gaussian 2% ≈ Poisson 0.5%
    conditions = [
        # Baseline: Gaussian 2%
        {'noise_type': 'gaussian', 'noise_level': 0.02, 'label': 'Gaussian 2%'},

        # RMS-equivalent Poisson
        {'noise_type': 'poisson', 'noise_level': 0.005, 'label': 'Poisson 0.5%'},
    ]

    loss_types = ['OLS', 'WLS', 'Poisson_NLL']

    # Run all combinations
    results = []

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"Noise: {cond['label']}")
        print(f"{'='*60}")

        for loss in loss_types:
            print(f"\n  Loss: {loss}...", end='', flush=True)
            start = time.time()

            result = run_benchmark(
                n_trials=n_trials,
                noise_type=cond['noise_type'],
                noise_level=cond['noise_level'],
                loss_type=loss,
                method='L1',
                alpha=1.0,
            )

            elapsed = time.time() - start
            print(f" done ({elapsed:.1f}s)")

            result['label'] = cond['label']
            results.append(result)

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Noise':<15} {'Loss':<15} {'median W1 (Å)':<15} {'Success Rate':<15}")
    print("-" * 60)

    for r in results:
        w1_str = f"{r['w1_median']:.2f} ± {r['w1_std']:.2f}" if not np.isnan(r['w1_median']) else "N/A"
        success_str = f"{r['success_rate']*100:.0f}%"
        print(f"{r['label']:<15} {r['loss_type']:<15} {w1_str:<15} {success_str:<15}")

    print("-" * 60)

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Compare within each noise type
    for cond in conditions:
        label = cond['label']
        print(f"\n{label}:")

        cond_results = [r for r in results if r['label'] == label]

        # Find best by success rate first, then W1
        best = min(cond_results, key=lambda x: (-x['success_rate'], x['w1_median']))

        print(f"  Best: {best['loss_type']} (W1={best['w1_median']:.2f} Å, Success={best['success_rate']*100:.0f}%)")

        # Compare Poisson handling improvement
        ols_result = next((r for r in cond_results if r['loss_type'] == 'OLS'), None)
        if ols_result and best['loss_type'] != 'OLS':
            if ols_result['success_rate'] > 0 and best['success_rate'] > 0:
                w1_improvement = (ols_result['w1_median'] - best['w1_median']) / ols_result['w1_median'] * 100
                sr_improvement = (best['success_rate'] - ols_result['success_rate']) * 100
                print(f"  vs OLS: W1 {w1_improvement:+.1f}%, Success {sr_improvement:+.0f}%")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
