"""
WLS Robustness Test for Poisson Noise
=====================================

Test WLS performance under more challenging conditions:
- Wider angle range (to test edge cases)
- More trials (to catch rare failures)
- Multiple thickness values

Compare: OLS vs WLS for Poisson 0.5% noise

Date: 2026-01-28
"""

import numpy as np
from scipy.special import erf
from typing import List, Tuple, Dict
import time


def create_sio2_si_profile(depth: np.ndarray, thickness: float = 30.0, interface_width: float = 3.0) -> np.ndarray:
    """Create SiO2/Si bilayer profile."""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))

    D = np.zeros((len(depth), 3))
    sio2 = 0.5 * (1 - erf((depth - thickness) / sigma))
    D[:, 0] = (1/3) * sio2
    D[:, 1] = (2/3) * sio2
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def add_poisson_noise(I_true: np.ndarray, noise_level: float, seed: int = None) -> np.ndarray:
    """Add Poisson noise."""
    if seed is not None:
        np.random.seed(seed)

    total_counts = 1.0 / (noise_level ** 2)
    counts = I_true * total_counts
    noisy_counts = np.random.poisson(counts)
    I_obs = noisy_counts / total_counts
    I_obs = np.clip(I_obs, 1e-10, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    return I_obs


def add_gaussian_noise(I_true: np.ndarray, noise_level: float, seed: int = None) -> np.ndarray:
    """Add Gaussian noise."""
    if seed is not None:
        np.random.seed(seed)

    noise = noise_level * np.random.randn(*I_true.shape) * I_true
    I_obs = I_true + noise
    I_obs = np.clip(I_obs, 1e-10, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    return I_obs


def compute_wasserstein(depth: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Wasserstein-1 distance."""
    p1_norm = p1 / (np.sum(p1) + 1e-10)
    p2_norm = p2 / (np.sum(p2) + 1e-10)
    cdf1 = np.cumsum(p1_norm)
    cdf2 = np.cumsum(p2_norm)
    step = depth[1] - depth[0] if len(depth) > 1 else 1.0
    return np.sum(np.abs(cdf1 - cdf2)) * step


def run_solver(I_obs, tm, elements, use_wls: bool = False):
    """Run SemiFreeSolver."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    semi_config = SemiFreeSolverConfig(
        thickness_init=30.0,
        abruptness_init=3.0,
        free_indices=[0],
        linked_indices=[1],
        substrate_indices=[2],
    )

    solver_config = SolverConfig(
        method='L1',
        alpha=1.0,
        max_iter=100,
        use_wls=use_wls,
    )

    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)
    result = solver.solve(I_obs, verbose=False)

    return result.D_opt, result.thickness, result.success


def run_robustness_test(
    n_trials: int = 50,
    thickness: float = 30.0,
    noise_type: str = 'poisson',
    noise_level: float = 0.005,
    angles: List[float] = None,
) -> Dict:
    """Run robustness test."""
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    if angles is None:
        angles = [0, 15, 30, 45, 60]

    depth = np.arange(0, 80, 2.0)

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_sio2_si_profile(depth, thickness=thickness)

    tm = TransferMatrix(depth=depth, step_size=2.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    results = {'OLS': {'w1': [], 'success': []}, 'WLS': {'w1': [], 'success': []}}

    for trial in range(n_trials):
        if noise_type == 'poisson':
            I_obs = add_poisson_noise(I_true, noise_level, seed=42 + trial)
        else:
            I_obs = add_gaussian_noise(I_true, noise_level, seed=42 + trial)

        for use_wls in [False, True]:
            name = 'WLS' if use_wls else 'OLS'

            D_opt, thickness_est, success = run_solver(I_obs, tm, elements, use_wls=use_wls)

            results[name]['success'].append(success)

            if success:
                w1_elements = [compute_wasserstein(depth, D_true[:, i], D_opt[:, i]) for i in range(3)]
                results[name]['w1'].append(np.mean(w1_elements))
            else:
                results[name]['w1'].append(np.nan)

    return results


def main():
    print("=" * 80)
    print("WLS Robustness Test for Poisson Noise")
    print("SemiFreeSolver, method='L1', α=1.0, D_prior=None")
    print("=" * 80)

    # Test conditions
    conditions = [
        # Standard angles, various thicknesses
        {'thickness': 10.0, 'angles': [0, 15, 30, 45, 60], 'label': '10Å, 0-60°'},
        {'thickness': 30.0, 'angles': [0, 15, 30, 45, 60], 'label': '30Å, 0-60°'},
        {'thickness': 50.0, 'angles': [0, 15, 30, 45, 60], 'label': '50Å, 0-60°'},

        # Challenging: wider angle range
        {'thickness': 30.0, 'angles': [0, 20, 40, 60, 70], 'label': '30Å, 0-70°'},

        # Challenging: many angles (more data points, higher sensitivity)
        {'thickness': 30.0, 'angles': [0, 10, 20, 30, 40, 50, 60], 'label': '30Å, 7 angles'},
    ]

    n_trials = 50

    print(f"\nRunning {n_trials} trials per condition...")
    print("Noise: Poisson 0.5% (≈Gaussian 2% RMS-equivalent)")

    all_results = []

    for cond in conditions:
        print(f"\n  {cond['label']}...", end='', flush=True)

        results = run_robustness_test(
            n_trials=n_trials,
            thickness=cond['thickness'],
            angles=cond['angles'],
            noise_type='poisson',
            noise_level=0.005,
        )

        # Compute statistics
        for name in ['OLS', 'WLS']:
            success_rate = np.mean(results[name]['success'])
            w1_valid = [w for w in results[name]['w1'] if not np.isnan(w)]
            w1_median = np.median(w1_valid) if w1_valid else np.nan
            w1_std = np.std(w1_valid) if w1_valid else np.nan

            all_results.append({
                'condition': cond['label'],
                'method': name,
                'success_rate': success_rate,
                'w1_median': w1_median,
                'w1_std': w1_std,
                'n_success': sum(results[name]['success']),
            })

        print(" done")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Condition':<20} {'Method':<8} {'Success':<10} {'median W1 (Å)':<15}")
    print("-" * 55)

    prev_cond = None
    for r in all_results:
        if r['condition'] != prev_cond:
            if prev_cond is not None:
                print("-" * 55)
            prev_cond = r['condition']

        w1_str = f"{r['w1_median']:.3f} ± {r['w1_std']:.3f}" if not np.isnan(r['w1_median']) else "N/A"
        success_str = f"{r['n_success']}/{n_trials}"
        print(f"{r['condition']:<20} {r['method']:<8} {success_str:<10} {w1_str:<15}")

    # Summary: OLS vs WLS
    print("\n" + "=" * 80)
    print("OLS vs WLS COMPARISON")
    print("=" * 80)

    for cond in conditions:
        ols = next(r for r in all_results if r['condition'] == cond['label'] and r['method'] == 'OLS')
        wls = next(r for r in all_results if r['condition'] == cond['label'] and r['method'] == 'WLS')

        if not np.isnan(ols['w1_median']) and not np.isnan(wls['w1_median']):
            improvement = (ols['w1_median'] - wls['w1_median']) / ols['w1_median'] * 100
            success_diff = wls['success_rate'] - ols['success_rate']
            print(f"{cond['label']:<20} WLS: W1 {improvement:+.1f}%, Success {success_diff*100:+.0f}%")
        else:
            print(f"{cond['label']:<20} Insufficient data")

    # Final recommendation
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Count improvements
    ols_better = 0
    wls_better = 0
    for cond in conditions:
        ols = next(r for r in all_results if r['condition'] == cond['label'] and r['method'] == 'OLS')
        wls = next(r for r in all_results if r['condition'] == cond['label'] and r['method'] == 'WLS')

        # Compare by success rate first, then W1
        if wls['success_rate'] > ols['success_rate']:
            wls_better += 1
        elif wls['success_rate'] < ols['success_rate']:
            ols_better += 1
        elif not np.isnan(wls['w1_median']) and not np.isnan(ols['w1_median']):
            if wls['w1_median'] < ols['w1_median']:
                wls_better += 1
            else:
                ols_better += 1

    print(f"\nWLS better: {wls_better}/{len(conditions)} conditions")
    print(f"OLS better: {ols_better}/{len(conditions)} conditions")

    if wls_better > ols_better:
        print("\n→ WLS recommended for Poisson noise")
    else:
        print("\n→ OLS is competitive; WLS optional")

    print("=" * 80)

    return all_results


if __name__ == "__main__":
    main()
