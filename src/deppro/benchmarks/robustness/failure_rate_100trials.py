"""
Failure Rate Test (100 trials)
==============================

Run 100 trials to estimate the failure rate, including rare failures.

Date: 2026-01-28
"""

import numpy as np
from scipy.special import erf
from typing import List


def create_sio2_si_profile(depth: np.ndarray, thickness: float) -> np.ndarray:
    """Create SiO2/Si bilayer profile."""
    sigma = 3.0 / (2 * np.sqrt(np.log(2)))
    D = np.zeros((len(depth), 3))
    sio2 = 0.5 * (1 - erf((depth - thickness) / sigma))
    D[:, 0] = (1/3) * sio2
    D[:, 1] = (2/3) * sio2
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))
    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def add_poisson_noise(I_true: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Add Poisson noise."""
    np.random.seed(seed)
    total_counts = 1.0 / (noise_level ** 2)
    counts = I_true * total_counts
    noisy_counts = np.random.poisson(counts)
    I_obs = noisy_counts / total_counts
    I_obs = np.clip(I_obs, 1e-10, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    return I_obs


def add_gaussian_noise(I_true: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Add Gaussian noise."""
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


def main():
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    print("=" * 70)
    print("Failure Rate Test (100 trials)")
    print("SemiFreeSolver, method='L1', α=1.0, D_prior=None")
    print("=" * 70)

    n_trials = 100
    thickness = 30.0
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

    # Test configurations
    configs = [
        {'noise_type': 'gaussian', 'noise_level': 0.02, 'use_wls': False, 'label': 'Gaussian 2%, OLS'},
        {'noise_type': 'gaussian', 'noise_level': 0.02, 'use_wls': True, 'label': 'Gaussian 2%, WLS'},
        {'noise_type': 'poisson', 'noise_level': 0.005, 'use_wls': False, 'label': 'Poisson 0.5%, OLS'},
        {'noise_type': 'poisson', 'noise_level': 0.005, 'use_wls': True, 'label': 'Poisson 0.5%, WLS'},
        {'noise_type': 'poisson', 'noise_level': 0.02, 'use_wls': False, 'label': 'Poisson 2%, OLS'},
        {'noise_type': 'poisson', 'noise_level': 0.02, 'use_wls': True, 'label': 'Poisson 2%, WLS'},
    ]

    results = []

    for cfg in configs:
        print(f"\n{cfg['label']}...", end='', flush=True)

        w1_list = []
        success_list = []

        for trial in range(n_trials):
            if cfg['noise_type'] == 'gaussian':
                I_obs = add_gaussian_noise(I_true, cfg['noise_level'], seed=42 + trial)
            else:
                I_obs = add_poisson_noise(I_true, cfg['noise_level'], seed=42 + trial)

            D_opt, _, success = run_solver(I_obs, tm, elements, use_wls=cfg['use_wls'])

            success_list.append(success)
            if success:
                w1 = np.mean([compute_wasserstein(depth, D_true[:, i], D_opt[:, i]) for i in range(3)])
                w1_list.append(w1)
            else:
                w1_list.append(np.nan)

        n_success = sum(success_list)
        w1_valid = [w for w in w1_list if not np.isnan(w)]
        w1_median = np.median(w1_valid) if w1_valid else np.nan
        w1_std = np.std(w1_valid) if w1_valid else np.nan

        print(f" Success: {n_success}/{n_trials}, W1: {w1_median:.3f} ± {w1_std:.3f} Å")

        results.append({
            'label': cfg['label'],
            'n_success': n_success,
            'failure_rate': 1 - n_success / n_trials,
            'w1_median': w1_median,
            'w1_std': w1_std,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Configuration':<25} {'Success':<12} {'Failure %':<12} {'median W1':<12}")
    print("-" * 60)

    for r in results:
        print(f"{r['label']:<25} {r['n_success']}/100      {r['failure_rate']*100:.0f}%          {r['w1_median']:.3f} Å")

    # Comparison
    print("\n" + "=" * 70)
    print("KEY COMPARISONS (RMS-equivalent noise)")
    print("=" * 70)

    # Gaussian 2% vs Poisson 0.5%
    g2_ols = next(r for r in results if 'Gaussian 2%' in r['label'] and 'OLS' in r['label'])
    p05_ols = next(r for r in results if 'Poisson 0.5%' in r['label'] and 'OLS' in r['label'])
    p05_wls = next(r for r in results if 'Poisson 0.5%' in r['label'] and 'WLS' in r['label'])

    print(f"\nGaussian 2% (OLS):   W1 = {g2_ols['w1_median']:.3f} Å, Failure = {g2_ols['failure_rate']*100:.0f}%")
    print(f"Poisson 0.5% (OLS):  W1 = {p05_ols['w1_median']:.3f} Å, Failure = {p05_ols['failure_rate']*100:.0f}%")
    print(f"Poisson 0.5% (WLS):  W1 = {p05_wls['w1_median']:.3f} Å, Failure = {p05_wls['failure_rate']*100:.0f}%")

    if p05_wls['w1_median'] < p05_ols['w1_median']:
        improvement = (p05_ols['w1_median'] - p05_wls['w1_median']) / p05_ols['w1_median'] * 100
        print(f"\n→ WLS improves Poisson 0.5% by {improvement:.0f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
