"""
All Loss Functions Comparison on 4-Layer System
================================================

Compare all available loss functions:
1. OLS (Ordinary Least Squares) - default
2. WLS (Weighted Least Squares) - use_wls=True
3. Poisson NLL - use_poisson_nll=True
4. Anscombe transform - use_anscombe=True

4-layer: C / Al2O3 / TiO2 / SiO2 / Si

Date: 2026-01-28
"""

import numpy as np
from scipy.special import erf
from typing import List, Dict
import time


def create_4layer_profile(depth: np.ndarray) -> np.ndarray:
    """Create 4-layer profile: C / Al2O3 / TiO2 / SiO2 / Si"""
    thicknesses = [5.0, 15.0, 35.0, 30.0]
    interface_widths = [2.0, 3.0, 4.0, 5.0]
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


def run_solver(I_obs, tm, elements, use_wls=False, use_poisson_nll=False, use_anscombe=False):
    """Run SemiFreeSolver with specified loss function."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    semi_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
    )

    solver_config = SolverConfig(
        method='L1',
        alpha=1.0,
        max_iter=100,
        use_wls=use_wls,
        use_poisson_nll=use_poisson_nll,
        use_anscombe=use_anscombe,
    )

    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)
    result = solver.solve(I_obs, verbose=False)

    return result.D_opt, result.thickness, result.success


def main():
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    print("=" * 70)
    print("All Loss Functions Comparison (4-Layer System)")
    print("C / Al2O3 / TiO2 / SiO2 / Si")
    print("SemiFreeSolver, method='L1', α=1.0")
    print("=" * 70)

    # Setup
    depth = np.arange(0, 120, 4.0)
    angles = [0, 10, 20, 30, 40, 50, 60]

    elements = [
        ElementConfig('C', '1s', 'C', oxidation_state=0),
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_4layer_profile(depth)

    tm = TransferMatrix(depth=depth, step_size=4.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    print(f"\n問題設定:")
    print(f"  Dynamic range: {I_true.max()/I_true.min():.0f}x")
    print(f"  データ点数: {I_true.size}")

    # Loss function configurations
    loss_configs = [
        {'name': 'OLS', 'use_wls': False, 'use_poisson_nll': False, 'use_anscombe': False},
        {'name': 'WLS', 'use_wls': True, 'use_poisson_nll': False, 'use_anscombe': False},
        {'name': 'Poisson NLL', 'use_wls': False, 'use_poisson_nll': True, 'use_anscombe': False},
        {'name': 'Anscombe', 'use_wls': False, 'use_poisson_nll': False, 'use_anscombe': True},
        {'name': 'Anscombe+WLS', 'use_wls': True, 'use_poisson_nll': False, 'use_anscombe': True},
    ]

    # Noise conditions
    noise_conditions = [
        {'type': 'gaussian', 'level': 0.02, 'label': 'Gaussian 2%'},
        {'type': 'poisson', 'level': 0.005, 'label': 'Poisson 0.5%'},
        {'type': 'poisson', 'level': 0.02, 'label': 'Poisson 2%'},
    ]

    n_trials = 30

    all_results = []

    for noise_cond in noise_conditions:
        print(f"\n{'='*60}")
        print(f"Noise: {noise_cond['label']}")
        print(f"{'='*60}")

        for loss_cfg in loss_configs:
            print(f"  {loss_cfg['name']}...", end='', flush=True)
            start = time.time()

            w1_list = []
            success_list = []

            for trial in range(n_trials):
                if noise_cond['type'] == 'gaussian':
                    I_obs = add_gaussian_noise(I_true, noise_cond['level'], seed=42 + trial)
                else:
                    I_obs = add_poisson_noise(I_true, noise_cond['level'], seed=42 + trial)

                try:
                    D_opt, _, success = run_solver(
                        I_obs, tm, elements,
                        use_wls=loss_cfg['use_wls'],
                        use_poisson_nll=loss_cfg['use_poisson_nll'],
                        use_anscombe=loss_cfg['use_anscombe'],
                    )

                    success_list.append(success)
                    if success:
                        w1 = np.mean([compute_wasserstein(depth, D_true[:, i], D_opt[:, i]) for i in range(6)])
                        w1_list.append(w1)
                    else:
                        w1_list.append(np.nan)
                except Exception as e:
                    success_list.append(False)
                    w1_list.append(np.nan)

            elapsed = time.time() - start
            n_success = sum(success_list)
            w1_valid = [w for w in w1_list if not np.isnan(w)]
            w1_median = np.median(w1_valid) if w1_valid else np.nan
            w1_std = np.std(w1_valid) if w1_valid else np.nan

            print(f" {elapsed:.1f}s | Success: {n_success}/{n_trials} | W1: {w1_median:.2f} ± {w1_std:.2f} Å")

            all_results.append({
                'noise': noise_cond['label'],
                'loss': loss_cfg['name'],
                'n_success': n_success,
                'n_trials': n_trials,
                'w1_median': w1_median,
                'w1_std': w1_std,
            })

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    # Header
    print(f"\n{'Loss Function':<15}", end='')
    for nc in noise_conditions:
        print(f" {nc['label']:<18}", end='')
    print()
    print("-" * 70)

    for loss_cfg in loss_configs:
        print(f"{loss_cfg['name']:<15}", end='')
        for nc in noise_conditions:
            r = next((x for x in all_results if x['loss'] == loss_cfg['name'] and x['noise'] == nc['label']), None)
            if r and not np.isnan(r['w1_median']):
                cell = f"{r['w1_median']:.2f}Å ({r['n_success']}/{r['n_trials']})"
            else:
                cell = "FAIL"
            print(f" {cell:<18}", end='')
        print()

    # Find best for each noise condition
    print("\n" + "=" * 70)
    print("BEST PERFORMERS")
    print("=" * 70)

    for nc in noise_conditions:
        results_for_noise = [r for r in all_results if r['noise'] == nc['label'] and not np.isnan(r['w1_median'])]
        if results_for_noise:
            # Sort by success rate first, then W1
            best = min(results_for_noise, key=lambda x: (-x['n_success']/x['n_trials'], x['w1_median']))
            print(f"{nc['label']:<15}: {best['loss']} (W1={best['w1_median']:.2f}Å, {best['n_success']}/{best['n_trials']})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
