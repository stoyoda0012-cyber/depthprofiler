"""
WLS Benchmark on 4-Layer System (Realistic)
===========================================

4-layer: C / Al2O3 / TiO2 / SiO2 / Si
This is a more challenging problem than SiO2/Si.

Date: 2026-01-28
"""

import numpy as np
from scipy.special import erf
from typing import List, Dict
import time


def create_4layer_profile(
    depth: np.ndarray,
    thicknesses: List[float] = [5.0, 15.0, 35.0, 30.0],
    interface_widths: List[float] = [2.0, 3.0, 4.0, 5.0],
) -> np.ndarray:
    """Create 4-layer profile: C / Al2O3 / TiO2 / SiO2 / Si

    Elements: [C, Al, Ti, Si_ox, O, Si_bulk]
    """
    sigma = [w / (2 * np.sqrt(np.log(2))) for w in interface_widths]
    z = [0] + list(np.cumsum(thicknesses))

    D = np.zeros((len(depth), 6))

    # C layer
    D[:, 0] = 0.5 * (1 - erf((depth - z[1]) / sigma[0]))

    # Al2O3 layer
    al2o3 = 0.5 * (1 + erf((depth - z[1]) / sigma[0])) * 0.5 * (1 - erf((depth - z[2]) / sigma[1]))
    D[:, 1] = (2/5) * al2o3  # Al
    D[:, 4] = (3/5) * al2o3  # O

    # TiO2 layer
    tio2 = 0.5 * (1 + erf((depth - z[2]) / sigma[1])) * 0.5 * (1 - erf((depth - z[3]) / sigma[2]))
    D[:, 2] = (1/3) * tio2   # Ti
    D[:, 4] += (2/3) * tio2  # O

    # SiO2 layer
    sio2 = 0.5 * (1 + erf((depth - z[3]) / sigma[2])) * 0.5 * (1 - erf((depth - z[4]) / sigma[3]))
    D[:, 3] = (1/3) * sio2   # Si_ox
    D[:, 4] += (2/3) * sio2  # O

    # Si substrate
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


def run_solver(I_obs, tm, elements, use_wls: bool = False):
    """Run SemiFreeSolver for 4-layer system."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    # 4-layer config
    # Elements: [C, Al, Ti, Si_ox, O, Si_bulk]
    # Free: C, Al, Ti, Si_ox (indices 0,1,2,3)
    # Linked: O (index 4) - linked to oxides
    # Substrate: Si_bulk (index 5)
    semi_config = SemiFreeSolverConfig(
        thickness_init=70.0,  # rough estimate of total film thickness
        abruptness_init=5.0,
        free_indices=[0, 1, 2, 3],  # C, Al, Ti, Si_ox
        linked_indices=[4],          # O
        substrate_indices=[5],       # Si_bulk
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
    print("WLS Benchmark on 4-Layer System")
    print("C / Al2O3 / TiO2 / SiO2 / Si")
    print("SemiFreeSolver, method='L1', α=1.0, D_prior=None")
    print("=" * 70)

    # Setup
    depth = np.arange(0, 120, 4.0)  # wider range for 4 layers
    angles = [0, 10, 20, 30, 40, 50, 60]

    thicknesses = [5.0, 15.0, 35.0, 30.0]  # total ~85Å

    elements = [
        ElementConfig('C', '1s', 'C', oxidation_state=0),
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_4layer_profile(depth, thicknesses)

    tm = TransferMatrix(depth=depth, step_size=4.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    # Problem complexity
    print(f"\n問題の複雑さ:")
    print(f"  元素数: 6 (C, Al, Ti, Si_ox, O, Si_bulk)")
    print(f"  自由パラメータ: ~{len(depth) * 4} (4 free elements × {len(depth)} depth points)")
    print(f"  データ点数: {I_true.size} ({len(angles)} angles × 6 elements)")
    print(f"  強度範囲: [{I_true.min():.4f}, {I_true.max():.4f}]")
    print(f"  Dynamic range: {I_true.max()/I_true.min():.0f}x")

    # Show intensity distribution
    print(f"\n元素別強度範囲:")
    el_names = ['C', 'Al', 'Ti', 'Si_ox', 'O', 'Si_bulk']
    for i, name in enumerate(el_names):
        I_el = I_true[:, i]
        print(f"  {name:<8}: [{I_el.min():.4f}, {I_el.max():.4f}]")

    # Run benchmark
    n_trials = 50

    configs = [
        {'noise_type': 'gaussian', 'noise_level': 0.02, 'use_wls': False, 'label': 'Gaussian 2%, OLS'},
        {'noise_type': 'gaussian', 'noise_level': 0.02, 'use_wls': True, 'label': 'Gaussian 2%, WLS'},
        {'noise_type': 'poisson', 'noise_level': 0.005, 'use_wls': False, 'label': 'Poisson 0.5%, OLS'},
        {'noise_type': 'poisson', 'noise_level': 0.005, 'use_wls': True, 'label': 'Poisson 0.5%, WLS'},
        {'noise_type': 'poisson', 'noise_level': 0.02, 'use_wls': False, 'label': 'Poisson 2%, OLS'},
        {'noise_type': 'poisson', 'noise_level': 0.02, 'use_wls': True, 'label': 'Poisson 2%, WLS'},
    ]

    results = []

    print(f"\n{n_trials} trials per condition...")

    for cfg in configs:
        print(f"\n  {cfg['label']}...", end='', flush=True)
        start = time.time()

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
                # Mean W1 across all elements
                w1 = np.mean([compute_wasserstein(depth, D_true[:, i], D_opt[:, i]) for i in range(6)])
                w1_list.append(w1)
            else:
                w1_list.append(np.nan)

        elapsed = time.time() - start
        n_success = sum(success_list)
        w1_valid = [w for w in w1_list if not np.isnan(w)]
        w1_median = np.median(w1_valid) if w1_valid else np.nan
        w1_std = np.std(w1_valid) if w1_valid else np.nan

        print(f" {elapsed:.1f}s, Success: {n_success}/{n_trials}, W1: {w1_median:.2f} ± {w1_std:.2f} Å")

        results.append({
            'label': cfg['label'],
            'n_success': n_success,
            'failure_rate': 1 - n_success / n_trials,
            'w1_median': w1_median,
            'w1_std': w1_std,
        })

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (4-layer system)")
    print("=" * 70)
    print(f"\n{'Configuration':<25} {'Success':<12} {'Failure':<10} {'median W1':<12}")
    print("-" * 60)

    for r in results:
        print(f"{r['label']:<25} {r['n_success']}/{n_trials}        {r['failure_rate']*100:.0f}%         {r['w1_median']:.2f} Å")

    # OLS vs WLS comparison
    print("\n" + "=" * 70)
    print("OLS vs WLS COMPARISON")
    print("=" * 70)

    for noise_label in ['Gaussian 2%', 'Poisson 0.5%', 'Poisson 2%']:
        ols = next((r for r in results if noise_label in r['label'] and 'OLS' in r['label']), None)
        wls = next((r for r in results if noise_label in r['label'] and 'WLS' in r['label']), None)

        if ols and wls and not np.isnan(ols['w1_median']) and not np.isnan(wls['w1_median']):
            w1_change = (wls['w1_median'] - ols['w1_median']) / ols['w1_median'] * 100
            print(f"{noise_label:<15}: OLS W1={ols['w1_median']:.2f}Å, WLS W1={wls['w1_median']:.2f}Å → {w1_change:+.0f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
