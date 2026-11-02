"""
Initial Value Sensitivity Benchmark
====================================

Compare solver performance with different initial value strategies:
- near_true: Near true values (similar to previous benchmarks)
- uniform: Uniform concentrations respecting charge balance
- random: Random concentrations respecting charge balance
"""

import sys
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig


def create_test_data(noise_level=0.02, seed=42):
    from scipy.special import erf

    np.random.seed(seed)
    depth = np.arange(0, 100, 4.0)
    angles = np.array([0, 10, 20, 30, 40, 50, 60])

    elements = [
        ElementConfig('C', '1s', 'C', oxidation_state=0),
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    thicknesses = [5.0, 15.0, 35.0, 30.0]
    interface_widths = [2.0, 3.0, 4.0, 5.0]
    sigma = [w / (2 * np.sqrt(np.log(2))) for w in interface_widths]
    z = [0] + list(np.cumsum(thicknesses))

    D_true = np.zeros((len(depth), 6))
    D_true[:, 0] = 0.5 * (1 - erf((depth - z[1]) / sigma[0]))
    al2o3 = 0.5 * (1 + erf((depth - z[1]) / sigma[0])) * 0.5 * (1 - erf((depth - z[2]) / sigma[1]))
    D_true[:, 1] = (2/5) * al2o3
    D_true[:, 4] = (3/5) * al2o3
    tio2 = 0.5 * (1 + erf((depth - z[2]) / sigma[1])) * 0.5 * (1 - erf((depth - z[3]) / sigma[2]))
    D_true[:, 2] = (1/3) * tio2
    D_true[:, 4] += (2/3) * tio2
    sio2 = 0.5 * (1 + erf((depth - z[3]) / sigma[2])) * 0.5 * (1 - erf((depth - z[4]) / sigma[3]))
    D_true[:, 3] = (1/3) * sio2
    D_true[:, 4] += (2/3) * sio2
    D_true[:, 5] = 0.5 * (1 + erf((depth - z[4]) / sigma[3]))

    D_true = np.clip(D_true, 1e-6, None)
    D_true = D_true / D_true.sum(axis=1, keepdims=True)

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

    return I_obs, tm, elements, D_true, depth


def wasserstein_distance(D_true, D_opt, depth):
    """Compute mean W1 distance between two depth profiles."""
    w1_total = 0.0
    dz = depth[1] - depth[0] if len(depth) > 1 else 1.0
    for e in range(D_true.shape[1]):
        # Normalize to get probability distributions
        p_true = D_true[:, e] / (np.sum(D_true[:, e]) + 1e-10)
        p_opt = D_opt[:, e] / (np.sum(D_opt[:, e]) + 1e-10)
        # CDF
        cdf_true = np.cumsum(p_true)
        cdf_opt = np.cumsum(p_opt)
        # W1 = integral of |CDF_true - CDF_opt| * dz
        w1_total += np.sum(np.abs(cdf_true - cdf_opt)) * dz
    return w1_total / D_true.shape[1]


def create_prior(depth, init_mode, seed):
    """Create D_prior based on initialization mode."""
    from scipy.special import erf

    n_depth = len(depth)
    n_free = 4  # C, Al, Ti, Si(oxide)

    thickness_init = 70.0
    abruptness_init = 5.0
    sigma = abruptness_init / (2 * np.sqrt(np.log(2)))
    substrate_profile = 0.5 * (1 + erf((depth - thickness_init) / sigma))

    D_prior = np.zeros((n_depth, 6))
    D_prior[:, 5] = substrate_profile  # Si substrate

    film_scale = np.maximum(1 - substrate_profile, 1e-6)

    # flc = [1, 2.5, 3, 3] for C, Al, Ti, Si
    flc = np.array([1.0, 2.5, 3.0, 3.0])

    if init_mode == 'uniform':
        # Uniform: all free elements have equal concentration
        # If all equal: x=y=z=w, then sum(x*flc) = x*9.5 = film_scale
        conc = film_scale / 9.5
        D_prior[:, 0] = conc  # C
        D_prior[:, 1] = conc  # Al
        D_prior[:, 2] = conc  # Ti
        D_prior[:, 3] = conc  # Si(ox)
        D_prior[:, 4] = conc * 5.5  # O (linked)

    elif init_mode == 'random':
        # Random: random concentrations respecting constraints
        np.random.seed(seed * 1000)

        for i in range(n_depth):
            # Random weights for each element
            weights = np.random.rand(n_free)
            weights = weights / np.sum(weights * flc)  # Normalize by charge
            conc = weights * film_scale[i]

            D_prior[i, 0] = conc[0]  # C
            D_prior[i, 1] = conc[1]  # Al
            D_prior[i, 2] = conc[2]  # Ti
            D_prior[i, 3] = conc[3]  # Si(ox)
            D_prior[i, 4] = np.sum(conc * (flc - 1))  # O

    return D_prior


def create_x0(depth, init_mode, seed, n_free, free_linked_const, thickness_init, abruptness_init, n_sub):
    """
    Create initial parameter vector x0 based on initialization mode.

    All modes use D_prior=None (zero prior) for regularization.
    Only the initial value x0 is changed.
    """
    from scipy.special import erf

    n_depth = len(depth)

    # Substrate profile at initial thickness
    sigma = abruptness_init / (2.0 * np.sqrt(np.log(2.0)))
    substrate_profile = 0.5 * (1 + erf((depth - thickness_init) / sigma))
    film_scale = np.maximum(1 - substrate_profile, 1e-6)

    # Film concentrations (normalized, before scaling by film_scale)
    film_conc = np.zeros((n_depth, n_free))

    if init_mode == 'zero':
        # Zero initial values (same as D_prior=None default)
        # film_conc already zeros
        pass

    elif init_mode == 'uniform':
        # Uniform: all free elements have equal concentration
        # If all equal: x=y=z=w, then sum(x*flc) = x*sum(flc) = film_scale
        # So x = film_scale / sum(flc)
        flc_sum = np.sum(free_linked_const)
        for i in range(n_depth):
            conc = film_scale[i] / flc_sum
            for j in range(n_free):
                # Normalize by film_scale to get the optimization variable
                film_conc[i, j] = conc / film_scale[i] if film_scale[i] > 1e-10 else 0.0

    elif init_mode == 'random':
        # Random: random concentrations respecting constraints
        np.random.seed(seed * 1000)
        for i in range(n_depth):
            weights = np.random.rand(n_free)
            # Normalize so sum(weights * flc) = 1
            weights = weights / np.sum(weights * free_linked_const)
            for j in range(n_free):
                film_conc[i, j] = weights[j]

    # Clip to valid range
    for j in range(n_free):
        if free_linked_const[j] > 0:
            max_val = 1.0 / free_linked_const[j]
            film_conc[:, j] = np.clip(film_conc[:, j], 1e-6, max_val)

    # Pack into x0
    film_size = n_depth * n_free
    x0 = np.zeros(film_size + 2 + n_sub)

    # Film concentrations (column-major order)
    for j in range(n_free):
        x0[j * n_depth:(j + 1) * n_depth] = film_conc[:, j]

    # Thickness and abruptness
    x0[film_size] = thickness_init
    x0[film_size + 1] = abruptness_init

    # Substrate ratios (equal)
    if n_sub > 0:
        x0[film_size + 2:film_size + 2 + n_sub] = 1.0 / n_sub

    return x0


def solve_single_distribution(args):
    seed, init_mode = args
    I_obs, tm, elements, D_true, depth = create_test_data(noise_level=0.02, seed=42 + seed)

    # All modes use D_prior=None (zero prior) for regularization
    # Only the initial value x0 is different
    D_prior = None

    config = SolverConfig(method='L1', alpha=1.0, max_iter=100)
    semi_free_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        thickness_bounds=(10.0, 90.0),
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
    )

    solver = SemiFreeSolver(tm, config, semi_free_config, elements=elements,
                           D_prior=D_prior, use_numba=True)

    # Create initial value based on mode
    n_free = len(semi_free_config.free_indices)
    n_sub = len(semi_free_config.substrate_indices)

    if init_mode == 'zero':
        # Use default x0 from solver (computed from D_prior=None -> zeros)
        x0_override = None
    else:
        x0_override = create_x0(
            depth, init_mode, seed, n_free,
            solver.free_linked_const,
            semi_free_config.thickness_init,
            semi_free_config.abruptness_init,
            n_sub,
        )

    t0 = time.perf_counter()
    result = solver.solve(I_obs, x0_override=x0_override)
    elapsed = time.perf_counter() - t0

    # Compute accuracy
    w1 = wasserstein_distance(D_true, result.D_opt, depth)

    return seed, elapsed, result.success, w1, result.n_iterations


def run_benchmark(n_distributions, init_mode, n_workers=16):
    args_list = [(i, init_mode) for i in range(n_distributions)]

    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(solve_single_distribution, args) for args in args_list]
        results = [f.result() for f in as_completed(futures)]

    parallel_time = time.perf_counter() - t0

    total_solve_time = sum(r[1] for r in results)
    success_count = sum(1 for r in results if r[2])
    w1_values = [r[3] for r in results]
    n_iters = [r[4] for r in results]

    rate = n_distributions / parallel_time
    efficiency = (total_solve_time / parallel_time) / n_workers * 100

    return {
        'rate': rate,
        'efficiency': efficiency,
        'success_rate': success_count / n_distributions * 100,
        'w1_mean': np.mean(w1_values),
        'w1_std': np.std(w1_values),
        'iter_mean': np.mean(n_iters),
        'parallel_time': parallel_time,
    }


def main():
    n_workers = min(16, os.cpu_count() or 4)

    print('=' * 70)
    print('INITIAL VALUE SENSITIVITY BENCHMARK')
    print('=' * 70)
    print(f'Workers: {n_workers}')
    print()

    for n_dist in [100, 500, 1000, 2000, 5000, 10000]:
        print(f'\n--- {n_dist} distributions ---')
        print(f"{'Mode':<12} {'Rate':>10} {'Eff':>8} {'Success':>10} {'W1 Mean':>10} {'W1 Std':>8} {'Iters':>8}")
        print('-' * 70)

        for mode in ['zero', 'uniform', 'random']:
            r = run_benchmark(n_dist, mode, n_workers)
            print(f"{mode:<12} {r['rate']:>10.1f} {r['efficiency']:>7.1f}% {r['success_rate']:>9.1f}% {r['w1_mean']:>10.2f} {r['w1_std']:>8.2f} {r['iter_mean']:>8.1f}")

    print('\n' + '=' * 70)
    print('BENCHMARK COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()
