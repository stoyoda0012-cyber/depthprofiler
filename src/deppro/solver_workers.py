"""
Solver Worker Functions for Parallel Depth Profiling
=====================================================

Module-level worker functions for multiprocessing-based parallel solving.
These functions must be at module level (not inside a class or function)
for pickle compatibility with multiprocessing.Pool.

Used by:
    - deppro.benchmarks.benchmark_dstk (Dstk benchmark)
    - deppro.streaming_pipeline (streaming H5 pipeline)

Functions:
    worker_initializer: Numba JIT warm-up (called once per worker process)
    worker_phase1: Phase 1 - iterative LLS/L1 solve for one pixel
    worker_phase2: Phase 2 - MEM refinement for one pixel
    build_first_neighbor_indices: N-ring spatial neighbor graph (Manhattan L1)
    neighbor_average_prior: Compute D_prior from neighbor-averaged D_opt

Date: 2026-02-14
"""

import numpy as np
from typing import List, Optional

from deppro import (
    SemiFreeSolver,
    SemiFreeSolverConfig,
    SolverConfig,
)


# ==============================================================================
# Numba JIT warm-up
# ==============================================================================

def worker_initializer():
    """Called once per worker process to warm up Numba JIT cache."""
    from deppro.numba_kernels import warmup_jit
    warmup_jit()


# ==============================================================================
# Phase 1: Iterative LLS/L1 solve
# ==============================================================================

def worker_phase1(args):
    """Worker for Phase 1: LLS/L1 iterative loop for a single pixel.

    Args:
        args: Tuple of:
            (local_idx, px, I_obs, trans_mat_data, elements, linkedconst,
             lambda_basic, method, alpha, n_loops, degree_decay, use_xreg,
             thickness_init, abruptness_init, solver_max_iter, solver_tol,
             semi_config_base_dict[, l1_smoothing, l1_beta])

    Returns:
        Tuple of (local_idx, D_opt, thickness, abruptness, chi2, objective,
                  success, warm_restarts, n_iters) -- nine values.
        On failure: (local_idx, None, 0.0, abruptness_init, 0.0, 0.0, False,
                     0, 0).
    """
    # Support both old (17-element) and new (19-element) arg tuples
    if len(args) >= 19:
        (local_idx, px, I_obs, trans_mat_data, elements, linkedconst,
         lambda_basic, method, alpha, n_loops, degree_decay, use_xreg,
         thickness_init, abruptness_init, solver_max_iter, solver_tol,
         semi_config_base_dict, l1_smoothing, l1_beta) = args
    else:
        (local_idx, px, I_obs, trans_mat_data, elements, linkedconst,
         lambda_basic, method, alpha, n_loops, degree_decay, use_xreg,
         thickness_init, abruptness_init, solver_max_iter, solver_tol,
         semi_config_base_dict) = args
        l1_smoothing = 'none'
        l1_beta = 0.01

    # Reconstruct TransferMatrix from serialized data
    tm = deserialize_transfer_matrix(trans_mat_data)

    # Film indices = free_indices (already 0-based in semi_config_base_dict)
    film_idx = semi_config_base_dict['free_indices']
    D_prior_loop = None
    thck_loop = thickness_init
    x0_override = None
    degree_loop = 1.0
    total_warm = 0

    try:
        for loop in range(n_loops):
            solver_config = SolverConfig(
                method=method, alpha=alpha, degree=degree_loop,
                max_iter=solver_max_iter, tol=solver_tol, scipy_method='SLSQP',
                use_sum_constraint=True, use_charge_constraint=True,
                max_warm_restarts=2, warm_restart_threshold=0.2,
                use_xreg_weighting=use_xreg,
                enable_fallback=False,
                l1_smoothing=l1_smoothing,
                l1_beta=l1_beta,
            )
            semi_config = SemiFreeSolverConfig(
                thickness_init=thck_loop, **semi_config_base_dict,
            )
            solver = SemiFreeSolver(
                transfer_matrix=tm, config=solver_config,
                semi_free_config=semi_config, D_prior=D_prior_loop,
                elements=elements,
            )
            result = solver.solve(I_obs, verbose=False, x0_override=x0_override)
            total_warm += result.warm_restarts

            # Updates for next loop
            x0_override = solver.build_x0(
                film_conc=result.D_opt[:, film_idx],
                thickness=result.thickness,
                abruptness=result.abruptness,
                substrate_ratios=result.substrate_ratios,
            )
            thck_loop = result.thickness
            D_prior_loop = result.D_opt.copy()

            if loop < n_loops - 1:
                lambda_obs = tm.calculate_lambda_obs(result.D_opt, lambda_basic)
                tm.trans_mat = tm.calculate_trans_mat(lambda_obs)

            if loop > 0:
                degree_loop *= degree_decay

        return (local_idx, result.D_opt, result.thickness, result.abruptness,
                result.chi2, result.objective, result.success, total_warm, result.n_iterations)
    except Exception:
        return (local_idx, None, 0.0, abruptness_init, 0.0, 0.0, False, 0, 0)


# ==============================================================================
# Phase 2: MEM refinement
# ==============================================================================

def worker_phase2(args):
    """Worker for Phase 2: MEM solve for a single pixel.

    Args:
        args: Tuple of:
            (local_idx, I_obs, D_opt_current, D_prior, thickness_current,
             abruptness_current, trans_mat_data, elements, linkedconst,
             lambda_basic, mem_alpha, use_xreg, solver_max_iter, solver_tol,
             mem_scipy_method, semi_config_base_dict, mem_space)

    Returns:
        Tuple of (local_idx, D_opt, thickness, abruptness, chi2, objective, success, warm_restarts).
        On failure: (local_idx, None, thickness_current, abruptness_current, 0.0, 0.0, False, 0).
    """
    # Support both old (16-element) and new (17-element with mem_space) arg tuples
    if len(args) == 17:
        (local_idx, I_obs, D_opt_current, D_prior, thickness_current,
         abruptness_current, trans_mat_data, elements, linkedconst,
         lambda_basic, mem_alpha, use_xreg, solver_max_iter, solver_tol,
         mem_scipy_method, semi_config_base_dict, mem_space) = args
    else:
        (local_idx, I_obs, D_opt_current, D_prior, thickness_current,
         abruptness_current, trans_mat_data, elements, linkedconst,
         lambda_basic, mem_alpha, use_xreg, solver_max_iter, solver_tol,
         mem_scipy_method, semi_config_base_dict) = args
        mem_space = 'concentration'

    from deppro import TransferMatrix

    # Film indices = free_indices (already 0-based in semi_config_base_dict)
    film_idx = semi_config_base_dict['free_indices']

    # Reconstruct TM and update IMFP
    tm = deserialize_transfer_matrix(trans_mat_data)

    lambda_obs = tm.calculate_lambda_obs(D_opt_current, lambda_basic)
    tm.trans_mat = tm.calculate_trans_mat(lambda_obs)

    try:
        mem_config = SolverConfig(
            method='MEM', alpha=mem_alpha, degree=1.0,
            max_iter=solver_max_iter, tol=solver_tol,
            scipy_method=mem_scipy_method,
            use_sum_constraint=True, use_charge_constraint=True,
            max_warm_restarts=2, warm_restart_threshold=0.2,
            use_xreg_weighting=use_xreg,
            enable_fallback=False,
            mem_space=mem_space,
        )
        semi_config = SemiFreeSolverConfig(
            thickness_init=thickness_current, **semi_config_base_dict,
        )
        solver = SemiFreeSolver(
            transfer_matrix=tm, config=mem_config,
            semi_free_config=semi_config, D_prior=D_prior,
            elements=elements,
        )
        x0 = solver.build_x0(
            film_conc=D_opt_current[:, film_idx],
            thickness=thickness_current,
            abruptness=abruptness_current,
        )
        result = solver.solve(I_obs, verbose=False, x0_override=x0)
        return (local_idx, result.D_opt, result.thickness, result.abruptness,
                result.chi2, result.objective, result.success, result.warm_restarts)
    except Exception:
        return (local_idx, None, thickness_current, abruptness_current, 0.0, 0.0, False, 0)


# ==============================================================================
# Spatial neighbor graph
# ==============================================================================

def build_first_neighbor_indices(
    pixel_indices: np.ndarray,
    grid_size: int = 50,
    nx: int = 5,
    ny: int = 5,
    n_ring: int = 1,
) -> list:
    """Build N-ring neighbor lists for pixels on a 3D grid (x, y, t).

    Pixel layout: pixel = t * (nx*ny) + y * nx + x

    Uses Manhattan (L1) distance: |dy| + |dx| <= n_ring.
    This matches MATLAB DepthProfiler.m Calculation_Callback exactly.

    N=1: 4 neighbors (up/down/left/right)
    N=2: 12 neighbors (+ diagonals + 2-step cardinal)
    N=11: ~484 neighbors (large diamond)

    Spatial neighbors only; pixels at different t are NOT neighbors.

    Args:
        pixel_indices: Global pixel indices to process
        grid_size: (deprecated, ignored) Use nx/ny instead
        nx: Spatial grid width (default: 5)
        ny: Spatial grid height (default: 5)
        n_ring: Neighbor ring radius (default: 1). Manhattan distance threshold.

    Returns:
        List of lists: neighbors[i] = list of indices *into pixel_indices* (not global)
    """
    nxy = nx * ny

    # Pre-compute offset list for the given n_ring (Manhattan diamond)
    offsets = []
    for dj in range(-n_ring, n_ring + 1):
        for di in range(-(n_ring - abs(dj)), n_ring - abs(dj) + 1):
            if dj != 0 or di != 0:
                offsets.append((dj, di))

    # Map global pixel index → local index in pixel_indices
    global_to_local = {int(px): i for i, px in enumerate(pixel_indices)}

    neighbors = []
    for i, px in enumerate(pixel_indices):
        px = int(px)
        t = px // nxy
        spatial = px % nxy
        y, x = spatial // nx, spatial % nx
        nb = []
        for dy, dx in offsets:
            y2, x2 = y + dy, x + dx
            if 0 <= y2 < ny and 0 <= x2 < nx:
                px2 = t * nxy + y2 * nx + x2
                if px2 in global_to_local:
                    nb.append(global_to_local[px2])
        neighbors.append(nb)
    return neighbors


def neighbor_average_prior(
    D_opt_all: np.ndarray,
    neighbors: list,
) -> np.ndarray:
    """Compute neighbor-averaged D_prior for each pixel.

    D_prior[i] = mean(D_opt[neighbors of i]).
    If no neighbors, D_prior[i] = D_opt[i] (self).

    Args:
        D_opt_all: Array of shape (n_pixels, n_depth, n_elements)
        neighbors: List of neighbor index lists (from build_first_neighbor_indices)

    Returns:
        D_prior: Same shape as D_opt_all
    """
    n_px = D_opt_all.shape[0]
    D_prior = np.zeros_like(D_opt_all)
    for i in range(n_px):
        nb = neighbors[i]
        if len(nb) > 0:
            D_prior[i] = np.mean(D_opt_all[nb], axis=0)
        else:
            D_prior[i] = D_opt_all[i]
    return D_prior


# ==============================================================================
# TransferMatrix serialization helpers
# ==============================================================================

def serialize_transfer_matrix(tm) -> dict:
    """Serialize a TransferMatrix to a pickle-friendly dict.

    Args:
        tm: TransferMatrix instance

    Returns:
        Dict with all fields needed to reconstruct the TransferMatrix
    """
    d = {
        'depth': tm.depth,
        'step_size': tm.step_size,
        'angles': tm.angles,
        'photon_energy': tm.photon_energy,
        'work_function': tm.work_function,
        'gamma_eff': tm.gamma_eff,
        'trans_mat': tm.trans_mat,
        'n_elements': tm.n_elements,
        '_step_sizes': tm._step_sizes,  # None for uniform grids
    }
    return d


def deserialize_transfer_matrix(tm_data: dict):
    """Reconstruct a TransferMatrix from serialized dict.

    Args:
        tm_data: Dict from serialize_transfer_matrix()

    Returns:
        TransferMatrix instance
    """
    from deppro import TransferMatrix

    tm = TransferMatrix.__new__(TransferMatrix)
    tm.depth = tm_data['depth']
    tm.step_size = tm_data['step_size']
    tm.angles = tm_data['angles']
    tm.photon_energy = tm_data['photon_energy']
    tm.work_function = tm_data['work_function']
    tm.gamma_eff = tm_data['gamma_eff']
    tm.trans_mat = tm_data['trans_mat'].copy()
    tm.n_elements = tm_data['n_elements']
    tm._step_sizes = tm_data.get('_step_sizes')  # Non-uniform grid support
    tm.n_depth = len(tm.depth)
    tm.n_angles = len(tm.angles)
    tm.lambda_basic = None
    tm.lambda_obs = None
    return tm
