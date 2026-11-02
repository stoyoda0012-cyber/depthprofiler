#!/usr/bin/env python3
"""NN Warm-Start + SLSQP Refinement hybrid solver.

Uses NN inference for the initial guess and SLSQP for the physically
constrained refinement, so that SLSQP starts close to a solution.

Usage:
    from deppro.hybrid_solver import solve_hybrid, solve_hybrid_batch

    result = solve_hybrid(I_obs, solver, nn_model, maxiter=10)
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class HybridResult:
    """Result from hybrid NN + SLSQP solve."""
    D_opt: np.ndarray           # (n_depth, n_elem) refined profile
    thickness: float
    abruptness: float
    chi2: float
    success: bool
    nn_prediction: np.ndarray   # (n_depth, n_elem) raw NN output
    n_iterations: int           # SLSQP iterations used
    cost_initial: float         # objective at NN init
    cost_final: float           # objective after refinement
    mode: str                   # 'hybrid', 'nn_only', 'slsqp_cold'
    time_nn_s: float = 0.0     # NN inference time
    time_slsqp_s: float = 0.0  # SLSQP refinement time


def _nn_infer_single(I_obs, nn_model, n_depth, n_elem):
    """Run NN inference for a single pixel.

    Parameters
    ----------
    I_obs : np.ndarray, shape (n_angles, n_elem)
    nn_model : DepthMLP
    n_depth, n_elem : int

    Returns
    -------
    D_nn : np.ndarray, shape (n_depth, n_elem)
    """
    import mlx.core as mx
    from deppro.inference import enforce_constraints

    I_flat = mx.array(I_obs.reshape(1, -1).astype(np.float32))
    D_flat = np.array(nn_model(I_flat))[0]  # (n_depth * n_elem,)
    mx.eval(nn_model.parameters())

    # Enforce sum-to-1 constraint per depth
    D_flat = enforce_constraints(D_flat[np.newaxis, :], n_depth, n_elem)[0]
    return D_flat.reshape(n_depth, n_elem)


def _nn_infer_batch(I_obs_batch, nn_model, n_depth, n_elem):
    """Run NN inference for a batch of pixels.

    Parameters
    ----------
    I_obs_batch : np.ndarray, shape (n_px, n_angles, n_elem)
    nn_model : DepthMLP
    n_depth, n_elem : int

    Returns
    -------
    D_nn_batch : np.ndarray, shape (n_px, n_depth, n_elem)
    """
    import mlx.core as mx
    from deppro.inference import enforce_constraints

    n_px = I_obs_batch.shape[0]
    I_flat = mx.array(I_obs_batch.reshape(n_px, -1).astype(np.float32))
    D_flat = np.array(nn_model(I_flat))  # (n_px, n_depth * n_elem)
    mx.eval(nn_model.parameters())

    D_flat = enforce_constraints(D_flat, n_depth, n_elem)
    return D_flat.reshape(n_px, n_depth, n_elem)


def solve_hybrid(
    I_obs: np.ndarray,
    solver,  # SemiFreeSolver instance (already configured)
    nn_model,
    maxiter: int = 10,
    mode: str = 'hybrid',
) -> HybridResult:
    """Solve one pixel using NN warm-start + SLSQP refinement.

    Parameters
    ----------
    I_obs : np.ndarray, shape (n_angles, n_elem)
        Observed intensities for one pixel.
    solver : SemiFreeSolver
        Pre-configured solver (with TransMat, config, etc.)
    nn_model : DepthMLP
        Trained NN model (MLX).
    maxiter : int
        Max SLSQP iterations for refinement.
    mode : str
        'hybrid'     - NN init + SLSQP refinement (default)
        'nn_only'    - NN only, no SLSQP
        'slsqp_cold' - SLSQP cold-start (default init, no NN)

    Returns
    -------
    HybridResult
    """
    n_depth = solver.n_depth
    n_elem = solver.n_elements

    # --- Phase 1: NN inference ---
    t0_nn = time.perf_counter()
    if mode in ('hybrid', 'nn_only'):
        D_nn = _nn_infer_single(I_obs, nn_model, n_depth, n_elem)
    else:
        D_nn = np.zeros((n_depth, n_elem))
    t_nn = time.perf_counter() - t0_nn

    if mode == 'nn_only':
        # Return NN result directly (no SLSQP)
        return HybridResult(
            D_opt=D_nn,
            thickness=getattr(solver.semi_free_config, 'thickness_init', 40.0),
            abruptness=getattr(solver.semi_free_config, 'abruptness_init', 8.0),
            chi2=0.0,
            success=True,
            nn_prediction=D_nn,
            n_iterations=0,
            cost_initial=0.0,
            cost_final=0.0,
            mode='nn_only',
            time_nn_s=t_nn,
        )

    # --- Phase 2: SLSQP refinement ---
    # Override maxiter in solver config
    orig_maxiter = solver.config.max_iter
    solver.config.max_iter = maxiter

    t0_slsqp = time.perf_counter()
    if mode == 'hybrid':
        # Build x0 from NN prediction
        # D_nn has all elements; solver.build_x0 expects film elements only
        film_conc = D_nn[:, solver.film_idx]

        try:
            x0 = solver.build_x0(film_conc=film_conc)
        except Exception:
            # Fallback to cold-start if build_x0 fails
            x0 = None

        result = solver.solve(I_obs, x0_override=x0)
    else:
        # slsqp_cold: use default initialization
        result = solver.solve(I_obs)

    t_slsqp = time.perf_counter() - t0_slsqp

    # Restore original maxiter
    solver.config.max_iter = orig_maxiter

    return HybridResult(
        D_opt=result.D_opt,
        thickness=result.thickness,
        abruptness=result.abruptness,
        chi2=result.chi2,
        success=result.success,
        nn_prediction=D_nn,
        n_iterations=result.n_iterations if hasattr(result, 'n_iterations') else 0,
        cost_initial=result.cost_initial if hasattr(result, 'cost_initial') else 0.0,
        cost_final=result.chi2,
        mode=mode,
        time_nn_s=t_nn,
        time_slsqp_s=t_slsqp,
    )


def solve_hybrid_batch(
    I_obs_batch: np.ndarray,
    solver_factory,
    nn_model,
    maxiter: int = 10,
    mode: str = 'hybrid',
    n_workers: int = 0,
    verbose: bool = True,
) -> list:
    """Solve a batch of pixels using hybrid NN + SLSQP.

    Phase 1: Batch NN inference (all pixels at once, MLX GPU)
    Phase 2: Parallel SLSQP refinement (multiprocessing)

    Parameters
    ----------
    I_obs_batch : np.ndarray, shape (n_px, n_angles, n_elem)
    solver_factory : callable
        Function that returns a configured SemiFreeSolver instance.
        Called once per worker process.
    nn_model : DepthMLP
    maxiter : int
    mode : str
    n_workers : int (0=auto)
    verbose : bool

    Returns
    -------
    list of HybridResult
    """
    import multiprocessing as mp

    n_px = I_obs_batch.shape[0]

    if n_workers == 0:
        n_workers = mp.cpu_count()

    # --- Phase 1: Batch NN inference (GPU) ---
    t0_nn = time.perf_counter()
    if mode in ('hybrid', 'nn_only'):
        solver_tmp = solver_factory()
        n_depth = solver_tmp.n_depth
        n_elem = solver_tmp.n_elements
        D_nn_batch = _nn_infer_batch(I_obs_batch, nn_model, n_depth, n_elem)
    else:
        D_nn_batch = None
    t_nn = time.perf_counter() - t0_nn

    if verbose:
        print(f'  NN batch inference: {n_px} pixels in {t_nn:.3f}s '
              f'({n_px/t_nn:.0f} px/s)')

    if mode == 'nn_only':
        results = []
        for px in range(n_px):
            results.append(HybridResult(
                D_opt=D_nn_batch[px],
                thickness=0.0,
                abruptness=0.0,
                chi2=0.0,
                success=True,
                nn_prediction=D_nn_batch[px],
                n_iterations=0,
                cost_initial=0.0,
                cost_final=0.0,
                mode='nn_only',
                time_nn_s=t_nn / n_px,
            ))
        return results

    # --- Phase 2: SLSQP refinement (parallel) ---
    t0_slsqp = time.perf_counter()

    # Serial execution for now (Phase A)
    # TODO Phase C: parallel SLSQP via multiprocessing
    results = []
    solver = solver_factory()
    solver.config.max_iter = maxiter

    for px in range(n_px):
        I_obs = I_obs_batch[px]

        if mode == 'hybrid' and D_nn_batch is not None:
            D_nn = D_nn_batch[px]
            film_conc = D_nn[:, solver.film_idx]
            try:
                x0 = solver.build_x0(film_conc=film_conc)
            except Exception:
                x0 = None
            result = solver.solve(I_obs, x0_override=x0)
        else:
            result = solver.solve(I_obs)

        results.append(HybridResult(
            D_opt=result.D_opt,
            thickness=result.thickness,
            abruptness=result.abruptness,
            chi2=result.chi2,
            success=result.success,
            nn_prediction=D_nn_batch[px] if D_nn_batch is not None else np.zeros(1),
            n_iterations=getattr(result, 'n_iterations', 0),
            cost_initial=getattr(result, 'cost_initial', 0.0),
            cost_final=result.chi2,
            mode=mode,
            time_nn_s=t_nn / n_px,
            time_slsqp_s=(time.perf_counter() - t0_slsqp) / (px + 1),
        ))

        if verbose and (px + 1) % max(1, n_px // 10) == 0:
            elapsed = time.perf_counter() - t0_slsqp
            rate = (px + 1) / elapsed
            print(f'  [{px+1}/{n_px}] SLSQP refinement: {rate:.1f} px/s')

    t_slsqp = time.perf_counter() - t0_slsqp
    if verbose:
        print(f'  SLSQP total: {n_px} pixels in {t_slsqp:.1f}s '
              f'({n_px/t_slsqp:.1f} px/s)')

    return results
