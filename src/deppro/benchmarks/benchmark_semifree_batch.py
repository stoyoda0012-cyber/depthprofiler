#!/usr/bin/env python
"""
SemiFreeSolver Batch Optimization Benchmark
============================================

Measures speedup from Phase 1-3 optimizations on 10x10x200 jitter+noisy data.

Usage:
    # Baseline (current sequential, 100 pixels)
    python -m deppro.benchmarks.benchmark_semifree_batch --baseline

    # Phase 1: warm-start (sequential, 100 pixels)
    python -m deppro.benchmarks.benchmark_semifree_batch --warmstart

    # Compare baseline vs warm-start
    python -m deppro.benchmarks.benchmark_semifree_batch --compare

    # Full run (all 20K pixels)
    python -m deppro.benchmarks.benchmark_semifree_batch --compare --full

Date: 2026-03-09
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from deppro import (
    SemiFreeSolver,
    SemiFreeSolverConfig,
    SolverConfig,
    TransferMatrix,
)
from deppro.benchmarks.benchmark_dstk import (
    DstkBenchmarkData,
    create_tm_from_h5,
    estimate_thickness,
    load_dstk_data,
    select_pixels,
)
from deppro.benchmarks.paths import GT_6ELEM_JITTER_NOISY
from deppro.depth_profile import wasserstein_distance_1d


# ==============================================================================
# ProcessPool worker (module-level for macOS spawn compatibility)
# ==============================================================================

# Per-worker globals (set by _pool_initializer, one per subprocess)
_pool_solver = None    # SemiFreeSolver (reused across tasks within a worker)
_pool_xstk = None      # np.ndarray backed by SharedMemory
_pool_dstk = None      # np.ndarray backed by SharedMemory
_pool_depth = None     # depth axis (regular ndarray copy)
_pool_shm_refs = []    # prevent GC from unmapping shared memory


def _pool_initializer(
    xstk_shm_name: str, xstk_shape: tuple, xstk_dtype: str,
    dstk_shm_name: str, dstk_shape: tuple, dstk_dtype: str,
    config: dict,
) -> None:
    """Attach SharedMemory + build SemiFreeSolver + JIT warmup.

    Called once per worker process before any tasks.
    """
    global _pool_solver, _pool_xstk, _pool_dstk, _pool_depth, _pool_shm_refs
    from multiprocessing.shared_memory import SharedMemory

    shm_x = SharedMemory(name=xstk_shm_name, create=False)
    _pool_xstk = np.ndarray(xstk_shape, dtype=xstk_dtype, buffer=shm_x.buf)

    shm_d = SharedMemory(name=dstk_shm_name, create=False)
    _pool_dstk = np.ndarray(dstk_shape, dtype=dstk_dtype, buffer=shm_d.buf)

    _pool_shm_refs = [shm_x, shm_d]
    _pool_depth = np.array(config["depth"])

    _pool_solver = _reconstruct_solver(config)

    # JIT warmup — loads Numba cache from disk
    _pool_solver.solve(_pool_xstk[0], verbose=False)


def _reconstruct_solver(config: dict):
    """Reconstruct SemiFreeSolver from picklable config dict."""
    from deppro import (
        ElementConfig,
        SemiFreeSolver,
        SemiFreeSolverConfig,
        SolverConfig,
        TransferMatrix,
    )

    depth = np.array(config["depth"])
    angles = np.array(config["angles"])
    trans_mat_np = np.array(config["trans_mat"])

    diffs = np.diff(depth)
    is_nonuniform = len(diffs) > 1 and (np.max(diffs) - np.min(diffs)) > 0.01

    if is_nonuniform:
        tm = TransferMatrix.from_nonuniform_grid(
            depth_grid=depth,
            angles=angles,
            photon_energy=config["photon_energy"],
            work_function=config["work_function"],
            gamma_eff=config["gamma_eff"],
        )
    else:
        tm = TransferMatrix(
            depth=depth,
            step_size=config["step_size"],
            angles=angles,
            photon_energy=config["photon_energy"],
            work_function=config["work_function"],
            gamma_eff=config["gamma_eff"],
        )

    tm.trans_mat = trans_mat_np
    tm.n_elements = config["n_elements"]

    elements = [ElementConfig(**ed) for ed in config["elements"]]

    semi_config = SemiFreeSolverConfig(
        thickness_init=config["thickness_init"],
        abruptness_init=config["abruptness_init"],
        free_indices=config["free_indices"],
        linked_indices=config["linked_indices"],
        substrate_indices=config["substrate_indices"],
        optimize_thickness=True,
        optimize_abruptness=False,
        thickness_bounds=tuple(config["thickness_bounds"]),
        abruptness_bounds=tuple(config["abruptness_bounds"]),
        free_linked_const_override=config["linkedconst"],
    )

    solver_config = SolverConfig(
        method=config["method"],
        alpha=config["alpha"],
        max_iter=200,
        tol=config.get("tol", 1e-6),
        scipy_method="SLSQP",
        use_sum_constraint=True,
        use_charge_constraint=True,
        max_warm_restarts=2,
        warm_restart_threshold=0.2,
        enable_fallback=False,
    )

    return SemiFreeSolver(
        transfer_matrix=tm,
        config=solver_config,
        semi_free_config=semi_config,
        D_prior=None,
        elements=elements,
    )


def _pool_solve_pixel(px_idx: int) -> tuple:
    """Solve one pixel in a worker subprocess.

    Returns (px_idx, rmse, thickness, n_iterations, success, warm_restarts).
    """
    try:
        I_obs = _pool_xstk[px_idx]
        D_true = _pool_dstk[px_idx]

        result = _pool_solver.solve(I_obs, verbose=False)

        diff = result.D_opt - D_true
        rmse = float(np.sqrt(np.mean(diff**2)))

        return (
            px_idx,
            rmse,
            float(result.thickness),
            int(result.n_iterations),
            bool(result.success),
            int(result.warm_restarts),
        )
    except Exception:
        return (px_idx, float("nan"), 0.0, 0, False, 0)


# ==============================================================================
# Result structure
# ==============================================================================

@dataclass
class BatchBenchmarkResult:
    """Result from a batch benchmark run."""

    label: str
    n_pixels: int
    total_time: float  # seconds
    ms_per_pixel: float
    total_iters: int
    iters_per_pixel: float
    rmse_mean: float
    rmse_std: float
    thickness_mae: float  # mean absolute error (A)
    success_rate: float
    warm_restarts_mean: float

    def print_report(self):
        print(f"\n{'='*60}")
        print(f"  {self.label}")
        print(f"{'='*60}")
        print(f"  Pixels:        {self.n_pixels}")
        print(f"  Total time:    {self.total_time:.1f}s")
        print(f"  ms/pixel:      {self.ms_per_pixel:.1f}")
        print(f"  iters/pixel:   {self.iters_per_pixel:.1f}")
        print(f"  Total iters:   {self.total_iters}")
        print(f"  RMSE:          {self.rmse_mean:.6f} +/- {self.rmse_std:.6f}")
        print(f"  Thickness MAE: {self.thickness_mae:.1f} A")
        print(f"  Success rate:  {self.success_rate:.1%}")
        print(f"  Warm restarts: {self.warm_restarts_mean:.2f}/pixel")
        print(f"{'='*60}")


# ==============================================================================
# Solver setup helpers
# ==============================================================================

def make_solver(
    data: DstkBenchmarkData,
    tm: TransferMatrix,
    method: str = "LLS",
    alpha: float = 0.0,
    thickness_init: float = 210.0,
    abruptness_init: float = 20.0,
    tol: float = 1e-6,
    scipy_method: str = "SLSQP",
) -> SemiFreeSolver:
    """Create a SemiFreeSolver from benchmark data (construct once)."""
    semi_config = SemiFreeSolverConfig(
        thickness_init=thickness_init,
        abruptness_init=abruptness_init,
        free_indices=data.free_indices,
        linked_indices=data.linked_indices,
        substrate_indices=data.substrate_indices,
        optimize_thickness=True,
        optimize_abruptness=False,
        thickness_bounds=(50.0, 500.0),
        abruptness_bounds=(1.0, 100.0),
        free_linked_const_override=data.linkedconst.tolist(),
    )

    solver_config = SolverConfig(
        method=method,
        alpha=alpha,
        max_iter=200,
        tol=tol,
        scipy_method=scipy_method,
        use_sum_constraint=True,
        use_charge_constraint=True,
        max_warm_restarts=2,
        warm_restart_threshold=0.2,
        enable_fallback=False,
    )

    return SemiFreeSolver(
        transfer_matrix=tm,
        config=solver_config,
        semi_free_config=semi_config,
        D_prior=None,
        elements=data.elements,
    )


# ==============================================================================
# Benchmark modes
# ==============================================================================

def _run_solve_loop(
    data: DstkBenchmarkData,
    pixel_indices: np.ndarray,
    method: str,
    alpha: float,
    reuse_solver: bool,
    verbose: bool,
    label: str,
    tol: float = 1e-6,
    scipy_method: str = "SLSQP",
) -> BatchBenchmarkResult:
    """Core solve loop with optional solver reuse."""
    n_px = len(pixel_indices)
    tm = create_tm_from_h5(data)

    rmse_px = np.zeros(n_px)
    thickness_err = np.zeros(n_px)
    iters_px = np.zeros(n_px, dtype=int)
    success_px = np.zeros(n_px, dtype=bool)
    warm_restarts_px = np.zeros(n_px, dtype=int)

    # Phase 1B: construct solver once (if reuse_solver)
    solver = make_solver(data, tm, method=method, alpha=alpha, tol=tol, scipy_method=scipy_method) if reuse_solver else None

    start = time.time()

    for i, px in enumerate(pixel_indices):
        I_obs = data.Xstk[px]
        D_true = data.Dstk[px]

        # Per-pixel construction (baseline) vs reuse
        if not reuse_solver:
            solver = make_solver(data, tm, method=method, alpha=alpha, tol=tol, scipy_method=scipy_method)

        try:
            result = solver.solve(I_obs, verbose=False)
            diff = result.D_opt - D_true
            rmse_px[i] = np.sqrt(np.mean(diff**2))
            thickness_true = estimate_thickness(D_true, data.depth)
            thickness_err[i] = abs(result.thickness - thickness_true)
            iters_px[i] = result.n_iterations
            success_px[i] = result.success
            warm_restarts_px[i] = result.warm_restarts
        except Exception as e:
            rmse_px[i] = np.nan
            success_px[i] = False
            if verbose:
                print(f"  pixel {px}: FAILED ({e})")

        if verbose and (i + 1) % max(1, n_px // 10) == 0:
            elapsed = time.time() - start
            print(
                f"  [{i+1}/{n_px}] {elapsed:.1f}s  "
                f"rmse={rmse_px[i]:.6f}  iters={iters_px[i]}",
                flush=True,
            )

    total_time = time.time() - start
    valid = ~np.isnan(rmse_px)

    return BatchBenchmarkResult(
        label=label,
        n_pixels=n_px,
        total_time=total_time,
        ms_per_pixel=total_time / n_px * 1000,
        total_iters=int(iters_px.sum()),
        iters_per_pixel=iters_px[valid].mean() if valid.any() else 0,
        rmse_mean=float(rmse_px[valid].mean()) if valid.any() else float("nan"),
        rmse_std=float(rmse_px[valid].std()) if valid.any() else float("nan"),
        thickness_mae=float(thickness_err[valid].mean()) if valid.any() else float("nan"),
        success_rate=float(success_px.mean()),
        warm_restarts_mean=float(warm_restarts_px.mean()),
    )


def run_baseline(
    data: DstkBenchmarkData,
    pixel_indices: np.ndarray,
    method: str = "LLS",
    alpha: float = 0.0,
    verbose: bool = True,
    tol: float = 1e-6,
) -> BatchBenchmarkResult:
    """Baseline: per-pixel SemiFreeSolver construction (current behavior)."""
    return _run_solve_loop(
        data, pixel_indices, method, alpha,
        reuse_solver=False, verbose=verbose,
        label=f"Baseline ({method}, alpha={alpha})",
        tol=tol,
    )


def run_reuse(
    data: DstkBenchmarkData,
    pixel_indices: np.ndarray,
    method: str = "LLS",
    alpha: float = 0.0,
    verbose: bool = True,
    tol: float = 1e-6,
) -> BatchBenchmarkResult:
    """Phase 1B: Solver constructed once, reused for all pixels."""
    return _run_solve_loop(
        data, pixel_indices, method, alpha,
        reuse_solver=True, verbose=verbose,
        label=f"Reuse ({method}, alpha={alpha})",
        tol=tol,
    )


def run_warmstart(
    data: DstkBenchmarkData,
    pixel_indices: np.ndarray,
    method: str = "LLS",
    alpha: float = 0.0,
    verbose: bool = True,
    mode: str = "structural",
    tol: float = 1e-6,
) -> BatchBenchmarkResult:
    """Phase 1: Solver reuse + cross-pixel warm-start.

    Warm-start modes:
      'full':       Pass entire x_opt from previous pixel (aggressive)
      'structural': Only warm-start thickness/abruptness, default film_conc (safe)
      'adaptive':   Try warm-start; if objective worse, re-solve with default x0
    """
    n_px = len(pixel_indices)
    tm = create_tm_from_h5(data)

    rmse_px = np.zeros(n_px)
    thickness_err = np.zeros(n_px)
    iters_px = np.zeros(n_px, dtype=int)
    success_px = np.zeros(n_px, dtype=bool)
    warm_restarts_px = np.zeros(n_px, dtype=int)
    fallback_count = 0

    # Phase 1B: construct solver ONCE
    solver = make_solver(data, tm, method=method, alpha=alpha, tol=tol)

    start = time.time()
    prev_result = None

    for i, px in enumerate(pixel_indices):
        I_obs = data.Xstk[px]
        D_true = data.Dstk[px]

        try:
            # Build warm-start x0 from previous pixel
            x0_warm = None
            if prev_result is not None:
                if mode == "full":
                    x0_warm = prev_result.x_opt
                elif mode == "structural":
                    # Only warm-start structural params, default film_conc
                    x0_warm = solver.build_x0(
                        thickness=prev_result.thickness,
                        abruptness=prev_result.abruptness,
                        substrate_ratios=prev_result.substrate_ratios,
                    )

            if mode == "adaptive" and prev_result is not None:
                # Try warm-start first
                x0_warm = prev_result.x_opt
                result_warm = solver.solve(I_obs, verbose=False, x0_override=x0_warm)
                # Also try default x0
                result_default = solver.solve(I_obs, verbose=False, x0_override=None)
                # Keep the one with lower objective
                if result_warm.objective <= result_default.objective:
                    result = result_warm
                else:
                    result = result_default
                    fallback_count += 1
            else:
                result = solver.solve(I_obs, verbose=False, x0_override=x0_warm)

            prev_result = result

            diff = result.D_opt - D_true
            rmse_px[i] = np.sqrt(np.mean(diff**2))
            thickness_true = estimate_thickness(D_true, data.depth)
            thickness_err[i] = abs(result.thickness - thickness_true)
            iters_px[i] = result.n_iterations
            success_px[i] = result.success
            warm_restarts_px[i] = result.warm_restarts
        except Exception as e:
            rmse_px[i] = np.nan
            success_px[i] = False
            prev_result = None
            if verbose:
                print(f"  pixel {px}: FAILED ({e})")

        if verbose and (i + 1) % max(1, n_px // 10) == 0:
            elapsed = time.time() - start
            print(
                f"  [{i+1}/{n_px}] {elapsed:.1f}s  "
                f"rmse={rmse_px[i]:.6f}  iters={iters_px[i]}",
                flush=True,
            )

    total_time = time.time() - start
    valid = ~np.isnan(rmse_px)

    label = f"Warm-Start-{mode} ({method}, alpha={alpha})"
    if mode == "adaptive":
        label += f" [fallback={fallback_count}/{n_px}]"

    return BatchBenchmarkResult(
        label=label,
        n_pixels=n_px,
        total_time=total_time,
        ms_per_pixel=total_time / n_px * 1000,
        total_iters=int(iters_px.sum()),
        iters_per_pixel=iters_px[valid].mean() if valid.any() else 0,
        rmse_mean=float(rmse_px[valid].mean()) if valid.any() else float("nan"),
        rmse_std=float(rmse_px[valid].std()) if valid.any() else float("nan"),
        thickness_mae=float(thickness_err[valid].mean()) if valid.any() else float("nan"),
        success_rate=float(success_px.mean()),
        warm_restarts_mean=float(warm_restarts_px.mean()),
    )


def run_alternating(
    data: DstkBenchmarkData,
    pixel_indices: np.ndarray,
    method: str = "LLS",
    alpha: float = 0.0,
    verbose: bool = True,
    tol: float = 1e-6,
) -> BatchBenchmarkResult:
    """Alternating optimization: Brent 1D thickness + QP film_conc."""
    n_px = len(pixel_indices)
    tm = create_tm_from_h5(data)

    rmse_px = np.zeros(n_px)
    thickness_err = np.zeros(n_px)
    iters_px = np.zeros(n_px, dtype=int)
    success_px = np.zeros(n_px, dtype=bool)
    brent_evals_px = np.zeros(n_px, dtype=int)

    solver = make_solver(data, tm, method=method, alpha=alpha, tol=tol)

    start = time.time()

    for i, px in enumerate(pixel_indices):
        I_obs = data.Xstk[px]
        D_true = data.Dstk[px]

        try:
            result = solver.solve_alternating(I_obs, verbose=False)
            diff = result.D_opt - D_true
            rmse_px[i] = np.sqrt(np.mean(diff**2))
            thickness_true = estimate_thickness(D_true, data.depth)
            thickness_err[i] = abs(result.thickness - thickness_true)
            iters_px[i] = result.n_iterations  # total inner SLSQP iters
            success_px[i] = result.success
            brent_evals_px[i] = result.warm_restarts  # repurposed: Brent evals
        except Exception as e:
            rmse_px[i] = np.nan
            success_px[i] = False
            if verbose:
                print(f"  pixel {px}: FAILED ({e})")

        if verbose and (i + 1) % max(1, n_px // 10) == 0:
            elapsed = time.time() - start
            print(
                f"  [{i+1}/{n_px}] {elapsed:.1f}s  "
                f"rmse={rmse_px[i]:.6f}  iters={iters_px[i]}  "
                f"brent={brent_evals_px[i]}",
                flush=True,
            )

    total_time = time.time() - start
    valid = ~np.isnan(rmse_px)

    return BatchBenchmarkResult(
        label=f"Alternating ({method}, alpha={alpha})",
        n_pixels=n_px,
        total_time=total_time,
        ms_per_pixel=total_time / n_px * 1000,
        total_iters=int(iters_px.sum()),
        iters_per_pixel=iters_px[valid].mean() if valid.any() else 0,
        rmse_mean=float(rmse_px[valid].mean()) if valid.any() else float("nan"),
        rmse_std=float(rmse_px[valid].std()) if valid.any() else float("nan"),
        thickness_mae=float(thickness_err[valid].mean()) if valid.any() else float("nan"),
        success_rate=float(success_px.mean()),
        warm_restarts_mean=float(brent_evals_px.mean()),
    )


def print_comparison(baseline: BatchBenchmarkResult, warmstart: BatchBenchmarkResult):
    """Print side-by-side comparison."""
    baseline.print_report()
    warmstart.print_report()

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    speedup = baseline.ms_per_pixel / warmstart.ms_per_pixel if warmstart.ms_per_pixel > 0 else float("inf")
    iter_reduction = 1 - warmstart.iters_per_pixel / baseline.iters_per_pixel if baseline.iters_per_pixel > 0 else 0
    rmse_delta = warmstart.rmse_mean - baseline.rmse_mean
    rmse_pct = rmse_delta / baseline.rmse_mean * 100 if baseline.rmse_mean > 0 else 0

    print(f"  Speedup:          {speedup:.2f}x")
    print(f"  Iter reduction:   {iter_reduction:.1%}")
    print(f"  RMSE delta:       {rmse_delta:+.6f} ({rmse_pct:+.2f}%)")
    print(f"  Thickness MAE:    {baseline.thickness_mae:.1f} -> {warmstart.thickness_mae:.1f} A")
    print(f"{'='*60}")


# ==============================================================================
# Phase 1D: ProcessPool parallel solve
# ==============================================================================

def _build_pool_config(
    data: DstkBenchmarkData,
    method: str = "LLS",
    alpha: float = 0.0,
    thickness_init: float = 210.0,
    abruptness_init: float = 20.0,
    tol: float = 1e-6,
) -> dict:
    """Serialize all solver configuration to a picklable dict for workers."""
    return {
        "method": method,
        "alpha": alpha,
        "depth": data.depth.tolist(),
        "angles": data.angles.tolist(),
        "trans_mat": data.trans_mat.tolist(),
        "step_size": float(data.step_size),
        "photon_energy": float(data.photon_energy),
        "work_function": float(data.work_function),
        "gamma_eff": float(data.gamma_eff),
        "n_elements": int(data.n_elements),
        "elements": [
            {
                "symbol": e.symbol,
                "orbital": e.orbital,
                "compound": e.compound,
                "binding_energy": e.binding_energy,
                "gaussian_fwhm": getattr(e, "gaussian_fwhm", 1.0),
                "lorentzian_fwhm": getattr(e, "lorentzian_fwhm", 0.3),
                "oxidation_state": getattr(e, "oxidation_state", None),
                "stoichiometry": getattr(e, "stoichiometry", None),
            }
            for e in data.elements
        ],
        "free_indices": data.free_indices,
        "linked_indices": data.linked_indices,
        "substrate_indices": data.substrate_indices,
        "linkedconst": data.linkedconst.tolist(),
        "thickness_init": thickness_init,
        "abruptness_init": abruptness_init,
        "thickness_bounds": [50.0, 500.0],
        "abruptness_bounds": [1.0, 100.0],
        "tol": tol,
    }


def run_processpool(
    data: DstkBenchmarkData,
    pixel_indices: np.ndarray,
    method: str = "LLS",
    alpha: float = 0.0,
    n_workers: int = 4,
    verbose: bool = True,
    tol: float = 1e-6,
) -> BatchBenchmarkResult:
    """Phase 1D: Parallel batch solve with ProcessPoolExecutor + SharedMemory.

    Each worker has its own SemiFreeSolver instance (no inter-process sharing).
    Xstk and Dstk are shared via POSIX SharedMemory (zero-copy reads).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing.shared_memory import SharedMemory

    n_px = len(pixel_indices)

    # ── SharedMemory for large arrays ────────────────────────────────
    shm_list = []

    shm_x = SharedMemory(create=True, size=data.Xstk.nbytes)
    buf_x = np.ndarray(data.Xstk.shape, dtype=data.Xstk.dtype, buffer=shm_x.buf)
    buf_x[:] = data.Xstk
    shm_list.append(shm_x)

    shm_d = SharedMemory(create=True, size=data.Dstk.nbytes)
    buf_d = np.ndarray(data.Dstk.shape, dtype=data.Dstk.dtype, buffer=shm_d.buf)
    buf_d[:] = data.Dstk
    shm_list.append(shm_d)

    config = _build_pool_config(data, method=method, alpha=alpha, tol=tol)

    try:
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_pool_initializer,
            initargs=(
                shm_x.name, data.Xstk.shape, str(data.Xstk.dtype),
                shm_d.name, data.Dstk.shape, str(data.Dstk.dtype),
                config,
            ),
        )

        # ── Warmup: ensure all workers are initialized ───────────────
        warmup_futs = [
            pool.submit(_pool_solve_pixel, int(pixel_indices[0]))
            for _ in range(n_workers)
        ]
        for f in warmup_futs:
            f.result()
        if verbose:
            print(f"  {n_workers} workers initialized + JIT warm")

        # ── Timed benchmark ──────────────────────────────────────────
        start = time.time()

        futures = {
            pool.submit(_pool_solve_pixel, int(px)): int(px)
            for px in pixel_indices
        }

        px_to_idx = {int(px): i for i, px in enumerate(pixel_indices)}
        rmse_px = np.full(n_px, np.nan)
        thickness_px = np.zeros(n_px)
        iters_px = np.zeros(n_px, dtype=int)
        success_px = np.zeros(n_px, dtype=bool)
        warm_restarts_px = np.zeros(n_px, dtype=int)

        done_count = 0
        for future in as_completed(futures):
            px_idx, rmse, thickness, n_iters, success, warm_restarts = (
                future.result()
            )
            i = px_to_idx.get(px_idx)
            if i is not None:
                rmse_px[i] = rmse
                thickness_px[i] = thickness
                iters_px[i] = n_iters
                success_px[i] = success
                warm_restarts_px[i] = warm_restarts

            done_count += 1
            if verbose and done_count % max(1, n_px // 10) == 0:
                elapsed = time.time() - start
                print(f"  [{done_count}/{n_px}] {elapsed:.1f}s", flush=True)

        total_time = time.time() - start
        pool.shutdown(wait=True)

    finally:
        for shm in shm_list:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    # ── Compute thickness error in main process ──────────────────────
    thickness_err = np.zeros(n_px)
    for i, px in enumerate(pixel_indices):
        D_true = data.Dstk[px]
        thickness_true = estimate_thickness(D_true, data.depth)
        thickness_err[i] = abs(thickness_px[i] - thickness_true)

    valid = ~np.isnan(rmse_px)

    return BatchBenchmarkResult(
        label=f"ProcessPool-{n_workers}w ({method}, alpha={alpha})",
        n_pixels=n_px,
        total_time=total_time,
        ms_per_pixel=total_time / n_px * 1000,
        total_iters=int(iters_px.sum()),
        iters_per_pixel=iters_px[valid].mean() if valid.any() else 0,
        rmse_mean=float(rmse_px[valid].mean()) if valid.any() else float("nan"),
        rmse_std=float(rmse_px[valid].std()) if valid.any() else float("nan"),
        thickness_mae=float(thickness_err[valid].mean()) if valid.any() else float("nan"),
        success_rate=float(success_px.mean()),
        warm_restarts_mean=float(warm_restarts_px.mean()),
    )


# ==============================================================================
# Phase 1C+D: ProcessPool with strip-based warm-start
# ==============================================================================

def _pool_solve_strip(strip_indices: list) -> list:
    """Solve a strip of pixels with solver reuse (no cross-pixel warm-start).

    The solver instance is reused across all pixels in the strip, saving
    construction overhead.  Cross-pixel warm-start is intentionally disabled
    because it degrades RMSE by preventing the warm-restart mechanism from
    re-classifying active/degenerate depths.

    Returns list of (px_idx, rmse, thickness, n_iterations, success, warm_restarts).
    """
    results = []

    for px_idx in strip_indices:
        try:
            I_obs = _pool_xstk[px_idx]
            D_true = _pool_dstk[px_idx]

            result = _pool_solver.solve(I_obs, verbose=False)

            diff = result.D_opt - D_true
            rmse = float(np.sqrt(np.mean(diff**2)))

            results.append((
                px_idx,
                rmse,
                float(result.thickness),
                int(result.n_iterations),
                bool(result.success),
                int(result.warm_restarts),
            ))
        except Exception:
            results.append((px_idx, float("nan"), 0.0, 0, False, 0))

    return results


def run_processpool_warmstart(
    data: DstkBenchmarkData,
    pixel_indices: np.ndarray,
    method: str = "LLS",
    alpha: float = 0.0,
    n_workers: int = 4,
    strip_size: int = 0,
    verbose: bool = True,
    tol: float = 1e-6,
) -> BatchBenchmarkResult:
    """Phase 1C+D: Parallel strip-based solve with warm-start chaining.

    Each strip is a contiguous range of pixels.  Within a strip, x_opt from
    the previous pixel is used as x0 for the next.  Strips are solved in
    parallel across workers.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing.shared_memory import SharedMemory

    n_px = len(pixel_indices)

    # Auto strip size: balance warm-start continuity vs load balancing
    if strip_size <= 0:
        strip_size = max(1, min(200, n_px // n_workers))

    strips = [
        [int(pixel_indices[j]) for j in range(i, min(i + strip_size, n_px))]
        for i in range(0, n_px, strip_size)
    ]

    if verbose:
        print(f"  {len(strips)} strips × ~{strip_size} pixels, {n_workers} workers")

    # ── SharedMemory for large arrays ────────────────────────────────
    shm_list = []

    shm_x = SharedMemory(create=True, size=data.Xstk.nbytes)
    buf_x = np.ndarray(data.Xstk.shape, dtype=data.Xstk.dtype, buffer=shm_x.buf)
    buf_x[:] = data.Xstk
    shm_list.append(shm_x)

    shm_d = SharedMemory(create=True, size=data.Dstk.nbytes)
    buf_d = np.ndarray(data.Dstk.shape, dtype=data.Dstk.dtype, buffer=shm_d.buf)
    buf_d[:] = data.Dstk
    shm_list.append(shm_d)

    config = _build_pool_config(data, method=method, alpha=alpha, tol=tol)

    try:
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_pool_initializer,
            initargs=(
                shm_x.name, data.Xstk.shape, str(data.Xstk.dtype),
                shm_d.name, data.Dstk.shape, str(data.Dstk.dtype),
                config,
            ),
        )

        # ── Warmup: ensure all workers are initialized ───────────────
        warmup_futs = [
            pool.submit(_pool_solve_pixel, int(pixel_indices[0]))
            for _ in range(n_workers)
        ]
        for f in warmup_futs:
            f.result()
        if verbose:
            print(f"  {n_workers} workers initialized + JIT warm")

        # ── Timed benchmark ──────────────────────────────────────────
        start = time.time()

        futures = {
            pool.submit(_pool_solve_strip, strip): strip
            for strip in strips
        }

        px_to_idx = {int(px): i for i, px in enumerate(pixel_indices)}
        rmse_px = np.full(n_px, np.nan)
        thickness_px = np.zeros(n_px)
        iters_px = np.zeros(n_px, dtype=int)
        success_px = np.zeros(n_px, dtype=bool)
        warm_restarts_px = np.zeros(n_px, dtype=int)

        done_count = 0
        for future in as_completed(futures):
            strip_results = future.result()
            for px_idx, rmse, thickness, n_iters, success, warm_restarts in strip_results:
                i = px_to_idx.get(px_idx)
                if i is not None:
                    rmse_px[i] = rmse
                    thickness_px[i] = thickness
                    iters_px[i] = n_iters
                    success_px[i] = success
                    warm_restarts_px[i] = warm_restarts
                done_count += 1

            if verbose and done_count % max(1, n_px // 10) == 0:
                elapsed = time.time() - start
                print(f"  [{done_count}/{n_px}] {elapsed:.1f}s", flush=True)

        total_time = time.time() - start
        pool.shutdown(wait=True)

    finally:
        for shm in shm_list:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    # ── Compute thickness error in main process ──────────────────────
    thickness_err = np.zeros(n_px)
    for i, px in enumerate(pixel_indices):
        D_true = data.Dstk[px]
        thickness_true = estimate_thickness(D_true, data.depth)
        thickness_err[i] = abs(thickness_px[i] - thickness_true)

    valid = ~np.isnan(rmse_px)

    return BatchBenchmarkResult(
        label=f"PP-{n_workers}w-warmstart-s{strip_size} ({method}, alpha={alpha})",
        n_pixels=n_px,
        total_time=total_time,
        ms_per_pixel=total_time / n_px * 1000,
        total_iters=int(iters_px.sum()),
        iters_per_pixel=iters_px[valid].mean() if valid.any() else 0,
        rmse_mean=float(rmse_px[valid].mean()) if valid.any() else float("nan"),
        rmse_std=float(rmse_px[valid].std()) if valid.any() else float("nan"),
        thickness_mae=float(thickness_err[valid].mean()) if valid.any() else float("nan"),
        success_rate=float(success_px.mean()),
        warm_restarts_mean=float(warm_restarts_px.mean()),
    )


# ==============================================================================
# Phase 3A: Constraint elimination + L-BFGS-B  [POC — REJECTED]
#
# Not used: see the root cause below.
# Root cause: at deep depths where erf_sub ≈ 1, sum constraint forces all
# film_conc → 0. Without explicit constraint enforcement, keep variables stay
# positive, making eliminated variable negative. No penalty/clip resolves this.
#
# Kept for reference — do NOT use in production.
# ==============================================================================

def _solve_eliminated_poc(
    solver: SemiFreeSolver,
    I_obs: np.ndarray,
    tol: float = 1e-4,
    penalty_weight: float = 0.0,  # unused — clip replaces penalty
) -> tuple:
    """Phase 3A POC: eliminate sum constraint analytically, use L-BFGS-B.

    Instead of SLSQP with N equality constraints (sum constraint at each
    active depth), solve for one free element analytically from the constraint
    and optimize remaining variables with L-BFGS-B (box bounds only).

    This removes the SLSQP QP solver step.

    Analytical gradient via:
    1. Forward pass → cache (shared with objective)
    2. gradient_backward_numba for ∂f/∂x_full (Numba backward kernel)
    3. Chain rule transform: x_full → x_red (constraint elimination)

    Returns:
        (D_opt, thickness, n_iterations, success, n_fev)
    """
    from scipy.optimize import minimize
    from scipy.special import erf as scipy_erf

    try:
        from deppro.numba_kernels import gradient_backward_numba
        _HAS_BACKWARD = True
    except ImportError:
        _HAS_BACKWARD = False

    n_depth = solver.n_depth
    n_elements = solver.n_elements
    n_film = len(solver.film_idx)
    n_sub = len(solver.sub_idx)
    flc = solver.free_linked_const.copy()
    min_conc = solver.config.min_conc

    # Substrate indices
    sub_idx_list = solver.sub_idx
    sub_idx_arr = np.array(sub_idx_list, dtype=np.int64)
    antisubst_idx = np.array(solver.free_idx + solver.linked_idx, dtype=np.int64)

    # Transfer matrices (n_elements, n_angles, n_depth)
    n_angles = len(solver.tm.angles)
    T_stacked = np.zeros((n_elements, n_angles, n_depth))
    for e in range(n_elements):
        T_stacked[e] = solver.tm.get_matrix(e)

    # Normalize observations by substrate
    I_sub = I_obs[:, sub_idx_arr].sum(axis=1, keepdims=True)
    I_norm = I_obs / np.maximum(I_sub, 1e-10)

    # Compute const_fix (constant element contribution, same as in solve())
    const_idx = solver.semi_free_config.const_indices or []
    const_fix = np.zeros(n_depth)
    if const_idx and solver.elements is not None:
        charges = np.array([
            getattr(e, 'oxidation_state', None) or 0 for e in solver.elements
        ])
        if solver.linked_idx:
            mean_lc = np.mean(charges[np.array(solver.linked_idx)])
            if mean_lc != 0:
                clc_arr = np.array([
                    max(1.0, 1.0 - charges[ci] / mean_lc) for ci in const_idx
                ])
            else:
                clc_arr = np.ones(len(const_idx))
        else:
            clc_arr = np.ones(len(const_idx))
        for i, ci in enumerate(const_idx):
            const_fix += solver.D_prior[:, ci] * clc_arr[i]

    # ── Element selection: eliminate the one with largest flc ──────────
    elim_j = int(np.argmax(flc))
    keep_j = np.array([j for j in range(n_film) if j != elim_j])
    n_keep = len(keep_j)
    flc_elim = flc[elim_j]
    flc_keep = flc[keep_j]
    elim_ub = 1.0 / flc_elim if flc_elim > 0 else 1.0

    # ── Reduced variable layout ───────────────────────────────────────
    # x_red = [film_conc_keep.ravel(), thickness, abruptness, sub_ratios]
    film_size_keep = n_depth * n_keep
    film_size_full = n_depth * n_film

    # Initial values (same as solve())
    thickness_init = solver.semi_free_config.thickness_init
    abruptness_init = solver.semi_free_config.abruptness_init
    sub_ratios_init = np.ones(n_sub) / n_sub if n_sub > 0 else np.array([])

    D_sub_init = solver._build_substrate_profile(
        thickness_init, abruptness_init, sub_ratios_init
    )
    sub_sum_init = D_sub_init.sum(axis=1, keepdims=True)
    film_frac_init = np.maximum(1.0 - sub_sum_init, min_conc)
    film_conc_init = solver.D_prior[:, solver.film_idx].copy() / film_frac_init
    for i in range(n_film):
        if flc[i] > 0:
            film_conc_init[:, i] = np.clip(
                film_conc_init[:, i], min_conc, 1.0 / flc[i]
            )

    x0 = np.concatenate([
        film_conc_init[:, keep_j].ravel(),
        [thickness_init, abruptness_init],
        sub_ratios_init,
    ])

    # Bounds for reduced variables
    keep_ub_2d = np.zeros((n_depth, n_keep))
    for ki, j in enumerate(keep_j):
        keep_ub_2d[:, ki] = 1.0 / flc[j] if flc[j] > 0 else 1.0

    lb = np.concatenate([
        np.full(film_size_keep, min_conc),
        [solver.semi_free_config.thickness_bounds[0], abruptness_init],
        np.full(n_sub, min_conc),
    ])
    ub = np.concatenate([
        keep_ub_2d.ravel(),
        [solver.semi_free_config.thickness_bounds[1], abruptness_init],
        np.full(n_sub, 1.0),
    ])
    bounds_list = list(zip(lb, ub))

    # ── Prior for regularization reference (LLS: unused but needed by kernel) ──
    m0_full = np.concatenate([
        solver.D_prior[:, solver.film_idx].ravel(),
        [solver.semi_free_config.thickness_init, abruptness_init],
        sub_ratios_init,
    ])
    m0_full[:film_size_full] = np.maximum(m0_full[:film_size_full], min_conc)

    # ── Constants for erf derivatives ─────────────────────────────────
    _2sqrt_ln2 = 2.0 * np.sqrt(np.log(2.0))
    _inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
    _depth = solver.depth.copy()
    _degree = solver.config.degree

    # ── Forward pass cache (shared between objective and gradient) ─────
    _cache = {}

    def _reconstruct_and_forward(x_red):
        """Reconstruct full profile and compute forward model."""
        fc_keep = x_red[:film_size_keep].reshape(n_depth, n_keep)
        thickness = x_red[film_size_keep]
        abruptness = x_red[film_size_keep + 1]
        sub_ratios = x_red[film_size_keep + 2: film_size_keep + 2 + n_sub]

        # Constraint elimination: eliminated element from sum constraint
        D_sub = solver._build_substrate_profile(thickness, abruptness, sub_ratios)
        erf_sub = D_sub.sum(axis=1)
        rhs = 1.0 - erf_sub - const_fix
        weighted_keep = (fc_keep * flc_keep[np.newaxis, :]).sum(axis=1)
        fc_elim = (rhs - weighted_keep) / flc_elim

        # Reconstruct full film_conc (clip eliminated element to valid bounds)
        fc_elim_clipped = np.clip(fc_elim, min_conc, elim_ub)
        film_conc = np.zeros((n_depth, n_film))
        film_conc[:, keep_j] = fc_keep
        film_conc[:, elim_j] = fc_elim_clipped

        # Build full profile
        D = solver._build_full_profile(film_conc, thickness, abruptness, sub_ratios)

        # Forward model
        I_pred = np.zeros((n_angles, n_elements))
        for e in range(n_elements):
            I_pred[:, e] = T_stacked[e] @ D[:, e]

        # Normalize by substrate
        I_pred_sub = I_pred[:, sub_idx_arr].sum(axis=1)
        I_pred_sub_safe = np.maximum(I_pred_sub, 1e-10)
        I_pred_norm = I_pred / I_pred_sub_safe[:, np.newaxis]

        # Return raw (unclipped) fc_elim for gradient clipping decisions
        return film_conc, thickness, abruptness, sub_ratios, fc_elim, D, I_pred, I_pred_norm, I_pred_sub_safe

    # ── Objective function ────────────────────────────────────────────
    def objective(x_red):
        fc_full, thick, abrp, sub_r, fc_elim_raw, D, I_pred, I_pred_norm, I_sub_sum = \
            _reconstruct_and_forward(x_red)

        # Chi2 (OLS)
        diff = I_norm[:, antisubst_idx] - I_pred_norm[:, antisubst_idx]
        chi2 = _degree * np.sum(diff ** 2)

        # Track which depths are clipped (for gradient masking)
        clipped_mask = (fc_elim_raw < min_conc) | (fc_elim_raw > elim_ub)

        # Cache for gradient
        _cache['film_conc'] = fc_full
        _cache['thickness'] = thick
        _cache['abruptness'] = abrp
        _cache['sub_ratios'] = sub_r
        _cache['fc_elim_raw'] = fc_elim_raw
        _cache['clipped_mask'] = clipped_mask
        _cache['I_pred'] = I_pred
        _cache['I_pred_norm'] = I_pred_norm
        _cache['I_sub_sum'] = I_sub_sum

        return 0.5 * chi2

    # ── Analytical gradient function ──────────────────────────────────
    def gradient(x_red):
        film_conc = _cache['film_conc']
        thickness = _cache['thickness']
        abruptness = _cache['abruptness']
        sub_ratios = _cache['sub_ratios']
        fc_elim = _cache['fc_elim_raw']
        clipped_mask = _cache['clipped_mask']
        I_pred = _cache['I_pred']
        I_pred_norm = _cache['I_pred_norm']
        I_sub_sum = _cache['I_sub_sum']

        # ── Step 1: dL/dI_pred_norm → dL/dI_pred ─────────────────────
        dL_dI_pred = np.zeros((n_angles, n_elements))
        for a in range(n_angles):
            S = I_sub_sum[a]
            S_sq = S * S
            # Accumulate for non-substrate elements
            for idx in range(len(antisubst_idx)):
                e = antisubst_idx[idx]
                r = I_norm[a, e] - I_pred_norm[a, e]
                dL_dI_pred[a, e] += -_degree * r / S
            # Substrate element contribution (normalization denominator)
            for si in range(len(sub_idx_arr)):
                sk = sub_idx_arr[si]
                grad_sk = 0.0
                for idx in range(len(antisubst_idx)):
                    e = antisubst_idx[idx]
                    r = I_norm[a, e] - I_pred_norm[a, e]
                    grad_sk += -_degree * r * (-I_pred[a, e] / S_sq)
                dL_dI_pred[a, sk] += grad_sk

        # ── Step 2: dL/dI_pred → dL/dD via T matrices ────────────────
        dL_dD = np.zeros((n_depth, n_elements))
        for e in range(n_elements):
            dL_dD[:, e] = T_stacked[e].T @ dL_dI_pred[:, e]

        # ── Step 3: dL/dD → dL/dx_full via Numba backward kernel ─────
        x_full = np.concatenate([
            film_conc.ravel(), [thickness, abruptness], sub_ratios,
        ])

        if _HAS_BACKWARD:
            grad_full = gradient_backward_numba(
                x_full, m0_full, dL_dD,
                film_conc, thickness, abruptness,
                np.asarray(sub_ratios),
                _depth, n_elements,
                solver._free_idx_array, solver._sub_idx_array,
                solver._linked_idx_array, solver.free_linked_const,
                min_conc, _degree, 0.0, 0,  # alpha=0, method=LLS
            )
        else:
            # Pure Python backward pass (slower but works without Numba)
            sigma = abruptness / _2sqrt_ln2
            erf_vals = 0.5 * (1 + scipy_erf((_depth - thickness) / sigma))
            substrate_sum_z = erf_vals * sub_ratios.sum()
            film_scale = np.maximum(1.0 - substrate_sum_z, min_conc)

            grad_full = np.zeros(len(x_full))
            n_linked = len(solver.linked_idx)
            for z in range(n_depth):
                for j in range(n_film):
                    free_e = solver.free_idx[j]
                    gv = dL_dD[z, free_e] * film_scale[z]
                    if n_linked > 0:
                        coeff = flc[j] - 1.0
                        for k in range(n_linked):
                            le = solver.linked_idx[k]
                            gv += dL_dD[z, le] * film_scale[z] * coeff / n_linked
                    grad_full[z * n_film + j] = gv
            # Thickness and sub_ratios gradients via profile build
            u = (_depth - thickness) / sigma
            d_erf_dt = (-1.0 / sigma) * _inv_sqrt_pi * np.exp(-u ** 2)
            for z in range(n_depth):
                if 1.0 - substrate_sum_z[z] >= min_conc:
                    d_fs = -sub_ratios.sum() * d_erf_dt[z]
                    for jf in range(n_film):
                        free_e = solver.free_idx[jf]
                        grad_full[film_size_full] += dL_dD[z, free_e] * film_conc[z, jf] * d_fs
                        if n_linked > 0:
                            c = flc[jf] - 1.0
                            for kl in range(n_linked):
                                le = solver.linked_idx[kl]
                                grad_full[film_size_full] += dL_dD[z, le] * film_conc[z, jf] * d_fs * c / n_linked
            # Sub ratio gradients
            for si in range(n_sub):
                for z in range(n_depth):
                    if 1.0 - substrate_sum_z[z] >= min_conc:
                        se = solver.sub_idx[si]
                        d_fs_sr = -erf_vals[z]
                        for jf in range(n_film):
                            free_e = solver.free_idx[jf]
                            grad_full[film_size_full + 2 + si] += dL_dD[z, free_e] * film_conc[z, jf] * d_fs_sr
                            if n_linked > 0:
                                c = flc[jf] - 1.0
                                for kl in range(n_linked):
                                    le = solver.linked_idx[kl]
                                    grad_full[film_size_full + 2 + si] += dL_dD[z, le] * film_conc[z, jf] * d_fs_sr * c / n_linked
                        grad_full[film_size_full + 2 + si] += dL_dD[z, se] * sub_ratios[si] * d_erf_dt[z]  # NOT correct—need ∂D_sub/∂ratio

        # ── Step 4: Chain rule transform grad_full → grad_red ─────────
        # Eliminated element depends on keep variables via constraint:
        #   fc_elim[z] = (rhs[z] - Σ flc_keep[k]*fc_keep[z,k]) / flc_elim
        # ∂fc_elim[z]/∂fc_keep[z,k] = -flc_keep[k] / flc_elim
        # ∂fc_elim[z]/∂thickness = -(∂erf_sub[z]/∂thickness) / flc_elim  (since rhs = 1 - erf_sub - cf)
        # ∂fc_elim[z]/∂sub_ratio[s] = -(∂erf_sub[z]/∂sub_ratio[s]) / flc_elim

        grad_red = np.zeros_like(x_red)

        # Film_conc keep variables
        # At clipped depths, the eliminated element is fixed (clamped),
        # so the chain rule contribution through it is zero.
        for z in range(n_depth):
            if clipped_mask[z]:
                # Eliminated element is clipped → no indirect gradient
                for ki in range(n_keep):
                    k = keep_j[ki]
                    grad_red[z * n_keep + ki] = grad_full[z * n_film + k]
            else:
                # Normal chain rule: direct + indirect through eliminated
                grad_elim_z = grad_full[z * n_film + elim_j]
                for ki in range(n_keep):
                    k = keep_j[ki]
                    grad_red[z * n_keep + ki] = (
                        grad_full[z * n_film + k]
                        + grad_elim_z * (-flc_keep[ki] / flc_elim)
                    )

        # Erf derivatives for thickness/sub_ratio chain rule
        sigma = abruptness / _2sqrt_ln2
        u = (_depth - thickness) / sigma
        exp_u2 = np.exp(-u ** 2)
        d_erf_sub_d_thickness = sub_ratios.sum() * (-1.0 / sigma) * _inv_sqrt_pi * exp_u2
        erf_profile = 0.5 * (1.0 + scipy_erf(u))

        # Thickness: grad_full already has ∂f/∂thickness via profile build.
        # Add chain rule through eliminated element (only at non-clipped depths).
        grad_red[film_size_keep] = grad_full[film_size_full]
        for z in range(n_depth):
            if not clipped_mask[z]:
                grad_red[film_size_keep] += (
                    grad_full[z * n_film + elim_j]
                    * (-d_erf_sub_d_thickness[z] / flc_elim)
                )

        # Abruptness (fixed)
        grad_red[film_size_keep + 1] = 0.0

        # Sub_ratios
        for si in range(n_sub):
            grad_red[film_size_keep + 2 + si] = grad_full[film_size_full + 2 + si]
            for z in range(n_depth):
                if not clipped_mask[z]:
                    grad_red[film_size_keep + 2 + si] += (
                        grad_full[z * n_film + elim_j]
                        * (-erf_profile[z] / flc_elim)
                    )

        return grad_red

    # ── L-BFGS-B optimization with analytical gradient ────────────────
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds_list,
        options={
            'maxiter': 200,
            'ftol': tol,   # relative function change tolerance
            'gtol': 1e-6,  # gradient norm tolerance
        },
    )

    # ── Reconstruct final D_opt ───────────────────────────────────────
    fc_keep = result.x[:film_size_keep].reshape(n_depth, n_keep)
    thickness = result.x[film_size_keep]
    abruptness = result.x[film_size_keep + 1]
    sub_ratios = result.x[film_size_keep + 2: film_size_keep + 2 + n_sub]

    D_sub = solver._build_substrate_profile(thickness, abruptness, sub_ratios)
    erf_sub = D_sub.sum(axis=1)
    rhs = 1.0 - erf_sub - const_fix
    weighted_keep = (fc_keep * flc_keep[np.newaxis, :]).sum(axis=1)
    fc_elim = (rhs - weighted_keep) / flc_elim

    film_conc = np.zeros((n_depth, n_film))
    film_conc[:, keep_j] = fc_keep
    film_conc[:, elim_j] = fc_elim

    D_opt = solver._build_full_profile(film_conc, thickness, abruptness, sub_ratios)

    return D_opt, float(thickness), int(result.nit), bool(result.success), int(result.nfev)


def run_trust_constr(
    data: DstkBenchmarkData,
    pixel_indices: np.ndarray,
    method: str = "LLS",
    alpha: float = 0.0,
    verbose: bool = True,
    tol: float = 1e-6,
) -> BatchBenchmarkResult:
    """Phase 3 alternative: trust-constr (interior point) instead of SLSQP.

    [POC — REJECTED] the Jacobian becomes singular.
    Kept for reference — do NOT use in production.
    """
    return _run_solve_loop(
        data, pixel_indices, method, alpha,
        reuse_solver=True, verbose=verbose,
        label=f"trust-constr ({method}, alpha={alpha})",
        tol=tol,
        scipy_method="trust-constr",
    )


def run_eliminated(
    data: DstkBenchmarkData,
    pixel_indices: np.ndarray,
    method: str = "LLS",
    alpha: float = 0.0,
    verbose: bool = True,
    tol: float = 1e-4,
) -> BatchBenchmarkResult:
    """Phase 3A: Constraint elimination + L-BFGS-B (sequential, single-thread).

    Eliminates sum constraint analytically and uses L-BFGS-B instead of SLSQP.
    Numerical gradient (finite differences) for initial POC validation.
    """
    n_px = len(pixel_indices)
    tm = create_tm_from_h5(data)

    rmse_px = np.zeros(n_px)
    thickness_err = np.zeros(n_px)
    iters_px = np.zeros(n_px, dtype=int)
    nfev_px = np.zeros(n_px, dtype=int)
    success_px = np.zeros(n_px, dtype=bool)

    # Build solver once (reuse across pixels)
    solver = make_solver(data, tm, method=method, alpha=alpha, tol=tol)

    start = time.time()

    for i, px in enumerate(pixel_indices):
        I_obs = data.Xstk[px]
        D_true = data.Dstk[px]

        try:
            D_opt, thickness, n_iters, success, n_fev = _solve_eliminated_poc(
                solver, I_obs, tol=tol,
            )
            diff = D_opt - D_true
            rmse_px[i] = np.sqrt(np.mean(diff ** 2))
            thickness_true = estimate_thickness(D_true, data.depth)
            thickness_err[i] = abs(thickness - thickness_true)
            iters_px[i] = n_iters
            nfev_px[i] = n_fev
            success_px[i] = success
        except Exception as e:
            rmse_px[i] = np.nan
            success_px[i] = False
            if verbose:
                print(f"  pixel {px}: FAILED ({e})")

        if verbose and (i + 1) % max(1, n_px // 10) == 0:
            elapsed = time.time() - start
            print(
                f"  [{i+1}/{n_px}] {elapsed:.1f}s  "
                f"rmse={rmse_px[i]:.6f}  iters={iters_px[i]}  nfev={nfev_px[i]}",
                flush=True,
            )

    total_time = time.time() - start
    valid = ~np.isnan(rmse_px)

    if verbose:
        print(f"  Mean nfev: {nfev_px[valid].mean():.0f}")

    return BatchBenchmarkResult(
        label=f"Eliminated-LBFGSB ({method}, alpha={alpha})",
        n_pixels=n_px,
        total_time=total_time,
        ms_per_pixel=total_time / n_px * 1000,
        total_iters=int(iters_px.sum()),
        iters_per_pixel=iters_px[valid].mean() if valid.any() else 0,
        rmse_mean=float(rmse_px[valid].mean()) if valid.any() else float("nan"),
        rmse_std=float(rmse_px[valid].std()) if valid.any() else float("nan"),
        thickness_mae=float(thickness_err[valid].mean()) if valid.any() else float("nan"),
        success_rate=float(success_px.mean()),
        warm_restarts_mean=0.0,  # no warm restarts in L-BFGS-B
    )


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SemiFreeSolver Batch Optimization Benchmark")
    parser.add_argument("--baseline", action="store_true", help="Run baseline only")
    parser.add_argument("--warmstart", action="store_true", help="Run warm-start only")
    parser.add_argument("--warmstart-mode", default="structural",
                        choices=["full", "structural", "adaptive"],
                        help="Warm-start strategy (default: structural)")
    parser.add_argument("--processpool", action="store_true", help="Run ProcessPool only")
    parser.add_argument("--compare", action="store_true",
                        help="Compare baseline vs ProcessPool")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all modes (baseline + reuse + ProcessPool scaling)")
    parser.add_argument("--method", default="LLS", help="Solver method (default: LLS)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Regularization alpha")
    parser.add_argument("--n-pixels", type=int, default=100, help="Number of pixels (default: 100)")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="ProcessPool workers (default: 4)")
    parser.add_argument("--full", action="store_true", help="Use all 20K pixels")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 pixels)")
    parser.add_argument("--start-px", type=int, default=0, help="Start pixel for consecutive selection")
    parser.add_argument("--h5", type=str, default=GT_6ELEM_JITTER_NOISY, help="H5 data path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pixel selection")
    parser.add_argument("--ftol", type=float, default=1e-6,
                        help="SLSQP convergence tolerance (default: 1e-6). "
                             "A looser tolerance trades accuracy for speed.")
    parser.add_argument("--warmstart-pool", action="store_true",
                        help="Run ProcessPool with strip-based warm-start")
    parser.add_argument("--strip-size", type=int, default=0,
                        help="Strip size for warm-start pool (0=auto)")
    parser.add_argument("--eliminated", action="store_true",
                        help="Run Phase 3A: constraint elimination + L-BFGS-B")
    parser.add_argument("--trust-constr", action="store_true",
                        help="Run Phase 3: trust-constr vs SLSQP comparison")
    parser.add_argument("--alternating", action="store_true",
                        help="Run alternating optimization: Brent 1D thickness + QP film_conc")
    args = parser.parse_args()

    # Load data
    print(f"Loading data: {args.h5}")
    data = load_dstk_data(args.h5)
    print(f"  {data.n_pixels} pixels, {data.n_elements} elements, "
          f"{data.n_angles} angles, {data.n_depth} depth points")

    # Select pixels — consecutive for spatial coherence
    if args.full:
        pixel_indices = np.arange(data.n_pixels)
    elif args.quick:
        n = 10
        pixel_indices = np.arange(args.start_px, min(args.start_px + n, data.n_pixels))
    else:
        n = min(args.n_pixels, data.n_pixels)
        pixel_indices = np.arange(args.start_px, min(args.start_px + n, data.n_pixels))

    print(f"  Selected {len(pixel_indices)} pixels")
    print()

    # Numba JIT warmup (first call compiles kernels → disk cache)
    print("Warming up Numba JIT...", flush=True)
    tm = create_tm_from_h5(data)
    warmup_solver = make_solver(data, tm, method=args.method, alpha=args.alpha, tol=args.ftol)
    _ = warmup_solver.solve(data.Xstk[pixel_indices[0]], verbose=False)
    print("  JIT warmup done\n")

    # Run benchmarks
    if args.compare or args.compare_all:
        results = []

        print("=" * 60)
        print("  Running: Baseline (sequential, per-pixel construction)")
        print("=" * 60)
        results.append(run_baseline(
            data, pixel_indices, method=args.method, alpha=args.alpha, tol=args.ftol))

        if args.compare_all:
            print(f"\n{'=' * 60}")
            print("  Running: Reuse (sequential, solver constructed once)")
            print(f"{'=' * 60}")
            results.append(run_reuse(
                data, pixel_indices, method=args.method, alpha=args.alpha, tol=args.ftol))

        print(f"\n{'=' * 60}")
        print(f"  Running: ProcessPool ({args.n_workers} workers)")
        print(f"{'=' * 60}")
        results.append(run_processpool(
            data, pixel_indices, method=args.method, alpha=args.alpha,
            n_workers=args.n_workers, tol=args.ftol,
        ))

        # Phase 1C+D: ProcessPool with strip-based warm-start
        print(f"\n{'=' * 60}")
        print(f"  Running: ProcessPool-WarmStart ({args.n_workers} workers)")
        print(f"{'=' * 60}")
        results.append(run_processpool_warmstart(
            data, pixel_indices, method=args.method, alpha=args.alpha,
            n_workers=args.n_workers, strip_size=args.strip_size,
            tol=args.ftol,
        ))

        if args.compare_all:
            # Test scaling with double workers
            n_w2 = min(args.n_workers * 2, 16)
            print(f"\n{'=' * 60}")
            print(f"  Running: ProcessPool ({n_w2} workers)")
            print(f"{'=' * 60}")
            results.append(run_processpool(
                data, pixel_indices, method=args.method, alpha=args.alpha,
                n_workers=n_w2, tol=args.ftol,
            ))

        # Print all results
        for r in results:
            r.print_report()

        # Comparison table
        print(f"\n{'='*75}")
        print(f"  {'Mode':<40s} {'ms/px':>8s} {'iter/px':>8s} {'RMSE':>10s} {'t_MAE':>8s}")
        print(f"  {'-'*71}")
        for r in results:
            print(f"  {r.label:<40s} {r.ms_per_pixel:>8.1f} {r.iters_per_pixel:>8.1f} "
                  f"{r.rmse_mean:>10.6f} {r.thickness_mae:>8.1f}")
        # Speedup vs baseline
        bl = results[0]
        for r in results[1:]:
            speedup = bl.ms_per_pixel / r.ms_per_pixel if r.ms_per_pixel > 0 else 0
            rmse_pct = (r.rmse_mean - bl.rmse_mean) / bl.rmse_mean * 100 if bl.rmse_mean > 0 else 0
            print(f"  → {r.label[:38]:<38s} speedup={speedup:.2f}x  RMSE={rmse_pct:+.1f}%")
        print(f"{'='*75}")

    elif getattr(args, 'trust_constr', False):
        # Phase 3: trust-constr vs SLSQP
        print("=" * 60)
        print("  Running: SLSQP (baseline)")
        print("=" * 60)
        bl = run_reuse(
            data, pixel_indices, method=args.method, alpha=args.alpha,
            tol=args.ftol,
        )
        bl.print_report()

        print(f"\n{'=' * 60}")
        print("  Running: trust-constr (interior point)")
        print(f"{'=' * 60}")
        tc = run_trust_constr(
            data, pixel_indices, method=args.method, alpha=args.alpha,
            tol=args.ftol,
        )
        tc.print_report()
        print_comparison(bl, tc)

    elif args.eliminated:
        # Phase 3A: constraint elimination + L-BFGS-B
        print("=" * 60)
        print("  Running: Baseline (SLSQP)")
        print("=" * 60)
        bl = run_reuse(
            data, pixel_indices, method=args.method, alpha=args.alpha,
            tol=args.ftol,
        )
        bl.print_report()

        print(f"\n{'=' * 60}")
        print("  Running: Eliminated (L-BFGS-B, numerical gradient)")
        print(f"{'=' * 60}")
        el = run_eliminated(
            data, pixel_indices, method=args.method, alpha=args.alpha,
            tol=args.ftol,
        )
        el.print_report()
        print_comparison(bl, el)

    elif args.alternating:
        # Alternating: Brent 1D thickness + QP film_conc vs baseline SLSQP
        print("=" * 60)
        print("  Running: Baseline (SLSQP joint)")
        print("=" * 60)
        bl = run_reuse(
            data, pixel_indices, method=args.method, alpha=args.alpha,
            tol=args.ftol,
        )
        bl.print_report()

        print(f"\n{'=' * 60}")
        print("  Running: Alternating (Brent + QP)")
        print(f"{'=' * 60}")
        alt = run_alternating(
            data, pixel_indices, method=args.method, alpha=args.alpha,
            tol=args.ftol,
        )
        alt.print_report()
        print_comparison(bl, alt)

    elif args.warmstart_pool:
        result = run_processpool_warmstart(
            data, pixel_indices, method=args.method, alpha=args.alpha,
            n_workers=args.n_workers, strip_size=args.strip_size,
            tol=args.ftol,
        )
        result.print_report()

    elif args.processpool:
        result = run_processpool(
            data, pixel_indices, method=args.method, alpha=args.alpha,
            n_workers=args.n_workers, tol=args.ftol,
        )
        result.print_report()

    elif args.warmstart:
        result = run_warmstart(
            data, pixel_indices, method=args.method, alpha=args.alpha,
            mode=args.warmstart_mode, tol=args.ftol,
        )
        result.print_report()

    elif args.baseline:
        result = run_baseline(data, pixel_indices, method=args.method, alpha=args.alpha,
                              tol=args.ftol)
        result.print_report()

    else:
        # Default: baseline
        result = run_baseline(data, pixel_indices, method=args.method, alpha=args.alpha,
                              tol=args.ftol)
        result.print_report()


if __name__ == "__main__":
    main()
