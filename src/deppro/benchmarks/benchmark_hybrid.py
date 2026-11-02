#!/usr/bin/env python3
"""Benchmark: NN Warm-Start + SLSQP Hybrid solver.

Compares three modes on the Checkered benchmark data:
  1. slsqp_cold  — standard SLSQP from default init (baseline)
  2. nn_only     — NN inference only (no SLSQP)
  3. hybrid      — NN init + SLSQP refinement

Usage:
    python -m deppro.benchmarks.benchmark_hybrid \
        --h5 path/to/dataset.h5 \
        --model outputs/depth_mlp_v3.npz \
        --max-pixels 100 --maxiter 10
"""

import argparse
import os
import sys
import time

import numpy as np

from deppro import (
    TransferMatrix,
    CompoundDatabase,
    SemiFreeSolver,
    SemiFreeSolverConfig,
    SolverConfig,
)
from deppro.benchmarks.benchmark_dstk import (
    load_dstk_data,
    create_tm_from_h5,
    DstkBenchmarkData,
)
from deppro.benchmarks.noise_sweep import compute_metrics_from_profiles
from deppro.hybrid_solver import solve_hybrid, HybridResult


from deppro.benchmarks.paths import GT_6ELEM

DEFAULT_H5 = GT_6ELEM
from deppro.paths import outputs_dir as _outputs_dir  # noqa: E402
DEFAULT_MODEL = str(_outputs_dir() / "depth_mlp_v3.npz")


def _create_solver(data: DstkBenchmarkData, method: str = "LLS",
                   alpha: float = 1.0, max_iter: int = 200,
                   use_xreg: bool = True) -> SemiFreeSolver:
    """Create a configured SemiFreeSolver from benchmark data."""
    tm = create_tm_from_h5(data)

    semi_config = SemiFreeSolverConfig(
        thickness_init=210.0,
        abruptness_init=20.0,
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
        max_iter=max_iter,
        scipy_method='SLSQP',
        use_sum_constraint=True,
        use_charge_constraint=True,
        max_warm_restarts=2,
        warm_restart_threshold=0.2,
        use_xreg_weighting=use_xreg,
        enable_fallback=False,
    )

    return SemiFreeSolver(
        transfer_matrix=tm,
        config=solver_config,
        semi_free_config=semi_config,
        D_prior=None,
        elements=data.elements,
    )


def run_hybrid_benchmark(
    h5_path: str,
    model_path: str,
    max_pixels: int = 100,
    maxiter_cold: int = 200,
    maxiter_hybrid: int = 10,
    method: str = "LLS",
    alpha: float = 1.0,
    n_loops: int = 10,
    use_xreg: bool = True,
    verbose: bool = True,
) -> dict:
    """Run hybrid benchmark comparing 3 modes.

    Parameters
    ----------
    h5_path : str
        Path to benchmark H5 file with Dstk/Xstk.
    model_path : str
        Path to trained NN model (.npz).
    max_pixels : int
        Number of pixels to test.
    maxiter_cold : int
        Max SLSQP iterations for cold-start.
    maxiter_hybrid : int
        Max SLSQP iterations for hybrid refinement.
    method : str
        SLSQP method (LLS, L1, etc.)
    alpha : float
        Regularization strength.
    n_loops : int
        Number of iterative loops for cold-start (with TransMat update).
    use_xreg : bool
        Use Xreg weighting.
    verbose : bool

    Returns
    -------
    dict with results for each mode
    """
    from deppro.inference import load_model

    # --- Load data ---
    print(f"Loading benchmark data: {h5_path}")
    data = load_dstk_data(h5_path)

    print(f"Loading NN model: {model_path}")
    nn_model = load_model(model_path)

    # Select pixels
    n_px = min(max_pixels, data.n_pixels)
    pixel_indices = np.arange(n_px)
    print(f"Testing {n_px} pixels, method={method}, alpha={alpha}")
    print(f"  Cold-start: maxiter={maxiter_cold}, loops={n_loops}")
    print(f"  Hybrid: maxiter={maxiter_hybrid}, loops=1")
    print()

    # Allocate storage
    modes = ['nn_only', 'slsqp_cold', 'hybrid']
    results = {}
    for mode in modes:
        results[mode] = {
            'D_opt': np.zeros((n_px, data.n_depth, data.n_elements)),
            'thickness': np.zeros(n_px),
            'chi2': np.zeros(n_px),
            'n_iterations': np.zeros(n_px, dtype=int),
            'time_nn': np.zeros(n_px),
            'time_slsqp': np.zeros(n_px),
            'success': np.zeros(n_px, dtype=bool),
        }

    # --- Mode 1: nn_only ---
    print("=" * 60)
    print("Mode: nn_only")
    print("=" * 60)
    t0 = time.perf_counter()
    solver = _create_solver(data, method, alpha, maxiter_cold, use_xreg)
    for i, px in enumerate(pixel_indices):
        I_obs = data.Xstk[px]
        hr = solve_hybrid(I_obs, solver, nn_model, maxiter=0, mode='nn_only')
        results['nn_only']['D_opt'][i] = hr.D_opt
        results['nn_only']['thickness'][i] = hr.thickness
        results['nn_only']['chi2'][i] = hr.chi2
        results['nn_only']['n_iterations'][i] = hr.n_iterations
        results['nn_only']['time_nn'][i] = hr.time_nn_s
        results['nn_only']['success'][i] = hr.success
    t_nn = time.perf_counter() - t0
    print(f"  {n_px} pixels in {t_nn:.3f}s ({n_px / t_nn:.0f} px/s)")
    print()

    # --- Mode 2: slsqp_cold (iterative with TransMat updates) ---
    print("=" * 60)
    print(f"Mode: slsqp_cold ({method}×{n_loops}+TM+Xreg)")
    print("=" * 60)

    # Pre-compute lambda_basic for TransMat updates
    compound_db = CompoundDatabase()
    tm_template = create_tm_from_h5(data)
    lambda_basic = tm_template.calculate_lambda_basic(data.elements, compound_db)

    t0 = time.perf_counter()
    for i, px in enumerate(pixel_indices):
        I_obs = data.Xstk[px]

        # Iterative loop (matching benchmark_dstk behavior)
        solver = _create_solver(data, method, alpha, maxiter_cold, use_xreg)
        x0_override = None

        for loop in range(n_loops):
            result = solver.solve(I_obs, x0_override=x0_override, verbose=False)

            # Warm-start for next loop
            x0_override = solver.build_x0(
                film_conc=result.D_opt[:, solver.film_idx],
                thickness=result.thickness,
                abruptness=result.abruptness,
                substrate_ratios=result.substrate_ratios,
            )

            # TransMat update
            if loop < n_loops - 1:
                lambda_obs = solver.tm.calculate_lambda_obs(
                    result.D_opt, lambda_basic
                )
                solver.tm.trans_mat = solver.tm.calculate_trans_mat(lambda_obs)

        results['slsqp_cold']['D_opt'][i] = result.D_opt
        results['slsqp_cold']['thickness'][i] = result.thickness
        results['slsqp_cold']['chi2'][i] = result.chi2
        results['slsqp_cold']['n_iterations'][i] = result.n_iterations
        results['slsqp_cold']['time_slsqp'][i] = time.perf_counter() - t0
        results['slsqp_cold']['success'][i] = result.success

        if verbose and (i + 1) % max(1, n_px // 10) == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{n_px}] {rate:.1f} px/s")

    t_cold = time.perf_counter() - t0
    print(f"  {n_px} pixels in {t_cold:.1f}s ({n_px / t_cold:.1f} px/s)")
    print()

    # --- Mode 3: hybrid (NN init + iterative SLSQP refinement) ---
    print("=" * 60)
    print(f"Mode: hybrid (NN init + {method}×{n_loops}, maxiter={maxiter_hybrid})")
    print("=" * 60)
    t0 = time.perf_counter()
    for i, px in enumerate(pixel_indices):
        I_obs = data.Xstk[px]

        # Phase 1: NN warm-start → first SLSQP solve
        solver = _create_solver(data, method, alpha, maxiter_hybrid, use_xreg)
        hr = solve_hybrid(I_obs, solver, nn_model, maxiter=maxiter_hybrid,
                          mode='hybrid')

        # Use the underlying SLSQP result for warm-start chain
        # Re-solve to get SemiFreeSolverResult with substrate_ratios
        # (hr already contains the optimized result)
        last_D_opt = hr.D_opt
        last_thickness = hr.thickness
        last_abruptness = hr.abruptness
        total_iters = hr.n_iterations

        # Phase 2: Iterative refinement with TransMat updates (like cold-start)
        for loop in range(1, n_loops):
            x0_override = solver.build_x0(
                film_conc=last_D_opt[:, solver.film_idx],
                thickness=last_thickness,
                abruptness=last_abruptness,
            )
            # TransMat update
            lambda_obs = solver.tm.calculate_lambda_obs(
                last_D_opt, lambda_basic
            )
            solver.tm.trans_mat = solver.tm.calculate_trans_mat(lambda_obs)

            slsqp_result = solver.solve(I_obs, x0_override=x0_override,
                                        verbose=False)
            last_D_opt = slsqp_result.D_opt
            last_thickness = slsqp_result.thickness
            last_abruptness = slsqp_result.abruptness
            total_iters += slsqp_result.n_iterations

        # Store final result
        results['hybrid']['D_opt'][i] = last_D_opt
        results['hybrid']['thickness'][i] = last_thickness
        results['hybrid']['chi2'][i] = slsqp_result.chi2 if n_loops > 1 else hr.chi2
        results['hybrid']['n_iterations'][i] = total_iters
        results['hybrid']['time_nn'][i] = hr.time_nn_s
        results['hybrid']['time_slsqp'][i] = (time.perf_counter() - t0) / (i + 1)
        results['hybrid']['success'][i] = (
            slsqp_result.success if n_loops > 1 else hr.success
        )

        if verbose and (i + 1) % max(1, n_px // 10) == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{n_px}] {rate:.1f} px/s, "
                  f"total_iters={total_iters}")

    t_hybrid = time.perf_counter() - t0
    print(f"  {n_px} pixels in {t_hybrid:.1f}s ({n_px / t_hybrid:.1f} px/s)")
    print()

    # --- Metrics ---
    D_true = data.Dstk[pixel_indices]
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    metrics = {}
    for mode in modes:
        m = compute_metrics_from_profiles(
            results[mode]['D_opt'], D_true, data.depth, data.n_elements
        )
        metrics[mode] = m

        n_iters = results[mode]['n_iterations']
        success_rate = results[mode]['success'].mean() * 100

        print(f"\n--- {mode} ---")
        print(f"  TOTAL RMSE: {m['total_rmse']:.4f}")
        print(f"  MEAN W1:    {np.nanmean(m['w1_per_element']):.1f} Å")
        print(f"  Success:    {success_rate:.1f}%")
        print(f"  Mean iters: {n_iters.mean():.1f}")

    # Element-level comparison table
    print(f"\n{'Element':<10} ", end="")
    for mode in modes:
        print(f"{'RMSE':>8s} {'W1(Å)':>7s}  ", end="")
    print()
    print("-" * 70)

    for e in range(data.n_elements):
        name = data.elements[e].symbol if hasattr(data.elements[e], 'symbol') else f"E{e}"
        print(f"{name:<10} ", end="")
        for mode in modes:
            rmse = metrics[mode]['rmse_per_element'][e]
            w1 = metrics[mode]['w1_per_element'][e]
            print(f"{rmse:8.4f} {w1:7.1f}  ", end="")
        print()

    # Timing summary
    print(f"\n{'Timing':<10} ", end="")
    times = {
        'nn_only': t_nn,
        'slsqp_cold': t_cold,
        'hybrid': t_hybrid,
    }
    for mode in modes:
        print(f"{'Time':>8s} {'px/s':>7s}  ", end="")
    print()
    for mode in modes:
        t = times[mode]
        print(f"{t:8.1f}s {n_px/t:7.1f}  ", end="")
    print()

    # Speedup
    if t_cold > 0:
        print(f"\nSpeedup: hybrid vs cold-start = {t_cold / t_hybrid:.1f}x")

    # Iteration reduction
    cold_iters = results['slsqp_cold']['n_iterations'].mean()
    hybrid_iters = results['hybrid']['n_iterations'].mean()
    print(f"Iteration reduction: {cold_iters:.1f} → {hybrid_iters:.1f} "
          f"({hybrid_iters/cold_iters*100:.0f}%)")

    return {
        'modes': modes,
        'metrics': metrics,
        'results': results,
        'times': times,
        'data': data,
        'pixel_indices': pixel_indices,
    }


def main():
    parser = argparse.ArgumentParser(description="Hybrid NN+SLSQP benchmark")
    parser.add_argument("--h5", default=DEFAULT_H5, help="Benchmark H5 path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NN model path")
    parser.add_argument("--max-pixels", type=int, default=100)
    parser.add_argument("--maxiter-cold", type=int, default=200,
                        help="Max SLSQP iters for cold-start")
    parser.add_argument("--maxiter-hybrid", type=int, default=10,
                        help="Max SLSQP iters for hybrid refinement")
    parser.add_argument("--method", default="LLS",
                        choices=["LLS", "L1", "L2", "MEM"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--loops", type=int, default=10,
                        help="Iterative loops for cold-start")
    parser.add_argument("--no-xreg", action="store_true")
    args = parser.parse_args()

    run_hybrid_benchmark(
        h5_path=args.h5,
        model_path=args.model,
        max_pixels=args.max_pixels,
        maxiter_cold=args.maxiter_cold,
        maxiter_hybrid=args.maxiter_hybrid,
        method=args.method,
        alpha=args.alpha,
        n_loops=args.loops,
        use_xreg=not args.no_xreg,
    )


if __name__ == "__main__":
    main()
