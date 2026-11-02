#!/usr/bin/env python3
"""Profile GUI Batch Solve pipeline — 20K pixel, LLS + L1.

Usage:
    cd depthprofiler/src
    PYTHONPATH=".:../toyomacro/src" \
        uv run --project ../toyomacro --python 3.12 \
        python -m deppro.benchmarks._profile_gui_pipeline [--n-pixels N] [--methods LLS,L1]
"""
from __future__ import annotations

import argparse
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory

from deppro.benchmarks.benchmark_dstk import load_dstk_data, create_tm_from_h5
from deppro.benchmarks.paths import GT_6ELEM_JITTER_NOISY
from deppro.benchmarks.benchmark_semifree_batch import make_solver


def run_pool_test(
    xstk_3d: np.ndarray,
    trans_mat_np: np.ndarray,
    config: dict,
    pixel_indices: np.ndarray,
    n_workers: int,
    label: str,
) -> dict:
    """Run ProcessPool batch solve and return timing dict."""
    from deppro.app._pixel_solve_worker import (
        pixel_solve_initializer,
        pixel_solve_strip,
    )

    n_test = len(pixel_indices)

    # SharedMemory
    shm_x = SharedMemory(create=True, size=xstk_3d.nbytes)
    buf_x = np.ndarray(xstk_3d.shape, dtype=xstk_3d.dtype, buffer=shm_x.buf)
    buf_x[:] = xstk_3d

    shm_t = SharedMemory(create=True, size=trans_mat_np.nbytes)
    buf_t = np.ndarray(trans_mat_np.shape, dtype=trans_mat_np.dtype, buffer=shm_t.buf)
    buf_t[:] = trans_mat_np

    # Strips — match GUI: max(1, min(200, n_px // n_workers))
    strip_size = max(1, min(200, n_test // n_workers))
    strips = []
    for i in range(0, n_test, strip_size):
        strips.append(pixel_indices[i : i + strip_size].tolist())

    initargs = (
        shm_x.name, xstk_3d.shape, str(xstk_3d.dtype),
        shm_t.name, trans_mat_np.shape, str(trans_mat_np.dtype),
        config,
    )

    print(f"\n{'=' * 60}")
    print(f"  {label}: {n_test} px, {n_workers} workers, {len(strips)} strips")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=pixel_solve_initializer,
        initargs=initargs,
    ) as pool:
        # Warmup
        warmup_futs = [
            pool.submit(pixel_solve_strip, [int(pixel_indices[0])])
            for _ in range(n_workers)
        ]
        for f in warmup_futs:
            f.result()
        t_warm = time.perf_counter()
        warmup_ms = (t_warm - t0) * 1000

        # Solve
        futs = {pool.submit(pixel_solve_strip, s): i for i, s in enumerate(strips)}
        n_solved = 0
        n_done_strips = 0
        for fut in as_completed(futs):
            res = fut.result()
            n_solved += sum(1 for _, d in res if d is not None)
            n_done_strips += 1
            if n_done_strips % max(1, len(strips) // 5) == 0:
                elapsed = time.perf_counter() - t_warm
                print(f"  [{n_done_strips}/{len(strips)} strips] {elapsed:.1f}s")

        t_done = time.perf_counter()

    solve_ms = (t_done - t_warm) * 1000
    total_ms = (t_done - t0) * 1000

    print(f"\n  Warmup:  {warmup_ms:.0f} ms")
    print(f"  Solve:   {solve_ms:.0f} ms ({solve_ms / n_test:.1f} ms/px wall)")
    print(f"  Total:   {total_ms:.0f} ms ({total_ms / n_test:.1f} ms/px)")
    print(f"  Success: {n_solved}/{n_test}")
    print(f"  Per-worker rate: {solve_ms / n_test * n_workers:.1f} ms/px")
    print(f"  Throughput: {n_test / (solve_ms / 1000):.0f} px/s")

    # Cleanup
    shm_x.close(); shm_x.unlink()
    shm_t.close(); shm_t.unlink()

    return {
        "label": label,
        "n_test": n_test,
        "warmup_ms": warmup_ms,
        "solve_ms": solve_ms,
        "total_ms": total_ms,
        "n_solved": n_solved,
        "per_worker_ms": solve_ms / n_test * n_workers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pixels", type=int, default=20000)
    parser.add_argument("--n-workers", type=int, default=10)
    parser.add_argument("--methods", default="LLS,L1",
                        help="Comma-separated methods (default: LLS,L1)")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]

    # Load data
    data = load_dstk_data(GT_6ELEM_JITTER_NOISY)
    n_test = min(args.n_pixels, data.n_pixels)
    pixel_indices = np.arange(n_test)
    print(f"Data: {data.n_pixels} pixels, using {n_test}")

    # Build TransferMatrix + JIT warmup
    tm = create_tm_from_h5(data)
    trans_mat_np = data.trans_mat.copy().astype(np.float64)
    warmup_solver = make_solver(data, tm, method="LLS", alpha=0.0, tol=1e-4)
    _ = warmup_solver.solve(data.Xstk[0], verbose=False)
    print("JIT warmup done")

    # xstk in GUI format: (n_px, n_elem, n_angles)
    xstk_3d = np.ascontiguousarray(
        data.Xstk.transpose(0, 2, 1).astype(np.float64)
    )

    # Base config
    base_config = {
        "solver_text": "SemiFreeSolver",
        "depth_axis": data.depth.tolist(),
        "angles": data.angles.tolist(),
        "photon_energy": data.photon_energy,
        "work_function": data.work_function,
        "gamma_eff": data.gamma_eff,
        "step_size": data.step_size,
        "n_elements": data.n_elements,
        "elements_config": None,
        "batch_ftol": 1e-4,
        "semifree_indices": {
            "free": data.free_indices,
            "linked": data.linked_indices,
            "subst": data.substrate_indices,
            "linkedconst": (
                data.linkedconst.tolist()
                if data.linkedconst is not None
                else []
            ),
        },
        "semifree_init_params": {
            "thickness_init": 210.0,
            "thickness_bounds": (50.0, 500.0),
            "abruptness_init": 20.0,
            "abruptness_bounds": (1.0, 100.0),
            "optimize_abruptness": False,
            "substrate_cutoff": None,
        },
    }

    results = []
    for method in methods:
        config = base_config.copy()
        # GUI always uses SemiFreeSolver, method selects regularization
        config["solver_text"] = "SemiFreeSolver"
        config["method"] = method
        config["alpha"] = 0.01 if method in ("L1", "MEM") else 0.0

        r = run_pool_test(
            xstk_3d, trans_mat_np, config,
            pixel_indices, args.n_workers,
            label=f"{method} ({n_test} px)",
        )
        results.append(r)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Method':<25} {'Warmup':>8} {'Solve':>10} {'Total':>10} "
          f"{'ms/px':>8} {'px/s':>8}")
    print(f"  {'-' * 65}")
    for r in results:
        throughput = r["n_test"] / (r["solve_ms"] / 1000)
        print(
            f"  {r['label']:<25} {r['warmup_ms']:>7.0f}ms "
            f"{r['solve_ms']:>9.0f}ms {r['total_ms']:>9.0f}ms "
            f"{r['total_ms'] / r['n_test']:>7.1f} "
            f"{throughput:>7.0f}"
        )
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
