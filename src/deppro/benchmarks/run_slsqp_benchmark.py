#!/usr/bin/env python
"""
Run Python SLSQP L1 benchmark and save to benchmark_results.h5.

Usage:
    python -m deppro.benchmarks.run_slsqp_benchmark
    python -m deppro.benchmarks.run_slsqp_benchmark --gt noisy   # GT_6ELEM_NOISY (default)
    python -m deppro.benchmarks.run_slsqp_benchmark --gt clean   # GT_6ELEM
    python -m deppro.benchmarks.run_slsqp_benchmark --method MEM --loops 3
"""

import argparse
import time

import numpy as np

from deppro.benchmarks.benchmark_dstk import load_dstk_data, run_benchmark
from deppro.benchmarks.benchmark_results_io import save_result, list_methods
from deppro.benchmarks.paths import GT_6ELEM, GT_6ELEM_NOISY


GT_MAP = {
    "noisy": GT_6ELEM_NOISY,
    "clean": GT_6ELEM,
}

# Method name mapping for benchmark_results.h5
METHOD_NAME_MAP = {
    "L1": "slsqp_L1",
    "LLS": "slsqp_LLS",
    "MEM": "slsqp_MEM",
    "L2": "slsqp_L2",
}


def main():
    parser = argparse.ArgumentParser(description="Run SLSQP benchmark → benchmark_results.h5")
    parser.add_argument("--gt", choices=["noisy", "clean"], default="noisy")
    parser.add_argument("--method", default="L1", choices=["L1", "LLS", "MEM", "L2"])
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    gt_path = GT_MAP[args.gt]
    result_name = METHOD_NAME_MAP[args.method]

    print(f"GT: {gt_path}")
    print(f"Method: {args.method} → save as '{result_name}'")
    print(f"Loops: {args.loops}, alpha: {args.alpha}")

    # Check existing
    existing = list_methods(gt_path)
    print(f"Existing methods: {existing}")
    if result_name in existing and not args.overwrite:
        print(f"'{result_name}' already exists. Use --overwrite to replace.")
        return

    # Load data
    data = load_dstk_data(gt_path)
    print(f"Loaded: {data.n_pixels} px, {data.n_elements} elem, "
          f"{data.n_depth} depth, {data.n_angles} angles")
    print(f"Elements: {data.element_names}")

    # Run benchmark
    t0 = time.time()
    result = run_benchmark(
        data,
        method=args.method,
        alpha=args.alpha,
        pixel_indices=np.arange(data.n_pixels),
        thickness_init=210.0,
        abruptness_init=20.0,
        verbose=True,
        store_profiles=True,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"SLSQP {args.method} completed in {elapsed:.1f}s "
          f"({data.n_pixels / elapsed:.1f} px/s)")
    print(f"Total RMSE: {result.total_rmse:.4f}")
    print(f"Total W1:   {result.total_w1:.1f} Å")
    print(f"Success:    {result.success_rate:.1%}")

    # Save to benchmark_results.h5
    save_result(
        gt_path,
        method=result_name,
        D=result.D_opt_all,
        time_sec=elapsed,
        loops=args.loops,
        config={
            "method": args.method,
            "alpha": args.alpha,
            "scipy_method": "SLSQP",
            "thickness_init": 210.0,
            "abruptness_init": 20.0,
            "total_rmse": float(result.total_rmse),
            "total_w1": float(result.total_w1),
            "success_rate": float(result.success_rate),
        },
        overwrite=args.overwrite,
    )

    print(f"\nAll methods now: {list_methods(gt_path)}")


if __name__ == "__main__":
    main()
