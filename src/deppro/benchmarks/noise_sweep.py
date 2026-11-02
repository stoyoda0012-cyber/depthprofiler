#!/usr/bin/env python3
"""Noise sweep benchmark: SLSQP vs NN across noise levels.

Systematically adds relative noise to Xstk and measures RMSE/W1 degradation
for both Python SLSQP (neighbor-MEM) and NN (MLX DepthMLP).

Results are cached as NPZ for rapid plot iteration.

Usage:
    python -m deppro.benchmarks.noise_sweep \
        --h5 path/to/dataset.h5 \
        --model outputs/depth_mlp_v3.npz \
        --output outputs/noise_sweep_results.npz \
        --noise-levels 0 0.01 0.03 0.05 0.10 \
        --workers 0
"""

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from .benchmark_dstk import (
    DstkBenchmarkData,
    add_noise_to_xstk,
    compute_matlab_baseline,
    load_dstk_data,
    run_neighbor_mem_benchmark,
    wasserstein_distance_1d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics_from_profiles(
    D_pred: np.ndarray,
    D_true: np.ndarray,
    depth: np.ndarray,
    n_elem: int,
) -> dict:
    """Compute per-element RMSE and W1 from (n_px, n_depth, n_elem) arrays."""
    n_px = D_true.shape[0]
    rmse_elem = np.zeros(n_elem)
    w1_elem = np.zeros(n_elem)

    for e in range(n_elem):
        rmse_sq = []
        w1_vals = []
        for px in range(n_px):
            d_true = D_true[px, :, e]
            d_pred = D_pred[px, :, e]
            rmse_sq.append(np.mean((d_pred - d_true) ** 2))
            if d_true.max() > 1e-4:
                w1_vals.append(wasserstein_distance_1d(depth, d_true, d_pred))
        rmse_elem[e] = np.sqrt(np.mean(rmse_sq))
        w1_elem[e] = np.mean(w1_vals) if w1_vals else 0.0

    total_rmse = np.sqrt(np.mean([
        np.mean((D_pred[px] - D_true[px]) ** 2) for px in range(n_px)
    ]))

    return {
        'rmse_per_element': rmse_elem,
        'w1_per_element': w1_elem,
        'total_rmse': total_rmse,
        'mean_w1': np.mean(w1_elem),
    }


def _load_py_drlt_from_h5(h5_path: str) -> Optional[np.ndarray]:
    """Load Py_Drlt (Python SLSQP result) from H5 if it exists."""
    with h5py.File(h5_path, 'r') as f:
        if 'Py_Drlt' in f:
            # MATLAB layout (px, elem, depth) → Python (px, depth, elem)
            return np.transpose(f['Py_Drlt'][:].astype(np.float64), (0, 2, 1))
    return None


# ---------------------------------------------------------------------------
# SLSQP sweep
# ---------------------------------------------------------------------------

def run_slsqp_sweep(
    data_clean: DstkBenchmarkData,
    noise_levels: List[float],
    h5_path: str,
    n_workers: int = 0,
    nx: int = 5,
    ny: int = 5,
) -> dict:
    """Run SLSQP neighbor-MEM at each noise level.

    Returns dict with arrays (n_levels, n_elem) for rmse/w1.
    """
    # Auto-detect workers
    if n_workers == 0:
        import multiprocessing as mp
        n_workers = mp.cpu_count()
        print(f"  Auto-detected {n_workers} CPU cores")

    n_levels = len(noise_levels)
    n_elem = data_clean.Dstk.shape[2]
    depth = data_clean.depth

    rmse_all = np.zeros((n_levels, n_elem))
    w1_all = np.zeros((n_levels, n_elem))
    total_rmse = np.zeros(n_levels)
    mean_w1 = np.zeros(n_levels)
    times = np.zeros(n_levels)

    for i, sigma in enumerate(noise_levels):
        print(f"\n{'='*60}")
        print(f"SLSQP noise={sigma:.2%} ({i+1}/{n_levels})")
        print(f"{'='*60}")

        if sigma == 0:
            # Try to reuse existing Py_Drlt
            D_py = _load_py_drlt_from_h5(h5_path)
            if D_py is not None:
                print("  Reusing Py_Drlt from H5 (noise=0 already computed)")
                t0 = time.perf_counter()
                metrics = compute_metrics_from_profiles(
                    D_py, data_clean.Dstk, depth, n_elem)
                elapsed = time.perf_counter() - t0
                rmse_all[i] = metrics['rmse_per_element']
                w1_all[i] = metrics['w1_per_element']
                total_rmse[i] = metrics['total_rmse']
                mean_w1[i] = metrics['mean_w1']
                times[i] = elapsed
                print(f"  RMSE={total_rmse[i]:.4f}, W1={mean_w1[i]:.1f}Å ({elapsed:.1f}s)")
                continue

        # Deep copy + add noise
        data = copy.deepcopy(data_clean)
        if sigma > 0:
            add_noise_to_xstk(data, sigma, "relative", seed=123)

        t0 = time.perf_counter()
        result = run_neighbor_mem_benchmark(
            data,
            phase1_method='LLS',
            phase1_loops=10,
            phase1_alpha=1.0,
            use_xreg=True,
            mem_alpha=0.01,
            mem_loops=5,
            nx=nx, ny=ny,
            n_workers=n_workers,
            verbose=True,
        )
        elapsed = time.perf_counter() - t0

        rmse_all[i] = result.rmse_per_element
        w1_all[i] = result.w1_per_element if result.w1_per_element is not None else 0
        total_rmse[i] = result.total_rmse
        mean_w1[i] = result.total_w1 if result.total_w1 is not None else 0
        times[i] = elapsed
        print(f"  RMSE={total_rmse[i]:.4f}, W1={mean_w1[i]:.1f}Å ({elapsed:.1f}s)")

    return {
        'rmse': rmse_all,
        'w1': w1_all,
        'total_rmse': total_rmse,
        'mean_w1': mean_w1,
        'time': times,
    }


# ---------------------------------------------------------------------------
# NN sweep
# ---------------------------------------------------------------------------

def run_nn_sweep(
    data_clean: DstkBenchmarkData,
    noise_levels: List[float],
    model_path: str,
) -> dict:
    """Run NN inference at each noise level.

    Returns dict with arrays (n_levels, n_elem) for rmse/w1.
    """
    import mlx.core as mx
    from deppro.inference import enforce_constraints, load_model

    model = load_model(model_path)
    n_levels = len(noise_levels)
    n_elem = data_clean.Dstk.shape[2]
    n_depth = data_clean.Dstk.shape[1]
    n_px = data_clean.Dstk.shape[0]
    depth = data_clean.depth

    # Xstk shape: (n_px, n_angles, n_elem)
    Xstk_clean = data_clean.Xstk.copy()

    rmse_all = np.zeros((n_levels, n_elem))
    w1_all = np.zeros((n_levels, n_elem))
    total_rmse = np.zeros(n_levels)
    mean_w1 = np.zeros(n_levels)

    for i, sigma in enumerate(noise_levels):
        print(f"  NN noise={sigma:.2%}...", end='', flush=True)

        # Apply noise (same formula as add_noise_to_xstk)
        Xstk = Xstk_clean.copy()
        if sigma > 0:
            rng = np.random.default_rng(123)
            noise = rng.normal(0, sigma, Xstk.shape)
            Xstk = Xstk * (1 + noise)
            np.maximum(Xstk, 0, out=Xstk)

        # Inference
        I_obs = Xstk.reshape(n_px, -1).astype(np.float32)
        D_nn_flat = np.array(model(mx.array(I_obs)))
        mx.eval(model.parameters())
        D_nn_flat = enforce_constraints(D_nn_flat, n_depth, n_elem)
        D_nn = D_nn_flat.reshape(n_px, n_depth, n_elem)

        # Metrics
        metrics = compute_metrics_from_profiles(D_nn, data_clean.Dstk, depth, n_elem)
        rmse_all[i] = metrics['rmse_per_element']
        w1_all[i] = metrics['w1_per_element']
        total_rmse[i] = metrics['total_rmse']
        mean_w1[i] = metrics['mean_w1']
        print(f" RMSE={total_rmse[i]:.4f}, W1={mean_w1[i]:.1f}Å")

    return {
        'rmse': rmse_all,
        'w1': w1_all,
        'total_rmse': total_rmse,
        'mean_w1': mean_w1,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_noise_sweep(
    h5_path: str,
    nn_model_path: str,
    output_path: str,
    noise_levels: List[float],
    n_workers: int = 0,
    skip_slsqp: bool = False,
    load_cache: Optional[str] = None,
):
    """Run full noise sweep: SLSQP + NN at each noise level."""
    print(f"Loading data from {h5_path}...")
    data_clean = load_dstk_data(h5_path)
    n_elem = data_clean.Dstk.shape[2]
    n_levels = len(noise_levels)
    element_names = data_clean.element_names

    print(f"  {data_clean.Dstk.shape[0]} pixels, {n_elem} elements, "
          f"{len(noise_levels)} noise levels")

    # MATLAB baseline (noise=0 only)
    print("\nComputing MATLAB baseline (noise=0)...")
    ml_metrics = compute_metrics_from_profiles(
        data_clean.Drlt, data_clean.Dstk, data_clean.depth, n_elem)
    print(f"  MATLAB: RMSE={ml_metrics['total_rmse']:.4f}, W1={ml_metrics['mean_w1']:.1f}Å")

    # Load cached SLSQP if requested
    slsqp_results = None
    if load_cache:
        print(f"\nLoading cached SLSQP results from {load_cache}...")
        cached = np.load(load_cache, allow_pickle=True)
        cached_levels = cached['noise_levels']
        if np.allclose(cached_levels, noise_levels):
            slsqp_results = {
                'rmse': cached['slsqp_rmse'],
                'w1': cached['slsqp_w1'],
                'total_rmse': cached['slsqp_total_rmse'],
                'mean_w1': cached['slsqp_mean_w1'],
                'time': cached['slsqp_time'],
            }
            print("  Cache loaded successfully.")
        else:
            print(f"  Cache noise levels {cached_levels} != requested {noise_levels}, ignoring.")

    # SLSQP sweep
    if slsqp_results is None and not skip_slsqp:
        print("\n" + "=" * 60)
        print("SLSQP NOISE SWEEP")
        print("=" * 60)
        slsqp_results = run_slsqp_sweep(
            data_clean, noise_levels, h5_path, n_workers=n_workers)

    # NN sweep
    print("\n" + "=" * 60)
    print("NN NOISE SWEEP")
    print("=" * 60)
    nn_results = run_nn_sweep(data_clean, noise_levels, nn_model_path)

    # Save NPZ
    save_dict = {
        'noise_levels': np.array(noise_levels),
        'element_names': np.array(element_names),
        # NN
        'nn_rmse': nn_results['rmse'],
        'nn_w1': nn_results['w1'],
        'nn_total_rmse': nn_results['total_rmse'],
        'nn_mean_w1': nn_results['mean_w1'],
        # MATLAB (noise=0 only)
        'matlab_rmse': ml_metrics['rmse_per_element'],
        'matlab_w1': ml_metrics['w1_per_element'],
        'matlab_total_rmse': np.float64(ml_metrics['total_rmse']),
        'matlab_mean_w1': np.float64(ml_metrics['mean_w1']),
    }

    if slsqp_results is not None:
        save_dict.update({
            'slsqp_rmse': slsqp_results['rmse'],
            'slsqp_w1': slsqp_results['w1'],
            'slsqp_total_rmse': slsqp_results['total_rmse'],
            'slsqp_mean_w1': slsqp_results['mean_w1'],
            'slsqp_time': slsqp_results['time'],
        })

    np.savez(output_path, **save_dict)
    print(f"\nSaved: {output_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("NOISE SWEEP SUMMARY")
    print(f"{'='*70}")
    header = f"{'Noise':>8s}"
    if slsqp_results is not None:
        header += f"  {'SLSQP RMSE':>11s} {'SLSQP W1':>9s} {'time':>6s}"
    header += f"  {'NN RMSE':>8s} {'NN W1':>6s}"
    print(header)
    print("-" * 70)

    for i, sigma in enumerate(noise_levels):
        row = f"{sigma:>7.0%}"
        if slsqp_results is not None:
            row += (f"  {slsqp_results['total_rmse'][i]:>10.4f} "
                    f"{slsqp_results['mean_w1'][i]:>8.1f}Å "
                    f"{slsqp_results['time'][i]:>5.0f}s")
        row += (f"  {nn_results['total_rmse'][i]:>7.4f} "
                f"{nn_results['mean_w1'][i]:>5.1f}Å")
        print(row)

    print("-" * 70)
    print(f"MATLAB (noise=0): RMSE={ml_metrics['total_rmse']:.4f}, "
          f"W1={ml_metrics['mean_w1']:.1f}Å")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Noise sweep benchmark: SLSQP vs NN across noise levels")
    parser.add_argument("--h5", type=str, required=True,
                        help="Path to benchmark H5 (with Dstk, Xstk, Drlt)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to NN model (.npz)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output NPZ cache path")
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0, 0.01, 0.03, 0.05, 0.10],
                        help="Relative noise levels (default: 0 0.01 0.03 0.05 0.10)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of workers (0=auto)")
    parser.add_argument("--skip-slsqp", action="store_true",
                        help="Skip SLSQP, only run NN")
    parser.add_argument("--load-cache", type=str, default=None,
                        help="Load cached SLSQP results from NPZ")

    args = parser.parse_args()

    run_noise_sweep(
        h5_path=args.h5,
        nn_model_path=args.model,
        output_path=args.output,
        noise_levels=args.noise_levels,
        n_workers=args.workers,
        skip_slsqp=args.skip_slsqp,
        load_cache=args.load_cache,
    )


if __name__ == "__main__":
    main()
