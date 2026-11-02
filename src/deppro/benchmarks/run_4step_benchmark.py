#!/usr/bin/env python
"""
4-Step Inverse Problem Benchmark
=================================

Compares four inversion methods on the same Checkered Dstk ground truth:

  Step 1: Python SemiFreeSolver (SLSQP L1)
  Step 2: MATLAB fmincon (Drlt from H5)
  Step 3: NN with MSE loss
  Step 4: NN with MSE + W1 CDF loss

All methods:  Xstk → D_reconstructed → compare with Dstk (ground truth)

Metrics:
  - Per-element RMSE
  - Per-element W1 (Wasserstein-1 distance, Å)
  - Total RMSE / W1
  - Throughput (pixels/sec)

Usage:
    python -m deppro.benchmarks.run_4step_benchmark
    python -m deppro.benchmarks.run_4step_benchmark --h5 /path/to/data.h5
    python -m deppro.benchmarks.run_4step_benchmark --skip-solver  # NN only
    python -m deppro.benchmarks.run_4step_benchmark --nn-epochs 100

Date: 2026-02-20
"""

import time
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from deppro.benchmarks.benchmark_dstk import (
    load_dstk_data,
    create_tm_from_h5,
    run_benchmark,
    DstkBenchmarkData,
)
from deppro.depth_profile import wasserstein_distance_1d


from deppro.benchmarks.paths import GT_50x50

# Default H5: Python-generated 50x50 Checkered
DEFAULT_H5 = GT_50x50

from deppro.paths import outputs_dir as _outputs_dir  # noqa: E402
OUTPUT_DIR = _outputs_dir()


# ==============================================================================
# Metric computation (shared)
# ==============================================================================

def compute_metrics(
    D_pred: np.ndarray,
    D_true: np.ndarray,
    depth: np.ndarray,
    element_names: list[str],
) -> dict:
    """Compute RMSE and W1 metrics for predicted vs true depth profiles.

    Args:
        D_pred: (n_pixels, n_depth, n_elements)
        D_true: (n_pixels, n_depth, n_elements)
        depth:  (n_depth,)
        element_names: list of element names

    Returns:
        dict with per-element and total metrics
    """
    n_px, n_depth, n_elem = D_true.shape

    rmse_elem = np.zeros(n_elem)
    w1_elem = np.zeros(n_elem)

    for e in range(n_elem):
        mse_vals = []
        w1_vals = []
        for px in range(n_px):
            diff_e = D_pred[px, :, e] - D_true[px, :, e]
            mse_vals.append(np.mean(diff_e ** 2))
            gt_e = D_true[px, :, e]
            if gt_e.max() > 1e-4:
                w1 = wasserstein_distance_1d(depth, gt_e, D_pred[px, :, e])
                if not np.isnan(w1):
                    w1_vals.append(w1)
        rmse_elem[e] = np.sqrt(np.mean(mse_vals))
        w1_elem[e] = np.mean(w1_vals) if w1_vals else 0.0

    rmse_px = np.array([
        np.sqrt(np.mean((D_pred[px] - D_true[px]) ** 2))
        for px in range(n_px)
    ])
    total_rmse = np.sqrt(np.mean(rmse_px ** 2))
    total_w1 = np.mean(w1_elem[w1_elem > 0]) if np.any(w1_elem > 0) else 0.0

    return {
        "rmse_per_element": rmse_elem,
        "w1_per_element": w1_elem,
        "total_rmse": total_rmse,
        "total_w1": total_w1,
        "element_names": element_names,
    }


# ==============================================================================
# Step 1: Python SemiFreeSolver
# ==============================================================================

def step1_python_solver(
    data: DstkBenchmarkData,
    pixel_indices: Optional[np.ndarray] = None,
    method: str = "L1",
    verbose: bool = True,
) -> dict:
    """Run Python SemiFreeSolver on benchmark data."""
    print("=" * 70)
    print("STEP 1: Python SemiFreeSolver (SLSQP L1)")
    print("=" * 70)

    if pixel_indices is None:
        pixel_indices = np.arange(data.n_pixels)

    t0 = time.time()
    result = run_benchmark(
        data,
        method=method,
        pixel_indices=pixel_indices,
        thickness_init=210.0,
        abruptness_init=20.0,
        verbose=verbose,
        store_profiles=True,
    )
    elapsed = time.time() - t0

    metrics = compute_metrics(
        result.D_opt_all, data.Dstk[pixel_indices], data.depth, data.element_names
    )
    metrics["elapsed"] = elapsed
    metrics["throughput"] = len(pixel_indices) / elapsed
    metrics["method"] = f"Python SLSQP {method}"
    metrics["D_pred"] = result.D_opt_all

    print(f"\n  Time: {elapsed:.1f}s ({metrics['throughput']:.1f} px/s)")
    print(f"  Total RMSE: {metrics['total_rmse']:.4f}")
    print(f"  Total W1:   {metrics['total_w1']:.1f} Å")
    print()

    return metrics


# ==============================================================================
# Step 2: MATLAB fmincon (from Drlt in H5)
# ==============================================================================

def step2_matlab_drlt(
    data: DstkBenchmarkData,
    pixel_indices: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate MATLAB fmincon results (Drlt stored in H5)."""
    print("=" * 70)
    print("STEP 2: MATLAB fmincon (Drlt from H5)")
    print("=" * 70)

    if pixel_indices is None:
        pixel_indices = np.arange(data.n_pixels)

    Drlt = data.Drlt[pixel_indices]
    Dstk = data.Dstk[pixel_indices]

    # Check if Drlt is just a copy of Dstk (Python-generated data has Drlt=Dstk)
    if np.allclose(Drlt, Dstk, atol=1e-6):
        print("  ⚠ Drlt ≡ Dstk (no MATLAB solver results available)")
        print("  → Skipping MATLAB evaluation (placeholder Drlt)")
        print()
        return {
            "rmse_per_element": np.zeros(data.n_elements),
            "w1_per_element": np.zeros(data.n_elements),
            "total_rmse": 0.0,
            "total_w1": 0.0,
            "element_names": data.element_names,
            "elapsed": 0.0,
            "throughput": 0.0,
            "method": "MATLAB fmincon (N/A)",
            "D_pred": Drlt,
            "skipped": True,
        }

    metrics = compute_metrics(Drlt, Dstk, data.depth, data.element_names)
    metrics["elapsed"] = 0.0  # MATLAB time not available
    metrics["throughput"] = 0.0
    metrics["method"] = "MATLAB fmincon"
    metrics["D_pred"] = Drlt

    print(f"  Total RMSE: {metrics['total_rmse']:.4f}")
    print(f"  Total W1:   {metrics['total_w1']:.1f} Å")
    print()

    return metrics


# ==============================================================================
# Step 3 & 4: NN (MSE / W1 CDF)
# ==============================================================================

def step3_nn_mse(
    data: DstkBenchmarkData,
    h5_path: str,
    pixel_indices: Optional[np.ndarray] = None,
    n_aug: int = 200_000,
    n_epochs: int = 50,
    hidden_dim: int = 256,
    n_hidden: int = 3,
    batch_size: int = 4096,
    lr: float = 1e-3,
    patience: int = 10,
) -> dict:
    """Train NN with MSE loss and evaluate."""
    print("=" * 70)
    print("STEP 3: NN with MSE loss")
    print("=" * 70)

    return _train_and_eval_nn(
        data, h5_path, pixel_indices,
        use_w1_loss=False, lambda_w1=0.0,
        n_aug=n_aug, n_epochs=n_epochs,
        hidden_dim=hidden_dim, n_hidden=n_hidden,
        batch_size=batch_size, lr=lr, patience=patience,
        label="NN (MSE)",
        model_suffix="mse",
    )


def step4_nn_w1(
    data: DstkBenchmarkData,
    h5_path: str,
    pixel_indices: Optional[np.ndarray] = None,
    n_aug: int = 200_000,
    n_epochs: int = 50,
    hidden_dim: int = 256,
    n_hidden: int = 3,
    batch_size: int = 4096,
    lr: float = 1e-3,
    patience: int = 10,
    lambda_w1: float = 1.0,
) -> dict:
    """Train NN with MSE + W1 CDF loss and evaluate."""
    print("=" * 70)
    print("STEP 4: NN with MSE + W1 CDF loss")
    print("=" * 70)

    return _train_and_eval_nn(
        data, h5_path, pixel_indices,
        use_w1_loss=True, lambda_w1=lambda_w1,
        n_aug=n_aug, n_epochs=n_epochs,
        hidden_dim=hidden_dim, n_hidden=n_hidden,
        batch_size=batch_size, lr=lr, patience=patience,
        label="NN (MSE+W1)",
        model_suffix="w1",
    )


def _train_and_eval_nn(
    data: DstkBenchmarkData,
    h5_path: str,
    pixel_indices: Optional[np.ndarray],
    use_w1_loss: bool,
    lambda_w1: float,
    n_aug: int,
    n_epochs: int,
    hidden_dim: int,
    n_hidden: int,
    batch_size: int,
    lr: float,
    patience: int,
    label: str,
    model_suffix: str,
) -> dict:
    """Train NN and evaluate on benchmark data."""
    import mlx.core as mx
    from deppro.inference.train import train_from_h5, load_model
    from deppro.inference.postprocess import enforce_constraints

    if pixel_indices is None:
        pixel_indices = np.arange(data.n_pixels)

    # Model output path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = str(OUTPUT_DIR / f"depth_mlp_7elem_{model_suffix}.npz")

    n_elem = data.n_elements
    n_depth = data.n_depth
    n_angles = data.n_angles
    input_dim = n_angles * n_elem
    output_dim = n_depth * n_elem

    # Train
    t_train_start = time.time()
    history = train_from_h5(
        h5_path=h5_path,
        output_path=model_path,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        n_aug=n_aug,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        use_w1_loss=use_w1_loss,
        lambda_w1=lambda_w1,
        verbose=True,
    )
    t_train = time.time() - t_train_start

    # Load best model
    model = load_model(model_path, input_dim, output_dim, hidden_dim, n_hidden)

    # Inference
    Xstk_sel = data.Xstk[pixel_indices]
    I_flat = Xstk_sel.reshape(len(pixel_indices), -1).astype(np.float32)

    t_infer_start = time.perf_counter()
    D_pred_flat = []
    for start in range(0, len(pixel_indices), 10000):
        end = min(start + 10000, len(pixel_indices))
        I_batch = mx.array(I_flat[start:end])
        D_batch = model(I_batch)
        mx.eval(D_batch)
        D_pred_flat.append(np.array(D_batch))
    D_pred_flat = np.concatenate(D_pred_flat, axis=0)
    t_infer = time.perf_counter() - t_infer_start
    throughput = len(pixel_indices) / t_infer

    # Post-process
    D_pred_flat = enforce_constraints(D_pred_flat, n_depth, n_elem)
    D_pred = D_pred_flat.reshape(len(pixel_indices), n_depth, n_elem)

    # Metrics
    metrics = compute_metrics(
        D_pred, data.Dstk[pixel_indices], data.depth, data.element_names
    )
    metrics["elapsed"] = t_train
    metrics["inference_time"] = t_infer
    metrics["throughput"] = throughput
    metrics["method"] = label
    metrics["D_pred"] = D_pred
    metrics["history"] = history
    metrics["model_path"] = model_path

    print(f"\n  Training: {t_train:.1f}s")
    print(f"  Inference: {t_infer:.3f}s ({throughput:,.0f} px/s)")
    print(f"  Total RMSE: {metrics['total_rmse']:.4f}")
    print(f"  Total W1:   {metrics['total_w1']:.1f} Å")
    print(f"  Best epoch: {history.get('best_epoch', '?')}")
    print(f"  Best val loss: {history.get('best_val_loss', '?'):.6f}")
    print()

    return metrics


# ==============================================================================
# Summary table
# ==============================================================================

def print_summary(results: dict, element_names: list[str]):
    """Print comparison table of all methods."""
    print()
    print("=" * 90)
    print("FINAL COMPARISON")
    print("=" * 90)

    methods = list(results.keys())
    n_elem = len(element_names)

    # Header
    hdr = f"{'Element':<12}"
    for m in methods:
        hdr += f" {m[:18]:>18}"
    print(hdr)
    print("-" * (12 + 19 * len(methods)))

    # RMSE per element
    print("RMSE:")
    for e in range(n_elem):
        row = f"  {element_names[e]:<10}"
        for m in methods:
            r = results[m]
            if r.get("skipped"):
                row += f" {'N/A':>18}"
            else:
                row += f" {r['rmse_per_element'][e]:18.4f}"
        print(row)

    row = f"  {'TOTAL':<10}"
    for m in methods:
        r = results[m]
        if r.get("skipped"):
            row += f" {'N/A':>18}"
        else:
            row += f" {r['total_rmse']:18.4f}"
    print(row)

    print()

    # W1 per element
    print("W1 (Å):")
    for e in range(n_elem):
        row = f"  {element_names[e]:<10}"
        for m in methods:
            r = results[m]
            if r.get("skipped"):
                row += f" {'N/A':>18}"
            else:
                row += f" {r['w1_per_element'][e]:18.1f}"
        print(row)

    row = f"  {'TOTAL':<10}"
    for m in methods:
        r = results[m]
        if r.get("skipped"):
            row += f" {'N/A':>18}"
        else:
            row += f" {r['total_w1']:18.1f}"
    print(row)

    print()

    # Throughput
    print("Throughput (px/s):")
    row = "  "
    for m in methods:
        r = results[m]
        if r.get("skipped"):
            row += f" {'N/A':>18}"
        elif r['throughput'] > 0:
            row += f" {r['throughput']:>18,.0f}"
        else:
            row += f" {'(precomputed)':>18}"
    print(row)

    print("=" * 90)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="4-Step Inverse Problem Benchmark"
    )
    parser.add_argument("--h5", default=DEFAULT_H5, help="H5 data path")
    parser.add_argument("--n-pixels", type=int, default=None,
                        help="Number of pixels to test (None=all)")
    parser.add_argument("--skip-solver", action="store_true",
                        help="Skip Step 1 (Python solver)")
    parser.add_argument("--skip-nn", action="store_true",
                        help="Skip Steps 3-4 (NN)")
    parser.add_argument("--nn-epochs", type=int, default=50)
    parser.add_argument("--nn-aug", type=int, default=200_000)
    parser.add_argument("--nn-hidden", type=int, default=256)
    parser.add_argument("--nn-layers", type=int, default=3)
    parser.add_argument("--lambda-w1", type=float, default=1.0)
    parser.add_argument("--solver-method", default="L1")
    args = parser.parse_args()

    # Load data
    print(f"Loading: {args.h5}")
    data = load_dstk_data(args.h5)
    print(f"  {data.n_pixels} pixels, {data.n_elements} elements, "
          f"{data.n_depth} depths, {data.n_angles} angles")
    print(f"  Elements: {data.element_names}")
    print()

    pixel_indices = None
    if args.n_pixels:
        pixel_indices = np.arange(min(args.n_pixels, data.n_pixels))

    results = {}

    # Step 1: Python solver
    if not args.skip_solver:
        r1 = step1_python_solver(data, pixel_indices, method=args.solver_method)
        results["PySolver"] = r1

    # Step 2: MATLAB Drlt
    r2 = step2_matlab_drlt(data, pixel_indices)
    results["MATLAB"] = r2

    # Step 3: NN MSE
    if not args.skip_nn:
        r3 = step3_nn_mse(
            data, args.h5, pixel_indices,
            n_aug=args.nn_aug, n_epochs=args.nn_epochs,
            hidden_dim=args.nn_hidden, n_hidden=args.nn_layers,
        )
        results["NN(MSE)"] = r3

    # Step 4: NN W1
    if not args.skip_nn:
        r4 = step4_nn_w1(
            data, args.h5, pixel_indices,
            n_aug=args.nn_aug, n_epochs=args.nn_epochs,
            hidden_dim=args.nn_hidden, n_hidden=args.nn_layers,
            lambda_w1=args.lambda_w1,
        )
        results["NN(W1)"] = r4

    # Summary
    print_summary(results, data.element_names)

    return results


if __name__ == "__main__":
    main()
