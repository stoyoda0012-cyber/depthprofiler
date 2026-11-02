#!/usr/bin/env python
"""
Run NN inference and save to benchmark_results.h5.

Usage:
    python -m deppro.benchmarks.run_nn_benchmark
    python -m deppro.benchmarks.run_nn_benchmark --gt clean
    python -m deppro.benchmarks.run_nn_benchmark --gt noisy
"""

import argparse
import time
from pathlib import Path

import numpy as np

from deppro.benchmarks.benchmark_dstk import load_dstk_data
from deppro.benchmarks.benchmark_results_io import save_result, list_methods
from deppro.benchmarks.paths import GT_6ELEM, GT_6ELEM_NOISY


GT_MAP = {
    "noisy": GT_6ELEM_NOISY,
    "clean": GT_6ELEM,
}

BASE_DIR = Path(__file__).resolve().parent.parent.parent
from deppro.paths import outputs_dir as _outputs_dir  # noqa: E402
DEFAULT_MODELS = {
    "nn_mse": str(_outputs_dir() / "depth_mlp_v3.npz"),
    "nn_w1":  str(_outputs_dir() / "depth_mlp_w1_0.03.npz"),
}


def run_nn_inference(data, model_path):
    """Run NN inference, return (n_px, n_depth, n_elem)."""
    import mlx.core as mx
    from deppro.inference import enforce_constraints_mlx, load_model

    model = load_model(model_path)
    n_px = data.Xstk.shape[0]
    n_depth = data.Dstk.shape[1]
    n_elem = data.Dstk.shape[2]
    I_obs = data.Xstk.reshape(n_px, -1).astype(np.float32)

    # End-to-end timed region: inference + on-device sum=1 enforcement,
    # then a single GPU->host transfer (enforce_constraints_mlx keeps the
    # normalization on the GPU instead of a separate numpy/CPU post-pass).
    Imx = mx.array(I_obs)
    t0 = time.perf_counter()
    D_mx = enforce_constraints_mlx(model(Imx), n_depth, n_elem)
    D_flat = np.array(D_mx)
    t_infer = time.perf_counter() - t0

    D_out = D_flat.reshape(n_px, n_depth, n_elem)

    return D_out, t_infer


def main():
    parser = argparse.ArgumentParser(description="Run NN inference → benchmark_results.h5")
    parser.add_argument("--gt", choices=["noisy", "clean"], default="clean")
    parser.add_argument("--model-mse", default=DEFAULT_MODELS["nn_mse"])
    parser.add_argument("--model-w1", default=DEFAULT_MODELS["nn_w1"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    gt_path = GT_MAP[args.gt]
    print(f"GT: {gt_path}")

    existing = list_methods(gt_path)
    print(f"Existing methods: {existing}")

    data = load_dstk_data(gt_path)
    print(f"Loaded: {data.n_pixels} px, {data.n_elements} elem, "
          f"{data.n_depth} depth, {data.n_angles} angles")

    models = {
        "nn_mse": args.model_mse,
        "nn_w1":  args.model_w1,
    }

    for method_name, model_path in models.items():
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}, skipping {method_name}")
            continue

        if method_name in existing and not args.overwrite:
            print(f"'{method_name}' already exists. Use --overwrite to replace.")
            continue

        print(f"\nRunning {method_name} inference from {Path(model_path).name}...")
        D_pred, t_infer = run_nn_inference(data, model_path)
        throughput = data.n_pixels / t_infer
        print(f"  Inference: {t_infer:.3f}s ({throughput:,.0f} px/s)")

        save_result(
            gt_path,
            method=method_name,
            D=D_pred,
            time_sec=t_infer,
            config={
                "model_path": str(Path(model_path).name),
                "type": "nn_inference",
            },
            overwrite=args.overwrite,
        )

    print(f"\nAll methods now: {list_methods(gt_path)}")


if __name__ == "__main__":
    main()
