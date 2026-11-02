#!/usr/bin/env python
"""
Load MATLAB CLI Benchmark Results
==================================

Reads H5 files produced by deppro.benchmark.runCheckeredBenchmark (MATLAB)
and converts to Python-compatible format for comparison with Python benchmark results.

Usage:
    from deppro.benchmarks.load_matlab_benchmark import load_matlab_benchmark
    data = load_matlab_benchmark('checkered_matlab_l1x3_100px.h5')
    print(data['total_rmse'], data['mean_w1'])

    # Quick CLI verification
    python -m deppro.benchmarks.load_matlab_benchmark --h5 path/to/file.h5
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any


def load_matlab_benchmark(h5_path: str) -> Dict[str, Any]:
    """Load MATLAB benchmark results from H5 file.

    Args:
        h5_path: Path to MATLAB benchmark H5 file.

    Returns:
        Dictionary with keys:
            Dopt_all: (n_pixels, n_depth, n_elements) reconstructed profiles
            Dstk_all: (n_pixels, n_depth, n_elements) ground truth
            rmse_px: (n_pixels,) per-pixel RMSE
            w1_px: (n_pixels, n_elements) per-pixel W1
            depth: (n_depth,) depth vector
            total_rmse: float
            mean_w1: float
            mean_w1_per_element: (n_elements,)
            pixel_indices: (n_pixels,)
            method: str
            loops: int
            time_sec: float
            element_names: list[str]
            source: 'MATLAB CLI'
    """
    data = {}
    with h5py.File(h5_path, "r") as f:
        bm = f["benchmark"]

        # Arrays (MATLAB stores in Fortran order, transpose if needed)
        data["Dopt_all"] = np.array(bm["Dopt_all"], dtype=np.float64)
        data["Dstk_all"] = np.array(bm["Dstk_all"], dtype=np.float64)
        data["rmse_px"] = np.array(bm["rmse_px"], dtype=np.float64).ravel()
        data["w1_px"] = np.array(bm["w1_px"], dtype=np.float64)
        data["depth"] = np.array(bm["depth"], dtype=np.float64).ravel()
        data["mean_w1_per_element"] = np.array(
            bm["mean_w1_per_element"], dtype=np.float64
        ).ravel()
        data["pixel_indices"] = np.array(bm["pixel_indices"], dtype=np.int32).ravel()

        # Scalars (h5create with size=1 creates 1-element arrays)
        data["total_rmse"] = float(np.array(bm["total_rmse"]).flat[0])
        data["mean_w1"] = float(np.array(bm["mean_w1"]).flat[0])
        data["time_sec"] = float(np.array(bm["time_sec"]).flat[0])
        data["loops"] = int(np.array(bm["loops"]).flat[0])

        # Attributes (may be bytes or str depending on HDF5 version)
        def _attr_str(val):
            if isinstance(val, bytes):
                return val.decode("utf-8")
            if isinstance(val, np.ndarray):
                return val.item().decode("utf-8") if val.dtype.kind == "S" else str(val.item())
            return str(val) if val is not None else ""

        data["method"] = _attr_str(bm.attrs.get("method", "unknown"))
        data["source"] = _attr_str(bm.attrs.get("source", "MATLAB CLI"))
        data["date"] = _attr_str(bm.attrs.get("date", ""))
        names_str = _attr_str(bm.attrs.get("element_names", ""))
        data["element_names"] = names_str.split(",") if names_str else []

    return data


def save_as_npz(data: Dict[str, Any], npz_path: str) -> None:
    """Convert MATLAB benchmark data to NPZ format."""
    np.savez_compressed(
        npz_path,
        Dopt_all=data["Dopt_all"],
        Dstk_all=data["Dstk_all"],
        rmse_px=data["rmse_px"],
        w1_px=data["w1_px"],
        depth=data["depth"],
        mean_w1_per_element=data["mean_w1_per_element"],
        pixel_indices=data["pixel_indices"],
        total_rmse=data["total_rmse"],
        mean_w1=data["mean_w1"],
        time_sec=data["time_sec"],
        loops=data["loops"],
        method=data["method"],
        element_names=data["element_names"],
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load MATLAB benchmark H5")
    parser.add_argument("--h5", required=True, help="Path to MATLAB benchmark H5")
    parser.add_argument("--npz", help="Optional: save as NPZ")
    args = parser.parse_args()

    data = load_matlab_benchmark(args.h5)

    print(f"Method: {data['method']} x {data['loops']} loops")
    print(f"Source: {data['source']} ({data['date']})")
    print(f"Pixels: {len(data['rmse_px'])}")
    print(f"Elements: {data['element_names']}")
    print(f"Depth: {len(data['depth'])} points, range [{data['depth'][0]:.0f}, {data['depth'][-1]:.0f}] Å")
    print()
    print(f"Total RMSE: {data['total_rmse']:.4f}")
    print(f"Mean W1: {data['mean_w1']:.2f} Å")
    print(f"Time: {data['time_sec']:.1f} sec")
    print()
    print("Per-element W1 (Å):")
    for name, w1 in zip(data["element_names"], data["mean_w1_per_element"]):
        print(f"  {name}: {w1:.2f}")

    if args.npz:
        save_as_npz(data, args.npz)
        print(f"\nSaved NPZ: {args.npz}")


if __name__ == "__main__":
    main()
