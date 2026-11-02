#!/usr/bin/env python
"""
Forward Model Validation: Python TM.forward(Dstk) vs MATLAB Xstk
=================================================================

Validates that the Python forward model reproduces MATLAB's Xstk from Dstk.

Two levels of validation:
  Level 1: Inject MATLAB TransMatObs → forward(Dstk) → compare with MATLAB Xstk
           (isolates forward multiplication from IMFP/λ differences)
  Level 2: Recompute TM from scratch via compute(Dstk, elements) → forward(Dstk)
           (full end-to-end validation including IMFP and λ_obs)

Usage:
    python -m deppro.benchmarks.validate_forward
    python -m deppro.benchmarks.validate_forward --pixels 0,1,100,2499
    python -m deppro.benchmarks.validate_forward --all

Date: 2026-02-20
"""

import numpy as np
import argparse
import sys
from pathlib import Path

from deppro import TransferMatrix, CompoundDatabase
from deppro.benchmarks.benchmark_dstk import (
    load_dstk_data,
    create_tm_from_h5,
    DEFAULT_H5_PATH,
)


def validate_forward(
    h5_path: str = DEFAULT_H5_PATH,
    pixel_indices: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """Run forward model validation.

    Args:
        h5_path: Path to ground truth H5 file
        pixel_indices: Pixel indices to validate (None = [0,1,100,1000,2499])
        verbose: Print detailed output

    Returns:
        Dictionary with validation metrics
    """
    # ── Load data ──────────────────────────────────────────────────────
    if verbose:
        print(f"Loading data from: {h5_path}")
    data = load_dstk_data(h5_path)

    if verbose:
        print(f"  Dstk:  {data.Dstk.shape}  (n_pixels, n_depth, n_elements)")
        print(f"  Xstk:  {data.Xstk.shape}  (n_pixels, n_angles, n_elements)")
        print(f"  TM:    {data.trans_mat.shape}  (n_angles, n_depth, n_elements)")
        print(f"  depth: {data.depth.shape}  step_size={data.step_size:.1f} Å")
        print(f"  angles: {data.angles}")
        print(f"  elements: {data.element_names}")
        print(f"  photon_energy={data.photon_energy:.1f} eV, "
              f"work_function={data.work_function:.1f} eV, "
              f"gamma_eff={data.gamma_eff:.2f}")
        print()

    if pixel_indices is None:
        # Default: representative pixels
        max_px = data.n_pixels - 1
        pixel_indices = sorted(set([0, 1, min(100, max_px), min(1000, max_px), max_px]))

    # ── Level 1: Inject MATLAB TM → forward ───────────────────────────
    if verbose:
        print("=" * 70)
        print("LEVEL 1: Inject MATLAB TransMatObs → forward(Dstk) vs MATLAB Xstk")
        print("=" * 70)

    tm_injected = create_tm_from_h5(data)

    l1_max_errors = []
    l1_rmse_list = []

    for px in pixel_indices:
        D_px = data.Dstk[px]       # (n_depth, n_elements)
        X_matlab = data.Xstk[px]   # (n_angles, n_elements)

        X_python = tm_injected.forward(D_px, normalize=True)

        diff = np.abs(X_python - X_matlab)
        max_err = diff.max()
        rmse = np.sqrt(np.mean(diff**2))

        l1_max_errors.append(max_err)
        l1_rmse_list.append(rmse)

        if verbose:
            print(f"  pixel {px:5d}: max_err={max_err:.2e}, rmse={rmse:.2e}")

    l1_max_errors = np.array(l1_max_errors)
    l1_rmse_list = np.array(l1_rmse_list)

    if verbose:
        print(f"  ─── Summary ───")
        print(f"  max_err across all pixels: {l1_max_errors.max():.2e}")
        print(f"  mean RMSE:                 {l1_rmse_list.mean():.2e}")
        status_l1 = "✓ PASS" if l1_max_errors.max() < 1e-10 else "✗ FAIL"
        print(f"  Status: {status_l1}  (threshold: 1e-10)")
        print()

    # ── Level 2: Recompute TM from scratch → forward ──────────────────
    if verbose:
        print("=" * 70)
        print("LEVEL 2: Recompute TM via compute(Dstk) → forward(Dstk) vs MATLAB Xstk")
        print("=" * 70)

    compound_db = CompoundDatabase()

    l2_max_errors = []
    l2_rmse_list = []
    tm_max_diffs = []

    for px in pixel_indices:
        D_px = data.Dstk[px]       # (n_depth, n_elements)
        X_matlab = data.Xstk[px]   # (n_angles, n_elements)

        # Create fresh TM and compute from scratch
        tm_fresh = TransferMatrix(
            depth=data.depth,
            step_size=data.step_size,
            angles=data.angles,
            photon_energy=data.photon_energy,
            work_function=data.work_function,
            gamma_eff=data.gamma_eff,
        )
        tm_fresh.compute(D_px, data.elements, compound_db)

        # Compare TM itself (only for first pixel — TM depends on D(z))
        tm_diff = np.abs(tm_fresh.trans_mat - data.trans_mat).max()
        tm_max_diffs.append(tm_diff)

        X_python = tm_fresh.forward(D_px, normalize=True)

        diff = np.abs(X_python - X_matlab)
        max_err = diff.max()
        rmse = np.sqrt(np.mean(diff**2))

        l2_max_errors.append(max_err)
        l2_rmse_list.append(rmse)

        if verbose:
            print(f"  pixel {px:5d}: max_err={max_err:.2e}, rmse={rmse:.2e}, "
                  f"TM_diff={tm_diff:.2e}")

    l2_max_errors = np.array(l2_max_errors)
    l2_rmse_list = np.array(l2_rmse_list)
    tm_max_diffs = np.array(tm_max_diffs)

    if verbose:
        print(f"  ─── Summary ───")
        print(f"  max_err across all pixels: {l2_max_errors.max():.2e}")
        print(f"  mean RMSE:                 {l2_rmse_list.mean():.2e}")
        print(f"  TM max diff (all pixels):  {tm_max_diffs.max():.2e}")
        # For Level 2, TM can differ because H5 stores single TM
        # computed from a reference Dstk, not per-pixel.
        print()

    # ── Level 2b: Use pixel 0 Dstk for TM (like MATLAB does) ─────────
    # MATLAB stores ONE TransMatObs in ezdeprof, computed from the
    # "template" Dstk. Let's see if Python TM recomputed from that
    # same profile matches the stored TM.
    if verbose:
        print("=" * 70)
        print("LEVEL 2b: Recompute TM from ezdeprof Dstk (pixel 0) vs stored TransMatObs")
        print("=" * 70)

    # The H5 TransMatObs is typically computed from ezdeprof/Dstk (the template)
    # Let's use pixel 0 as proxy (or we can load ezdeprof/Dstk directly)
    import h5py
    with h5py.File(h5_path, "r") as f:
        if "ezdeprof/Dstk" in f:
            Dstk_template_raw = f["ezdeprof/Dstk"][:].astype(np.float64)
            # MATLAB layout: (n_elements, n_depth) → Python: (n_depth, n_elements)
            Dstk_template = Dstk_template_raw.T
            source = "ezdeprof/Dstk"
        else:
            Dstk_template = data.Dstk[0]
            source = "Dstk[0] (fallback)"

    if verbose:
        print(f"  Template source: {source}")
        print(f"  Template shape: {Dstk_template.shape}")

    tm_template = TransferMatrix(
        depth=data.depth,
        step_size=data.step_size,
        angles=data.angles,
        photon_energy=data.photon_energy,
        work_function=data.work_function,
        gamma_eff=data.gamma_eff,
    )
    tm_template.compute(Dstk_template, data.elements, compound_db)

    tm_template_diff = np.abs(tm_template.trans_mat - data.trans_mat).max()
    tm_template_rmse = np.sqrt(np.mean((tm_template.trans_mat - data.trans_mat)**2))

    if verbose:
        print(f"  TM max diff:  {tm_template_diff:.2e}")
        print(f"  TM RMSE:      {tm_template_rmse:.2e}")
        status_2b = "✓ PASS" if tm_template_diff < 1e-6 else "~ CLOSE" if tm_template_diff < 0.01 else "✗ FAIL"
        print(f"  Status: {status_2b}  (threshold: 1e-6 / 0.01)")

        # Per-element breakdown
        print(f"\n  Per-element TM comparison:")
        for e, name in enumerate(data.element_names):
            tm_e_diff = np.abs(
                tm_template.trans_mat[:, :, e] - data.trans_mat[:, :, e]
            ).max()
            print(f"    {name:6s}: max_diff={tm_e_diff:.2e}")

        # Lambda comparison
        print(f"\n  Lambda_basic (Python computed):")
        for i, name_i in enumerate(data.element_names):
            row = [f"{tm_template.lambda_basic[i, j]:.2f}"
                   for j in range(len(data.element_names))]
            print(f"    {name_i:6s}: [{', '.join(row)}] Å")

        print(f"\n  Lambda_obs at surface (d=0):")
        for e, name in enumerate(data.element_names):
            print(f"    {name:6s}: {tm_template.lambda_obs[0, e]:.2f} Å")

    print()

    # ── Final summary ─────────────────────────────────────────────────
    results = {
        "level1_max_error": float(l1_max_errors.max()),
        "level1_mean_rmse": float(l1_rmse_list.mean()),
        "level2_max_error": float(l2_max_errors.max()),
        "level2_mean_rmse": float(l2_rmse_list.mean()),
        "level2b_tm_max_diff": float(tm_template_diff),
        "level2b_tm_rmse": float(tm_template_rmse),
        "n_pixels_tested": len(pixel_indices),
        "pixel_indices": pixel_indices,
    }

    if verbose:
        print("=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"  Level 1 (TM injected):    max_err={results['level1_max_error']:.2e}")
        print(f"  Level 2 (TM recomputed):  max_err={results['level2_max_error']:.2e}")
        print(f"  Level 2b (TM template):   TM_diff={results['level2b_tm_max_diff']:.2e}")

        all_pass = (
            results["level1_max_error"] < 1e-10
            and results["level2b_tm_max_diff"] < 1e-6
        )
        print(f"\n  Overall: {'✓ ALL PASS' if all_pass else '→ See details above'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate Python forward model against MATLAB Xstk"
    )
    parser.add_argument(
        "--h5", default=DEFAULT_H5_PATH,
        help="Path to ground truth H5 file"
    )
    parser.add_argument(
        "--pixels", default=None,
        help="Comma-separated pixel indices (default: 0,1,100,1000,2499)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Validate all pixels"
    )
    args = parser.parse_args()

    if args.all:
        import h5py
        with h5py.File(args.h5, "r") as f:
            n_px = f["Dstk"].shape[0]
        pixel_indices = list(range(n_px))
    elif args.pixels:
        pixel_indices = [int(x) for x in args.pixels.split(",")]
    else:
        pixel_indices = None

    results = validate_forward(args.h5, pixel_indices)
    sys.exit(0 if results["level1_max_error"] < 1e-10 else 1)


if __name__ == "__main__":
    main()
