#!/usr/bin/env python
"""
Dstk Ground Truth Benchmark for SemiFreeSolver
===============================================

Evaluates SemiFreeSolver against MATLAB simulation ground truth (Dstk).
Compares thickness accuracy and element-wise RMSE with MATLAB solver (Drlt).

Ground truth data:
    See deppro.benchmarks.paths for canonical paths.

Usage:
    # Quick benchmark (10 random pixels)
    python -m deppro.benchmarks.benchmark_dstk --quick

    # Full benchmark (all 2500 pixels)
    python -m deppro.benchmarks.benchmark_dstk --method L1

    # Subset benchmark (50 representative pixels)
    python -m deppro.benchmarks.benchmark_dstk --subset

    # With plots
    python -m deppro.benchmarks.benchmark_dstk --quick --plot

Date: 2026-02-11
"""

import numpy as np
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path
import h5py

from deppro.depth_grid import GRID_TYPES
from deppro import (
    TransferMatrix,
    ElementConfig,
    CompoundDatabase,
    SemiFreeSolver,
    SemiFreeSolverConfig,
    SolverConfig,
)
from deppro.depth_profile import wasserstein_distance_1d


# ==============================================================================
# Default paths
# ==============================================================================

from deppro.benchmarks.paths import GT_6ELEM

DEFAULT_H5_PATH = GT_6ELEM

from deppro.paths import outputs_dir as _outputs_dir  # noqa: E402
DEFAULT_OUTPUT_DIR = _outputs_dir()


# Data containers, the H5 reader, the transfer-matrix injection and the
# thickness rule live in the body (deppro.io.dstk) so that the wheel, which
# excludes deppro.benchmarks, does not depend on this package. They are
# re-exported here so existing imports keep working.
from deppro.io.dstk import (  # noqa: E402
    BenchmarkResult,
    DstkBenchmarkData,
    create_tm_from_h5,
    estimate_thickness,
    load_dstk_data as _load_dstk_data,
)


def load_dstk_data(h5_path: str = DEFAULT_H5_PATH) -> DstkBenchmarkData:
    """Load ground truth data from HDF5 (``deppro.io.dstk.load_dstk_data``).

    Defaults to ``GT_6ELEM`` from ``deppro.benchmarks.paths``.
    """
    return _load_dstk_data(h5_path)


def dstk_data_from_simulation(sim: dict) -> DstkBenchmarkData:
    """Convert generate_simulation() output to DstkBenchmarkData.

    Enables an H5-free roundtrip pipeline:
        generate_simulation() -> DstkBenchmarkData -> run_benchmark() -> plot

    Args:
        sim: Dictionary returned by generate_simulation() or generate_checkered(),
             containing: Dstk, Xstk, depth, angles, element_configs, config, charges

    Returns:
        DstkBenchmarkData populated from simulation data.
    """
    config = sim["config"]
    Dstk = sim["Dstk"]
    Xstk = sim["Xstk"]
    depth = sim["depth"]
    angles = sim["angles"]
    element_configs = sim["element_configs"]
    charges = sim["charges"]

    n_pixels, n_depth, n_elements = Dstk.shape
    n_angles = angles.shape[0]

    # Build element names: "Si(0)", "Si(+4)", "Ti(+4)", etc.
    element_names = []
    for ec in element_configs:
        ox = int(ec.oxidation_state)
        if ox > 0:
            element_names.append(f"{ec.symbol}(+{ox})")
        elif ox < 0:
            element_names.append(f"{ec.symbol}({ox})")
        else:
            element_names.append(f"{ec.symbol}(0)")

    # Convert 0-based indices to MATLAB 1-based
    free_matlab = np.array(config.free_idx, dtype=int) + 1
    subst_matlab = np.array(config.subst_idx, dtype=int) + 1
    linked_matlab = np.array(config.linked_idx, dtype=int) + 1

    # Compute representative transfer matrix from pixel 0
    tm = TransferMatrix(
        depth=depth,
        step_size=config.step_size,
        angles=angles,
        photon_energy=config.photon_energy,
        work_function=config.work_function,
        gamma_eff=config.gamma_eff,
    )
    compound_db = CompoundDatabase()
    tm.compute(Dstk[0], element_configs, compound_db)
    trans_mat = tm.trans_mat  # (n_angles, n_depth, n_elements)

    # Drlt placeholder (no MATLAB results in generate mode)
    Drlt = np.zeros_like(Dstk)

    # linkedconst
    linkedconst = np.array(config.linked_const, dtype=np.float64)

    return DstkBenchmarkData(
        Dstk=Dstk, Xstk=Xstk, Drlt=Drlt,
        trans_mat=trans_mat,
        depth=depth, angles=angles,
        elements=element_configs, element_names=element_names,
        charges=charges, linkedconst=linkedconst,
        step_size=config.step_size, gamma_eff=config.gamma_eff,
        photon_energy=config.photon_energy, work_function=config.work_function,
        n_pixels=n_pixels, n_depth=n_depth,
        n_angles=n_angles, n_elements=n_elements,
        free_matlab=free_matlab, subst_matlab=subst_matlab,
        linked_matlab=linked_matlab,
    )


# ==============================================================================
# Noise injection
# ==============================================================================

def add_noise_to_xstk(
    data: DstkBenchmarkData,
    noise_level: float,
    noise_type: str = "relative",
    seed: int = 123,
) -> DstkBenchmarkData:
    """Add noise to Xstk intensities (in-place modification of data).

    Args:
        data: Benchmark data (modified in place)
        noise_level: Noise magnitude (fraction, e.g. 0.02 = 2%)
        noise_type: 'relative' = X*(1 + σ*randn), 'poisson' = X + sqrt(|X|)*σ*randn
        seed: Random seed for reproducibility

    Returns:
        Same data object with noisy Xstk
    """
    if noise_level <= 0:
        return data

    rng = np.random.default_rng(seed)
    shape = data.Xstk.shape  # (n_pixels, n_angles, n_elements)

    if noise_type == "relative":
        noise = rng.normal(0, noise_level, shape)
        data.Xstk = data.Xstk * (1 + noise)
    elif noise_type == "poisson":
        noise = rng.normal(0, noise_level, shape)
        data.Xstk = data.Xstk + np.sqrt(np.abs(data.Xstk)) * noise
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Ensure non-negative intensities
    data.Xstk = np.maximum(data.Xstk, 0.0)
    return data


def load_noisy_xstk_from_ismp(
    data: DstkBenchmarkData,
    ismp_h5_path: str,
    ref_xstk_h5_path: str,
) -> DstkBenchmarkData:
    """Replace Xstk with realistic noisy data reconstructed from Ismp.

    Ismp (peak-fitted intensities) = raw_intensity * cross_section.
    Reconstruction: Xsmp = Ismp / cs, Xstk = Xsmp / sum(Xsmp, axis=elements).
    Cross-sections are recovered from the reference Xstk (1 pixel processed by
    runFullPipeline) via fixed-point iteration.

    Args:
        data: Benchmark data (modified in place)
        ismp_h5_path: Path to H5 with /Ismp (n_pixels, n_elements, n_angles)
        ref_xstk_h5_path: Path to H5 with /Xstk (1, n_elements, n_angles)
            from runFullPipeline (1 stack processed) for cross-section recovery

    Returns:
        Same data object with noisy Xstk
    """
    with h5py.File(ismp_h5_path, "r") as f:
        # Ismp: MATLAB layout (n_pixels, n_elements, n_angles)
        Ismp = f["Ismp"][:].astype(np.float64)

    with h5py.File(ref_xstk_h5_path, "r") as f:
        # Reference Xstk from 1 processed pixel: (1, n_elements, n_angles)
        Xstk_ref = f["Xstk"][0].astype(np.float64)  # (n_elements, n_angles)

    n_elements = Ismp.shape[1]

    # Recover cross-sections via fixed-point iteration on pixel 0
    cs = Ismp[0, :, 0] / Xstk_ref[:, 0]  # initial guess from angle 0
    for _ in range(50):
        Xsmp0 = Ismp[0] / cs[:, np.newaxis]  # (n_elements, n_angles)
        Xstk_calc = Xsmp0 / Xsmp0.sum(axis=0, keepdims=True)
        err = np.abs(Xstk_calc - Xstk_ref).max()
        if err < 1e-12:
            break
        for e in range(n_elements):
            denom = Xstk_ref[e, :] * Xsmp0.sum(axis=0)
            cs[e] = np.mean(Ismp[0, e, :] / (denom + 1e-30))

    # Apply to all pixels: Ismp → Xsmp → Xstk
    Xsmp_all = Ismp / cs[:, np.newaxis]  # (n_pixels, n_elements, n_angles)
    Xstk_noisy = Xsmp_all / Xsmp_all.sum(axis=1, keepdims=True)

    # Transpose from MATLAB (pixels, elements, angles) to Python (pixels, angles, elements)
    data.Xstk = np.transpose(Xstk_noisy, (0, 2, 1))
    return data




# ==============================================================================
# Non-uniform grid remapping
# ==============================================================================

def remap_data_to_grid(
    data: DstkBenchmarkData,
    new_depth: np.ndarray,
) -> DstkBenchmarkData:
    """Remap benchmark data from original uniform grid to a new depth grid.

    Interpolates Dstk, Drlt, and recomputes TransferMatrix on the new grid.
    Xstk (angle-resolved intensities) is NOT changed since it's the input data.

    Args:
        data: Original benchmark data (modified in place)
        new_depth: New depth grid (n_depth_new,) in Angstroms

    Returns:
        Same data object remapped to new grid
    """
    from deppro.depth_grid import interpolate_profiles_to_grid

    old_depth = data.depth
    n_new = len(new_depth)

    # Interpolate ground truth and MATLAB results to new grid
    data.Dstk = interpolate_profiles_to_grid(data.Dstk, old_depth, new_depth)
    data.Drlt = interpolate_profiles_to_grid(data.Drlt, old_depth, new_depth)

    # Build new TransferMatrix on the non-uniform grid
    tm = TransferMatrix.from_nonuniform_grid(
        depth_grid=new_depth,
        angles=data.angles,
        photon_energy=data.photon_energy,
        work_function=data.work_function,
        gamma_eff=data.gamma_eff,
    )

    # Compute lambda_basic and initial trans_mat from pixel 0 profile
    compound_db = CompoundDatabase()
    tm.compute(data.Dstk[0], data.elements, compound_db)

    # Update data fields
    data.depth = new_depth
    data.step_size = float(np.mean(tm.step_sizes))
    data.trans_mat = tm.trans_mat
    data.n_depth = n_new

    return data




# ==============================================================================
# Pixel selection
# ==============================================================================

def select_pixels(data: DstkBenchmarkData, mode: str = "full", seed: int = 42) -> np.ndarray:
    """Select pixel indices for benchmark.

    Args:
        data: Benchmark data
        mode: 'full' (2500), 'subset' (50 representative), 'quick' (10 random)

    Returns:
        Array of pixel indices
    """
    if mode == "full":
        return np.arange(data.n_pixels)

    rng = np.random.default_rng(seed)

    if mode == "quick":
        return rng.choice(data.n_pixels, 10, replace=False)

    if mode == "subset":
        # Pick pixels spanning the thickness range
        thicknesses = np.array([
            estimate_thickness(data.Dstk[px], data.depth)
            for px in range(data.n_pixels)
        ])
        # Bin by 10Å intervals and pick ~5 from each bin
        bins = np.arange(200, 340, 10)
        indices = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (thicknesses >= lo) & (thicknesses < hi)
            candidates = np.where(mask)[0]
            if len(candidates) > 0:
                picked = rng.choice(candidates, min(5, len(candidates)), replace=False)
                indices.extend(picked)
        return np.array(indices)

    raise ValueError(f"Unknown mode: {mode}")


# ==============================================================================
# Core benchmark
# ==============================================================================

def run_benchmark(
    data: DstkBenchmarkData,
    method: str = "LLS",
    alpha: float = 1.0,
    pixel_indices: Optional[np.ndarray] = None,
    thickness_init: float = 210.0,
    abruptness_init: float = 20.0,
    verbose: bool = True,
    store_profiles: bool = False,
) -> BenchmarkResult:
    """Run SemiFreeSolver benchmark on selected pixels.

    Args:
        data: Benchmark data from load_dstk_data()
        method: Optimization method ('LLS', 'L1', 'L2', 'MEM')
        alpha: Regularization strength
        pixel_indices: Pixel indices to benchmark (None = all)
        thickness_init: Initial thickness guess (Å)
        verbose: Print per-pixel progress
        store_profiles: Store full D_opt for all pixels (memory intensive)

    Returns:
        BenchmarkResult with metrics
    """
    if pixel_indices is None:
        pixel_indices = np.arange(data.n_pixels)

    n_px = len(pixel_indices)
    tm = create_tm_from_h5(data)

    # SemiFreeSolver configuration — indices computed dynamically from H5
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
        scipy_method='SLSQP',
        use_sum_constraint=True,
        use_charge_constraint=True,
        max_warm_restarts=2,
        warm_restart_threshold=0.2,
        enable_fallback=False,
    )

    # Allocate result arrays
    rmse_px = np.zeros(n_px)
    thickness_opt = np.zeros(n_px)
    thickness_true = np.zeros(n_px)
    thickness_err = np.zeros(n_px)
    chi2_px = np.zeros(n_px)
    success_px = np.zeros(n_px, dtype=bool)
    warm_restarts_px = np.zeros(n_px, dtype=int)
    rmse_elem = np.zeros((n_px, data.n_elements))
    w1_elem = np.zeros((n_px, data.n_elements))

    D_opt_all = np.zeros((n_px, data.n_depth, data.n_elements)) if store_profiles else None

    start_time = time.time()

    for i, px in enumerate(pixel_indices):
        I_obs = data.Xstk[px]   # (n_angles, n_elements)
        D_true = data.Dstk[px]  # (n_depth, n_elements)

        # Create solver for this pixel
        solver = SemiFreeSolver(
            transfer_matrix=tm,
            config=solver_config,
            semi_free_config=semi_config,
            D_prior=None,
            elements=data.elements,
        )

        try:
            result = solver.solve(I_obs, verbose=False)

            thickness_opt[i] = result.thickness
            chi2_px[i] = result.chi2
            success_px[i] = result.success
            warm_restarts_px[i] = result.warm_restarts

            # RMSE vs ground truth
            diff = result.D_opt - D_true
            rmse_px[i] = np.sqrt(np.mean(diff ** 2))

            # Per-element RMSE and W1
            for e in range(data.n_elements):
                rmse_elem[i, e] = np.sqrt(np.mean(diff[:, e] ** 2))
                w1_elem[i, e] = wasserstein_distance_1d(data.depth, D_true[:, e], result.D_opt[:, e])

            if store_profiles:
                D_opt_all[i] = result.D_opt

        except Exception as exc:
            if verbose:
                print(f"  pixel {px}: FAILED ({exc})")
            rmse_px[i] = np.nan
            success_px[i] = False
            w1_elem[i, :] = np.nan

        # Ground truth thickness
        thickness_true[i] = estimate_thickness(D_true, data.depth)
        thickness_err[i] = abs(thickness_opt[i] - thickness_true[i])

        if verbose and (i + 1) % max(1, n_px // 20) == 0:
            elapsed = time.time() - start_time
            print(
                f"  [{i+1}/{n_px}] px={px:4d}  "
                f"t_true={thickness_true[i]:.0f}  t_opt={thickness_opt[i]:.0f}  "
                f"err={thickness_err[i]:.1f}Å  rmse={rmse_px[i]:.4f}  "
                f"warm={warm_restarts_px[i]}  "
                f"({elapsed:.1f}s)"
            )

    total_time = time.time() - start_time

    # Aggregate element RMSE and W1
    valid = ~np.isnan(rmse_px)
    rmse_per_element = np.sqrt(np.nanmean(rmse_elem[valid] ** 2, axis=0))
    total_rmse = np.sqrt(np.nanmean(rmse_px[valid] ** 2))
    w1_per_element = np.nanmean(w1_elem[valid], axis=0)
    total_w1 = np.nanmean(w1_per_element)

    return BenchmarkResult(
        method=method,
        n_pixels=n_px,
        rmse_per_pixel=rmse_px,
        thickness_opt=thickness_opt,
        thickness_true=thickness_true,
        thickness_error=thickness_err,
        chi2_per_pixel=chi2_px,
        success_per_pixel=success_px,
        warm_restarts_per_pixel=warm_restarts_px,
        rmse_per_element=rmse_per_element,
        w1_per_element=w1_per_element,
        element_names=data.element_names,
        total_rmse=total_rmse,
        total_w1=total_w1,
        mean_thickness_error=np.nanmean(thickness_err[valid]),
        success_rate=np.mean(success_px),
        total_time=total_time,
        D_opt_all=D_opt_all,
    )


# ==============================================================================
# L1 → MEM pipeline
# ==============================================================================

def run_l1_mem_pipeline(
    data: DstkBenchmarkData,
    pixel_indices: Optional[np.ndarray] = None,
    l1_alpha: float = 1.0,
    mem_alpha: float = 0.0003,
    thickness_init: float = 210.0,
    abruptness_init: float = 20.0,
    verbose: bool = True,
    store_profiles: bool = False,
) -> BenchmarkResult:
    """Run L1 → MEM chaining benchmark.

    Step 1: SemiFreeSolver L1 for coarse reconstruction
    Step 2: SemiFreeSolver MEM using L1 result as D_prior and thickness_init
    """
    if pixel_indices is None:
        pixel_indices = np.arange(data.n_pixels)

    n_px = len(pixel_indices)
    tm = create_tm_from_h5(data)

    # Shared semi-free config template — indices computed dynamically from H5
    semi_config_base = dict(
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

    l1_config = SolverConfig(
        method='L1', alpha=l1_alpha, max_iter=200,
        scipy_method='SLSQP',
        use_sum_constraint=True, use_charge_constraint=True,
        max_warm_restarts=2, warm_restart_threshold=0.2,
        enable_fallback=False,
    )
    mem_config = SolverConfig(
        method='MEM', alpha=mem_alpha, max_iter=200,
        scipy_method='SLSQP',
        use_sum_constraint=True, use_charge_constraint=True,
        max_warm_restarts=2, warm_restart_threshold=0.2,
        enable_fallback=False,
        mem_space=mem_space,
    )

    # Allocate
    rmse_px = np.zeros(n_px)
    thickness_opt = np.zeros(n_px)
    thickness_true = np.zeros(n_px)
    thickness_err = np.zeros(n_px)
    chi2_px = np.zeros(n_px)
    success_px = np.zeros(n_px, dtype=bool)
    warm_restarts_px = np.zeros(n_px, dtype=int)
    rmse_elem = np.zeros((n_px, data.n_elements))
    w1_elem = np.zeros((n_px, data.n_elements))
    D_opt_all = np.zeros((n_px, data.n_depth, data.n_elements)) if store_profiles else None

    start_time = time.time()

    for i, px in enumerate(pixel_indices):
        I_obs = data.Xstk[px]
        D_true = data.Dstk[px]

        try:
            # Step 1: L1 coarse
            semi_l1 = SemiFreeSolverConfig(
                thickness_init=thickness_init, **semi_config_base
            )
            solver_l1 = SemiFreeSolver(
                transfer_matrix=tm, config=l1_config,
                semi_free_config=semi_l1, D_prior=None,
                elements=data.elements,
            )
            result_l1 = solver_l1.solve(I_obs, verbose=False)

            # Step 2: MEM refinement using L1 result
            semi_mem = SemiFreeSolverConfig(
                thickness_init=result_l1.thickness, **semi_config_base
            )
            solver_mem = SemiFreeSolver(
                transfer_matrix=tm, config=mem_config,
                semi_free_config=semi_mem,
                D_prior=result_l1.D_opt,
                elements=data.elements,
            )
            # Warm-start from L1 solution
            x0 = solver_mem.build_x0(
                thickness=result_l1.thickness,
                abruptness=result_l1.abruptness,
                substrate_ratios=result_l1.substrate_ratios,
            )
            result = solver_mem.solve(I_obs, verbose=False, x0_override=x0)

            thickness_opt[i] = result.thickness
            chi2_px[i] = result.chi2
            success_px[i] = result.success
            warm_restarts_px[i] = result_l1.warm_restarts + result.warm_restarts

            diff = result.D_opt - D_true
            rmse_px[i] = np.sqrt(np.mean(diff ** 2))
            for e in range(data.n_elements):
                rmse_elem[i, e] = np.sqrt(np.mean(diff[:, e] ** 2))
                w1_elem[i, e] = wasserstein_distance_1d(data.depth, D_true[:, e], result.D_opt[:, e])

            if store_profiles:
                D_opt_all[i] = result.D_opt

        except Exception as exc:
            if verbose:
                print(f"  pixel {px}: FAILED ({exc})")
            rmse_px[i] = np.nan
            success_px[i] = False
            w1_elem[i, :] = np.nan

        thickness_true[i] = estimate_thickness(D_true, data.depth)
        thickness_err[i] = abs(thickness_opt[i] - thickness_true[i])

        if verbose and (i + 1) % max(1, n_px // 20) == 0:
            elapsed = time.time() - start_time
            print(
                f"  [{i+1}/{n_px}] px={px:4d}  "
                f"t_true={thickness_true[i]:.0f}  t_opt={thickness_opt[i]:.0f}  "
                f"err={thickness_err[i]:.1f}Å  rmse={rmse_px[i]:.4f}  "
                f"warm={warm_restarts_px[i]}  "
                f"({elapsed:.1f}s)"
            )

    total_time = time.time() - start_time
    valid = ~np.isnan(rmse_px)
    rmse_per_element = np.sqrt(np.nanmean(rmse_elem[valid] ** 2, axis=0))
    total_rmse = np.sqrt(np.nanmean(rmse_px[valid] ** 2))
    w1_per_element = np.nanmean(w1_elem[valid], axis=0)
    total_w1 = np.nanmean(w1_per_element)

    return BenchmarkResult(
        method="L1→MEM",
        n_pixels=n_px,
        rmse_per_pixel=rmse_px,
        thickness_opt=thickness_opt,
        thickness_true=thickness_true,
        thickness_error=thickness_err,
        chi2_per_pixel=chi2_px,
        success_per_pixel=success_px,
        warm_restarts_per_pixel=warm_restarts_px,
        rmse_per_element=rmse_per_element,
        w1_per_element=w1_per_element,
        element_names=data.element_names,
        total_rmse=total_rmse,
        total_w1=total_w1,
        mean_thickness_error=np.nanmean(thickness_err[valid]),
        success_rate=np.mean(success_px),
        total_time=total_time,
        D_opt_all=D_opt_all,
    )


# ==============================================================================
# Iterative multi-loop benchmark (MATLAB Run_DepthProfiling.m equivalent)
# ==============================================================================

def run_iterative_benchmark(
    data: DstkBenchmarkData,
    method: str = "LLS",
    alpha: float = 1.0,
    n_loops: int = 10,
    degree_decay: float = 0.8,
    update_prior: bool = True,
    update_transmat: bool = True,
    use_xreg: bool = False,
    pixel_indices: Optional[np.ndarray] = None,
    thickness_init: float = 210.0,
    abruptness_init: float = 20.0,
    solver_max_iter: int = 200,
    solver_tol: float = 1e-6,
    scipy_method: str = 'SLSQP',
    l1_smoothing: str = 'none',
    l1_beta: float = 0.01,
    verbose: bool = True,
    store_profiles: bool = False,
) -> BenchmarkResult:
    """Run MATLAB-equivalent iterative multi-loop benchmark.

    Implements the same iterative scheme as MATLAB Run_DepthProfiling.m:
      1. x0 warm-start: previous result → next initial guess
      2. D_prior update: D_opt → D_prior (regularization reference)
      3. TransferMatrix update: recompute IMFP from D_opt composition
      4. degree decay: degree *= 0.8 each loop (simulated annealing)

    Args:
        data: Benchmark data from load_dstk_data()
        method: Optimization method ('LLS', 'L1', 'L2', 'MEM')
        alpha: Regularization strength
        n_loops: Number of optimization loops per pixel (MATLAB default: 10)
        degree_decay: Degree reduction factor per loop (MATLAB: 0.8)
        update_prior: Update D_prior from D_opt each loop
        update_transmat: Recompute TransferMatrix from D_opt each loop
        pixel_indices: Pixel indices to benchmark (None = all)
        thickness_init: Initial thickness guess (Å)
        verbose: Print per-pixel progress
        store_profiles: Store full D_opt for all pixels

    Returns:
        BenchmarkResult with metrics
    """
    if pixel_indices is None:
        pixel_indices = np.arange(data.n_pixels)

    n_px = len(pixel_indices)

    # Pre-compute lambda_basic once (element-pair IMFP, doesn't change)
    compound_db = CompoundDatabase()
    tm_template = create_tm_from_h5(data)
    lambda_basic = tm_template.calculate_lambda_basic(data.elements, compound_db)

    # Shared semi-free config template — indices computed dynamically from H5
    semi_config_base = dict(
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

    film_idx = data.film_idx

    # Allocate result arrays
    rmse_px = np.zeros(n_px)
    thickness_opt = np.zeros(n_px)
    thickness_true = np.zeros(n_px)
    thickness_err = np.zeros(n_px)
    chi2_px = np.zeros(n_px)
    success_px = np.zeros(n_px, dtype=bool)
    warm_restarts_px = np.zeros(n_px, dtype=int)
    rmse_elem = np.zeros((n_px, data.n_elements))
    w1_elem = np.zeros((n_px, data.n_elements))
    D_opt_all = np.zeros((n_px, data.n_depth, data.n_elements)) if store_profiles else None

    start_time = time.time()

    for i, px in enumerate(pixel_indices):
        I_obs = data.Xstk[px]   # (n_angles, n_elements)
        D_true = data.Dstk[px]  # (n_depth, n_elements)

        # Per-pixel state: reset for each pixel
        tm = create_tm_from_h5(data)           # Start with H5 TransMat
        D_prior_loop = None                    # L1/LLS: sparsity prior
        thck_loop = thickness_init
        x0_override = None
        degree_loop = 1.0
        total_warm = 0

        try:
            for loop in range(n_loops):
                # --- Build solver config with current degree ---
                solver_config = SolverConfig(
                    method=method,
                    alpha=alpha,
                    degree=degree_loop,
                    max_iter=solver_max_iter,
                    tol=solver_tol,
                    scipy_method=scipy_method,
                    use_sum_constraint=True,
                    use_charge_constraint=True,
                    max_warm_restarts=2,
                    warm_restart_threshold=0.2,
                    use_xreg_weighting=use_xreg,
                    enable_fallback=False,
                    l1_smoothing=l1_smoothing,
                    l1_beta=l1_beta,
                )

                semi_config = SemiFreeSolverConfig(
                    thickness_init=thck_loop,
                    **semi_config_base,
                )

                solver = SemiFreeSolver(
                    transfer_matrix=tm,
                    config=solver_config,
                    semi_free_config=semi_config,
                    D_prior=D_prior_loop,
                    elements=data.elements,
                )

                result = solver.solve(I_obs, verbose=False, x0_override=x0_override)
                total_warm += result.warm_restarts

                # --- Updates for next loop ---
                # 1. x0 warm-start from result
                x0_override = solver.build_x0(
                    film_conc=result.D_opt[:, film_idx],
                    thickness=result.thickness,
                    abruptness=result.abruptness,
                    substrate_ratios=result.substrate_ratios,
                )

                # 2. Update thickness init
                thck_loop = result.thickness

                # 3. Update D_prior (regularization reference)
                if update_prior:
                    D_prior_loop = result.D_opt.copy()

                # 4. Update TransferMatrix from D_opt composition
                if update_transmat and loop < n_loops - 1:
                    lambda_obs = tm.calculate_lambda_obs(result.D_opt, lambda_basic)
                    tm.trans_mat = tm.calculate_trans_mat(lambda_obs)

                # 5. Degree decay (simulated annealing)
                if loop > 0:
                    degree_loop *= degree_decay

            # Store final result
            thickness_opt[i] = result.thickness
            chi2_px[i] = result.chi2
            success_px[i] = result.success
            warm_restarts_px[i] = total_warm

            # RMSE and W1 vs ground truth
            diff = result.D_opt - D_true
            rmse_px[i] = np.sqrt(np.mean(diff ** 2))
            for e in range(data.n_elements):
                rmse_elem[i, e] = np.sqrt(np.mean(diff[:, e] ** 2))
                w1_elem[i, e] = wasserstein_distance_1d(data.depth, D_true[:, e], result.D_opt[:, e])

            if store_profiles:
                D_opt_all[i] = result.D_opt

        except Exception as exc:
            if verbose:
                print(f"  pixel {px}: FAILED ({exc})")
            rmse_px[i] = np.nan
            success_px[i] = False
            w1_elem[i, :] = np.nan

        # Ground truth thickness
        thickness_true[i] = estimate_thickness(D_true, data.depth)
        thickness_err[i] = abs(thickness_opt[i] - thickness_true[i])

        if verbose and (i + 1) % max(1, n_px // 20) == 0:
            elapsed = time.time() - start_time
            print(
                f"  [{i+1}/{n_px}] px={px:4d}  "
                f"t_true={thickness_true[i]:.0f}  t_opt={thickness_opt[i]:.0f}  "
                f"err={thickness_err[i]:.1f}Å  rmse={rmse_px[i]:.4f}  "
                f"warm={warm_restarts_px[i]}  "
                f"({elapsed:.1f}s)"
            )

    total_time = time.time() - start_time

    # Aggregate element RMSE and W1
    valid = ~np.isnan(rmse_px)
    rmse_per_element = np.sqrt(np.nanmean(rmse_elem[valid] ** 2, axis=0))
    total_rmse = np.sqrt(np.nanmean(rmse_px[valid] ** 2))
    w1_per_element = np.nanmean(w1_elem[valid], axis=0)
    total_w1 = np.nanmean(w1_per_element)

    method_label = f"{method}×{n_loops}"
    if update_transmat:
        method_label += "+TM"
    if update_prior:
        method_label += "+Prior"
    if use_xreg:
        method_label += "+Xreg"

    return BenchmarkResult(
        method=method_label,
        n_pixels=n_px,
        rmse_per_pixel=rmse_px,
        thickness_opt=thickness_opt,
        thickness_true=thickness_true,
        thickness_error=thickness_err,
        chi2_per_pixel=chi2_px,
        success_per_pixel=success_px,
        warm_restarts_per_pixel=warm_restarts_px,
        rmse_per_element=rmse_per_element,
        w1_per_element=w1_per_element,
        element_names=data.element_names,
        total_rmse=total_rmse,
        total_w1=total_w1,
        mean_thickness_error=np.nanmean(thickness_err[valid]),
        success_rate=np.mean(success_px),
        total_time=total_time,
        D_opt_all=D_opt_all,
    )


# ==============================================================================
# Neighbor-average MEM pipeline (LLS → neighbor-avg → MEM)
# ==============================================================================

def _compute_grid_positions(n_pixels: int, grid_size: int = 50,
                            nx: int = 5, ny: int = 5) -> np.ndarray:
    """Compute 3D grid positions (x, y, t) for each pixel.

    Layout: pixel = t * (nx*ny) + y * nx + x
    """
    nxy = nx * ny
    return np.array([
        [px % nx, (px % nxy) // nx, px // nxy]
        for px in range(n_pixels)
    ])


# ==============================================================================
# Worker functions (imported from solver_workers for pickle compatibility)
# ==============================================================================
from deppro.solver_workers import (
    worker_initializer as _worker_initializer,
    worker_phase1 as _worker_phase1,
    worker_phase2 as _worker_phase2,
    build_first_neighbor_indices as _build_first_neighbor_indices,
    neighbor_average_prior as _neighbor_average_prior,
)


# ==============================================================================
# Main pipeline (serial or parallel)
# ==============================================================================

def run_neighbor_mem_benchmark(
    data: DstkBenchmarkData,
    pixel_indices: Optional[np.ndarray] = None,
    # Phase 1 params (LLS)
    phase1_method: str = "LLS",
    phase1_loops: int = 10,
    phase1_alpha: float = 1.0,
    use_xreg: bool = True,
    # Phase 2 params (MEM)
    mem_alpha: float = 0.0003,
    mem_loops: int = 3,
    mem_scipy_method: str = 'SLSQP',
    mem_space: str = 'concentration',
    # Shared
    thickness_init: float = 210.0,
    abruptness_init: float = 20.0,
    solver_max_iter: int = 200,
    solver_tol: float = 1e-6,
    grid_size: int = 50,
    nx: int = 5,
    ny: int = 5,
    n_ring: int = 1,
    n_workers: int = 1,
    l1_smoothing: str = 'none',
    l1_beta: float = 0.01,
    verbose: bool = True,
    store_profiles: bool = False,
) -> BenchmarkResult:
    """Run LLS → neighbor-average → MEM pipeline (serial or parallel).

    Phase 1: LLS×N+TM+Xreg for all pixels (get D_opt)
    Phase 2: Neighbor-average D_opt → MEM D_prior, then MEM×M loops

    Pixel layout: pixel = t * (nx*ny) + y * nx + x
    Neighbors use Manhattan (L1) distance: |dy|+|dx| <= n_ring.

    When n_workers > 1, both phases are parallelized with multiprocessing.

    Args:
        data: Benchmark data
        pixel_indices: Which pixels to process (None = all 2500)
        phase1_method: Method for phase 1 ('LLS' or 'L1')
        phase1_loops: Number of loops for phase 1
        mem_alpha: MEM regularization strength
        mem_loops: Number of MEM refinement loops per pixel
        grid_size: Grid dimension (50 for 50×50)
        n_workers: Number of parallel workers (1 = serial)
    """
    if pixel_indices is None:
        pixel_indices = np.arange(data.n_pixels)
    n_px = len(pixel_indices)
    use_parallel = n_workers > 1

    if use_parallel:
        import multiprocessing as mp

    # Pre-compute shared resources
    compound_db = CompoundDatabase()
    tm_template = create_tm_from_h5(data)
    lambda_basic = tm_template.calculate_lambda_basic(data.elements, compound_db)

    # Serialize TransferMatrix data for pickling
    from deppro.solver_workers import serialize_transfer_matrix
    tm_data = serialize_transfer_matrix(tm_template)

    semi_config_base_dict = dict(
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
    film_idx = data.film_idx

    # =====================================================================
    # Phase 1: LLS iterative (parallel or serial)
    # =====================================================================
    if verbose:
        mode_str = f"{n_workers} workers" if use_parallel else "serial"
        print(f"=== Phase 1: {phase1_method}×{phase1_loops}+TM+Xreg "
              f"({n_px} pixels, {mode_str}) ===")

    start_phase1 = time.time()

    # Build Phase 1 args for all pixels
    phase1_args = []
    for i, px in enumerate(pixel_indices):
        phase1_args.append((
            i, int(px), data.Xstk[px], tm_data, data.elements, data.linkedconst,
            lambda_basic, phase1_method, phase1_alpha, phase1_loops, 0.8, use_xreg,
            thickness_init, abruptness_init, solver_max_iter, solver_tol,
            semi_config_base_dict, l1_smoothing, l1_beta,
        ))

    D_opt_all = np.zeros((n_px, data.n_depth, data.n_elements))
    thickness_all = np.zeros(n_px)
    abruptness_all = np.full(n_px, abruptness_init)
    chi2_p1 = np.zeros(n_px)
    obj_p1 = np.zeros(n_px)
    success_phase1 = np.zeros(n_px, dtype=bool)
    warm_phase1 = np.zeros(n_px, dtype=int)
    total_iters = 0

    if use_parallel:
        with mp.Pool(n_workers, initializer=_worker_initializer) as pool:
            results_iter = pool.imap_unordered(_worker_phase1, phase1_args)
            done = 0
            for res in results_iter:
                idx, D_opt, thk, abrp, chi2, obj, suc, warm, nit = res
                if D_opt is not None:
                    D_opt_all[idx] = D_opt
                    thickness_all[idx] = thk
                    abruptness_all[idx] = abrp
                    chi2_p1[idx] = chi2
                    obj_p1[idx] = obj
                    success_phase1[idx] = suc
                    warm_phase1[idx] = warm
                    total_iters += nit
                done += 1
                if verbose and done % max(1, n_px // 20) == 0:
                    elapsed = time.time() - start_phase1
                    print(f"  Phase 1: [{done}/{n_px}] ({elapsed:.1f}s)")
    else:
        for args in phase1_args:
            res = _worker_phase1(args)
            idx, D_opt, thk, abrp, chi2, obj, suc, warm, nit = res
            if D_opt is not None:
                D_opt_all[idx] = D_opt
                thickness_all[idx] = thk
                abruptness_all[idx] = abrp
                chi2_p1[idx] = chi2
                obj_p1[idx] = obj
                success_phase1[idx] = suc
                warm_phase1[idx] = warm
                total_iters += nit
            if verbose and (idx + 1) % max(1, n_px // 20) == 0:
                elapsed = time.time() - start_phase1
                px = pixel_indices[idx]
                print(f"  [{idx+1}/{n_px}] px={px:4d}  "
                      f"t_opt={thk:.0f}  warm={warm}  ({elapsed:.1f}s)")

    phase1_time = time.time() - start_phase1

    # Phase 1 RMSE for logging
    if verbose:
        p1_rmse_px = np.zeros(n_px)
        for i, px in enumerate(pixel_indices):
            diff = D_opt_all[i] - data.Dstk[px]
            p1_rmse_px[i] = np.sqrt(np.mean(diff ** 2))
        p1_valid = success_phase1
        p1_rmse = np.sqrt(np.mean(p1_rmse_px[p1_valid] ** 2)) if np.any(p1_valid) else float('nan')
        print(f"\nPhase 1 done: RMSE={p1_rmse:.4f}, "
              f"success={np.mean(success_phase1)*100:.1f}%, "
              f"time={phase1_time:.1f}s")

    # =====================================================================
    # Phase 2: Neighbor-average → MEM refinement (parallel or serial)
    # =====================================================================
    if verbose:
        print(f"\n=== Phase 2: {n_ring}-ring neighbor-avg → MEM×{mem_loops} "
              f"(alpha={mem_alpha}, {mode_str if use_parallel else 'serial'}) ===")

    neighbors = _build_first_neighbor_indices(pixel_indices, grid_size, nx=nx, ny=ny, n_ring=n_ring)
    n_with_neighbors = sum(1 for nb in neighbors if len(nb) > 0)
    if verbose:
        print(f"  Neighbor graph: {n_with_neighbors}/{n_px} pixels have neighbors "
              f"(avg {np.mean([len(nb) for nb in neighbors]):.1f} neighbors/pixel)")

    # Preserve Phase 1 results (Dtrn = before MEM, Drlt = after MEM)
    D_opt_phase1 = D_opt_all.copy()   # Dtrn equivalent

    # State arrays
    D_opt_current = D_opt_all.copy()
    thickness_current = thickness_all.copy()
    abruptness_current = abruptness_all.copy()

    # Result arrays
    rmse_px = np.zeros(n_px)
    thickness_opt = np.zeros(n_px)
    thickness_true = np.zeros(n_px)
    for i, px in enumerate(pixel_indices):
        thickness_true[i] = estimate_thickness(data.Dstk[px], data.depth)
    thickness_err = np.zeros(n_px)
    chi2_px = np.zeros(n_px)
    obj_px = np.zeros(n_px)
    success_px = np.zeros(n_px, dtype=bool)
    warm_restarts_px = warm_phase1.copy()
    rmse_elem = np.zeros((n_px, data.n_elements))

    start_phase2 = time.time()

    for mem_loop in range(mem_loops):
        # Serial: compute neighbor-averaged D_prior
        D_prior_neighbor = _neighbor_average_prior(D_opt_current, neighbors)

        if verbose:
            print(f"\n  MEM loop {mem_loop+1}/{mem_loops}...")

        # Build Phase 2 args
        phase2_args = []
        for i, px in enumerate(pixel_indices):
            phase2_args.append((
                i, data.Xstk[px], D_opt_current[i], D_prior_neighbor[i],
                float(thickness_current[i]), float(abruptness_current[i]),
                tm_data, data.elements, data.linkedconst, lambda_basic,
                mem_alpha, use_xreg, solver_max_iter, solver_tol,
                mem_scipy_method, semi_config_base_dict, mem_space,
            ))

        if use_parallel:
            with mp.Pool(n_workers, initializer=_worker_initializer) as pool:
                results_iter = pool.imap_unordered(_worker_phase2, phase2_args)
                done = 0
                for res in results_iter:
                    idx, D_opt, thk, abrp, chi2, obj, suc, warm = res
                    if D_opt is not None:
                        D_opt_current[idx] = D_opt
                        thickness_current[idx] = thk
                        abruptness_current[idx] = abrp
                        chi2_px[idx] = chi2
                        obj_px[idx] = obj
                        warm_restarts_px[idx] += warm
                    done += 1
                    if verbose and done % max(1, n_px // 10) == 0:
                        elapsed = time.time() - start_phase2
                        print(f"    [{done}/{n_px}] ({elapsed:.1f}s)")
        else:
            for args_i in phase2_args:
                res = _worker_phase2(args_i)
                idx, D_opt, thk, abrp, chi2, obj, suc, warm = res
                if D_opt is not None:
                    D_opt_current[idx] = D_opt
                    thickness_current[idx] = thk
                    abruptness_current[idx] = abrp
                    chi2_px[idx] = chi2
                    obj_px[idx] = obj
                    warm_restarts_px[idx] += warm
                if verbose and (idx + 1) % max(1, n_px // 10) == 0:
                    elapsed = time.time() - start_phase2
                    px_id = pixel_indices[idx]
                    print(f"    [{idx+1}/{n_px}] px={px_id:4d}  "
                          f"t_opt={thk:.0f}  ({elapsed:.1f}s)")

    # Compute final metrics (RMSE and W1)
    w1_elem = np.zeros((n_px, data.n_elements))
    for i, px in enumerate(pixel_indices):
        D_true = data.Dstk[px]
        diff = D_opt_current[i] - D_true
        rmse_px[i] = np.sqrt(np.mean(diff ** 2))
        for e in range(data.n_elements):
            rmse_elem[i, e] = np.sqrt(np.mean(diff[:, e] ** 2))
            w1_elem[i, e] = wasserstein_distance_1d(data.depth, D_true[:, e], D_opt_current[i, :, e])
        thickness_opt[i] = thickness_current[i]
        thickness_err[i] = abs(thickness_current[i] - thickness_true[i])
        # chi2_px and obj_px already populated from Phase 2 worker results
        success_px[i] = not np.isnan(rmse_px[i])

    total_time = phase1_time + (time.time() - start_phase2)
    valid = ~np.isnan(rmse_px)
    rmse_per_element = np.sqrt(np.nanmean(rmse_elem[valid] ** 2, axis=0))
    total_rmse = np.sqrt(np.nanmean(rmse_px[valid] ** 2))
    w1_per_element = np.nanmean(w1_elem[valid], axis=0)
    total_w1 = np.nanmean(w1_per_element)

    method_label = (
        f"{phase1_method}×{phase1_loops}+TM+Xreg → "
        f"neighbor-MEM×{mem_loops}(α={mem_alpha})"
    )
    if use_parallel:
        method_label += f" [{n_workers}w]"

    D_opt_final = D_opt_current if store_profiles else None
    D_opt_p1 = D_opt_phase1 if store_profiles else None

    if verbose:
        print(f"\nPhase 2 done: RMSE={total_rmse:.4f}, "
              f"thickness={np.nanmean(thickness_err[valid]):.1f}Å, "
              f"time={time.time()-start_phase2:.1f}s")

    return BenchmarkResult(
        method=method_label,
        n_pixels=n_px,
        rmse_per_pixel=rmse_px,
        thickness_opt=thickness_opt,
        thickness_true=thickness_true,
        thickness_error=thickness_err,
        chi2_per_pixel=chi2_px,
        success_per_pixel=success_px,
        warm_restarts_per_pixel=warm_restarts_px,
        rmse_per_element=rmse_per_element,
        w1_per_element=w1_per_element,
        element_names=data.element_names,
        total_rmse=total_rmse,
        total_w1=total_w1,
        mean_thickness_error=np.nanmean(thickness_err[valid]),
        success_rate=np.mean(success_px),
        total_time=total_time,
        D_opt_all=D_opt_final,
        D_opt_phase1=D_opt_p1,
        abruptness_opt=abruptness_current.copy() if store_profiles else None,
        objective_per_pixel=obj_px.copy() if store_profiles else None,
        thickness_phase1=thickness_all.copy() if store_profiles else None,
        abruptness_phase1=abruptness_all.copy() if store_profiles else None,
        chi2_phase1=chi2_p1.copy() if store_profiles else None,
        objective_phase1=obj_p1.copy() if store_profiles else None,
        n_iters_total=total_iters,
    )


# ==============================================================================
# MATLAB baseline metrics
# ==============================================================================

def compute_matlab_baseline(data: DstkBenchmarkData, pixel_indices: np.ndarray) -> dict:
    """Compute MATLAB Drlt metrics on the same pixel set."""
    n_px = len(pixel_indices)
    rmse_px = np.zeros(n_px)
    rmse_elem = np.zeros((n_px, data.n_elements))
    w1_elem = np.zeros((n_px, data.n_elements))
    thickness_err = np.zeros(n_px)

    for i, px in enumerate(pixel_indices):
        D_true = data.Dstk[px]
        D_matlab = data.Drlt[px]
        diff = D_matlab - D_true
        rmse_px[i] = np.sqrt(np.mean(diff ** 2))
        for e in range(data.n_elements):
            rmse_elem[i, e] = np.sqrt(np.mean(diff[:, e] ** 2))
            w1_elem[i, e] = wasserstein_distance_1d(data.depth, D_true[:, e], D_matlab[:, e])
        t_true = estimate_thickness(D_true, data.depth)
        t_matlab = estimate_thickness(D_matlab, data.depth)
        thickness_err[i] = abs(t_matlab - t_true)

    w1_per_element = np.nanmean(w1_elem, axis=0)

    return {
        "total_rmse": np.sqrt(np.mean(rmse_px ** 2)),
        "rmse_per_element": np.sqrt(np.mean(rmse_elem ** 2, axis=0)),
        "w1_per_element": w1_per_element,
        "total_w1": np.nanmean(w1_per_element),
        "mean_thickness_error": np.mean(thickness_err),
    }


# ==============================================================================
# Report
# ==============================================================================

def print_report(
    result: BenchmarkResult,
    matlab: dict,
    data: DstkBenchmarkData,
):
    """Print formatted comparison report."""
    w = 80
    print("=" * w)
    print("Dstk Benchmark Results")
    print("=" * w)
    print(f"Method:     SemiFreeSolver + {result.method}")
    print(f"Pixels:     {result.n_pixels}")
    print(f"Time:       {result.total_time:.1f}s ({result.total_time/result.n_pixels*1000:.0f}ms/pixel)")
    print(f"Success:    {result.success_rate*100:.1f}%")
    print(f"Warm restarts: {np.mean(result.warm_restarts_per_pixel):.2f} avg")

    # --- RMSE table ---
    print("-" * w)
    print(f"{'Element':<10} {'Python RMSE':>12} {'MATLAB RMSE':>12} {'Δ':>8}")
    print("-" * w)
    for i, name in enumerate(result.element_names):
        py_rmse = result.rmse_per_element[i]
        ml_rmse = matlab["rmse_per_element"][i]
        delta = py_rmse - ml_rmse
        marker = " <<" if delta < -0.01 else (" >>" if delta > 0.01 else "")
        print(f"{name:<10} {py_rmse:>12.4f} {ml_rmse:>12.4f} {delta:>+8.4f}{marker}")
    print("-" * w)
    py_total = result.total_rmse
    ml_total = matlab["total_rmse"]
    delta = py_total - ml_total
    print(f"{'TOTAL':<10} {py_total:>12.4f} {ml_total:>12.4f} {delta:>+8.4f}")

    # --- W1 table ---
    has_w1 = (result.w1_per_element is not None and "w1_per_element" in matlab)
    if has_w1:
        print()
        print("-" * w)
        print(f"{'Element':<10} {'Python W1(Å)':>12} {'MATLAB W1(Å)':>12} {'Δ(Å)':>8}")
        print("-" * w)
        for i, name in enumerate(result.element_names):
            py_w1 = result.w1_per_element[i]
            ml_w1 = matlab["w1_per_element"][i]
            delta_w1 = py_w1 - ml_w1
            marker = " <<" if delta_w1 < -0.5 else (" >>" if delta_w1 > 0.5 else "")
            print(f"{name:<10} {py_w1:>12.2f} {ml_w1:>12.2f} {delta_w1:>+8.2f}{marker}")
        print("-" * w)
        py_w1_total = result.total_w1
        ml_w1_total = matlab["total_w1"]
        delta_w1_total = py_w1_total - ml_w1_total
        print(f"{'MEAN':<10} {py_w1_total:>12.2f} {ml_w1_total:>12.2f} {delta_w1_total:>+8.2f}")

    # --- Thickness ---
    print("-" * w)
    print(f"{'Thickness':<10} {result.mean_thickness_error:>10.1f} Å {matlab['mean_thickness_error']:>10.1f} Å")
    print("=" * w)


# ==============================================================================
# Plotting
# ==============================================================================

def plot_results(
    result: BenchmarkResult,
    data: DstkBenchmarkData,
    pixel_indices: np.ndarray,
    output_dir: Path,
):
    """Generate benchmark plots."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Representative profiles ---
    n_show = min(4, result.n_pixels)
    if result.D_opt_all is not None:
        fig, axes = plt.subplots(n_show, 1, figsize=(10, 3 * n_show), sharex=True)
        if n_show == 1:
            axes = [axes]

        # Pick pixels spread across thickness range
        sort_idx = np.argsort(result.thickness_true)
        show_indices = sort_idx[np.linspace(0, len(sort_idx) - 1, n_show, dtype=int)]

        for ax_i, si in enumerate(show_indices):
            px = pixel_indices[si]
            D_true = data.Dstk[px]
            D_opt = result.D_opt_all[si]
            D_matlab = data.Drlt[px]

            ax = axes[ax_i]
            for e in range(data.n_elements):
                color = f"C{e}"
                ax.plot(data.depth, D_true[:, e], "-", color=color, alpha=0.5, linewidth=2)
                ax.plot(data.depth, D_opt[:, e], "--", color=color, linewidth=1.5,
                        label=data.element_names[e] if ax_i == 0 else None)
                ax.plot(data.depth, D_matlab[:, e], ":", color=color, linewidth=1, alpha=0.5)

            ax.set_ylabel("Concentration")
            t_true = result.thickness_true[si]
            t_opt = result.thickness_opt[si]
            ax.set_title(
                f"px={px}  t_true={t_true:.0f}Å  t_opt={t_opt:.0f}Å  "
                f"err={result.thickness_error[si]:.1f}Å  rmse={result.rmse_per_pixel[si]:.4f}"
            )
            ax.axvline(t_true, color="gray", linestyle="-", alpha=0.3)
            ax.axvline(t_opt, color="red", linestyle="--", alpha=0.3)

        axes[0].legend(loc="upper right", fontsize=8, ncol=3)
        axes[-1].set_xlabel("Depth (Å)")
        fig.suptitle(f"SemiFreeSolver {result.method} — solid=truth, dashed=Python, dotted=MATLAB")
        fig.tight_layout()
        fig.savefig(output_dir / "benchmark_dstk_profiles.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / 'benchmark_dstk_profiles.png'}")

    # --- Plot 2: Thickness error histogram ---
    fig, ax = plt.subplots(figsize=(8, 4))
    valid = result.success_per_pixel
    ax.hist(result.thickness_error[valid], bins=30, alpha=0.7, label="Python")
    ax.axvline(result.mean_thickness_error, color="red", linestyle="--",
               label=f"Mean={result.mean_thickness_error:.1f}Å")
    ax.set_xlabel("Thickness Error (Å)")
    ax.set_ylabel("Count")
    ax.set_title(f"Thickness Error Distribution — {result.method}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_dstk_thickness_hist.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'benchmark_dstk_thickness_hist.png'}")


# ==============================================================================
# H5 profile saving
# ==============================================================================

def _save_profiles_to_h5(
    h5_path: str,
    result: BenchmarkResult,
    data: DstkBenchmarkData,
    pixel_indices: np.ndarray,
    mode: str,
):
    """Save Python Dtrn (Phase1) and Drlt (Phase2/final) to the H5 file.

    Arrays are stored in MATLAB layout: (n_pixels, n_elements, n_depth) as float32.
    For 'full' mode, writes all pixels. For subset/quick, skips saving.
    """
    if mode != "full":
        print(f"[save-h5] Skipping H5 save (mode={mode}, need 'full' for complete profiles)")
        return

    n_px = len(pixel_indices)
    n_depth = data.n_depth
    n_elements = data.n_elements

    # Drlt = final result (Phase 2 if neighbor-MEM, or Phase 1 only)
    D_final = result.D_opt_all  # (n_px, n_depth, n_elements)
    # Dtrn = Phase 1 pre-MEM (if available)
    D_phase1 = result.D_opt_phase1  # (n_px, n_depth, n_elements) or None

    # Transpose to MATLAB layout: (n_pixels, n_elements, n_depth)
    Drlt_h5 = np.transpose(D_final, (0, 2, 1)).astype(np.float32)

    if D_phase1 is not None:
        Dtrn_h5 = np.transpose(D_phase1, (0, 2, 1)).astype(np.float32)
    else:
        # If no Phase 1 (single-phase pipeline), Dtrn = Drlt
        Dtrn_h5 = Drlt_h5.copy()

    # Also save Qtrn/Qrlt (reconstructed intensities) if available
    # For now, just save Dtrn/Drlt

    print(f"[save-h5] Saving to {h5_path}")
    print(f"  Dtrn: {Dtrn_h5.shape} (Phase 1, MATLAB layout)")
    print(f"  Drlt: {Drlt_h5.shape} (Final, MATLAB layout)")

    with h5py.File(h5_path, "a") as f:
        for key, arr in [("Dtrn", Dtrn_h5), ("Drlt", Drlt_h5)]:
            if key in f:
                # Check if it's MATLAB data we shouldn't overwrite
                existing = f[key]
                if existing.shape == arr.shape:
                    # Replace Python results
                    prefix = "Py_" + key
                    if prefix in f:
                        del f[prefix]
                    f.create_dataset(prefix, data=arr, compression="gzip")
                    print(f"  {prefix}: saved ({key} already exists with MATLAB data)")
                else:
                    print(f"  {key}: shape mismatch! existing={existing.shape}, "
                          f"new={arr.shape} — saving as Py_{key}")
                    prefix = "Py_" + key
                    if prefix in f:
                        del f[prefix]
                    f.create_dataset(prefix, data=arr, compression="gzip")
            else:
                f.create_dataset(key, data=arr, compression="gzip")
                print(f"  {key}: saved")

    print(f"[save-h5] Done!")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dstk ground truth benchmark")
    parser.add_argument("--h5", default=DEFAULT_H5_PATH, help="Path to HDF5 file")
    parser.add_argument("--method", default="LLS", choices=["LLS", "L1", "L2", "MEM"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--mem-alpha", type=float, default=0.0003, help="MEM alpha for pipeline")
    parser.add_argument("--thickness-init", type=float, default=210.0)
    parser.add_argument("--abruptness-init", type=float, default=20.0,
                        help="Initial abruptness (FWHM). MATLAB CalcDistrib uses 20 as sigma directly; "
                             "to match, use 33.3 = 20*2*sqrt(ln2)")
    parser.add_argument("--pipeline", choices=["l1-mem", "iterative", "neighbor-mem"],
                        help="Pipeline: l1-mem (L1→MEM), iterative (multi-loop), "
                             "neighbor-mem (LLS→neighbor-avg→MEM)")
    parser.add_argument("--loops", type=int, default=10, help="Number of loops for iterative pipeline")
    parser.add_argument("--mem-loops", type=int, default=3, help="MEM refinement loops for neighbor-mem")
    parser.add_argument("--mem-space", type=str, default="concentration",
                        choices=["concentration", "log"],
                        help="MEM optimization space: concentration (Python I-div) or log (MATLAB entropy)")
    parser.add_argument("--n-ring", type=int, default=1,
                        help="Neighbor ring radius (Manhattan L1 distance). "
                             "N=1: 4 neighbors, N=2: 12, N=11: ~484")
    parser.add_argument("--degree-decay", type=float, default=0.8, help="Degree decay per loop")
    parser.add_argument("--no-update-transmat", action="store_true", help="Disable TransMat update")
    parser.add_argument("--no-update-prior", action="store_true", help="Disable Prior update")
    parser.add_argument("--xreg", action="store_true", help="Enable Xreg weighting (MATLAB-style)")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations per solve")
    parser.add_argument("--tol", type=float, default=1e-6, help="Solver tolerance (MATLAB uses 1e-12)")
    parser.add_argument("--l1-smoothing", default="none", choices=["none", "huber", "log_cosh"],
                        help="L1 smoothing: none (FD gradient), log_cosh (analytical gradient)")
    parser.add_argument("--l1-beta", type=float, default=0.01,
                        help="Smoothness parameter for log_cosh L1 (smaller = closer to L1)")
    parser.add_argument("--scipy-method", default="SLSQP", choices=["SLSQP", "trust-constr"],
                        help="Scipy optimizer method")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (1=serial, 0=auto-detect)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Noise level (fraction, e.g. 0.02 = 2%%. 0 = no noise)")
    parser.add_argument("--noise-type", default="relative", choices=["relative", "poisson"],
                        help="Noise model: relative = X*(1+σ*randn), poisson = X+sqrt(X)*σ*randn")
    parser.add_argument("--noisy-ismp", type=str, default=None,
                        help="Path to noisy Ismp H5 (e.g. .../<main>.h5 with /Ismp)")
    parser.add_argument("--noisy-ref", type=str, default=None,
                        help="Path to reference H5 with 1-pixel Xstk (required with --noisy-ismp)")
    parser.add_argument("--nx", type=int, default=None,
                        help="Spatial grid width (auto-detect from n_pixels if not specified)")
    parser.add_argument("--ny", type=int, default=None,
                        help="Spatial grid height (auto-detect from n_pixels if not specified)")
    parser.add_argument("--quick", action="store_true", help="10 random pixels")
    parser.add_argument("--subset", action="store_true", help="50 representative pixels")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--save-h5", action="store_true",
                        help="Save Dtrn (Phase1) and Drlt (Phase2/final) to the input H5 file")
    parser.add_argument("--save-full-h5", action="store_true",
                        help="Save full MATLAB GUI-compatible H5 (all 24 datasets + derived metrics)")
    parser.add_argument("--output-h5", type=str, default=None,
                        help="Output H5 path for --save-full-h5 (default: same dir, *_python.h5)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--grid", default=None,
                        choices=list(GRID_TYPES),
                        help="Depth grid type. None = use original H5 grid. "
                             "Remaps Dstk/TransMat to non-uniform grid for comparison.")
    parser.add_argument("--n-grid", type=int, default=None,
                        help="Number of grid points for --grid (default: use original n_depth)")
    args = parser.parse_args()

    # --save-h5 or --save-full-h5 implies store_profiles
    need_profiles = args.plot or args.save_h5 or args.save_full_h5

    mode = "quick" if args.quick else ("subset" if args.subset else "full")

    print(f"Loading data from {args.h5}...")
    data = load_dstk_data(args.h5)
    print(f"  {data.n_pixels} pixels, {data.n_elements} elements, "
          f"{data.n_depth} depths, {data.n_angles} angles")
    print(f"  free={data.free_matlab.tolist()} (MATLAB 1-based), "
          f"linked={data.linked_matlab.tolist()}, subst={data.subst_matlab.tolist()}")

    # Auto-detect nx/ny if not provided
    if args.nx is None or args.ny is None:
        # Heuristic: try common grid sizes
        n = data.n_pixels
        for try_nx in [5, 10, 20, 25, 50]:
            for try_ny in [5, 10, 20, 25, 50]:
                if n % (try_nx * try_ny) == 0:
                    nt = n // (try_nx * try_ny)
                    if nt >= 1:
                        nx = args.nx if args.nx is not None else try_nx
                        ny = args.ny if args.ny is not None else try_ny
                        break
            else:
                continue
            break
        else:
            nx = args.nx or 5
            ny = args.ny or 5
    else:
        nx = args.nx
        ny = args.ny
    nt = data.n_pixels // (nx * ny)
    print(f"  Grid: {nx}×{ny} spatial × {nt} temporal = {nx*ny*nt} pixels")

    # Noise injection
    if args.noisy_ismp:
        ref_path = args.noisy_ref
        if ref_path is None:
            raise ValueError("--noisy-ref is required with --noisy-ismp")
        print(f"  Loading noisy Xstk from Ismp: {args.noisy_ismp}")
        print(f"  Reference Xstk: {ref_path}")
        load_noisy_xstk_from_ismp(data, args.noisy_ismp, ref_path)
        # Report noise level
        orig_data = load_dstk_data(args.h5)
        for e in range(data.n_elements):
            rel = (data.Xstk[:, :, e] - orig_data.Xstk[:, :, e]) / (orig_data.Xstk[:, :, e] + 1e-15)
            print(f"    {data.element_names[e]}: bias={np.mean(rel):+.3f}, noise(std)={np.std(rel):.3f}")
    elif args.noise > 0:
        print(f"  Adding {args.noise_type} noise: {args.noise*100:.1f}%")
        add_noise_to_xstk(data, args.noise, args.noise_type)

    # Non-uniform grid remapping — must happen BEFORE pixel selection and
    # MATLAB baseline so that all comparisons use the same grid.
    # Pixel selection uses the original grid to find representative pixels
    # (since thickness estimation degrades on sparse grids).
    if args.grid is not None:
        from deppro.depth_grid import make_grid
        n_grid = args.n_grid or data.n_depth
        z_max = float(data.depth[-1])
        # Estimate mean IMFP (passed to grid types that use it)
        lambda_eff = 25.0  # default estimate
        try:
            with h5py.File(args.h5, 'r') as f:
                lambda_obs_h5 = f['ezdeprof/lambdaobs'][:].squeeze().astype(np.float64)
                lambda_eff = float(np.mean(lambda_obs_h5))
        except Exception:
            pass

        # Select pixels on original grid (before remapping)
        pixel_indices = select_pixels(data, mode)

        new_depth = make_grid(
            args.grid, n_grid, z_max,
            lambda_eff=lambda_eff,
            angles_deg=data.angles,
        )
        print(f"  Remapping to {args.grid} grid: n_depth {data.n_depth} → {n_grid}")
        print(f"    z_max={z_max:.1f}Å, lambda_eff={lambda_eff:.1f}Å")
        print(f"    Grid points: {new_depth[:5]}... {new_depth[-2:]}")
        remap_data_to_grid(data, new_depth)
        print(f"    New data shape: Dstk={data.Dstk.shape}, depth={data.depth.shape}")
    else:
        pixel_indices = select_pixels(data, mode)

    print(f"Selected {len(pixel_indices)} pixels (mode={mode})")

    # MATLAB baseline (on current grid — remapped if --grid was used)
    print("Computing MATLAB baseline...")
    matlab = compute_matlab_baseline(data, pixel_indices)

    # Run benchmark
    # Auto-detect workers
    n_workers = args.workers
    if n_workers == 0:
        import multiprocessing as mp
        n_workers = mp.cpu_count()
        print(f"Auto-detected {n_workers} CPU cores")

    if args.pipeline == "neighbor-mem":
        print(f"Running neighbor-MEM pipeline "
              f"(phase1={args.method}×{args.loops}, MEM α={args.mem_alpha}, "
              f"MEM loops={args.mem_loops}, MEM space={args.mem_space}, "
              f"Xreg={args.xreg}, scipy={args.scipy_method}, workers={n_workers})...")
        result = run_neighbor_mem_benchmark(
            data,
            pixel_indices=pixel_indices,
            phase1_method=args.method,
            phase1_loops=args.loops,
            phase1_alpha=args.alpha,
            use_xreg=args.xreg,
            mem_alpha=args.mem_alpha,
            mem_loops=args.mem_loops,
            mem_scipy_method=args.scipy_method,
            thickness_init=args.thickness_init,
            abruptness_init=args.abruptness_init,
            solver_max_iter=args.max_iter,
            solver_tol=args.tol,
            nx=nx,
            ny=ny,
            n_ring=args.n_ring,
            n_workers=n_workers,
            l1_smoothing=args.l1_smoothing,
            l1_beta=args.l1_beta,
            mem_space=args.mem_space,
            verbose=True,
            store_profiles=need_profiles,
        )
    elif args.pipeline == "l1-mem":
        print(f"Running L1→MEM pipeline (L1 alpha={args.alpha}, MEM alpha={args.mem_alpha}, "
              f"abruptness={args.abruptness_init})...")
        result = run_l1_mem_pipeline(
            data,
            pixel_indices=pixel_indices,
            l1_alpha=args.alpha,
            mem_alpha=args.mem_alpha,
            thickness_init=args.thickness_init,
            abruptness_init=args.abruptness_init,
            verbose=True,
            store_profiles=need_profiles,
        )
    elif args.pipeline == "iterative":
        update_tm = not args.no_update_transmat
        update_pr = not args.no_update_prior
        print(f"Running iterative {args.method} x{args.loops} "
              f"(alpha={args.alpha}, degree_decay={args.degree_decay}, "
              f"TM={update_tm}, Prior={update_pr}, Xreg={args.xreg}, "
              f"abruptness={args.abruptness_init}, "
              f"max_iter={args.max_iter}, tol={args.tol:.0e}, "
              f"scipy={args.scipy_method})...")
        result = run_iterative_benchmark(
            data,
            method=args.method,
            alpha=args.alpha,
            n_loops=args.loops,
            degree_decay=args.degree_decay,
            update_prior=update_pr,
            update_transmat=update_tm,
            use_xreg=args.xreg,
            pixel_indices=pixel_indices,
            thickness_init=args.thickness_init,
            abruptness_init=args.abruptness_init,
            solver_max_iter=args.max_iter,
            solver_tol=args.tol,
            scipy_method=args.scipy_method,
            l1_smoothing=args.l1_smoothing,
            l1_beta=args.l1_beta,
            verbose=True,
            store_profiles=need_profiles,
        )
    else:
        print(f"Running SemiFreeSolver {args.method} (alpha={args.alpha}, "
              f"abruptness={args.abruptness_init})...")
        result = run_benchmark(
            data,
            method=args.method,
            alpha=args.alpha,
            pixel_indices=pixel_indices,
            thickness_init=args.thickness_init,
            abruptness_init=args.abruptness_init,
            verbose=True,
            store_profiles=need_profiles,
        )

    # Report
    print()
    print_report(result, matlab, data)

    # Save Dtrn/Drlt to H5 (append to input)
    if args.save_h5 and result.D_opt_all is not None:
        _save_profiles_to_h5(args.h5, result, data, pixel_indices, mode)

    # Save full GUI-compatible H5 (standalone file)
    if args.save_full_h5 and result.D_opt_all is not None:
        from deppro.benchmark_h5_writer import save_full_benchmark_h5, verify_h5
        output_h5 = args.output_h5
        if output_h5 is None:
            # Default: same dir, *_python.h5
            from pathlib import Path as P
            stem = P(args.h5).stem
            output_h5 = str(P(args.h5).parent / f"{stem}_python.h5")
        save_full_benchmark_h5(
            input_h5_path=args.h5,
            output_h5_path=output_h5,
            result=result,
            data=data,
            pixel_indices=pixel_indices,
            config={
                'method': args.method,
                'alpha': args.alpha,
                'loops': getattr(args, 'loops', 10),
                'degree_decay': 0.8,
                'n_ring': getattr(args, 'n_ring', 1),
                'mem_alpha': getattr(args, 'mem_alpha', 0.01),
                'mem_loops': getattr(args, 'mem_loops', 5),
                'l1_smoothing': getattr(args, 'l1_smoothing', 'none'),
                'l1_beta': getattr(args, 'l1_beta', 0.01),
                'pipeline': getattr(args, 'pipeline', 'iterative'),
            },
        )
        verify_h5(output_h5)

    # Plots
    if args.plot:
        plot_results(result, data, pixel_indices, Path(args.output))


if __name__ == "__main__":
    main()
