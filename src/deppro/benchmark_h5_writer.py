"""
Full HDF5 Writer for Benchmark Results
========================================

Creates a standalone, MATLAB GUI-compatible HDF5 file from benchmark results.
Includes all 24 datasets + derived metrics + ezdeprof metadata.

Q values follow MATLAB convention (Run_DepthProfiling.m L143, L400-434):
    Q = [0.5*Chi2, ObjFnc - 0.5*Chi2, ObjFnc]
    Q[0] = data-fit term (Xreg-weighted, degree-scaled Chi2)
    Q[1] = regularization term
    Q[2] = objective function value

Used by benchmark_dstk.py --save-full-h5 to produce Python-only H5 files.

Date: 2026-02-15
"""

import json
import time
import numpy as np
import h5py
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from deppro.io.dstk import BenchmarkResult, DstkBenchmarkData


def save_full_benchmark_h5(
    input_h5_path: str,
    output_h5_path: str,
    result: 'BenchmarkResult',
    data: 'DstkBenchmarkData',
    pixel_indices: np.ndarray,
    config: dict,
) -> None:
    """Save full MATLAB GUI-compatible H5 from benchmark results.

    Creates a standalone H5 file with all 24 datasets needed by
    DepthProfiler GUI: ground truth (copied), solver results (transposed),
    and derived metrics (computed).

    Args:
        input_h5_path: Path to input H5 (MATLAB format, contains Dstk etc.)
        output_h5_path: Path to output H5 (will be created/overwritten)
        result: BenchmarkResult with D_opt_all, D_opt_phase1, Q values
        data: DstkBenchmarkData with metadata
        pixel_indices: Array of pixel indices used in benchmark
        config: Dict with benchmark settings (method, alpha, loops, n_ring, etc.)
    """
    from deppro.derived_metrics import (
        DerivedMetricsConfig, compute_all_derived_batch,
    )

    n_px = result.n_pixels
    n_elem = data.n_elements
    n_depth = data.n_depth
    comp_kw = dict(compression='gzip', compression_opts=1)

    print(f"\n=== Saving full H5: {output_h5_path} ===")
    t0 = time.time()

    with h5py.File(input_h5_path, 'r') as src, \
         h5py.File(output_h5_path, 'w') as dst:

        # ==================================================================
        # 1. Copy ezdeprof group (all 99 datasets)
        # ==================================================================
        src.copy('ezdeprof', dst)
        print(f"  Copied ezdeprof ({len(dst['ezdeprof'])} datasets)")

        # ==================================================================
        # 2. Copy GUI metadata (top-level)
        # ==================================================================
        gui_meta = ['thetaobs', 'tsmp', 'Col', 'StrData',
                    'FermiEnergy', 'BindingEnergy', 'SubIndex', 'UnlockIndex']
        for name in gui_meta:
            if name in src:
                src.copy(name, dst)

        # ==================================================================
        # 3. Copy Category A: ground truth / source datasets
        # ==================================================================
        cat_a_3d = ['Dstk', 'Xstk', 'Ismp']
        cat_a_2d = ['FV0stk', 'FVstk', 'SVstk', 'SPSstk', 'Cstk', 'Qstk', 'NSstk']
        for name in cat_a_3d + cat_a_2d:
            if name in src:
                src.copy(name, dst)
        print(f"  Copied Category A datasets")

        # ==================================================================
        # 4. Write Category B: solver results (Dtrn/Drlt, Qtrn/Qrlt)
        # ==================================================================

        # D_opt: Python (n_px, n_depth, n_elem) → H5 (n_px, n_elem, n_depth)
        D_phase1 = result.D_opt_phase1 if result.D_opt_phase1 is not None else result.D_opt_all
        Dtrn_h5 = np.transpose(D_phase1, (0, 2, 1)).astype(np.float32)
        dst.create_dataset('Dtrn', data=Dtrn_h5, **comp_kw)

        Drlt_h5 = np.transpose(result.D_opt_all, (0, 2, 1)).astype(np.float32)
        dst.create_dataset('Drlt', data=Drlt_h5, **comp_kw)

        # Q: [0.5*chi2, obj - 0.5*chi2, obj]  shape (3, n_px)
        chi2_final = result.chi2_per_pixel
        obj_final = result.objective_per_pixel if result.objective_per_pixel is not None else chi2_final * 0.5
        Qrlt = np.zeros((3, n_px), dtype=np.float64)
        Qrlt[0, :] = 0.5 * chi2_final
        Qrlt[1, :] = obj_final - 0.5 * chi2_final
        Qrlt[2, :] = obj_final
        dst.create_dataset('Qrlt', data=Qrlt, **comp_kw)

        chi2_p1 = result.chi2_phase1 if result.chi2_phase1 is not None else chi2_final
        obj_p1 = result.objective_phase1 if result.objective_phase1 is not None else obj_final
        Qtrn = np.zeros((3, n_px), dtype=np.float64)
        Qtrn[0, :] = 0.5 * chi2_p1
        Qtrn[1, :] = obj_p1 - 0.5 * chi2_p1
        Qtrn[2, :] = obj_p1
        dst.create_dataset('Qtrn', data=Qtrn, **comp_kw)

        print(f"  Wrote Dtrn/Drlt {Dtrn_h5.shape}, Qtrn/Qrlt (3, {n_px})")

        # ==================================================================
        # 5. Compute and write Category C: derived metrics
        # ==================================================================

        # Build DerivedMetricsConfig
        trans_mat_opt = None
        n_angles_opt = 0
        if 'TransMatOpt' in src['ezdeprof']:
            # MATLAB stores as (n_elem, n_depth, n_angles_opt) in H5
            # → Python needs (n_angles_opt, n_depth, n_elem)
            tm_opt_raw = src['ezdeprof/TransMatOpt'][:].astype(np.float64)
            trans_mat_opt = np.transpose(tm_opt_raw, (2, 1, 0))
            n_angles_opt = trans_mat_opt.shape[0]

        all_indices = set(range(n_elem))
        used = set(data.free_indices) | set(data.substrate_indices) | set(data.linked_indices)
        const_indices = sorted(all_indices - used)

        derived_config = DerivedMetricsConfig(
            depth=data.depth,
            n_elements=n_elem,
            free_indices=data.free_indices,
            subst_indices=data.substrate_indices,
            linked_indices=data.linked_indices,
            const_indices=const_indices,
            trans_mat_opt=trans_mat_opt,
            n_angles_opt=n_angles_opt,
        )

        # Phase 1 derived (trn)
        thk_p1 = result.thickness_phase1 if result.thickness_phase1 is not None else result.thickness_opt
        abrp_p1 = result.abruptness_phase1 if result.abruptness_phase1 is not None else np.full(n_px, 20.0)
        derived_trn = compute_all_derived_batch(D_phase1, thk_p1, abrp_p1, derived_config)

        # Phase 2 derived (rlt)
        abrp_rlt = result.abruptness_opt if result.abruptness_opt is not None else np.full(n_px, 20.0)
        derived_rlt = compute_all_derived_batch(result.D_opt_all, result.thickness_opt, abrp_rlt, derived_config)

        # Xrlt: (n_px, n_angles_opt, n_elem) → H5 (n_px, n_elem, n_angles_opt)
        if derived_rlt.Xrlt is not None:
            Xrlt_h5 = np.transpose(derived_rlt.Xrlt, (0, 2, 1)).astype(np.float32)
            dst.create_dataset('Xrlt', data=Xrlt_h5, **comp_kw)
            print(f"  Wrote Xrlt {Xrlt_h5.shape}")

        # 2D derived: (n_px, dim) → H5 (dim, n_px)
        def write_2d(name, arr):
            if arr is not None:
                dst.create_dataset(name, data=arr.T.astype(np.float32), **comp_kw)

        write_2d('Ctrn', derived_trn.C)
        write_2d('Crlt', derived_rlt.C)
        write_2d('FVtrn', derived_trn.FV)
        write_2d('FVrlt', derived_rlt.FV)
        write_2d('SVtrn', derived_trn.SV)
        write_2d('SVrlt', derived_rlt.SV)
        write_2d('SPStrn', derived_trn.SPS)
        write_2d('SPSrlt', derived_rlt.SPS)
        print(f"  Wrote C/FV/SV/SPS trn+rlt")

        # alpha and degree: (3, n_px)
        alpha_val = config.get('alpha', 1.0)
        degree_decay = config.get('degree_decay', 0.8)
        n_loops = config.get('loops', 10)
        final_degree = degree_decay ** max(0, n_loops - 1)

        alpha_arr = np.zeros((3, n_px), dtype=np.float64)
        alpha_arr[0, :] = alpha_val   # Phase 1 alpha
        alpha_arr[1, :] = config.get('mem_alpha', 0.01)  # Phase 2 (MEM) alpha
        alpha_arr[2, :] = alpha_val   # Overall alpha
        dst.create_dataset('alpha', data=alpha_arr, **comp_kw)

        degree_arr = np.zeros((3, n_px), dtype=np.float64)
        degree_arr[0, :] = final_degree  # Final degree after decay
        degree_arr[1, :] = 1.0           # MEM degree (always 1.0)
        degree_arr[2, :] = final_degree
        dst.create_dataset('degree', data=degree_arr, **comp_kw)

        print(f"  Wrote alpha/degree (3, {n_px})")

        # ==================================================================
        # 6. Update ezdeprof metadata with Python-specific values
        # ==================================================================
        ez = dst['ezdeprof']

        # iterationtime
        _overwrite_scalar(ez, 'iterationtime', np.float32(result.total_time))

        # iterationtotal
        n_iters = result.n_iters_total if result.n_iters_total else 0
        _overwrite_scalar(ez, 'iterationtotal', np.uint64(n_iters))

        # NumberofNearestNeighbor
        _overwrite_scalar(ez, 'NumberofNearestNeighbor', np.uint16(config.get('n_ring', 1)))

        # Dopt: last pixel (n_depth, n_elem) → H5 (n_elem, n_depth) = transposed
        if result.D_opt_all is not None:
            last_D = result.D_opt_all[-1]  # (n_depth, n_elem)
            _overwrite_dataset(ez, 'Dopt', last_D.T.astype(np.float32))

        # Xopt: last pixel forward model at thetaopt
        if derived_rlt.Xrlt is not None:
            last_Xrlt = derived_rlt.Xrlt[-1]  # (n_angles_opt, n_elem)
            _overwrite_dataset(ez, 'Xopt', last_Xrlt.T.astype(np.float32))

        # Qopt: [0.5*chi2, reg, obj] for last pixel
        last_q = Qrlt[:, -1]  # (3,)
        _overwrite_dataset(ez, 'Qopt', last_q.reshape(1, 3).astype(np.float32))

        # FeatureVector_Calc, SupportVector_Calc, Conc_Calc, Sprs_Calc
        if derived_rlt.FV is not None:
            last_fv = derived_rlt.FV[-1]
            _overwrite_dataset(ez, 'FeatureVector_Calc', last_fv.reshape(1, -1).astype(np.float32))
        if derived_rlt.SV is not None:
            last_sv = derived_rlt.SV[-1]
            _overwrite_dataset(ez, 'SupportVector_Calc', last_sv.reshape(1, -1).astype(np.float32))
        if derived_rlt.C is not None:
            last_c = derived_rlt.C[-1]
            _overwrite_dataset(ez, 'Conc_Calc', last_c.reshape(-1, 1).astype(np.float32))
        if derived_rlt.SPS is not None:
            last_sps = derived_rlt.SPS[-1]
            _overwrite_dataset(ez, 'Sprs_Calc', last_sps.reshape(1, -1).astype(np.float32))

        print(f"  Updated ezdeprof metadata")

    elapsed = time.time() - t0
    print(f"=== H5 saved in {elapsed:.1f}s: {output_h5_path} ===\n")


def _overwrite_scalar(group: h5py.Group, name: str, value) -> None:
    """Overwrite a scalar dataset in an HDF5 group."""
    if name in group:
        group[name][...] = value
    else:
        group.create_dataset(name, data=np.array([value]))


def _overwrite_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """Overwrite or create a dataset in an HDF5 group."""
    if name in group:
        ds = group[name]
        if ds.shape == data.shape:
            ds[...] = data
        else:
            del group[name]
            group.create_dataset(name, data=data)
    else:
        group.create_dataset(name, data=data)


def verify_h5(h5_path: str) -> None:
    """Quick verification of saved H5 file."""
    with h5py.File(h5_path, 'r') as f:
        top_datasets = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        n_ez = len(f['ezdeprof']) if 'ezdeprof' in f else 0

        print(f"\n=== H5 Verification: {h5_path} ===")
        print(f"  Top-level datasets: {len(top_datasets)}")
        print(f"  ezdeprof datasets: {n_ez}")

        # Check critical datasets
        critical = ['Dstk', 'Dtrn', 'Drlt', 'Xstk', 'Qtrn', 'Qrlt',
                     'Ctrn', 'Crlt', 'FVtrn', 'FVrlt', 'SVtrn', 'SVrlt',
                     'SPStrn', 'SPSrlt', 'alpha', 'degree']
        missing = [k for k in critical if k not in f]
        if missing:
            print(f"  MISSING: {missing}")
        else:
            print(f"  All {len(critical)} critical datasets present")

        # Check Q consistency: Q[2] = Q[0] + Q[1]
        if 'Qrlt' in f:
            Q = f['Qrlt'][:]
            residual = np.abs(Q[2, :] - (Q[0, :] + Q[1, :])).max()
            print(f"  Qrlt consistency (Q[2]=Q[0]+Q[1]): max residual = {residual:.2e}")

        # Shapes
        for name in ['Dtrn', 'Drlt', 'Qtrn', 'Qrlt']:
            if name in f:
                print(f"  {name}: {f[name].shape} {f[name].dtype}")

        print("===\n")
