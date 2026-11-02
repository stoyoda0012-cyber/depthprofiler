"""
Evaluation of NN Predictions against Benchmark Data
====================================================

Computes RMSE and W1 metrics, compares with SLSQP baseline.
"""

import time
from typing import Optional

import numpy as np

import mlx.core as mx

from .model import DepthMLP
from .postprocess import enforce_constraints


# Element order mapping: H5 -> Simulator
# H5:  [Si(0), Si(+4), Ti(+4), Al(+3), C(0), O(-2)]
# Sim: [C(0),  Al(+3), Ti(+4), Si(+4), O(-2), Si(0)]
H5_TO_SIM = np.array([5, 3, 2, 1, 0, 4])  # h5[i] = sim[H5_TO_SIM[i]]
SIM_TO_H5 = np.array([4, 3, 2, 1, 5, 0])  # sim[j] = h5[SIM_TO_H5[j]]


def evaluate_on_dstk(
    model: DepthMLP,
    h5_path: str,
    batch_size: int = 10000,
    h5_element_order: bool = False,
    verbose: bool = True,
) -> dict:
    """Evaluate NN predictions against benchmark ground truth.

    Parameters
    ----------
    model : DepthMLP
        Trained model.
    h5_path : str
        Path to benchmark H5 file (with Dstk, Xstk, Drlt).
    batch_size : int
        Inference batch size.
    h5_element_order : bool
        If True, model was trained in H5 element order (train-h5 mode).
        No element reordering is applied.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        nn_rmse_per_element, nn_w1_per_element, nn_total_rmse, nn_total_w1,
        matlab_rmse_per_element, matlab_w1_per_element, matlab_total_rmse,
        matlab_total_w1, element_names, throughput, n_pixels
    """
    import h5py
    from ..depth_profile import wasserstein_distance_1d

    # Load benchmark data (Python layout: pixels, depth/angles, elements)
    with h5py.File(h5_path, 'r') as f:
        Dstk = np.transpose(f['Dstk'][:].astype(np.float64), (0, 2, 1))
        Xstk = np.transpose(f['Xstk'][:].astype(np.float64), (0, 2, 1))
        Drlt = np.transpose(f['Drlt'][:].astype(np.float64), (0, 2, 1))
        depth = f['ezdeprof/depth'][:].flatten().astype(np.float64)

        elem_raw = f['ezdeprof/element'][:].flatten()
        element_names = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in elem_raw]

    n_pixels, n_angles, n_elem = Xstk.shape
    n_depth = Dstk.shape[1]

    if verbose:
        print(f"Benchmark Evaluation")
        print(f"  H5:       {h5_path}")
        print(f"  Pixels:   {n_pixels}")
        print(f"  Angles:   {n_angles}, Depths: {n_depth}, Elements: {n_elem}")
        print()

    # Prepare input: Xstk is already per-angle normalized (sum=1)
    if h5_element_order:
        # Model trained in H5 element order (train-h5 mode) — no reordering
        I_obs_flat = Xstk.reshape(n_pixels, -1).astype(np.float32)
    else:
        # Model trained in simulator element order — reorder H5→sim
        Xstk_sim = Xstk[:, :, H5_TO_SIM]
        I_obs_flat = Xstk_sim.reshape(n_pixels, -1).astype(np.float32)

    # NN Inference
    t0 = time.perf_counter()
    D_pred_flat = []
    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        I_batch = mx.array(I_obs_flat[start:end])
        D_batch = model(I_batch)
        mx.eval(D_batch)
        D_pred_flat.append(np.array(D_batch))
    D_pred_flat = np.concatenate(D_pred_flat, axis=0)
    t_infer = time.perf_counter() - t0
    throughput = n_pixels / t_infer

    # Post-process: enforce constraints
    D_pred_flat = enforce_constraints(D_pred_flat, n_depth, n_elem)

    # Reshape to (n_px, n_depth, n_elem)
    D_pred_reshaped = D_pred_flat.reshape(n_pixels, n_depth, n_elem)

    if h5_element_order:
        # Already in H5 order
        D_pred_h5 = D_pred_reshaped
    else:
        # Reorder from sim order back to H5 order for comparison with Dstk
        D_pred_h5 = D_pred_reshaped[:, :, SIM_TO_H5]

    if verbose:
        print(f"  Inference: {t_infer:.3f}s ({throughput:,.0f} px/s)")
        print()

    # Compute NN metrics
    nn_rmse_elem = np.zeros(n_elem)
    nn_w1_elem = np.zeros(n_elem)
    nn_rmse_px = np.zeros(n_pixels)

    for px in range(n_pixels):
        diff = D_pred_h5[px] - Dstk[px]  # (n_depth, n_elem)
        nn_rmse_px[px] = np.sqrt(np.mean(diff ** 2))

    for e in range(n_elem):
        elem_rmse_sq = []
        w1_vals = []
        for px in range(n_pixels):
            diff_e = D_pred_h5[px, :, e] - Dstk[px, :, e]
            elem_rmse_sq.append(np.mean(diff_e ** 2))
            # W1 only where ground truth has signal
            gt_e = Dstk[px, :, e]
            if gt_e.max() > 1e-4:
                w1 = wasserstein_distance_1d(depth, gt_e, D_pred_h5[px, :, e])
                w1_vals.append(w1)
        nn_rmse_elem[e] = np.sqrt(np.mean(elem_rmse_sq))
        nn_w1_elem[e] = np.mean(w1_vals) if w1_vals else 0.0

    nn_total_rmse = np.sqrt(np.mean(nn_rmse_px ** 2))
    nn_total_w1 = np.mean(nn_w1_elem)

    # Compute MATLAB baseline metrics
    ml_rmse_elem = np.zeros(n_elem)
    ml_w1_elem = np.zeros(n_elem)
    ml_rmse_px = np.zeros(n_pixels)

    for px in range(n_pixels):
        diff = Drlt[px] - Dstk[px]
        ml_rmse_px[px] = np.sqrt(np.mean(diff ** 2))

    for e in range(n_elem):
        elem_rmse_sq = []
        w1_vals = []
        for px in range(n_pixels):
            diff_e = Drlt[px, :, e] - Dstk[px, :, e]
            elem_rmse_sq.append(np.mean(diff_e ** 2))
            gt_e = Dstk[px, :, e]
            if gt_e.max() > 1e-4:
                w1 = wasserstein_distance_1d(depth, gt_e, Drlt[px, :, e])
                w1_vals.append(w1)
        ml_rmse_elem[e] = np.sqrt(np.mean(elem_rmse_sq))
        ml_w1_elem[e] = np.mean(w1_vals) if w1_vals else 0.0

    ml_total_rmse = np.sqrt(np.mean(ml_rmse_px ** 2))
    ml_total_w1 = np.mean(ml_w1_elem)

    # Print comparison table
    if verbose:
        print(f"{'Element':<12} {'NN RMSE':>10} {'ML RMSE':>10} {'Δ':>8} "
              f"{'NN W1(Å)':>10} {'ML W1(Å)':>10} {'Δ W1':>8}")
        print("-" * 70)
        for e in range(n_elem):
            d_rmse = nn_rmse_elem[e] - ml_rmse_elem[e]
            d_w1 = nn_w1_elem[e] - ml_w1_elem[e]
            marker = " <<" if d_rmse < -0.005 else (" >>" if d_rmse > 0.005 else "")
            print(f"  {element_names[e]:<10} {nn_rmse_elem[e]:10.4f} "
                  f"{ml_rmse_elem[e]:10.4f} {d_rmse:+8.4f} "
                  f"{nn_w1_elem[e]:10.1f} {ml_w1_elem[e]:10.1f} "
                  f"{d_w1:+8.1f}{marker}")
        print("-" * 70)
        print(f"  {'TOTAL':<10} {nn_total_rmse:10.4f} "
              f"{ml_total_rmse:10.4f} {nn_total_rmse - ml_total_rmse:+8.4f} "
              f"{nn_total_w1:10.1f} {ml_total_w1:10.1f} "
              f"{nn_total_w1 - ml_total_w1:+8.1f}")
        print()
        print(f"  Throughput: {throughput:,.0f} px/s (this run)")

    return {
        'nn_rmse_per_element': nn_rmse_elem,
        'nn_w1_per_element': nn_w1_elem,
        'nn_total_rmse': nn_total_rmse,
        'nn_total_w1': nn_total_w1,
        'matlab_rmse_per_element': ml_rmse_elem,
        'matlab_w1_per_element': ml_w1_elem,
        'matlab_total_rmse': ml_total_rmse,
        'matlab_total_w1': ml_total_w1,
        'element_names': element_names,
        'throughput': throughput,
        'n_pixels': n_pixels,
        'D_pred': D_pred_h5,
    }
