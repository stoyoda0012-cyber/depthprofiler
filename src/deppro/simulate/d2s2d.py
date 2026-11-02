#!/usr/bin/env python
"""
D2S2D Pipeline — Depth to Spectra to Depth
============================================

Complete pipeline for generating synthetic ARXPS data and evaluating
inverse reconstruction quality. Runs entirely in NumPy (no H5 I/O).

Pipeline:
    1. Generate Dstk (ground truth) with spatial/temporal FV modulation
    2. Forward model: D(z) → I(θ) via TransferMatrix (lambda_basic cached)
    3. Add Poisson/relative noise
    4. Inverse: NN (DepthMLP) or solver (FreeSolver/SemiFreeSolver)
    5. Evaluate: per-element RMSE + W1

Usage:
    python -m deppro.simulate.d2s2d --preset 7elem_checkered --nx 5 --ny 5 --dt 1 --noise 0.0 --inverse free
    python -m deppro.simulate.d2s2d --preset 6elem_standard --noise 0.03 --inverse nn --model model.npz

Date: 2026-02-26
"""

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np

from scipy.special import erf
from scipy.optimize import minimize_scalar

from deppro import TransferMatrix, ElementConfig, CompoundDatabase
from deppro.depth_profile import wasserstein_distance_1d

# 配布対象の共有カーネルから取る。以前は deppro.benchmarks.generate_checkered から
# import していたが、benchmarks は配布に含まれないので wheel では落ちていた。
from deppro.simulate.scene_kernel import (
    SimulationConfig,
    PRESET_6ELEM_STANDARD,
    PRESET_7ELEM_CHECKERED,
    update_feature_vector,
    fv_to_profiles,
    calculate_distribution,
    calculate_distribution_gaussian,
)

try:
    from numba import jit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    prange = range
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


# ==============================================================================
# Numba batch kernels for Dstk generation + forward model
# ==============================================================================

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(math.log(2.0)))


@jit(nopython=True, cache=True, fastmath=True)
def _charge_balance_analytical(D, charges, subst_idx, antisubst):
    """Analytical single-pass charge balance (Numba nopython).

    For each depth, solves charge neutrality exactly by independently
    scaling positive-charge and negative-charge antisubst elements,
    preserving ratios within each group.

    Properties vs iterative (30-iter):
    - Charge balance: ~1e-14 (vs ~4e-05) — essentially perfect
    - D normalization: exactly 1.0 (vs up to 1.047) — physically correct
    - Speed: faster (one pass instead of 30 iterations)
    """
    n_depth = D.shape[0]
    n_elem = D.shape[1]
    n_subst = len(subst_idx)
    n_as = len(antisubst)

    for d in range(n_depth):
        # Classify antisubst elements by charge sign
        pos_charge_weighted = 0.0  # Σ D[d,e]*|charge[e]| for charge > 0
        neg_charge_weighted = 0.0  # Σ D[d,e]*|charge[e]| for charge < 0
        pos_sum = 0.0              # Σ D[d,e] for charge > 0
        neg_sum = 0.0              # Σ D[d,e] for charge < 0
        neutral_sum = 0.0          # Σ D[d,e] for charge == 0

        for i in range(n_as):
            e_idx = antisubst[i]
            c = charges[e_idx]
            d_val = D[d, e_idx]
            if c > 0.0:
                pos_charge_weighted += d_val * c
                pos_sum += d_val
            elif c < 0.0:
                neg_charge_weighted += d_val * (-c)
                neg_sum += d_val
            else:
                neutral_sum += d_val

        # Available concentration for charged elements
        subst_sum = 0.0
        for si in range(n_subst):
            subst_sum += D[d, subst_idx[si]]
        available = 1.0 - subst_sum - neutral_sum
        if available < 0.0:
            available = 0.0

        # Solve: α * pos_charge_weighted = β * neg_charge_weighted  (charge neutrality)
        #        α * pos_sum + β * neg_sum = available               (normalization)
        if pos_charge_weighted > 1e-15 and neg_charge_weighted > 1e-15:
            denom = neg_charge_weighted * pos_sum / pos_charge_weighted + neg_sum
            if denom > 1e-15:
                beta = available / denom
                alpha = beta * neg_charge_weighted / pos_charge_weighted
            else:
                alpha = 1.0
                beta = 1.0
        elif pos_sum + neg_sum > 1e-15:
            scale = available / (pos_sum + neg_sum)
            alpha = scale if pos_sum > 0.0 else 1.0
            beta = scale if neg_sum > 0.0 else 1.0
        else:
            alpha = 1.0
            beta = 1.0

        # Apply scaling
        for i in range(n_as):
            e_idx = antisubst[i]
            c = charges[e_idx]
            if c > 0.0:
                D[d, e_idx] *= alpha
            elif c < 0.0:
                D[d, e_idx] *= beta


@jit(nopython=True, cache=True, fastmath=True)
def _charge_balance_iterative(D, charges, antisubst, as_pos_mask, as_neg_mask):
    """Iterative 30-step charge balance matching MATLAB algorithm (Numba nopython).

    Legacy algorithm. Use _charge_balance_analytical for better accuracy and speed.
    """
    n_depth = D.shape[0]
    n_elem = D.shape[1]
    n_as = len(antisubst)

    for iteration in range(1, 31):
        factor = 1.0 + 100.0 * math.exp(-iteration / 2.0)
        for d in range(n_depth):
            charge_sum = 0.0
            for e in range(n_elem):
                charge_sum += D[d, e] * charges[e]

            if charge_sum < 0.0:
                for i in range(n_as):
                    if as_pos_mask[i]:
                        D[d, antisubst[i]] *= factor
            elif charge_sum > 0.0:
                for i in range(n_as):
                    if as_neg_mask[i]:
                        D[d, antisubst[i]] *= factor

            row_sum = 0.0
            for e in range(n_elem):
                row_sum += D[d, e]
            if row_sum > 0.0:
                for i in range(n_as):
                    D[d, antisubst[i]] /= row_sum


@jit(nopython=True, cache=True, fastmath=True)
def _calc_distribution_gaussian_single(
    D,               # (n_depth, n_elem) output — zeroed on entry
    depth,           # (n_depth,)
    thickness,       # scalar
    abruptness_fwhm, # scalar
    # FCL entries (sorted by element index)
    fcl_elem_idx,    # (n_fcl,) int64
    fcl_pos,         # (n_fcl,) float64
    fcl_width,       # (n_fcl,) float64
    fcl_height,      # (n_fcl,) float64
    # Element structure
    charges,         # (n_elem,) float64
    subst_idx,       # (n_subst,) int64
    antisubst,       # (n_antisubst,) int64
    as_pos_mask,     # (n_antisubst,) bool
    as_neg_mask,     # (n_antisubst,) bool
    use_analytical_cb,  # bool — True=analytical, False=iterative (MATLAB compat)
):
    """Gaussian distribution for a single pixel (Numba nopython)."""
    n_depth = len(depth)
    n_elem = len(charges)
    n_fcl = len(fcl_elem_idx)
    n_subst = len(subst_idx)

    abrp_sigma = abruptness_fwhm * _FWHM_TO_SIGMA

    # Substrate + Gaussian peaks
    for d in range(n_depth):
        # Substrate: erf rising
        for si in range(n_subst):
            D[d, subst_idx[si]] = 0.5 * abs(
                1.0 + math.erf((depth[d] - thickness) / abrp_sigma))

        # Film space
        subst_sum = 0.0
        for si in range(n_subst):
            subst_sum += D[d, subst_idx[si]]
        film_space = abs(1.0 - subst_sum)

        # Gaussian peaks for each FCL element
        for k in range(n_fcl):
            S_k = fcl_width[k] * _FWHM_TO_SIGMA
            height_frac = fcl_height[k] / 100.0
            val = height_frac * math.exp(
                -(depth[d] - fcl_pos[k])**2 / (S_k * S_k)
            ) * film_space
            # Clip to [0, 1]
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            D[d, fcl_elem_idx[k]] = val

    # Charge balancing
    if use_analytical_cb:
        _charge_balance_analytical(D, charges, subst_idx, antisubst)
    else:
        _charge_balance_iterative(D, charges, antisubst, as_pos_mask, as_neg_mask)

    # Final clip
    for d in range(n_depth):
        for e in range(n_elem):
            if D[d, e] < 0.0:
                D[d, e] = 0.0
            elif D[d, e] > 1.0:
                D[d, e] = 1.0


@jit(nopython=True, cache=True, fastmath=True)
def _calc_distribution_erf_single(
    D,               # (n_depth, n_elem) output — zeroed on entry
    depth,           # (n_depth,)
    thickness,       # scalar
    abruptness_fwhm, # scalar
    # FCL entries (sorted by element index)
    fcl_elem_idx,    # (n_fcl,) int64
    fcl_pos,         # (n_fcl,) float64
    fcl_width,       # (n_fcl,) float64
    fcl_height,      # (n_fcl,) float64
    fcl_is_neutral,  # (n_fcl,) bool — charge == 0
    fcl_half_thresh, # float — n_fcl / 2.0 + 1 (MATLAB direction threshold)
    # Element structure
    charges,         # (n_elem,) float64
    subst_idx,       # (n_subst,) int64
    antisubst,       # (n_antisubst,) int64
    as_pos_mask,     # (n_antisubst,) bool
    as_neg_mask,     # (n_antisubst,) bool
    use_analytical_cb,  # bool — True=analytical, False=iterative (MATLAB compat)
):
    """Error-function distribution for a single pixel (Numba nopython)."""
    n_depth = len(depth)
    n_elem = len(charges)
    n_fcl = len(fcl_elem_idx)
    n_subst = len(subst_idx)

    abrp_sigma = abruptness_fwhm * _FWHM_TO_SIGMA

    # Substrate: erf rising
    for d in range(n_depth):
        for si in range(n_subst):
            D[d, subst_idx[si]] = 0.5 * abs(
                1.0 + math.erf((depth[d] - thickness) / abrp_sigma))

    # D0 temporary for FCL
    D0 = np.zeros((n_depth, n_fcl))

    # Neutral elements first
    for k in range(n_fcl):
        if not fcl_is_neutral[k]:
            continue
        sigma_k = fcl_width[k] * _FWHM_TO_SIGMA
        height_frac = fcl_height[k] / 100.0
        for d in range(n_depth):
            subst_sum = 0.0
            for si in range(n_subst):
                subst_sum += D[d, subst_idx[si]]
            film_space = abs(1.0 - subst_sum)
            D0[d, k] = height_frac * 0.5 * abs(
                1.0 + math.erf(-(depth[d] - fcl_pos[k]) / sigma_k)
            ) * film_space

    # Charged elements
    for k in range(n_fcl):
        if fcl_is_neutral[k]:
            continue
        sigma_k = fcl_width[k] * _FWHM_TO_SIGMA
        height_frac = fcl_height[k] / 100.0
        for d in range(n_depth):
            subst_sum = 0.0
            for si in range(n_subst):
                subst_sum += D[d, subst_idx[si]]
            neutral_sum = 0.0
            for kk in range(n_fcl):
                if fcl_is_neutral[kk]:
                    neutral_sum += D0[d, kk]
            film_space = abs(1.0 - subst_sum - neutral_sum)
            # Direction: k+1 < n_fcl/2 + 1 → rising erf
            if (k + 1) < fcl_half_thresh:
                D0[d, k] = height_frac * 0.5 * abs(
                    1.0 + math.erf((depth[d] - fcl_pos[k]) / sigma_k)
                ) * film_space
            else:
                D0[d, k] = height_frac * 0.5 * abs(
                    1.0 + math.erf(-(depth[d] - fcl_pos[k]) / sigma_k)
                ) * film_space

    # Assign D0 → D
    for k in range(n_fcl):
        for d in range(n_depth):
            D[d, fcl_elem_idx[k]] = D0[d, k]

    # Clip
    for d in range(n_depth):
        for e in range(n_elem):
            if D[d, e] < 0.0:
                D[d, e] = 0.0
            elif D[d, e] > 1.0:
                D[d, e] = 1.0

    # Charge balancing
    if use_analytical_cb:
        _charge_balance_analytical(D, charges, subst_idx, antisubst)
    else:
        _charge_balance_iterative(D, charges, antisubst, as_pos_mask, as_neg_mask)

    # Final clip
    for d in range(n_depth):
        for e in range(n_elem):
            if D[d, e] < 0.0:
                D[d, e] = 0.0
            elif D[d, e] > 1.0:
                D[d, e] = 1.0


@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def _generate_dstk_batch(
    Dstk,            # (n_px, n_depth, n_elem) output
    FV_all,          # (n_px, n_fv) pre-computed feature vectors
    depth,           # (n_depth,)
    n_free,          # int
    use_gaussian,    # bool — Gaussian vs Error fcn
    # FCL structure (pre-computed)
    fcl_elem_idx,    # (n_fcl,) int64
    fcl_from_fv,     # (n_fcl,) bool — True if from FV (free element)
    fcl_fv_base,     # (n_fcl,) int64 — FV offset (for free elements)
    fcl_linked_src,  # (n_fcl,) int64 — index into linked arrays
    linked_pos,      # (n_linked,) float64
    linked_width,    # (n_linked,) float64
    linked_height,   # (n_linked,) float64
    fcl_is_neutral,  # (n_fcl,) bool
    fcl_half_thresh, # float
    # Element structure
    charges,         # (n_elem,)
    subst_idx,       # (n_subst,)
    antisubst,       # (n_antisubst,)
    as_pos_mask,     # (n_antisubst,) bool
    as_neg_mask,     # (n_antisubst,) bool
    use_analytical_cb,  # bool — True=analytical, False=iterative charge balance
):
    """Batch Dstk generation with parallel pixels."""
    n_px = FV_all.shape[0]
    n_fcl = len(fcl_elem_idx)

    for px in prange(n_px):
        # Extract thickness, abruptness from FV
        thickness = FV_all[px, 0]
        abruptness = FV_all[px, 1]

        # Build per-pixel FCL profiles
        fcl_pos = np.empty(n_fcl)
        fcl_width = np.empty(n_fcl)
        fcl_height = np.empty(n_fcl)
        for k in range(n_fcl):
            if fcl_from_fv[k]:
                base = fcl_fv_base[k]
                fcl_pos[k] = FV_all[px, base]
                fcl_width[k] = FV_all[px, base + 1]
                fcl_height[k] = FV_all[px, base + 2]
            else:
                li = fcl_linked_src[k]
                fcl_pos[k] = linked_pos[li]
                fcl_width[k] = linked_width[li]
                fcl_height[k] = linked_height[li]

        # Zero output
        D = Dstk[px]
        for d in range(D.shape[0]):
            for e in range(D.shape[1]):
                D[d, e] = 0.0

        if use_gaussian:
            _calc_distribution_gaussian_single(
                D, depth, thickness, abruptness,
                fcl_elem_idx, fcl_pos, fcl_width, fcl_height,
                charges, subst_idx, antisubst, as_pos_mask, as_neg_mask,
                use_analytical_cb,
            )
        else:
            _calc_distribution_erf_single(
                D, depth, thickness, abruptness,
                fcl_elem_idx, fcl_pos, fcl_width, fcl_height,
                fcl_is_neutral, fcl_half_thresh,
                charges, subst_idx, antisubst, as_pos_mask, as_neg_mask,
                use_analytical_cb,
            )


@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def _batch_forward(
    Xstk,           # (n_px, n_angles, n_elem) output
    Dstk,           # (n_px, n_depth, n_elem)
    lambda_basic,   # (n_elem, n_elem)
    angles_rad,     # (n_angles,)
    step_size,      # float
    gamma_eff,      # float
):
    """Batch forward model: Dstk → Xstk for all pixels (parallel).

    Optimizations:
    - Pre-compute decay_base[d,e] = -step_size * Σ(D/λ_basic) / γ
      (eliminates 2 divisions per inner iteration → 1 multiply)
    - Multiplicative decay accumulation (avoids exp of large cumulative sums)
    - Pre-compute inv_cos_theta (1 division per angle vs per iteration)
    """
    n_px, n_depth, n_elem = Dstk.shape
    n_angles = len(angles_rad)

    # Pre-compute inv_cos_theta (n_angles values)
    inv_cos = np.empty(n_angles)
    for a in range(n_angles):
        inv_cos[a] = 1.0 / math.cos(angles_rad[a])

    for px in prange(n_px):
        D = Dstk[px]

        # Fused lambda_obs → decay base:
        # decay_base[d,e] = -step_size / lambda_obs[d,e]
        #                  = -step_size * (Σ_i D[d,i]/λ_basic[e,i]) / γ
        decay_base = np.empty((n_depth, n_elem))
        for e in range(n_elem):
            for d in range(n_depth):
                s = 0.0
                for i in range(n_elem):
                    s += D[d, i] / lambda_basic[e, i]
                if s == 0.0:
                    s = 1e-10
                decay_base[d, e] = -step_size * s / gamma_eff

        # Multiplicative decay forward model
        for a in range(n_angles):
            ict = inv_cos[a]
            for e in range(n_elem):
                trans_val = 1.0
                intensity = D[0, e]  # first depth: trans_val = 1.0
                for d in range(1, n_depth):
                    trans_val *= math.exp(decay_base[d - 1, e] * ict)
                    intensity += trans_val * D[d, e]
                Xstk[px, a, e] = intensity

        # Normalize per angle
        for a in range(n_angles):
            total = 0.0
            for e in range(n_elem):
                total += Xstk[px, a, e]
            if total > 0.0:
                for e in range(n_elem):
                    Xstk[px, a, e] /= total


def _batch_forward_mlx(
    Dstk_np: np.ndarray,      # (n_px, n_depth, n_elem) float64
    lambda_basic_np: np.ndarray,  # (n_elem, n_elem) float64
    angles_rad_np: np.ndarray,    # (n_angles,) float64
    step_size: float,
    gamma_eff: float,
    chunk_size: int = 500_000,
) -> np.ndarray:
    """Metal GPU accelerated forward model using MLX.

    Fully vectorized: no Python loops over pixels.
    Uses float32 on GPU for throughput, converts result back to float64.

    Computation:
        1. decay_base[px,d,e] = -step_size/gamma * (D @ inv_lambda^T)[px,d,e]
        2. cum_decay = cumsum(decay_base, axis=depth)  — parallel prefix sum
        3. trans[px,d,a,e] = exp(cum_decay_shifted[px,d,e] * inv_cos[a])
        4. Xstk[px,a,e] = sum_d(trans * D)
        5. Normalize per angle

    Memory: peak = n_px * n_depth * n_angles * n_elem * 4B (float32)
    Auto-chunks if peak > ~2 GB.

    Args:
        Dstk_np: Ground truth depth profiles (n_px, n_depth, n_elem)
        lambda_basic_np: IMFP matrix (n_elem, n_elem)
        angles_rad_np: Measurement angles in radians (n_angles,)
        step_size: Depth step size [Angstrom]
        gamma_eff: Effective attenuation parameter
        chunk_size: Max pixels per GPU batch (auto-tuned by default)

    Returns:
        Xstk: Normalized angle-resolved intensities (n_px, n_angles, n_elem) float64
    """
    n_px, n_depth, n_elem = Dstk_np.shape
    n_angles = len(angles_rad_np)

    # Pre-compute constants on GPU (float32 for throughput)
    inv_lambda = 1.0 / mx.array(lambda_basic_np, dtype=mx.float32)  # (n_elem, n_elem)
    inv_lambda_T = mx.transpose(inv_lambda)  # (n_elem, n_elem)
    inv_cos = 1.0 / mx.cos(mx.array(angles_rad_np, dtype=mx.float32))  # (n_angles,)
    coeff = mx.array(-step_size / gamma_eff, dtype=mx.float32)

    # Auto-chunk: estimate peak memory for trans tensor
    # peak = n_chunk * n_depth * n_angles * n_elem * 4 bytes
    bytes_per_px = n_depth * n_angles * n_elem * 4
    max_chunk = max(1, int(2e9 / bytes_per_px))  # ~2 GB limit
    actual_chunk = min(chunk_size, max_chunk, n_px)

    if n_px <= actual_chunk:
        # Single batch — no chunking
        Xstk_mlx = _forward_mlx_batch(
            mx.array(Dstk_np, dtype=mx.float32),
            inv_lambda_T, inv_cos, coeff, n_depth)
        mx.eval(Xstk_mlx)
        return np.array(Xstk_mlx, dtype=np.float64)
    else:
        # Chunked processing
        Xstk_out = np.empty((n_px, n_angles, n_elem), dtype=np.float64)
        for start in range(0, n_px, actual_chunk):
            end = min(start + actual_chunk, n_px)
            D_chunk = mx.array(Dstk_np[start:end], dtype=mx.float32)
            X_chunk = _forward_mlx_batch(
                D_chunk, inv_lambda_T, inv_cos, coeff, n_depth)
            mx.eval(X_chunk)
            Xstk_out[start:end] = np.array(X_chunk, dtype=np.float64)
        return Xstk_out


def _forward_mlx_batch(D, inv_lambda_T, inv_cos, coeff, n_depth):
    """Core MLX forward model for a batch of pixels.

    Args:
        D: (n_chunk, n_depth, n_elem) float32 MLX array
        inv_lambda_T: (n_elem, n_elem) transposed inverse IMFP
        inv_cos: (n_angles,) inverse cosine of angles
        coeff: scalar -step_size / gamma_eff
        n_depth: number of depth points

    Returns:
        Xstk: (n_chunk, n_angles, n_elem) float32 MLX array (normalized)
    """
    n_chunk = D.shape[0]
    n_elem = D.shape[2]

    # Step 1: decay_base = coeff * (D @ inv_lambda^T)
    # D: (n_chunk, n_depth, n_elem), inv_lambda_T: (n_elem, n_elem)
    # Reshape for batch matmul: (n_chunk * n_depth, n_elem) @ (n_elem, n_elem)
    D_2d = mx.reshape(D, (-1, n_elem))
    col_2d = mx.matmul(D_2d, inv_lambda_T)  # (n_chunk * n_depth, n_elem)
    col = mx.reshape(col_2d, (n_chunk, n_depth, n_elem))
    col = mx.maximum(col, 1e-10)
    decay_base = coeff * col  # (n_chunk, n_depth, n_elem)

    # Step 2: Cumulative sum along depth → transmission exponents
    cum_decay = mx.cumsum(decay_base, axis=1)  # (n_chunk, n_depth, n_elem)

    # Shift: trans_val[d=0] = 1.0 (no attenuation at surface)
    # cum_decay_shifted[d] = cum_decay[d-1], cum_decay_shifted[0] = 0
    zeros = mx.zeros((n_chunk, 1, n_elem))
    cum_decay_shifted = mx.concatenate([zeros, cum_decay[:, :-1, :]], axis=1)

    # Step 3: Broadcast over angles
    # trans[px, d, a, e] = exp(cum_decay_shifted[px,d,e] * inv_cos[a])
    # Shape: (n_chunk, n_depth, 1, n_elem) * (1, 1, n_angles, 1)
    #      → (n_chunk, n_depth, n_angles, n_elem)
    trans = mx.exp(
        mx.expand_dims(cum_decay_shifted, axis=2) * mx.reshape(inv_cos, (1, 1, -1, 1))
    )

    # Step 4: Weighted sum over depth → intensities
    # (n_chunk, n_depth, n_angles, n_elem) * (n_chunk, n_depth, 1, n_elem)
    Xstk = mx.sum(trans * mx.expand_dims(D, axis=2), axis=1)
    # → (n_chunk, n_angles, n_elem)

    # Step 5: Normalize per angle
    Xstk_sum = mx.sum(Xstk, axis=2, keepdims=True)  # (n_chunk, n_angles, 1)
    Xstk = Xstk / mx.maximum(Xstk_sum, 1e-30)

    return Xstk


def _precompute_fv_vectorized(config: 'SimulationConfig') -> np.ndarray:
    """Vectorized FV computation for all pixels (numpy broadcasting).

    Replaces per-pixel update_feature_vector() loop with batch numpy ops.
    Supports all 13 modifiers. Returns None if flags contain unsupported ones.

    Returns:
        FV_all: (n_px, n_fv) or None if fallback needed
    """
    n_px = config.n_pixels
    base_fv = config.base_fv()
    n_fv = len(base_fv)
    n_free = (n_fv - 2) // 3
    flags = config.flags
    nx, ny, dt = config.nx, config.ny, config.dt
    # Use H5 metadata dx/dy for modulation rate (NOT grid nx/ny).
    # For 6-elem: dx=1,dy=1 (H5 metadata). For 7-elem: dx=10,dy=10.
    dx = config.eff_dx
    dy = config.eff_dy
    dxdy = dx * dy
    dxdydt = dxdy * dt

    FV_all = np.tile(base_fv, (n_px, 1))
    sk = np.arange(1, n_px + 1, dtype=np.float64)

    # === A. DistribBottomLayer (spatial, mutually exclusive) ===
    if flags.get('DistribBottomLayerTrigonometric'):
        mod_val = (2.0
                   - np.cos(2.0 * np.pi * (sk - 1) / (dx / 3.0))
                   - np.cos(2.0 * np.pi * np.floor((sk - 1) / dy) / (dy / 3.0))
                   ) / 4.0
        FV_all[:, 0] += 50.0 * mod_val
        FV_all[:, 2] += 50.0 * mod_val
        FV_all[:, 3] += 75.0 * mod_val
    elif flags.get('DistribBottomLayerLinear'):
        scale_x = dx / 6.0
        scale_y = dy / 6.0
        id1 = ~(np.floor((sk - 1) / scale_x).astype(np.int64) % 2).astype(bool)
        id2 = ~(np.floor(np.floor((sk - 1) / dy) / scale_y).astype(np.int64) % 2).astype(bool)
        x_local = (sk - 1) - np.floor((sk - 1) / scale_x) * scale_x
        y_local = np.floor((sk - 1) / dy) - np.floor(np.floor((sk - 1) / dy) / scale_y) * scale_y
        delta_fv1 = np.zeros(n_px)
        delta_fv2 = np.zeros(n_px)
        m = id1 & id2
        delta_fv1[m] = x_local[m] * (25.0 / scale_x) + y_local[m] * (25.0 / scale_y)
        delta_fv2[m] = 1.5 * x_local[m] * (25.0 / scale_x) + 1.5 * y_local[m] * (25.0 / scale_y)
        m = id1 & ~id2
        delta_fv1[m] = x_local[m] * (25.0 / scale_x) + 25.0 + (-y_local[m]) * (25.0 / scale_x)
        delta_fv2[m] = 1.5 * x_local[m] * (25.0 / scale_x) + 37.5 + 1.5 * (-y_local[m]) * (25.0 / scale_x)
        m = ~id1 & id2
        delta_fv1[m] = 25.0 + (-x_local[m]) * (25.0 / scale_x) + y_local[m] * (25.0 / scale_y)
        delta_fv2[m] = 37.5 + 1.5 * (-x_local[m]) * (25.0 / scale_x) + 1.5 * y_local[m] * (25.0 / scale_y)
        m = ~id1 & ~id2
        delta_fv1[m] = 25.0 + (-x_local[m]) * (25.0 / scale_x) + 25.0 + (-y_local[m]) * (25.0 / scale_x)
        delta_fv2[m] = 37.5 + 1.5 * (-x_local[m]) * (25.0 / scale_x) + 37.5 + 1.5 * (-y_local[m]) * (25.0 / scale_x)
        FV_all[:, 0] += delta_fv1
        FV_all[:, 2] += delta_fv1
        FV_all[:, 3] += delta_fv2
    elif flags.get('DistribBottomLayerExponential'):
        scale_x = dx / 6.0
        scale_y = dy / 6.0
        tau = dx / 30.0
        id1 = ~(np.floor((sk - 1) / scale_x).astype(np.int64) % 2).astype(bool)
        id2 = ~(np.floor(np.floor((sk - 1) / dy) / scale_y).astype(np.int64) % 2).astype(bool)
        x_local = -sk + 1 + np.floor((sk - 1) / scale_x) * scale_x
        y_local = -np.floor((sk - 1) / dy) + np.floor(np.floor((sk - 1) / dy) / scale_y) * scale_y
        exp_x_rise = 1.0 - np.exp(x_local / tau)
        exp_x_fall = np.exp(x_local / tau)
        exp_y_rise = 1.0 - np.exp(y_local / tau)
        exp_y_fall = np.exp(y_local / tau)
        delta_fv1 = np.zeros(n_px)
        delta_fv2 = np.zeros(n_px)
        m = id1 & id2;  delta_fv1[m] = 25.0 * exp_x_rise[m] + 25.0 * exp_y_rise[m]; delta_fv2[m] = 37.5 * exp_x_rise[m] + 37.5 * exp_y_rise[m]
        m = id1 & ~id2; delta_fv1[m] = 25.0 * exp_x_rise[m] + 25.0 * exp_y_rise[m]; delta_fv2[m] = 37.5 * exp_x_rise[m] + 37.5 * exp_y_rise[m]
        m = ~id1 & id2; delta_fv1[m] = 25.0 * exp_x_fall[m] + 25.0 * exp_y_rise[m]; delta_fv2[m] = 37.5 * exp_x_fall[m] + 37.5 * exp_y_rise[m]
        m = ~id1 & ~id2; delta_fv1[m] = 25.0 * exp_x_fall[m] + 25.0 * exp_y_fall[m]; delta_fv2[m] = 37.5 * exp_x_fall[m] + 37.5 * exp_y_fall[m]
        FV_all[:, 0] += delta_fv1
        FV_all[:, 2] += delta_fv1
        FV_all[:, 3] += delta_fv2

    # === B. DistribTopLayerCheckered (spatial) ===
    if flags.get('DistribTopLayerCheckered'):
        idx_h1 = 3 * (n_free - 2) + 4
        idx_h2 = 3 * (n_free - 1) + 4
        id1 = ~(np.ceil(6.0 * sk / dx).astype(np.int64) % 2).astype(bool)
        id2 = ~(np.ceil(6.0 * sk / (dx * dy)).astype(np.int64) % 2).astype(bool)
        pattern_a = (id1 & id2) | (~id1 & ~id2)
        FV_all[pattern_a, idx_h1] = 0.0
        FV_all[pattern_a, idx_h2] = 100.0
        FV_all[~pattern_a, idx_h1] = 100.0
        FV_all[~pattern_a, idx_h2] = 0.0

    # === C. PhenomInterfRedox (temporal, mutually exclusive) ===
    # Common phase computation
    phase_block = np.floor((sk - 1) / dxdydt).astype(np.int64)
    is_forward = ~(phase_block % 2).astype(bool)
    sk_local = sk - 1 - phase_block * dxdydt
    t_step = np.floor(sk_local / dxdy).astype(np.int64)

    if flags.get('PhenomInterfRedoxExponential'):
        tau = dxdydt / 5.0
        prog_fwd = 1.0 - np.exp(-t_step * dxdy / tau)
        progress = sk - 1 - phase_block * dxdydt
        back_quant = np.ceil(-progress / dxdy) * dxdy
        prog_bwd = np.exp(back_quant / tau)
        prog = np.where(is_forward, prog_fwd, prog_bwd)
        FV_all[:, 0] += 100.0 * prog
        FV_all[:, 2] += 100.0 * prog
        FV_all[:, 3] += 150.0 * prog
    elif flags.get('PhenomInterfRedoxLinear'):
        fwd_val = t_step.astype(np.float64)
        back_t = np.ceil((-sk + 1 + phase_block * dxdydt) / dxdy)
        delta_fv1 = np.where(is_forward, fwd_val, 100.0 + back_t)
        delta_fv2 = np.where(is_forward, 1.5 * fwd_val, 150.0 + 1.5 * back_t)
        FV_all[:, 0] += delta_fv1
        FV_all[:, 2] += delta_fv1
        FV_all[:, 3] += delta_fv2
    elif flags.get('PhenomInterfRedoxTrigonometric'):
        t_step_trig = np.floor((sk - 1) / dxdy).astype(np.int64)
        mod_val = (1.0 - np.cos(np.pi * t_step_trig / dt)) / 2.0
        FV_all[:, 0] += 100.0 * mod_val
        FV_all[:, 2] += 100.0 * mod_val
        FV_all[:, 3] += 150.0 * mod_val

    # === D. PhenomEtchGrow (temporal, mutually exclusive) ===
    etchgrow_idx = [0] + [2 + 3 * k for k in range(n_free)]

    if flags.get('PhenomEtchGrowLinear'):
        fwd_shift = -np.floor(sk_local / (2 * dxdy))
        back_val = np.ceil((-sk + 1 + phase_block * dxdydt) / (2 * dxdy))
        shift = np.where(is_forward, fwd_shift, -50.0 - back_val)
        for idx in etchgrow_idx:
            FV_all[:, idx] += shift
    elif flags.get('PhenomEtchGrowExponential'):
        tau = dxdydt / 5.0
        prog_fwd = 1.0 - np.exp(-t_step * dxdy / tau)
        back_local = -sk + 1 + phase_block * dxdydt
        prog_bwd = np.exp(back_local / tau)
        prog = np.where(is_forward, prog_fwd, prog_bwd)
        for idx in etchgrow_idx:
            FV_all[:, idx] -= 50.0 * prog
    elif flags.get('PhenomEtchGrowTrigonometric'):
        t_step_trig = np.floor((sk - 1) / dxdy).astype(np.int64)
        mod_val = (1.0 - np.cos(np.pi * t_step_trig / dt)) / 2.0
        for idx in etchgrow_idx:
            FV_all[:, idx] -= 50.0 * mod_val

    # === E. PhenomIntDifSep (temporal, mutually exclusive) ===
    if n_free <= 4:
        fvid1_ids = [3 * k - 1 for k in range(2, n_free - 1)]
        last_m = 3 * (n_free - 1)
        fvid2_ids = [last_m - 1, last_m]
    else:
        fvid1_ids = [3 * k - 1 for k in range(2, n_free - 2)]
        last_m = 3 * (n_free - 2)
        fvid2_ids = [last_m - 1, last_m]

    if flags.get('PhenomIntDifSepLinear'):
        fwd_shift = 0.8 * np.floor(sk_local / (2 * dxdy))
        back_val = 0.8 * np.ceil((-sk + 1 + phase_block * dxdydt) / (2 * dxdy))
        # MATLAB: fvid1 = FV0 - 40 - 0.8*ceil(...), fvid2 = FV0 + 40 + 0.8*ceil(...)
        shift1 = np.where(is_forward, -fwd_shift, -40.0 - back_val)
        shift2 = np.where(is_forward, fwd_shift, 40.0 + back_val)
        for idx in fvid1_ids:
            FV_all[:, idx] += shift1
        for idx in fvid2_ids:
            FV_all[:, idx] += shift2
    elif flags.get('PhenomIntDifSepExponential'):
        tau = dxdydt / 5.0
        prog_fwd = 1.0 - np.exp(-t_step * dxdy / tau)
        back_local = -sk + 1 + phase_block * dxdydt
        prog_bwd = np.exp(back_local / tau)
        prog = np.where(is_forward, prog_fwd, prog_bwd)
        for idx in fvid1_ids:
            FV_all[:, idx] -= 40.0 * prog
        for idx in fvid2_ids:
            FV_all[:, idx] += 40.0 * prog
    elif flags.get('PhenomIntDifSepTrigonometric'):
        t_step_trig = np.floor((sk - 1) / dxdy).astype(np.int64)
        mod_val = (1.0 - np.cos(np.pi * t_step_trig / dt)) / 2.0
        for idx in fvid1_ids:
            FV_all[:, idx] -= 40.0 * mod_val
        for idx in fvid2_ids:
            FV_all[:, idx] += 40.0 * mod_val

    return FV_all


def _prepare_batch_constants(config: 'SimulationConfig') -> dict:
    """Pre-compute Numba-compatible arrays from SimulationConfig.

    Returns dict of arrays suitable for passing to _generate_dstk_batch.
    """
    n_free = len(config.free_idx)
    charges = np.array([e["charge"] for e in config.elements], dtype=np.float64)

    # Build FCL entries: merge free + linked, sort by element index
    fcl_entries = []
    for i, fi in enumerate(config.free_idx):
        fcl_entries.append({
            'elem_idx': fi,
            'from_fv': True,
            'fv_base': 2 + 3 * i,  # offset in FV array
            'linked_src': -1,
        })
    for i, li in enumerate(config.linked_idx):
        fcl_entries.append({
            'elem_idx': li,
            'from_fv': False,
            'fv_base': -1,
            'linked_src': i,
        })
    fcl_entries.sort(key=lambda x: x['elem_idx'])

    n_fcl = len(fcl_entries)
    fcl_elem_idx = np.array([e['elem_idx'] for e in fcl_entries], dtype=np.int64)
    fcl_from_fv = np.array([e['from_fv'] for e in fcl_entries], dtype=np.bool_)
    fcl_fv_base = np.array([e['fv_base'] for e in fcl_entries], dtype=np.int64)
    fcl_linked_src = np.array([e['linked_src'] for e in fcl_entries], dtype=np.int64)

    # Linked profiles as arrays
    n_linked = len(config.linked_profiles)
    linked_pos = np.array([p["pos"] for p in config.linked_profiles], dtype=np.float64) if n_linked > 0 else np.empty(0, dtype=np.float64)
    linked_width = np.array([p["width"] for p in config.linked_profiles], dtype=np.float64) if n_linked > 0 else np.empty(0, dtype=np.float64)
    linked_height = np.array([p["height"] for p in config.linked_profiles], dtype=np.float64) if n_linked > 0 else np.empty(0, dtype=np.float64)

    # Neutral mask (for Error fcn mode)
    fcl_charges = np.array([charges[e['elem_idx']] for e in fcl_entries])
    fcl_is_neutral = np.array([c == 0 for c in fcl_charges], dtype=np.bool_)
    fcl_half_thresh = n_fcl / 2.0 + 1.0

    # Antisubst: sorted free + linked indices
    antisubst = np.array(sorted(config.free_idx + config.linked_idx), dtype=np.int64)
    antisubst_charges = charges[antisubst]
    as_pos_mask = np.array(antisubst_charges >= 0, dtype=np.bool_)
    as_neg_mask = np.array(antisubst_charges < 0, dtype=np.bool_)

    subst_idx = np.array(config.subst_idx, dtype=np.int64)

    return {
        'n_free': n_free,
        'use_gaussian': getattr(config, 'dist_type', 'Error fcn.') == 'Gaussian',
        'fcl_elem_idx': fcl_elem_idx,
        'fcl_from_fv': fcl_from_fv,
        'fcl_fv_base': fcl_fv_base,
        'fcl_linked_src': fcl_linked_src,
        'linked_pos': linked_pos,
        'linked_width': linked_width,
        'linked_height': linked_height,
        'fcl_is_neutral': fcl_is_neutral,
        'fcl_half_thresh': fcl_half_thresh,
        'charges': charges,
        'subst_idx': subst_idx,
        'antisubst': antisubst,
        'as_pos_mask': as_pos_mask,
        'as_neg_mask': as_neg_mask,
    }


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class D2S2DConfig:
    """D2S2D pipeline configuration."""

    # Simulation (Dstk generation)
    sim_config: SimulationConfig = field(default_factory=lambda: PRESET_7ELEM_CHECKERED)

    # Charge balance algorithm
    # 'analytical': 1-pass direct solution (exact charge balance, one pass)
    # 'iterative': 30-step multiplicative update (MATLAB compatible)
    charge_balance: Literal['analytical', 'iterative'] = 'analytical'

    # Forward model backend
    # 'auto': MLX (Metal GPU) if available, else Numba CPU
    # 'mlx': Force Metal GPU (requires MLX)
    # 'numba': Force Numba CPU
    forward_backend: Literal['auto', 'mlx', 'numba'] = 'auto'

    # Noise injection
    noise_level: float = 0.0
    noise_type: Literal['poisson', 'relative'] = 'poisson'
    noise_seed: int = 42

    # Inverse reconstruction
    inverse_method: Literal['nn', 'free', 'semifree', 'nn_semifree',
                            'mlx_free', 'mlx_semifree'] = 'free'
    nn_model_path: Optional[str] = None
    solver_method: str = 'L1'
    solver_alpha: float = 0.01
    solver_max_iter: int = 200
    solver_loops: int = 3

    # SemiFreeSolver-specific settings
    semifree_init: Literal['gt_fv', 'base_fv', 'default', 'nn'] = 'gt_fv'
    semifree_thickness_bounds: tuple = (50.0, 500.0)
    semifree_abruptness_bounds: tuple = (1.0, 50.0)
    semifree_optimize_abruptness: bool = False


@dataclass
class D2S2DResult:
    """D2S2D pipeline results."""

    # Arrays
    Dstk: np.ndarray        # (n_px, n_depth, n_elem) ground truth
    Xstk: np.ndarray        # (n_px, n_angles, n_elem) clean observations
    Xstk_noisy: np.ndarray  # (n_px, n_angles, n_elem) noisy observations
    Drlt: np.ndarray        # (n_px, n_depth, n_elem) reconstruction

    # Per-element metrics
    rmse_per_element: np.ndarray  # (n_elem,)
    rmse_total: float
    w1_per_element: np.ndarray    # (n_elem,)
    w1_mean: float

    # Metadata
    depth: np.ndarray
    angles: np.ndarray
    element_names: list
    config: D2S2DConfig
    elapsed_generate: float
    elapsed_inverse: float
    elapsed_total: float
    FV_all: Optional[np.ndarray] = None  # (n_px, n_fv) GT feature vectors


# ==============================================================================
# Phase 1: Generate Dstk + Xstk (optimized)
# ==============================================================================

def generate_dstk_xstk(
    config: SimulationConfig,
    verbose: bool = True,
    charge_balance: str = 'analytical',
    forward_backend: str = 'auto',
) -> dict:
    """Generate ground truth Dstk and angle-resolved Xstk.

    Uses Numba batch kernels when available (parallel prange):
    - Phase 1: Pre-compute FV_all in Python (update_feature_vector per pixel)
    - Phase 2: Numba batch Dstk generation (parallel, all pixels)
    - Phase 3: Forward model (MLX Metal GPU or Numba CPU)

    Falls back to per-pixel Python loop when Numba is not available.

    Args:
        config: SimulationConfig with spatial/temporal flags
        verbose: Print progress
        charge_balance: 'analytical' (fast, exact) or 'iterative' (MATLAB compat)
        forward_backend: 'auto' (MLX if available), 'mlx', or 'numba'

    Returns:
        dict with Dstk, Xstk, depth, angles, element_configs, charges
    """
    n_pixels = config.n_pixels
    depth = np.arange(config.n_depth, dtype=np.float64) * config.step_size
    angles = np.linspace(config.angle_min, config.angle_max, config.n_angles)
    n_elements = len(config.elements)
    n_free = len(config.free_idx)
    charges = np.array([e["charge"] for e in config.elements], dtype=np.float64)
    use_analytical_cb = (charge_balance == 'analytical')

    element_configs = [
        ElementConfig(
            symbol=e["symbol"], orbital=e["orbital"], compound=e["compound"],
            binding_energy=e["BE"], oxidation_state=e["charge"],
        )
        for e in config.elements
    ]
    compound_db = CompoundDatabase()

    # Compute lambda_basic once (expensive IMFP lookup)
    tm = TransferMatrix(
        depth=depth, step_size=config.step_size, angles=angles,
        photon_energy=config.photon_energy, work_function=config.work_function,
        gamma_eff=config.gamma_eff,
    )
    lambda_basic = tm.calculate_lambda_basic(element_configs, compound_db)

    # Resolve forward backend
    use_mlx_forward = False
    if forward_backend == 'auto':
        use_mlx_forward = _HAS_MLX
    elif forward_backend == 'mlx':
        if not _HAS_MLX:
            raise RuntimeError("MLX not available. Install with: pip install mlx")
        use_mlx_forward = True
    # else: 'numba' → use_mlx_forward = False

    if verbose:
        mode = "Numba parallel" if _HAS_NUMBA else "Python (slow)"
        fwd_mode = "MLX Metal GPU" if use_mlx_forward else "Numba CPU"
        cb_mode = "analytical" if use_analytical_cb else "iterative"
        print(f"D2S2D Generate [{mode}, fwd={fwd_mode}, cb={cb_mode}]: {config.nx}x{config.ny}"
              f" (dt={config.dt}, phases={config.n_phases}) = {n_pixels} pixels")
        print(f"  depth: {config.n_depth} pts, step={config.step_size} A")
        print(f"  angles: {config.n_angles} pts, {config.angle_min}-{config.angle_max} deg")
        print(f"  elements: {[e['symbol'] for e in config.elements]} ({n_elements})")
        flags_on = [k for k, v in config.flags.items() if v]
        print(f"  flags: {flags_on}")
        print(f"  dist_type: {getattr(config, 'dist_type', 'Error fcn.')}")

    t0 = time.perf_counter()

    if _HAS_NUMBA:
        result = _generate_fast(
            config, depth, angles, n_pixels, n_elements, n_free,
            charges, lambda_basic, verbose, use_analytical_cb, use_mlx_forward)
    else:
        result = _generate_slow(
            config, depth, angles, n_pixels, n_elements, n_free,
            charges, lambda_basic, tm, verbose)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  Generation done: {elapsed:.1f}s ({n_pixels/elapsed:.0f} px/s)")

    return {
        "Dstk": result[0],
        "Xstk": result[1],
        "FV_all": result[2],
        "depth": depth,
        "angles": angles,
        "element_configs": element_configs,
        "charges": charges,
        "elapsed": elapsed,
    }


def _generate_fast(config, depth, angles, n_pixels, n_elements, n_free,
                   charges, lambda_basic, verbose, use_analytical_cb=True,
                   use_mlx_forward=False):
    """Numba-accelerated generation: FV pre-compute → batch Dstk → batch forward."""
    t1 = time.perf_counter()

    # Phase 1: Pre-compute all FVs (vectorized numpy)
    FV_all = _precompute_fv_vectorized(config)

    t_fv = time.perf_counter() - t1
    if verbose:
        print(f"  FV vectorized: {n_pixels} pixels in {t_fv*1000:.1f}ms"
              f" ({n_pixels/t_fv:.0f} px/s)")

    # Phase 2: Batch Dstk generation (Numba parallel)
    bc = _prepare_batch_constants(config)
    Dstk = np.empty((n_pixels, config.n_depth, n_elements), dtype=np.float64)

    _generate_dstk_batch(
        Dstk, FV_all, depth, n_free, bc['use_gaussian'],
        bc['fcl_elem_idx'], bc['fcl_from_fv'], bc['fcl_fv_base'],
        bc['fcl_linked_src'], bc['linked_pos'], bc['linked_width'],
        bc['linked_height'], bc['fcl_is_neutral'], bc['fcl_half_thresh'],
        bc['charges'], bc['subst_idx'], bc['antisubst'],
        bc['as_pos_mask'], bc['as_neg_mask'],
        use_analytical_cb,
    )

    t2 = time.perf_counter()
    if verbose:
        print(f"  Dstk batch: {t2-t1:.3f}s ({n_pixels/(t2-t1):.0f} px/s)")

    # Phase 3: Forward model (MLX Metal GPU or Numba CPU)
    angles_rad = np.deg2rad(angles).astype(np.float64)

    if use_mlx_forward and _HAS_MLX:
        Xstk = _batch_forward_mlx(
            Dstk,
            np.ascontiguousarray(lambda_basic),
            angles_rad,
            float(config.step_size),
            float(config.gamma_eff),
        )
        backend_label = "MLX Metal"
    else:
        Xstk = np.empty((n_pixels, config.n_angles, n_elements), dtype=np.float64)
        _batch_forward(
            Xstk, Dstk,
            np.ascontiguousarray(lambda_basic),
            angles_rad,
            float(config.step_size),
            float(config.gamma_eff),
        )
        backend_label = "Numba CPU"

    t3 = time.perf_counter()
    if verbose:
        print(f"  Xstk {backend_label}: {t3-t2:.3f}s ({n_pixels/(t3-t2):.0f} px/s)")

    return Dstk, Xstk, FV_all


def _generate_slow(config, depth, angles, n_pixels, n_elements, n_free,
                   charges, lambda_basic, tm, verbose):
    """Python fallback: per-pixel loop (original implementation)."""
    dist_func = calculate_distribution
    if getattr(config, 'dist_type', 'Error fcn.') == 'Gaussian':
        dist_func = calculate_distribution_gaussian

    base_fv = config.base_fv()
    tm.lambda_basic = lambda_basic
    tm.n_elements = n_elements

    n_fv = len(base_fv)
    Dstk = np.zeros((n_pixels, config.n_depth, n_elements))
    Xstk = np.zeros((n_pixels, config.n_angles, n_elements))
    FV_all = np.zeros((n_pixels, n_fv))

    t0 = time.perf_counter()
    for px in range(n_pixels):
        sk = px + 1
        FV = update_feature_vector(
            base_fv, sk, config.nx, config.ny, config.dt, config.flags)
        FV_all[px] = FV
        thickness, abruptness, free_profiles = fv_to_profiles(FV, n_free)
        D = dist_func(
            depth=depth, thickness=thickness, abruptness_fwhm=abruptness,
            free_profiles=free_profiles, linked_profiles=config.linked_profiles,
            charges=charges, subst_idx=config.subst_idx,
            free_idx=config.free_idx, linked_idx=config.linked_idx,
        )
        Dstk[px] = D
        tm.lambda_obs = tm.calculate_lambda_obs(D, lambda_basic)
        tm.trans_mat = tm.calculate_trans_mat(tm.lambda_obs)
        Xstk[px] = tm.forward(D, normalize=True)

        if verbose and (px % max(1, n_pixels // 10) == 0 or px == n_pixels - 1):
            elapsed = time.perf_counter() - t0
            rate = (px + 1) / elapsed if elapsed > 0 else 0
            print(f"  pixel {px+1}/{n_pixels} ({rate:.0f} px/s)")

    return Dstk, Xstk, FV_all


# ==============================================================================
# Phase 2: Noise injection
# ==============================================================================

def add_noise(
    Xstk: np.ndarray,
    noise_level: float,
    noise_type: str = 'poisson',
    seed: int = 42,
) -> np.ndarray:
    """Add noise to angle-resolved intensities.

    Args:
        Xstk: Clean intensities (n_px, n_angles, n_elem)
        noise_level: Noise level (0 = no noise)
        noise_type: 'poisson' or 'relative'
        seed: Random seed

    Returns:
        Noisy Xstk (same shape, clipped to [0, inf))
    """
    if noise_level <= 0:
        return Xstk.copy()

    rng = np.random.default_rng(seed)

    if noise_type == 'poisson':
        # Scale so that Poisson noise gives relative std ~ noise_level
        scale = 1.0 / noise_level**2
        Xstk_scaled = np.clip(Xstk * scale, 0, None)
        Xstk_noisy = rng.poisson(Xstk_scaled).astype(np.float64) / scale
    else:  # relative
        Xstk_noisy = Xstk * (1.0 + noise_level * rng.standard_normal(Xstk.shape))

    return np.clip(Xstk_noisy, 0, None)


# ==============================================================================
# Phase 3: Inverse reconstruction
# ==============================================================================

def inverse_reconstruct(
    Xstk_noisy: np.ndarray,
    config: D2S2DConfig,
    depth: np.ndarray,
    angles: np.ndarray,
    element_configs: list,
    charges: np.ndarray,
    verbose: bool = True,
    FV_all: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reconstruct depth profiles from noisy observations.

    Args:
        Xstk_noisy: Noisy intensities (n_px, n_angles, n_elem)
        config: D2S2D config
        depth, angles, element_configs, charges: Physics params
        verbose: Print progress
        FV_all: GT feature vectors (n_px, n_fv) for semifree init

    Returns:
        Drlt: Reconstructed profiles (n_px, n_depth, n_elem)
    """
    if config.inverse_method == 'nn':
        return _inverse_nn(Xstk_noisy, config, depth, verbose)
    elif config.inverse_method == 'nn_semifree':
        return _inverse_nn_semifree(
            Xstk_noisy, config, depth, angles, element_configs, charges,
            verbose, FV_all)
    elif config.inverse_method == 'semifree':
        return _inverse_semifree(
            Xstk_noisy, config, depth, angles, element_configs, charges,
            verbose, FV_all)
    elif config.inverse_method == 'mlx_free':
        return _inverse_mlx_free(
            Xstk_noisy, config, depth, angles, element_configs, verbose)
    elif config.inverse_method == 'mlx_semifree':
        return _inverse_mlx_semifree(
            Xstk_noisy, config, depth, angles, element_configs, charges,
            verbose, FV_all)
    else:
        return _inverse_solver(
            Xstk_noisy, config, depth, angles, element_configs, charges, verbose)


def _inverse_mlx_free(
    Xstk_noisy: np.ndarray,
    config: D2S2DConfig,
    depth: np.ndarray,
    angles: np.ndarray,
    element_configs: list,
    verbose: bool,
) -> np.ndarray:
    """MLX batched Free solver (Stage A): all pixels in one GPU batch.

    Mirrors _inverse_solver's setup (uniform D_init, one shared transfer
    matrix) but replaces the per-pixel SLSQP loop with solve_free_batch.
    """
    from deppro.inverse_solver import SolverConfig
    from deppro.solvers_mlx import MLXBatchFreeSolver

    n_px, _, n_elem = Xstk_noisy.shape
    n_depth = len(depth)

    D_init = np.ones((n_depth, n_elem)) / n_elem
    tm = TransferMatrix(
        depth=depth, step_size=config.sim_config.step_size, angles=angles,
        photon_energy=config.sim_config.photon_energy,
        work_function=config.sim_config.work_function,
        gamma_eff=config.sim_config.gamma_eff,
    )
    tm.compute(D_init, element_configs, CompoundDatabase())

    solver_config = SolverConfig(
        method=config.solver_method,
        alpha=config.solver_alpha,
        max_iter=config.solver_max_iter,
        n_loops=config.solver_loops,
    )
    solver = MLXBatchFreeSolver(tm, solver_config)

    if verbose:
        print(f"  MLX batch inverse: {n_px} px "
              f"(method={config.solver_method}, alpha={config.solver_alpha})")

    t0 = time.perf_counter()
    batch = solver.solve_batch(Xstk_noisy, D_init=D_init)
    if verbose:
        dt = time.perf_counter() - t0
        print(f"    done in {dt:.2f}s ({n_px / dt:.0f} px/s), "
              f"converged {int(batch.converged.sum())}/{n_px}")
    return batch.D_opt


def _inverse_mlx_semifree(
    Xstk_noisy: np.ndarray,
    config: D2S2DConfig,
    depth: np.ndarray,
    angles: np.ndarray,
    element_configs: list,
    charges: np.ndarray,
    verbose: bool,
    FV_all: Optional[np.ndarray] = None,
) -> np.ndarray:
    """MLX batched SemiFree solver (Stage C): all pixels in one GPU batch.

    Mirrors _inverse_semifree's per-pixel setup (same SolverConfig /
    SemiFreeSolverConfig, same thickness/abruptness init modes) but
    replaces the SLSQP loop with the block-coordinate batch solver.

    semifree_init='nn' additionally warm-starts from NN-predicted depth
    profiles (nn_model_path) via estimate_init_from_drlt.
    """
    from deppro.inverse_solver import (
        SemiFreeSolverConfig, SolverConfig,
    )
    from deppro.solvers_mlx import MLXBatchSemiFreeSolver

    n_px, n_angles, n_elem = Xstk_noisy.shape
    n_depth = len(depth)
    step_size = float(depth[1] - depth[0]) if len(depth) > 1 else 30.0
    sim = config.sim_config

    solver_config = SolverConfig(
        method=config.solver_method,
        alpha=config.solver_alpha,
        max_iter=config.solver_max_iter,
        n_loops=config.solver_loops,
        scipy_method='SLSQP',
        use_xreg_weighting=True,
        enable_fallback=True,
        fallback_method='trust-constr',
    )

    init_mode = config.semifree_init
    if init_mode == 'gt_fv' and FV_all is None:
        init_mode = 'base_fv'
        if verbose:
            print("  WARNING: FV_all not available, falling back to base_fv init")

    # thickness/abruptness init per pixel (same modes as _inverse_semifree)
    film_conc_init = None
    if init_mode == 'gt_fv':
        thickness_init = FV_all[:n_px, 0].astype(np.float64)
        abruptness_init = FV_all[:n_px, 1].astype(np.float64)
    elif init_mode == 'nn':
        # NN warm-start: predict profiles, extract (t, a, film_conc)
        Drlt_nn = _inverse_nn(Xstk_noisy, config, depth, verbose)
        flc_arr = np.maximum(np.asarray(sim.linked_const, dtype=np.float64),
                             1.0)
        est = estimate_init_from_drlt(
            Drlt_nn, depth, list(sim.subst_idx), list(sim.free_idx),
            flc_arr)
        thickness_init = est['thickness']
        abruptness_init = est['abruptness']
        film_conc_init = est['film_conc']
    elif init_mode == 'base_fv':
        thickness_init = np.full(n_px, sim.thickness)
        abruptness_init = np.full(n_px, sim.abruptness)
    else:  # 'default'
        thickness_init = np.full(n_px, 30.0)
        abruptness_init = np.full(n_px, 5.0)

    t_lo, t_hi = config.semifree_thickness_bounds
    thickness_init = np.clip(thickness_init, t_lo, t_hi)

    semi_cfg = SemiFreeSolverConfig(
        thickness_init=float(np.median(thickness_init)),
        abruptness_init=float(np.median(abruptness_init)),
        thickness_bounds=config.semifree_thickness_bounds,
        abruptness_bounds=config.semifree_abruptness_bounds,
        optimize_thickness=True,
        optimize_abruptness=config.semifree_optimize_abruptness,
        free_indices=list(sim.free_idx),
        linked_indices=list(sim.linked_idx),
        substrate_indices=list(sim.subst_idx),
        free_linked_const_override=list(sim.linked_const),
    )

    D_init = np.ones((n_depth, n_elem)) / n_elem
    tm = TransferMatrix(
        depth=depth, step_size=step_size, angles=angles,
        photon_energy=sim.photon_energy,
        work_function=sim.work_function,
        gamma_eff=sim.gamma_eff,
    )
    tm.compute(D_init, element_configs, CompoundDatabase())

    # NN warm-start: the NN gives a thickness estimate, so search a +-50 A
    # bracket around it in fewer rounds instead of the full bounds.
    mlx_kw = (dict(t_radius_init=50.0, outer_rounds=6)
              if init_mode == 'nn' else {})
    solver = MLXBatchSemiFreeSolver(
        tm, solver_config, semi_cfg, D_prior=None, elements=element_configs,
        **mlx_kw,
    )

    if verbose:
        print(f"  MLX batch SemiFree inverse: {n_px} px "
              f"(init={init_mode}, method={config.solver_method}, "
              f"alpha={config.solver_alpha})")

    t0 = time.perf_counter()
    batch = solver.solve_batch(
        Xstk_noisy,
        thickness_init=thickness_init,
        abruptness_init=abruptness_init,
        film_conc_init=film_conc_init,
    )
    if verbose:
        dt = time.perf_counter() - t0
        print(f"    done in {dt:.2f}s ({n_px / dt:.0f} px/s), "
              f"outer rounds {batch.n_outer}, "
              f"thk med={np.median(batch.thickness):.1f} A")
    return batch.D_opt


def _inverse_nn(
    Xstk_noisy: np.ndarray,
    config: D2S2DConfig,
    depth: np.ndarray,
    verbose: bool,
) -> np.ndarray:
    """NN-based inverse using DepthMLP."""
    try:
        from deppro.inference.model import DepthMLP
        from deppro.inference.postprocess import enforce_constraints
        import mlx.core as mx
    except ImportError as e:
        raise ImportError(f"NN inverse requires mlx: {e}")

    if config.nn_model_path is None:
        raise ValueError("nn_model_path required for NN inverse")

    n_px, n_angles, n_elem = Xstk_noisy.shape
    n_depth = len(depth)

    # Load model (auto-detect architecture from weights)
    weights = mx.load(config.nn_model_path)
    weight_dict = dict(weights) if not isinstance(weights, dict) else weights
    # Infer hidden_dim from first Linear layer, n_hidden from layer count
    hidden_dim = weight_dict['layers.0.weight'].shape[0]
    # Count Linear layers (weight shape is 2D for Linear, 1D for LayerNorm)
    n_linear = sum(1 for k, v in weight_dict.items()
                   if 'weight' in k and len(v.shape) == 2)
    n_hidden = n_linear - 1  # exclude final output layer

    model = DepthMLP(
        input_dim=n_angles * n_elem,
        output_dim=n_depth * n_elem,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
    )
    model.load_weights(list(weight_dict.items()))
    mx.eval(model.parameters())

    if verbose:
        print(f"  NN inverse: {config.nn_model_path}")
        print(f"    model: {n_angles*n_elem} -> {model.hidden_dim}x{model.n_hidden} -> {n_depth*n_elem}")
        print(f"    params: {model.n_parameters():,}")

    # Flatten input: (n_px, n_angles * n_elem)
    X_flat = Xstk_noisy.reshape(n_px, -1)
    X_mx = mx.array(X_flat.astype(np.float32))

    # Batch predict
    t0 = time.perf_counter()
    D_mx = model(X_mx)
    mx.eval(D_mx)
    D_flat = np.array(D_mx)
    t_infer = time.perf_counter() - t0

    # Post-processing: enforce sum-to-1 normalization at each depth
    D_flat = enforce_constraints(D_flat, n_depth, n_elem)
    Drlt = D_flat.reshape(n_px, n_depth, n_elem)

    if verbose:
        throughput = n_px / t_infer if t_infer > 0 else float('inf')
        print(f"    inference: {t_infer:.4f}s ({throughput:.0f} px/s)")

    return Drlt


def _inverse_solver(
    Xstk_noisy: np.ndarray,
    config: D2S2DConfig,
    depth: np.ndarray,
    angles: np.ndarray,
    element_configs: list,
    charges: np.ndarray,
    verbose: bool,
) -> np.ndarray:
    """Solver-based inverse using FreeSolver or SemiFreeSolver."""
    from deppro.inverse_solver import FreeSolver, SolverConfig

    n_px, n_angles, n_elem = Xstk_noisy.shape
    n_depth = len(depth)
    step_size = float(depth[1] - depth[0]) if len(depth) > 1 else 30.0

    solver_config = SolverConfig(
        method=config.solver_method,
        alpha=config.solver_alpha,
        max_iter=config.solver_max_iter,
        n_loops=config.solver_loops,
    )

    compound_db = CompoundDatabase()

    Drlt = np.zeros((n_px, n_depth, n_elem))

    if verbose:
        print(f"  Solver inverse: {config.inverse_method} "
              f"(method={config.solver_method}, alpha={config.solver_alpha})")

    t0 = time.perf_counter()

    for px in range(n_px):
        I_obs = Xstk_noisy[px]  # (n_angles, n_elem)

        # Build TransferMatrix for this pixel
        # Use uniform D_init for initial transfer matrix
        D_init = np.ones((n_depth, n_elem)) / n_elem
        tm = TransferMatrix(
            depth=depth, step_size=step_size, angles=angles,
            photon_energy=config.sim_config.photon_energy,
            work_function=config.sim_config.work_function,
            gamma_eff=config.sim_config.gamma_eff,
        )
        tm.compute(D_init, element_configs, compound_db)

        solver = FreeSolver(tm, solver_config)
        result = solver.solve(I_obs, D_init=D_init, verbose=False)

        # Always use solver result (even if not converged, it's better than D_init)
        Drlt[px] = result.D_opt

        if verbose and (px % max(1, n_px // 10) == 0 or px == n_px - 1):
            elapsed = time.perf_counter() - t0
            rate = (px + 1) / elapsed if elapsed > 0 else 0
            print(f"    pixel {px+1}/{n_px} ({rate:.1f} px/s)"
                  f" success={result.success}")

    return Drlt


def _inverse_semifree(
    Xstk_noisy: np.ndarray,
    config: D2S2DConfig,
    depth: np.ndarray,
    angles: np.ndarray,
    element_configs: list,
    charges: np.ndarray,
    verbose: bool,
    FV_all: Optional[np.ndarray] = None,
) -> np.ndarray:
    """SemiFreeSolver-based inverse with GT FV initialization."""
    from deppro.inverse_solver import (
        SemiFreeSolver, SemiFreeSolverConfig, SolverConfig,
    )

    n_px, n_angles, n_elem = Xstk_noisy.shape
    n_depth = len(depth)
    step_size = float(depth[1] - depth[0]) if len(depth) > 1 else 30.0
    sim = config.sim_config

    solver_config = SolverConfig(
        method=config.solver_method,
        alpha=config.solver_alpha,
        max_iter=config.solver_max_iter,
        n_loops=config.solver_loops,
        scipy_method='SLSQP',
        use_xreg_weighting=True,
        enable_fallback=True,
        fallback_method='trust-constr',
    )

    compound_db = CompoundDatabase()
    Drlt = np.zeros((n_px, n_depth, n_elem))

    init_mode = config.semifree_init
    if init_mode == 'gt_fv' and FV_all is None:
        init_mode = 'base_fv'
        if verbose:
            print("  WARNING: FV_all not available, falling back to base_fv init")

    if verbose:
        print(f"  SemiFreeSolver: init={init_mode}, "
              f"method={config.solver_method}, alpha={config.solver_alpha}, "
              f"loops={config.solver_loops}")

    t0 = time.perf_counter()

    for px in range(n_px):
        I_obs = Xstk_noisy[px]  # (n_angles, n_elem)

        # Determine thickness/abruptness init for this pixel
        if init_mode == 'gt_fv':
            thickness_init = float(FV_all[px, 0])
            abruptness_init = float(FV_all[px, 1])
        elif init_mode == 'base_fv':
            thickness_init = sim.thickness
            abruptness_init = sim.abruptness
        else:  # 'default'
            thickness_init = 30.0
            abruptness_init = 5.0

        # Clamp thickness to bounds
        t_lo, t_hi = config.semifree_thickness_bounds
        thickness_init = max(t_lo, min(t_hi, thickness_init))

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=thickness_init,
            abruptness_init=abruptness_init,
            thickness_bounds=config.semifree_thickness_bounds,
            abruptness_bounds=config.semifree_abruptness_bounds,
            optimize_thickness=True,
            optimize_abruptness=config.semifree_optimize_abruptness,
            free_indices=list(sim.free_idx),
            linked_indices=list(sim.linked_idx),
            substrate_indices=list(sim.subst_idx),
            free_linked_const_override=list(sim.linked_const),
        )

        # Build TransferMatrix
        D_init = np.ones((n_depth, n_elem)) / n_elem
        tm = TransferMatrix(
            depth=depth, step_size=step_size, angles=angles,
            photon_energy=sim.photon_energy,
            work_function=sim.work_function,
            gamma_eff=sim.gamma_eff,
        )
        tm.compute(D_init, element_configs, compound_db)

        solver = SemiFreeSolver(
            tm, solver_config, semi_cfg,
            D_prior=None,
            elements=element_configs,
        )
        result = solver.solve(I_obs, verbose=False)
        Drlt[px] = result.D_opt

        if verbose and (px % max(1, n_px // 10) == 0 or px == n_px - 1):
            elapsed = time.perf_counter() - t0
            rate = (px + 1) / elapsed if elapsed > 0 else 0
            print(f"    pixel {px+1}/{n_px} ({rate:.1f} px/s)"
                  f" ok={result.success}"
                  f" thk={result.thickness:.1f} abrp={result.abruptness:.1f}")

    return Drlt


# ==============================================================================
# Phase 3b: Hybrid NN → SemiFree inverse
# ==============================================================================

def estimate_init_from_drlt(
    Drlt: np.ndarray,
    depth: np.ndarray,
    sub_idx: list,
    free_idx: list,
    free_linked_const: np.ndarray,
    min_conc: float = 1e-6,
) -> dict:
    """Estimate SemiFreeSolver init params from NN-predicted depth profiles.

    For each pixel, extracts thickness + abruptness by fitting erf to the
    substrate profile, then computes normalized free-element concentrations.

    Parameters
    ----------
    Drlt : (n_px, n_depth, n_elem)
        NN-predicted depth profiles (sum-to-1 at each depth).
    depth : (n_depth,)
        Depth grid in Angstrom.
    sub_idx : list of int
        Substrate element indices.
    free_idx : list of int
        Free element indices.
    free_linked_const : (n_free,)
        Charge balance coefficients.
    min_conc : float
        Minimum concentration clamp.

    Returns
    -------
    dict with:
        thickness : (n_px,)
        abruptness : (n_px,)
        film_conc : (n_px, n_depth, n_free)  — normalized free concentrations
    """
    n_px, n_depth, n_elem = Drlt.shape
    n_free = len(free_idx)

    thickness_arr = np.empty(n_px)
    abruptness_arr = np.empty(n_px)
    film_conc_arr = np.empty((n_px, n_depth, n_free))

    for px in range(n_px):
        D = Drlt[px]  # (n_depth, n_elem)

        # --- Substrate sum profile ---
        sub_sum = D[:, sub_idx].sum(axis=1)  # (n_depth,)

        # Estimate thickness: depth where substrate crosses 0.5
        thickness_est = _estimate_thickness(depth, sub_sum)

        # Estimate abruptness: fit erf, extract FWHM
        abruptness_est = _estimate_abruptness(depth, sub_sum, thickness_est)

        thickness_arr[px] = thickness_est
        abruptness_arr[px] = abruptness_est

        # --- Film concentrations (normalized) ---
        film_scale = np.maximum(1.0 - sub_sum, min_conc)  # (n_depth,)
        D_free = D[:, free_idx]  # (n_depth, n_free)
        # Normalized: what SemiFreeSolver optimizes
        film_conc = D_free / film_scale[:, np.newaxis]
        # Clip to valid bounds
        for i, flc in enumerate(free_linked_const):
            if flc > 0:
                film_conc[:, i] = np.clip(film_conc[:, i], min_conc, 1.0 / flc)
        film_conc_arr[px] = film_conc

    return {
        'thickness': thickness_arr,
        'abruptness': abruptness_arr,
        'film_conc': film_conc_arr,
    }


def _estimate_thickness(depth: np.ndarray, sub_sum: np.ndarray) -> float:
    """Estimate thickness from substrate sum via 0.5 crossing (linear interp)."""
    n = len(depth)
    # Find crossing of 0.5
    for i in range(n - 1):
        if sub_sum[i] <= 0.5 <= sub_sum[i + 1]:
            # Linear interpolation
            frac = (0.5 - sub_sum[i]) / max(sub_sum[i + 1] - sub_sum[i], 1e-12)
            return depth[i] + frac * (depth[i + 1] - depth[i])
    # Fallback: if substrate never crosses 0.5, use weighted mean
    if sub_sum.max() > 0.1:
        # Use gradient peak (steepest part of the erf)
        grad = np.abs(np.gradient(sub_sum, depth))
        return float(depth[np.argmax(grad)])
    return float(depth[n // 2])  # Last resort: midpoint


def _estimate_abruptness(
    depth: np.ndarray, sub_sum: np.ndarray, thickness: float,
) -> float:
    """Estimate abruptness (FWHM) by fitting erf to substrate profile.

    Uses scalar optimization (Brent) on MSE between observed substrate
    and erf model: 0.5 * (1 + erf((z - thickness) / sigma)).
    """
    def _erf_mse(log_sigma):
        sigma = np.exp(log_sigma)
        model = 0.5 * (1.0 + erf((depth - thickness) / sigma))
        return np.mean((sub_sum - model) ** 2)

    # Search sigma in [1, 200] Angstrom (log scale)
    result = minimize_scalar(_erf_mse, bounds=(np.log(1.0), np.log(200.0)),
                             method='bounded')
    sigma_opt = np.exp(result.x)
    # Convert sigma to FWHM
    abruptness = sigma_opt * 2.0 * np.sqrt(np.log(2.0))
    return float(np.clip(abruptness, 1.0, 100.0))


def _inverse_nn_semifree(
    Xstk_noisy: np.ndarray,
    config: D2S2DConfig,
    depth: np.ndarray,
    angles: np.ndarray,
    element_configs: list,
    charges: np.ndarray,
    verbose: bool,
    FV_all: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Hybrid NN→SemiFree inverse: NN provides initial guess, SemiFree refines.

    Stage 1: NN fast inference → Drlt_init
    Stage 2: Extract thickness/abruptness/film_conc from Drlt_init
    Stage 3: SemiFree optimization with NN-derived initial guess
    """
    from deppro.inverse_solver import (
        SemiFreeSolver, SemiFreeSolverConfig, SolverConfig,
    )

    sim = config.sim_config
    n_px, n_angles, n_elem = Xstk_noisy.shape
    n_depth = len(depth)
    step_size = float(depth[1] - depth[0]) if len(depth) > 1 else 30.0
    n_free = len(sim.free_idx)

    # --- Stage 1: NN inference ---
    if verbose:
        print("  Stage 1: NN inference")
    t0 = time.perf_counter()
    Drlt_nn = _inverse_nn(Xstk_noisy, config, depth, verbose)
    t_nn = time.perf_counter() - t0

    # --- Stage 2: Extract init params from NN output ---
    if verbose:
        print("  Stage 2: Extracting init params from NN output")
    t1 = time.perf_counter()
    linked_const = np.array(sim.linked_const, dtype=np.float64)
    linked_const = np.maximum(linked_const, 1.0)
    init_params = estimate_init_from_drlt(
        Drlt_nn, depth, sim.subst_idx, sim.free_idx, linked_const)
    t_extract = time.perf_counter() - t1

    if verbose:
        thk_mean = init_params['thickness'].mean()
        thk_std = init_params['thickness'].std()
        abrp_mean = init_params['abruptness'].mean()
        print(f"    thickness: {thk_mean:.1f} ± {thk_std:.1f} Å")
        print(f"    abruptness: {abrp_mean:.1f} Å (mean)")
        print(f"    extract time: {t_extract:.3f}s")

    # --- Stage 3: SemiFree refinement ---
    if verbose:
        print(f"  Stage 3: SemiFree refinement ({n_px} pixels)")

    solver_config = SolverConfig(
        method=config.solver_method,
        alpha=config.solver_alpha,
        max_iter=config.solver_max_iter,
        n_loops=config.solver_loops,
        scipy_method='SLSQP',
        use_xreg_weighting=True,
        enable_fallback=True,
        fallback_method='trust-constr',
    )

    compound_db = CompoundDatabase()
    Drlt = np.zeros((n_px, n_depth, n_elem))

    t2 = time.perf_counter()

    for px in range(n_px):
        I_obs = Xstk_noisy[px]

        thickness_init = float(init_params['thickness'][px])
        abruptness_init = float(init_params['abruptness'][px])

        # Clamp thickness to bounds
        t_lo, t_hi = config.semifree_thickness_bounds
        thickness_init = max(t_lo, min(t_hi, thickness_init))

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=thickness_init,
            abruptness_init=abruptness_init,
            thickness_bounds=config.semifree_thickness_bounds,
            abruptness_bounds=config.semifree_abruptness_bounds,
            optimize_thickness=True,
            optimize_abruptness=config.semifree_optimize_abruptness,
            free_indices=list(sim.free_idx),
            linked_indices=list(sim.linked_idx),
            substrate_indices=list(sim.subst_idx),
            free_linked_const_override=list(sim.linked_const),
        )

        # Build TransferMatrix
        D_init = np.ones((n_depth, n_elem)) / n_elem
        tm = TransferMatrix(
            depth=depth, step_size=step_size, angles=angles,
            photon_energy=sim.photon_energy,
            work_function=sim.work_function,
            gamma_eff=sim.gamma_eff,
        )
        tm.compute(D_init, element_configs, compound_db)

        solver = SemiFreeSolver(
            tm, solver_config, semi_cfg,
            D_prior=None,
            elements=element_configs,
        )

        # Build x0 from NN-derived film_conc + thickness + abruptness
        film_conc = init_params['film_conc'][px]  # (n_depth, n_free)
        x0 = solver.build_x0(
            film_conc=film_conc,
            thickness=thickness_init,
            abruptness=abruptness_init,
        )

        result = solver.solve(I_obs, verbose=False, x0_override=x0)

        # Per-pixel best-of selection: keep NN if SemiFree is worse
        # Compare forward-model residuals (no GT needed)
        tm.lambda_obs = tm.calculate_lambda_obs(result.D_opt, tm.lambda_basic)
        tm.trans_mat = tm.calculate_trans_mat(tm.lambda_obs)
        I_semi = tm.forward(result.D_opt, normalize=True)
        loss_semi = np.sum((I_obs - I_semi) ** 2)

        tm.lambda_obs = tm.calculate_lambda_obs(Drlt_nn[px], tm.lambda_basic)
        tm.trans_mat = tm.calculate_trans_mat(tm.lambda_obs)
        I_nn = tm.forward(Drlt_nn[px], normalize=True)
        loss_nn = np.sum((I_obs - I_nn) ** 2)

        if loss_semi <= loss_nn:
            Drlt[px] = result.D_opt
            chose_semi = True
        else:
            Drlt[px] = Drlt_nn[px]
            chose_semi = False

        if verbose and (px % max(1, n_px // 10) == 0 or px == n_px - 1):
            elapsed = time.perf_counter() - t2
            rate = (px + 1) / elapsed if elapsed > 0 else 0
            sel = "SF" if chose_semi else "NN"
            print(f"    pixel {px+1}/{n_px} ({rate:.1f} px/s)"
                  f" ok={result.success}"
                  f" thk={result.thickness:.1f} abrp={result.abruptness:.1f}"
                  f" sel={sel}")

    t_semi = time.perf_counter() - t2

    # Count selection stats
    # (re-compute quickly — not in the hot loop for clarity)
    n_kept_nn = 0
    n_kept_semi = 0
    for px in range(n_px):
        if np.allclose(Drlt[px], Drlt_nn[px], atol=1e-10):
            n_kept_nn += 1
        else:
            n_kept_semi += 1

    if verbose:
        total = time.perf_counter() - t0
        print(f"  Selection: {n_kept_semi} SemiFree, {n_kept_nn} NN-kept "
              f"({100*n_kept_nn/n_px:.0f}% NN)")
        print(f"  Timing: NN={t_nn:.3f}s, extract={t_extract:.3f}s, "
              f"SemiFree={t_semi:.1f}s, total={total:.1f}s")

    return Drlt


# ==============================================================================
# Phase 4: Evaluation
# ==============================================================================

def evaluate_reconstruction(
    Dstk: np.ndarray,
    Drlt: np.ndarray,
    depth: np.ndarray,
    element_names: list,
    verbose: bool = True,
) -> dict:
    """Evaluate reconstruction quality.

    Args:
        Dstk: Ground truth (n_px, n_depth, n_elem)
        Drlt: Reconstruction (n_px, n_depth, n_elem)
        depth: Depth array
        element_names: Element labels
        verbose: Print results

    Returns:
        dict with rmse_per_element, rmse_total, w1_per_element, w1_mean
    """
    n_px, n_depth, n_elem = Dstk.shape

    # Per-element RMSE (across all pixels and depth points)
    rmse_per_element = np.zeros(n_elem)
    for e in range(n_elem):
        diff = Dstk[:, :, e] - Drlt[:, :, e]
        rmse_per_element[e] = np.sqrt(np.mean(diff**2))

    rmse_total = np.sqrt(np.mean((Dstk - Drlt)**2))

    # Per-element W1 (averaged over pixels)
    w1_per_element = np.zeros(n_elem)
    for e in range(n_elem):
        w1_values = []
        for px in range(n_px):
            w1 = wasserstein_distance_1d(depth, Dstk[px, :, e], Drlt[px, :, e])
            if not np.isnan(w1):
                w1_values.append(w1)
        if w1_values:
            w1_per_element[e] = np.mean(w1_values)
        else:
            w1_per_element[e] = np.nan

    w1_mean = np.nanmean(w1_per_element)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  RMSE total: {rmse_total:.4f} ({rmse_total*100:.2f}%)")
        print(f"  W1 mean:    {w1_mean:.1f} A")
        print(f"  {'Element':<12s} {'RMSE':>8s} {'W1(A)':>8s}")
        print(f"  {'-'*28}")
        for e in range(n_elem):
            name = element_names[e] if e < len(element_names) else f"E{e}"
            w1_str = f"{w1_per_element[e]:.1f}" if not np.isnan(w1_per_element[e]) else "N/A"
            print(f"  {name:<12s} {rmse_per_element[e]:>8.4f} {w1_str:>8s}")
        print(f"{'='*60}")

    return {
        "rmse_per_element": rmse_per_element,
        "rmse_total": rmse_total,
        "w1_per_element": w1_per_element,
        "w1_mean": w1_mean,
    }


# ==============================================================================
# Main orchestrator
# ==============================================================================

def run_d2s2d(config: D2S2DConfig, verbose: bool = True) -> D2S2DResult:
    """Run complete D2S2D pipeline.

    Args:
        config: Pipeline configuration
        verbose: Print progress

    Returns:
        D2S2DResult with all arrays and metrics
    """
    t_total = time.perf_counter()

    if verbose:
        print("=" * 60)
        print("D2S2D Pipeline")
        print("=" * 60)

    # Phase 1: Generate Dstk + Xstk
    sim_result = generate_dstk_xstk(
        config.sim_config, verbose=verbose,
        charge_balance=config.charge_balance,
        forward_backend=config.forward_backend)
    Dstk = sim_result["Dstk"]
    Xstk = sim_result["Xstk"]
    FV_all = sim_result.get("FV_all")
    depth = sim_result["depth"]
    angles = sim_result["angles"]
    element_configs = sim_result["element_configs"]
    charges = sim_result["charges"]
    elapsed_gen = sim_result["elapsed"]

    element_names = [
        f"{e['symbol']}({'+' if e['charge']>0 else ''}{e['charge']})"
        for e in config.sim_config.elements
    ]

    # Phase 2: Noise
    if verbose and config.noise_level > 0:
        print(f"\nNoise: {config.noise_type} level={config.noise_level}")
    Xstk_noisy = add_noise(
        Xstk, config.noise_level, config.noise_type, config.noise_seed)

    # Phase 3: Inverse
    if verbose:
        print(f"\nInverse reconstruction ({config.inverse_method}):")
    t_inv = time.perf_counter()
    Drlt = inverse_reconstruct(
        Xstk_noisy, config, depth, angles, element_configs, charges, verbose,
        FV_all=FV_all)
    elapsed_inv = time.perf_counter() - t_inv

    # Phase 4: Evaluate
    if verbose:
        print(f"\nEvaluation:")
    metrics = evaluate_reconstruction(
        Dstk, Drlt, depth, element_names, verbose=verbose)

    elapsed_total = time.perf_counter() - t_total

    if verbose:
        print(f"\nTiming: generate={elapsed_gen:.1f}s, inverse={elapsed_inv:.1f}s, "
              f"total={elapsed_total:.1f}s")

    return D2S2DResult(
        Dstk=Dstk,
        Xstk=Xstk,
        Xstk_noisy=Xstk_noisy,
        Drlt=Drlt,
        rmse_per_element=metrics["rmse_per_element"],
        rmse_total=metrics["rmse_total"],
        w1_per_element=metrics["w1_per_element"],
        w1_mean=metrics["w1_mean"],
        depth=depth,
        angles=angles,
        element_names=element_names,
        config=config,
        elapsed_generate=elapsed_gen,
        elapsed_inverse=elapsed_inv,
        elapsed_total=elapsed_total,
        FV_all=FV_all,
    )


# ==============================================================================
# Preset configs
# ==============================================================================

PRESETS = {
    '7elem_checkered': PRESET_7ELEM_CHECKERED,
    '6elem_standard': PRESET_6ELEM_STANDARD,
}


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="D2S2D Pipeline")
    parser.add_argument("--preset", type=str, default="7elem_checkered",
                        choices=list(PRESETS.keys()),
                        help="Simulation preset")
    parser.add_argument("--nx", type=int, default=None, help="Override X pixels")
    parser.add_argument("--ny", type=int, default=None, help="Override Y pixels")
    parser.add_argument("--dt", type=int, default=None, help="Override time steps")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level")
    parser.add_argument("--noise-type", type=str, default="poisson",
                        choices=["poisson", "relative"])
    parser.add_argument("--charge-balance", type=str, default="analytical",
                        choices=["analytical", "iterative"],
                        help="Charge balance algorithm")
    parser.add_argument("--forward-backend", type=str, default="auto",
                        choices=["auto", "mlx", "numba"],
                        help="Forward model backend (auto=MLX if available)")
    parser.add_argument("--inverse", type=str, default="free",
                        choices=["nn", "free", "semifree", "nn_semifree",
                                 "mlx_free", "mlx_semifree"])
    parser.add_argument("--model", type=str, default=None, help="NN model path")
    parser.add_argument("--solver-method", type=str, default="L1")
    parser.add_argument("--solver-alpha", type=float, default=0.01)
    parser.add_argument("--solver-loops", type=int, default=3)
    parser.add_argument("--semifree-init", type=str, default="gt_fv",
                        choices=["gt_fv", "base_fv", "default", "nn"],
                        help="SemiFreeSolver init mode ('nn' = NN warm-start"
                             " via --model, mlx_semifree only)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save result to .npz")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Build SimulationConfig from preset + overrides
    from dataclasses import replace
    sim_config = PRESETS[args.preset]
    if args.nx is not None:
        sim_config = replace(sim_config, nx=args.nx)
    if args.ny is not None:
        sim_config = replace(sim_config, ny=args.ny)
    if args.dt is not None:
        sim_config = replace(sim_config, dt=args.dt)

    config = D2S2DConfig(
        sim_config=sim_config,
        charge_balance=args.charge_balance,
        forward_backend=args.forward_backend,
        noise_level=args.noise,
        noise_type=args.noise_type,
        inverse_method=args.inverse,
        nn_model_path=args.model,
        solver_method=args.solver_method,
        solver_alpha=args.solver_alpha,
        solver_loops=args.solver_loops,
        semifree_init=args.semifree_init,
    )

    result = run_d2s2d(config, verbose=not args.quiet)

    if args.output:
        np.savez_compressed(
            args.output,
            Dstk=result.Dstk,
            Xstk=result.Xstk,
            Xstk_noisy=result.Xstk_noisy,
            Drlt=result.Drlt,
            depth=result.depth,
            angles=result.angles,
            rmse_per_element=result.rmse_per_element,
            rmse_total=result.rmse_total,
            w1_per_element=result.w1_per_element,
            w1_mean=result.w1_mean,
        )
        print(f"Saved to: {args.output}")

    return result


if __name__ == "__main__":
    main()
