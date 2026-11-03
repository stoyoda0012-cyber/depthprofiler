"""P1 contract: reimplementation of the metrics used in the existing benchmarks (RMSE, W1, thickness) without changing
their definitions, and the observation residual, **recorded only**.

The main metrics are the overall RMSE (mean over all pixels, depths and elements) and the
canonical W1 (``depth_profile.wasserstein_distance_1d`` vectorized over (px, elem); equivalence is pinned by a test).
Thickness is the linear interpolation of the substrate 0.5 crossing (same rule as ``deppro.io.dstk.estimate_thickness``).
NaN is null in JSON, with the valid count alongside. No third definition is introduced.

Observation residual (re-forward). The generation formula declared by the producer
(``CalculateDistribution.m`` / ``TransformationMatrix.m``):

    λ(z,i) = γ_eff / Σ_a D(z,a)/λ_basic(a,i)      (harmonic mean of the local composition; T differs per pixel)
    T(θ,z,i) = cumprod_z(exp(−Δz/(λ(z,i) cosθ))) / exp(·)   (only the layers above z)
    I(θ,i) = Σ_z T(θ,z,i)·D(z,i); the noise is on I
    X(θ,i) = I(θ,i) / Σ_j I(θ,j)                  (normalized over elements per angle)

``composition_transfer_matrix`` + ``forward`` are this formula as is.
The file's ``TransMatObs`` is **the T of a single stack** (at save time), so reusing it for every pixel
leaves a composition-dependent residual. That comes from the wrong T, not from a model floor.
**Pass/fail** is assigned by ``closure_gate``: PASS/FAIL only for datasets with noise = 0 (the tolerance is an
absolute value with margin over float32 storage precision); with noise > 0 it is RECORDED (the noise model
version and effective value await the producer contract). The GT must not be normalized to sum 1 (``Dstk`` is compared as stored).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence

import numpy as np
from ..._compat import trapezoid

from .dataset_files import jsonable
from .problem import GroundTruth, ProblemSpec

METRIC_VERSION = "deppro-contract-metrics/1"
REFORWARD_FORMULA_VERSION = (
    "producer-declared/1 (2026-09-08): lambda(z,i)=gamma/sum_a D(z,a)/lambda_basic(a,i); "
    "T=cumprod_z(exp(-dz/(lambda cos theta)))/exp(.); I=sum_z T*D; noise on I; "
    "X=I/sum_j I per angle. "
    "Stored TransMatObs is a single stack's T, so a shared-T residual is expected.")
# Closure-gate tolerance for noise = 0. About 10x the float32 storage precision (eps32·max X ≈ 9e-8),
# and about 3x the largest residual observed on noise-free generated data, 3e-7.
CLOSURE_TOL_ABS_NOISEFREE = 1e-6
GT_EMPTY_THRESHOLD = 1e-6          # same as depth_profile.wasserstein_distance_1d


# --------------------------------------------------------------------------
def w1_matrix(D: np.ndarray, Dgt: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Canonical W1 [Å] per (px, elem). NaN where the GT mass is below the threshold (deppro convention).

    Agrees with ``wasserstein_distance_1d(depth, gt, est)`` applied to each (px, elem)
    (pinned by a test).
    """
    D = np.asarray(D, dtype=np.float64)
    Dgt = np.asarray(Dgt, dtype=np.float64)
    dz = np.gradient(depth)
    ig = trapezoid(Dgt, depth, axis=1)                          # (px, elem)
    ir = trapezoid(D, depth, axis=1)
    cg = np.cumsum(Dgt / np.maximum(ig, 1e-10)[:, None, :], axis=1) * dz[None, :, None]
    cr = np.cumsum(D / np.maximum(ir, 1e-10)[:, None, :], axis=1) * dz[None, :, None]
    cg = cg / np.maximum(cg[:, -1:, :], 1e-10)
    cr = cr / np.maximum(cr[:, -1:, :], 1e-10)
    w1 = trapezoid(np.abs(cg - cr), depth, axis=1)
    w1[trapezoid(np.abs(Dgt), depth, axis=1) < GT_EMPTY_THRESHOLD] = np.nan
    return w1


def thickness_half_crossing(D_px: np.ndarray, depth: np.ndarray, substrate_idx: int) -> float:
    """Linear interpolation of the first position where the substrate concentration exceeds 0.5 (same rule as deppro.io.dstk.estimate_thickness)."""
    D_sub = np.asarray(D_px)[:, substrate_idx]
    crossings = np.where(D_sub >= 0.5)[0]
    if len(crossings) == 0:
        return float(depth[-1])
    idx = int(crossings[0])
    if idx > 0:
        z0, z1 = depth[idx - 1], depth[idx]
        d0, d1 = D_sub[idx - 1], D_sub[idx]
        denom = d1 - d0
        if abs(denom) < 1e-12:
            return float((z0 + z1) / 2)
        return float(z0 + (0.5 - d0) / denom * (z1 - z0))
    return float(depth[0])


def thickness_array(D: np.ndarray, depth: np.ndarray, substrate_idx: int) -> np.ndarray:
    return np.array([thickness_half_crossing(D[q], depth, substrate_idx) for q in range(D.shape[0])])


# --------------------------------------------------------------------------
def composition_transfer_matrix(D: np.ndarray, lambda_basic: np.ndarray, angles_deg: np.ndarray,
                                step: float, gamma_eff: float) -> np.ndarray:
    """Composition-dependent transfer matrix T(D). Same formula as the uniform-grid kernel of ``deppro.transfer_matrix``.

    D: (n, depth, elem) → T: (n, angle, depth, elem)。
    λ_obs(z,e) = γ / Σ_i D(z,i)/λ_basic(e,i) (Σ=0 gives 1e-10),
    T(a,z,e) = exp(Σ_{z'<z} −step / (λ_obs(z',e) cos a)) = exp(cumsum(t1) − t1)。
    """
    D = np.asarray(D, dtype=np.float64)
    inv = np.einsum("nzi,ei->nze", D, 1.0 / np.asarray(lambda_basic, dtype=np.float64))
    inv = np.where(inv == 0.0, 1e-10, inv)
    lam = gamma_eff / inv                                          # (n, Z, E)
    cos = np.cos(np.deg2rad(np.asarray(angles_deg, dtype=np.float64)))
    t1 = -step / lam[:, None, :, :] / cos[None, :, None, None]    # (n, A, Z, E)
    return np.exp(np.cumsum(t1, axis=2) - t1)


def forward(T: np.ndarray, D: np.ndarray) -> np.ndarray:
    """X̂ = normalize_angle(Σ_z T·D). T is shared (A,Z,E) or per pixel (n,A,Z,E). Provisional formula (see docstring)."""
    D = np.asarray(D, dtype=np.float64)
    if T.ndim == 3:
        Xp = np.einsum("aze,nze->nae", T, D)
    else:
        Xp = np.einsum("naze,nze->nae", T, D)
    s = Xp.sum(axis=2, keepdims=True)
    return np.divide(Xp, s, out=np.zeros_like(Xp), where=s > 0)


# --------------------------------------------------------------------------
def _nan_stats(w1: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN slice → NaN is intended
        return np.nanmean(w1, axis=0), np.nanmedian(w1, axis=0), np.sum(~np.isnan(w1), axis=0)


def _metrics_block(D: np.ndarray, Dgt: np.ndarray, depth: np.ndarray, labels: Sequence[str],
                   free: Sequence[int], substrate_idx: int) -> Dict[str, Any]:
    diff = D - Dgt
    rmse_e = np.sqrt(np.mean(diff ** 2, axis=(0, 1)))
    w1 = w1_matrix(D, Dgt, depth)
    w1_mean, w1_med, w1_valid = _nan_stats(w1)
    thk = np.abs(thickness_array(D, depth, substrate_idx) - thickness_array(Dgt, depth, substrate_idx))
    free = list(free)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {
            "n_pixels": int(D.shape[0]),
            "rmse_total": float(np.sqrt(np.mean(diff ** 2))),
            "rmse_free_mean": float(rmse_e[free].mean()) if free else None,
            "rmse_per_element": dict(zip(labels, map(float, rmse_e))),
            "w1_meanpx_per_element_A": dict(zip(labels, map(float, w1_mean))),
            "w1_medianpx_per_element_A": dict(zip(labels, map(float, w1_med))),
            "w1_valid_px_per_element": dict(zip(labels, map(int, w1_valid))),
            "w1_meanpx_all_A": float(np.nanmean(w1_mean)),
            "w1_meanpx_free_A": float(np.nanmean(w1_mean[free])) if free else None,
            "w1_medianpx_all_A": float(np.nanmean(w1_med)),      # per-element pixel median, averaged over elements
            "w1_medianpx_free_A": float(np.nanmean(w1_med[free])) if free else None,
            "thickness_mae_A": float(thk.mean()),
            "thickness_err_median_A": float(np.median(thk)),
        }
    if len(free) == 5:                                           # also under the key names of the autumn table
        out["rmse_free5_mean"] = out["rmse_free_mean"]
        out["w1_meanpx_free5_A"] = out["w1_meanpx_free_A"]
        out["w1_medianpx_free5_A"] = out["w1_medianpx_free_A"]
    return out


def evaluate(D_est: np.ndarray, gt: GroundTruth, spec: ProblemSpec, *,
             free: Optional[Sequence[int]] = None, substrate_idx: Optional[int] = None,
             per_frame: bool = False) -> Dict[str, Any]:
    """Compute the existing metrics as they are. The GT is used as stored (not renormalized). NaN is null in JSON."""
    D_est = np.asarray(D_est, dtype=np.float64)
    if D_est.shape != gt.D.shape:
        raise ValueError(f"D_est {D_est.shape} vs GT {gt.D.shape}")
    if not np.array_equal(np.asarray(gt.pixel_ids), np.asarray(spec.pixel_ids)):
        raise ValueError("GroundTruth and ProblemSpec pixel_ids differ")
    free = list(spec.elements.free if free is None else free)
    sub = spec.elements.subst[0] if substrate_idx is None else int(substrate_idx)
    if substrate_idx is None and not spec.elements.subst:
        raise ValueError("substrate index unknown: pass substrate_idx")
    labels = spec.elements.labels
    depth = spec.axes.depth_A
    out = {"metric_version": METRIC_VERSION, "gt_source": gt.source,
           "gt_renormalised": False, "free_indices": free, "substrate_index": sub}
    out.update(_metrics_block(D_est, gt.D, depth, labels, free, sub))
    if per_frame:
        frames = spec.frame_of()
        out["per_frame"] = {}
        for fr in np.unique(frames):
            m = frames == fr
            out["per_frame"][str(int(fr))] = _metrics_block(D_est[m], gt.D[m], depth, labels, free, sub)
    return jsonable(out)


def observation_residual(spec: ProblemSpec, D: np.ndarray, *, mask_X_gt: float = 0.02,
                         mode: str = "stored_T") -> Dict[str, Any]:
    """**Record** the residual of the re-forward (pass/fail is ``closure_gate``).

    mode: ``stored_T`` = the stored TransMatObs (operator of a single stack) reused for every pixel — a residual
    appears to the extent of the composition dependence (explained above). ``composition_T`` = T(D) rebuilt
    from each pixel's D, the generation formula itself (needs lambda_basic and gamma_eff).
    """
    D = np.asarray(D, dtype=np.float64)
    if mode == "stored_T":
        T = spec.stored_T
    elif mode == "composition_T":
        if spec.lambda_basic is None or spec.gamma_eff is None:
            return {"status": "NA", "mode": mode, "reason": "lambda_basic/gamma_eff missing"}
        T = composition_transfer_matrix(D, spec.lambda_basic, spec.axes.angles_deg,
                                        spec.axes.step_A, spec.gamma_eff)
    else:
        raise ValueError(mode)
    Xp = forward(T, D)
    X = spec.X
    m = X > mask_X_gt
    rel_full = np.where(m, np.abs(Xp - X) / np.where(m, X, 1.0), 0.0)
    rel = rel_full[m]
    per_px = np.array([np.median(rel_full[i][m[i]]) if m[i].any() else np.nan for i in range(X.shape[0])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {
            "status": "RECORDED", "mode": mode, "formula_version": REFORWARD_FORMULA_VERSION,
            "n_px": int(X.shape[0]), "mask_X_gt": mask_X_gt, "n_entries": int(m.sum()),
            "rel_residual_median": float(np.median(rel)) if rel.size else None,
            "rel_residual_p95": float(np.percentile(rel, 95)) if rel.size else None,
            "rel_residual_max": float(rel.max()) if rel.size else None,
            "abs_residual_max": float(np.abs(Xp - X).max()),
            "rel_residual_by_angle_max": rel_full.max(axis=(0, 2)).tolist(),
            "per_px_median_rel": {"median": float(np.nanmedian(per_px)),
                                  "p95": float(np.nanpercentile(per_px, 95)),
                                  "max": float(np.nanmax(per_px))} if per_px.size else None,
            "note": ("stored_T: single-stack T shared across pixels, residual explained by the "
                     "composition dependence of lambda; composition_T: the producer's forward itself. "
                     "Pass/fail is decided by closure_gate (noise = 0 only)."
                     if mode == "stored_T" else
                     "the producer-declared forward; at noise 0 the residual is float32 storage error; "
                     "at noise > 0 it is the noise realisation (noise model version pending)"),
        }
    return jsonable(out)


def closure_gate(spec: ProblemSpec, D_gt: np.ndarray, *, tol_abs: float = CLOSURE_TOL_ABS_NOISEFREE,
                 mask_X_gt: float = 0.02) -> Dict[str, Any]:
    """GT → observation closure gate. Builds X from the GT with the generation formula (producer-declared/1) and compares with the stored X.

    * noise == 0: PASS if max|X̂−X| ≤ tol_abs, FAIL otherwise. The tolerance is based on float32 storage precision.
    * noise > 0 : RECORDED (the noise realization is on top, so no pass/fail. The noise model version awaits the contract).
    * lambda_basic / gamma_eff missing: NA.
    The GT is used as stored (not renormalized).
    """
    D_gt = np.asarray(D_gt, dtype=np.float64)
    if spec.lambda_basic is None or spec.gamma_eff is None:
        return {"status": "NA", "reason": "lambda_basic/gamma_eff missing", "formula_version": REFORWARD_FORMULA_VERSION}
    T = composition_transfer_matrix(D_gt, spec.lambda_basic, spec.axes.angles_deg, spec.axes.step_A, spec.gamma_eff)
    Xp = forward(T, D_gt)
    X = spec.X
    diff = np.abs(Xp - X)
    m = X > mask_X_gt
    rel = diff[m] / X[m] if m.any() else np.array([])
    per_px_max = diff.reshape(X.shape[0], -1).max(axis=1)
    out: Dict[str, Any] = {
        "formula_version": REFORWARD_FORMULA_VERSION, "n_px": int(X.shape[0]), "noise": spec.noise,
        "max_abs": float(diff.max()), "median_abs": float(np.median(diff)),
        "rel_residual_median": float(np.median(rel)) if rel.size else None,
        "rel_residual_max": float(rel.max()) if rel.size else None,
        "n_px_over_tol": int((per_px_max > tol_abs).sum()), "tol_abs": tol_abs,
        "float32_floor_abs": float(np.finfo(np.float32).eps * X.max()),
        "gt_renormalised": False,
    }
    if spec.noise is None:
        out.update(status="UNRESOLVED", reason="noise level unknown; cannot decide exact closure")
    elif spec.noise == 0:
        if diff.max() <= tol_abs:
            out.update(status="PASS", note="exact closure of the producer-declared forward at noise 0")
        else:
            out.update(status="FAIL",
                       note=f"noise 0 but max|X_hat-X| exceeds tol {tol_abs:g}: generator version or "
                            "forward differs from producer-declared/1 (do not renormalise GT to fit)")
            # Diagnostic aid (does not change pass/fail): whether it reproduces when the stored lambdabasic is
            # oriented opposite to the MATLAB dialect (transposed). Real case of the Python producer on 2026-09-09 (values matched the manifest; only the H5 orientation was reversed).
            if spec.lambda_basic.shape[0] == spec.lambda_basic.shape[1]:
                T_t = composition_transfer_matrix(D_gt, spec.lambda_basic.T, spec.axes.angles_deg,
                                                  spec.axes.step_A, spec.gamma_eff)
                d_t = float(np.abs(forward(T_t, D_gt) - X).max())
                out["max_abs_with_transposed_lambda"] = d_t
                out["transposed_lambda_would_pass"] = bool(d_t <= tol_abs)
                if d_t <= tol_abs:
                    out["note"] += (f". With lambda_basic transposed the closure holds (max {d_t:.2e}): "
                                    "the stored ezdeprof/lambdabasic is probably written in the wrong "
                                    "orientation for the MATLAB HDF5 dialect (producer writer), not a "
                                    "different forward")
    else:
        out.update(status="RECORDED",
                   note="noise > 0: residual includes the noise realisation; noise model version pending")
    return jsonable(out)
