"""Closed-form projections for batched MLX solvers.

All projections are raggedness-free: no data-dependent control flow, pure
elementwise/reduce ops, fully vmap/compile-compatible, GPU-only. The dual
variable is found EXACTLY by breakpoint evaluation + linear interpolation
(the piecewise-linear monotone dual has breakpoints only at (V-hi)/s and
(V-lo)/s) — no sequential bisection chain.
"""
from typing import Optional

import mlx.core as mx

# Retained for API compatibility; the exact breakpoint method needs no
# iteration count.
N_BISECT = 30


def project_capped_simplex(
    V: mx.array,
    lo: float = 1e-6,
    hi: float = 1.0,
    target: float = 1.0,
    weights: Optional[mx.array] = None,
    metric: Optional[mx.array] = None,
    n_bisect: int = N_BISECT,
) -> mx.array:
    """Project rows of V onto {x : sum(w*x) = target, lo <= x <= hi}.

    Solves, independently for every leading index of ``V``:

        min_x 0.5 * sum(m * (x - v)^2)
        s.t.  sum(w * x) = target,  lo <= x <= hi

    The KKT solution is x(lam) = clip(v - lam * w / m, lo, hi) where the
    scalar dual lam is the root of the monotone (non-increasing) piecewise
    linear g(lam) = sum(w * x(lam)) - target, found exactly by breakpoint
    evaluation + interpolation.

    Args:
        V: (..., n) points to project (projection over the last axis).
        lo, hi: box bounds (scalars).
        target: equality-constraint right-hand side.
        weights: (n,) constraint coefficients w (default: ones). Must be > 0.
        metric: (n,) diagonal metric m (default: ones = Euclidean).
            The preconditioner form is diag(T^T T) + alpha*mu; the
            default is the identity.
        n_bisect: unused (kept for API compatibility with the earlier
            bisection implementation).

    Returns:
        Projected array, same shape as V.
    """
    w = mx.ones(V.shape[-1], dtype=V.dtype) if weights is None else weights
    m = mx.ones(V.shape[-1], dtype=V.dtype) if metric is None else metric
    s = w / m  # step direction per unit dual

    # g(lam) = sum(w * clip(V - lam*s, lo, hi)) - target is piecewise linear
    # and non-increasing, with breakpoints where coordinates enter/leave the
    # box: lam = (V - hi)/s and (V - lo)/s. Evaluating g at all 2n
    # breakpoints (one broadcast op) and interpolating on the bracketing
    # (linear) segment gives the EXACT dual.
    #
    # Implementation note: g is evaluated at all 2n breakpoints with one
    # broadcast clip/sum; no sort or cumulative sum is needed.
    bp = mx.concatenate([(V - hi) / s, (V - lo) / s], axis=-1)   # (..., 2n)
    # s may be (n,) or fully batched (..., n) (variable-metric callers);
    # insert the breakpoint axis explicitly so both broadcast.
    x_bp = mx.clip(V[..., None, :] - bp[..., :, None] * s[..., None, :],
                   lo, hi)
    g_bp = mx.sum(w * x_bp, axis=-1) - target                    # (..., 2n)

    pos = g_bp > 0
    neg_inf = mx.array(float('-inf'), dtype=V.dtype)
    pos_inf = mx.array(float('inf'), dtype=V.dtype)
    # Tightest bracket: largest bp with g>0, smallest bp with g<=0
    lam_lo = mx.max(mx.where(pos, bp, neg_inf), axis=-1, keepdims=True)
    lam_hi = mx.min(mx.where(pos, pos_inf, bp), axis=-1, keepdims=True)
    g_lo = mx.min(mx.where(pos, g_bp, pos_inf), axis=-1, keepdims=True)
    g_hi = mx.max(mx.where(pos, neg_inf, g_bp), axis=-1, keepdims=True)

    # Interpolate on the linear segment; guard the degenerate cases
    # (bracket outside breakpoint range, or g_lo == g_hi).
    denom = g_lo - g_hi
    frac = mx.where(denom > 1e-30, g_lo / mx.maximum(denom, 1e-30), 0.5)
    lam = lam_lo + frac * (lam_hi - lam_lo)
    lam = mx.where(mx.isinf(lam_lo), lam_hi, lam)   # all g <= 0
    lam = mx.where(mx.isinf(lam_hi), mx.where(mx.isinf(lam_lo), 0.0, lam_lo),
                   lam)                             # all g > 0 / empty
    return mx.clip(V - lam * s, lo, hi)
