"""Batched block-coordinate solver for the SemiFreeSolver problem class (Stage C).

Solves, for B independent pixels sharing one transfer matrix T, the
SemiFreeSolver objective (charge-balance mode, substrate normalization):

    min_{C, t}  0.5 * degree * sum_{a, e in antisubst} (xw_e * (I_obs - I_pred_norm))^2
                + alpha * R(C, t)
    s.t.        sum_j flc_j * C[z, j] = 1   (each active depth z)
                min_conc <= C[z, j] <= 1/flc_j,   t_lo <= t <= t_hi

where D is built exactly as SemiFreeSolver._build_full_profile:

    erf_sub(z)  = 0.5 * (1 + erf((z - t) / (a / (2 sqrt(ln 2)))))
    D_sub       = erf_sub x sub_ratios
    film_scale  = max(1 - sum(D_sub), min_conc)
    D_free      = C * film_scale
    D_linked    = sum_j D_free_j * (flc_j - 1) / n_linked
    D           = clip(D, min_conc, max_conc)

    I_pred_norm = (T @ D) / sum_{e in sub}(T @ D)     ('substrate' mode)

R matches SemiFreeSolver's regularization on the *packed x* (normalized
film concentrations C, thickness t; abruptness fixed here):
    L2:  sum((C - m0_C)^2) + 100*((t - m0_t)/100)^2 (+ fixed-a term, const)
    MEM: sum(C log(C/m0_C) - C + m0_C) + (t log(t/m0_t) - t + m0_t)/10
    L1:  sum(|C - m0_C|) + |t - m0_t|/100      (exact; see free_batch note)
    LLS: 0

Block-coordinate structure (a batched solve_alternating):
  - C-block: with (t, a, s) fixed, the normalization denominator depends
    only on the substrate profile, so I_pred_norm[antisubst] is LINEAR in
    C -> the C subproblem is convex WLS + convex reg on a per-depth
    weighted capped simplex. Solved with the Stage A FISTA machinery
    (MFISTA best-tracking, masked convergence, round-level step adapt).
  - t-block: 1-D per pixel, but every candidate t is scored AFTER a short
    warm-started C refinement (the reference solve_alternating re-solves
    C per t via Brent x SLSQP). Scoring at fixed C is invalid: the C
    optimum at a wrong t is a genuine constrained stationary point with f
    orders of magnitude above the joint optimum.
    Bracket-zoom grid per outer round (the round-0 full-range grid doubles
    as a multistart); derivative-free, so the |t - m0_t| kink in the L1 reg needs no
    smoothing.
  - a-block (optional, optimize_abruptness): same 1-D machinery on a.
  - s-block (n_sub >= 2, optional): few projected-gradient steps on the
    substrate ratios (simplex, autodiff gradient).

Monotone safety: every block update is accepted per pixel only if it
lowers the true objective f (elementwise where), so f is non-increasing
across blocks and outer rounds by construction.

Raggedness-free: fixed-count rounds, masked freezes, host sync only at
round boundaries; grid/golden evaluations are fully batched.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mlx.core as mx

from .projections import project_capped_simplex

ROUND_SIZE = 50
POWER_ITERS = 15
STEP_SAFETY = 0.9
_FWHM2SIG = 1.0 / (2.0 * np.sqrt(np.log(2.0)))


@dataclass
class SemiFreeBatchConfig:
    method: str = 'LLS'            # 'LLS' | 'L1' | 'L2' | 'MEM' (conc-space)
    alpha: float = 0.01
    degree: float = 1.0
    min_conc: float = 1e-6
    max_conc: float = 1.0
    # C-block (FISTA)
    max_iter: int = 350            # FISTA iters per C-block sweep
    conv_tol: float = 1e-3         # projected-gradient-mapping residual
    round_size: int = ROUND_SIZE
    compile_step: bool = True
    # Diagnostic controls.  Defaults preserve the production per-pixel,
    # power-iteration step and convergence-driven stopping.  A caller may
    # supply a scalar step derived once for a whole cell to test whether
    # chunk boundaries alter the optimization path.
    fixed_step: Optional[float] = None
    capture_initial_step: bool = False
    fixed_iterations: bool = False
    # R1 (chunk-invariance remediation, PAPER3 CHUNK-2): make the
    # power-iteration initial vector a stateless function of
    # (base seed, global pixel id, invocation counter) so a pixel's step
    # estimate no longer depends on batch position, chunk membership, or
    # the global RNG stream.  Also replaces the batch-global L floor
    # (1e-12 * max over the chunk) with a per-pixel absolute floor.
    # Default False keeps the earlier behavior bit-for-bit.
    deterministic_power_init: bool = False
    power_init_seed: int = 0
    # t-block (batched solve_alternating equivalent: every candidate t is
    # scored AFTER a short warm-started C refinement — scoring at fixed C
    # is invalid, the blocks are too strongly coupled)
    optimize_thickness: bool = True
    thickness_bounds: tuple = (50.0, 500.0)
    t_grid: int = 8                # round-0 candidates (config-able, v2)
    t_refine_iters: int = 100      # round-0 inner C iterations per candidate
    t_grid_later: int = 5          # rounds >= 1: bracket is narrow, fewer
    t_refine_later: int = 40       #   candidates / shallower refines suffice
    t_bracket_min: float = 1.0     # A; bracket floor for the zoom schedule
    # warm-start bracket: when the caller trusts t_init (NN warm-start,
    # known-structure runs), search only +- this radius instead of the
    # full bounds. None = full range (tier-condition default).
    t_radius_init: Optional[float] = None
    # a-block
    optimize_abruptness: bool = False
    abruptness_bounds: tuple = (1.0, 50.0)
    a_grid: int = 5
    a_refine_iters: int = 40
    # s-block (substrate ratios, n_sub >= 2)
    optimize_substrate_ratio: bool = True
    s_pg_iters: int = 20           # projected-gradient steps per round
    # outer loop
    outer_rounds: int = 6
    t_conv_tol: float = 0.05       # Angstrom; pixel converged when t stalls
    # degenerate-depth handling (SemiFreeSolver degen_depth_threshold);
    # active mask is refreshed from the current (t, a, s) each outer round
    # (batched equivalent of the reference's warm restart).
    degen_threshold: float = 0.0
    # Explicit thickness prior (opt-in; t_prior_center=None keeps the legacy objective).
    # When t_prior_center is set, P(t) = (t - center)² / (2·sigma²) is added to
    # every objective evaluation (t-candidate racing and the final float64
    # re-evaluation) OUTSIDE alpha·reg. None (default) = off = legacy behavior.
    # t_prior_sigma has no default and must be given whenever t_prior_center
    # is set; it is a penalty width in Å, not a statistical standard
    # deviation (see deppro.inverse_solver.SemiFreeSolverConfig).
    t_prior_center: Optional[float] = None
    t_prior_sigma: Optional[float] = None

    def __post_init__(self):
        if self.t_prior_center is not None and self.t_prior_sigma is None:
            raise ValueError(
                "t_prior_sigma must be given when t_prior_center is set "
                "(it has no default; see SemiFreeSolverConfig)")


@dataclass
class SemiFreeBatchResult:
    D_opt: np.ndarray              # (B, n_depth, n_elem)
    I_opt: np.ndarray              # (B, n_angles, n_elem), total-normalized (reporting)
    film_conc: np.ndarray          # (B, n_depth, n_free) normalized concentrations
    thickness: np.ndarray          # (B,)
    abruptness: np.ndarray         # (B,)
    sub_ratios: np.ndarray         # (B, n_sub)
    chi2: np.ndarray               # (B,) xreg-weighted data chi2 (reference convention)
    objective: np.ndarray          # (B,) float64 CPU re-evaluation of f
    n_outer: int
    converged: np.ndarray          # (B,) bool
    elapsed_time: float = 0.0
    timings: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Batched physics (mirrors SemiFreeSolver._build_full_profile, charge balance)
# ---------------------------------------------------------------------------

def _make_forward(T_anti, T_sub, depth, n_free, n_linked, flc,
                  min_conc, max_conc):
    """Batched forward returning (I_anti, I_sub) — no gather/scatter ops
    (mx.take's backward is a scatter, whose JVP is unimplemented in MLX
    0.30; grouping the columns up front avoids it and saves a stack).

    T_anti: (A, Z, n_anti) transfer matrix, antisubst columns in
        free+linked order. T_sub: (A, Z, S) substrate columns.
    """
    coeff = flc - 1.0                          # (n_free,)
    nl = max(n_linked, 1)

    def forward(C, t, a, s):
        """C: (B, Z, F), t/a: (B,), s: (B, S) -> I_anti (B,A,n_anti),
        I_sub (B,A,1)."""
        sigma = mx.maximum(a * _FWHM2SIG, 1e-10)
        u = (depth[None, :] - t[:, None]) / sigma[:, None]      # (B, Z)
        erf_prof = 0.5 * (1.0 + mx.erf(u))                       # (B, Z)
        sub_cols = erf_prof[:, :, None] * s[:, None, :]          # (B, Z, S)
        sub_sum = mx.sum(sub_cols, axis=-1)                      # (B, Z)
        film_scale = mx.maximum(1.0 - sub_sum, min_conc)         # (B, Z)
        D_free = C * film_scale[:, :, None]                      # (B, Z, F)
        linked_val = (mx.sum(D_free * coeff, axis=-1)
                      / nl)[:, :, None]                          # (B, Z, 1)
        if n_linked > 0:
            linked_cols = mx.broadcast_to(
                linked_val, (*linked_val.shape[:2], n_linked))
            D_anti = mx.concatenate([D_free, linked_cols], axis=-1)
        else:
            D_anti = D_free
        D_anti = mx.clip(D_anti, min_conc, max_conc)
        D_sub = mx.clip(sub_cols, min_conc, max_conc)
        I_anti = mx.einsum('ade,bde->bae', T_anti, D_anti)
        I_sub = mx.sum(mx.einsum('ads,bds->bas', T_sub, D_sub),
                       axis=-1, keepdims=True)
        return I_anti, I_sub

    return forward


def _make_reg_fn(method, m0_C, m0_t, m0_a, alpha, t_prior=None):
    """alpha * R(C, t, a) per pixel, matching SemiFreeSolver's reg on packed x.

    m0_C: (Z, F) or (B, Z, F); m0_t / m0_a: (B,). Only L2 has an abruptness
    term (reference lines 3131-3133); MEM/L1 regularize C and t only.
    """
    if method == 'L2':
        def reg(C, t, a):
            dc = C - m0_C
            reg_t = 100.0 * ((t - m0_t) / 100.0) ** 2
            reg_a = 100.0 * ((a - m0_a) / 10.0) ** 2
            return alpha * (mx.sum(dc * dc, axis=(1, 2)) + reg_t + reg_a)
    elif method == 'MEM':
        m0s = mx.maximum(m0_C, 1e-10)
        def reg(C, t, a):
            Cs = mx.maximum(C, 1e-10)
            rc = mx.sum(Cs * mx.log(Cs / m0s) - Cs + m0s, axis=(1, 2))
            ts = mx.maximum(t, 1e-10)
            m0ts = mx.maximum(m0_t, 1e-10)
            rt = (ts * mx.log(ts / m0ts) - ts + m0ts) / 10.0
            return alpha * (rc + rt)
    elif method == 'L1':
        def reg(C, t, a):
            rc = mx.sum(mx.abs(C - m0_C), axis=(1, 2))
            return alpha * (rc + mx.abs(t - m0_t) / 100.0)
    else:  # LLS
        def reg(C, t, a):
            return mx.zeros(t.shape, dtype=C.dtype)
    if t_prior is None:
        return reg
    # Explicit thickness prior, additive and independent of alpha/method.
    t0, sig2 = t_prior
    reg_base = reg

    def reg_with_t_prior(C, t, a):
        return reg_base(C, t, a) + (t - t0) ** 2 / (2.0 * sig2)

    return reg_with_t_prior


def solve_semifree_batch(
    T: np.ndarray,
    I_obs_norm: np.ndarray,
    C_init: np.ndarray,
    t_init: np.ndarray,
    a_init: np.ndarray,
    s_init: np.ndarray,
    depth: np.ndarray,
    free_idx: list,
    linked_idx: list,
    sub_idx: list,
    free_linked_const: np.ndarray,
    m0_C: np.ndarray,
    m0_t: np.ndarray,
    config: SemiFreeBatchConfig,
    xreg_weights: Optional[np.ndarray] = None,
    verbose: bool = False,
    pixel_ids: Optional[np.ndarray] = None,
) -> SemiFreeBatchResult:
    """Solve B independent SemiFree problems with shared T.

    Args:
        T: (n_angles, n_depth, n_elem) shared transfer matrix.
        I_obs_norm: (B, n_angles, n_elem) substrate-normalized observations.
        C_init: (B, n_depth, n_free) normalized film concentrations x0.
        t_init, a_init: (B,) thickness / abruptness initial values.
        s_init: (B, n_sub) substrate ratios (sum-to-one rows).
        depth: (n_depth,) depth grid [A].
        free_idx / linked_idx / sub_idx: element classification.
        free_linked_const: (n_free,) flc coefficients (>= 1).
        m0_C: (n_depth, n_free) or (B, n_depth, n_free) prior for C
            (= D_prior[:, free_idx], the reference packs raw prior values).
        m0_t: (B,) thickness prior (reference: thickness_init unless the
            prior profile is ~0, then 0.0).
        config: solver configuration.
        xreg_weights: (B, n_antisubst) or None (plain OLS misfit).
    """
    import time as _time
    t_start = _time.perf_counter()

    if config.method not in ('LLS', 'L1', 'L2', 'MEM'):
        raise NotImplementedError(f"unknown method {config.method!r}")

    f32 = mx.float32
    B, n_angles, n_elem = I_obs_norm.shape
    n_depth = len(depth)
    n_free = len(free_idx)
    n_sub = len(sub_idx)
    antisubst_idx = list(free_idx) + list(linked_idx)

    T_np = np.asarray(T, dtype=np.float64)
    T_anti = mx.array(np.ascontiguousarray(T_np[:, :, antisubst_idx]),
                      dtype=f32)
    T_sub = mx.array(np.ascontiguousarray(T_np[:, :, sub_idx]), dtype=f32)
    depth_mx = mx.array(np.asarray(depth), dtype=f32)
    flc = mx.array(np.asarray(free_linked_const), dtype=f32)
    I_anti = mx.array(np.ascontiguousarray(
        np.asarray(I_obs_norm, dtype=np.float64)[:, :, antisubst_idx]), dtype=f32)
    if xreg_weights is not None:
        XW = mx.array(np.ascontiguousarray(xreg_weights), dtype=f32)[:, None, :]
    else:
        XW = mx.ones((1, 1, 1), dtype=f32)

    # R1: stable global pixel ids for the deterministic power init.
    # Default arange(B) is only correct for an unchunked, unpermuted call;
    # chunked/permuted callers MUST pass the global ids explicitly.
    _pids_np = (np.arange(B, dtype=np.float64) if pixel_ids is None
                else np.asarray(pixel_ids, dtype=np.float64))
    pids = mx.array(_pids_np, dtype=f32)

    C = mx.array(np.ascontiguousarray(C_init), dtype=f32)
    t = mx.array(np.ascontiguousarray(t_init), dtype=f32)
    a = mx.array(np.ascontiguousarray(a_init), dtype=f32)
    s = mx.array(np.ascontiguousarray(s_init), dtype=f32)
    m0C = mx.array(np.ascontiguousarray(m0_C), dtype=f32)
    m0t = mx.array(np.ascontiguousarray(m0_t), dtype=f32)

    lo = config.min_conc
    # per-element upper bound 1/flc (reference free_ub), shared across depths
    ub_row = mx.minimum(1.0 / mx.maximum(flc, 1e-12), config.max_conc)

    # abruptness prior = abruptness_init (reference m0 packs it)
    m0a = mx.array(np.ascontiguousarray(a_init), dtype=f32)

    forward = _make_forward(T_anti, T_sub, depth_mx, n_free,
                            len(linked_idx), flc, config.min_conc,
                            config.max_conc)
    t_prior = None
    if config.t_prior_center is not None:
        t0_val = float(config.t_prior_center)
        if not np.isfinite(t0_val):
            raise ValueError(f"t_prior_center must be finite, got {t0_val}")
        if config.t_prior_sigma is None:
            raise ValueError(
                "t_prior_sigma must be given when t_prior_center is set")
        sig_val = float(config.t_prior_sigma)
        if not (np.isfinite(sig_val) and sig_val > 0):
            raise ValueError(
                f"t_prior_sigma must be finite and > 0, got {sig_val}")
        t_prior = (mx.array(t0_val, dtype=f32), sig_val ** 2)
    reg_fn = _make_reg_fn(config.method, m0C, m0t, m0a, config.alpha,
                          t_prior=t_prior)
    degree = mx.array(config.degree, dtype=f32)

    def obj_px(C, t, a, s):
        """True per-pixel objective f (B,). Mirrors SemiFreeSolver.objective."""
        I_pred_anti, I_sub = forward(C, t, a, s)
        I_pred_anti = I_pred_anti / mx.maximum(I_sub, 1e-10)
        diff = (I_pred_anti - I_anti) * XW
        mis = 0.5 * degree * mx.sum(diff * diff, axis=(1, 2))
        return mis + reg_fn(C, t, a)

    grad_C = mx.grad(lambda C, t, a, s: mx.sum(obj_px(C, t, a, s)), argnums=0)
    if n_sub >= 2:
        grad_s = mx.grad(lambda s, C, t, a: mx.sum(obj_px(C, t, a, s)), argnums=0)

    # ------------------------------------------------------------------
    # Diagonal metric (always used for SemiFree).
    #
    # The C subproblem is multi-scale: misfit curvature carries
    # film_scale(z)^2 which spans [1e-12, 1] across depths, so a single
    # Lipschitz step makes deep-slice reg mass redistribution O(1/k)-slow
    # (thousands of iters). M = diag(J^T W J) + alpha*mu_R equalizes the
    # coordinate scales; for L1/LLS mu_R = 0, so the anti-Jacobi effect of
    # the L2 case (alpha*mu*I flattening being re-spread) does not apply.
    # diag(J^T W J) is closed-form because I_sub is C-independent:
    #   dI_norm[a,e]/dC[z,f] = g_z/I_sub[a] * (T[a,z,f] delta_{ef}
    #                          + T[a,z,l] coeff_f/n_l)
    # ------------------------------------------------------------------
    Tf2 = T_anti[:, :, :n_free] ** 2                       # (A, Z, F)
    n_linked_ = len(linked_idx)
    Tl2 = T_anti[:, :, n_free:] ** 2 if n_linked_ else None  # (A, Z, L)
    xw_f2 = (XW[:, 0, :n_free] ** 2 if xreg_weights is not None
             else mx.ones((1, n_free), dtype=f32))          # (B|1, F)
    xw_l2 = (XW[:, 0, n_free:] ** 2 if (xreg_weights is not None
             and n_linked_) else
             (mx.ones((1, n_linked_), dtype=f32) if n_linked_ else None))
    coeff2 = ((flc - 1.0) / max(n_linked_, 1)) ** 2         # (F,)

    def make_metric(t, a, s, C_ref_):
        """(B, Z, F) diagonal curvature; None disables preconditioning."""
        sigma = mx.maximum(a * _FWHM2SIG, 1e-10)
        u = (depth_mx[None, :] - t[:, None]) / sigma[:, None]
        erf_sub = 0.5 * (1.0 + mx.erf(u)) * mx.sum(s, axis=-1)[:, None]
        g = mx.maximum(1.0 - erf_sub, config.min_conc)      # (B, Z)
        # I_sub from the substrate profile (C-independent)
        sub_cols = (0.5 * (1.0 + mx.erf(u)))[:, :, None] * s[:, None, :]
        sub_cols = mx.clip(sub_cols, config.min_conc, config.max_conc)
        I_sub = mx.sum(mx.einsum('ads,bds->bas', T_sub, sub_cols), axis=-1)
        inv2 = 1.0 / mx.maximum(I_sub, 1e-10) ** 2          # (B, A)
        m_free = mx.einsum('azf,ba->bzf', Tf2, inv2) * xw_f2[:, None, :]
        if n_linked_:
            s_l = mx.einsum('azl,ba->bzl', Tl2, inv2) * xw_l2[:, None, :]
            m_link = mx.sum(s_l, axis=-1)[:, :, None] * coeff2
        else:
            m_link = 0.0
        M = degree * (g ** 2)[:, :, None] * (m_free + m_link)
        if config.method == 'L2':
            M = M + 2.0 * config.alpha
        elif config.method == 'MEM':
            M = M + config.alpha / mx.maximum(C_ref_, 1e-6)
        # relative floor: keeps deep-slice steps large but finite (f32)
        M_max = mx.max(M, axis=(1, 2), keepdims=True)
        return mx.maximum(M, 1e-6 * M_max)

    def active_mask_fn(t, a, s):
        """(B, Z) bool: depths where the sum constraint is meaningful.

        rhs = 1 - erf_sub >= degen_threshold (const elements unsupported).
        """
        sigma = mx.maximum(a * _FWHM2SIG, 1e-10)
        u = (depth_mx[None, :] - t[:, None]) / sigma[:, None]
        erf_sub = 0.5 * (1.0 + mx.erf(u)) * mx.sum(s, axis=-1)[:, None]
        return (1.0 - erf_sub) >= config.degen_threshold

    def project_C(V, active, M):
        """Weighted capped-simplex projection per depth row + degen fallback.

        Per-element upper bounds ub_j = 1/flc_j enter as a vector `hi`
        (all projection ops broadcast over the last axis). On active rows
        the equality sum(flc*C)=1 with C >= min_conc already implies
        C_j < 1/flc_j, so the vector cap only matters for degenerate rows.
        The diagonal metric M makes the projection consistent with the
        preconditioned gradient step (variable-metric PG).
        """
        P = project_capped_simplex(V, lo=lo, hi=ub_row, target=1.0,
                                   weights=flc, metric=M)
        clipped = mx.clip(V, lo, ub_row)
        return mx.where(active[:, :, None], P, clipped)

    # ------------------------------------------------------------------
    # C-block: FISTA sweep (adapted from free_batch.fista_sweep)
    # ------------------------------------------------------------------
    timings = {'c_block': 0.0, 'ta_block': 0.0, 'setup': 0.0, 's_block': 0.0}
    total_c_iters = 0

    def estimate_step(C0, t, a, s, M, n_iters=POWER_ITERS, ctx=0.0):
        """Per-pixel 1/L of the M-preconditioned gradient map.

        ctx is a STRUCTURED call context (outer round / call site /
        candidate index), NOT a running counter: a counter depends on how
        many calls the chunk executed before (early-break history), which
        couples a pixel's init vector to its companions."""
        if config.fixed_step is not None:
            return mx.full((B, 1, 1), float(config.fixed_step), dtype=f32)
        if config.deterministic_power_init:
            # R1: stateless per-pixel init from (seed, pixel id, context).
            # Shader-style hash; power iteration only needs a non-degenerate
            # start, not normality.  No global RNG stream, no batch coupling.
            ctx = float(ctx)
            zi = mx.arange(C0.shape[1], dtype=f32)[None, :, None]
            ei = mx.arange(C0.shape[2], dtype=f32)[None, None, :]
            ph = (pids[:, None, None] * 12.9898 + zi * 78.233 + ei * 37.719
                  + ctx * 0.61803398875
                  + float(config.power_init_seed) * 3.14159265)
            v = mx.sin(ph) * 43758.5453
            v = v - mx.floor(v) - 0.5
        else:
            v = mx.random.normal(C0.shape)
        v = v / mx.maximum(mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True)),
                           1e-30)
        fn = lambda C: grad_C(C, t, a, s) / M
        L = mx.ones((B,))
        for _ in range(n_iters):
            jv = mx.jvp(fn, [C0], [v])[1][0]
            L = mx.sqrt(mx.sum(jv * jv, axis=(1, 2)))
            v = jv / mx.maximum(L[:, None, None], 1e-30)
        if config.deterministic_power_init:
            # per-pixel absolute floor: the archived relative floor couples
            # every pixel to the chunk maximum (batch-global reduction)
            L = mx.maximum(L, 1e-12)
        else:
            L = mx.maximum(L, 1e-12 * mx.max(L))
        return (STEP_SAFETY / L)[:, None, None]

    def c_step_fn(C, Y, tm_, mask, f_best, C_best, step, active, M, t, a, s):
        G = grad_C(Y, t, a, s) / M
        C_new = project_C(Y - step * G, active, M)
        f_new = obj_px(C_new, t, a, s)
        better = mx.logical_and(f_new < f_best, mx.logical_not(mask))
        b3 = better[:, None, None]
        C_best = mx.where(b3, C_new, C_best)
        f_best = mx.where(better, f_new, f_best)
        restart = mx.sum((Y - C_new) * (C_new - C), axis=(1, 2)) > 0.0
        t_eff = mx.where(restart, mx.ones_like(tm_), tm_)
        t_new = 0.5 * (1.0 + mx.sqrt(1.0 + 4.0 * t_eff * t_eff))
        Y_new = C_new + ((t_eff - 1.0) / t_new)[:, None, None] * (C_new - C)
        m3 = mask[:, None, None]
        return (mx.where(m3, C, C_new), mx.where(m3, Y, Y_new),
                mx.where(mask, tm_, t_new), mask, f_best, C_best)

    c_step_c = mx.compile(c_step_fn) if config.compile_step else c_step_fn

    def residual_map(C, step, active, M, t, a, s):
        G = grad_C(C, t, a, s) / M
        P = project_C(C - step * G, active, M)
        num = mx.sqrt(mx.sum((C - P) ** 2, axis=(1, 2)))
        den = mx.sqrt(mx.sum((step * G) ** 2, axis=(1, 2))) + 1e-12
        return num / den

    def c_block(C, f_in, active, t, a, s, frozen, ctx=0.0):
        """FISTA on C with (t, a, s) fixed; returns per-pixel best C, f."""
        nonlocal total_c_iters
        M = make_metric(t, a, s, C)
        step = estimate_step(C, t, a, s, M, ctx=ctx)
        mx.eval(step)
        if config.capture_initial_step and "initial_step_global_min" not in timings:
            timings["initial_step_global_min"] = float(
                np.min(np.asarray(step)))
            timings["initial_step_global_max"] = float(
                np.max(np.asarray(step)))
        Y = C
        t_mom = mx.ones((B,))
        mask = (mx.zeros((B,), dtype=mx.bool_) if config.fixed_iterations
                else frozen)               # fixed-iteration diagnostic: no early freeze
        f_best = f_in
        C_best = C
        step_min = step * (2.0 ** -8)
        res_prev = mx.full((B,), np.inf)
        n_rounds = max(1, (config.max_iter + config.round_size - 1)
                       // config.round_size)
        it = 0
        for _ in range(n_rounds):
            k = min(config.round_size, config.max_iter - it)
            for _ in range(k):
                C, Y, t_mom, mask, f_best, C_best = c_step_c(
                    C, Y, t_mom, mask, f_best, C_best, step, active, M,
                    t, a, s)
            mx.eval(C, C_best)
            it += k
            total_c_iters += k
            res = residual_map(C_best, step, active, M, t, a, s)
            if not config.fixed_iterations:
                mask = mx.logical_or(mask, res < config.conv_tol)
            stalled = mx.logical_and(res >= 1.05 * res_prev,
                                     mx.logical_not(mask))
            s3 = stalled[:, None, None]
            step = mx.maximum(mx.where(s3, step * 0.5, step), step_min)
            C = mx.where(s3, C_best, C)
            Y = mx.where(s3, C_best, Y)
            t_mom = mx.where(stalled, mx.ones_like(t_mom), t_mom)
            res_prev = res
            all_done = mx.all(mask)
            mx.eval(all_done)
            if bool(all_done.item()):
                break
        return C_best, f_best

    # ------------------------------------------------------------------
    # 1-D blocks (t and a): batched solve_alternating equivalent.
    #
    # Scoring a candidate t at FIXED C is invalid — the blocks are too
    # strongly coupled (film_scale multiplies every free concentration,
    # and the C optimum at a wrong t is a genuine stationary point with
    # f orders of magnitude above the joint optimum).
    # The reference solve_alternating re-solves C per t (Brent x SLSQP);
    # here every candidate gets a short warm-started C refinement before
    # scoring, and the winner's refined C is kept. Bracket shrinks each
    # outer round (grid zoom) instead of golden section — golden's
    # unimodality argument only holds on the partially-minimized f*(t).
    # ------------------------------------------------------------------
    def refine_candidate(C0, t_g, a_g, K, ctx=0.0):
        """K FISTA iters on C at fixed (t_g, a_g, s); returns best (f, C)."""
        act = active_mask_fn(t_g, a_g, s)
        M = make_metric(t_g, a_g, s, C0)
        step = estimate_step(C0, t_g, a_g, s, M, n_iters=5, ctx=ctx)
        f0 = obj_px(C0, t_g, a_g, s)
        Cl, Y = C0, C0
        tm_ = mx.ones((B,))
        mask = mx.zeros((B,), dtype=mx.bool_)
        f_best, C_best = f0, C0
        for _ in range(K):
            Cl, Y, tm_, mask, f_best, C_best = c_step_c(
                Cl, Y, tm_, mask, f_best, C_best, step, act, M, t_g, a_g, s)
        return f_best, C_best

    def search_1d(x_cur, f_in, C_in, lo_b, hi_b, radius, n_cand, K,
                  which, frozen, ctx_base=0.0):
        """One zoom round over [x-radius, x+radius] cap [lo_b, hi_b].

        Fair race: the CURRENT x is re-scored with the same K-iteration
        refinement as the candidates. Ranking a fully-converged current f
        against shallow-refined candidates biases against moving and
        freezes x when K is small. Global monotonicity is enforced by the caller's
        best-point tracking, not per move.
        """
        lo_v = mx.maximum(x_cur - radius, lo_b)
        hi_v = mx.minimum(x_cur + radius, hi_b)

        def score(xg, ci):
            if which == 't':
                return refine_candidate(C_in, xg, a, K, ctx=ctx_base + ci)
            return refine_candidate(C_in, t, xg, K, ctx=ctx_base + ci)

        best_x = x_cur
        best_f, best_C = score(x_cur, 9.0)     # same-depth baseline
        mx.eval(best_f, best_C)
        for i in range(n_cand):
            frac = i / max(n_cand - 1, 1)
            xg = lo_v + (hi_v - lo_v) * frac
            fg, Cg = score(xg, float(i))
            better = fg < best_f
            b3 = better[:, None, None]
            best_x = mx.where(better, xg, best_x)
            best_f = mx.where(better, fg, best_f)
            best_C = mx.where(b3, Cg, best_C)
            mx.eval(best_x, best_f, best_C)   # bound the lazy graph
        # frozen pixels keep their state untouched
        best_x = mx.where(frozen, x_cur, best_x)
        best_f = mx.where(frozen, f_in, best_f)
        best_C = mx.where(frozen[:, None, None], C_in, best_C)
        # zoom: the winner lies within +-spacing of the sampled optimum,
        # so the next bracket is one spacing. (2*spacing stalls: with
        # n_cand=5, 2*spacing == radius -> zoom factor 1, no shrinkage.)
        spacing = (hi_v - lo_v) / max(n_cand - 1, 1)
        next_radius = mx.maximum(spacing, config.t_bracket_min)
        return best_x, best_f, best_C, next_radius

    # ------------------------------------------------------------------
    # Outer block-coordinate loop
    # ------------------------------------------------------------------
    timings['setup'] = _time.perf_counter() - t_start

    active = active_mask_fn(t, a, s)
    C = project_C(C, active, mx.ones((1, 1, n_free), dtype=f32))  # feasible start
    f_cur = obj_px(C, t, a, s)
    mx.eval(C, f_cur)

    # Global best across all blocks/rounds: every tracked f is the true
    # objective at a concrete (C, t, a, s), so they are all comparable.
    # This is what guarantees monotone output now that the 1-D search
    # ranks with shallow (fair-race) refinements.
    f_gb, C_gb, t_gb, a_gb, s_gb = f_cur, C, t, a, s

    def update_gbest():
        nonlocal f_gb, C_gb, t_gb, a_gb, s_gb
        better = f_cur < f_gb
        b3 = better[:, None, None]
        C_gb = mx.where(b3, C, C_gb)
        t_gb = mx.where(better, t, t_gb)
        a_gb = mx.where(better, a, a_gb)
        s_gb = mx.where(better[:, None], s, s_gb)
        f_gb = mx.where(better, f_cur, f_gb)

    if verbose:
        mx.eval(f_cur)
        print(f"  [mlx_semifree] init: f max={float(mx.max(f_cur).item()):.4g}"
              f" med={float(np.median(np.array(f_cur))):.4g}")

    frozen = mx.zeros((B,), dtype=mx.bool_)
    t_lo, t_hi = config.thickness_bounds
    a_lo, a_hi = config.abruptness_bounds
    r0 = (0.5 * (t_hi - t_lo) if config.t_radius_init is None
          else float(config.t_radius_init))
    t_radius = mx.full((B,), r0)
    a_radius = mx.full((B,), 0.5 * (a_hi - a_lo))
    n_outer_done = 0
    for outer in range(config.outer_rounds):
        t_prev = t

        t0 = _time.perf_counter()
        C, f_cur = c_block(C, f_cur, active, t, a, s, frozen,
                           ctx=outer * 100.0)
        update_gbest()
        timings['c_block'] += _time.perf_counter() - t0
        if verbose:
            mx.eval(f_cur)
            print(f"  [mlx_semifree] round {outer} C-block:"
                  f" f max={float(mx.max(f_cur).item()):.4g}"
                  f" med={float(np.median(np.array(f_cur))):.4g}")

        t0 = _time.perf_counter()
        n_cand = config.t_grid if outer == 0 else config.t_grid_later
        k_ref = (config.t_refine_iters if outer == 0
                 else config.t_refine_later)
        if config.optimize_thickness:
            t, f_cur, C, t_radius = search_1d(
                t, f_cur, C, t_lo, t_hi, t_radius,
                n_cand, k_ref, 't', frozen, ctx_base=outer * 100.0 + 10.0)
        if config.optimize_abruptness:
            a, f_cur, C, a_radius = search_1d(
                a, f_cur, C, a_lo, a_hi, a_radius,
                config.a_grid, config.a_refine_iters, 'a', frozen,
                ctx_base=outer * 100.0 + 50.0)
        update_gbest()
        timings['ta_block'] += _time.perf_counter() - t0
        if verbose:
            mx.eval(f_cur, t)
            print(f"  [mlx_semifree] round {outer} t-block:"
                  f" f max={float(mx.max(f_cur).item()):.4g}"
                  f" med={float(np.median(np.array(f_cur))):.4g}"
                  f" t med={float(np.median(np.array(t))):.1f}")

        if n_sub >= 2 and config.optimize_substrate_ratio:
            t0 = _time.perf_counter()
            # few projected-gradient steps on s (simplex, sum-to-one)
            step_s = mx.full((B, 1), 1e-2)
            for _ in range(config.s_pg_iters):
                Gs = grad_s(s, C, t, a)
                s_try = project_capped_simplex(
                    s - step_s * Gs, lo=config.min_conc, hi=config.max_conc,
                    target=1.0)
                f_try = obj_px(C, t, a, s_try)
                ok = mx.logical_and(f_try < f_cur, mx.logical_not(frozen))
                s = mx.where(ok[:, None], s_try, s)
                f_cur = mx.where(ok, f_try, f_cur)
                step_s = mx.where(ok[:, None], step_s * 1.2, step_s * 0.5)
            update_gbest()
            timings['s_block'] += _time.perf_counter() - t0

        # refresh active mask from the new (t, a, s) — batched warm restart
        active_new = active_mask_fn(t, a, s)
        active = active_new

        n_outer_done = outer + 1
        # freeze only once the bracket has collapsed AND t stopped moving
        # (dt alone is not convergence: early rounds legitimately move
        # little while C is still adapting)
        dt_move = mx.abs(t - t_prev)
        bracket_done = t_radius <= (config.t_bracket_min + 1e-6)
        if not config.fixed_iterations:
            frozen = mx.logical_or(
                frozen, mx.logical_and(dt_move < config.t_conv_tol,
                                       bracket_done))
        all_frozen = mx.all(frozen)
        mx.eval(all_frozen, f_cur)
        if (not config.fixed_iterations
                and outer >= 1 and bool(all_frozen.item())):
            break

    # final C polish at the global-best (t, a, s)
    C, t, a, s = C_gb, t_gb, a_gb, s_gb
    f_cur = f_gb
    active = active_mask_fn(t, a, s)
    C, f_cur = c_block(C, f_cur, active,
                       t, a, s, mx.zeros((B,), dtype=mx.bool_),
                       ctx=1e6)
    update_gbest()
    C, t, a, s, f_cur = C_gb, t_gb, a_gb, s_gb, f_gb
    mx.eval(C, t, a, s, f_cur)

    # ------------------------------------------------------------------
    # Outputs: float64 CPU re-evaluation (equivalence layer 1 currency)
    # ------------------------------------------------------------------
    C_np = np.array(C, dtype=np.float64)
    t_np = np.array(t, dtype=np.float64)
    a_np = np.array(a, dtype=np.float64)
    s_np = np.array(s, dtype=np.float64)

    D_np = _build_D_np(C_np, t_np, a_np, s_np, np.asarray(depth),
                       list(free_idx), list(linked_idx), list(sub_idx),
                       np.asarray(free_linked_const),
                       config.min_conc, config.max_conc)
    T64 = np.asarray(T, dtype=np.float64)
    I_pred = np.einsum('ade,bde->bae', T64, D_np)
    I_sub = np.maximum(I_pred[:, :, sub_idx].sum(axis=-1, keepdims=True),
                       1e-10)
    I_pred_norm = I_pred / I_sub
    I_obs64 = np.asarray(I_obs_norm, dtype=np.float64)
    diff = I_pred_norm[:, :, antisubst_idx] - I_obs64[:, :, antisubst_idx]
    if xreg_weights is not None:
        diff = diff * np.asarray(xreg_weights, dtype=np.float64)[:, None, :]
    chi2 = config.degree * np.sum(diff ** 2, axis=(1, 2))

    m0C_np = np.broadcast_to(np.asarray(m0_C, dtype=np.float64), C_np.shape)
    m0t_np = np.asarray(m0_t, dtype=np.float64)
    m0a_np = np.asarray(a_init, dtype=np.float64)
    if config.method == 'L2':
        reg = np.sum((C_np - m0C_np) ** 2, axis=(1, 2))
        reg = reg + 100.0 * ((t_np - m0t_np) / 100.0) ** 2
        reg = reg + 100.0 * ((a_np - m0a_np) / 10.0) ** 2
    elif config.method == 'MEM':
        Cs = np.maximum(C_np, 1e-10)
        m0s = np.maximum(m0C_np, 1e-10)
        reg = np.sum(Cs * np.log(Cs / m0s) - Cs + m0s, axis=(1, 2))
        ts = np.maximum(t_np, 1e-10)
        m0ts = np.maximum(m0t_np, 1e-10)
        reg = reg + (ts * np.log(ts / m0ts) - ts + m0ts) / 10.0
    elif config.method == 'L1':
        reg = np.sum(np.abs(C_np - m0C_np), axis=(1, 2))
        reg = reg + np.abs(t_np - m0t_np) / 100.0
    else:
        reg = np.zeros(B)
    objective = 0.5 * chi2 + config.alpha * reg
    if config.t_prior_center is not None:
        objective = objective + (
            (t_np - float(config.t_prior_center)) ** 2
            / (2.0 * float(config.t_prior_sigma) ** 2)
        )

    # reporting I_opt: total normalization (reference convention)
    I_tot = np.maximum(I_pred.sum(axis=-1, keepdims=True), 1e-10)

    return SemiFreeBatchResult(
        D_opt=D_np,
        I_opt=I_pred / I_tot,
        film_conc=C_np,
        thickness=t_np,
        abruptness=a_np,
        sub_ratios=s_np,
        chi2=chi2,
        objective=objective,
        n_outer=n_outer_done,
        converged=np.array(frozen),
        elapsed_time=_time.perf_counter() - t_start,
        timings=timings,
    )


def _build_D_np(C, t, a, s, depth, free_idx, linked_idx, sub_idx, flc,
                min_conc, max_conc):
    """float64 reference-exact profile builder (batched numpy)."""
    B, Z, F = C.shape
    n_elem = F + len(linked_idx) + len(sub_idx)
    sigma = np.maximum(a * _FWHM2SIG, 1e-10)
    erf_prof = 0.5 * (1.0 + _np_erf((depth[None, :] - t[:, None])
                                    / sigma[:, None]))
    D = np.zeros((B, Z, n_elem))
    sub_cols = erf_prof[:, :, None] * s[:, None, :]
    D[:, :, sub_idx] = sub_cols
    film_scale = np.maximum(1.0 - sub_cols.sum(axis=-1), min_conc)
    D_free = C * film_scale[:, :, None]
    D[:, :, free_idx] = D_free
    if linked_idx:
        linked_val = (D_free * (flc - 1.0)).sum(axis=-1) / len(linked_idx)
        for e in linked_idx:
            D[:, :, e] = linked_val
    return np.clip(D, min_conc, max_conc)


def _np_erf(x):
    from scipy.special import erf as _erf
    return _erf(x)
