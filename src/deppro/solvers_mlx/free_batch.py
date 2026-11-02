"""Batched FISTA solver for the FreeSolver problem class (Stage A).

Solves, for B independent pixels sharing one transfer matrix T:

    min_D  0.5 * degree * chi2(D) + alpha * R(D)
    s.t.   sum_e D(z, e) = 1  (each depth z),   min_conc <= D <= max_conc

    chi2(D) = sum_{a,e} w * (I_obs_norm - normalize(T @ D))^2
    R(D)    = ||D - D_prior||^2  (L2)  |  0  (LLS)

The *objective* matches ``FreeSolver._objective`` exactly (this is what
the reference SLSQP evaluates and what equivalence layer 1 compares).
The *gradient* is the exact autodiff gradient of that objective.
Historical note: ``FreeSolver._gradient`` used to ignore the
normalization derivative ("simplified"), which made the reference SLSQP
stop above the true objective minimum; the simplified map's fixed points
were NOT minimizers of f.
``FreeSolver._gradient`` has since been fixed to include the normalization
chain rule (see tests/solver/test_freesolver_gradient_consistency.py), so
both now agree.

Raggedness-free design:
  - fixed-count inner rounds (ROUND_SIZE iters), host sync only at round
    boundaries (single mx.eval + one bool readback)
  - per-pixel convergence mask freezes finished pixels inside the graph
  - projection is closed-form (fixed-count bisection)
  - step size from per-pixel power iteration on the gradient-map Jacobian
    (mx.jvp), once per loop — no line search
  - float32 GPU throughout; final objective for acceptance tests is
    re-evaluated in float64 on the CPU by the caller (adapter)
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mlx.core as mx

from .projections import project_capped_simplex

ROUND_SIZE = 50          # iters between host syncs
POWER_ITERS = 15         # power-iteration count for step estimation
STEP_SAFETY = 0.9        # step = SAFETY / L_hat


@dataclass
class FreeBatchConfig:
    """Stage A: {'LLS','L2'} x OLS/WLS. Stage B adds 'L1' (log-cosh),
    'MEM' (I-divergence), and misfit='poisson' (IRLS majorization)."""
    method: str = 'L2'             # 'LLS' | 'L2' | 'L1' | 'MEM'
    alpha: float = 1.0
    degree: float = 1.0
    max_iter: int = 350            # per loop (operational default)
    n_loops: int = 1
    degree_decay: float = 0.8
    min_conc: float = 1e-6
    max_conc: float = 1.0
    conv_tol: float = 1e-3         # projected-gradient-mapping residual
    round_size: int = ROUND_SIZE
    compile_step: bool = True      # benchmark records on/off difference
    # Optional preconditioning (default off):
    # diagonal metric M = diag(T^T T) + alpha * mu_R for the projection.
    use_diag_metric: bool = False
    # --- Stage B ---
    l1_beta: float = 0.01          # log-cosh smoothing (FreeSolver l1_beta)
    misfit: str = 'ols'            # 'ols' | 'poisson' (WLS via `weights` arg)
    # Poisson-NLL via IRLS majorization: outer Fisher-
    # scoring weight updates w = 1/max(I_pred, floor), inner WLS-FISTA.
    # Fixed outer cap (raggedness-free); early break when weights settle.
    # The IRLS fixed point is a stationary point of the true deviance, but
    # the iteration converges only linearly: the max relative weight change
    # between rounds shrinks ~3x per round (0.77, 0.13, 0.057, 0.018,
    # 0.005, 0.002, 7e-4 on the 6elem_standard 5x5 test problem, both
    # compound tables, 2026-10-18) and the solution stops moving at round
    # 7-8, where it reaches the floor set by the inner conv_tol. The former
    # cap of 4 rounds stopped early (D still ~2% from the converged
    # reference on one pixel); 0.02 would have stopped at round 5 (1.3%).
    # 1e-3 is where D has settled; 16 bounds the cost if the weights never
    # settle below it (the floor of the weight change is ~conv_tol). Both
    # are accuracy/cost tuning values with no physical meaning.
    poisson_outer: int = 16
    poisson_floor: float = 1e-6    # tied to the physical box lower bound
    poisson_wtol: float = 1e-3     # relative weight change to declare settled
    # Observable-subspace preconditioner:
    # per-element P_e = V_r diag(s1^2/s_i^2) V_r^T + (I - V_r V_r^T) built
    # once from the shared T's SVD (CPU, float64) and applied to the
    # gradient as one matvec. Equalizes the observable modes' convergence;
    # null-space modes keep unit metric (they don't affect f).
    subspace_precond: Optional[np.ndarray] = None   # (n_depth, n_depth, n_elem)


@dataclass
class FreeBatchResult:
    D_opt: np.ndarray              # (B, n_depth, n_elem)
    I_opt: np.ndarray              # (B, n_angles, n_elem), normalized prediction
    chi2: np.ndarray               # (B,) float64, at final degree weighting = 1
    objective: np.ndarray          # (B,) float64 reference re-evaluation
    n_iterations: int              # total iterations executed (all loops)
    converged: np.ndarray          # (B,) bool at final round
    elapsed_time: float = 0.0
    timings: dict = field(default_factory=dict)  # component budget


def _forward_norm(T: mx.array, D: mx.array) -> mx.array:
    """I_pred_norm[b,a,e] = normalize_a(sum_d T[a,d,e] D[b,d,e])."""
    I = mx.einsum('ade,bde->bae', T, D)
    s = mx.sum(I, axis=-1, keepdims=True)
    return I / mx.maximum(s, 1e-10)


def _make_reg_px_fn(D_prior, method, alpha, l1_beta):
    """Per-pixel regularization alpha*R_b(D). Conventions = FreeSolver."""
    if method == 'L2':
        def reg(D):
            dd = D - D_prior
            return alpha * mx.sum(dd * dd, axis=(1, 2))
    elif method == 'L1':
        # Exact |D - m0|_1 (FreeSolver default l1_smoothing='none').
        # On the box D >= min_conc >= m0 this is smooth (= D - m0); with the
        # sum-to-one constraint and m0 = min_conc it is affine (~constant),
        # i.e. L1-Free with a near-zero prior is mathematically LLS + const.
        # The log-cosh variant (l1_beta) is deliberately NOT used here: its
        # clip(±20) plateau makes the reference objective non-convex and its
        # minima are smoothing artifacts.
        def reg(D):
            return alpha * mx.sum(mx.abs(D - D_prior), axis=(1, 2))
    elif method == 'MEM':         # I-divergence (concentration space)
        Dp = mx.maximum(D_prior, 1e-10)
        def reg(D):
            Ds = mx.maximum(D, 1e-10)
            return alpha * mx.sum(Ds * mx.log(Ds / Dp) - Ds + Dp, axis=(1, 2))
    else:                         # LLS
        def reg(D):
            return mx.zeros(D.shape[0], dtype=D.dtype)
    return reg


def _make_obj_px_fn(T, I_obs, W, reg_fn, misfit, poisson_floor=1e-6):
    """Per-pixel objective f_b(D) = 0.5*degree*misfit_b + alpha*R_b.

    misfit 'ols'/'wls': sum(W * (I_pred_norm - I_obs)^2)
    misfit 'poisson':   Poisson deviance
        2*sum[I_obs*log(I_obs/I_pred) - I_obs + I_pred]  (0 where I_obs=0)
    """
    if misfit == 'poisson':
        I_safe = mx.maximum(I_obs, 0.0)
        log_obs = mx.log(mx.maximum(I_obs, poisson_floor))
        def obj_px(D, degree):
            Ip = mx.maximum(_forward_norm(T, D), poisson_floor)
            dev = I_safe * (log_obs - mx.log(Ip)) - I_safe + Ip
            return degree * mx.sum(dev, axis=(1, 2)) + reg_fn(D)
    else:
        def obj_px(D, degree):
            R = _forward_norm(T, D) - I_obs
            return 0.5 * degree * mx.sum(W * R * R, axis=(1, 2)) + reg_fn(D)
    return obj_px


def _make_grad_fn(obj_px_fn):
    """Exact batched gradient: pixels are independent, so grad of the batch
    sum gives every per-pixel gradient in one backward pass."""
    return mx.grad(lambda D, degree: mx.sum(obj_px_fn(D, degree)))


def _estimate_step(grad_fn, D0, degree, key=None) -> mx.array:
    """Per-pixel 1/L via power iteration on J_g at D0 using mx.jvp."""
    B = D0.shape[0]
    v = mx.random.normal(D0.shape, key=key)
    v = v / mx.maximum(mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True)), 1e-30)
    fn = lambda D: grad_fn(D, degree)
    L = mx.ones((B,))
    for _ in range(POWER_ITERS):
        jv = mx.jvp(fn, [D0], [v])[1][0]
        L = mx.sqrt(mx.sum(jv * jv, axis=(1, 2)))
        v = jv / mx.maximum(L[:, None, None], 1e-30)
    # Guard degenerate pixels (L ~ 0): fall back to batch max
    L = mx.maximum(L, 1e-12 * mx.max(L))
    return (STEP_SAFETY / L)[:, None, None]


def solve_free_batch(
    T: np.ndarray,
    I_obs_norm: np.ndarray,
    D_init: np.ndarray,
    D_prior: np.ndarray,
    config: FreeBatchConfig,
    weights: Optional[np.ndarray] = None,
) -> FreeBatchResult:
    """Solve B independent Free problems with shared T.

    Args:
        T: (n_angles, n_depth, n_elem) shared transfer matrix.
        I_obs_norm: (B, n_angles, n_elem) normalized observations.
        D_init: (B, n_depth, n_elem) initial guess.
        D_prior: (n_depth, n_elem) or (B, n_depth, n_elem) prior model.
        config: solver configuration.
        weights: optional WLS weights, broadcastable to (B, n_angles, n_elem).
            chi2 = sum(weights * residual^2); gradient uses the same weights.
    """
    import time as _time
    t_start = _time.perf_counter()

    if config.method not in ('L2', 'LLS', 'L1', 'MEM'):
        raise NotImplementedError(f"unknown method {config.method!r}")
    if config.misfit not in ('ols', 'poisson'):
        raise NotImplementedError(f"unknown misfit {config.misfit!r}")

    f32 = mx.float32
    T_mx = mx.array(np.ascontiguousarray(T), dtype=f32)
    I_mx = mx.array(np.ascontiguousarray(I_obs_norm), dtype=f32)
    D = mx.array(np.ascontiguousarray(D_init), dtype=f32)
    Dp = mx.array(np.ascontiguousarray(np.broadcast_to(
        D_prior, D_init.shape) if D_prior.ndim == 2 else D_prior), dtype=f32)
    # WLS enters chi2 as w*r^2 -> gradient carries w once
    W = mx.ones((1, 1, 1), dtype=f32) if weights is None else \
        mx.array(np.ascontiguousarray(weights), dtype=f32)

    B = D.shape[0]
    lo, hi = config.min_conc, config.max_conc

    metric = None
    if config.use_diag_metric:
        # Full-Hessian diagonal: diag(T^T T) summed misfit + reg curvature
        diag_TtT = mx.sum(T_mx * T_mx, axis=0)              # (n_depth, n_elem)
        mu_R = 2.0 if config.method == 'L2' else 0.0
        metric = config.degree * diag_TtT + config.alpha * mu_R + 1e-12

    reg_fn = _make_reg_px_fn(Dp, config.method, config.alpha, config.l1_beta)
    # True objective — used for best-tracking and final reporting always
    # (for misfit='poisson' the inner gradient comes from an IRLS-WLS
    # surrogate, but the tracked/returned solution is judged on the true f).
    obj_fn = _make_obj_px_fn(T_mx, I_mx, W, reg_fn, config.misfit,
                             config.poisson_floor)

    P_pre = None
    if config.subspace_precond is not None:
        P_pre = mx.array(np.ascontiguousarray(config.subspace_precond),
                         dtype=f32)          # (n_depth, n_depth, n_elem)

    def apply_precond(grad_fn):
        if P_pre is None:
            return grad_fn
        def g(D, degree):
            return mx.einsum('dfe,bfe->bde', P_pre, grad_fn(D, degree))
        return g

    def project(V):
        return project_capped_simplex(V, lo=lo, hi=hi, target=1.0, metric=metric)

    # Start feasible: the reference's x0 may violate the sum constraint
    # (e.g. LLS prior = min_conc everywhere); best-tracking must never
    # compare against an infeasible point's objective.
    D = project(D)

    timings = {'step_total': 0.0, 'check_total': 0.0, 'setup': 0.0}
    total_iters = 0
    converged = mx.zeros((B,), dtype=mx.bool_)
    t_setup = _time.perf_counter()

    def fista_sweep(D, grad_fn, degree, max_iter):
        """One full FISTA run (rounds + masked convergence + step adapt).

        MFISTA-style best-tracking makes the returned solution immune to
        the limit cycles a slightly overestimated step can induce — the
        momentum path stays pure FISTA, but the output is the best point
        visited (per pixel, elementwise, no host sync inside rounds).
        """
        nonlocal total_iters, timings

        def step_fn(D, Y, t, mask, f_best, D_best, step, degree):
            G = grad_fn(Y, degree)
            D_new = project(Y - step * G)
            f_new = obj_fn(D_new, degree)
            better = mx.logical_and(f_new < f_best, mx.logical_not(mask))
            b3 = better[:, None, None]
            D_best = mx.where(b3, D_new, D_best)
            f_best = mx.where(better, f_new, f_best)
            # Adaptive restart (per pixel): momentum against descent direction
            restart = mx.sum((Y - D_new) * (D_new - D), axis=(1, 2)) > 0.0
            t_eff = mx.where(restart, mx.ones_like(t), t)
            t_new = 0.5 * (1.0 + mx.sqrt(1.0 + 4.0 * t_eff * t_eff))
            Y_new = D_new + ((t_eff - 1.0) / t_new)[:, None, None] * (D_new - D)
            m3 = mask[:, None, None]
            return (mx.where(m3, D, D_new),
                    mx.where(m3, Y, Y_new),
                    mx.where(mask, t, t_new),
                    mask, f_best, D_best)

        step_c = mx.compile(step_fn) if config.compile_step else step_fn

        def residual_map(D, step, degree):
            """Projected-gradient-mapping residual, scale-normalized."""
            G = grad_fn(D, degree)
            P = project(D - step * G)
            num = mx.sqrt(mx.sum((D - P) ** 2, axis=(1, 2)))
            den = mx.sqrt(mx.sum((step * G) ** 2, axis=(1, 2))) + 1e-12
            return num / den

        step = _estimate_step(grad_fn, D, degree)
        mx.eval(step)
        Y = D
        t_mom = mx.ones((B,))
        mask = mx.zeros((B,), dtype=mx.bool_)
        degree_mx = mx.array(degree, dtype=f32)
        f_best = obj_fn(D, degree_mx)
        D_best = D
        step_min = step * (2.0 ** -8)
        res_prev = mx.full((B,), np.inf)
        n_rounds = max(1, (max_iter + config.round_size - 1)
                       // config.round_size)
        it = 0
        for _ in range(n_rounds):
            k = min(config.round_size, max_iter - it)
            t0 = _time.perf_counter()
            for _ in range(k):
                D, Y, t_mom, mask, f_best, D_best = step_c(
                    D, Y, t_mom, mask, f_best, D_best, step, degree_mx)
            mx.eval(D, D_best)               # single sync per round
            timings['step_total'] += _time.perf_counter() - t0
            it += k
            total_iters += k

            t0 = _time.perf_counter()
            res = residual_map(D_best, step, degree_mx)
            mask = mx.logical_or(mask, res < config.conv_tol)
            # Round-level step adaptation (elementwise, reuses this sync):
            # a pixel whose residual INCREASED over a round is in a limit
            # cycle from an overestimated step — halve its step and restart
            # from its best point. (A slower threshold like res >= 0.7*prev
            # misfires on legitimately slow ill-conditioned problems — LLS —
            # cascading the step to its floor and freezing progress.)
            stalled = mx.logical_and(res >= 1.05 * res_prev,
                                     mx.logical_not(mask))
            s3 = stalled[:, None, None]
            step = mx.maximum(mx.where(s3, step * 0.5, step), step_min)
            D = mx.where(s3, D_best, D)
            Y = mx.where(s3, D_best, Y)
            t_mom = mx.where(stalled, mx.ones_like(t_mom), t_mom)
            res_prev = res
            all_done = mx.all(mask)
            mx.eval(all_done)
            timings['check_total'] += _time.perf_counter() - t0
            if bool(all_done.item()):        # round-boundary early exit
                break
        return D_best, mask

    if config.misfit == 'poisson':
        # IRLS majorization: Fisher-scoring weights
        # w = 1/max(I_pred, floor), inner WLS-FISTA, fixed outer cap.
        w_prev = None
        for outer in range(config.poisson_outer):
            Ip = mx.maximum(_forward_norm(T_mx, D), config.poisson_floor)
            w_irls = 1.0 / Ip
            surrogate_obj = _make_obj_px_fn(T_mx, I_mx, w_irls, reg_fn, 'ols')
            grad_fn = apply_precond(_make_grad_fn(surrogate_obj))
            if outer == 0:
                timings['setup'] = _time.perf_counter() - t_setup
            D, converged = fista_sweep(D, grad_fn, config.degree,
                                       config.max_iter)
            if w_prev is not None:
                relw = mx.max(mx.abs(w_irls - w_prev) / w_prev)
                mx.eval(relw)
                if float(relw.item()) < config.poisson_wtol:
                    break
            w_prev = w_irls
        degree = config.degree                # loops/decay not used for poisson
    else:
        grad_fn = apply_precond(_make_grad_fn(obj_fn))
        degree = config.degree
        timings['setup'] = _time.perf_counter() - t_setup
        for loop in range(config.n_loops):
            D, converged = fista_sweep(D, grad_fn, degree, config.max_iter)
            degree *= config.degree_decay

    # Outputs: final prediction + float64 CPU re-evaluation of the objective
    I_opt = _forward_norm(T_mx, D)
    mx.eval(I_opt, converged)

    D_np = np.array(D, dtype=np.float64)
    I_opt_np = np.array(I_opt, dtype=np.float64)
    I_obs64 = np.asarray(I_obs_norm, dtype=np.float64)
    W_np = np.ones((1, 1, 1)) if weights is None else np.asarray(weights, dtype=np.float64)
    if config.misfit == 'poisson':
        Ip = np.maximum(I_opt_np, config.poisson_floor)
        Io = np.maximum(I_obs64, 0.0)
        log_ratio = np.log(np.maximum(I_obs64, config.poisson_floor)) - np.log(Ip)
        half_mis = np.sum(Io * log_ratio - Io + Ip, axis=(1, 2))
        chi2 = 2.0 * half_mis                 # full deviance in the result
    else:
        resid = I_opt_np - I_obs64
        chi2 = np.sum(W_np * resid ** 2, axis=(1, 2))
        half_mis = 0.5 * chi2
    Dp_np = np.asarray(np.broadcast_to(D_prior, D_np.shape), dtype=np.float64)
    if config.method == 'L2':
        reg = np.sum((D_np - Dp_np) ** 2, axis=(1, 2))
    elif config.method == 'L1':
        reg = np.sum(np.abs(D_np - Dp_np), axis=(1, 2))
    elif config.method == 'MEM':
        Ds = np.maximum(D_np, 1e-10)
        Dps = np.maximum(Dp_np, 1e-10)
        reg = np.sum(Ds * np.log(Ds / Dps) - Ds + Dps, axis=(1, 2))
    else:
        reg = np.zeros(B)
    # Reference objective at the *final* loop's degree (matches FreeSolver's
    # last minimize call convention)
    n_decays = 0 if config.misfit == 'poisson' else config.n_loops - 1
    final_degree = config.degree * (config.degree_decay ** n_decays)
    objective = final_degree * half_mis + config.alpha * reg

    return FreeBatchResult(
        D_opt=D_np,
        I_opt=I_opt_np,
        chi2=chi2,
        objective=objective,
        n_iterations=total_iters,
        converged=np.array(converged),
        elapsed_time=_time.perf_counter() - t_start,
        timings=timings,
    )
