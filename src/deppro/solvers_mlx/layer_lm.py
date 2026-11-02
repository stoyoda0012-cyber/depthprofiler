"""Batched Levenberg-Marquardt for the GenericLayerSolver problem (Stage D).

Solves, for B independent pixels sharing one transfer matrix T, the
GenericLayerSolver objective (plain OLS, total-normalized):

    min_x  chi2(x) = sum_{a,e} (I_pred_norm(x) - I_obs_norm)^2
    s.t.   thickness/width bounds

where x = [thicknesses (n_layers), (widths if optimize_widths)] and the
depth profile is built exactly as GenericLayerSolver._build_profile for a
FIXED stoichiometry: each layer contributes an erf window, the substrate an
erf step, weighted by precomputed per-element coefficients, then clipped to
[min_conc, inf) and normalized to sum 1 at each depth.

For fixed stoichiometry the profile is LINEAR in the per-layer erf windows:

    D[b,z,e] = sum_i window_i(x)[b,z] * Ccoef[i,e] + step(x)[b,z] * Csub[e]

so a scatter-free, autodiff-friendly forward is available (avoids the MLX
0.30 mx.take-JVP gap hit in Stage C). The nonlinearity is entirely in the
erf window positions/widths — a small (~1-8) bounded NLLS, the natural home
of Levenberg-Marquardt.

Design:
  - bounds via sigmoid reparametrization x = lb + (ub-lb)*sigmoid(u); LM runs
    on the unconstrained u. Init u maps x0 to the bound interior and every
    iterate is clipped to |u| <= U_CLIP so the sigmoid gradient never
    vanishes (saturation guard).
  - Jacobian J = dr/du via forward-mode (mx.jvp, one push per parameter);
    n_p is tiny so this beats reverse-mode over the ~n_angles*n_elem outputs.
  - normal equations (J^T J + lambda * diag(J^T J)) delta = -J^T r solved on
    the CPU stream (mx.linalg is CPU-only; the batched n_p x n_p solve is
    small, so this is the one deliberate CPU step in the hot path).
    Marquardt diagonal scaling (not lambda*I) because sigmoid columns differ
    in scale by orders of magnitude (thickness vs width sensitivity).
  - lambda update = masked accept/reject (fixed rounds, no ragged control
    flow): accept where chi2 drops (lambda /= DOWN), reject elsewhere
    (lambda *= UP, iterate restored). All batched, host sync once per round.

Raggedness-free: fixed iteration count, masked accept/reject, per-pixel
convergence freeze; only the CPU solve leaves the GPU (unified memory).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mlx.core as mx

_FWHM2SIG = 1.0 / (2.0 * np.sqrt(np.log(2.0)))
U_CLIP = 8.0            # |u| clip: sigmoid'(8) ~ 3e-4, still nonzero
LAMBDA0 = 1.0          # conservative start: weakly-constrained directions
                       # (e.g. a buried layer's thickness) are near-flat, so
                       # an aggressive first step overshoots to a bound and
                       # lands at a worse minimum than the reference. Heavy
                       # initial damping keeps early steps gradient-like.
LAMBDA_UP = 3.0
LAMBDA_DOWN = 3.0
LAMBDA_MIN = 1e-9
LAMBDA_MAX = 1e9


@dataclass
class LayerLMConfig:
    optimize_widths: bool = False
    thickness_bounds: tuple = (1.0, 200.0)
    width_bounds: tuple = (0.5, 30.0)
    # LM usually converges well before this cap; the default leaves headroom
    # for harder initial values and more layers at some cost in throughput.
    max_iter: int = 30
    min_conc: float = 1e-6
    lambda0: float = LAMBDA0
    conv_tol: float = 1e-9       # relative chi2 decrease to freeze a pixel
    round_check: int = 5         # host-sync convergence check cadence


@dataclass
class LayerLMResult:
    D_opt: np.ndarray               # (B, n_depth, n_elem)
    I_opt: np.ndarray               # (B, n_angles, n_elem), total-normalized
    layer_thicknesses: np.ndarray   # (B, n_layers)
    interface_widths: np.ndarray    # (B, n_layers)
    chi2: np.ndarray                # (B,) float64
    n_iterations: int
    converged: np.ndarray           # (B,) bool
    elapsed_time: float = 0.0
    timings: dict = field(default_factory=dict)


def build_layer_coefficients(layer_config, n_elem):
    """Precompute (Ccoef, Csub) matching GenericLayerSolver._build_profile
    for FIXED stoichiometry (charge-conservation reset included).

    Ccoef: (n_layers, n_elem) — layer window weights.
    Csub:  (n_elem,)          — substrate step weights.
    Returns also profile_types list (must all be 'erf' for this solver).
    """
    cfg = layer_config
    n_layers = cfg.n_layers
    Ccoef = np.zeros((n_layers, n_elem))
    for i, layer in enumerate(cfg.layers):
        if layer.profile_type not in ('erf',):
            raise NotImplementedError(
                f"layer '{layer.name}' profile_type={layer.profile_type!r}: "
                "Stage D batched LM supports 'erf' windows only")
        total = sum(layer.stoichiometry.values())
        for e, s in layer.stoichiometry.items():
            Ccoef[i, e] += layer.base_concentration * (s / total)
    Csub = np.zeros(n_elem)
    total_sub = sum(cfg.substrate.stoichiometry.values())
    for e, s in cfg.substrate.stoichiometry.items():
        Csub[e] = s / total_sub
    # linked element: the reference resets D[:, linked] before summing the
    # layer contributions, erasing any substrate contribution to it.
    if cfg.linked_element_index is not None:
        Csub[cfg.linked_element_index] = 0.0
    return Ccoef, Csub


def _make_forward(T, depth, Ccoef, Csub, n_layers, optimize_widths,
                  fixed_sigma, min_conc):
    """Batched differentiable forward. u -> (I_pred_norm_flat, D).

    Returns a function of the unconstrained parameter u (B, n_p) plus the
    sigmoid map arrays (lb, span), producing normalized predictions.
    """
    z = depth[None, :]                              # (1, Z)

    def build_D(x):
        """x: (B, n_p) physical params -> D (B, Z, E)."""
        thick = x[:, :n_layers]                     # (B, L)
        if optimize_widths:
            width = x[:, n_layers:2 * n_layers]
            sigma = mx.maximum(width * _FWHM2SIG, 1e-6)
        else:
            sigma = fixed_sigma[None, :]            # (1, L)
            sigma = mx.broadcast_to(sigma, thick.shape)
        z_end = mx.cumsum(thick, axis=1)            # (B, L)
        windows = []
        for i in range(n_layers):
            sig_end = sigma[:, i:i + 1]
            falling = 0.5 * (1.0 + mx.erf((z - z_end[:, i:i + 1]) / sig_end))
            if i == 0:
                prof = mx.maximum(1.0 - falling, 0.0)
            else:
                sig_start = sigma[:, i - 1:i]
                rising = 0.5 * (1.0 + mx.erf(
                    (z - z_end[:, i - 1:i]) / sig_start))
                prof = mx.maximum(rising - falling, 0.0)
            windows.append(prof)
        W = mx.stack(windows, axis=-1)              # (B, Z, L)
        sig_sub = sigma[:, n_layers - 1:n_layers]
        step = 0.5 * (1.0 + mx.erf((z - z_end[:, n_layers - 1:n_layers])
                                   / sig_sub))       # (B, Z)
        D = mx.einsum('bzi,ie->bze', W, Ccoef) + step[:, :, None] * Csub
        D = mx.maximum(D, min_conc)
        D = D / mx.maximum(mx.sum(D, axis=-1, keepdims=True), 1e-10)
        return D

    def forward_norm(x):
        D = build_D(x)
        I = mx.einsum('ade,bde->bae', T, D)         # (B, A, E)
        I = I / mx.maximum(mx.sum(I, axis=-1, keepdims=True), 1e-10)
        return I

    return build_D, forward_norm


def solve_layer_lm(
    T: np.ndarray,
    I_obs_norm: np.ndarray,
    layer_config,
    x0: np.ndarray,
    config: LayerLMConfig,
) -> LayerLMResult:
    """Batched LM for B GenericLayer pixels sharing T.

    Args:
        T: (n_angles, n_depth, n_elem) shared transfer matrix.
        I_obs_norm: (B, n_angles, n_elem) total-normalized observations.
        layer_config: GenericLayerConfig (fixed stoichiometry).
        x0: (n_p,) or (B, n_p) initial physical params
            [thicknesses, (widths)].
        config: LayerLMConfig.
    """
    import time as _time
    t_start = _time.perf_counter()

    if layer_config.optimize_stoichiometry and \
            layer_config.has_variable_stoichiometry:
        raise NotImplementedError(
            "Stage D batched LM supports fixed stoichiometry only "
            "(optimize_stoichiometry=True is out of scope)")

    f32 = mx.float32
    B, n_angles, n_elem = I_obs_norm.shape
    n_depth = np.asarray(T).shape[1]
    n_layers = layer_config.n_layers
    optimize_widths = config.optimize_widths
    n_p = n_layers * (2 if optimize_widths else 1)

    Ccoef_np, Csub_np = build_layer_coefficients(layer_config, n_elem)

    # depth grid: use the transfer matrix's own depth via caller (passed in
    # x0 space is params). depth must match _build_profile — supplied by the
    # adapter as layer_config carries none; we reconstruct from T alignment.
    depth_np = _resolve_depth(layer_config, n_depth)

    T_mx = mx.array(np.ascontiguousarray(T), dtype=f32)
    depth_mx = mx.array(depth_np, dtype=f32)
    Ccoef = mx.array(Ccoef_np, dtype=f32)
    Csub = mx.array(Csub_np, dtype=f32)
    I_obs = mx.array(np.ascontiguousarray(I_obs_norm), dtype=f32)

    # Bounds
    lb_list = [config.thickness_bounds[0]] * n_layers
    ub_list = [config.thickness_bounds[1]] * n_layers
    if optimize_widths:
        lb_list += [config.width_bounds[0]] * n_layers
        ub_list += [config.width_bounds[1]] * n_layers
    lb = mx.array(np.array(lb_list), dtype=f32)[None, :]     # (1, n_p)
    span = mx.array(np.array(ub_list) - np.array(lb_list), dtype=f32)[None, :]

    fixed_sigma = None
    if not optimize_widths:
        fixed_sigma = mx.array(
            np.asarray(layer_config.initial_widths) * _FWHM2SIG, dtype=f32)

    _, forward_norm = _make_forward(
        T_mx, depth_mx, Ccoef, Csub, n_layers, optimize_widths,
        fixed_sigma, config.min_conc)

    # x = lb + span * sigmoid(u); optimize u.
    def x_of_u(u):
        return lb + span * mx.sigmoid(u)

    def resid(u):
        x = x_of_u(u)
        r = forward_norm(x) - I_obs               # (B, A, E)
        return mx.reshape(r, (B, n_angles * n_elem))

    def chi2_of(u):
        r = resid(u)
        return mx.sum(r * r, axis=1)              # (B,)

    # Init u from x0 (mapped into the interior, clipped away from saturation)
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.ndim == 1:
        x0 = np.broadcast_to(x0, (B, n_p))
    lb_np = np.array(lb_list)
    span_np = np.array(ub_list) - lb_np
    frac = np.clip((x0 - lb_np) / np.maximum(span_np, 1e-12), 1e-3, 1 - 1e-3)
    u0 = np.log(frac / (1.0 - frac))
    u = mx.array(np.clip(u0, -U_CLIP, U_CLIP), dtype=f32)

    lam = mx.full((B,), config.lambda0, dtype=f32)
    chi2 = chi2_of(u)
    mx.eval(u, chi2)
    frozen = mx.zeros((B,), dtype=mx.bool_)
    eye = mx.eye(n_p, dtype=f32)[None, :, :]       # (1, p, p)

    timings = {'jac': 0.0, 'solve': 0.0, 'eval': 0.0}

    def jacobian(u):
        """J = d resid / d u, (B, n_res, n_p) via n_p forward pushes."""
        cols = []
        for j in range(n_p):
            tan = np.zeros((B, n_p), dtype=np.float32)
            tan[:, j] = 1.0
            _, jv = mx.jvp(resid, [u], [mx.array(tan)])
            cols.append(jv[0])                     # (B, n_res)
        return mx.stack(cols, axis=-1)             # (B, n_res, n_p)

    n_iter = 0
    for it in range(config.max_iter):
        t0 = _time.perf_counter()
        r = resid(u)                               # (B, n_res)
        J = jacobian(u)                            # (B, n_res, n_p)
        mx.eval(r, J)
        timings['jac'] += _time.perf_counter() - t0

        g = mx.einsum('brp,br->bp', J, r)          # (B, p)  = J^T r
        H = mx.einsum('brp,brq->bpq', J, J)        # (B, p, p) = J^T J
        # Levenberg damping A = H + lambda * I.
        # Marquardt lambda*diag(H) was tried first but on a near-flat
        # direction diag(H) -> 0 makes the damping vanish there, letting the
        # step run to a bound and settle at a worse minimum than the
        # reference (for example, a buried-layer thickness running to its
        # upper bound). lambda*I damps flat directions uniformly.
        A = H + lam[:, None, None] * eye + 1e-12 * eye

        t0 = _time.perf_counter()
        # CPU-stream batched solve: A delta = -g  (plan's hot-path exception)
        delta = mx.linalg.solve(A, -g[:, :, None], stream=mx.cpu)[:, :, 0]
        mx.eval(delta)
        timings['solve'] += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        u_new = mx.clip(u + delta, -U_CLIP, U_CLIP)
        chi2_new = chi2_of(u_new)
        accept = mx.logical_and(chi2_new < chi2, mx.logical_not(frozen))
        a1 = accept[:, None]
        u = mx.where(a1, u_new, u)
        # lambda: down on accept (toward Gauss-Newton), up on reject
        lam = mx.where(accept, lam / LAMBDA_DOWN, lam * LAMBDA_UP)
        lam = mx.clip(lam, LAMBDA_MIN, LAMBDA_MAX)
        rel_drop = (chi2 - chi2_new) / mx.maximum(chi2, 1e-30)
        chi2 = mx.where(accept, chi2_new, chi2)
        mx.eval(u, lam, chi2)
        timings['eval'] += _time.perf_counter() - t0
        n_iter += 1

        # Freeze pixels whose accepted step barely moved chi2 (converged).
        newly = mx.logical_and(accept, rel_drop < config.conv_tol)
        frozen = mx.logical_or(frozen, newly)
        if (it + 1) % config.round_check == 0:
            all_done = mx.all(frozen)
            mx.eval(all_done)
            if bool(all_done.item()):
                break

    # Final products in float64 (acceptance-test currency).
    x_final = np.array(x_of_u(u), dtype=np.float64)
    thick = x_final[:, :n_layers]
    if optimize_widths:
        width = x_final[:, n_layers:2 * n_layers]
    else:
        width = np.broadcast_to(
            np.asarray(layer_config.initial_widths, dtype=np.float64),
            (B, n_layers)).copy()

    D_np, I_np, chi2_np = _forward_np(
        np.asarray(T, dtype=np.float64), depth_np, Ccoef_np, Csub_np,
        thick, width, np.asarray(I_obs_norm, dtype=np.float64),
        config.min_conc, n_layers)

    return LayerLMResult(
        D_opt=D_np,
        I_opt=I_np,
        layer_thicknesses=thick,
        interface_widths=width,
        chi2=chi2_np,
        n_iterations=n_iter,
        converged=np.array(frozen),
        elapsed_time=_time.perf_counter() - t_start,
        timings=timings,
    )


def _resolve_depth(layer_config, n_depth):
    """The GenericLayer profile is built on the transfer matrix's depth grid.
    The adapter attaches it as ``layer_config._depth`` (set at construction);
    fall back to an index grid only if absent (tests set it explicitly)."""
    d = getattr(layer_config, '_depth', None)
    if d is not None:
        return np.asarray(d, dtype=np.float64)
    raise ValueError("layer_config._depth not set; the adapter must attach "
                     "the transfer matrix depth grid before solving")


def _forward_np(T, depth, Ccoef, Csub, thick, width, I_obs_norm,
                min_conc, n_layers):
    """float64 reference-exact batched forward (mirrors build_D + normalize)."""
    B = thick.shape[0]
    n_elem = Ccoef.shape[1]
    sigma = np.maximum(width * _FWHM2SIG, 1e-6)          # (B, L)
    z_end = np.cumsum(thick, axis=1)                      # (B, L)
    z = depth[None, :]
    from scipy.special import erf
    W = np.zeros((B, len(depth), n_layers))
    for i in range(n_layers):
        falling = 0.5 * (1 + erf((z - z_end[:, i:i + 1]) / sigma[:, i:i + 1]))
        if i == 0:
            W[:, :, i] = np.maximum(1 - falling, 0.0)
        else:
            rising = 0.5 * (1 + erf(
                (z - z_end[:, i - 1:i]) / sigma[:, i - 1:i]))
            W[:, :, i] = np.maximum(rising - falling, 0.0)
    step = 0.5 * (1 + erf((z - z_end[:, n_layers - 1:n_layers])
                          / sigma[:, n_layers - 1:n_layers]))
    D = np.einsum('bzi,ie->bze', W, Ccoef) + step[:, :, None] * Csub
    D = np.maximum(D, min_conc)
    D = D / np.maximum(D.sum(axis=-1, keepdims=True), 1e-10)
    I = np.einsum('ade,bde->bae', T, D)
    I = I / np.maximum(I.sum(axis=-1, keepdims=True), 1e-10)
    chi2 = np.sum((I - I_obs_norm) ** 2, axis=(1, 2))
    return D, I, chi2
