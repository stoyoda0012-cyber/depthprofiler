"""Adapter: FreeSolver/SemiFreeSolver/GenericLayerSolver-compatible facades
over the MLX batched solvers.

Selection is explicit (``MLXBatchFreeSolver`` / ``MLXBatchSemiFreeSolver`` /
``MLXBatchGenericLayerSolver``) — the SciPy paths remain untouched defaults
everywhere else.
"""
import dataclasses
from typing import List, Optional

import numpy as np

from ..inverse_solver import (
    GenericLayerResult, SemiFreeSolver, SemiFreeSolverConfig, SolverConfig,
    SolverResult,
)
from .free_batch import FreeBatchConfig, FreeBatchResult, solve_free_batch
from .semifree_batch import (
    SemiFreeBatchConfig, SemiFreeBatchResult, solve_semifree_batch,
)
from .layer_lm import LayerLMConfig, LayerLMResult, solve_layer_lm, \
    build_layer_coefficients

# FISTA iteration budget is a different currency from SLSQP's SQP-iteration
# max_iter; 350 is the operational default.
DEFAULT_FISTA_MAX_ITER = 350

# Auto chunk size: large batches are split to limit the peak memory;
# 10K pixels per chunk is the default.
CHUNK_PX_AUTO = 10_000


class MLXBatchFreeSolver:
    """Batched drop-in for FreeSolver (Stage A: methods 'L2' / 'LLS').

    Usage (batch — the intended path):
        solver = MLXBatchFreeSolver(tm, config, D_prior=None)
        batch = solver.solve_batch(I_obs_stack)   # (B, n_angles, n_elem)

    Usage (single pixel, drop-in FreeSolver signature):
        result = solver.solve(I_obs)              # SolverResult
    """

    def __init__(
        self,
        transfer_matrix,
        config: Optional[SolverConfig] = None,
        D_prior: Optional[np.ndarray] = None,
        elements: Optional[List] = None,
        fista_max_iter: int = DEFAULT_FISTA_MAX_ITER,
        conv_tol: float = 1e-3,
        compile_step: bool = True,
        misfit: str = 'ols',
        chunk_px: Optional[int] = None,
    ):
        # chunk_px: split batches larger than this to bound peak GPU memory
        # None = auto (CHUNK_PX_AUTO).
        self.chunk_px = chunk_px if chunk_px is not None else CHUNK_PX_AUTO
        self.tm = transfer_matrix
        self.config = config or SolverConfig()
        if elements is not None and self.config.use_charge_constraint:
            # Charge constraint needs a 2nd equality per depth row: Stage B/C
            # scope (Dykstra or 2-D dual). The d2s2d free path does not use it.
            raise NotImplementedError(
                "mlx_batch Stage A supports the sum-to-one constraint only; "
                "pass elements=None or use_charge_constraint=False")

        n_depth = len(transfer_matrix.depth)
        n_elem = transfer_matrix.n_elements
        if D_prior is not None:
            self.D_prior = np.maximum(D_prior.copy(), self.config.min_conc)
        elif self.config.method in ('L1', 'LLS'):
            self.D_prior = np.full((n_depth, n_elem), self.config.min_conc)
        else:
            self.D_prior = np.ones((n_depth, n_elem)) / n_elem

        self.batch_config = FreeBatchConfig(
            method=self.config.method,
            alpha=self.config.alpha,
            degree=self.config.degree,
            max_iter=fista_max_iter,
            n_loops=self.config.n_loops,
            degree_decay=self.config.degree_decay,
            min_conc=self.config.min_conc,
            max_conc=self.config.max_conc,
            conv_tol=conv_tol,
            compile_step=compile_step,
            l1_beta=self.config.l1_beta,
            misfit=misfit,
        )

    def _stacked_T(self) -> np.ndarray:
        """(n_angles, n_depth, n_elem) — same matrices FreeSolver extracts."""
        return np.stack(
            [self.tm.get_matrix(e) for e in range(self.tm.n_elements)], axis=-1)

    @staticmethod
    def _normalize(I_obs: np.ndarray) -> np.ndarray:
        """FreeSolver.solve() convention (substrate_indices=None path)."""
        return I_obs / np.maximum(I_obs.sum(axis=-1, keepdims=True), 1e-10)

    def solve_batch(
        self,
        I_obs: np.ndarray,
        D_init: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> FreeBatchResult:
        """Solve B pixels at once.

        Args:
            I_obs: (B, n_angles, n_elem) raw observations (normalized inside).
            D_init: (B, n_depth, n_elem) or (n_depth, n_elem); default D_prior.
            weights: optional WLS weights (see solve_free_batch).
        """
        I_norm = self._normalize(np.asarray(I_obs, dtype=np.float64))
        B = I_norm.shape[0]
        if D_init is None:
            D_init = np.broadcast_to(self.D_prior, (B, *self.D_prior.shape))
        elif D_init.ndim == 2:
            D_init = np.broadcast_to(D_init, (B, *D_init.shape))
        D_init = np.array(D_init, dtype=np.float64)
        T = self._stacked_T()

        if B <= self.chunk_px:
            return solve_free_batch(T, I_norm, D_init, self.D_prior,
                                    self.batch_config, weights=weights)
        # Chunked path: bounds peak GPU memory; per-chunk results
        # are concatenated (pixels are independent, so this is exact).
        parts = []
        for s in range(0, B, self.chunk_px):
            e = min(s + self.chunk_px, B)
            w = None if weights is None or np.ndim(weights) < 3 \
                else weights[s:e]
            parts.append(solve_free_batch(T, I_norm[s:e], D_init[s:e],
                                          self.D_prior, self.batch_config,
                                          weights=w))
        return FreeBatchResult(
            D_opt=np.concatenate([p.D_opt for p in parts]),
            I_opt=np.concatenate([p.I_opt for p in parts]),
            chi2=np.concatenate([p.chi2 for p in parts]),
            objective=np.concatenate([p.objective for p in parts]),
            n_iterations=max(p.n_iterations for p in parts),
            converged=np.concatenate([p.converged for p in parts]),
            elapsed_time=sum(p.elapsed_time for p in parts),
            timings={k: sum(p.timings.get(k, 0.0) for p in parts)
                     for k in parts[0].timings},
        )

    def solve(
        self,
        I_obs: np.ndarray,
        D_init: Optional[np.ndarray] = None,
        substrate_indices: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> SolverResult:
        """Single-pixel drop-in for FreeSolver.solve()."""
        if substrate_indices is not None:
            raise NotImplementedError("substrate normalization: Stage B scope")
        r = self.solve_batch(I_obs[None])
        reg = 0.0
        if self.config.method == 'L2':
            reg = float(np.sum((r.D_opt[0] - self.D_prior) ** 2))
        return SolverResult(
            D_opt=r.D_opt[0], I_opt=r.I_opt[0],
            chi2=float(r.chi2[0]), regularization=reg,
            objective=float(r.objective[0]),
            n_iterations=r.n_iterations,
            elapsed_time=r.elapsed_time,
            success=bool(r.converged[0]),
        )


class MLXBatchSemiFreeSolver:
    """Batched drop-in for SemiFreeSolver (Stage C, charge-balance mode).

    Reuses a real ``SemiFreeSolver`` instance for all derived quantities
    (free_linked_const, index validation, D_prior default) so the setup is
    consistent with the reference by construction; the optimization itself
    runs on the MLX block-coordinate solver (semifree_batch).

    Usage (batch — the intended path):
        solver = MLXBatchSemiFreeSolver(tm, config, semi_cfg,
                                        elements=element_configs)
        batch = solver.solve_batch(I_obs_stack,          # (B, n_angles, n_elem)
                                   thickness_init=thk,   # (B,) or scalar
                                   abruptness_init=abrp) # (B,) or scalar
    """

    def __init__(
        self,
        transfer_matrix,
        config: Optional[SolverConfig] = None,
        semi_free_config: Optional[SemiFreeSolverConfig] = None,
        D_prior: Optional[np.ndarray] = None,
        elements: Optional[List] = None,
        # Defaults chosen by the authors as a balance of throughput and
        # agreement with the SLSQP reference: 8 rounds, round-0 grid 8 x
        # refine 100, later rounds 5 x 40, 250 C-iters per round.
        fista_max_iter: int = 250,
        conv_tol: float = 1e-3,
        outer_rounds: int = 8,
        t_grid: int = 8,
        t_refine_iters: int = 100,
        t_grid_later: int = 5,
        t_refine_later: int = 40,
        t_radius_init: Optional[float] = None,
        compile_step: bool = True,
        chunk_px: Optional[int] = None,
        fixed_c_step: Optional[float] = None,
        capture_initial_step: bool = False,
        fixed_iterations: bool = False,
        deterministic_power_init: bool = False,
        power_init_seed: int = 0,
        canonical_chunk_pad: bool = False,
    ):
        self.chunk_px = chunk_px if chunk_px is not None else CHUNK_PX_AUTO
        # R1b (PAPER3 CHUNK-2): Metal kernels are batch-shape specialized,
        # so float32 arithmetic differs between chunk shapes and strong
        # any-chunk-size invariance is unattainable.  With
        # canonical_chunk_pad=True every chunk (including the remainder and
        # the small-B path) is padded to EXACTLY chunk_px rows with dummy
        # pixels (results discarded), so each pixel is always processed
        # under the identical kernel shape: results become bit-independent
        # of total batch size, pixel order, and companion pixels
        # (requires deterministic_power_init + explicit pixel_ids).
        self.canonical_chunk_pad = canonical_chunk_pad
        self.tm = transfer_matrix
        self.config = config or SolverConfig()
        self.semi_free_config = semi_free_config or SemiFreeSolverConfig()

        cfg, scfg = self.config, self.semi_free_config
        unsupported = {
            'gradient_mode': scfg.gradient_mode,
            "stoichiometric_mode != 'charge_balance'":
                scfg.stoichiometric_mode != 'charge_balance',
            'normalize_free_conc': scfg.normalize_free_conc,
            'substrate_cutoff': scfg.substrate_cutoff is not None,
            'use_legacy_sum_constraint': scfg.use_legacy_sum_constraint,
            'use_nonlinear_sum_constraint': scfg.use_nonlinear_sum_constraint,
            'smooth_degenerate': scfg.smooth_degenerate,
            'const_indices': bool(scfg.const_indices),
            "intensity_normalization != 'substrate'":
                scfg.intensity_normalization != 'substrate',
            'use_anscombe': cfg.use_anscombe,
            'use_wls': cfg.use_wls,
            'use_poisson_nll': cfg.use_poisson_nll,
            "MEM mem_space='log'":
                cfg.method == 'MEM' and cfg.mem_space == 'log',
            "l1_smoothing='log_cosh'":
                cfg.method == 'L1' and cfg.l1_smoothing == 'log_cosh',
        }
        bad = [k for k, v in unsupported.items() if v]
        if bad:
            raise NotImplementedError(
                f"mlx_batch SemiFree does not support: {', '.join(bad)}")

        # Reference instance: derives free_linked_const / indices / D_prior
        # exactly as the SLSQP path does (its __init__ is pure setup).
        self._ref = SemiFreeSolver(
            transfer_matrix, config=self.config,
            semi_free_config=self.semi_free_config,
            D_prior=D_prior, elements=elements, use_numba=False,
        )
        self.free_idx = list(self._ref.free_idx)
        self.linked_idx = list(self._ref.linked_idx)
        self.sub_idx = list(self._ref.sub_idx)
        self.antisubst_idx = self.free_idx + self.linked_idx
        self.free_linked_const = np.asarray(self._ref.free_linked_const,
                                            dtype=np.float64)
        self.D_prior = self._ref.D_prior
        self.depth = np.asarray(transfer_matrix.depth, dtype=np.float64)
        if not self.sub_idx:
            raise NotImplementedError(
                "mlx_batch SemiFree requires substrate elements "
                "(substrate normalization mode)")

        self.batch_config = SemiFreeBatchConfig(
            method=cfg.method,
            alpha=cfg.alpha,
            degree=cfg.degree,
            min_conc=cfg.min_conc,
            max_conc=cfg.max_conc,
            max_iter=fista_max_iter,
            conv_tol=conv_tol,
            compile_step=compile_step,
            optimize_thickness=scfg.optimize_thickness,
            thickness_bounds=tuple(scfg.thickness_bounds),
            t_grid=t_grid,
            t_refine_iters=t_refine_iters,
            t_grid_later=t_grid_later,
            t_refine_later=t_refine_later,
            t_radius_init=t_radius_init,
            fixed_step=fixed_c_step,
            capture_initial_step=capture_initial_step,
            fixed_iterations=fixed_iterations,
            deterministic_power_init=deterministic_power_init,
            power_init_seed=power_init_seed,
            optimize_abruptness=scfg.optimize_abruptness,
            abruptness_bounds=tuple(scfg.abruptness_bounds),
            optimize_substrate_ratio=scfg.optimize_substrate_ratio,
            outer_rounds=outer_rounds,
            degen_threshold=cfg.degen_depth_threshold,
            t_prior_center=scfg.t_prior_center,
            t_prior_sigma=scfg.t_prior_sigma,
        )

    def _stacked_T(self) -> np.ndarray:
        return np.stack(
            [self.tm.get_matrix(e) for e in range(self.tm.n_elements)],
            axis=-1)

    def _prepare_init(self, B, thickness_init, abruptness_init,
                      film_conc_init):
        """Replicate SemiFreeSolver.solve()'s x0 construction (batched)."""
        scfg = self.semi_free_config
        cfg = self.config
        n_free = len(self.free_idx)
        n_depth = len(self.depth)
        flc = self.free_linked_const

        t0 = np.broadcast_to(np.asarray(
            thickness_init if thickness_init is not None
            else scfg.thickness_init, dtype=np.float64), (B,)).copy()
        a0 = np.broadcast_to(np.asarray(
            abruptness_init if abruptness_init is not None
            else scfg.abruptness_init, dtype=np.float64), (B,)).copy()
        t_lo, t_hi = scfg.thickness_bounds
        t0 = np.clip(t0, t_lo, t_hi)

        n_sub = len(self.sub_idx)
        s0 = np.full((B, n_sub), 1.0 / n_sub)

        if film_conc_init is None:
            # reference: film_conc = D_prior[:, free] / (1 - erf_sub), clip,
            # then row-normalized to satisfy sum(flc*C) = 1
            sigma = a0 / (2.0 * np.sqrt(np.log(2.0)))
            erf_sub = 0.5 * (1.0 + _erf_np(
                (self.depth[None, :] - t0[:, None])
                / np.maximum(sigma[:, None], 1e-10)))       # (B, Z)
            film_frac = np.maximum(1.0 - erf_sub, cfg.min_conc)
            C0 = (self.D_prior[None, :, self.free_idx]
                  / film_frac[:, :, None])                   # (B, Z, F)
            for i, f in enumerate(flc):
                if f > 0:
                    C0[:, :, i] = np.clip(C0[:, :, i], cfg.min_conc, 1.0 / f)
            row = (C0 * flc).sum(axis=-1)                    # (B, Z)
            uniform = 1.0 / flc.sum()
            C0 = np.where((row < 1e-6)[:, :, None], uniform,
                          C0 / np.maximum(row, 1e-6)[:, :, None])
        else:
            C0 = np.broadcast_to(
                np.asarray(film_conc_init, dtype=np.float64),
                (B, n_depth, n_free)).copy()
            for i, f in enumerate(flc):
                if f > 0:
                    C0[:, :, i] = np.clip(C0[:, :, i], cfg.min_conc, 1.0 / f)

        # priors (reference m0): m0_C = raw D_prior[:, free];
        # m0_t = thickness_init unless prior profile ~0 (then 0.0)
        m0_C = self.D_prior[:, self.free_idx]
        m0_t = t0.copy()
        if np.all(self.D_prior <= 1e-12):
            m0_t = np.zeros(B)
        return C0, t0, a0, s0, m0_C, m0_t

    def _xreg(self, I_norm: np.ndarray) -> Optional[np.ndarray]:
        """Per-pixel xreg weights (reference solve() lines 2703-2713)."""
        if not self.config.use_xreg_weighting:
            return None
        xreg = np.mean(I_norm[:, :, self.antisubst_idx], axis=1)  # (B, n_anti)
        xreg[xreg == 0] = 1.0
        w = 1.0 / xreg
        if np.isfinite(self.config.xreg_max_weight):
            w = np.minimum(w, self.config.xreg_max_weight)
        return w

    def solve_batch(
        self,
        I_obs: np.ndarray,
        thickness_init=None,
        abruptness_init=None,
        film_conc_init: Optional[np.ndarray] = None,
        pixel_ids: Optional[np.ndarray] = None,
    ) -> SemiFreeBatchResult:
        """Solve B pixels at once.

        Args:
            I_obs: (B, n_angles, n_elem) raw observations.
            thickness_init: scalar or (B,) [A]; default semi_free_config.
            abruptness_init: scalar or (B,) [A]; default semi_free_config.
            film_conc_init: optional (B, n_depth, n_free) or (n_depth,
                n_free) normalized film concentrations (e.g. NN warm-start
                via ``estimate_init_from_drlt``); default = reference x0.
            pixel_ids: optional (B,) stable global pixel ids for the R1
                deterministic power init; default arange(B).  The chunked
                path slices these, so a pixel keeps its id regardless of
                chunk layout.
        """
        I_obs = np.asarray(I_obs, dtype=np.float64)
        B = I_obs.shape[0]
        I_sub = I_obs[:, :, self.sub_idx].sum(axis=-1, keepdims=True)
        I_norm = I_obs / np.maximum(I_sub, 1e-10)

        C0, t0, a0, s0, m0_C, m0_t = self._prepare_init(
            B, thickness_init, abruptness_init, film_conc_init)
        xw = self._xreg(I_norm)
        T = self._stacked_T()

        pids = (np.arange(B, dtype=np.float64) if pixel_ids is None
                else np.asarray(pixel_ids, dtype=np.float64))

        def _pad(a, n):
            if len(a) == n:
                return a
            reps = np.repeat(a[-1:], n - len(a), axis=0)
            return np.concatenate([a, reps], axis=0)

        def run(sl, n_real=None):
            args = [I_norm[sl], C0[sl], t0[sl], a0[sl], s0[sl]]
            xws = None if xw is None else xw[sl]
            pd = pids[sl]
            m0ts = m0_t[sl]
            if self.canonical_chunk_pad:
                n_real = len(args[0]) if n_real is None else n_real
                args = [_pad(a, self.chunk_px) for a in args]
                xws = None if xws is None else _pad(xws, self.chunk_px)
                # dummy rows get distinct negative ids so their power-init
                # vectors cannot alias a real pixel's
                extra = self.chunk_px - n_real
                pd = np.concatenate([pd, -np.arange(1, extra + 1, dtype=float)]) \
                    if extra else pd
                m0ts = _pad(m0ts, self.chunk_px)
            r = solve_semifree_batch(
                T, *args,
                self.depth, self.free_idx, self.linked_idx, self.sub_idx,
                self.free_linked_const, m0_C, m0ts,
                self.batch_config,
                xreg_weights=xws,
                pixel_ids=pd,
            )
            if self.canonical_chunk_pad and len(r.D_opt) != n_real:
                r = dataclasses.replace(
                    r,
                    **{f: getattr(r, f)[:n_real]
                       for f in ("D_opt", "I_opt", "film_conc", "thickness",
                                 "abruptness", "sub_ratios", "chi2",
                                 "objective", "converged")})
            return r

        if B <= self.chunk_px:
            return run(slice(None))
        parts = [run(slice(s, min(s + self.chunk_px, B)))
                 for s in range(0, B, self.chunk_px)]
        return SemiFreeBatchResult(
            D_opt=np.concatenate([p.D_opt for p in parts]),
            I_opt=np.concatenate([p.I_opt for p in parts]),
            film_conc=np.concatenate([p.film_conc for p in parts]),
            thickness=np.concatenate([p.thickness for p in parts]),
            abruptness=np.concatenate([p.abruptness for p in parts]),
            sub_ratios=np.concatenate([p.sub_ratios for p in parts]),
            chi2=np.concatenate([p.chi2 for p in parts]),
            objective=np.concatenate([p.objective for p in parts]),
            n_outer=max(p.n_outer for p in parts),
            converged=np.concatenate([p.converged for p in parts]),
            elapsed_time=sum(p.elapsed_time for p in parts),
            timings={k: sum(p.timings.get(k, 0.0) for p in parts)
                     for k in parts[0].timings},
        )


def _erf_np(x):
    from scipy.special import erf
    return erf(x)


class MLXBatchGenericLayerSolver:
    """Batched drop-in for GenericLayerSolver (Stage D, fixed stoichiometry).

    Optimizes layer thicknesses (and optionally interface widths) for B
    pixels sharing one transfer matrix, via batched Levenberg-Marquardt.
    Concentrations follow the fixed stoichiometry, exactly as the reference
    GenericLayerSolver._build_profile (erf windows + substrate step).

    Usage (batch):
        solver = MLXBatchGenericLayerSolver(tm, layer_config)
        b = solver.solve_batch(I_obs_stack)   # (B, n_angles, n_elem)

    Usage (single pixel, GenericLayerSolver.solve()-shaped result):
        r = solver.solve(I_obs)               # GenericLayerResult
    """

    def __init__(
        self,
        transfer_matrix,
        layer_config,
        config: Optional[SolverConfig] = None,
        elements: Optional[List] = None,
        max_iter: int = 30,
        chunk_px: Optional[int] = None,
    ):
        if layer_config is None:
            raise ValueError("layer_config is required")
        if layer_config.optimize_stoichiometry and \
                layer_config.has_variable_stoichiometry:
            raise NotImplementedError(
                "mlx_batch GenericLayer supports fixed stoichiometry only")
        self.tm = transfer_matrix
        self.layer_config = layer_config
        self.config = config or SolverConfig()
        self.elements = elements
        self.chunk_px = chunk_px if chunk_px is not None else CHUNK_PX_AUTO
        self.depth = np.asarray(transfer_matrix.depth, dtype=np.float64)
        self.n_elem = transfer_matrix.n_elements
        # validate coefficients build (also rejects non-erf profiles early)
        build_layer_coefficients(layer_config, self.n_elem)

        # attach the depth grid the profile is built on (layer_lm reads it)
        layer_config._depth = self.depth

        n_layers = layer_config.n_layers
        x0_parts = [np.asarray(layer_config.initial_thicknesses, float)]
        if layer_config.optimize_widths:
            x0_parts.append(np.asarray(layer_config.initial_widths, float))
        self.x0 = np.concatenate(x0_parts)

        self.lm_config = LayerLMConfig(
            optimize_widths=layer_config.optimize_widths,
            thickness_bounds=tuple(layer_config.thickness_bounds),
            width_bounds=tuple(layer_config.width_bounds),
            max_iter=max_iter,
            min_conc=self.config.min_conc,
        )

    def _stacked_T(self) -> np.ndarray:
        return np.stack(
            [self.tm.get_matrix(e) for e in range(self.tm.n_elements)],
            axis=-1)

    @staticmethod
    def _normalize(I_obs):
        return I_obs / np.maximum(I_obs.sum(axis=-1, keepdims=True), 1e-10)

    def solve_batch(
        self,
        I_obs: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> LayerLMResult:
        """Solve B pixels at once.

        Args:
            I_obs: (B, n_angles, n_elem) raw observations (normalized inside).
            x0: (n_p,) or (B, n_p) initial [thicknesses, (widths)];
                default = layer_config initial values (NN/known-structure
                warm-starts pass per-pixel x0 here).
        """
        I_norm = self._normalize(np.asarray(I_obs, dtype=np.float64))
        B = I_norm.shape[0]
        T = self._stacked_T()
        x0v = self.x0 if x0 is None else np.asarray(x0, dtype=np.float64)

        def run(sl):
            xs = x0v if x0v.ndim == 1 else x0v[sl]
            return solve_layer_lm(T, I_norm[sl], self.layer_config, xs,
                                  self.lm_config)

        if B <= self.chunk_px:
            return run(slice(None))
        parts = [run(slice(s, min(s + self.chunk_px, B)))
                 for s in range(0, B, self.chunk_px)]
        return LayerLMResult(
            D_opt=np.concatenate([p.D_opt for p in parts]),
            I_opt=np.concatenate([p.I_opt for p in parts]),
            layer_thicknesses=np.concatenate(
                [p.layer_thicknesses for p in parts]),
            interface_widths=np.concatenate(
                [p.interface_widths for p in parts]),
            chi2=np.concatenate([p.chi2 for p in parts]),
            n_iterations=max(p.n_iterations for p in parts),
            converged=np.concatenate([p.converged for p in parts]),
            elapsed_time=sum(p.elapsed_time for p in parts),
            timings={k: sum(p.timings.get(k, 0.0) for p in parts)
                     for k in parts[0].timings},
        )

    def solve(self, I_obs: np.ndarray, verbose: bool = False
              ) -> GenericLayerResult:
        """Single-pixel drop-in for GenericLayerSolver.solve()."""
        b = self.solve_batch(I_obs[None])
        return GenericLayerResult(
            D_opt=b.D_opt[0], I_opt=b.I_opt[0],
            layer_thicknesses=b.layer_thicknesses[0],
            interface_widths=b.interface_widths[0],
            chi2=float(b.chi2[0]), objective=float(b.chi2[0]),
            n_iterations=b.n_iterations, elapsed_time=b.elapsed_time,
            success=bool(b.converged[0]),
        )
