"""
Inverse Solver for Depth Profile Reconstruction
================================================

Reconstructs depth profiles D(z) from angle-resolved intensities I(θ).

This is the inverse problem:
    Given: I(θ) observed intensities at multiple angles
    Find:  D(z) depth-dependent concentrations

The forward model is:
    I(θ, e) = TransMat(θ, z, e) @ D(z, e)

where TransMat encodes the Beer-Lambert attenuation.

Optimization methods:
    - L2 (Tikhonov): ||D - D_prior||² regularization (smooth)
    - L1: ||D - D_prior||₁ regularization (sparse)
    - MEM: Maximum Entropy Method (positive, smooth)
    - LLS: Linear Least Squares (no regularization)

Usage:
    from deppro.inverse_solver import FreeSolver, SolverConfig

    solver = FreeSolver(
        transfer_matrix=tm,
        method='L2',
        alpha=1.0,
    )

    D_reconstructed = solver.solve(I_observed)

Date: 2026-01-22
"""

import numpy as np
from dataclasses import dataclass, field, field
from typing import Dict, List, Optional, Tuple, Union, Literal, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import warnings

from scipy.optimize import minimize, minimize_scalar, Bounds
from scipy.sparse import diags
from scipy.special import erf

from .transfer_matrix import TransferMatrix
from .semifree_parametrization import (
    build_full_profile_core as _build_full_profile_core,
    build_substrate_profile as _build_substrate_profile_pure,
)

# Try to import Numba-accelerated kernels
try:
    from .numba_kernels import (
        build_substrate_profile_numba,
        build_full_profile_numba,
        compute_forward_model_numba,
        normalize_by_substrate_numba,
        compute_chi2_numba,
        compute_chi2_wls_numba,
        compute_poisson_nll_numba,
        compute_regularization_l1_numba,
        compute_regularization_l1_logcosh_numba,
        compute_regularization_l2_numba,
        compute_regularization_mem_numba,
        gradient_kernel_numba,
        gradient_backward_numba,
        gradient_kernel_wls_numba,
        gradient_kernel_poisson_numba,
        anscombe_transform,
        anscombe_inverse,
        warmup_jit,
        NUMBA_AVAILABLE,
    )
except ImportError:
    NUMBA_AVAILABLE = False


# Optimization methods
OptMethod = Literal['L2', 'L1', 'MEM', 'LLS']

# Prior model update strategies (for MEM)
PriorUpdateStrategy = Literal['fixed', 'iterative', 'neighbor', 'l1_init']


# Features held back from the public release. Their code stays in this module,
# but they refuse to run unless the distribution enables them; the private
# tree does so, the public release does not. Not a public API: an enabled
# held feature carries no support or guarantee of correct results.
_ENABLED_HELD_FEATURES: set = set()


def _enable_held_feature(name: str) -> None:
    """Enable a feature that is held back from this release (internal)."""
    _ENABLED_HELD_FEATURES.add(name)


# scipy.optimize.minimize methods that honour `constraints`, keyed by upper
# case, with the spelling the rest of this module compares against (SLSQP
# fallback routing, trust-constr options). COBYLA and COBYQA are accepted by
# scipy but are not exercised by this package's tests.
_CONSTRAINT_CAPABLE_METHODS = {
    'SLSQP': 'SLSQP', 'TRUST-CONSTR': 'trust-constr',
    'COBYLA': 'COBYLA', 'COBYQA': 'COBYQA',
}


@dataclass
class SolverConfig:
    """Configuration for inverse solver."""
    method: OptMethod = 'L2'
    alpha: float = 1.0              # Regularization strength
    degree: float = 1.0             # Data fit weight (Chi² multiplier)
    max_iter: int = 100             # Maximum iterations per pixel
    tol: float = 1e-6               # Convergence tolerance
    n_loops: int = 1                # Number of optimization loops
    degree_decay: float = 0.8       # Degree reduction per loop

    # Bounds
    min_conc: float = 1e-6          # Minimum concentration
    max_conc: float = 1.0           # Maximum concentration

    # Parallelization
    n_workers: Optional[int] = None  # None = use all CPUs
    batch_size: int = 100           # Pixels per batch

    # Scipy optimizer settings. SLSQP is the default because both FreeSolver
    # and SemiFreeSolver solve under equality constraints (stoichiometry,
    # mass balance): FreeSolver switches to SLSQP whenever it has
    # constraints, and SemiFreeSolver replaces a method that cannot handle
    # constraints by SLSQP with a warning. 'trust-constr' is also accepted;
    # 'L-BFGS-B' only takes bounds (2026-10-20: the default was 'L-BFGS-B',
    # which silently dropped SemiFreeSolver's constraints).
    scipy_method: str = 'SLSQP'  # 'SLSQP', 'trust-constr', 'L-BFGS-B'

    # Fallback routing on SLSQP failure. With scipy_method='SLSQP' a solve is
    # not always a single SLSQP run: FreeSolver reroutes a failed SLSQP, or a
    # "success" whose relative KKT residual exceeds fallback_kkt_threshold, to
    # fallback_method and then polishes that result with a short SLSQP
    # restart; SemiFreeSolver reroutes status 8 / status 4 / an
    # ill-conditioned constraint Jacobian, and also an SLSQP "success" that
    # took no step at all (x unchanged after more than one iteration), on
    # which route it polishes the fallback result with SLSQP as well. A
    # rerouted result is kept only if it succeeds or lowers the objective;
    # SemiFreeSolverResult.fallback_used records that it was kept, and the
    # reported success/status are those of the last optimizer that ran
    # (a kept, improved result can still report success=False).
    enable_fallback: bool = True             # Retry with fallback method on failure
    fallback_method: str = 'trust-constr'    # Method to fall back to
    fallback_condition_threshold: float = 1e12  # Condition number threshold for fallback
    fallback_kkt_threshold: float = 0.1      # Relative KKT residual above which an SLSQP
                                             # "success" is treated as a false convergence
                                             # (FreeSolver only)

    # Warm restart: re-run optimization with updated active depths when
    # thickness changes significantly (degenerate depth rows become stale)
    max_warm_restarts: int = 2               # Max warm restart iterations (0 = disabled)
    warm_restart_threshold: float = 0.2      # Relative thickness change to trigger (20%)

    # Degenerate depth detection threshold. At depths where erf_sub ≈ 1
    # (deep substrate), the sum constraint RHS ≈ 0, causing rank-deficient
    # constraint Jacobian. Set to 0.0 to disable (all depths active).
    # WARNING: enabling this (>0) makes the solver thickness_init-sensitive
    # due to the coupling between active depths and thickness.
    degen_depth_threshold: float = 0.0       # Default: disabled (all depths active)

    # Constraint options (MATLAB-compatible: always enabled by default)
    use_sum_constraint: bool = True     # Enforce Σconcentration = 1 at each depth
    use_charge_constraint: bool = True  # Enforce charge neutrality
    use_stoich_constraint: bool = False  # Enforce stoichiometry ratios

    # Weighted Least Squares for Poisson noise
    use_wls: bool = False               # Use WLS with weight = 1/I_obs for Poisson noise

    # MATLAB-compatible Xreg weighting (Chi² = sum((X-Xsim)²/Xreg²))
    # Xreg = mean(Xdata) where Xdata = substrate-normalized film intensities
    # This weights elements by their relative intensity, giving more weight
    # to trace elements (Si_ox, Ti) and less to dominant ones (C, O)
    use_xreg_weighting: bool = False
    # Maximum Xreg weight (1/Xreg). Caps the amplification of very weak
    # elements to prevent over-weighting when their signal is near the
    # detection limit. Default inf = no cap (original MATLAB behavior).
    xreg_max_weight: float = float('inf')

    # Anscombe variance-stabilizing transform for Poisson noise
    use_anscombe: bool = False          # Apply Y = 2*sqrt(I + 3/8) transform

    # Poisson negative log-likelihood (recommended for Poisson noise)
    use_poisson_nll: bool = False       # Use NLL = Σ[I_pred - I_obs*log(I_pred)]

    # Prior model update (for MEM)
    prior_update: str = 'fixed'         # 'fixed', 'iterative', 'neighbor', 'l1_init'
    neighbor_radius: int = 1            # Radius for nearest-neighbor averaging

    # L1 smoothing strategy for analytical gradient compatibility
    # 'none': standard |x-m0| (requires FD for gradient)
    # 'huber': d/max(|d|, delta) smoothing (found insufficient)
    # 'log_cosh': β·log(cosh((x-m0)/β)) — fully smooth, grad=tanh (recommended)
    l1_smoothing: str = 'none'
    l1_beta: float = 0.01            # Smoothness for log_cosh (smaller = closer to L1)

    # MEM optimization space (only used when method='MEM')
    # 'concentration': I-divergence in concentration space (current Python)
    #   R(D) = Σ(D·log(D/m0) - D + m0), grad = log(D/m0)
    # 'log': MATLAB-style log-space entropy
    #   x = log(D/m0), R(x) = Σ m0·(eˣ-1-x·eˣ), grad = m0·x·eˣ
    #   Preserves checker pattern boundaries (C=0/B=0)
    mem_space: str = 'concentration'

    # Force numerical gradient (finite differences) instead of analytical
    use_numerical_gradient: bool = False

    # Substrate fixing
    substrate_indices: Optional[List[int]] = None  # Indices of substrate elements
    substrate_depth_start: Optional[float] = None  # Depth where substrate starts (Å)
    fix_substrate_deep: bool = False               # Fix substrate to 1.0 at deep depths

    @classmethod
    def recommended(cls, alpha: float = 0.0003) -> 'SolverConfig':
        """
        Recommended configuration for FreeSolver with Exponential Prior.

        Chosen in a verification with an Al2O3/SiO2/Si 3-layer system
        (2026-01-29). The RMSE ≈ 0.056 recorded then does not reproduce:
        examples/freesolver_recommended.py gives a concentration RMSE of about
        0.13 (see docs/api_reference.md, "FreeSolver example").

        Usage:
            config = SolverConfig.recommended()
            D_prior = create_exponential_prior(depth, n_elements, substrate_idx,
                                                d_penetration=imfp_substrate)
            solver = FreeSolver(tm, config, D_prior, elements=elements)

        Args:
            alpha: Regularization strength. The default 0.0003 is a
                   starting value, not a tuned optimum; choose alpha for
                   the noise level of your data.

        Returns:
            SolverConfig with MEM + Sum + Charge constraints enabled.
        """
        return cls(
            method='MEM',
            alpha=alpha,
            max_iter=500,
            use_sum_constraint=True,
            use_charge_constraint=True,
        )

    @classmethod
    def quick(cls) -> 'SolverConfig':
        """
        Quick configuration for fast iteration during development.

        Uses LLS (no regularization) with constraints.
        Faster but less accurate for deep interfaces.
        """
        return cls(
            method='LLS',
            max_iter=200,
            use_sum_constraint=True,
            use_charge_constraint=True,
        )


@dataclass
class SolverResult:
    """Result from inverse solver."""
    D_opt: np.ndarray               # Reconstructed depth profile (n_depth, n_elements)
    I_opt: np.ndarray               # Forward model prediction (n_angles, n_elements)
    chi2: float                     # Final Chi² value
    regularization: float           # Final regularization term
    objective: float                # Final objective value
    n_iterations: int               # Total iterations
    elapsed_time: float             # Total time in seconds
    success: bool                   # Convergence status

    @property
    def rmse(self) -> float:
        """Root mean square error of reconstruction."""
        return np.sqrt(self.chi2 / self.I_opt.size)


class FreeSolver:
    """
    Free model inverse solver for depth profile reconstruction.

    Optimizes concentration at each depth point independently.
    High flexibility but may be unstable without good priors.

    Solves: min_D  0.5 * degree * ||I_obs - T @ D||² + alpha * R(D)

    where R(D) is the regularization term depending on method.

    Note: For better stability, consider SemiFreeSolver (substrate fixed)
    or GenericLayerSolver (layer thickness optimization).
    """

    def __init__(
        self,
        transfer_matrix: TransferMatrix,
        config: Optional[SolverConfig] = None,
        D_prior: Optional[np.ndarray] = None,
        elements: Optional[List] = None,  # List[ElementConfig] for constraints
    ):
        """
        Initialize inverse solver.

        Args:
            transfer_matrix: Pre-computed transfer matrix
            config: Solver configuration
            D_prior: Prior depth profile for regularization (n_depth, n_elements)
            elements: List of ElementConfig for charge/stoichiometry constraints
        """
        self.tm = transfer_matrix
        self.config = config or SolverConfig()
        self.elements = elements

        # Prior model: L1/LLS use near-zero (sparsity), L2/MEM use uniform
        if D_prior is not None:
            self.D_prior = D_prior.copy()
        else:
            n_depth = len(transfer_matrix.depth)
            n_elements = transfer_matrix.n_elements
            if self.config.method in ('L1', 'LLS'):
                self.D_prior = np.full((n_depth, n_elements), self.config.min_conc)
            else:
                self.D_prior = np.ones((n_depth, n_elements)) / n_elements

        # Ensure prior is positive
        self.D_prior = np.maximum(self.D_prior, self.config.min_conc)

    def solve(
        self,
        I_obs: np.ndarray,
        D_init: Optional[np.ndarray] = None,
        substrate_indices: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> SolverResult:
        """
        Solve inverse problem for single pixel.

        Args:
            I_obs: Observed intensities (n_angles, n_elements)
            D_init: Initial guess (n_depth, n_elements), default=D_prior
            substrate_indices: Indices of substrate elements (for normalization)
            verbose: Print progress

        Returns:
            SolverResult with reconstructed profile
        """
        start_time = time.time()

        # Normalize observations
        if substrate_indices is not None:
            I_substrate = I_obs[:, substrate_indices].sum(axis=1, keepdims=True)
            I_normalized = I_obs / np.maximum(I_substrate, 1e-10)
        else:
            I_normalized = I_obs / np.maximum(I_obs.sum(axis=1, keepdims=True), 1e-10)

        # Initial guess
        if D_init is None:
            D_init = self.D_prior.copy()

        n_depth, n_elements = D_init.shape

        # Apply substrate fixing to initial guess if configured
        if self.config.fix_substrate_deep and self.config.substrate_indices is not None:
            D_init = self._apply_substrate_constraint(D_init.copy())

        # Flatten for optimizer
        x0 = D_init.flatten()

        # Bounds - with substrate fixing
        lb = np.full_like(x0, self.config.min_conc)
        ub = np.full_like(x0, self.config.max_conc)

        if self.config.fix_substrate_deep and self.config.substrate_indices is not None:
            lb, ub = self._apply_substrate_bounds(lb, ub, n_depth, n_elements)

        bounds = Bounds(lb=lb, ub=ub)

        # Get transfer matrices for each element
        T_matrices = []
        for e in range(n_elements):
            T_matrices.append(self.tm.get_matrix(e))

        # Build constraints
        constraints = self._build_constraints(n_depth, n_elements)

        # Choose optimizer method based on constraints
        scipy_method = self.config.scipy_method
        if constraints:
            # SLSQP supports equality/inequality constraints
            scipy_method = 'SLSQP'

        # Optimization
        degree = self.config.degree
        total_iterations = 0

        for loop in range(self.config.n_loops):
            # Use analytical gradient for methods that support it
            use_gradient = (
                self.config.method in ['L2', 'LLS', 'MEM']
                and not self.config.use_numerical_gradient
            )

            result = minimize(
                self._objective,
                x0,
                args=(I_normalized, T_matrices, n_depth, n_elements, degree),
                method=scipy_method,
                bounds=bounds,
                jac=self._gradient if use_gradient else None,
                constraints=constraints,
                options={
                    'maxiter': self.config.max_iter,
                    'ftol': self.config.tol,
                    'disp': verbose,
                }
            )

            # False-convergence detection: on flat degenerate valleys (e.g.
            # LLS, where the regularization term is zero) SLSQP can stall on
            # ftol and report success far from stationarity. Verified via the
            # relative KKT residual with the exact analytical gradient
            # (available for L2/LLS/MEM even when the optimizer used FD).
            # Pure diagnostic: genuinely converged points pass untouched.
            _kkt_stalled = False
            if (result.success
                    and self.config.enable_fallback
                    and scipy_method == 'SLSQP'
                    and self.config.method in ('L2', 'LLS', 'MEM')):
                _kkt = self._kkt_stationarity_residual(
                    result.x,
                    (I_normalized, T_matrices, n_depth, n_elements, degree),
                    constraints, lb, ub)
                _kkt_stalled = _kkt > self.config.fallback_kkt_threshold

            # Condition-based fallback routing (same pattern as SemiFreeSolver)
            if ((not result.success or _kkt_stalled)
                    and self.config.enable_fallback
                    and scipy_method == 'SLSQP'):

                # Check constraint Jacobian condition number
                _max_condition = 0.0
                for c in constraints:
                    if 'jac' in c and c['jac'] is not None:
                        try:
                            J = np.atleast_2d(c['jac'](result.x))
                            if J.shape[0] > 0:
                                s = np.linalg.svd(J, compute_uv=False)
                                cond = s[0] / max(s[-1], 1e-15)
                                _max_condition = max(_max_condition, cond)
                        except Exception:
                            pass

                _should_fallback = (
                    _kkt_stalled  # false SLSQP success (non-stationary point)
                    or _max_condition > self.config.fallback_condition_threshold
                    or result.status == 8  # positive directional derivative
                    or result.status == 4  # inequality constraints incompatible
                    or result.status == 9  # iteration limit reached
                )

                if _should_fallback:
                    if verbose:
                        print(f"[FreeSolver] SLSQP failed (status={result.status}, "
                              f"stalled={_kkt_stalled}, cond={_max_condition:.1e}), "
                              f"retrying with {self.config.fallback_method}")

                    fallback_options = {
                        'maxiter': self.config.max_iter,
                        'disp': verbose,
                        'gtol': self.config.tol,
                    }
                    result_fb = minimize(
                        self._objective,
                        x0,  # restart from x0, not result.x
                        args=(I_normalized, T_matrices, n_depth, n_elements, degree),
                        method=self.config.fallback_method,
                        bounds=bounds,
                        jac=self._gradient if use_gradient else None,
                        constraints=constraints,
                        options=fallback_options,
                    )

                    # Polish: trust-constr typically exhausts its iteration
                    # budget just short of stationarity on flat valleys
                    # (success=False, status=0). A short SLSQP restart from
                    # its point converges cleanly with success=True.
                    result_pl = minimize(
                        self._objective,
                        result_fb.x,
                        args=(I_normalized, T_matrices, n_depth, n_elements, degree),
                        method='SLSQP',
                        bounds=bounds,
                        jac=self._gradient if use_gradient else None,
                        constraints=constraints,
                        options={
                            'maxiter': self.config.max_iter,
                            'ftol': self.config.tol,
                            'disp': verbose,
                        },
                    )
                    if result_pl.success or result_pl.fun < result_fb.fun:
                        result_fb = result_pl

                    # Accept fallback if it improved
                    if result_fb.success or result_fb.fun < result.fun:
                        result = result_fb
                        if verbose:
                            print(f"[FreeSolver] Fallback succeeded: "
                                  f"obj={result.fun:.6e}, nit={result.nit}")

            x0 = result.x
            total_iterations += result.nit

            # Decay degree for next loop
            degree *= self.config.degree_decay

        # Reshape result
        D_opt = result.x.reshape(n_depth, n_elements)

        # Compute forward model
        I_opt = np.zeros_like(I_normalized)
        for e in range(n_elements):
            I_opt[:, e] = T_matrices[e] @ D_opt[:, e]

        # Normalize predicted intensities
        I_opt = I_opt / np.maximum(I_opt.sum(axis=1, keepdims=True), 1e-10)

        # Compute final statistics
        chi2 = np.sum((I_normalized - I_opt) ** 2)
        reg = self._regularization(D_opt)

        elapsed = time.time() - start_time

        return SolverResult(
            D_opt=D_opt,
            I_opt=I_opt,
            chi2=chi2,
            regularization=reg,
            objective=result.fun,
            n_iterations=total_iterations,
            elapsed_time=elapsed,
            success=result.success,
        )

    def _objective(
        self,
        x: np.ndarray,
        I_obs: np.ndarray,
        T_matrices: List[np.ndarray],
        n_depth: int,
        n_elements: int,
        degree: float,
    ) -> float:
        """Objective function: 0.5 * degree * Chi² + alpha * R(D)"""
        D = x.reshape(n_depth, n_elements)

        # Forward model
        I_pred = np.zeros_like(I_obs)
        for e in range(n_elements):
            I_pred[:, e] = T_matrices[e] @ D[:, e]

        # Normalize
        I_pred = I_pred / np.maximum(I_pred.sum(axis=1, keepdims=True), 1e-10)

        # Chi² term
        chi2 = np.sum((I_obs - I_pred) ** 2)

        # Regularization
        reg = self._regularization(D)

        return 0.5 * degree * chi2 + self.config.alpha * reg

    def _gradient(
        self,
        x: np.ndarray,
        I_obs: np.ndarray,
        T_matrices: List[np.ndarray],
        n_depth: int,
        n_elements: int,
        degree: float,
    ) -> np.ndarray:
        """Gradient of objective (for L2/LLS/MEM), exact w.r.t. _objective."""
        D = x.reshape(n_depth, n_elements)

        # Forward model
        I_pred = np.zeros_like(I_obs)
        for e in range(n_elements):
            I_pred[:, e] = T_matrices[e] @ D[:, e]

        # Normalize (must mirror _objective exactly)
        I_sum = I_pred.sum(axis=1, keepdims=True)
        I_sum_clip = np.maximum(I_sum, 1e-10)
        I_pred_norm = I_pred / I_sum_clip

        # Residual
        residual = I_pred_norm - I_obs  # (n_angles, n_elements)

        # Gradient of the 0.5·degree·Chi² term through the per-angle
        # normalization I_norm[a,e] = I_pred[a,e]/S[a], S[a] = Σ_e I_pred[a,e]:
        #   d/dI_pred[a,e'] = degree·(r[a,e'] − Σ_e r[a,e]·I_norm[a,e]) / S[a]
        # Where S[a] hits the 1e-10 clip the denominator is constant, so the
        # projection term vanishes there.
        proj = (residual * I_pred_norm).sum(axis=1, keepdims=True)
        dchi2_dI = degree * (residual - proj * (I_sum > 1e-10)) / I_sum_clip

        grad = np.zeros_like(D)
        for e in range(n_elements):
            grad[:, e] = T_matrices[e].T @ dchi2_dI[:, e]

        # Gradient of regularization
        if self.config.method == 'L2':
            # d/dD ||D - D_prior||² = 2 * (D - D_prior)
            grad += 2 * self.config.alpha * (D - self.D_prior)
        elif self.config.method == 'MEM':
            # Bregman div D_φ(D‖m0), φ=x·log(x): grad = log(D/m0)
            D_safe = np.maximum(D, 1e-10)
            D_prior_safe = np.maximum(self.D_prior, 1e-10)
            grad += self.config.alpha * np.log(D_safe / D_prior_safe)
        # LLS has no regularization gradient

        return grad.flatten()

    def _kkt_stationarity_residual(
        self,
        x: np.ndarray,
        args: Tuple,
        constraints: List[Dict],
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> float:
        """
        Relative KKT stationarity residual at x (dimensionless).

        Least-squares fits Lagrange multipliers for the equality-constraint
        Jacobians and active bounds, then returns the unexplained fraction
        of the gradient: ||g - A^T lambda|| / max(1, ||g||). Near a true
        constrained optimum this is ~0; at an SLSQP ftol stall it is O(1).
        Sign conditions on the multipliers are ignored (a relaxation), so a
        small residual is necessary but not sufficient for optimality --
        adequate for routing, since it is only used to trigger a fallback
        probe whose result is accepted only if it improves.
        """
        g = self._gradient(x, *args)
        rows = []
        for c in constraints:
            if 'jac' in c and c['jac'] is not None:
                try:
                    J = np.atleast_2d(np.asarray(c['jac'](x), dtype=float))
                    if J.shape[0] > 0:
                        rows.append(J)
                except Exception:
                    pass
        active = (x <= lb + 1e-8) | (x >= ub - 1e-8)
        if active.any():
            rows.append(np.eye(x.size)[active])
        if rows:
            A = np.vstack(rows)
            lam, *_ = np.linalg.lstsq(A.T, g, rcond=None)
            r = g - A.T @ lam
        else:
            r = g
        return float(np.linalg.norm(r) / max(1.0, np.linalg.norm(g)))

    def _regularization(self, D: np.ndarray) -> float:
        """Compute regularization term R(D)."""
        if self.config.method == 'LLS':
            return 0.0

        elif self.config.method == 'L2':
            # Tikhonov: ||D - D_prior||²
            return np.sum((D - self.D_prior) ** 2)

        elif self.config.method == 'L1':
            if self.config.l1_smoothing == 'log_cosh':
                # Smooth L1: β·log(cosh((x-m0)/β)) → |x-m0| as β→0
                beta = self.config.l1_beta
                d = (D - self.D_prior) / beta
                return beta * np.sum(np.log(np.cosh(np.clip(d, -20, 20))))
            else:
                # Standard L1-norm: ||D - D_prior||₁
                return np.sum(np.abs(D - self.D_prior))

        elif self.config.method == 'MEM':
            # Bregman divergence D_φ(D‖m0) = Σ(D·log(D/m0) - D + m0) ≥ 0
            D_safe = np.maximum(D, 1e-10)
            D_prior_safe = np.maximum(self.D_prior, 1e-10)
            return np.sum(D_safe * np.log(D_safe / D_prior_safe) - D_safe + D_prior_safe)

        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def _build_constraints(
        self,
        n_depth: int,
        n_elements: int,
    ) -> List[Dict]:
        """
        Build scipy constraints from config.

        Returns:
            List of constraint dictionaries for scipy.optimize.minimize
        """
        from .depth_profile import (
            create_concentration_sum_constraint_matrix,
            create_charge_constraint_matrix,
            create_stoichiometry_constraint_matrix,
        )

        constraints = []

        # Concentration sum constraint: Σ D_e = 1 at each depth
        if self.config.use_sum_constraint:
            A_sum = create_concentration_sum_constraint_matrix(n_depth, n_elements)
            b_sum = np.ones(n_depth)
            constraints.append({
                'type': 'eq',
                'fun': lambda x, A=A_sum, b=b_sum: A @ x - b,
                'jac': lambda x, A=A_sum: A,
            })

        # Charge neutrality constraint: Σ (oxidation_state_e × D_e) = 0
        if self.config.use_charge_constraint and self.elements is not None:
            A_charge = create_charge_constraint_matrix(self.elements, n_depth)
            if A_charge is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x, A=A_charge: A @ x,
                    'jac': lambda x, A=A_charge: A,
                })

        # Stoichiometry constraint: maintain element ratios
        if self.config.use_stoich_constraint and self.elements is not None:
            A_stoich = create_stoichiometry_constraint_matrix(self.elements, n_depth)
            if A_stoich is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x, A=A_stoich: A @ x,
                    'jac': lambda x, A=A_stoich: A,
                })

        return constraints

    def _apply_substrate_constraint(self, D: np.ndarray) -> np.ndarray:
        """
        Apply substrate constraint to depth profile.

        At depths beyond substrate_depth_start, substrate elements sum to 1.0
        and non-substrate elements are 0.

        Args:
            D: Depth profile (n_depth, n_elements)

        Returns:
            Modified depth profile
        """
        if self.config.substrate_indices is None:
            return D

        depth = self.tm.depth
        n_elements = D.shape[1]

        # Find depth index where substrate starts
        if self.config.substrate_depth_start is not None:
            deep_idx = np.searchsorted(depth, self.config.substrate_depth_start)
        else:
            # Default: last 20% of depth range
            deep_idx = int(len(depth) * 0.8)

        # Get substrate indices
        sub_idx = self.config.substrate_indices
        n_substrate = len(sub_idx)

        # At deep depths: substrate = 1/n_substrate each, others = 0
        for z in range(deep_idx, len(depth)):
            for e in range(n_elements):
                if e in sub_idx:
                    D[z, e] = 1.0 / n_substrate
                else:
                    D[z, e] = self.config.min_conc

        return D

    def _apply_substrate_bounds(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        n_depth: int,
        n_elements: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply substrate bounds to fix deep substrate concentrations.

        Args:
            lb: Lower bounds (flattened)
            ub: Upper bounds (flattened)
            n_depth: Number of depth points
            n_elements: Number of elements

        Returns:
            Modified (lb, ub) tuple
        """
        if self.config.substrate_indices is None:
            return lb, ub

        depth = self.tm.depth

        # Find depth index where substrate starts
        if self.config.substrate_depth_start is not None:
            deep_idx = np.searchsorted(depth, self.config.substrate_depth_start)
        else:
            # Default: last 20% of depth range
            deep_idx = int(len(depth) * 0.8)

        sub_idx = self.config.substrate_indices
        n_substrate = len(sub_idx)

        # Reshape bounds for easier manipulation
        lb_2d = lb.reshape(n_depth, n_elements)
        ub_2d = ub.reshape(n_depth, n_elements)

        # At deep depths: fix substrate to 1/n_substrate, others to min_conc
        for z in range(deep_idx, n_depth):
            for e in range(n_elements):
                idx = z * n_elements + e
                if e in sub_idx:
                    # Fix substrate element to 1/n_substrate
                    lb[idx] = 1.0 / n_substrate - 1e-6
                    ub[idx] = 1.0 / n_substrate + 1e-6
                else:
                    # Fix non-substrate to minimum
                    lb[idx] = self.config.min_conc
                    ub[idx] = self.config.min_conc + 1e-6

        return lb, ub


def solve_single_pixel(args: Tuple) -> Tuple[int, SolverResult]:
    """
    Solve for a single pixel (for parallel execution).

    Args:
        args: (pixel_idx, I_obs, tm, config, D_prior, D_init, substrate_indices)

    Returns:
        (pixel_idx, SolverResult)
    """
    pixel_idx, I_obs, tm, config, D_prior, D_init, substrate_indices = args

    solver = FreeSolver(tm, config, D_prior)
    result = solver.solve(I_obs, D_init, substrate_indices)

    return pixel_idx, result


class BatchInverseSolver:
    """
    Batch inverse solver with parallel execution.

    Processes multiple pixels in parallel using ProcessPoolExecutor.
    Supports chunked processing for memory efficiency.
    """

    def __init__(
        self,
        transfer_matrix: TransferMatrix,
        config: Optional[SolverConfig] = None,
        D_prior: Optional[np.ndarray] = None,
        elements: Optional[List] = None,
    ):
        self.tm = transfer_matrix
        self.config = config or SolverConfig()
        self.D_prior = D_prior
        self.elements = elements

    def solve_batch(
        self,
        I_obs_batch: np.ndarray,
        D_init_batch: Optional[np.ndarray] = None,
        substrate_indices: Optional[List[int]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[SolverResult]:
        """
        Solve for multiple pixels in parallel.

        Args:
            I_obs_batch: Observed intensities (n_pixels, n_angles, n_elements)
            D_init_batch: Initial guesses (n_pixels, n_depth, n_elements)
            substrate_indices: Substrate element indices
            progress_callback: Called with (completed, total)

        Returns:
            List of SolverResult, one per pixel
        """
        n_pixels = I_obs_batch.shape[0]

        # Prepare arguments
        args_list = []
        for i in range(n_pixels):
            D_init = D_init_batch[i] if D_init_batch is not None else None
            args_list.append((
                i,
                I_obs_batch[i],
                self.tm,
                self.config,
                self.D_prior,
                D_init,
                substrate_indices,
            ))

        # Parallel execution with chunking
        results = [None] * n_pixels
        n_workers = self.config.n_workers
        batch_size = self.config.batch_size

        completed = 0

        # Process in chunks
        for chunk_start in range(0, n_pixels, batch_size):
            chunk_end = min(chunk_start + batch_size, n_pixels)
            chunk_args = args_list[chunk_start:chunk_end]

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(solve_single_pixel, args): args[0]
                          for args in chunk_args}

                for future in as_completed(futures):
                    pixel_idx, result = future.result()
                    results[pixel_idx] = result
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, n_pixels)

        return results

    def solve_batch_sequential(
        self,
        I_obs_batch: np.ndarray,
        D_init_batch: Optional[np.ndarray] = None,
        substrate_indices: Optional[List[int]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[SolverResult]:
        """
        Solve for multiple pixels sequentially (for debugging or small batches).

        Args:
            I_obs_batch: Observed intensities (n_pixels, n_angles, n_elements)
            D_init_batch: Initial guesses (n_pixels, n_depth, n_elements)
            substrate_indices: Substrate element indices
            progress_callback: Called with (completed, total)

        Returns:
            List of SolverResult, one per pixel
        """
        n_pixels = I_obs_batch.shape[0]
        results = []

        solver = FreeSolver(self.tm, self.config, self.D_prior, self.elements)

        for i in range(n_pixels):
            D_init = D_init_batch[i] if D_init_batch is not None else None
            result = solver.solve(I_obs_batch[i], D_init, substrate_indices)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, n_pixels)

        return results


class IterativeMEMSolver:
    """
    Iterative MEM solver with Prior model updating.

    Implements the MATLAB DepthProfiler approach:
    1. L1 initialization: Run L1 solver first, use result as MEM prior
    2. Iterative update: D_prior = mean(D_opt, D_prior_prev)
    3. Neighbor averaging: Average nearby pixels for spatial smoothing

    Usage:
        solver = IterativeMEMSolver(tm, config, D_prior)
        result = solver.solve(I_obs)
    """

    def __init__(
        self,
        transfer_matrix: TransferMatrix,
        config: Optional[SolverConfig] = None,
        D_prior: Optional[np.ndarray] = None,
        elements: Optional[List] = None,
    ):
        """
        Initialize iterative MEM solver.

        Args:
            transfer_matrix: Pre-computed transfer matrix
            config: Solver configuration (should have method='MEM')
            D_prior: Initial prior depth profile
            elements: List of ElementConfig for constraints
        """
        self.tm = transfer_matrix
        self.config = config or SolverConfig(method='MEM')
        self.elements = elements

        # Ensure method is MEM
        if self.config.method != 'MEM':
            self.config.method = 'MEM'

        # Prior model
        if D_prior is not None:
            self.D_prior = D_prior.copy()
        else:
            n_depth = len(transfer_matrix.depth)
            n_elements = transfer_matrix.n_elements
            self.D_prior = np.ones((n_depth, n_elements)) / n_elements

    def solve(
        self,
        I_obs: np.ndarray,
        D_init: Optional[np.ndarray] = None,
        substrate_indices: Optional[List[int]] = None,
        n_outer_loops: int = 3,
        verbose: bool = False,
    ) -> SolverResult:
        """
        Solve inverse problem with iterative Prior updating.

        Args:
            I_obs: Observed intensities (n_angles, n_elements)
            D_init: Initial guess (n_depth, n_elements)
            substrate_indices: Indices of substrate elements
            n_outer_loops: Number of outer loops for Prior updating
            verbose: Print progress

        Returns:
            SolverResult with reconstructed profile
        """
        start_time = time.time()

        # Step 1: L1 initialization (if configured)
        if self.config.prior_update == 'l1_init':
            if verbose:
                print("Step 1: L1 initialization...")

            # Run L1 solver first
            l1_config = SolverConfig(
                method='L1',
                alpha=self.config.alpha,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
            )
            l1_solver = FreeSolver(self.tm, l1_config, self.D_prior, self.elements)
            l1_result = l1_solver.solve(I_obs, D_init, substrate_indices)

            # Use L1 result as MEM prior
            D_prior_current = l1_result.D_opt.copy()
            D_current = l1_result.D_opt.copy()

            if verbose:
                print(f"   L1 result: Chi²={l1_result.chi2:.6f}")
        else:
            D_prior_current = self.D_prior.copy()
            D_current = D_init if D_init is not None else self.D_prior.copy()

        # Step 2: Iterative MEM optimization with Prior updating
        total_iterations = 0

        for outer_loop in range(n_outer_loops):
            if verbose:
                print(f"Outer loop {outer_loop + 1}/{n_outer_loops}...")

            # Create MEM solver with current prior
            mem_solver = FreeSolver(self.tm, self.config, D_prior_current, self.elements)
            result = mem_solver.solve(I_obs, D_current, substrate_indices)
            D_opt = result.D_opt

            total_iterations += result.n_iterations

            if verbose:
                print(f"   Chi²={result.chi2:.6f}, Reg={result.regularization:.6f}")

            # Update Prior model based on strategy
            if self.config.prior_update == 'iterative':
                # Iterative: average with previous prior
                # Dmod = mean(Dopt, DrltprevPar) from MATLAB
                D_prior_current = 0.5 * (D_opt + D_prior_current)
            elif self.config.prior_update in ['l1_init', 'fixed']:
                # Fixed: use L1 result or original prior (no update)
                pass

            # Update initial guess for next iteration
            D_current = D_opt.copy()

        elapsed = time.time() - start_time

        return SolverResult(
            D_opt=D_opt,
            I_opt=result.I_opt,
            chi2=result.chi2,
            regularization=result.regularization,
            objective=result.objective,
            n_iterations=total_iterations,
            elapsed_time=elapsed,
            success=result.success,
        )

    def solve_batch_with_neighbor(
        self,
        I_obs_batch: np.ndarray,
        D_init_batch: Optional[np.ndarray] = None,
        substrate_indices: Optional[List[int]] = None,
        pixel_positions: Optional[np.ndarray] = None,
        n_outer_loops: int = 3,
        verbose: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[SolverResult]:
        """
        Solve for multiple pixels with nearest-neighbor Prior averaging.

        This implements the MATLAB approach:
        - After each outer loop, update Prior using spatial average of neighbors
        - D_prior[i] = mean(D_opt[neighbors of i])

        Args:
            I_obs_batch: Observed intensities (n_pixels, n_angles, n_elements)
            D_init_batch: Initial guesses (n_pixels, n_depth, n_elements)
            substrate_indices: Substrate element indices
            pixel_positions: 2D positions for neighbor finding (n_pixels, 2)
                             If None, treats pixels as 1D sequence
            n_outer_loops: Number of outer loops for Prior updating
            verbose: Print progress
            progress_callback: Called with (completed, total)

        Returns:
            List of SolverResult, one per pixel
        """
        n_pixels = I_obs_batch.shape[0]
        n_depth = len(self.tm.depth)
        n_elements = self.tm.n_elements

        # Initialize
        if D_init_batch is not None:
            D_all = D_init_batch.copy()
        else:
            D_all = np.broadcast_to(
                self.D_prior,
                (n_pixels, n_depth, n_elements)
            ).copy()

        D_prior_all = np.broadcast_to(
            self.D_prior,
            (n_pixels, n_depth, n_elements)
        ).copy()

        # Build neighbor indices
        neighbors = self._build_neighbor_indices(n_pixels, pixel_positions)

        # Step 1: L1 initialization if configured
        if self.config.prior_update == 'l1_init':
            if verbose:
                print("Step 1: L1 initialization for all pixels...")

            l1_config = SolverConfig(
                method='L1',
                alpha=self.config.alpha,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
            )

            for i in range(n_pixels):
                l1_solver = FreeSolver(self.tm, l1_config, self.D_prior)
                D_init = D_all[i] if D_init_batch is not None else None
                l1_result = l1_solver.solve(I_obs_batch[i], D_init, substrate_indices)
                D_all[i] = l1_result.D_opt
                D_prior_all[i] = l1_result.D_opt

                if progress_callback:
                    progress_callback(i + 1, n_pixels * (n_outer_loops + 1))

        # Outer loops with Prior updating
        results = [None] * n_pixels

        for outer_loop in range(n_outer_loops):
            if verbose:
                print(f"Outer loop {outer_loop + 1}/{n_outer_loops}...")

            # Solve for each pixel
            for i in range(n_pixels):
                mem_solver = FreeSolver(self.tm, self.config, D_prior_all[i])
                result = mem_solver.solve(I_obs_batch[i], D_all[i], substrate_indices)
                D_all[i] = result.D_opt
                results[i] = result

                if progress_callback:
                    base = n_pixels if self.config.prior_update == 'l1_init' else 0
                    completed = base + outer_loop * n_pixels + i + 1
                    total = n_pixels * (n_outer_loops + (1 if self.config.prior_update == 'l1_init' else 0))
                    progress_callback(completed, total)

            # Update Prior models using neighbor averaging
            if self.config.prior_update == 'neighbor':
                # Pure neighbor averaging: use average of neighbor optimization results
                D_prior_all = self._update_prior_neighbors_only(D_all, neighbors)
            elif self.config.prior_update == 'iterative':
                # Iterative: include previous prior in average (MATLAB style)
                D_prior_all = self._update_prior_with_neighbors(D_all, D_prior_all, neighbors)

        return results

    def _build_neighbor_indices(
        self,
        n_pixels: int,
        pixel_positions: Optional[np.ndarray],
    ) -> List[List[int]]:
        """
        Build list of neighbor indices for each pixel.

        Args:
            n_pixels: Number of pixels
            pixel_positions: 2D positions (n_pixels, 2) or None for 1D

        Returns:
            List of neighbor index lists for each pixel
        """
        radius = self.config.neighbor_radius
        neighbors = []

        if pixel_positions is None:
            # 1D: neighbors are adjacent pixels (excluding self)
            for i in range(n_pixels):
                nb = []
                for offset in range(-radius, radius + 1):
                    if offset == 0:
                        continue  # Exclude self
                    j = i + offset
                    if 0 <= j < n_pixels:
                        nb.append(j)
                neighbors.append(nb)
        else:
            # 2D: neighbors within radius (excluding self)
            for i in range(n_pixels):
                pos_i = pixel_positions[i]
                nb = []
                for j in range(n_pixels):
                    if i == j:
                        continue  # Exclude self
                    pos_j = pixel_positions[j]
                    dist = np.sqrt(np.sum((pos_i - pos_j) ** 2))
                    if dist <= radius:
                        nb.append(j)
                neighbors.append(nb)

        return neighbors

    def _update_prior_neighbors_only(
        self,
        D_all: np.ndarray,
        neighbors: List[List[int]],
    ) -> np.ndarray:
        """
        Update Prior models using only neighbor optimization results.

        Args:
            D_all: Current optimized profiles (n_pixels, n_depth, n_elements)
            neighbors: List of neighbor indices for each pixel

        Returns:
            Updated prior models (n_pixels, n_depth, n_elements)
        """
        n_pixels = D_all.shape[0]
        D_prior_new = np.zeros_like(D_all)

        for i in range(n_pixels):
            nb_indices = neighbors[i]
            if len(nb_indices) > 0:
                D_neighbors = D_all[nb_indices]
                D_prior_new[i] = np.mean(D_neighbors, axis=0)
            else:
                D_prior_new[i] = D_all[i]

        return D_prior_new

    def _update_prior_with_neighbors(
        self,
        D_all: np.ndarray,
        D_prior_all: np.ndarray,
        neighbors: List[List[int]],
    ) -> np.ndarray:
        """
        Update Prior models using neighbor averaging with previous prior.

        Uses ONLY neighbors (excluding self) for spatial correlation:
        - D_prior[i] = mean(D_opt[neighbors] + D_prior_prev[neighbors])

        This ensures self is NOT included in its own prior model.

        Args:
            D_all: Current optimized profiles (n_pixels, n_depth, n_elements)
            D_prior_all: Previous prior models (n_pixels, n_depth, n_elements)
            neighbors: List of neighbor indices for each pixel (excluding self)

        Returns:
            Updated prior models (n_pixels, n_depth, n_elements)
        """
        n_pixels = D_all.shape[0]
        D_prior_new = np.zeros_like(D_prior_all)

        for i in range(n_pixels):
            nb_indices = neighbors[i]

            if len(nb_indices) > 0:
                # Use only neighbors (self is excluded in _build_neighbor_indices)
                # Combine neighbor optimization results with neighbor previous priors
                D_combined = np.concatenate([
                    D_all[nb_indices],        # Neighbor optimization results
                    D_prior_all[nb_indices],  # Neighbor previous prior values
                ], axis=0)
                D_prior_new[i] = np.mean(D_combined, axis=0)
            else:
                # No neighbors: keep previous prior (don't use self)
                D_prior_new[i] = D_prior_all[i]

        return D_prior_new


class VectorizedBatchSolver:
    """
    Vectorized batch solver using custom Gauss-Newton optimization.

    Processes all pixels simultaneously using NumPy vectorization.
    Much faster than ProcessPoolExecutor for small per-pixel computations.
    """

    def __init__(
        self,
        transfer_matrix: TransferMatrix,
        config: Optional[SolverConfig] = None,
        D_prior: Optional[np.ndarray] = None,
    ):
        self.tm = transfer_matrix
        self.config = config or SolverConfig()

        # Build transfer matrix array: (n_elements, n_angles, n_depth)
        self.T = np.stack([self.tm.get_matrix(e) for e in range(self.tm.n_elements)])
        self.n_elements = self.tm.n_elements
        self.n_depth = len(self.tm.depth)
        self.n_angles = self.tm.n_angles

        # Prior
        if D_prior is not None:
            self.D_prior = D_prior.copy()
        else:
            self.D_prior = np.ones((self.n_depth, self.n_elements)) / self.n_elements

    def solve_batch(
        self,
        I_obs_batch: np.ndarray,
        D_init_batch: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve for multiple pixels using vectorized Gauss-Newton.

        Args:
            I_obs_batch: Observed intensities (n_pixels, n_angles, n_elements)
            D_init_batch: Initial guesses (n_pixels, n_depth, n_elements)
            progress_callback: Called with (iteration, max_iter)

        Returns:
            Tuple of:
                D_opt: Optimized profiles (n_pixels, n_depth, n_elements)
                chi2: Final Chi² values (n_pixels,)
                n_iter: Number of iterations used
        """
        n_pixels = I_obs_batch.shape[0]

        # Normalize observations
        I_sum = I_obs_batch.sum(axis=2, keepdims=True)
        I_norm = I_obs_batch / np.maximum(I_sum, 1e-10)

        # Initialize D
        if D_init_batch is not None:
            D = D_init_batch.copy()
        else:
            D = np.broadcast_to(self.D_prior, (n_pixels, self.n_depth, self.n_elements)).copy()

        # Gauss-Newton iterations
        alpha = self.config.alpha
        degree = self.config.degree

        for iteration in range(self.config.max_iter):
            # Forward model: I_pred[p, a, e] = sum_d T[e, a, d] * D[p, d, e]
            I_pred = self._forward_batch(D)

            # Normalize predictions
            I_pred_sum = I_pred.sum(axis=2, keepdims=True)
            I_pred_norm = I_pred / np.maximum(I_pred_sum, 1e-10)

            # Residual
            residual = I_pred_norm - I_norm  # (n_pixels, n_angles, n_elements)

            # Gradient
            grad = self._gradient_batch(D, residual, degree, alpha)

            # Compute step using gradient descent
            # (For Gauss-Newton, we would compute Hessian, but gradient descent is simpler)
            step_size = 0.1 / (iteration + 1)  # Decreasing step size
            D_new = D - step_size * grad

            # Apply bounds
            D_new = np.clip(D_new, self.config.min_conc, self.config.max_conc)

            # Check convergence
            change = np.max(np.abs(D_new - D))
            D = D_new

            if progress_callback:
                progress_callback(iteration + 1, self.config.max_iter)

            if change < self.config.tol:
                break

        # Compute final Chi²
        I_pred = self._forward_batch(D)
        I_pred_norm = I_pred / np.maximum(I_pred.sum(axis=2, keepdims=True), 1e-10)
        chi2 = np.sum((I_norm - I_pred_norm) ** 2, axis=(1, 2))

        return D, chi2, iteration + 1

    def _forward_batch(self, D: np.ndarray) -> np.ndarray:
        """
        Vectorized forward model.

        Args:
            D: Depth profiles (n_pixels, n_depth, n_elements)

        Returns:
            I_pred: Predicted intensities (n_pixels, n_angles, n_elements)
        """
        # T: (n_elem, n_angles, n_depth)
        # D: (n_pixels, n_depth, n_elem)
        # I: (n_pixels, n_angles, n_elem)
        I_pred = np.zeros((D.shape[0], self.n_angles, self.n_elements))
        for e in range(self.n_elements):
            I_pred[:, :, e] = D[:, :, e] @ self.T[e].T
        return I_pred

    def _gradient_batch(
        self,
        D: np.ndarray,
        residual: np.ndarray,
        degree: float,
        alpha: float,
    ) -> np.ndarray:
        """
        Vectorized gradient computation.

        Args:
            D: Current profiles (n_pixels, n_depth, n_elements)
            residual: I_pred - I_obs (n_pixels, n_angles, n_elements)
            degree: Data fit weight
            alpha: Regularization weight

        Returns:
            grad: Gradient (n_pixels, n_depth, n_elements)
        """
        n_pixels = D.shape[0]

        # Data term gradient
        # residual: (n_pixels, n_angles, n_elem)
        # T: (n_elem, n_angles, n_depth)
        # grad_data: (n_pixels, n_depth, n_elem)
        grad_data = np.zeros_like(D)
        for e in range(self.n_elements):
            grad_data[:, :, e] = degree * residual[:, :, e] @ self.T[e]

        # Regularization gradient
        if self.config.method == 'L2':
            D_prior_batch = np.broadcast_to(self.D_prior, D.shape)
            grad_reg = 2 * alpha * (D - D_prior_batch)
        elif self.config.method == 'MEM':
            # Bregman div D_φ(D‖m0), φ=x·log(x): grad = log(D/m0)
            D_safe = np.maximum(D, 1e-10)
            D_prior_safe = np.maximum(self.D_prior, 1e-10)
            D_prior_batch = np.broadcast_to(D_prior_safe, D.shape)
            grad_reg = alpha * np.log(D_safe / D_prior_batch)
        else:
            grad_reg = 0

        return grad_data + grad_reg


@dataclass
class SemiFreeSolverConfig:
    """Configuration for semi-free model solver.

    Semi-free model: substrate follows erf profile, film elements freely optimized.
    This provides a good balance between flexibility and stability.

    This enables joint optimization of:
    - Film layer concentrations (free at each depth)
    - Film thickness (erf interface position)
    - Interface abruptness (erf width) - NOTE: fixed by default
    - Substrate composition ratios

    Element Classification (following MATLAB Run_DepthProfiling.m):
    - free: Film elements freely optimized (e.g., Si⁴⁺ in SiO2)
    - linked: Elements determined by charge balance (e.g., O²⁻ in SiO2)
    - const: Fixed concentration elements
    - substrate: Substrate elements following erf profile (e.g., Si⁰)

    Constraints:
    - Charge conservation: Σ(charge × concentration) = 0 at each depth
    - Concentration sum: Σconcentration = 1 at each depth

    NOTE on abruptness optimization:
    ARXPS is sensitive to interface position but insensitive to interface width.
    Therefore, abruptness optimization is disabled by default (optimize_abruptness=False).
    If the interface width is known from other measurements, provide it via abruptness_init.
    """
    # Film structure parameters
    optimize_thickness: bool = True      # Optimize thickness parameter
    optimize_abruptness: bool = False    # Optimize abruptness parameter (disabled by default)
    optimize_substrate_ratio: bool = True  # Optimize substrate element ratios

    # Initial values
    thickness_init: float = 30.0         # Initial thickness (Å)
    abruptness_init: float = 5.0         # Initial abruptness FWHM (Å)

    # Bounds
    thickness_bounds: Tuple[float, float] = (5.0, 100.0)  # Min/max thickness
    abruptness_bounds: Tuple[float, float] = (1.0, 20.0)  # Min/max abruptness

    # Element classification (indices) - MATLAB style
    film_indices: Optional[List[int]] = None       # Film element indices (free)
    substrate_indices: Optional[List[int]] = None  # Substrate element indices

    # MATLAB-style element classification
    free_indices: Optional[List[int]] = None       # Freely optimized (cations in film)
    linked_indices: Optional[List[int]] = None     # Determined by charge balance (anions)
    const_indices: Optional[List[int]] = None      # Fixed concentration

    # Constraint settings
    use_charge_constraint: bool = True   # Enforce charge neutrality

    # Normalization mode for observed intensities.
    # 'substrate': divide by per-angle substrate signal (MATLAB default)
    # 'scalar_substrate': divide by substrate signal at reference angle only
    #   (preserves [0,∞) range and angle-dependent info, avoids blow-up)
    # 'total': divide by total signal (loses thickness sensitivity)
    intensity_normalization: str = 'substrate'

    # Override for freelinkedconst (from HDF5 or user-specified)
    # When set, bypasses charge-based computation entirely.
    # Length must match len(free_indices).
    free_linked_const_override: Optional[List[float]] = None

    # Constraint settings
    use_sum_constraint: bool = True      # Enforce Σconcentration = 1
    # Sum constraint mode: 'equality' only (penalty removed)
    sum_constraint_mode: str = 'equality'

    # Normalize free concentrations so sum(D_free * free_linked_const) = 1
    # This enforces D_free + D_linked = 1 - D_sub by construction.
    normalize_free_conc: bool = False

    # Threshold to force film scale to zero when substrate dominates.
    substrate_cutoff: Optional[float] = None

    # Smooth degenerate depth handling (replaces binary active/degenerate mask)
    # When True, uses sigmoid weighting for constraint rows near the
    # film/substrate boundary, reducing rank discontinuity.
    smooth_degenerate: bool = False          # Default False (opt-in)
    smooth_degenerate_scale: float = 1e-4   # Sigmoid transition width

    # Stoichiometric mode (for 090317-style O/N independent optimization)
    stoichiometric_mode: str = 'charge_balance'
    # Options:
    #   'charge_balance' (default): D_linked = charge-balanced from D_free
    #   'stoichiometric': D_derived calculated from explicit stoichiometry rules
    #                     (e.g., D_SiO = 0.5*D_O + 0.75*D_N for SiON system)

    # Stoichiometric rules for 'stoichiometric' mode
    # Dict mapping derived_element_idx -> (source_indices, coefficients)
    # Example: {1: ([2, 3], [0.5, 0.75])} means D[1] = 0.5*D[2] + 0.75*D[3]
    stoichiometric_rules: Optional[Dict[int, Tuple[List[int], List[float]]]] = None

    # Stoichiometric constraint coefficients for 'stoichiometric' mode
    # Constraint: sum(coeff[i] * D[free[i]]) = 1 - D_substrate at each depth
    # For SiON: coeffs = [1.5, 1.75] means (3/2)*D_O + (7/4)*D_N = 1 - D_SiB
    stoichiometric_constraint_coeffs: Optional[List[float]] = None

    # ========================================
    # Composition Gradient Mode (Phase 11.1)
    # ========================================
    # For layers with continuous composition variation (e.g., a mixed oxide
    # A_xB_{1-x}O_y whose cation ratio changes with depth)
    # Unlike stoichiometric_mode (discrete composition per layer),
    # gradient mode handles smooth composition changes within a layer.
    #
    # Use cases:
    #   - Mixed-oxide interface: the cation ratio varies continuously with depth
    #   - Graded oxide/nitride interfaces
    #   - Diffusion profiles
    #
    # Comparison:
    #   stoichiometric_mode: layer-to-layer discrete (e.g. an oxynitride layer on SiON)
    #   gradient_mode: within-layer continuous (graded mixed oxide)

    gradient_mode: bool = False  # Enable composition gradient optimization

    gradient_type: str = 'linear'
    # Options:
    #   'linear': c(z) = c0 + slope * z (2 params per element)
    #   'quadratic': c(z) = c0 + c1*z + c2*z² (3 params per element)

    # Which elements have gradient (by index)
    # If None, all free_indices elements can have gradients
    gradient_element_indices: Optional[List[int]] = None

    # Gradient slope bounds per element: {element_idx: (min_slope, max_slope)}
    # Unit: concentration change per Å (e.g., 0.01 means 1% change per Å)
    # If None, defaults to (-0.05, 0.05) for all gradient elements
    gradient_bounds: Optional[Dict[int, Tuple[float, float]]] = None

    # Initial gradient slopes per element: {element_idx: initial_slope}
    # If None, defaults to 0.0 (start from uniform composition)
    gradient_init: Optional[Dict[int, float]] = None

    # Legacy sum constraint: include erf_sub in the equality constraint.
    # Old: Σ flc*film_conc + erf_sub = 1  (film_conc sums to 1-erf_sub)
    # New: Σ flc*film_conc = 1            (film_conc always sums to 1)
    # The old constraint was physically incorrect (D_opt rows don't sum to 1
    # at the interface), but it provided implicit thickness regularization
    # that made SLSQP more robust. Set True to use the old behavior.
    use_legacy_sum_constraint: bool = False

    # Nonlinear sum constraint: use _build_full_profile().sum(axis=1) = 1.
    # This was the original Python constraint.
    # Provides strong thickness coupling (all film depths affected) but
    # requires SLSQP to compute numerical Jacobian.
    # Takes precedence over use_legacy_sum_constraint if both are True.
    use_nonlinear_sum_constraint: bool = False

    # Explicit thickness prior (opt-in; t_prior_center=None keeps the legacy objective).
    # When t_prior_center is set, adds P(t) = (t - center)² / (2·sigma²) to the
    # objective OUTSIDE alpha·reg, so its strength is independent of alpha and
    # method. None (default) = off = exact legacy behavior.
    #
    # t_prior_sigma has no default: it must be given whenever t_prior_center
    # is set. It is a width in Å of a quadratic penalty, NOT a statistical
    # standard deviation. With use_xreg_weighting=True the data term 0.5·chi2
    # is a sum of squared residuals scaled by degree/Xreg² (Xreg = angle-mean
    # of the normalized observations), not by the measurement-noise variance,
    # so the objective is not a negative log-likelihood and P(t) is not a calibrated
    # Gaussian prior with standard deviation sigma. How strongly the penalty
    # acts relative to the data depends on degree, the residual normalization
    # and the number of angles and elements, so a sigma chosen under one
    # objective convention (e.g. degree=1, use_xreg_weighting=True, per-angle
    # substrate normalization) must be chosen again if that convention changes.
    t_prior_center: Optional[float] = None
    t_prior_sigma: Optional[float] = None

    def __post_init__(self):
        if self.t_prior_center is not None and self.t_prior_sigma is None:
            raise ValueError(
                "t_prior_sigma must be given when t_prior_center is set "
                "(it has no default; see SemiFreeSolverConfig)")


class SemiFreeSolver:
    """
    Semi-free model solver (MATLAB Run_DepthProfiling.m style).

    Substrate follows erf profile (physically constrained),
    film elements are freely optimized at each depth.

    This solver jointly optimizes:
    1. Film layer concentrations (normalized) for "free" elements
    2. Thickness parameter (erf position)
    3. Abruptness parameter (erf width)
    4. Substrate element ratios

    Element Classification (MATLAB style):
    - free: Film cations, freely optimized (e.g., Si⁴⁺ in SiO2)
    - linked: Film anions, determined by charge balance (e.g., O²⁻ in SiO2)
    - substrate: Substrate elements, follows erf profile (e.g., Si⁰)

    The substrate profile is modeled as:
        D_substrate(z) = substrate_ratio × 0.5 × (1 + erf((z - thickness) / sigma))

    The free element concentrations are:
        D_free(z) = normalized_conc × (1 - sum(D_substrate))

    The linked element concentrations are computed from charge balance:
        D_linked(z) = D_free(z) × (1 - charge_free / charge_linked)

    This ensures charge neutrality: Σ(charge × concentration) = 0 at each depth.

    Recommended for most use cases due to better stability than FreeSolver.
    """

    def __init__(
        self,
        transfer_matrix: TransferMatrix,
        config: Optional[SolverConfig] = None,
        semi_free_config: Optional['SemiFreeSolverConfig'] = None,
        D_prior: Optional[np.ndarray] = None,
        elements: Optional[List] = None,
        use_numba: bool = True,
    ):
        """
        Initialize semi-free solver.

        Args:
            transfer_matrix: Pre-computed transfer matrix
            config: Solver configuration
            semi_free_config: Semi-free model configuration
            D_prior: Prior depth profile (n_depth, n_elements). Serves two roles:

                1. **Initial guess source** — film concentrations at each depth
                   are extracted from D_prior[:, film_idx] to construct x0.
                2. **Regularization reference** (m0) — used as the prior model
                   for L2 (‖D - m0‖²) and MEM (KL divergence) regularization.

                The solver creates an internal copy; the caller's array is
                never modified.

                **Recommended usage by method**:

                - L1/LLS: ``D_prior=None`` (sparsity prior: min_conc everywhere)
                - L2/MEM: supply expected profile shape as D_prior

                If None, defaults to min_conc (≈1e-6) for all elements/depths,
                which promotes sparsity.
            elements: Element configurations (with oxidation_state) for charge
                      and stoichiometric constraints
            use_numba: Use Numba-accelerated kernels if available
        """
        self.tm = transfer_matrix
        self.config = config or SolverConfig()
        self.semi_free_config = semi_free_config or SemiFreeSolverConfig()
        self.elements = elements
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self._degree_scale = 1.0  # set by solve() for scalar_substrate mode

        if self.semi_free_config.normalize_free_conc:
            # Disable Numba kernels because gradient/forward assume unnormalized variables.
            self.use_numba = False

        self.depth = transfer_matrix.depth
        self.n_depth = len(self.depth)
        self.n_elements = transfer_matrix.n_elements

        # Prior model
        # Default: zero prior for L1 sparsity (sum|D|)
        # This promotes sparsity and suppresses false peaks in deep regions
        if D_prior is not None:
            self.D_prior = D_prior.copy()
        else:
            # Zero prior (with small epsilon to avoid numerical issues)
            self.D_prior = np.full((self.n_depth, self.n_elements), 1e-6)

        # Element classification
        # For backward compatibility, support both old and new naming
        cfg = self.semi_free_config
        self.free_idx = cfg.free_indices or cfg.film_indices or []
        self.linked_idx = cfg.linked_indices or []
        self.sub_idx = cfg.substrate_indices or []

        # For backward compatibility
        self.film_idx = self.free_idx

        # Validate indices - no overlap allowed
        all_idx = set(self.free_idx) | set(self.linked_idx) | set(self.sub_idx)
        total_count = len(self.free_idx) + len(self.linked_idx) + len(self.sub_idx)
        if len(all_idx) != total_count:
            raise ValueError("free, linked, and substrate indices must not overlap")

        # Pre-compute numpy arrays for Numba kernels (int64 required)
        self._free_idx_array = np.array(self.free_idx, dtype=np.int64)
        self._sub_idx_array = np.array(self.sub_idx, dtype=np.int64)
        self._linked_idx_array = np.array(self.linked_idx, dtype=np.int64)

        # Compute charge-based coefficients for linked elements
        self._compute_charge_coefficients()

        # Setup stoichiometric mode
        self._setup_stoichiometric_mode()

        # Setup gradient mode (Phase 11.1)
        self._setup_gradient_mode()

    def _compute_charge_coefficients(self):
        """
        Compute coefficients for linked element calculation based on charge balance.

        MATLAB formula (line 446): D_linked = sum(D_free × (freelinkedconst - 1))
        where freelinkedconst = 1 - charge_free / mean(charge_linked)

        For SiO2 example:
        - free: Si⁴⁺ (charge=+4)
        - linked: O²⁻ (charge=-2)
        - freelinkedconst = 1 - (+4) / (-2) = 1 + 2 = 3
        - D_O = D_Si × (3 - 1) = D_Si × 2
        This gives [O] = 2 × [Si], which satisfies SiO2 stoichiometry!
        """
        self.free_linked_const = np.ones(len(self.free_idx))

        # Use override if provided (e.g., from HDF5 ezdeprof/linkedconst)
        if self.semi_free_config.free_linked_const_override is not None:
            override = np.asarray(self.semi_free_config.free_linked_const_override, dtype=np.float64)
            if len(override) == len(self.free_idx):
                self.free_linked_const = np.maximum(override, 1.0)
                return
            # Length mismatch: fall through to charge-based computation

        if not self.linked_idx or self.elements is None:
            return

        # Get oxidation states
        charges = []
        for elem in self.elements:
            ox = getattr(elem, 'oxidation_state', None)
            charges.append(ox if ox is not None else 0)
        charges = np.array(charges)

        # Get linked charge (average if multiple linked elements)
        linked_charges = charges[self.linked_idx]
        mean_linked_charge = np.mean(linked_charges) if len(linked_charges) > 0 else 0

        if mean_linked_charge == 0:
            return  # No charge constraint possible

        # Compute freelinkedconst for each free element
        # MATLAB: freelinkedconst = 1 - charge(free) / mean(charge(linked))
        for i, free_e in enumerate(self.free_idx):
            self.free_linked_const[i] = 1 - charges[free_e] / mean_linked_charge

        # MATLAB line 33: freelinkedconst(freelinkedconst<1) = 1
        # Clamp to minimum of 1.0 to prevent negative linked concentrations
        self.free_linked_const = np.maximum(self.free_linked_const, 1.0)

    def _setup_stoichiometric_mode(self):
        """
        Setup stoichiometric mode for 090317-style O/N independent optimization.

        In stoichiometric mode:
        - Multiple elements (e.g., O and N) are optimized independently as 'free'
        - Derived elements (e.g., Si-O) are computed from stoichiometry rules
        - A stoichiometric constraint ensures mass conservation

        Example for SiON system (NON pattern):
        - free_indices = [O_idx, N_idx]
        - substrate_indices = [Si_bulk_idx]
        - stoichiometric_rules = {Si_ox_idx: ([O_idx, N_idx], [0.5, 0.75])}
          → D_SiO = 0.5*D_O + 0.75*D_N
        - stoichiometric_constraint_coeffs = [1.5, 1.75]
          → (3/2)*D_O + (7/4)*D_N = 1 - D_SiB
        """
        cfg = self.semi_free_config

        self._stoichiometric_mode = cfg.stoichiometric_mode
        self._stoichiometric_rules = cfg.stoichiometric_rules or {}
        self._stoichiometric_constraint_coeffs = cfg.stoichiometric_constraint_coeffs

        # Identify derived element indices (those computed from rules)
        self._derived_idx = list(self._stoichiometric_rules.keys())

        if self._stoichiometric_mode == 'stoichiometric':
            # Validate configuration
            if not self._stoichiometric_rules:
                raise ValueError(
                    "stoichiometric_mode='stoichiometric' requires stoichiometric_rules"
                )
            if self._stoichiometric_constraint_coeffs is None:
                raise ValueError(
                    "stoichiometric_mode='stoichiometric' requires stoichiometric_constraint_coeffs"
                )
            if len(self._stoichiometric_constraint_coeffs) != len(self.free_idx):
                raise ValueError(
                    f"stoichiometric_constraint_coeffs length ({len(self._stoichiometric_constraint_coeffs)}) "
                    f"must match free_indices length ({len(self.free_idx)})"
                )

    def _setup_gradient_mode(self):
        """
        Setup gradient mode for continuous composition variation within layers.

        In gradient mode:
        - Film element concentrations vary linearly/quadratically with depth
        - c(z) = c0 + slope * (z - z_surface) for linear mode
        - Useful for graded interfaces (mixed oxides) or diffusion profiles

        Gradient parameters are added to the optimization vector.
        """
        cfg = self.semi_free_config

        if cfg.gradient_mode and 'gradient_mode' not in _ENABLED_HELD_FEATURES:
            raise NotImplementedError(
                "SemiFreeSolverConfig(gradient_mode=True) is not available in this "
                "release of deppro; the composition-gradient mode is held back for a "
                "later version.")

        self._gradient_mode = cfg.gradient_mode
        self._gradient_type = cfg.gradient_type

        if not self._gradient_mode:
            # Gradient mode disabled
            self._gradient_element_idx = []
            self._gradient_bounds = {}
            self._gradient_init = {}
            self._n_gradient_params = 0
            return

        # Determine which elements have gradients
        if cfg.gradient_element_indices is not None:
            self._gradient_element_idx = cfg.gradient_element_indices
        else:
            # Default: all free elements can have gradients
            self._gradient_element_idx = list(self.free_idx)

        # Validate gradient elements are in free_idx
        for idx in self._gradient_element_idx:
            if idx not in self.free_idx:
                raise ValueError(
                    f"gradient_element_indices contains {idx} which is not in free_indices"
                )

        # Setup gradient bounds
        default_bounds = (-0.02, 0.02)  # ±2% per Å default
        if cfg.gradient_bounds is not None:
            self._gradient_bounds = cfg.gradient_bounds
        else:
            self._gradient_bounds = {
                idx: default_bounds for idx in self._gradient_element_idx
            }

        # Setup initial values
        if cfg.gradient_init is not None:
            self._gradient_init = cfg.gradient_init
        else:
            self._gradient_init = {idx: 0.0 for idx in self._gradient_element_idx}

        # Number of gradient parameters
        if self._gradient_type == 'linear':
            # One slope per gradient element
            self._n_gradient_params = len(self._gradient_element_idx)
        elif self._gradient_type == 'quadratic':
            # Two coefficients (c1, c2) per gradient element
            self._n_gradient_params = 2 * len(self._gradient_element_idx)
        else:
            raise ValueError(f"Unknown gradient_type: {self._gradient_type}")

    def _apply_gradient_to_profile(
        self,
        free_conc: np.ndarray,
        gradient_params: np.ndarray,
    ) -> np.ndarray:
        """
        Apply gradient correction to free element concentrations.

        Args:
            free_conc: Base concentration profile (n_depth, n_free)
            gradient_params: Gradient parameters (slopes or polynomial coeffs)

        Returns:
            Modified concentration profile with gradient applied
        """
        if not self._gradient_mode or len(gradient_params) == 0:
            return free_conc

        # Make a copy to avoid modifying original
        conc_with_gradient = free_conc.copy()

        # Depth relative to surface (z=0 at surface)
        z_rel = self.depth - self.depth[0]

        if self._gradient_type == 'linear':
            # c(z) = c0 + slope * z
            for i, elem_idx in enumerate(self._gradient_element_idx):
                # Find position in free_idx
                free_pos = self.free_idx.index(elem_idx)
                slope = gradient_params[i]
                conc_with_gradient[:, free_pos] += slope * z_rel

        elif self._gradient_type == 'quadratic':
            # c(z) = c0 + c1*z + c2*z²
            for i, elem_idx in enumerate(self._gradient_element_idx):
                free_pos = self.free_idx.index(elem_idx)
                c1 = gradient_params[2 * i]
                c2 = gradient_params[2 * i + 1]
                conc_with_gradient[:, free_pos] += c1 * z_rel + c2 * z_rel**2

        # Clip to valid range
        conc_with_gradient = np.clip(
            conc_with_gradient,
            self.config.min_conc,
            self.config.max_conc
        )

        return conc_with_gradient

    def _dump_diagnostics(self, x0, result, constraints, bounds):
        """Save diagnostic info on SLSQP failure for debugging.

        Creates an .npz file with x0, x_opt, bounds, constraint violations,
        and SVD of constraint Jacobians (for detecting rank deficiency).
        Only called when verbose=True and optimization fails.
        """
        import json
        import datetime
        from pathlib import Path

        diag = {
            'timestamp': datetime.datetime.now().isoformat(),
            'status': int(result.status),
            'message': str(result.message),
            'n_iterations': int(result.nit),
            'objective': float(result.fun),
            'x0_norm': float(np.linalg.norm(x0)),
            'x_opt_norm': float(np.linalg.norm(result.x)),
            'n_params': len(x0),
            'n_constraints': len(constraints),
        }

        # Constraint violation and SVD at final x
        for i, c in enumerate(constraints):
            try:
                val = np.atleast_1d(c['fun'](result.x))
                diag[f'constraint_{i}_max_violation'] = float(np.max(np.abs(val)))
                diag[f'constraint_{i}_n_rows'] = int(len(val))
            except Exception as e:
                diag[f'constraint_{i}_error'] = str(e)

            if 'jac' in c and c['jac'] is not None:
                try:
                    J = np.atleast_2d(c['jac'](result.x))
                    if J.shape[0] > 0:
                        s = np.linalg.svd(J, compute_uv=False)
                        diag[f'constraint_{i}_svd_min'] = float(s[-1])
                        diag[f'constraint_{i}_svd_max'] = float(s[0])
                        diag[f'constraint_{i}_condition'] = float(
                            s[0] / max(s[-1], 1e-15)
                        )
                        diag[f'constraint_{i}_rank'] = int(np.sum(s > 1e-10))
                except Exception as e:
                    diag[f'constraint_{i}_svd_error'] = str(e)

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        diag_dir = Path(__file__).resolve().parents[2] / '_archived' / 'slsqp_diagnostics'
        diag_dir.mkdir(parents=True, exist_ok=True)
        fname = diag_dir / f'_slsqp_diagnostic_{ts}.npz'
        try:
            np.savez(
                fname,
                x0=x0,
                x_opt=result.x,
                bounds_lb=np.array(bounds.lb) if hasattr(bounds, 'lb') else np.array([]),
                bounds_ub=np.array(bounds.ub) if hasattr(bounds, 'ub') else np.array([]),
                diag_json=json.dumps(diag, indent=2),
            )
            print(f"[SemiFreeSolver] SLSQP diagnostic saved: {fname}")
        except Exception as e:
            print(f"[SemiFreeSolver] Failed to save diagnostic: {e}")

        # Print summary to console
        print(f"[SemiFreeSolver] SLSQP status={result.status}: {result.message}")
        print(f"  iterations={result.nit}, objective={result.fun:.6e}")
        for i in range(len(constraints)):
            key = f'constraint_{i}_max_violation'
            if key in diag:
                cond_key = f'constraint_{i}_condition'
                cond_str = f", cond={diag[cond_key]:.1e}" if cond_key in diag else ""
                print(f"  constraint[{i}]: max_violation={diag[key]:.2e}{cond_str}")

    def _build_substrate_profile(
        self,
        thickness: float,
        abruptness: float,
        substrate_ratios: np.ndarray,
    ) -> np.ndarray:
        """
        Build substrate profile using error function.

        D_sub(z, e) = ratio_e × 0.5 × (1 + erf((z - thickness) / sigma))

        Args:
            thickness: Film thickness (Å)
            abruptness: Interface abruptness FWHM (Å)
            substrate_ratios: Concentration ratios for substrate elements (sum=1)

        Returns:
            Substrate profile (n_depth, n_substrate_elements)
        """
        # Stage -1 extraction: the body moved to semifree_parametrization (behavior unchanged).
        return _build_substrate_profile_pure(
            self.depth, thickness, abruptness, substrate_ratios,
            use_numba=self.use_numba,
        )

    def _build_full_profile(
        self,
        free_conc: np.ndarray,
        thickness: float,
        abruptness: float,
        substrate_ratios: np.ndarray,
        gradient_params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build full depth profile from optimized parameters.

        Follows MATLAB CalcDistrib function (line 437-451):
        1. D_substrate = ratio × erf(...)
        2. D_free = normalized_conc × (1 - Σ D_substrate)
        3. D_linked = Σ (D_free × (freelinkedconst - 1))

        If gradient_mode is enabled, applies depth-dependent composition changes.

        Args:
            free_conc: Normalized free element concentrations (n_depth, n_free)
            thickness: Film thickness (Å)
            abruptness: Interface abruptness (Å)
            substrate_ratios: Substrate element ratios
            gradient_params: Gradient coefficients (slopes, etc.)

        Returns:
            Full depth profile (n_depth, n_elements)
        """
        # Apply gradient correction if enabled
        if self._gradient_mode and gradient_params is not None and len(gradient_params) > 0:
            free_conc = self._apply_gradient_to_profile(free_conc, gradient_params)

        # Use stoichiometric mode if configured
        # NOTE: normalize_free_conc is applied on the charge_balance side (the extraction target), so
        # it is applied here before entering stoichiometric (same order as before the extraction).
        if self._stoichiometric_mode == 'stoichiometric':
            if self.semi_free_config.normalize_free_conc:
                weighted_sum = np.sum(free_conc * self.free_linked_const, axis=1, keepdims=True)
                weighted_sum = np.maximum(weighted_sum, self.config.min_conc)
                free_conc = free_conc / weighted_sum
            return self._build_full_profile_stoichiometric(
                free_conc, thickness, abruptness, substrate_ratios
            )

        # Stage -1 extraction: the charge_balance body moved to semifree_parametrization (behavior unchanged).
        return _build_full_profile_core(
            free_conc, thickness, abruptness, substrate_ratios,
            depth=self.depth,
            n_elements=self.n_elements,
            free_idx=self.free_idx,
            sub_idx=self.sub_idx,
            linked_idx=self.linked_idx,
            free_linked_const=self.free_linked_const,
            min_conc=self.config.min_conc,
            max_conc=self.config.max_conc,
            substrate_cutoff=self.semi_free_config.substrate_cutoff,
            normalize_free_conc=self.semi_free_config.normalize_free_conc,
            free_idx_array=self._free_idx_array,
            sub_idx_array=self._sub_idx_array,
            linked_idx_array=self._linked_idx_array,
            use_numba=self.use_numba,
        )

    def _build_full_profile_stoichiometric(
        self,
        free_conc: np.ndarray,
        thickness: float,
        abruptness: float,
        substrate_ratios: np.ndarray,
    ) -> np.ndarray:
        """
        Build full depth profile using stoichiometric mode (090317-style).

        In this mode:
        - free_conc contains raw concentrations for free elements (O, N in SiON)
        - Derived elements (Si-O) are computed from stoichiometry rules
        - Substrate follows erf profile

        For SiON example:
        - free_conc[:, 0] = D_O (oxygen concentration)
        - free_conc[:, 1] = D_N (nitrogen concentration)
        - D_SiO = 0.5*D_O + 0.75*D_N (from Si3N4 and SiO2 stoichiometry)
        """
        D = np.zeros((self.n_depth, self.n_elements))

        # 1. Substrate profile (same as charge_balance mode)
        D_sub = self._build_substrate_profile(thickness, abruptness, substrate_ratios)
        D[:, self.sub_idx] = D_sub

        # 2. Free element concentrations (directly from optimization)
        # Note: In stoichiometric mode, free_conc is NOT normalized
        # It's the raw concentration that satisfies stoichiometric constraint
        for i, free_e in enumerate(self.free_idx):
            D[:, free_e] = free_conc[:, i]

        # 3. Derived element concentrations from stoichiometry rules
        # Example: D_SiO = 0.5*D_O + 0.75*D_N
        for derived_idx, (source_indices, coeffs) in self._stoichiometric_rules.items():
            D[:, derived_idx] = sum(
                D[:, src] * c for src, c in zip(source_indices, coeffs)
            )

        # Clip to valid range
        D = np.clip(D, self.config.min_conc, self.config.max_conc)

        return D

    def _pack_parameters(
        self,
        film_conc: np.ndarray,
        thickness: float,
        abruptness: float,
        substrate_ratios: np.ndarray,
        gradient_params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pack parameters into optimization vector.

        Vector layout:
        [film_conc.flatten(), thickness, abruptness, substrate_ratios, gradient_params]
        """
        parts = [
            film_conc.flatten(),
            [thickness, abruptness],
            substrate_ratios,
        ]
        if self._gradient_mode and gradient_params is not None:
            parts.append(gradient_params)
        elif self._gradient_mode:
            # Default: zero gradients
            parts.append(np.zeros(self._n_gradient_params))

        x = np.concatenate(parts)
        return x

    def _unpack_parameters(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        """Unpack optimization vector to parameters.

        Returns:
            film_conc: Film concentrations (n_depth, n_film)
            thickness: Film thickness
            abruptness: Interface abruptness
            substrate_ratios: Substrate element ratios
            gradient_params: Gradient coefficients (empty if gradient_mode=False)
        """
        n_film = len(self.film_idx)
        n_sub = len(self.sub_idx)

        # Film concentrations
        film_size = self.n_depth * n_film
        film_conc = x[:film_size].reshape(self.n_depth, n_film)

        # Structure parameters
        thickness = x[film_size]
        abruptness = x[film_size + 1]

        # Substrate ratios
        sub_start = film_size + 2
        sub_end = sub_start + n_sub
        substrate_ratios = x[sub_start:sub_end]

        # Gradient parameters
        if self._gradient_mode and self._n_gradient_params > 0:
            gradient_params = x[sub_end:sub_end + self._n_gradient_params]
        else:
            gradient_params = np.array([])

        return film_conc, thickness, abruptness, substrate_ratios, gradient_params

    def build_x0(
        self,
        film_conc: Optional[np.ndarray] = None,
        thickness: Optional[float] = None,
        abruptness: Optional[float] = None,
        substrate_ratios: Optional[np.ndarray] = None,
        gradient_params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build an x0 parameter vector for use with ``solve(x0_override=...)``.

        Each argument is optional. When ``None``, the value is taken from the
        solver's defaults (``semi_free_config`` and ``D_prior``).

        Args:
            film_conc: Film concentrations (n_depth, n_film). If ``None``,
                derived from ``D_prior[:, film_idx]``.
            thickness: Film thickness [Å]. Default: ``semi_free_config.thickness_init``.
            abruptness: Interface abruptness [Å]. Default: ``semi_free_config.abruptness_init``.
            substrate_ratios: Substrate element ratios (n_sub,). Default: from D_prior
                or uniform ``1/n_sub``.
            gradient_params: Gradient coefficients. Default: zeros.

        Returns:
            1-D parameter vector ``x0`` with layout::

                [film_conc.ravel(), thickness, abruptness, sub_ratios, (grad_params)]

        Example::

            solver = SemiFreeSolver(tm, cfg, semi_cfg, elements=elements)
            # Start from a different thickness
            x0 = solver.build_x0(thickness=30.0)
            result = solver.solve(I_obs, x0_override=x0)
        """
        n_film = len(self.film_idx)
        n_sub = len(self.sub_idx)

        # Film concentrations
        if film_conc is None:
            film_conc = self.D_prior[:, self.film_idx].copy()
            film_conc = np.clip(film_conc, self.config.min_conc, self.config.max_conc)

        if film_conc.shape != (self.n_depth, n_film):
            raise ValueError(
                f"film_conc shape {film_conc.shape} != expected ({self.n_depth}, {n_film})"
            )

        # Thickness
        if thickness is None:
            thickness = self.semi_free_config.thickness_init

        # Abruptness
        if abruptness is None:
            abruptness = self.semi_free_config.abruptness_init

        # Substrate ratios
        if substrate_ratios is None:
            if n_sub > 0:
                sub_total = self.D_prior[:, self.sub_idx].mean(axis=0)
                sub_sum = sub_total.sum()
                if sub_sum > 0:
                    substrate_ratios = sub_total / sub_sum
                else:
                    substrate_ratios = np.ones(n_sub) / n_sub
            else:
                substrate_ratios = np.array([])

        if len(substrate_ratios) != n_sub:
            raise ValueError(
                f"substrate_ratios length {len(substrate_ratios)} != expected {n_sub}"
            )

        # Gradient params
        if gradient_params is None and self._gradient_mode:
            gradient_params = np.zeros(self._n_gradient_params)

        return self._pack_parameters(
            film_conc, thickness, abruptness, substrate_ratios, gradient_params
        )

    def _gradient_pure_python(
        self,
        x: np.ndarray,
        m0: np.ndarray,
        I_obs_norm: np.ndarray,
        T_matrices_list: list,
        T_matrices_stacked: np.ndarray,
        antisubst_idx: np.ndarray,
        xreg_weights: np.ndarray,
        weights: np.ndarray,
        method_int: int,
        n_angles: int,
        film_size: int,
    ) -> np.ndarray:
        """Pure Python analytical gradient (equivalent to gradient_kernel_numba).

        Used when Numba is not available (e.g. in uv .venv environments).
        This provides the same analytical gradient as the Numba kernel but
        implemented with vectorized NumPy operations.
        """
        n_depth = self.n_depth
        n_elements = self.n_elements
        n_free = len(self.film_idx)
        n_sub = len(self.sub_idx)
        n_linked = len(self.linked_idx)
        free_idx = self.free_idx
        sub_idx = self.sub_idx
        linked_idx = self.linked_idx
        degree = self.config.degree
        alpha = self.config.alpha
        min_conc = self.config.min_conc

        # Unpack parameters
        film_conc, thickness, abruptness, sub_ratios, grad_params = self._unpack_parameters(x)
        D = self._build_full_profile(film_conc, thickness, abruptness, sub_ratios, grad_params)

        # Forward model: I_pred[a,e] = T[e] @ D[:,e]
        I_pred = np.zeros((n_angles, n_elements), dtype=np.float64)
        for e in range(n_elements):
            I_pred[:, e] = T_matrices_list[e] @ D[:, e]

        # Normalize by substrate
        if sub_idx:
            I_sub_sum = I_pred[:, sub_idx].sum(axis=1)  # (n_angles,)
            I_sub_sum = np.maximum(I_sub_sum, 1e-10)
            I_pred_norm = I_pred / I_sub_sum[:, np.newaxis]
        else:
            I_total = I_pred.sum(axis=1)
            I_total = np.maximum(I_total, 1e-10)
            I_pred_norm = I_pred / I_total[:, np.newaxis]

        # Apply Anscombe transform if enabled
        if self.config.use_anscombe and self._anscombe_scale is not None:
            I_pred_scaled = I_pred_norm * self._anscombe_scale
            I_pred_transformed = 2.0 * np.sqrt(I_pred_scaled + 3.0 / 8.0)
            I_pred_norm = (I_pred_transformed - self._anscombe_min) / (
                self._anscombe_max - self._anscombe_min + 1e-10)

        # Residual: r[a,e] = I_obs[a,e] - I_pred_norm[a,e] for antisubst elements
        # dL/dI_pred_norm[a,e] = -degree * r[a,e]  (for e in antisubst_idx)
        # Loss = 0.5 * chi²

        # Gradient w.r.t. I_pred (pre-normalization)
        # I_pred_norm[a,e] = I_pred[a,e] / S[a]  where S = Σ_{sub} I_pred[a,sub]
        dL_dI_pred = np.zeros((n_angles, n_elements), dtype=np.float64)

        # Precompute Xreg weight² per antisubst element (0 if no xreg)
        has_xreg = xreg_weights.size > 0
        w2 = xreg_weights ** 2 if has_xreg else np.ones(len(antisubst_idx))

        for a in range(n_angles):
            S = I_sub_sum[a] if sub_idx else I_pred.sum(axis=1)[a]
            S_sq = S * S

            for i_as in range(len(antisubst_idx)):
                e = antisubst_idx[i_as]
                r = I_obs_norm[a, e] - I_pred_norm[a, e]
                # Chi² = degree * Σ (r * w)² → dChi²/dI_norm = -2 * degree * r * w²
                dL_dI_norm = -degree * r * w2[i_as]

                # Direct contribution: dI_pred_norm[a,e]/dI_pred[a,e] = 1/S
                dL_dI_pred[a, e] += dL_dI_norm / S

            # Indirect contribution from substrate normalization
            for j in range(n_sub):
                sub_k = sub_idx[j]
                grad_sub_k = 0.0
                for i_as in range(len(antisubst_idx)):
                    e = antisubst_idx[i_as]
                    r = I_obs_norm[a, e] - I_pred_norm[a, e]
                    dL_dI_norm = -degree * r * w2[i_as]
                    grad_sub_k += dL_dI_norm * (-I_pred[a, e] / S_sq)
                dL_dI_pred[a, sub_k] += grad_sub_k

        # Gradient w.r.t. D: dL/dD[z,e] = Σ_a dL/dI_pred[a,e] * T[e,a,z]
        dL_dD = np.zeros((n_depth, n_elements), dtype=np.float64)
        for e in range(n_elements):
            # T_matrices_stacked[e] has shape (n_angles, n_depth)
            dL_dD[:, e] = T_matrices_stacked[e].T @ dL_dI_pred[:, e]

        # Compute substrate erf and film_scale
        _2sqrt_ln2 = 2.0 * np.sqrt(np.log(2.0))
        sigma = abruptness / _2sqrt_ln2
        u = (self.depth - thickness) / sigma
        erf_vals = 0.5 * (1.0 + erf(u))  # (n_depth,)

        substrate_sum = erf_vals * sub_ratios.sum()  # (n_depth,)
        film_scale_raw = 1.0 - substrate_sum
        film_scale = np.maximum(film_scale_raw, min_conc)
        film_scale_clipped = film_scale_raw < min_conc  # boolean mask

        # Build gradient vector
        total_size = film_size + 2 + n_sub
        grad = np.zeros(total_size, dtype=np.float64)

        # 1. Gradient w.r.t. free_conc
        for j in range(n_free):
            free_e = free_idx[j]
            coeff = self.free_linked_const[j] - 1.0

            for z in range(n_depth):
                grad_val = dL_dD[z, free_e] * film_scale[z]

                if n_linked > 0:
                    for k in range(n_linked):
                        linked_e = linked_idx[k]
                        grad_val += dL_dD[z, linked_e] * film_scale[z] * coeff / n_linked

                grad[z * n_free + j] = grad_val

        # 2. Gradient w.r.t. thickness
        _inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
        exp_u2 = np.exp(-u * u)
        d_erf_d_t = -1.0 / sigma * _inv_sqrt_pi * exp_u2  # (n_depth,)

        grad_thickness = 0.0
        for z in range(n_depth):
            if not film_scale_clipped[z]:
                d_film_scale = -sub_ratios.sum() * d_erf_d_t[z]
                for jf in range(n_free):
                    free_e = free_idx[jf]
                    grad_thickness += dL_dD[z, free_e] * film_conc[z, jf] * d_film_scale
                    if n_linked > 0:
                        coeff = self.free_linked_const[jf] - 1.0
                        for k in range(n_linked):
                            linked_e = linked_idx[k]
                            grad_thickness += dL_dD[z, linked_e] * film_conc[z, jf] * d_film_scale * coeff / n_linked

            for j in range(n_sub):
                sub_e = sub_idx[j]
                grad_thickness += dL_dD[z, sub_e] * sub_ratios[j] * d_erf_d_t[z]

        grad[film_size] = grad_thickness

        # 3. Gradient w.r.t. abruptness
        d_erf_d_a = (-u / abruptness) * _inv_sqrt_pi * exp_u2  # (n_depth,)

        grad_abruptness = 0.0
        for z in range(n_depth):
            if not film_scale_clipped[z]:
                d_film_scale = -sub_ratios.sum() * d_erf_d_a[z]
                for jf in range(n_free):
                    free_e = free_idx[jf]
                    grad_abruptness += dL_dD[z, free_e] * film_conc[z, jf] * d_film_scale
                    if n_linked > 0:
                        coeff = self.free_linked_const[jf] - 1.0
                        for k in range(n_linked):
                            linked_e = linked_idx[k]
                            grad_abruptness += dL_dD[z, linked_e] * film_conc[z, jf] * d_film_scale * coeff / n_linked

            for j in range(n_sub):
                sub_e = sub_idx[j]
                grad_abruptness += dL_dD[z, sub_e] * sub_ratios[j] * d_erf_d_a[z]

        grad[film_size + 1] = grad_abruptness

        # 4. Gradient w.r.t. substrate_ratios
        for j in range(n_sub):
            sub_e = sub_idx[j]
            grad_ratio = 0.0
            for z in range(n_depth):
                grad_ratio += dL_dD[z, sub_e] * erf_vals[z]
                if not film_scale_clipped[z]:
                    d_film_scale = -erf_vals[z]
                    for jf in range(n_free):
                        free_e = free_idx[jf]
                        grad_ratio += dL_dD[z, free_e] * film_conc[z, jf] * d_film_scale
                        if n_linked > 0:
                            coeff = self.free_linked_const[jf] - 1.0
                            for k in range(n_linked):
                                linked_e = linked_idx[k]
                                grad_ratio += dL_dD[z, linked_e] * film_conc[z, jf] * d_film_scale * coeff / n_linked
            grad[film_size + 2 + j] = grad_ratio

        # 5. Regularization gradients
        if method_int == 1:  # L2
            for i in range(film_size):
                grad[i] += alpha * 2.0 * (x[i] - m0[i])
            grad[film_size] += alpha * 0.02 * (thickness - m0[film_size])
            grad[film_size + 1] += alpha * 2.0 * (abruptness - m0[film_size + 1])
        elif method_int == 2:  # L1 — smoothed subgradient (Huber-like)
            # Using d/max(|d|, delta) instead of sign(d) to avoid non-smooth
            # kinks that confuse SLSQP's BFGS Hessian approximation.
            # FD naturally smooths these kinks; this gives similar behavior.
            _l1_delta = 1e-3
            for i in range(film_size):
                d = x[i] - m0[i]
                grad[i] += alpha * d / max(abs(d), _l1_delta)
            dt = thickness - m0[film_size]
            grad[film_size] += alpha * 0.01 * dt / max(abs(dt), _l1_delta)
        elif method_int == 3:  # MEM — d/dx[Σ(x·log(x/m0) - x + m0)] = log(x/m0)
            for i in range(film_size):
                x_safe = max(x[i], 1e-10)
                m0_safe = max(m0[i], 1e-10)
                grad[i] += alpha * np.log(x_safe / m0_safe)
            t_safe = max(thickness, 1e-10)
            m0_t = max(m0[film_size], 1e-10)
            grad[film_size] += alpha * np.log(t_safe / m0_t) / 10.0
        elif method_int == 4:  # L1 log-cosh: grad = tanh((x-m0)/β)
            _l1_beta = 0.01
            for i in range(film_size):
                d = (x[i] - m0[i]) / _l1_beta
                grad[i] += alpha * np.tanh(d)
            d_t = (thickness - m0[film_size]) / (100.0 * _l1_beta)
            grad[film_size] += alpha * np.tanh(d_t) * 0.01
        elif method_int == 5:  # MEM log-space: d(-S)/dx = x·m0·exp(x)
            for i in range(film_size):
                x_clip = min(max(x[i], -100.0), 100.0)
                m0_safe = max(m0[i], 1e-10)
                exp_x = np.exp(x_clip)
                grad[i] += alpha * x_clip * m0_safe * exp_x
            # thickness: concentration space I-divergence gradient
            t_safe = max(thickness, 1e-10)
            m0_t = max(m0[film_size], 1e-10)
            grad[film_size] += alpha * np.log(t_safe / m0_t) / 10.0

        return grad

    def _gradient_split(
        self,
        x: np.ndarray,
        m0: np.ndarray,
        I_obs_norm: np.ndarray,
        T_matrices_stacked: np.ndarray,
        antisubst_idx: np.ndarray,
        xreg_weights: np.ndarray,
        method_int: int,
        n_angles: int,
        film_size: int,
        cache: dict,
    ) -> np.ndarray:
        """Split gradient: NumPy forward pass (from objective cache) + Numba backward pass.

        Uses intermediates cached by objective() to compute dL/dI_pred and dL/dD
        in NumPy, then calls gradient_backward_numba for the chain rule through
        _build_full_profile parameters. This ensures exact forward pass consistency
        with the objective while keeping Numba speed for the backward pass.
        """
        n_elements = self.n_elements
        n_depth = self.n_depth
        sub_idx = self.sub_idx

        # Get cached forward pass results
        I_pred = cache.get('I_pred')
        I_pred_norm = cache.get('I_pred_norm')
        I_sub_sum = cache.get('I_sub_sum')
        film_conc = cache.get('film_conc')
        thickness = cache.get('thickness')
        abruptness = cache.get('abruptness')
        sub_ratios = cache.get('sub_ratios')

        if I_pred is None:
            # Cache miss (shouldn't happen in normal SLSQP flow where
            # objective is always called before gradient).
            # Fall back to pure Python using T_matrices_stacked as list.
            T_list = [T_matrices_stacked[e] for e in range(n_elements)]
            return self._gradient_pure_python(
                x, m0, I_obs_norm, T_list, T_matrices_stacked,
                antisubst_idx, xreg_weights, np.empty(0),
                method_int, n_angles, film_size,
            )

        # --- Step 1: dL/dI_pred_norm → dL/dI_pred (NumPy vectorized) ---
        # Precompute Xreg weight²
        w2 = xreg_weights ** 2

        dL_dI_pred = np.zeros((n_angles, n_elements), dtype=np.float64)

        for a in range(n_angles):
            S = I_sub_sum[a]
            S_sq = S * S

            for i_as in range(len(antisubst_idx)):
                e = antisubst_idx[i_as]
                r = I_obs_norm[a, e] - I_pred_norm[a, e]
                _eff_deg = self.config.degree * self._degree_scale
                dL_dI_norm = -_eff_deg * r * w2[i_as]
                dL_dI_pred[a, e] += dL_dI_norm / S

            for j in range(len(sub_idx)):
                sub_k = sub_idx[j]
                grad_sub_k = 0.0
                for i_as in range(len(antisubst_idx)):
                    e = antisubst_idx[i_as]
                    r = I_obs_norm[a, e] - I_pred_norm[a, e]
                    dL_dI_norm = -_eff_deg * r * w2[i_as]
                    grad_sub_k += dL_dI_norm * (-I_pred[a, e] / S_sq)
                dL_dI_pred[a, sub_k] += grad_sub_k

        # --- Step 2: dL/dI_pred → dL/dD via T matrices (NumPy matmul) ---
        dL_dD = np.zeros((n_depth, n_elements), dtype=np.float64)
        for e in range(n_elements):
            dL_dD[:, e] = T_matrices_stacked[e].T @ dL_dI_pred[:, e]

        # --- Step 3: dL/dD → dL/dx via Numba backward kernel ---
        return gradient_backward_numba(
            x, m0, dL_dD,
            film_conc, thickness, abruptness,
            np.asarray(sub_ratios) if not isinstance(sub_ratios, np.ndarray) else sub_ratios,
            self.depth, n_elements,
            self._free_idx_array, self._sub_idx_array, self._linked_idx_array,
            self.free_linked_const,
            self.config.min_conc,
            self.config.degree * self._degree_scale, self.config.alpha,
            method_int,
        )

    def solve(
        self,
        I_obs: np.ndarray,
        verbose: bool = False,
        x0_override: Optional[np.ndarray] = None,
        _warm_restart_count: int = 0,
    ) -> 'SemiFreeSolverResult':
        """
        Solve inverse problem with film structure optimization.

        Args:
            I_obs: Observed intensities (n_angles, n_elements)
            verbose: Print progress
            x0_override: Optional initial parameter vector to override default x0.

                If provided, this vector is used directly as the starting point
                instead of constructing x0 from D_prior. The regularization
                reference (m0) is **always** derived from D_prior, so
                x0_override only changes the starting point, not the prior.

                Use ``build_x0()`` to construct a valid x0_override from
                component arrays (film_conc, thickness, abruptness, etc.).
            _warm_restart_count: Internal counter for warm restart recursion
                (do not set manually).

        Returns:
            SemiFreeSolverResult with optimized profile and parameters

        Notes:
            With ``config.scipy_method == 'SLSQP'`` and
            ``config.enable_fallback`` (the default) the solve is not
            guaranteed to be a single SLSQP run. SLSQP results with status 8
            or 4, with an ill-conditioned constraint Jacobian, or reported as
            a success at the unchanged starting point after more than one
            iteration are rerouted to ``config.fallback_method`` restarted
            from x0; on the last route the fallback result is then polished
            by an SLSQP restart with the same ``max_iter`` budget. The
            rerouted result is kept only if it succeeds or lowers the
            objective (``fallback_used`` in the result), and the returned
            ``success`` / ``opt_status`` / ``n_iterations`` are then those of
            the last optimizer that ran: a kept polished result can carry
            ``success=False`` (status 8) at a point that improved on the
            SLSQP one. The no-progress test only sees that SLSQP did not
            move, so it also fires when x0 already is the optimum; the result
            is then unchanged. On the rerouted route the output is not bit-
            reproducible from run to run. A thickness change
            above ``config.warm_restart_threshold`` that also changes the
            active depth set triggers a warm restart
            (``config.max_warm_restarts``), whose result is the one returned.
        """
        start_time = time.time()

        n_film = len(self.film_idx)
        n_sub = len(self.sub_idx)

        # Normalize observations
        _norm_mode = self.semi_free_config.intensity_normalization
        self._degree_scale = 1.0  # scale factor for degree in scalar_substrate mode
        if _norm_mode == 'scalar_substrate' and self.sub_idx:
            # Use substrate signal at reference angle (most normal = index 0)
            # as a single scalar divisor — preserves [0, ∞) and angle info
            I_sub_all = I_obs[:, self.sub_idx].sum(axis=1)  # (n_angles,)
            _sub_ref = np.max(I_sub_all)  # most normal angle has strongest sub
            _sub_ref = max(_sub_ref, 1e-10)
            I_norm = I_obs / _sub_ref
        elif _norm_mode != 'total' and self.sub_idx:
            # Per-angle substrate normalization (MATLAB default)
            I_sub = I_obs[:, self.sub_idx].sum(axis=1, keepdims=True)
            I_norm = I_obs / np.maximum(I_sub, 1e-10)
        else:
            I_norm = I_obs / np.maximum(I_obs.sum(axis=1, keepdims=True), 1e-10)

        # Compute Xreg weighting (MATLAB-compatible)
        # MATLAB: Xreg = mean(Xdata) where Xdata = I_film / I_substrate
        # Chi² = sum((Xdata - Xsimu)² / Xreg²) × degree
        # This gives more weight to trace elements and less to dominant ones
        if self.config.use_xreg_weighting and self.sub_idx:
            # antisubst = film elements (free + linked)
            _antisubst = self.free_idx + self.linked_idx
            # Xreg = mean over angles of substrate-normalized film intensities
            xreg = np.mean(I_norm[:, _antisubst], axis=0)  # (n_film_elements,)
            xreg[xreg == 0] = 1.0  # MATLAB-compatible: zero means "no data" → weight=1
            # Broadcast to (n_angles, n_antisubst) for per-element weighting
            xreg_weights = 1.0 / xreg  # (n_antisubst,)
            # Cap weights to prevent over-weighting near-detection-limit elements
            if np.isfinite(self.config.xreg_max_weight):
                xreg_weights = np.minimum(xreg_weights, self.config.xreg_max_weight)
        else:
            xreg_weights = np.empty(0)

        # Compute WLS weights for Poisson noise
        # For Poisson: variance = mean (in counts), relative error = 1/sqrt(N)
        #
        # After normalization to relative intensities, the variance structure changes.
        # A common approach is to use weight = I (not 1/I) which gives more weight
        # to high-intensity signals (which have better signal-to-noise ratio).
        #
        # Alternative: weight = sqrt(I) as a compromise.
        #
        # Here we use weight = I (normalized), which empirically works better.
        if self.config.use_wls:
            # Weight proportional to intensity (high intensity = better S/N = higher weight)
            raw_weights = np.maximum(I_norm, 1e-10)
            # Normalize so sum(weights) = n_data_points
            weights = raw_weights * (raw_weights.size / raw_weights.sum())
        else:
            weights = np.empty(0)  # Empty array signals uniform weighting

        # Apply Anscombe variance-stabilizing transform for Poisson noise
        # Y = 2 * sqrt(I + 3/8) transforms Poisson to ~Gaussian with variance ≈ 1
        #
        # For normalized intensities (0-1 range), we need to scale to equivalent counts
        # before applying Anscombe, then rescale back to maintain optimization balance.
        # Typical XPS: ~2500 counts for 2% relative noise (σ = 1/sqrt(N))
        if self.config.use_anscombe:
            # Scale factor: equivalent total counts per angle
            # For 2% relative noise: N = 1/(0.02)^2 = 2500
            anscombe_scale = 2500.0
            I_scaled = I_norm * anscombe_scale
            I_transformed = anscombe_transform(I_scaled)
            # Rescale to [0, 1] range to maintain optimization balance
            # Y_min = 2*sqrt(0 + 3/8) ≈ 1.22, Y_max = 2*sqrt(2500 + 3/8) ≈ 100
            # For our data range, min/max will vary
            self._anscombe_min = I_transformed.min()
            self._anscombe_max = I_transformed.max()
            I_norm = (I_transformed - self._anscombe_min) / (self._anscombe_max - self._anscombe_min + 1e-10)
            self._anscombe_scale = anscombe_scale
        else:
            self._anscombe_scale = None
            self._anscombe_min = None
            self._anscombe_max = None

        # Initial values
        thickness_init = self.semi_free_config.thickness_init
        abruptness_init = self.semi_free_config.abruptness_init

        # Initial substrate ratios (equal distribution or from prior)
        if n_sub > 0:
            sub_ratios_init = np.ones(n_sub, dtype=np.float64) / n_sub
        else:
            sub_ratios_init = np.array([], dtype=np.float64)

        # Initial film concentrations from prior
        # First compute initial substrate profile
        D_sub_init = self._build_substrate_profile(thickness_init, abruptness_init, sub_ratios_init)
        substrate_sum_init = D_sub_init.sum(axis=1, keepdims=True)
        film_fraction_init = np.maximum(1.0 - substrate_sum_init, self.config.min_conc)

        if self._stoichiometric_mode == 'stoichiometric':
            # Stoichiometric mode: free_conc is raw concentration (not normalized)
            # Initial values come directly from D_prior
            film_conc_init = self.D_prior[:, self.film_idx].copy()
            # Clip to reasonable range
            film_conc_init = np.clip(film_conc_init, self.config.min_conc, self.config.max_conc)
        else:
            # Charge balance mode: normalized concentration
            # The optimization variable free_conc is the NORMALIZED concentration
            # (range [0, 1/freelinkedconst]), which gets multiplied by (1-substrate)
            # to get the final concentration.
            #
            # D_prior contains final concentrations, so we need to convert:
            #   free_conc = D_prior[:, free] / (1 - substrate_sum)
            film_conc_init = self.D_prior[:, self.film_idx].copy() / film_fraction_init
            for i, flc in enumerate(self.free_linked_const):
                if flc > 0:
                    max_val = 1.0 / flc
                    film_conc_init[:, i] = np.clip(film_conc_init[:, i], self.config.min_conc, max_val)

            # Ensure initial film_conc satisfies sum=1 constraint at
            # every active depth.  SLSQP requires a feasible starting
            # point; without this normalization, D_prior≈0 gives sum≈0
            # and FreeSolver-derived priors give arbitrary sums, both
            # causing SLSQP to fail immediately (status 4 or 8).
            flc = self.free_linked_const
            row_sums = (film_conc_init * flc).sum(axis=1)
            for z in range(self.n_depth):
                if row_sums[z] < 1e-6:
                    # No prior info: uniform
                    film_conc_init[z, :] = 1.0 / np.sum(flc)
                else:
                    # Scale to satisfy Σ flc[j]*conc[j] = 1
                    film_conc_init[z, :] *= 1.0 / row_sums[z]

        # Initial gradient parameters (zero slopes)
        if self._gradient_mode:
            gradient_init = np.array([
                self._gradient_init.get(idx, 0.0)
                for idx in self._gradient_element_idx
            ])
            if self._gradient_type == 'quadratic':
                # For quadratic: [c1_0, c2_0, c1_1, c2_1, ...]
                gradient_init = np.zeros(self._n_gradient_params)
        else:
            gradient_init = None

        # Pack initial parameters
        if x0_override is not None:
            x0 = x0_override
        else:
            x0 = self._pack_parameters(
                film_conc_init, thickness_init, abruptness_init, sub_ratios_init,
                gradient_init
            )

        # If abruptness optimization is disabled, fix abruptness to initial value
        # Use local variable to avoid mutating self.semi_free_config
        if not self.semi_free_config.optimize_abruptness:
            _abruptness_bounds = (abruptness_init, abruptness_init)
        else:
            _abruptness_bounds = self.semi_free_config.abruptness_bounds

        # If thickness optimization is disabled, fix thickness to initial value
        if not self.semi_free_config.optimize_thickness:
            _thickness_bounds = (thickness_init, thickness_init)
        else:
            _thickness_bounds = self.semi_free_config.thickness_bounds

        # If substrate ratio optimization is disabled, fix to initial values
        if not self.semi_free_config.optimize_substrate_ratio:
            _sub_ratio_lb = sub_ratios_init.copy()
            _sub_ratio_ub = sub_ratios_init.copy()
        else:
            _sub_ratio_lb = np.full(n_sub, self.config.min_conc)
            _sub_ratio_ub = np.full(n_sub, self.config.max_conc)

        # Bounds
        film_size = self.n_depth * n_film

        if self._stoichiometric_mode == 'stoichiometric':
            # Stoichiometric mode: simple bounds [min_conc, max_conc]
            # The stoichiometric constraint will enforce mass balance
            free_ub = np.full(film_size, self.config.max_conc)
        else:
            # Charge balance mode: upper bound from freelinkedconst
            # MATLAB line 58: ub0 = 1./freelinkedconst
            # Upper bound for free element concentration to ensure sum = 1
            # When D_free = 1/flc, D_linked = D_free*(flc-1) = (flc-1)/flc
            # So D_free + D_linked = 1/flc + (flc-1)/flc = 1
            # Note: x[:film_size] is stored depth-major (C order):
            #   x[d*n_film + e] = concentration of element e at depth d
            free_ub_2d = np.zeros((self.n_depth, n_film))
            for i, flc in enumerate(self.free_linked_const):
                if flc > 0:
                    free_ub_2d[:, i] = 1.0 / flc
                else:
                    free_ub_2d[:, i] = self.config.max_conc
            free_ub = free_ub_2d.flatten()  # depth-major (C order)

        # Base bounds (film_conc, thickness, abruptness, substrate_ratios)
        lb_parts = [
            np.full(film_size, self.config.min_conc),
            [_thickness_bounds[0], _abruptness_bounds[0]],
            _sub_ratio_lb,
        ]
        ub_parts = [
            free_ub,  # Upper bound (mode-dependent)
            [_thickness_bounds[1], _abruptness_bounds[1]],
            _sub_ratio_ub,
        ]

        # Add gradient bounds
        if self._gradient_mode and self._n_gradient_params > 0:
            default_bounds = (-0.02, 0.02)
            gradient_lb = []
            gradient_ub = []
            for idx in self._gradient_element_idx:
                bounds_pair = self._gradient_bounds.get(idx, default_bounds)
                if self._gradient_type == 'linear':
                    gradient_lb.append(bounds_pair[0])
                    gradient_ub.append(bounds_pair[1])
                elif self._gradient_type == 'quadratic':
                    # c1 and c2 bounds (use same bounds for both coefficients)
                    gradient_lb.extend([bounds_pair[0], bounds_pair[0] / 10])  # c2 scaled down
                    gradient_ub.extend([bounds_pair[1], bounds_pair[1] / 10])
            lb_parts.append(np.array(gradient_lb))
            ub_parts.append(np.array(gradient_ub))

        lb = np.concatenate(lb_parts)
        ub = np.concatenate(ub_parts)

        # ================================================================
        # Log-space MEM transformation (MATLAB-style).
        #
        # Transform film_conc part of x0 and bounds to log space:
        #   x_log = log(film_conc / m0_film)
        #   D = m0_film * exp(x_log)
        #
        # Only film_conc is transformed; thickness, abruptness, sub_ratios
        # stay in their original space (matches MATLAB behavior).
        #
        # Key: objective/gradient/constraints must convert x back to
        # concentration space before physics computations.
        # ================================================================
        _use_log_space = (self.config.method == 'MEM'
                          and self.config.mem_space == 'log')

        # ================================================================
        # Identify degenerate constraint depths and pin film_conc there.
        #
        # At depths where erf_sub ≈ 1 (deep into substrate), the sum
        # constraint rhs = 1 - erf_sub - const_fix ≈ 0, forcing film_conc
        # to ~min_conc.  These constraint rows have near-zero Jacobian
        # entries, making the constraint matrix rank-deficient and causing
        # SLSQP to fail ("Positive directional derivative" / status 8).
        #
        # Fix: evaluate erf_sub at x0, mark depths with rhs < threshold
        # as "degenerate", and exclude those rows from the equality
        # constraint.  A warm restart is triggered if the optimized
        # thickness differs significantly from x0 (see end of solve()).
        # ================================================================
        _DEGEN_THRESHOLD = self.config.degen_depth_threshold

        # Pre-compute const element contribution (DoptFix in MATLAB)
        # Need this early for degenerate depth detection.
        # MATLAB line 62-66: if ~isempty(const), DoptFix = sum(Dini(depind,const).*constlinkedconst)
        _const_idx_early = self.semi_free_config.const_indices or []
        const_fix = np.zeros(self.n_depth)
        if _const_idx_early and self.elements is not None:
            charges_early = np.array([
                getattr(e, 'oxidation_state', None) or 0 for e in self.elements
            ])
            if self.linked_idx:
                mean_lc = np.mean(charges_early[self.linked_idx])
                if mean_lc != 0:
                    clc = np.array([
                        max(1.0, 1.0 - charges_early[ci] / mean_lc)
                        for ci in _const_idx_early
                    ])
                else:
                    clc = np.ones(len(_const_idx_early))
            else:
                clc = np.ones(len(_const_idx_early))
            for i, ci in enumerate(_const_idx_early):
                const_fix += self.D_prior[:, ci] * clc[i]

        # Compute erf_sub at initial parameters
        _, thickness_init_val, abruptness_init_val, sub_ratios_init_val, _ = \
            self._unpack_parameters(x0)
        D_sub_init = self._build_substrate_profile(
            thickness_init_val, abruptness_init_val, sub_ratios_init_val
        )
        erf_sub_init = D_sub_init.sum(axis=1)  # (n_depth,)
        rhs_init = 1.0 - erf_sub_init - const_fix  # available "room" for film

        # Active depths: rhs > threshold (constraint is meaningful)
        active_depth_mask = rhs_init >= _DEGEN_THRESHOLD
        active_depths = np.where(active_depth_mask)[0]
        degen_depths = np.where(~active_depth_mask)[0]

        if verbose and len(degen_depths) > 0:
            print(f"[SemiFreeSolver] Degenerate depths ({len(degen_depths)}/{self.n_depth}): "
                  f"z={degen_depths.tolist()}, rhs<{_DEGEN_THRESHOLD}")
            print(f"[SemiFreeSolver] Active depths ({len(active_depths)}/{self.n_depth}): "
                  f"z={active_depths.tolist()}")

        bounds = Bounds(lb=lb, ub=ub)

        # Get transfer matrices
        T_matrices_list = [self.tm.get_matrix(e) for e in range(self.n_elements)]

        # Stack transfer matrices for Numba kernel: (n_elements, n_angles, n_depth)
        n_angles = len(self.tm.angles)
        T_matrices_stacked = np.zeros((self.n_elements, n_angles, self.n_depth))
        for e in range(self.n_elements):
            T_matrices_stacked[e] = T_matrices_list[e]

        # Pre-compute antisubst indices for Numba
        antisubst_idx = np.array(self.free_idx + self.linked_idx, dtype=np.int64)

        # Prior for regularization
        thickness_prior = self.semi_free_config.thickness_init
        if np.all(self.D_prior <= 1e-12):
            thickness_prior = 0.0

        m0 = self._pack_parameters(
            self.D_prior[:, self.film_idx],
            thickness_prior,
            self.semi_free_config.abruptness_init,
            sub_ratios_init,
            gradient_init,  # Include gradient params in prior
        )

        # Explicit thickness prior P(t) = (t - t0)²/(2σ²), added outside
        # alpha·reg (see SemiFreeSolverConfig.t_prior_center).
        _t_prior_t0 = self.semi_free_config.t_prior_center
        _t_prior_sig2 = None
        if _t_prior_t0 is not None:
            if not np.isfinite(_t_prior_t0):
                raise ValueError(
                    f"t_prior_center must be finite, got {_t_prior_t0}")
            if self.semi_free_config.t_prior_sigma is None:
                raise ValueError(
                    "t_prior_sigma must be given when t_prior_center is set")
            _t_prior_sigma = float(self.semi_free_config.t_prior_sigma)
            if not (np.isfinite(_t_prior_sigma) and _t_prior_sigma > 0):
                raise ValueError(
                    f"t_prior_sigma must be finite and > 0, "
                    f"got {_t_prior_sigma}")
            _t_prior_sig2 = _t_prior_sigma ** 2

        # Log-space MEM: transform x0 and bounds for film_conc part
        if _use_log_space:
            m0_film = np.maximum(m0[:film_size], self.config.min_conc)
            # x0: film_conc → log(film_conc / m0)
            x0[:film_size] = np.log(
                np.maximum(x0[:film_size], self.config.min_conc) / m0_film
            )
            # bounds: lb/ub → log(lb/m0), log(ub/m0)
            lb[:film_size] = np.log(
                np.maximum(lb[:film_size], self.config.min_conc) / m0_film
            )
            ub[:film_size] = np.log(
                np.maximum(ub[:film_size], self.config.min_conc) / m0_film
            )
            bounds = Bounds(lb=lb, ub=ub)

        # Log-space helper: convert x (possibly log-space) to concentration space
        if _use_log_space:
            _m0_film_log = m0[:film_size].copy()  # cache for closure

            def _x_to_conc(x):
                """Convert x from log-space to concentration-space."""
                x_conc = x.copy()
                x_conc[:film_size] = _m0_film_log * np.exp(
                    np.clip(x[:film_size], -100, 100)
                )
                return x_conc
        else:
            def _x_to_conc(x):
                """No-op: already in concentration space."""
                return x

        # Forward pass cache for split gradient architecture.
        # Objective caches intermediates; gradient reuses them to avoid
        # the forward pass mismatch that exists in gradient_kernel_numba.
        _cache = {}

        # Unit weights for OLS split gradient (same interface as xreg path)
        _unit_xreg_weights = np.ones(len(antisubst_idx), dtype=np.float64)

        # Objective function
        def objective(x):
            # Log-space: convert film_conc from log to concentration before physics
            x_conc = _x_to_conc(x)
            film_conc, thickness, abruptness, sub_ratios, grad_params = self._unpack_parameters(x_conc)
            D = self._build_full_profile(film_conc, thickness, abruptness, sub_ratios, grad_params)

            # Forward model (Numba-accelerated if available)
            if self.use_numba:
                I_pred = compute_forward_model_numba(D, T_matrices_stacked, n_angles, self.n_elements)
            else:
                I_pred = np.zeros_like(I_norm)
                for e in range(self.n_elements):
                    I_pred[:, e] = T_matrices_list[e] @ D[:, e]

            # Normalize predicted intensities (must match I_obs normalization)
            if _norm_mode == 'scalar_substrate' and self.sub_idx:
                I_pred_sub_all = I_pred[:, self.sub_idx].sum(axis=1)
                _pred_sub_ref = np.max(I_pred_sub_all)
                _pred_sub_ref = max(_pred_sub_ref, 1e-10)
                I_pred_norm = I_pred / _pred_sub_ref
            elif _norm_mode != 'total' and self.sub_idx:
                if self.use_numba:
                    I_pred_norm = normalize_by_substrate_numba(I_pred, self._sub_idx_array)
                else:
                    I_pred_sub = I_pred[:, self.sub_idx].sum(axis=1, keepdims=True)
                    I_pred_norm = I_pred / np.maximum(I_pred_sub, 1e-10)
            else:
                I_pred_norm = I_pred / np.maximum(I_pred.sum(axis=1, keepdims=True), 1e-10)

            # Apply Anscombe transform to predicted intensities (if enabled)
            # Use same scale factor and normalization as was applied to observed intensities
            if self.config.use_anscombe and self._anscombe_scale is not None:
                I_pred_scaled = I_pred_norm * self._anscombe_scale
                I_pred_transformed = anscombe_transform(I_pred_scaled)
                # Rescale using same min/max as observed data
                I_pred_norm = (I_pred_transformed - self._anscombe_min) / (self._anscombe_max - self._anscombe_min + 1e-10)

            # Data fit term (Chi² or Poisson deviance, Numba-accelerated if available)
            # Scale degree for scalar_substrate mode so Chi² magnitude matches
            # per-angle substrate normalization
            _eff_degree = self.config.degree * self._degree_scale

            if self.config.use_xreg_weighting and xreg_weights.size > 0:
                # MATLAB-compatible: Chi² = sum((Xdata - Xsimu)² / Xreg²) × degree
                # xreg_weights = 1/Xreg, so (diff/Xreg)² = (diff × xreg_weights)²
                diff = I_norm[:, antisubst_idx] - I_pred_norm[:, antisubst_idx]
                chi2 = _eff_degree * np.sum((diff * xreg_weights[np.newaxis, :]) ** 2)
            elif self.config.use_poisson_nll:
                # Poisson deviance: D = 2 * Σ[I_obs * log(I_obs/I_pred) - (I_obs - I_pred)]
                # This is properly scaled to be comparable with Chi²
                if self.use_numba:
                    chi2 = compute_poisson_nll_numba(I_norm, I_pred_norm, antisubst_idx, _eff_degree)
                else:
                    I_obs = I_norm[:, antisubst_idx]
                    I_obs_safe = np.maximum(I_obs, 1e-10)
                    I_pred_safe = np.maximum(I_pred_norm[:, antisubst_idx], 1e-10)
                    chi2 = _eff_degree * 2.0 * np.sum(
                        I_obs_safe * np.log(I_obs_safe / I_pred_safe) - (I_obs - I_pred_safe)
                    )
            elif self.config.use_wls:
                # Weighted least squares
                if self.use_numba:
                    chi2 = compute_chi2_wls_numba(I_norm, I_pred_norm, antisubst_idx, _eff_degree, weights)
                else:
                    diff = I_norm[:, antisubst_idx] - I_pred_norm[:, antisubst_idx]
                    chi2 = _eff_degree * np.sum(weights[:, antisubst_idx] * diff ** 2)
            else:
                # Ordinary least squares
                if self.use_numba:
                    chi2 = compute_chi2_numba(I_norm, I_pred_norm, antisubst_idx, _eff_degree)
                else:
                    diff = I_norm[:, antisubst_idx] - I_pred_norm[:, antisubst_idx]
                    chi2 = _eff_degree * np.sum(diff ** 2)

            # Regularization (Numba-accelerated if available)
            if self.config.method == 'L2':
                if self.use_numba:
                    reg = compute_regularization_l2_numba(
                        x, m0, film_size, thickness, abruptness, m0[film_size], m0[film_size + 1]
                    )
                else:
                    reg = np.sum((x[:film_size] - m0[:film_size]) ** 2)
                    reg += 100 * ((thickness - m0[film_size]) / 100) ** 2
                    reg += 100 * ((abruptness - m0[film_size + 1]) / 10) ** 2
            elif self.config.method == 'MEM':
                if _use_log_space:
                    # MATLAB-style log-space entropy:
                    # S = Σ m0·(eˣ-1-x·eˣ) ≤ 0 where x = log(D/m0)
                    # We use reg = -S ≥ 0 so that f = 0.5*chi2 + alpha*reg
                    # is minimized by pushing x toward 0 (D toward m0).
                    # x[:film_size] is still in LOG SPACE (not converted by _x_to_conc)
                    x_log = x[:film_size]  # original log-space values
                    m0_safe = np.maximum(m0[:film_size], 1e-10)
                    exp_x = np.exp(np.clip(x_log, -100, 100))
                    reg = -np.sum(m0_safe * (exp_x - 1.0 - x_log * exp_x))
                    # Thickness: I-divergence in concentration space (MATLAB style)
                    t_safe = max(thickness, 1e-10)
                    m0_t_safe = max(m0[film_size], 1e-10)
                    reg += (t_safe * np.log(t_safe / m0_t_safe) - t_safe + m0_t_safe) / 10
                elif self.use_numba:
                    reg = compute_regularization_mem_numba(x_conc, m0, film_size, thickness, m0[film_size])
                else:
                    # Bregman divergence from φ(x)=x·log(x):
                    # D_φ(x‖m0) = Σ(x·log(x/m0) - x + m0) ≥ 0 always
                    x_safe = np.maximum(x_conc[:film_size], 1e-10)
                    m0_safe = np.maximum(m0[:film_size], 1e-10)
                    reg = np.sum(x_safe * np.log(x_safe / m0_safe) - x_safe + m0_safe)
                    t_safe = max(thickness, 1e-10)
                    m0_t_safe = max(m0[film_size], 1e-10)
                    reg += (t_safe * np.log(t_safe / m0_t_safe) - t_safe + m0_t_safe) / 10
            elif self.config.method == 'L1':
                if self.config.l1_smoothing == 'log_cosh':
                    if self.use_numba:
                        reg = compute_regularization_l1_logcosh_numba(
                            x, m0, film_size, thickness, m0[film_size],
                            self.config.l1_beta)
                    else:
                        beta = self.config.l1_beta
                        d = (x[:film_size] - m0[:film_size]) / beta
                        reg = beta * np.sum(np.log(np.cosh(np.clip(d, -20, 20))))
                        d_t = (thickness - m0[film_size]) / (100 * beta)
                        reg += beta * np.log(np.cosh(np.clip(d_t, -20, 20)))
                elif self.use_numba:
                    reg = compute_regularization_l1_numba(x, m0, film_size, thickness, m0[film_size])
                else:
                    reg = np.sum(np.abs(x[:film_size] - m0[:film_size]))
                    reg += np.abs(thickness - m0[film_size]) / 100
            else:  # LLS
                reg = 0.0

            # Cache forward pass intermediates for split gradient
            _cache['D'] = D
            _cache['I_pred'] = I_pred
            _cache['I_pred_norm'] = I_pred_norm
            _cache['film_conc'] = film_conc
            _cache['thickness'] = thickness
            _cache['abruptness'] = abruptness
            _cache['sub_ratios'] = sub_ratios
            if self.sub_idx:
                _cache['I_sub_sum'] = I_pred[:, self.sub_idx].sum(axis=1)
                _cache['I_sub_sum'] = np.maximum(_cache['I_sub_sum'], 1e-10)

            f_total = 0.5 * chi2 + self.config.alpha * reg
            if _t_prior_t0 is not None:
                f_total += (thickness - _t_prior_t0) ** 2 / (2.0 * _t_prior_sig2)
            return f_total

        # Analytical gradient function (Numba-accelerated)
        # Map method name to integer for Numba
        # 0=LLS, 1=L2, 2=L1, 3=MEM(conc), 4=L1-logcosh, 5=MEM(log-space)
        method_map = {'LLS': 0, 'L2': 1, 'L1': 2, 'MEM': 3}
        method_int = method_map.get(self.config.method, 0)
        # L1+log_cosh uses method=4 in Numba kernels
        if self.config.method == 'L1' and self.config.l1_smoothing == 'log_cosh':
            method_int = 4
        # MEM log-space uses method=5
        if _use_log_space:
            method_int = 5

        def gradient(x):
            if self.config.use_xreg_weighting and self.use_numba:
                # Split gradient: forward pass reused from objective cache,
                # backward pass in Numba. This avoids the forward pass mismatch
                # in gradient_kernel_numba while keeping Numba speed.
                return self._gradient_split(
                    x, m0, I_norm, T_matrices_stacked,
                    antisubst_idx, xreg_weights,
                    method_int, n_angles, film_size, _cache,
                )
            if self.use_numba:
                use_sum_penalty = False
                if self.config.use_poisson_nll:
                    _eff_deg = self.config.degree * self._degree_scale
                    return gradient_kernel_poisson_numba(
                        x, m0, I_norm, T_matrices_stacked,
                        self.depth, self.n_elements, n_angles,
                        self._free_idx_array, self._sub_idx_array, self._linked_idx_array,
                        self.free_linked_const, antisubst_idx,
                        self.config.min_conc, self.config.max_conc,
                        _eff_deg, self.config.alpha,
                        method_int, use_sum_penalty,
                    )
                elif self.config.use_wls:
                    _eff_deg = self.config.degree * self._degree_scale
                    return gradient_kernel_wls_numba(
                        x, m0, I_norm, T_matrices_stacked,
                        self.depth, self.n_elements, n_angles,
                        self._free_idx_array, self._sub_idx_array, self._linked_idx_array,
                        self.free_linked_const, antisubst_idx,
                        self.config.min_conc, self.config.max_conc,
                        _eff_deg, self.config.alpha,
                        method_int, use_sum_penalty,
                        weights,
                    )
                else:
                    # OLS: prefer split gradient (reuses objective's cached
                    # forward pass, avoiding the duplicate T@D computation
                    # in gradient_kernel_numba).  Falls back to monolithic
                    # kernel on the first call before objective populates cache.
                    if _cache.get('I_pred') is not None:
                        return self._gradient_split(
                            x, m0, I_norm, T_matrices_stacked,
                            antisubst_idx, _unit_xreg_weights,
                            method_int, n_angles, film_size, _cache,
                        )
                    _eff_deg = self.config.degree * self._degree_scale
                    return gradient_kernel_numba(
                        x, m0, I_norm, T_matrices_stacked,
                        self.depth, self.n_elements, n_angles,
                        self._free_idx_array, self._sub_idx_array, self._linked_idx_array,
                        self.free_linked_const, antisubst_idx,
                        self.config.min_conc, self.config.max_conc,
                        _eff_deg, self.config.alpha,
                        method_int, use_sum_penalty,
                        np.empty(0, dtype=np.float64),
                    )
            else:
                # Pure Python analytical gradient (no Numba)
                return self._gradient_pure_python(
                    x, m0, I_norm, T_matrices_list, T_matrices_stacked,
                    antisubst_idx, xreg_weights, weights,
                    method_int, n_angles, film_size,
                )

        if _t_prior_t0 is not None:
            # Wrap so the explicit thickness prior reaches every gradient
            # path (numba kernels, split, pure python) without touching them.
            # x[film_size] is thickness in every mode (log-space MEM only
            # transforms the film-concentration block).
            _gradient_base = gradient

            def gradient(x):
                grad = _gradient_base(x)
                grad[film_size] += (x[film_size] - _t_prior_t0) / _t_prior_sig2
                return grad

        # ================================================================
        # Build equality constraints for substrate and charge balance.
        #
        # MATLAB-compatible constraint formulation.
        # Ref: Run_DepthProfiling.m Const() function (line 481-506)
        #
        # Key difference from previous implementation:
        # MATLAB uses LINEAR constraints on x_free directly:
        #   Σ_j freelinkedconst(j) * x_free(z,j) + erf_sub(z) + const_fix(z) = 1
        # This is linear in x_free; only erf_sub depends on thickness nonlinearly.
        #
        # The previous Python code used _build_full_profile(x).sum() which is
        # NONLINEAR (free_conc * film_scale), causing singular constraint Jacobian
        # in SLSQP and leading to immediate convergence failure.
        # ================================================================
        constraints = []

        # const_fix was already computed above (for degenerate depth detection)
        _const_idx = self.semi_free_config.const_indices or []
        _const_fix = const_fix.copy()
        _flc = self.free_linked_const.copy()  # (n_free,)

        # Shared helpers for the constraint closures below. They are defined
        # unconditionally because the charge, substrate and stoichiometric
        # constraints use them too, not only the sum constraint.

        # --- Analytical Jacobian pieces for erf_sub (sum / stoichiometric) ---
        # constraint(z) = Σ_j w(j)*film_conc(z,j) - 1 + erf_sub(z) + cf(z)
        #
        # erf_sub(z) = Σ_s ratio_s * 0.5*(1+erf(u(z)))
        #   where u(z) = (depth(z) - thickness) / sigma
        #         sigma = abruptness / (2*sqrt(ln2))
        #
        # ∂erf_sub/∂thickness  = Σ_s ratio_s * (-1/sigma) * (1/√π) * exp(-u²)
        # ∂erf_sub/∂abruptness = Σ_s ratio_s * (-u/abruptness) * (1/√π) * exp(-u²)
        # ∂erf_sub/∂ratio_s    = 0.5*(1+erf(u))
        _depth_arr = self.depth.copy()
        _n_depth_c = self.n_depth
        _n_film_c = n_film
        _n_sub_c = n_sub
        _film_size_c = film_size
        _inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
        _2sqrt_ln2 = 2.0 * np.sqrt(np.log(2.0))
        _n_grad = self._n_gradient_params if self._gradient_mode else 0

        def _erf_derivs(thickness, abruptness, sub_ratios):
            """Compute erf_sub and its derivatives w.r.t. thickness, abruptness, sub_ratios."""
            sigma = abruptness / _2sqrt_ln2
            u = (_depth_arr - thickness) / sigma
            erf_u = erf(u)
            exp_u2 = np.exp(-u * u)
            erf_profile = 0.5 * (1.0 + erf_u)          # (n_depth,)
            erf_sub = erf_profile * sub_ratios.sum()     # (n_depth,)

            # ∂erf_sub/∂thickness = Σ_s ratio_s * (-1/sigma) * (1/√π) * exp(-u²)
            d_thickness = sub_ratios.sum() * (-1.0 / sigma) * _inv_sqrt_pi * exp_u2

            # ∂erf_sub/∂abruptness = Σ_s ratio_s * (-u/(sigma * 2√ln2)) * (1/√π) * exp(-u²)
            #   = Σ_s ratio_s * (-u/abruptness) * (1/√π) * exp(-u²)
            d_abruptness = sub_ratios.sum() * (-u / abruptness) * _inv_sqrt_pi * exp_u2

            # ∂erf_sub/∂ratio_s = erf_profile  for each substrate element
            d_sub_ratios = erf_profile  # same for all sub elements

            return erf_sub, d_thickness, d_abruptness, d_sub_ratios

        # Active depths for constraint row filtering
        _active = active_depths.copy()
        _n_active = len(_active)

        # --- Constraint 1: Sum constraint (MATLAB-compatible linear form) ---
        # MATLAB Const() line 502 (linked case):
        #   CnstMat(:,:,1)*x_free - 1 + DoptSub + DoptFix = 0
        # where CnstMat(:,:,1) = block-diag with freelinkedconst weights
        #
        # Meaning: Σ_j freelinkedconst(j) * x_free(z,j) = 1 - erf_sub(z) - const_fix(z)
        #
        # When linked is empty (MATLAB line 504):
        #   CnstMat(:,:,2)*x_free - 1 + DoptSub + DoptFix = 0
        # where CnstMat(:,:,2) uses ones (unit weights)
        #
        # Note: erf_sub depends on thickness/abruptness, so this constraint is
        # still nonlinear overall, but it is MUCH better conditioned because
        # x_free enters linearly (not multiplied by film_scale as before).
        if (self.semi_free_config.use_sum_constraint
                and self.semi_free_config.sum_constraint_mode != 'equality'):
            # Only 'equality' is implemented (the penalty form was removed).
            # Fail here rather than silently solving without the constraint.
            raise ValueError(
                "SemiFreeSolverConfig.sum_constraint_mode must be 'equality', "
                f"got {self.semi_free_config.sum_constraint_mode!r}")
        if self.semi_free_config.use_sum_constraint:
            if self.linked_idx:
                # Linked exists: use freelinkedconst weights
                #
                # Constraint: Σ_j flc[j] * film_conc[z,j] + const_fix[z] = 1
                #
                # This ensures D_opt sums to 1 at each depth because
                # _build_full_profile computes:
                #   D_free[j] = film_conc[j] * (1 - erf_sub)
                #   D_linked  = Σ D_free[j] * (flc[j] - 1)
                #   Total = (1 - erf_sub) * Σ flc[j]*film_conc[j] + erf_sub
                #         = (1 - erf_sub) * 1 + erf_sub = 1
                #
                # Note: The original MATLAB code (and earlier Python) had
                # erf_sub in the constraint (Σ flc*x + erf_sub = 1), which
                # double-counted the substrate contribution and caused
                # D_opt row sums of (1-erf_sub)² + erf_sub ≠ 1 at the
                # film/substrate interface.
                _legacy_constraint = self.semi_free_config.use_legacy_sum_constraint
                _nonlinear_constraint = self.semi_free_config.use_nonlinear_sum_constraint

                if _nonlinear_constraint:
                    # Original Python constraint:
                    # D_opt.sum(axis=1) = 1 at each active depth.
                    # Nonlinear in x because D = f(film_conc, thickness).
                    # Strong thickness coupling: all film depths affected.
                    # No analytical Jacobian — SLSQP uses finite differences.
                    def sum_constraint_nonlinear(x, az=_active):
                        film_conc, thickness, abruptness, sub_ratios, gp = \
                            self._unpack_parameters(_x_to_conc(x))
                        D = self._build_full_profile(
                            film_conc, thickness, abruptness, sub_ratios, gp)
                        return (D.sum(axis=1) - 1.0)[az]

                    constraints.append({
                        'type': 'eq',
                        'fun': sum_constraint_nonlinear,
                    })
                else:
                    def sum_constraint_matlab(x, flc=_flc, cf=_const_fix, az=_active,
                                              legacy=_legacy_constraint):
                        film_conc, thickness, abruptness, sub_ratios, _ = \
                            self._unpack_parameters(_x_to_conc(x))
                        weighted_sum = (film_conc * flc[np.newaxis, :]).sum(axis=1)
                        if legacy:
                            # Old: Σ flc*film_conc + erf_sub + const_fix = 1
                            # Provides implicit thickness regularization
                            erf_sub, _, _, _ = _erf_derivs(thickness, abruptness, sub_ratios)
                            full = weighted_sum + erf_sub - 1.0 + cf
                        else:
                            # New: Σ flc*film_conc + const_fix = 1
                            full = weighted_sum - 1.0 + cf
                        return full[az]

                    def sum_constraint_jac(x, flc=_flc, az=_active, na=_n_active,
                                           legacy=_legacy_constraint):
                        film_conc, thickness, abruptness, sub_ratios, _ = \
                            self._unpack_parameters(_x_to_conc(x))

                        n_x = len(x)
                        jac = np.zeros((na, n_x))

                        # ∂constraint/∂film_conc: block-diagonal with flc weights
                        for row_i, z in enumerate(az):
                            for j in range(_n_film_c):
                                jac[row_i, z * _n_film_c + j] = flc[j]

                        # Log-space chain rule: ∂conc/∂x_log = conc = m0*exp(x_log)
                        if _use_log_space:
                            x_conc = _x_to_conc(x)
                            for row_i, z in enumerate(az):
                                for j in range(_n_film_c):
                                    jac[row_i, z * _n_film_c + j] *= x_conc[z * _n_film_c + j]

                        if legacy:
                            # Legacy: erf_sub depends on thickness → add derivatives
                            _, d_thk, d_abrp, _ = _erf_derivs(thickness, abruptness, sub_ratios)
                            thk_idx = _n_film_c * self.n_depth
                            for row_i, z in enumerate(az):
                                jac[row_i, thk_idx] = d_thk[z]
                            if self.semi_free_config.optimize_abruptness:
                                abrp_idx = thk_idx + 1
                                for row_i, z in enumerate(az):
                                    jac[row_i, abrp_idx] = d_abrp[z]

                        return jac

                    constraints.append({
                        'type': 'eq',
                        'fun': sum_constraint_matlab,
                        'jac': sum_constraint_jac,
                    })
            else:
                # No linked: use unit weights
                # Constraint: Σ_j film_conc[z,j] + const_fix[z] = 1
                # (same fix as linked branch — no erf_sub in constraint)
                def sum_constraint_nolinked(x, cf=_const_fix, az=_active):
                    film_conc, thickness, abruptness, sub_ratios, _ = \
                        self._unpack_parameters(_x_to_conc(x))
                    conc_sum = film_conc.sum(axis=1)
                    full = conc_sum - 1.0 + cf
                    return full[az]

                def sum_constraint_nolinked_jac(x, az=_active, na=_n_active):
                    film_conc, thickness, abruptness, sub_ratios, _ = \
                        self._unpack_parameters(_x_to_conc(x))

                    n_x = len(x)
                    jac = np.zeros((na, n_x))

                    for row_i, z in enumerate(az):
                        for j in range(_n_film_c):
                            jac[row_i, z * _n_film_c + j] = 1.0

                    # Log-space chain rule: ∂conc/∂x_log = conc
                    if _use_log_space:
                        x_conc = _x_to_conc(x)
                        for row_i, z in enumerate(az):
                            for j in range(_n_film_c):
                                jac[row_i, z * _n_film_c + j] *= x_conc[z * _n_film_c + j]

                    # No erf_sub derivatives needed

                    return jac

                constraints.append({
                    'type': 'eq',
                    'fun': sum_constraint_nolinked,
                    'jac': sum_constraint_nolinked_jac,
                })

        # --- Charge balance (only when no linked elements) ---
        # MATLAB Const() line 504: only adds CnstMat(:,:,3) when linked is empty.
        # When linked elements exist, charge balance is implicitly enforced
        # via D_linked = D_free * (freelinkedconst - 1) in _build_full_profile.
        # MATLAB: CnstMat(:,:,3)*x + CnstChrg = 0
        if (self.semi_free_config.use_charge_constraint
                and not self.linked_idx
                and self.elements is not None):
            charges = np.array([
                getattr(e, 'oxidation_state', 0) or 0 for e in self.elements
            ])
            freecharge = charges[np.array(self.free_idx)]

            # Check if free elements have mixed-sign charges
            # MATLAB line 37: only add charge constraint if mixed signs
            has_positive = np.any(freecharge >= 0)
            has_negative = np.any(freecharge < 0)

            if has_positive and has_negative:
                # Const element charge contribution (CnstChrg in MATLAB)
                if _const_idx:
                    const_charge_vec = sum(
                        self.D_prior[:, ci] * charges[ci] for ci in _const_idx
                    )
                else:
                    const_charge_vec = np.zeros(self.n_depth)

                _fc = freecharge.copy()
                _cc = const_charge_vec.copy()

                def charge_constraint(x, fc=_fc, cc=_cc):
                    film_conc, _, _, _, _ = self._unpack_parameters(_x_to_conc(x))
                    # CnstMat(:,:,3) * x + CnstChrg = 0
                    return (film_conc * fc).sum(axis=1) + cc

                def charge_constraint_jac(x, fc=_fc):
                    """Analytical Jacobian for charge balance constraint.

                    ∂constraint(z)/∂film_conc[z,j] = fc[j] (block-diagonal)
                    All other partials are zero (no dependence on thickness/abruptness/sub_ratios).
                    """
                    n_x = len(x)
                    jac = np.zeros((_n_depth_c, n_x))
                    for z in range(_n_depth_c):
                        for j in range(_n_film_c):
                            jac[z, z * _n_film_c + j] = fc[j]
                    # Log-space chain rule
                    if _use_log_space:
                        x_conc = _x_to_conc(x)
                        for z in range(_n_depth_c):
                            for j in range(_n_film_c):
                                jac[z, z * _n_film_c + j] *= x_conc[z * _n_film_c + j]
                    return jac

                constraints.append({
                    'type': 'eq',
                    'fun': charge_constraint,
                    'jac': charge_constraint_jac,
                })

        # --- Constraint 3: Substrate ratios sum to 1 ---
        # MATLAB: CnstSub(1,:)*sub_ratios - 1 = 0
        if n_sub > 0:
            def substrate_sum_constraint(x):
                return x[_film_size_c + 2:_film_size_c + 2 + _n_sub_c].sum() - 1.0

            def substrate_sum_jac(x):
                """Analytical Jacobian for substrate sum constraint.

                ∂constraint/∂sub_ratios[s] = 1.0 for all s; all others zero.
                """
                jac = np.zeros((1, len(x)))
                jac[0, _film_size_c + 2:_film_size_c + 2 + _n_sub_c] = 1.0
                return jac

            constraints.append({
                'type': 'eq',
                'fun': substrate_sum_constraint,
                'jac': substrate_sum_jac,
            })

        # --- Constraint 4: Substrate charge = 0 ---
        # MATLAB: CnstSub(2,:)*sub_ratios = 0
        if (n_sub > 0 and self.semi_free_config.use_charge_constraint
                and self.elements is not None):
            sub_charges = np.array([
                getattr(self.elements[i], 'oxidation_state', 0) or 0 for i in self.sub_idx
            ])
            if np.any(sub_charges != 0):
                _sc = sub_charges.copy()

                def substrate_charge_constraint(x, sc=_sc):
                    sub_ratios = x[_film_size_c + 2:_film_size_c + 2 + _n_sub_c]
                    return np.dot(sc, sub_ratios)

                def substrate_charge_jac(x, sc=_sc):
                    """Analytical Jacobian for substrate charge constraint.

                    ∂constraint/∂sub_ratios[s] = sc[s] (oxidation state); all others zero.
                    """
                    jac = np.zeros((1, len(x)))
                    jac[0, _film_size_c + 2:_film_size_c + 2 + _n_sub_c] = sc
                    return jac

                constraints.append({
                    'type': 'eq',
                    'fun': substrate_charge_constraint,
                    'jac': substrate_charge_jac,
                })

        # --- Stoichiometric constraint for 090317-style mode ---
        if self._stoichiometric_mode == 'stoichiometric':
            coeffs = np.array(self._stoichiometric_constraint_coeffs)
            _stoich_active = active_depths.copy()
            _stoich_n_active = len(_stoich_active)

            def stoichiometric_constraint(x, az=_stoich_active):
                """Stoichiometric constraint: sum(coeff * D_free) = 1 - D_sub."""
                film_conc, thickness, abruptness, sub_ratios, grad_params = \
                    self._unpack_parameters(_x_to_conc(x))

                if self._gradient_mode and len(grad_params) > 0:
                    film_conc = self._apply_gradient_to_profile(film_conc, grad_params)

                D_sub = self._build_substrate_profile(thickness, abruptness, sub_ratios)
                D_sub_sum = D_sub.sum(axis=1)

                weighted_sum = (film_conc * coeffs).sum(axis=1)
                full = weighted_sum - (1 - D_sub_sum)
                return full[az]

            def stoichiometric_jac(x, az=_stoich_active, na=_stoich_n_active):
                """Analytical Jacobian for stoichiometric constraint.

                constraint(z) = Σ_j(coeffs[j] × film_conc[z,j]) - 1 + erf_sub(z)
                Same structure as sum constraint but with stoichiometric coefficients.
                """
                film_conc, thickness, abruptness, sub_ratios, grad_params = \
                    self._unpack_parameters(_x_to_conc(x))
                erf_sub, d_th, d_ab, d_sr = _erf_derivs(thickness, abruptness, sub_ratios)

                n_x = len(x)
                jac = np.zeros((na, n_x))

                # ∂/∂film_conc[z,j] = coeffs[j]
                for row_i, z in enumerate(az):
                    for j in range(_n_film_c):
                        jac[row_i, z * _n_film_c + j] = coeffs[j]

                # Log-space chain rule
                if _use_log_space:
                    x_conc = _x_to_conc(x)
                    for row_i, z in enumerate(az):
                        for j in range(_n_film_c):
                            jac[row_i, z * _n_film_c + j] *= x_conc[z * _n_film_c + j]

                jac[:, _film_size_c] = d_th[az]
                jac[:, _film_size_c + 1] = d_ab[az]
                for s in range(_n_sub_c):
                    jac[:, _film_size_c + 2 + s] = d_sr[az]

                # ∂/∂gradient_params (if gradient_mode enabled)
                if self._gradient_mode and _n_grad > 0:
                    z_rel = _depth_arr - _depth_arr[0]
                    grad_offset = _film_size_c + 2 + _n_sub_c

                    if self._gradient_type == 'linear':
                        for i, elem_idx in enumerate(self._gradient_element_idx):
                            free_pos = self.free_idx.index(elem_idx)
                            jac[:, grad_offset + i] = coeffs[free_pos] * z_rel[az]
                    elif self._gradient_type == 'quadratic':
                        for i, elem_idx in enumerate(self._gradient_element_idx):
                            free_pos = self.free_idx.index(elem_idx)
                            jac[:, grad_offset + 2 * i] = coeffs[free_pos] * z_rel[az]
                            jac[:, grad_offset + 2 * i + 1] = coeffs[free_pos] * z_rel[az] ** 2

                return jac

            constraints.append({
                'type': 'eq',
                'fun': stoichiometric_constraint,
                'jac': stoichiometric_jac,
            })

        # Optimization with analytical Jacobian.
        # Xreg weighting is now supported in the pure Python gradient kernel.
        # SLSQP *requires* analytical gradient for large parameter spaces with
        # equality constraints; with finite differences it can stop after one
        # iteration.
        # L1 subgradient creates non-smooth kinks that interact badly with
        # SLSQP's BFGS Hessian approximation, causing convergence to worse
        # local minima. FD naturally smooths these kinks. LLS/MEM are smooth
        # and benefit from the analytical gradient.
        if self.config.use_numerical_gradient:
            use_analytical_jac = False
        elif _use_log_space:
            # Log-space MEM: use FD for now (Numba kernels don't support method=5 yet)
            # Pure Python gradient supports it, but Numba forward model doesn't handle
            # log→conc transform, so FD is safer for correctness validation.
            use_analytical_jac = False
        elif self.config.method == 'L1' and self.config.use_xreg_weighting:
            # log_cosh is fully smooth → analytical gradient works
            if self.config.l1_smoothing == 'log_cosh':
                use_analytical_jac = True
            else:
                use_analytical_jac = False
        else:
            use_analytical_jac = True
        scipy_method = self.config.scipy_method
        # One spelling per method, so that 'slsqp' keeps the SLSQP fallback
        # routing and 'TRUST-CONSTR' gets the trust-constr options.
        scipy_method = _CONSTRAINT_CAPABLE_METHODS.get(scipy_method.upper(), scipy_method)
        if constraints and scipy_method.upper() not in _CONSTRAINT_CAPABLE_METHODS:
            # scipy.optimize.minimize ignores `constraints` for methods that
            # cannot handle them (it only warns), so the stoichiometric
            # equality constraints would be dropped without notice. The
            # method was named explicitly, so say what replaces it.
            warnings.warn(
                f"SemiFreeSolver: scipy_method={scipy_method!r} cannot handle "
                f"the equality constraints of this problem; using 'SLSQP' "
                f"instead (set scipy_method='SLSQP' or 'trust-constr' to "
                f"silence this warning).",
                UserWarning, stacklevel=2)
            scipy_method = 'SLSQP'
        options = {
            'maxiter': self.config.max_iter,
            'disp': verbose,
        }
        if scipy_method == 'trust-constr':
            options['gtol'] = self.config.tol
        else:
            options['ftol'] = self.config.tol
        result = minimize(
            objective,
            x0,
            method=scipy_method,
            jac=gradient if use_analytical_jac else None,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        # Condition-based fallback routing
        _fallback_used = False
        # SLSQP can report success without having moved at all. When its QP
        # subproblem (Kraft's LSQ) returns a non-descent step -- observed
        # with badly scaled gradients (|grad| ~ 1e5 on film concentrations
        # whose bound distances are ~0.1; scaling the objective by 0.1
        # makes the same solve converge normally) -- the solver resets its
        # BFGS matrix five times and then exits through its relaxed test,
        # which holds trivially because no step was ever taken
        # (|f - f0| = 0 < 10*ftol). The result is mode 0 at the unchanged
        # x0 with nit > 1. The test below looks only at "did not move": it
        # also fires when x0 already is the optimum and SLSQP leaves through
        # the same relaxed exit (restarting from a converged solution
        # returns success with x unchanged). Such a misfire costs
        # one fallback run (<= max_iter iterations) plus one SLSQP restart
        # and changes nothing, because the fallback result is kept only if
        # it succeeds or lowers the objective.
        _no_progress = bool(
            result.success
            and result.nit > 1
            and np.array_equal(np.asarray(result.x, dtype=float),
                               np.clip(x0, bounds.lb, bounds.ub))
        )
        if ((not result.success or _no_progress)
                and self.config.enable_fallback
                and scipy_method == 'SLSQP'):

            # Check constraint Jacobian condition number
            _max_condition = 0.0
            for c in constraints:
                if 'jac' in c and c['jac'] is not None:
                    try:
                        J = np.atleast_2d(c['jac'](result.x))
                        if J.shape[0] > 0:
                            s = np.linalg.svd(J, compute_uv=False)
                            cond = s[0] / max(s[-1], 1e-15)
                            _max_condition = max(_max_condition, cond)
                    except Exception:
                        pass

            _should_fallback = (
                _no_progress  # SLSQP "success" without any step (see above)
                or _max_condition > self.config.fallback_condition_threshold
                or result.status == 8  # positive directional derivative
                or result.status == 4  # inequality constraints incompatible
            )

            if _should_fallback:
                if verbose:
                    print(f"[SemiFreeSolver] SLSQP failed (status={result.status}, "
                          f"no_progress={_no_progress}, "
                          f"cond={_max_condition:.1e}), "
                          f"retrying with {self.config.fallback_method}")

                # The fallback is a second attempt, never a reason to lose the
                # first: if the fallback optimizer itself raises inside scipy
                # (seen with trust-constr in gradient_mode, scipy 1.16 and
                # 1.18: a dimension mismatch in its equality-constrained SQP
                # step), keep the SLSQP result and say so. Only that call is
                # guarded; errors from the polish below still propagate.
                result_fb = None
                try:
                    fallback_options = {
                        'maxiter': self.config.max_iter,
                        'disp': verbose,
                        'gtol': self.config.tol,
                    }
                    result_fb = minimize(
                        objective,
                        x0,  # restart from x0, not result.x
                        method=self.config.fallback_method,
                        jac=gradient if use_analytical_jac else None,
                        bounds=bounds,
                        constraints=constraints,
                        options=fallback_options,
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    warnings.warn(
                        f"SemiFreeSolver: fallback method "
                        f"{self.config.fallback_method!r} raised "
                        f"{type(exc).__name__}: {exc}; keeping the SLSQP result "
                        f"(success={bool(result.success)}, status={result.status}).",
                        RuntimeWarning, stacklevel=2)

                if result_fb is not None:
                    # Polish on the no-progress route (same pattern as
                    # FreeSolver): the fallback starts from x0, which SLSQP never
                    # left, so trust-constr typically exhausts its iteration
                    # budget short of stationarity (success=False) at a point
                    # that satisfies the equality constraints only to ~1e-7. A
                    # short SLSQP restart from there is meant to converge and
                    # satisfy the constraints. The status-8/4 and condition routes
                    # keep their previous behaviour.
                    if _no_progress:
                        result_pl = minimize(
                            objective,
                            np.clip(result_fb.x, bounds.lb, bounds.ub),
                            method='SLSQP',
                            jac=gradient if use_analytical_jac else None,
                            bounds=bounds,
                            constraints=constraints,
                            options=options,
                        )
                        if result_pl.success or result_pl.fun < result_fb.fun:
                            result_fb = result_pl

                    # Accept fallback if it improved
                    if result_fb.success or result_fb.fun < result.fun:
                        result = result_fb
                        _fallback_used = True
                        if verbose:
                            print(f"[SemiFreeSolver] Fallback succeeded: "
                                  f"obj={result.fun:.6e}, nit={result.nit}")

        # Diagnostic dump on failure (only when verbose)
        if not result.success and verbose:
            self._dump_diagnostics(x0, result, constraints, bounds)

        # ================================================================
        # Warm restart: re-run with updated active depths
        #
        # Degenerate depths are computed from x0 at the start of solve().
        # If thickness changed significantly, the set of degenerate depths
        # may be wrong (e.g., initially 11/20 degenerate but after
        # optimization only 5/20 should be degenerate).  Re-running with
        # the optimized x as x0 recomputes active_depths correctly.
        # ================================================================
        # Convert result to concentration space for unpacking
        result_x_conc = _x_to_conc(result.x)

        if (self.config.max_warm_restarts > 0
                and _warm_restart_count < self.config.max_warm_restarts):
            _, thickness_opt_check, abruptness_opt_check, sub_ratios_opt_check, _ = \
                self._unpack_parameters(result_x_conc)

            # Recompute active depths with optimized parameters
            D_sub_new = self._build_substrate_profile(
                thickness_opt_check, abruptness_opt_check, sub_ratios_opt_check
            )
            erf_sub_new = D_sub_new.sum(axis=1)
            rhs_new = 1.0 - erf_sub_new - const_fix
            new_active = np.where(rhs_new >= _DEGEN_THRESHOLD)[0]

            # Check if active depths changed
            active_changed = (len(new_active) != len(active_depths)
                              or not np.array_equal(new_active, active_depths))

            # Also check relative thickness change
            rel_change = abs(thickness_opt_check - thickness_init_val) / max(thickness_init_val, 1.0)

            if active_changed and rel_change > self.config.warm_restart_threshold:
                if verbose:
                    print(f"[SemiFreeSolver] Warm restart #{_warm_restart_count + 1}: "
                          f"thickness {thickness_init_val:.1f} → {thickness_opt_check:.1f} Å "
                          f"(Δ={rel_change:.1%}), "
                          f"active depths {len(active_depths)} → {len(new_active)}")

                # Re-run solve() with optimized x as starting point.
                # x0_override in concentration space (solve() applies log transform)
                return self.solve(
                    I_obs,
                    verbose=verbose,
                    x0_override=result_x_conc,
                    _warm_restart_count=_warm_restart_count + 1,
                )

        # Unpack result (in concentration space)
        film_conc_opt, thickness_opt, abruptness_opt, sub_ratios_opt, grad_params_opt = self._unpack_parameters(result_x_conc)
        D_opt = self._build_full_profile(film_conc_opt, thickness_opt, abruptness_opt, sub_ratios_opt, grad_params_opt)

        # Compute forward model
        I_opt = np.zeros_like(I_obs)
        for e in range(self.n_elements):
            I_opt[:, e] = T_matrices_list[e] @ D_opt[:, e]
        # Use total normalization for I_opt_norm (reporting only)
        I_opt_norm = I_opt / np.maximum(I_opt.sum(axis=1, keepdims=True), 1e-10)

        # Statistics: compute data-fit chi2 (unscaled — for reporting)
        if self.config.use_xreg_weighting and xreg_weights.size > 0:
            diff = I_norm[:, antisubst_idx] - I_opt_norm[:, antisubst_idx]
            chi2 = self.config.degree * np.sum((diff * xreg_weights[np.newaxis, :]) ** 2)
        elif self.config.use_poisson_nll:
            I_obs_sub = I_norm[:, antisubst_idx]
            I_obs_safe = np.maximum(I_obs_sub, 1e-10)
            I_pred_safe = np.maximum(I_opt_norm[:, antisubst_idx], 1e-10)
            chi2 = self.config.degree * 2.0 * np.sum(
                I_obs_safe * np.log(I_obs_safe / I_pred_safe) - (I_obs_sub - I_pred_safe)
            )
        elif self.config.use_wls:
            diff = I_norm[:, antisubst_idx] - I_opt_norm[:, antisubst_idx]
            chi2 = self.config.degree * np.sum(weights[:, antisubst_idx] * diff ** 2)
        else:
            diff = I_norm[:, antisubst_idx] - I_opt_norm[:, antisubst_idx]
            chi2 = self.config.degree * np.sum(diff ** 2)

        elapsed = time.time() - start_time

        # Safely extract jacobian diagnostics (result.jac may have different structures)
        opt_jac_norm = None
        opt_jac_thickness = None
        if hasattr(result, 'jac') and result.jac is not None:
            try:
                jac_arr = np.asarray(result.jac)
                opt_jac_norm = float(np.linalg.norm(jac_arr))
            except Exception:
                try:
                    # try concatenate list-like jac
                    flattened = np.concatenate([np.ravel(x) for x in result.jac])
                    opt_jac_norm = float(np.linalg.norm(flattened))
                except Exception:
                    opt_jac_norm = None

            try:
                if isinstance(result.jac, (list, tuple)):
                    flat = np.concatenate([np.ravel(x) for x in result.jac])
                    opt_jac_thickness = float(flat[film_size])
                else:
                    jac_arr = np.asarray(result.jac)
                    opt_jac_thickness = float(jac_arr[film_size])
            except Exception:
                opt_jac_thickness = None

        return SemiFreeSolverResult(
            D_opt=D_opt,
            I_opt=I_opt_norm,
            thickness=thickness_opt,
            abruptness=abruptness_opt,
            substrate_ratios=sub_ratios_opt,
            gradient_params=grad_params_opt if self._gradient_mode else None,
            gradient_element_indices=self._gradient_element_idx if self._gradient_mode else None,
            chi2=chi2,
            objective=result.fun,
            n_iterations=result.nit,
            elapsed_time=elapsed,
            success=result.success,
            opt_status=getattr(result, 'status', None),
            opt_message=getattr(result, 'message', None),
            opt_jac_norm=opt_jac_norm,
            opt_jac_thickness=opt_jac_thickness,
            fallback_used=_fallback_used,
            warm_restarts=_warm_restart_count,
            x_opt=result_x_conc.copy(),
        )

    def solve_alternating(
        self,
        I_obs: np.ndarray,
        verbose: bool = False,
        x0_override: Optional[np.ndarray] = None,
        max_brent_iter: int = 20,
        thickness_atol: float = 0.5,
    ) -> 'SemiFreeSolverResult':
        """Alternating optimization: Brent 1D thickness search + film_conc solve.

        Outer loop: ``scipy.optimize.minimize_scalar`` (Brent) over thickness.
        Inner loop: ``solve()`` with ``optimize_thickness=False`` at the trial
        thickness, warm-started from the previous inner solution. The number
        of inner iterations depends on the configuration and on convergence.

        Args:
            I_obs: Observed intensities (n_angles, n_elements)
            verbose: Print progress
            x0_override: Optional initial parameter vector
            max_brent_iter: Maximum Brent iterations (default 20)
            thickness_atol: Absolute tolerance for thickness (Å, default 0.5)

        Returns:
            SemiFreeSolverResult with optimized profile and parameters
        """
        start_time = time.time()

        # Save original state
        saved_optimize_thickness = self.semi_free_config.optimize_thickness
        saved_thickness_init = self.semi_free_config.thickness_init
        saved_warm_restarts = self.config.max_warm_restarts
        saved_enable_fallback = self.config.enable_fallback
        thickness_bounds = self.semi_free_config.thickness_bounds

        try:
            # Inner solve config: fix thickness, no warm restarts
            self.semi_free_config.optimize_thickness = False
            self.config.max_warm_restarts = 0
            self.config.enable_fallback = False

            n_film = len(self.film_idx)
            film_size = self.n_depth * n_film

            # Track last solution for warm-starting across Brent evals
            last_x_conc = x0_override
            eval_count = 0
            total_inner_iters = 0

            # Thickness L2 penalty: inner solve has thickness=m0 so penalty=0.
            # Add it back in the outer loop using the ORIGINAL thickness_init
            # as the reference (prior).
            thickness_prior = saved_thickness_init
            alpha = self.config.alpha

            def thickness_objective(t):
                nonlocal last_x_conc, eval_count, total_inner_iters
                eval_count += 1

                self.semi_free_config.thickness_init = t

                # Warm-start: reuse film_conc from previous eval, update thickness
                x0 = None
                if last_x_conc is not None:
                    x0 = last_x_conc.copy()
                    x0[film_size] = t  # update thickness in x vector
                    x0[film_size + 1] = self.semi_free_config.abruptness_init  # keep abruptness

                result = self.solve(I_obs, verbose=False, x0_override=x0)
                last_x_conc = result.x_opt
                total_inner_iters += result.n_iterations

                # Outer objective: inner objective + thickness L2 penalty
                # (inner obj already includes film_conc L2 but NOT thickness L2)
                obj = result.objective
                if self.config.method == 'L2' and alpha > 0:
                    obj += alpha * 100 * ((t - thickness_prior) / 100) ** 2
                elif self.config.method == 'MEM' and alpha > 0:
                    t_safe = max(t, 1e-10)
                    tp_safe = max(thickness_prior, 1e-10)
                    obj += alpha * (t_safe * np.log(t_safe / tp_safe) - t_safe + tp_safe) / 10

                if verbose:
                    print(f"  Brent #{eval_count}: t={t:.2f} Å, "
                          f"obj={obj:.6e}, chi2={result.chi2:.6e}, "
                          f"nit={result.n_iterations}")

                return obj

            opt = minimize_scalar(
                thickness_objective,
                bounds=thickness_bounds,
                method='bounded',
                options={'maxiter': max_brent_iter, 'xatol': thickness_atol},
            )

            # Final solve at optimal thickness
            self.semi_free_config.thickness_init = opt.x
            x0_final = None
            if last_x_conc is not None:
                x0_final = last_x_conc.copy()
                x0_final[film_size] = opt.x
            final = self.solve(I_obs, verbose=False, x0_override=x0_final)

            elapsed = time.time() - start_time

            if verbose:
                print(f"  Alternating done: {eval_count} evals, "
                      f"{total_inner_iters} total inner iters, "
                      f"t*={opt.x:.2f} Å, chi2={final.chi2:.6e}, "
                      f"{elapsed*1000:.1f} ms")

            # Override timing in result
            final.elapsed_time = elapsed
            # Store total iterations (Brent evals + inner SLSQP iters)
            final.n_iterations = total_inner_iters
            final.warm_restarts = eval_count  # repurpose: number of Brent evals

            return final

        finally:
            # Restore original state
            self.semi_free_config.optimize_thickness = saved_optimize_thickness
            self.semi_free_config.thickness_init = saved_thickness_init
            self.config.max_warm_restarts = saved_warm_restarts
            self.config.enable_fallback = saved_enable_fallback


@dataclass
class SemiFreeSolverResult:
    """Result from film structure solver."""
    D_opt: np.ndarray                # Optimized depth profile (n_depth, n_elements)
    I_opt: np.ndarray                # Forward model prediction (n_angles, n_elements)
    thickness: float                 # Optimized film thickness (Å)
    abruptness: float                # Optimized interface abruptness (Å)
    substrate_ratios: np.ndarray     # Optimized substrate ratios
    chi2: float                      # Final Chi² value
    objective: float                 # Final objective value
    n_iterations: int                # Number of iterations
    elapsed_time: float              # Elapsed time (seconds)
    success: bool                    # Convergence status
    opt_status: Optional[int] = None  # SciPy OptimizeResult.status
    opt_message: Optional[str] = None  # SciPy OptimizeResult.message
    opt_jac_norm: Optional[float] = None  # Norm of returned jac (if available)
    opt_jac_thickness: Optional[float] = None  # Jacobian component for thickness (if available)
    gradient_params: Optional[np.ndarray] = None  # Gradient coefficients (slopes)
    gradient_element_indices: Optional[List[int]] = None  # Element indices with gradients
    fallback_used: bool = False              # Whether fallback method was used
    warm_restarts: int = 0                   # Number of warm restarts performed
    x_opt: Optional[np.ndarray] = None       # Raw optimization vector for warm-start

    @property
    def rmse(self) -> float:
        """Root mean square error."""
        return np.sqrt(self.chi2 / self.I_opt.size)

    @property
    def has_gradient(self) -> bool:
        """Whether gradient optimization was used."""
        return self.gradient_params is not None and len(self.gradient_params) > 0

    def get_gradient_slopes(self) -> Dict[int, float]:
        """Get gradient slopes as dict {element_idx: slope}."""
        if not self.has_gradient:
            return {}
        return {
            idx: self.gradient_params[i]
            for i, idx in enumerate(self.gradient_element_indices)
        }


def reconstruct_depth_profile(
    I_obs: np.ndarray,
    transfer_matrix: TransferMatrix,
    method: OptMethod = 'L2',
    alpha: float = 1.0,
    D_prior: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> SolverResult:
    """
    Convenience function for depth profile reconstruction.

    Args:
        I_obs: Observed intensities (n_angles, n_elements)
        transfer_matrix: Pre-computed transfer matrix
        method: Optimization method ('L2', 'L1', 'MEM', 'LLS')
        alpha: Regularization strength
        D_prior: Prior depth profile
        verbose: Print progress

    Returns:
        SolverResult with reconstructed profile
    """
    config = SolverConfig(method=method, alpha=alpha)
    solver = FreeSolver(transfer_matrix, config, D_prior)
    return solver.solve(I_obs, verbose=verbose)


# =============================================================================
# Generic Layer Solver (2026-01-24)
# =============================================================================

@dataclass
class LayerDefinition:
    """
    Definition of one layer

    Parameters
    ----------
    name : str
        Layer name (e.g., "SiO2", "contamination", "SiOxNy")
    element_indices : List[int]
        Indices of the elements present in this layer
    stoichiometry : Dict[int, float]
        Mapping element index → stoichiometric ratio
        Example: SiO2 = {0: 1, 1: 2} (Si:O = 1:2)
    stoichiometry_bounds : Dict[int, Tuple[float, float]], optional
        Mapping element index → (min, max) range of the stoichiometric ratio
        The stoichiometric ratio of the given elements becomes an optimization parameter
        Example: {1: (1.5, 2.0)} → optimize the O ratio within 1.5-2.0 (oxygen deficiency)
    profile_type : str
        "erf" (default), "gaussian", "exponential"
    base_concentration : float
        Reference concentration within the layer (before normalization)

    Examples
    --------
    >>> # SiO2 layer (fixed composition)
    >>> sio2_layer = LayerDefinition(
    ...     name="SiO2",
    ...     element_indices=[0, 1],  # Si_ox, O
    ...     stoichiometry={0: 1, 1: 2},
    ... )

    >>> # SiOx layer (oxygen deficient, variable composition)
    >>> siox_layer = LayerDefinition(
    ...     name="SiOx",
    ...     element_indices=[0, 1],  # Si_ox, O
    ...     stoichiometry={0: 1, 1: 2.0},  # initial value
    ...     stoichiometry_bounds={1: (1.5, 2.0)},  # optimize the O ratio
    ... )

    >>> # SiOxNy layer (nitride, variable N/O ratio)
    >>> sion_layer = LayerDefinition(
    ...     name="SiOxNy",
    ...     element_indices=[0, 1, 2],  # Si, O, N
    ...     stoichiometry={0: 3, 1: 1, 2: 3},  # initial value
    ...     stoichiometry_bounds={1: (0, 2), 2: (2, 4)},  # optimize the O and N ratios
    ... )
    """
    name: str
    element_indices: List[int]
    stoichiometry: Dict[int, float]
    stoichiometry_bounds: Optional[Dict[int, Tuple[float, float]]] = None
    profile_type: str = "erf"
    base_concentration: float = 1.0

    @property
    def has_variable_stoichiometry(self) -> bool:
        """Whether there are variable-composition parameters"""
        return self.stoichiometry_bounds is not None and len(self.stoichiometry_bounds) > 0

    @property
    def n_stoich_params(self) -> int:
        """Number of variable-composition parameters"""
        if self.stoichiometry_bounds is None:
            return 0
        return len(self.stoichiometry_bounds)


@dataclass
class GenericLayerConfig:
    """
    Configuration of the generic layer-structure solver

    Any number of layers. The profile is built automatically from the layer definitions.

    Parameters
    ----------
    layers : List[LayerDefinition]
        List of layer definitions, from the surface toward the substrate
    substrate : LayerDefinition
        Definition of the substrate
    initial_thicknesses : List[float]
        Initial thickness of each layer [Å]
    initial_widths : List[float]
        Initial width of each interface [Å]
    optimize_widths : bool
        Whether to optimize the interface widths (default: False; keeping them fixed is recommended)
    thickness_bounds : Tuple[float, float]
        Thickness bounds (min, max)
    width_bounds : Tuple[float, float]
        Interface-width bounds (min, max)
    linked_element_index : Optional[int]
        Index of the element determined by charge conservation (e.g., O)
        With None, it is not computed automatically from the stoichiometry

    Examples
    --------
    >>> # SiO2/Si single-layer system
    >>> config = GenericLayerConfig(
    ...     layers=[
    ...         LayerDefinition(
    ...             name="SiO2",
    ...             element_indices=[0, 1],
    ...             stoichiometry={0: 1, 1: 2},
    ...         ),
    ...     ],
    ...     substrate=LayerDefinition(
    ...         name="Si",
    ...         element_indices=[2],
    ...         stoichiometry={2: 1},
    ...     ),
    ...     initial_thicknesses=[30.0],
    ...     initial_widths=[5.0],
    ...     linked_element_index=1,  # O
    ... )
    """
    layers: List[LayerDefinition]
    substrate: LayerDefinition
    initial_thicknesses: List[float]
    initial_widths: List[float]
    optimize_widths: bool = False
    optimize_stoichiometry: bool = False  # whether to optimize the stoichiometric ratios
    thickness_bounds: Tuple[float, float] = (1.0, 200.0)
    width_bounds: Tuple[float, float] = (0.5, 30.0)
    linked_element_index: Optional[int] = None

    @property
    def n_layers(self) -> int:
        """Number of layers (excluding the substrate)"""
        return len(self.layers)

    @property
    def n_stoich_params(self) -> int:
        """Total number of variable-composition parameters over all layers"""
        return sum(layer.n_stoich_params for layer in self.layers)

    @property
    def has_variable_stoichiometry(self) -> bool:
        """Whether there are variable-composition parameters"""
        return any(layer.has_variable_stoichiometry for layer in self.layers)

    def __post_init__(self):
        """Validation"""
        if len(self.initial_thicknesses) != len(self.layers):
            raise ValueError(
                f"initial_thicknesses ({len(self.initial_thicknesses)}) must match "
                f"number of layers ({len(self.layers)})"
            )
        if len(self.initial_widths) != len(self.layers):
            raise ValueError(
                f"initial_widths ({len(self.initial_widths)}) must match "
                f"number of layers ({len(self.layers)})"
            )
        # Warn when optimize_stoichiometry is True but there are no variable parameters
        if self.optimize_stoichiometry and not self.has_variable_stoichiometry:
            import warnings
            warnings.warn(
                "optimize_stoichiometry=True but no layers have stoichiometry_bounds defined. "
                "Define stoichiometry_bounds in LayerDefinition to enable composition optimization."
            )


@dataclass
class GenericLayerResult:
    """Result of the generic layer-structure solver"""
    D_opt: np.ndarray                    # optimized depth profile
    I_opt: np.ndarray                    # forward prediction
    layer_thicknesses: np.ndarray        # optimized thicknesses
    interface_widths: np.ndarray         # optimized interface widths
    chi2: float                          # Chi² value
    objective: float                     # objective value
    n_iterations: int                    # number of iterations
    elapsed_time: float                  # computation time
    success: bool                        # convergence flag
    # Composition optimization results (optional)
    layer_stoichiometries: Optional[List[Dict[int, float]]] = None  # optimized stoichiometric ratios of each layer
    assumptions: Optional[Dict[str, any]] = None  # the assumptions used (for display)

    @property
    def rmse(self) -> float:
        """Root mean square error"""
        return np.sqrt(self.chi2 / self.I_opt.size)

    def summary(self) -> str:
        """Return a summary of the result as a string (including the assumptions)"""
        lines = [
            "=== GenericLayerSolver Result ===",
            f"Thicknesses: {self.layer_thicknesses}",
            f"Interface widths: {self.interface_widths}",
            f"Chi²: {self.chi2:.6f}",
            f"RMSE: {self.rmse:.6f}",
            f"Converged: {self.success}",
        ]
        if self.layer_stoichiometries:
            lines.append("Stoichiometric ratios:")
            for i, stoich in enumerate(self.layer_stoichiometries):
                lines.append(f"  Layer {i}: {stoich}")
        if self.assumptions:
            lines.append("Assumptions used:")
            for key, val in self.assumptions.items():
                lines.append(f"  {key}: {val}")
        return "\n".join(lines)


class GenericLayerSolver:
    """
    Generic layer-structure solver

    A layer-structure optimization solver for any number of layers.
    Optimizes only the thicknesses (and interface widths); concentrations follow from the stoichiometry.

    Handles simple systems such as SiO2/Si as well as complex multilayers such as
    C/Al2O3/TiO2/SiO2/Si.

    Parameters
    ----------
    transfer_matrix : TransferMatrix
        Transfer matrix
    config : SolverConfig
        Solver configuration
    layer_config : GenericLayerConfig
        Layer-structure configuration
    elements : List[ElementConfig], optional
        Element configuration (for the IMFP calculation)

    Examples
    --------
    >>> # SiO2/Si single-layer system
    >>> config = GenericLayerConfig(
    ...     layers=[LayerDefinition("SiO2", [0, 1], {0: 1, 1: 2})],
    ...     substrate=LayerDefinition("Si", [2], {2: 1}),
    ...     initial_thicknesses=[30.0],
    ...     initial_widths=[5.0],
    ...     linked_element_index=1,
    ... )
    >>> solver = GenericLayerSolver(tm, SolverConfig(method='LLS'), config)
    >>> result = solver.solve(I_obs)
    >>> print(f"Thickness: {result.layer_thicknesses[0]:.2f} Å")
    """

    def __init__(
        self,
        transfer_matrix: TransferMatrix,
        config: Optional[SolverConfig] = None,
        layer_config: Optional[GenericLayerConfig] = None,
        elements: Optional[List] = None,
    ):
        self.tm = transfer_matrix
        self.config = config or SolverConfig()
        self.layer_config = layer_config
        self.elements = elements

        self.depth = transfer_matrix.depth
        self.n_depth = len(self.depth)
        self.n_elements = transfer_matrix.n_elements

        if layer_config is None:
            raise ValueError("layer_config is required")

    def _build_profile(
        self,
        thicknesses: np.ndarray,
        widths: np.ndarray,
        layer_stoichiometries: Optional[List[Dict[int, float]]] = None,
    ) -> np.ndarray:
        """
        Build the depth profile from the structure parameters

        Parameters
        ----------
        thicknesses : ndarray
            Thickness of each layer [n_layers]
        widths : ndarray
            Width of each interface [n_layers]
        layer_stoichiometries : List[Dict[int, float]], optional
            Stoichiometric ratios of each layer (for variable composition). With None, the LayerDefinition values are used

        Returns
        -------
        D : ndarray
            Depth profile [n_depth, n_elements]
        """
        cfg = self.layer_config
        z = self.depth
        n_layers = cfg.n_layers

        # Use the defaults when no composition is given
        if layer_stoichiometries is None:
            layer_stoichiometries = [dict(layer.stoichiometry) for layer in cfg.layers]

        # Initialize
        D = np.zeros((self.n_depth, self.n_elements))

        # Cumulative thicknesses
        cumulative_thicknesses = np.cumsum(thicknesses)

        # sigma = FWHM / (2 * sqrt(ln(2)))
        sigmas = widths / (2 * np.sqrt(np.log(2)))

        # Profile of each layer
        layer_profiles = []

        for i, layer in enumerate(cfg.layers):
            # Start and end of the layer
            z_start = 0 if i == 0 else cumulative_thicknesses[i - 1]
            z_end = cumulative_thicknesses[i]

            # sigma
            sigma_start = sigmas[0] if i == 0 else sigmas[i - 1]
            sigma_end = sigmas[i]

            if layer.profile_type == "erf":
                # Define the layer with erf
                rising = 0.5 * (1 + erf((z - z_start) / sigma_start)) if i > 0 else np.ones_like(z)
                falling = 0.5 * (1 + erf((z - z_end) / sigma_end))
                profile = np.maximum(rising - falling, 0)
            elif layer.profile_type == "exponential":
                # Exponential decay (for surface contamination layers)
                decay = np.exp(-z / max(thicknesses[i], 1.0))
                cutoff = 1 - 0.5 * (1 + erf((z - z_end) / sigma_end))
                profile = decay * cutoff
            else:
                # erf by default
                rising = 0.5 * (1 + erf((z - z_start) / sigma_start)) if i > 0 else np.ones_like(z)
                falling = 0.5 * (1 + erf((z - z_end) / sigma_end))
                profile = np.maximum(rising - falling, 0)

            layer_profiles.append(profile)

            # Use the variable composition
            stoich = layer_stoichiometries[i]
            total_stoich = sum(stoich.values())
            for elem_idx, elem_stoich in stoich.items():
                if cfg.linked_element_index is not None and elem_idx == cfg.linked_element_index:
                    continue  # linked is computed later
                D[:, elem_idx] += layer.base_concentration * (elem_stoich / total_stoich) * profile

        # Substrate
        z_sub = cumulative_thicknesses[-1] if n_layers > 0 else 0
        sigma_sub = sigmas[-1] if n_layers > 0 else 1.0
        substrate_profile = 0.5 * (1 + erf((z - z_sub) / sigma_sub))

        total_stoich = sum(cfg.substrate.stoichiometry.values())
        for elem_idx, stoich in cfg.substrate.stoichiometry.items():
            D[:, elem_idx] += (stoich / total_stoich) * substrate_profile

        # Linked elements (charge conservation)
        if cfg.linked_element_index is not None:
            linked_idx = cfg.linked_element_index

            # Simplified: linked = sum(layer_profiles * linked_stoich / total_stoich)
            D[:, linked_idx] = 0
            for i, layer in enumerate(cfg.layers):
                stoich = layer_stoichiometries[i]
                if linked_idx in stoich:
                    total_stoich = sum(stoich.values())
                    linked_stoich = stoich[linked_idx]
                    D[:, linked_idx] += layer.base_concentration * (linked_stoich / total_stoich) * layer_profiles[i]

        # Clip and normalize
        D = np.clip(D, 1e-6, None)
        total = D.sum(axis=1, keepdims=True)
        D = D / np.maximum(total, 1e-10)

        return D

    def _pack_stoichiometry_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Pack the variable-composition parameters into a vector

        Returns
        -------
        stoich_init : ndarray
            Initial-value vector
        stoich_lb : ndarray
            Lower-bound vector
        stoich_ub : ndarray
            Upper-bound vector
        stoich_map : List[Tuple[int, int]]
            Mapping of (layer_idx, element_idx)
        """
        cfg = self.layer_config
        stoich_init = []
        stoich_lb = []
        stoich_ub = []
        stoich_map = []  # (layer_idx, element_idx)

        for layer_idx, layer in enumerate(cfg.layers):
            if layer.stoichiometry_bounds:
                for elem_idx, (lb, ub) in layer.stoichiometry_bounds.items():
                    stoich_init.append(layer.stoichiometry[elem_idx])
                    stoich_lb.append(lb)
                    stoich_ub.append(ub)
                    stoich_map.append((layer_idx, elem_idx))

        return (
            np.array(stoich_init) if stoich_init else np.array([]),
            np.array(stoich_lb) if stoich_lb else np.array([]),
            np.array(stoich_ub) if stoich_ub else np.array([]),
            stoich_map,
        )

    def _unpack_stoichiometry_params(
        self,
        stoich_params: np.ndarray,
        stoich_map: List[Tuple[int, int]],
    ) -> List[Dict[int, float]]:
        """
        Unpack the composition parameters into per-layer dicts

        Returns
        -------
        layer_stoichiometries : List[Dict[int, float]]
            Stoichiometric ratios of each layer (updated)
        """
        cfg = self.layer_config
        # Copy the initial values
        layer_stoichiometries = [dict(layer.stoichiometry) for layer in cfg.layers]

        # Update with the optimization parameters
        for i, (layer_idx, elem_idx) in enumerate(stoich_map):
            layer_stoichiometries[layer_idx][elem_idx] = stoich_params[i]

        return layer_stoichiometries

    def _build_assumptions_dict(self) -> Dict[str, any]:
        """Collect the assumptions used into a dict (for display)"""
        cfg = self.layer_config
        assumptions = {
            'optimize_widths': cfg.optimize_widths,
            'optimize_stoichiometry': cfg.optimize_stoichiometry,
            'interface_model': 'erf',
            'fixed_parameters': [],
            'variable_parameters': [],
        }

        # Fixed parameters
        if not cfg.optimize_widths:
            assumptions['fixed_parameters'].append(f'interface_widths={cfg.initial_widths}')
        if not cfg.optimize_stoichiometry:
            for i, layer in enumerate(cfg.layers):
                assumptions['fixed_parameters'].append(f'layer[{i}].stoichiometry={layer.stoichiometry}')

        # Variable parameters
        assumptions['variable_parameters'].append('thicknesses')
        if cfg.optimize_widths:
            assumptions['variable_parameters'].append('interface_widths')
        if cfg.optimize_stoichiometry:
            for i, layer in enumerate(cfg.layers):
                if layer.stoichiometry_bounds:
                    for elem_idx, bounds in layer.stoichiometry_bounds.items():
                        assumptions['variable_parameters'].append(
                            f'layer[{i}].stoichiometry[{elem_idx}] in {bounds}'
                        )

        return assumptions

    def solve(
        self,
        I_obs: np.ndarray,
        verbose: bool = False,
    ) -> GenericLayerResult:
        """
        Solve the inverse problem

        Parameters
        ----------
        I_obs : ndarray
            Observed intensities [n_angles, n_elements]
        verbose : bool
            Verbose output

        Returns
        -------
        GenericLayerResult
        """
        start_time = time.time()
        cfg = self.layer_config
        n_layers = cfg.n_layers

        # Normalize the observations
        I_obs_norm = I_obs / np.maximum(I_obs.sum(axis=1, keepdims=True), 1e-10)

        # Initial values
        thicknesses_init = np.array(cfg.initial_thicknesses)
        widths_init = np.array(cfg.initial_widths)

        # Composition parameters
        stoich_init, stoich_lb, stoich_ub, stoich_map = self._pack_stoichiometry_params()
        n_stoich = len(stoich_init)
        optimize_stoich = cfg.optimize_stoichiometry and n_stoich > 0

        # Pack the optimization variables
        # Structure: [thicknesses, (widths), (stoichiometry)]
        x0_parts = [thicknesses_init]
        lb_parts = [np.full(n_layers, cfg.thickness_bounds[0])]
        ub_parts = [np.full(n_layers, cfg.thickness_bounds[1])]

        if cfg.optimize_widths:
            x0_parts.append(widths_init)
            lb_parts.append(np.full(n_layers, cfg.width_bounds[0]))
            ub_parts.append(np.full(n_layers, cfg.width_bounds[1]))

        if optimize_stoich:
            x0_parts.append(stoich_init)
            lb_parts.append(stoich_lb)
            ub_parts.append(stoich_ub)

        x0 = np.concatenate(x0_parts)
        lb = np.concatenate(lb_parts)
        ub = np.concatenate(ub_parts)
        bounds = Bounds(lb=lb, ub=ub)

        # Transfer matrix
        T_matrices = [self.tm.get_matrix(e) for e in range(self.n_elements)]

        # For index arithmetic
        idx_thickness_end = n_layers
        idx_width_end = idx_thickness_end + (n_layers if cfg.optimize_widths else 0)
        idx_stoich_end = idx_width_end + (n_stoich if optimize_stoich else 0)

        def objective(x):
            """Objective function"""
            # Unpack the parameters
            thicknesses = x[:idx_thickness_end]

            if cfg.optimize_widths:
                widths = x[idx_thickness_end:idx_width_end]
            else:
                widths = widths_init

            if optimize_stoich:
                stoich_params = x[idx_width_end:idx_stoich_end]
                layer_stoichs = self._unpack_stoichiometry_params(stoich_params, stoich_map)
            else:
                layer_stoichs = None

            # Build the profile (variable composition supported)
            D = self._build_profile(thicknesses, widths, layer_stoichs)

            # Forward problem
            I_pred = np.zeros((len(self.tm.angles), self.n_elements))
            for e in range(self.n_elements):
                I_pred[:, e] = T_matrices[e] @ D[:, e]

            # Normalize
            I_pred_norm = I_pred / np.maximum(I_pred.sum(axis=1, keepdims=True), 1e-10)

            # Chi²
            chi2 = np.sum((I_obs_norm - I_pred_norm)**2)

            return chi2

        # Optimization
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.config.max_iter,
                'ftol': self.config.tol,
            },
        )

        # Unpack the result
        thicknesses_opt = result.x[:idx_thickness_end]

        if cfg.optimize_widths:
            widths_opt = result.x[idx_thickness_end:idx_width_end]
        else:
            widths_opt = widths_init

        if optimize_stoich:
            stoich_params_opt = result.x[idx_width_end:idx_stoich_end]
            layer_stoichs_opt = self._unpack_stoichiometry_params(stoich_params_opt, stoich_map)
        else:
            layer_stoichs_opt = [dict(layer.stoichiometry) for layer in cfg.layers]

        # Final profile
        D_opt = self._build_profile(thicknesses_opt, widths_opt, layer_stoichs_opt)

        # Final prediction
        I_opt = np.zeros((len(self.tm.angles), self.n_elements))
        for e in range(self.n_elements):
            I_opt[:, e] = T_matrices[e] @ D_opt[:, e]

        elapsed = time.time() - start_time

        # Build the assumptions dict
        assumptions = self._build_assumptions_dict()

        if verbose:
            print(f"GenericLayerSolver result:")
            print(f"  converged: {result.success}")
            print(f"  thicknesses: {thicknesses_opt}")
            print(f"  interface widths: {widths_opt}")
            if optimize_stoich:
                print(f"  stoichiometric ratios:")
                for i, stoich in enumerate(layer_stoichs_opt):
                    print(f"    Layer {i}: {stoich}")
            print(f"  Chi²: {result.fun:.6f}")
            print(f"  time: {elapsed:.3f} s")
            print(f"  assumptions: {assumptions['fixed_parameters']}")

        return GenericLayerResult(
            D_opt=D_opt,
            I_opt=I_opt,
            layer_thicknesses=thicknesses_opt,
            interface_widths=widths_opt,
            chi2=result.fun,
            objective=result.fun,
            n_iterations=result.nit if hasattr(result, 'nit') else 0,
            elapsed_time=elapsed,
            success=result.success,
            layer_stoichiometries=layer_stoichs_opt,
            assumptions=assumptions,
        )


# =============================================================================
# Backward compatibility aliases (deprecated, will be removed in future)
# =============================================================================

# Solver classes
InverseSolver = FreeSolver
FilmStructureSolver = SemiFreeSolver
# Note: LayerStructureSolver removed (2026-01-24) - use GenericLayerSolver

# Config classes
FilmStructureConfig = SemiFreeSolverConfig

# Result classes
FilmStructureResult = SemiFreeSolverResult
