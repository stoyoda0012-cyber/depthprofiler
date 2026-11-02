"""
Fisher Information Matrix and Cramér-Rao Lower Bound for ARXPS Depth Profiling
================================================================================

Computes the statistical precision limits (CRLB) for depth profile estimation
from angle-resolved XPS data, enabling quantitative assessment of solver
efficiency relative to the theoretical optimum.

Mathematical framework:
    Observation model:  X_obs(θ, e) = I(θ,e) / Σ_e' I(θ,e')  + noise
    Forward model:      I(θ, e) = T(θ, :, e) · D(:, e)
    Fisher matrix:      F_ij = Σ_{θ,e} (1/σ²) · (∂X/∂p_i) · (∂X/∂p_j)
    CRLB:              σ_i ≥ sqrt([F^{-1}]_ii)

Two parameter spaces are supported:
    1. "full" — all n_depth × n_free film concentrations + thickness + abruptness
       + substrate ratios (the SLSQP x-vector)
    2. "concentration" — only film concentrations at each depth (n_depth × n_free),
       marginalised over structural parameters

Usage:
    from deppro.fisher_information import FisherInformation
    from deppro.io.dstk import load_dstk_data, create_tm_from_h5

    data = load_dstk_data(h5_path)
    tm = create_tm_from_h5(data)
    fi = FisherInformation(tm, data)
    result = fi.compute_crlb(pixel_idx=0)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from math import erf

from .inverse_solver import (
    SemiFreeSolver,
    SemiFreeSolverConfig,
    SolverConfig,
)
from .transfer_matrix import TransferMatrix


@dataclass
class CRLBResult:
    """Result container for CRLB analysis."""

    # Fisher matrix and its properties
    fisher_matrix: np.ndarray          # (n_params, n_params)
    condition_number: float
    singular_values: np.ndarray        # (n_params,)
    effective_rank: int

    # CRLB per parameter
    crlb: np.ndarray                   # (n_params,)

    # Reshaped views for convenience
    crlb_film_conc: np.ndarray         # (n_depth, n_free)
    crlb_thickness: float
    crlb_abruptness: float
    crlb_sub_ratios: np.ndarray        # (n_sub,)

    # Parameters used
    profile: np.ndarray                # (n_depth, n_elements) — D(z) at which CRLB is evaluated
    x_vector: np.ndarray               # Parameter vector at evaluation point
    n_counts: float                    # Total count scaling
    svd_threshold: float               # SVD truncation threshold used

    # Jacobian
    jacobian: np.ndarray               # (n_obs, n_params) — ∂X_norm/∂x


class FisherInformation:
    """Fisher Information Matrix and CRLB calculator for ARXPS depth profiling.

    Works with the SemiFreeSolver parameter space (x-vector):
        x = [film_conc.ravel(), thickness, abruptness, sub_ratios]
    and computes the Jacobian of the *normalised* forward model X_norm(θ,e)
    with respect to x.
    """

    def __init__(
        self,
        transfer_matrix: TransferMatrix,
        benchmark_data,  # DstkBenchmarkData
        n_counts: float = 1e6,
    ):
        """
        Parameters
        ----------
        transfer_matrix : TransferMatrix
            Pre-computed transfer matrix (from H5).
        benchmark_data : DstkBenchmarkData
            Benchmark data with Dstk, Xstk, element info, etc.
        n_counts : float
            Total count number for Poisson noise scaling.
            CRLB ∝ 1/sqrt(n_counts).
        """
        self.tm = transfer_matrix
        self.data = benchmark_data
        self.n_counts = n_counts

        # Dimensions (needed before _create_solver)
        self.n_depth = benchmark_data.n_depth
        self.n_angles = benchmark_data.n_angles
        self.n_elements = benchmark_data.n_elements

        # Build solver (for parameter unpacking and profile construction)
        self._solver = self._create_solver()

        # Derived dimensions from solver
        self.n_free = len(self._solver.free_idx)
        self.n_sub = len(self._solver.sub_idx)
        self.n_linked = len(self._solver.linked_idx)

        # Number of observations = n_angles × n_elements (normalised intensities)
        # We exclude substrate elements from the chi² (antisubst), but include
        # them in the Jacobian since they contribute through normalisation.
        # For CRLB we use the full observation vector.
        self.n_obs = self.n_angles * self.n_elements

        # Number of parameters in x-vector
        self.film_size = self.n_depth * self.n_free
        self.n_params = self.film_size + 2 + self.n_sub

    def _create_solver(self) -> SemiFreeSolver:
        """Create a SemiFreeSolver configured from benchmark data."""
        d = self.data
        cfg = SolverConfig(method='LLS', degree=1.0)
        semi_cfg = SemiFreeSolverConfig(
            free_indices=d.free_indices,
            substrate_indices=d.substrate_indices,
            linked_indices=d.linked_indices,
            thickness_init=40.0,
            abruptness_init=8.0,
            thickness_bounds=(10, 500),
            abruptness_bounds=(1, 50),
        )
        solver = SemiFreeSolver(
            self.tm, cfg, semi_cfg,
            D_prior=None,
            elements=d.elements,
            use_numba=False,  # Need pure Python for FD Jacobian
        )
        # Override free_linked_const from benchmark data if available
        n_free = len(solver.free_idx)
        if hasattr(d, 'linkedconst') and d.linkedconst is not None and len(d.linkedconst) >= n_free:
            solver.free_linked_const = np.maximum(d.linkedconst[:n_free], 1.0)
        return solver

    def _build_x_from_profile(
        self,
        D: np.ndarray,
        thickness: float,
        abruptness: float,
    ) -> np.ndarray:
        """Build parameter vector x from a known profile D(z) and structural params.

        Parameters
        ----------
        D : (n_depth, n_elements)
        thickness : float (Å)
        abruptness : float (Å)
        """
        solver = self._solver
        film_conc = D[:, solver.free_idx].copy()

        # Substrate profile at each depth
        _2sqrt_ln2 = 2.0 * np.sqrt(np.log(2.0))
        sigma = abruptness / _2sqrt_ln2
        depth = solver.depth
        erf_vals = 0.5 * (1.0 + np.array([erf((z - thickness) / sigma) for z in depth]))

        # Substrate ratios (sum of substrate concentrations at deepest point)
        if solver.sub_idx:
            D_sub_deep = D[-1, solver.sub_idx]
            sub_sum = D_sub_deep.sum()
            if sub_sum > 1e-10:
                sub_ratios = D_sub_deep / sub_sum * erf_vals[-1]
                # Actually, sub_ratios = D_sub / erf
                sub_ratios = D_sub_deep  # Direct ratio
            else:
                sub_ratios = np.ones(len(solver.sub_idx)) / len(solver.sub_idx)
        else:
            sub_ratios = np.array([])

        # Invert film_scale: film_conc_x = D_free / film_scale
        substrate_sum = np.zeros(self.n_depth)
        for j, si in enumerate(solver.sub_idx):
            substrate_sum += D[:, si]
        film_scale = np.maximum(1.0 - substrate_sum, solver.config.min_conc)
        film_conc_x = film_conc / film_scale[:, np.newaxis]

        # Pack into x
        x = np.zeros(self.n_params)
        x[:self.film_size] = film_conc_x.ravel()
        x[self.film_size] = thickness
        x[self.film_size + 1] = abruptness
        if self.n_sub > 0:
            x[self.film_size + 2:] = sub_ratios

        return x

    def _forward_normalised(self, x: np.ndarray) -> np.ndarray:
        """Compute normalised forward model X_norm(θ,e) from parameter vector x.

        Returns shape (n_angles, n_elements).
        """
        solver = self._solver
        film_conc, thickness, abruptness, sub_ratios, grad_params = solver._unpack_parameters(x)
        D = solver._build_full_profile(film_conc, thickness, abruptness, sub_ratios, grad_params)

        # Forward model: I(θ, e) = T[:, :, e] @ D[:, e]
        I_pred = np.zeros((self.n_angles, self.n_elements))
        for e in range(self.n_elements):
            I_pred[:, e] = self.tm.trans_mat[:, :, e] @ D[:, e]

        # Substrate-based normalisation
        if solver.sub_idx:
            I_sub_sum = I_pred[:, solver.sub_idx].sum(axis=1)
            I_sub_sum = np.maximum(I_sub_sum, 1e-30)
            X_norm = I_pred / I_sub_sum[:, np.newaxis]
        else:
            I_total = I_pred.sum(axis=1)
            I_total = np.maximum(I_total, 1e-30)
            X_norm = I_pred / I_total[:, np.newaxis]

        return X_norm

    def compute_jacobian(
        self,
        x: np.ndarray,
        eps: float = 1e-7,
    ) -> np.ndarray:
        """Compute Jacobian of normalised forward model by finite differences.

        J[k, j] = ∂X_norm_k / ∂x_j

        where k indexes (angle, element) pairs and j indexes parameters.

        Parameters
        ----------
        x : (n_params,) parameter vector
        eps : float, finite difference step size

        Returns
        -------
        J : (n_obs, n_params) Jacobian matrix
        """
        X0 = self._forward_normalised(x).ravel()  # (n_obs,)
        J = np.zeros((self.n_obs, self.n_params))

        for j in range(self.n_params):
            x_plus = x.copy()
            x_plus[j] += eps

            X_plus = self._forward_normalised(x_plus).ravel()
            J[:, j] = (X_plus - X0) / eps

        return J

    def compute_fisher_matrix(
        self,
        x: np.ndarray,
        jacobian: Optional[np.ndarray] = None,
        noise_model: str = 'poisson',
    ) -> Tuple[np.ndarray, float, np.ndarray, int]:
        """Compute Fisher Information Matrix.

        F = J^T · W · J

        where W = diag(1/σ²) depends on the noise model:
        - 'poisson': σ² = X_norm / n_counts  (variance of normalised intensity)
        - 'gaussian': σ² = σ_fixed²  (uniform noise)

        Parameters
        ----------
        x : parameter vector
        jacobian : pre-computed Jacobian (optional)
        noise_model : 'poisson' or 'gaussian'

        Returns
        -------
        F : (n_params, n_params) Fisher matrix
        condition_number : float
        singular_values : (n_params,)
        effective_rank : int
        """
        if jacobian is None:
            jacobian = self.compute_jacobian(x)

        X_norm = self._forward_normalised(x).ravel()

        if noise_model == 'poisson':
            # Poisson: variance = signal / n_counts
            # Weight = n_counts / signal
            sigma2 = np.maximum(X_norm, 1e-30) / self.n_counts
            W = 1.0 / sigma2
        elif noise_model == 'gaussian':
            # Gaussian: uniform variance ~ 1% of mean signal
            sigma_fixed = 0.01 * np.mean(np.maximum(X_norm, 1e-30))
            W = np.full_like(X_norm, 1.0 / (sigma_fixed ** 2))
        else:
            raise ValueError(f"Unknown noise_model: {noise_model}")

        # Fisher matrix: F = J^T · diag(W) · J
        JtW = jacobian.T * W[np.newaxis, :]  # (n_params, n_obs)
        F = JtW @ jacobian

        # SVD analysis
        U, S, Vt = np.linalg.svd(F, full_matrices=False)
        cond = S[0] / S[-1] if S[-1] > 0 else np.inf

        # Effective rank (singular values above threshold)
        threshold = S[0] * 1e-10
        eff_rank = int(np.sum(S > threshold))

        return F, cond, S, eff_rank

    def compute_crlb(
        self,
        pixel_idx: int = 0,
        thickness: Optional[float] = None,
        abruptness: Optional[float] = None,
        profile: Optional[np.ndarray] = None,
        noise_model: str = 'poisson',
        svd_threshold: float = 1e-10,
    ) -> CRLBResult:
        """Compute CRLB for a given pixel or profile.

        Parameters
        ----------
        pixel_idx : int
            Index into benchmark data Dstk (ground truth).
        thickness, abruptness : float, optional
            Structural parameters. If None, estimated from profile.
        profile : (n_depth, n_elements), optional
            Override profile instead of using Dstk[pixel_idx].
        noise_model : str
            'poisson' or 'gaussian'
        svd_threshold : float
            Relative SVD truncation threshold for pseudo-inverse.

        Returns
        -------
        CRLBResult
        """
        if profile is None:
            profile = self.data.Dstk[pixel_idx]

        # Estimate thickness and abruptness from profile if not given
        if thickness is None or abruptness is None:
            t_est, a_est = self._estimate_structural_params(profile)
            if thickness is None:
                thickness = t_est
            if abruptness is None:
                abruptness = a_est

        # Build parameter vector
        x = self._build_x_from_profile(profile, thickness, abruptness)

        # Compute Jacobian
        J = self.compute_jacobian(x)

        # Compute Fisher matrix
        F, cond, S, eff_rank = self.compute_fisher_matrix(x, J, noise_model)

        # Pseudo-inverse via SVD truncation
        U, S_full, Vt = np.linalg.svd(F, full_matrices=False)
        threshold = S_full[0] * svd_threshold
        S_inv = np.where(S_full > threshold, 1.0 / S_full, 0.0)
        F_inv = (Vt.T * S_inv[np.newaxis, :]) @ U.T

        # CRLB = sqrt(diag(F^{-1}))
        crlb = np.sqrt(np.maximum(np.diag(F_inv), 0.0))

        # Reshape film concentration CRLB
        crlb_film = crlb[:self.film_size].reshape(self.n_depth, self.n_free)
        crlb_thickness = crlb[self.film_size]
        crlb_abruptness = crlb[self.film_size + 1]
        crlb_sub = crlb[self.film_size + 2:self.film_size + 2 + self.n_sub]

        return CRLBResult(
            fisher_matrix=F,
            condition_number=cond,
            singular_values=S,
            effective_rank=eff_rank,
            crlb=crlb,
            crlb_film_conc=crlb_film,
            crlb_thickness=crlb_thickness,
            crlb_abruptness=crlb_abruptness,
            crlb_sub_ratios=crlb_sub,
            profile=profile,
            x_vector=x,
            n_counts=self.n_counts,
            svd_threshold=svd_threshold,
            jacobian=J,
        )

    def compute_crlb_vs_counts(
        self,
        pixel_idx: int = 0,
        count_range: Optional[np.ndarray] = None,
        profile: Optional[np.ndarray] = None,
        thickness: Optional[float] = None,
        abruptness: Optional[float] = None,
    ) -> Dict:
        """Compute CRLB as a function of total count number.

        Verifies the expected 1/sqrt(N) scaling.

        Returns
        -------
        dict with keys:
            'count_range': (n_counts_values,)
            'crlb_film': (n_counts_values, n_depth, n_free)
            'crlb_thickness': (n_counts_values,)
            'crlb_abruptness': (n_counts_values,)
            'mean_crlb_film': (n_counts_values,)
        """
        if count_range is None:
            count_range = np.logspace(3, 8, 12)

        if profile is None:
            profile = self.data.Dstk[pixel_idx]

        if thickness is None or abruptness is None:
            t_est, a_est = self._estimate_structural_params(profile)
            if thickness is None:
                thickness = t_est
            if abruptness is None:
                abruptness = a_est

        x = self._build_x_from_profile(profile, thickness, abruptness)
        J = self.compute_jacobian(x)

        results = {
            'count_range': count_range,
            'crlb_film': [],
            'crlb_thickness': [],
            'crlb_abruptness': [],
            'mean_crlb_film': [],
        }

        for n_c in count_range:
            old_counts = self.n_counts
            self.n_counts = n_c

            F, _, S, _ = self.compute_fisher_matrix(x, J, 'poisson')

            U, S_full, Vt = np.linalg.svd(F, full_matrices=False)
            threshold = S_full[0] * 1e-10
            S_inv = np.where(S_full > threshold, 1.0 / S_full, 0.0)
            F_inv = (Vt.T * S_inv[np.newaxis, :]) @ U.T
            crlb = np.sqrt(np.maximum(np.diag(F_inv), 0.0))

            crlb_film = crlb[:self.film_size].reshape(self.n_depth, self.n_free)
            results['crlb_film'].append(crlb_film)
            results['crlb_thickness'].append(crlb[self.film_size])
            results['crlb_abruptness'].append(crlb[self.film_size + 1])
            results['mean_crlb_film'].append(np.mean(crlb_film))

            self.n_counts = old_counts

        for key in ['crlb_film', 'crlb_thickness', 'crlb_abruptness', 'mean_crlb_film']:
            results[key] = np.array(results[key])

        return results

    def compute_crlb_sxps(
        self,
        pixel_idx: int = 0,
        lambda_ratio: float = 1.0 / 3.0,
        profile: Optional[np.ndarray] = None,
        thickness: Optional[float] = None,
        abruptness: Optional[float] = None,
    ) -> Dict:
        """Compare CRLB under conventional XPS vs SXPS conditions.

        SXPS uses shorter wavelength X-rays → shorter IMFP → better depth resolution.
        This is modelled by scaling the transfer matrix attenuation.

        Parameters
        ----------
        lambda_ratio : float
            SXPS λ / conventional λ ratio (default 1/3).

        Returns
        -------
        dict with keys:
            'crlb_conv': CRLBResult for conventional
            'crlb_sxps': CRLBResult for SXPS
            'improvement_film': (n_depth, n_free) ratio conv/sxps
            'improvement_thickness': float
        """
        # Conventional CRLB
        result_conv = self.compute_crlb(
            pixel_idx=pixel_idx, profile=profile,
            thickness=thickness, abruptness=abruptness,
        )

        # Modify transfer matrix for SXPS (shorter λ)
        tm_sxps = TransferMatrix(
            depth=self.tm.depth,
            step_size=self.tm.step_size,
            angles=self.tm.angles,
            photon_energy=self.tm.photon_energy,
            work_function=self.tm.work_function,
            gamma_eff=self.tm.gamma_eff,
        )
        # Scale attenuation: shorter λ → more attenuation per step
        # trans_mat = exp(cumsum(-step/λ/cosθ))
        # Reducing λ by factor r → exponent scales by 1/r → more attenuation
        # Equivalent: trans_mat_sxps = trans_mat_conv ^ (1/r)
        tm_sxps.trans_mat = np.power(
            np.maximum(self.tm.trans_mat, 1e-30),
            1.0 / lambda_ratio
        )
        tm_sxps.n_elements = self.tm.n_elements

        # Temporarily swap transfer matrix
        old_tm = self.tm
        self.tm = tm_sxps
        self._solver.tm = tm_sxps

        result_sxps = self.compute_crlb(
            pixel_idx=pixel_idx, profile=profile,
            thickness=thickness, abruptness=abruptness,
        )

        # Restore
        self.tm = old_tm
        self._solver.tm = old_tm

        # Compute improvement ratios (clamp to avoid inf)
        sxps_crlb_safe = np.maximum(result_sxps.crlb_film_conc, 1e-20)
        improvement_film = np.where(
            result_sxps.crlb_film_conc > 1e-15,
            result_conv.crlb_film_conc / sxps_crlb_safe,
            np.nan,
        )
        improvement_thickness = (
            result_conv.crlb_thickness / max(result_sxps.crlb_thickness, 1e-30)
        )

        return {
            'crlb_conv': result_conv,
            'crlb_sxps': result_sxps,
            'improvement_film': improvement_film,
            'improvement_thickness': improvement_thickness,
            'lambda_ratio': lambda_ratio,
        }

    def compute_efficiency(
        self,
        rmse_actual: np.ndarray,
        crlb_result: CRLBResult,
    ) -> np.ndarray:
        """Compute statistical efficiency = CRLB / RMSE.

        Efficiency of 1.0 means the estimator achieves the CRLB (optimal).
        Efficiency < 1.0 means there is room for improvement.
        Efficiency > 1.0 should not happen (indicates a calculation error).

        Parameters
        ----------
        rmse_actual : (n_depth, n_free) or (n_free,) actual RMSE from solver
        crlb_result : CRLBResult

        Returns
        -------
        efficiency : same shape as rmse_actual
        """
        crlb = crlb_result.crlb_film_conc
        if rmse_actual.ndim == 1 and crlb.ndim == 2:
            # Per-element average
            crlb = np.mean(crlb, axis=0)

        return np.where(rmse_actual > 1e-30, crlb / rmse_actual, 0.0)

    def _estimate_structural_params(
        self,
        D: np.ndarray,
    ) -> Tuple[float, float]:
        """Estimate thickness and abruptness from a depth profile.

        Uses the substrate profile (erf shape) to estimate parameters.

        Returns
        -------
        thickness : float (Å)
        abruptness : float (Å)
        """
        solver = self._solver
        depth = solver.depth

        if not solver.sub_idx:
            return 40.0, 8.0

        # Sum of substrate concentrations
        D_sub_sum = D[:, solver.sub_idx].sum(axis=1)

        # Find 50% point → thickness
        half_max = 0.5 * D_sub_sum[-1]
        idx_50 = np.argmin(np.abs(D_sub_sum - half_max))
        thickness = depth[idx_50]

        # Estimate abruptness from slope at interface
        if 0 < idx_50 < len(depth) - 1:
            slope = np.gradient(D_sub_sum, depth)
            max_slope = np.max(np.abs(slope))
            if max_slope > 1e-10:
                # For erf: max slope = D_max / (sigma * sqrt(pi))
                D_max = D_sub_sum[-1]
                sigma = D_max / (max_slope * np.sqrt(np.pi))
                _2sqrt_ln2 = 2.0 * np.sqrt(np.log(2.0))
                abruptness = sigma * _2sqrt_ln2
            else:
                abruptness = 8.0
        else:
            abruptness = 8.0

        # Clamp to reasonable range
        thickness = np.clip(thickness, 10.0, 500.0)
        abruptness = np.clip(abruptness, 1.0, 50.0)

        return float(thickness), float(abruptness)

    def compute_crlb_multi_pixel(
        self,
        pixel_indices: Optional[List[int]] = None,
        n_pixels: int = 100,
    ) -> Dict:
        """Compute CRLB statistics over multiple pixels.

        Returns
        -------
        dict with per-element statistics
        """
        if pixel_indices is None:
            n_total = self.data.n_pixels
            step = max(1, n_total // n_pixels)
            pixel_indices = list(range(0, n_total, step))[:n_pixels]

        all_crlb_film = []
        all_crlb_thickness = []
        all_crlb_abruptness = []
        all_cond = []

        for px in pixel_indices:
            try:
                result = self.compute_crlb(pixel_idx=px)
                all_crlb_film.append(result.crlb_film_conc)
                all_crlb_thickness.append(result.crlb_thickness)
                all_crlb_abruptness.append(result.crlb_abruptness)
                all_cond.append(result.condition_number)
            except Exception:
                continue

        return {
            'pixel_indices': pixel_indices,
            'crlb_film_mean': np.mean(all_crlb_film, axis=0),
            'crlb_film_std': np.std(all_crlb_film, axis=0),
            'crlb_thickness_mean': np.mean(all_crlb_thickness),
            'crlb_abruptness_mean': np.mean(all_crlb_abruptness),
            'condition_number_median': np.median(all_cond),
            'n_computed': len(all_crlb_film),
        }

    def svd_threshold_sensitivity(
        self,
        pixel_idx: int = 0,
        thresholds: Optional[np.ndarray] = None,
    ) -> Dict:
        """Test sensitivity of CRLB to SVD truncation threshold.

        Returns
        -------
        dict with 'thresholds' and 'mean_crlb_film' arrays
        """
        if thresholds is None:
            thresholds = np.array([1e-14, 1e-12, 1e-10, 1e-8, 1e-6])

        profile = self.data.Dstk[pixel_idx]
        t, a = self._estimate_structural_params(profile)
        x = self._build_x_from_profile(profile, t, a)
        J = self.compute_jacobian(x)
        F, _, _, _ = self.compute_fisher_matrix(x, J)

        U, S_full, Vt = np.linalg.svd(F, full_matrices=False)

        results = {'thresholds': thresholds, 'mean_crlb_film': []}

        for th in thresholds:
            cutoff = S_full[0] * th
            S_inv = np.where(S_full > cutoff, 1.0 / S_full, 0.0)
            F_inv = (Vt.T * S_inv[np.newaxis, :]) @ U.T
            crlb = np.sqrt(np.maximum(np.diag(F_inv), 0.0))
            crlb_film = crlb[:self.film_size].reshape(self.n_depth, self.n_free)
            results['mean_crlb_film'].append(np.mean(crlb_film))

        results['mean_crlb_film'] = np.array(results['mean_crlb_film'])
        return results
