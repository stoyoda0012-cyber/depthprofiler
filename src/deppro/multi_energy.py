"""
Multi-Energy CRLB Analysis for ARXPS Depth Profiling
====================================================

Reusable module for multi-energy Fisher information analysis.
Extracts the ad-hoc patterns from plot_fisher_fundamentals.py and
plot_nonuniform_grid.py into a composable API.

Fisher matrices of independent measurements are additive:
F_combined = F_conv + F_sxps.

Usage:
    from deppro.multi_energy import MultiEnergyCRLB, EnergyConfig
    from deppro.io.dstk import load_dstk_data, create_tm_from_h5
    from deppro.fisher_information import FisherInformation

    data = load_dstk_data(h5_path)
    tm = create_tm_from_h5(data)
    fi = FisherInformation(tm, data, n_counts=1e6)
    me = MultiEnergyCRLB(fi)

    # Single energy
    result = me.analyze_single(EnergyConfig("Conv", 1.0, 1e6))

    # Combined Fisher
    result = me.analyze_combination([
        EnergyConfig("Conv", 1.0, 1e6),
        EnergyConfig("SXPS(1/2)", 2.0, 1e6),
    ])

    # Lambda ratio sweep
    sweep = me.sweep_lambda_ratios([1, 1.5, 2, 3, 5, 10])
"""

import numpy as np
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple, Dict

from .fisher_information import FisherInformation
from .transfer_matrix import TransferMatrix


# =========================================================================
# Photon source presets (from SESSAAnalyzer/database/bulk_7sources)
# =========================================================================

# 7 typical X-ray sources spanning soft X-ray to HAXPES range
PHOTON_SOURCES = {
    "SR_620":  {"energy_eV":  620.0, "label": "SR 620eV",   "color": "red"},
    "SR_800":  {"energy_eV":  800.0, "label": "SR 800eV",   "color": "orange"},
    "Mg_Ka":   {"energy_eV": 1253.6, "label": "Mg Kα",      "color": "gold"},
    "Al_Ka":   {"energy_eV": 1486.6, "label": "Al Kα",      "color": "green"},
    "Ag_La":   {"energy_eV": 2984.2, "label": "Ag Lα",      "color": "blue"},
    "Cr_Ka":   {"energy_eV": 5414.7, "label": "Cr Kα",      "color": "indigo"},
    "Ga_Ka":   {"energy_eV": 9251.7, "label": "Ga Kα",      "color": "violet"},
}

# Curated presets for common experimental configurations
ENERGY_PRESETS = {
    "7sources": {
        "description": "All 7 X-ray sources (SESSAAnalyzer database)",
        "sources": list(PHOTON_SOURCES.keys()),
    },
    "lab": {
        "description": "Common lab sources: Al Kα, Ag Lα, Cr Kα, Ga Kα",
        "sources": ["Al_Ka", "Ag_La", "Cr_Ka", "Ga_Ka"],
    },
    "haxpes": {
        "description": "HAXPES sources: Ag Lα, Cr Kα, Ga Kα",
        "sources": ["Ag_La", "Cr_Ka", "Ga_Ka"],
    },
    "dual_anode": {
        "description": "Typical dual-anode: Al Kα + Cr Kα (ULVAC style)",
        "sources": ["Al_Ka", "Cr_Ka"],
    },
    "soft_hard": {
        "description": "Soft + Hard: SR 620eV + Ga Kα (max contrast)",
        "sources": ["SR_620", "Ga_Ka"],
    },
}


def make_energies_from_preset(preset_name: str, n_counts: float = 1e6) -> 'List[EnergyConfig]':
    """Create EnergyConfig list from a named preset.

    Parameters
    ----------
    preset_name : str
        Key in ENERGY_PRESETS or a single source key in PHOTON_SOURCES.
    n_counts : float
        Counts per energy.

    Returns
    -------
    list of EnergyConfig
    """
    if preset_name in ENERGY_PRESETS:
        source_keys = ENERGY_PRESETS[preset_name]["sources"]
    elif preset_name in PHOTON_SOURCES:
        source_keys = [preset_name]
    else:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(ENERGY_PRESETS.keys())} or {list(PHOTON_SOURCES.keys())}"
        )

    energies = []
    for key in source_keys:
        src = PHOTON_SOURCES[key]
        energies.append(EnergyConfig(
            label=src["label"],
            lambda_ratio=0.0,
            n_counts=n_counts,
            photon_energy_eV=src["energy_eV"],
        ))
    return energies


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class EnergyConfig:
    """Configuration for a single photon energy measurement.

    Parameters
    ----------
    label : str
        Display name, e.g. "Conv", "SXPS(1/2)", "Ga-Ka 9.25keV".
    lambda_ratio : float
        lambda_conv / lambda_this. Convention:
          1.0 = conventional (same IMFP as reference H5)
          2.0 = SXPS with half the IMFP
          3.0 = SXPS with 1/3 the IMFP
        NOTE: this is the "magnification" convention (higher =
        shorter lambda), the inverse of an earlier convention where
        lambda_ratio=1/3 meant SXPS.
    n_counts : float
        Total counts allocated to this energy.
    photon_energy_eV : float, optional
        If set, use full IMFP recalculation via TPP-2M instead of
        power-law trans_mat scaling. lambda_ratio is ignored.
    """
    label: str
    lambda_ratio: float
    n_counts: float
    photon_energy_eV: Optional[float] = None


@dataclass
class MultiEnergyResult:
    """Result from multi-energy CRLB analysis."""
    # Configuration
    energies: List[EnergyConfig]

    # Per-energy Fisher matrices and derived quantities
    fisher_matrices: List[np.ndarray]       # each (n_params, n_params)
    per_energy_ranks: List[int]
    per_energy_crlbs: List[float]           # mean film CRLB (%)

    # Combined results
    combined_fisher: np.ndarray             # (n_params, n_params)
    combined_rank: int
    combined_crlb_film: np.ndarray          # (n_depth, n_free)
    combined_crlb_per_element: np.ndarray   # (n_free,) mean over depths
    mean_crlb_film_pct: float               # scalar, mean over film

    # Improvement
    reference_crlb_pct: float               # Conv-only CRLB for comparison
    improvement_factor: float               # reference / combined

    # Metadata
    element_names: List[str]
    film_mask: np.ndarray                   # (n_depth,) bool


# =========================================================================
# MultiEnergyCRLB class
# =========================================================================

class MultiEnergyCRLB:
    """Multi-energy CRLB analysis tool.

    Wraps a FisherInformation instance and provides composable primitives
    for multi-energy Fisher analysis via Fisher matrix additivity.
    """

    def __init__(
        self,
        fi: FisherInformation,
        pixel_idx: int = 0,
    ):
        """
        Parameters
        ----------
        fi : FisherInformation
            Pre-configured single-energy Fisher calculator.
        pixel_idx : int
            Reference pixel for ground-truth profile.
        """
        self.fi = fi
        self.pixel_idx = pixel_idx

        # Cache the profile and structural parameters
        self._D = fi.data.Dstk[pixel_idx]
        self._t_est, self._a_est = fi._estimate_structural_params(self._D)
        self._x = fi._build_x_from_profile(self._D, self._t_est, self._a_est)

        # Pre-compute conventional Jacobian and X_norm (reused often)
        self._J_conv = fi.compute_jacobian(self._x)
        self._X_conv = fi._forward_normalised(self._x).ravel()

        # Dimensions
        self.film_size = fi.film_size
        self.n_depth = fi.n_depth
        self.n_free = fi.n_free
        self.n_params = fi.n_params

        # Film mask: depths where free elements have significant concentration
        free_idx = fi._solver.free_idx
        self.film_mask = np.any(self._D[:, free_idx] > 0.001, axis=1)
        self.element_names = [fi.data.element_names[i] for i in free_idx]

    # -----------------------------------------------------------------
    # Core primitives
    # -----------------------------------------------------------------

    def compute_jacobian_at_energy(
        self,
        lambda_ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jacobian and X_norm for a given lambda ratio.

        Uses power-law trans_mat scaling: T_sxps = T_conv^(1/r)
        where r is the ratio lambda_sxps/lambda_conv (the earlier convention).
        We convert from our convention: r_internal = 1/lambda_ratio.

        Parameters
        ----------
        lambda_ratio : float
            lambda_conv / lambda_this.  1.0 = conventional.

        Returns
        -------
        J : (n_obs, n_params) Jacobian
        X_norm : (n_obs,) normalized forward model
        """
        if abs(lambda_ratio - 1.0) < 1e-10:
            return self._J_conv.copy(), self._X_conv.copy()

        fi = self.fi
        tm = fi.tm

        # Build scaled transfer matrix
        tm_s = TransferMatrix(
            depth=tm.depth, step_size=tm.step_size, angles=tm.angles,
            photon_energy=tm.photon_energy, work_function=tm.work_function,
            gamma_eff=tm.gamma_eff,
        )
        # Power scaling: T^(lambda_ratio) for shorter lambda
        tm_s.trans_mat = np.power(
            np.maximum(tm.trans_mat, 1e-30),
            lambda_ratio,
        )
        tm_s.n_elements = tm.n_elements

        # Temporarily swap transfer matrix
        old_tm = fi.tm
        fi.tm = tm_s
        fi._solver.tm = tm_s
        try:
            J = fi.compute_jacobian(self._x)
            X = fi._forward_normalised(self._x).ravel()
        finally:
            fi.tm = old_tm
            fi._solver.tm = old_tm

        return J, X

    def compute_jacobian_at_photon_energy(
        self,
        photon_energy_eV: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jacobian at a specific photon energy via full IMFP recalculation.

        Creates a new TransferMatrix with the given photon energy and
        recalculates lambda_basic, lambda_obs, and trans_mat from scratch
        using TPP-2M IMFP.

        Parameters
        ----------
        photon_energy_eV : float
            Photon energy in eV.

        Returns
        -------
        J : (n_obs, n_params) Jacobian
        X_norm : (n_obs,) normalized forward model
        """
        fi = self.fi
        old_tm = fi.tm

        # Build new transfer matrix with different photon energy
        tm_new = TransferMatrix(
            depth=old_tm.depth,
            step_size=old_tm.step_size,
            angles=old_tm.angles,
            photon_energy=photon_energy_eV,
            work_function=old_tm.work_function,
            gamma_eff=old_tm.gamma_eff,
        )
        # Full compute: IMFP -> lambda_obs -> trans_mat
        tm_new.compute(self._D, fi.data.elements)

        # Validate: NaN/Inf in trans_mat means photon energy is too low
        # for some elements (KE < 0 → invalid IMFP)
        if not np.all(np.isfinite(tm_new.trans_mat)):
            import warnings
            warnings.warn(
                f"TransferMatrix at {photon_energy_eV:.1f} eV contains NaN/Inf "
                f"(photon energy too low for some elements). "
                f"Replacing invalid entries with 0."
            )
            tm_new.trans_mat = np.nan_to_num(tm_new.trans_mat, nan=0.0, posinf=0.0, neginf=0.0)

        # Swap and compute
        fi.tm = tm_new
        fi._solver.tm = tm_new
        try:
            J = fi.compute_jacobian(self._x)
            X = fi._forward_normalised(self._x).ravel()
        finally:
            fi.tm = old_tm
            fi._solver.tm = old_tm

        return J, X

    @staticmethod
    def build_fisher_matrix(
        J: np.ndarray,
        X_norm: np.ndarray,
        n_counts: float,
    ) -> np.ndarray:
        """Build Fisher matrix: F = J^T * diag(N/X) * J

        Parameters
        ----------
        J : (n_obs, n_params)
        X_norm : (n_obs,)
        n_counts : float

        Returns
        -------
        F : (n_params, n_params) symmetric positive semi-definite
        """
        W = n_counts / np.maximum(X_norm, 1e-30)
        return (J.T * W[np.newaxis, :]) @ J

    @staticmethod
    def extract_crlb(
        F: np.ndarray,
        film_size: int,
        n_depth: int,
        n_free: int,
        svd_threshold: float = 1e-10,
        film_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, np.ndarray, float, np.ndarray]:
        """Extract rank and CRLB from Fisher matrix.

        Parameters
        ----------
        F : (n_params, n_params) Fisher matrix
        film_size : int, size of film concentration block
        n_depth, n_free : int, profile dimensions
        svd_threshold : float, relative SVD truncation
        film_mask : (n_depth,) bool, optional, depths in film region

        Returns
        -------
        rank : int, effective rank
        crlb_film : (n_depth, n_free), film concentration CRLB
        mean_crlb_pct : float, mean film CRLB in percent
        singular_values : (n_params,)
        """
        # Sanitise Fisher matrix (NaN/Inf from invalid eV paths)
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

        U, S, Vt = np.linalg.svd(F, full_matrices=False)
        threshold = S[0] * svd_threshold if S[0] > 0 else svd_threshold
        rank = int(np.sum(S > threshold))

        S_inv = np.where(S > threshold, 1.0 / S, 0.0)
        F_inv = (Vt.T * S_inv[np.newaxis, :]) @ U.T
        crlb = np.sqrt(np.maximum(np.diag(F_inv), 0.0))

        crlb_film = crlb[:film_size].reshape(n_depth, n_free)

        if film_mask is not None and film_mask.any():
            mean_crlb_pct = np.mean(crlb_film[film_mask, :]) * 100
        else:
            mean_crlb_pct = np.mean(crlb_film) * 100

        return rank, crlb_film, mean_crlb_pct, S

    # -----------------------------------------------------------------
    # High-level analysis
    # -----------------------------------------------------------------

    def _get_jacobian_and_xnorm(
        self, energy: EnergyConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get Jacobian and X_norm for an EnergyConfig."""
        if energy.photon_energy_eV is not None:
            return self.compute_jacobian_at_photon_energy(energy.photon_energy_eV)
        else:
            return self.compute_jacobian_at_energy(energy.lambda_ratio)

    def analyze_single(self, energy: EnergyConfig) -> dict:
        """Analyze a single energy configuration.

        Returns
        -------
        dict with keys: F, J, X_norm, rank, crlb_film, crlb_pct, crlb_per_element
        """
        J, X = self._get_jacobian_and_xnorm(energy)
        F = self.build_fisher_matrix(J, X, energy.n_counts)
        rank, crlb_film, mean_pct, sv = self.extract_crlb(
            F, self.film_size, self.n_depth, self.n_free,
            film_mask=self.film_mask,
        )

        # Per-element CRLB (mean over film depths)
        if self.film_mask.any():
            crlb_per_elem = np.mean(crlb_film[self.film_mask, :], axis=0) * 100
        else:
            crlb_per_elem = np.mean(crlb_film, axis=0) * 100

        return {
            'F': F, 'J': J, 'X_norm': X,
            'rank': rank,
            'crlb_film': crlb_film,
            'crlb_pct': mean_pct,
            'crlb_per_element': crlb_per_elem,
            'singular_values': sv,
        }

    def analyze_combination(
        self,
        energies: List[EnergyConfig],
    ) -> MultiEnergyResult:
        """Analyze a combination of energies via Fisher additivity.

        F_combined = sum(F_i) for independent measurements at each energy.

        Parameters
        ----------
        energies : list of EnergyConfig
            One or more energy configurations.

        Returns
        -------
        MultiEnergyResult
        """
        fisher_matrices = []
        per_ranks = []
        per_crlbs = []

        for energy in energies:
            r = self.analyze_single(energy)
            fisher_matrices.append(r['F'])
            per_ranks.append(r['rank'])
            per_crlbs.append(r['crlb_pct'])

        # Combine Fisher matrices
        F_combined = sum(fisher_matrices)
        rank, crlb_film, mean_pct, _ = self.extract_crlb(
            F_combined, self.film_size, self.n_depth, self.n_free,
            film_mask=self.film_mask,
        )

        # Per-element combined CRLB
        if self.film_mask.any():
            crlb_per_elem = np.mean(crlb_film[self.film_mask, :], axis=0) * 100
        else:
            crlb_per_elem = np.mean(crlb_film, axis=0) * 100

        # Reference: Conv-only (lambda_ratio=1.0)
        conv_energies = [e for e in energies if abs(e.lambda_ratio - 1.0) < 1e-10
                         and e.photon_energy_eV is None]
        if conv_energies:
            ref_pct = per_crlbs[energies.index(conv_energies[0])]
        else:
            # No explicit Conv in the list; compute it
            ref = self.analyze_single(EnergyConfig("Conv_ref", 1.0, energies[0].n_counts))
            ref_pct = ref['crlb_pct']

        improvement = ref_pct / mean_pct if mean_pct > 0 else np.inf

        return MultiEnergyResult(
            energies=energies,
            fisher_matrices=fisher_matrices,
            per_energy_ranks=per_ranks,
            per_energy_crlbs=per_crlbs,
            combined_fisher=F_combined,
            combined_rank=rank,
            combined_crlb_film=crlb_film,
            combined_crlb_per_element=crlb_per_elem,
            mean_crlb_film_pct=mean_pct,
            reference_crlb_pct=ref_pct,
            improvement_factor=improvement,
            element_names=self.element_names,
            film_mask=self.film_mask,
        )

    # -----------------------------------------------------------------
    # Sweep methods
    # -----------------------------------------------------------------

    def sweep_lambda_ratios(
        self,
        ratios: List[float],
        n_counts: float = 1e6,
    ) -> dict:
        """Sweep lambda ratios and compute rank + CRLB for each.

        Parameters
        ----------
        ratios : list of float
            Lambda ratio values (1.0 = conventional).
        n_counts : float
            Counts per energy.

        Returns
        -------
        dict with keys: ratios, ranks, crlb_pcts
        """
        ranks = []
        crlbs = []
        for r in ratios:
            energy = EnergyConfig(f"r={r:.2f}", r, n_counts)
            result = self.analyze_single(energy)
            ranks.append(result['rank'])
            crlbs.append(result['crlb_pct'])
        return {
            'ratios': np.array(ratios),
            'ranks': np.array(ranks),
            'crlb_pcts': np.array(crlbs),
        }

    def sweep_count_allocation(
        self,
        ratio_a: float,
        ratio_b: float,
        total_counts: float,
        n_steps: int = 21,
    ) -> dict:
        """Sweep count allocation between two energies.

        Varies fraction f from 0 to 1: energy_a gets f*N, energy_b gets (1-f)*N.

        Parameters
        ----------
        ratio_a, ratio_b : float
            Lambda ratios for the two energies.
        total_counts : float
            Total counts to distribute.
        n_steps : int
            Number of allocation steps.

        Returns
        -------
        dict with keys: fractions, crlb_pcts, ranks, optimal_fraction, optimal_crlb
        """
        fractions = np.linspace(0, 1, n_steps)
        crlbs = []
        ranks = []

        # Pre-compute Jacobians once
        J_a, X_a = self.compute_jacobian_at_energy(ratio_a)
        J_b, X_b = self.compute_jacobian_at_energy(ratio_b)

        for f in fractions:
            n_a = total_counts * f
            n_b = total_counts * (1 - f)
            F = np.zeros((self.n_params, self.n_params))
            if n_a > 0:
                F += self.build_fisher_matrix(J_a, X_a, n_a)
            if n_b > 0:
                F += self.build_fisher_matrix(J_b, X_b, n_b)
            rank, _, mean_pct, _ = self.extract_crlb(
                F, self.film_size, self.n_depth, self.n_free,
                film_mask=self.film_mask,
            )
            crlbs.append(mean_pct)
            ranks.append(rank)

        crlbs = np.array(crlbs)
        ranks = np.array(ranks)
        best_idx = np.argmin(crlbs)

        return {
            'fractions': fractions,
            'crlb_pcts': crlbs,
            'ranks': ranks,
            'optimal_fraction': fractions[best_idx],
            'optimal_crlb': crlbs[best_idx],
            'ratio_a': ratio_a,
            'ratio_b': ratio_b,
            'total_counts': total_counts,
        }

    def optimize_energy_set(
        self,
        candidate_ratios: List[float],
        n_energies: int = 2,
        total_counts: float = 2e6,
    ) -> dict:
        """Find optimal energy combination from candidates.

        Exhaustive search over all n_energies-combinations with equal
        count allocation.

        Parameters
        ----------
        candidate_ratios : list of float
            Candidate lambda ratios to choose from.
        n_energies : int
            How many energies to combine.
        total_counts : float
            Total counts distributed equally.

        Returns
        -------
        dict with keys: best_combo, best_crlb, best_rank, all_results
        """
        # Pre-compute all Jacobians
        jac_cache = {}
        for r in candidate_ratios:
            jac_cache[r] = self.compute_jacobian_at_energy(r)

        counts_per = total_counts / n_energies
        all_results = []

        for combo in combinations(candidate_ratios, n_energies):
            F = np.zeros((self.n_params, self.n_params))
            for r in combo:
                J, X = jac_cache[r]
                F += self.build_fisher_matrix(J, X, counts_per)
            rank, _, mean_pct, _ = self.extract_crlb(
                F, self.film_size, self.n_depth, self.n_free,
                film_mask=self.film_mask,
            )
            all_results.append({
                'combo': combo,
                'rank': rank,
                'crlb_pct': mean_pct,
            })

        all_results.sort(key=lambda x: x['crlb_pct'])
        best = all_results[0]

        return {
            'best_combo': best['combo'],
            'best_rank': best['rank'],
            'best_crlb': best['crlb_pct'],
            'all_results': all_results,
            'n_energies': n_energies,
            'total_counts': total_counts,
        }

    def sweep_n_angles(
        self,
        n_angles_range: List[int],
        lambda_ratio: float = 1.0,
        n_counts: float = 1e6,
    ) -> dict:
        """Sweep number of angles for a given energy.

        Selects evenly-spaced subsets of available angles.

        Parameters
        ----------
        n_angles_range : list of int
        lambda_ratio : float
        n_counts : float

        Returns
        -------
        dict with keys: n_angles, ranks, crlb_pcts
        """
        fi = self.fi
        tm = fi.tm
        total_angles = len(tm.angles)

        ranks = []
        crlbs = []

        for n_a in n_angles_range:
            if n_a >= total_angles:
                indices = np.arange(total_angles)
            else:
                indices = np.unique(
                    np.round(np.linspace(0, total_angles - 1, n_a)).astype(int)
                )

            # Build sub-angle transfer matrix
            tm_sub = TransferMatrix(
                depth=tm.depth, step_size=tm.step_size,
                angles=tm.angles[indices],
                photon_energy=tm.photon_energy,
                work_function=tm.work_function,
                gamma_eff=tm.gamma_eff,
            )

            # Apply lambda scaling if needed
            if abs(lambda_ratio - 1.0) < 1e-10:
                tm_sub.trans_mat = tm.trans_mat[indices, :, :]
            else:
                tm_sub.trans_mat = np.power(
                    np.maximum(tm.trans_mat[indices, :, :], 1e-30),
                    lambda_ratio,
                )
            tm_sub.n_elements = tm.n_elements

            # Create sub-FisherInformation
            fi_sub = FisherInformation(tm_sub, fi.data, n_counts=n_counts)
            fi_sub.n_angles = len(indices)
            fi_sub.n_obs = len(indices) * fi_sub.n_elements

            x_sub = fi_sub._build_x_from_profile(self._D, self._t_est, self._a_est)
            J_sub = fi_sub.compute_jacobian(x_sub)
            X_sub = fi_sub._forward_normalised(x_sub).ravel()
            F_sub = self.build_fisher_matrix(J_sub, X_sub, n_counts)

            rank, _, mean_pct, _ = self.extract_crlb(
                F_sub, self.film_size, self.n_depth, self.n_free,
                film_mask=self.film_mask,
            )
            ranks.append(rank)
            crlbs.append(mean_pct)

        return {
            'n_angles': np.array(n_angles_range),
            'ranks': np.array(ranks),
            'crlb_pcts': np.array(crlbs),
        }
