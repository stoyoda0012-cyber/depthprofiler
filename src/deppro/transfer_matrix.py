"""
Transfer Matrix Calculation for Depth Profiling
================================================

Implements the transformation from depth profiles to angle-resolved
intensities using the Beer-Lambert law.

The transformation matrix converts depth profiles D(z) to angle-resolved
intensities I(θ) using:

    I(θ, e) = TransMat(θ, :, e) @ D(:, e)

where TransMat is computed from the escape depth and observation angles.

Usage:
    from deppro.transfer_matrix import TransferMatrix

    tm = TransferMatrix(
        depth=profile.depth,
        step_size=1.0,
        angles=np.array([0, 30, 45, 60, 75]),
        photon_energy=1486.6
    )

    # Set escape depth and compute matrix
    tm.compute(profile.concentrations, compound_db)

    # Apply forward model
    intensities = tm.forward(profile.concentrations)
"""

import numpy as np
from dataclasses import dataclass, field
from math import exp as _math_exp, cos as _math_cos
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .imfp import IMFP, CompoundParams, CompoundDatabase
from .depth_profile import DepthProfile, ElementConfig

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# =========================================================================
# Numba-accelerated kernels for TransferMatrix
# =========================================================================

@jit(nopython=True, cache=True, fastmath=True)
def _calculate_lambda_obs_numba(
    D: np.ndarray,
    lambda_basic: np.ndarray,
    gamma_eff: float,
) -> np.ndarray:
    """Numba kernel for effective escape depth calculation.

    λ_obs(d, e) = γ_eff / Σ_i[D(d, i) / λ_basic(e, i)]

    Parameters
    ----------
    D : (n_depth, n_elem) float64
    lambda_basic : (n_elem, n_elem) float64
    gamma_eff : float

    Returns
    -------
    lambda_obs : (n_depth, n_elem) float64
    """
    n_depth, n_elem = D.shape
    lambda_obs = np.empty((n_depth, n_elem))

    for e in range(n_elem):
        for d in range(n_depth):
            s = 0.0
            for i in range(n_elem):
                s += D[d, i] / lambda_basic[e, i]
            if s == 0.0:
                s = 1e-10
            lambda_obs[d, e] = gamma_eff / s

    return lambda_obs


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_trans_mat_numba(
    lambda_obs: np.ndarray,
    angles_rad: np.ndarray,
    step_size: float,
) -> np.ndarray:
    """Numba kernel for transformation matrix (uniform grid).

    Uses cumsum trick: cumprod(exp(t1)) / exp(t1) = exp(cumsum(t1) - t1)
    This replaces n_depth individual exp() calls with a single cumsum + exp.

    Parameters
    ----------
    lambda_obs : (n_depth, n_elem) float64
    angles_rad : (n_angles,) float64  — angles already in radians
    step_size : float

    Returns
    -------
    trans_mat : (n_angles, n_depth, n_elem) float64
    """
    n_depth, n_elem = lambda_obs.shape
    n_angles = len(angles_rad)
    trans_mat = np.empty((n_angles, n_depth, n_elem))

    for a in range(n_angles):
        cos_theta = _math_cos(angles_rad[a])

        for e in range(n_elem):
            # cumsum of t1 = -step_size / lambda_obs / cos_theta
            cumsum_val = 0.0
            for d in range(n_depth):
                t1 = -step_size / lambda_obs[d, e] / cos_theta
                # trans_mat = exp(cumsum(t1) - t1)
                # i.e., cumulative attenuation up to but not including depth d
                trans_mat[a, d, e] = _math_exp(cumsum_val)
                cumsum_val += t1

    return trans_mat


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_trans_mat_nonuniform_numba(
    lambda_obs: np.ndarray,
    angles_rad: np.ndarray,
    step_sizes: np.ndarray,
) -> np.ndarray:
    """Numba kernel for transformation matrix (non-uniform grid).

    Same as _calculate_trans_mat_numba but with per-depth step sizes.

    Parameters
    ----------
    lambda_obs : (n_depth, n_elem) float64
    angles_rad : (n_angles,) float64
    step_sizes : (n_depth,) float64 — per-depth step sizes

    Returns
    -------
    trans_mat : (n_angles, n_depth, n_elem) float64
    """
    n_depth, n_elem = lambda_obs.shape
    n_angles = len(angles_rad)
    trans_mat = np.empty((n_angles, n_depth, n_elem))

    for a in range(n_angles):
        cos_theta = _math_cos(angles_rad[a])

        for e in range(n_elem):
            cumsum_val = 0.0
            for d in range(n_depth):
                t1 = -step_sizes[d] / lambda_obs[d, e] / cos_theta
                trans_mat[a, d, e] = _math_exp(cumsum_val)
                cumsum_val += t1

    return trans_mat


@jit(nopython=True, cache=True, fastmath=True)
def _forward_numba(
    trans_mat: np.ndarray,
    D: np.ndarray,
    normalize: bool,
) -> np.ndarray:
    """Numba kernel for forward model: I(θ, e) = T(θ, :, e) · D(:, e).

    Parameters
    ----------
    trans_mat : (n_angles, n_depth, n_elem) float64
    D : (n_depth, n_elem) float64
    normalize : bool

    Returns
    -------
    intensities : (n_angles, n_elem) float64
    """
    n_angles, n_depth, n_elem = trans_mat.shape
    intensities = np.zeros((n_angles, n_elem))

    for e in range(n_elem):
        for a in range(n_angles):
            s = 0.0
            for d in range(n_depth):
                s += trans_mat[a, d, e] * D[d, e]
            intensities[a, e] = s

    if normalize:
        for a in range(n_angles):
            total = 0.0
            for e in range(n_elem):
                total += intensities[a, e]
            if total > 0.0:
                for e in range(n_elem):
                    intensities[a, e] /= total

    return intensities


@dataclass
class TransferMatrixParams:
    """Parameters for transfer matrix calculation."""
    gamma_eff: float = 1.0          # Effective gamma factor
    photon_energy: float = 1486.6   # X-ray photon energy (eV, Al Kα)
    work_function: float = 4.5      # Work function (eV)


class TransferMatrix:
    """
    Transfer matrix for depth profile to angle-resolved intensity conversion.

    This implements the Beer-Lambert attenuation model for XPS:

        TransMat(θ, d, e) = cumprod(exp(-Δz / λ(d,e) / cos(θ)))

    where:
        - θ is the observation angle
        - d is the depth index
        - e is the element index
        - λ(d,e) is the escape depth at depth d for element e
        - Δz is the depth step size
    """

    def __init__(
        self,
        depth: np.ndarray,
        step_size: float,
        angles: np.ndarray,
        photon_energy: float = 1486.6,
        work_function: float = 4.5,
        gamma_eff: float = 1.0,
    ):
        """
        Initialize the transfer matrix calculator.

        Args:
            depth: Depth array in Angstroms [nDepth]
            step_size: Depth step size in Angstroms (scalar for uniform grid).
                       For non-uniform grids, this is used as the nominal step
                       size for API compatibility; actual per-depth step sizes
                       are computed from the depth array.
            angles: Observation angles in degrees [nAngles]
            photon_energy: X-ray photon energy (eV)
            work_function: Work function (eV)
            gamma_eff: Effective gamma factor (typically 1.0)
        """
        self.depth = np.asarray(depth, dtype=np.float64)
        self.step_size = step_size
        self.angles = np.asarray(angles)
        self.photon_energy = photon_energy
        self.work_function = work_function
        self.gamma_eff = gamma_eff

        self.n_depth = len(depth)
        self.n_angles = len(angles)

        # Per-depth step sizes for non-uniform grids.
        # step_sizes[0] = depth[0] (distance from surface to first grid point)
        # step_sizes[i] = depth[i] - depth[i-1] for i > 0
        self._step_sizes: Optional[np.ndarray] = None

        # These will be set by compute()
        self.n_elements: int = 0
        self.lambda_basic: Optional[np.ndarray] = None  # [nElement, nElement]
        self.lambda_obs: Optional[np.ndarray] = None    # [nDepth, nElement]
        self.trans_mat: Optional[np.ndarray] = None     # [nAngle, nDepth, nElement]

    @property
    def step_sizes(self) -> np.ndarray:
        """Per-depth step sizes. For uniform grids, all equal to step_size."""
        if self._step_sizes is not None:
            return self._step_sizes
        return np.full(self.n_depth, self.step_size)

    @property
    def is_nonuniform(self) -> bool:
        """True if using a non-uniform depth grid."""
        return self._step_sizes is not None

    @classmethod
    def from_nonuniform_grid(
        cls,
        depth_grid: np.ndarray,
        angles: np.ndarray,
        photon_energy: float = 1486.6,
        work_function: float = 4.5,
        gamma_eff: float = 1.0,
    ) -> 'TransferMatrix':
        """Create TransferMatrix from a non-uniform depth grid.

        Args:
            depth_grid: Non-uniform depth points in Angstroms [nDepth].
                        Must be sorted and positive (first point > 0).
            angles: Observation angles in degrees [nAngles]
            photon_energy: X-ray photon energy (eV)
            work_function: Work function (eV)
            gamma_eff: Effective gamma factor

        Returns:
            TransferMatrix configured for the non-uniform grid
        """
        depth_grid = np.asarray(depth_grid, dtype=np.float64)
        step_sizes = np.empty(len(depth_grid), dtype=np.float64)
        step_sizes[0] = depth_grid[0]
        step_sizes[1:] = np.diff(depth_grid)

        # Nominal step_size = mean step for API compatibility
        nominal_step = float(np.mean(step_sizes))

        tm = cls(
            depth=depth_grid,
            step_size=nominal_step,
            angles=angles,
            photon_energy=photon_energy,
            work_function=work_function,
            gamma_eff=gamma_eff,
        )
        tm._step_sizes = step_sizes
        return tm

    def calculate_lambda_basic(
        self,
        elements: List[ElementConfig],
        compound_db: Optional[CompoundDatabase] = None,
    ) -> np.ndarray:
        """
        Calculate basic IMFP values for all element pairs.

        The basic lambda matrix represents the IMFP of element i measured
        through material j: lambda_basic[i, j] = IMFP(KE_i, material_j)

        Args:
            elements: List of element configurations
            compound_db: Compound database for IMFP calculation

        Returns:
            lambda_basic: [nElement, nElement] IMFP matrix in Angstroms
        """
        if compound_db is None:
            compound_db = CompoundDatabase()

        n_elements = len(elements)
        lambda_basic = np.zeros((n_elements, n_elements))

        for i, elem_i in enumerate(elements):
            # Kinetic energy for element i
            ke_i = self.photon_energy - elem_i.binding_energy - self.work_function

            for j, elem_j in enumerate(elements):
                # IMFP of element i through material j
                compound_j = compound_db.get(elem_j.compound)
                lambda_basic[i, j] = IMFP.from_compound(ke_i, compound_j)

        return lambda_basic

    def calculate_lambda_obs(
        self,
        D: np.ndarray,
        lambda_basic: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate effective escape depth at each depth.

        The effective escape depth is the harmonic average of basic IMFP
        values weighted by the local composition:

            λ_obs(d, e) = γ_eff / Σ_i[D(d, i) / λ_basic(e, i)]

        Args:
            D: Depth profile concentration [nDepth, nElement]
            lambda_basic: Basic IMFP matrix [nElement, nElement]

        Returns:
            lambda_obs: Effective escape depth [nDepth, nElement]
        """
        if NUMBA_AVAILABLE:
            return _calculate_lambda_obs_numba(
                np.ascontiguousarray(D),
                np.ascontiguousarray(lambda_basic),
                self.gamma_eff,
            )

        # Pure NumPy fallback
        n_depth, n_elements = D.shape
        lambda_obs = np.zeros((n_depth, n_elements))

        for e in range(n_elements):
            weighted_sum = np.zeros(n_depth)
            for i in range(n_elements):
                weighted_sum += D[:, i] / lambda_basic[e, i]
            weighted_sum[weighted_sum == 0] = 1e-10
            lambda_obs[:, e] = self.gamma_eff / weighted_sum

        return lambda_obs

    def calculate_trans_mat(
        self,
        lambda_obs: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the transformation matrix.

        The transformation matrix encodes the Beer-Lambert attenuation:

            t1(θ, d, e) = -Δz(d) / λ_obs(d, e) / cos(θ)
            TransMat(θ, d, e) = cumprod(exp(t1), axis=d) / exp(t1)

        Supports both uniform (scalar step_size) and non-uniform grids
        (per-depth step_sizes array).

        Args:
            lambda_obs: Effective escape depth [nDepth, nElement]

        Returns:
            trans_mat: Transformation matrix [nAngle, nDepth, nElement]
        """
        angles_rad = np.deg2rad(self.angles).astype(np.float64)

        if NUMBA_AVAILABLE:
            if self.is_nonuniform:
                return _calculate_trans_mat_nonuniform_numba(
                    np.ascontiguousarray(lambda_obs),
                    angles_rad,
                    np.ascontiguousarray(self.step_sizes),
                )
            else:
                return _calculate_trans_mat_numba(
                    np.ascontiguousarray(lambda_obs),
                    angles_rad,
                    float(self.step_size),
                )

        # Pure NumPy fallback (supports both uniform and non-uniform)
        n_depth, n_elements = lambda_obs.shape
        n_angles = len(self.angles)
        trans_mat = np.zeros((n_angles, n_depth, n_elements))
        sz = self.step_sizes  # (n_depth,)

        for a, theta in enumerate(self.angles):
            cos_theta = np.cos(np.deg2rad(theta))
            for e in range(n_elements):
                t1 = -sz / lambda_obs[:, e] / cos_theta
                exp_t1 = np.exp(t1)
                cumprod_exp = np.cumprod(exp_t1)
                trans_mat[a, :, e] = cumprod_exp / exp_t1

        return trans_mat

    def compute(
        self,
        D: np.ndarray,
        elements: List[ElementConfig],
        compound_db: Optional[CompoundDatabase] = None,
        L_correction: Optional[np.ndarray] = None,
    ) -> 'TransferMatrix':
        """
        Compute the transfer matrix for a given depth profile.

        Args:
            D: Depth profile concentration [nDepth, nElement]
            elements: List of element configurations
            compound_db: Compound database (optional)
            L_correction: Non-dipole angular correction factors
                [nAngles, nElements]. If provided, the transfer matrix
                is multiplied by L so that forward() produces
                I(θ,e) = L(θ,e) × Σ_d T(θ,d,e) × D(d,e).

        Returns:
            self (for chaining)
        """
        if compound_db is None:
            compound_db = CompoundDatabase()

        self.n_elements = len(elements)

        # Step 1: Calculate basic IMFP matrix
        self.lambda_basic = self.calculate_lambda_basic(elements, compound_db)

        # Step 2: Calculate effective escape depth
        self.lambda_obs = self.calculate_lambda_obs(D, self.lambda_basic)

        # Step 3: Calculate transformation matrix
        self.trans_mat = self.calculate_trans_mat(self.lambda_obs)

        # Step 4: Apply non-dipole angular correction (if provided)
        self.L_correction = L_correction
        if L_correction is not None:
            # L_correction: (n_angles, n_elements)
            # trans_mat:    (n_angles, n_depth, n_elements)
            self.trans_mat *= L_correction[:, np.newaxis, :]

        return self

    def forward(
        self,
        D: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Apply forward model: depth profile → angle-resolved intensities.

        I(θ, e) = TransMat(θ, :, e) @ D(:, e)
        X(θ, e) = I(θ, e) / Σ_e'[I(θ, e')]  (if normalize=True)

        Args:
            D: Depth profile concentration [nDepth, nElement]
            normalize: If True, normalize intensities at each angle

        Returns:
            intensities: Angle-resolved intensities [nAngle, nElement]
        """
        if self.trans_mat is None:
            raise RuntimeError("Must call compute() before forward()")

        if NUMBA_AVAILABLE:
            return _forward_numba(
                np.ascontiguousarray(self.trans_mat),
                np.ascontiguousarray(D),
                normalize,
            )

        # Pure NumPy fallback
        n_angles = self.trans_mat.shape[0]
        n_elements = D.shape[1]
        intensities = np.zeros((n_angles, n_elements))

        for e in range(n_elements):
            intensities[:, e] = self.trans_mat[:, :, e] @ D[:, e]

        if normalize:
            total = intensities.sum(axis=1, keepdims=True)
            total[total == 0] = 1
            intensities = intensities / total

        return intensities

    def forward_full(
        self,
        profile: DepthProfile,
        normalize: bool = True,
        recompute: bool = True,
        L_correction: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Full forward model using DepthProfile object.

        Args:
            profile: DepthProfile object
            normalize: If True, normalize intensities
            recompute: If True, recompute transfer matrix
            L_correction: Non-dipole angular correction [nAngles, nElements]

        Returns:
            intensities: [nAngle, nElement]
        """
        if recompute or self.trans_mat is None:
            self.compute(profile.concentrations, profile.elements,
                         L_correction=L_correction)

        return self.forward(profile.concentrations, normalize)

    def get_matrix(self, element_idx: int) -> np.ndarray:
        """
        Get the transfer matrix for a specific element.

        Args:
            element_idx: Element index (0-based)

        Returns:
            Transfer matrix [nAngle, nDepth] for the specified element
        """
        if self.trans_mat is None:
            raise ValueError("Transfer matrix not computed. Call compute() first.")

        return self.trans_mat[:, :, element_idx]


class AngleResolvedData:
    """
    Container for angle-resolved measurement data.

    Stores intensities at multiple angles for multiple elements.
    """

    def __init__(
        self,
        angles: np.ndarray,
        intensities: np.ndarray,
        element_labels: List[str],
        normalized: bool = True,
    ):
        """
        Initialize angle-resolved data.

        Args:
            angles: Observation angles in degrees [nAngles]
            intensities: Intensity data [nAngles, nElements]
            element_labels: List of element labels
            normalized: Whether intensities are normalized
        """
        self.angles = np.asarray(angles)
        self.intensities = np.asarray(intensities)
        self.element_labels = element_labels
        self.normalized = normalized

    @property
    def n_angles(self) -> int:
        return len(self.angles)

    @property
    def n_elements(self) -> int:
        return len(self.element_labels)

    def get_element(self, label: str) -> np.ndarray:
        """Get intensity profile for a specific element."""
        idx = self.element_labels.index(label)
        return self.intensities[:, idx]

    def normalize(self) -> 'AngleResolvedData':
        """Normalize intensities to sum to 1 at each angle."""
        if self.normalized:
            return self

        total = self.intensities.sum(axis=1, keepdims=True)
        total[total == 0] = 1
        return AngleResolvedData(
            angles=self.angles.copy(),
            intensities=self.intensities / total,
            element_labels=self.element_labels.copy(),
            normalized=True,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'angles': self.angles.tolist(),
            'intensities': self.intensities.tolist(),
            'element_labels': self.element_labels,
            'normalized': self.normalized,
        }


def compute_angle_profile(
    profile: DepthProfile,
    angles: np.ndarray,
    photon_energy: float = 1486.6,
    work_function: float = 4.5,
    normalize: bool = True,
) -> AngleResolvedData:
    """
    Convenience function to compute angle-resolved profile from depth profile.

    Args:
        profile: DepthProfile object
        angles: Observation angles in degrees
        photon_energy: X-ray photon energy (eV)
        work_function: Work function (eV)
        normalize: Whether to normalize intensities

    Returns:
        AngleResolvedData with intensities at each angle
    """
    tm = TransferMatrix(
        depth=profile.depth,
        step_size=profile.step_size,
        angles=angles,
        photon_energy=photon_energy,
        work_function=work_function,
    )

    tm.compute(profile.concentrations, profile.elements)
    intensities = tm.forward(profile.concentrations, normalize=normalize)

    return AngleResolvedData(
        angles=angles,
        intensities=intensities,
        element_labels=profile.element_labels,
        normalized=normalize,
    )


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from .depth_profile import create_five_layer_profile

    print("Transfer Matrix Test")
    print("=" * 50)

    # Create test profile
    profile = create_five_layer_profile()
    print(f"Profile: {profile.element_labels}")
    print(f"Depth: {profile.depth[0]:.1f} - {profile.depth[-1]:.1f} Å")

    # Define observation angles
    angles = np.array([0, 15, 30, 45, 60, 75])
    print(f"Angles: {angles}")

    # Compute angle profile
    ar_data = compute_angle_profile(
        profile,
        angles,
        photon_energy=1486.6,  # Al Kα
    )

    print(f"\nAngle-resolved intensities:")
    print(f"Shape: {ar_data.intensities.shape}")

    # Print intensities table
    print(f"\n{'Angle':>6} | " + " | ".join(f"{l:>6}" for l in ar_data.element_labels))
    print("-" * (8 + 9 * len(ar_data.element_labels)))
    for i, theta in enumerate(ar_data.angles):
        vals = " | ".join(f"{v:>6.3f}" for v in ar_data.intensities[i])
        print(f"{theta:>6.0f} | {vals}")

    # Plot angle profile
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Depth profile
    for i, label in enumerate(profile.element_labels):
        ax1.plot(profile.depth, profile.concentrations[:, i], label=label, linewidth=2)
    ax1.set_xlabel('Depth (Å)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('Depth Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)

    # Right: Angle profile
    for i, label in enumerate(ar_data.element_labels):
        ax2.plot(ar_data.angles, ar_data.intensities[:, i], 'o-', label=label, linewidth=2)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Normalized Intensity')
    ax2.set_title('Angle-Resolved Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 75)

    plt.tight_layout()
    plt.savefig('/tmp/transfer_matrix_test.png', dpi=150)
    print(f"\nSaved plot to /tmp/transfer_matrix_test.png")


# =============================================================================
# Condition Number Analysis Utilities
# =============================================================================

def analyze_transfer_matrix_condition(
    tm: TransferMatrix,
    verbose: bool = True,
) -> Dict:
    """
    Analyze the condition number of a transfer matrix.

    Args:
        tm: Transfer matrix to analyze
        verbose: Print analysis results

    Returns:
        Dictionary with condition analysis results
    """
    if tm.trans_mat is None:
        raise ValueError("Transfer matrix not computed. Call tm.compute() first.")

    # Get combined transfer matrix
    T_combined = np.vstack([tm.get_matrix(e) for e in range(tm.n_elements)])

    # Compute SVD
    U, S, Vt = np.linalg.svd(T_combined, full_matrices=False)

    cond = S[0] / S[-1] if S[-1] > 0 else np.inf
    rank = np.sum(S > S[0] * 1e-10)

    # Effective information depth (from largest singular value)
    effective_depth = tm.step_size * rank / tm.n_elements

    results = {
        'condition_number': cond,
        'rank': rank,
        'matrix_shape': T_combined.shape,
        'singular_values': S,
        'effective_rank_ratio': rank / min(T_combined.shape),
        'effective_info_depth': effective_depth,
        'n_angles': tm.n_angles,
        'n_depth': tm.n_depth,
        'n_elements': tm.n_elements,
        'step_size': tm.step_size,
    }

    if verbose:
        print("=" * 50)
        print("Transfer Matrix Condition Analysis")
        print("=" * 50)
        print(f"Matrix shape: {T_combined.shape}")
        print(f"Condition number: {cond:.2e}")
        print(f"Rank: {rank}/{min(T_combined.shape)}")
        print(f"Effective rank ratio: {results['effective_rank_ratio']:.2%}")
        print(f"Effective info depth: {effective_depth:.1f} Å")
        print(f"\nTop 5 singular values: {S[:5]}")
        print(f"Bottom 5 singular values: {S[-5:]}")

        if cond > 1e15:
            print("\n⚠ WARNING: Very high condition number!")
            print("  Consider:")
            print("  - Increasing step_size (coarser depth resolution)")
            print("  - Using regularization (L2 or MEM)")
            print("  - Reducing depth range")

    return results


def suggest_optimal_parameters(
    target_info_depth: float = 100.0,
    angles: Optional[np.ndarray] = None,
    photon_energy: float = 1486.6,
    target_condition: float = 1e10,
) -> Dict:
    """
    Suggest optimal transfer matrix parameters for a given target.

    Args:
        target_info_depth: Target information depth in Angstroms
        angles: Observation angles (default: 0-75° in 9 steps)
        photon_energy: X-ray photon energy (eV)
        target_condition: Target condition number

    Returns:
        Dictionary with suggested parameters
    """
    if angles is None:
        angles = np.linspace(0, 75, 9)

    # Estimate typical IMFP (~20-30 Å for organic materials with Al Kα)
    typical_imfp = 25.0

    # Information depth is typically 3× IMFP
    info_depth = 3 * typical_imfp

    # Suggested step size: coarser steps improve the conditioning. The
    # sqrt(info_depth) below is a heuristic default, not a derived rule.
    suggested_step = max(5.0, np.sqrt(target_info_depth))

    # Suggested depth range
    suggested_depth_end = min(target_info_depth, 5 * typical_imfp)

    # Number of depth points
    n_depth = int(suggested_depth_end / suggested_step) + 1

    suggestions = {
        'angles': angles,
        'step_size': suggested_step,
        'depth_end': suggested_depth_end,
        'n_depth_points': n_depth,
        'estimated_info_depth': info_depth,
        'typical_imfp': typical_imfp,
        'notes': [
            f"Using step_size={suggested_step:.1f}Å for improved conditioning",
            f"Depth range 0-{suggested_depth_end:.0f}Å based on info depth",
            f"Expected condition number: ~10^{np.log10(target_condition):.0f}",
        ],
    }

    print("=" * 50)
    print("Suggested Transfer Matrix Parameters")
    print("=" * 50)
    print(f"Angles: {angles}")
    print(f"Step size: {suggested_step:.1f} Å")
    print(f"Depth range: 0-{suggested_depth_end:.0f} Å")
    print(f"Number of depth points: {n_depth}")
    print(f"\nNotes:")
    for note in suggestions['notes']:
        print(f"  • {note}")

    return suggestions
