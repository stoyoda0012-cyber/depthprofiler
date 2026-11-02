"""
Derived Metrics from Depth Profile Results
============================================

Pure computation functions to calculate MATLAB-compatible derived quantities
from D_opt (depth profiles). No I/O — all functions take arrays and return arrays.

These metrics populate the GUI-required datasets:
    Xrlt  — forward model intensity at thetaopt angles
    C     — depth-integrated concentration per element
    FV    — feature vector: [thickness, abruptness, p/w/h per free+const element]
    SV    — support vector: [p/w/h per substrate element, p/w/h per linked element]
    SPS   — sparsity: [count(x<1e-5), n_free * n_depth]

Reference: DepthProfiler.m lines 2523-2528 (parfor optimization loop)

Date: 2026-02-15
"""

import numpy as np
from ._compat import trapezoid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DerivedMetricsConfig:
    """Configuration for derived metrics computation.

    All indices are 0-based (Python convention).
    """
    depth: np.ndarray               # (n_depth,) depth grid in Angstroms
    n_elements: int                  # total number of elements
    free_indices: List[int]          # 0-based indices of free elements
    subst_indices: List[int]         # 0-based indices of substrate elements
    linked_indices: List[int]        # 0-based indices of linked elements
    const_indices: List[int]         # 0-based indices of constant elements (often empty)
    trans_mat_opt: Optional[np.ndarray] = None  # (n_angles_opt, n_depth, n_elements) or None
    n_angles_opt: int = 0            # number of optimization angles (for Xrlt)

    @property
    def fv_indices(self) -> List[int]:
        """Sorted free + const indices (for FeatureVector)."""
        return sorted(self.free_indices + self.const_indices)

    @property
    def antisubst_indices(self) -> List[int]:
        """All non-substrate indices (for concentration normalization)."""
        subst_set = set(self.subst_indices)
        return [i for i in range(self.n_elements) if i not in subst_set]

    @property
    def n_fv(self) -> int:
        """Length of feature vector: 2 + 3 * len(free + const)."""
        return 2 + 3 * len(self.fv_indices)

    @property
    def n_sv(self) -> int:
        """Length of support vector: 3 * n_subst + 3 * n_linked."""
        return 3 * len(self.subst_indices) + 3 * len(self.linked_indices)


def compute_xrlt(
    D_opt: np.ndarray,
    trans_mat_opt: np.ndarray,
) -> np.ndarray:
    """Compute forward model intensity at optimization angles.

    MATLAB equivalent (CalcDistrib_Parallel.m line 27):
        for i = 1:n_elements
            I2(:, i) = TransMatOpt(:, :, i) * Dopt(:, i)
        Xrlt = I2 / sum(I2, 2)

    Args:
        D_opt: (n_depth, n_elements) depth profile
        trans_mat_opt: (n_angles_opt, n_depth, n_elements) transfer matrix at thetaopt

    Returns:
        Xrlt: (n_angles_opt, n_elements) normalized intensity
    """
    n_angles, n_depth, n_elem = trans_mat_opt.shape

    # I2[:, i] = TransMatOpt[:, :, i] @ D_opt[:, i]
    I2 = np.zeros((n_angles, n_elem))
    for i in range(n_elem):
        I2[:, i] = trans_mat_opt[:, :, i] @ D_opt[:, i]

    # Normalize: each angle sums to 1
    row_sums = I2.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-30)  # avoid division by zero
    Xrlt = I2 / row_sums

    return Xrlt


def compute_concentration(
    D_opt: np.ndarray,
    antisubst_indices: List[int],
) -> np.ndarray:
    """Compute depth-integrated concentration per element.

    MATLAB equivalent (DepthProfiler.m line 2528):
        Conc_Calc = 100 * sum(Dopt) / sum(Dopt(:, antisubst), 'all')

    Args:
        D_opt: (n_depth, n_elements) depth profile
        antisubst_indices: 0-based indices of non-substrate elements

    Returns:
        C: (n_elements,) concentration in percent
    """
    total_per_elem = D_opt.sum(axis=0)  # (n_elements,)
    film_total = D_opt[:, antisubst_indices].sum()

    if film_total < 1e-30:
        return np.zeros(D_opt.shape[1])

    return 100.0 * total_per_elem / film_total


def _compute_pwh(
    D_opt: np.ndarray,
    depth: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute position/width/height for all elements.

    MATLAB equivalent (DepthProfiler.m line 2525):
        p = sum(Dopt .* repmat(depth, 1, n_elem)) ./ sum(Dopt)
        h = 100 * max(Dopt)
        w = 2 * sqrt(log(2)/pi) * trapz(depth, 100*Dopt) ./ h

    Args:
        D_opt: (n_depth, n_elements) depth profile
        depth: (n_depth,) depth grid

    Returns:
        p: (n_elements,) position (center of mass)
        w: (n_elements,) width (FWHM-like)
        h: (n_elements,) height (max concentration × 100)
    """
    n_depth, n_elem = D_opt.shape

    # Position: weighted mean depth
    depth_2d = depth[:, np.newaxis]  # (n_depth, 1)
    col_sums = D_opt.sum(axis=0)  # (n_elements,)
    safe_sums = np.where(col_sums > 1e-30, col_sums, 1.0)
    p = (D_opt * depth_2d).sum(axis=0) / safe_sums

    # Height: max concentration × 100
    h = 100.0 * D_opt.max(axis=0)

    # Width: area-based FWHM
    area = trapezoid(100.0 * D_opt, depth, axis=0)  # (n_elements,)
    safe_h = np.where(h > 1e-30, h, 1.0)
    w = 2.0 * np.sqrt(np.log(2.0) / np.pi) * area / safe_h

    return p, w, h


def compute_feature_vector(
    D_opt: np.ndarray,
    thickness: float,
    abruptness: float,
    depth: np.ndarray,
    fv_indices: List[int],
) -> np.ndarray:
    """Compute feature vector for free+const elements.

    MATLAB equivalent (DepthProfiler.m line 2526):
        idfv = sort([free const])
        FeatureVector_Calc = [thckfin abrpfin reshape([p(idfv)' w(idfv)' h(idfv)']', 1, 3*length(idfv))]

    Layout: [thickness, abruptness, p1, w1, h1, p2, w2, h2, ...]
    Note: MATLAB interleaves as p/w/h per element (not all p then all w then all h).

    Args:
        D_opt: (n_depth, n_elements) depth profile
        thickness: film thickness (Angstroms)
        abruptness: interface abruptness (Angstroms)
        depth: (n_depth,) depth grid
        fv_indices: sorted 0-based indices of free+const elements

    Returns:
        FV: (2 + 3*len(fv_indices),) feature vector
    """
    p, w, h = _compute_pwh(D_opt, depth)

    n_fv = len(fv_indices)
    FV = np.zeros(2 + 3 * n_fv, dtype=np.float32)
    FV[0] = thickness
    FV[1] = abruptness

    # Interleave: [p0, w0, h0, p1, w1, h1, ...]
    for k, idx in enumerate(fv_indices):
        FV[2 + 3 * k + 0] = p[idx]
        FV[2 + 3 * k + 1] = w[idx]
        FV[2 + 3 * k + 2] = h[idx]

    return FV


def compute_support_vector(
    D_opt: np.ndarray,
    depth: np.ndarray,
    subst_indices: List[int],
    linked_indices: List[int],
) -> np.ndarray:
    """Compute support vector for substrate+linked elements.

    MATLAB equivalent (DepthProfiler.m line 2527):
        SupportVector_Calc = [reshape([p(subst)' w(subst)' h(subst)']', 1, 3*length(subst))
                              reshape([p(linked)' w(linked)' h(linked)']', 1, 3*length(linked))]

    Layout: [p_sub1, w_sub1, h_sub1, ..., p_lnk1, w_lnk1, h_lnk1, ...]

    Args:
        D_opt: (n_depth, n_elements)
        depth: (n_depth,) depth grid
        subst_indices: 0-based substrate indices
        linked_indices: 0-based linked indices

    Returns:
        SV: (3*n_subst + 3*n_linked,) support vector
    """
    p, w, h = _compute_pwh(D_opt, depth)

    n_subst = len(subst_indices)
    n_linked = len(linked_indices)
    SV = np.zeros(3 * n_subst + 3 * n_linked, dtype=np.float32)

    for k, idx in enumerate(subst_indices):
        SV[3 * k + 0] = p[idx]
        SV[3 * k + 1] = w[idx]
        SV[3 * k + 2] = h[idx]

    offset = 3 * n_subst
    for k, idx in enumerate(linked_indices):
        SV[offset + 3 * k + 0] = p[idx]
        SV[offset + 3 * k + 1] = w[idx]
        SV[offset + 3 * k + 2] = h[idx]

    return SV


def compute_sparsity(
    D_opt: np.ndarray,
    free_indices: List[int],
) -> np.ndarray:
    """Compute sparsity statistics.

    MATLAB equivalent (DepthProfiler.m line 2528):
        Sprs_Calc = [length(find(x(1:m*length(free)) < 1e-5))  length(x(1:m*length(free)))]

    Args:
        D_opt: (n_depth, n_elements) depth profile
        free_indices: 0-based indices of free elements

    Returns:
        SPS: (2,) = [n_sparse, n_total]
    """
    n_depth = D_opt.shape[0]
    n_free = len(free_indices)

    # Extract free element concentrations
    free_conc = D_opt[:, free_indices].ravel()  # (n_depth * n_free,)

    n_sparse = int(np.sum(free_conc < 1e-5))
    n_total = n_depth * n_free

    return np.array([n_sparse, n_total], dtype=np.float32)


# ==============================================================================
# Batch computation (for streaming pipeline)
# ==============================================================================

@dataclass
class DerivedResults:
    """Derived metrics for a batch of pixels.

    All arrays have first dimension = n_pixels_in_batch.
    """
    # 3D: (batch, n_angles_opt, n_elements) → stored as (batch, n_elem, n_angles_opt) in H5
    Xrlt: Optional[np.ndarray] = None

    # 2D: (batch, n_elements) → stored as (n_elements, batch) in H5
    C: Optional[np.ndarray] = None

    # 2D: (batch, n_fv) → stored as (n_fv, batch) in H5
    FV: Optional[np.ndarray] = None

    # 2D: (batch, n_sv) → stored as (n_sv, batch) in H5
    SV: Optional[np.ndarray] = None

    # 2D: (batch, 2) → stored as (2, batch) in H5
    SPS: Optional[np.ndarray] = None


def compute_all_derived_batch(
    D_opts: np.ndarray,
    thicknesses: np.ndarray,
    abruptnesses: np.ndarray,
    config: DerivedMetricsConfig,
) -> DerivedResults:
    """Compute all derived metrics for a batch of pixels.

    This is the main entry point used by the streaming pipeline.
    Called once per chunk in the main process (not in workers).

    Args:
        D_opts: (n_pixels, n_depth, n_elements) depth profiles (Python layout)
        thicknesses: (n_pixels,) film thicknesses
        abruptnesses: (n_pixels,) interface abruptnesses
        config: Metric computation configuration

    Returns:
        DerivedResults with all derived quantities
    """
    n_px, n_depth, n_elem = D_opts.shape
    depth = config.depth
    fv_indices = config.fv_indices
    antisubst = config.antisubst_indices

    # Pre-allocate outputs
    n_fv = config.n_fv
    n_sv = config.n_sv

    has_xrlt = (config.trans_mat_opt is not None and config.n_angles_opt > 0)
    n_angles_opt = config.n_angles_opt if has_xrlt else 0

    Xrlt_batch = np.zeros((n_px, n_angles_opt, n_elem), dtype=np.float32) if has_xrlt else None
    C_batch = np.zeros((n_px, n_elem), dtype=np.float32)
    FV_batch = np.zeros((n_px, n_fv), dtype=np.float32)
    SV_batch = np.zeros((n_px, n_sv), dtype=np.float32)
    SPS_batch = np.zeros((n_px, 2), dtype=np.float32)

    for i in range(n_px):
        D = D_opts[i]  # (n_depth, n_elements)

        # Skip zero/invalid pixels
        if D.sum() < 1e-30:
            continue

        # Xrlt
        if has_xrlt:
            Xrlt_batch[i] = compute_xrlt(D, config.trans_mat_opt)

        # Concentration
        C_batch[i] = compute_concentration(D, antisubst)

        # Feature Vector
        FV_batch[i] = compute_feature_vector(
            D, thicknesses[i], abruptnesses[i], depth, fv_indices
        )

        # Support Vector
        SV_batch[i] = compute_support_vector(
            D, depth, config.subst_indices, config.linked_indices
        )

        # Sparsity
        SPS_batch[i] = compute_sparsity(D, config.free_indices)

    return DerivedResults(
        Xrlt=Xrlt_batch,
        C=C_batch,
        FV=FV_batch,
        SV=SV_batch,
        SPS=SPS_batch,
    )
