"""
HAXPES Data Preprocessing
==========================

Angle geometry, background subtraction, peak integration, and cross-section
correction for AR-HAXPES data.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from deppro.depth_profile import ElementConfig

from .config import SAMPLE_POLAR_DEG, _polar_offset_from

logger = logging.getLogger(__name__)


# ============================================================
# Angle Geometry
# ============================================================

def detector_to_emission(
    n_detector: int = 1001,
    polar_offset: float = SAMPLE_POLAR_DEG,
    det_range: Tuple[float, float] = (-36.84, 36.84),
    *,
    tilt: Optional[float] = None,
) -> np.ndarray:
    """Convert detector pixel positions to emission angles.

    Default range +/-36.84 deg comes from IBW wave header:
        sfA[1] = 0.07368511, sfB[1] = -36.8425532

    Parameters
    ----------
    n_detector : int
        Number of detector channels (1001 for Angular56).
    polar_offset : float
        Polar angle [degrees] from the analyzer central axis (lens axis, detector angle = 0)
        to the sample normal. Emission angle = detector angle + polar_offset. It does not
        coincide with the center angle of the kept bins (0.18 deg lower, from the 996/1001 channels).
    tilt : float, optional
        Old name of ``polar_offset`` (deprecated).
    det_range : tuple
        Detector angular range (lo, hi) in degrees.

    Returns
    -------
    emission : ndarray
        (n_detector,) emission angles in degrees.
    """
    det_angle = np.linspace(det_range[0], det_range[1], n_detector)
    # flip is a reinterpretation of detector rows as analyzer channels.
    # The 2D detector is mounted below the analyzer, so the row order is reversed relative to the
    # angular-dispersion axis (not needed for a top mount). It is not a sign flip of the angle (polar_offset - det).
    # Here, though, the default full 1001-channel grid being flipped (±36.84 deg) is symmetric about 0, so at this point the two coincide.
    # They differ on the 12-bin pre-binned grid (emission_angles_prebinned, load_prebinned_pxt), which comes
    # from 996/1001 channels and is off-center by 2.5 ch = 0.184 deg; read as a sign flip, every bin
    # shifts by a uniform 0.368 deg (happened on 2026-07-31).
    polar_offset = _polar_offset_from(polar_offset, tilt, "detector_to_emission")
    return np.flip(det_angle) + polar_offset


def make_uniform_bins(
    n_bins: int,
    angle_lo: float,
    angle_hi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create uniform-width angle bins.

    Returns
    -------
    bin_edges : ndarray
        (n_bins + 1,) bin edge angles.
    bin_centers : ndarray
        (n_bins,) bin center angles.
    """
    edges = np.linspace(angle_lo, angle_hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers


def bin_detector_data(
    raw_data: np.ndarray,
    emission_angles: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Bin raw detector data into angle bins by averaging.

    Parameters
    ----------
    raw_data : ndarray
        (n_energy, 1001) or (n_energy, 1001, n_frames).
    emission_angles : ndarray
        (1001,) emission angles.
    bin_edges : ndarray
        (n_bins + 1,) bin boundaries.

    Returns
    -------
    binned : ndarray
        (n_energy, n_bins) or (n_energy, n_bins, n_frames).
    """
    n_bins = len(bin_edges) - 1

    if raw_data.ndim == 2:
        n_energy = raw_data.shape[0]
        binned = np.zeros((n_energy, n_bins))
        for i in range(n_bins):
            mask = ((emission_angles >= bin_edges[i])
                    & (emission_angles < bin_edges[i + 1]))
            if mask.sum() > 0:
                binned[:, i] = raw_data[:, mask].mean(axis=1)
    elif raw_data.ndim == 3:
        n_energy, _, n_frames = raw_data.shape
        binned = np.zeros((n_energy, n_bins, n_frames))
        for i in range(n_bins):
            mask = ((emission_angles >= bin_edges[i])
                    & (emission_angles < bin_edges[i + 1]))
            if mask.sum() > 0:
                binned[:, i, :] = raw_data[:, mask, :].mean(axis=1)
    else:
        raise ValueError(f"Unexpected raw_data shape: {raw_data.shape}")

    return binned


# ============================================================
# Background Subtraction
# ============================================================

def subtract_background(
    spectrum: np.ndarray,
    method: str = 'linear_endpoints',
    margin_frac: float = 0.05,
) -> np.ndarray:
    """Subtract background from spectra.

    Parameters
    ----------
    spectrum : ndarray
        (n_energy,) single spectrum or (n_energy, n_angles) multi-angle.
    method : str
        'linear_endpoints', 'shirley', or 'none'.
    margin_frac : float
        Fraction of endpoints for baseline averaging.

    Returns
    -------
    bg_sub : ndarray
        Background-subtracted spectrum, same shape as input.
    """
    if spectrum.ndim == 1:
        if method == 'linear_endpoints':
            return _linear_bg_subtract_1d(spectrum, margin_frac)
        elif method == 'shirley':
            return _shirley_bg_subtract(spectrum)
        elif method == 'none':
            return spectrum.copy()
        else:
            raise ValueError(f"Unknown bg method: {method}")

    # 2D: (n_energy, n_angles)
    n_energy, n_cols = spectrum.shape
    bg_sub = np.zeros_like(spectrum)

    for j in range(n_cols):
        if method == 'linear_endpoints':
            bg_sub[:, j] = _linear_bg_subtract_1d(
                spectrum[:, j], margin_frac)
        elif method == 'shirley':
            bg_sub[:, j] = _shirley_bg_subtract(spectrum[:, j])
        elif method == 'none':
            bg_sub[:, j] = spectrum[:, j]
        else:
            raise ValueError(f"Unknown bg method: {method}")

    return bg_sub


def _linear_bg_subtract_1d(
    spec: np.ndarray,
    margin_frac: float = 0.05,
) -> np.ndarray:
    """Linear endpoint background subtraction for a single spectrum."""
    n = len(spec)
    n_margin = max(int(n * margin_frac), 3)
    lo_val = spec[:n_margin].mean()
    hi_val = spec[-n_margin:].mean()
    bg = np.linspace(lo_val, hi_val, n)
    return np.maximum(spec - bg, 0)


def _shirley_bg_subtract(
    spectrum: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> np.ndarray:
    """Iterative Shirley background subtraction for a single spectrum."""
    n = len(spectrum)
    bg = np.zeros(n)

    left_val = np.mean(spectrum[:5])
    right_val = np.mean(spectrum[-5:])

    for _ in range(max_iter):
        bg_old = bg.copy()

        # Cumulative integral from right
        cum_right = np.cumsum(np.maximum(spectrum - bg, 0)[::-1])[::-1]
        total = cum_right[0]

        if total > 0:
            bg = right_val + (left_val - right_val) * cum_right / total
        else:
            bg = np.linspace(left_val, right_val, n)

        if np.max(np.abs(bg - bg_old)) < tol:
            break

    return np.maximum(spectrum - bg, 0)


# ============================================================
# Peak Area Integration
# ============================================================

def integrate_peak_areas(
    binned_spectra: Dict[str, np.ndarray],
    energy_axes: Dict[str, np.ndarray],
    elem_names: List[str],
    bg_method: str = 'linear_endpoints',
    bg_margin_frac: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Background-subtract and integrate over energy to get peak areas.

    Parameters
    ----------
    binned_spectra : dict
        {elem: (n_energy, n_bins)} angle-binned spectra.
    energy_axes : dict
        {elem: (n_energy,)} kinetic energy axes.
    elem_names : list
        Ordered element names.
    bg_method : str
        Background subtraction method.
    bg_margin_frac : float
        Fraction of energy range for baseline averaging.

    Returns
    -------
    Ismp : ndarray
        (n_elements, n_bins) peak areas.
    spectra_bg_sub : dict
        {elem: (n_energy, n_bins)} background-subtracted spectra.
    """
    Ismp_list = []
    spectra_bg_sub = {}

    for elem in elem_names:
        if elem not in binned_spectra:
            n_bins = next(iter(binned_spectra.values())).shape[1]
            Ismp_list.append(np.zeros(n_bins))
            continue

        spectrum = binned_spectra[elem]  # (n_energy, n_bins)
        energy = energy_axes[elem]

        # Background subtraction
        bg_sub = subtract_background(spectrum, bg_method, bg_margin_frac)
        spectra_bg_sub[elem] = bg_sub

        # Energy step size
        de = abs(energy[1] - energy[0]) if len(energy) > 1 else 1.0
        peak_areas = bg_sub.sum(axis=0) * de  # (n_bins,)

        Ismp_list.append(peak_areas)
        logger.info(
            f"  {elem}: peak_area range = "
            f"[{peak_areas.min():.1f}, {peak_areas.max():.1f}] "
            f"(mean={peak_areas.mean():.1f})"
        )

    return np.array(Ismp_list), spectra_bg_sub


# ============================================================
# Cross Section Correction
# ============================================================

def _compute_non_dipole_factors(
    elements: List[ElementConfig],
    photon_energy: float,
    emission_angles: np.ndarray,
    xray_from_normal_deg: float = 88.0,
    polarized: bool = True,
) -> np.ndarray:
    """Compute non-dipole angular correction factors L(θ) for each element.

    Supports two modes:
    - polarized=True  (synchrotron): L_full with ψ = α_xray − θ
    - polarized=False (lab source):  L_unpolarized with α = α_xray − θ

    Parameters
    ----------
    elements : list
        ElementConfig list.
    photon_energy : float
        Excitation energy in eV.
    emission_angles : ndarray
        (n_bins,) emission angles in degrees.
    xray_from_normal_deg : float
        X-ray incidence angle from surface normal (deg).
    polarized : bool
        True for synchrotron (linearly polarized), False for lab
        X-ray sources (unpolarized).

    Returns
    -------
    L_factors : ndarray
        (n_elements, n_bins) angular correction factors.
    """
    from toyomacro.data.angular_correction import AngularCorrection

    n_elem = len(elements)
    n_bins = len(emission_angles)
    L_factors = np.ones((n_elem, n_bins))

    mode_str = "polarized" if polarized else "unpolarized"

    # Map common orbital names to Trzhaskovskaya notation
    _ORBITAL_MAP = {
        '1s': '1s1/2', '2s': '2s1/2', '3s': '3s1/2', '4s': '4s1/2',
        '2p': '2p3/2', '2p3/2': '2p3/2', '2p1/2': '2p1/2',
        '3p': '3p3/2', '3p3/2': '3p3/2', '3p1/2': '3p1/2',
        '3d': '3d5/2', '3d5/2': '3d5/2', '3d3/2': '3d3/2',
        '4d': '4d5/2', '4d5/2': '4d5/2', '4d3/2': '4d3/2',
        '4f': '4f7/2', '4f7/2': '4f7/2', '4f5/2': '4f5/2',
    }

    for i, elem in enumerate(elements):
        orbital_trzh = _ORBITAL_MAP.get(elem.orbital, elem.orbital)

        if polarized:
            result = AngularCorrection.angular_distribution(
                elem.symbol, orbital_trzh, photon_energy,
                emission_angles_deg=emission_angles,
                xray_from_normal_deg=xray_from_normal_deg,
            )
            if result is not None:
                L_factors[i, :] = result['L_full']
                L_key = 'L_full'
            else:
                result = None
        else:
            result = AngularCorrection.angular_distribution_unpolarized(
                elem.symbol, orbital_trzh, photon_energy,
                emission_angles_deg=emission_angles,
                xray_from_normal_deg=xray_from_normal_deg,
            )
            if result is not None:
                L_factors[i, :] = result['L_unpolarized']
                L_key = 'L_unpolarized'

        if result is not None:
            L = L_factors[i, :]
            logger.info(
                f"  Non-dipole ({mode_str}) {elem.symbol} {elem.orbital}: "
                f"β={result['beta']:.3f}, γ={result['gamma']:.3f}, "
                f"δ={result['delta']:.4f}, "
                f"L range=[{L.min():.3f}, {L.max():.3f}]"
            )
        else:
            logger.warning(
                f"  Non-dipole {elem.symbol} {orbital_trzh}: "
                f"not found in Trzhaskovskaya tables, using L=1"
            )

    return L_factors


def compute_L_factors(
    elements: List[ElementConfig],
    photon_energy: float,
    emission_angles: np.ndarray,
    xray_from_normal_deg: float = 88.0,
    polarized: bool = True,
) -> np.ndarray:
    """Compute non-dipole angular correction factors for TransferMatrix.

    Returns L in TransferMatrix-compatible shape (n_angles, n_elements).

    Parameters
    ----------
    elements : list
        ElementConfig list.
    photon_energy : float
        Excitation energy in eV.
    emission_angles : ndarray
        (n_bins,) emission angles in degrees.
    xray_from_normal_deg : float
        X-ray incidence angle from surface normal (deg).
    polarized : bool
        True for synchrotron, False for lab X-ray.

    Returns
    -------
    L : ndarray
        (n_angles, n_elements) angular correction factors.
        Pass directly to TransferMatrix.compute(L_correction=L).
    """
    L = _compute_non_dipole_factors(
        elements, photon_energy, emission_angles,
        xray_from_normal_deg=xray_from_normal_deg,
        polarized=polarized,
    )
    return L.T  # (n_elem, n_bins) → (n_bins, n_elem)


def apply_cross_section_correction(
    Ismp: np.ndarray,
    elements: List[ElementConfig],
    photon_energy: float,
    source: str = 'yeh_lindau_extrap',
    poly_order: int = 3,
    emission_angles: Optional[np.ndarray] = None,
    non_dipole: bool = False,
    xray_from_normal_deg: float = 88.0,
    polarized: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Divide Ismp by photoionization cross section.

    Parameters
    ----------
    Ismp : ndarray
        (n_elements, n_bins) peak area intensities.
    elements : list
        ElementConfig list matching Ismp row order.
    photon_energy : float
        Excitation energy in eV.
    source : str
        'yeh_lindau_extrap' (recommended for HAXPES), 'scofield', 'yeh_lindau'.
    poly_order : int
        Polynomial order for Yeh-Lindau extrapolation.
    emission_angles : ndarray, optional
        (n_bins,) emission angles in degrees. Required when non_dipole=True.
    non_dipole : bool
        If True, apply non-dipole angular correction (Cooper formula).
    xray_from_normal_deg : float
        X-ray incidence angle from surface normal (deg). Default 88.0
        (grazing ~2° incidence, typical of a synchrotron HAXPES beamline).
    polarized : bool
        True for synchrotron (linearly polarized), False for lab
        X-ray sources (unpolarized, e.g. Ga Kα tube).

    Returns
    -------
    Ismp_corrected : ndarray
        (n_elements, n_bins) cross-section corrected intensities.
    sigmas : ndarray
        (n_elements,) total cross sections in Mb (angle-independent).
    """
    from deppro.haxpes.cross_section_loader import CrossSectionDatabase

    _ELEMENT_Z = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6,
        'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12,
        'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Ti': 22,
    }

    db = CrossSectionDatabase()
    sigmas = np.zeros(len(elements))

    logger.info(f"Cross section correction ({source})")
    for i, elem in enumerate(elements):
        if source == 'yeh_lindau_extrap':
            z = _ELEMENT_Z[elem.symbol]
            if z <= 12:
                r = db.get_cross_section(
                    elem.symbol, elem.orbital,
                    photon_energy, source='yeh_lindau')
                sigmas[i] = r.cross_section_Mb
            else:
                from .cross_section_extrap import yeh_lindau_extrapolate_1s
                sigmas[i] = yeh_lindau_extrapolate_1s(
                    z, photon_energy,
                    poly_order_energy=poly_order,
                    poly_order_z=poly_order)
            logger.info(f"  {elem.symbol} {elem.orbital}: "
                        f"sigma={sigmas[i]:.4e} Mb")
        else:
            r = db.get_cross_section(
                elem.symbol, elem.orbital,
                photon_energy, source=source)
            sigmas[i] = r.cross_section_Mb
            logger.info(f"  {elem.symbol} {elem.orbital}: "
                        f"sigma={r.cross_section_Mb:.4e} Mb")

    if non_dipole and emission_angles is not None:
        L_factors = _compute_non_dipole_factors(
            elements, photon_energy, emission_angles,
            xray_from_normal_deg=xray_from_normal_deg,
            polarized=polarized,
        )
        # σ_eff(elem, angle) = σ_total × L(angle)
        sigma_angle = sigmas[:, np.newaxis] * L_factors
        Ismp_corrected = Ismp / sigma_angle
    else:
        Ismp_corrected = Ismp / sigmas[:, np.newaxis]

    return Ismp_corrected, sigmas


# ============================================================
# Quality Metrics
# ============================================================

def compute_si_normalized_rrmse(
    I_obs: np.ndarray,
    I_opt: np.ndarray,
    substrate_index: int = 0,
) -> float:
    """Compute Si-normalized relative RMSE.

    Divides each element's angular profile by the substrate (Si),
    then computes RMSE over all element/angle pairs.

    Parameters
    ----------
    I_obs : ndarray
        (n_angles, n_elements) observed intensities.
    I_opt : ndarray
        (n_angles, n_elements) optimized forward-model intensities.
    substrate_index : int
        Index of substrate element (default: 0 for Si).

    Returns
    -------
    rrmse : float
        Relative RMSE (a fraction, e.g. 0.05 for 5%).
    """
    n_elements = I_obs.shape[1]
    obs_parts, opt_parts = [], []
    for i in range(n_elements):
        if i == substrate_index:
            continue
        obs_parts.append(I_obs[:, i] / I_obs[:, substrate_index])
        opt_parts.append(I_opt[:, i] / I_opt[:, substrate_index])
    obs_all = np.concatenate(obs_parts)
    opt_all = np.concatenate(opt_parts)
    rmse = np.sqrt(np.mean((obs_all - opt_all) ** 2))
    return rmse / np.mean(np.abs(obs_all))
