"""
HAXPES Bootstrap Methods
=========================

Per-bin shuffle and frame bootstrap for HAXPES variance estimation.

Per-bin shuffle assigns independent random temporal indices to each
(energy, angle) bin, effectively averaging over sqrt(n_channels) time
points through peak area integration.

Mechanism (Jensen's inequality):
    per-bin shuffle: each of ~100 energy channels picks independent time indices
    peak area = sum over 100 channels = implicit 100-point temporal average
    improvement factor ~ sqrt(n_energy_channels) ~ sqrt(100) ~ 10x
"""
import logging
from dataclasses import dataclass
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from deppro.depth_profile import ElementConfig

logger = logging.getLogger(__name__)


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class BootstrapResult:
    """Results from bootstrap analysis.

    Attributes:
        D_map: (n_pixels, n_depth, n_elements) depth profiles.
        thk_map: (n_pixels,) thickness per pixel.
        rrmse_map: (n_pixels,) rRMSE per pixel.
        n_converged: Number of converged pixels.
        method: 'frame' or 'perbin'.
        K: Number of bootstrap samples per pixel.
        elapsed_seconds: Total processing time.
    """
    D_map: np.ndarray
    thk_map: np.ndarray
    rrmse_map: np.ndarray
    n_converged: int
    method: str
    K: int
    elapsed_seconds: float


# ============================================================
# Pixel averaging
# ============================================================

def average_pixels(
    pixels: List[Dict[str, np.ndarray]],
    indices: np.ndarray,
    elem_names: List[str],
) -> Dict[str, np.ndarray]:
    """Average spectra from selected pixel indices.

    Parameters
    ----------
    pixels : list
        List of {elem: (n_energy, n_angles)} pixel dicts.
    indices : ndarray
        Pixel indices to average.
    elem_names : list
        Ordered element names.

    Returns
    -------
    avg : dict
        {elem: (n_energy, n_angles)} averaged spectra.
    """
    avg = {}
    for elem in elem_names:
        spectra = []
        for idx in indices:
            if elem in pixels[idx]:
                spectra.append(pixels[idx][elem])
        if spectra:
            avg[elem] = np.mean(spectra, axis=0)
        else:
            # Fallback shape from first available pixel
            for p in pixels:
                if elem in p:
                    avg[elem] = np.zeros_like(p[elem])
                    break
    return avg


# ============================================================
# Per-bin bootstrap pixel building
# ============================================================

def build_perbin_pixels(
    elem_stacks: Dict[str, np.ndarray],
    angle_mask: np.ndarray,
    elem_names: List[str],
    nx: int,
    ny: int,
    K: int = 50,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """Build per-bin bootstrap averaged pixel data.

    For each pixel, for each element, for each (energy, angle) bin,
    independently pick K random time indices and average. This is the
    MATLAB-style bootstrap that produces implicit temporal averaging
    through peak area integration.

    Parameters
    ----------
    elem_stacks : dict
        {elem: (n_time, n_energy, n_angles)} raw data.
    angle_mask : ndarray
        Boolean mask for valid angles.
    elem_names : list
        Ordered element names.
    nx, ny : int
        Output grid dimensions.
    K : int
        Number of bootstrap draws per bin.
    seed : int
        Random seed.

    Returns
    -------
    pixel_data_list : list
        List of {elem: (n_energy, n_angles_used)} dicts.
    """
    rng = np.random.default_rng(seed)
    first_elem = next(iter(elem_stacks))
    n_time = elem_stacks[first_elem].shape[0]
    n_energy = elem_stacks[first_elem].shape[1]
    angle_indices = np.where(angle_mask)[0]
    n_a_used = len(angle_indices)
    n_pixels = nx * ny

    # Pre-extract angle-masked data
    elem_masked = {}
    for elem in elem_names:
        if elem in elem_stacks:
            elem_masked[elem] = elem_stacks[elem][:, :, angle_indices]

    # Broadcasting indices for gather
    E_bc = np.arange(n_energy)[None, :, None]   # (1, n_energy, 1)
    A_bc = np.arange(n_a_used)[None, None, :]   # (1, 1, n_a_used)

    pixel_data_list = []
    t0 = time()

    for px_idx in range(n_pixels):
        pixel_data = {}
        for elem in elem_names:
            if elem not in elem_masked:
                continue
            stack = elem_masked[elem]  # (n_time, n_energy, n_a_used)
            # Independent random time index per (K, energy, angle)
            rand_t = rng.integers(0, n_time, size=(K, n_energy, n_a_used))
            samples = stack[rand_t, E_bc, A_bc]  # (K, n_energy, n_a_used)
            pixel_data[elem] = samples.mean(axis=0)  # (n_energy, n_a_used)

        pixel_data_list.append(pixel_data)

        if (px_idx + 1) % 500 == 0:
            elapsed = time() - t0
            logger.info(f"  Built {px_idx+1}/{n_pixels} pixels ({elapsed:.1f}s)")

    logger.info(f"  All {n_pixels} per-bin pixels built ({time()-t0:.1f}s)")
    return pixel_data_list


def build_frame_pixels(
    all_pixels: List[Dict[str, np.ndarray]],
    angle_mask: np.ndarray,
    elem_names: List[str],
    nx: int,
    ny: int,
    K: int = 50,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """Build frame-level bootstrap averaged pixel data.

    Standard bootstrap: draw K complete frames and average.

    Parameters
    ----------
    all_pixels : list
        List of {elem: (n_energy, n_angles)} pixel dicts (all temporal frames).
    angle_mask : ndarray
        Boolean mask for valid angles.
    elem_names : list
        Ordered element names.
    nx, ny : int
        Output grid dimensions.
    K : int
        Number of bootstrap draws.
    seed : int
        Random seed.

    Returns
    -------
    pixel_data_list : list
        List of {elem: (n_energy, n_angles_used)} dicts.
    """
    rng = np.random.default_rng(seed)
    n_total = len(all_pixels)
    n_pixels = nx * ny

    pixel_data_list = []
    for _ in range(n_pixels):
        indices = rng.choice(n_total, size=K, replace=True)
        avg_pixel = average_pixels(all_pixels, indices, elem_names)
        pixel_data = {}
        for elem in elem_names:
            if elem in avg_pixel:
                pixel_data[elem] = avg_pixel[elem][:, angle_mask]
        pixel_data_list.append(pixel_data)

    return pixel_data_list


# ============================================================
# Frame bootstrap (single-point, B repetitions)
# ============================================================

def frame_bootstrap(
    all_pixels: List[Dict[str, np.ndarray]],
    elements: List[ElementConfig],
    angles: np.ndarray,
    sigmas: np.ndarray,
    angle_mask: np.ndarray,
    elem_names: List[str],
    n_bootstrap: int = 100,
    k_sample: int = 50,
    alpha: float = 0.01,
    thickness_init: float = 240.0,
    seed: int = 42,
) -> List[dict]:
    """Run frame-level bootstrap.

    Draw K pixels, average spectra, solve -> repeat B times.
    This is solve(mean(x)) -- average BEFORE optimization.

    Parameters
    ----------
    all_pixels : list
        All temporal frames as pixel dicts.
    elements : list
        ElementConfig list.
    angles : ndarray
        Emission angles (already masked).
    sigmas : ndarray
        Cross sections.
    angle_mask : ndarray
        Boolean mask for angle selection.
    elem_names : list
        Ordered element names.
    n_bootstrap : int
        Number of bootstrap repetitions.
    k_sample : int
        Number of frames to average per sample.
    alpha : float
        Regularization.
    thickness_init : float
        Initial thickness.
    seed : int
        Random seed.

    Returns
    -------
    results : list
        List of {'D_opt', 'thickness', 'rrmse'} dicts.
    """
    from .pipeline import process_single_pixel

    rng = np.random.default_rng(seed)
    n_total = len(all_pixels)

    results = []
    t0 = time()

    for b in range(n_bootstrap):
        indices = rng.choice(n_total, size=k_sample, replace=True)
        avg_pixel = average_pixels(all_pixels, indices, elem_names)

        pixel_data = {}
        for elem in elem_names:
            if elem in avg_pixel:
                pixel_data[elem] = avg_pixel[elem][:, angle_mask]

        D_opt, thickness, rrmse = process_single_pixel(
            pixel_data, elements, angles, sigmas,
            alpha=alpha, thickness_init=thickness_init,
        )

        results.append({
            'D_opt': D_opt,
            'thickness': thickness,
            'rrmse': rrmse,
        })

        if (b + 1) % 20 == 0:
            elapsed = time() - t0
            rate = (b + 1) / elapsed
            logger.info(f"  [{b+1}/{n_bootstrap}] "
                        f"thk={thickness:.0f}A rRMSE={rrmse:.1%} "
                        f"({rate:.1f} samples/s)")

    return results


# ============================================================
# Block bootstrap (coarser spatial map)
# ============================================================

def block_bootstrap(
    all_pixels: List[Dict[str, np.ndarray]],
    elements: List[ElementConfig],
    angles: np.ndarray,
    sigmas: np.ndarray,
    angle_mask: np.ndarray,
    elem_names: List[str],
    block_size: int = 100,
    n_bootstrap_per_block: int = 20,
    alpha: float = 0.01,
    thickness_init: float = 240.0,
    seed: int = 42,
) -> List[dict]:
    """Run block bootstrap for coarser spatial maps.

    Divide pixels into spatial blocks, bootstrap within each block.

    Parameters
    ----------
    all_pixels : list
        All temporal frames.
    elements : list
        ElementConfig list.
    angles : ndarray
        Emission angles (already masked).
    sigmas : ndarray
        Cross sections.
    angle_mask : ndarray
        Boolean mask.
    elem_names : list
        Element names.
    block_size : int
        Pixels per block.
    n_bootstrap_per_block : int
        Bootstrap samples per block.
    alpha : float
        Regularization.
    thickness_init : float
        Initial thickness.
    seed : int
        Random seed.

    Returns
    -------
    block_results : list
        List of {'D_mean', 'thk_mean', 'D_opts', 'thicknesses'} per block.
    """
    from .pipeline import process_single_pixel

    rng = np.random.default_rng(seed)
    n_total = len(all_pixels)
    n_blocks = n_total // block_size
    grid_side = int(np.sqrt(n_blocks))
    n_blocks = grid_side * grid_side

    logger.info(f"Block bootstrap: {grid_side}x{grid_side} blocks, "
                f"{block_size} px/block, {n_bootstrap_per_block}/block")

    block_results = []
    t0 = time()

    for blk in range(n_blocks):
        blk_lo = blk * block_size
        blk_hi = min(blk_lo + block_size, n_total)

        D_opts = []
        thicknesses = []
        for _ in range(n_bootstrap_per_block):
            indices = rng.choice(
                range(blk_lo, blk_hi),
                size=block_size, replace=True)
            avg_pixel = average_pixels(all_pixels, indices, elem_names)
            pixel_data = {}
            for elem in elem_names:
                if elem in avg_pixel:
                    pixel_data[elem] = avg_pixel[elem][:, angle_mask]

            D_opt, thickness, rrmse = process_single_pixel(
                pixel_data, elements, angles, sigmas,
                alpha=alpha, thickness_init=thickness_init,
            )

            if D_opt is not None:
                D_opts.append(D_opt)
                thicknesses.append(thickness)

        if D_opts:
            D_mean = np.median(D_opts, axis=0)
            thk_mean = np.median(thicknesses)
        else:
            D_mean = np.zeros((20, 5))
            thk_mean = 0.0

        block_results.append({
            'D_mean': D_mean,
            'thk_mean': thk_mean,
            'D_opts': D_opts,
            'thicknesses': thicknesses,
        })

    elapsed = time() - t0
    logger.info(f"Block bootstrap done: {elapsed:.1f}s")
    return block_results
