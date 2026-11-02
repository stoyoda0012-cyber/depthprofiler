#!/usr/bin/env python
"""
Roundtrip Benchmark for ARXPS Depth Profile Analysis
=====================================================

Tests the complete cycle:
    Depth Profile (100x100 pixels)
        -> Angle-resolved spectra (14 angles) + Poisson noise
        -> Inverse solver
        -> Reconstructed profile

Goal: Generate a visible "film-like" 100x100 image showing reconstruction quality.

Usage:
    # Quick benchmark
    python -m deppro.benchmarks.roundtrip_benchmark --quick

    # Full benchmark with image output
    python -m deppro.benchmarks.roundtrip_benchmark --noise 0.01 --output outputs/

    # Using reference H5 profile
    python -m deppro.benchmarks.roundtrip_benchmark --h5 outputs/reproduced_dataset.h5

Date: 2026-02-04
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import h5py
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

from deppro import (
    TransferMatrix,
    ElementConfig,
    FreeSolver,
    SemiFreeSolver,
    SemiFreeSolverConfig,
    GenericLayerSolver,
    GenericLayerConfig,
    LayerDefinition,
    SolverConfig,
    create_multilayer_profile,
    CompoundDatabase,
)


# ==============================================================================
# Configuration
# ==============================================================================

# Default 14 angles (standard ARXPS measurement)
DEFAULT_ANGLES = np.array([0, 5, 15, 23, 30, 38, 45, 53, 60, 67, 72, 77, 82, 85])

# 5-element system: O, C, Al, Ti, Si (Si-ox and Si-bulk merged)
ELEMENT_CONFIGS_5 = [
    ElementConfig(symbol='O', orbital='1s', compound='SiO2', binding_energy=532.0),
    ElementConfig(symbol='C', orbital='1s', compound='C', binding_energy=285.0),
    ElementConfig(symbol='Al', orbital='2p', compound='Al2O3', binding_energy=74.0),
    ElementConfig(symbol='Ti', orbital='2p', compound='TiO2', binding_energy=459.0),
    ElementConfig(symbol='Si', orbital='2p', compound='Si', binding_energy=99.0),
]


@dataclass
class RoundtripConfig:
    """Configuration for roundtrip benchmark."""
    # Grid size
    n_x: int = 100
    n_y: int = 100

    # Depth profile
    n_depth: int = 100  # Depth points
    max_depth: float = 200.0  # Maximum depth in Angstroms

    # Angles
    angles: np.ndarray = None  # Defaults to DEFAULT_ANGLES

    # Noise
    noise_level: float = 0.01  # 1% Poisson noise

    # Solver
    solver_type: str = 'free'  # 'free', 'semi-free', 'generic'
    method: str = 'L2'  # L2 is more stable than MEM for this test
    alpha: float = 0.01  # Moderate regularization for stability
    max_iter: int = 200

    # Parallelization
    parallel: bool = False  # Use parallel processing
    n_workers: int = None  # Number of workers (None = CPU count - 1)

    # Elements
    n_elements: int = 5
    elements: List[ElementConfig] = None

    def __post_init__(self):
        if self.angles is None:
            self.angles = DEFAULT_ANGLES.copy()
        if self.elements is None:
            self.elements = ELEMENT_CONFIGS_5[:self.n_elements]


@dataclass
class RoundtripResult:
    """Result from roundtrip benchmark."""
    profile_true: np.ndarray      # (n_pixels, n_depth, n_elements)
    profile_recon: np.ndarray     # (n_pixels, n_depth, n_elements)
    I_obs: np.ndarray             # (n_pixels, n_angles, n_elements)
    I_recon: np.ndarray           # (n_pixels, n_angles, n_elements)

    depth: np.ndarray
    angles: np.ndarray
    elements: List[str]

    # Metrics
    rmse_per_pixel: np.ndarray    # (n_pixels,)
    chi2_per_pixel: np.ndarray    # (n_pixels,)
    success_rate: float

    # Timing
    total_time: float
    time_per_pixel: float

    @property
    def rmse_total(self) -> float:
        return float(np.sqrt(np.mean((self.profile_recon - self.profile_true)**2)))

    @property
    def rmse_image(self) -> np.ndarray:
        """RMSE as 2D image."""
        n_x = int(np.sqrt(len(self.rmse_per_pixel)))
        return self.rmse_per_pixel.reshape(n_x, n_x)


# ==============================================================================
# Profile Generation
# ==============================================================================

def create_5layer_profile(
    depth: np.ndarray,
    thicknesses: List[float],
    widths: List[float],
) -> np.ndarray:
    """
    Create a 5-layer profile: C / Al2O3 / TiO2 / SiO2 / Si

    Args:
        depth: Depth array in Angstroms
        thicknesses: [d_C, d_Al2O3, d_TiO2, d_SiO2] layer thicknesses
        widths: [w_C, w_Al2O3, w_TiO2, w_SiO2] interface widths

    Returns:
        profile: (n_depth, 5) array [O, C, Al, Ti, Si]
    """
    n_depth = len(depth)
    profile = np.zeros((n_depth, 5))  # O, C, Al, Ti, Si

    d_C, d_Al2O3, d_TiO2, d_SiO2 = thicknesses
    w_C, w_Al2O3, w_TiO2, w_SiO2 = widths

    # Cumulative depths
    z1 = d_C
    z2 = z1 + d_Al2O3
    z3 = z2 + d_TiO2
    z4 = z3 + d_SiO2

    def sigmoid(z, z0, w):
        return 1 / (1 + np.exp(-(z - z0) / max(w, 0.1)))

    for i, z in enumerate(depth):
        # Layer fractions
        f_C = 1 - sigmoid(z, z1, w_C)                              # C surface
        f_Al2O3 = sigmoid(z, z1, w_C) * (1 - sigmoid(z, z2, w_Al2O3))  # Al2O3
        f_TiO2 = sigmoid(z, z2, w_Al2O3) * (1 - sigmoid(z, z3, w_TiO2))  # TiO2
        f_SiO2 = sigmoid(z, z3, w_TiO2) * (1 - sigmoid(z, z4, w_SiO2))  # SiO2
        f_Si = sigmoid(z, z4, w_SiO2)                              # Si substrate

        # Element concentrations (stoichiometry)
        # O: from Al2O3 (3/5=0.6), TiO2 (2/3=0.667), SiO2 (2/3=0.667)
        c_O = 0.6 * f_Al2O3 + 0.667 * f_TiO2 + 0.667 * f_SiO2

        # C: contamination layer
        c_C = f_C

        # Al: from Al2O3 (2/5=0.4)
        c_Al = 0.4 * f_Al2O3

        # Ti: from TiO2 (1/3=0.333)
        c_Ti = 0.333 * f_TiO2

        # Si: from SiO2 (1/3=0.333) + Si substrate (1.0)
        c_Si = 0.333 * f_SiO2 + f_Si

        # Normalize
        total = c_O + c_C + c_Al + c_Ti + c_Si + 1e-10
        profile[i] = [c_O/total, c_C/total, c_Al/total, c_Ti/total, c_Si/total]

    return profile


def load_profile_from_h5(h5_path: str, element_indices: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load depth profile from H5 file.

    Args:
        h5_path: Path to H5 file
        element_indices: Indices of elements to extract (default: first 5)

    Returns:
        profile: (n_depth, n_elements)
        depth: depth array
    """
    with h5py.File(h5_path, 'r') as f:
        # Drlt: (batch, elements, depth) - take first batch
        Drlt = f['Drlt'][0, :, :]  # (elements, depth)

        if element_indices is not None:
            Drlt = Drlt[element_indices, :]
        else:
            # Default: first 5 elements
            Drlt = Drlt[:5, :]

        # Transpose to (depth, elements)
        profile = Drlt.T

        # Create depth array (assuming uniform spacing)
        n_depth = profile.shape[0]
        max_depth = 200.0  # Default assumption
        depth = np.linspace(0, max_depth, n_depth)

    return profile, depth


def generate_varied_profiles(
    config: RoundtripConfig,
    variation_type: str = 'thickness',
    base_thicknesses: List[float] = None,
    base_widths: List[float] = None,
) -> np.ndarray:
    """
    Generate spatially varying depth profiles for 2D image.

    Args:
        config: Roundtrip configuration
        variation_type: 'thickness', 'composition', or 'both'
        base_thicknesses: Base layer thicknesses
        base_widths: Base interface widths

    Returns:
        profiles: (n_pixels, n_depth, n_elements)
    """
    n_pixels = config.n_x * config.n_y
    depth = np.linspace(0, config.max_depth, config.n_depth)

    if base_thicknesses is None:
        base_thicknesses = [5.0, 15.0, 20.0, 15.0]  # C, Al2O3, TiO2, SiO2
    if base_widths is None:
        base_widths = [2.0, 3.0, 3.0, 5.0]

    profiles = np.zeros((n_pixels, config.n_depth, config.n_elements))

    for idx in range(n_pixels):
        # 2D position
        ix = idx % config.n_x
        iy = idx // config.n_x

        # Normalized position [0, 1] - handle n_x=1 or n_y=1 case
        nx = ix / (config.n_x - 1) if config.n_x > 1 else 0.5
        ny = iy / (config.n_y - 1) if config.n_y > 1 else 0.5

        # Vary thicknesses across the image
        if variation_type in ('thickness', 'both'):
            # Linear variation: top-left thin, bottom-right thick
            scale = 0.5 + 1.0 * (nx + ny) / 2  # 0.5 to 1.5
            thicknesses = [t * scale for t in base_thicknesses]
        else:
            thicknesses = base_thicknesses.copy()

        # Vary widths (roughness) for 'both' mode
        if variation_type == 'both':
            width_scale = 0.8 + 0.4 * nx
            widths = [w * width_scale for w in base_widths]
        else:
            widths = base_widths.copy()

        profiles[idx] = create_5layer_profile(depth, thicknesses, widths)

    return profiles


# ==============================================================================
# Forward Model
# ==============================================================================

def forward_model_batch(
    profiles: np.ndarray,
    tm: TransferMatrix,
    elements: List[ElementConfig],
    noise_level: float = 0.0,
) -> np.ndarray:
    """
    Apply forward model to batch of profiles with Poisson noise.

    Args:
        profiles: (n_pixels, n_depth, n_elements)
        tm: TransferMatrix
        elements: Element configurations
        noise_level: Poisson noise level (0-1, relative to max intensity)

    Returns:
        I_obs: (n_pixels, n_angles, n_elements)
    """
    n_pixels = profiles.shape[0]
    n_angles = len(tm.angles)
    n_elements = profiles.shape[2]

    I_obs = np.zeros((n_pixels, n_angles, n_elements))

    for idx in range(n_pixels):
        D = profiles[idx]  # (n_depth, n_elements)

        # Compute transfer matrix for this profile
        tm.compute(D, elements)

        # Forward model
        I = tm.forward(D, normalize=True)  # (n_angles, n_elements)

        # Add Poisson noise
        if noise_level > 0:
            # Scale to simulate photon counts
            scale = 1.0 / (noise_level ** 2)  # Higher scale = lower relative noise
            I_counts = I * scale

            # Poisson sampling
            I_noisy = np.random.poisson(np.maximum(I_counts, 1e-10).astype(np.float64))
            I = I_noisy / scale

            # Re-normalize
            I = np.clip(I, 1e-10, None)
            I = I / I.sum(axis=1, keepdims=True)

        I_obs[idx] = I

    return I_obs


# ==============================================================================
# Inverse Solver
# ==============================================================================

def _solve_single_pixel(args):
    """Worker function for parallel processing."""
    idx, I, tm_params, solver_config_dict, D_prior, elements_dict, solver_type = args

    # Reconstruct TransferMatrix (can't pickle the object directly)
    from deppro import (
        TransferMatrix, FreeSolver, SemiFreeSolver, SemiFreeSolverConfig,
        SolverConfig, ElementConfig
    )

    tm = TransferMatrix(
        depth=tm_params['depth'],
        step_size=tm_params['step_size'],
        angles=tm_params['angles'],
        photon_energy=tm_params['photon_energy'],
    )

    # Reconstruct elements
    elements = [ElementConfig(**e) for e in elements_dict]

    # Reconstruct solver config
    solver_config = SolverConfig(**solver_config_dict)

    try:
        if solver_type == 'semi-free':
            n_elements = len(elements)
            semi_config = SemiFreeSolverConfig(
                thickness_init=30.0,
                abruptness_init=5.0,
                free_indices=list(range(n_elements - 1)),  # All except substrate
                linked_indices=[],
                substrate_indices=[n_elements - 1],  # Si is substrate
                optimize_thickness=True,
                optimize_abruptness=False,
            )
            solver = SemiFreeSolver(tm, solver_config, semi_config, D_prior, elements)
        else:
            solver = FreeSolver(tm, solver_config, D_prior, elements)
        result = solver.solve(I)
        return idx, result.D_opt, result.chi2, result.success
    except Exception as e:
        return idx, D_prior, np.nan, False


def inverse_solve_batch(
    I_obs: np.ndarray,
    tm: TransferMatrix,
    elements: List[ElementConfig],
    config: RoundtripConfig,
    D_priors: np.ndarray = None,
    verbose: bool = False,
    n_workers: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve inverse problem for batch of observations (sequential).

    Args:
        I_obs: (n_pixels, n_angles, n_elements)
        tm: TransferMatrix
        elements: Element configurations
        config: Roundtrip configuration
        D_priors: (n_pixels, n_depth, n_elements) prior profiles
        verbose: Print progress
        n_workers: Not used (for API compatibility)

    Returns:
        profiles_recon: (n_pixels, n_depth, n_elements)
        chi2_values: (n_pixels,)
        success_flags: (n_pixels,)
    """
    n_pixels = I_obs.shape[0]
    n_depth = tm.n_depth
    n_elements = len(elements)

    profiles_recon = np.zeros((n_pixels, n_depth, n_elements))
    chi2_values = np.zeros(n_pixels)
    success_flags = np.zeros(n_pixels, dtype=bool)

    # Use recommended settings for stability
    solver_config = SolverConfig(
        method=config.method,
        alpha=config.alpha,
        max_iter=config.max_iter,
        use_sum_constraint=True,
        use_charge_constraint=True,  # Improve stability
        degree=1.0,  # Standard data fit weight
    )

    for idx in range(n_pixels):
        if verbose and idx % 100 == 0:
            print(f"  Processing pixel {idx}/{n_pixels}...")

        I = I_obs[idx]  # (n_angles, n_elements)

        # Prior - use uniform prior for fair test (not true profile)
        if D_priors is not None:
            # Use a smoothed version of prior, not the exact true profile
            D_prior = D_priors[idx].copy()
            # Add small uniform component to avoid getting stuck
            D_prior = 0.9 * D_prior + 0.1 / n_elements
            D_prior = D_prior / D_prior.sum(axis=1, keepdims=True)
        else:
            D_prior = np.ones((n_depth, n_elements)) / n_elements

        # Solve based on solver type
        try:
            if config.solver_type == 'semi-free':
                semi_config = SemiFreeSolverConfig(
                    thickness_init=30.0,
                    abruptness_init=5.0,
                    free_indices=list(range(n_elements - 1)),  # All except substrate
                    linked_indices=[],
                    substrate_indices=[n_elements - 1],  # Si is substrate
                    optimize_thickness=True,
                    optimize_abruptness=False,
                )
                solver = SemiFreeSolver(tm, solver_config, semi_config, D_prior, elements)
            else:
                solver = FreeSolver(tm, solver_config, D_prior, elements)
            result = solver.solve(I)

            profiles_recon[idx] = result.D_opt
            chi2_values[idx] = result.chi2
            success_flags[idx] = result.success
        except Exception as e:
            if verbose:
                print(f"  Pixel {idx} failed: {e}")
            profiles_recon[idx] = D_prior
            chi2_values[idx] = np.nan
            success_flags[idx] = False

    return profiles_recon, chi2_values, success_flags


def inverse_solve_batch_parallel(
    I_obs: np.ndarray,
    tm: TransferMatrix,
    elements: List[ElementConfig],
    config: RoundtripConfig,
    D_priors: np.ndarray = None,
    verbose: bool = False,
    n_workers: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve inverse problem for batch of observations (parallel).

    Args:
        I_obs: (n_pixels, n_angles, n_elements)
        tm: TransferMatrix
        elements: Element configurations
        config: Roundtrip configuration
        D_priors: (n_pixels, n_depth, n_elements) prior profiles
        verbose: Print progress
        n_workers: Number of parallel workers (default: CPU count - 1)

    Returns:
        profiles_recon: (n_pixels, n_depth, n_elements)
        chi2_values: (n_pixels,)
        success_flags: (n_pixels,)
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_pixels = I_obs.shape[0]
    n_depth = tm.n_depth
    n_elements = len(elements)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    profiles_recon = np.zeros((n_pixels, n_depth, n_elements))
    chi2_values = np.zeros(n_pixels)
    success_flags = np.zeros(n_pixels, dtype=bool)

    # Serialize TransferMatrix parameters
    tm_params = {
        'depth': tm.depth,
        'step_size': tm.step_size,
        'angles': tm.angles,
        'photon_energy': tm.photon_energy,
    }

    # Serialize solver config
    solver_config_dict = {
        'method': config.method,
        'alpha': config.alpha,
        'max_iter': config.max_iter,
        'use_sum_constraint': True,
        'use_charge_constraint': True,
        'degree': 1.0,
    }

    # Serialize elements
    elements_dict = [
        {'symbol': e.symbol, 'orbital': e.orbital, 'compound': e.compound,
         'binding_energy': e.binding_energy}
        for e in elements
    ]

    # Prepare arguments
    args_list = []
    for idx in range(n_pixels):
        if D_priors is not None:
            D_prior = D_priors[idx].copy()
            D_prior = 0.9 * D_prior + 0.1 / n_elements
            D_prior = D_prior / D_prior.sum(axis=1, keepdims=True)
        else:
            D_prior = np.ones((n_depth, n_elements)) / n_elements

        args_list.append((
            idx, I_obs[idx], tm_params, solver_config_dict, D_prior, elements_dict,
            config.solver_type
        ))

    # Process in parallel
    completed = 0
    if verbose:
        print(f"  Using {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_solve_single_pixel, args): args[0]
                  for args in args_list}

        for future in as_completed(futures):
            idx, D_opt, chi2, success = future.result()
            profiles_recon[idx] = D_opt
            chi2_values[idx] = chi2
            success_flags[idx] = success
            completed += 1

            if verbose and completed % 500 == 0:
                print(f"  Completed {completed}/{n_pixels}...")

    return profiles_recon, chi2_values, success_flags


# ==============================================================================
# Roundtrip Benchmark
# ==============================================================================

def run_roundtrip_benchmark(
    config: RoundtripConfig = None,
    h5_path: str = None,
    verbose: bool = True,
    save_dir: str = None,
) -> RoundtripResult:
    """
    Run complete roundtrip benchmark.

    Args:
        config: Benchmark configuration
        h5_path: Optional H5 file to load base profile from
        verbose: Print progress
        save_dir: Directory to save results

    Returns:
        RoundtripResult with all data and metrics
    """
    if config is None:
        config = RoundtripConfig()

    if verbose:
        print("=" * 60)
        print("ARXPS Roundtrip Benchmark")
        print("=" * 60)
        print(f"  Grid: {config.n_x} x {config.n_y} = {config.n_x * config.n_y} pixels")
        print(f"  Depth: 0-{config.max_depth} A, {config.n_depth} points")
        print(f"  Angles: {len(config.angles)} ({config.angles[0]}-{config.angles[-1]} deg)")
        print(f"  Elements: {config.n_elements}")
        print(f"  Noise: {config.noise_level*100:.1f}%")
        print(f"  Solver: {config.solver_type} / {config.method}")
        print(f"  Parallel: {config.parallel} (workers: {config.n_workers or 'auto'})")
        print()

    # Setup
    n_pixels = config.n_x * config.n_y
    depth = np.linspace(0, config.max_depth, config.n_depth)
    step_size = depth[1] - depth[0]
    elements = config.elements[:config.n_elements]
    element_names = [e.symbol for e in elements]

    # Transfer matrix
    tm = TransferMatrix(
        depth=depth,
        step_size=step_size,
        angles=config.angles,
        photon_energy=1486.6,
    )

    # Step 1: Generate profiles
    t0 = time.perf_counter()

    if verbose:
        print("1. Generating depth profiles...")

    if h5_path:
        # Load base profile and replicate with variations
        base_profile, _ = load_profile_from_h5(h5_path)
        # Interpolate to target depth resolution
        from scipy.interpolate import interp1d
        old_depth = np.linspace(0, config.max_depth, base_profile.shape[0])
        profiles_true = np.zeros((n_pixels, config.n_depth, config.n_elements))
        for elem_idx in range(min(base_profile.shape[1], config.n_elements)):
            f = interp1d(old_depth, base_profile[:, elem_idx], kind='linear', fill_value='extrapolate')
            for px in range(n_pixels):
                profiles_true[px, :, elem_idx] = f(depth)
        # Normalize
        profiles_true = np.clip(profiles_true, 0, None)
        sums = profiles_true.sum(axis=2, keepdims=True)
        profiles_true = profiles_true / np.maximum(sums, 1e-10)
    else:
        profiles_true = generate_varied_profiles(config, variation_type='thickness')

    t1 = time.perf_counter()
    if verbose:
        print(f"   Generated {n_pixels} profiles in {t1-t0:.2f}s")

    # Step 2: Forward model with noise
    if verbose:
        print("2. Computing forward model with Poisson noise...")

    I_obs = forward_model_batch(profiles_true, tm, elements, config.noise_level)

    t2 = time.perf_counter()
    if verbose:
        print(f"   Forward model: {t2-t1:.2f}s")

    # Step 3: Inverse solver
    if verbose:
        print("3. Solving inverse problem...")

    if config.parallel:
        profiles_recon, chi2_values, success_flags = inverse_solve_batch_parallel(
            I_obs, tm, elements, config, D_priors=profiles_true.copy(),
            verbose=verbose, n_workers=config.n_workers
        )
    else:
        profiles_recon, chi2_values, success_flags = inverse_solve_batch(
            I_obs, tm, elements, config, D_priors=profiles_true.copy(), verbose=verbose
        )

    t3 = time.perf_counter()
    if verbose:
        print(f"   Inverse solve: {t3-t2:.2f}s ({(t3-t2)/n_pixels*1000:.2f} ms/pixel)")

    # Step 4: Compute reconstructed intensities
    if verbose:
        print("4. Computing reconstructed intensities...")

    I_recon = np.zeros_like(I_obs)
    for idx in range(n_pixels):
        tm.compute(profiles_recon[idx], elements)
        I_recon[idx] = tm.forward(profiles_recon[idx], normalize=True)

    t4 = time.perf_counter()

    # Step 5: Compute metrics
    if verbose:
        print("5. Computing metrics...")

    rmse_per_pixel = np.sqrt(np.mean((profiles_recon - profiles_true)**2, axis=(1, 2)))
    success_rate = np.mean(success_flags)

    total_time = t4 - t0
    time_per_pixel = total_time / n_pixels

    if verbose:
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"  Total RMSE: {np.mean(rmse_per_pixel):.4f}")
        print(f"  Min/Max RMSE: {rmse_per_pixel.min():.4f} / {rmse_per_pixel.max():.4f}")
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Time per pixel: {time_per_pixel*1000:.2f} ms")

    result = RoundtripResult(
        profile_true=profiles_true,
        profile_recon=profiles_recon,
        I_obs=I_obs,
        I_recon=I_recon,
        depth=depth,
        angles=config.angles,
        elements=element_names,
        rmse_per_pixel=rmse_per_pixel,
        chi2_per_pixel=chi2_values,
        success_rate=success_rate,
        total_time=total_time,
        time_per_pixel=time_per_pixel,
    )

    # Save results
    if save_dir:
        save_results(result, config, save_dir, verbose)

    return result


# ==============================================================================
# Visualization
# ==============================================================================

def plot_roundtrip_result(
    result: RoundtripResult,
    config: RoundtripConfig,
    output_path: str = None,
    show: bool = True,
):
    """
    Create visualization of roundtrip benchmark results.

    Shows:
    - RMSE image (100x100)
    - Sample profile comparisons
    - Intensity residuals
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.2, 1, 1])

    n_x = config.n_x
    n_y = config.n_y
    n_elements = len(result.elements)

    # 1. RMSE Image (top row, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    rmse_img = result.rmse_per_pixel.reshape(n_y, n_x)
    im = ax1.imshow(rmse_img, cmap='viridis', origin='lower')
    ax1.set_title(f'RMSE per Pixel (Mean: {np.mean(rmse_img):.4f})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im, ax=ax1, label='RMSE')

    # 2. Chi2 Image
    ax2 = fig.add_subplot(gs[0, 2:])
    chi2_img = result.chi2_per_pixel.reshape(n_y, n_x)
    chi2_img_clipped = np.clip(chi2_img, 0, np.nanpercentile(chi2_img, 95))
    im2 = ax2.imshow(chi2_img_clipped, cmap='hot', origin='lower')
    ax2.set_title(f'Chi2 per Pixel (Mean: {np.nanmean(chi2_img):.4f})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Chi2')

    # 3. Sample profiles (middle row)
    sample_indices = [0, n_x//2, n_x*n_y//2, n_x*n_y-1]  # corners and center
    colors = plt.cm.Set1(np.linspace(0, 1, n_elements))

    for i, (idx, ax_pos) in enumerate(zip(sample_indices, range(4))):
        ax = fig.add_subplot(gs[1, ax_pos])

        ix = idx % n_x
        iy = idx // n_x

        for e in range(n_elements):
            ax.plot(result.depth, result.profile_true[idx, :, e], '-',
                   color=colors[e], linewidth=1.5, label=result.elements[e])
            ax.plot(result.depth, result.profile_recon[idx, :, e], '--',
                   color=colors[e], linewidth=1.5, alpha=0.7)

        ax.set_title(f'Pixel ({ix}, {iy})\nRMSE={result.rmse_per_pixel[idx]:.4f}')
        ax.set_xlabel('Depth (A)')
        ax.set_ylabel('Concentration')
        ax.set_xlim([0, result.depth[-1]])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    # 4. Element concentration images (bottom row)
    for e in range(min(4, n_elements)):
        ax = fig.add_subplot(gs[2, e])

        # Average concentration at surface (depth 0-10 A)
        surface_idx = result.depth < 10
        conc_surface_true = result.profile_true[:, surface_idx, e].mean(axis=1)
        conc_surface_recon = result.profile_recon[:, surface_idx, e].mean(axis=1)

        diff = (conc_surface_recon - conc_surface_true).reshape(n_y, n_x)
        vmax = np.abs(diff).max()

        im = ax.imshow(diff, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{result.elements[e]} Surface Diff')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)

    plt.suptitle(
        f'Roundtrip Benchmark: {config.n_x}x{config.n_y}, Noise={config.noise_level*100:.0f}%',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    if show:
        plt.show()

    return fig


def plot_3d_isosurface(
    result: RoundtripResult,
    config: RoundtripConfig,
    element_idx: int = 0,
    iso_values: List[float] = None,
    output_path: str = None,
    show: bool = True,
):
    """
    Create 3D isosurface visualization of depth profile.

    Shows the spatial distribution of element concentration as 3D isosurfaces.

    Args:
        result: RoundtripResult from benchmark
        config: RoundtripConfig
        element_idx: Index of element to visualize
        iso_values: List of isosurface values (default: [0.2, 0.4, 0.6, 0.8])
        output_path: Path to save PNG
        show: Show interactive viewer
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed. Install with: pip install pyvista")
        return None

    if iso_values is None:
        iso_values = [0.2, 0.4, 0.6, 0.8]

    n_x = config.n_x
    n_y = config.n_y
    n_depth = config.n_depth

    # Reshape profile to 3D volume: (x, y, z)
    # profile_recon: (n_pixels, n_depth, n_elements)
    volume = result.profile_recon[:, :, element_idx].reshape(n_x, n_y, n_depth)

    # Create coordinate arrays
    x = np.arange(n_x)
    y = np.arange(n_y)
    z = result.depth  # Depth in Angstroms

    # Create structured grid
    grid = pv.RectilinearGrid(x, y, z)
    grid["concentration"] = volume.flatten(order='F')

    # Create plotter
    plotter = pv.Plotter(off_screen=not show)
    plotter.set_background('white')

    # Add isosurfaces with different opacities
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(iso_values)))

    for i, iso_val in enumerate(iso_values):
        try:
            contour = grid.contour([iso_val], scalars="concentration")
            if contour.n_points > 0:
                plotter.add_mesh(
                    contour,
                    color=colors[i][:3],
                    opacity=0.3 + 0.15 * i,
                    label=f'{result.elements[element_idx]}={iso_val:.1f}'
                )
        except Exception:
            pass  # Skip if contour fails

    # Add bounding box
    plotter.add_mesh(grid.outline(), color='black', line_width=1)

    # Labels
    plotter.add_axes()
    plotter.add_title(f'{result.elements[element_idx]} Concentration Isosurfaces')

    if output_path:
        plotter.screenshot(output_path)
        print(f"Saved 3D view to {output_path}")

    if show:
        plotter.show()

    return plotter


def plot_3d_slices(
    result: RoundtripResult,
    config: RoundtripConfig,
    element_idx: int = 0,
    output_path: str = None,
    show: bool = True,
):
    """
    Create 3D visualization with orthogonal slices through the volume.

    Args:
        result: RoundtripResult from benchmark
        config: RoundtripConfig
        element_idx: Index of element to visualize
        output_path: Path to save PNG
        show: Show interactive viewer
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed")
        return None

    n_x = config.n_x
    n_y = config.n_y
    n_depth = config.n_depth

    # Reshape to 3D: (x, y, z)
    volume = result.profile_recon[:, :, element_idx].reshape(n_x, n_y, n_depth)

    # Create grid
    x = np.arange(n_x)
    y = np.arange(n_y)
    z = result.depth

    grid = pv.RectilinearGrid(x, y, z)
    grid["concentration"] = volume.flatten(order='F')

    plotter = pv.Plotter(off_screen=not show)
    plotter.set_background('white')

    # Add orthogonal slices
    slices = grid.slice_orthogonal(x=n_x//2, y=n_y//2, z=result.depth[n_depth//4])
    plotter.add_mesh(slices, cmap='viridis', clim=[0, 1])

    # Add semi-transparent volume rendering
    plotter.add_volume(
        grid,
        scalars="concentration",
        cmap='viridis',
        opacity='sigmoid',
        shade=True,
    )

    plotter.add_axes()
    plotter.add_title(f'{result.elements[element_idx]} 3D Volume')

    if output_path:
        plotter.screenshot(output_path)
        print(f"Saved 3D view to {output_path}")

    if show:
        plotter.show()

    return plotter


def plot_film_structure_3d(
    result: RoundtripResult,
    config: RoundtripConfig,
    output_path: str = None,
    show: bool = True,
):
    """
    Create 3D visualization showing film layer structure.

    Uses isocaps to show where each element dominates, creating a
    realistic "film stack" appearance.

    Args:
        result: RoundtripResult from benchmark
        config: RoundtripConfig
        output_path: Path to save PNG
        show: Show interactive viewer
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed")
        return None

    n_x = config.n_x
    n_y = config.n_y
    n_depth = config.n_depth
    n_elements = len(result.elements)

    # Create plotter
    plotter = pv.Plotter(off_screen=not show)
    plotter.set_background('white')

    # Element colors
    element_colors = {
        'O': 'red',
        'C': 'gray',
        'Al': 'silver',
        'Ti': 'purple',
        'Si': 'blue',
    }

    x = np.arange(n_x)
    y = np.arange(n_y)
    z = result.depth

    # Add each element as isosurface where it's dominant (> threshold)
    threshold = 0.3  # Show where element > 30%

    for elem_idx in range(min(n_elements, 5)):
        volume = result.profile_recon[:, :, elem_idx].reshape(n_x, n_y, n_depth)

        grid = pv.RectilinearGrid(x, y, z)
        grid["conc"] = volume.flatten(order='F')

        elem_name = result.elements[elem_idx]
        color = element_colors.get(elem_name, 'white')

        try:
            # Create isosurface at threshold
            contour = grid.contour([threshold], scalars="conc")
            if contour.n_points > 0:
                plotter.add_mesh(
                    contour,
                    color=color,
                    opacity=0.6,
                    label=elem_name,
                    smooth_shading=True,
                )
        except Exception:
            pass

    # Add bounding box
    bbox = pv.RectilinearGrid(x, y, z)
    plotter.add_mesh(bbox.outline(), color='black', line_width=2)

    # Labels and legend
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Depth (A)')
    plotter.add_legend()
    plotter.add_title('Film Layer Structure')

    # Set good viewing angle
    plotter.camera_position = [(n_x*2, n_y*2, -z.max()*1.5),
                               (n_x/2, n_y/2, z.max()/2),
                               (0, 0, -1)]

    if output_path:
        plotter.screenshot(output_path)
        print(f"Saved 3D film structure to {output_path}")

    if show:
        plotter.show()

    return plotter


def plot_profile_comparison(
    result: RoundtripResult,
    config: RoundtripConfig,
    n_profiles: int = 10,
    output_path: str = None,
    show: bool = True,
    D_initial: np.ndarray = None,
):
    """
    Plot comparison of true vs reconstructed vs initial profiles.

    Args:
        result: RoundtripResult from benchmark
        config: RoundtripConfig
        n_profiles: Number of profiles to plot
        output_path: Path to save PNG
        show: Show plot
        D_initial: Initial/prior profiles (n_pixels, n_depth, n_elements)
    """
    n_pixels = result.profile_true.shape[0]
    n_elements = len(result.elements)

    # Select profiles spread across the image
    if n_pixels <= n_profiles:
        indices = list(range(n_pixels))
    else:
        indices = np.linspace(0, n_pixels - 1, n_profiles, dtype=int)

    # Create figure: n_profiles rows x n_elements columns
    fig, axes = plt.subplots(n_profiles, n_elements, figsize=(3 * n_elements, 2.5 * n_profiles))
    if n_profiles == 1:
        axes = axes.reshape(1, -1)
    if n_elements == 1:
        axes = axes.reshape(-1, 1)

    for row, idx in enumerate(indices):
        ix = idx % config.n_x if config.n_x > 1 else 0
        iy = idx // config.n_x if config.n_x > 0 else 0
        rmse = result.rmse_per_pixel[idx]

        for col in range(n_elements):
            ax = axes[row, col]

            # Plot true profile
            ax.plot(result.depth, result.profile_true[idx, :, col],
                   'b-', linewidth=2, label='True')

            # Plot reconstructed profile
            ax.plot(result.depth, result.profile_recon[idx, :, col],
                   'r--', linewidth=2, label='Recon')

            # Plot initial/prior profile if provided
            if D_initial is not None:
                ax.plot(result.depth, D_initial[idx, :, col],
                       'g:', linewidth=1.5, label='Initial')

            ax.set_xlim([0, result.depth[-1]])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)

            # Labels
            if row == 0:
                ax.set_title(result.elements[col], fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'Px({ix},{iy})\nRMSE={rmse:.3f}', fontsize=9)
            if row == n_profiles - 1:
                ax.set_xlabel('Depth (Å)')

            # Legend only on first plot
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')

    plt.suptitle(
        f'Profile Comparison: {config.solver_type} / {config.method} (α={config.alpha})\n'
        f'Total RMSE: {np.mean(result.rmse_per_pixel):.4f}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved profile comparison to {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def run_comparison_test(
    n_profiles: int = 10,
    noise_level: float = 0.01,
    output_dir: str = None,
    verbose: bool = True,
):
    """
    Run comparison test with FreeSolver+L2 and SemiFreeSolver+L1.

    Shows profiles for each solver to verify correctness.

    Args:
        n_profiles: Number of profiles to test
        noise_level: Poisson noise level
        output_dir: Directory to save results
        verbose: Print progress
    """
    from pathlib import Path

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Profile Comparison Test")
    print("=" * 70)

    # Common configuration
    base_config = {
        'n_x': n_profiles,
        'n_y': 1,  # 1D array of profiles
        'n_depth': 100,
        'max_depth': 200.0,
        'noise_level': noise_level,
        'n_elements': 5,
        'parallel': False,
    }

    # Generate profiles once
    config_gen = RoundtripConfig(**base_config)
    depth = np.linspace(0, config_gen.max_depth, config_gen.n_depth)
    step_size = depth[1] - depth[0]

    profiles_true = generate_varied_profiles(config_gen, variation_type='thickness')

    # Create transfer matrix
    tm = TransferMatrix(
        depth=depth,
        step_size=step_size,
        angles=config_gen.angles,
        photon_energy=1486.6,
    )
    elements = config_gen.elements[:config_gen.n_elements]

    # Forward model with noise
    print(f"\nGenerating {n_profiles} profiles with {noise_level*100:.0f}% noise...")
    I_obs = forward_model_batch(profiles_true, tm, elements, noise_level)

    # Create uniform initial prior
    n_elements = config_gen.n_elements
    n_depth = config_gen.n_depth
    D_initial = np.ones((n_profiles, n_depth, n_elements)) / n_elements

    results = {}

    # Test 1: FreeSolver + L2
    print("\n" + "-" * 40)
    print("Test 1: FreeSolver + L2 (α=0.01)")
    print("-" * 40)

    config1 = RoundtripConfig(
        **base_config,
        solver_type='free',
        method='L2',
        alpha=0.01,
    )

    profiles_recon1, chi2_1, success_1 = inverse_solve_batch(
        I_obs, tm, elements, config1,
        D_priors=profiles_true.copy(),
        verbose=verbose
    )

    rmse1 = np.sqrt(np.mean((profiles_recon1 - profiles_true)**2, axis=(1, 2)))
    print(f"  RMSE: {np.mean(rmse1):.4f} (min: {rmse1.min():.4f}, max: {rmse1.max():.4f})")
    print(f"  Success rate: {np.mean(success_1)*100:.1f}%")

    result1 = RoundtripResult(
        profile_true=profiles_true,
        profile_recon=profiles_recon1,
        I_obs=I_obs,
        I_recon=np.zeros_like(I_obs),  # Not used
        depth=depth,
        angles=config1.angles,
        elements=[e.symbol for e in elements],
        rmse_per_pixel=rmse1,
        chi2_per_pixel=chi2_1,
        success_rate=np.mean(success_1),
        total_time=0,
        time_per_pixel=0,
    )
    results['FreeSolver_L2'] = result1

    if output_dir:
        plot_profile_comparison(
            result1, config1, n_profiles=n_profiles,
            output_path=str(output_dir / 'comparison_free_L2.png'),
            show=False,
            D_initial=D_initial,
        )

    # Test 2: SemiFreeSolver + L1
    print("\n" + "-" * 40)
    print("Test 2: SemiFreeSolver + L1 (α=0.01)")
    print("-" * 40)

    config2 = RoundtripConfig(
        **base_config,
        solver_type='semi-free',
        method='L1',
        alpha=0.01,
    )

    profiles_recon2, chi2_2, success_2 = inverse_solve_batch(
        I_obs, tm, elements, config2,
        D_priors=profiles_true.copy(),
        verbose=verbose
    )

    rmse2 = np.sqrt(np.mean((profiles_recon2 - profiles_true)**2, axis=(1, 2)))
    print(f"  RMSE: {np.mean(rmse2):.4f} (min: {rmse2.min():.4f}, max: {rmse2.max():.4f})")
    print(f"  Success rate: {np.mean(success_2)*100:.1f}%")

    result2 = RoundtripResult(
        profile_true=profiles_true,
        profile_recon=profiles_recon2,
        I_obs=I_obs,
        I_recon=np.zeros_like(I_obs),
        depth=depth,
        angles=config2.angles,
        elements=[e.symbol for e in elements],
        rmse_per_pixel=rmse2,
        chi2_per_pixel=chi2_2,
        success_rate=np.mean(success_2),
        total_time=0,
        time_per_pixel=0,
    )
    results['SemiFreeSolver_L1'] = result2

    if output_dir:
        plot_profile_comparison(
            result2, config2, n_profiles=n_profiles,
            output_path=str(output_dir / 'comparison_semi_L1.png'),
            show=False,
            D_initial=D_initial,
        )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Solver':<25} {'RMSE':>10} {'Success':>10}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<25} {res.rmse_total:>10.4f} {res.success_rate*100:>9.1f}%")

    return results


def save_results(
    result: RoundtripResult,
    config: RoundtripConfig,
    save_dir: str,
    verbose: bool = True,
):
    """Save benchmark results to HDF5 and PNG."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save H5
    h5_path = save_dir / 'roundtrip_result.h5'
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('profile_true', data=result.profile_true.astype(np.float32))
        f.create_dataset('profile_recon', data=result.profile_recon.astype(np.float32))
        f.create_dataset('I_obs', data=result.I_obs.astype(np.float32))
        f.create_dataset('I_recon', data=result.I_recon.astype(np.float32))
        f.create_dataset('depth', data=result.depth)
        f.create_dataset('angles', data=result.angles)
        f.create_dataset('rmse_per_pixel', data=result.rmse_per_pixel)
        f.create_dataset('chi2_per_pixel', data=result.chi2_per_pixel)

        f.attrs['n_x'] = config.n_x
        f.attrs['n_y'] = config.n_y
        f.attrs['noise_level'] = config.noise_level
        f.attrs['method'] = config.method
        f.attrs['alpha'] = config.alpha
        f.attrs['rmse_total'] = result.rmse_total
        f.attrs['success_rate'] = result.success_rate
        f.attrs['total_time'] = result.total_time

    if verbose:
        print(f"Saved H5 to {h5_path}")

    # Save PNG
    png_path = save_dir / 'roundtrip_result.png'
    plot_roundtrip_result(result, config, output_path=str(png_path), show=False)


# ==============================================================================
# CLI
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='ARXPS Roundtrip Benchmark')
    parser.add_argument('--quick', action='store_true', help='Quick test (20x20)')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison test (FreeSolver+L2 vs SemiFreeSolver+L1)')
    parser.add_argument('--nx', type=int, default=100, help='X grid size')
    parser.add_argument('--ny', type=int, default=100, help='Y grid size')
    parser.add_argument('--noise', type=float, default=0.01, help='Noise level (0-1)')
    parser.add_argument('--h5', type=str, help='H5 file with base profile')
    parser.add_argument('--output', '-o', type=str, help='Output directory')
    parser.add_argument('--method', type=str, default='L2', choices=['MEM', 'L2', 'L1', 'LLS'])
    parser.add_argument('--alpha', type=float, default=0.01, help='Regularization')
    parser.add_argument('--solver', type=str, default='free',
                       choices=['free', 'semi-free'], help='Solver type')
    parser.add_argument('--parallel', '-p', action='store_true', help='Use parallel processing')
    parser.add_argument('--workers', '-w', type=int, default=None, help='Number of workers')
    parser.add_argument('--no-show', action='store_true', help='Do not show plot')

    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        n_profiles = args.nx if args.nx <= 20 else 10
        output_dir = args.output or '/tmp/roundtrip_compare'
        return run_comparison_test(
            n_profiles=n_profiles,
            noise_level=args.noise,
            output_dir=output_dir,
            verbose=True,
        )

    # Quick mode
    if args.quick:
        args.nx = 20
        args.ny = 20
        args.output = args.output or '/tmp/roundtrip_quick'

    config = RoundtripConfig(
        n_x=args.nx,
        n_y=args.ny,
        noise_level=args.noise,
        solver_type=args.solver,
        method=args.method,
        alpha=args.alpha,
        parallel=args.parallel,
        n_workers=args.workers,
    )

    result = run_roundtrip_benchmark(
        config=config,
        h5_path=args.h5,
        verbose=True,
        save_dir=args.output,
    )

    if not args.no_show and args.output:
        plot_roundtrip_result(result, config, show=True)

    return result


if __name__ == '__main__':
    main()
