"""
Simulation Configuration
========================

Dataclasses for configuring the Dstk simulator.
"""

from dataclasses import dataclass, field
from typing import Tuple, Literal


@dataclass
class SimulationConfig:
    """Top-level configuration for Dstk simulator.

    Parameters
    ----------
    n_samples : int
        Number of random profiles to generate.
    seed : int
        Random seed for reproducibility.

    Depth Grid
    ----------
    depth_end : float
        Maximum depth in Angstroms.
    step_size : float
        Depth step size in Angstroms.
        n_depth = int(depth_end / step_size) + 1

    Physics
    -------
    photon_energy : float
        X-ray photon energy in eV. 1486.6 for Al Ka, 8000 for HAXPES.
    angles : tuple of float
        Observation angles in degrees.
    work_function : float
        Work function in eV.
    gamma_eff : float
        Effective gamma factor (typically 1.0).

    Sampling Ranges
    ---------------
    n_layers_range : (int, int)
        Min/max number of overlay film layers.
    thickness_range : (float, float)
        Min/max layer thickness in Angstroms.
    abruptness_range : (float, float)
        Min/max interface width FWHM in Angstroms.
    min_conc : float
        Minimum concentration floor for numerical stability.
    surface_contamination_prob : float
        Probability of adding surface C contamination layer.
    contamination_decay_range : (float, float)
        Min/max decay length for C contamination in Angstroms.

    Noise
    -----
    noise_level : float
        Noise level (0 = no noise). For relative: fractional std dev.
    noise_type : str
        'relative' (multiplicative) or 'poisson' (shot noise).

    Output
    ------
    output_path : str
        Output file path.
    output_format : str
        'parquet', 'h5', or 'both'.
    batch_size : int
        Samples per write batch (controls memory usage).
    """

    # Generation
    n_samples: int = 10_000
    seed: int = 42

    # Depth grid
    depth_end: float = 100.0
    step_size: float = 5.0

    # Physics
    photon_energy: float = 1486.6
    angles: Tuple[float, ...] = (0, 15, 30, 45, 60, 75)
    work_function: float = 4.5
    gamma_eff: float = 1.0

    # Sampling ranges
    n_layers_range: Tuple[int, int] = (1, 4)
    thickness_range: Tuple[float, float] = (10, 500)
    abruptness_range: Tuple[float, float] = (3, 15)
    max_abruptness_ratio: float = 0.3  # abruptness <= thickness * ratio (Cumpson limit)
    min_conc: float = 1e-6
    surface_contamination_prob: float = 0.7
    contamination_decay_range: Tuple[float, float] = (2, 10)

    # Noise
    noise_level: float = 0.0
    noise_type: Literal['relative', 'poisson'] = 'relative'

    # Output
    output_path: str = 'train.parquet'
    output_format: Literal['parquet', 'h5', 'both'] = 'parquet'
    batch_size: int = 10_000
