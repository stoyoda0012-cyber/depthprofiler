"""
HAXPES Experiment Configuration
================================

Centralized configuration dataclasses for AR-HAXPES experiments.
Replaces constants that were hardcoded across analysis scripts.

Three-level configuration:
    HAXPESInstrumentConfig: Hardware parameters (fixed per beamline)
    HAXPESExperimentConfig: Sample/element parameters (fixed per sample)
    HAXPESSolverConfig: Tunable analysis parameters

Pre-built instances:
    FIX_IBW_INSTRUMENT, FIX_PXT_INSTRUMENT: Scienta Omicron Angular56 analyzer, fix mode,
        IBW export (1001 detector channels) and pre-binned PXT export (12 angle channels)
    CALTIOSI_EXPERIMENT: C/Al₂O₃/TiO₂/Si multilayer sample
"""
import functools
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from deppro.depth_profile import ElementConfig
from deppro.paths import env_path_or_placeholder


# ============================================================
# Instrument-level config (frozen: hardware doesn't change)
# ============================================================

#: The polar angle from the analyzer's central (lens) axis, detector angle
#: ang = 0, to the sample normal. Emission angle = ang + this value. The sample
#: is rotated about the polar axis only. It is NOT the center of the retained
#: bins: the retained window (996 of 1001 channels) is off-center by 0.184 deg.
#: Its value is 30 deg. Earlier analyses and their deposited arrays and figures
#: were made with 26.0 and carry 26-based angle labels (see
#: docs/REPRODUCTION_JVST_SIA.md). The formula
#: ``flip(detector) + polar_offset`` is unchanged; the value also moves any
#: mask applied in emission-angle space (e.g. 5-55 deg), so the selected bins
#: change with it.
#:
#: It used to be called ``SAMPLE_TILT_DEG``; that name is kept as a deprecated
#: alias. The sample's tilt axis is a different rotation, see
#: ``deppro.haxpes.geometry``.
SAMPLE_POLAR_DEG = 30.0

_DEPRECATED_NAMES = {"SAMPLE_TILT_DEG": "SAMPLE_POLAR_DEG"}


def _polar_offset_from(polar_offset, tilt, where):
    """``polar_offset``, or the deprecated ``tilt=`` keyword of ``where``."""
    if tilt is None:
        return polar_offset
    warnings.warn(f"{where}(tilt=...) is deprecated; use polar_offset=...",
                  DeprecationWarning, stacklevel=3)
    if polar_offset != SAMPLE_POLAR_DEG and polar_offset != tilt:
        raise TypeError(f"{where}: give polar_offset or the deprecated tilt, not both")
    # A polar_offset equal to the default cannot be told from an omitted one, so
    # polar_offset=SAMPLE_POLAR_DEG together with tilt=x silently uses x.
    return tilt


def __getattr__(name):
    """Deprecated module-level aliases (``SAMPLE_TILT_DEG``)."""
    if name in _DEPRECATED_NAMES:
        new = _DEPRECATED_NAMES[name]
        warnings.warn(f"deppro.haxpes.config.{name} is deprecated; use {new}",
                      DeprecationWarning, stacklevel=2)
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class HAXPESInstrumentConfig:
    """Instrument-level parameters (fixed per beamline/analyzer).

    Attributes:
        photon_energy: Excitation energy in eV (e.g., 9251.74 for Ga Ka).
        polar_offset: polar angle [degrees] from the analyzer central axis (lens axis,
                detector angle ang = 0) to the sample normal. Emission = ang + polar_offset.
                Note: it does not coincide with the center angle of the kept bins (0.18 deg lower, from the 996/1001 channels).
        tilt_angle: old name of ``polar_offset`` (deprecated; not a dataclass field).
                Passing ``tilt_angle=`` at construction emits a DeprecationWarning and stores
                it in ``polar_offset``. Passing it together with a different ``polar_offset``
                is a TypeError. Reading ``.tilt_angle`` also emits a DeprecationWarning and
                returns ``polar_offset``.
        detector_range: Detector angular range (lo, hi) in degrees.
        n_detector_channels: Number of detector pixels (1001 for Angular56).
        n_prebinned_angles: Number of pre-binned angle channels (12 for PXT).
        detector_bin_lo: Pre-binned detector lower edge.
        detector_bin_hi: Pre-binned detector upper edge.
    """
    photon_energy: float = 9251.74
    polar_offset: float = SAMPLE_POLAR_DEG
    detector_range: Tuple[float, float] = (-36.84, 36.84)
    n_detector_channels: int = 1001
    n_prebinned_angles: int = 12
    detector_bin_lo: float = -33.82
    detector_bin_hi: float = 33.45

    def emission_angles_from_detector(
        self, n_channels: Optional[int] = None
    ) -> np.ndarray:
        """Compute emission angles from raw detector channel positions.

        For Fix mode with 1001 detector pixels.
        """
        n = n_channels or self.n_detector_channels
        det_angle = np.linspace(self.detector_range[0],
                                self.detector_range[1], n)
        # flip is a reinterpretation of detector rows as analyzer channels.
        # The 2D detector is mounted below the analyzer, so the row order is reversed relative to the
        # angular-dispersion axis (not needed for a top mount). It is not a sign flip of the angle (polar_offset - det).
        # Here, though, the default full 1001-channel grid being flipped (±36.84 deg) is symmetric about 0, so at this point the two coincide.
        # They differ on the 12-bin pre-binned grid (emission_angles_prebinned, load_prebinned_pxt), which comes
        # from 996/1001 channels and is off-center by 2.5 ch = 0.184 deg; read as a sign flip, every bin
        # shifts by a uniform 0.368 deg.
        return np.flip(det_angle) + self.polar_offset

    def emission_angles_prebinned(self) -> np.ndarray:
        """Compute emission angles for pre-binned PXT data (12 channels).

        Uses detector_bin_lo/hi as bin centers, matching PXT file x_ini[1].
        The pseudo-GT, DNN models, and benchmarks of the original analyses
        were computed with this definition, which was recorded there as the
        correct interpretation; the bin-edge offset previously applied here
        was judged incorrect.
        """
        det_centers = np.linspace(
            self.detector_bin_lo,
            self.detector_bin_hi,
            self.n_prebinned_angles,
        )
        # flip is a reinterpretation of detector rows as analyzer channels.
        # The 2D detector is mounted below the analyzer, so the row order is reversed relative to the
        # angular-dispersion axis (not needed for a top mount). It is not a sign flip of the angle (polar_offset - det).
        # The grid comes from 996/1001 channels and is off-center by 2.5 ch = 0.184 deg, so the two do not coincide.
        # Reading it as a sign flip shifts every bin by a uniform 0.368 deg.
        return np.flip(det_centers) + self.polar_offset



# The deprecated name tilt_angle is handled outside the dataclass, so that it is
# not a field: dataclasses.replace() and asdict() then see polar_offset only.
_instrument_init = HAXPESInstrumentConfig.__init__


@functools.wraps(_instrument_init)
def _instrument_init_with_alias(self, *args, tilt_angle=None, **kwargs):
    if tilt_angle is not None:
        warnings.warn("HAXPESInstrumentConfig(tilt_angle=...) is deprecated; "
                      "use polar_offset=...", DeprecationWarning, stacklevel=2)
        if len(args) >= 2:  # photon_energy, polar_offset given positionally
            given = args[1]
        else:
            given = kwargs.get("polar_offset", tilt_angle)
        if given != tilt_angle:
            raise TypeError("give polar_offset or the deprecated tilt_angle, not both")
        if len(args) < 2:
            kwargs["polar_offset"] = float(tilt_angle)
    _instrument_init(self, *args, **kwargs)


def _deprecated_tilt_angle(self):
    warnings.warn("HAXPESInstrumentConfig.tilt_angle is deprecated; use polar_offset",
                  DeprecationWarning, stacklevel=2)
    return self.polar_offset


HAXPESInstrumentConfig.__init__ = _instrument_init_with_alias
HAXPESInstrumentConfig.tilt_angle = property(_deprecated_tilt_angle)

# ============================================================
# Experiment-level config (sample + elements)
# ============================================================

@dataclass
class HAXPESExperimentConfig:
    """Experiment-level parameters (fixed per sample/measurement).

    Attributes:
        name: Experiment identifier.
        elem_names: Ordered element names.
        elem_colors: Plotting colors per element.
        ibw_patterns: IBW filename patterns {elem -> label}.
        region_to_elem: PXT region name -> element name.
        step_times_ms: Exposure time per element in milliseconds.
        free_indices: Element indices freely optimized in SemiFreeSolver.
        linked_indices: Element indices determined by charge balance.
        substrate_indices: Substrate element indices.
        substrate_element: Substrate element name for Si-normalization.
    """
    name: str = 'C_AlOx_TiOx_Si'
    elem_names: List[str] = field(
        default_factory=lambda: ['Si', 'Ti', 'Al', 'C', 'O']
    )
    elem_colors: Dict[str, str] = field(default_factory=lambda: {
        'Si': '#1f77b4', 'Ti': '#d62728', 'Al': '#2ca02c',
        'C': '#333333', 'O': '#ff7f0e',
    })
    ibw_patterns: Dict[str, str] = field(default_factory=lambda: {
        'Si': 'Si 1s', 'Ti': 'Ti 1s', 'Al': 'Al 1s',
        'C': 'C 1s', 'O': 'O 1s',
    })
    region_to_elem: Dict[str, str] = field(default_factory=lambda: {
        'C 1s': 'C', 'O 1s': 'O', 'Al 1s': 'Al',
        'Si 1s': 'Si', 'Ti 1s': 'Ti',
    })
    step_times_ms: Dict[str, int] = field(default_factory=lambda: {
        'Si': 20000, 'Ti': 10000, 'Al': 20000, 'C': 40000, 'O': 30000,
    })
    # SemiFreeSolver element classification
    # Default order: [Si=0, Ti=1, Al=2, C=3, O=4]
    free_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    linked_indices: List[int] = field(default_factory=lambda: [4])
    substrate_indices: List[int] = field(default_factory=lambda: [0])
    substrate_element: str = 'Si'

    def make_elements(self) -> List[ElementConfig]:
        """Build ElementConfig list matching elem_names order."""
        return [
            ElementConfig('Si', '1s', 'Si',    binding_energy=1839.0, oxidation_state=0),
            ElementConfig('Ti', '1s', 'TiO2',  binding_energy=4969.0, oxidation_state=+4),
            ElementConfig('Al', '1s', 'Al2O3', binding_energy=1562.0, oxidation_state=+3),
            ElementConfig('C',  '1s', 'C',     binding_energy=285.0,  oxidation_state=0),
            ElementConfig('O',  '1s', 'Al2O3', binding_energy=531.0,  oxidation_state=-2),
        ]

    @property
    def n_elements(self) -> int:
        return len(self.elem_names)

    @property
    def n_film(self) -> int:
        return len(self.free_indices)


# ============================================================
# Solver-level config (tunable per analysis)
# ============================================================

@dataclass
class HAXPESSolverConfig:
    """Tunable solver parameters for HAXPES depth profiling.

    Attributes:
        alpha: L1/L2 regularization strength.
        thickness_init: Initial film thickness estimate (Angstroms).
        thickness_bounds: (min, max) for thickness optimization.
        abruptness_init: Initial interface width (Angstroms).
        optimize_abruptness: Whether to optimize interface width.
        n_depth: Number of depth grid points.
        totaldepth: Maximum analysis depth (Angstroms).
        method: Optimization method ('L1', 'LLS', 'MEM', 'L2').
        scipy_method: Scipy optimizer ('SLSQP' or 'trust-constr'; SemiFreeSolver
            replaces a method that cannot handle constraints by 'SLSQP').
        tol: Convergence tolerance.
        max_iter: Maximum optimizer iterations.
        n_bins: Number of angle bins for detector binning.
        angle_range: (lo, hi) emission angle range for binning.
        angle_trim: Optional (lo, hi) for angle trimming in solver.
        bg_method: Background subtraction method.
        bg_margin_frac: Fraction of energy range for baseline endpoints.
        cross_section_source: Cross section database source.
        cross_section_poly_order: Polynomial order for YL extrapolation.
        non_dipole: Enable non-dipole angular correction (Cooper formula).
        xray_from_normal_deg: X-ray incidence angle from surface normal (deg).
        polarized: True for synchrotron (linearly polarized), False for lab
            X-ray sources (unpolarized, e.g. Ga Kα tube).
        n_workers: Number of parallel workers.
    """
    # Solver
    alpha: float = 0.01
    thickness_init: float = 240.0
    thickness_bounds: Tuple[float, float] = (10.0, 500.0)
    abruptness_init: float = 5.0
    optimize_abruptness: bool = False
    method: str = 'L1'
    scipy_method: str = 'SLSQP'
    tol: float = 1e-6
    max_iter: int = 200

    # Depth grid
    n_depth: int = 20
    totaldepth: float = 600.0

    # Angle binning
    n_bins: int = 12
    angle_range: Tuple[float, float] = (2.0, 50.0)
    angle_trim: Optional[Tuple[float, float]] = None

    # Preprocessing
    bg_method: str = 'linear_endpoints'
    bg_margin_frac: float = 0.05
    cross_section_source: str = 'yeh_lindau_extrap'
    cross_section_poly_order: int = 3

    # Non-dipole correction (Cooper formula: β + γ + δ)
    non_dipole: bool = False
    xray_from_normal_deg: float = 88.0  # X-ray incidence from surface normal
    polarized: bool = True  # True=synchrotron (linear), False=lab source (unpolarized)

    # Parallelization
    n_workers: int = 4

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.n_depth <= 0:
            raise ValueError(f"n_depth must be > 0, got {self.n_depth}")
        if self.totaldepth <= 0:
            raise ValueError(f"totaldepth must be > 0, got {self.totaldepth}")
        if self.angle_range[0] >= self.angle_range[1]:
            raise ValueError(
                f"angle_range[0] must be < angle_range[1], "
                f"got {self.angle_range}"
            )

    @property
    def stepsize(self) -> float:
        return self.totaldepth / self.n_depth

    @property
    def depth(self) -> np.ndarray:
        return np.arange(1, self.n_depth + 1) * self.stepsize


# ============================================================
# Pre-built instrument configs
# ============================================================

FIX_IBW_INSTRUMENT = HAXPESInstrumentConfig(
    photon_energy=9251.74,
    polar_offset=SAMPLE_POLAR_DEG,
    detector_range=(-36.84, 36.84),
    n_detector_channels=1001,
)

FIX_PXT_INSTRUMENT = HAXPESInstrumentConfig(
    photon_energy=9251.74,
    polar_offset=SAMPLE_POLAR_DEG,
    detector_range=(-36.84, 36.84),
    n_detector_channels=1001,
    n_prebinned_angles=12,
    detector_bin_lo=-33.82,
    detector_bin_hi=33.45,
)

# Pre-built experiment config
CALTIOSI_EXPERIMENT = HAXPESExperimentConfig(name='C_AlOx_TiOx_Si')


# ============================================================
# Data path constants (following benchmarks/paths.py pattern)
# ============================================================

# Root of all raw experiment data. There is no meaningful default the repo
# can ship, so unset means "no data available": imports succeed, but any
# loader hitting these paths fails with a missing-file error naming the env
# var to set.
# Each location has its own variable (2026-10-20): the directory layout of the
# authors' measurement campaigns is not part of the package.
FIX_IBW_DIR = env_path_or_placeholder("DEPPRO_FIX_IBW_DIR")   # fix-mode IBW frames
FIX_PXT_DIR = env_path_or_placeholder("DEPPRO_FIX_PXT_DIR")   # fix-mode pre-binned PXT files
MAP_H5 = env_path_or_placeholder("DEPPRO_MAP_H5")             # a fitted map HDF5 file

# Output directory for all experiment scripts.
# Resolved through deppro.paths (DEPPRO_OUTPUTS, else <repo>/outputs). Kept as a
# module constant because deppro.haxpes re-exports it; it is read once at import.
from deppro.paths import outputs_dir as _outputs_dir  # noqa: E402
OUTPUT_DIR = _outputs_dir()
