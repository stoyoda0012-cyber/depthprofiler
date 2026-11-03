"""
HAXPES Data Loaders
====================

Unified data loading for IBW and PXT HAXPES data formats.

Supported formats:
    - IBW: Igor Binary Wave (fix mode, 1001 detector pixels per frame)
    - PXT: Scienta format (fix mode, pre-binned 12 angle channels)
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SAMPLE_POLAR_DEG, HAXPESExperimentConfig, HAXPESInstrumentConfig, _polar_offset_from

logger = logging.getLogger(__name__)


# ============================================================
# Data classes
# ============================================================

@dataclass
class RawElementData:
    """Raw data for a single element from one or more measurements.

    Attributes:
        element: Element symbol.
        raw: Raw detector data array.
            IBW: (n_energy, n_detector, n_frames)
            PXT summed: (n_energy, n_angles)
        energy_ke: Kinetic energy axis (eV).
        step_time_ms: Exposure time per step.
        center_ke: Center kinetic energy.
        n_frames_per_meas: Number of frames from each measurement file.
    """
    element: str
    raw: np.ndarray
    energy_ke: np.ndarray
    step_time_ms: int = 10000
    center_ke: float = 0.0
    n_frames_per_meas: List[int] = field(default_factory=list)


# ============================================================
# IBW Loader (fix mode)
# ============================================================

def load_fix_ibw(
    base_path: Path,
    experiment: Optional[HAXPESExperimentConfig] = None,
    measurement_indices: Tuple[int, ...] = (1, 2),
) -> Dict[str, RawElementData]:
    """Load and merge Fix mode IBW data from multiple measurements.

    Parameters
    ----------
    base_path : Path
        Path to the fix-mode IBW folder.
    experiment : HAXPESExperimentConfig, optional
        Experiment config. Uses default CALTIOSI if None.
    measurement_indices : tuple
        Measurement indices (e.g., (1, 2) for 0001+0002).

    Returns
    -------
    data : dict
        {element: RawElementData} with raw shape (n_energy, 1001, n_total_frames).
    """
    import igor2.binarywave

    if experiment is None:
        from .config import CALTIOSI_EXPERIMENT
        experiment = CALTIOSI_EXPERIMENT

    data: Dict[str, RawElementData] = {}

    for meas_idx in measurement_indices:
        for elem, label in experiment.ibw_patterns.items():
            pattern = f"*{meas_idx:04d}{label}.ibw"
            files = list(base_path.glob(pattern))
            if not files:
                logger.warning(f"No file for {elem} (meas={meas_idx}, {pattern})")
                continue

            ibw_path = files[0]
            logger.info(f"Loading: {ibw_path.name}")
            wave = igor2.binarywave.load(str(ibw_path))
            raw = wave['wave']['wData']  # (n_energy, 1001, n_frames)

            # Parse metadata
            note = wave['wave'].get('note', b'').decode('utf-8', errors='ignore')
            meta = {}
            for line in note.replace('\r', '\n').split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    meta[k.strip()] = v.strip()

            center_ke = float(meta.get('Center Energy', 0))
            low_ke = float(meta.get('Low Energy', center_ke - 10))
            high_ke = float(meta.get('High Energy', center_ke + 10))
            energy_ke = np.linspace(low_ke, high_ke, raw.shape[0])
            step_time = int(meta.get('Step Time', 10000))

            logger.info(f"  {elem}: shape={raw.shape}, "
                        f"E=[{low_ke:.1f}, {high_ke:.1f}] eV")

            if elem not in data:
                data[elem] = RawElementData(
                    element=elem,
                    raw=raw,
                    energy_ke=energy_ke,
                    step_time_ms=step_time,
                    center_ke=center_ke,
                    n_frames_per_meas=[raw.shape[2]],
                )
            else:
                # Concatenate frames along axis 2
                data[elem].raw = np.concatenate(
                    [data[elem].raw, raw], axis=2
                )
                data[elem].n_frames_per_meas.append(raw.shape[2])

    # Summary
    for elem in experiment.elem_names:
        if elem in data:
            d = data[elem]
            total_frames = d.raw.shape[2]
            logger.info(
                f"  {elem}: {d.raw.shape}, "
                f"frames={d.n_frames_per_meas} (total={total_frames}), "
                f"step_time={d.step_time_ms}ms"
            )

    return data


# ============================================================
# PXT Loader (fix mode, pre-binned)
# ============================================================

def load_prebinned_pxt(
    pxt_dir: Path,
    experiment: Optional[HAXPESExperimentConfig] = None,
    instrument: Optional[HAXPESInstrumentConfig] = None,
    file_filter: str = 'Narrow',
    polar_offset: float = SAMPLE_POLAR_DEG,
    *,
    tilt: Optional[float] = None,
) -> Dict[str, dict]:
    """Load fix-mode pre-binned PXT data, summed over spatial and temporal dims.

    Parameters
    ----------
    pxt_dir : Path
        Directory containing PXT files.
    experiment : HAXPESExperimentConfig, optional
        Experiment config.
    instrument : HAXPESInstrumentConfig, optional
        Instrument config.
    file_filter : str
        Filter string for PXT filenames (default: 'Narrow').
    polar_offset : float
        Polar angle from the analyser's lens axis to the sample normal
        (degrees); emission angle = detector angle + polar_offset.
    tilt : float, optional
        Deprecated name of ``polar_offset``.

    Returns
    -------
    result : dict
        {elem: {'spectra': (n_energy, 12), 'energy': (n_energy,),
                 'emission_angles': (12,), ...}}
    """
    polar_offset = _polar_offset_from(polar_offset, tilt, "load_prebinned_pxt")
    from toyomacro.io.readers.pxt_reader import PXTReader, _parse_wave_notes

    if experiment is None:
        from .config import CALTIOSI_EXPERIMENT
        experiment = CALTIOSI_EXPERIMENT

    pxt_files = sorted(pxt_dir.glob("*.pxt"))
    pxt_files = [f for f in pxt_files if file_filter in f.name]

    region_to_elem = experiment.region_to_elem
    elem_names = experiment.elem_names

    data: Dict[str, dict] = {}

    for pxt_path in pxt_files:
        logger.info(f"Reading: {pxt_path.name}")
        reader = PXTReader(pxt_path)
        reader._ensure_parsed()

        for region in reader._regions:
            notes = _parse_wave_notes(region.wave_notes)
            region_name = notes.get('Region Name',
                                    f'Region {region.region_number}')
            elem = region_to_elem.get(region_name)
            if elem is None:
                continue

            spec = region.data.copy().astype(np.float64)

            # Detector angle info
            det_lo = region.x_ini[1]
            n_angle = region.wave_header.n_dim[1]
            angle_delta = region.wave_header.sf_a[1]
            det_bin_centers = det_lo + np.arange(n_angle) * angle_delta

            # Energy axis
            n_energy = region.wave_header.n_dim[0]
            energy = np.linspace(region.x_ini[0], region.x_fin[0], n_energy)

            if elem not in data:
                data[elem] = {
                    'frames': [spec],
                    'energy': energy,
                    'det_bin_centers': det_bin_centers,
                    'n_angle': n_angle,
                }
            else:
                data[elem]['frames'].append(spec)

    # Sum all frames and spatial pixels
    result = {}
    for elem in elem_names:
        if elem not in data:
            continue

        d = data[elem]
        all_frames = d['frames']

        total = np.zeros_like(all_frames[0])
        for frame in all_frames:
            total += frame

        if total.ndim == 4:
            n_pixels = total.shape[2] * total.shape[3]
            spectra = total.sum(axis=(2, 3))
        elif total.ndim == 3:
            n_pixels = total.shape[2]
            spectra = total.sum(axis=2)
        else:
            n_pixels = 1
            spectra = total

        # flip is a reinterpretation of detector rows as analyzer channels.
        # The 2D detector is mounted below the analyzer, so the row order is reversed relative to the
        # angular-dispersion axis (not needed for a top mount). It is not a sign flip of the angle (polar_offset - det).
        # The grid comes from 996/1001 channels and is off-center by 2.5 ch = 0.184 deg, so the two do not coincide.
        # Reading it as a sign flip shifts every bin by a uniform 0.368 deg (happened on 2026-07-31).
        emission = np.flip(d['det_bin_centers']) + polar_offset

        logger.info(f"  {elem}: {len(all_frames)} files, sum → {spectra.shape}")

        result[elem] = {
            'spectra': spectra,
            'energy': d['energy'],
            'emission_angles': emission,
            'det_bin_centers': d['det_bin_centers'],
            'n_files': len(all_frames),
            'n_pixels': n_pixels,
        }

    return result


# ============================================================
# PXT Pixel Stack Loader (for bootstrap methods)
# ============================================================

def load_prebinned_pixel_stacks(
    pxt_dir: Path,
    experiment: Optional[HAXPESExperimentConfig] = None,
    file_filter: str = 'Narrow',
) -> Dict[str, np.ndarray]:
    """Load all raw data as {elem: (n_time, n_energy, n_angles)} stacks.

    Used by bootstrap methods that need per-pixel temporal data.
    Each spatial pixel (nx * ny per file) becomes a separate time index.

    Parameters
    ----------
    pxt_dir : Path
        Directory containing PXT files.
    experiment : HAXPESExperimentConfig, optional
        Experiment config.
    file_filter : str
        Filter string for PXT filenames.

    Returns
    -------
    elem_stacks : dict
        {elem: (n_time, n_energy, n_angles)} arrays.
    """
    from toyomacro.io.readers.pxt_reader import PXTReader, _parse_wave_notes

    if experiment is None:
        from .config import CALTIOSI_EXPERIMENT
        experiment = CALTIOSI_EXPERIMENT

    pxt_files = sorted(pxt_dir.glob("*.pxt"))
    pxt_files = [f for f in pxt_files if file_filter in f.name]

    region_to_elem = experiment.region_to_elem
    elem_stacks: Dict[str, List[np.ndarray]] = {}

    for pf in pxt_files:
        logger.info(f"Reading: {pf.name}")
        reader = PXTReader(pf)
        reader._ensure_parsed()
        for region in reader._regions:
            notes = _parse_wave_notes(region.wave_notes)
            rname = notes.get('Region Name', '')
            elem = region_to_elem.get(rname)
            if elem is None:
                continue
            data = region.data.copy().astype(np.float64)
            # (n_energy, n_angles, nx, ny)
            n_e, n_a, n_x, n_y = data.shape
            # Reshape to (n_pixels, n_energy, n_angles)
            reshaped = data.transpose(2, 3, 0, 1).reshape(n_x * n_y, n_e, n_a)
            if elem not in elem_stacks:
                elem_stacks[elem] = []
            elem_stacks[elem].append(reshaped)

    # Concatenate along time axis
    for elem in elem_stacks:
        elem_stacks[elem] = np.concatenate(elem_stacks[elem], axis=0)
        logger.info(f"  {elem}: stack shape = {elem_stacks[elem].shape}")

    return elem_stacks


# ============================================================
# Sum and Bin (IBW → angle-binned spectra)
# ============================================================

def sum_and_bin(
    data_dict: Dict[str, RawElementData],
    elem_names: List[str],
    n_bins: int = 12,
    angle_lo: float = 2.0,
    angle_hi: float = 50.0,
    polar_offset: float = SAMPLE_POLAR_DEG,
    n_detector: int = 1001,
    *,
    tilt: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray],
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum all frames and bin 1001 detector channels to angle bins.

    Parameters
    ----------
    data_dict : dict
        {elem: RawElementData} from load_fix_ibw().
    elem_names : list
        Ordered element names.
    n_bins : int
        Number of angle bins.
    angle_lo, angle_hi : float
        Angle range for binning.
    polar_offset : float
        Polar angle from the analyser's lens axis to the sample normal
        (degrees); emission angle = detector angle + polar_offset.
    n_detector : int
        Number of detector channels.
    tilt : float, optional
        Deprecated name of ``polar_offset``.

    Returns
    -------
    binned_spectra : dict
        {elem: (n_energy, n_bins)} binned energy spectra.
    energy_axes : dict
        {elem: (n_energy,)} energy axes.
    bin_edges : ndarray
        (n_bins + 1,).
    bin_centers : ndarray
        (n_bins,).
    emission_angles : ndarray
        (n_detector,) full emission angle array.
    channels_per_bin : ndarray
        (n_bins,) channels per bin.
    """
    from .preprocessor import detector_to_emission, make_uniform_bins, bin_detector_data

    polar_offset = _polar_offset_from(polar_offset, tilt, "sum_and_bin")
    emission_angles = detector_to_emission(n_detector, polar_offset)
    bin_edges, bin_centers = make_uniform_bins(n_bins, angle_lo, angle_hi)

    # Count channels per bin
    channels_per_bin = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = ((emission_angles >= bin_edges[i])
                & (emission_angles < bin_edges[i + 1]))
        channels_per_bin[i] = mask.sum()

    binned_spectra = {}
    energy_axes = {}

    for elem in elem_names:
        if elem not in data_dict:
            continue

        d = data_dict[elem]
        raw = d.raw  # (n_energy, 1001, n_frames)

        # Sum over all frames
        frame_sum = raw.sum(axis=2)  # (n_energy, 1001)

        # Bin detector channels → angle bins
        binned = bin_detector_data(frame_sum, emission_angles, bin_edges)

        binned_spectra[elem] = binned
        energy_axes[elem] = d.energy_ke

        logger.info(f"  {elem}: {raw.shape} → sum → "
                    f"{frame_sum.shape} → bin → {binned.shape}")

    return (binned_spectra, energy_axes, bin_edges, bin_centers,
            emission_angles, channels_per_bin)
