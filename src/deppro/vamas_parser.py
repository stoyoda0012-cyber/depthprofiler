"""
VAMAS (ISO 14976) File Parser for XPS Data
==========================================

VAMAS (Versailles Project on Advanced Materials and Standards) is the
ISO standard format (ISO 14976:1998) for surface chemical analysis data.

This parser extracts:
- Experiment metadata
- Block (spectrum) information
- Energy axis and intensity data
- Acquisition parameters

Usage:
    from deppro.vamas_parser import load_vamas, VamasFile

    # Load VAMAS file
    vamas = load_vamas('data.vms')

    # Access metadata
    print(vamas.institution)
    print(vamas.operator)

    # Iterate over blocks (spectra)
    for block in vamas.blocks:
        print(f"{block.species_label} @ {block.analysis_source_energy} eV")
        energy = block.get_energy_axis()
        counts = block.get_counts()

    # Convert to IntensityData for ARXPS analysis
    intensity_data = vamas.to_intensity_data(angles=[0, 30, 45, 60, 75])

References:
    - ISO 14976:1998 Surface chemical analysis
    - VAMAS format specification: https://www.sssj.org/VAMAS/

Date: 2026-01-23
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from enum import Enum
import re


class ScanMode(Enum):
    """VAMAS scan mode identifiers."""
    REGULAR = 'REGULAR'
    IRREGULAR = 'IRREGULAR'
    MAPPING = 'MAPPING'


class SignalMode(Enum):
    """VAMAS signal mode identifiers."""
    ANALOG = 'analog'
    PULSE_COUNTING = 'pulse counting'
    DETECTOR_MODE = 'detector mode'


@dataclass
class VamasBlock:
    """Single data block (spectrum) in VAMAS file.

    Each block represents one measurement, typically one element
    at one analysis condition (angle, energy, etc.).
    """
    # Block identification
    block_id: str = ''
    sample_id: str = ''

    # Analysis parameters
    analysis_source_label: str = ''           # e.g., 'Al Ka', 'Mg Ka'
    analysis_source_energy: float = 0.0       # eV
    analysis_source_strength: float = 0.0     # W or mA
    analysis_width_x: float = 0.0             # mm
    analysis_width_y: float = 0.0             # mm

    # Sample orientation
    analysis_take_off_polar_angle: float = 0.0    # degrees from normal
    analysis_azimuth: float = 0.0                 # degrees

    # Species information
    species_label: str = ''                   # e.g., 'C', 'Si', 'O'
    transition_label: str = ''                # e.g., '1s', '2p', '2p3/2'

    # Analyzer settings
    analyser_mode: str = ''                   # e.g., 'FAT', 'FRR'
    analyser_pass_energy: float = 0.0         # eV (for FAT mode)
    analyser_retard_ratio: float = 0.0        # (for FRR mode)
    analyser_work_function: float = 0.0       # eV

    # Energy scale
    abscissa_label: str = 'kinetic energy'    # or 'binding energy'
    abscissa_units: str = 'eV'
    abscissa_start: float = 0.0
    abscissa_increment: float = 0.0

    # Data
    num_ordinate_values: int = 0              # Number of data points
    signal_mode: str = 'pulse counting'
    dwell_time: float = 0.0                   # seconds
    num_scans: int = 1

    # Raw data
    ordinate_values: np.ndarray = field(default_factory=lambda: np.array([]))

    # Additional variables (optional)
    additional_params: Dict[str, float] = field(default_factory=dict)

    def get_energy_axis(self) -> np.ndarray:
        """Get energy axis array."""
        n_points = len(self.ordinate_values)
        return self.abscissa_start + np.arange(n_points) * self.abscissa_increment

    def get_counts(self) -> np.ndarray:
        """Get intensity/counts array."""
        return self.ordinate_values.copy()

    def get_binding_energy(self, photon_energy: Optional[float] = None) -> np.ndarray:
        """Convert to binding energy scale.

        Args:
            photon_energy: X-ray photon energy (eV). If None, uses analysis_source_energy.

        Returns:
            Binding energy array (eV)
        """
        if photon_energy is None:
            photon_energy = self.analysis_source_energy

        ke = self.get_energy_axis()

        if 'binding' in self.abscissa_label.lower():
            return ke  # Already in BE
        else:
            # Convert KE to BE: BE = hv - KE - phi
            return photon_energy - ke - self.analyser_work_function

    def get_kinetic_energy(self, photon_energy: Optional[float] = None) -> np.ndarray:
        """Get kinetic energy scale.

        Args:
            photon_energy: X-ray photon energy (eV). If None, uses analysis_source_energy.

        Returns:
            Kinetic energy array (eV)
        """
        if photon_energy is None:
            photon_energy = self.analysis_source_energy

        energy = self.get_energy_axis()

        if 'kinetic' in self.abscissa_label.lower():
            return energy
        else:
            # Convert BE to KE: KE = hv - BE - phi
            return photon_energy - energy - self.analyser_work_function

    @property
    def integrated_intensity(self) -> float:
        """Total integrated intensity."""
        return np.sum(self.ordinate_values) * abs(self.abscissa_increment)

    @property
    def peak_intensity(self) -> float:
        """Maximum intensity."""
        return np.max(self.ordinate_values) if len(self.ordinate_values) > 0 else 0.0


@dataclass
class VamasFile:
    """Complete VAMAS file representation."""

    # File header
    format_identifier: str = 'VAMAS Surface Chemical Analysis Standard Data Transfer Format 1988 May 4'
    institution_identifier: str = ''
    instrument_model_identifier: str = ''
    operator_identifier: str = ''
    experiment_identifier: str = ''

    # Comment
    num_comment_lines: int = 0
    comment_lines: List[str] = field(default_factory=list)

    # Experiment mode
    experiment_mode: str = 'NORM'  # NORM, SDP, MAP, etc.
    scan_mode: str = 'REGULAR'

    # Number of spectral regions and blocks
    num_spectral_regions: int = 0
    num_blocks: int = 0

    # Experiment variables (for MAPDP, MAPSVDP, etc.)
    num_experiment_variables: int = 0
    experiment_variable_labels: List[str] = field(default_factory=list)
    experiment_variable_units: List[str] = field(default_factory=list)

    # Blocks (spectra)
    blocks: List[VamasBlock] = field(default_factory=list)

    # Source path
    source_path: str = ''

    def get_block_by_species(self, species: str) -> List[VamasBlock]:
        """Get all blocks for a given species/element."""
        return [b for b in self.blocks if species.lower() in b.species_label.lower()]

    def get_block_by_angle(self, angle: float, tolerance: float = 1.0) -> List[VamasBlock]:
        """Get all blocks at a specific take-off angle."""
        return [b for b in self.blocks
                if abs(b.analysis_take_off_polar_angle - angle) < tolerance]

    def get_unique_species(self) -> List[str]:
        """Get list of unique species labels."""
        species = set()
        for block in self.blocks:
            if block.species_label:
                species.add(block.species_label)
        return sorted(species)

    def get_unique_angles(self) -> List[float]:
        """Get list of unique take-off angles."""
        angles = set()
        for block in self.blocks:
            angles.add(block.analysis_take_off_polar_angle)
        return sorted(angles)

    def to_intensity_data(
        self,
        angles: Optional[List[float]] = None,
        elements: Optional[List[str]] = None,
        use_integrated: bool = True,
    ) -> 'IntensityData':
        """Convert to IntensityData for ARXPS analysis.

        Args:
            angles: List of angles to extract. If None, uses all unique angles.
            elements: List of element labels to extract. If None, uses all unique species.
            use_integrated: If True, use integrated intensity; else use peak intensity.

        Returns:
            IntensityData object for inverse solver
        """
        from .data_io import IntensityData

        if angles is None:
            angles = self.get_unique_angles()
        if elements is None:
            elements = self.get_unique_species()

        n_angles = len(angles)
        n_elements = len(elements)

        intensities = np.zeros((n_angles, n_elements))
        kinetic_energies = np.zeros(n_elements)

        for j, elem in enumerate(elements):
            elem_blocks = self.get_block_by_species(elem)

            # Get kinetic energy from first block
            if elem_blocks:
                ke = elem_blocks[0].get_kinetic_energy()
                kinetic_energies[j] = np.mean(ke)  # Center of spectrum

            for i, angle in enumerate(angles):
                # Find block matching element and angle
                for block in elem_blocks:
                    if abs(block.analysis_take_off_polar_angle - angle) < 1.0:
                        if use_integrated:
                            intensities[i, j] = block.integrated_intensity
                        else:
                            intensities[i, j] = block.peak_intensity
                        break

        return IntensityData(
            angles=np.array(angles),
            element_labels=elements,
            intensities=intensities,
            kinetic_energies=kinetic_energies,
        )


def load_vamas(path: Union[str, Path]) -> VamasFile:
    """Load and parse VAMAS file.

    Args:
        path: Path to VAMAS file (.vms)

    Returns:
        VamasFile object with all parsed data
    """
    path = Path(path)

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    return _parse_vamas(lines, str(path))


def _parse_vamas(lines: List[str], source_path: str = '') -> VamasFile:
    """Parse VAMAS file content.

    VAMAS format is line-based with specific field order.
    This parser follows the ISO 14976 specification.
    """
    vamas = VamasFile(source_path=source_path)

    idx = 0

    def read_line() -> str:
        nonlocal idx
        if idx < len(lines):
            line = lines[idx].rstrip('\n\r')
            idx += 1
            return line
        return ''

    def read_int() -> int:
        return int(read_line().strip())

    def read_float() -> float:
        line = read_line().strip()
        # Handle scientific notation variants
        line = line.replace('D', 'E').replace('d', 'e')
        try:
            return float(line)
        except ValueError:
            return 0.0

    # === File Header ===
    vamas.format_identifier = read_line()
    vamas.institution_identifier = read_line()
    vamas.instrument_model_identifier = read_line()
    vamas.operator_identifier = read_line()
    vamas.experiment_identifier = read_line()

    # Number of comment lines
    vamas.num_comment_lines = read_int()
    vamas.comment_lines = [read_line() for _ in range(vamas.num_comment_lines)]

    # Experiment mode
    vamas.experiment_mode = read_line().strip()
    vamas.scan_mode = read_line().strip()

    # Number of spectral regions (for specific modes)
    if vamas.experiment_mode in ('MAP', 'MAPDP', 'NORM', 'SDP'):
        vamas.num_spectral_regions = read_int()

    # Number of experiment variables
    if vamas.experiment_mode in ('MAP', 'MAPDP', 'MAPSVDP', 'SDP', 'SDPSV'):
        vamas.num_experiment_variables = read_int()
        for _ in range(vamas.num_experiment_variables):
            vamas.experiment_variable_labels.append(read_line())
            vamas.experiment_variable_units.append(read_line())

    # Number of entries in parameter inclusion/exclusion list
    # (we skip these for now - they define which optional fields are present)
    num_entries = read_int()
    inclusion_list = []
    for _ in range(num_entries):
        inclusion_list.append(read_int())

    # Number of manually entered items
    num_manual = read_int()
    for _ in range(num_manual):
        read_int()  # Future use field indices

    # Number of future upgrade experiment entries
    num_future_exp = read_int()
    for _ in range(num_future_exp):
        read_line()  # Future upgrade block entries

    # Number of future upgrade block entries
    num_future_block = read_int()
    # These will be read per block

    # Number of blocks
    vamas.num_blocks = read_int()

    # === Parse Each Block ===
    for block_num in range(vamas.num_blocks):
        block = _parse_block(
            read_line, read_int, read_float,
            vamas.experiment_mode,
            vamas.num_experiment_variables,
            num_future_block,
            inclusion_list,
        )
        vamas.blocks.append(block)

    return vamas


def _parse_block(
    read_line,
    read_int,
    read_float,
    experiment_mode: str,
    num_exp_vars: int,
    num_future_block: int,
    inclusion_list: List[int],
) -> VamasBlock:
    """Parse a single VAMAS block (spectrum)."""
    block = VamasBlock()

    # Block identifier
    block.block_id = read_line()
    block.sample_id = read_line()

    # Year, month, day, hours, minutes, seconds, timezone
    year = read_int()
    month = read_int()
    day = read_int()
    hours = read_int()
    minutes = read_int()
    seconds = read_int()
    num_hours_advance = read_int()

    # Number of comment lines for this block
    num_block_comments = read_int()
    for _ in range(num_block_comments):
        read_line()  # Comment lines

    # Technique
    technique = read_line()

    # For MAP/MAPDP modes
    if experiment_mode in ('MAP', 'MAPDP'):
        x_coord = read_float()
        y_coord = read_float()

    # Experiment variable values (for SDP, MAP, etc.)
    for _ in range(num_exp_vars):
        read_float()

    # Analysis source label
    block.analysis_source_label = read_line()

    # Analysis source energy (for XPS techniques)
    if technique.upper() in ('XPS', 'AES', 'UPS', 'FABMS', 'FABMS ENERGY SPEC', 'ISS', 'SIMS', 'SIMS ENERGY SPEC', 'SNMS', 'SNMS ENERGY SPEC'):
        block.analysis_source_energy = read_float()

    # Analysis source strength
    block.analysis_source_strength = read_float()

    # Analysis source beam dimensions
    block.analysis_width_x = read_float()
    block.analysis_width_y = read_float()

    # Field of view (for some modes)
    if experiment_mode in ('MAP', 'MAPDP', 'MAPSVDP', 'SEM'):
        fov_x = read_float()
        fov_y = read_float()

    # First/last linescan X coordinate (for MAP)
    if experiment_mode in ('MAP', 'MAPDP', 'MAPSVDP'):
        first_x = read_int()
        last_x = read_int()
        first_y = read_int()
        last_y = read_int()

    # Analysis source polar/azimuth angles
    source_polar = read_float()
    source_azimuth = read_float()

    # Analyzer mode
    block.analyser_mode = read_line()

    # Analyzer resolution
    resolution = read_float()

    # Analyzer mode parameters
    if block.analyser_mode.upper() == 'FAT' or block.analyser_mode.upper() == 'FRR':
        if block.analyser_mode.upper() == 'FAT':
            block.analyser_pass_energy = read_float()
        else:
            block.analyser_retard_ratio = read_float()

    # Magnification analyzer transfer lens
    magnification = read_float()

    # Analyzer work function
    block.analyser_work_function = read_float()

    # Target bias
    target_bias = read_float()

    # Analysis width X, Y (again?)
    analysis_width = read_float()
    analysis_height = read_float()

    # Analysis take-off angle and azimuth
    block.analysis_take_off_polar_angle = read_float()  # From surface normal
    block.analysis_azimuth = read_float()

    # Species label (element)
    block.species_label = read_line()

    # Transition/charge state label
    block.transition_label = read_line()

    # Charge of detected species
    charge_detected = read_int()

    # Abscissa (energy) axis label and units
    block.abscissa_label = read_line()
    block.abscissa_units = read_line()

    # Abscissa start and increment
    block.abscissa_start = read_float()
    block.abscissa_increment = read_float()

    # Number of corresponding variables (typically 1)
    num_corr_vars = read_int()

    # Corresponding variable info
    for _ in range(num_corr_vars):
        corr_label = read_line()
        corr_units = read_line()

    # Signal mode
    block.signal_mode = read_line()

    # Dwell time
    block.dwell_time = read_float()

    # Number of scans
    block.num_scans = read_int()

    # Signal collection time
    collection_time = read_float()

    # Number of scans for background
    num_bg_scans = read_int()

    # Number of additional numerical parameters
    num_add_params = read_int()
    for _ in range(num_add_params):
        param_label = read_line()
        param_unit = read_line()
        param_value = read_float()
        block.additional_params[param_label] = param_value

    # Number of ordinate values (data points)
    block.num_ordinate_values = read_int()

    # Minimum and maximum ordinate values
    min_ordinate = read_float()
    max_ordinate = read_float()

    # Read ordinate data
    ordinate_values = []
    for _ in range(block.num_ordinate_values):
        ordinate_values.append(read_float())

    block.ordinate_values = np.array(ordinate_values)

    # Future upgrade block entries
    for _ in range(num_future_block):
        read_line()

    return block


def convert_vamas_to_csv(
    vamas_path: Union[str, Path],
    output_path: Union[str, Path],
    angles: Optional[List[float]] = None,
) -> None:
    """Convert VAMAS file to CSV format for ARXPS analysis.

    Args:
        vamas_path: Path to VAMAS file
        output_path: Path for output CSV file
        angles: Specific angles to extract (None = all)
    """
    from .data_io import save_intensity_csv

    vamas = load_vamas(vamas_path)
    intensity_data = vamas.to_intensity_data(angles=angles)
    save_intensity_csv(intensity_data, output_path)


# ============================================================================
# Kratos Format Parser (Basic Implementation)
# ============================================================================

@dataclass
class KratosSpectrum:
    """Single spectrum from Kratos format."""
    name: str = ''
    element: str = ''
    orbital: str = ''
    pass_energy: float = 0.0
    dwell_time: float = 0.0
    num_scans: int = 1
    start_energy: float = 0.0
    step_size: float = 0.0
    data: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def energy_axis(self) -> np.ndarray:
        n = len(self.data)
        return self.start_energy + np.arange(n) * self.step_size


@dataclass
class KratosFile:
    """Kratos XPS file representation."""
    filename: str = ''
    operator: str = ''
    sample_id: str = ''
    spectra: List[KratosSpectrum] = field(default_factory=list)

    def to_intensity_data(
        self,
        angles: Optional[List[float]] = None,
    ) -> 'IntensityData':
        """Convert to IntensityData (limited - Kratos doesn't always store angles)."""
        from .data_io import IntensityData

        # Kratos files don't always have angle information
        # This is a basic conversion assuming single-angle data
        elements = [s.element for s in self.spectra if s.element]
        unique_elements = list(dict.fromkeys(elements))

        intensities = []
        for elem in unique_elements:
            for spec in self.spectra:
                if spec.element == elem:
                    intensities.append(np.sum(spec.data))
                    break

        # Default to single angle if not specified
        if angles is None:
            angles = [0.0]

        n_angles = len(angles)
        n_elements = len(unique_elements)

        # Repeat intensity for each angle (placeholder)
        int_matrix = np.tile(np.array(intensities), (n_angles, 1))

        return IntensityData(
            angles=np.array(angles),
            element_labels=unique_elements,
            intensities=int_matrix,
        )


def load_kratos(path: Union[str, Path]) -> KratosFile:
    """Load Kratos XPS file.

    Note: This is a basic implementation. Kratos uses various proprietary
    formats (.dset, .xml, etc.) that may require additional parsing.

    Args:
        path: Path to Kratos file

    Returns:
        KratosFile object
    """
    path = Path(path)

    # Try to determine format from extension
    suffix = path.suffix.lower()

    if suffix == '.xml':
        return _parse_kratos_xml(path)
    else:
        # Try generic text format
        return _parse_kratos_text(path)


def _parse_kratos_xml(path: Path) -> KratosFile:
    """Parse Kratos XML format."""
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("xml.etree.ElementTree required for Kratos XML parsing")

    tree = ET.parse(path)
    root = tree.getroot()

    kratos = KratosFile(filename=str(path))

    # Parse based on Kratos XML structure
    # (This is a simplified parser - actual format may vary)
    for spectrum_elem in root.findall('.//Spectrum'):
        spec = KratosSpectrum()
        spec.name = spectrum_elem.get('name', '')

        # Extract element from name (e.g., "C 1s" -> "C")
        parts = spec.name.split()
        if parts:
            spec.element = parts[0]
            if len(parts) > 1:
                spec.orbital = parts[1]

        # Find data
        data_elem = spectrum_elem.find('Data')
        if data_elem is not None and data_elem.text:
            values = [float(x) for x in data_elem.text.split()]
            spec.data = np.array(values)

        kratos.spectra.append(spec)

    return kratos


def _parse_kratos_text(path: Path) -> KratosFile:
    """Parse Kratos text export format."""
    kratos = KratosFile(filename=str(path))

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Very basic parsing - look for common patterns
    lines = content.split('\n')

    current_spectrum = None
    data_mode = False
    data_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Look for spectrum header
        if 'Region' in line or 'Spectrum' in line:
            if current_spectrum and data_lines:
                current_spectrum.data = np.array([float(x.split()[-1]) for x in data_lines if x.strip()])
                kratos.spectra.append(current_spectrum)

            current_spectrum = KratosSpectrum()
            current_spectrum.name = line
            data_mode = False
            data_lines = []

        elif 'Element' in line:
            if current_spectrum:
                parts = line.split(':')
                if len(parts) > 1:
                    current_spectrum.element = parts[1].strip()

        elif line[0].isdigit() or (line[0] == '-' and len(line) > 1 and line[1].isdigit()):
            # Data line
            data_lines.append(line)

    # Don't forget last spectrum
    if current_spectrum and data_lines:
        try:
            current_spectrum.data = np.array([float(x.split()[-1]) for x in data_lines if x.strip()])
            kratos.spectra.append(current_spectrum)
        except ValueError:
            pass

    return kratos


if __name__ == '__main__':
    print("VAMAS/Kratos Parser Module")
    print("=" * 40)
    print("\nSupported formats:")
    print("  - VAMAS (.vms) - ISO 14976 standard")
    print("  - Kratos (.xml, text export)")
    print("\nUsage:")
    print("  from deppro.vamas_parser import load_vamas, load_kratos")
    print("  vamas = load_vamas('data.vms')")
    print("  intensity_data = vamas.to_intensity_data()")
