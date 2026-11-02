"""
Data I/O for ARXPS Depth Profile Analysis
==========================================

Handles input/output of measurement data and results.

Supported formats:
    - CSV: Angle-resolved intensity data
    - YAML: Configuration and model definition
    - JSON: Results output
    - VAMAS: Standard XPS format (future)

Date: 2026-01-23
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml


# ============================================================================
# Input Data Structures
# ============================================================================

@dataclass
class IntensityData:
    """Angle-resolved intensity data for inverse problem.

    Attributes:
        angles: Observation angles in degrees (n_angles,)
        element_labels: Element labels (n_elements,)
        intensities: Intensity matrix (n_angles, n_elements)
        kinetic_energies: Kinetic energies per element (optional)
        uncertainties: Measurement uncertainties (optional)
    """
    angles: np.ndarray
    element_labels: List[str]
    intensities: np.ndarray
    kinetic_energies: Optional[np.ndarray] = None
    uncertainties: Optional[np.ndarray] = None

    @property
    def n_angles(self) -> int:
        return len(self.angles)

    @property
    def n_elements(self) -> int:
        return len(self.element_labels)

    def normalize(self, mode: str = 'angle') -> 'IntensityData':
        """Normalize intensities.

        Args:
            mode: 'angle' (sum per angle = 1) or 'element' (max per element = 1)

        Returns:
            New IntensityData with normalized intensities
        """
        if mode == 'angle':
            sums = self.intensities.sum(axis=1, keepdims=True)
            norm_int = self.intensities / np.maximum(sums, 1e-10)
        elif mode == 'element':
            maxes = self.intensities.max(axis=0, keepdims=True)
            norm_int = self.intensities / np.maximum(maxes, 1e-10)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        return IntensityData(
            angles=self.angles.copy(),
            element_labels=self.element_labels.copy(),
            intensities=norm_int,
            kinetic_energies=self.kinetic_energies.copy() if self.kinetic_energies is not None else None,
            uncertainties=self.uncertainties.copy() if self.uncertainties is not None else None,
        )


# ============================================================================
# CSV I/O
# ============================================================================

def load_intensity_csv(
    path: Union[str, Path],
    angle_column: str = 'angle',
    delimiter: str = ',',
    skip_header: int = 0,
) -> IntensityData:
    """Load angle-resolved intensity data from CSV.

    Expected CSV format:
        angle,C,Al,Ti,Si_ox,O,Si_bulk
        0,0.15,0.12,0.18,0.20,0.25,0.10
        30,0.14,0.11,0.17,0.21,0.26,0.11
        ...

    Or with kinetic energies (header row 2):
        angle,C,Al,Ti,Si_ox,O,Si_bulk
        KE,1200,1403,1028,1385,956,1385
        0,0.15,0.12,0.18,0.20,0.25,0.10
        ...

    Args:
        path: Path to CSV file
        angle_column: Name of angle column
        delimiter: CSV delimiter
        skip_header: Number of header lines to skip

    Returns:
        IntensityData object
    """
    path = Path(path)

    with open(path, 'r') as f:
        lines = f.readlines()

    # Skip header lines and comment lines
    lines = lines[skip_header:]
    lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]

    if not lines:
        raise ValueError(f"No data found in {path}")

    # Parse header
    header = lines[0].strip().split(delimiter)

    # Find angle column index
    try:
        angle_idx = header.index(angle_column)
    except ValueError:
        angle_idx = 0  # Default to first column

    # Element labels (all columns except angle)
    element_labels = [h.strip() for i, h in enumerate(header) if i != angle_idx]

    # Check for kinetic energy row
    kinetic_energies = None
    data_start = 1

    if len(lines) > 1:
        first_data = lines[1].strip().split(delimiter)
        if first_data[0].strip().upper() in ('KE', 'KINETIC_ENERGY', 'E_KIN'):
            ke_values = [float(v.strip()) for i, v in enumerate(first_data) if i != angle_idx]
            kinetic_energies = np.array(ke_values)
            data_start = 2

    # Parse data
    angles = []
    intensities = []

    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        values = line.split(delimiter)
        angles.append(float(values[angle_idx].strip()))

        row = []
        for i, v in enumerate(values):
            if i != angle_idx:
                row.append(float(v.strip()))
        intensities.append(row)

    return IntensityData(
        angles=np.array(angles),
        element_labels=element_labels,
        intensities=np.array(intensities),
        kinetic_energies=kinetic_energies,
    )


def save_intensity_csv(
    data: IntensityData,
    path: Union[str, Path],
    include_ke: bool = True,
    delimiter: str = ',',
) -> None:
    """Save intensity data to CSV.

    Args:
        data: IntensityData object
        path: Output path
        include_ke: Include kinetic energy row
        delimiter: CSV delimiter
    """
    path = Path(path)

    with open(path, 'w') as f:
        # Header
        header = ['angle'] + data.element_labels
        f.write(delimiter.join(header) + '\n')

        # Kinetic energies
        if include_ke and data.kinetic_energies is not None:
            ke_row = ['KE'] + [f'{ke:.1f}' for ke in data.kinetic_energies]
            f.write(delimiter.join(ke_row) + '\n')

        # Data
        for i, angle in enumerate(data.angles):
            row = [f'{angle:.1f}'] + [f'{v:.6f}' for v in data.intensities[i]]
            f.write(delimiter.join(row) + '\n')


# ============================================================================
# YAML Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration for depth profile analysis.

    Defines the physical model for inverse problem solving.
    """
    # Model type
    model_type: str = 'multilayer'  # 'multilayer', 'film_structure', 'generic'

    # Depth grid
    depth_start: float = 0.0
    depth_end: float = 100.0
    depth_step: float = 1.0

    # Physical parameters
    photon_energy: float = 1486.6  # Al Ka

    # Layer structure (for multilayer model)
    n_layers: int = 4
    layer_thicknesses: Tuple[float, ...] = (5.0, 15.0, 35.0, 65.0)
    interface_widths: Tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)

    # Element classification
    free_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    linked_indices: List[int] = field(default_factory=lambda: [4])
    substrate_indices: List[int] = field(default_factory=lambda: [5])

    # Stoichiometry
    oxygen_coefficients: Tuple[float, ...] = (0.0, 1.5, 2.0, 2.0)

    # Optimization
    optimize_thicknesses: bool = True
    optimize_widths: bool = False  # Fixed interface width is recommended

    # Solver settings
    solver_method: str = 'SLSQP'  # 'SLSQP', 'L-BFGS-B', 'trust-constr'
    regularization_method: str = 'L2'  # 'L2', 'L1', 'MEM', 'LLS'
    regularization_alpha: float = 0.001
    max_iterations: int = 100
    tolerance: float = 1e-8

    # SemiFreeSolver: initial guess for substrate interface depth
    thickness_init: Optional[float] = None  # Å; if None, use depth_end / 3
    abruptness_init: Optional[float] = None  # Å; if None, use interface_widths[0] or 30.0

    # Bounds
    thickness_bounds: Tuple[float, float] = (1.0, 150.0)
    width_bounds: Tuple[float, float] = (0.5, 20.0)

    # Cross-section correction
    cross_section_correction: bool = False  # Apply PICS correction (like MATLAB GUI)
    cross_section_source: str = 'auto'  # 'auto', 'scofield', 'yeh_lindau'

    # MATLAB-compatible Xreg weighting
    # Weights Chi² by 1/mean(Xdata)² per element, emphasizing trace elements
    xreg_weighting: bool = False

    # Constraint options (MATLAB-compatible: always enabled by default)
    use_sum_constraint: bool = True     # Enforce Σconcentration = 1 at each depth
    use_charge_constraint: bool = True  # Enforce charge neutrality


@dataclass
class ElementDef:
    """Element definition for model configuration."""
    label: str
    symbol: str
    orbital: str = '1s'
    compound: str = ''
    binding_energy: float = 0.0
    kinetic_energy: float = 0.0
    oxidation_state: int = 0
    is_substrate: bool = False


@dataclass
class AnalysisConfig:
    """Complete analysis configuration."""
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # Elements
    elements: List[ElementDef] = field(default_factory=list)

    # Input/Output
    input_file: str = ''
    output_dir: str = './results'

    # Options
    normalize_input: bool = True
    normalization_mode: str = 'angle'  # 'angle' or 'element'

    # Reporting
    save_profile_json: bool = True
    save_profile_csv: bool = True
    save_plot: bool = True
    verbose: bool = False


def load_config_yaml(path: Union[str, Path]) -> AnalysisConfig:
    """Load analysis configuration from YAML file.

    Example YAML:
        model:
          model_type: multilayer
          depth_end: 100.0
          n_layers: 4
          layer_thicknesses: [5.0, 15.0, 35.0, 65.0]
          interface_widths: [2.0, 3.0, 4.0, 5.0]
          optimize_thicknesses: true
          optimize_widths: false
          solver_method: SLSQP

        elements:
          - label: C
            symbol: C
            orbital: 1s
            binding_energy: 284.8
          - label: Al
            symbol: Al
            orbital: 2p
            binding_energy: 74.7
            oxidation_state: 3

        input_file: data.csv
        output_dir: ./results

    Args:
        path: Path to YAML configuration file

    Returns:
        AnalysisConfig object
    """
    path = Path(path)

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # Parse model config
    model_data = data.get('model', {})

    # Convert lists to tuples for immutable fields
    if 'layer_thicknesses' in model_data:
        model_data['layer_thicknesses'] = tuple(model_data['layer_thicknesses'])
    if 'interface_widths' in model_data:
        model_data['interface_widths'] = tuple(model_data['interface_widths'])
    if 'oxygen_coefficients' in model_data:
        model_data['oxygen_coefficients'] = tuple(model_data['oxygen_coefficients'])
    if 'thickness_bounds' in model_data:
        model_data['thickness_bounds'] = tuple(model_data['thickness_bounds'])
    if 'width_bounds' in model_data:
        model_data['width_bounds'] = tuple(model_data['width_bounds'])

    model = ModelConfig(**model_data)

    # Parse elements
    elements = []
    for elem_data in data.get('elements', []):
        elements.append(ElementDef(**elem_data))

    # Build config
    config = AnalysisConfig(
        model=model,
        elements=elements,
        input_file=data.get('input_file', ''),
        output_dir=data.get('output_dir', './results'),
        normalize_input=data.get('normalize_input', True),
        normalization_mode=data.get('normalization_mode', 'angle'),
        save_profile_json=data.get('save_profile_json', True),
        save_profile_csv=data.get('save_profile_csv', True),
        save_plot=data.get('save_plot', True),
        verbose=data.get('verbose', False),
    )

    return config


def save_config_yaml(config: AnalysisConfig, path: Union[str, Path]) -> None:
    """Save analysis configuration to YAML file.

    Args:
        config: AnalysisConfig object
        path: Output path
    """
    path = Path(path)

    # Convert dataclasses to dicts
    data = {
        'model': asdict(config.model),
        'elements': [asdict(e) for e in config.elements],
        'input_file': config.input_file,
        'output_dir': config.output_dir,
        'normalize_input': config.normalize_input,
        'normalization_mode': config.normalization_mode,
        'save_profile_json': config.save_profile_json,
        'save_profile_csv': config.save_profile_csv,
        'save_plot': config.save_plot,
        'verbose': config.verbose,
    }

    # Convert tuples to lists for YAML
    model = data['model']
    for key in ['layer_thicknesses', 'interface_widths', 'oxygen_coefficients',
                'thickness_bounds', 'width_bounds']:
        if key in model and isinstance(model[key], tuple):
            model[key] = list(model[key])

    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def create_default_config(
    model_type: str = 'multilayer',
    elements: Optional[List[str]] = None,
) -> AnalysisConfig:
    """Create default configuration for common use cases.

    Args:
        model_type: 'multilayer' for C/Al2O3/TiO2/SiO2/Si,
                   'sio2_si' for simple SiO2/Si
        elements: Optional element labels

    Returns:
        AnalysisConfig with default settings
    """
    if model_type == 'multilayer':
        elements = [
            ElementDef(label='C', symbol='C', orbital='1s',
                      binding_energy=284.8, oxidation_state=0),
            ElementDef(label='Al', symbol='Al', orbital='2p',
                      binding_energy=74.7, compound='Al2O3', oxidation_state=3),
            ElementDef(label='Ti', symbol='Ti', orbital='2p',
                      binding_energy=458.5, compound='TiO2', oxidation_state=4),
            ElementDef(label='Si_ox', symbol='Si', orbital='2p',
                      binding_energy=103.5, compound='SiO2', oxidation_state=4),
            ElementDef(label='O', symbol='O', orbital='1s',
                      binding_energy=531.0, oxidation_state=-2),
            ElementDef(label='Si_bulk', symbol='Si', orbital='2p',
                      binding_energy=99.3, is_substrate=True, oxidation_state=0),
        ]

        model = ModelConfig(
            model_type='multilayer',
            n_layers=4,
            layer_thicknesses=(5.0, 15.0, 35.0, 65.0),
            interface_widths=(2.0, 3.0, 4.0, 5.0),
            free_indices=[0, 1, 2, 3],
            linked_indices=[4],
            substrate_indices=[5],
            oxygen_coefficients=(0.0, 1.5, 2.0, 2.0),
            optimize_thicknesses=True,
            optimize_widths=False,
        )

    elif model_type == 'sio2_si':
        elements = [
            ElementDef(label='Si_ox', symbol='Si', orbital='2p',
                      binding_energy=103.5, compound='SiO2', oxidation_state=4),
            ElementDef(label='O', symbol='O', orbital='1s',
                      binding_energy=531.0, oxidation_state=-2),
            ElementDef(label='Si_bulk', symbol='Si', orbital='2p',
                      binding_energy=99.3, is_substrate=True, oxidation_state=0),
        ]

        model = ModelConfig(
            model_type='film_structure',
            n_layers=1,
            layer_thicknesses=(30.0,),
            interface_widths=(5.0,),
            free_indices=[0],
            linked_indices=[1],
            substrate_indices=[2],
            oxygen_coefficients=(2.0,),
            optimize_thicknesses=True,
            optimize_widths=False,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return AnalysisConfig(model=model, elements=elements)


# ============================================================================
# Result Output
# ============================================================================

@dataclass
class SolverResultData:
    """Serializable solver result for JSON output."""
    # Depth profile
    depth: List[float]
    concentrations: Dict[str, List[float]]  # element_label -> concentration array

    # Fit quality
    chi2: float
    rmse: float

    # Optimized parameters
    layer_thicknesses: List[float]
    interface_widths: List[float]

    # Metadata
    n_iterations: int
    elapsed_time: float
    success: bool
    solver_method: str


_RESULT_JSON_FIELDS = ('D_opt', 'chi2', 'rmse', 'layer_thicknesses', 'interface_widths',
                       'n_iterations', 'elapsed_time', 'success')


def save_result_json(
    result: 'GenericLayerResult',
    depth: np.ndarray,
    element_labels: List[str],
    path: Union[str, Path],
    config: Optional[AnalysisConfig] = None,
) -> None:
    """Save a layer-model solver result to JSON.

    Args:
        result: a result with the fields in ``_RESULT_JSON_FIELDS``, e.g.
            ``GenericLayerResult`` (GenericLayerSolver and the MLX adapter).
            ``SemiFreeSolverResult`` and ``SolverResult`` have no layer
            thicknesses / interface widths; save them with
            :func:`save_profile_csv`.
        depth: Depth array
        element_labels: Element labels
        path: Output path
        config: Optional configuration used

    Raises:
        TypeError: if ``result`` lacks any of those fields (nothing is written).
    """
    missing = [f for f in _RESULT_JSON_FIELDS if not hasattr(result, f)]
    if missing:
        raise TypeError(
            f"save_result_json() needs a layer-model result (e.g. GenericLayerResult); "
            f"{type(result).__name__} has no {', '.join(missing)}. "
            f"Use save_profile_csv() for other results.")

    path = Path(path)

    # Build concentrations dict
    concentrations = {}
    for i, label in enumerate(element_labels):
        concentrations[label] = result.D_opt[:, i].tolist()

    data = {
        'depth': depth.tolist(),
        'concentrations': concentrations,
        'chi2': float(result.chi2),
        'rmse': float(result.rmse),
        'layer_thicknesses': result.layer_thicknesses.tolist(),
        'interface_widths': result.interface_widths.tolist(),
        'n_iterations': int(result.n_iterations),
        'elapsed_time': float(result.elapsed_time),
        'success': bool(result.success),
    }

    if config is not None:
        data['config'] = {
            'model_type': config.model.model_type,
            'solver_method': config.model.solver_method,
            'regularization': config.model.regularization_method,
            'alpha': config.model.regularization_alpha,
        }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_profile_csv(
    depth: np.ndarray,
    concentrations: np.ndarray,
    element_labels: List[str],
    path: Union[str, Path],
) -> None:
    """Save depth profile to CSV.

    Args:
        depth: Depth array (n_depth,)
        concentrations: Concentration matrix (n_depth, n_elements)
        element_labels: Element labels
        path: Output path
    """
    path = Path(path)

    with open(path, 'w') as f:
        # Header
        header = ['depth'] + element_labels
        f.write(','.join(header) + '\n')

        # Data
        for i, z in enumerate(depth):
            row = [f'{z:.2f}'] + [f'{c:.6f}' for c in concentrations[i]]
            f.write(','.join(row) + '\n')


# ============================================================================
# VAMAS Format
# ============================================================================

def load_vamas(
    path: Union[str, Path],
    angles: Optional[List[float]] = None,
) -> IntensityData:
    """Load data from VAMAS format.

    VAMAS (Versailles Project on Advanced Materials and Standards) is the
    ISO standard format (ISO 14976) for surface analysis data.

    Args:
        path: Path to VAMAS file (.vms)
        angles: Specific angles to extract (None = all available)

    Returns:
        IntensityData object
    """
    from .vamas_parser import load_vamas as _load_vamas

    vamas = _load_vamas(path)
    return vamas.to_intensity_data(angles=angles)


def load_kratos(
    path: Union[str, Path],
    angles: Optional[List[float]] = None,
) -> IntensityData:
    """Load data from Kratos XPS format.

    Args:
        path: Path to Kratos file
        angles: Specific angles to extract (None = use default)

    Returns:
        IntensityData object
    """
    from .vamas_parser import load_kratos as _load_kratos

    kratos = _load_kratos(path)
    return kratos.to_intensity_data(angles=angles)
