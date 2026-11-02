"""
Flexible Multi-Layer Model Definition
======================================

Hybrid approach: YAML-driven configuration + Python extensibility.

Design Goals:
1. Users define layer structure in YAML (no code changes needed)
2. Automatic constraint generation from compound formulas
3. Extensible profile functions (erf, exponential, gaussian, etc.)
4. Python API for programmatic model building

Usage:
    # YAML-driven
    model = LayerModel.from_yaml('my_structure.yaml')
    solver = model.create_solver(transfer_matrix)
    result = solver.solve(I_obs)

    # Builder pattern (Python)
    model = (
        LayerModelBuilder()
        .add_contamination('C', decay_length=3.0)
        .add_oxide('Al2O3', thickness=10.0, width=3.0)
        .add_oxide('TiO2', thickness=20.0, width=4.0)
        .add_oxide('SiO2', thickness=25.0, width=5.0)
        .set_substrate('Si')
        .build()
    )

Date: 2026-01-23
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Callable
from enum import Enum
import re
from scipy.special import erf
import yaml
from pathlib import Path


# ============================================================================
# Enums and Constants
# ============================================================================

class ProfileType(Enum):
    """Layer profile function types."""
    ERF = 'erf'                 # Error function (smooth interface)
    EXPONENTIAL = 'exponential' # Exponential decay (contamination)
    GAUSSIAN = 'gaussian'       # Gaussian peak
    STEP = 'step'               # Sharp step function
    SUBSTRATE = 'substrate'     # Substrate (1 at deep)
    CUSTOM = 'custom'           # User-defined function


class ElementRole(Enum):
    """Element role in the model."""
    FREE = 'free'           # Freely optimized (cations)
    LINKED = 'linked'       # Determined by stoichiometry (anions)
    SUBSTRATE = 'substrate' # Substrate element
    FIXED = 'fixed'         # Fixed concentration


# Common oxidation states for automatic inference
OXIDATION_STATES = {
    'Li': +1, 'Na': +1, 'K': +1, 'Rb': +1, 'Cs': +1,
    'Be': +2, 'Mg': +2, 'Ca': +2, 'Sr': +2, 'Ba': +2,
    'Al': +3, 'Ga': +3, 'In': +3,
    'Si': +4, 'Ge': +4, 'Sn': +4, 'Ti': +4, 'Zr': +4, 'Hf': +4,
    'V': +5, 'Nb': +5, 'Ta': +5,
    'Cr': +6, 'Mo': +6, 'W': +6,
    'Mn': +4, 'Fe': +3, 'Co': +3, 'Ni': +2, 'Cu': +2, 'Zn': +2,
    'O': -2, 'S': -2, 'Se': -2, 'Te': -2,
    'F': -1, 'Cl': -1, 'Br': -1, 'I': -1,
    'N': -3, 'P': +5, 'As': +5,
    'C': 0,  # Contamination carbon is neutral
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Element:
    """Element definition with physical properties."""
    symbol: str
    label: str                          # Display label (e.g., 'Si_ox', 'Si_bulk')
    binding_energy: float = 0.0         # eV
    kinetic_energy: float = 0.0         # eV (computed from photon energy if not set)
    oxidation_state: int = 0            # For charge balance
    role: ElementRole = ElementRole.FREE
    compound: str = ''                  # Parent compound (e.g., 'SiO2')

    def __post_init__(self):
        # Auto-infer oxidation state if not set
        if self.oxidation_state == 0 and self.symbol in OXIDATION_STATES:
            self.oxidation_state = OXIDATION_STATES[self.symbol]


@dataclass
class Layer:
    """Layer definition with profile parameters."""
    name: str
    compound: str                       # e.g., 'Al2O3', 'TiO2', 'C'
    elements: List[str]                 # Element labels in this layer
    stoichiometry: List[float]          # Molar ratios
    profile_type: ProfileType = ProfileType.ERF
    thickness: float = 10.0             # Initial thickness (Å)
    width: float = 3.0                  # Interface width FWHM (Å)
    optimize_thickness: bool = True
    optimize_width: bool = False        # Recommend: False
    concentration: float = 1.0          # Max concentration in layer
    decay_length: float = 3.0           # For exponential profile

    @classmethod
    def from_compound(cls, compound: str, **kwargs) -> 'Layer':
        """Create layer from compound formula (e.g., 'Al2O3', 'TiO2')."""
        elements, stoich = parse_compound(compound)
        return cls(
            name=compound,
            compound=compound,
            elements=elements,
            stoichiometry=stoich,
            **kwargs,
        )


@dataclass
class LayerModel:
    """Complete multi-layer model definition."""
    layers: List[Layer]
    elements: List[Element]
    substrate_element: str              # Label of substrate element
    photon_energy: float = 1486.6       # Al Kα default
    depth_start: float = 0.0
    depth_end: float = 100.0
    depth_step: float = 1.0

    def __post_init__(self):
        self._build_element_index()
        self._compute_oxygen_coefficients()

    def _build_element_index(self):
        """Build element label -> index mapping."""
        self.element_index = {e.label: i for i, e in enumerate(self.elements)}
        self.n_elements = len(self.elements)

    def _compute_oxygen_coefficients(self):
        """Compute oxygen stoichiometry coefficients for linked elements."""
        self.oxygen_coefficients = []

        # Find oxygen element index
        self.oxygen_idx = None
        for i, e in enumerate(self.elements):
            if e.symbol == 'O' and e.role == ElementRole.LINKED:
                self.oxygen_idx = i
                break

        if self.oxygen_idx is None:
            return

        # For each free element, compute O coefficient from parent compound
        for e in self.elements:
            if e.role == ElementRole.FREE:
                coeff = self._get_oxygen_ratio(e)
                self.oxygen_coefficients.append(coeff)
            else:
                self.oxygen_coefficients.append(0.0)

    def _get_oxygen_ratio(self, element: Element) -> float:
        """Get O/cation ratio from compound formula."""
        if not element.compound:
            return 0.0

        elements, stoich = parse_compound(element.compound)

        # Find oxygen and cation in formula
        o_ratio = 0.0
        cation_ratio = 0.0

        for el, st in zip(elements, stoich):
            if el == 'O':
                o_ratio = st
            elif el == element.symbol:
                cation_ratio = st

        if cation_ratio > 0:
            return o_ratio / cation_ratio
        return 0.0

    @property
    def free_indices(self) -> List[int]:
        return [i for i, e in enumerate(self.elements) if e.role == ElementRole.FREE]

    @property
    def linked_indices(self) -> List[int]:
        return [i for i, e in enumerate(self.elements) if e.role == ElementRole.LINKED]

    @property
    def substrate_indices(self) -> List[int]:
        return [i for i, e in enumerate(self.elements) if e.role == ElementRole.SUBSTRATE]

    @property
    def n_layers(self) -> int:
        return len([l for l in self.layers if l.profile_type != ProfileType.SUBSTRATE])

    @property
    def layer_thicknesses(self) -> Tuple[float, ...]:
        """Get cumulative layer interface positions."""
        thicknesses = []
        cumulative = 0.0
        for layer in self.layers:
            if layer.profile_type not in (ProfileType.SUBSTRATE, ProfileType.EXPONENTIAL):
                cumulative += layer.thickness
                thicknesses.append(cumulative)
            elif layer.profile_type == ProfileType.EXPONENTIAL:
                # Contamination layer: use decay_length as "thickness"
                thicknesses.append(layer.decay_length)
        return tuple(thicknesses)

    @property
    def interface_widths(self) -> Tuple[float, ...]:
        return tuple(l.width for l in self.layers if l.profile_type != ProfileType.SUBSTRATE)

    def build_depth_profile(
        self,
        thicknesses: np.ndarray,
        widths: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """Build concentration profile from structural parameters.

        Args:
            thicknesses: Layer interface positions
            widths: Interface widths FWHM
            depth: Depth grid

        Returns:
            Concentration array [n_depth, n_elements]
        """
        n_depth = len(depth)
        D = np.zeros((n_depth, self.n_elements))

        # Convert FWHM to sigma
        sigmas = widths / (2 * np.sqrt(np.log(2)))

        layer_idx = 0
        prev_interface = 0.0

        for layer in self.layers:
            if layer.profile_type == ProfileType.SUBSTRATE:
                # Substrate: rises at last interface
                last_interface = thicknesses[-1] if len(thicknesses) > 0 else 0
                last_sigma = sigmas[-1] if len(sigmas) > 0 else 3.0
                profile = 0.5 * (1 + erf((depth - last_interface) / last_sigma))

                sub_idx = self.element_index.get(self.substrate_element)
                if sub_idx is not None:
                    D[:, sub_idx] = profile

            elif layer.profile_type == ProfileType.EXPONENTIAL:
                # Exponential decay (contamination)
                decay = layer.decay_length
                interface = thicknesses[layer_idx] if layer_idx < len(thicknesses) else decay
                sigma = sigmas[layer_idx] if layer_idx < len(sigmas) else 2.0

                # Exponential decay cut off at interface
                cutoff = 0.5 * (1 - erf((depth - interface) / sigma))
                profile = layer.concentration * np.exp(-depth / decay) * cutoff

                for el_label in layer.elements:
                    if el_label in self.element_index:
                        D[:, self.element_index[el_label]] += profile

                layer_idx += 1

            elif layer.profile_type == ProfileType.ERF:
                # Error function layer (oxide)
                if layer_idx >= len(thicknesses):
                    continue

                z_end = thicknesses[layer_idx]
                sigma_end = sigmas[layer_idx]
                sigma_start = sigmas[layer_idx - 1] if layer_idx > 0 else sigmas[0]

                # Layer bounded by two error functions
                rising = 0.5 * (1 + erf((depth - prev_interface) / sigma_start))
                falling = 0.5 * (1 + erf((depth - z_end) / sigma_end))
                profile = np.maximum(rising - falling, 0)

                # Apply concentration to cations
                for el_label, stoich in zip(layer.elements, layer.stoichiometry):
                    if el_label in self.element_index:
                        el = self.elements[self.element_index[el_label]]
                        if el.role == ElementRole.FREE:
                            # Cation concentration from stoichiometry
                            total_stoich = sum(layer.stoichiometry)
                            conc = stoich / total_stoich * layer.concentration
                            D[:, self.element_index[el_label]] += conc * profile

                prev_interface = z_end
                layer_idx += 1

        # Compute linked elements (oxygen) from stoichiometry
        if self.oxygen_idx is not None:
            o_coeffs = np.array(self.oxygen_coefficients)
            D_free = D[:, self.free_indices]
            D[:, self.oxygen_idx] = (D_free * o_coeffs[self.free_indices]).sum(axis=1)

        # Normalize
        D = np.clip(D, 1e-6, 1.0)
        total = D.sum(axis=1, keepdims=True)
        D = D / np.maximum(total, 1e-10)

        return D

    def to_solver_config(self):
        """Convert to MultiLayerFilmConfig for solver (unavailable).

        Raises:
            NotImplementedError: always. MultiLayerFilmConfig / MultiLayerFilmSolver
            have been removed from ``deppro.inverse_solver``. Configure a current
            solver (e.g. ``SemiFreeSolverConfig``) directly.
            ``deppro.MultiLayerFilmConfig`` (from ``deppro.depth_profile``) is a
            different class and is not accepted by any solver.
        """
        raise NotImplementedError(
            "LayerModel.to_solver_config() builds a MultiLayerFilmConfig for "
            "MultiLayerFilmSolver, which has been removed from deppro.inverse_solver. "
            "Configure a current solver (e.g. SemiFreeSolverConfig) directly; "
            "deppro.MultiLayerFilmConfig (deppro.depth_profile) is a different class.")
        # The body below targets the removed MultiLayerFilmSolver; kept for porting.
        from .inverse_solver import MultiLayerFilmConfig

        # Get oxygen coefficients for free elements only
        free_o_coeffs = [self.oxygen_coefficients[i] for i in self.free_indices]

        return MultiLayerFilmConfig(
            n_layers=self.n_layers,
            layer_thicknesses=self.layer_thicknesses,
            interface_widths=self.interface_widths,
            optimize_thicknesses=True,
            optimize_abruptnesses=False,  # Fixed recommended
            free_indices=self.free_indices,
            linked_indices=self.linked_indices,
            substrate_indices=self.substrate_indices,
            oxygen_coefficients=tuple(free_o_coeffs),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'LayerModel':
        """Load model definition from YAML file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict) -> 'LayerModel':
        """Create model from dictionary (parsed YAML)."""
        # Parse elements
        elements = []
        for el_data in data.get('elements', []):
            role = ElementRole(el_data.get('role', 'free'))
            elements.append(Element(
                symbol=el_data['symbol'],
                label=el_data.get('label', el_data['symbol']),
                binding_energy=el_data.get('binding_energy', 0.0),
                kinetic_energy=el_data.get('kinetic_energy', 0.0),
                oxidation_state=el_data.get('oxidation_state', 0),
                role=role,
                compound=el_data.get('compound', ''),
            ))

        # Parse layers
        layers = []
        for layer_data in data.get('layers', []):
            profile_type = ProfileType(layer_data.get('profile', 'erf'))

            # Auto-parse compound if elements not specified
            if 'elements' in layer_data:
                el_labels = layer_data['elements']
                stoich = layer_data.get('stoichiometry', [1.0] * len(el_labels))
            else:
                compound = layer_data.get('compound', layer_data.get('name', ''))
                el_symbols, stoich = parse_compound(compound)
                # Map symbols to labels
                el_labels = []
                for sym in el_symbols:
                    # Find matching element label
                    for e in elements:
                        if e.symbol == sym and e.compound == compound:
                            el_labels.append(e.label)
                            break
                    else:
                        # Fallback: use symbol as label
                        el_labels.append(sym)

            layers.append(Layer(
                name=layer_data.get('name', ''),
                compound=layer_data.get('compound', ''),
                elements=el_labels,
                stoichiometry=stoich,
                profile_type=profile_type,
                thickness=layer_data.get('thickness', 10.0),
                width=layer_data.get('width', 3.0),
                optimize_thickness=layer_data.get('optimize_thickness', True),
                optimize_width=layer_data.get('optimize_width', False),
                concentration=layer_data.get('concentration', 1.0),
                decay_length=layer_data.get('decay_length', 3.0),
            ))

        return cls(
            layers=layers,
            elements=elements,
            substrate_element=data.get('substrate_element', 'Si'),
            photon_energy=data.get('photon_energy', 1486.6),
            depth_start=data.get('depth_start', 0.0),
            depth_end=data.get('depth_end', 100.0),
            depth_step=data.get('depth_step', 1.0),
        )

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save model definition to YAML file."""
        data = {
            'photon_energy': self.photon_energy,
            'depth_start': self.depth_start,
            'depth_end': self.depth_end,
            'depth_step': self.depth_step,
            'substrate_element': self.substrate_element,
            'elements': [
                {
                    'symbol': e.symbol,
                    'label': e.label,
                    'binding_energy': e.binding_energy,
                    'kinetic_energy': e.kinetic_energy,
                    'oxidation_state': e.oxidation_state,
                    'role': e.role.value,
                    'compound': e.compound,
                }
                for e in self.elements
            ],
            'layers': [
                {
                    'name': l.name,
                    'compound': l.compound,
                    'elements': l.elements,
                    'stoichiometry': l.stoichiometry,
                    'profile': l.profile_type.value,
                    'thickness': l.thickness,
                    'width': l.width,
                    'optimize_thickness': l.optimize_thickness,
                    'optimize_width': l.optimize_width,
                    'concentration': l.concentration,
                    'decay_length': l.decay_length,
                }
                for l in self.layers
            ],
        }

        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ============================================================================
# Builder Pattern
# ============================================================================

class LayerModelBuilder:
    """Fluent builder for LayerModel."""

    def __init__(self):
        self._layers: List[Layer] = []
        self._elements: List[Element] = []
        self._substrate_element: str = ''
        self._photon_energy: float = 1486.6
        self._depth_end: float = 100.0
        self._depth_step: float = 1.0

        # Track element labels to avoid duplicates
        self._element_labels: set = set()

    def set_photon_energy(self, energy: float) -> 'LayerModelBuilder':
        """Set X-ray photon energy (eV)."""
        self._photon_energy = energy
        return self

    def set_depth_range(self, end: float, step: float = 1.0) -> 'LayerModelBuilder':
        """Set depth grid parameters."""
        self._depth_end = end
        self._depth_step = step
        return self

    def add_contamination(
        self,
        symbol: str = 'C',
        decay_length: float = 3.0,
        binding_energy: float = 284.8,
    ) -> 'LayerModelBuilder':
        """Add contamination layer (exponential decay)."""
        label = symbol
        if label in self._element_labels:
            label = f'{symbol}_cont'

        self._elements.append(Element(
            symbol=symbol,
            label=label,
            binding_energy=binding_energy,
            oxidation_state=0,
            role=ElementRole.FREE,
        ))
        self._element_labels.add(label)

        self._layers.append(Layer(
            name='contamination',
            compound=symbol,
            elements=[label],
            stoichiometry=[1.0],
            profile_type=ProfileType.EXPONENTIAL,
            decay_length=decay_length,
            width=2.0,
        ))

        return self

    def add_oxide(
        self,
        compound: str,
        thickness: float,
        width: float = 3.0,
        cation_binding_energy: float = 0.0,
    ) -> 'LayerModelBuilder':
        """Add oxide layer (e.g., 'Al2O3', 'TiO2', 'SiO2').

        Automatically:
        - Parses compound formula
        - Creates cation element (FREE role)
        - Adds oxygen contribution to shared O element
        """
        el_symbols, stoich = parse_compound(compound)

        # Find cation and oxygen
        cation_sym = None
        cation_stoich = 0
        o_stoich = 0

        for sym, st in zip(el_symbols, stoich):
            if sym == 'O':
                o_stoich = st
            else:
                cation_sym = sym
                cation_stoich = st

        if cation_sym is None:
            raise ValueError(f"No cation found in {compound}")

        # Create unique cation label
        cation_label = f'{cation_sym}_{compound}'
        if cation_label in self._element_labels:
            cation_label = f'{cation_sym}_{compound}_{len(self._layers)}'

        # Add cation element
        ox_state = OXIDATION_STATES.get(cation_sym, 0)
        self._elements.append(Element(
            symbol=cation_sym,
            label=cation_label,
            binding_energy=cation_binding_energy,
            oxidation_state=ox_state,
            role=ElementRole.FREE,
            compound=compound,
        ))
        self._element_labels.add(cation_label)

        # Add oxygen element (shared, LINKED role)
        o_label = 'O'
        if o_label not in self._element_labels:
            self._elements.append(Element(
                symbol='O',
                label=o_label,
                binding_energy=531.0,  # Typical O 1s
                oxidation_state=-2,
                role=ElementRole.LINKED,
            ))
            self._element_labels.add(o_label)

        # Add layer
        self._layers.append(Layer(
            name=compound,
            compound=compound,
            elements=[cation_label, o_label],
            stoichiometry=[cation_stoich, o_stoich],
            profile_type=ProfileType.ERF,
            thickness=thickness,
            width=width,
        ))

        return self

    def set_substrate(
        self,
        symbol: str,
        binding_energy: float = 99.3,
    ) -> 'LayerModelBuilder':
        """Set substrate element."""
        label = f'{symbol}_bulk'
        if label in self._element_labels:
            label = f'{symbol}_substrate'

        self._elements.append(Element(
            symbol=symbol,
            label=label,
            binding_energy=binding_energy,
            oxidation_state=0,
            role=ElementRole.SUBSTRATE,
        ))
        self._element_labels.add(label)
        self._substrate_element = label

        # Add substrate "layer" (special profile)
        self._layers.append(Layer(
            name='substrate',
            compound=symbol,
            elements=[label],
            stoichiometry=[1.0],
            profile_type=ProfileType.SUBSTRATE,
        ))

        return self

    def build(self) -> LayerModel:
        """Build the LayerModel."""
        if not self._substrate_element:
            raise ValueError("Substrate element not set. Call set_substrate() first.")

        return LayerModel(
            layers=self._layers,
            elements=self._elements,
            substrate_element=self._substrate_element,
            photon_energy=self._photon_energy,
            depth_end=self._depth_end,
            depth_step=self._depth_step,
        )


# ============================================================================
# Utility Functions
# ============================================================================

def parse_compound(formula: str) -> Tuple[List[str], List[float]]:
    """Parse compound formula into elements and stoichiometry.

    Examples:
        'Al2O3' -> (['Al', 'O'], [2.0, 3.0])
        'TiO2'  -> (['Ti', 'O'], [1.0, 2.0])
        'SiO2'  -> (['Si', 'O'], [1.0, 2.0])
        'C'     -> (['C'], [1.0])

    Args:
        formula: Chemical formula string

    Returns:
        Tuple of (element symbols, stoichiometry coefficients)
    """
    # Regex: element symbol (1-2 letters) followed by optional number
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)

    elements = []
    stoich = []

    for element, count in matches:
        if element:
            elements.append(element)
            stoich.append(float(count) if count else 1.0)

    return elements, stoich


# ============================================================================
# Preset Models
# ============================================================================

def create_standard_multilayer() -> LayerModel:
    """Create standard C/Al2O3/TiO2/SiO2/Si model."""
    return (
        LayerModelBuilder()
        .set_photon_energy(1486.6)
        .set_depth_range(100.0, 1.0)
        .add_contamination('C', decay_length=3.0, binding_energy=284.8)
        .add_oxide('Al2O3', thickness=10.0, width=3.0, cation_binding_energy=74.7)
        .add_oxide('TiO2', thickness=20.0, width=4.0, cation_binding_energy=458.5)
        .add_oxide('SiO2', thickness=25.0, width=5.0, cation_binding_energy=103.5)
        .set_substrate('Si', binding_energy=99.3)
        .build()
    )


def create_sio2_si_model(thickness: float = 30.0, width: float = 5.0) -> LayerModel:
    """Create simple SiO2/Si model."""
    return (
        LayerModelBuilder()
        .set_photon_energy(1486.6)
        .set_depth_range(80.0, 0.5)
        .add_oxide('SiO2', thickness=thickness, width=width, cation_binding_energy=103.5)
        .set_substrate('Si', binding_energy=99.3)
        .build()
    )


# ============================================================================
# Example YAML Template
# ============================================================================

EXAMPLE_YAML = """
# Flexible Layer Model Definition
# ================================
# This YAML defines a multi-layer structure for ARXPS depth profiling.

photon_energy: 1486.6  # Al Ka (eV)
depth_start: 0.0
depth_end: 100.0
depth_step: 1.0
substrate_element: Si_bulk

# Element definitions
# role: free (optimized cation), linked (from stoichiometry), substrate
elements:
  - symbol: C
    label: C
    binding_energy: 284.8
    role: free

  - symbol: Al
    label: Al_Al2O3
    binding_energy: 74.7
    oxidation_state: 3
    role: free
    compound: Al2O3

  - symbol: Ti
    label: Ti_TiO2
    binding_energy: 458.5
    oxidation_state: 4
    role: free
    compound: TiO2

  - symbol: Si
    label: Si_SiO2
    binding_energy: 103.5
    oxidation_state: 4
    role: free
    compound: SiO2

  - symbol: O
    label: O
    binding_energy: 531.0
    oxidation_state: -2
    role: linked

  - symbol: Si
    label: Si_bulk
    binding_energy: 99.3
    oxidation_state: 0
    role: substrate

# Layer stack (top to bottom)
layers:
  - name: contamination
    compound: C
    elements: [C]
    stoichiometry: [1.0]
    profile: exponential
    decay_length: 3.0
    width: 2.0

  - name: Al2O3
    compound: Al2O3
    elements: [Al_Al2O3, O]
    stoichiometry: [2.0, 3.0]
    profile: erf
    thickness: 10.0
    width: 3.0
    optimize_thickness: true
    optimize_width: false

  - name: TiO2
    compound: TiO2
    elements: [Ti_TiO2, O]
    stoichiometry: [1.0, 2.0]
    profile: erf
    thickness: 20.0
    width: 4.0

  - name: SiO2
    compound: SiO2
    elements: [Si_SiO2, O]
    stoichiometry: [1.0, 2.0]
    profile: erf
    thickness: 25.0
    width: 5.0

  - name: substrate
    compound: Si
    elements: [Si_bulk]
    stoichiometry: [1.0]
    profile: substrate
"""


if __name__ == '__main__':
    # Demo: create model using builder
    print("Creating model via Builder pattern...")
    model = create_standard_multilayer()

    print(f"\nElements ({model.n_elements}):")
    for i, e in enumerate(model.elements):
        print(f"  {i}: {e.label} ({e.symbol}) - {e.role.value}")

    print(f"\nLayers ({model.n_layers}):")
    for l in model.layers:
        print(f"  {l.name}: {l.compound} ({l.profile_type.value})")

    print(f"\nFree indices: {model.free_indices}")
    print(f"Linked indices: {model.linked_indices}")
    print(f"Substrate indices: {model.substrate_indices}")
    print(f"Layer thicknesses: {model.layer_thicknesses}")
    print(f"Interface widths: {model.interface_widths}")
    print(f"Oxygen coefficients: {model.oxygen_coefficients}")

    # Save example YAML
    print("\n\nSaving example YAML...")
    with open('/tmp/example_model.yaml', 'w') as f:
        f.write(EXAMPLE_YAML)
    print("Saved: /tmp/example_model.yaml")

    # Load and verify
    print("\nLoading from YAML...")
    model2 = LayerModel.from_yaml('/tmp/example_model.yaml')
    print(f"Loaded model with {model2.n_elements} elements, {model2.n_layers} layers")
