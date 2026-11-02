"""
Depth Profile Data Structures and Distribution Functions
========================================================

Provides data structures for representing depth profiles and
distribution functions for generating concentration profiles.

Usage:
    from deppro.depth_profile import DepthProfile, LayerConfig, create_five_layer_profile

    # Create a multi-layer profile
    profile = DepthProfile.from_json('profile.json')

    # Or programmatically
    profile = create_five_layer_profile()
"""

import numpy as np
from ._compat import trapezoid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Tuple
from scipy.special import erf
import json
import h5py

from .imfp import CompoundParams, CompoundDatabase, BindingEnergyDatabase


# Distribution function types
DistributionType = Literal[
    'substrate',     # Error function (substrate)
    'layer',         # Gaussian layer
    'error_func',    # Error function layer
    'exponential',   # Exponential decay
    'rectangle',     # Rectangular profile
    'triangle',      # Triangular profile
    'homogeneous',   # Constant concentration
]


@dataclass
class ElementConfig:
    """Configuration for a single element in the depth profile."""
    symbol: str              # Element symbol (e.g., 'Si', 'O')
    orbital: str             # Orbital (e.g., '2p', '1s')
    compound: str            # Compound name for IMFP (e.g., 'Si', 'SiO2')
    binding_energy: Optional[float] = None  # Override BE (eV), auto-lookup if None

    # Peak parameters for spectrum generation
    gaussian_fwhm: float = 1.0   # Gaussian FWHM (eV)
    lorentzian_fwhm: float = 0.3  # Lorentzian FWHM (eV)

    # Charge/stoichiometry for constraints
    oxidation_state: Optional[float] = None  # e.g., +4 for Ti⁴⁺, -2 for O²⁻
    stoichiometry: Optional[float] = None    # e.g., 1 for Ti in TiO2, 2 for O in TiO2

    def __post_init__(self):
        """Validate and lookup binding energy if needed."""
        if self.binding_energy is None:
            try:
                be_db = BindingEnergyDatabase()
                self.binding_energy = be_db.get(self.symbol, self.orbital)
            except (FileNotFoundError, KeyError):
                pass  # Will be set later or raise error

    @property
    def label(self) -> str:
        """Return element label like 'Si2p' or 'O1s'."""
        return f"{self.symbol}{self.orbital}"

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'orbital': self.orbital,
            'compound': self.compound,
            'binding_energy': self.binding_energy,
            'gaussian_fwhm': self.gaussian_fwhm,
            'lorentzian_fwhm': self.lorentzian_fwhm,
            'oxidation_state': self.oxidation_state,
            'stoichiometry': self.stoichiometry,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ElementConfig':
        return cls(**d)


@dataclass
class LayerConfig:
    """Configuration for a layer in the depth profile."""
    element_index: int           # Index into elements list
    distribution: DistributionType
    concentration: float = 1.0   # Peak concentration (0-1)

    # Position parameters (in Angstroms)
    position: float = 0.0        # Center position
    thickness: float = 10.0      # Layer thickness or interface width
    abruptness: float = 5.0      # Interface abruptness (FWHM for Gaussian)

    # For substrate: these control the error function
    # For layer: position=center, thickness=width

    def to_dict(self) -> dict:
        return {
            'element_index': self.element_index,
            'distribution': self.distribution,
            'concentration': self.concentration,
            'position': self.position,
            'thickness': self.thickness,
            'abruptness': self.abruptness,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'LayerConfig':
        return cls(**d)


@dataclass
class DepthProfile:
    """
    Complete depth profile definition.

    Attributes:
        depth: Depth array in Angstroms [nDepth]
        elements: List of element configurations
        concentrations: Concentration matrix [nDepth, nElements]
        step_size: Depth step size in Angstroms
    """
    depth: np.ndarray
    elements: List[ElementConfig]
    concentrations: np.ndarray
    step_size: float = 1.0

    @property
    def n_depth(self) -> int:
        return len(self.depth)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def element_labels(self) -> List[str]:
        return [e.label for e in self.elements]

    def get_element_index(self, label: str) -> int:
        """Get element index by label (e.g., 'Si2p')."""
        for i, e in enumerate(self.elements):
            if e.label == label or e.symbol == label:
                return i
        raise KeyError(f"Element '{label}' not found")

    def normalize(self, in_place: bool = True) -> 'DepthProfile':
        """
        Normalize concentrations to sum to 1 at each depth.

        Args:
            in_place: Modify this object or return a copy

        Returns:
            Normalized DepthProfile
        """
        if in_place:
            total = self.concentrations.sum(axis=1, keepdims=True)
            total[total == 0] = 1  # Avoid division by zero
            self.concentrations /= total
            return self
        else:
            new_conc = self.concentrations.copy()
            total = new_conc.sum(axis=1, keepdims=True)
            total[total == 0] = 1
            new_conc /= total
            return DepthProfile(
                depth=self.depth.copy(),
                elements=self.elements.copy(),
                concentrations=new_conc,
                step_size=self.step_size
            )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'depth': {
                'start': float(self.depth[0]),
                'end': float(self.depth[-1]),
                'step': float(self.step_size),
            },
            'elements': [e.to_dict() for e in self.elements],
            'concentrations': self.concentrations.tolist(),
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """Save profile to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_h5(self, path: Union[str, Path]) -> None:
        """Save profile to HDF5 file."""
        with h5py.File(path, 'w') as f:
            f.create_dataset('depth', data=self.depth)
            f.create_dataset('concentrations', data=self.concentrations)
            f.attrs['step_size'] = self.step_size
            f.attrs['n_elements'] = self.n_elements

            # Save element configs as group
            elem_grp = f.create_group('elements')
            for i, e in enumerate(self.elements):
                g = elem_grp.create_group(f'element_{i}')
                g.attrs['symbol'] = e.symbol
                g.attrs['orbital'] = e.orbital
                g.attrs['compound'] = e.compound
                g.attrs['binding_energy'] = e.binding_energy or 0.0
                g.attrs['gaussian_fwhm'] = e.gaussian_fwhm
                g.attrs['lorentzian_fwhm'] = e.lorentzian_fwhm

    @classmethod
    def from_dict(cls, d: dict) -> 'DepthProfile':
        """Create from dictionary."""
        depth_cfg = d['depth']
        depth = np.arange(
            depth_cfg['start'],
            depth_cfg['end'] + depth_cfg['step'],
            depth_cfg['step']
        )
        elements = [ElementConfig.from_dict(e) for e in d['elements']]
        concentrations = np.array(d['concentrations'])
        return cls(
            depth=depth,
            elements=elements,
            concentrations=concentrations,
            step_size=depth_cfg['step']
        )

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> 'DepthProfile':
        """Load profile from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_h5(cls, path: Union[str, Path]) -> 'DepthProfile':
        """Load profile from HDF5 file."""
        with h5py.File(path, 'r') as f:
            depth = f['depth'][:]
            concentrations = f['concentrations'][:]
            step_size = f.attrs['step_size']

            elements = []
            elem_grp = f['elements']
            for i in range(f.attrs['n_elements']):
                g = elem_grp[f'element_{i}']
                elements.append(ElementConfig(
                    symbol=g.attrs['symbol'],
                    orbital=g.attrs['orbital'],
                    compound=g.attrs['compound'],
                    binding_energy=g.attrs['binding_energy'] or None,
                    gaussian_fwhm=g.attrs['gaussian_fwhm'],
                    lorentzian_fwhm=g.attrs['lorentzian_fwhm'],
                ))

            return cls(
                depth=depth,
                elements=elements,
                concentrations=concentrations,
                step_size=step_size
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DepthProfile':
        """Load profile from file (auto-detect format)."""
        path = Path(path)
        if path.suffix == '.json':
            return cls.load_json(path)
        elif path.suffix in ('.h5', '.hdf5'):
            return cls.load_h5(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")


class DistributionFunction:
    """
    Distribution functions for generating concentration profiles.

    These mirror the MATLAB CalculateDistribution.m functions.
    """

    @staticmethod
    def error_function(
        depth: np.ndarray,
        thickness: float,
        abruptness: float,
        rising: bool = True
    ) -> np.ndarray:
        """
        Error function distribution (substrate interface).

        Args:
            depth: Depth array (Å)
            thickness: Interface position (Å)
            abruptness: Interface width FWHM (Å)
            rising: If True, increases with depth (substrate)

        Returns:
            Concentration profile [0, 1]
        """
        # Convert FWHM to sigma: sigma = FWHM / (2 * sqrt(ln(2)))
        sigma = abruptness / (2 * np.sqrt(np.log(2)))

        if rising:
            # Substrate: rises from 0 to 1 with depth
            return 0.5 * np.abs(1 + erf((depth - thickness) / sigma))
        else:
            # Layer: decreases from 1 to 0 with depth
            return 0.5 * np.abs(1 + erf(-(depth - thickness) / sigma))

    @staticmethod
    def gaussian(
        depth: np.ndarray,
        position: float,
        fwhm: float,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Gaussian distribution.

        Args:
            depth: Depth array (Å)
            position: Center position (Å)
            fwhm: Full width at half maximum (Å)
            amplitude: Peak amplitude

        Returns:
            Concentration profile
        """
        sigma = fwhm / (2 * np.sqrt(np.log(2)))
        return amplitude * np.exp(-(depth - position)**2 / sigma**2)

    @staticmethod
    def exponential(
        depth: np.ndarray,
        position: float,
        decay_length: float,
        amplitude: float = 1.0,
        rising: bool = False
    ) -> np.ndarray:
        """
        Exponential distribution.

        Args:
            depth: Depth array (Å)
            position: Reference position (Å)
            decay_length: Decay constant (Å)
            amplitude: Peak amplitude
            rising: If True, exponentially increases

        Returns:
            Concentration profile
        """
        if rising:
            return amplitude * np.exp((depth - position) / decay_length)
        else:
            return amplitude * np.exp(-(depth - position) / decay_length)

    @staticmethod
    def rectangle(
        depth: np.ndarray,
        position: float,
        width: float,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Rectangular (box) distribution.

        Args:
            depth: Depth array (Å)
            position: Center position (Å)
            width: Total width (Å)
            amplitude: Amplitude inside the box

        Returns:
            Concentration profile
        """
        result = np.zeros_like(depth)
        mask = (position - width/2 <= depth) & (depth <= position + width/2)
        result[mask] = amplitude
        return result

    @staticmethod
    def triangle(
        depth: np.ndarray,
        position: float,
        width: float,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Triangular distribution.

        Args:
            depth: Depth array (Å)
            position: Peak position (Å)
            width: Half-width at base (Å)
            amplitude: Peak amplitude

        Returns:
            Concentration profile
        """
        result = np.zeros_like(depth)

        # Left side (rising)
        left_mask = (position - width <= depth) & (depth <= position)
        result[left_mask] = amplitude * (1/width * (depth[left_mask] - position) + 1)

        # Right side (falling)
        right_mask = (position <= depth) & (depth <= position + width)
        result[right_mask] = amplitude * (-1/width * (depth[right_mask] - position) + 1)

        return result


class DepthProfileBuilder:
    """
    Builder for creating depth profiles from layer configurations.
    """

    def __init__(
        self,
        depth_start: float = 0.0,
        depth_end: float = 100.0,
        step_size: float = 1.0
    ):
        self.depth = np.arange(depth_start, depth_end + step_size, step_size)
        self.step_size = step_size
        self.elements: List[ElementConfig] = []
        self.layers: List[LayerConfig] = []

    def add_element(self, config: ElementConfig) -> int:
        """Add an element and return its index."""
        self.elements.append(config)
        return len(self.elements) - 1

    def add_layer(self, config: LayerConfig) -> 'DepthProfileBuilder':
        """Add a layer configuration."""
        self.layers.append(config)
        return self

    def build(self, normalize: bool = True) -> DepthProfile:
        """
        Build the depth profile from layer configurations.

        Args:
            normalize: If True, normalize concentrations to sum to 1

        Returns:
            DepthProfile object
        """
        n_depth = len(self.depth)
        n_elements = len(self.elements)
        concentrations = np.zeros((n_depth, n_elements))

        # Process each layer
        for layer in self.layers:
            idx = layer.element_index
            profile = self._calculate_distribution(layer)
            concentrations[:, idx] += profile

        # Create profile object
        profile = DepthProfile(
            depth=self.depth.copy(),
            elements=self.elements.copy(),
            concentrations=concentrations,
            step_size=self.step_size
        )

        if normalize:
            profile.normalize()

        return profile

    def _calculate_distribution(self, layer: LayerConfig) -> np.ndarray:
        """Calculate concentration distribution for a layer."""
        df = DistributionFunction

        if layer.distribution == 'substrate':
            return layer.concentration * df.error_function(
                self.depth, layer.thickness, layer.abruptness, rising=True
            )

        elif layer.distribution == 'layer' or layer.distribution == 'error_func':
            return layer.concentration * df.error_function(
                self.depth, layer.position, layer.abruptness, rising=False
            )

        elif layer.distribution == 'gaussian':
            return df.gaussian(
                self.depth, layer.position, layer.abruptness, layer.concentration
            )

        elif layer.distribution == 'exponential':
            return df.exponential(
                self.depth, layer.position, layer.thickness, layer.concentration
            )

        elif layer.distribution == 'rectangle':
            return df.rectangle(
                self.depth, layer.position, layer.thickness, layer.concentration
            )

        elif layer.distribution == 'triangle':
            return df.triangle(
                self.depth, layer.position, layer.thickness, layer.concentration
            )

        elif layer.distribution == 'homogeneous':
            return np.full_like(self.depth, layer.concentration)

        else:
            raise ValueError(f"Unknown distribution: {layer.distribution}")


def create_five_layer_profile(
    depth_end: float = 100.0,
    step_size: float = 1.0
) -> DepthProfile:
    """
    Create a five-layer test profile (Si/SiO2/TiO2/Al2O3/C).

    This represents a typical test structure:
    - Surface contamination (C): 0-5 Å
    - Al2O3 layer: 5-15 Å
    - TiO2 layer: 15-35 Å
    - SiO2 layer: 35-60 Å
    - Si substrate: 60+ Å

    Args:
        depth_end: Maximum depth (Å)
        step_size: Depth step (Å)

    Returns:
        DepthProfile for the five-layer structure
    """
    builder = DepthProfileBuilder(0, depth_end, step_size)

    # Add elements (order matters for color mapping consistency)
    idx_c = builder.add_element(ElementConfig('C', '1s', 'C', gaussian_fwhm=1.2))
    idx_o = builder.add_element(ElementConfig('O', '1s', 'SiO2', gaussian_fwhm=1.5))
    idx_al = builder.add_element(ElementConfig('Al', '2p', 'Al2O3', gaussian_fwhm=0.8))
    idx_ti = builder.add_element(ElementConfig('Ti', '2p', 'TiO2', gaussian_fwhm=1.0))
    idx_si = builder.add_element(ElementConfig('Si', '2p', 'Si', gaussian_fwhm=0.9))

    # Add layers (from surface to substrate)

    # 1. Surface carbon contamination (exponential decay from surface)
    builder.add_layer(LayerConfig(
        element_index=idx_c,
        distribution='exponential',
        concentration=0.15,
        position=0,
        thickness=5.0,  # decay length
    ))

    # 2. Al2O3 layer (Gaussian centered at 10 Å)
    builder.add_layer(LayerConfig(
        element_index=idx_al,
        distribution='gaussian',
        concentration=0.25,
        position=10.0,
        abruptness=8.0,  # FWHM
    ))

    # 3. O in Al2O3 region (same profile as Al, scaled for stoichiometry)
    builder.add_layer(LayerConfig(
        element_index=idx_o,
        distribution='gaussian',
        concentration=0.375,  # Al2O3 stoichiometry: 3O / 2Al
        position=10.0,
        abruptness=8.0,
    ))

    # 4. TiO2 layer (Gaussian centered at 25 Å)
    builder.add_layer(LayerConfig(
        element_index=idx_ti,
        distribution='gaussian',
        concentration=0.20,
        position=25.0,
        abruptness=15.0,
    ))

    # 5. O in TiO2 region
    builder.add_layer(LayerConfig(
        element_index=idx_o,
        distribution='gaussian',
        concentration=0.40,  # TiO2 stoichiometry
        position=25.0,
        abruptness=15.0,
    ))

    # 6. SiO2 layer (Gaussian centered at 47 Å)
    builder.add_layer(LayerConfig(
        element_index=idx_si,
        distribution='gaussian',
        concentration=0.20,  # Si in SiO2
        position=47.0,
        abruptness=20.0,
    ))

    # 7. O in SiO2 region
    builder.add_layer(LayerConfig(
        element_index=idx_o,
        distribution='gaussian',
        concentration=0.40,  # SiO2 stoichiometry
        position=47.0,
        abruptness=20.0,
    ))

    # 8. Si substrate (error function rising)
    builder.add_layer(LayerConfig(
        element_index=idx_si,
        distribution='substrate',
        concentration=1.0,
        thickness=60.0,  # Interface position
        abruptness=5.0,   # Interface sharpness
    ))

    return builder.build(normalize=True)


def create_simple_sio2_si_profile(
    sio2_thickness: float = 30.0,
    interface_width: float = 5.0,
    depth_end: float = 100.0,
    step_size: float = 1.0
) -> DepthProfile:
    """
    Create a simple SiO2/Si bilayer profile.

    Args:
        sio2_thickness: SiO2 layer thickness (Å)
        interface_width: Interface width FWHM (Å)
        depth_end: Maximum depth (Å)
        step_size: Depth step (Å)

    Returns:
        DepthProfile for SiO2/Si structure
    """
    builder = DepthProfileBuilder(0, depth_end, step_size)

    idx_o = builder.add_element(ElementConfig('O', '1s', 'SiO2'))
    idx_si = builder.add_element(ElementConfig('Si', '2p', 'Si'))

    # SiO2 layer (O component)
    builder.add_layer(LayerConfig(
        element_index=idx_o,
        distribution='layer',
        concentration=0.667,  # SiO2: 2/3 oxygen
        position=sio2_thickness,
        abruptness=interface_width,
    ))

    # SiO2 layer (Si component)
    builder.add_layer(LayerConfig(
        element_index=idx_si,
        distribution='layer',
        concentration=0.333,  # SiO2: 1/3 silicon
        position=sio2_thickness,
        abruptness=interface_width,
    ))

    # Si substrate
    builder.add_layer(LayerConfig(
        element_index=idx_si,
        distribution='substrate',
        concentration=1.0,
        thickness=sio2_thickness,
        abruptness=interface_width,
    ))

    return builder.build(normalize=True)


def create_sio2_si_with_chemical_states(
    sio2_thickness: float = 30.0,
    interface_width: float = 5.0,
    depth_end: float = 100.0,
    step_size: float = 1.0
) -> DepthProfile:
    """
    Create SiO2/Si profile with separate chemical states.

    Elements:
        0: O²⁻ (oxygen in SiO2, oxidation_state=-2)
        1: Si⁴⁺ (silicon in SiO2, oxidation_state=+4)
        2: Si⁰ (bulk silicon substrate, oxidation_state=0)

    Constraints:
        - Charge conservation: 2×[Si⁴⁺] = [O²⁻]  (from -2×[O] + 4×[Si-O] = 0)
        - Concentration sum = 1 at each depth

    Args:
        sio2_thickness: SiO2 layer thickness (Å)
        interface_width: Interface width FWHM (Å)
        depth_end: Maximum depth (Å)
        step_size: Depth step (Å)

    Returns:
        DepthProfile with 3 chemical states
    """
    builder = DepthProfileBuilder(0, depth_end, step_size)

    # Element 0: O²⁻ in SiO2
    idx_o = builder.add_element(ElementConfig(
        'O', '1s', 'SiO2',
        oxidation_state=-2,
        stoichiometry=2,  # 2 O per SiO2
    ))

    # Element 1: Si⁴⁺ in SiO2
    idx_si_ox = builder.add_element(ElementConfig(
        'Si', '2p', 'SiO2',
        oxidation_state=+4,
        stoichiometry=1,  # 1 Si per SiO2
    ))

    # Element 2: Si⁰ bulk substrate
    idx_si_bulk = builder.add_element(ElementConfig(
        'Si', '2p', 'Si',
        oxidation_state=0,
        stoichiometry=None,  # Not part of oxide
    ))

    # In SiO2 layer: [O] = 2/3, [Si-O] = 1/3, [Si-B] ≈ 0
    # Charge balance: -2 × (2/3) + 4 × (1/3) = -4/3 + 4/3 = 0 ✓

    # O²⁻ in SiO2 layer (decreases at interface)
    builder.add_layer(LayerConfig(
        element_index=idx_o,
        distribution='layer',
        concentration=0.667,  # 2/3 for SiO2
        position=sio2_thickness,
        abruptness=interface_width,
    ))

    # Si⁴⁺ in SiO2 layer (decreases at interface)
    builder.add_layer(LayerConfig(
        element_index=idx_si_ox,
        distribution='layer',
        concentration=0.333,  # 1/3 for SiO2
        position=sio2_thickness,
        abruptness=interface_width,
    ))

    # Si⁰ substrate (rises at interface)
    builder.add_layer(LayerConfig(
        element_index=idx_si_bulk,
        distribution='substrate',
        concentration=1.0,
        thickness=sio2_thickness,
        abruptness=interface_width,
    ))

    return builder.build(normalize=True)


# =============================================================================
# Charge Conservation / Stoichiometry Utilities
# =============================================================================

# Common oxidation states for elements
COMMON_OXIDATION_STATES: Dict[str, float] = {
    # Alkali metals (always +1)
    'Li': +1, 'Na': +1, 'K': +1,
    # Alkaline earth metals (always +2)
    'Be': +2, 'Mg': +2, 'Ca': +2, 'Sr': +2, 'Ba': +2,
    # Transition metals (common states)
    'Ti': +4, 'V': +5, 'Cr': +3, 'Mn': +2, 'Fe': +3,
    'Co': +2, 'Ni': +2, 'Cu': +2, 'Zn': +2,
    'Zr': +4, 'Nb': +5, 'Mo': +6, 'Ru': +4, 'Pd': +2,
    'Ag': +1, 'Cd': +2, 'Hf': +4, 'Ta': +5, 'W': +6,
    'Pt': +4, 'Au': +3,
    # Main group elements
    'Al': +3, 'Si': +4, 'Sn': +4, 'Pb': +2,
    'B': +3, 'Ga': +3, 'In': +3,
    # Non-metals (common negative states)
    'O': -2, 'S': -2, 'Se': -2, 'Te': -2,
    'N': -3, 'P': +5, 'As': +3,
    'F': -1, 'Cl': -1, 'Br': -1, 'I': -1,
    # Carbon can be variable
    'C': +4,
    'H': +1,
}


def parse_compound_formula(formula: str) -> Dict[str, float]:
    """
    Parse chemical formula to extract element stoichiometry.

    Examples:
        'TiO2' -> {'Ti': 1, 'O': 2}
        'SiO2' -> {'Si': 1, 'O': 2}
        'Al2O3' -> {'Al': 2, 'O': 3}
        'Ta2O5' -> {'Ta': 2, 'O': 5}

    Args:
        formula: Chemical formula string

    Returns:
        Dictionary mapping element symbols to their count
    """
    import re

    # Pattern: element symbol (1-2 letters) optionally followed by a number
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)

    result = {}
    for symbol, count_str in matches:
        if not symbol:
            continue
        count = float(count_str) if count_str else 1.0
        result[symbol] = result.get(symbol, 0) + count

    return result


def infer_oxidation_states(
    elements: List[ElementConfig],
    compound_formula: Optional[str] = None,
) -> List[ElementConfig]:
    """
    Infer oxidation states for elements based on common values or compound.

    Args:
        elements: List of element configurations
        compound_formula: Optional compound formula to verify charge balance

    Returns:
        List of ElementConfig with oxidation_state filled in
    """
    for elem in elements:
        if elem.oxidation_state is None:
            elem.oxidation_state = COMMON_OXIDATION_STATES.get(elem.symbol, 0)

    return elements


def create_charge_constraint_matrix(
    elements: List[ElementConfig],
    n_depth: int,
) -> Optional[np.ndarray]:
    """
    Create constraint matrix for charge conservation at each depth.

    The constraint enforces: Σ_e (oxidation_state_e × concentration_e) = 0 at each depth

    Args:
        elements: List of element configurations with oxidation_state set
        n_depth: Number of depth points

    Returns:
        Constraint matrix A of shape (n_depth, n_depth * n_elements)
        such that A @ x = 0 enforces charge neutrality
        Returns None if no oxidation states are defined
    """
    n_elements = len(elements)

    # Check if oxidation states are defined
    oxidation_states = [e.oxidation_state for e in elements]
    if all(o is None or o == 0 for o in oxidation_states):
        return None

    # Replace None with 0
    oxidation_states = [o if o is not None else 0 for o in oxidation_states]

    # Build constraint matrix
    # For flattened D (d*n_elem + e), constraint at depth d is:
    # sum_e(oxidation_state[e] * D[d, e]) = 0
    A = np.zeros((n_depth, n_depth * n_elements))
    for d in range(n_depth):
        for e in range(n_elements):
            A[d, d * n_elements + e] = oxidation_states[e]

    return A


def create_stoichiometry_constraint_matrix(
    elements: List[ElementConfig],
    n_depth: int,
    stoichiometry_groups: Optional[List[List[int]]] = None,
) -> Optional[np.ndarray]:
    """
    Create constraint matrix for stoichiometry (ratio constraints).

    For elements with stoichiometry defined, enforces ratio constraints.
    Example: For TiO2 with Ti(stoich=1) and O(stoich=2), enforces D_O = 2 * D_Ti

    Args:
        elements: List of element configurations
        n_depth: Number of depth points
        stoichiometry_groups: List of element index groups that should maintain ratios
                             If None, all elements with stoichiometry form one group

    Returns:
        Constraint matrix A such that A @ x = 0
        Returns None if no stoichiometry is defined
    """
    n_elements = len(elements)
    stoich = [e.stoichiometry for e in elements]

    # Check if stoichiometry is defined
    defined_idx = [i for i, s in enumerate(stoich) if s is not None and s > 0]
    if len(defined_idx) < 2:
        return None

    if stoichiometry_groups is None:
        stoichiometry_groups = [defined_idx]

    constraints = []

    for group in stoichiometry_groups:
        if len(group) < 2:
            continue

        # Use first element as reference
        ref_idx = group[0]
        ref_stoich = stoich[ref_idx]
        if ref_stoich is None or ref_stoich == 0:
            continue

        for other_idx in group[1:]:
            other_stoich = stoich[other_idx]
            if other_stoich is None or other_stoich == 0:
                continue

            # Constraint: D[other]/stoich[other] = D[ref]/stoich[ref]
            # => stoich[ref] * D[other] - stoich[other] * D[ref] = 0
            for d in range(n_depth):
                row = np.zeros(n_depth * n_elements)
                row[d * n_elements + other_idx] = ref_stoich
                row[d * n_elements + ref_idx] = -other_stoich
                constraints.append(row)

    if not constraints:
        return None

    return np.array(constraints)


def create_concentration_sum_constraint_matrix(
    n_depth: int,
    n_elements: int,
) -> np.ndarray:
    """
    Create constraint matrix for concentration sum = 1 at each depth.

    Args:
        n_depth: Number of depth points
        n_elements: Number of elements

    Returns:
        Constraint matrix A of shape (n_depth, n_depth * n_elements)
        such that A @ x = 1 (vector of ones)
    """
    A = np.zeros((n_depth, n_depth * n_elements))
    for d in range(n_depth):
        A[d, d * n_elements:(d + 1) * n_elements] = 1

    return A


def create_multilayer_profile(
    layer_thicknesses: List[float] = [5.0, 10.0, 20.0, 30.0],
    interface_widths: List[float] = [2.0, 3.0, 4.0, 5.0],
    depth_end: float = 100.0,
    step_size: float = 1.0,
) -> DepthProfile:
    """
    Create C/Al₂O₃/TiO₂/SiO₂/Si multilayer profile with chemical states.

    Structure (from surface to bulk):
        - C (contamination): 0-5 Å
        - Al₂O₃ layer: 5-15 Å
        - TiO₂ layer: 15-35 Å
        - SiO₂ layer: 35-65 Å
        - Si substrate: 65+ Å

    Chemical states (elements):
        0: C (carbon contamination, free)
        1: Al³⁺ (aluminum in Al₂O₃, free)
        2: Ti⁴⁺ (titanium in TiO₂, free)
        3: Si⁴⁺ (silicon in SiO₂, free)
        4: O²⁻ (oxygen in all oxides, linked)
        5: Si⁰ (bulk silicon substrate)

    Constraints:
        - Charge balance: 3×[Al] + 4×[Ti] + 4×[Si-O] = 2×[O]
        - Concentration sum = 1 at each depth
        - O²⁻ linked via stoichiometry to cations

    Args:
        layer_thicknesses: [C, Al2O3, TiO2, SiO2] thicknesses in Å
        interface_widths: Interface widths (FWHM) in Å for each boundary
        depth_end: Maximum depth (Å)
        step_size: Depth step (Å)

    Returns:
        DepthProfile with 6 chemical states
    """
    builder = DepthProfileBuilder(0, depth_end, step_size)

    # Calculate interface positions from layer thicknesses
    t_C, t_Al2O3, t_TiO2, t_SiO2 = layer_thicknesses
    w_C, w_Al2O3, w_TiO2, w_SiO2 = interface_widths

    pos_Al2O3_start = t_C                           # C/Al2O3 interface
    pos_TiO2_start = pos_Al2O3_start + t_Al2O3      # Al2O3/TiO2 interface
    pos_SiO2_start = pos_TiO2_start + t_TiO2        # TiO2/SiO2 interface
    pos_Si_start = pos_SiO2_start + t_SiO2          # SiO2/Si interface

    # Element 0: C (carbon contamination)
    idx_c = builder.add_element(ElementConfig(
        'C', '1s', 'C',
        oxidation_state=0,       # Neutral carbon
        stoichiometry=None,
        gaussian_fwhm=1.2,
        lorentzian_fwhm=0.3,
    ))

    # Element 1: Al³⁺ in Al₂O₃
    idx_al = builder.add_element(ElementConfig(
        'Al', '2p', 'Al2O3',
        oxidation_state=+3,
        stoichiometry=2,         # 2 Al per Al₂O₃
        gaussian_fwhm=0.8,
        lorentzian_fwhm=0.3,
    ))

    # Element 2: Ti⁴⁺ in TiO₂
    idx_ti = builder.add_element(ElementConfig(
        'Ti', '2p', 'TiO2',
        oxidation_state=+4,
        stoichiometry=1,         # 1 Ti per TiO₂
        gaussian_fwhm=1.0,
        lorentzian_fwhm=0.4,
    ))

    # Element 3: Si⁴⁺ in SiO₂
    idx_si_ox = builder.add_element(ElementConfig(
        'Si', '2p', 'SiO2',
        oxidation_state=+4,
        stoichiometry=1,         # 1 Si per SiO₂
        gaussian_fwhm=0.9,
        lorentzian_fwhm=0.3,
    ))

    # Element 4: O²⁻ (combined from all oxide layers)
    idx_o = builder.add_element(ElementConfig(
        'O', '1s', 'SiO2',       # Use SiO2 as reference for IMFP
        oxidation_state=-2,
        stoichiometry=None,      # Linked element, computed from charge balance
        gaussian_fwhm=1.5,
        lorentzian_fwhm=0.4,
    ))

    # Element 5: Si⁰ (bulk silicon)
    idx_si_bulk = builder.add_element(ElementConfig(
        'Si', '2p', 'Si',
        oxidation_state=0,
        stoichiometry=None,
        gaussian_fwhm=0.9,
        lorentzian_fwhm=0.3,
    ))

    depth = builder.depth

    # Calculate profiles for each layer using error functions
    sigma_C = w_C / (2 * np.sqrt(np.log(2)))
    sigma_Al2O3 = w_Al2O3 / (2 * np.sqrt(np.log(2)))
    sigma_TiO2 = w_TiO2 / (2 * np.sqrt(np.log(2)))
    sigma_SiO2 = w_SiO2 / (2 * np.sqrt(np.log(2)))

    # Define layer profiles using difference of error functions
    def erf_layer(z, z_start, z_end, sigma_start, sigma_end):
        """Create a layer profile bounded by two error functions."""
        # Rising edge at z_start (layer begins)
        rising = 0.5 * (1 + erf((z - z_start) / sigma_start))
        # Falling edge at z_end (layer ends)
        falling = 0.5 * (1 + erf((z - z_end) / sigma_end))
        return rising - falling

    # Carbon contamination layer (surface to C/Al2O3 interface)
    # Exponential decay from surface
    D_C = 0.3 * np.exp(-depth / 3.0) * (1 - 0.5 * (1 + erf((depth - pos_Al2O3_start) / sigma_C)))

    # Al2O3 layer (between C/Al2O3 and Al2O3/TiO2)
    # Al: 2/(2+3) = 0.4, O: 3/(2+3) = 0.6 in Al2O3
    layer_Al2O3 = erf_layer(depth, pos_Al2O3_start, pos_TiO2_start, sigma_Al2O3, sigma_TiO2)
    D_Al = 0.4 * layer_Al2O3

    # TiO2 layer (between Al2O3/TiO2 and TiO2/SiO2)
    # Ti: 1/(1+2) = 0.333, O: 2/(1+2) = 0.667 in TiO2
    layer_TiO2 = erf_layer(depth, pos_TiO2_start, pos_SiO2_start, sigma_TiO2, sigma_SiO2)
    D_Ti = 0.333 * layer_TiO2

    # SiO2 layer (between TiO2/SiO2 and SiO2/Si)
    # Si: 1/(1+2) = 0.333, O: 2/(1+2) = 0.667 in SiO2
    layer_SiO2 = erf_layer(depth, pos_SiO2_start, pos_Si_start, sigma_SiO2, sigma_SiO2)
    D_Si_ox = 0.333 * layer_SiO2

    # Si substrate (rises at SiO2/Si interface)
    D_Si_bulk = 0.5 * (1 + erf((depth - pos_Si_start) / sigma_SiO2))

    # Oxygen from all oxide layers (charge-balanced)
    # O in Al2O3: 3/2 × Al (to give Al₂O₃ stoichiometry)
    # O in TiO2: 2 × Ti (to give TiO₂ stoichiometry)
    # O in SiO2: 2 × Si-ox (to give SiO₂ stoichiometry)
    D_O = (3/2) * D_Al + 2 * D_Ti + 2 * D_Si_ox

    # Build concentration matrix
    concentrations = np.zeros((len(depth), 6))
    concentrations[:, idx_c] = D_C
    concentrations[:, idx_al] = D_Al
    concentrations[:, idx_ti] = D_Ti
    concentrations[:, idx_si_ox] = D_Si_ox
    concentrations[:, idx_o] = D_O
    concentrations[:, idx_si_bulk] = D_Si_bulk

    # Clip to valid range
    concentrations = np.clip(concentrations, 1e-6, 1.0)

    # Normalize to sum = 1 at each depth
    total = concentrations.sum(axis=1, keepdims=True)
    concentrations = concentrations / np.maximum(total, 1e-10)

    # Create profile
    profile = DepthProfile(
        depth=depth,
        elements=builder.elements,
        concentrations=concentrations,
        step_size=step_size
    )

    return profile


@dataclass
class MultiLayerFilmConfig:
    """Configuration for multi-layer film structure optimization.

    This extends FilmStructureConfig for systems with multiple interfaces:
    C/Al₂O₃/TiO₂/SiO₂/Si

    Element Classification:
        free_indices: Freely optimized cations [Al³⁺, Ti⁴⁺, Si⁴⁺, C]
        linked_indices: Charge-balanced anions [O²⁻]
        substrate_indices: Substrate elements [Si⁰]

    Interface Parameters:
        - thickness[i]: Position of interface i
        - abruptness[i]: Width of interface i

    Constraints:
        - Charge neutrality: Σ(charge × conc) = 0 at each depth
        - Concentration sum = 1 at each depth
        - Stoichiometry within each oxide layer
    """
    # Number of layers (excluding substrate)
    n_layers: int = 4  # C, Al2O3, TiO2, SiO2

    # Interface positions (thickness of each layer from surface)
    layer_thicknesses: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0, 30.0])

    # Interface widths (FWHM for error functions)
    interface_widths: List[float] = field(default_factory=lambda: [2.0, 3.0, 4.0, 5.0])

    # Optimization flags
    optimize_thicknesses: bool = True
    optimize_abruptnesses: bool = True

    # Bounds for parameters
    thickness_bounds: Tuple[float, float] = (1.0, 100.0)
    abruptness_bounds: Tuple[float, float] = (0.5, 20.0)

    # Element classification (indices)
    free_indices: Optional[List[int]] = None        # [C, Al, Ti, Si-ox]
    linked_indices: Optional[List[int]] = None      # [O]
    substrate_indices: Optional[List[int]] = None   # [Si-bulk]

    # Constraint settings
    use_charge_constraint: bool = True
    use_sum_constraint: bool = True

    # Stoichiometry coefficients for oxygen calculation
    # O = sum(coeff[i] * free[i]) for each cation
    oxygen_stoich_coeffs: Optional[List[float]] = None  # [0, 1.5, 2.0, 2.0] for C, Al, Ti, Si


def verify_multilayer_charge_balance(profile: DepthProfile) -> Dict:
    """
    Verify charge balance in a multilayer profile.

    For the C/Al₂O₃/TiO₂/SiO₂/Si structure:
    - Total positive charge: 3×[Al] + 4×[Ti] + 4×[Si-ox]
    - Total negative charge: 2×[O]
    - Charge balance error: positive - negative

    Args:
        profile: DepthProfile with chemical states

    Returns:
        Dictionary with charge balance analysis
    """
    # Get oxidation states
    charges = []
    for elem in profile.elements:
        ox = getattr(elem, 'oxidation_state', None)
        charges.append(ox if ox is not None else 0)
    charges = np.array(charges)

    # Calculate total charge at each depth
    D = profile.concentrations
    total_charge = (D * charges).sum(axis=1)

    # Analyze
    max_error = np.max(np.abs(total_charge))
    mean_error = np.mean(np.abs(total_charge))
    rmse = np.sqrt(np.mean(total_charge ** 2))

    results = {
        'total_charge': total_charge,
        'max_error': max_error,
        'mean_error': mean_error,
        'rmse': rmse,
        'is_balanced': max_error < 0.01,
        'charges': charges,
    }

    return results


def verify_multilayer_stoichiometry(profile: DepthProfile) -> Dict:
    """
    Verify stoichiometry in a multilayer profile.

    Checks that O²⁻ concentration satisfies:
        [O] = (3/2)×[Al] + 2×[Ti] + 2×[Si-ox]

    This ensures:
        - Al₂O₃ stoichiometry: 2 Al + 3 O
        - TiO₂ stoichiometry: 1 Ti + 2 O
        - SiO₂ stoichiometry: 1 Si + 2 O

    Args:
        profile: DepthProfile with chemical states

    Returns:
        Dictionary with stoichiometry analysis
    """
    D = profile.concentrations
    labels = profile.element_labels

    # Find element indices by label
    idx_map = {label: i for i, label in enumerate(labels)}

    # Get cation concentrations
    D_Al = D[:, idx_map.get('Al2p', -1)] if 'Al2p' in idx_map else np.zeros(len(profile.depth))
    D_Ti = D[:, idx_map.get('Ti2p', -1)] if 'Ti2p' in idx_map else np.zeros(len(profile.depth))

    # Si can appear twice (oxide and bulk) - need to identify by compound
    D_Si_ox = np.zeros(len(profile.depth))
    for i, elem in enumerate(profile.elements):
        if elem.symbol == 'Si' and elem.compound == 'SiO2':
            D_Si_ox = D[:, i]
            break

    # Get O concentration
    D_O = D[:, idx_map.get('O1s', -1)] if 'O1s' in idx_map else np.zeros(len(profile.depth))

    # Expected O from stoichiometry
    D_O_expected = (3/2) * D_Al + 2 * D_Ti + 2 * D_Si_ox

    # Error analysis
    error = D_O - D_O_expected
    max_error = np.max(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    results = {
        'D_O_actual': D_O,
        'D_O_expected': D_O_expected,
        'error': error,
        'max_error': max_error,
        'rmse': rmse,
        'is_satisfied': max_error < 0.01,
    }

    return results


# =============================================================================
# Profile Comparison Metrics
# =============================================================================

def wasserstein_distance_1d(
    depth: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """
    Compute 1D Wasserstein distance (W1) between two depth distributions.

    W1 = ∫|F_p1(z) - F_p2(z)| dz

    where F is the cumulative distribution function.

    If the ground-truth distribution (p1) has negligible total mass
    (integral < 1e-6), returns NaN. This avoids meaningless W1 values
    for checker-pattern elements like C/B where GT ≡ 0 at certain pixels.

    Args:
        depth: Depth array (Å)
        p1: First distribution (ground truth concentration profile)
        p2: Second distribution (reconstructed concentration profile)

    Returns:
        W1 distance in Å (same unit as depth). NaN if p1 integral ≈ 0.
    """
    # Skip if ground truth is essentially zero (checker pattern C=0 / B=0 pixels)
    p1_integral = trapezoid(np.abs(p1), depth)
    if p1_integral < 1e-6:
        return np.nan

    # Normalize to probability distributions
    p1_norm = p1 / np.maximum(trapezoid(p1, depth), 1e-10)
    p2_norm = p2 / np.maximum(trapezoid(p2, depth), 1e-10)

    # Compute CDFs
    cdf1 = np.cumsum(p1_norm) * np.gradient(depth)
    cdf2 = np.cumsum(p2_norm) * np.gradient(depth)

    # Normalize CDFs to [0, 1]
    cdf1 = cdf1 / np.maximum(cdf1[-1], 1e-10)
    cdf2 = cdf2 / np.maximum(cdf2[-1], 1e-10)

    # W1 = integral of |CDF1 - CDF2|
    w1 = trapezoid(np.abs(cdf1 - cdf2), depth)

    return w1


def centroid(depth: np.ndarray, profile: np.ndarray) -> float:
    """
    Compute centroid (center of mass) of a depth distribution.

    centroid = ∫z·c(z)dz / ∫c(z)dz

    Args:
        depth: Depth array (Å)
        profile: Concentration profile

    Returns:
        Centroid position in Å
    """
    total = trapezoid(profile, depth)
    if total < 1e-10:
        return 0.0
    return trapezoid(depth * profile, depth) / total


def profile_moments(
    depth: np.ndarray,
    profile: np.ndarray,
    max_order: int = 3,
) -> Dict[str, float]:
    """
    Compute moments of a depth distribution.

    Args:
        depth: Depth array (Å)
        profile: Concentration profile
        max_order: Maximum moment order (1=centroid, 2=variance, 3=skewness)

    Returns:
        Dictionary with moments:
            - m1: Centroid (Å)
            - m2: Variance (Å²)
            - sigma: Standard deviation (Å)
            - m3: Third central moment
            - skewness: Normalized skewness
    """
    # Normalize
    total = trapezoid(profile, depth)
    if total < 1e-10:
        return {'m1': 0.0, 'm2': 0.0, 'sigma': 0.0, 'm3': 0.0, 'skewness': 0.0}

    p_norm = profile / total

    # First moment (centroid)
    m1 = trapezoid(depth * p_norm, depth)

    # Second central moment (variance)
    m2 = trapezoid((depth - m1)**2 * p_norm, depth)
    sigma = np.sqrt(max(m2, 0))

    # Third central moment
    m3 = trapezoid((depth - m1)**3 * p_norm, depth) if max_order >= 3 else 0.0
    skewness = m3 / (sigma**3) if sigma > 1e-10 else 0.0

    return {
        'm1': m1,
        'm2': m2,
        'sigma': sigma,
        'm3': m3,
        'skewness': skewness,
    }


@dataclass
class ProfileComparisonResult:
    """Result of comparing two depth profiles."""
    # Per-element metrics
    w1_per_element: Dict[str, float]          # W1 distance per element (Å)
    centroid_shift: Dict[str, float]          # Centroid difference per element (Å)
    rmse_per_element: Dict[str, float]        # RMSE per element

    # Aggregate metrics
    w1_total: float                           # Sum of W1 across elements (Å)
    w1_mean: float                            # Mean W1 across elements (Å)
    centroid_shift_mean: float                # Mean absolute centroid shift (Å)
    rmse_total: float                         # Total RMSE

    # Optional: interface positions if available
    interface_errors: Optional[np.ndarray] = None


def compare_profiles(
    profile1: 'DepthProfile',
    profile2: 'DepthProfile',
    element_indices: Optional[List[int]] = None,
) -> ProfileComparisonResult:
    """
    Compare two depth profiles using multiple distance metrics.

    Computes:
        - W1 (Wasserstein-1) distance per element
        - Centroid shift per element
        - RMSE per element

    Args:
        profile1: First (reference/true) profile
        profile2: Second (reconstructed) profile
        element_indices: Optional list of element indices to compare.
                        If None, compares all elements.

    Returns:
        ProfileComparisonResult with all metrics

    Example:
        >>> true_profile = create_multilayer_profile()
        >>> recon_profile = solver.solve(I_obs).D_opt
        >>> result = compare_profiles(true_profile, recon_profile)
        >>> print(f"W1 mean: {result.w1_mean:.2f} Å")
    """
    depth = profile1.depth
    D1 = profile1.concentrations

    # Handle both DepthProfile and raw array
    if hasattr(profile2, 'concentrations'):
        D2 = profile2.concentrations
        labels = profile2.element_labels
    else:
        D2 = profile2
        labels = profile1.element_labels

    if element_indices is None:
        element_indices = list(range(D1.shape[1]))

    # Per-element metrics
    w1_per_element = {}
    centroid_shift = {}
    rmse_per_element = {}

    for idx in element_indices:
        label = labels[idx] if idx < len(labels) else f"elem_{idx}"
        p1 = D1[:, idx]
        p2 = D2[:, idx]

        # W1 distance
        w1_per_element[label] = wasserstein_distance_1d(depth, p1, p2)

        # Centroid shift
        c1 = centroid(depth, p1)
        c2 = centroid(depth, p2)
        centroid_shift[label] = c2 - c1

        # RMSE
        rmse_per_element[label] = np.sqrt(np.mean((p1 - p2)**2))

    # Aggregate metrics
    w1_values = list(w1_per_element.values())
    w1_total = sum(w1_values)
    w1_mean = np.mean(w1_values)

    centroid_values = list(centroid_shift.values())
    centroid_shift_mean = np.mean(np.abs(centroid_values))

    rmse_values = list(rmse_per_element.values())
    rmse_total = np.sqrt(np.mean(np.array(rmse_values)**2))

    return ProfileComparisonResult(
        w1_per_element=w1_per_element,
        centroid_shift=centroid_shift,
        rmse_per_element=rmse_per_element,
        w1_total=w1_total,
        w1_mean=w1_mean,
        centroid_shift_mean=centroid_shift_mean,
        rmse_total=rmse_total,
    )


def compare_with_array(
    depth: np.ndarray,
    D_true: np.ndarray,
    D_recon: np.ndarray,
    labels: Optional[List[str]] = None,
) -> ProfileComparisonResult:
    """
    Compare two concentration arrays directly.

    Convenience function when you have raw arrays instead of DepthProfile objects.

    Args:
        depth: Depth array (Å)
        D_true: True concentration array [n_depth, n_elements]
        D_recon: Reconstructed concentration array [n_depth, n_elements]
        labels: Optional element labels

    Returns:
        ProfileComparisonResult with all metrics
    """
    n_elements = D_true.shape[1]
    if labels is None:
        labels = [f"elem_{i}" for i in range(n_elements)]

    w1_per_element = {}
    centroid_shift = {}
    rmse_per_element = {}

    for i in range(n_elements):
        label = labels[i]
        p1 = D_true[:, i]
        p2 = D_recon[:, i]

        w1_per_element[label] = wasserstein_distance_1d(depth, p1, p2)

        c1 = centroid(depth, p1)
        c2 = centroid(depth, p2)
        centroid_shift[label] = c2 - c1

        rmse_per_element[label] = np.sqrt(np.mean((p1 - p2)**2))

    w1_values = list(w1_per_element.values())
    w1_total = sum(w1_values)
    w1_mean = np.mean(w1_values)

    centroid_values = list(centroid_shift.values())
    centroid_shift_mean = np.mean(np.abs(centroid_values))

    rmse_values = list(rmse_per_element.values())
    rmse_total = np.sqrt(np.mean(np.array(rmse_values)**2))

    return ProfileComparisonResult(
        w1_per_element=w1_per_element,
        centroid_shift=centroid_shift,
        rmse_per_element=rmse_per_element,
        w1_total=w1_total,
        w1_mean=w1_mean,
        centroid_shift_mean=centroid_shift_mean,
        rmse_total=rmse_total,
    )


# =============================================================================
# Model-Free Interface Detection
# =============================================================================


def detect_interfaces_gradient(
    depth: np.ndarray,
    profile: np.ndarray,
    threshold: float = 0.1,
    min_separation: float = 3.0,
) -> np.ndarray:
    """
    Detect interface positions using gradient maximum method.

    Finds positions where |∂c/∂z| is locally maximum, indicating
    rapid concentration changes at interfaces.

    Args:
        depth: Depth array (Å)
        profile: Concentration profile [n_depth] or [n_depth, n_elements]
        threshold: Minimum gradient magnitude (relative to max) to consider
        min_separation: Minimum separation between interfaces (Å)

    Returns:
        Array of interface positions (Å)
    """
    if profile.ndim == 2:
        # Sum absolute gradients across all elements
        grad_sum = np.zeros_like(depth)
        for i in range(profile.shape[1]):
            grad = np.abs(np.gradient(profile[:, i], depth))
            grad_sum += grad
        grad = grad_sum
    else:
        grad = np.abs(np.gradient(profile, depth))

    # Normalize gradient
    grad_max = np.max(grad)
    if grad_max < 1e-10:
        return np.array([])
    grad_norm = grad / grad_max

    # Find local maxima above threshold
    interfaces = []
    for i in range(1, len(depth) - 1):
        if grad_norm[i] > threshold:
            if grad_norm[i] > grad_norm[i-1] and grad_norm[i] > grad_norm[i+1]:
                # Check minimum separation
                if len(interfaces) == 0 or depth[i] - interfaces[-1] >= min_separation:
                    interfaces.append(depth[i])

    return np.array(interfaces)


def detect_interfaces_50percent(
    depth: np.ndarray,
    profile: np.ndarray,
    element_index: int = 0,
    direction: str = 'increasing',
) -> np.ndarray:
    """
    Detect interface positions using 50% concentration point.

    Finds positions where concentration crosses 50% of its range,
    which corresponds to the interface center for erf profiles.

    Args:
        depth: Depth array (Å)
        profile: Concentration profile [n_depth, n_elements]
        element_index: Which element to analyze
        direction: 'increasing', 'decreasing', or 'both'

    Returns:
        Array of interface positions (Å)
    """
    if profile.ndim == 1:
        conc = profile
    else:
        conc = profile[:, element_index]

    c_min, c_max = np.min(conc), np.max(conc)
    c_50 = 0.5 * (c_min + c_max)

    interfaces = []
    for i in range(len(depth) - 1):
        crossed = False
        if direction in ['increasing', 'both']:
            if conc[i] < c_50 <= conc[i+1]:
                crossed = True
        if direction in ['decreasing', 'both']:
            if conc[i] > c_50 >= conc[i+1]:
                crossed = True

        if crossed:
            # Linear interpolation to find exact crossing
            t = (c_50 - conc[i]) / (conc[i+1] - conc[i])
            z_cross = depth[i] + t * (depth[i+1] - depth[i])
            interfaces.append(z_cross)

    return np.array(interfaces)


def detect_interfaces_inflection(
    depth: np.ndarray,
    profile: np.ndarray,
    threshold: float = 0.05,
    min_separation: float = 3.0,
) -> np.ndarray:
    """
    Detect interface positions using inflection points (∂²c/∂z² = 0).

    For erf profiles, inflection points occur at interface centers.

    Args:
        depth: Depth array (Å)
        profile: Concentration profile [n_depth] or [n_depth, n_elements]
        threshold: Minimum first derivative magnitude to consider
        min_separation: Minimum separation between interfaces (Å)

    Returns:
        Array of interface positions (Å)
    """
    if profile.ndim == 2:
        # Use first element or sum
        conc = profile[:, 0]
    else:
        conc = profile

    # Compute first and second derivatives
    grad1 = np.gradient(conc, depth)
    grad2 = np.gradient(grad1, depth)

    # Normalize
    grad1_max = np.max(np.abs(grad1))
    if grad1_max < 1e-10:
        return np.array([])
    grad1_norm = np.abs(grad1) / grad1_max

    # Find zero crossings of second derivative where first derivative is significant
    interfaces = []
    for i in range(1, len(depth) - 1):
        if grad1_norm[i] > threshold:
            if grad2[i-1] * grad2[i+1] < 0:  # Sign change
                # Linear interpolation to find zero crossing
                t = -grad2[i-1] / (grad2[i] - grad2[i-1]) if abs(grad2[i] - grad2[i-1]) > 1e-10 else 0.5
                z_inflect = depth[i-1] + t * (depth[i] - depth[i-1])

                # Check minimum separation
                if len(interfaces) == 0 or z_inflect - interfaces[-1] >= min_separation:
                    interfaces.append(z_inflect)

    return np.array(interfaces)


@dataclass
class InterfaceDetectionResult:
    """Result of model-free interface detection."""
    gradient_positions: np.ndarray  # Positions from gradient method
    percent50_positions: np.ndarray  # Positions from 50% method
    inflection_positions: np.ndarray  # Positions from inflection method
    consensus_positions: np.ndarray  # Average of all methods
    consensus_std: np.ndarray  # Standard deviation across methods

    def to_dict(self) -> dict:
        return {
            'gradient_positions': self.gradient_positions.tolist(),
            'percent50_positions': self.percent50_positions.tolist(),
            'inflection_positions': self.inflection_positions.tolist(),
            'consensus_positions': self.consensus_positions.tolist(),
            'consensus_std': self.consensus_std.tolist(),
        }


def detect_interfaces(
    depth: np.ndarray,
    profile: np.ndarray,
    n_interfaces: Optional[int] = None,
    labels: Optional[List[str]] = None,
) -> InterfaceDetectionResult:
    """
    Detect interface positions using multiple model-free methods.

    Combines gradient, 50% crossing, and inflection point methods
    to provide robust interface position estimates.

    Args:
        depth: Depth array (Å)
        profile: Concentration profile [n_depth, n_elements]
        n_interfaces: Expected number of interfaces (for filtering)
        labels: Element labels

    Returns:
        InterfaceDetectionResult with positions from each method
    """
    # Method 1: Gradient maximum
    grad_pos = detect_interfaces_gradient(depth, profile)

    # Method 2: 50% crossing (for first element, assuming it's a layer element)
    if profile.ndim == 2 and profile.shape[1] > 1:
        pos_50 = detect_interfaces_50percent(depth, profile, element_index=1, direction='both')
    else:
        pos_50 = detect_interfaces_50percent(depth, profile if profile.ndim == 1 else profile[:, 0])

    # Method 3: Inflection points
    inflect_pos = detect_interfaces_inflection(depth, profile)

    # Compute consensus
    all_positions = []
    for pos in [grad_pos, pos_50, inflect_pos]:
        all_positions.extend(pos.tolist())

    if len(all_positions) == 0:
        return InterfaceDetectionResult(
            gradient_positions=grad_pos,
            percent50_positions=pos_50,
            inflection_positions=inflect_pos,
            consensus_positions=np.array([]),
            consensus_std=np.array([]),
        )

    # Cluster nearby positions
    all_positions = np.sort(all_positions)
    clusters = []
    current_cluster = [all_positions[0]]

    for pos in all_positions[1:]:
        if pos - current_cluster[-1] < 5.0:  # 5 Å clustering threshold
            current_cluster.append(pos)
        else:
            clusters.append(current_cluster)
            current_cluster = [pos]
    clusters.append(current_cluster)

    # Compute consensus from clusters
    consensus_positions = np.array([np.mean(c) for c in clusters])
    consensus_std = np.array([np.std(c) if len(c) > 1 else 0.0 for c in clusters])

    # Filter by expected number of interfaces
    if n_interfaces is not None and len(consensus_positions) > n_interfaces:
        # Keep the most prominent ones (largest clusters)
        cluster_sizes = np.array([len(c) for c in clusters])
        indices = np.argsort(cluster_sizes)[::-1][:n_interfaces]
        indices = np.sort(indices)  # Keep in depth order
        consensus_positions = consensus_positions[indices]
        consensus_std = consensus_std[indices]

    return InterfaceDetectionResult(
        gradient_positions=grad_pos,
        percent50_positions=pos_50,
        inflection_positions=inflect_pos,
        consensus_positions=consensus_positions,
        consensus_std=consensus_std,
    )


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("Creating five-layer profile...")
    profile = create_five_layer_profile()

    print(f"Depth: {profile.depth[0]} to {profile.depth[-1]} Å, {profile.n_depth} points")
    print(f"Elements: {profile.element_labels}")
    print(f"Concentrations shape: {profile.concentrations.shape}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(profile.element_labels):
        ax.plot(profile.depth, profile.concentrations[:, i], label=label, linewidth=2)

    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title('Five-layer Depth Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('/tmp/five_layer_profile.png', dpi=150)
    print("Saved plot to /tmp/five_layer_profile.png")

    # Test save/load
    profile.save_json('/tmp/five_layer_profile.json')
    print("Saved JSON to /tmp/five_layer_profile.json")

    loaded = DepthProfile.load_json('/tmp/five_layer_profile.json')
    print(f"Loaded profile: {loaded.element_labels}")

    # Test multilayer profile
    print("\n" + "=" * 60)
    print("Creating C/Al₂O₃/TiO₂/SiO₂/Si Multilayer Profile...")
    print("=" * 60)

    ml_profile = create_multilayer_profile(
        layer_thicknesses=[5.0, 10.0, 20.0, 30.0],
        interface_widths=[2.0, 3.0, 4.0, 5.0],
    )

    print(f"Elements: {ml_profile.element_labels}")
    print(f"  - C: carbon contamination (free)")
    print(f"  - Al2p: Al³⁺ in Al₂O₃ (free)")
    print(f"  - Ti2p: Ti⁴⁺ in TiO₂ (free)")
    print(f"  - Si2p (SiO2): Si⁴⁺ in SiO₂ (free)")
    print(f"  - O1s: O²⁻ in all oxides (linked)")
    print(f"  - Si2p (Si): Si⁰ substrate")

    # Verify charge balance
    print("\nCharge balance verification:")
    charge_results = verify_multilayer_charge_balance(ml_profile)
    print(f"  Max error: {charge_results['max_error']:.6f}")
    print(f"  RMSE: {charge_results['rmse']:.6f}")
    print(f"  Balanced: {charge_results['is_balanced']}")

    # Verify stoichiometry
    print("\nStoichiometry verification:")
    stoich_results = verify_multilayer_stoichiometry(ml_profile)
    print(f"  Max O error: {stoich_results['max_error']:.6f}")
    print(f"  RMSE: {stoich_results['rmse']:.6f}")
    print(f"  Satisfied: {stoich_results['is_satisfied']}")

    # Plot multilayer profile
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Full profile
    ax1 = axes[0, 0]
    colors = ['gray', 'green', 'blue', 'red', 'orange', 'purple']
    for i, label in enumerate(ml_profile.element_labels):
        ax1.plot(ml_profile.depth, ml_profile.concentrations[:, i],
                 label=label, linewidth=2, color=colors[i % len(colors)])
    ax1.set_xlabel('Depth (Å)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('C/Al₂O₃/TiO₂/SiO₂/Si Multilayer Profile')
    ax1.legend(loc='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.05)

    # Add layer boundaries
    layer_positions = [5, 15, 35, 65]
    layer_labels = ['C/Al₂O₃', 'Al₂O₃/TiO₂', 'TiO₂/SiO₂', 'SiO₂/Si']
    for pos, lbl in zip(layer_positions, layer_labels):
        ax1.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)
        ax1.text(pos, 0.95, lbl, rotation=90, fontsize=8, ha='right', va='top')

    # Top-right: Zoom on oxide layers
    ax2 = axes[0, 1]
    for i, label in enumerate(ml_profile.element_labels):
        ax2.plot(ml_profile.depth, ml_profile.concentrations[:, i],
                 label=label, linewidth=2, color=colors[i % len(colors)])
    ax2.set_xlabel('Depth (Å)')
    ax2.set_ylabel('Concentration')
    ax2.set_title('Oxide Layers (0-80 Å)')
    ax2.legend(loc='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 80)
    ax2.set_ylim(0, 0.8)

    # Bottom-left: Charge balance
    ax3 = axes[1, 0]
    ax3.plot(ml_profile.depth, charge_results['total_charge'], 'b-', linewidth=2)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Depth (Å)')
    ax3.set_ylabel('Total Charge')
    ax3.set_title(f'Charge Balance (RMSE={charge_results["rmse"]:.4f})')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)

    # Bottom-right: Oxygen stoichiometry
    ax4 = axes[1, 1]
    ax4.plot(ml_profile.depth, stoich_results['D_O_actual'], 'b-', linewidth=2, label='Actual O')
    ax4.plot(ml_profile.depth, stoich_results['D_O_expected'], 'r--', linewidth=2, label='Expected O')
    ax4.set_xlabel('Depth (Å)')
    ax4.set_ylabel('O²⁻ Concentration')
    ax4.set_title(f'Oxygen Stoichiometry (Max Error={stoich_results["max_error"]:.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig('/tmp/multilayer_profile.png', dpi=150)
    print(f"\nSaved multilayer plot to /tmp/multilayer_profile.png")


def create_exponential_prior(
    depth: np.ndarray,
    n_elements: int,
    substrate_idx: int,
    d_penetration: float = 30.0,
    surface_conc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create Exponential Prior for MEM optimization (Livesey & Smith, 1994).

    This prior assumes overlayer elements decay exponentially from the surface,
    while the substrate element fills the remainder (approaches 1.0 at depth).

    Reference:
        A.K. Livesey and G.C. Smith, "The determination of depth profiles from
        angle-dependent XPS using maximum entropy data analysis",
        J. Electron Spectrosc. Relat. Phenom. 67 (1994) 439-461, Equation 19.

    Args:
        depth: Depth array in Angstroms
        n_elements: Number of elements
        substrate_idx: Index of the substrate element (e.g., Si bulk)
        d_penetration: Penetration depth for exponential decay (Å).
                       Recommended: λ_substrate (IMFP of substrate element).
                       Typical values: 25-35 Å for Al Kα XPS.
        surface_conc: Surface concentration for each element.
                      If None, overlayer elements are equally distributed.

    Returns:
        D_prior: Prior depth profile array (n_depth, n_elements)

    Example:
        >>> from deppro.imfp import IMFP
        >>> # Calculate substrate IMFP
        >>> imfp_si = IMFP.tpp2m(KE=1383, Nv=4, density=2.33, Aw=28.09, Eg=1.1)
        >>> # Create prior for 4-element system (Al, Si_ox, Si_bulk, O)
        >>> D_prior = create_exponential_prior(depth, 4, substrate_idx=2,
        ...                                     d_penetration=imfp_si)

    Notes:
        - Overlayer elements: D_j(z) = c_j * exp(-z / d_penetration)
        - Substrate element: D_sub(z) = 1 - Σ D_j(z), approaches 1.0 at depth
        - The prior is normalized so that Σ D_e(z) = 1 at each depth
    """
    D_prior = np.zeros((len(depth), n_elements))

    # Set surface concentrations
    if surface_conc is None:
        # Equal distribution among overlayer elements
        n_overlayer = n_elements - 1
        surface_conc = np.ones(n_elements) / n_overlayer
        surface_conc[substrate_idx] = 0.0

    # Overlayer elements: exponential decay from surface
    for j in range(n_elements):
        if j != substrate_idx:
            D_prior[:, j] = surface_conc[j] * np.exp(-depth / d_penetration)

    # Substrate: fills the remainder
    D_prior[:, substrate_idx] = 1.0 - D_prior.sum(axis=1)
    D_prior[:, substrate_idx] = np.maximum(D_prior[:, substrate_idx], 1e-6)

    # Normalize to ensure sum = 1 at each depth
    return D_prior / D_prior.sum(axis=1, keepdims=True)
