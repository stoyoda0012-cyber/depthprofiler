"""
Element System Presets
=====================

Pre-built element configurations for common ARXPS measurement systems.
Each preset bundles elements, constraint indices, and stoichiometry
in a format compatible with MATLAB DepthProfiler's ezdeprof structure.

Usage:
    from deppro.simulate.presets import get_preset, list_presets

    preset = get_preset('layered_6elem')
    print(preset.elements)
    print(preset.free_indices)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ..depth_profile import ElementConfig


@dataclass
class ElementSystemPreset:
    """Complete element system for ARXPS simulation.

    Bundles element definitions, constraint classification, and
    stoichiometry coefficients. This parallels the MATLAB ezdeprof
    metadata structure.

    Parameters
    ----------
    name : str
        Preset identifier (e.g., 'layered_6elem').
    elements : list of ElementConfig
        Ordered element list. Order must match MATLAB convention.
    free_indices : list of int
        0-based indices of freely optimized film elements.
    linked_indices : list of int
        0-based indices of charge-balance-determined elements (e.g., O).
    substrate_indices : list of int
        0-based indices of substrate elements.
    linkedconst : list of float
        MATLAB ezdeprof/linkedconst format. Length = len(free_indices).
        D_linked = sum_i(D_free[i] * linkedconst[i]) where the factor
        accounts for stoichiometry. In MATLAB code, the actual O content
        is computed as: freelinkedconst = 1 + |charge_free/charge_linked|.
    charges : list of float
        Oxidation states for each element (+4 for Si in SiO2, etc.).
    min_photon_energy : float
        Minimum photon energy (eV) required for all orbital transitions.
        Computed as max(binding_energy) + work_function + margin.
    description : str
        Human-readable description.
    """

    name: str
    elements: List[ElementConfig]
    free_indices: List[int]
    linked_indices: List[int]
    substrate_indices: List[int]
    linkedconst: List[float]
    charges: List[float]
    min_photon_energy: float = 0.0
    description: str = ""

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def element_names(self) -> List[str]:
        return [e.label for e in self.elements]

    @property
    def free_matlab(self) -> np.ndarray:
        """1-based free indices for MATLAB/H5 compatibility."""
        return np.array([i + 1 for i in self.free_indices], dtype=np.float64)

    @property
    def subst_matlab(self) -> np.ndarray:
        """1-based substrate indices for MATLAB/H5 compatibility."""
        return np.array([i + 1 for i in self.substrate_indices], dtype=np.float64)

    @property
    def linked_matlab(self) -> np.ndarray:
        """1-based linked indices for MATLAB/H5 compatibility."""
        return np.array([i + 1 for i in self.linked_indices], dtype=np.float64)


# =============================================================================
# Pre-built Presets
# =============================================================================

# 6-element system matching existing 5x5x200 benchmark
# Element order: C(0), Al(+3), Ti(+4), Si(+4), O(-2), Si(0)
# free=[0,1,2,3], linked=[4], subst=[5]
LAYERED_6ELEM = ElementSystemPreset(
    name='layered_6elem',
    description='C/Al2O3/TiO2/SiO2/Si - 6 elements (HAXPES)',
    elements=[
        ElementConfig('C',  '1s', 'C',    oxidation_state=0,  stoichiometry=None),
        ElementConfig('Al', '1s', 'Al2O3', oxidation_state=3,  stoichiometry=2),
        ElementConfig('Ti', '1s', 'TiO2', oxidation_state=4,  stoichiometry=1),
        ElementConfig('Si', '1s', 'SiO2', oxidation_state=4,  stoichiometry=1),
        ElementConfig('O',  '1s', 'SiO2', oxidation_state=-2, stoichiometry=None),
        ElementConfig('Si', '1s', 'Si',   oxidation_state=0,  stoichiometry=None),
    ],
    free_indices=[0, 1, 2, 3],       # C, Al, Ti, Si(+4)
    linked_indices=[4],               # O(-2)
    substrate_indices=[5],            # Si(0)
    # linkedconst: freelinkedconst = 1 + |charge_free / charge_linked|
    # C(0):   1 + |0/2| = 1.0
    # Al(+3): 1 + |3/2| = 2.5
    # Ti(+4): 1 + |4/2| = 3.0
    # Si(+4): 1 + |4/2| = 3.0
    linkedconst=[1.0, 2.5, 3.0, 3.0],
    charges=[0, 3, 4, 4, -2, 0],
    min_photon_energy=5100.0,  # Ti 1s BE=4966 eV + margin
)


# 7-element system matching existing 10x10x100 checker benchmark
# Element order: Si(0), Si(+4), Ti(+4), Al(+3), C(0), B(0), O(-2)
# free=[1,2,3,4,5], linked=[6], subst=[0]
CHECKER_7ELEM = ElementSystemPreset(
    name='checker_7elem',
    description='Si(0)/Si(+4)/Ti(+4)/Al(+3)/C(0)/B(0)/O(-2) - 7 elements (checker)',
    elements=[
        ElementConfig('Si', '1s', 'Si',   oxidation_state=0,  stoichiometry=None),
        ElementConfig('Si', '1s', 'SiO2', oxidation_state=4,  stoichiometry=1),
        ElementConfig('Ti', '1s', 'TiO2', oxidation_state=4,  stoichiometry=1),
        ElementConfig('Al', '1s', 'Al2O3', oxidation_state=3,  stoichiometry=2),
        ElementConfig('C',  '1s', 'C',    oxidation_state=0,  stoichiometry=None),
        ElementConfig('B',  '1s', 'B',    oxidation_state=0,  stoichiometry=None),
        ElementConfig('O',  '1s', 'SiO2', oxidation_state=-2, stoichiometry=None),
    ],
    free_indices=[1, 2, 3, 4, 5],    # Si(+4), Ti, Al, C, B
    linked_indices=[6],               # O(-2)
    substrate_indices=[0],            # Si(0)
    # Si(+4): 1+|4/2|=3.0, Ti(+4): 3.0, Al(+3): 2.5, C(0): 1.0, B(0): 1.0
    linkedconst=[3.0, 3.0, 2.5, 1.0, 1.0],
    charges=[0, 4, 4, 3, 0, 0, -2],
    min_photon_energy=5100.0,  # Ti 1s BE=4966 eV + margin
)


# 3-element system: simplest test case
# Element order: O(-2), Si(+4), Si(0)
# free=[1], linked=[0], subst=[2]
SIO2_SI = ElementSystemPreset(
    name='sio2_si',
    description='O/SiO2/Si bilayer - 3 elements (simplest case)',
    elements=[
        ElementConfig('O',  '1s', 'SiO2', oxidation_state=-2, stoichiometry=None),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4,  stoichiometry=1),
        ElementConfig('Si', '2p', 'Si',   oxidation_state=0,  stoichiometry=None),
    ],
    free_indices=[1],                 # Si(+4)
    linked_indices=[0],               # O(-2)
    substrate_indices=[2],            # Si(0)
    # Si(+4): 1+|4/2|=3.0
    linkedconst=[3.0],
    charges=[-2, 4, 0],
    min_photon_energy=650.0,  # O 1s BE=543 eV + margin
)


# =============================================================================
# Registry
# =============================================================================

_PRESETS: Dict[str, ElementSystemPreset] = {
    'layered_6elem': LAYERED_6ELEM,
    'checker_7elem': CHECKER_7ELEM,
    'sio2_si': SIO2_SI,
}


def get_preset(name: str) -> ElementSystemPreset:
    """Get an element system preset by name.

    Parameters
    ----------
    name : str
        Preset name (e.g., 'layered_6elem', 'sio2_si').

    Returns
    -------
    ElementSystemPreset

    Raises
    ------
    KeyError
        If preset name not found.
    """
    if name not in _PRESETS:
        available = ', '.join(sorted(_PRESETS.keys()))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return _PRESETS[name]


def list_presets() -> List[str]:
    """List all available preset names."""
    return sorted(_PRESETS.keys())


def register_preset(preset: ElementSystemPreset) -> None:
    """Register a custom element system preset.

    Parameters
    ----------
    preset : ElementSystemPreset
        The preset to register. Will overwrite if name exists.
    """
    _PRESETS[preset.name] = preset
