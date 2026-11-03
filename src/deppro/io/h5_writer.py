"""
H5Writer - HDF5 writer fully compatible with the MATLAB DepthProfiler GUI

Produces H5 files that DepthProfiler.m's ReadResult_Callback can load,
in the same format as MATLAB's h5write().

Compatibility rules:
- strings: vlen-string, shape (N, 1) column vector
- scalars: shape (1,) arrays (the h5write convention)
- vectors: (N, 1) column vectors as a rule
- multi-dimensional arrays: transposed for MATLAB column-major

Example:
    from deppro.io import H5Writer, create_gui_compatible_h5

    # Short form: required parameters only
    create_gui_compatible_h5(
        filepath='result.h5',
        depth=np.linspace(0, 150, 76),
        angles=np.array([15, 30, 45, 60, 75, 85]),
        elements=['C', 'O', 'Si'],
        Drlt=profile_recon,
        Dtrn=profile_true,
    )

    # Long form: control over every parameter
    writer = H5Writer('result.h5')
    writer.set_basic_params(depth, angles, elements, ...)
    writer.set_profiles(Drlt, Dtrn, Dstk)
    writer.write()
"""

import numpy as np
import h5py
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from deppro.imfp import IMFP

# =============================================================================
# Element database (equivalent of BindingEnergyTable)
# =============================================================================

# Element symbol → atomic number
ELEMENT_TO_ATOMIC_NUMBER = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92,
}

# Orbital name → orbital number (same order as MATLAB orbitallist)
ORBITAL_LIST = ['1s', '2s', '2p', '3s', '3p', '3d', '4s', '4p', '4d', '4f',
                '5s', '5p', '5d', '5f', '6s', '6p', '6d', '7s', '7p']
ORBITAL_TO_NUMBER = {orb: i + 1 for i, orb in enumerate(ORBITAL_LIST)}

# =============================================================================
# Compound database (equivalent of CompoundTable)
# =============================================================================

COMPOUND_PROPERTIES = {
    # compound: (valencecharge, density, molecularweight, bandgap)
    'Si': (4, 2.329, 28.08553, 1.12),
    'SiO2': (16, 2.2, 60.08, 8.9),
    'Oxide': (6, 2.2, 16.0, 6.0),
    'O': (6, 1.429e-3, 16.0, 0.0),  # oxygen gas
    'Hf': (4, 13.31, 178.49, 0.0),
    'Al': (3, 2.7, 26.98, 0.0),
    'Al2O3': (24, 3.95, 101.96, 8.8),
    'Ti': (4, 4.5, 47.87, 0.0),
    'TiO2': (24, 4.23, 79.87, 3.2),
    'C': (4, 2.267, 12.01, 5.5),  # diamond
    'N': (5, 1.25e-3, 14.01, 0.0),
    'Ta': (5, 16.65, 180.95, 0.0),
    'Ta2O5': (32, 8.2, 441.89, 4.4),
    'Zr': (4, 6.52, 91.22, 0.0),
    'ZrO2': (16, 5.68, 123.22, 5.0),
    # Default average values
    'AVERAGE': (10, 3.0, 50.0, 3.0),
}

def get_compound_properties(compound: str) -> tuple:
    """Get the properties of a compound (valencecharge, density, molecularweight, bandgap)"""
    if compound in COMPOUND_PROPERTIES:
        return COMPOUND_PROPERTIES[compound]
    # Try the element name
    if compound in ELEMENT_TO_ATOMIC_NUMBER:
        # Element defaults
        return COMPOUND_PROPERTIES.get('AVERAGE', (10, 3.0, 50.0, 3.0))
    return COMPOUND_PROPERTIES['AVERAGE']

def estimate_charge_from_compound(compound: str, element: str) -> float:
    """Estimate the oxidation state from the compound and the element"""
    compound = str(compound)
    element = str(element)

    # Oxygen is always -2
    if element == 'O':
        return -2.0

    # Common oxide patterns
    oxide_charges = {
        'SiO2': 4.0,
        'Al2O3': 3.0,
        'TiO2': 4.0,
        'Ta2O5': 5.0,
        'ZrO2': 4.0,
    }
    if compound in oxide_charges and element != 'O':
        return oxide_charges[compound]

    # Pure elements are 0
    if compound == element or compound in ELEMENT_TO_ATOMIC_NUMBER:
        return 0.0

    # Oxide generic
    if compound == 'Oxide':
        return -2.0

    return 0.0

def format_stringcharge(charge: float) -> str:
    """Format a charge as a string: 4 → '+4', -2 → '-2', 0 → '0'"""
    if charge > 0:
        return f'+{int(charge)}'
    elif charge < 0:
        return f'{int(charge)}'
    else:
        return '0'

def format_element_with_charge(element: str, charge: float) -> str:
    """Element with charge as one string: 'Si', 4 → 'Si(+4)'"""
    sc = format_stringcharge(charge)
    return f'{element}({sc})'

def determine_linked_const(charge: np.ndarray, free: np.ndarray, const: np.ndarray, linked: np.ndarray) -> np.ndarray:
    """
    Equivalent of MATLAB's DeterminingLinkedConst.m

    linkedconst = 1 - charge(freeconst) / mean(charge(linked))
    linkedconst(linkedconst < 1) = 1

    Args:
        charge: charges of all elements (n_elements,)
        free: free element indices (1-based)
        const: fixed element indices (1-based)
        linked: linked element indices (1-based)

    Returns:
        linkedconst: array of shape (len(freeconst),)
    """
    # freeconst = sort([free, const]) (1-based indices)
    freeconst = np.sort(np.concatenate([np.atleast_1d(free), np.atleast_1d(const)]))
    freeconst = freeconst[freeconst > 0]  # drop zeros

    if len(linked) == 0 or (len(linked) == 1 and linked[0] == 0):
        # No linked element: all ones
        return np.ones(len(freeconst), dtype=np.float32)

    # Only the valid indices of linked
    linked_valid = linked[linked > 0]
    if len(linked_valid) == 0:
        return np.ones(len(freeconst), dtype=np.float32)

    # Charges (converted to 0-based)
    charge_freeconst = charge[freeconst.astype(int) - 1]
    charge_linked = charge[linked_valid.astype(int) - 1]
    mean_charge_linked = np.mean(charge_linked)

    if mean_charge_linked == 0:
        return np.ones(len(freeconst), dtype=np.float32)

    linkedconst = 1.0 - charge_freeconst / mean_charge_linked
    linkedconst[linkedconst < 1] = 1.0

    return linkedconst.astype(np.float32)


class H5Writer:
    """
    HDF5 writer fully compatible with the MATLAB DepthProfiler GUI

    Produces H5 files in the format that MATLAB's
    deppro.io.DataStore.loadFullEzdeprofFromH5() reads.

    MATLAB h5write() compatibility:
    - scalars are stored as (1,) arrays
    - strings as vlen-string, (N, 1)
    - vectors as (N, 1) column vectors as a rule
    """

    # vlen-string dtype (compatible with MATLAB cellstr)
    VLEN_STR_DTYPE = h5py.special_dtype(vlen=str)

    def __init__(self, filepath: Union[str, Path]):
        """
        Args:
            filepath: output H5 file path
        """
        self.filepath = Path(filepath)
        self._ezdeprof: Dict[str, Any] = {}
        self._profiles: Dict[str, np.ndarray] = {}
        self._objectives: Dict[str, np.ndarray] = {}

    def set_basic_params(
        self,
        depth: np.ndarray,
        angles: np.ndarray,
        elements: List[str],
        angles_opt: Optional[np.ndarray] = None,
        orbitals: Optional[List[str]] = None,
        compounds: Optional[List[str]] = None,
        binding_energies: Optional[np.ndarray] = None,
        photon_energy: float = 1486.6,  # Al Kα
        simulation: bool = True,
        slice_size: Optional[float] = None,  # film slice size [Å]
        # Element property settings
        subst: Optional[List[int]] = None,    # substrate element indices (1-based)
        free: Optional[List[int]] = None,      # free element indices (1-based)
        linked: Optional[List[int]] = None,    # linked element indices (1-based, 0 = none)
        prpties: Optional[List[str]] = None,   # property strings
        # Charges (estimated from the compounds when omitted)
        charges: Optional[List[float]] = None,
    ):
        """
        Set the basic parameters

        Args:
            depth: depth axis [Å] (n_depth,)
            angles: measurement angles [°] (n_angles,) - thetaobs
            elements: list of element symbols ['Si', 'Si', 'O', 'Al']
            angles_opt: optimization angles [°] - thetaopt (default: the standard 16 points)
            orbitals: orbitals ['1s', '1s', '1s', '2p'] (estimated automatically when omitted)
            compounds: compound names ['Si', 'SiO2', 'Oxide', 'Al2O3'] (default: same as the elements)
            binding_energies: binding energies [eV] (default values when omitted)
            photon_energy: photon energy [eV]
            simulation: whether this is simulation mode
            slice_size: film slice size [Å] (default: about totaldepth/10)
            charges: charge of each element [0, 4, -2, 4] (estimated from the compounds when omitted)
        """
        n_depth = len(depth)
        n_angles = len(angles)
        n_elements = len(elements)

        # Estimate the orbitals
        if orbitals is None:
            orbitals = self._estimate_orbitals(elements)

        # Compound names
        if compounds is None:
            compounds = elements.copy()

        # Binding energies
        if binding_energies is None:
            binding_energies = self._estimate_binding_energies(elements, orbitals)

        # Basic parameters
        ez = self._ezdeprof
        ez['SimulationOrExperiment'] = simulation
        ez['depth'] = depth.astype(np.float32)
        ez['totaldepth'] = float(depth[-1])
        ez['stepsize'] = float(depth[1] - depth[0]) if len(depth) > 1 else 2.0
        ez['m'] = n_depth
        # slice: film slice size (independent of stepsize)
        # The reference file has, for example, slice=20 and stepsize=30
        if slice_size is None:
            # Default: about 1/10 of totaldepth (at least 10 Å)
            slice_size = max(10.0, float(depth[-1]) / 10.0)
        ez['slice'] = float(slice_size)

        # Angles
        ez['thetaobs'] = angles.astype(np.float32)
        # thetaopt: optimization angles (default: the standard 16 points, -10 to 90 degrees)
        if angles_opt is None:
            angles_opt = np.array([-10, 0, 10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90], dtype=np.float32)
        ez['thetaopt'] = angles_opt.astype(np.float32)
        ez['angle'] = float(angles[-1]) if len(angles) > 0 else 60.0  # maximum angle

        # Element information
        ez['elementsymbol'] = elements
        ez['orbital'] = orbitals
        ez['compound'] = compounds
        ez['bindingenergy'] = binding_energies.astype(np.float32)
        # photonenergy: source energy for each element (normally the same value for all)
        ez['photonenergy'] = np.full(n_elements, photon_energy, dtype=np.float32)

        # ★ Element number: atomic number (index into BindingEnergyTable)
        elementnumber = np.zeros(n_elements, dtype=np.uint8)
        for i, elem in enumerate(elements):
            elementnumber[i] = ELEMENT_TO_ATOMIC_NUMBER.get(elem, 1)
        ez['elementnumber'] = elementnumber

        # ★ Orbital number: index into the orbital list (1-based)
        orbitalnumber = np.zeros(n_elements, dtype=np.uint8)
        for i, orb in enumerate(orbitals):
            orbitalnumber[i] = ORBITAL_TO_NUMBER.get(orb, 1)
        ez['orbitalnumber'] = orbitalnumber

        # ★ Charges: estimated from the compounds
        if charges is not None:
            charge_arr = np.array(charges, dtype=np.float32)
        else:
            charge_arr = np.zeros(n_elements, dtype=np.float32)
            for i, (comp, elem) in enumerate(zip(compounds, elements)):
                charge_arr[i] = estimate_charge_from_compound(comp, elem)
        ez['charge'] = charge_arr

        # ★ stringcharge and element (with charge)
        stringcharge = [format_stringcharge(c) for c in charge_arr]
        element_with_charge = [format_element_with_charge(elem, c) for elem, c in zip(elements, charge_arr)]
        ez['stringcharge'] = stringcharge
        ez['element'] = element_with_charge

        # Substrate / free / linked settings
        if subst is not None:
            ez['subst'] = np.array(subst, dtype=np.uint8)
        else:
            ez['subst'] = np.array([1], dtype=np.uint8)  # default: the first element is the substrate

        if free is not None:
            ez['free'] = np.array(free, dtype=np.uint8)
        else:
            # Default: everything but the substrate is free
            subst_set = set(ez['subst'].flatten())
            ez['free'] = np.array([i for i in range(1, n_elements + 1) if i not in subst_set], dtype=np.uint8)

        if linked is not None:
            ez['linked'] = np.array(linked, dtype=np.uint8)
        else:
            ez['linked'] = np.array([0], dtype=np.uint8)  # default: no linked element

        ez['const'] = np.array([0], dtype=np.uint8)

        # ★ linkedconst: the logic of DeterminingLinkedConst
        # freeconst = sort([free, const])
        free_arr = np.atleast_1d(ez['free'])
        const_arr = np.atleast_1d(ez['const'])
        linked_arr = np.atleast_1d(ez['linked'])
        linkedconst = determine_linked_const(charge_arr, free_arr, const_arr, linked_arr)
        ez['linkedconst'] = linkedconst

        # antisubst: the elements other than the substrate
        subst_set = set(ez['subst'].flatten())
        ez['antisubst'] = np.array([i for i in range(1, n_elements + 1) if i not in subst_set], dtype=np.uint8)

        # Colors (RGB, a different color for each element)
        default_colors = np.array([
            [0.0, 0.765, 0.0],   # Green
            [1.0, 0.0, 0.0],     # Red
            [0.0, 0.0, 0.765],   # Blue
            [1.0, 0.5, 0.0],     # Orange
            [0.5, 0.0, 0.5],     # Purple
        ], dtype=np.float32)
        ez['Col'] = default_colors[:n_elements] if n_elements <= 5 else \
                    np.random.rand(n_elements, 3).astype(np.float32)

        # ★ Material parameters: from the compound table
        valencecharge = np.zeros(n_elements, dtype=np.float32)
        density = np.zeros(n_elements, dtype=np.float32)
        molecularweight = np.zeros(n_elements, dtype=np.float32)
        bandgap = np.zeros(n_elements, dtype=np.float32)
        for i, comp in enumerate(compounds):
            vc, dens, mw, bg = get_compound_properties(comp)
            valencecharge[i] = vc
            density[i] = dens
            molecularweight[i] = mw
            bandgap[i] = bg
        ez['valencecharge'] = valencecharge
        ez['density'] = density
        ez['molecularweight'] = molecularweight
        ez['bandgap'] = bandgap
        ez['workfunction'] = 5.0

        # Property strings
        if prpties is not None:
            ez['prpties'] = prpties
        else:
            # Default: subst=Subst., linked=Linked, otherwise Free
            _prpties = ['Free'] * n_elements
            for i in ez['subst'].flatten():
                if 1 <= i <= n_elements:
                    _prpties[i - 1] = 'Subst.'
            for i in ez['linked'].flatten():
                if 1 <= i <= n_elements:
                    _prpties[i - 1] = 'Linked'
            ez['prpties'] = _prpties

        # TPP-2M / IMFP settings
        ez['TPPver'] = 3  # as in the reference file
        ez['gammaeff'] = 0.8  # as in the reference file

        # Other settings
        # NP_Set: number of elements (NP_Set=4 in the 4-component reference H5 written by MATLAB)
        # NP_Calc: number of layers used in the calculation (initially 2)
        ez['NP_Set'] = n_elements
        ez['NP_Calc'] = 2
        # NP_Max: ceil(length(elementsymbol)/2)*2
        ez['NP_Max'] = int(np.ceil(n_elements / 2) * 2)
        ez['crossecapproximord'] = 3  # as in the reference file
        ez['crossecapproximview'] = 0  # as in the reference file
        ez['photon'] = photon_energy
        ez['deltaphoton'] = 1000.0  # as in the reference file
        ez['noise'] = 1.0  # as in the reference file
        ez['dxdydt'] = np.array([1, 1, 100], dtype=np.uint16)  # as in the reference file
        # datapoints: [n_angles_obs, n_samples, n_angles_opt]
        ez['datapoints'] = np.array([n_angles, 1, len(angles_opt)], dtype=np.uint16)

        # Loop and stack settings
        ez['loop'] = 10  # as in the reference file
        ez['stack'] = 1
        ez['currentloop'] = 10  # as in the reference file
        ez['currentstack'] = 1
        ez['iterationinterval'] = 1  # as in the reference file
        ez['numberofmodeling'] = 1

        # Initial thickness and abruptness
        # thckini/abrpini: [initial, lower bound, upper bound]
        # The reference file has thckini=[210, 0, 600], abrpini=[20, 20, 20]
        total_depth = float(depth[-1]) if len(depth) > 0 else 600.0
        thck_init = total_depth / 3  # about 1/3 of the depth as the initial thickness
        ez['thckini'] = np.array([thck_init, 0, total_depth], dtype=np.float32)
        ez['abrpini'] = np.array([20.0, 20.0, 20.0], dtype=np.float32)
        ez['thckmod'] = np.array([thck_init, 0, total_depth], dtype=np.float32)
        ez['abrpmod'] = np.array([20.0, 20.0, 20.0], dtype=np.float32)

        # Initial values of FeatureVector_Source
        # freeconst = sort([free, const]) - used inside ezdeprof
        # freeconstlinked = sort([free, const, linked]) - used by the root FV0stk
        free_arr = np.atleast_1d(ez['free'])
        const_arr = np.atleast_1d(ez['const'])
        linked_arr = np.atleast_1d(ez['linked'])
        free_valid = [x for x in free_arr if x > 0]
        const_valid = [x for x in const_arr if x > 0]
        linked_valid = [x for x in linked_arr if x > 0]

        # freeconst: used by ezdeprof/FeatureVector_*
        freeconst = sorted(free_valid + const_valid)
        n_freeconst = len(freeconst)

        # FeatureVector_Source: [thck, abrp, (pos, width, height) × n_freeconst]
        # 4-component reference H5 written by MATLAB: [190, 20, 180, 60, 33.33, 0, 60, 33.33]
        # position: first = thck*0.95, second = 0 (surface)
        fv_source = np.zeros(2 + 3 * n_freeconst, dtype=np.float32)
        fv_source[0] = thck_init  # thck
        fv_source[1] = 20.0       # abrp (FWHM)
        # Initial parameters of each element (position, width, height)
        for i in range(n_freeconst):
            if i == 0:
                # First free element: near the film thickness (thck - 10)
                # MATLAB: round(thck*0.95) but in practice uses thck-10 pattern
                fv_source[2 + 3*i + 0] = round(thck_init - 10)  # position
            else:
                # Second and later: near the surface (0)
                fv_source[2 + 3*i + 0] = 0.0  # position
            fv_source[2 + 3*i + 1] = 60.0       # width
            fv_source[2 + 3*i + 2] = 100.0 / 3  # height: 33.3%
        ez['FeatureVector_Source'] = fv_source

        # FeatureVector_Set and FeatureVector_Calc are the same
        ez['FeatureVector_Set'] = fv_source.copy()
        ez['FeatureVector_Calc'] = fv_source.copy()

        # SupportVector_Set: [3×subst + 3×linked], each (pos, width, height)
        subst_arr = np.atleast_1d(ez['subst'])
        n_subst = len(subst_arr)
        n_linked = len(linked_valid)
        sv_len = 3 * (n_subst + n_linked)
        sv_set = np.zeros(sv_len, dtype=np.float32)
        # Substrate parameters
        for i in range(n_subst):
            sv_set[3*i + 0] = total_depth / 2   # position
            sv_set[3*i + 1] = total_depth / 2   # width
            sv_set[3*i + 2] = 100.0             # height: 100%
        # Linked parameters
        for i in range(n_linked):
            offset = 3 * n_subst
            sv_set[offset + 3*i + 0] = thck_init / 2   # position
            sv_set[offset + 3*i + 1] = thck_init       # width
            sv_set[offset + 3*i + 2] = 100.0 / 3       # height: 33.3%
        ez['SupportVector_Set'] = sv_set
        ez['SupportVector_Calc'] = sv_set.copy()
        # subsini/subsmod: (3, 1) form [initial concentration; lower bound; upper bound]
        # The reference file has [1, 1, 1] (all 1.0)
        ez['subsini'] = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)  # (3, 1)
        ez['subsmod'] = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)  # (3, 1)

        # Optimization settings
        ez['optimizationmethodvalue'] = 1
        ez['initialguessvalue'] = 1
        ez['priormodelvalue'] = 2  # as in the reference file
        ez['transfermatrixupdatevalue'] = 2  # as in the reference file
        ez['photoionizationcrosssection'] = 1
        ez['verificationschemevalue'] = 1
        ez['verificationmethod'] = 2  # as in the reference file
        ez['holdout'] = 1  # as in the reference file
        ez['Kfold'] = 1  # as in the reference file
        ez['NumberofNearestNeighbor'] = 1  # as in the reference file
        ez['degreecontrol'] = 0  # as in the reference file
        ez['alphacontrol'] = 0  # as in the reference file
        ez['verificationstatus'] = 0
        ez['calculationinitialized'] = np.array([2, 1, 1], dtype=np.uint64)  # as in the reference file
        ez['iterationtime'] = 0.0
        ez['iterationtotal'] = 0

        # times: empty array (MATLAB compatible)
        ez['times'] = np.zeros((1, 0), dtype=np.float32)

        # Compute the IMFP values automatically
        self.compute_lambda_values()

    def set_transfer_matrix(
        self,
        TransMat: np.ndarray,
        TransMatOpt: Optional[np.ndarray] = None,
        lambda_basic: Optional[np.ndarray] = None,
        lambda_obs: Optional[np.ndarray] = None,
        lambda_cos_theta_obs: Optional[np.ndarray] = None,
        lambda_cos_theta_opt: Optional[np.ndarray] = None,
    ):
        """
        Set the transfer matrices

        Args:
            TransMat: observation transfer matrix (n_angles_obs, n_depth, n_elements) - TransMatObs
            TransMatOpt: optimization transfer matrix (n_angles_opt, n_depth, n_elements) - generated when omitted
            lambda_basic: basic IMFP (n_elements, n_elements, 1) → H5: (1, n_elements, n_elements)
            lambda_obs: depth-wise IMFP (n_elements, n_depth) → H5: (1, n_elements, n_depth)
            lambda_cos_theta_obs: λcosθ for observation (n_angles_obs, n_elements)
            lambda_cos_theta_opt: λcosθ for optimization (n_angles_opt, n_elements) - generated when omitted
        """
        ez = self._ezdeprof
        ez['TransMatObs'] = TransMat.astype(np.float32)

        # TransMatOpt: used if given, otherwise generated by _compute_derived_params
        if TransMatOpt is not None:
            ez['TransMatOpt'] = TransMatOpt.astype(np.float32)
        # else: _compute_derived_params creates a zero matrix of thetaopt size

        if lambda_basic is not None:
            # Stored as (n_elements, n_elements, 1), transposed on write
            ez['lambdabasic'] = lambda_basic.astype(np.float32)

        if lambda_obs is not None:
            # (n_elements, n_depth) → (1, n_elements, n_depth)
            if lambda_obs.ndim == 2:
                lambda_obs = lambda_obs[np.newaxis, :, :]  # (1, n_elements, n_depth)
            ez['lambdaobs'] = lambda_obs.astype(np.float32)
            ez['lambdaopt'] = lambda_obs.astype(np.float32)

        if lambda_cos_theta_obs is not None:
            ez['lambdacosinethetaobs'] = lambda_cos_theta_obs.astype(np.float32)

        if lambda_cos_theta_opt is not None:
            ez['lambdacosinethetaopt'] = lambda_cos_theta_opt.astype(np.float32)
        # else: _compute_derived_params creates a zero matrix of thetaopt size

    def compute_lambda_values(self, Dini: Optional[np.ndarray] = None):
        """
        Compute the IMFP-related values from the material parameters

        From the binding_energy, photon_energy, valencecharge, density,
        molecularweight and bandgap given to set_basic_params, computes:
        - lambdabasic: (n_elements, n_elements) IMFP of each element's photoelectron through each compound
        - lambdaobs/lambdaopt: (n_depth, n_elements) depth-wise IMFP (harmonic mean)
        - lambdacosinethetaobs/opt: (n_angles, n_elements) mean(λ)×cosθ

        Args:
            Dini: initial depth profile (n_depth, n_elements)
                        Used for the same harmonic-mean calculation as MATLAB.
                        When omitted, the diagonal (uniform) is used.

        Note:
            Called automatically after set_basic_params; calling it again
            after set_profiles has set Dini gives the MATLAB-compatible
            result.
        """
        ez = self._ezdeprof

        # Required parameters
        photon = float(ez.get('photon', 8000.0))
        binding_energy = np.atleast_1d(ez.get('bindingenergy', np.zeros(1))).flatten()
        valencecharge = np.atleast_1d(ez.get('valencecharge', np.ones(1) * 4)).flatten()
        density = np.atleast_1d(ez.get('density', np.ones(1) * 2.3)).flatten()
        molecularweight = np.atleast_1d(ez.get('molecularweight', np.ones(1) * 28)).flatten()
        bandgap = np.atleast_1d(ez.get('bandgap', np.zeros(1))).flatten()
        gammaeff = float(ez.get('gammaeff', 0.8))

        n_elements = len(binding_energy)
        kinetic_energies = photon - binding_energy  # kinetic energies

        # lambdabasic: (n_elements, n_elements)
        # Row i: photoelectron of element i (KE_i); column j: IMFP computed with the properties of compound j
        lambda_basic = np.zeros((n_elements, n_elements), dtype=np.float32)
        for i, ke in enumerate(kinetic_energies):
            for j in range(n_elements):
                lambda_basic[i, j] = IMFP.tpp2m(
                    ke,
                    valencecharge[j],
                    density[j],
                    molecularweight[j],
                    bandgap[j]
                )

        # H5 form: (1, n_elements, n_elements)
        ez['lambdabasic'] = lambda_basic.reshape(1, n_elements, n_elements)

        # lambdaobs/lambdaopt: depth-wise IMFP
        depth = np.atleast_1d(ez.get('depth', np.array([0]))).flatten()
        n_depth = len(depth)

        if Dini is not None:
            # MATLAB compatible: effective IMFP from the harmonic mean
            # lambdaobs[d, e] = gammaeff / sum(Dini[d, j] / lambdabasic[e, j])
            Dmat = Dini if Dini.shape[0] == n_depth else Dini.T  # (n_depth, n_elements)
            lambda_obs = np.zeros((n_depth, n_elements), dtype=np.float32)
            for d in range(n_depth):
                for e in range(n_elements):
                    denom = np.sum(Dmat[d, :] / lambda_basic[e, :])
                    if denom > 1e-10:
                        lambda_obs[d, e] = gammaeff / denom
                    else:
                        lambda_obs[d, e] = lambda_basic[e, 0]  # fallback
        else:
            # Simple form: diagonal (uniform)
            lambda_diag = np.diag(lambda_basic)  # (n_elements,)
            lambda_obs = np.tile(lambda_diag[np.newaxis, :], (n_depth, 1))  # (n_depth, n_elements)

        # H5 form: (1, n_elements, n_depth) - transposed
        ez['lambdaobs'] = lambda_obs.T.reshape(1, n_elements, n_depth).astype(np.float32)
        ez['lambdaopt'] = lambda_obs.T.reshape(1, n_elements, n_depth).astype(np.float32)

        # lambdacosinethetaobs/opt: mean(λobs) × cosθ
        thetaobs = np.atleast_1d(ez.get('thetaobs', np.array([0]))).flatten()
        thetaopt = np.atleast_1d(ez.get('thetaopt', np.array([0]))).flatten()

        # Average over depth
        lambda_mean = np.mean(lambda_obs, axis=0)  # (n_elements,)

        # (n_angles, n_elements) = mean(λ) × cosθ
        cos_obs = np.cos(np.radians(thetaobs))[:, np.newaxis]
        cos_opt = np.cos(np.radians(thetaopt))[:, np.newaxis]

        lambda_cos_obs = lambda_mean[np.newaxis, :] * cos_obs  # (n_angles, n_elements)
        lambda_cos_opt = lambda_mean[np.newaxis, :] * cos_opt

        # Internal form: (n_angles, n_elements) - transposed on write
        ez['lambdacosinethetaobs'] = lambda_cos_obs.astype(np.float32)
        ez['lambdacosinethetaopt'] = lambda_cos_opt.astype(np.float32)

    def set_profiles(
        self,
        Drlt: np.ndarray,
        Dtrn: Optional[np.ndarray] = None,
        Dstk: Optional[np.ndarray] = None,
        Xobs: Optional[np.ndarray] = None,
        Xopt: Optional[np.ndarray] = None,
    ):
        """
        Set the depth profiles

        Args:
            Drlt: result profile (n_depth, n_elements) or (n_depth, n_elements, n_stack)
            Dtrn: true profile (for simulation)
            Dstk: initial profile
            Xobs: observed intensities (n_angles_obs, n_elements) - thetaobs size
            Xopt: optimization intensities (n_angles_opt, n_elements) - thetaopt size
        """
        ez = self._ezdeprof

        # Convert to 3D arrays
        if Drlt.ndim == 2:
            Drlt = Drlt[:, :, np.newaxis]
        self._profiles['Drlt'] = Drlt.astype(np.float32)

        if Dtrn is not None:
            if Dtrn.ndim == 2:
                Dtrn = Dtrn[:, :, np.newaxis]
            self._profiles['Dtrn'] = Dtrn.astype(np.float32)
        else:
            self._profiles['Dtrn'] = np.zeros_like(self._profiles['Drlt'])

        if Dstk is not None:
            if Dstk.ndim == 2:
                Dstk = Dstk[:, :, np.newaxis]
            self._profiles['Dstk'] = Dstk.astype(np.float32)
        else:
            self._profiles['Dstk'] = np.zeros_like(self._profiles['Drlt'])

        # Number of angles
        n_angles_obs = len(ez.get('thetaobs', [0]))
        n_angles_opt = len(ez.get('thetaopt', [0]))
        n_elements = self._profiles['Drlt'].shape[1]
        n_stack = self._profiles['Drlt'].shape[2]

        # Xobs: observed intensities (thetaobs size)
        if Xobs is not None:
            if Xobs.ndim == 2:
                Xobs = Xobs[:, :, np.newaxis]
            self._profiles['Xobs'] = Xobs.astype(np.float32)
        else:
            self._profiles['Xobs'] = np.zeros((n_angles_obs, n_elements, n_stack), dtype=np.float32)

        # Xopt: optimization intensities (thetaopt size)
        if Xopt is not None:
            if Xopt.ndim == 2:
                Xopt = Xopt[:, :, np.newaxis]
            self._profiles['Xopt'] = Xopt.astype(np.float32)
        else:
            self._profiles['Xopt'] = np.zeros((n_angles_opt, n_elements, n_stack), dtype=np.float32)

        # Root-level Xrlt, Xstk
        # Xrlt: thetaopt size (optimization / display)
        # Xstk: thetaobs size (observed data)
        self._profiles['Xrlt'] = self._profiles['Xopt'].copy()  # thetaopt
        self._profiles['Xstk'] = self._profiles['Xobs'].copy()  # thetaobs

        # Also set Dini, Dmod, Dobs, Dopt inside ezdeprof
        n_depth = Drlt.shape[0]
        # Dini: initial estimate (Dstk or zeros)
        ez['Dini'] = self._profiles['Dstk'][:, :, 0].astype(np.float32)
        # Dmod: model (Dtrn if present, otherwise Drlt)
        ez['Dmod'] = self._profiles['Dtrn'][:, :, 0].astype(np.float32)
        # Dobs: observation (Dtrn in simulation, Drlt in experiment)
        ez['Dobs'] = self._profiles['Dtrn'][:, :, 0].astype(np.float32)
        # Dopt: optimization result
        ez['Dopt'] = Drlt[:, :, 0].astype(np.float32)

        # Xini, Xmod: thetaopt size (optimization / display)
        ez['Xini'] = self._profiles['Xopt'][:, :, 0]
        ez['Xmod'] = self._profiles['Xopt'][:, :, 0]
        # Xobs: thetaobs size (observed data)
        ez['Xobs'] = self._profiles['Xobs'][:, :, 0]
        # Xopt: thetaopt size (optimization result)
        ez['Xopt'] = self._profiles['Xopt'][:, :, 0]

        # Update the stack count
        ez['stack'] = n_stack
        # calculationinitialized: [loop_initialized, calc_done, stack_count]
        # 4-component reference H5 written by MATLAB: [2, 1, 1]
        ez['calculationinitialized'] = np.array([2, 1, n_stack], dtype=np.uint64)

        # Recompute the lambda values with Dini (MATLAB-compatible harmonic mean)
        Dini = ez.get('Dini', None)
        if Dini is not None and Dini.size > 0:
            self.compute_lambda_values(Dini.T)  # (n_elem, n_depth) → (n_depth, n_elem)
            # Compute TransMatOpt and Xini
            self.compute_transfer_matrix_and_xini()

        # Initial display values from Dtrn (for display before Set)
        self._compute_initial_display_values()

    def _compute_initial_display_values(self):
        """
        Compute the initial display values from Dtrn (the true profile)

        Sets the following for the GUI display before Set:
        - Conc_Set/Conc_Calc: mean concentration of each element [%]
        - Sprs_Set/Sprs_Calc: sparsity [number of non-zero depths, number of depths]
        - Xobs: angle intensities computed from the true profile
        - noise: default 1.0
        """
        ez = self._ezdeprof

        # Dtrn (n_depth, n_elem) - Python internal form
        Dtrn = self._profiles.get('Dtrn', None)
        if Dtrn is None or Dtrn.size == 0:
            return

        Dtrn_2d = Dtrn[:, :, 0] if Dtrn.ndim == 3 else Dtrn  # (n_depth, n_elem)
        n_depth, n_elem = Dtrn_2d.shape

        # Conc: mean concentration of each element × 100 [%]
        # MATLAB computes stepsize * sum(D) (the integral)
        stepsize = float(ez.get('stepsize', 30.0))
        conc = stepsize * np.sum(Dtrn_2d, axis=0)  # (n_elem,)
        ez['Conc_Set'] = conc.astype(np.float32)
        ez['Conc_Calc'] = conc.astype(np.float32)

        # Sparsity: [number of non-zero depths, number of depths]
        # Count the film region (depths where something other than the substrate exists)
        subst = np.atleast_1d(ez.get('subst', np.array([1]))).flatten()
        antisubst_mask = np.ones(n_elem, dtype=bool)
        for s in subst:
            if 1 <= s <= n_elem:
                antisubst_mask[s - 1] = False

        # Count the depths where the film components sum to at least the threshold
        film_sum = np.sum(Dtrn_2d[:, antisubst_mask], axis=1)  # (n_depth,)
        nonzero_depth = np.sum(film_sum > 0.01)  # 1 % or more counts as non-zero
        ez['Sprs_Set'] = np.array([nonzero_depth, n_depth], dtype=np.float32)
        ez['Sprs_Calc'] = np.array([nonzero_depth, n_depth], dtype=np.float32)

        # noise: default 1.0 (if not set)
        if ez.get('noise', 0) == 0:
            ez['noise'] = 1.0

        # Xobs: angle intensities computed from Dtrn (uses TransMatObs if present)
        TransMatObs = ez.get('TransMatObs', None)
        if TransMatObs is not None and TransMatObs.size > 0:
            # TransMatObs: (n_angles, n_depth, n_elem) Python internal form
            n_angles = TransMatObs.shape[0]
            I = np.zeros((n_angles, n_elem), dtype=np.float32)
            for e in range(n_elem):
                I[:, e] = TransMatObs[:, :, e] @ Dtrn_2d[:, e]

            # Normalize
            I_sum = np.sum(I, axis=1, keepdims=True)
            I_sum = np.where(I_sum > 1e-10, I_sum, 1.0)
            Xobs_calc = I / I_sum

            ez['Xobs'] = Xobs_calc.astype(np.float32)
            # Update _profiles as well
            self._profiles['Xobs'][:, :, 0] = Xobs_calc

    def compute_transfer_matrix_and_xini(self):
        """
        Compute the transfer matrices TransMatObs/TransMatOpt and the initial angle profile Xini

        Computes the Beer-Lambert transfer matrices from lambdaobs and Dini,
        and sets Xini = normalize(TransMatOpt × Dini).

        Python internal form:
        - Dini: (n_depth, n_elem)
        - lambdaobs: (1, n_elem, n_depth)

        After the calculation (transposed on write):
        - TransMatObs: (n_angles_obs, n_depth, n_elem) → H5: (n_elem, n_depth, n_angles_obs)
        - TransMatOpt: (n_angles_opt, n_depth, n_elem) → H5: (n_elem, n_depth, n_angles_opt)
        - Xini: (n_angles_opt, n_elem) → H5: (n_elem, n_angles_opt)
        """
        ez = self._ezdeprof

        # Required parameters
        lambdaobs = ez.get('lambdaobs', None)  # (1, n_elem, n_depth)
        if lambdaobs is None or lambdaobs.size == 0:
            return

        Dini = ez.get('Dini', None)  # Python internal: (n_depth, n_elem)
        if Dini is None or Dini.size == 0:
            return

        stepsize = float(ez.get('stepsize', 30.0))
        thetaobs = np.atleast_1d(ez.get('thetaobs', np.array([0]))).flatten()
        thetaopt = np.atleast_1d(ez.get('thetaopt', np.array([0]))).flatten()
        n_angles_obs = len(thetaobs)
        n_angles_opt = len(thetaopt)
        n_depth, n_elem = Dini.shape

        # lambdaobs: (1, n_elem, n_depth) → (n_depth, n_elem)
        lambda_obs = lambdaobs[0].T  # (n_depth, n_elem)

        def compute_transmat(theta_arr):
            """Transfer matrix from the Beer-Lambert law"""
            n_angles = len(theta_arr)
            TransMat = np.zeros((n_angles, n_depth, n_elem), dtype=np.float32)
            cos_theta = np.cos(np.radians(theta_arr))

            for a in range(n_angles):
                if abs(cos_theta[a]) > 1e-6:
                    for e in range(n_elem):
                        t1 = -stepsize / (lambda_obs[:, e] * cos_theta[a])
                        exp_t1 = np.exp(t1)
                        cumprod_exp = np.cumprod(exp_t1)
                        TransMat[a, :, e] = cumprod_exp / exp_t1
            return TransMat

        # TransMatObs (observation angles)
        TransMatObs = compute_transmat(thetaobs)
        ez['TransMatObs'] = TransMatObs.astype(np.float32)

        # TransMatOpt (optimization angles)
        TransMatOpt = compute_transmat(thetaopt)
        ez['TransMatOpt'] = TransMatOpt.astype(np.float32)

        # Xini = normalize(TransMatOpt × Dini)
        # I[angle, elem] = sum_depth(TransMat[angle, depth, elem] * D[depth, elem])
        I = np.zeros((n_angles_opt, n_elem), dtype=np.float32)
        for e in range(n_elem):
            for a in range(n_angles_opt):
                I[a, e] = np.sum(TransMatOpt[a, :, e] * Dini[:, e])

        # Normalize: X = I / sum(I, axis=elem)
        I_sum = np.sum(I, axis=1, keepdims=True)  # (n_angles_opt, 1)
        I_sum = np.where(I_sum > 1e-10, I_sum, 1.0)
        X = I / I_sum  # (n_angles_opt, n_elem)

        # Internal form: (n_angles_opt, n_elem) - transposed on write
        ez['Xini'] = X.astype(np.float32)
        ez['Xmod'] = X.astype(np.float32)
        ez['Xopt'] = X.astype(np.float32)

    def set_objectives(
        self,
        Qrlt: np.ndarray,
        Qtrn: Optional[np.ndarray] = None,
        Qstk: Optional[np.ndarray] = None,
    ):
        """
        Set the objective-function values

        Args:
            Qrlt: objective of the result (n_stack, 3) [chi2, regularization, objective]
            Qtrn: objective of the truth
            Qstk: initial objective
        """
        if Qrlt.ndim == 1:
            Qrlt = Qrlt.reshape(1, -1)
        self._objectives['Qrlt'] = Qrlt.astype(np.float64)

        if Qtrn is not None:
            if Qtrn.ndim == 1:
                Qtrn = Qtrn.reshape(1, -1)
            self._objectives['Qtrn'] = Qtrn.astype(np.float64)
        else:
            self._objectives['Qtrn'] = np.zeros_like(self._objectives['Qrlt'])

        if Qstk is not None:
            if Qstk.ndim == 1:
                Qstk = Qstk.reshape(1, -1)
            self._objectives['Qstk'] = Qstk.astype(np.float64)
        else:
            self._objectives['Qstk'] = np.zeros_like(self._objectives['Qrlt'])

        # Also set Qini, Qopt inside ezdeprof
        ez = self._ezdeprof
        ez['Qini'] = self._objectives['Qstk'][0].astype(np.float32)
        ez['Qopt'] = self._objectives['Qrlt'][0].astype(np.float32)

    def write(self):
        """
        Write the H5 file
        """
        ez = self._ezdeprof

        # Check the required parameters
        if 'depth' not in ez:
            raise ValueError("depth is required. Call set_basic_params() first.")
        if 'Drlt' not in self._profiles:
            raise ValueError("Drlt is required. Call set_profiles() first.")

        # Derived parameters
        self._compute_derived_params()

        with h5py.File(self.filepath, 'w') as f:
            # Write the ezdeprof group
            self._write_ezdeprof(f)

            # Write the root datasets
            self._write_root_datasets(f)

        print(f"Saved GUI-compatible H5 file: {self.filepath}")
        n_depth, n_elements, n_stack = self._profiles['Drlt'].shape
        print(f"  depth: {n_depth}, elements: {n_elements}, stack: {n_stack}")

    def _compute_derived_params(self):
        """
        Compute the derived parameters
        """
        ez = self._ezdeprof
        n_elements = len(ez['elementsymbol'])
        n_depth = len(ez['depth'])
        n_angles_obs = len(ez['thetaobs'])
        n_angles_opt = len(ez['thetaopt'])
        n_photon = 1

        # freeconst = sort([free, const]) (used by FV0stk/FeatureVector_Source)
        # Reference: in the 4-component H5 written by MATLAB, FV0stk=8 = 2 + 3*2
        free = ez.get('free', np.array([]))
        const = ez.get('const', np.array([0]))
        linked = ez.get('linked', np.array([0]))

        # Keep only the valid indices (0 means none)
        free_valid = [x for x in np.atleast_1d(free) if x > 0]
        const_valid = [x for x in np.atleast_1d(const) if x > 0]
        linked_valid = [x for x in np.atleast_1d(linked) if x > 0]

        # freeconst: the elements used by FV0stk/FeatureVector
        freeconst = sorted(free_valid + const_valid)
        n_freeconst = len(freeconst)

        # Size of FV0stk: 2 + 3 * len(freeconst)
        # (thck, abrp) + 3 × (position, width, height per element)
        fv_len = 2 + 3 * n_freeconst

        # SVrlt/SVstk have length 3 × (number of subst + number of linked)
        # MATLAB reference: SVrlt shape=(6,1) = 3 × 2 (subst=1, linked=1)
        subst = ez.get('subst', np.array([1]))
        n_subst = len(np.atleast_1d(subst))
        n_linked = len(linked_valid)
        sv_len = 3 * (n_subst + n_linked)  # 3 × (number of subst + number of linked)

        if 'FeatureVector_Source' not in ez:
            ez['FeatureVector_Source'] = np.zeros(fv_len, dtype=np.float32)
        if 'FeatureVector_Set' not in ez:
            ez['FeatureVector_Set'] = np.zeros(fv_len, dtype=np.float32)
        if 'FeatureVector_Calc' not in ez:
            ez['FeatureVector_Calc'] = np.zeros(fv_len, dtype=np.float32)
        if 'SupportVector_Set' not in ez:
            ez['SupportVector_Set'] = np.zeros(sv_len, dtype=np.float32)
        if 'SupportVector_Calc' not in ez:
            ez['SupportVector_Calc'] = np.zeros(sv_len, dtype=np.float32)
        if 'Conc_Set' not in ez:
            ez['Conc_Set'] = np.zeros(n_elements, dtype=np.float32)
        if 'Conc_Calc' not in ez:
            ez['Conc_Calc'] = np.zeros(n_elements, dtype=np.float32)
        if 'Sprs_Set' not in ez:
            ez['Sprs_Set'] = np.zeros(2, dtype=np.float32)
        if 'Sprs_Calc' not in ez:
            ez['Sprs_Calc'] = np.zeros(2, dtype=np.float32)

        # Transfer matrices (zeros if not set)
        if 'TransMatObs' not in ez:
            ez['TransMatObs'] = np.zeros((n_angles_obs, n_depth, n_elements), dtype=np.float32)
            ez['TransMatOpt'] = np.zeros((n_angles_opt, n_depth, n_elements), dtype=np.float32)
        if 'lambdabasic' not in ez:
            # lambdabasic: (n_elements, n_elements, 1) → transposed to (1, n_elements, n_elements)
            ez['lambdabasic'] = np.zeros((n_elements, n_elements, n_photon), dtype=np.float32)
        if 'lambdaobs' not in ez:
            # lambdaobs/opt: (1, n_elements, n_depth) depth-wise IMFP
            ez['lambdaobs'] = np.zeros((1, n_elements, n_depth), dtype=np.float32)
            ez['lambdaopt'] = np.zeros((1, n_elements, n_depth), dtype=np.float32)
        if 'lambdacosinethetaobs' not in ez:
            ez['lambdacosinethetaobs'] = np.zeros((n_angles_obs, n_elements), dtype=np.float32)
            ez['lambdacosinethetaopt'] = np.zeros((n_angles_opt, n_elements), dtype=np.float32)

        # Objective functions
        if 'Qini' not in ez:
            ez['Qini'] = np.zeros(3, dtype=np.float32)
        if 'Qopt' not in ez:
            ez['Qopt'] = np.zeros(3, dtype=np.float32)

    # =========================================================================
    # MATLAB-compatible write helpers
    # =========================================================================

    def _write_scalar(self, f: h5py.File, path: str, value, dtype):
        """
        Write a scalar in MATLAB h5write-compatible (1,) form
        """
        f.create_dataset(path, data=np.array([value], dtype=dtype))

    def _write_vlen_string_column(self, f: h5py.File, path: str, strings: List[str]):
        """
        Write a string array as MATLAB cellstr-compatible vlen-string (N, 1)
        """
        # Build an object array of shape (N, 1)
        data = np.array([[s] for s in strings], dtype=object)
        f.create_dataset(path, data=data, dtype=self.VLEN_STR_DTYPE)

    def _write_column_vector(self, f: h5py.File, path: str, arr: np.ndarray, dtype=None):
        """
        Write a 1D array as a MATLAB-compatible (N, 1) column vector
        """
        arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        f.create_dataset(path, data=arr.reshape(-1, 1))

    def _write_row_vector(self, f: h5py.File, path: str, arr: np.ndarray, dtype=None):
        """
        Write a 1D array as a MATLAB-compatible (1, N) row vector
        """
        arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        f.create_dataset(path, data=arr.reshape(1, -1))

    # =========================================================================
    # Writing the ezdeprof group
    # =========================================================================

    def _write_ezdeprof(self, f: h5py.File):
        """
        Write the /ezdeprof/ group (fully compatible with MATLAB h5write)
        """
        ez = self._ezdeprof

        # -----------------------------------------------------------------
        # Scalars (MATLAB: shape (1,))
        # -----------------------------------------------------------------

        # uint8 scalars
        for key in ['SimulationOrExperiment', 'NP_Set', 'NP_Calc', 'NP_Max',
                    'crossecapproximord', 'crossecapproximview', 'TPPver',
                    'optimizationmethodvalue', 'initialguessvalue', 'priormodelvalue',
                    'transfermatrixupdatevalue', 'photoionizationcrosssection',
                    'verificationschemevalue', 'verificationmethod',
                    'degreecontrol', 'alphacontrol', 'verificationstatus']:
            val = ez.get(key, 0)
            self._write_scalar(f, f'/ezdeprof/{key}', val, np.uint8)

        # uint16 scalars
        for key in ['m', 'loop', 'currentloop', 'Kfold', 'NumberofNearestNeighbor']:
            val = ez.get(key, 0)
            self._write_scalar(f, f'/ezdeprof/{key}', val, np.uint16)

        # uint64 scalars
        for key in ['iterationinterval', 'numberofmodeling', 'stack', 'currentstack',
                    'holdout', 'iterationtotal']:
            val = ez.get(key, 0)
            self._write_scalar(f, f'/ezdeprof/{key}', val, np.uint64)

        # float32 scalars
        for key in ['workfunction', 'gammaeff', 'angle', 'photon', 'deltaphoton',
                    'noise', 'slice', 'totaldepth', 'stepsize', 'iterationtime']:
            val = ez.get(key, 0.0)
            self._write_scalar(f, f'/ezdeprof/{key}', val, np.float32)

        # -----------------------------------------------------------------
        # String arrays (MATLAB cellstr: vlen-string, shape (N, 1))
        # -----------------------------------------------------------------
        for key in ['elementsymbol', 'orbital', 'stringcharge', 'element', 'prpties', 'compound']:
            arr = ez.get(key, [''])
            self._write_vlen_string_column(f, f'/ezdeprof/{key}', arr)

        # -----------------------------------------------------------------
        # 1D arrays - column vectors (N, 1)
        # -----------------------------------------------------------------

        # uint8 column vectors
        for key in ['subst', 'free', 'const', 'linked', 'antisubst']:
            arr = ez.get(key, [0])
            self._write_column_vector(f, f'/ezdeprof/{key}', arr, np.uint8)

        for key in ['elementnumber', 'orbitalnumber']:
            arr = ez.get(key, [0])
            self._write_column_vector(f, f'/ezdeprof/{key}', arr, np.uint8)

        # uint16 column vectors
        for key in ['dxdydt', 'datapoints']:
            arr = ez.get(key, [1, 1, 1])
            self._write_column_vector(f, f'/ezdeprof/{key}', arr, np.uint16)

        # uint64 column vectors
        for key in ['calculationinitialized']:
            arr = ez.get(key, [0, 1, 1])
            self._write_column_vector(f, f'/ezdeprof/{key}', arr, np.uint64)

        # float32 column vectors - per-element parameters
        for key in ['charge', 'linkedconst', 'photonenergy', 'bindingenergy',
                    'valencecharge', 'density', 'molecularweight', 'bandgap']:
            arr = ez.get(key, [0.0])
            self._write_column_vector(f, f'/ezdeprof/{key}', arr, np.float32)

        # float32 column vectors - thckini and the like (3, 1)
        for key in ['thckini', 'abrpini', 'thckmod', 'abrpmod']:
            arr = ez.get(key, [0.0, 0.0, 0.0])
            arr = np.array(arr, dtype=np.float32)[:3]
            self._write_column_vector(f, f'/ezdeprof/{key}', arr, np.float32)

        # float32 column vectors - Qini, Qopt (3, 1)
        for key in ['Qini', 'Qopt']:
            arr = ez.get(key, [0.0, 0.0, 0.0])
            arr = np.array(arr, dtype=np.float32)[:3]
            self._write_column_vector(f, f'/ezdeprof/{key}', arr, np.float32)

        # float32 column vectors - Sprs (2, 1)
        for key in ['Sprs_Set', 'Sprs_Calc']:
            arr = ez.get(key, [0.0, 0.0])
            arr = np.array(arr, dtype=np.float32)[:2]
            self._write_column_vector(f, f'/ezdeprof/{key}', arr, np.float32)

        # float32 column vectors - feature vectors
        for key in ['FeatureVector_Source', 'FeatureVector_Set', 'FeatureVector_Calc',
                    'SupportVector_Set', 'SupportVector_Calc', 'Conc_Set', 'Conc_Calc']:
            arr = ez.get(key, [0.0])
            self._write_column_vector(f, f'/ezdeprof/{key}', arr, np.float32)

        # -----------------------------------------------------------------
        # 1D arrays - row vectors (1, N)
        # -----------------------------------------------------------------

        # float32 row vectors - depth, theta
        for key in ['depth', 'thetaobs', 'thetaopt']:
            arr = ez.get(key, [0.0])
            self._write_row_vector(f, f'/ezdeprof/{key}', arr, np.float32)

        # times: empty array (1, 0)
        times = ez.get('times', np.zeros((1, 0), dtype=np.float32))
        f.create_dataset('/ezdeprof/times', data=times.astype(np.float32))

        # -----------------------------------------------------------------
        # 2D arrays
        # -----------------------------------------------------------------

        # Col: (3, n_elements) = RGB x elements (Python internal is (n_elements, 3))
        arr = ez.get('Col', np.zeros((3, 1)))
        f.create_dataset('/ezdeprof/Col', data=np.array(arr, dtype=np.float32).T)

        # subsini/subsmod: (3, n_subst)
        for key in ['subsini', 'subsmod']:
            arr = ez.get(key, np.array([[1.0], [0.0], [1.0]]))
            f.create_dataset(f'/ezdeprof/{key}', data=np.array(arr, dtype=np.float32))

        # Dini and the like: (elements, depth) - Python internal is (depth, elements), so transposed
        for key in ['Dini', 'Dmod', 'Dobs', 'Dopt']:
            arr = ez.get(key, np.zeros((1, 1)))
            arr = np.array(arr, dtype=np.float32)
            arr_t = arr.T  # (depth, elements) → (elements, depth)
            f.create_dataset(f'/ezdeprof/{key}', data=arr_t,
                           maxshape=(None, None), chunks=arr_t.shape)

        # Xini and the like: (elements, angles) - Python internal is (angles, elements), so transposed
        for key in ['Xini', 'Xmod', 'Xobs', 'Xopt']:
            arr = ez.get(key, np.zeros((1, 1)))
            arr = np.array(arr, dtype=np.float32)
            arr_t = arr.T  # (angles, elements) → (elements, angles)
            f.create_dataset(f'/ezdeprof/{key}', data=arr_t,
                           maxshape=(None, None), chunks=arr_t.shape)

        # lambdacosinetheta: (n_elements, n_angles)
        for key in ['lambdacosinethetaobs', 'lambdacosinethetaopt']:
            arr = ez.get(key, np.zeros((1, 1)))
            arr = np.array(arr, dtype=np.float32)
            f.create_dataset(f'/ezdeprof/{key}', data=arr.T)

        # -----------------------------------------------------------------
        # 3D arrays - transfer matrices
        # -----------------------------------------------------------------

        # TransMat: (n_elements, n_depth, n_angles)
        # Python internal: (n_angles, n_depth, n_elements)
        for key in ['TransMatObs', 'TransMatOpt']:
            arr = ez.get(key, np.zeros((1, 1, 1)))
            arr = np.array(arr, dtype=np.float32)
            arr_t = np.transpose(arr, (2, 1, 0))  # → (n_elements, n_depth, n_angles)
            f.create_dataset(f'/ezdeprof/{key}', data=arr_t)

        # lambdabasic: (1, n_elements, n_elements)
        arr = ez.get('lambdabasic', np.zeros((1, 1, 1)))
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.transpose(arr, (2, 0, 1))  # (e, e, 1) → (1, e, e)
        f.create_dataset('/ezdeprof/lambdabasic', data=arr)

        # lambdaobs/opt: (1, n_elements, n_depth)
        for key in ['lambdaobs', 'lambdaopt']:
            arr = ez.get(key, np.zeros((1, 1, 1)))
            arr = np.array(arr, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[2] == 1 and arr.shape[0] == arr.shape[1]:
                arr = np.transpose(arr, (2, 0, 1))
            f.create_dataset(f'/ezdeprof/{key}', data=arr)

    # =========================================================================
    # Writing the root datasets
    # =========================================================================

    def _write_root_datasets(self, f: h5py.File):
        """
        Write the root-level datasets (fully compatible with MATLAB h5write)

        Note: MATLAB is column-major and HDF5 row-major, so transposition is needed
        Python: (depth, elements, stack) → H5: (stack, elements, depth)
        → MATLAB: (depth, elements, stack)

        Every stack-related dataset is made extensible (maxshape)
        """
        max_stack = None  # unlimited

        # Depth profiles
        # Python: (depth, elements, stack) → transpose(2,1,0) → H5: (stack, elements, depth)
        for key in ['Drlt', 'Dtrn', 'Dstk']:
            arr = self._profiles.get(key)
            if arr is not None:
                arr_h5 = np.transpose(arr, (2, 1, 0)).astype(np.float32)
                n_s, n_e, n_d = arr_h5.shape
                f.create_dataset(f'/{key}', data=arr_h5,
                               maxshape=(max_stack, n_e, n_d), chunks=(1, n_e, n_d))

        # Intensities
        # Python: (angles, elements, stack) → (stack, elements, angles)
        for key in ['Xrlt', 'Xstk']:
            arr = self._profiles.get(key)
            if arr is not None:
                arr_h5 = np.transpose(arr, (2, 1, 0)).astype(np.float32)
                n_s, n_e, n_a = arr_h5.shape
                f.create_dataset(f'/{key}', data=arr_h5,
                               maxshape=(max_stack, n_e, None), chunks=(1, n_e, n_a))

        # Objective functions
        # Python: (stack, 3) → (3, stack) → MATLAB: (stack, 3)
        for key in ['Qrlt', 'Qtrn', 'Qstk']:
            arr = self._objectives.get(key)
            if arr is not None:
                arr_h5 = arr.T.astype(np.float64)  # (stack, 3) → (3, stack)
                f.create_dataset(f'/{key}', data=arr_h5,
                               maxshape=(3, max_stack), chunks=(3, 1))

        # Feature vectors and the like (2D: features, stack) - column-vector form
        ez = self._ezdeprof
        n_stack = self._profiles['Drlt'].shape[2]
        n_elements = self._profiles['Drlt'].shape[1]

        # freeconst (FV0stk is based on freeconst)
        # Reference: in the 4-component H5 written by MATLAB, FV0stk=8 = 2 + 3*2 (freeconst has 2 elements)
        free = ez.get('free', np.array([]))
        const = ez.get('const', np.array([0]))
        linked = ez.get('linked', np.array([0]))

        free_valid = [x for x in np.atleast_1d(free) if x > 0]
        const_valid = [x for x in np.atleast_1d(const) if x > 0]
        linked_valid = [x for x in np.atleast_1d(linked) if x > 0]

        # freeconst: used by FV0stk
        freeconst = sorted(free_valid + const_valid)
        n_freeconst = len(freeconst)

        # Size of FV0stk: 2 + 3 * len(freeconst)
        fv_len_calc = 2 + 3 * n_freeconst
        if fv_len_calc < 2:
            fv_len_calc = 8  # default

        # Length of SVrlt: 3 × (number of subst + number of linked)
        subst = ez.get('subst', np.array([1]))
        n_subst = len(np.atleast_1d(subst))
        n_linked = len(linked_valid)
        sv_len_calc = 3 * (n_subst + n_linked)
        if sv_len_calc == 0:
            sv_len_calc = 3  # at least 3

        # Initial values of FV, SV, Conc, SPS
        fv_source = np.array(ez.get('FeatureVector_Source', np.zeros(fv_len_calc)), dtype=np.float32)
        # Extend fv_source if it is too short
        if len(fv_source) < fv_len_calc:
            fv_source = np.pad(fv_source, (0, fv_len_calc - len(fv_source)), constant_values=0.0)
        sv_set = np.array(ez.get('SupportVector_Set', np.zeros(sv_len_calc)), dtype=np.float32)
        # Extend sv_set if it is too short
        if len(sv_set) < sv_len_calc:
            sv_set = np.pad(sv_set, (0, sv_len_calc - len(sv_set)), constant_values=100.0)
        conc_set = np.array(ez.get('Conc_Set', np.zeros(n_elements)), dtype=np.float32)
        sprs_set = np.array(ez.get('Sprs_Set', np.zeros(2)), dtype=np.float32)

        # FV family: (features, stack) form
        fv_init = np.tile(fv_source.reshape(-1, 1), (1, n_stack))
        fv_len = fv_init.shape[0]
        f.create_dataset('/FV0stk', data=fv_init, maxshape=(fv_len, max_stack), chunks=(fv_len, 1))
        f.create_dataset('/NSstk', data=np.zeros((1, n_stack), dtype=np.float32), maxshape=(1, max_stack), chunks=(1, 1))
        f.create_dataset('/FVstk', data=fv_init.copy(), maxshape=(fv_len, max_stack), chunks=(fv_len, 1))
        f.create_dataset('/FVrlt', data=fv_init.copy(), maxshape=(fv_len, max_stack), chunks=(fv_len, 1))
        f.create_dataset('/FVtrn', data=fv_init.copy(), maxshape=(fv_len, max_stack), chunks=(fv_len, 1))

        # SV family: (features, stack) form
        sv_init = np.tile(sv_set.reshape(-1, 1), (1, n_stack))
        sv_len = sv_init.shape[0]
        f.create_dataset('/SVstk', data=sv_init, maxshape=(sv_len, max_stack), chunks=(sv_len, 1))
        f.create_dataset('/SVrlt', data=sv_init.copy(), maxshape=(sv_len, max_stack), chunks=(sv_len, 1))
        f.create_dataset('/SVtrn', data=sv_init.copy(), maxshape=(sv_len, max_stack), chunks=(sv_len, 1))

        # Conc family: (n_elements, stack) form
        conc_init = np.tile(conc_set.reshape(-1, 1), (1, n_stack))
        conc_len = conc_init.shape[0]
        f.create_dataset('/Cstk', data=conc_init, maxshape=(conc_len, max_stack), chunks=(conc_len, 1))
        f.create_dataset('/Crlt', data=conc_init.copy(), maxshape=(conc_len, max_stack), chunks=(conc_len, 1))
        f.create_dataset('/Ctrn', data=conc_init.copy(), maxshape=(conc_len, max_stack), chunks=(conc_len, 1))

        # SPS family: (2, stack) form
        sps_init = np.tile(sprs_set.reshape(-1, 1), (1, n_stack))
        sps_len = sps_init.shape[0]
        f.create_dataset('/SPSstk', data=sps_init, maxshape=(sps_len, max_stack), chunks=(sps_len, 1))
        f.create_dataset('/SPSrlt', data=sps_init.copy(), maxshape=(sps_len, max_stack), chunks=(sps_len, 1))
        f.create_dataset('/SPStrn', data=sps_init.copy(), maxshape=(sps_len, max_stack), chunks=(sps_len, 1))

        # degree and alpha: (3, stack)
        f.create_dataset('/degree', data=np.ones((3, n_stack), dtype=np.float64), maxshape=(3, max_stack), chunks=(3, 1))
        f.create_dataset('/alpha', data=np.ones((3, n_stack), dtype=np.float64), maxshape=(3, max_stack), chunks=(3, 1))

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _estimate_orbitals(elements: List[str]) -> List[str]:
        """
        Estimate the orbital from the element
        """
        orbital_map = {
            'C': '1s', 'N': '1s', 'O': '1s', 'F': '1s',
            'Si': '2p', 'P': '2p', 'S': '2p', 'Cl': '2p',
            'Al': '2p', 'Ti': '2p', 'Fe': '2p', 'Ni': '2p',
            'Cu': '2p', 'Zn': '2p', 'Ga': '2p', 'Ge': '3d',
            'Hf': '4f',
        }
        return [orbital_map.get(e, '2p') for e in elements]

    @staticmethod
    def _estimate_binding_energies(elements: List[str], orbitals: List[str]) -> np.ndarray:
        """
        Estimate the binding energy from the element and the orbital

        Uses data equivalent to MATLAB's BindingEnergyTable.
        HXPS (8 keV) mainly uses the 1s orbital.
        """
        # Data equivalent to MATLAB's BindingEnergyTable
        # (element, orbital) → binding energy [eV]
        be_map = {
            # Light elements - 1s
            ('C', '1s'): 284.4,
            ('N', '1s'): 399.8,
            ('O', '1s'): 531.0,
            ('F', '1s'): 684.9,

            # Si
            ('Si', '1s'): 1839.1,
            ('Si', '2s'): 150.0,
            ('Si', '2p'): 99.0,

            # Al
            ('Al', '1s'): 1561.8,
            ('Al', '2s'): 118.0,
            ('Al', '2p'): 74.0,

            # Ti
            ('Ti', '1s'): 4968.7,
            ('Ti', '2s'): 560.0,
            ('Ti', '2p'): 459.0,
            ('Ti', '3p'): 34.0,

            # Fe
            ('Fe', '1s'): 7114.0,
            ('Fe', '2p'): 711.0,
            ('Fe', '3p'): 55.0,

            # Hf
            ('Hf', '1s'): 65351.0,
            ('Hf', '4f'): 18.0,
            ('Hf', '4d'): 214.0,

            # Other common elements
            ('Na', '1s'): 1072.0,
            ('Mg', '1s'): 1305.0,
            ('P', '1s'): 2149.0,
            ('S', '1s'): 2472.0,
            ('Cl', '1s'): 2823.0,
            ('K', '1s'): 3610.0,
            ('Ca', '1s'): 4041.0,
            ('Cr', '1s'): 5989.0,
            ('Mn', '1s'): 6541.0,
            ('Co', '1s'): 7711.0,
            ('Ni', '1s'): 8335.0,
            ('Cu', '1s'): 8984.0,
            ('Zn', '1s'): 9663.0,
            ('Ga', '1s'): 10372.0,
            ('Ge', '1s'): 11104.0,
            ('As', '1s'): 11864.0,
            ('Zr', '1s'): 18000.0,
            ('Nb', '1s'): 18989.0,
            ('Mo', '1s'): 20004.0,
            ('Ta', '1s'): 67419.0,
            ('W', '1s'): 69525.0,
        }
        return np.array([be_map.get((e, o), 100.0) for e, o in zip(elements, orbitals)])


def create_gui_compatible_h5(
    filepath: Union[str, Path],
    depth: np.ndarray,
    angles: np.ndarray,
    elements: List[str],
    Drlt: np.ndarray,
    Dtrn: Optional[np.ndarray] = None,
    Dstk: Optional[np.ndarray] = None,
    Qrlt: Optional[np.ndarray] = None,
    TransMat: Optional[np.ndarray] = None,
    orbitals: Optional[List[str]] = None,
    compounds: Optional[List[str]] = None,
    binding_energies: Optional[np.ndarray] = None,
    photon_energy: float = 1486.6,
    simulation: bool = True,
    # Element property settings
    subst: Optional[List[int]] = None,
    free: Optional[List[int]] = None,
    linked: Optional[List[int]] = None,
    prpties: Optional[List[str]] = None,
    charges: Optional[List[float]] = None,
) -> Path:
    """
    Create a GUI-compatible H5 file in one call

    Args:
        filepath: output path
        depth: depth axis [Å]
        angles: measurement angles [°]
        elements: element symbols ['Si', 'Si', 'O', 'Al']
        Drlt: result profile (n_depth, n_elements) or (n_depth, n_elements, n_stack)
        Dtrn: true profile
        Dstk: initial profile
        Qrlt: objective function (n_stack, 3) or (3,)
        TransMat: transfer matrix
        orbitals: orbitals ['1s', '1s', '1s', '2p']
        compounds: compound names ['Si', 'SiO2', 'Oxide', 'Al2O3']
        subst: substrate element indices (1-based), default=[1]
        free: free element indices (1-based)
        linked: linked element indices (1-based, 0 = none)
        binding_energies: binding energies
        photon_energy: photon energy
        simulation: simulation mode
        charges: charge of each element [0, 4, -2, 4] (estimated from the compounds when omitted)

    Returns:
        Output file path
    """
    writer = H5Writer(filepath)

    writer.set_basic_params(
        depth=depth,
        angles=angles,
        elements=elements,
        orbitals=orbitals,
        compounds=compounds,
        binding_energies=binding_energies,
        photon_energy=photon_energy,
        simulation=simulation,
        subst=subst,
        free=free,
        linked=linked,
        prpties=prpties,
        charges=charges,
    )

    if TransMat is not None:
        writer.set_transfer_matrix(TransMat)

    writer.set_profiles(Drlt=Drlt, Dtrn=Dtrn, Dstk=Dstk)

    if Qrlt is not None:
        writer.set_objectives(Qrlt=Qrlt)
    else:
        # Default objective function
        n_stack = Drlt.shape[2] if Drlt.ndim == 3 else 1
        writer.set_objectives(Qrlt=np.zeros((n_stack, 3)))

    writer.write()

    return Path(filepath)
