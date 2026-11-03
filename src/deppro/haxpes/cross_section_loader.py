#!/usr/bin/env python3
"""
Photoionization Cross Section Database Loader.

Loads and compares cross sections from multiple databases:
- Scofield (1973): Wide energy range (1 keV - 1.5 MeV), tabulated values
- Yeh-Lindau (1985): Standard XPS range (10 eV - 8 keV), tabulated values
- Trzhaskovskaya (2018): Fitting parameters, primarily outer shells

The databases cover different energy ranges and can disagree; above the
range of a table, its values are extrapolated. Record which source a result
used.

Usage:
    from deppro.haxpes.cross_section_loader import CrossSectionDatabase

    db = CrossSectionDatabase()

    # Get cross section at specific energy
    sigma = db.get_cross_section('Si', '1s', 9251.7, source='scofield')

    # Compare databases
    comparison = db.compare_at_energy(['C', 'O', 'Si', 'Ti'], '1s', 9251.7)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Tuple, Union
from scipy.interpolate import interp1d
import warnings

from deppro.paths import env_value


# =============================================================================
# Configuration
# =============================================================================

ENV_CROSS_SECTION_DIR = "DEPPRO_CROSS_SECTION_DIR"
SCOFIELD_WORKBOOK = "excel_scofield_1973_pics.xlsx"
YEH_LINDAU_WORKBOOK = "excel_yeh_lindau_1985_pics-4.xlsx"


def cross_section_dir() -> Optional[Path]:
    """Directory holding the Scofield and Yeh–Lindau workbooks, or ``None``.

    ``DEPPRO_CROSS_SECTION_DIR`` if set (``~`` is expanded); an empty value
    counts as unset. There is no default directory: no neighboring checkout
    is searched, so a source checkout, an installed package and CI resolve the
    same way. Resolved when called, not when imported.
    """
    value = env_value(ENV_CROSS_SECTION_DIR)
    if value:
        return Path(value).expanduser()
    return None


def _require_workbook(path: Path) -> Path:
    """Return ``path``, or raise ``FileNotFoundError`` that names the variable."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Cross-section workbook not found at {path}. Set "
            f"{ENV_CROSS_SECTION_DIR} to the directory that holds "
            f"{SCOFIELD_WORKBOOK} and {YEH_LINDAU_WORKBOOK}, or install toyomacro "
            f"(extra ``fit``), whose bundled tables are the default source."
        )
    return path


def _toyomacro_cross_section():
    """``toyomacro.data.cross_section.CrossSection`` if toyomacro is installed, else ``None``."""
    try:
        from toyomacro.data.cross_section import CrossSection
    except ImportError:
        return None
    return CrossSection


def resolve_source(db_path: Optional[Path] = None) -> Tuple[str, Optional[Path]]:
    """Where the tables come from: ``('excel', directory)`` or ``('toyomacro', None)``.

    toyomacro keeps the complete, reviewed tables (provenance and data-rights
    review in its ``docs/DATA_SOURCES.md``); deppro extracts what it needs from
    them. The Excel workbooks remain available for cross-checks. Precedence:

    1. an explicit ``db_path`` -> the workbooks in that directory;
    2. ``DEPPRO_CROSS_SECTION_DIR`` set -> the workbooks there;
    3. toyomacro installed -> its bundled Scofield and Yeh-Lindau tables;
    4. otherwise ``FileNotFoundError`` naming ``DEPPRO_CROSS_SECTION_DIR`` and
       toyomacro. No neighboring checkout is searched.

    Only the *source* differs between 2 and 3: deppro keeps its own
    interpolation. ``tests/unit/test_cross_section_sources.py`` pins the
    production values the workbooks gave and requires the toyomacro source to
    reproduce them; its docstring lists where the two sources are known to
    differ (above 30 keV, a few non-production Yeh-Lindau cells, the reported
    ``energy_range``).
    """
    if db_path is not None:
        return "excel", Path(db_path)
    if env_value(ENV_CROSS_SECTION_DIR):
        return "excel", cross_section_dir()
    if _toyomacro_cross_section() is not None:
        return "toyomacro", None
    raise FileNotFoundError(
        f"No cross-section table source: {ENV_CROSS_SECTION_DIR} is not set and "
        f"toyomacro is not installed. Install toyomacro (pip install 'deppro[fit]'), "
        f"whose bundled tables are the default source, or set {ENV_CROSS_SECTION_DIR} "
        f"to the directory that holds {SCOFIELD_WORKBOOK} and {YEH_LINDAU_WORKBOOK}."
    )


def _frame_from_toyomacro(table: str, element: str) -> pd.DataFrame:
    """One element of toyomacro's ``table`` shaped like the Excel sheets.

    Columns: ``PhotonEnergy_eV`` and one column per orbital, in Mb, ``NaN``
    where the source does not tabulate a value.
    """
    cross_section = _toyomacro_cross_section()
    if cross_section is None:  # pragma: no cover - resolve_source() never selects this backend then
        raise ImportError("toyomacro is not installed")
    # ``to_dict`` is in toyomacro's supported API layer (docs/API.md): shape
    # {"photon_energies": [...], "data": {element: {orbital: {"binding_energy",
    # "cross_sections": [...]}}}}, one entry per photon energy, None where the
    # table has no value, Scofield already converted to Mb. It returns the live
    # in-process cache, not a copy: read it, never mutate it (we copy into new
    # arrays below). Reading the raw grid and interpolating ourselves is the
    # intended use; structural changes are announced in toyomacro's CHANGELOG.
    data: Dict[str, Any] = cross_section.to_dict(table)
    orbitals = data["data"].get(element)
    if orbitals is None:
        raise ValueError(f"Element {element} not in toyomacro's {table} table")
    frame: Dict[str, np.ndarray] = {
        "PhotonEnergy_eV": np.asarray(data["photon_energies"], dtype=float),
    }
    for orbital, entry in orbitals.items():
        frame[orbital] = np.array(
            [np.nan if v is None else v for v in entry["cross_sections"]], dtype=float
        )
    return pd.DataFrame(frame)

# Element name to atomic number mapping
ELEMENT_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49,
    'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77,
    'Pt': 78, 'Au': 79, 'Pb': 82, 'Bi': 83,
}

# Orbital to column name mapping for different databases
ORBITAL_COLUMNS = {
    'scofield': {
        '1s': '1s1/2',
        '2s': '2s1/2',
        '2p': '2p3/2',  # Usually use 2p3/2 for XPS
        '2p1/2': '2p1/2',
        '2p3/2': '2p3/2',
        '3s': '3s1/2',
        '3p': '3p3/2',
        '3d': '3d5/2',
        '3d3/2': '3d3/2',
        '3d5/2': '3d5/2',
        '4s': '4s1/2',
        '4p': '4p3/2',
        '4d': '4d5/2',
        '4d3/2': '4d3/2',
        '4d5/2': '4d5/2',
        '4f': '4f7/2',
        '4f5/2': '4f5/2',
        '4f7/2': '4f7/2',
    },
    'yeh_lindau': {
        '1s': '1s',
        '2s': '2s',
        '2p': '2p',
        '3s': '3s',
        '3p': '3p',
        '3d': '3d',
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CrossSectionResult:
    """Cross section query result."""
    element: str
    orbital: str
    photon_energy_eV: float
    cross_section_Mb: float  # Megabarns (1 Mb = 10^-18 cm^2)
    source: str
    is_interpolated: bool
    is_extrapolated: bool
    energy_range: Tuple[float, float]  # Available data range


@dataclass
class DatabaseComparison:
    """Comparison of cross sections across databases."""
    elements: List[str]
    orbital: str
    photon_energy_eV: float

    # Cross sections by source
    scofield: Dict[str, float]
    yeh_lindau: Dict[str, float]
    trzhaskovskaya: Optional[Dict[str, float]]

    # Ratios
    yl_to_scofield: Dict[str, float]

    # Concentration impact (relative to Scofield)
    concentration_change_pct: Dict[str, float]


# =============================================================================
# Database Loaders
# =============================================================================

class ScofieldLoader:
    """Load Scofield (1973) cross section data."""

    def __init__(self, db_path: Optional[Path] = None):
        self.backend, self.db_path = resolve_source(db_path)
        self.file_path = self.db_path / SCOFIELD_WORKBOOK if self.db_path is not None else None
        # The Excel sheets are in barns per atom; toyomacro's table is already in Mb.
        self._to_Mb = 1e-6 if self.backend == "excel" else 1.0
        self._cache: Dict[int, pd.DataFrame] = {}
        self._sheet_names: Optional[List[str]] = None

    def _get_sheet_names(self) -> List[str]:
        """Get list of sheet names."""
        if self._sheet_names is None:
            xl = pd.ExcelFile(_require_workbook(self.file_path))
            self._sheet_names = xl.sheet_names
        return self._sheet_names

    def _get_sheet_name(self, z: int, element: str) -> str:
        """Get sheet name for element.

        Sheet naming varies: Z__6_C_, Z_10_Ne, Z_13_Al, etc.
        """
        # Search for matching sheet
        sheets = self._get_sheet_names()
        for sheet in sheets:
            # Pattern: Z_XX_El or Z__X_El
            if f"_{z}_" in sheet or f"__{z}_" in sheet:
                if element in sheet:
                    return sheet
        raise ValueError(f"Sheet not found for Z={z}, {element}")

    def load_element(self, element: str) -> pd.DataFrame:
        """Load cross section data for an element."""
        z = ELEMENT_Z.get(element)
        if z is None:
            raise ValueError(f"Unknown element: {element}")

        if z in self._cache:
            return self._cache[z]

        if self.backend == "toyomacro":
            df = _frame_from_toyomacro("scofield", element)
            self._cache[z] = df
            return df

        sheet_name = self._get_sheet_name(z, element)
        df = pd.read_excel(_require_workbook(self.file_path), sheet_name=sheet_name)

        # Convert energy column to numeric and to eV
        energy_col = 'PhotonEnergy[keV]'
        df[energy_col] = pd.to_numeric(df[energy_col], errors='coerce')
        df['PhotonEnergy_eV'] = df[energy_col] * 1000

        # Remove rows with NaN energy
        df = df.dropna(subset=['PhotonEnergy_eV'])

        self._cache[z] = df
        return df

    def get_cross_section(
        self,
        element: str,
        orbital: str,
        photon_energy_eV: float,
    ) -> CrossSectionResult:
        """Get cross section at specified energy."""
        df = self.load_element(element)

        # Get column name
        col = ORBITAL_COLUMNS['scofield'].get(orbital, orbital)
        if col not in df.columns:
            raise ValueError(f"Orbital {orbital} ({col}) not found for {element}")

        energies = df['PhotonEnergy_eV'].values
        cross_sections = df[col].values

        # Keep only tabulated, positive values. The Excel sheets hold 0 below
        # the ionisation threshold, toyomacro holds NaN there; both mean
        # "not tabulated" and must not enter the log-log interpolation.
        valid = ~np.isnan(cross_sections) & (cross_sections > 0)
        energies = energies[valid]
        cross_sections = cross_sections[valid]

        e_min, e_max = energies.min(), energies.max()
        is_extrapolated = photon_energy_eV < e_min or photon_energy_eV > e_max

        # Interpolate (log-log scale for better accuracy)
        log_e = np.log(energies)
        log_cs = np.log(cross_sections)

        interp_func = interp1d(
            log_e, log_cs,
            kind='linear',
            fill_value='extrapolate',
        )

        log_result = interp_func(np.log(photon_energy_eV))
        result_barns = np.exp(log_result)

        # Excel: barns -> Megabarns (1 barn = 10^-24 cm^2, 1 Mb = 10^-18 cm^2);
        # toyomacro: already Mb (factor 1.0).
        result_Mb = result_barns * self._to_Mb
        if photon_energy_eV < e_min:
            # Below the lowest tabulated value the subshell cannot be ionised
            # (or the table does not cover it): no number is defensible, so
            # return NaN on every source instead of a downward extrapolation.
            result_Mb = float("nan")

        return CrossSectionResult(
            element=element,
            orbital=orbital,
            photon_energy_eV=photon_energy_eV,
            cross_section_Mb=result_Mb,
            source='Scofield (1973)',
            is_interpolated=not is_extrapolated,
            is_extrapolated=is_extrapolated,
            energy_range=(e_min, e_max),
        )


class YehLindauLoader:
    """Load Yeh-Lindau (1985) cross section data."""

    def __init__(self, db_path: Optional[Path] = None):
        self.backend, self.db_path = resolve_source(db_path)
        self.file_path = self.db_path / YEH_LINDAU_WORKBOOK if self.db_path is not None else None
        self._cache: Dict[int, pd.DataFrame] = {}

    def _get_sheet_name(self, z: int, element: str) -> str:
        """Get sheet name for element."""
        return f"{z}_{element}"

    def load_element(self, element: str) -> pd.DataFrame:
        """Load cross section data for an element."""
        z = ELEMENT_Z.get(element)
        if z is None:
            raise ValueError(f"Unknown element: {element}")

        if z in self._cache:
            return self._cache[z]

        if self.backend == "toyomacro":
            df = _frame_from_toyomacro("yeh_lindau", element)
            self._cache[z] = df
            return df

        sheet_name = self._get_sheet_name(z, element)
        df = pd.read_excel(_require_workbook(self.file_path), sheet_name=sheet_name)

        # Rename energy column
        df = df.rename(columns={'Photon Energy / eV': 'PhotonEnergy_eV'})

        # Remove metadata rows (like "#electrons")
        df = df[df['PhotonEnergy_eV'].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x))]

        self._cache[z] = df
        return df

    def get_cross_section(
        self,
        element: str,
        orbital: str,
        photon_energy_eV: float,
    ) -> CrossSectionResult:
        """Get cross section at specified energy."""
        df = self.load_element(element)

        # Get column name
        col = ORBITAL_COLUMNS['yeh_lindau'].get(orbital, orbital)
        if col not in df.columns:
            raise ValueError(f"Orbital {orbital} ({col}) not found for {element}")

        energies = df['PhotonEnergy_eV'].values.astype(float)
        cross_sections = df[col].values.astype(float)

        # Keep only tabulated, positive values (see ScofieldLoader).
        valid = ~np.isnan(cross_sections) & ~np.isnan(energies) & (cross_sections > 0)
        energies = energies[valid]
        cross_sections = cross_sections[valid]

        if len(energies) < 2:
            raise ValueError(f"Insufficient data for {element} {orbital}")

        e_min, e_max = energies.min(), energies.max()
        is_extrapolated = photon_energy_eV < e_min or photon_energy_eV > e_max

        # Interpolate (log-log scale)
        # Yeh-Lindau data is already in Megabarns
        log_e = np.log(energies)
        log_cs = np.log(cross_sections)

        interp_func = interp1d(
            log_e, log_cs,
            kind='linear',
            fill_value='extrapolate',
        )

        log_result = interp_func(np.log(photon_energy_eV))
        result_Mb = np.exp(log_result)
        if photon_energy_eV < e_min:
            result_Mb = float("nan")  # below threshold / not tabulated: see ScofieldLoader

        return CrossSectionResult(
            element=element,
            orbital=orbital,
            photon_energy_eV=photon_energy_eV,
            cross_section_Mb=result_Mb,
            source='Yeh-Lindau (1985)',
            is_interpolated=not is_extrapolated,
            is_extrapolated=is_extrapolated,
            energy_range=(e_min, e_max),
        )


# =============================================================================
# Unified Database Interface
# =============================================================================

class CrossSectionDatabase:
    """Unified interface for cross section databases."""

    def __init__(self, db_path: Optional[Path] = None):
        self.backend, self.db_path = resolve_source(db_path)
        self._scofield = ScofieldLoader(self.db_path)
        self._yeh_lindau = YehLindauLoader(self.db_path)

    def get_cross_section(
        self,
        element: str,
        orbital: str,
        photon_energy_eV: float,
        source: str = 'auto',
    ) -> CrossSectionResult:
        """
        Get cross section for element/orbital at given photon energy.

        Args:
            element: Element symbol (e.g., 'Si', 'O')
            orbital: Orbital name (e.g., '1s', '2p')
            photon_energy_eV: Photon energy in eV
            source: 'scofield', 'yeh_lindau', or 'auto'
                   'auto' uses Scofield for >5 keV, Yeh-Lindau otherwise

        Returns:
            CrossSectionResult with cross section and metadata
        """
        if source == 'auto':
            # Use Scofield for HAXPES (>5 keV), Yeh-Lindau for standard XPS
            source = 'scofield' if photon_energy_eV > 5000 else 'yeh_lindau'

        if source == 'scofield':
            return self._scofield.get_cross_section(element, orbital, photon_energy_eV)
        elif source == 'yeh_lindau':
            return self._yeh_lindau.get_cross_section(element, orbital, photon_energy_eV)
        else:
            raise ValueError(f"Unknown source: {source}")

    def compare_at_energy(
        self,
        elements: List[str],
        orbital: str,
        photon_energy_eV: float,
    ) -> DatabaseComparison:
        """
        Compare cross sections from different databases.

        Args:
            elements: List of element symbols
            orbital: Orbital name
            photon_energy_eV: Photon energy in eV

        Returns:
            DatabaseComparison with values and ratios
        """
        scofield = {}
        yeh_lindau = {}
        yl_to_scofield = {}
        concentration_change = {}

        for elem in elements:
            try:
                sc_result = self._scofield.get_cross_section(elem, orbital, photon_energy_eV)
                scofield[elem] = sc_result.cross_section_Mb
            except Exception as e:
                warnings.warn(f"Scofield data not available for {elem} {orbital}: {e}")
                scofield[elem] = np.nan

            try:
                yl_result = self._yeh_lindau.get_cross_section(elem, orbital, photon_energy_eV)
                yeh_lindau[elem] = yl_result.cross_section_Mb
            except Exception as e:
                warnings.warn(f"Yeh-Lindau data not available for {elem} {orbital}: {e}")
                yeh_lindau[elem] = np.nan

            # Calculate ratio
            if scofield[elem] > 0 and not np.isnan(scofield[elem]):
                if not np.isnan(yeh_lindau[elem]):
                    yl_to_scofield[elem] = yeh_lindau[elem] / scofield[elem]
                    # Concentration ∝ 1/σ, so Δc/c = Δσ/σ
                    concentration_change[elem] = (1 - yl_to_scofield[elem]) * 100
                else:
                    yl_to_scofield[elem] = np.nan
                    concentration_change[elem] = np.nan
            else:
                yl_to_scofield[elem] = np.nan
                concentration_change[elem] = np.nan

        return DatabaseComparison(
            elements=elements,
            orbital=orbital,
            photon_energy_eV=photon_energy_eV,
            scofield=scofield,
            yeh_lindau=yeh_lindau,
            trzhaskovskaya=None,  # Not implemented yet
            yl_to_scofield=yl_to_scofield,
            concentration_change_pct=concentration_change,
        )

    def get_energy_dependence(
        self,
        element: str,
        orbital: str,
        energy_range: Tuple[float, float] = (1000, 15000),
        n_points: int = 50,
        source: str = 'both',
    ) -> Dict[str, np.ndarray]:
        """
        Get cross section vs energy curve.

        Args:
            element: Element symbol
            orbital: Orbital name
            energy_range: (min, max) energy in eV
            n_points: Number of points
            source: 'scofield', 'yeh_lindau', or 'both'

        Returns:
            Dictionary with 'energy', 'scofield', 'yeh_lindau' arrays
        """
        energies = np.logspace(
            np.log10(energy_range[0]),
            np.log10(energy_range[1]),
            n_points
        )

        result = {'energy': energies}

        if source in ('scofield', 'both'):
            scofield = []
            for e in energies:
                try:
                    r = self._scofield.get_cross_section(element, orbital, e)
                    scofield.append(r.cross_section_Mb)
                except:
                    scofield.append(np.nan)
            result['scofield'] = np.array(scofield)

        if source in ('yeh_lindau', 'both'):
            yeh_lindau = []
            for e in energies:
                try:
                    r = self._yeh_lindau.get_cross_section(element, orbital, e)
                    yeh_lindau.append(r.cross_section_Mb)
                except:
                    yeh_lindau.append(np.nan)
            result['yeh_lindau'] = np.array(yeh_lindau)

        return result


# =============================================================================
# Visualization
# =============================================================================

def plot_database_comparison(
    comparison: DatabaseComparison,
    output_path: Optional[Path] = None,
) -> None:
    """Plot database comparison."""
    import matplotlib.pyplot as plt

    elements = comparison.elements
    n = len(elements)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(n)
    width = 0.35

    # Cross sections
    ax = axes[0]
    sc_vals = [comparison.scofield.get(e, 0) for e in elements]
    yl_vals = [comparison.yeh_lindau.get(e, 0) for e in elements]

    ax.bar(x - width/2, sc_vals, width, label='Scofield')
    ax.bar(x + width/2, yl_vals, width, label='Yeh-Lindau')
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_ylabel('Cross section (Mb)')
    ax.set_title(f'{comparison.orbital} @ {comparison.photon_energy_eV/1000:.1f} keV')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Ratios
    ax = axes[1]
    ratios = [comparison.yl_to_scofield.get(e, np.nan) for e in elements]
    colors = ['green' if 0.95 <= r <= 1.05 else 'orange' if 0.9 <= r <= 1.1 else 'red'
              for r in ratios]
    ax.bar(x, ratios, color=colors)
    ax.axhline(1.0, color='k', linestyle='--', label='Perfect agreement')
    ax.axhline(0.95, color='g', linestyle=':', alpha=0.5)
    ax.axhline(1.05, color='g', linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_ylabel('YL / Scofield ratio')
    ax.set_title('Database Ratio')
    ax.set_ylim([0.7, 1.3])
    ax.grid(True, alpha=0.3)

    # Concentration impact
    ax = axes[2]
    conc_change = [comparison.concentration_change_pct.get(e, 0) for e in elements]
    colors = ['green' if abs(c) < 5 else 'orange' if abs(c) < 10 else 'red'
              for c in conc_change]
    ax.bar(x, conc_change, color=colors)
    ax.axhline(0, color='k', linestyle='-')
    ax.axhline(5, color='g', linestyle=':', alpha=0.5)
    ax.axhline(-5, color='g', linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_ylabel('Concentration change (%)')
    ax.set_title('If using Scofield instead of YL')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_energy_dependence(
    db: CrossSectionDatabase,
    element: str,
    orbital: str,
    energy_range: Tuple[float, float] = (500, 15000),
    output_path: Optional[Path] = None,
) -> None:
    """Plot cross section vs energy for both databases."""
    import matplotlib.pyplot as plt

    data = db.get_energy_dependence(element, orbital, energy_range, source='both')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Cross section vs energy
    ax = axes[0]
    if 'scofield' in data:
        ax.loglog(data['energy'], data['scofield'], 'b-', label='Scofield', linewidth=2)
    if 'yeh_lindau' in data:
        ax.loglog(data['energy'], data['yeh_lindau'], 'r--', label='Yeh-Lindau', linewidth=2)

    # Mark common energies
    common_energies = [1486.6, 8047.8, 9251.7]  # Al Kα, Cu Kα, Ga Kα
    for e in common_energies:
        if energy_range[0] <= e <= energy_range[1]:
            ax.axvline(e, color='gray', linestyle=':', alpha=0.5)
            ax.text(e, ax.get_ylim()[1], f'{e/1000:.1f}keV', rotation=90, va='top', fontsize=8)

    ax.set_xlabel('Photon energy (eV)')
    ax.set_ylabel('Cross section (Mb)')
    ax.set_title(f'{element} {orbital}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ratio vs energy
    ax = axes[1]
    if 'scofield' in data and 'yeh_lindau' in data:
        ratio = data['yeh_lindau'] / data['scofield']
        ax.semilogx(data['energy'], ratio, 'k-', linewidth=2)
        ax.axhline(1.0, color='gray', linestyle='--')
        ax.fill_between(data['energy'], 0.95, 1.05, alpha=0.2, color='green', label='±5%')
        ax.fill_between(data['energy'], 0.9, 1.1, alpha=0.1, color='orange', label='±10%')

        for e in common_energies:
            if energy_range[0] <= e <= energy_range[1]:
                ax.axvline(e, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Photon energy (eV)')
    ax.set_ylabel('YL / Scofield ratio')
    ax.set_title('Database comparison')
    ax.set_ylim([0.7, 1.3])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()


def print_comparison_table(comparison: DatabaseComparison) -> None:
    """Print comparison as formatted table."""
    print("=" * 75)
    print(f"Cross Section Comparison @ {comparison.photon_energy_eV/1000:.2f} keV")
    print(f"Orbital: {comparison.orbital}")
    print("=" * 75)
    print(f"{'Element':<8} {'Z':>4} {'Scofield':>12} {'Yeh-Lindau':>12} {'YL/Sc':>8} {'Δc (%)':>8}")
    print("-" * 75)

    for elem in comparison.elements:
        z = ELEMENT_Z.get(elem, 0)
        sc = comparison.scofield.get(elem, np.nan)
        yl = comparison.yeh_lindau.get(elem, np.nan)
        ratio = comparison.yl_to_scofield.get(elem, np.nan)
        dc = comparison.concentration_change_pct.get(elem, np.nan)

        sc_str = f"{sc:.2e}" if not np.isnan(sc) else "N/A"
        yl_str = f"{yl:.2e}" if not np.isnan(yl) else "N/A"
        ratio_str = f"{ratio:.3f}" if not np.isnan(ratio) else "N/A"
        dc_str = f"{dc:+.1f}" if not np.isnan(dc) else "N/A"

        print(f"{elem:<8} {z:>4} {sc_str:>12} {yl_str:>12} {ratio_str:>8} {dc_str:>8}")

    print("=" * 75)

    # Statistics
    ratios = [r for r in comparison.yl_to_scofield.values() if not np.isnan(r)]
    if ratios:
        print(f"YL/Scofield ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
        print(f"Range: {min(ratios):.3f} - {max(ratios):.3f}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Demonstrate cross section database usage."""
    from deppro.paths import ensure_outputs_dir
    output_dir = ensure_outputs_dir()

    db = CrossSectionDatabase()

    # Compare at Ga Kα energy (HAXPES)
    print("\n" + "=" * 75)
    print("Cross Section Comparison for C/Al2O3/TiO2/SiO2/Si @ Ga Kα (9251.7 eV)")
    print("=" * 75)

    elements = ['C', 'O', 'Al', 'Si', 'Ti']
    comparison = db.compare_at_energy(elements, '1s', 9251.7)
    print_comparison_table(comparison)

    # Plot comparison
    plot_database_comparison(comparison, output_dir / "cross_section_comparison_9keV.png")

    # Plot energy dependence for key elements
    print("\nGenerating energy dependence plots...")
    for elem in ['Si', 'Ti', 'O']:
        plot_energy_dependence(
            db, elem, '1s',
            energy_range=(500, 15000),
            output_path=output_dir / f"cross_section_{elem}_1s_energy.png"
        )

    # Compare at standard XPS energy
    print("\n" + "=" * 75)
    print("Cross Section Comparison @ Al Kα (1486.6 eV)")
    print("=" * 75)

    comparison_xps = db.compare_at_energy(elements, '1s', 1486.6)
    print_comparison_table(comparison_xps)
    plot_database_comparison(comparison_xps, output_dir / "cross_section_comparison_AlKa.png")


if __name__ == '__main__':
    main()
