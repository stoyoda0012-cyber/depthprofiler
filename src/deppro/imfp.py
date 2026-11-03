"""
IMFP (Inelastic Mean Free Path) Calculator
==========================================

Calculates IMFP using TPP (Tanuma-Powell-Penn) and JTP formulas.

References:
    TPP:   S. Tanuma et al. Surface Science 192, L849 (1987)
           S. Tanuma et al. Surf. Interface Anal. 11, 577 (1988)
    TPP-2: S. Tanuma et al. Surf. Interface Anal. 17, 911 (1991)
           S. Tanuma et al. Surf. Interface Anal. 17, 927 (1991)
           S. Tanuma et al. Surf. Interface Anal. 20, 77 (1993)
    TPP-2M: S. Tanuma et al. Surf. Interface Anal. 21, 165 (1994)
            H. Shinotsuka et al. Surf. Interface Anal. 47, 871 (2015)
    JTP:   A. Jablonski, S. Tanuma, C.J. Powell, Surf. Interface Anal. 55, 609 (2023)

Usage:
    from deppro.imfp import IMFP, CompoundDatabase

    # Direct calculation
    lambda_imfp = IMFP.tpp2m(KE=1000, Nv=4, density=2.329, Aw=28.09, Eg=1.12)

    # Using compound database
    db = CompoundDatabase()
    lambda_imfp = db.calculate_imfp('Si', kinetic_energy=1000)
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import csv

# How to install toyomacro (not on PyPI). Pinned to the release this package was
# checked against in a clean venv, the same tag the ``fit`` extra in
# pyproject.toml names; CI installs toyomacro through that extra too (a weekly
# job also tests against toyomacro's main).
TOYOMACRO_INSTALL = "pip install 'toyomacro @ git+https://github.com/stoyoda0012-cyber/toyomacro@v0.3.1'"


@dataclass
class CompoundParams:
    """Material parameters for IMFP calculation."""
    name: str
    valence_charge: float    # Nv - Number of valence electrons
    density: float           # g/cm³
    molecular_weight: float  # g/mol (Aw)
    band_gap: float         # eV (Eg)

    def __repr__(self) -> str:
        return (f"CompoundParams({self.name}: Nv={self.valence_charge}, "
                f"ρ={self.density} g/cm³, Aw={self.molecular_weight} g/mol, "
                f"Eg={self.band_gap} eV)")


class IMFP:
    """
    Inelastic Mean Free Path calculator.

    Implements TPP, TPP-2, and TPP-2M formulas for IMFP calculation.
    """

    @staticmethod
    def tpp(
        KE: Union[float, np.ndarray],
        Nv: float,
        density: float,
        Aw: float,
        Eg: float = 0.0
    ) -> Union[float, np.ndarray]:
        """
        Original TPP formula (1987-1988).

        Args:
            KE: Kinetic energy (eV)
            Nv: Number of valence electrons
            density: Density (g/cm³)
            Aw: Molecular weight (g/mol)
            Eg: Band gap (eV), default 0 for metals

        Returns:
            IMFP in Angstroms
        """
        gamma = 0.151 * density**(-0.49)
        Ep = 28.8 * (density * Nv / Aw)**(1/2)
        beta = -0.0252 + 1.05 / (Ep**2 + Eg**2)**(1/2) + 8.10e-4 * density

        return KE / (Ep**2 * beta * np.log(gamma * KE))

    @staticmethod
    def tpp2(
        KE: Union[float, np.ndarray],
        Nv: float,
        density: float,
        Aw: float,
        Eg: float = 0.0
    ) -> Tuple[Union[float, np.ndarray], Dict[str, float]]:
        """
        TPP-2 formula (1991-1993).

        Args:
            KE: Kinetic energy (eV)
            Nv: Number of valence electrons
            density: Density (g/cm³)
            Aw: Molecular weight (g/mol)
            Eg: Band gap (eV)

        Returns:
            Tuple of (IMFP in Angstroms, dict of intermediate parameters)
        """
        gamma = 0.191 * density**(-0.50)
        Ep = 28.816 * (density * Nv / Aw)**(1/2)
        beta = -0.0216 + 0.944 / (Ep**2 + Eg**2)**(1/2) + 7.39e-4 * density
        U = (Ep / 28.816)**2
        C = 1.97 - 0.91 * U
        D = 53.4 - 20.8 * U

        imfp = KE / (Ep**2 * (beta * np.log(gamma * KE) - C / KE + D / KE**2))

        params = {'Ep': Ep, 'U': U, 'beta': beta, 'gamma': gamma, 'C': C, 'D': D}
        return imfp, params

    @staticmethod
    def tpp2m(
        KE: Union[float, np.ndarray],
        Nv: float,
        density: float,
        Aw: float,
        Eg: float = 0.0
    ) -> Union[float, np.ndarray]:
        """
        TPP-2M formula (1994-2015) - Most commonly used.

        Args:
            KE: Kinetic energy (eV)
            Nv: Number of valence electrons
            density: Density (g/cm³)
            Aw: Molecular weight (g/mol)
            Eg: Band gap (eV)

        Returns:
            IMFP in Angstroms
        """
        gamma = 0.191 * density**(-0.50)
        Ep = 28.816 * (density * Nv / Aw)**(1/2)
        beta = -0.10 + 0.944 / (Ep**2 + Eg**2)**(1/2) + 0.0691 * density**0.1
        U = (Ep / 28.816)**2
        C = 1.97 - 0.91 * U
        D = 53.4 - 20.8 * U

        return KE / (Ep**2 * (beta * np.log(gamma * KE) - C / KE + D / KE**2))

    @staticmethod
    def tpp2m_with_params(
        KE: Union[float, np.ndarray],
        Nv: float,
        density: float,
        Aw: float,
        Eg: float = 0.0
    ) -> Tuple[Union[float, np.ndarray], Dict[str, float]]:
        """
        TPP-2M formula with intermediate parameters returned.

        Returns:
            Tuple of (IMFP in Angstroms, dict of intermediate parameters)
        """
        gamma = 0.191 * density**(-0.50)
        Ep = 28.816 * (density * Nv / Aw)**(1/2)
        beta = -0.10 + 0.944 / (Ep**2 + Eg**2)**(1/2) + 0.0691 * density**0.1
        U = (Ep / 28.816)**2
        C = 1.97 - 0.91 * U
        D = 53.4 - 20.8 * U

        imfp = KE / (Ep**2 * (beta * np.log(gamma * KE) - C / KE + D / KE**2))

        params = {'Ep': Ep, 'U': U, 'beta': beta, 'gamma': gamma, 'C': C, 'D': D}
        return imfp, params

    @staticmethod
    def jtp(
        KE: Union[float, np.ndarray],
        Nv: float,
        density: float,
        Aw: float,
        Eg: float = 0.0
    ) -> Union[float, np.ndarray]:
        """
        JTP formula (2023) — refitted modified Bethe equation with
        relativistic correction.

            A. Jablonski, S. Tanuma, C.J. Powell,
            Surf. Interface Anal. 55, 609 (2023).

        Fitted to 8300 IMFPs (100 materials, 83 energies, 50 eV – 200 keV)
        with RMS deviation 10.2% (vs TPP-2M 16.0%).

        Args:
            KE: Kinetic energy (eV)
            Nv: Number of valence electrons
            density: Density (g/cm³)
            Aw: Molecular weight (g/mol)
            Eg: Band gap (eV)

        Returns:
            IMFP in Angstroms
        """
        # Relativistic correction α(E) — Eq. (9a)
        alpha = (1.0 + KE / 1021999.8) / (1.0 + KE / 510998.9) ** 2

        Ep = 28.816 * (density * Nv / Aw) ** 0.5
        U  = (Ep / 28.816) ** 2

        # JTP coefficients in nm-units (Eq. 12a-d, Table 1)
        beta_nm  = 0.0539 + 17.0 / (Ep ** 2 + Eg ** 2) ** 0.639 \
                   - 0.252 * density ** (-0.463)
        gamma    = 0.115 * density ** (-0.253)
        C_nm     = 9.76 + 2.09 * U
        D_nm     = 97.5 + 223.0 * U

        # Convert nm → Å (factor 10) for β, C, D
        beta = beta_nm / 10.0
        C    = C_nm    / 10.0
        D    = D_nm    / 10.0

        # Modified Bethe with relativistic correction — Eq. (8), result in Å
        return alpha * KE / (
            Ep ** 2
            * (beta * np.log(gamma * alpha * KE) - C / KE + D / KE ** 2)
        )

    @staticmethod
    def calculate(
        KE: Union[float, np.ndarray],
        Nv: float,
        density: float,
        Aw: float,
        Eg: float = 0.0,
        version: int = 3
    ) -> Union[float, np.ndarray]:
        """
        Calculate IMFP with specified version.

        Args:
            KE: Kinetic energy (eV)
            Nv: Number of valence electrons
            density: Density (g/cm³)
            Aw: Molecular weight (g/mol)
            Eg: Band gap (eV)
            version: 1 (TPP), 2 (TPP-2), 3 (TPP-2M, default), or 4 (JTP 2023)

        Returns:
            IMFP in Angstroms
        """
        if version == 1:
            return IMFP.tpp(KE, Nv, density, Aw, Eg)
        elif version == 2:
            return IMFP.tpp2(KE, Nv, density, Aw, Eg)[0]
        elif version == 3:
            return IMFP.tpp2m(KE, Nv, density, Aw, Eg)
        elif version == 4:
            return IMFP.jtp(KE, Nv, density, Aw, Eg)
        else:
            raise ValueError(
                f"Version must be 1 (TPP), 2 (TPP-2), 3 (TPP-2M), or 4 (JTP), got {version}"
            )

    @staticmethod
    def from_compound(
        KE: Union[float, np.ndarray],
        compound: CompoundParams,
        version: int = 3
    ) -> Union[float, np.ndarray]:
        """
        Calculate IMFP using CompoundParams object.

        Args:
            KE: Kinetic energy (eV)
            compound: CompoundParams with material properties
            version: TPP version (1, 2, or 3)

        Returns:
            IMFP in Angstroms
        """
        return IMFP.calculate(
            KE,
            compound.valence_charge,
            compound.density,
            compound.molecular_weight,
            compound.band_gap,
            version
        )


class CompoundDatabase:
    """
    Database of compound parameters for IMFP calculation.

    Source of the parameters (N_v, density, M_w, E_g), in order:

    1. ``DEPPRO_COMPOUND_TABLE_PATH``, or an explicit ``csv_path=`` —
       a CompoundTable.csv in the MATLAB-compatible layout.
    2. ``toyomacro.data.CompoundDB`` — the default.

    If neither resolves, construction raises and names both.

    Why toyomacro is the default
    ----------------------------
    The table comes from an explicit setting or from toyomacro, never from a
    file that happens to sit next to the checkout, so every machine and CI
    compute from the same table unless an operator chooses otherwise.

    The two sources are separate compilations and are not identical. The
    MATLAB-compatible CSV can hold compounds that toyomacro does not, and
    some shared entries differ (mostly band gaps), which may reflect
    different phases (for example SiC polytypes, or α versus amorphous
    Al2O3). IMFPs, and depth profiles computed from them, can therefore
    differ between the two sources: record which table a result used.
    toyomacro records per-entry provenance (``CompoundDB.get_provenance``,
    ``get_comparisons``); most of its default values are not traced to a
    primary source.
    """

    # Legacy names → current CSV entries. The table splits 'Si3N4' into
    # aSi3N4 (amorphous, the former unprefixed entry) and bSi3N4 (β-crystal).
    ALIASES = {
        'Si3N4': 'aSi3N4',
    }

    @staticmethod
    def _get_override_csv_path() -> Optional[Path]:
        """CSV override from the environment, or None to use toyomacro.

        Reading the env var here (not at import time) keeps this correct in
        ProcessPoolExecutor workers, which re-import but inherit the
        environment.
        """
        import os
        env_path = os.environ.get('DEPPRO_COMPOUND_TABLE_PATH')
        return Path(env_path) if env_path else None

    def _load_from_toyomacro(self) -> None:
        """Load parameters from ``toyomacro.data.CompoundDB``.

        toyomacro is the optional extra ``fit``. As for every optional dependency, the import
        error surfaces here, at the point of reading data, and names both the
        extra and the env var — importing deppro stays possible without it.
        """
        try:
            from toyomacro.data import CompoundDB
        except ImportError as exc:
            # The ``fit`` extra pins the same release tag as TOYOMACRO_INSTALL
            # (a direct git reference, since toyomacro is not on PyPI).
            raise FileNotFoundError(
                "No compound parameter source available. Either install "
                f"toyomacro (`pip install 'deppro[fit]'`, or {TOYOMACRO_INSTALL}"
                " — it provides toyomacro.data.CompoundDB), or "
                "point DEPPRO_COMPOUND_TABLE_PATH at a CompoundTable.csv. "
                f"Underlying import error: {exc}"
            ) from exc

        for name in CompoundDB.list_compounds():
            props = CompoundDB.get_properties(name)
            self._compounds[name] = CompoundParams(
                name=name,
                valence_charge=float(props['Nv']),
                density=float(props['density']),
                molecular_weight=float(props['Mw']),
                band_gap=float(props['Eg']),
            )

    def __init__(self, csv_path: Optional[Path] = None):
        """
        Initialize the compound database.

        Args:
            csv_path: Path to CompoundTable.csv. If None, uses default path.
        """
        if csv_path is not None:
            self.csv_path = Path(csv_path)
        else:
            self.csv_path = self._get_override_csv_path()
        self._compounds: Dict[str, CompoundParams] = {}
        if self.csv_path is not None:
            self._load_database()
            self.source = f"csv:{self.csv_path}"
        else:
            self._load_from_toyomacro()
            self.source = "toyomacro.data.CompoundDB"

    def _load_database(self) -> None:
        """Load compound data from CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CompoundTable.csv not found at {self.csv_path}")

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['CompoundName']
                self._compounds[name] = CompoundParams(
                    name=name,
                    valence_charge=float(row['ValenceCharge']),
                    density=float(row['Density [g/cm3]']),
                    molecular_weight=float(row['MolecularWeight [g/mol]']),
                    band_gap=float(row['BandGap [eV]'])
                )

    def get(self, name: str) -> CompoundParams:
        """
        Get compound parameters by name.

        Args:
            name: Compound name (e.g., 'Si', 'SiO2', 'TiO2')

        Returns:
            CompoundParams for the compound

        Raises:
            KeyError: If compound not found
        """
        if name not in self._compounds:
            name = self.ALIASES.get(name, name)
        if name not in self._compounds:
            available = ', '.join(sorted(self._compounds.keys())[:10])
            raise KeyError(f"Compound '{name}' not found. Available: {available}...")
        return self._compounds[name]

    def calculate_imfp(
        self,
        compound_name: str,
        kinetic_energy: Union[float, np.ndarray],
        version: int = 3
    ) -> Union[float, np.ndarray]:
        """
        Calculate IMFP for a compound.

        Args:
            compound_name: Name of compound in database
            kinetic_energy: Kinetic energy (eV)
            version: TPP version (default: 3 for TPP-2M)

        Returns:
            IMFP in Angstroms
        """
        compound = self.get(compound_name)
        return IMFP.from_compound(kinetic_energy, compound, version)

    def list_compounds(self) -> list:
        """Return list of all compound names."""
        return sorted(self._compounds.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._compounds or self.ALIASES.get(name) in self._compounds

    def __len__(self) -> int:
        return len(self._compounds)


class BindingEnergyDatabase:
    """
    Database of binding energies for elements.

    Source, in order: ``DEPPRO_BINDING_ENERGY_TABLE_PATH`` or an explicit
    ``csv_path=``, then ``toyomacro.data.BindingEnergy``. Same reasoning as
    :class:`CompoundDatabase` — the old default was a ``BindingEnergyTable.csv``
    in an unversioned sibling directory next to the repository, which CI cannot get.

    Unlike the compound table, **this swap changes no numbers.** Every
    (element, orbital) pair either source holds, NaN entries excluded:
    1361 agree, none disagree, and neither side has an entry the other
    lacks. Both are the standard elemental compilation (toyomacro's
    DATA_SOURCES.md traces its copy to Bearden & Burr 1967 and Fuggle &
    Mårtensson 1980, cross-checked against the LBNL X-ray Data Booklet).

    That count replaces the "1266 pairs" figure claimed when this class was
    switched over. The earlier comparison walked only the orbitals named in
    ORBITAL_COLUMNS — the same subset the loader had been restricted to — so
    it checked the loader against its own restriction and reported full
    agreement while 316 pairs were unreachable. The population has to be
    taken from the data, not from the code being tested.

    Note for debugging: ``ElementConfig.__post_init__`` catches
    FileNotFoundError here and leaves ``binding_energy`` as None, so a
    missing table used to surface far away as
    ``TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'``
    in ``transfer_matrix.calculate_lambda_basic``. That is how this one was
    found on 2026-08-03.
    """

    @staticmethod
    def _get_override_csv_path() -> Optional[Path]:
        """CSV override from the environment, or None to use toyomacro."""
        import os
        env_path = os.environ.get('DEPPRO_BINDING_ENERGY_TABLE_PATH')
        return Path(env_path) if env_path else None

    def _load_from_toyomacro(self) -> None:
        """Load binding energies from ``toyomacro.data.BindingEnergy``."""
        try:
            from toyomacro.data.binding_energy import BindingEnergy
        except ImportError as exc:
            raise FileNotFoundError(
                "No binding-energy source available. Either install "
                f"toyomacro (`pip install 'deppro[fit]'`, or {TOYOMACRO_INSTALL}), "
                "or point DEPPRO_BINDING_ENERGY_TABLE_PATH at a "
                f"BindingEnergyTable.csv. Underlying import error: {exc}"
            ) from exc

        # Keep the CSV-shaped {symbol: {column: eV}} layout so `get()` and its
        # ORBITAL_COLUMNS mapping work unchanged for both sources.
        #
        # Load **every** orbital the source offers, not the subset named in
        # ORBITAL_COLUMNS. That mapping exists to resolve convenience aliases
        # ('2p' -> '2p3/2'); it is not the set of orbitals callers may ask for,
        # because `get()` falls back to the raw name when the alias table has
        # no entry. Restricting the load to ORBITAL_COLUMNS.values() therefore
        # silently dropped every orbital without an alias — 316 (element,
        # orbital) pairs, among them Au 4f5/2, Au 4d3/2, Au 4p1/2, U 6s and the
        # 5p1/2, 5d3/2, 6p1/2, 6p3/2, 7s series. The CSV loader had always read
        # all of its columns, so those lookups worked before 1be222e and raised
        # KeyError after it, while the data sat in toyomacro unread.
        for symbol in BindingEnergy.get_all_elements():
            row = {}
            for column in BindingEnergy.get_orbitals(symbol):
                value = BindingEnergy.lookup(symbol, column)
                if value is not None:
                    row[column] = float(value)
            if row:
                self._data[symbol] = row

    # Orbital column mapping
    ORBITAL_COLUMNS = {
        '1s': '1s',
        '2s': '2s',
        '2p': '2p3/2',  # Default to 3/2 for 2p
        '2p1/2': '2p1/2',
        '2p3/2': '2p3/2',
        '3s': '3s',
        '3p': '3p3/2',
        '3p1/2': '3p1/2',
        '3p3/2': '3p3/2',
        '3d': '3d5/2',
        '3d3/2': '3d3/2',
        '3d5/2': '3d5/2',
        '4s': '4s',
        '4p': '4p3/2',
        '4d': '4d5/2',
        '4f': '4f7/2',
        '5s': '5s',
        '5p': '5p3/2',
        '5d': '5d5/2',
    }

    def __init__(self, csv_path: Optional[Path] = None):
        """Initialize the binding energy database."""
        if csv_path is not None:
            self.csv_path = Path(csv_path)
        else:
            self.csv_path = self._get_override_csv_path()
        self._data: Dict[str, Dict[str, float]] = {}
        if self.csv_path is not None:
            self._load_database()
            self.source = f"csv:{self.csv_path}"
        else:
            self._load_from_toyomacro()
            self.source = "toyomacro.data.BindingEnergy"

    def _load_database(self) -> None:
        """Load binding energy data from CSV."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"BindingEnergyTable.csv not found at {self.csv_path}")

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row['ElementSymbol']
                self._data[symbol] = {}
                for col, value in row.items():
                    if col not in ('ElementNumber', 'ElementSymbol'):
                        try:
                            be = float(value)
                            if not np.isnan(be):
                                self._data[symbol][col] = be
                        except (ValueError, TypeError):
                            pass

    def get(self, symbol: str, orbital: str) -> float:
        """
        Get binding energy for an element and orbital.

        Args:
            symbol: Element symbol (e.g., 'Si', 'O')
            orbital: Orbital name (e.g., '2p', '1s', '2p3/2')

        Returns:
            Binding energy in eV
        """
        if symbol not in self._data:
            raise KeyError(f"Element '{symbol}' not found")

        # Map orbital to column name
        col = self.ORBITAL_COLUMNS.get(orbital, orbital)

        if col not in self._data[symbol]:
            available = ', '.join(sorted(self._data[symbol].keys()))
            raise KeyError(f"Orbital '{orbital}' not found for {symbol}. Available: {available}")

        return self._data[symbol][col]

    def list_elements(self) -> list:
        """Return list of all element symbols."""
        return sorted(self._data.keys())

    def list_orbitals(self, symbol: str) -> list:
        """Return available orbitals for an element."""
        if symbol not in self._data:
            raise KeyError(f"Element '{symbol}' not found")
        return sorted(self._data[symbol].keys())


class CompoundBindingEnergyDatabase:
    """
    NIST compound binding energy database.

    Loads measured BE values for elements across thousands of compounds
    from CoreLevelBindingEnergyCompoundTable_NIST.csv (~29k entries, 91 elements).

    Primary use: identify the likely compound/chemical state from a
    measured binding energy, or look up BE reference values for peak fitting.
    """

    @staticmethod
    def _get_default_csv_path() -> Path:
        """``DEPPRO_COMPOUND_BE_TABLE_PATH``; there is no default table.

        No neighboring checkout is searched and no package bundles this table,
        so without the variable (or an explicit ``csv_path=``) construction
        raises ``FileNotFoundError`` naming the variable.
        """
        import os
        env_path = os.environ.get('DEPPRO_COMPOUND_BE_TABLE_PATH')
        if env_path:
            return Path(env_path)
        raise FileNotFoundError(
            "No NIST compound binding-energy table: set DEPPRO_COMPOUND_BE_TABLE_PATH "
            "to the path of CoreLevelBindingEnergyCompoundTable_NIST.csv, or pass "
            "csv_path=. No package bundles this table and no neighbouring checkout "
            "is searched.")

    def __init__(self, csv_path: Optional[Path] = None):
        if csv_path is not None:
            self.csv_path = Path(csv_path)
        else:
            self.csv_path = self._get_default_csv_path()
        self._records: list = []
        self._by_element: Dict[str, list] = {}
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"NIST BE table not found at {self.csv_path}")
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = {
                    'element': row['Element'],
                    'line': row['SpectralLine'],
                    'formula': row['Formula'],
                    'be': float(row['BE_eV']),
                }
                self._records.append(rec)
                self._by_element.setdefault(rec['element'], []).append(rec)

    def __len__(self) -> int:
        return len(self._records)

    def list_elements(self) -> list:
        return sorted(self._by_element.keys())

    def lookup(
        self,
        element: str,
        orbital: Optional[str] = None,
        formula: Optional[str] = None,
    ) -> list:
        """Return records matching element, optional orbital, optional formula."""
        records = self._by_element.get(element, [])
        if orbital is not None:
            records = [r for r in records if r['line'] == orbital or r['line'].startswith(orbital)]
        if formula is not None:
            records = [r for r in records if r['formula'] == formula]
        return records

    def identify(
        self,
        element: str,
        binding_energy: float,
        orbital: Optional[str] = None,
        tol: float = 0.5,
    ) -> list:
        """
        Identify candidate compounds matching a measured BE.

        Args:
            element: Element symbol (e.g. 'Si')
            binding_energy: Measured BE in eV
            orbital: Optional orbital filter (e.g. '2p', '2p3/2')
            tol: Matching tolerance in eV (default 0.5)

        Returns:
            List of {formula, line, be, delta} sorted by |delta|.
        """
        records = self.lookup(element, orbital=orbital)
        matches = []
        for r in records:
            delta = r['be'] - binding_energy
            if abs(delta) <= tol:
                matches.append({
                    'formula': r['formula'],
                    'line': r['line'],
                    'be': r['be'],
                    'delta': delta,
                })
        matches.sort(key=lambda x: abs(x['delta']))
        return matches


# Convenience function
def calculate_imfp(
    kinetic_energy: Union[float, np.ndarray],
    compound_name: str,
    csv_path: Optional[Path] = None,
    version: int = 3
) -> Union[float, np.ndarray]:
    """
    Convenience function to calculate IMFP.

    Args:
        kinetic_energy: Kinetic energy in eV
        compound_name: Compound name from database
        csv_path: Optional path to CompoundTable.csv
        version: TPP version (default: 3 for TPP-2M)

    Returns:
        IMFP in Angstroms
    """
    db = CompoundDatabase(csv_path)
    return db.calculate_imfp(compound_name, kinetic_energy, version)


if __name__ == "__main__":
    # Quick test
    print("IMFP Calculator Test")
    print("=" * 50)

    # Test TPP-2M for Si at 1000 eV
    # Si: Nv=4, density=2.329, Aw=28.09, Eg=1.12
    ke = 1000.0
    imfp_si = IMFP.tpp2m(ke, Nv=4, density=2.329, Aw=28.08553, Eg=1.12)
    print(f"Si at KE={ke} eV: λ = {imfp_si:.2f} Å")

    # Test with database
    try:
        db = CompoundDatabase()
        print(f"\nLoaded {len(db)} compounds")

        for compound in ['Si', 'SiO2', 'TiO2', 'Al2O3', 'C']:
            if compound in db:
                imfp = db.calculate_imfp(compound, ke)
                print(f"{compound} at KE={ke} eV: λ = {imfp:.2f} Å")
    except FileNotFoundError as e:
        print(f"Database not found: {e}")

    # Test binding energy
    try:
        be_db = BindingEnergyDatabase()
        print(f"\nBinding Energies:")
        for elem, orb in [('Si', '2p'), ('O', '1s'), ('Ti', '2p'), ('Al', '2p'), ('C', '1s')]:
            be = be_db.get(elem, orb)
            print(f"  {elem} {orb}: {be} eV")
    except FileNotFoundError as e:
        print(f"BE Database not found: {e}")
