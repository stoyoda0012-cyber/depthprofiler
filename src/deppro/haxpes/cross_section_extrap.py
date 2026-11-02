"""Yeh–Lindau 1s photoionization cross section, extrapolated in energy and Z.

Moved from an untracked analysis script (function
``yeh_lindau_extrapolate_1s``) so that the production path
``deppro.haxpes.preprocessor.apply_cross_section_correction(...,
source='yeh_lindau_extrap')`` depends only on tracked code. The algorithm is
unchanged; only the location of the table is now resolved through
``DEPPRO_YEH_LINDAU_TABLE_PATH``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from deppro.paths import env_value

__all__ = ["yeh_lindau_table_path", "resolve_yeh_lindau_source", "yeh_lindau_extrapolate_1s"]

_ENV = "DEPPRO_YEH_LINDAU_TABLE_PATH"
_TABLE_NAME = "CrossSectionTable_Yeh=Lindau.csv"


def yeh_lindau_table_path() -> Optional[Path]:
    """Path of the Yeh–Lindau cross-section CSV, or ``None``.

    ``DEPPRO_YEH_LINDAU_TABLE_PATH`` if set (``~`` is expanded); an empty value
    counts as unset. There is no default path: no neighbouring checkout is
    searched.
    """
    value = env_value(_ENV)
    if value:
        return Path(value).expanduser()
    return None


def resolve_yeh_lindau_source() -> Tuple[str, Optional[Path]]:
    """Where the Yeh–Lindau table comes from: ``('csv', path)`` or ``('toyomacro', None)``.

    ``DEPPRO_YEH_LINDAU_TABLE_PATH`` set -> that CSV. Otherwise toyomacro's
    bundled Yeh–Lindau table (generated from the same CSV; 7172 values compared
    equal on 2026-10-02). Otherwise ``FileNotFoundError`` naming the variable and
    toyomacro; no neighbouring checkout is searched. toyomacro keeps the
    complete, reviewed tables; deppro extracts the 1s rows it needs.
    """
    if env_value(_ENV):
        return "csv", yeh_lindau_table_path()
    try:
        from toyomacro.data.cross_section import CrossSection  # noqa: F401
    except ImportError:
        raise FileNotFoundError(
            f"No Yeh–Lindau table source: {_ENV} is not set and toyomacro is not "
            f"installed. Install toyomacro (pip install 'deppro[fit]'), whose bundled "
            f"table is the default source, or set {_ENV} to the path of {_TABLE_NAME}."
        ) from None
    return "toyomacro", None


def _yeh_lindau_1s_rows() -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """``(Z, photon_energies_eV, sigma_Mb)`` for every element with a tabulated 1s
    cross section, ordered by Z, from :func:`resolve_yeh_lindau_source`.

    Raises ``FileNotFoundError`` naming ``DEPPRO_YEH_LINDAU_TABLE_PATH`` when the
    CSV is the source and is missing.
    """
    source, csv_path = resolve_yeh_lindau_source()
    rows: List[Tuple[int, np.ndarray, np.ndarray]] = []
    if source == "toyomacro":
        from toyomacro.data.cross_section import CrossSection
        from .cross_section_loader import ELEMENT_Z
        data = CrossSection.to_dict("yeh_lindau")
        pe_values = np.asarray(data["photon_energies"], dtype=float)
        for element, orbitals in data["data"].items():
            if "1s" not in orbitals or element not in ELEMENT_Z:
                continue
            cs_vals = np.array(
                [np.nan if v is None else v for v in orbitals["1s"]["cross_sections"]], dtype=float
            )
            rows.append((ELEMENT_Z[element], pe_values, cs_vals))
    else:
        import pandas as pd
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Yeh–Lindau cross-section table not found at {csv_path}. "
                f"Set {_ENV} to the path of {_TABLE_NAME}, or install toyomacro "
                f"(extra ``fit``), whose bundled table is the default source.")
        df = pd.read_csv(csv_path)
        df = df[df['AtomicNumber'] > 0]  # skip header row
        pe_cols = [c for c in df.columns if c.startswith('CrossSec_PE_')]
        pe_values = np.array([float(c.replace('CrossSec_PE_', '').replace('_', '.'))
                              for c in pe_cols])
        for _, row in df[df['AtomicOrbital'] == '1s'].iterrows():
            rows.append((int(row['AtomicNumber']), pe_values, row[pe_cols].values.astype(float)))
    rows.sort(key=lambda r: r[0])
    return rows


def yeh_lindau_extrapolate_1s(target_z: int, photon_energy: float,
                              poly_order_energy: int = 3,
                              poly_order_z: int = 3) -> float:
    """Extrapolate the Yeh–Lindau 1s cross section (Mb) to Z > 12.

    MATLAB equivalent: ``xps.CrossSection.extrapolate()``. Two steps:

      1. Energy: log-log polynomial from the tabulated energies to the
         target energy, for each tabulated Z (Z = 1–12).
      2. Atomic number: polynomial in Z of log10(sigma) to the target Z.

    The MATLAB default polyOrder = 6 oscillates (Runge) when extrapolating
    beyond 8048 eV; order 3 is stable.

    The table comes from :func:`resolve_yeh_lindau_source`. Raises
    ``FileNotFoundError`` naming ``DEPPRO_YEH_LINDAU_TABLE_PATH`` if a CSV is
    the source and cannot be found.
    """
    # Step 1: for each Z, extrapolate the cross section to the target energy
    z_vals = []
    sigma_at_pe = []
    for z, pe_values, cs_vals in _yeh_lindau_1s_rows():
        valid = ~np.isnan(cs_vals) & (cs_vals > 0)
        if valid.sum() < 2:
            continue

        pe_valid = pe_values[valid]
        cs_valid = cs_vals[valid]

        # Log-log polynomial fit in energy (low order to avoid Runge)
        order = min(poly_order_energy, len(pe_valid) - 1)
        p = np.polyfit(np.log10(pe_valid), np.log10(cs_valid), order)
        sigma = 10 ** np.polyval(p, np.log10(photon_energy))

        z_vals.append(z)
        sigma_at_pe.append(sigma)

    z_arr = np.array(z_vals)
    sigma_arr = np.array(sigma_at_pe)

    # Step 2: extrapolate from Z = 1–12 to the target Z
    order_z = min(poly_order_z, len(z_arr) - 1)
    p_z = np.polyfit(z_arr, np.log10(sigma_arr), order_z)
    sigma_target = 10 ** np.polyval(p_z, target_z)

    return sigma_target
