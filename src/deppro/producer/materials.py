"""Material table -> constants -> lambda_basic.

MATLAB counterpart (`+deppro/+core/DeterminingLambdaBasic.m`):

    lambdabasic(ii,i,k) = xps.IMFP.calculate(KE(i), Nv(ii), rho(ii), Mw(ii), Eg(ii), TPPver)
    KE(i) = photonenergy(k) - bindingenergy(i) - workfunction

`ii` is the **material**, `i` the **emitting element**. Transposing shifts the values, so
the axis names (a, i) of the return value are the contract. TPPver=3 is TPP-2M.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from deppro.imfp import IMFP
from deppro.paths import env_path_or_placeholder

# Column names of CompoundTable.csv. A changed table changes the hash, so the column assumptions live here too.
_COLUMNS = {
    "valence_charge": "ValenceCharge",
    "density": "Density [g/cm3]",
    "molecular_weight": "MolecularWeight [g/mol]",
    "band_gap": "BandGap [eV]",
}

TPP_VERSIONS = {3: "TPP-2M"}


class CompoundTableError(LookupError):
    """The table cannot be read, or a required compound is missing."""


@dataclass(frozen=True)
class MaterialConstants:
    """Material constants of one series, and where they came from.

    `source_sha256` exists so that "which table was used" can be stated afterwards.
    **Never apply the hash of the current table retroactively to files generated earlier**
    (it is no evidence that the table was the same at generation time).
    """

    compounds: tuple
    valence_charge: np.ndarray
    density: np.ndarray
    molecular_weight: np.ndarray
    band_gap: np.ndarray
    source_path: str
    source_sha256: str
    source_bytes: int


def compound_table_path() -> Path:
    """`DEPPRO_COMPOUND_TABLE_PATH`。

    Importing works with it unset; the error, naming the variable, comes when this is called.
    No implicit dependence on a default path under the MATLAB app.
    """
    return env_path_or_placeholder("DEPPRO_COMPOUND_TABLE_PATH")


def read_compound_table(compounds: Sequence[str], path: Path | str | None = None) -> MaterialConstants:
    """Look up the table by compound name; return the constants and the table hash."""
    p = Path(path).expanduser() if path is not None else compound_table_path()
    if not p.is_file():
        raise CompoundTableError(
            f"compound table not found: {p} (set DEPPRO_COMPOUND_TABLE_PATH)"
        )
    raw = p.read_bytes()
    with p.open(newline="", encoding="utf-8-sig") as fh:
        rows = {r["CompoundName"]: r for r in csv.DictReader(fh)}
    missing = [c for c in compounds if c not in rows]
    if missing:
        raise CompoundTableError(f"compounds absent from {p.name}: {missing}")
    cols: Dict[str, np.ndarray] = {}
    for field, column in _COLUMNS.items():
        try:
            cols[field] = np.array([float(rows[c][column]) for c in compounds], dtype=float)
        except KeyError as exc:
            raise CompoundTableError(f"{p.name} has no column {column!r}") from exc
    return MaterialConstants(
        compounds=tuple(compounds),
        source_path=str(p),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_bytes=len(raw),
        **cols,
    )


def lambda_basic(
    constants: MaterialConstants,
    binding_energy: Sequence[float],
    photon_energy: float,
    work_function: float,
    tpp_version: int = 3,
) -> np.ndarray:
    """``L[a, i]`` = IMFP (Å) of a photoelectron of element i traveling through material a.

    Same axis orientation as MATLAB's `lambdabasic(ii,i)`. Not transposed.
    """
    if tpp_version not in TPP_VERSIONS:
        raise NotImplementedError(
            f"TPPver={tpp_version} is not implemented here (have {sorted(TPP_VERSIONS)})"
        )
    be = np.asarray(binding_energy, dtype=float)
    n = len(constants.compounds)
    if be.shape != (n,):
        raise ValueError(f"binding_energy has {be.shape}, expected ({n},)")
    ke = float(photon_energy) - be - float(work_function)
    if np.any(ke <= 0):
        bad = [i for i, v in enumerate(ke) if v <= 0]
        raise ValueError(f"non-positive kinetic energy at element index {bad}")
    out = np.empty((n, n), dtype=float)
    for a in range(n):
        out[a, :] = IMFP.tpp2m(
            ke,
            constants.valence_charge[a],
            constants.density[a],
            constants.molecular_weight[a],
            constants.band_gap[a],
        )
    return out
