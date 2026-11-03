"""Resolution of binding energies.

Two tables are consulted:

* `CoreLevelBindingEnergyCompoundTable_NIST.csv`  (Element, SpectralLine, Formula, BE_eV)
* `BindingEnergyTable.csv`                        (ElementSymbol, 1s, 2s, ...)

**The values of existing MATLAB datasets cannot be fully reconstructed from these two tables alone.**
A species with several rows in the table, with no row, or edited in the GUI
may not be reconstructible from the tables.
Binding energies are editable in the GUI, so they are also per-dataset state. Therefore:

* in normal generation they **can be given explicitly in cfg**, and the origin is recorded in the manifest;
* what can be looked up is looked up, and `compound` / `element` / `explicit` is kept per species;
* "looked up" and "given" are never mixed into one reported number.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class BindingEnergyError(LookupError):
    pass


@dataclass(frozen=True)
class BindingEnergies:
    values: np.ndarray                 # (ne,) eV
    provenance: List[str]              # per species: "compound" | "element" | "explicit"
    detail: List[str]
    sources: Dict[str, Dict[str, object]]


def _hash(p: Path) -> Dict[str, object]:
    raw = p.read_bytes()
    return {"path": str(p), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def compound_be_path() -> Optional[Path]:
    v = os.environ.get("DEPPRO_COMPOUND_BE_TABLE_PATH")
    return Path(v).expanduser() if v else None


def element_be_path() -> Optional[Path]:
    v = os.environ.get("DEPPRO_ELEMENT_BE_TABLE_PATH")
    return Path(v).expanduser() if v else None


def resolve(
    symbols: Sequence[str],
    orbitals: Sequence[str],
    compounds: Sequence[str],
    *,
    explicit: Optional[Sequence[Optional[float]]] = None,
    compound_table: Path | str | None = None,
    element_table: Path | str | None = None,
) -> BindingEnergies:
    """Decide the BE of each species. An `explicit` value, where given, takes precedence."""
    n = len(symbols)
    explicit = list(explicit) if explicit is not None else [None] * n
    if not (len(orbitals) == len(compounds) == len(explicit) == n):
        raise ValueError("symbols/orbitals/compounds/explicit lengths differ")

    sources: Dict[str, Dict[str, object]] = {}
    comp_idx: Dict[tuple, List[float]] = collections.defaultdict(list)
    cp = Path(compound_table).expanduser() if compound_table else compound_be_path()
    if cp and cp.is_file():
        sources["compoundBindingEnergyTable"] = _hash(cp)
        with cp.open(newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                try:
                    comp_idx[(r["Element"], r["SpectralLine"], r["Formula"])].append(float(r["BE_eV"]))
                except (KeyError, ValueError):
                    continue
    elem_idx: Dict[tuple, float] = {}
    ep = Path(element_table).expanduser() if element_table else element_be_path()
    if ep and ep.is_file():
        sources["elementBindingEnergyTable"] = _hash(ep)
        with ep.open(newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                sym = r.get("ElementSymbol")
                for line, v in r.items():
                    if line in ("ElementNumber", "ElementSymbol"):
                        continue
                    try:
                        elem_idx[(sym, line)] = float(v)
                    except (TypeError, ValueError):
                        continue

    values = np.empty(n, dtype=float)
    prov: List[str] = []
    detail: List[str] = []
    missing: List[str] = []
    for i in range(n):
        if explicit[i] is not None:
            values[i] = float(explicit[i]); prov.append("explicit")
            detail.append(f"{symbols[i]} {orbitals[i]} in {compounds[i]}: given in cfg")
            continue
        key = (symbols[i], orbitals[i], compounds[i])
        hits = comp_idx.get(key, [])
        if len(hits) == 1:
            values[i] = hits[0]; prov.append("compound")
            detail.append(f"{key}: unique row in the compound table")
            continue
        if len(hits) > 1:
            # The selection rule is undeclared, so **do not silently take the median or similar**
            missing.append(f"{key}: {len(hits)} compound rows ({min(hits)}..{max(hits)}); "
                           "selection rule is not declared -- pass it in cfg")
            prov.append("unresolved"); detail.append("ambiguous"); values[i] = np.nan
            continue
        ev = elem_idx.get((symbols[i], orbitals[i]))
        if ev is not None:
            values[i] = ev; prov.append("element")
            detail.append(f"{symbols[i]} {orbitals[i]}: element table (no compound row for {compounds[i]})")
            continue
        missing.append(f"{key}: absent from both tables")
        prov.append("unresolved"); detail.append("absent"); values[i] = np.nan
    if missing:
        raise BindingEnergyError(
            "binding energies could not be resolved:\n  " + "\n  ".join(missing)
            + "\n(pass them in cfg as bindingEnergy; they are GUI-editable state, "
              "so the tables are not always authoritative)"
        )
    return BindingEnergies(values=values, provenance=prov, detail=detail, sources=sources)
