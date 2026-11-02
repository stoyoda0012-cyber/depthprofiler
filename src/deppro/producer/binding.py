"""結合エネルギーの解決。

2 つの表を引く:

* `CoreLevelBindingEnergyCompoundTable_NIST.csv`  (Element, SpectralLine, Formula, BE_eV)
* `BindingEnergyTable.csv`                        (ElementSymbol, 1s, 2s, ...)

**既存 MATLAB データセットの値は、この 2 表だけからは完全には再構成できない。**
表に複数の行がある、行が無い、あるいは GUI で編集された化学種は、
表だけからは再構成できないことがある。
結合エネルギーは GUI で編集できる値なので、データセットごとの状態でもある。したがって:

* 通常生成では **cfg で明示指定**でき、由来を manifest に記録する。
* 表から引けたものは引き、化学種ごとに `compound` / `element` / `explicit` を残す。
* 「表から引いた」と「与えられた」を混ぜて 1 つの数字として報告しない。
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
    provenance: List[str]              # 化学種ごとに "compound" | "element" | "explicit"
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
    """化学種ごとに BE を決める。`explicit` が与えられた要素はそれを最優先する。"""
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
            # 選択規則が未宣言なので、**黙って中央値などを取らない**
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
