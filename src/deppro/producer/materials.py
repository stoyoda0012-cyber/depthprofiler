"""材料表 -> 定数 -> lambda_basic。

MATLAB 側の対応（`+deppro/+core/DeterminingLambdaBasic.m`）:

    lambdabasic(ii,i,k) = xps.IMFP.calculate(KE(i), Nv(ii), rho(ii), Mw(ii), Eg(ii), TPPver)
    KE(i) = photonenergy(k) - bindingenergy(i) - workfunction

`ii` が**物質**、`i` が**放出元素**。転置すると値がずれるので、
戻り値の軸名 (a, i) をそのまま契約とする。TPPver=3 が TPP-2M。
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

# CompoundTable.csv の列名。表が変われば hash が変わるので、列名の想定もここに置く。
_COLUMNS = {
    "valence_charge": "ValenceCharge",
    "density": "Density [g/cm3]",
    "molecular_weight": "MolecularWeight [g/mol]",
    "band_gap": "BandGap [eV]",
}

TPP_VERSIONS = {3: "TPP-2M"}


class CompoundTableError(LookupError):
    """表が読めない、または必要な化合物が無い。"""


@dataclass(frozen=True)
class MaterialConstants:
    """1 系列ぶんの材料定数と、その出所。

    `source_sha256` は「どの表を使ったか」を後から言えるようにするためのもの。
    **過去に生成されたファイルへ遡及して現在の表の hash を当ててはいけない**
    （生成時点の表が同じだったことの証拠にはならない）。
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

    未設定でも import は通り、ここを呼んだ段で変数名を含むエラーになる。
    MATLAB アプリ配下の既定パスへ暗黙に依存しない。
    """
    return env_path_or_placeholder("DEPPRO_COMPOUND_TABLE_PATH")


def read_compound_table(compounds: Sequence[str], path: Path | str | None = None) -> MaterialConstants:
    """化合物名で表を引き、定数と表の hash を返す。"""
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
    """``L[a, i]`` = 元素 i の光電子が物質 a 中を進むときの IMFP (Å)。

    軸の向きは MATLAB の `lambdabasic(ii,i)` と同じ。転置しない。
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
