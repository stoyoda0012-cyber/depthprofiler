"""データセットフォルダの規約と主 H5 の特定（preflight と contract の共通部品）。

規約: ``{Standard,Checkered}/{Nx}x{Ny}x{Nt}_{N}elem[_variant]/``（benchmarks/paths.py）。
variant の語彙は検証しない（``noisy_cli`` も variant として受ける。差分表の合意）。
主 H5 の規則: ``.prj`` と同じ stem の非スペクトル H5 → 無ければ非スペクトル H5 が 1 本ならそれ →
それでも無ければ producer 命名 ``<6 桁日付><元素記号列>.h5`` に合うものが 1 本ならそれ。
どれにも当たらなければ特定不能。採用した規則は必ず記録する。派生 H5（``_mlx.h5`` 等）は
独立標本として数えない。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

FAMILIES = ("Standard", "Checkered")
FOLDER_RE = re.compile(
    r"^(?P<nx>\d+)x(?P<ny>\d+)x(?P<nt>\d+)_(?P<ncomp>\d+)elem(?:_(?P<variant>.+))?$")
# 元素別スペクトルファイル: Si1s_..., Al1s_..., Ti2p3_... のような接頭辞
SPECTRA_RE = re.compile(r"^[A-Z][a-z]?\d[spdf]\d?_")
# producer の主 H5 名: 6 桁日付 + 元素記号列（例 YYMMDD + OCAlTiSi → 250101OCAlTiSi.h5）
PRODUCER_NAME_RE = re.compile(r"^\d{6}[A-Za-z]+\.h5$")


@dataclass
class FolderSpec:
    family: Optional[str]
    nx: int
    ny: int
    nt: int
    ncomp: int
    variant: Optional[str]

    @property
    def px_per_frame(self) -> int:
        return self.nx * self.ny

    @property
    def n_px(self) -> int:
        return self.nx * self.ny * self.nt


def parse_dataset_folder(folder: Path) -> Optional[FolderSpec]:
    """``{Nx}x{Ny}x{Nt}_{N}elem[_variant]`` を解釈する。合わなければ None。"""
    m = FOLDER_RE.match(Path(folder).name)
    if not m:
        return None
    parent = Path(folder).parent.name
    return FolderSpec(parent if parent in FAMILIES else None,
                      int(m["nx"]), int(m["ny"]), int(m["nt"]), int(m["ncomp"]),
                      m["variant"])


def discover_files(folder: Path) -> Dict[str, Any]:
    """主 H5・派生 H5・元素別スペクトル・prj・manifest を分ける（規則は ``rule`` に記録）。"""
    folder = Path(folder)
    h5s = sorted(folder.glob("*.h5"))
    spectra = [p for p in h5s if SPECTRA_RE.match(p.name)]
    prj = sorted(folder.glob("*.prj"))
    prj_stems = {p.stem for p in prj}
    cands = [p for p in h5s if p not in spectra]
    main = [p for p in cands if p.stem in prj_stems]
    rule = "prj-stem"
    if not main and len(cands) == 1:
        rule, main = "unique-non-spectra", list(cands)
    if not main:
        named = [p for p in cands if PRODUCER_NAME_RE.match(p.name)]
        if len(named) == 1:
            rule, main = "producer-name-pattern", named
    derived = [p for p in cands if p not in main]
    jsons = sorted(folder.glob("*.json"))
    return {
        "main": main, "derived": derived, "spectra": spectra, "prj": prj,
        "rule": rule, "manifest": folder / "manifest.json",
        "derived_view": folder / "derived_view.json",   # dpIngest の GUI 検証用コピーの印
        "other_json": [p for p in jsons if p.name not in ("manifest.json", "derived_view.json")],
    }


def decode_strings(v: Any) -> Any:
    """MATLAB 由来の文字列配列（bytes/object）を Python 文字列リストへ。数値はそのまま。"""
    a = np.asarray(v)
    if a.dtype.kind in "OSU":
        out: List[str] = []
        for x in a.ravel():
            if isinstance(x, bytes):
                x = x.decode("utf-8", "replace")
            out.append(str(x))
        return out
    return a


def matlab_index_list(v: Any) -> List[int]:
    """MATLAB 由来の 1-based インデックス配列（0 は「なし」の番兵）を 1-based の int リストに。"""
    a = np.asarray(v).ravel().astype(int)
    return [int(i) for i in a if i != 0]


def jsonable(o: Any) -> Any:
    """numpy 型を素の Python へ。非有限 float は None（NaN/Inf を JSON に書かない）。"""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o
