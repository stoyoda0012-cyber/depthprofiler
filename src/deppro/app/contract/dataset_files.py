"""Dataset-folder conventions and identification of the main H5 (shared by preflight and contract).

Convention: ``{Standard,Checkered}/{Nx}x{Ny}x{Nt}_{N}elem[_variant]/`` (benchmarks/paths.py).
The variant vocabulary is not validated (``noisy_cli`` is accepted as a variant; agreed in the diff table).
Rule for the main H5: the non-spectrum H5 with the same stem as the ``.prj`` → else the only non-spectrum H5, if there is one →
else the only file matching the producer naming ``<6-digit date><element symbols>.h5``, if there is one.
If none applies, it cannot be identified. The rule used is always recorded. Derived H5 files (``_mlx.h5`` and the like) are
not counted as independent samples.
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
# Per-element spectrum files: prefixes such as Si1s_..., Al1s_..., Ti2p3_...
SPECTRA_RE = re.compile(r"^[A-Z][a-z]?\d[spdf]\d?_")
# Producer's main H5 name: 6-digit date + element symbols (e.g. YYMMDD + OCAlTiSi → 250101OCAlTiSi.h5)
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
    """Parse ``{Nx}x{Ny}x{Nt}_{N}elem[_variant]``. None if it does not match."""
    m = FOLDER_RE.match(Path(folder).name)
    if not m:
        return None
    parent = Path(folder).parent.name
    return FolderSpec(parent if parent in FAMILIES else None,
                      int(m["nx"]), int(m["ny"]), int(m["nt"]), int(m["ncomp"]),
                      m["variant"])


def discover_files(folder: Path) -> Dict[str, Any]:
    """Separate main H5, derived H5, per-element spectra, prj and manifest (the rule is recorded in ``rule``)."""
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
        "derived_view": folder / "derived_view.json",   # marker of a GUI-check copy made by dpIngest
        "other_json": [p for p in jsons if p.name not in ("manifest.json", "derived_view.json")],
    }


def decode_strings(v: Any) -> Any:
    """MATLAB string arrays (bytes/object) to a list of Python strings. Numbers are left as they are."""
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
    """MATLAB 1-based index arrays (0 is the "none" sentinel) to a list of 1-based ints."""
    a = np.asarray(v).ravel().astype(int)
    return [int(i) for i in a if i != 0]


def jsonable(o: Any) -> Any:
    """numpy types to plain Python. Non-finite floats become None (no NaN/Inf in JSON)."""
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
