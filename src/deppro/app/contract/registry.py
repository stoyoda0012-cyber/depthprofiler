"""Registry of the benchmark datasets (the include / exclude decisions with their reasons and evidence, machine-readable).

The registry contents are not in code. They are read from a JSON file (``schema: deppro-dataset-registry/1``):

    {"schema": "deppro-dataset-registry/1", "as_of": "YYYY-MM-DD",
     "entries": [{"folder_rel": "Family/name", "status": "INCLUDED", "reason": "...",
                  "evidence": ["<preflight item id>", ...], "notes": "", "identity": null}, ...]}

Read it explicitly with ``load_registry(path)``, or set its path in ``DEPPRO_DATASET_REGISTRY``.
With it unset the default registry is empty and every dataset is "not in the registry" (importing always works).
The registry does not move, delete or modify dataset files. Its state is updated by a
preflight rescan (``deppro.diagnostics.dataset_preflight``).

status:
  INCLUDED                   benchmark target. No preflight FAIL (a WARN is not a reason for exclusion)
  EXCLUDED                   preflight FAIL. Carries a reason. Files are left as they are
  DERIVED                    a copy of an existing reconstruction (_NN/_SLSQP) or a derived view. Not counted as an independent sample
  SEPARATE_REGIME_CANDIDATE  old writer conventions (fencepost grid and the like). Not included in the size-series comparison
  UNRESOLVED_PROVENANCE      contradictory noise label, or missing prj/spectra. Excluded until the provenance is confirmed
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ...paths import env_value

SCHEMA = "deppro-dataset-registry/1"
ENV_VAR = "DEPPRO_DATASET_REGISTRY"
PREFLIGHT_TOOL = "deppro.diagnostics.dataset_preflight"

INCLUDED = "INCLUDED"
EXCLUDED = "EXCLUDED"
DERIVED = "DERIVED"
SEPARATE_REGIME_CANDIDATE = "SEPARATE_REGIME_CANDIDATE"
UNRESOLVED_PROVENANCE = "UNRESOLVED_PROVENANCE"
STATUSES = (INCLUDED, EXCLUDED, DERIVED, SEPARATE_REGIME_CANDIDATE, UNRESOLVED_PROVENANCE)


@dataclass(frozen=True)
class DatasetEntry:
    folder_rel: str
    status: str
    reason: str = ""
    evidence: Tuple[str, ...] = ()      # preflight item ids / codes
    notes: str = ""
    identity: Optional[str] = None      # "main=<main H5 name>; gen=<generationId>" — a folder name can change contents on regeneration

    @property
    def family(self) -> str:
        return self.folder_rel.split("/")[0]


@dataclass(frozen=True)
class Registry:
    entries: Tuple[DatasetEntry, ...]
    as_of: Optional[str] = None
    source: Optional[str] = None        # path of the file read. None for the empty default registry

    def __post_init__(self):
        object.__setattr__(self, "entries", tuple(self.entries))
        names = [e.folder_rel for e in self.entries]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"duplicate folder in registry: {dup}")
        bad = sorted({e.status for e in self.entries} - set(STATUSES))
        if bad:
            raise ValueError(f"unknown status in registry: {bad}")
        missing = [e.folder_rel for e in self.entries if e.status != INCLUDED and not e.reason]
        if missing:
            raise ValueError(f"non-INCLUDED entries need a reason: {missing}")

    def entry(self, folder_rel: str) -> Optional[DatasetEntry]:
        return self._by_name().get(folder_rel)

    def _by_name(self) -> Dict[str, DatasetEntry]:
        return {e.folder_rel: e for e in self.entries}


def load_registry(path: Union[str, Path]) -> Registry:
    """Read the registry JSON. ValueError if the schema differs."""
    p = Path(path).expanduser()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        got = data.get("schema") if isinstance(data, dict) else type(data).__name__
        raise ValueError(f"{p}: schema is {got!r}, expected {SCHEMA!r}")
    if not isinstance(data.get("entries"), list):
        raise ValueError(f"{p}: 'entries' must be a list")
    entries = []
    for i, e in enumerate(data["entries"]):
        if not isinstance(e, dict) or "folder_rel" not in e or "status" not in e:
            raise ValueError(f"{p}: entries[{i}] needs 'folder_rel' and 'status'")
        evidence = e.get("evidence", [])
        if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
            raise ValueError(f"{p}: entries[{i}].evidence must be a list of strings")
        entries.append(DatasetEntry(e["folder_rel"], e["status"], e.get("reason", ""), tuple(evidence),
                                    e.get("notes", ""), e.get("identity")))
    return Registry(tuple(entries), data.get("as_of"), str(p))


def default_registry() -> Registry:
    """The registry of ``DEPPRO_DATASET_REGISTRY``. Empty registry when unset (including an empty string).

    The file is read on every call (no caching).
    """
    path = env_value(ENV_VAR)
    if path is None:
        return Registry(())
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"{ENV_VAR} points to {p}, which is not a file")
    return load_registry(p)


def _resolve(registry: Optional[Registry]) -> Registry:
    return default_registry() if registry is None else registry


def entry(folder_rel: str, registry: Optional[Registry] = None) -> Optional[DatasetEntry]:
    return _resolve(registry).entry(folder_rel)


def benchmark_targets(family: Optional[str] = None, registry: Optional[Registry] = None) -> List[DatasetEntry]:
    """Benchmark targets = INCLUDED only."""
    return [e for e in _resolve(registry).entries
            if e.status == INCLUDED and (family is None or e.family == family)]


def excluded(registry: Optional[Registry] = None) -> List[DatasetEntry]:
    """Everything but INCLUDED. Each carries a reason (checked when the ``Registry`` is built)."""
    return [e for e in _resolve(registry).entries if e.status != INCLUDED]


def explain(folder_rel: str, registry: Optional[Registry] = None) -> str:
    reg = _resolve(registry)
    e = reg.entry(folder_rel)
    if e is None:
        where = f"registry {reg.source} (as of {reg.as_of})" if reg.source else f"registry (none loaded; set {ENV_VAR})"
        return f"{folder_rel}: not in {where}; run {PREFLIGHT_TOOL} first"
    return f"{folder_rel}: {e.status} — {e.reason}" + (f" [{'; '.join(e.evidence)}]" if e.evidence else "")
