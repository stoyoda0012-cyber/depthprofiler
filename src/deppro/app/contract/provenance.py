"""P1 contract: SolverRun / RunManifest and provenance.

A run carries the method, config hash, dataset reference, pixel selection, status, stop reason,
timing category, metrics, observation residual (recorded only), artifact references and the warm-start
lineage. A manifest carries the code revision and dirty state, environment, contract version, metrics
version, command and output hashes. **Other people's working-tree diffs are never snapshotted** (dirty is
a flag and a count only). JSON is committed by tmp → rename. Large arrays go to npz; the manifest holds only the index and aggregates. NaN is null.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .dataset_files import jsonable

CONTRACT_VERSION = "p1/0.1.0"
SCHEMA_VERSION = "run-manifest/0.2.0"   # 0.2.0: appended axes (depth_A and other effective axes).
                                                                                # Default empty, so 0.1.0 manifests still read as they are.
# Versions load_run_manifest accepts. A version not listed here, or a wrong type, is a ValueError, not a silent read (audit B-2).
#   run-manifest/0.1.0: no axes (Atlas sets Å metrics to null or recovers them from depth_step)
#   run-manifest/0.2.0: axes appended (depth_A and other effective axes). The metrics computation reads them
SUPPORTED_SCHEMA_VERSIONS = ("run-manifest/0.1.0", "run-manifest/0.2.0")


def config_hash(obj: Any) -> str:
    """Content hash of the config (sha256 of canonical JSON independent of key order, first 16 digits)."""
    s = json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                   default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_revision(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """git HEAD and dirty state. Neither the diff contents nor paths are recorded (other people's work is not copied)."""
    if repo_root is None:
        from deppro.paths import repo_root as _rr
        repo_root = _rr()
    out: Dict[str, Any] = {"rev": None, "dirty": None, "n_dirty_paths": None, "error": None}
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                             text=True, timeout=10)
        st = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True,
                            text=True, timeout=10)
        if rev.returncode == 0:
            out["rev"] = rev.stdout.strip()
        if st.returncode == 0:
            lines = [ln for ln in st.stdout.splitlines() if ln.strip()]
            out["dirty"] = bool(lines)
            out["n_dirty_paths"] = len(lines)
        if rev.returncode != 0 or st.returncode != 0:
            out["error"] = (rev.stderr or st.stderr).strip()[:200]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def environment() -> Dict[str, Any]:
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "machine": platform.machine(), "system": platform.system(),
           "executable": sys.executable}
    for mod in ("h5py", "scipy", "mlx.core"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            env[mod] = getattr(m, "__version__", "present")
        except Exception:  # noqa: BLE001
            env[mod] = None
    env["deppro_env"] = {k: bool(os.environ.get(k)) for k in
                         ("DEPPRO_SIMULATION_ROOT", "DEPPRO_COMPOUND_TABLE_PATH",
                          "DEPPRO_FIX_IBW_DIR", "DEPPRO_FIX_PXT_DIR", "DEPPRO_MAP_H5",
                          "DEPPRO_OUTPUTS")}   # only whether it is set; the value is not written
    return env


def new_run_id(prefix: str = "run") -> str:
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{ts}-{secrets.token_hex(3)}"   # no collision even within one second


@dataclass
class SolverRun:
    run_id: str
    method: str                       # e.g. "slsqp/semifree", "mlx/semifree", "nn/march_mlp"
    config: Dict[str, Any]
    dataset: Dict[str, Any]           # DatasetRef as dict
    pixel_selection: Dict[str, Any]   # {"n_px", "frames", "pixel_ids_ref"}
    status: str = "pending"           # pending|completed|failed|unresolved|budget_exhausted
    stop_reason: Optional[str] = None
    timing_s: Dict[str, float] = field(default_factory=dict)   # {"solve", "io", "eval", ...}
    metrics: Optional[Dict[str, Any]] = None
    observation_residual: Optional[Dict[str, Any]] = None       # RECORDED only
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    artifact_ref: Optional[str] = None
    warm_start_lineage: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not self.config_hash:
            self.config_hash = config_hash(self.config)


@dataclass
class RunManifest:
    run_id: str
    dataset: Dict[str, Any]
    command: str
    # Effective axes (depth_A holds the real Å values; axes with a non-zero origin or non-uniform spacing are kept as they are).
    # Append-only with an empty default, so an older manifest without them still reads.
    # Not only the display but **the metrics computation** reads these, so they are kept with the run artifacts.
    axes: Dict[str, Any] = field(default_factory=dict)
    runs: List[SolverRun] = field(default_factory=list)
    preflight_ref: Optional[str] = None
    code: Dict[str, Any] = field(default_factory=code_revision)
    environment: Dict[str, Any] = field(default_factory=environment)
    outputs: Dict[str, Dict[str, str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    contract_version: str = CONTRACT_VERSION
    metric_version: Optional[str] = None
    generated: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return jsonable(d)


def save_arrays(out_dir: Path, name: str, **arrays: np.ndarray) -> Dict[str, str]:
    """Save arrays to npz and return the path and sha256 (listed under the manifest's outputs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    return {"path": str(path), "sha256": sha256_file(path)}


def write_run_manifest(manifest: RunManifest, out_dir: Path) -> Path:
    """Commit by tmp → rename. No partial JSON is left behind."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest.run_id}.manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_run_manifest(path: Path) -> Dict[str, Any]:
    """Read a run manifest. ValueError if schema_version is not in the accepted set or a required key has the wrong type.

    The minimum check that stops an automated loop from silently reading a future version or broken JSON.
    The meaning of the contents (metrics version, config contents) is not checked here.
    """
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"{path.name}: run manifest must be a JSON object, got {type(d).__name__}")
    sv = d.get("schema_version")
    if not isinstance(sv, str):
        raise ValueError(f"{path.name}: schema_version must be a string, got {type(sv).__name__}")
    if sv not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"{path.name}: unsupported run manifest schema_version {sv!r}; "
                         f"supported: {', '.join(SUPPORTED_SCHEMA_VERSIONS)}")
    for key, typ in (("run_id", str), ("dataset", dict), ("runs", list)):
        if not isinstance(d.get(key), typ):
            raise ValueError(f"{path.name}: {key!r} must be a JSON {typ.__name__}, got {type(d.get(key)).__name__}")
    # axes was appended in 0.2.0. Missing or null means "effective axes unknown" and is left to the reader (Atlas).
    # If present it must be a JSON object.
    if d.get("axes") is not None and not isinstance(d["axes"], dict):
        raise ValueError(f"{path.name}: 'axes' must be a JSON object, got {type(d['axes']).__name__}")
    return d
