"""P1 契約: SolverRun / RunManifest と provenance（履歴管理）。

run は method・config hash・データセット参照・画素選択・状態・停止理由・計測区分・指標・
観測残差（記録のみ）・artifact 参照・warm start の系譜を持つ。manifest は code revision と
dirty 状態、環境、契約版、指標版、コマンド、出力の hash を持つ。**他者の作業ツリー差分を
取り込んでスナップショット化しない**（dirty は真偽と件数だけ）。JSON は tmp → rename で確定する。
巨大配列は npz へ、manifest は索引と集約のみ。NaN は null。
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
SCHEMA_VERSION = "run-manifest/0.2.0"   # 0.2.0: 追記の axes（depth_A ほか実効軸）。
                                        # 既定は空なので 0.1.0 の manifest もそのまま読める。
# load_run_manifest が受理する版。ここに無い版・型不正は黙って読まず ValueError にする（監査 B-2）。
#   run-manifest/0.1.0: axes なし（Atlas は Å 指標を null にするか depth_step から復元する）
#   run-manifest/0.2.0: axes（depth_A ほか実効軸）を追記。指標計算がこれを見る
SUPPORTED_SCHEMA_VERSIONS = ("run-manifest/0.1.0", "run-manifest/0.2.0")


def config_hash(obj: Any) -> str:
    """設定の内容ハッシュ（キー順に依存しない canonical JSON の sha256、先頭 16 桁）。"""
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
    """git の HEAD と dirty 状態。差分の中身やパスは記録しない（他者の作業を写さない）。"""
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
                          "DEPPRO_OUTPUTS")}   # 設定有無だけ、値は書かない
    return env


def new_run_id(prefix: str = "run") -> str:
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{ts}-{secrets.token_hex(3)}"   # 同一秒内でも衝突しない


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
    # 実効の軸（depth_A は Å の実値。原点が0でない軸・非一様な軸もそのまま持つ）。
    # 追記のみで既定は空なので、これを持たない古い manifest もそのまま読める。
    # 表示だけでなく **指標計算** がここを見るため、run 成果物に残す。
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
    """配列を npz に保存し、パスと sha256 を返す（manifest の outputs に載せる）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    return {"path": str(path), "sha256": sha256_file(path)}


def write_run_manifest(manifest: RunManifest, out_dir: Path) -> Path:
    """tmp → rename で確定。途中 JSON を残さない。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest.run_id}.manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_run_manifest(path: Path) -> Dict[str, Any]:
    """run manifest を読む。schema_version が受理集合に無い、または必須キーの型が違えば ValueError。

    自動ループが将来の版や壊れた JSON を黙って読み進めないための最小検査。中身の意味（指標の版、
    config の内容）はここでは検査しない。
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
    # axes は 0.2.0 の追記。無い／null は「実軸が分からない」として読み手（Atlas）が扱うので通す。
    # あるなら JSON object でなければならない。
    if d.get("axes") is not None and not isinstance(d["axes"], dict):
        raise ValueError(f"{path.name}: 'axes' must be a JSON object, got {type(d['axes']).__name__}")
    return d
