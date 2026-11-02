"""ベンチマーク対象データセットの台帳（含める / 除く判断とその理由・根拠を機械可読に持つ）。

台帳の中身はコードに持たない。JSON ファイル（``schema: deppro-dataset-registry/1``）から読む:

    {"schema": "deppro-dataset-registry/1", "as_of": "YYYY-MM-DD",
     "entries": [{"folder_rel": "Family/name", "status": "INCLUDED", "reason": "...",
                  "evidence": ["<preflight item id>", ...], "notes": "", "identity": null}, ...]}

``load_registry(path)`` で明示的に読むか、``DEPPRO_DATASET_REGISTRY`` にそのパスを設定する。
未設定なら既定の台帳は空で、どのデータセットも「台帳にない」と答える（import は常に通る）。
台帳はデータセットのファイルを移動・削除・修正しない。状態は preflight
（``deppro.diagnostics.dataset_preflight``）の再走査で更新する。

status:
  INCLUDED                   ベンチマーク対象。preflight FAIL なし（WARN は排除理由にしない）
  EXCLUDED                   preflight FAIL。理由を持つ。ファイルはそのまま
  DERIVED                    既存再構成のコピー（_NN/_SLSQP）や派生ビュー。独立標本として数えない
  SEPARATE_REGIME_CANDIDATE  旧 writer 規約（fencepost 格子等）。サイズ系列の比較に入れない
  UNRESOLVED_PROVENANCE      noise ラベル矛盾や prj/スペクトル欠落。由来確認まで対象外
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
    identity: Optional[str] = None      # "main=<主 H5 名>; gen=<generationId>" — フォルダ名は再生成で中身が変わりうる

    @property
    def family(self) -> str:
        return self.folder_rel.split("/")[0]


@dataclass(frozen=True)
class Registry:
    entries: Tuple[DatasetEntry, ...]
    as_of: Optional[str] = None
    source: Optional[str] = None        # 読んだファイルのパス。空の既定台帳なら None

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
    """台帳 JSON を読む。schema が違えば ValueError。"""
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
    """``DEPPRO_DATASET_REGISTRY`` の台帳。未設定（空文字列を含む）なら空の台帳。

    呼ぶたびにファイルを読む（キャッシュしない）。
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
    """ベンチマーク対象 = INCLUDED のみ。"""
    return [e for e in _resolve(registry).entries
            if e.status == INCLUDED and (family is None or e.family == family)]


def excluded(registry: Optional[Registry] = None) -> List[DatasetEntry]:
    """INCLUDED 以外。どれも理由を持つ（``Registry`` の構築時に検査済み）。"""
    return [e for e in _resolve(registry).entries if e.status != INCLUDED]


def explain(folder_rel: str, registry: Optional[Registry] = None) -> str:
    reg = _resolve(registry)
    e = reg.entry(folder_rel)
    if e is None:
        where = f"registry {reg.source} (as of {reg.as_of})" if reg.source else f"registry (none loaded; set {ENV_VAR})"
        return f"{folder_rel}: not in {where}; run {PREFLIGHT_TOOL} first"
    return f"{folder_rel}: {e.status} — {e.reason}" + (f" [{'; '.join(e.evidence)}]" if e.evidence else "")
