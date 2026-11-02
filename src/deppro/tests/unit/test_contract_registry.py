"""台帳を読む仕組み: 架空の小さな台帳で、登録・除外・重複防止・既定の空台帳を確かめる。"""
from __future__ import annotations

import json

import pytest

from deppro.app.contract import registry as R
from deppro.app.contract import benchmark_targets, default_registry, entry, excluded, explain, load_registry

ENTRIES = [
    {"folder_rel": "Alpha/10x10x20_3elem", "status": "INCLUDED", "reason": "preflight PASS",
     "evidence": ["closure.per_pixel_T:PASS"], "identity": "main=alpha.h5; gen=0001"},
    {"folder_rel": "Alpha/10x10x20_3elem_noisy", "status": "INCLUDED", "reason": "preflight PASS"},
    {"folder_rel": "Alpha/5x5x10_3elem", "status": "EXCLUDED", "reason": "main H5 missing",
     "evidence": ["files.main_h5_unique:FAIL"]},
    {"folder_rel": "Beta/10x10x20_4elem", "status": "SEPARATE_REGIME_CANDIDATE", "reason": "fencepost depth grid",
     "evidence": ["meta.depth_grid:WARN"]},
    {"folder_rel": "Beta/10x10x20_4elem_view", "status": "DERIVED", "reason": "derived copy of the Beta main H5"},
]


def _write(path, entries=ENTRIES, schema=R.SCHEMA, as_of="2001-01-01"):
    path.write_text(json.dumps({"schema": schema, "as_of": as_of, "entries": entries}), encoding="utf-8")
    return path


@pytest.fixture
def ledger(tmp_path):
    return load_registry(_write(tmp_path / "ledger.json"))


def test_load_keeps_order_and_fields(ledger):
    assert [e.folder_rel for e in ledger.entries] == [e["folder_rel"] for e in ENTRIES]
    a = ledger.entry("Alpha/10x10x20_3elem")
    assert a.status == R.INCLUDED and a.evidence == ("closure.per_pixel_T:PASS",)
    assert a.identity == "main=alpha.h5; gen=0001" and a.family == "Alpha"
    assert ledger.as_of == "2001-01-01" and ledger.source.endswith("ledger.json")


def test_targets_are_included_only(ledger):
    assert {e.folder_rel for e in benchmark_targets(registry=ledger)} == {
        "Alpha/10x10x20_3elem", "Alpha/10x10x20_3elem_noisy"}
    assert benchmark_targets(family="Beta", registry=ledger) == []
    assert {e.status for e in excluded(registry=ledger)} == {
        R.EXCLUDED, R.SEPARATE_REGIME_CANDIDATE, R.DERIVED}


def test_explain(ledger):
    assert explain("Alpha/5x5x10_3elem", registry=ledger).startswith("Alpha/5x5x10_3elem: EXCLUDED — main H5 missing")
    assert "[files.main_h5_unique:FAIL]" in explain("Alpha/5x5x10_3elem", registry=ledger)
    msg = explain("Alpha/nope", registry=ledger)
    assert "not in registry" in msg and "2001-01-01" in msg and R.PREFLIGHT_TOOL in msg
    assert entry("Alpha/nope", registry=ledger) is None


@pytest.mark.parametrize("bad, match", [
    (ENTRIES + [ENTRIES[0]], "duplicate folder"),
    ([{"folder_rel": "X/y", "status": "MAYBE", "reason": "r"}], "unknown status"),
    ([{"folder_rel": "X/y", "status": "EXCLUDED"}], "need a reason"),
    ([{"status": "INCLUDED"}], "needs 'folder_rel' and 'status'"),
    ([{"folder_rel": "X/y", "status": "INCLUDED", "evidence": "closure.per_pixel_T:PASS"}], "list of strings"),
])
def test_invalid_ledgers_are_rejected(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        load_registry(_write(tmp_path / "bad.json", bad))


def test_entries_are_immutable(ledger):
    assert isinstance(ledger.entries, tuple) and isinstance(ledger.entries[0].evidence, tuple)


def test_default_registry_rereads_the_file(tmp_path, monkeypatch):
    path = _write(tmp_path / "ledger.json")
    monkeypatch.setenv(R.ENV_VAR, str(path))
    assert entry("Alpha/5x5x10_3elem").status == R.EXCLUDED
    _write(path, [dict(ENTRIES[2], status="INCLUDED")])
    assert entry("Alpha/5x5x10_3elem").status == R.INCLUDED


def test_schema_is_checked(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        load_registry(_write(tmp_path / "old.json", schema="something-else/1"))


@pytest.mark.parametrize("value", [None, ""])
def test_default_registry_is_empty_when_unset(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(R.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(R.ENV_VAR, value)
    assert default_registry().entries == ()
    assert benchmark_targets() == [] and entry("Alpha/10x10x20_3elem") is None
    assert R.ENV_VAR in explain("Alpha/10x10x20_3elem")


def test_default_registry_reads_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(R.ENV_VAR, str(_write(tmp_path / "env_ledger.json")))
    assert [e.folder_rel for e in benchmark_targets()] == ["Alpha/10x10x20_3elem", "Alpha/10x10x20_3elem_noisy"]


def test_missing_file_fails_at_use_with_the_variable_name(tmp_path, monkeypatch):
    monkeypatch.setenv(R.ENV_VAR, str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError, match=R.ENV_VAR):
        entry("Alpha/10x10x20_3elem")
