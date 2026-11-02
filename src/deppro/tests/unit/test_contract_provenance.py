"""SolverRun / RunManifest / provenance の契約。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from deppro.app.contract import (
    CONTRACT_VERSION, RunManifest, SolverRun, code_revision, config_hash, environment,
    load_run_manifest, new_run_id, save_arrays, write_run_manifest,
)
from deppro.app.contract.provenance import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS, sha256_file


def test_config_hash_is_order_insensitive_and_numpy_safe():
    a = {"alpha": 2.0, "cap": np.float64(40), "loops": [1, 2], "nested": {"x": np.int64(1)}}
    b = {"nested": {"x": 1}, "loops": [1, 2], "cap": 40.0, "alpha": 2.0}
    assert config_hash(a) == config_hash(b)
    assert config_hash({"alpha": 2.0}) != config_hash({"alpha": 2.1})


def test_solver_run_hashes_its_config_and_defaults():
    run = SolverRun(run_id="r1", method="mlx/semifree", config={"alpha": 2.0},
                    dataset={"folder_rel": "Checkered/x"}, pixel_selection={"n_px": 300})
    assert run.config_hash == config_hash({"alpha": 2.0})
    assert run.status == "pending" and run.metrics is None and run.observation_residual is None


def test_manifest_round_trip_is_atomic(tmp_path):
    run = SolverRun(run_id="r1", method="nn/march", config={"model": "x.npz"},
                    dataset={"folder_rel": "Checkered/x"}, pixel_selection={"n_px": 4},
                    status="completed", metrics={"rmse_total": 0.1, "nan_metric": float("nan")},
                    timing_s={"solve": 0.5})
    m = RunManifest(run_id="r1", dataset={"folder_rel": "Checkered/x"}, command="pytest",
                    runs=[run], metric_version="v1")
    path = write_run_manifest(m, tmp_path / "runs")
    assert path.name == "r1.manifest.json" and not list((tmp_path / "runs").glob("*.tmp"))
    d = load_run_manifest(path)
    assert d["schema_version"].startswith("run-manifest/") and d["contract_version"] == CONTRACT_VERSION
    assert d["runs"][0]["metrics"]["nan_metric"] is None       # NaN は null
    assert d["runs"][0]["config_hash"] == run.config_hash
    assert set(d["code"]) >= {"rev", "dirty", "n_dirty_paths", "error"}
    assert "deppro_env" in d["environment"]
    json.dumps(d)


def test_code_revision_records_only_flags_not_diffs():
    c = code_revision()
    assert set(c) == {"rev", "dirty", "n_dirty_paths", "error"}
    assert c["rev"] is None or len(c["rev"]) == 40
    assert not any(isinstance(v, (list, dict)) for v in c.values())   # 差分やパスの列挙はしない


def test_environment_does_not_leak_env_values():
    e = environment()
    assert all(isinstance(v, bool) for v in e["deppro_env"].values())


def test_save_arrays_hash_and_run_id_uniqueness(tmp_path):
    ref = save_arrays(tmp_path, "D_opt", D=np.zeros((2, 3)), ids=np.arange(2))
    assert ref["sha256"] == sha256_file(ref["path"])
    ids = {new_run_id("t") for _ in range(20)}
    assert len(ids) == 20


# ---- schema_version の検査（監査 B-2: 自動ループで将来の版を黙って読まない） --------------------

def _write_manifest(tmp_path, d, name="x.manifest.json"):
    p = tmp_path / name
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_load_run_manifest_accepts_exactly_the_supported_versions(tmp_path):
    base = {"run_id": "r", "dataset": {"folder_rel": "Checkered/x"}, "runs": []}
    # 0.1.0 は axes を持たない（Atlas は Å 指標を null にするか step から復元する）
    p010 = _write_manifest(tmp_path, {**base, "schema_version": "run-manifest/0.1.0"}, "a.manifest.json")
    assert load_run_manifest(p010)["schema_version"] == "run-manifest/0.1.0"
    # 0.2.0 は axes を追記
    p020 = _write_manifest(tmp_path, {**base, "schema_version": "run-manifest/0.2.0",
                                      "axes": {"depth_A": [0.0, 30.0]}}, "b.manifest.json")
    assert load_run_manifest(p020)["axes"]["depth_A"] == [0.0, 30.0]
    # 0.2.0 でも axes が無い／null は通す（実軸不明として Atlas が null 指標にする経路。監査 fixture が使う）
    for i, axes in enumerate((None, "absent")):
        m = {**base, "schema_version": "run-manifest/0.2.0"}
        if axes is None:
            m["axes"] = None
        pth = _write_manifest(tmp_path, m, f"c{i}.manifest.json")
        assert load_run_manifest(pth).get("axes") is None
    # writer が書く版は必ず受理集合に入っている（書いたものが読めない、を防ぐ）
    assert SCHEMA_VERSION in SUPPORTED_SCHEMA_VERSIONS
    assert set(SUPPORTED_SCHEMA_VERSIONS) == {"run-manifest/0.1.0", "run-manifest/0.2.0"}


@pytest.mark.parametrize("bad, needle", [
    ({"schema_version": "run-manifest/9.9.9"}, "9.9.9"),                  # 未知の将来版
    ({"schema_version": "p1/0.1.0"}, "p1/0.1.0"),                          # 別の版体系（contract_version の取り違え）
    ({"schema_version": 123}, "schema_version"),                            # 型不正
    ({"schema_version": None}, "schema_version"),                           # 欠落相当
    ({"schema_version": "run-manifest/0.2.0", "axes": [0.0, 30.0]}, "axes"),  # axes が object でない
    ({"schema_version": "run-manifest/0.1.0", "runs": "nope"}, "runs"),    # 必須キーの型不正
    ({"schema_version": "run-manifest/0.1.0", "dataset": []}, "dataset"),
])
def test_load_run_manifest_rejects_unknown_version_and_bad_types(tmp_path, bad, needle):
    d = {"run_id": "r", "dataset": {"folder_rel": "Checkered/x"}, "runs": [], **bad}
    if bad.get("schema_version", "x") is None:
        del d["schema_version"]
    p = _write_manifest(tmp_path, d)
    with pytest.raises(ValueError, match=needle):
        load_run_manifest(p)


def test_load_run_manifest_rejects_non_object_json(tmp_path):
    p = tmp_path / "list.manifest.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_run_manifest(p)
