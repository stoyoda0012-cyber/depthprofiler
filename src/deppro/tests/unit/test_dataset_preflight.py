"""``deppro.diagnostics.dataset_preflight`` の契約テスト。

実データには依存しない。MATLAB writer と同じレイアウトの小さな合成データセットを
tmp_path に書き、構造・意味・恒等式・閉包・台帳の各判定が期待どおりに出ることを見る。
「壊した入力で落ちる」ことを各判定について 1 つ以上確認する（lesson: hollow gates）。
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from deppro.diagnostics import dataset_preflight as dp

from deppro.tests.unit._synthetic_dataset import STEM, SYMS, make_dataset  # noqa: E402


def _item(rep, iid):
    return next(it for it in rep["items"] if it["id"] == iid)


# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("10x10x200_7elem_noisy", (10, 10, 200, 7, "noisy")),
    ("30x30x200_6elem", (30, 30, 200, 6, None)),
    ("10x10x200_6elem_jitter_noisy", (10, 10, 200, 6, "jitter_noisy")),
    ("50x50x200_7elem_noisy_cli", (50, 50, 200, 7, "noisy_cli")),
])
def test_parse_dataset_folder(tmp_path, name, expected):
    spec = dp.parse_dataset_folder(tmp_path / "Checkered" / name)
    assert (spec.nx, spec.ny, spec.nt, spec.ncomp, spec.variant) == expected
    assert spec.family == "Checkered"
    assert spec.n_px == expected[0] * expected[1] * expected[2]


@pytest.mark.parametrize("name", ["NeighborMEM_Refinement", "_cli_test5", "backup",
                                  "matlab_cli_optimization", "10x10_7elem"])
def test_parse_rejects_non_dataset_folders(tmp_path, name):
    assert dp.parse_dataset_folder(tmp_path / "Checkered" / name) is None


def test_parse_family_none_outside_standard_checkered(tmp_path):
    spec = dp.parse_dataset_folder(tmp_path / "Other" / "2x2x3_7elem")
    assert spec is not None and spec.family is None


# --------------------------------------------------------------------------
def test_clean_dataset_passes_with_expected_unresolved(tmp_path):
    folder = make_dataset(tmp_path)
    rep = dp.run_preflight(folder, root=tmp_path)
    assert rep["verdict"]["overall"] == dp.PASS, rep["verdict"]
    assert rep["verdict"]["fail_ids"] == []
    for iid in ("files.main_h5_unique", "structure.shapes", "structure.folder_vs_h5",
                "meta.element_roles", "meta.depth_grid", "meta.angles",
                "sample.finite", "sample.value_range", "sample.xstk_sum_to_one",
                "sample.dstk_composition_sum", "sample.zero_pixels",
                "files.spectra_vs_symbols", "meta.noise_vs_variant"):
        assert _item(rep, iid)["status"] == dp.PASS, iid
    # dxdydt(3) は TimeSize。manifest が無いので照合先なし = RECORDED（PASS にも FAIL にもしない）
    assert _item(rep, "structure.dxdydt")["status"] == dp.RECORDED
    # legacy dataset: provenance items stay UNRESOLVED, never PASS
    for iid in ("manifest.present", "roles.declared", "noise.version", "meta.angle_reference"):
        assert _item(rep, iid)["status"] == dp.UNRESOLVED, iid
    # closure at px0 with the very T that produced X: float32 rounding only
    cl = _item(rep, "closure.px0_stored_T")
    assert cl["status"] == dp.RECORDED
    assert cl["detail"]["rel_residual_median"] < 1e-4
    assert cl["detail"]["n_entries"] > 0
    # scope is explicit and bounded
    assert rep["scope"]["n_px_checked"] <= rep["scope"]["n_px_total"]
    assert rep["scope"]["frames_checked"] == [0, 1, 2]
    assert rep["folder_rel"] == "Checkered/2x2x3_7elem_noisy"
    json.dumps(rep)   # no numpy types / NaN leak


def test_dxdydt_third_component_is_timesize_not_nt(tmp_path):
    """dxdydt = [nx, ny, TimeSize]。第 3 成分は nt と違ってよい（producer 2026-09-08）。"""
    folder = make_dataset(tmp_path, dxdydt=[2, 2, 100])          # nt は 3、TimeSize は 100
    rep = dp.run_preflight(folder)
    it = _item(rep, "structure.dxdydt")
    assert it["status"] == dp.RECORDED and it["detail"]["code"] == dp.DXDYDT_T_IS_TIMESIZE
    assert it["detail"]["timeSize_from_h5"] == 100 and it["detail"]["nt_from_content"] == 3
    assert rep["verdict"]["overall"] == dp.PASS
    # manifest が axes.timeSize を宣言していれば照合する
    mf = {"schemaVersion": "1.1.0", "generationId": "g", "mainFile": f"/x/{STEM}.h5",
          "resolved": {"axes": {"nx": 2, "ny": 2, "nt": 3, "timeSize": 100}}}
    rep2 = dp.run_preflight(make_dataset(tmp_path, nx=3, dxdydt=[3, 2, 100], manifest=mf | {"resolved": {"axes": {"nx": 3, "ny": 2, "nt": 3, "timeSize": 100}}}))
    assert _item(rep2, "structure.dxdydt")["status"] == dp.PASS
    mf_bad = {"schemaVersion": "1.1.0", "generationId": "g", "mainFile": f"/x/{STEM}.h5",
              "resolved": {"axes": {"nx": 4, "ny": 2, "nt": 3, "timeSize": 7}}}
    rep3 = dp.run_preflight(make_dataset(tmp_path, nx=4, dxdydt=[4, 2, 100], manifest=mf_bad))
    assert _item(rep3, "structure.dxdydt")["status"] == dp.FAIL


def test_dxdydt_grid_mismatch_fails(tmp_path):
    folder = make_dataset(tmp_path, dxdydt=[3, 2, 3])
    rep = dp.run_preflight(folder)
    assert _item(rep, "structure.dxdydt")["status"] == dp.FAIL
    assert rep["verdict"]["overall"] == dp.FAIL


def test_nan_fails_finite(tmp_path):
    folder = make_dataset(tmp_path, nan_at=5)
    rep = dp.run_preflight(folder)
    assert _item(rep, "sample.finite")["status"] == dp.FAIL
    assert rep["verdict"]["overall"] == dp.FAIL
    assert _item(rep, "closure.px0_stored_T")["status"] == dp.NA


def test_folder_pixel_count_mismatch_fails(tmp_path):
    folder = make_dataset(tmp_path, n_px_override=13)
    rep = dp.run_preflight(folder)
    it = _item(rep, "structure.folder_vs_h5")
    assert it["status"] == dp.FAIL
    assert it["detail"]["conditions"]["nx*ny*nt == Dstk.px"] is False
    assert _item(rep, "structure.shapes")["status"] == dp.PASS   # stack==P still holds


def test_two_main_candidates_without_prj_fail(tmp_path):
    # both candidates look like producer output and there is no prj to disambiguate
    folder = make_dataset(tmp_path, write_prj=False, extra_main=True,
                          extra_main_name="000001SyntheticStack.h5")
    rep = dp.run_preflight(folder)
    it = _item(rep, "files.main_h5_unique")
    assert it["status"] == dp.FAIL and it["detail"]["n_main"] == 0
    assert rep["verdict"]["overall"] == dp.FAIL
    assert _item(rep, "manifest.present")["status"] == dp.UNRESOLVED   # still reported


def test_prj_stem_disambiguates_derived_files(tmp_path):
    folder = make_dataset(tmp_path, extra_main=True)
    rep = dp.run_preflight(folder)
    it = _item(rep, "files.main_h5_unique")
    assert it["status"] == dp.PASS and it["detail"]["rule"] == "prj-stem"
    assert it["detail"]["derived"] == ["other_result.h5"]
    assert rep["verdict"]["overall"] == dp.PASS


def test_manifest_present_passes(tmp_path):
    folder = make_dataset(tmp_path, manifest={"schema_version": "0.1", "dataset_id": "x"})
    rep = dp.run_preflight(folder)
    it = _item(rep, "manifest.present")
    assert it["status"] == dp.PASS and it["detail"]["schemaVersion"] == "0.1"
    assert it["detail"]["generationId"] is None          # 合成 manifest に generationId は無い


def test_unreadable_manifest_fails(tmp_path):
    folder = make_dataset(tmp_path)
    (folder / "manifest.json").write_text("{not json")
    rep = dp.run_preflight(folder)
    assert _item(rep, "manifest.present")["status"] == dp.FAIL


def test_all_zero_pixel_is_unresolved_and_excluded_from_sum(tmp_path):
    folder = make_dataset(tmp_path, zero_px=7)
    rep = dp.run_preflight(folder)
    zp = _item(rep, "sample.zero_pixels")
    assert zp["status"] == dp.UNRESOLVED and zp["detail"]["all_zero_Dstk"] == 1
    assert _item(rep, "sample.dstk_composition_sum")["status"] == dp.PASS
    assert all(it["id"] != "sample.gt_all_zero" for it in rep["items"])
    assert rep["verdict"]["overall"] == dp.PASS


def test_spectra_count_mismatch_is_warn(tmp_path):
    folder = make_dataset(tmp_path, spectra=False)
    rep = dp.run_preflight(folder)
    assert _item(rep, "files.spectra_vs_symbols")["status"] == dp.WARN
    assert rep["verdict"]["overall"] == dp.PASS


def test_broken_element_roles_fail(tmp_path):
    folder = make_dataset(tmp_path)
    with h5py.File(folder / f"{STEM}.h5", "r+") as f:
        del f["ezdeprof/subst"]
        f["ezdeprof/subst"] = np.array([[2]], dtype=np.uint8)   # duplicates a free index
    rep = dp.run_preflight(folder)
    it = _item(rep, "meta.element_roles")
    assert it["status"] == dp.FAIL and it["detail"]["disjoint"] is False


def test_depth_grid_violation_fails(tmp_path):
    folder = make_dataset(tmp_path)
    with h5py.File(folder / f"{STEM}.h5", "r+") as f:
        f["ezdeprof/depth"][0, 1] = 45.0
    rep = dp.run_preflight(folder)
    assert _item(rep, "meta.depth_grid")["status"] == dp.FAIL


def test_missing_required_dataset_fails_early(tmp_path):
    folder = make_dataset(tmp_path)
    with h5py.File(folder / f"{STEM}.h5", "r+") as f:
        del f["Xstk"]
    rep = dp.run_preflight(folder)
    it = _item(rep, "structure.datasets_present")
    assert it["status"] == dp.FAIL and it["detail"]["missing_datasets"] == ["Xstk"]


def test_producer_name_pattern_disambiguates_without_prj(tmp_path):
    folder = make_dataset(tmp_path, write_prj=False, extra_main=True)   # other_result.h5
    rep = dp.run_preflight(folder)
    it = _item(rep, "files.main_h5_unique")
    assert it["status"] == dp.PASS and it["detail"]["rule"] == "producer-name-pattern"
    assert _item(rep, "files.prj_present")["status"] == dp.WARN


def test_dstk_sum_not_one_is_unresolved_not_fail(tmp_path):
    folder = make_dataset(tmp_path, d_scale=(2, 1.05))
    rep = dp.run_preflight(folder)
    it = _item(rep, "sample.dstk_composition_sum")
    assert it["status"] == dp.UNRESOLVED
    assert it["detail"]["code"] == dp.DSTK_SUM_EXCEEDS_1_LOCALIZED
    assert it["detail"]["depth_indices_deviating"] == [2]
    assert it["detail"]["depth_A_deviating"] == [60.0]
    assert abs(it["detail"]["max_dev"] - 0.05) < 1e-3
    assert _item(rep, "sample.xstk_sum_to_one")["status"] == dp.PASS
    assert rep["verdict"]["overall"] == dp.PASS      # unresolved is not failure


def test_dstk_sum_below_one_everywhere_is_not_localized(tmp_path):
    folder = make_dataset(tmp_path)
    with h5py.File(folder / f"{STEM}.h5", "r+") as f:
        f["Dstk"][...] = f["Dstk"][...] * 0.9
    rep = dp.run_preflight(folder)
    it = _item(rep, "sample.dstk_composition_sum")
    assert it["status"] == dp.UNRESOLVED and it["detail"]["code"] == dp.DSTK_SUM_NOT_1


def test_gt_all_zero_fails(tmp_path):
    folder = make_dataset(tmp_path, zero_all_D=True)
    rep = dp.run_preflight(folder)
    it = _item(rep, "sample.gt_all_zero")
    assert it["status"] == dp.FAIL and it["detail"]["all_zero_Dstk"] == rep["scope"]["n_px_checked"]
    assert _item(rep, "sample.xstk_sum_to_one")["status"] == dp.PASS
    assert rep["verdict"]["overall"] == dp.FAIL


def test_noise_variant_contradiction_warns(tmp_path):
    folder = make_dataset(tmp_path, variant="noisy", noise=0.0)
    rep = dp.run_preflight(folder)
    it = _item(rep, "meta.noise_vs_variant")
    assert it["status"] == dp.WARN and it["detail"]["code"] == dp.NOISE_VARIANT_CONTRADICTION
    folder2 = make_dataset(tmp_path, variant=None, noise=1.0, nx=3)
    assert _item(dp.run_preflight(folder2), "meta.noise_vs_variant")["status"] == dp.WARN


def test_fencepost_depth_grid_is_warn(tmp_path):
    folder = make_dataset(tmp_path, fencepost=True)
    rep = dp.run_preflight(folder)
    it = _item(rep, "meta.depth_grid")
    assert it["status"] == dp.WARN and it["detail"]["code"] == dp.DEPTH_GRID_FENCEPOST_LEGACY
    assert "(Z-1)" in it["detail"]["totaldepth_convention"]
    assert rep["verdict"]["overall"] == dp.PASS


def test_generation_progress_is_recorded_only(tmp_path):
    folder = make_dataset(tmp_path, currentstack=1)
    rep = dp.run_preflight(folder)
    it = _item(rep, "meta.generation_progress")
    assert it["status"] == dp.RECORDED and it["detail"]["code"] == dp.GEN_PROGRESS_MISMATCH
    assert it["detail"]["equal"] is False and rep["verdict"]["overall"] == dp.PASS
    folder3 = make_dataset(tmp_path, currentstack=12, nx=2, ny=2, nt=3, variant="x")  # 12 == stack
    it3 = _item(dp.run_preflight(folder3), "meta.generation_progress")
    assert it3["status"] == dp.RECORDED and it3["detail"]["equal"] is True


def test_dxdydt_reference_is_content_not_folder(tmp_path):
    # file holds twice the frames the folder name and dxdydt claim (the 200x200 case)
    folder = make_dataset(tmp_path, n_px_override=2 * 2 * 6, dxdydt=[2, 2, 3])
    rep = dp.run_preflight(folder)
    dx = _item(rep, "structure.dxdydt")
    assert dx["status"] == dp.RECORDED and dx["detail"]["nt_from_content"] == 6   # dxdydt は nt を主張しない
    fv = _item(rep, "structure.folder_vs_h5")
    assert fv["status"] == dp.FAIL and fv["detail"]["h5_nt_from_content"] == 6


def test_invalid_hdf5_signature_fails(tmp_path):
    folder = make_dataset(tmp_path)
    (folder / f"Zr3d_{STEM}.h5").write_text("not an hdf5 file")     # spectra-named, wrong content
    rep = dp.run_preflight(folder)
    it = _item(rep, "files.hdf5_signature")
    assert it["status"] == dp.FAIL and it["detail"]["invalid"] == [f"Zr3d_{STEM}.h5"]
    assert _item(rep, "structure.shapes")["status"] == dp.PASS      # main file still checked


def test_invalid_main_hdf5_returns_early_without_error(tmp_path):
    folder = make_dataset(tmp_path)
    (folder / f"{STEM}.h5").write_text("truncated")
    rep = dp.run_preflight(folder)
    assert rep["verdict"]["overall"] == dp.FAIL
    assert "files.hdf5_signature" in rep["verdict"]["fail_ids"]
    assert rep["verdict"]["error_ids"] == []


def test_closure_records_obs_pixel_identity_and_scale_fit(tmp_path):
    folder = make_dataset(tmp_path)
    with h5py.File(folder / f"{STEM}.h5", "r+") as f:
        f["ezdeprof/Xobs"] = f["Xstk"][0]
        f["ezdeprof/Dobs"] = f["Dstk"][0]
    rep = dp.run_preflight(folder)
    d = _item(rep, "closure.px0_stored_T")["detail"]
    assert d["obs_pixel_identity"] == {"Xobs_equals_Xstk0": True, "Dobs_equals_Dstk0": True}
    assert d["after_per_element_scale"]["rel_residual_median"] < 1e-4
    assert len(d["rel_residual_by_angle_max"]) == 3 and set(d["rel_residual_by_element_max"]) == set(SYMS)


def test_derived_view_copy_is_flagged_not_treated_as_dataset(tmp_path):
    folder = make_dataset(tmp_path, spectra=False, write_prj=False)
    (folder / "derived_view.json").write_text(json.dumps(
        {"kind": "derived_view", "purpose": "human GUI verification",
         "parent": {"folder": "/x/parent", "generationId": "20260907T000000000-abcdef"}}))
    rep = dp.run_preflight(folder)
    assert rep["dataset_kind"] == "derived_view"
    it = _item(rep, "files.derived_view")
    assert it["status"] == dp.WARN and it["detail"]["parent_generationId"] == "20260907T000000000-abcdef"
    assert _item(rep, "manifest.present")["status"] == dp.UNRESOLVED
    folder2 = make_dataset(tmp_path, nx=3)
    assert dp.run_preflight(folder2)["dataset_kind"] == "dataset"


def test_manifest_identity_and_main_file_consistency(tmp_path):
    good = {"schemaVersion": "1.1.0", "generationId": "20260907T123456789-0abcde", "mainFile": f"/any/where/{STEM}.h5",
            "confirmed": {"status": "ok"}}
    folder = make_dataset(tmp_path, manifest=good)
    rep = dp.run_preflight(folder)
    it = _item(rep, "manifest.present")
    assert it["detail"]["schemaVersion"] == "1.1.0" and it["detail"]["generationId"].startswith("20260907T")
    assert it["detail"]["confirmed"] == "ok" and rep["files"]["manifest_generationId"] == good["generationId"]
    assert _item(rep, "manifest.main_matches")["status"] == dp.PASS
    bad = dict(good, mainFile="/any/where/other.h5")
    rep2 = dp.run_preflight(make_dataset(tmp_path, nx=3, manifest=bad))
    assert _item(rep2, "manifest.main_matches")["status"] == dp.FAIL


def test_producer_role_declaration_is_compared_with_the_assumption(tmp_path):
    decl = {"schemaVersion": "1.1.0", "generationId": "g", "mainFile": f"/x/{STEM}.h5",
            "resolved": {"roleDeclarations": {"declaredBy": "producer", "groundTruth": {"array": "/Dstk"},
                                              "stillUndeclared": []},
                         "datasets": [{"name": "Xstk", "role": "observation side (ground truth)"}]}}
    rep = dp.run_preflight(make_dataset(tmp_path, manifest=decl))
    it = _item(rep, "roles.declared")
    assert it["status"] == dp.PASS and it["detail"]["producer_groundTruth_array"] == "/Dstk"
    assert it["detail"]["producer_roles"]["Xstk"].startswith("observation")
    decl2 = json.loads(json.dumps(decl)); decl2["resolved"]["roleDeclarations"]["groundTruth"]["array"] = "/Dtrn"
    rep2 = dp.run_preflight(make_dataset(tmp_path, nx=3, manifest=decl2))
    assert _item(rep2, "roles.declared")["status"] == dp.WARN
    rep3 = dp.run_preflight(make_dataset(tmp_path, nx=4))          # no manifest
    assert _item(rep3, "roles.declared")["status"] == dp.UNRESOLVED


def test_closure_gate_item_pass_fail_recorded(tmp_path):
    folder = make_dataset(tmp_path, noise=0.0, variant=None, x_from_composition=True)
    rep = dp.run_preflight(folder)
    it = _item(rep, "closure.per_pixel_T")
    assert it["status"] == dp.PASS and it["detail"]["max_abs"] <= it["detail"]["tol_abs"]
    assert it["detail"]["pixel_ranges"] and it["detail"]["gt_renormalised"] is False
    assert _item(rep, "closure.px0_stored_T")["status"] == dp.RECORDED
    assert rep["verdict"]["overall"] == dp.PASS
    # noise 0 but X not from the producer forward -> FAIL (and the dataset verdict becomes FAIL)
    rep2 = dp.run_preflight(make_dataset(tmp_path, nx=3, noise=0.0, variant=None, x_from_composition=False))
    assert _item(rep2, "closure.per_pixel_T")["status"] == dp.FAIL and rep2["verdict"]["overall"] == dp.FAIL
    # noise > 0 -> recorded only, verdict unaffected
    rep3 = dp.run_preflight(make_dataset(tmp_path, nx=4, noise=1.0, x_from_composition=True))
    assert _item(rep3, "closure.per_pixel_T")["status"] == dp.RECORDED and rep3["verdict"]["overall"] == dp.PASS


# --------------------------------------------------------------------------
def test_scan_root_and_cli_ledger(tmp_path):
    make_dataset(tmp_path)
    make_dataset(tmp_path, family="Standard", nx=3, ny=1, nt=2, variant=None)
    (tmp_path / "Checkered" / "_cli_test").mkdir()
    (tmp_path / "Checkered" / "NeighborMEM_Refinement").mkdir()
    out = tmp_path / "out"
    rc = dp.main(["--all", "--root", str(tmp_path), "--out-dir", str(out), "--quiet"])
    assert rc == 0
    ledger = json.loads((out / "preflight_ledger.json").read_text())
    assert len(ledger["reports"]) == 2
    skipped = {d["folder"] for d in ledger["not_scanned"]}
    assert skipped == {"Checkered/_cli_test", "Checkered/NeighborMEM_Refinement"}
    md = (out / "preflight_ledger.md").read_text()
    assert "Checkered/2x2x3_7elem_noisy" in md and "Standard/3x1x2_7elem" in md
    assert (out / "preflight_Checkered__2x2x3_7elem_noisy.json").exists()


def test_cli_exit_code_reflects_fail(tmp_path):
    folder = make_dataset(tmp_path, nan_at=0)
    rc = dp.main([str(folder), "--out-dir", str(tmp_path / "out"), "--quiet"])
    assert rc == 1


def test_cli_missing_root_returns_2(tmp_path):
    rc = dp.main(["--all", "--root", str(tmp_path / "nowhere"), "--quiet"])
    assert rc == 2
