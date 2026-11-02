"""ProblemSpec / GroundTruth 読み取り契約: GT 隔離、配置転置、画素選択、内容由来の nt。"""
from __future__ import annotations

import dataclasses
import json
import os

import h5py
import numpy as np
import pytest

from deppro.app.contract import (
    GroundTruth, ProblemSpec, load_existing_result, load_problem, select_pixel_ids,
)
from deppro.tests.unit._synthetic_dataset import STEM, make_dataset


def test_problem_spec_carries_no_ground_truth(tmp_path):
    folder = make_dataset(tmp_path)
    spec, gt = load_problem(folder, frames=[0, 2], root=tmp_path)
    assert isinstance(spec, ProblemSpec) and isinstance(gt, GroundTruth)
    names = {f.name for f in dataclasses.fields(spec)}
    assert "D" not in names and "Dstk" not in names and not hasattr(spec, "D")
    assert spec.roles_declared is False and "Dstk" in spec.roles_assumed
    assert gt.D.shape == (8, 4, 7) and spec.X.shape == (8, 3, 7)
    assert spec.pixel_ids.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
    assert np.array_equal(gt.pixel_ids, spec.pixel_ids)


def test_layout_is_px_depth_elem_and_matches_file(tmp_path):
    folder = make_dataset(tmp_path)
    spec, gt = load_problem(folder, frames=[1])
    with h5py.File(folder / f"{STEM}.h5", "r") as f:
        D_file = np.asarray(f["Dstk"][4:8], dtype=np.float64)          # (px, elem, depth)
        X_file = np.asarray(f["Xstk"][4:8], dtype=np.float64)          # (px, elem, angle)
        T_file = np.asarray(f["ezdeprof/TransMatObs"][()], dtype=np.float64)
    assert np.array_equal(gt.D, np.transpose(D_file, (0, 2, 1)))
    assert np.array_equal(spec.X, np.transpose(X_file, (0, 2, 1)))
    assert np.array_equal(spec.stored_T, np.transpose(T_file, (2, 1, 0)))
    assert spec.stored_T.shape == (3, 4, 7)
    assert spec.lambda_basic.shape == (7, 7) and spec.gamma_eff == 1.0


def test_roles_are_zero_based_and_labelled(tmp_path):
    spec, _ = load_problem(make_dataset(tmp_path), frames=[0])
    e = spec.elements
    assert e.free == [1, 2, 3, 4, 5] and e.linked == [6] and e.const == [] and e.subst == [0]
    assert e.labels == ["Si(0)", "Si(+4)", "Ti(+4)", "Al(+3)", "C(0)", "B(0)", "O(-2)"]
    assert e.linked_const.tolist()[:5] == [3.0, 3.0, 2.5, 1.0, 1.0]


def test_nt_comes_from_content_not_folder_or_dxdydt(tmp_path):
    folder = make_dataset(tmp_path, n_px_override=2 * 2 * 6, dxdydt=[2, 2, 100])
    spec, _ = load_problem(folder, frames=[5])
    assert spec.axes.nt == 6 and spec.ref.nt == 6
    assert spec.pixel_ids.tolist() == [20, 21, 22, 23]
    assert spec.frame_of().tolist() == [5, 5, 5, 5]


def test_select_pixel_ids_rules():
    assert select_pixel_ids(12, 4, pixel_ids=[5, 1, 5, 9]).tolist() == [1, 5, 9]
    assert select_pixel_ids(12, 4, frames=[2], max_px_per_frame=2).tolist() == [8, 9]
    assert select_pixel_ids(12, 4).size == 12
    with pytest.raises(IndexError):
        select_pixel_ids(12, 4, pixel_ids=[12])


def test_fingerprint_is_cheap_and_changes_with_file_stat(tmp_path):
    folder = make_dataset(tmp_path)
    spec1, _ = load_problem(folder, frames=[0])
    spec2, _ = load_problem(folder, frames=[0])
    assert spec1.ref.fingerprint == spec2.ref.fingerprint
    assert "content" not in spec1.ref.fingerprint_kind and "mtime" in spec1.ref.fingerprint_kind
    p = folder / f"{STEM}.h5"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    spec3, _ = load_problem(folder, frames=[0])
    assert spec3.ref.fingerprint != spec1.ref.fingerprint


def test_summary_is_json_serialisable_and_has_no_arrays(tmp_path):
    spec, _ = load_problem(make_dataset(tmp_path), frames=[0], root=tmp_path)
    s = spec.summary()
    json.dumps(s)
    assert s["dataset"]["folder_rel"] == "Checkered/2x2x3_7elem_noisy"
    assert s["X_shape"] == [4, 3, 7] and s["roles_declared"] is False
    assert s["dataset"]["manifest_present"] is False


def test_existing_result_and_without_gt(tmp_path):
    folder = make_dataset(tmp_path)
    spec, gt = load_problem(folder, frames=[0], with_ground_truth=False)
    assert gt is None
    D_prev = load_existing_result(folder, spec.pixel_ids)
    assert D_prev.shape == (4, 4, 7) and not np.any(D_prev)
    with pytest.raises(KeyError):
        load_existing_result(folder, spec.pixel_ids, dataset="Dnope")


def test_unidentifiable_main_raises(tmp_path):
    folder = make_dataset(tmp_path, write_prj=False, extra_main=True,
                          extra_main_name="000001SyntheticStack.h5")
    with pytest.raises(FileNotFoundError):
        load_problem(folder, frames=[0])
