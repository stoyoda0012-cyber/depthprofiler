"""既存指標の等価性（定義を変えていないことの固定）と、記録のみの観測残差。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from deppro.app.contract import (
    CLOSURE_TOL_ABS_NOISEFREE, METRIC_VERSION, closure_gate, composition_transfer_matrix, evaluate,
    forward, load_problem, observation_residual, thickness_array, w1_matrix,
)
from deppro.depth_profile import wasserstein_distance_1d
from deppro.tests.unit._synthetic_dataset import make_dataset


def test_w1_matrix_equals_canonical_per_pixel_element():
    rng = np.random.default_rng(1)
    n, Z, E = 5, 12, 4
    depth = np.arange(Z) * 30.0
    Dgt = rng.random((n, Z, E)); D = rng.random((n, Z, E))
    Dgt[:2, :, 3] = 0.0                         # GT 質量ゼロ → NaN（deppro 慣例）
    w = w1_matrix(D, Dgt, depth)
    for p in range(n):
        for e in range(E):
            ref = wasserstein_distance_1d(depth, Dgt[p, :, e], D[p, :, e])
            if np.isnan(ref):
                assert np.isnan(w[p, e])
            else:
                assert abs(w[p, e] - ref) < 1e-12, (p, e, w[p, e], ref)


def test_thickness_rule_equals_benchmark_dstk():
    from deppro.io.dstk import estimate_thickness
    rng = np.random.default_rng(2)
    depth = np.arange(20) * 30.0
    D = rng.random((6, 20, 3))
    D[:, :, 0] = np.clip(np.linspace(0, 1.2, 20)[None, :] + 0.1 * rng.standard_normal((6, 20)), 0, 1)
    D[5, :, 0] = 0.2                            # 交差なし → depth[-1]
    D[4, 0, 0] = 0.9                            # 先頭で既に超える → depth[0]
    mine = thickness_array(D, depth, 0)
    ref = np.array([estimate_thickness(D[q], depth, 0) for q in range(6)])
    assert np.allclose(mine, ref, atol=0, rtol=0)


def test_composition_transfer_matrix_equals_deppro_kernels():
    from deppro.transfer_matrix import _calculate_lambda_obs_numba, _calculate_trans_mat_numba
    rng = np.random.default_rng(3)
    n, Z, E, A = 3, 6, 4, 5
    D = rng.random((n, Z, E)); D /= D.sum(axis=2, keepdims=True)
    D[1, 2, :] = 0.0                            # 総和ゼロの深さ → 1e-10 分岐も一致すること
    lam_b = rng.random((E, E)) * 20 + 10
    angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
    T = composition_transfer_matrix(D, lam_b, angles, 30.0, 1.3)
    assert T.shape == (n, A, Z, E)
    for p in range(n):
        lam_obs = _calculate_lambda_obs_numba(np.ascontiguousarray(D[p]), lam_b, 1.3)
        ref = _calculate_trans_mat_numba(lam_obs, np.deg2rad(angles), 30.0)
        assert np.allclose(T[p], ref, rtol=1e-12, atol=0)


def test_forward_is_angle_normalised_for_shared_and_per_pixel_T():
    rng = np.random.default_rng(4)
    D = rng.random((3, 6, 4)); T3 = rng.random((5, 6, 4)); T4 = rng.random((3, 5, 6, 4))
    for T in (T3, T4):
        X = forward(T, D)
        assert X.shape == (3, 5, 4) and np.allclose(X.sum(axis=2), 1.0)


def test_evaluate_identity_and_keys(tmp_path):
    spec, gt = load_problem(make_dataset(tmp_path), frames=[0, 1])
    m = evaluate(gt.D, gt, spec, per_frame=True)
    assert m["metric_version"] == METRIC_VERSION and m["gt_renormalised"] is False
    assert m["rmse_total"] == 0.0 and m["thickness_mae_A"] == 0.0
    assert m["free_indices"] == [1, 2, 3, 4, 5] and m["substrate_index"] == 0
    for k in ("rmse_free5_mean", "w1_medianpx_free5_A", "w1_meanpx_free5_A",
              "w1_valid_px_per_element", "w1_medianpx_per_element_A"):
        assert k in m
    assert all(v == 8 for v in m["w1_valid_px_per_element"].values())
    # JSON 契約: per_frame のキーはフレーム番号の文字列
    assert set(m["per_frame"]) == {"0", "1"} and m["per_frame"]["0"]["n_pixels"] == 4
    json.dumps(m)


def test_evaluate_rmse_formula_and_nan_to_null(tmp_path):
    spec, gt = load_problem(make_dataset(tmp_path), frames=[0])
    rng = np.random.default_rng(5)
    D_est = np.clip(gt.D + 0.05 * rng.standard_normal(gt.D.shape), 0, None)
    m = evaluate(D_est, gt, spec)
    assert abs(m["rmse_total"] - float(np.sqrt(np.mean((D_est - gt.D) ** 2)))) < 1e-15
    gt.D[:, :, 4] = 0.0                          # C(0) の GT を空にする → W1 は null、valid 0
    m2 = evaluate(D_est, gt, spec)
    assert m2["w1_meanpx_per_element_A"]["C(0)"] is None
    assert m2["w1_valid_px_per_element"]["C(0)"] == 0
    json.dumps(m2)


def test_evaluate_refuses_mismatched_pixels(tmp_path):
    folder = make_dataset(tmp_path)
    spec, _ = load_problem(folder, frames=[0])
    _, gt_other = load_problem(folder, frames=[1])
    with pytest.raises(ValueError):
        evaluate(gt_other.D, gt_other, spec)


def test_observation_residual_is_recorded_only(tmp_path):
    spec, gt = load_problem(make_dataset(tmp_path), frames=[0, 2])
    r = observation_residual(spec, gt.D, mode="stored_T")
    assert r["status"] == "RECORDED" and r["formula_version"].startswith("producer-declared/1")
    assert "closure_gate" in r["note"] and "pass" not in {k.lower() for k in r}
    assert r["rel_residual_median"] < 1e-4          # X は同じ T から作ったので丸めだけ
    assert len(r["rel_residual_by_angle_max"]) == 3
    r2 = observation_residual(spec, gt.D, mode="composition_T")
    assert r2["status"] == "RECORDED" and r2["rel_residual_median"] > 1e-3   # 別の T → 残差は残る
    spec.lambda_basic = None
    assert observation_residual(spec, gt.D, mode="composition_T")["status"] == "NA"
    json.dumps(r); json.dumps(r2)


# --------------------------------------------------------------------------
def test_closure_gate_passes_for_producer_forward_at_noise_zero(tmp_path):
    """生成式 (producer-declared/1) で作った X は float32 格納精度で再現される → PASS。"""
    spec, gt = load_problem(make_dataset(tmp_path, noise=0.0, x_from_composition=True), frames=[0, 2])
    g = closure_gate(spec, gt.D)
    assert g["status"] == "PASS" and g["max_abs"] <= CLOSURE_TOL_ABS_NOISEFREE and g["n_px_over_tol"] == 0
    assert g["max_abs"] <= 20 * g["float32_floor_abs"] and g["gt_renormalised"] is False
    assert g["formula_version"].startswith("producer-declared/1")


def test_closure_gate_fails_when_gt_or_forward_disagrees(tmp_path):
    spec, gt = load_problem(make_dataset(tmp_path, noise=0.0, x_from_composition=True), frames=[0])
    D_bad = gt.D.copy(); D_bad[:, 1, :] *= 1.02                  # GT を 2% だけ動かす
    g = closure_gate(spec, D_bad)
    assert g["status"] == "FAIL" and g["n_px_over_tol"] == 4 and "do not renormalise" in g["note"]
    # X を共有 T で作った合成データ（生成式に従っていない）も noise 0 なら FAIL する
    spec2, gt2 = load_problem(make_dataset(tmp_path, nx=3, noise=0.0, x_from_composition=False), frames=[0])
    assert closure_gate(spec2, gt2.D)["status"] == "FAIL"


def test_closure_gate_records_only_at_noise_and_na_without_lambda(tmp_path):
    spec, gt = load_problem(make_dataset(tmp_path, noise=1.0, x_from_composition=True), frames=[0])
    g = closure_gate(spec, gt.D)
    assert g["status"] == "RECORDED" and "noise" in g["note"]
    spec.lambda_basic = None
    assert closure_gate(spec, gt.D)["status"] == "NA"
    spec2, gt2 = load_problem(make_dataset(tmp_path, nx=3, noise=0.0, x_from_composition=True), frames=[0])
    spec2.noise = None
    assert closure_gate(spec2, gt2.D)["status"] == "UNRESOLVED"


def test_closure_gate_hints_at_a_transposed_stored_lambda_but_stays_fail(tmp_path):
    """Python producer の実例（2026-09-09）: forward も値も正しいのに H5 の lambdabasic の向きが逆。

    ゲートは FAIL のまま（緑化しない）。診断補助として「転置なら再現する」を detail に残す。
    """
    from deppro.app.contract import closure_gate, load_problem
    from deppro.tests.unit._synthetic_dataset import make_dataset
    ok = make_dataset(tmp_path / "ok", nx=2, ny=2, nt=2, variant="noisefree", noise=0.0, x_from_composition=True)
    spec, gt = load_problem(ok)
    r = closure_gate(spec, gt.D)
    assert r["status"] == "PASS" and "transposed_lambda_would_pass" not in r
    bad = make_dataset(tmp_path / "bad", nx=2, ny=2, nt=2, variant="noisefree", noise=0.0,
                       x_from_composition=True, lambda_transposed=True)
    spec, gt = load_problem(bad)
    r = closure_gate(spec, gt.D)
    assert r["status"] == "FAIL" and r["transposed_lambda_would_pass"] is True
    assert r["max_abs_with_transposed_lambda"] <= r["tol_abs"] < r["max_abs"]
    assert "orientation" in r["note"]


def test_closure_gate_does_not_blame_orientation_for_a_real_forward_mismatch(tmp_path):
    """本当に forward が違う（X が別物）ときは転置でも再現しないので、補助は False のまま。"""
    from deppro.app.contract import closure_gate, load_problem
    from deppro.tests.unit._synthetic_dataset import make_dataset
    folder = make_dataset(tmp_path, nx=2, ny=2, nt=2, variant="noisefree", noise=0.0)   # X は共有 T から、式は別
    spec, gt = load_problem(folder)
    r = closure_gate(spec, gt.D)
    assert r["status"] == "FAIL" and r["transposed_lambda_would_pass"] is False
    assert "orientation" not in r["note"]
