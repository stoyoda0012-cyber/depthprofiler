"""``deppro.haxpes.deposit_replay`` — the metric definitions and the guard.

These tests need no copy of the Zenodo deposit: the aggregation functions are
checked on small arrays whose answer is known by hand, and the SHA-256 guard
on files written here. A run of the module against the deposit itself is a
separate, recorded exercise (see ``docs/REPRODUCTION_JVST_SIA.md``); it is not
hidden behind a skip here.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from deppro.haxpes import deposit_replay as dr


# ---------------------------------------------------------------------------
# metric definitions
# ---------------------------------------------------------------------------

def _two_box_profiles():
    """Reference: one box per element on the 20-point grid. Shape (20, 2)."""
    D_ref = np.zeros((20, 2))
    D_ref[2:6, 0] = 1.0     # element 0 occupies depth points 2..5
    D_ref[8:14, 1] = 0.5    # element 1 occupies 8..13
    return D_ref


def test_depth_rmse_percent_is_mean_of_per_pixel_rmse_times_100():
    D_ref = _two_box_profiles()
    # pixel 0: constant offset 0.02 everywhere -> per-pixel RMSE 0.02
    # pixel 1: constant offset 0.04 everywhere -> per-pixel RMSE 0.04
    D_map = np.stack([D_ref + 0.02, D_ref + 0.04])
    assert dr.depth_rmse_percent(D_map, D_ref) == pytest.approx(3.0)
    # and it differs from the pooled form (sqrt of the mean of all squares),
    # which here is sqrt((0.02**2 + 0.04**2)/2)*100 = 3.162...
    pooled = np.sqrt(np.mean((D_map - D_ref) ** 2)) * 100
    assert pooled == pytest.approx(3.1623, abs=1e-3)
    assert dr.depth_rmse_percent(D_map, D_ref) < pooled


def test_per_pixel_rmse_percent_vector():
    D_ref = _two_box_profiles()
    D_map = np.stack([D_ref + 0.02, D_ref + 0.04, D_ref])
    np.testing.assert_allclose(dr.per_pixel_rmse_percent(D_map, D_ref), [2.0, 4.0, 0.0], atol=1e-12)


def test_depth_rmse_percent_ignores_nan_entries():
    D_ref = _two_box_profiles()
    D_map = np.stack([D_ref + 0.02])
    D_map[0, 0, 0] = np.nan
    assert dr.depth_rmse_percent(D_map, D_ref) == pytest.approx(2.0)


def test_w1_median_mean_recovers_a_pure_shift():
    """A box shifted by one grid step (30 Å) has W1 = 30 Å against the
    unshifted box; the element that is not shifted contributes 0; the mean
    over the two elements is 15 Å. Three identical pixels -> median = value."""
    D_ref = _two_box_profiles()
    shifted = D_ref.copy()
    shifted[:, 0] = np.roll(D_ref[:, 0], 1)
    D_map = np.stack([shifted, shifted, shifted])
    assert dr.w1_median_mean(D_map, D_ref) == pytest.approx(15.0, rel=1e-6)


def test_w1_median_mean_takes_median_over_pixels_then_mean_over_elements():
    D_ref = _two_box_profiles()
    s1, s2, s3 = D_ref.copy(), D_ref.copy(), D_ref.copy()
    s1[:, 0] = np.roll(D_ref[:, 0], 1)   # 30 Å in element 0
    s2[:, 0] = np.roll(D_ref[:, 0], 2)   # 60 Å in element 0
    s3[:, 0] = np.roll(D_ref[:, 0], 5)   # 150 Å in element 0
    D_map = np.stack([s1, s2, s3])
    # element 0: median(30, 60, 150) = 60; element 1: 0 -> mean 30
    assert dr.w1_median_mean(D_map, D_ref) == pytest.approx(30.0, rel=1e-6)


def test_w1_matches_the_deposited_copy_of_the_function():
    """The deposit's ``arhaxpes_denoise.metrics.wasserstein_w1`` is documented
    as a copy of ``deppro.depth_profile.wasserstein_distance_1d``. Re-state
    that copy's arithmetic here (it is five lines) and check agreement, so a
    change in deppro's function would be noticed by this replay."""
    from deppro.depth_profile import wasserstein_distance_1d

    def deposited_w1(depth, p1, p2):
        _trapz = np.trapezoid
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        depth = np.asarray(depth, dtype=float)
        if _trapz(np.abs(p1), depth) < 1e-6:
            return np.nan
        p1n = p1 / np.maximum(_trapz(p1, depth), 1e-10)
        p2n = p2 / np.maximum(_trapz(p2, depth), 1e-10)
        c1 = np.cumsum(p1n) * np.gradient(depth)
        c2 = np.cumsum(p2n) * np.gradient(depth)
        c1 = c1 / np.maximum(c1[-1], 1e-10)
        c2 = c2 / np.maximum(c2[-1], 1e-10)
        return float(_trapz(np.abs(c1 - c2), depth))

    rng = np.random.default_rng(0)
    for _ in range(20):
        p1 = rng.random(20)
        p2 = rng.random(20)
        assert deposited_w1(dr.DEPTH, p1, p2) == pytest.approx(
            wasserstein_distance_1d(dr.DEPTH, p1, p2), rel=1e-12, abs=1e-12)
    assert np.isnan(deposited_w1(dr.DEPTH, np.zeros(20), p2))
    assert np.isnan(wasserstein_distance_1d(dr.DEPTH, np.zeros(20), p2))


# ---------------------------------------------------------------------------
# ladder fit and its inverse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["rmse", "w1"])
def test_fit_then_equivalent_k_round_trips_on_noiseless_ladder(metric):
    k = np.array([1, 5, 10, 25, 50, 125], dtype=float)
    C = dr.SOFT_FLOOR[metric]
    if metric == "rmse":
        y = dr._rmse_offset_floor(k, A=100.0, gamma=0.5, N0=2.0, C=C)
    else:
        y = dr._w1_offset_floor(k, B=20.0, gamma=0.5, N0=2.0, C=C)
    popt = dr.fit_frame_average(k, y, metric)
    assert popt is not None and popt["mode"] == metric and popt["C"] == C
    assert popt["gamma"] == pytest.approx(0.5, abs=1e-3)
    assert popt["N0"] == pytest.approx(2.0, abs=1e-2)
    # inverting the model at the value of the ladder at K=7 gives 7
    target = (dr._rmse_offset_floor(7.0, 100.0, 0.5, 2.0, C) if metric == "rmse"
              else dr._w1_offset_floor(7.0, 20.0, 0.5, 2.0, C))
    assert dr.equivalent_k(popt, target) == pytest.approx(7.0, abs=1e-2)


def test_equivalent_k_is_none_below_floor_and_outside_range():
    popt = {"mode": "rmse", "A": 100.0, "gamma": 0.5, "N0": 0.0, "C": 0.5}
    assert dr.equivalent_k(popt, 0.5) is None          # at the floor
    assert dr.equivalent_k(popt, 0.1) is None          # below the floor
    assert dr.equivalent_k(None, 3.0) is None          # no fit
    assert dr.equivalent_k(popt, 3.0) == pytest.approx(100.0 / (3.0 ** 2 - 0.25))
    assert dr.equivalent_k(popt, 0.6) is None          # inverse > n_max=500


def test_fit_needs_three_valid_points():
    k = np.array([1, 5, 10], dtype=float)
    y = np.array([5.0, np.nan, 2.0])
    assert dr.fit_frame_average(k, y, "rmse") is None


# ---------------------------------------------------------------------------
# the deposit guard
# ---------------------------------------------------------------------------

def test_verify_files_accepts_matching_and_reports_every_problem(tmp_path):
    (tmp_path / "a.npz").write_bytes(b"alpha")
    (tmp_path / "b.npz").write_bytes(b"beta")
    expected = {
        "a.npz": hashlib.sha256(b"alpha").hexdigest(),
        "b.npz": hashlib.sha256(b"not beta").hexdigest(),
        "c.npz": hashlib.sha256(b"gamma").hexdigest(),
    }
    assert dr.verify_files(tmp_path, {"a.npz": expected["a.npz"]}) == {"a.npz": expected["a.npz"]}
    with pytest.raises(dr.DepositMismatch) as ei:
        dr.verify_files(tmp_path, expected)
    msg = str(ei.value)
    assert "mismatch: b.npz" in msg and "missing: c.npz" in msg and dr.DEPOSIT_VERSION_DOI in msg


def test_replay_refuses_a_directory_that_is_not_the_deposit(tmp_path):
    with pytest.raises(dr.DepositMismatch):
        dr.replay_sia_fig4(tmp_path)


def test_pinned_tables_are_self_consistent():
    """Every file the SIA replay reads is pinned, and every pinned hash is a
    64-hex SHA-256 — a typo here would void the comparison silently."""
    for name, h in list(dr.SIA_FILES.items()) + list(dr.JVST_FILES.items()):
        assert len(h) == 64 and int(h, 16) >= 0, name
    for ds, spec in dr.SIA_DATASETS.items():
        assert spec["gt"] in dr.SIA_FILES
        for k in spec["k_values"]:
            assert f"outputs/hax26_sysavg_cache/{ds}_k{k}.npz" in dr.SIA_FILES
        assert f"outputs/hax26_selfdn_cache/{ds}_dn_W1.npz" in dr.SIA_FILES
        for src in dr.SIA_DATASETS:
            if src != ds:
                assert f"outputs/hax26_selfdn_cache/{ds}_cross_from_{src}.npz" in dr.SIA_FILES
    for (apply_ds, train) in dr.PUBLISHED_SIA["table1"]:
        assert apply_ds in dr.SIA_DATASETS and (train == "self" or train in dr.SIA_DATASETS)


def test_jvst_group_mapping():
    assert dr._jvst_group_for("frame_N10", "session102", "fix25") == "session102u_ladder_k10"
    assert dr._jvst_group_for("perbin_B5", "session102", "fix23") == "session102f_perbin_B5"
    assert dr._jvst_group_for("dn_W5", "session103", "fix25") == "session103u_dn_W5"
    assert dr._jvst_group_for("pbdn_B5_W1", "session103", "fix23") == "session103f_pbdn_B5_W1"
    # the session-103 script reused the session-102 arrays for these three kinds
    assert dr._jvst_group_for("frame_N10", "session103", "fix25") == "session102u_ladder_k10"
    assert dr._jvst_group_for("perbin_B5", "session103", "fix23") == "session102f_perbin_B5"
    assert dr._jvst_group_for("frame_raw", "session103", "fix25") == "session102u_frame_raw"


def test_across_seed_sd_is_the_sample_standard_deviation():
    assert dr.across_seed_sd([1.0, 2.0, 3.0]) == pytest.approx(1.0)       # ddof=1
    assert dr.across_seed_sd([5.0]) is None
    assert dr.across_seed_sd([]) is None


def test_archive_md5_names_only_the_archives_the_module_reads():
    assert set(dr.ARCHIVE_MD5) == {"sia_figure_data.zip", "jvst_figure_data.zip"}
    for h in dr.ARCHIVE_MD5.values():
        assert len(h) == 32 and int(h, 16) >= 0
    for (apply_ds, train) in (list(dr.PUBLISHED_SIA["text"]) + list(dr.PUBLISHED_SIA["seed_sd"])
                              + list(dr.PUBLISHED_SIA["fig_s9_caption"])):
        assert apply_ds in dr.SIA_DATASETS and (train == "self" or train in dr.SIA_DATASETS)
    for caption in dr.PUBLISHED_SIA["fig_s9_caption"].values():
        assert set(caption["seeds"]) == {0, 1, 2} and caption["fig5_reference"] > 0
