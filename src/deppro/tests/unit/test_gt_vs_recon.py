"""``deppro.tools.gt_vs_recon`` — 真値と再構成を並べる薄い合成層の契約。

守りたいこと:

- :func:`load_profile` は H5 の dialect（``/Dstk`` か ``/D_est/data``）を見て
  ローダを選ぶ。どちらも無ければ何があったかを挙げて拒否する。
- :func:`check_compatible` は grid・深さ軸・元素列の**すべて**の不一致を挙げる。
- 指標は既存の定義の呼び出しであって、ここで新しい数学を持たない:
  RMSE は :func:`deppro.tools.triptych.rmse_percent`、W1 は
  :func:`deppro.depth_profile.wasserstein_distance_1d` の画素中央値・元素平均。
- 描画ライブラリ無しで指標と入力検査が動く（この部分は skip しない）。

実際の描画（pyvista + imageio）は末尾の 1 本だけで、無い環境では skip される。
skip は合格ではない。
"""
from __future__ import annotations

import numpy as np
import pytest

from deppro.app import write_depth_estimate
from deppro.simulate import CHECKER_7ELEM, H5Writer
from deppro.tools.gt_vs_recon import (
    GtVsReconConfig,
    build_parser,
    check_compatible,
    config_from_args,
    frame_metrics,
    load_profile,
    mean_profile,
)
from deppro.visualize_4d import DepthProfile4DData

LABELS = ["Si(0)", "Si(+4)", "Ti(+4)", "Al(+3)", "C(0)", "B(0)", "O(-2)"]
DEPTH = np.arange(0.0, 100.0, 5.0)          # 20 points, 5 Å
ANGLES = np.array([10.0, 30.0, 50.0])
NX, NY, NT = 3, 2, 2
N_PX = NX * NY * NT


def _box(lo: int, hi: int) -> np.ndarray:
    """深さ bin [lo, hi) に濃度 1 の箱。"""
    p = np.zeros(len(DEPTH))
    p[lo:hi] = 1.0
    return p


def _profiles(shift: int = 0) -> np.ndarray:
    """(N_PX, n_depth, 7): 元素 0 (Si(0)) は基板、元素 1 は箱、元素 4 (C) は空。"""
    D = np.zeros((N_PX, len(DEPTH), 7))
    for px in range(N_PX):
        D[px, :, 1] = _box(4 + shift, 8 + shift)          # 箱 20 Å 幅
        D[px, :, 0] = 1.0 - D[px, :, 1]                   # 残りは基板
    return D


def _write_gt(path, D) -> str:
    w = H5Writer(path, CHECKER_7ELEM)
    w.add_batch(D, np.zeros((N_PX, len(ANGLES), 7)))
    w.finalize(depth=DEPTH, angles=ANGLES, step_size=5.0, photon_energy=1486.6, nx=NX, ny=NY)
    return str(path)


def _write_est(out_dir, D, labels=LABELS, depth=DEPTH, grid=(NX, NY)) -> str:
    return write_depth_estimate(
        D, {"pixel_shape": grid, "elements": labels, "depth_axis": depth, "step_size": 5.0},
        str(out_dir), solver="TEST",
    )


# ---------------------------------------------------------------------------
# load_profile
# ---------------------------------------------------------------------------


def test_load_profile_picks_the_loader_from_the_file(tmp_path):
    D = _profiles()
    gt = load_profile(_write_gt(tmp_path / "gt.h5", D))
    est = load_profile(_write_est(tmp_path, D))
    for d in (gt, est):
        assert isinstance(d, DepthProfile4DData)
        assert (d.nx, d.ny, d.nt) == (NX, NY, NT)
        assert d.element_names == LABELS
    np.testing.assert_allclose(gt.data, est.data, atol=1e-6)


def test_load_profile_rejects_a_file_with_neither_dataset(tmp_path):
    import h5py

    p = tmp_path / "other.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("Xstk", data=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="Xstk"):
        load_profile(str(p))


# ---------------------------------------------------------------------------
# check_compatible
# ---------------------------------------------------------------------------


def test_check_compatible_passes_for_matching_files(tmp_path):
    D = _profiles()
    check_compatible(load_profile(_write_gt(tmp_path / "gt.h5", D)),
                     load_profile(_write_est(tmp_path, D)))


def test_check_compatible_lists_every_mismatch(tmp_path):
    D = _profiles()
    gt = load_profile(_write_gt(tmp_path / "gt.h5", D))
    other = LABELS[:-1] + ["O(0)"]
    bad = load_profile(_write_est(tmp_path, D[: NX * NY], labels=other, grid=(NX, NY)))  # nt=1
    with pytest.raises(ValueError) as ei:
        check_compatible(gt, bad)
    msg = str(ei.value)
    assert "grid" in msg and "elements" in msg and "O(0)" in msg


# ---------------------------------------------------------------------------
# 指標・プロファイル（描画ライブラリ不要）
# ---------------------------------------------------------------------------


def test_mean_profile_is_the_xy_average_in_percent(tmp_path):
    gt = load_profile(_write_gt(tmp_path / "gt.h5", _profiles()))
    prof = mean_profile(gt, 0)
    assert prof.shape == (len(DEPTH), 7)
    np.testing.assert_allclose(prof[:, 1], _box(4, 8) * 100.0, atol=1e-4)
    np.testing.assert_allclose(prof[:, 0] + prof[:, 1], 100.0, atol=1e-4)


def test_identical_data_gives_zero_metrics_and_nan_for_empty_elements(tmp_path):
    D = _profiles()
    gt = load_profile(_write_gt(tmp_path / "gt.h5", D))
    est = load_profile(_write_est(tmp_path, D))
    m = frame_metrics(gt, est, 0)
    assert m["rmse_percent"] == pytest.approx(0.0, abs=1e-4)
    assert m["w1_median_px_A"] == pytest.approx(0.0, abs=1e-6)
    per = m["w1_per_element_A"]
    assert per.shape == (7,)
    assert np.isnan(per[4]) and np.isnan(per[5])          # C, B: 真値が空 → NaN
    assert per[0] == pytest.approx(0.0, abs=1e-6)


def test_a_shifted_box_moves_w1_by_the_shift_and_matches_the_canonical_w1(tmp_path):
    from deppro.depth_profile import wasserstein_distance_1d
    from deppro.tools.triptych import rmse_percent

    gt = load_profile(_write_gt(tmp_path / "gt.h5", _profiles(shift=0)))
    est = load_profile(_write_est(tmp_path, _profiles(shift=1)))   # 1 bin = 5 Å
    m = frame_metrics(gt, est, 0)
    # 元素 1 の箱は丸ごと 5 Å 下へ: W1 は定義の関数そのもの（全画素同一なので中央値 = 1 画素の値）
    expected = wasserstein_distance_1d(DEPTH, _box(4, 8), _box(5, 9))
    assert m["w1_per_element_A"][1] == pytest.approx(expected)
    assert expected > 0
    assert m["rmse_percent"] == pytest.approx(rmse_percent(est, gt, 0))
    assert m["rmse_percent"] > 0


# ---------------------------------------------------------------------------
# 設定・CLI の組み立て（描画はしない）
# ---------------------------------------------------------------------------


def test_defaults_follow_triptych_and_the_cli_maps_onto_them():
    from deppro.tools.triptych import TriptychConfig

    base, cfg = TriptychConfig(), GtVsReconConfig()
    assert (cfg.azimuth, cfg.elevation, cfg.lighting, cfg.plot_box_aspect) == (
        base.azimuth, base.elevation, base.lighting, base.plot_box_aspect
    )
    args = build_parser().parse_args(
        ["--gt", "a.h5", "--recon", "b.h5", "--label", "NN", "--hide", "6",
         "--no-metrics", "--elevation", "30", "--fps", "5"]
    )
    c = config_from_args(args)
    assert c.recon_label == "NN" and c.invisible_elements == [6]
    assert c.show_metrics is False and c.profile_row is True
    assert c.elevation == 30 and c.fps == 5 and c.azimuth == base.azimuth
    assert c.depth_peeling is True
    assert config_from_args(build_parser().parse_args(
        ["--gt", "a.h5", "--recon", "b.h5", "--no-depth-peeling"])).depth_peeling is False


# ---------------------------------------------------------------------------
# 描画（pyvista + imageio がある環境だけ。skip は合格ではない）
# ---------------------------------------------------------------------------


def test_render_writes_a_png_for_one_frame(tmp_path):
    pytest.importorskip("pyvista")
    imageio = pytest.importorskip("imageio")
    from deppro.tools.gt_vs_recon import render_gt_vs_recon

    D = _profiles()
    gt = load_profile(_write_gt(tmp_path / "gt.h5", D))
    est = load_profile(_write_est(tmp_path, _profiles(shift=1)))
    # depth_peeling=False: ソフトウェア OpenGL（CI / Xvfb）では peeling が面を落とす
    cfg = GtVsReconConfig(title="test", panel_px=(320, 220), dpi=72, depth_peeling=False)
    out = render_gt_vs_recon(gt, est, cfg, output=str(tmp_path / "frame.png"), frames=[0])
    assert out == tmp_path / "frame.png" and out.exists()
    img = imageio.imread(out)
    assert img.ndim == 3 and img.shape[0] > 100 and img.shape[1] > 100
    assert (img[:, :, :3] < 200).any()        # 真っ白ではない
