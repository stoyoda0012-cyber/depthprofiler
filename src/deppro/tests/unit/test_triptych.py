"""
三点比較図（triptych）と ``DEPPRO_SIMULATION_ROOT`` の契約
==========================================================

守りたいこと:

- ``simulation_root()`` は **env 未設定でも import が通る**。
  失敗するのは実際にデータを読む段で、変数名を含むエラーになる。
- ``TriptychConfig`` の既定が EzDeP4D の本番動画の設定から動かない
  （照明オフ・横長プロットボックス）。値は MATLAB の描画と比べた実測。
- 3 枚組は 3 枚（左＝真値／外部参照）。

pyvista を要求しない範囲だけを見る。実際の描画は
test_visualize_4d_matlab_mode.py が担当。
"""

import os

import numpy as np
import pytest

from deppro.paths import simulation_root
from deppro.tools.triptych import (
    MATLAB_TILE_PLOT_BOX,
    TriptychConfig,
    load_matlab_h5,
    rmse_percent,
)


# ---------------------------------------------------------------------------
# DEPPRO_SIMULATION_ROOT
# ---------------------------------------------------------------------------


def test_simulation_root_importable_without_env(monkeypatch):
    # 未設定でも import も呼び出しも通ること。値は「設定されていない」と
    # 読み取れる文字列で、実際に読む段で初めて失敗する。
    monkeypatch.delenv("DEPPRO_SIMULATION_ROOT", raising=False)
    root = simulation_root()
    assert "DEPPRO_SIMULATION_ROOT" in str(root)


def test_simulation_root_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DEPPRO_SIMULATION_ROOT", str(tmp_path))
    assert simulation_root() == tmp_path


def test_simulation_root_expands_user(monkeypatch):
    monkeypatch.setenv("DEPPRO_SIMULATION_ROOT", "~/sim-root")
    assert "~" not in str(simulation_root())


def test_missing_h5_error_names_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("DEPPRO_SIMULATION_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="DEPPRO_SIMULATION_ROOT"):
        load_matlab_h5("nope/missing.h5", "Dstk")


# ---------------------------------------------------------------------------
# TriptychConfig の既定
# ---------------------------------------------------------------------------


def test_defaults_match_ezdep4d_production_movies():
    cfg = TriptychConfig()
    # EzDeP4D は 3D Lighting トグルが off のとき lightangle を呼ばない。
    # 本番 mp4 の面色を実測した結果もそちらと一致した（docs §5）。
    assert cfg.lighting is False
    # 2x2 タイル (327x220 px) の実測 PlotBoxAspectRatio。立方体ではない。
    assert cfg.plot_box_aspect == MATLAB_TILE_PLOT_BOX
    assert MATLAB_TILE_PLOT_BOX == pytest.approx((0.7127, 1.0, 0.6855))
    # 方位は MATLAB の view([-105 ...]) のまま
    assert cfg.azimuth == pytest.approx(-105.0)


def test_face_alpha_default_is_matlab_opacity():
    # 実効不透明度 = 0.75 * face_alpha。MATLAB の FaceAlpha 既定は 0.75。
    assert 0.75 * TriptychConfig().face_alpha == pytest.approx(0.75)


def test_cube_mode_is_selectable():
    # 正方 axes の MATLAB は PlotBoxAspectRatio = [1 1 1] なので、
    # 立方体も正当な MATLAB 忠実設定。
    assert TriptychConfig(plot_box_aspect=None).plot_box_aspect is None


# ---------------------------------------------------------------------------
# 指標とパネル数
# ---------------------------------------------------------------------------


def _tiny(nt=2, seed=0):
    from deppro.visualize_4d import DepthProfile4DData

    rng = np.random.default_rng(seed)
    return DepthProfile4DData(
        data=rng.random((nt, 3, 3, 4, 2)),
        depth=np.arange(4) * 30.0,
        element_names=["A", "B"],
        colors=np.array([[1.0, 0, 0], [0, 0, 1.0]]),
        step_size=30.0, nx=3, ny=3, nt=nt, n_depth=4, n_elem=2,
    )


def test_rmse_percent_zero_for_identical():
    d = _tiny()
    assert rmse_percent(d, d, 0) == pytest.approx(0.0)


def test_rmse_percent_is_in_percent():
    a, b = _tiny(seed=1), _tiny(seed=2)
    v = rmse_percent(a, b, 0)
    # 濃度は 0-1 なので RMSE は 0-100% に収まる（0 より大きい）
    assert 0.0 < v < 100.0


def test_render_triptych_requires_exactly_three_panels():
    pytest.importorskip("pyvista")
    from deppro.tools.triptych import render_triptych

    d = _tiny()
    with pytest.raises(ValueError, match="3"):
        render_triptych([("only", d), ("two", d)])
