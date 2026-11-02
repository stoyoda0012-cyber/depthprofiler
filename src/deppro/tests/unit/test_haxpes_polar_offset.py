"""アナライザー中心軸から試料法線までの polar 角（コード上の ``polar_offset``、定数 ``SAMPLE_POLAR_DEG``）が 30° であることを固定する。

なぜ要るか
----------
放出角は ``flip(検出器角) + polar_offset`` で作られる。この式は正しいが、その値が
2026-09-27 まで 26.0 だった。正しい値は 30° である
（出版された JVST 本文 §II.B は "the central emission angle is 30° from the
surface normal" と書く）。

``polar_offset`` の意味は「アナライザー中心軸（レンズ軸、検出器角 0）から試料法線までの
polar 角」である。試料は polar 一軸しか回らないので試料傾斜角と
数値は一致するが、保持 bin の中心角とは一致しない（0.184° ずれる）。

26 は ``config`` のデータクラス既定値、FIX_IBW/FIX_PXT の構成、``preprocessor`` と
``loader`` の関数既定値の 4 か所に別々に書かれていた。それが、値の誤りが
どこか 1 か所だけ直されて残る形を許していた。ここでは、全員が同じ 1 つの定数を
読んでいることと、その値で決まる角度を固定する。

``polar_offset`` は角度ラベルだけでなく、放出角空間のマスク（5–55°）が選ぶ bin も動かす。
26 では bin 1–8（10.53–53.34°）、30 では bin 2–9（8.41–51.22°）になる。
後者の値は、生の角度軸（sfB = −33.8215、sfA = 6.1159、12 bins）から独立に
再構成された確定値と一致する。

この量は以前 ``tilt_angle`` / ``SAMPLE_TILT_DEG`` / 引数 ``tilt`` と呼ばれていたが、
試料の tilt 軸（``deppro.haxpes.geometry`` の ``tilt_deg``）とは別の回転なので
改名した。旧名は非推奨の別名として残り、使うと DeprecationWarning を出す。
"""

import inspect
import warnings

import numpy as np
import pytest

from deppro.haxpes import config, loader, preprocessor


def test_the_polar_offset_is_30_degrees():
    assert config.SAMPLE_POLAR_DEG == 30.0


def test_every_default_reads_the_one_constant():
    assert config.HAXPESInstrumentConfig().polar_offset == config.SAMPLE_POLAR_DEG
    assert config.FIX_IBW_INSTRUMENT.polar_offset == config.SAMPLE_POLAR_DEG
    assert config.FIX_PXT_INSTRUMENT.polar_offset == config.SAMPLE_POLAR_DEG
    for function in (
        preprocessor.detector_to_emission,
        loader.load_prebinned_pxt,
        loader.sum_and_bin,
    ):
        default = inspect.signature(function).parameters["polar_offset"].default
        assert default == config.SAMPLE_POLAR_DEG, function.__name__


def test_the_5_to_55_degree_mask_selects_bins_2_to_9():
    angles = config.FIX_PXT_INSTRUMENT.emission_angles_prebinned()
    selected = np.flatnonzero((angles >= 5.0) & (angles <= 55.0))
    assert selected.tolist() == [2, 3, 4, 5, 6, 7, 8, 9]
    np.testing.assert_allclose(
        angles[selected][[0, -1]], [51.22, 8.41], atol=0.005
    )


def test_the_raw_detector_axis_spans_plus_minus_36_84_about_the_polar_offset():
    emission = preprocessor.detector_to_emission()
    assert emission[0] == 36.84 + config.SAMPLE_POLAR_DEG
    assert emission[-1] == -36.84 + config.SAMPLE_POLAR_DEG


# polar offset を表す名前（旧名を含む）。値に数値を直接書くことを禁じる
# （``SAMPLE_POLAR_DEG`` を参照させる）。
_POLAR_NAMES = {"polar_offset", "tilt", "tilt_angle", "tilt_deg", "sample_tilt"}


def _literal_polar_offsets(path):
    """``path`` の中で、polar offset の名前に数値リテラルを与えている箇所 (行, 種類, 名前, 値)。"""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []

    def is_number(node):
        return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            positional = a.posonlyargs + a.args
            pairs = list(zip(positional[len(positional) - len(a.defaults):], a.defaults))
            pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            for arg, default in pairs:
                if arg.arg in _POLAR_NAMES and is_number(default):
                    found.append((default.lineno, "default", arg.arg, default.value))
        elif isinstance(node, ast.keyword):
            if node.arg in _POLAR_NAMES and is_number(node.value):
                found.append((node.value.lineno, "keyword", node.arg, node.value.value))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            for target in targets:
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name in _POLAR_NAMES and node.value is not None and is_number(node.value):
                    found.append((node.lineno, "assign", name, node.value.value))
    return found


def test_no_polar_offset_is_written_as_a_number_anywhere_in_src():
    """``haxpes/`` の外にも 26.0 が 8 か所残っていた（2026-10-01 に直した）。

    公開版に含めない読み込みスクリプトが関数既定値・データクラスの既定値・
    キーワード引数で 26.0 を直接書き、当時の ``SAMPLE_TILT_DEG``（現 ``SAMPLE_POLAR_DEG``）を読んでいなかった。
    上の ``test_every_default_reads_the_one_constant`` は名前を挙げた関数しか
    見ないので、それを捕まえられなかった。ここでは ``src/`` の追跡ファイル
    全部を構文木で調べる（git の無い場所では ``src/`` の全 .py を調べる）。過去の解析を再現するために 26.0 を渡したいときは、
    呼び出す側（スクリプトの外）で明示する。

    捕まえない形（網羅ではない）: 負の数（``-26.0`` は単項マイナスで、定数として
    現れない）、位置引数で渡した値、``_POLAR_NAMES`` に無い名前（例:
    ``tilt_angle_deg``）。
    """
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "src"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Not a git checkout (an extracted sdist or a copied tree), or no git:
        # scan every .py under src/ instead. Nothing is excluded (the git
        # route does not exclude _archived/ either), so this is a superset of
        # the tracked set and never weaker.
        listed = [p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")]
    offenders = {}
    for rel in listed:
        if not rel.endswith(".py") or rel.endswith("test_haxpes_polar_offset.py"):
            continue
        hits = _literal_polar_offsets(root / rel)
        if hits:
            offenders[rel] = hits
    assert not offenders, offenders


# 旧名は非推奨の別名として残す。値は同じで、使うと DeprecationWarning を出す。

def test_old_constant_name_warns_and_gives_the_same_value():
    with pytest.warns(DeprecationWarning, match="SAMPLE_POLAR_DEG"):
        assert config.SAMPLE_TILT_DEG == config.SAMPLE_POLAR_DEG


def test_old_constant_name_is_importable_with_a_warning():
    with pytest.warns(DeprecationWarning):
        from deppro.haxpes.config import SAMPLE_TILT_DEG
    assert SAMPLE_TILT_DEG == 30.0


def test_unknown_module_attribute_still_raises():
    with pytest.raises(AttributeError):
        config.NOT_A_NAME


def test_old_dataclass_keyword_warns_and_sets_polar_offset():
    with pytest.warns(DeprecationWarning, match="tilt_angle"):
        c = config.HAXPESInstrumentConfig(tilt_angle=26.0)
    assert c.polar_offset == 26.0
    with pytest.warns(DeprecationWarning, match="tilt_angle"):
        assert c.tilt_angle == 26.0


def test_old_dataclass_keyword_conflicting_with_polar_offset_raises():
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        config.HAXPESInstrumentConfig(polar_offset=28.0, tilt_angle=26.0)


def test_new_names_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        c = config.HAXPESInstrumentConfig(polar_offset=26.0)
        assert c.polar_offset == 26.0
        c.emission_angles_prebinned()
        preprocessor.detector_to_emission(1001, polar_offset=26.0)


@pytest.mark.parametrize("function", [
    preprocessor.detector_to_emission,
])
def test_old_tilt_keyword_warns_and_gives_the_same_angles(function):
    new = function(polar_offset=26.0)
    with pytest.warns(DeprecationWarning, match="polar_offset"):
        old = function(tilt=26.0)
    np.testing.assert_array_equal(old, new)
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        function(polar_offset=28.0, tilt=26.0)


def test_old_dataclass_keyword_with_explicit_default_polar_offset_raises():
    """生成時は明示されたかどうかが分かるので、既定値と同じ値でも食い違いを検出する。"""
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        config.HAXPESInstrumentConfig(polar_offset=config.SAMPLE_POLAR_DEG, tilt_angle=26.0)
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        config.HAXPESInstrumentConfig(9251.74, 28.0, tilt_angle=26.0)


def test_replace_and_asdict_see_polar_offset_only():
    import dataclasses

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        base = config.HAXPESInstrumentConfig(polar_offset=26.0)
        assert dataclasses.replace(base, polar_offset=20.0).polar_offset == 20.0
        assert dataclasses.replace(base, photon_energy=1.0).polar_offset == 26.0
        assert "tilt_angle" not in dataclasses.asdict(base)
        assert [f.name for f in dataclasses.fields(base)].count("polar_offset") == 1
        assert "tilt_angle" not in [f.name for f in dataclasses.fields(base)]


def test_sum_and_bin_old_tilt_keyword_is_used():
    new = loader.sum_and_bin({}, [], polar_offset=26.0)[4]
    with pytest.warns(DeprecationWarning, match="polar_offset"):
        old = loader.sum_and_bin({}, [], tilt=26.0)[4]
    np.testing.assert_array_equal(old, new)
    assert not np.array_equal(new, loader.sum_and_bin({}, [])[4])
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        loader.sum_and_bin({}, [], polar_offset=28.0, tilt=26.0)


def test_load_prebinned_pxt_old_tilt_keyword_warns_before_reading(tmp_path):
    with pytest.warns(DeprecationWarning, match="polar_offset"), pytest.raises(TypeError):
        loader.load_prebinned_pxt(tmp_path, polar_offset=28.0, tilt=26.0)


@pytest.mark.parametrize("function", [loader.load_prebinned_pxt, loader.sum_and_bin])
def test_loader_old_tilt_keyword_is_keyword_only(function):
    params = inspect.signature(function).parameters
    assert params["tilt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["tilt"].default is None
