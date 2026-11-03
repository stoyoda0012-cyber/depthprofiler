"""matlab_generation の契約テスト。

**模擬 CLI で回す通常の回帰**と、実 MATLAB を起動する**受入ゲート**を分ける。
後者は ``slow`` マーカで既定では収集されず、走らせるときは
依存が無ければ skip ではなく**失敗**する側に置く。

受入ゲートの実行:
    pytest -m slow src/deppro/app/tests/test_matlab_generation.py
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from deppro.app import matlab_generation as mg


# --------------------------------------------------------------------------
# 模擬 CLI で作る fixture
# --------------------------------------------------------------------------

def _write_dataset(folder: Path, *, generation_id="20260907T000000000-0abcde",
                   stack=6, nx=3, ny=2, nt=1, layout_type="rectangular_grid",
                   confirmed_generation=None, status="ok", schema="1.0.0",
                   with_stackxyt=True, byte_lie=False, layout=None) -> Path:
    """生成側が書くものと同じ形の manifest + h5 を作る。"""
    h5py = pytest.importorskip("h5py")
    folder.mkdir(parents=True, exist_ok=True)
    main = folder / "main.h5"

    s = np.arange(1, stack + 1)
    with h5py.File(main, "w") as f:
        f.create_dataset("Dstk", data=np.ones((stack, 2, 4), dtype="f4"))
        if with_stackxyt:
            # 生成側と同じく、h5py からは (3, stack) に見える形で書く
            xyt = np.stack([
                (s - 1) % nx + 1,
                ((s - 1) % (nx * ny)) // nx + 1,
                (s - 1) // (nx * ny) + 1,
            ]).astype("f4")
            f.create_dataset("stackxyt", data=xyt)

    manifest = {
        "status": status,
        "schemaVersion": schema,
        "generationId": generation_id,
        "mainFile": str(main),
        "resolved": {
            "axes": {"nx": nx, "ny": ny, "nt": nt, "stack": stack},
            "files": [{"name": "main.h5",
                       "bytes": main.stat().st_size + (1 if byte_lie else 0)}],
        },
        "layout": (layout if layout is not None
                   else {"type": layout_type, "nx": nx, "ny": ny, "nt": nt, "complete": True}),
        "sampleId": {"rule": "<generationId>#<stackIndex>", "generationId": generation_id},
        "confirmed": {
            "status": "ok",
            "generationId": confirmed_generation or generation_id,
        },
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return main


# --------------------------------------------------------------------------
# 受け入れ条件
# --------------------------------------------------------------------------

def test_import_works_without_the_env_var(monkeypatch):
    """未設定でも import は通り、使う段で変数名を含むエラーになる。"""
    monkeypatch.delenv("DEPPRO_DEPTHPROFILER_CLI", raising=False)
    assert "DEPPRO_DEPTHPROFILER_CLI" in str(mg.matlab_cli())
    with pytest.raises(mg.MatlabGenerationError, match="DEPPRO_DEPTHPROFILER_CLI"):
        mg.generate({"outputFolder": "/nonexistent"})


def test_accepts_a_confirmed_manifest(tmp_path):
    _write_dataset(tmp_path)
    result = mg.load_confirmed_manifest(tmp_path)
    assert result.generation_id == "20260907T000000000-0abcde"
    assert result.layout["type"] == "rectangular_grid"


def test_rejects_a_missing_manifest(tmp_path):
    with pytest.raises(mg.ManifestRejected, match="manifest.json"):
        mg.load_confirmed_manifest(tmp_path)


def test_rejects_a_truncated_manifest(tmp_path):
    _write_dataset(tmp_path)
    (tmp_path / "manifest.json").write_text('{"status": "ok"', encoding="utf-8")
    with pytest.raises(mg.ManifestRejected, match="parse"):
        mg.load_confirmed_manifest(tmp_path)


def test_rejects_an_unknown_schema_major(tmp_path):
    _write_dataset(tmp_path, schema="2.0.0")
    with pytest.raises(mg.ManifestRejected, match="schemaVersion"):
        mg.load_confirmed_manifest(tmp_path)


def test_rejects_a_failed_run(tmp_path):
    _write_dataset(tmp_path, status="error")
    with pytest.raises(mg.ManifestRejected, match="status"):
        mg.load_confirmed_manifest(tmp_path)


def test_rejects_an_unconfirmed_manifest(tmp_path):
    _write_dataset(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["confirmed"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(mg.ManifestRejected, match="confirmed"):
        mg.load_confirmed_manifest(tmp_path)


def test_rejects_a_confirmation_from_another_generation(tmp_path):
    """別世代の確認を流用させない。"""
    _write_dataset(tmp_path, confirmed_generation="20260101T000000000-999999")
    with pytest.raises(mg.ManifestRejected, match="another generation"):
        mg.load_confirmed_manifest(tmp_path)


def test_rejects_a_file_whose_size_moved(tmp_path):
    _write_dataset(tmp_path, byte_lie=True)
    with pytest.raises(mg.ManifestRejected, match="bytes"):
        mg.load_confirmed_manifest(tmp_path)


# --------------------------------------------------------------------------
# レイアウトと座標
# --------------------------------------------------------------------------

def test_stack_coordinates_come_back_as_stack_by_three(tmp_path):
    """h5py が見る (3, stack) を (stack, 3) に直すのがこの層の契約。"""
    h5py = pytest.importorskip("h5py")
    main = _write_dataset(tmp_path, stack=6, nx=3, ny=2, nt=1)
    with h5py.File(main, "r") as f:
        assert f["stackxyt"].shape == (3, 6), "fixture が生成側と同じ向きでない"

    xyt = mg.stack_coordinates(mg.load_confirmed_manifest(tmp_path))
    assert xyt.shape == (6, 3)
    # x = mod(s-1,nx)+1, y = floor(mod(s-1,nx*ny)/nx)+1, t = floor((s-1)/(nx*ny))+1
    expected = np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1],
                         [1, 2, 1], [2, 2, 1], [3, 2, 1]], dtype="f4")
    np.testing.assert_array_equal(xyt, expected)


def test_an_unknown_layout_is_refused_not_reshaped(tmp_path):
    """未対応レイアウトは明示的に止まる。黙って矩形にしない。"""
    _write_dataset(tmp_path, layout_type="wafer_scatter")
    with pytest.raises(mg.UnsupportedLayout, match="wafer_scatter"):
        mg.stack_coordinates(mg.load_confirmed_manifest(tmp_path))


def test_a_dataset_without_stackxyt_is_refused(tmp_path):
    """2026-09-07 より前の生成物には入っていない。順序を仮定して補わない。"""
    _write_dataset(tmp_path, with_stackxyt=False)
    with pytest.raises(mg.ManifestRejected, match="stackxyt"):
        mg.stack_coordinates(mg.load_confirmed_manifest(tmp_path))


def test_sample_ids_keep_the_original_index(tmp_path):
    """行番号は位置であって識別子ではない。抜き出しても元の番号が残る。"""
    _write_dataset(tmp_path, stack=6)
    result = mg.load_confirmed_manifest(tmp_path)
    assert mg.sample_ids(result) == [f"20260907T000000000-0abcde#{i}" for i in range(1, 7)]
    # 部分抽出しても 3 番目の点は #3 のまま
    assert mg.sample_ids(result, [3, 5]) == [
        "20260907T000000000-0abcde#3",
        "20260907T000000000-0abcde#5",
    ]


def test_rejects_a_manifest_that_names_no_files(tmp_path):
    """H5 が 1 つも無くても内部だけ整合した manifest を受理していた（再現済み）。

    検査対象が 0 件だったことと、検査に通ったことは別。
    """
    _write_dataset(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["resolved"]["files"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(mg.ManifestRejected, match="resolved.files"):
        mg.load_confirmed_manifest(tmp_path)


def test_rejects_a_manifest_whose_main_file_is_gone(tmp_path):
    _write_dataset(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["mainFile"] = str(tmp_path / "never_written.h5")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(mg.ManifestRejected, match="mainFile"):
        mg.load_confirmed_manifest(tmp_path)


def test_rejects_a_main_file_that_the_list_does_not_cover(tmp_path):
    """一覧に載っていない artifact を、検査済みのつもりで開かない。"""
    _write_dataset(tmp_path)
    other = tmp_path / "unlisted.h5"
    other.write_bytes(b"not in resolved.files")
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["mainFile"] = str(other)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(mg.ManifestRejected, match="none of resolved.files"):
        mg.load_confirmed_manifest(tmp_path)


def test_stack_coordinates_orientation_comes_from_the_schema(tmp_path):
    """stack==3 だと (3,3) が正方で、形から向きを決められない。

    形で判断していたころは転置したものを黙って返していた。保存形は
    (3, stack) と宣言されているので、そちらで決める。
    """
    _write_dataset(tmp_path, stack=3, nx=3, ny=1, nt=1)
    xyt = mg.stack_coordinates(mg.load_confirmed_manifest(tmp_path))
    np.testing.assert_array_equal(
        xyt, np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1]], dtype="f4"))


def test_stack_coordinates_reject_a_permuted_stackxyt(tmp_path):
    """座標が自分自身と無矛盾なだけでは、並びが宣言どおりの証拠にならない。"""
    h5py = pytest.importorskip("h5py")
    main = _write_dataset(tmp_path, stack=6, nx=3, ny=2, nt=1)
    with h5py.File(main, "r+") as f:
        xyt = f["stackxyt"][()]
        del f["stackxyt"]
        f.create_dataset("stackxyt", data=xyt[:, ::-1])   # 行を並べ替える
    with pytest.raises(mg.ManifestRejected, match="declared formula"):
        mg.stack_coordinates(mg.load_confirmed_manifest(tmp_path))


def test_stack_coordinates_reject_out_of_range_values(tmp_path):
    h5py = pytest.importorskip("h5py")
    main = _write_dataset(tmp_path, stack=6, nx=3, ny=2, nt=1)
    with h5py.File(main, "r+") as f:
        xyt = f["stackxyt"][()]
        xyt[0, 0] = 99
        del f["stackxyt"]
        f.create_dataset("stackxyt", data=xyt)
    with pytest.raises(mg.ManifestRejected, match="out of range"):
        mg.stack_coordinates(mg.load_confirmed_manifest(tmp_path))


def test_sample_ids_reject_bool_and_non_integer(tmp_path):
    """`[1.9, True]` が黙って `#1, #1` になっていた。

    別々に指定した 2 点が同じ id になると、正当な重複抽出と型変換で生まれた
    衝突を区別できない。
    """
    _write_dataset(tmp_path, stack=6)
    result = mg.load_confirmed_manifest(tmp_path)
    with pytest.raises(ValueError, match="bool"):
        mg.sample_ids(result, [True])
    with pytest.raises(ValueError, match="not an integer"):
        mg.sample_ids(result, [1.9])
    with pytest.raises(ValueError, match="str"):
        mg.sample_ids(result, ["2"])
    # 整数値の float は位置指定として妥当なので通す
    assert mg.sample_ids(result, [2.0]) == ["20260907T000000000-0abcde#2"]


def test_sample_ids_reject_indices_that_are_not_1_based(tmp_path):
    """0-based の行番号を渡す取り違えを黙って通さない。

    `#0` は 1-based では存在しない。素通しすると体裁だけ整った無意味な id に
    なるので、範囲外はここで止める。
    """
    _write_dataset(tmp_path, stack=6)
    result = mg.load_confirmed_manifest(tmp_path)
    with pytest.raises(ValueError, match="1-based"):
        mg.sample_ids(result, [0, 1, 2])
    with pytest.raises(ValueError, match="1-based"):
        mg.sample_ids(result, [7])
    # 負の添字も位置指定であって識別子ではない
    with pytest.raises(ValueError, match="1-based"):
        mg.sample_ids(result, [-1])
    # 境界は通る
    assert mg.sample_ids(result, [1, 6])[-1].endswith("#6")


def test_rejects_a_manifest_whose_output_folder_is_elsewhere(tmp_path):
    """置き違えた manifest を、読んだ場所の生成物として扱わない。"""
    _write_dataset(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["outputFolder"] = str(tmp_path.parent / "somewhere_else")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(mg.ManifestRejected, match="outputFolder"):
        mg.load_confirmed_manifest(tmp_path)


def test_rejects_a_relative_main_file(tmp_path):
    """相対パスは cwd 次第で受理/拒否が変わる。黙って解決しない。"""
    _write_dataset(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["mainFile"] = "main.h5"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(mg.ManifestRejected, match="relative path"):
        mg.load_confirmed_manifest(tmp_path)


def test_stackxyt_message_does_not_promise_a_date_boundary(tmp_path):
    """境界は生成器の版であって日付ではない（09-07 午前の生成物にも欠けている）。"""
    _write_dataset(tmp_path, with_stackxyt=False)
    with pytest.raises(mg.ManifestRejected) as exc:
        mg.stack_coordinates(mg.load_confirmed_manifest(tmp_path))
    assert "generator version" in str(exc.value)


# --------------------------------------------------------------------------
# 模擬 CLI 経由
# --------------------------------------------------------------------------

def test_generate_runs_the_cli_and_returns_the_manifest(tmp_path):
    out = tmp_path / "out"
    fake = tmp_path / "fake_dp"
    fake.write_text(
        "#!/bin/sh\n"
        f"exec {os.sys.executable} -c \""
        "import sys, json, pathlib; "
        f"sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        "from test_matlab_generation import _write_dataset; "
        f"_write_dataset(pathlib.Path({str(out)!r}))\"\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    result = mg.generate({"outputFolder": str(out)}, cli=fake)
    assert result.generation_id == "20260907T000000000-0abcde"
    assert result.folder == out


def test_generate_raises_when_the_cli_exits_non_zero(tmp_path):
    fake = tmp_path / "failing_dp"
    fake.write_text("#!/bin/sh\necho boom >&2\nexit 3\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(mg.MatlabGenerationError, match="returncode=3"):
        mg.generate({"outputFolder": str(tmp_path / "out")}, cli=fake)


def test_generate_raises_on_timeout(tmp_path):
    fake = tmp_path / "slow_dp"
    fake.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(mg.MatlabGenerationError, match="did not finish within"):
        mg.generate({"outputFolder": str(tmp_path / "out")}, cli=fake, timeout=0.5)


# --------------------------------------------------------------------------
# 受入ゲート（実 MATLAB）
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_acceptance_real_matlab_round_trip(tmp_path):
    """Python cfg -> 実 MATLAB CLI -> 確認済み manifest -> 座標読取。

    物理精度は見ない。実行・記録経路が通ることだけを判定する。
    依存が無ければ skip ではなく**失敗**させる。
    """
    cli = mg.matlab_cli()
    assert cli.exists(), (
        f"DEPPRO_DEPTHPROFILER_CLI が指す CLI が無い: {cli}. "
        "受入ゲートは依存が無いときに skip せず失敗する"
    )
    out = tmp_path / "Checkered" / "_acceptance"
    cfg = {
        "gridX": 10, "gridY": 10, "timeSteps": 2, "timeSize": 10,
        "totalDepth": 600, "noise": 1,
        "topLayer": "Checkered", "bottomLayer": "Trigonometric",
        "interfRedox": "Exponential", "etchGrow": "Linear",
        "parallel": False, "loop": 10,
        "editElements": [
            {"index": 5, "height": 50},
            {"index": 6, "symbol": "B", "charge": 0, "property": "Free",
             "position": 0, "width": 70, "height": 50},
        ],
        "addElements": [{"symbol": "O", "charge": -2, "property": "Linked"}],
        "outputFolder": str(out),
    }
    result = mg.generate(cfg, timeout=900)
    assert result.layout["type"] == "rectangular_grid"

    xyt = mg.stack_coordinates(result)
    assert xyt.shape == (200, 3)
    nx, ny = result.layout["nx"], result.layout["ny"]
    s = np.arange(1, 201)
    np.testing.assert_array_equal(xyt[:, 0], (s - 1) % nx + 1)
    np.testing.assert_array_equal(xyt[:, 1], ((s - 1) % (nx * ny)) // nx + 1)
    np.testing.assert_array_equal(xyt[:, 2], (s - 1) // (nx * ny) + 1)
    assert mg.sample_ids(result, [1])[0].startswith(result.generation_id)


# --------------------------------------------------------------------------
# 「この cfg の生成物か」
# --------------------------------------------------------------------------

def _do_nothing_cli(tmp_path: Path) -> Path:
    exe = tmp_path / "cli_that_ignores_cfg"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_generate_rejects_a_manifest_that_does_not_match_the_cfg(tmp_path):
    """終了コード 0 + confirmed の内部整合だけでは、CLI が cfg を無視して古い
    確定済み manifest を返した場合を見分けられない。点数を突き合わせる。
    """
    out = tmp_path / "out"
    _write_dataset(out, stack=6, nx=3, ny=2, nt=1)
    exe = _do_nothing_cli(tmp_path)
    with pytest.raises(mg.ManifestRejected, match="not the product of this request"):
        mg.generate({"outputFolder": str(out), "gridX": 40, "gridY": 40, "timeSteps": 200}, cli=exe)
    # 一致していれば通る
    ok = mg.generate({"outputFolder": str(out), "gridX": 3, "gridY": 2, "timeSteps": 1}, cli=exe)
    assert ok.generation_id == "20260907T000000000-0abcde"


def test_generate_kills_the_whole_process_group_on_timeout(tmp_path):
    """`_cli/dp` は生成段の MATLAB を exec せず子として起こす。直接の子だけを
    殺すと MATLAB が走り続け、「timeout = 失敗」のあとに成功物が現れる。
    """
    import time
    out = tmp_path / "out"
    _write_dataset(out, stack=6, nx=3, ny=2, nt=1)
    marker = tmp_path / "written_after_the_timeout"
    exe = tmp_path / "cli_with_a_grandchild"
    exe.write_text(
        f"#!/bin/sh\n/bin/sh -c 'sleep 5; echo late > {marker}' \nexit 0\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(mg.MatlabGenerationError, match="did not finish within"):
        mg.generate({"outputFolder": str(out), "gridX": 3, "gridY": 2, "timeSteps": 1},
                    cli=exe, timeout=1.0)
    time.sleep(6.0)
    assert not marker.exists(), "孫プロセスが timeout 後も走り続けている"


# --------------------------------------------------------------------------
# stack が nx*ny の倍数でないとき（2026-09-08）
#
# 生成器の場面は flat index の関数で、フレーム境界という概念を持たない
# （DistribTopLayerCheckered(FV, sk, dx, dy)）。だから素数の stack でも生成は
# 通る。壊れるのは解釈する側なので、ここで 3 通りの結果を固定する。
# --------------------------------------------------------------------------

def _partial(tmp_path, *, nx=3, ny=2, stack=7):
    """nx*ny=6 に対して stack=7 → 最後のフレームは 1 点だけ。"""
    return _write_dataset(tmp_path, stack=stack, nx=nx, ny=ny, nt=2,
                          layout={"type": "rectangular_grid_partial", "nx": nx, "ny": ny,
                                  "nt": 2, "complete": False, "fullFrames": 1,
                                  "lastFrameCount": stack - nx * ny})


def test_grid_coverage_reports_a_complete_grid(tmp_path):
    _write_dataset(tmp_path, stack=6, nx=3, ny=2, nt=1)
    cov = mg.grid_coverage(mg.load_confirmed_manifest(tmp_path))
    assert cov["type"] == "rectangular_grid" and cov["consistent"] and not cov["partial"]


def test_a_partial_grid_is_not_returned_silently(tmp_path):
    """完全な格子として黙って渡さない。明示の opt-in を要求する。"""
    _partial(tmp_path)
    result = mg.load_confirmed_manifest(tmp_path)
    cov = mg.grid_coverage(result)
    assert cov["partial"] and cov["consistent"] and cov["last_frame_count"] == 1
    with pytest.raises(mg.ManifestRejected, match="allow_partial"):
        mg.stack_coordinates(result)
    xyt = mg.stack_coordinates(result, allow_partial=True)
    assert xyt.shape == (7, 3)
    # 末尾は途中で切れたフレームの 1 点目
    np.testing.assert_array_equal(xyt[-1], np.array([1, 1, 2], dtype="f4"))


def test_sample_ids_still_work_without_any_grid(tmp_path):
    """格子が無くても識別子は使える ── 実験データの通常経路。"""
    _write_dataset(tmp_path, stack=7, nx=3, ny=2, nt=2,
                   layout={"type": "flat_samples", "stack": 7})
    result = mg.load_confirmed_manifest(tmp_path)
    # 「未対応レイアウト」の一般分岐でも UnsupportedLayout は出るので、それでは
    # 専用分岐を通ったことの証拠にならない。flat_samples 固有の理由を求める。
    with pytest.raises(mg.UnsupportedLayout, match="number of measurement points"):
        mg.stack_coordinates(result)
    assert mg.sample_ids(result)[-1] == "20260907T000000000-0abcde#7"
    assert mg.grid_coverage(result)["type"] == "flat_samples"


def test_an_inconsistent_grid_declaration_is_refused(tmp_path):
    """完全格子を名乗りながら数が合わないものは受けない。"""
    _write_dataset(tmp_path, stack=7, nx=3, ny=2, nt=1)   # 3*2*1 = 6 != 7
    result = mg.load_confirmed_manifest(tmp_path)
    assert not mg.grid_coverage(result)["consistent"]
    with pytest.raises(mg.ManifestRejected, match="sample_ids"):
        mg.stack_coordinates(result)
