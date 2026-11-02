"""``deppro.app`` は兄弟パッケージ toyomacro が無い環境でも import できる。

toyomacro は optional extra ``fit`` で、``deppro.app.fitpara_reader`` だけがそれを必要とする
（2026-10-19 からは関数内で遅延 import し、呼んだ時点で名前付きエラー）。
この環境には toyomacro が入っているので、子プロセスで ``sys.modules['toyomacro'] = None`` を
置いて import を遮断し、その状態で ``deppro.app`` とその CLI-only 部分（契約層・MATLAB 生成
アダプタ）が import できることを確かめる。遮断が本当に効いていることは対照で示す。
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

BLOCK = "import sys; sys.modules['toyomacro'] = None\n"   # 以後 `import toyomacro...` は ImportError


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCK + textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=300)


def test_control_toyomacro_is_really_blocked_in_the_child():
    r = _run("import toyomacro.io.schema")
    assert r.returncode != 0 and "toyomacro" in r.stderr


def test_app_and_its_cli_only_parts_import_without_toyomacro():
    r = _run("""
        import deppro.app
        import deppro.app.contract
        import deppro.app.matlab_generation
        from deppro.app import DEPPRO_AVAILABLE, batch_solve, read_ezdeprof, run_pipeline
        print("ok", DEPPRO_AVAILABLE)
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["ok", "True"]


def test_read_fitpara_areas_is_the_only_part_that_needs_toyomacro_and_says_so():
    """2026-10-19 から fitpara_reader は toyomacro を関数内で遅延 import する: 属性の
    解決と module の import は通り、呼んだ時点で `deppro[fit]` を名指しする ImportError。"""
    r = _run("""
        import deppro.app
        import deppro.app.fitpara_reader
        fn = deppro.app.read_fitpara_areas
        try:
            fn(["/nonexistent.h5"], pixel_shape=None)
        except ImportError as e:
            print("ImportError:", e)
        else:
            print("no error")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ImportError:") and "toyomacro" in r.stdout and "deppro[fit]" in r.stdout


def test_unknown_attribute_is_still_an_attribute_error():
    r = _run("""
        import deppro.app
        try:
            deppro.app.no_such_symbol
        except AttributeError:
            print("AttributeError")
    """)
    assert r.returncode == 0 and r.stdout.strip() == "AttributeError"
