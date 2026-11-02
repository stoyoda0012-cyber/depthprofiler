"""同梱の theory の例は import しても何も実行・書き出ししない。

2026-10-18 の clean venv 検証で、import 走査がこれらの module を読んだだけで
venv の中に PNG が書かれた。実行は ``python <file>`` または ``main()`` だけ。
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from deppro.paths import repo_root

SHIPPED = ["deppro.examples.theory.theory_comparison", "deppro.examples.theory.xps_noise_summary"]


@pytest.mark.parametrize("module", SHIPPED)
def test_import_writes_nothing_and_exposes_main(module, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    env = dict(os.environ, PYTHONPATH=str(repo_root() / "src"), DEPPRO_OUTPUTS=str(out), MPLBACKEND="Agg")
    code = f"import importlib; m = importlib.import_module({module!r}); print(callable(getattr(m, 'main', None)))"
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "True", r.stderr
    assert list(out.iterdir()) == [], f"import wrote files: {list(out.iterdir())}"
    assert list(tmp_path.iterdir()) == [out], f"import wrote into cwd: {list(tmp_path.iterdir())}"
