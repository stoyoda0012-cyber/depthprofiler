"""``deppro.visualizer_3d`` imports without its optional rendering backends.

pyvista, plotly and napari are optional. The module sets them to ``None``
when absent, so it must not evaluate annotations such as ``-> pv.PolyData``
at import time. Checked in a fresh interpreter with the three packages
blocked, so the import state of this session is not disturbed.
"""
from __future__ import annotations

import os
import subprocess
import sys

from deppro.paths import repo_root

_BLOCK = ("import sys\n"
          "for m in ('pyvista', 'plotly', 'plotly.graph_objects', 'napari'):\n"
          "    sys.modules[m] = None\n")


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root() / "src")
    return subprocess.run([sys.executable, "-c", _BLOCK + code], capture_output=True, text=True, env=env)


def test_imports_without_backends():
    r = _run("import deppro.visualizer_3d as v\n"
             "assert (v.HAS_PYVISTA, v.HAS_PLOTLY, v.HAS_NAPARI) == (False, False, False)\n"
             "import deppro.run_napari\n"
             "print('ok')")
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_pyvista_use_still_raises_a_named_import_error():
    r = _run("import deppro.visualizer_3d as v\n"
             "try:\n"
             "    v.MatlabStyleVisualizer(sim=None)\n"
             "except ImportError as e:\n"
             "    print(e)\n")
    assert r.returncode == 0 and r.stdout.strip() == "PyVista required: pip install pyvista", r.stderr
