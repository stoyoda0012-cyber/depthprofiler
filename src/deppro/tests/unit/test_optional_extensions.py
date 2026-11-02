"""deppro and deppro.inference must import without the optional _extensions.

The public release does not contain ``deppro/_extensions.py`` or
``deppro/inference/_extensions.py``. Copy the package without them and
import it in a fresh interpreter; nothing else may register extra grid
types or architectures.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import deppro

PKG = Path(deppro.__file__).parent
DROP = {'_extensions.py'}


def _copy_without_extensions(dst):
    def ignore(_dir, names):
        return [n for n in names if n in DROP or n == '__pycache__']
    shutil.copytree(PKG, dst / 'deppro', ignore=ignore)


def _run(code, root):
    return subprocess.run(
        [sys.executable, '-c', code], cwd=root, capture_output=True,
        text=True, env={'PYTHONPATH': str(root), 'PYTHONDONTWRITEBYTECODE': '1',
                        'PATH': '/usr/bin:/bin'}, timeout=300)


def test_import_deppro_without_extensions(tmp_path):
    _copy_without_extensions(tmp_path)
    r = _run(
        "import deppro\n"
        "from deppro.depth_grid import GRID_TYPES\n"
        "assert deppro.__file__.startswith(%r), deppro.__file__\n"
        "assert list(GRID_TYPES) == ['uniform', 'geometric', 'log'], GRID_TYPES\n"
        % str(tmp_path), tmp_path)
    assert r.returncode == 0, r.stderr


def test_import_inference_without_extensions(tmp_path):
    pytest.importorskip('mlx.core')
    _copy_without_extensions(tmp_path)
    r = _run(
        "import deppro.inference as inf\n"
        "from deppro.inference.model import MODEL_REGISTRY\n"
        "assert inf.__file__.startswith(%r), inf.__file__\n"
        "assert list(MODEL_REGISTRY) == ['mlp', 'resmlp', 'conv1d', 'elemnet', "
        "'transformer'], list(MODEL_REGISTRY)\n"
        % str(tmp_path), tmp_path)
    assert r.returncode == 0, r.stderr


def test_error_inside_extensions_is_not_hidden(tmp_path):
    _copy_without_extensions(tmp_path)
    (tmp_path / 'deppro' / '_extensions.py').write_text(
        'import deppro_no_such_module_for_test  # noqa: F401\n')
    r = _run('import deppro\n', tmp_path)
    assert r.returncode != 0
    assert "No module named 'deppro_no_such_module_for_test'" in r.stderr
