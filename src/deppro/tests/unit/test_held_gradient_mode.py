"""SemiFreeSolver's gradient_mode is held back from the public release.

The code stays in ``inverse_solver.py`` (shared by the private and the public
tree), but ``gradient_mode=True`` raises NotImplementedError unless the
distribution enables the feature. The private tree enables it from
``deppro/_extensions.py``, which the public release does not ship.

The public behaviour is checked in a fresh interpreter in which
``deppro._extensions`` cannot be imported.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import deppro

SRC = str(Path(deppro.__file__).resolve().parents[1])

PROBE = textwrap.dedent("""
    import sys
    sys.modules['deppro._extensions'] = None          # as in the public release
    import importlib.util as u
    real = u.find_spec
    u.find_spec = lambda name, *a, **k: None if name == 'deppro._extensions' else real(name, *a, **k)
    import numpy as np
    import deppro.inverse_solver as inv
    from deppro.transfer_matrix import TransferMatrix
    depth = np.arange(0, 40, 4.0)
    tm = TransferMatrix(depth, 4.0, [0, 30, 60])
    tm.n_elements = 3
    tm.trans_mat = np.ones((3, len(depth), 3))
    assert 'gradient_mode' not in inv._ENABLED_HELD_FEATURES
    plain = inv.SemiFreeSolverConfig(free_indices=[0], linked_indices=[1], substrate_indices=[2])
    inv.SemiFreeSolver(tm, inv.SolverConfig(method='LLS'), plain)        # must construct
    graded = inv.SemiFreeSolverConfig(free_indices=[0], linked_indices=[1], substrate_indices=[2],
                                      gradient_mode=True, gradient_element_indices=[0])
    try:
        inv.SemiFreeSolver(tm, inv.SolverConfig(method='LLS'), graded)
    except NotImplementedError as e:
        print('RAISED:', e)
    else:
        print('NOT RAISED')
""")


def _run_public_probe():
    env = {"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"}
    return subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env, timeout=120)


def test_without_the_private_extensions_gradient_mode_raises():
    out = _run_public_probe()
    assert out.returncode == 0, out.stderr
    assert "RAISED:" in out.stdout and "not available in this release" in out.stdout, out.stdout


def test_gradient_mode_is_enabled_exactly_when_the_private_extensions_exist():
    """Private tree: enabled by deppro._extensions. Public release: not enabled."""
    import deppro.inverse_solver as inv
    from importlib.util import find_spec
    has_extensions = find_spec("deppro._extensions") is not None
    assert ("gradient_mode" in inv._ENABLED_HELD_FEATURES) == has_extensions
