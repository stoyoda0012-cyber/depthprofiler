"""``deppro.haxpes`` imports without its DNN denoiser wrapper.

``haxpes/denoiser.py`` was private until 2026-10-18 and a checkout may still
omit it. The package ``__init__`` therefore tolerates its absence and
exposes ``HaxpesDenoiser = None`` in that case, while any *other* missing
module still raises. Both are checked in a fresh interpreter so the import
state of this test session is not disturbed. The case with the module present
is tested where the module exists.
"""
from __future__ import annotations

import os
import subprocess
import sys

from deppro.paths import repo_root


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root() / "src")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_package_imports_when_the_denoiser_module_is_absent():
    code = (
        "import sys; sys.modules['deppro.haxpes.denoiser'] = None\n"
        "import deppro.haxpes as h\n"
        "assert h.HaxpesDenoiser is None, h.HaxpesDenoiser\n"
        "assert h.HAXPESPipeline is not None\n"
        "print('ok')"
    )
    r = _run(code)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_other_missing_modules_still_raise():
    code = (
        "import sys; sys.modules['deppro.haxpes.bootstrap'] = None\n"
        "import deppro.haxpes\n"
    )
    r = _run(code)
    assert r.returncode != 0
    assert "ModuleNotFoundError" in r.stderr and "bootstrap" in r.stderr, r.stderr
