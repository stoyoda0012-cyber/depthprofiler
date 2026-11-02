"""``deppro.haxpes.cross_section_loader``: table location and import boundary.

The loader used to live in ``deppro.experiments`` and located the Scofield /
Yeh–Lindau workbooks relative to its own ``__file__`` (a directory outside the
repository). It now lives in the tracked body and resolves that directory
through ``DEPPRO_CROSS_SECTION_DIR`` at call time. These tests
pin that behaviour, and the boundary itself: the body must not import
``deppro.experiments``, ``deppro.benchmarks`` or ``deppro.diagnostics`` (none is
part of the publishable body; all three are excluded from the wheel). Unlike
``test_repo_hygiene``, which checks module-level imports of shipped modules,
this scan also sees imports inside functions.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from deppro.paths import repo_root

ENV = "DEPPRO_CROSS_SECTION_DIR"


# ---------------------------------------------------------------------------
# Import boundary: the body does not import deppro.experiments
# ---------------------------------------------------------------------------

# Top-level packages under src/deppro that are *not* the body.
_NOT_BODY = {"experiments", "tests", "benchmarks", "diagnostics", "_debug"}


def _body_files():
    root = repo_root() / "src" / "deppro"
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if parts[0] in _NOT_BODY:
            continue
        if "tests" in parts[:-1] or "diagnostics" in parts[:-1]:
            # deppro.app.tests / deppro.app.diagnostics are excluded from the wheel too
            continue
        yield path


_DYNAMIC_IMPORTERS = {"import_module", "__import__"}


def _imported_names_in(source: str, filename: str = "<string>"):
    """Every module name a source text imports: ``import a.b``, ``from a.b import c``
    (both ``a.b`` and ``a.b.c``, so ``from deppro import experiments`` and
    ``from . import benchmarks`` are seen), and string literals handed to
    ``importlib.import_module`` / ``__import__``. Relative imports keep their dots.
    """
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            yield base
            sep = "" if base.endswith(".") or not base else "."
            for alias in node.names:
                yield f"{base}{sep}{alias.name}"
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _DYNAMIC_IMPORTERS:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        yield arg.value


def _imported_names(path: Path):
    yield from _imported_names_in(path.read_text(encoding="utf-8"), str(path))


def _targets(forbidden: str):
    short = forbidden.split(".")[-1]
    return (forbidden, forbidden + ".", "." + short, "." + short + ".", ".." + short, ".." + short + ".")


@pytest.mark.parametrize("forbidden", ["deppro.experiments", "deppro.benchmarks", "deppro.diagnostics"])
def test_body_does_not_import(forbidden):
    """Every import of ``forbidden`` from a body module is listed, with its file."""
    offenders = []
    for path in _body_files():
        for name in _offending(_imported_names(path), forbidden):
            offenders.append(f"{path.relative_to(repo_root())}: {name}")
    assert offenders == [], "\n".join(offenders)


def _offending(names, forbidden: str):
    heads = _targets(forbidden)
    return [n for n in names if n in heads or any(n.startswith(h) for h in heads if h.endswith("."))]


def test_scanner_sees_indirect_import_forms():
    """Self-check of the detector on forms that hide the package name from a naive scan."""
    src = (
        "from deppro import experiments\n"
        "from . import benchmarks\n"
        "from .. import diagnostics\n"
        "import importlib\n"
        "def f():\n"
        "    importlib.import_module('deppro.experiments.cross_section_loader')\n"
        "    __import__('deppro.benchmarks')\n"
        "from deppro.io import dstk\n"          # not forbidden
        "from .paths import env_value\n"        # not forbidden
    )
    names = list(_imported_names_in(src))
    assert _offending(names, "deppro.experiments") == ["deppro.experiments", "deppro.experiments.cross_section_loader"]
    assert _offending(names, "deppro.benchmarks") == [".benchmarks", "deppro.benchmarks"]
    assert _offending(names, "deppro.diagnostics") == ["..diagnostics"]
    # the two allowed imports are seen by the scanner, and (above) are not flagged
    assert "deppro.io.dstk" in names and ".paths.env_value" in names


def test_boundary_scan_sees_the_body():
    """Guard against an empty scan passing vacuously."""
    files = list(_body_files())
    names = {p.name for p in files}
    assert "cli_depth.py" in names and "preprocessor.py" in names
    assert not any("experiments" in p.parts for p in files)


# ---------------------------------------------------------------------------
# Table location
# ---------------------------------------------------------------------------

@pytest.fixture
def csl():
    """The loader module, imported lazily so the boundary tests above never depend on it."""
    from deppro.haxpes import cross_section_loader
    return cross_section_loader


def test_env_variable_sets_the_directory(csl, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path))
    assert csl.cross_section_dir() == tmp_path


def test_tilde_in_the_variable_is_expanded(csl, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(ENV, "~/xs")
    assert csl.cross_section_dir() == tmp_path / "xs"


def test_empty_variable_means_unset(csl, monkeypatch):
    """Unset (or empty) gives no directory at all: no neighbouring directory is assumed."""
    monkeypatch.setenv(ENV, "")
    assert csl.cross_section_dir() is None
    monkeypatch.delenv(ENV)
    assert csl.cross_section_dir() is None


def test_directory_is_resolved_at_call_time(csl, tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    default = csl.cross_section_dir()
    monkeypatch.setenv(ENV, str(tmp_path))
    assert csl.cross_section_dir() == tmp_path
    assert default != tmp_path


@pytest.mark.parametrize("loader", ["ScofieldLoader", "YehLindauLoader"])
def test_missing_workbook_names_the_variable(csl, tmp_path, monkeypatch, loader):
    monkeypatch.setenv(ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match=ENV):
        getattr(csl, loader)().load_element("Si")


def test_database_without_argument_follows_the_variable(csl, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path))
    db = csl.CrossSectionDatabase()
    assert db.db_path == tmp_path
    with pytest.raises(FileNotFoundError, match=ENV):
        db.get_cross_section("Si", "1s", 9251.7, source="scofield")


def test_explicit_directory_wins_over_the_variable(csl, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path / "from-env"))
    explicit = tmp_path / "explicit"
    db = csl.CrossSectionDatabase(explicit)
    assert db.db_path == explicit
    assert db._scofield.file_path == explicit / csl.SCOFIELD_WORKBOOK
    assert db._yeh_lindau.file_path == explicit / csl.YEH_LINDAU_WORKBOOK


def _write_scofield(csl, directory: Path) -> None:
    df = pd.DataFrame({"PhotonEnergy[keV]": [1.0, 5.0, 10.0], "1s": [10.0, 2.0, 1.0]})
    with pd.ExcelWriter(directory / csl.SCOFIELD_WORKBOOK) as xw:
        df.to_excel(xw, sheet_name="Z_14_Si", index=False)


def _write_yeh_lindau(csl, directory: Path) -> None:
    df = pd.DataFrame({"Photon Energy / eV": [100.0, 1000.0, 8000.0], "1s": [5.0, 1.0, 0.1]})
    with pd.ExcelWriter(directory / csl.YEH_LINDAU_WORKBOOK) as xw:
        df.to_excel(xw, sheet_name="14_Si", index=False)


def test_scofield_loader_reads_the_workbook_in_the_directory(csl, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path))
    _write_scofield(csl, tmp_path)
    df = csl.ScofieldLoader().load_element("Si")
    assert list(df["PhotonEnergy_eV"]) == [1000.0, 5000.0, 10000.0]


def test_yeh_lindau_loader_reads_the_workbook_in_the_directory(csl, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path))
    _write_yeh_lindau(csl, tmp_path)
    df = csl.YehLindauLoader().load_element("Si")
    assert list(df["PhotonEnergy_eV"]) == [100.0, 1000.0, 8000.0]
