"""deppro.haxpes.cross_section_extrap: table location and the extrapolation itself.

The real Yeh–Lindau table is not in the repository, so the algorithm is checked
on a synthetic table whose extrapolation is exact:
log10(sigma) = 2 - 2*log10(E / 1000) + 0.1*Z is linear in log10(E) for each Z and
linear in Z at fixed E, so both polynomial fits reproduce it.
"""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from deppro.haxpes import cross_section_extrap as cse
from deppro.haxpes import preprocessor

ENV = "DEPPRO_YEH_LINDAU_TABLE_PATH"
ENERGIES = [1000.0, 1486.6, 2000.0, 4000.0, 8048.0]


def _log_sigma(z, e):
    return 2.0 - 2.0 * np.log10(e / 1000.0) + 0.1 * z


def _write_table(path: Path) -> Path:
    cols = ["AtomicNumber", "Element", "AtomicOrbital"] + [
        "CrossSec_PE_" + (f"{e:g}".replace(".", "_")) for e in ENERGIES]
    lines = [",".join(cols), ",".join(["0", "x", "x"] + ["0"] * len(ENERGIES))]
    for z in range(1, 13):
        vals = [f"{10 ** _log_sigma(z, e):.17g}" for e in ENERGIES]
        lines.append(",".join([str(z), f"E{z}", "1s"] + vals))
        lines.append(",".join([str(z), f"E{z}", "2p"] + ["7"] * len(ENERGIES)))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_env_variable_sets_the_table_path(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path / "t.csv"))
    assert cse.yeh_lindau_table_path() == tmp_path / "t.csv"


def test_empty_env_variable_means_unset(monkeypatch):
    """Unset (or empty) gives no path at all: no neighbouring directory is assumed."""
    monkeypatch.setenv(ENV, "")
    assert cse.yeh_lindau_table_path() is None
    monkeypatch.delenv(ENV)
    assert cse.yeh_lindau_table_path() is None


def test_missing_table_names_the_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path / "absent.csv"))
    with pytest.raises(FileNotFoundError, match=ENV):
        cse.yeh_lindau_extrapolate_1s(14, 9251.74)


@pytest.mark.parametrize("z,e", [(13, 9251.74), (14, 8048.0), (22, 5414.7)])
def test_extrapolation_on_an_exact_synthetic_table(tmp_path, monkeypatch, z, e):
    monkeypatch.setenv(ENV, str(_write_table(tmp_path / "t.csv")))
    got = cse.yeh_lindau_extrapolate_1s(z, e, poly_order_energy=3, poly_order_z=3)
    assert got == pytest.approx(10 ** _log_sigma(z, e), rel=1e-8)


def test_preprocessor_no_longer_imports_an_untracked_module():
    tree = ast.parse(inspect.getsource(preprocessor))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any("session92" in m for m in modules)
    assert "cross_section_extrap" in modules


def _write_curved_table(path: Path):
    """log10(sigma) = 2 - 2*u + 0.3*u**2 + 0.1*Z - 0.004*Z**2, u = log10(E / 1000).

    Quadratic in log10(E) and in Z, so order 1 cannot reproduce it and orders 2
    and 3 reproduce it exactly. This makes the two polynomial orders observable.
    """
    def log_sigma(z, e):
        u = np.log10(e / 1000.0)
        return 2.0 - 2.0 * u + 0.3 * u ** 2 + 0.1 * z - 0.004 * z ** 2

    cols = ["AtomicNumber", "Element", "AtomicOrbital"] + [
        "CrossSec_PE_" + (f"{e:g}".replace(".", "_")) for e in ENERGIES]
    lines = [",".join(cols), ",".join(["0", "x", "x"] + ["0"] * len(ENERGIES))]
    for z in range(1, 13):
        vals = [f"{10 ** log_sigma(z, e):.17g}" for e in ENERGIES]
        lines.append(",".join([str(z), f"E{z}", "1s"] + vals))
    path.write_text("\n".join(lines) + "\n")
    return path, log_sigma


@pytest.mark.parametrize("order_e,order_z,exact", [
    (2, 2, True), (3, 3, True), (1, 3, False), (3, 1, False)])
def test_polynomial_orders_matter_on_a_curved_table(tmp_path, monkeypatch, order_e, order_z, exact):
    path, log_sigma = _write_curved_table(tmp_path / "c.csv")
    monkeypatch.setenv(ENV, str(path))
    z, e = 22, 9251.74
    got = cse.yeh_lindau_extrapolate_1s(z, e, poly_order_energy=order_e, poly_order_z=order_z)
    want = 10 ** log_sigma(z, e)
    if exact:
        assert got == pytest.approx(want, rel=1e-7)
    else:
        assert abs(got / want - 1) > 1e-3


def test_tilde_in_the_variable_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(ENV, "~/t.csv")
    assert cse.yeh_lindau_table_path() == tmp_path / "t.csv"
