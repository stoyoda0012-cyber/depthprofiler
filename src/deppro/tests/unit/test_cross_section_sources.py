"""Cross-section tables: toyomacro's bundled data is the default source.

Owner's decision (2026-10-02): toyomacro keeps the complete, reviewed tables
(provenance and data-rights review in its ``docs/DATA_SOURCES.md``); deppro
extracts what it needs from them. The Excel workbooks and the CSV stay
available through ``DEPPRO_CROSS_SECTION_DIR`` / ``DEPPRO_YEH_LINDAU_TABLE_PATH``
for cross-checks. Only the *source* changed: deppro keeps its own
interpolation, so the production cross sections must come out as before.

The ``PINNED_*`` values below were produced by the Excel / CSV implementation
at main ``771f531`` (before this change), except four rows read from the same
workbooks later (see the comment above ``PINNED_LOADER``). They are the gate: the
toyomacro-backed source must reproduce them to 1e-12 (the measured difference
is at most 1.5e-15, mainly from multiplying by 1e-6 before rather than after
the log-log interpolation; one case keeps a last-bit difference from the
stored values themselves).

Where the two sources are *known* to differ, outside the pinned cases:

- toyomacro's Scofield table stops at 30 keV (the workbooks go to 1.5 MeV), so
  a query above 30 keV is an extrapolation on the default source;
- a few non-production Yeh-Lindau cells differ between the workbook and
  toyomacro (transcription differences, reported to toyomacro for primary-
  source checking); the default source carries toyomacro's values;
- the reported ``energy_range`` starts at the first tabulated value instead of
  the first row of the sheet.

Below the ionisation threshold both sources now return NaN (the workbook
path used to reach NaN only through the log of its zero cells).
"""
from __future__ import annotations

import math
import os
import sys

import pytest

from deppro.haxpes import cross_section_extrap as xs_extrap
from deppro.haxpes import cross_section_loader as xs

try:
    import toyomacro  # noqa: F401  (optional sibling package, extra ``fit``)
    TOYOMACRO = True
except ImportError:
    TOYOMACRO = False


def _require_toyomacro():
    """The gate needs toyomacro. In CI its absence is a failure, not a skip
    (a gate without its dependency must go red); locally it is a
    skip, since toyomacro is an optional sibling checkout."""
    if TOYOMACRO:
        return
    if os.environ.get("CI"):
        pytest.fail("toyomacro is not installed: the cross-section gate cannot run in CI")
    pytest.skip("toyomacro not installed locally; the gate runs in CI's pytest job")


needs_toyomacro = pytest.mark.usefixtures("_toyomacro_required")


@pytest.fixture
def _toyomacro_required():
    _require_toyomacro()

ENV_DIR = "DEPPRO_CROSS_SECTION_DIR"
ENV_CSV = "DEPPRO_YEH_LINDAU_TABLE_PATH"
RTOL = 1e-12

# (source, element, orbital, photon_energy_eV) -> sigma in Mb, Excel workbooks, main 771f531.
# The W 3d5/2 and Ag 2p3/2 rows were read from the same workbooks on 2026-10-21
# (toyomacro agreed to <= 1.5e-15 relative); they replace two earlier rows.
PINNED_LOADER = {
    ('yeh_lindau', 'C', '1s', 1486.6): 0.012999999999999998,
    ('yeh_lindau', 'C', '1s', 5414.7): 0.00026414326184793054,
    ('yeh_lindau', 'C', '1s', 9251.7): 5.2553069121106675e-05,
    ('yeh_lindau', 'O', '1s', 1486.6): 0.04000000000000001,
    ('yeh_lindau', 'O', '1s', 5414.7): 0.0008723503409896015,
    ('yeh_lindau', 'O', '1s', 9251.7): 0.00017872552355075426,
    ('yeh_lindau', 'N', '1s', 9251.7): 9.866319024081429e-05,
    ('yeh_lindau', 'F', '1s', 9251.7): 0.000300477749449934,
    ('yeh_lindau', 'Na', '1s', 9251.7): 0.0006752085286475407,
    ('yeh_lindau', 'Mg', '1s', 9251.7): 0.0009506013642717644,
    ('scofield', 'C', '1s', 1486.6): 0.013634109634987639,
    ('scofield', 'C', '1s', 5414.7): 0.0002757812599680053,
    ('scofield', 'C', '1s', 9251.7): 5.0478004293827386e-05,
    ('scofield', 'O', '1s', 1486.6): 0.0399376587770616,
    ('scofield', 'O', '1s', 5414.7): 0.0009321647441428156,
    ('scofield', 'O', '1s', 9251.7): 0.00017852356193391442,
    ('scofield', 'Al', '1s', 5414.7): 0.006372605342808842,
    ('scofield', 'Al', '1s', 9251.7): 0.0013415315499531288,
    ('scofield', 'Si', '1s', 5414.7): 0.008392677887936326,
    ('scofield', 'Si', '1s', 9251.7): 0.0017984761236851522,
    ('scofield', 'Ti', '1s', 5414.7): 0.039239614237640615,
    ('scofield', 'Ti', '1s', 9251.7): 0.00965196183363545,
    ('scofield', 'W', '3d5/2', 5414.7): 0.03158596806839605,
    ('scofield', 'W', '3d5/2', 9251.7): 0.005412306291943844,
    ('scofield', 'Ag', '2p3/2', 5414.7): 0.04379540990038714,
    ('scofield', 'Ag', '2p3/2', 9251.7): 0.009086119144692257,
    ('scofield', 'Au', '4f7/2', 1486.6): 0.1304965138039196,
    ('scofield', 'Au', '4f7/2', 5414.7): 0.0013446534342226547,
    ('scofield', 'Au', '4f7/2', 9251.7): 0.00015056535836049142,
}

# (Z, photon_energy_eV) -> sigma in Mb from the CSV, main 771f531. Only the
# production-relevant region: the Z polynomial diverges for Z >= 40 and near
# the 1s threshold (Z = 22 at Cr K-alpha gives 1.3e4 Mb), which is a property
# of the algorithm, not of the source.
PINNED_EXTRAP = {
    (13, 5414.7): 0.007145550109678092,
    (13, 9251.7): 0.0010746550295288792,
    (14, 5414.7): 0.012675750990817681,
    (14, 9251.7): 0.0012769551233890346,
    (22, 9251.7): 0.008645431569828447,
}


def _no_env(monkeypatch):
    monkeypatch.delenv(ENV_DIR, raising=False)
    monkeypatch.delenv(ENV_CSV, raising=False)


def _hide_toyomacro(monkeypatch):
    for name in ("toyomacro", "toyomacro.data", "toyomacro.data.cross_section"):
        monkeypatch.setitem(sys.modules, name, None)


# ---------------------------------------------------------------- precedence

@needs_toyomacro
def test_default_source_is_toyomacro_when_installed(monkeypatch):
    _no_env(monkeypatch)
    assert xs.resolve_source() == ("toyomacro", None)
    assert xs.CrossSectionDatabase().backend == "toyomacro"
    assert xs_extrap.resolve_yeh_lindau_source() == ("toyomacro", None)


def test_env_variable_selects_the_workbooks_over_toyomacro(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_DIR, str(tmp_path))
    assert xs.resolve_source() == ("excel", tmp_path)
    db = xs.CrossSectionDatabase()
    assert db.backend == "excel" and db.db_path == tmp_path
    monkeypatch.setenv(ENV_CSV, str(tmp_path / "t.csv"))
    assert xs_extrap.resolve_yeh_lindau_source() == ("csv", tmp_path / "t.csv")


def test_explicit_directory_selects_the_workbooks_over_toyomacro(tmp_path, monkeypatch):
    _no_env(monkeypatch)
    db = xs.CrossSectionDatabase(tmp_path)
    assert db.backend == "excel" and db.db_path == tmp_path


def test_without_toyomacro_or_variables_resolution_stops_and_names_both(monkeypatch):
    """Order is variable -> toyomacro -> stop. No neighbouring checkout is searched,
    so the result does not depend on where the repository happens to sit."""
    _no_env(monkeypatch)
    _hide_toyomacro(monkeypatch)
    with pytest.raises(FileNotFoundError, match=ENV_DIR) as info:
        xs.resolve_source()
    assert "toyomacro" in str(info.value)
    with pytest.raises(FileNotFoundError, match=ENV_DIR):
        xs.CrossSectionDatabase()
    with pytest.raises(FileNotFoundError, match=ENV_CSV) as info:
        xs_extrap.resolve_yeh_lindau_source()
    assert "toyomacro" in str(info.value)
    with pytest.raises(FileNotFoundError, match=ENV_CSV):
        xs_extrap.yeh_lindau_extrapolate_1s(14, 9251.7)


def test_missing_workbooks_name_the_variable_and_toyomacro(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_DIR, str(tmp_path))
    with pytest.raises(FileNotFoundError, match=ENV_DIR) as info:
        xs.CrossSectionDatabase().get_cross_section("O", "1s", 9251.7, source="scofield")
    assert "toyomacro" in str(info.value)
    monkeypatch.setenv(ENV_CSV, str(tmp_path / "absent.csv"))
    with pytest.raises(FileNotFoundError, match=ENV_CSV) as info:
        xs_extrap.yeh_lindau_extrapolate_1s(14, 9251.7)
    assert "toyomacro" in str(info.value)


# ------------------------------------------------------ the gate: same numbers

@needs_toyomacro
def test_toyomacro_source_reproduces_the_pinned_production_values(monkeypatch):
    _no_env(monkeypatch)
    db = xs.CrossSectionDatabase()
    assert db.backend == "toyomacro"
    worst_rel, worst_case = 0.0, None
    for (source, element, orbital, energy), pinned in PINNED_LOADER.items():
        value = db.get_cross_section(element, orbital, energy, source=source).cross_section_Mb
        rel = abs(value - pinned) / pinned
        if rel > worst_rel:
            worst_rel, worst_case = rel, (source, element, orbital, energy)
        assert value == pytest.approx(pinned, rel=RTOL), (source, element, orbital, energy, value, pinned)
    assert worst_rel <= RTOL, (worst_rel, worst_case)


@needs_toyomacro
def test_toyomacro_source_reproduces_the_pinned_extrapolations(monkeypatch):
    _no_env(monkeypatch)
    assert xs_extrap.resolve_yeh_lindau_source()[0] == "toyomacro"
    for (z, energy), pinned in PINNED_EXTRAP.items():
        value = xs_extrap.yeh_lindau_extrapolate_1s(z, energy)
        assert value == pytest.approx(pinned, rel=RTOL), (z, energy, value, pinned)


SUB_THRESHOLD = [("Al", "1s", 1486.6), ("Ag", "1s", 9251.7), ("W", "2p3/2", 5414.7)]


@needs_toyomacro
def test_below_threshold_queries_are_nan_on_the_default_source(monkeypatch):
    """hv below the subshell's lowest tabulated value: NaN, flagged, never a number."""
    _no_env(monkeypatch)
    db = xs.CrossSectionDatabase()
    for element, orbital, energy in SUB_THRESHOLD:
        r = db.get_cross_section(element, orbital, energy, source="scofield")
        assert math.isnan(r.cross_section_Mb), (element, orbital, energy, r.cross_section_Mb)
        assert r.is_extrapolated and energy < r.energy_range[0]


@needs_toyomacro
def test_toyomacro_rows_cover_the_same_elements_as_the_algorithm_expects(monkeypatch):
    _no_env(monkeypatch)
    rows = xs_extrap._yeh_lindau_1s_rows()
    assert [z for z, _, _ in rows] == list(range(1, 13))   # H..Mg, ordered by Z
    assert all(len(pe) == 16 and len(cs) == 16 for _, pe, cs in rows)


# --------------------------------------------- local cross-checks (need the files)

def _excel_dir():
    """The workbook directory named by ``DEPPRO_CROSS_SECTION_DIR``, if both workbooks are there."""
    c = xs.cross_section_dir()
    if c is not None and (c / xs.SCOFIELD_WORKBOOK).is_file() and (c / xs.YEH_LINDAU_WORKBOOK).is_file():
        return c
    return None


@needs_toyomacro
def test_workbooks_and_toyomacro_agree_on_the_pinned_values(monkeypatch):
    """Local cross-check against the Excel workbooks named by DEPPRO_CROSS_SECTION_DIR.

    Skipped where that variable is unset or the workbooks are absent (CI, a
    public checkout): the private tables are not available there. Not a pass."""
    excel = _excel_dir()
    if excel is None:
        pytest.skip(f"private tables absent: {ENV_DIR} unset or the workbooks not found; "
                    "the pinned-value test above is the gate")
    _no_env(monkeypatch)
    via_toyomacro = xs.CrossSectionDatabase()
    via_excel = xs.CrossSectionDatabase(excel)
    assert via_toyomacro.backend == "toyomacro" and via_excel.backend == "excel"
    for (source, element, orbital, energy) in PINNED_LOADER:
        a = via_excel.get_cross_section(element, orbital, energy, source=source).cross_section_Mb
        b = via_toyomacro.get_cross_section(element, orbital, energy, source=source).cross_section_Mb
        assert b == pytest.approx(a, rel=RTOL), (source, element, orbital, energy, a, b)
    for element, orbital, energy in SUB_THRESHOLD:
        a = via_excel.get_cross_section(element, orbital, energy, source="scofield").cross_section_Mb
        assert math.isnan(a), ("workbook path must agree on NaN below threshold", element, orbital, energy, a)


@needs_toyomacro
def test_csv_and_toyomacro_agree_on_the_pinned_extrapolations(monkeypatch):
    """Local cross-check against the CSV named by DEPPRO_YEH_LINDAU_TABLE_PATH.

    Skipped where that variable is unset or the file is absent (CI, a public
    checkout): the private table is not available there. Not a pass."""
    csv = xs_extrap.yeh_lindau_table_path()   # the variable only; None when unset
    if csv is None or not csv.is_file():
        pytest.skip(f"private table absent: {ENV_CSV} unset or the file not found; "
                    "the pinned-value test above is the gate")
    _no_env(monkeypatch)
    via_toyomacro = {k: xs_extrap.yeh_lindau_extrapolate_1s(*k) for k in PINNED_EXTRAP}
    monkeypatch.setenv(ENV_CSV, str(csv))
    assert xs_extrap.resolve_yeh_lindau_source()[0] == "csv"
    for k, b in via_toyomacro.items():
        a = xs_extrap.yeh_lindau_extrapolate_1s(*k)
        assert b == pytest.approx(a, rel=RTOL), (k, a, b)
