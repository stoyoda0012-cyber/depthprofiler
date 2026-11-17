"""The fixtures in ``src/deppro/tests/fixtures/web/`` against the code that wrote them.

Each claim fails on its own kind of drift:

- the source files behind the numbers are the ones the fixtures were made
  from (``source_sha256``): an edit to deppro's forward model or solver fails
  here first, until the fixtures are regenerated and the diff reviewed;
- ``source_commit`` is the last commit that touched them (only where a full
  history exists);
- a fresh generation, on deppro's NumPy path, reproduces the committed files:
  inputs and structure exactly, computed values within the stored tolerance;
- the index lists every case with its digest;
- the case whose inversion assumes everything the data were made with
  recovers the generating thickness (the fixtures are not only
  self-consistent with themselves);
- where toyomacro is installed, deppro and toyomacro still give the input
  numbers the generator holds (binding energies, compound parameters,
  angular factors).

How the cases, the expected values and the tolerances are defined is
documented in ``web_fixtures.py``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import numpy as np
import pytest

from deppro.tests import web_fixtures as wf

CASES = wf.case_names()


def _load(case: str) -> dict:
    return json.loads((wf.FIXTURE_DIR / f"{case}.json").read_text(encoding="utf-8"))


def test_source_files_are_those_the_fixtures_were_made_from():
    index = wf.committed_index()
    assert index is not None, "no fixtures; run `python -m deppro.tests.web_fixtures --write`"
    assert index["source_sha256"] == wf.source_sha256(), (
        "deppro source behind the web fixtures changed: commit it, run "
        "`python -m deppro.tests.web_fixtures --write`, and review the diff")


def test_source_commit_matches_the_history():
    commit = wf.git_source_commit()
    if commit is None:
        pytest.skip("no full git history or uncommitted source files: source_commit is checked only "
                    "against a full history (source_sha256 still is)")
    assert wf.committed_index()["source_commit"] == commit


def test_index_lists_every_case_with_its_digest():
    index = wf.committed_index()
    assert [c["case"] for c in index["cases"]] == CASES
    on_disk = sorted(p.name for p in wf.FIXTURE_DIR.glob("*.json"))
    assert on_disk == sorted([c["file"] for c in index["cases"]] + ["index.json"])
    for entry in index["cases"]:
        data = (wf.FIXTURE_DIR / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        provenance = json.loads(data)["provenance"]
        assert provenance["source_sha256"] == index["source_sha256"]
        assert provenance["source_commit"] == index["source_commit"]


def test_a_fresh_generation_reproduces_the_fixtures():
    # In a child process: generation blocks numba before deppro is imported,
    # which this process (pytest, with deppro already loaded) cannot do.
    cmd, env = wf.child_command(["--emit"])
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    # the child must run this tree's deppro, not another installed one
    assert f"deppro imported from {wf.DEPPRO / '__init__.py'}" in out.stderr, out.stderr[-2000:]
    fresh = {name: text.encode("utf-8") for name, text in json.loads(out.stdout).items()}
    problems = wf.compare_files(wf.committed_files(), fresh)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("case", CASES)
def test_cases_are_synthetic_or_counts_and_say_where_inputs_come_from(case):
    data = _load(case)
    assert data["schema"] == wf.SCHEMA and data["schema_version"] == wf.SCHEMA_VERSION
    assert data["input"]["data"]["kind"] in ("synthetic", "counts")
    assert data["provenance"]["runtime_path"].startswith("NumPy")
    assert set(data["tolerance"]) == set(data["expected"])
    n_angles, n_el = len(data["input"]["theta_deg"]), len(data["input"]["elements"])
    assert len(data["expected"]["I_fit"]) == n_angles and len(data["expected"]["I_fit"][0]) == n_el
    assert len(data["expected"]["D_fit"]) == data["input"]["depth"]["n_points"]


def test_the_fully_matched_case_recovers_the_generating_thickness():
    # Same angular factor, same depth grid and lambda from the generating
    # composition: what is left is where the Brent search over thickness stops
    # (thickness_atol 0.5 A). Measured 40.0067 A for 40 A (40.0000 with a tight
    # search); the bound guards against a broken case and is not an accuracy claim.
    data = _load("eh2_hfo2_lambda_truth")
    assert data["input"]["inversion"]["L"] == data["input"]["data"]["L"]
    assert abs(data["expected"]["thickness_A"] - data["input"]["data"]["thickness_A"]) < 0.05


def test_inputs_are_what_deppro_and_toyomacro_give(monkeypatch):
    # The fixtures hold these inputs as numbers and never read toyomacro; this
    # reports when the tables they came from have moved. Skipped without the
    # ``fit`` extra.
    pytest.importorskip("toyomacro")
    import os

    from deppro.depth_profile import ElementConfig
    from deppro.haxpes.preprocessor import compute_L_factors
    from deppro.imfp import CompoundDatabase

    for key in [k for k in os.environ if k.startswith(("DEPPRO_", "TOYOMACRO_"))]:
        monkeypatch.delenv(key)
    compounds = CompoundDatabase()
    for name, params in wf.COMPOUNDS.items():
        c = compounds.get(name)
        assert (c.valence_charge, c.density, c.molecular_weight, c.band_gap) == params, name
    for (symbol, orbital), be in wf.BINDING_ENERGY_EV.items():
        assert ElementConfig(symbol, orbital, "Si").binding_energy == be, (symbol, orbital)
    for name, (elements, setup, xray, polarized) in wf.L_SOURCES.items():
        configs = [ElementConfig(e["symbol"], e["orbital"], e["compound"], oxidation_state=e["oxidation_state"])
                   for e in elements]
        L = compute_L_factors(configs, setup["hv"], np.array(setup["angles"], dtype=float),
                              xray_from_normal_deg=xray, polarized=polarized)
        np.testing.assert_allclose(L, getattr(wf, name), rtol=1e-11, atol=1e-12, err_msg=name)
