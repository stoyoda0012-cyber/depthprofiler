"""Language-neutral fixtures for ports of the single-point depth inversion.

A port of deppro to another runtime (a browser build under Pyodide, for
instance) needs to know what deppro returns, not only that it imports.
This module writes that to ``src/deppro/tests/fixtures/web/`` as JSON, one
file per case, and checks the committed files against a fresh generation.

What a case pins
----------------
Every case is one measurement point: angle-resolved intensities of a film on
a substrate, inverted for the film thickness. The procedure, which a port
must follow to reproduce the numbers, is:

1. Elements are ``ElementConfig(symbol, orbital, compound,
   binding_energy=..., oxidation_state=...)`` with the binding energy given,
   so no table is read. The compound parameters (N_v, density, M, E_g) are
   given too and read through ``CompoundDatabase(csv_path=...)``.
2. The depth grid is ``z = step * (1, 2, ..., n)`` (the first point one step
   below the surface, as ``deppro.haxpes.pipeline``). A transfer matrix is
   ``TransferMatrix(z, step, theta, photon_energy=hv).compute(D, elements,
   compound_db, L_correction=L)`` with the default work function (4.5 eV)
   and TPP-2M (version 3); ``L`` is the angular factor, or None.
3. Data are either synthetic or counts:

   - synthetic: the substrate columns of ``D_true`` are
     ``build_substrate_profile(z, t, fwhm, substrate_fractions)`` (NumPy
     path), and each film column is ``(1 - sum of substrate columns) *
     fraction``, film fractions in the order the film elements appear in
     ``elements``. ``I_obs = transfer(D_true, L_synthetic).forward(D_true)``;
   - counts: integer counts per angle and element, ``I_obs = counts /
     counts.sum(axis=1)``.

4. The inversion builds a transfer matrix from a composition ``D_lambda`` --
   uniform ``1 / n_elements`` everywhere, or ``D_true`` ('truth', synthetic
   only) -- with ``L_inversion``, and runs ``SemiFreeSolver(tm,
   SolverConfig(method, alpha), SemiFreeSolverConfig(free/linked/substrate
   indices, thickness_init, thickness_bounds, abruptness_init),
   D_prior=None, elements).solve_alternating(I_obs)``. With ``rebuilds = k``
   it then sets ``D_lambda = D_opt`` and solves again, k times. The matrix
   is rebuilt from D_opt as in ``cli_depth``'s loop; unlike that loop, L is
   kept, alpha is not decayed, and ``solve_alternating`` is used.

``expected`` holds lambda_basic, I_obs (synthetic cases), the fraction of
the signal still left at the deepest grid point of the first inversion
matrix (``max T[:, -1, :] / T[:, 0, :]``), and the solver result: final
thickness, the thickness after each solve, the interface width (held at
its initial value: ``optimize_abruptness`` is off by default), D_opt,
I_opt, the RMS of ``I_obs - I_opt`` and the chi2 the solver reports. That
chi2 compares the substrate-normalized observation with the
total-normalized prediction at this commit, so it is pinned as a number,
not as a fit-quality figure.

The thickness is where ``solve_alternating``'s bounded Brent search stops,
at its default ``thickness_atol`` of 0.5 A, not the minimizer of the
objective: on these cases a tight search moves it by up to about 0.08 A.
Pinning it to 1e-6 A pins deppro's iterate path, which a faithful port
follows; it is not a statement that the thickness is known that well.

``informational`` holds the iteration count and the final inner SLSQP
status, which are not compared: the status flips at the same converged
point between platforms, and on one platform under perturbations of the
inputs as small as their 12-digit rounding, while thickness, D and I agree
to about 1e-11.

Where the inputs come from
--------------------------
Binding energies, compound parameters and angular factors are numbers in
this file. They were taken from toyomacro v0.7.0 (4645e20), through
``ElementConfig``, ``CompoundDatabase`` and
``deppro.haxpes.preprocessor.compute_L_factors`` at deppro c9eef51, and
rounded to 12 significant digits. The fixtures do not read toyomacro, so
they do not move when another toyomacro is installed; a port that computes
these inputs itself checks them against these numbers separately. Where
toyomacro is installed, ``test_web_fixtures.py`` also checks that deppro and
toyomacro still give these numbers (``L_SOURCES``), so a change to the
tables is reported here rather than only downstream.

Runtime path
------------
The values are those of deppro's NumPy path: generation runs with numba
blocked, as Pyodide has no numba. ``python -m deppro.tests.web_fixtures``
re-runs itself in a child process with numba blocked when deppro was
already imported with it.

Tolerances and reproducibility
------------------------------
The tolerances are meant for a port running the same algorithm on another
interpreter, NumPy, SciPy and BLAS. In one such comparison made outside
this repository -- CPython 3.12 / NumPy 2.3 / SciPy 1.16 on macOS against
Pyodide 314.0.0 (Python 3.14) / NumPy 2.4 / SciPy 1.17 -- the forward model
agreed to 3e-14 relative and no difference reached 1e-5 of these
tolerances. The tolerances are far above that and far below the 0.5 A
stopping tolerance of the thickness search. A failure means a different
code path or a changed number, not rounding. They say nothing about
accuracy: the synthetic data are made by the model that inverts them.

Computed values are stored rounded to 12 significant digits; that rounding
uses under 5% of each tolerance (lambda_basic, about 100 A, is the
largest). Regenerating on the same platform gives the same bytes; on
another, a computed value may move in its last digit, so the contract test
compares inputs and structure exactly and computed values within
tolerance. The fixture files are marked ``-text`` in git so that their
bytes, and the digests in index.json, survive a checkout with line-ending
conversion; ``source_sha256`` reads the source files with line endings
normalized to LF for the same reason.

``source_sha256`` is a digest of ``SOURCE_FILES`` as they are on disk, and
``source_commit`` the last commit that touched them. Writing requires a
full history and no uncommitted change to those files. A change to them
fails the contract test until the fixtures are regenerated: commit the
change, regenerate, and read the diff -- it is the record of what moved.
Regenerate after the branch's final rebase, and merge such a branch with a
merge commit: a squash or rebase on merge leaves ``source_commit`` naming a
commit that is not on main (checked only where a full history exists, so
not in CI's shallow clones).

Usage::

    python -m deppro.tests.web_fixtures           # check against a fresh generation
    python -m deppro.tests.web_fixtures --write   # rewrite the fixtures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

DEPPRO = Path(__file__).resolve().parents[1]          # src/deppro
REPO = DEPPRO.parents[1]
FIXTURE_DIR = DEPPRO / "tests" / "fixtures" / "web"
SCHEMA = "deppro-web-fixture"
SCHEMA_VERSION = 1
GENERATOR = "src/deppro/tests/web_fixtures.py"
DIGITS = 12

# deppro source behind every number here, relative to src/deppro.
SOURCE_FILES = (
    "_compat.py",
    "depth_profile.py",
    "imfp.py",
    "inverse_solver.py",
    "numba_kernels.py",
    "semifree_parametrization.py",
    "transfer_matrix.py",
)

# field: (atol, rtol); |fresh - committed| <= atol + rtol * |committed|
TOLERANCE = {
    "lambda_basic": (1e-9, 1e-10),
    "I_obs": (1e-12, 1e-9),
    "signal_left": (1e-12, 1e-9),
    "thickness_A": (1e-6, 1e-9),
    "thickness_history_A": (1e-6, 1e-9),
    "interface_fwhm_A": (1e-6, 1e-9),
    "D_fit": (1e-7, 1e-6),
    "I_fit": (1e-9, 1e-7),
    "residual_rms": (1e-7, 1e-6),   # never tighter than I_fit, which it is computed from
    "chi2_engine_reported": (1e-9, 1e-6),
}

# ----------------------------------------------------------------------
# Inputs (numbers only; see "Where the inputs come from")
# ----------------------------------------------------------------------

BINDING_ENERGY_EV = {("Hf", "4f"): 14.0, ("O", "1s"): 531.0, ("Si", "2p"): 99.0, ("Ti", "2p"): 454.0}

COMPOUNDS = {  # N_v, density g/cm3, M g/mol, E_g eV
    "HfO2": (16.0, 9.68, 210.49, 5.8),
    "Si": (4.0, 2.329, 28.08553, 1.12),
    "SiO2": (16.0, 2.2, 60.08, 8.9),
    "TiO2": (16.0, 4.23, 79.866, 3.1),
}

# compute_L_factors(elements, hv, theta, xray_from_normal_deg=..., polarized=...)
L_HFO2_7940_POL_M40P8 = [
    [0.544068024367, 0.950548627649, 0.653743967677], [0.607510159056, 1.10012922908, 0.693460314983],
    [0.663956484925, 1.23060144092, 0.729038268026], [0.710794583049, 1.33729394897, 0.758705627759],
    [0.745870328426, 1.41637361529, 0.780998766997], [0.767577561136, 1.46499223226, 0.794825148275],
    [0.774925, 1.4813958532, 0.79951], [0.767577561136, 1.46499223226, 0.794825148275],
    [0.745870328426, 1.41637361529, 0.780998766997], [0.710794583049, 1.33729394897, 0.758705627759],
    [0.663956484925, 1.23060144092, 0.729038268026], [0.607510159056, 1.10012922908, 0.693460314983],
    [0.544068024367, 0.950548627649, 0.653743967677],
]
L_HFO2_7940_POL_88 = [
    [1.61704253527, 1.53861035715, 1.42827801655], [1.6342064887, 1.5372265571, 1.44707100542],
    [1.64570060671, 1.52599361983, 1.46231661219], [1.64933692606, 1.50244792935, 1.47231665439],
    [1.64313080969, 1.46447505187, 1.47551079676], [1.62540682956, 1.41045178367, 1.47055486139],
    [1.59489202329, 1.33936840818, 1.45639022216], [1.55079214607, 1.25092488161, 1.43230110557],
    [1.49284727768, 1.14559579607, 1.39795714607], [1.42136404748, 1.02466034238, 1.35343918832],
    [1.33722276879, 0.890195050498, 1.29924706322], [1.24185888208, 0.74502875627, 1.23628885749],
    [1.13721924612, 0.592660954328, 1.16585201206],
]
L_TIO2_7940_POL_M40P8 = [
    [0.720476439848, 0.950548627649, 0.653743967677], [0.786460110916, 1.10012922908, 0.693460314983],
    [0.844783060125, 1.23060144092, 0.729038268026], [0.89294786937, 1.33729394897, 0.758705627759],
    [0.928897214821, 1.41637361529, 0.780998766997], [0.951098152969, 1.46499223226, 0.794825148275],
    [0.958605, 1.4813958532, 0.79951], [0.951098152969, 1.46499223226, 0.794825148275],
    [0.928897214821, 1.41637361529, 0.780998766997], [0.89294786937, 1.33729394897, 0.758705627759],
    [0.844783060125, 1.23060144092, 0.729038268026], [0.786460110916, 1.10012922908, 0.693460314983],
    [0.720476439848, 0.950548627649, 0.653743967677],
]
L_HFO2_925174_UNPOL_120 = [
    [0.60635706541, 0.779472915676, 0.709833134755], [0.671186501687, 0.886066114642, 0.749468665612],
    [0.741556011895, 0.994696728523, 0.793702420297], [0.816300305705, 1.10295857411, 0.841821537345],
    [0.894030484856, 1.20821927626, 0.892940877181], [0.973179987011, 1.30770428533, 0.946033443583],
    [1.05205965326, 1.39859262736, 0.999967526236], [1.128919665, 1.47812065, 1.053549015],
    [1.20201576479, 1.54368956074, 1.10556709777], [1.26967695544, 1.59297226532, 1.15484139234],
    [1.33037177323, 1.62401491777, 1.20026848606], [1.38277025868, 1.63532869678, 1.24086586832],
    [1.42579890205, 1.6259676227, 1.27581134083],
]
L_SIO2_14866_UNPOL_120 = [
    [0.9233, 1.00837558, 0.9233], [1.05461691844, 1.22960080488, 1.05461691844],
    [1.16924231407, 1.40059967668, 1.16924231407], [1.2515, 1.49775, 1.2515],
    [1.28825576831, 1.50484632931, 1.28825576831], [1.27186361389, 1.41654505981, 1.27186361389],
    [1.20245, 1.24049942, 1.20245],
]

# Poisson counts (1e5 per angle, numpy default_rng(7)) drawn from the
# eh2_hfo2_uniform synthetic intensities; stored as integers so that no
# random-number stream is part of the contract.
COUNTS_EH2_HFO2 = [
    [7905, 27313, 64804], [8369, 29521, 61910], [8700, 31665, 59556], [9122, 33954, 56981],
    [9646, 35826, 54765], [10020, 37933, 52074], [10482, 40046, 49223], [11062, 42192, 46642],
    [12047, 44867, 43192], [12860, 48146, 39717], [14128, 50290, 35485], [15373, 54458, 30545],
    [17176, 58272, 24416],
]


def _angles(start, stop, step):
    n = int(math.floor((stop - start) / step + 1e-9)) + 1
    return [round(start + i * step, 6) for i in range(n)]


def _element(symbol, orbital, compound, q, role):
    return {"symbol": symbol, "orbital": orbital, "compound": compound,
            "binding_energy_eV": BINDING_ENERGY_EV[(symbol, orbital)], "oxidation_state": q, "role": role}


HFO2_SI = [_element("Hf", "4f", "HfO2", 4, "free"), _element("O", "1s", "HfO2", -2, "linked"),
           _element("Si", "2p", "Si", 0, "substrate")]
TIO2_SI = [_element("Ti", "2p", "TiO2", 4, "free"), _element("O", "1s", "TiO2", -2, "linked"),
           _element("Si", "2p", "Si", 0, "substrate")]
SIO2_SI = [_element("Si", "2p", "SiO2", 4, "free"), _element("O", "1s", "SiO2", -2, "linked"),
           _element("Si", "2p", "Si", 0, "substrate")]
FILM = [1 / 3, 2 / 3]


def _case(description, hv, angles, elements, depth, data, inversion):
    return {"description": description, "photon_energy_eV": hv, "theta_deg": angles,
            "elements": elements, "compounds": {c: dict(zip(("valence_charge", "density", "molecular_weight",
                                                               "band_gap"), COMPOUNDS[c]))
                                                for c in sorted({e["compound"] for e in elements})},
            "depth": depth, "data": data, "inversion": inversion}


def _synthetic(t, w, L, film=FILM, substrate=(1.0,)):
    return {"kind": "synthetic", "thickness_A": t, "interface_fwhm_A": w, "film_fractions": list(film),
            "substrate_fractions": list(substrate), "L": L}


def _inversion(L, method="LLS", alpha=None, t0=25.0, bounds=(5.0, 150.0), w=5.0, lam="uniform", rebuilds=0):
    return {"L": L, "method": method, "alpha": alpha, "thickness_init_A": t0, "thickness_bounds_A": list(bounds),
            "interface_fwhm_A": w, "lambda_composition": lam, "rebuilds": rebuilds}


EH2 = dict(hv=7940.0, angles=_angles(25.2, 73.2, 4), depth={"step_A": 2.0, "n_points": 300})
GAKA = dict(hv=9251.74, angles=_angles(2, 50, 4), depth={"step_A": 2.0, "n_points": 400})
ALKA = dict(hv=1486.6, angles=_angles(0, 60, 10), depth={"step_A": 1.0, "n_points": 200})
EH2_DATA = _synthetic(40.0, 5.0, L_HFO2_7940_POL_M40P8)

# How each angular-factor table was computed:
# compute_L_factors(elements, hv, theta, xray_from_normal_deg=..., polarized=...)
L_SOURCES = {
    "L_HFO2_7940_POL_M40P8": (HFO2_SI, EH2, -40.8, True),
    "L_HFO2_7940_POL_88": (HFO2_SI, EH2, 88.0, True),
    "L_TIO2_7940_POL_M40P8": (TIO2_SI, EH2, -40.8, True),
    "L_HFO2_925174_UNPOL_120": (HFO2_SI, GAKA, 120.0, False),
    "L_SIO2_14866_UNPOL_120": (SIO2_SI, ALKA, 120.0, False),
}


def _cases():
    eh2 = lambda d, data, inv, els=HFO2_SI: _case(d, EH2["hv"], EH2["angles"], els, EH2["depth"], data, inv)  # noqa: E731
    gaka_data = _synthetic(40.0, 5.0, L_HFO2_925174_UNPOL_120)
    return {
        "eh2_hfo2_uniform": eh2(
            "HfO2/Si at 7.94 keV, linear polarization (source at -40.8 deg); lambda from a uniform "
            "composition, as deppro.haxpes.pipeline", EH2_DATA, _inversion(L_HFO2_7940_POL_M40P8)),
        "eh2_hfo2_lambda_truth": eh2(
            "as eh2_hfo2_uniform, lambda from the generating composition", EH2_DATA,
            _inversion(L_HFO2_7940_POL_M40P8, lam="truth")),
        "eh2_hfo2_rebuild2": eh2(
            "as eh2_hfo2_uniform, then the transfer matrix rebuilt from the solution twice", EH2_DATA,
            _inversion(L_HFO2_7940_POL_M40P8, rebuilds=2)),
        "eh2_hfo2_inversion_without_l": eh2(
            "data with the angular factor, inverted without it", EH2_DATA, _inversion(None)),
        "eh2_hfo2_inversion_source_88": eh2(
            "data with the source at -40.8 deg, inverted with it at 88 deg (compute_L_factors' default)",
            EH2_DATA, _inversion(L_HFO2_7940_POL_88)),
        "eh2_hfo2_l2": eh2("as eh2_hfo2_uniform, L2 with alpha 1", EH2_DATA,
                           _inversion(L_HFO2_7940_POL_M40P8, method="L2", alpha=1.0)),
        "eh2_hfo2_mem": eh2("as eh2_hfo2_uniform, MEM with alpha 1", EH2_DATA,
                            _inversion(L_HFO2_7940_POL_M40P8, method="MEM", alpha=1.0)),
        "eh2_hfo2_counts": eh2(
            "Poisson counts drawn from the eh2_hfo2_uniform intensities, inverted as eh2_hfo2_uniform",
            {"kind": "counts", "counts": COUNTS_EH2_HFO2}, _inversion(L_HFO2_7940_POL_M40P8)),
        "eh2_tio2_uniform": eh2(
            "TiO2/Si, otherwise as eh2_hfo2_uniform", _synthetic(40.0, 5.0, L_TIO2_7940_POL_M40P8),
            _inversion(L_TIO2_7940_POL_M40P8), els=TIO2_SI),
        "gaka_hfo2_unpolarized": _case(
            "HfO2/Si at 9251.74 eV (Ga K-alpha), unpolarized (beam direction 120 deg), 800 A grid",
            GAKA["hv"], GAKA["angles"], HFO2_SI, GAKA["depth"], gaka_data, _inversion(L_HFO2_925174_UNPOL_120)),
        "gaka_hfo2_unpolarized_rebuild2": _case(
            "as gaka_hfo2_unpolarized, rebuilt from the solution twice", GAKA["hv"], GAKA["angles"], HFO2_SI,
            GAKA["depth"], gaka_data, _inversion(L_HFO2_925174_UNPOL_120, rebuilds=2)),
        "alka_sio2_unpolarized": _case(
            "SiO2/Si at 1486.6 eV (Al K-alpha), unpolarized (beam direction 120 deg); the angular-factor "
            "table is clamped below 1.5 keV", ALKA["hv"], ALKA["angles"], SIO2_SI, ALKA["depth"],
            _synthetic(15.0, 3.0, L_SIO2_14866_UNPOL_120),
            _inversion(L_SIO2_14866_UNPOL_120, t0=10.0, bounds=(2.0, 60.0), w=3.0)),
    }


def case_names() -> list[str]:
    return list(_cases())


# ----------------------------------------------------------------------
# The procedure (NumPy path)
# ----------------------------------------------------------------------

# Run in a fresh interpreter: block numba before deppro is imported, then run
# this module as __main__.
BOOT = '''
import runpy, sys
class _BlockNumba:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "numba":
            raise ImportError("numba is blocked: the web fixtures pin deppro's NumPy path")
sys.meta_path.insert(0, _BlockNumba())
sys.argv[0] = "deppro.tests.web_fixtures"
import deppro
print("deppro imported from", deppro.__file__, file=sys.stderr)
runpy.run_module("deppro.tests.web_fixtures", run_name="__main__", alter_sys=True)
'''


CHILD_FLAG = "DEPPRO_WEB_FIXTURES_CHILD"


def child_command(args):
    """(argv, env) that run this module with numba blocked, importing deppro from this tree."""
    import os
    path = os.pathsep.join(p for p in (str(DEPPRO.parent), os.environ.get("PYTHONPATH", "")) if p)
    return [sys.executable, "-c", BOOT, *args], dict(os.environ, PYTHONPATH=path, **{CHILD_FLAG: "1"})


def numba_loaded() -> bool:
    return any(m.split(".")[0] == "numba" for m in sys.modules)


def _require_numpy_path():
    from deppro import numba_kernels, transfer_matrix
    if numba_loaded() or numba_kernels.NUMBA_AVAILABLE or transfer_matrix.NUMBA_AVAILABLE:
        raise RuntimeError("numba is loaded; generate through main(), which blocks it")


def _round(x):
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    return float(f"{float(x):.{DIGITS}g}")


def compound_db(compounds, tmp: Path):
    """CompoundDatabase read from the case's numbers (the public csv_path route)."""
    from deppro.imfp import CompoundDatabase
    path = tmp / "compounds.csv"
    rows = ["CompoundName,ValenceCharge,Density [g/cm3],MolecularWeight [g/mol],BandGap [eV]"]
    rows += [f"{n},{p['valence_charge']!r},{p['density']!r},{p['molecular_weight']!r},{p['band_gap']!r}"
             for n, p in compounds.items()]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return CompoundDatabase(csv_path=path)


def element_configs(case):
    from deppro.depth_profile import ElementConfig
    return [ElementConfig(e["symbol"], e["orbital"], e["compound"], binding_energy=e["binding_energy_eV"],
                          oxidation_state=e["oxidation_state"]) for e in case["elements"]]


def depth_grid(case):
    import numpy as np
    step = case["depth"]["step_A"]
    return step * np.arange(1, case["depth"]["n_points"] + 1), step


def roles(case):
    return {r: [i for i, e in enumerate(case["elements"]) if e["role"] == r] for r in ("free", "linked", "substrate")}


def truth_profile(case, z):
    import numpy as np
    from deppro.semifree_parametrization import build_substrate_profile
    data, idx = case["data"], roles(case)
    D = np.zeros((len(z), len(case["elements"])))
    D_sub = build_substrate_profile(z, data["thickness_A"], data["interface_fwhm_A"],
                                    np.array(data["substrate_fractions"]), use_numba=False)
    D[:, idx["substrate"]] = D_sub
    share = 1.0 - D_sub.sum(axis=1)
    for k, i in enumerate(sorted(idx["free"] + idx["linked"])):
        D[:, i] = share * data["film_fractions"][k]
    return D


def solve(case):
    """Run the procedure on one case; returns (expected, informational), unrounded."""
    import numpy as np
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig
    from deppro.transfer_matrix import TransferMatrix
    _require_numpy_path()
    hv, theta = case["photon_energy_eV"], np.array(case["theta_deg"], dtype=float)
    els, idx = element_configs(case), roles(case)
    z, step = depth_grid(case)
    arr = lambda L: None if L is None else np.array(L, dtype=float)  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        db = compound_db(case["compounds"], Path(tmp))
        transfer = lambda D, L: TransferMatrix(z, step, theta, photon_energy=hv).compute(  # noqa: E731
            D, els, compound_db=db, L_correction=L)
        expected = {}
        data = case["data"]
        if data["kind"] == "synthetic":
            D_true = truth_profile(case, z)
            I_obs = transfer(D_true, arr(data["L"])).forward(D_true)
            expected["I_obs"] = I_obs.tolist()
        else:
            counts = np.array(data["counts"], dtype=float)
            I_obs, D_true = counts / counts.sum(axis=1, keepdims=True), None
        inv = case["inversion"]
        D_lam = D_true if inv["lambda_composition"] == "truth" else np.full((len(z), len(els)), 1.0 / len(els))
        cfg = SemiFreeSolverConfig(free_indices=idx["free"], linked_indices=idx["linked"],
                                   substrate_indices=idx["substrate"], thickness_init=inv["thickness_init_A"],
                                   thickness_bounds=tuple(inv["thickness_bounds_A"]),
                                   abruptness_init=inv["interface_fwhm_A"])
        alpha = 1.0 if inv["alpha"] is None else inv["alpha"]
        history = []
        for k in range(inv["rebuilds"] + 1):
            tm = transfer(D_lam, arr(inv["L"]))
            if k == 0:
                expected["lambda_basic"] = tm.lambda_basic.tolist()
                expected["signal_left"] = float(np.max(tm.trans_mat[:, -1, :] / tm.trans_mat[:, 0, :]))
            solver = SemiFreeSolver(transfer_matrix=tm, config=SolverConfig(method=inv["method"], alpha=alpha),
                                    semi_free_config=cfg, D_prior=None, elements=els)
            r = solver.solve_alternating(I_obs, verbose=False)
            history.append(float(r.thickness))
            D_lam = np.asarray(r.D_opt)
    I_fit = np.asarray(r.I_opt)
    expected.update({
        "thickness_A": float(r.thickness),
        "thickness_history_A": history,
        "interface_fwhm_A": float(r.abruptness),
        "D_fit": np.asarray(r.D_opt).tolist(),
        "I_fit": I_fit.tolist(),
        "residual_rms": float(np.sqrt(np.mean((I_obs - I_fit) ** 2))),
        "chi2_engine_reported": float(r.chi2),
    })
    info = {"n_iterations": int(r.n_iterations), "success": bool(r.success), "message": str(r.opt_message)}
    return expected, info


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

def _git(*args) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_sha256() -> str:
    h = hashlib.sha256()
    for rel in SOURCE_FILES:
        text = (DEPPRO / rel).read_bytes().replace(b"\r\n", b"\n")
        h.update(rel.encode() + b"\0" + text + b"\0")
    return h.hexdigest()


def git_source_commit() -> str | None:
    """Last commit touching SOURCE_FILES, or None (shallow clone, no git, or uncommitted edits)."""
    if _git("rev-parse", "--is-shallow-repository") != "false":
        return None
    paths = [f"src/deppro/{p}" for p in SOURCE_FILES]
    if _git("status", "--porcelain", "--", *paths):
        return None
    return _git("log", "-1", "--format=%H", "--", *paths) or None


def committed_index() -> dict | None:
    path = FIXTURE_DIR / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ----------------------------------------------------------------------
# Generation and comparison
# ----------------------------------------------------------------------

def _dump(obj) -> bytes:
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def generate(commit: str) -> dict[str, bytes]:
    """All fixture files, name -> bytes."""
    digest = source_sha256()
    provenance = {"source_commit": commit, "source_sha256": digest, "source_files": list(SOURCE_FILES),
                  "generator": GENERATOR, "runtime_path": "NumPy (numba blocked), as under Pyodide"}
    files, index = {}, []
    for name, case in _cases().items():
        expected, info = solve(case)
        doc = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "case": name,
               "provenance": provenance, "input": case,
               "expected": {k: _round(v) for k, v in expected.items()},
               "tolerance": {k: {"absolute": a, "relative": r} for k, (a, r) in TOLERANCE.items() if k in expected},
               "informational": info}
        files[f"{name}.json"] = _dump(doc)
        index.append({"case": name, "file": f"{name}.json",
                      "sha256": hashlib.sha256(files[f"{name}.json"]).hexdigest()})
    files["index.json"] = _dump({"schema": SCHEMA + "-index", "schema_version": SCHEMA_VERSION,
                                 "source_commit": commit, "source_sha256": digest, "cases": index})
    return files


def _flat(x):
    if isinstance(x, list):
        for v in x:
            yield from _flat(v)
    else:
        yield x


def within(expected, actual, atol, rtol) -> float:
    """max |actual - expected| / (atol + rtol |expected|); inf on a shape or finiteness mismatch."""
    e, a = list(_flat(expected)), list(_flat(actual))
    if len(e) != len(a) or not all(math.isfinite(v) for v in e + a):
        return math.inf
    return max((abs(x - y) / (atol + rtol * abs(x)) for x, y in zip(e, a)), default=0.0)


def compare_files(committed: dict[str, bytes], fresh: dict[str, bytes]) -> list[str]:
    """Differences that matter: anything but computed values exactly, computed values beyond tolerance."""
    problems = []
    if sorted(committed) != sorted(fresh):
        return [f"file sets differ: committed {sorted(committed)}, fresh {sorted(fresh)}"]
    for name in sorted(fresh):
        if name == "index.json":
            c, f = json.loads(committed[name]), json.loads(fresh[name])
            for doc in (c, f):
                for entry in doc["cases"]:
                    entry.pop("sha256")
            if c != f:
                problems.append("index.json differs beyond the file digests")
            continue
        c, f = json.loads(committed[name]), json.loads(fresh[name])
        ce, fe = c.pop("expected"), f.pop("expected")
        c.pop("informational"), f.pop("informational")
        if c != f:
            keys = sorted(k for k in set(c) | set(f) if c.get(k) != f.get(k))
            problems.append(f"{name}: differs in {keys}")
        if sorted(ce) != sorted(fe):
            problems.append(f"{name}: expected fields differ: {sorted(ce)} vs {sorted(fe)}")
            continue
        for key in ce:
            atol, rtol = TOLERANCE[key]
            ratio = within(ce[key], fe[key], atol, rtol)
            if ratio > 1:
                problems.append(f"{name}: {key} max |diff|/tol = {ratio:.3g}")
    return problems


def committed_files() -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in FIXTURE_DIR.glob("*.json")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite the fixtures")
    parser.add_argument("--emit", action="store_true", help="print a fresh generation as JSON (for the tests)")
    args = parser.parse_args(argv)
    if numba_loaded():   # deppro was imported with numba: run again with it blocked
        import os
        if os.environ.get(CHILD_FLAG):
            raise RuntimeError("numba is loaded although it was blocked (imported before the blocker?)")
        cmd, env = child_command(argv if argv is not None else sys.argv[1:])
        return subprocess.call(cmd, env=env)
    if args.write:
        commit = git_source_commit()
        if commit is None:
            print("cannot write: need a full git history and no uncommitted change to "
                  + ", ".join(SOURCE_FILES), file=sys.stderr)
            return 2
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        files = generate(commit)
        for old in FIXTURE_DIR.glob("*.json"):
            if old.name not in files:
                old.unlink()
        for name, data in files.items():
            (FIXTURE_DIR / name).write_bytes(data)
        print(f"wrote {len(files)} files to {FIXTURE_DIR.relative_to(REPO)} (source {commit[:10]})")
        return 0
    index = committed_index()
    commit = index["source_commit"] if index else git_source_commit()
    if commit is None:
        print("no fixtures and no exact source commit; run --write", file=sys.stderr)
        return 2
    fresh = generate(commit)
    if args.emit:
        sys.stdout.write(json.dumps({k: v.decode("utf-8") for k, v in fresh.items()}))
        return 0
    problems = compare_files(committed_files(), fresh)
    for p in problems:
        print(p)
    print("fixtures match a fresh generation" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
