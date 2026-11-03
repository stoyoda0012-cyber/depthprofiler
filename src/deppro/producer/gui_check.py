"""Checks of a GUI profile. **Catch on the Python side what can be caught before starting MATLAB.**

There are two:

* `compare_profiles` — checks that the light profile and the GUI profile agree in GT, coordinates,
    roles and forward conditions. Evidence that the export did not alter the generated data.
    Differences that are allowed are named in `DECLARED_DIFFERENCES`; any other difference disqualifies.
* `readresult_preflight` — **mimics as is** the `h5read` calls that `loadFullEzdeprofFromH5` in
    `+deppro/+io/DataStore.m` makes. The MATLAB side collapses everything in a try/catch into the single line
    "Please read a "*.h5" file of depth-profiling results!", which does not say which field failed, so this
    reports name, type and shape.

Pass/fail of both is judged by the returned `ok`, not by the look of the standard output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

#: What is changed on purpose in the GUI profile. Declared together with the reason.
DECLARED_DIFFERENCES: Dict[str, str] = {
    "/ezdeprof/TransMatObs":
        "light is T(D of sample 1), the GUI is T(Dini). MATLAB holds T(Dopt)=T(Dini) before analysis."
        "Generation rebuilds T per pixel, so changing the stored value changes neither Dstk nor Xstk.",
    "/ezdeprof/elementsymbol": "shape only, (ne,) -> (ne,1). Values and order are the same.",
    "/ezdeprof/orbital": "shape only, (ne,) -> (ne,1). Values and order are the same.",
    "/ezdeprof/compound": "shape only, (ne,) -> (ne,1). Values and order are the same.",
    "/ezdeprof/prpties": "shape only, (ne,) -> (ne,1). Values and order are the same.",
}

#: What must agree: GT, coordinates, roles, forward conditions.
IDENTICAL_DATASETS: Sequence[str] = (
    "/Dstk", "/Xstk", "/NSstk", "/stackxyt",
    "/ezdeprof/subst", "/ezdeprof/free", "/ezdeprof/const", "/ezdeprof/linked",
    "/ezdeprof/charge", "/ezdeprof/linkedconst",
    "/ezdeprof/depth", "/ezdeprof/thetaobs", "/ezdeprof/stepsize", "/ezdeprof/totaldepth",
    "/ezdeprof/m", "/ezdeprof/noise", "/ezdeprof/stack", "/ezdeprof/dxdydt",
    "/ezdeprof/bindingenergy", "/ezdeprof/photonenergy", "/ezdeprof/workfunction",
    "/ezdeprof/gammaeff", "/ezdeprof/TPPver", "/ezdeprof/lambdabasic",
    "/ezdeprof/FeatureVector_Source", "/ezdeprof/SupportVector_Set",
    "/ezdeprof/SimulationOrExperiment",
)

#: What is compared by value only (the shape has a declared difference).
VALUE_ONLY_DATASETS: Sequence[str] = (
    "/ezdeprof/elementsymbol", "/ezdeprof/orbital", "/ezdeprof/compound", "/ezdeprof/prpties",
)


def _decode(a: np.ndarray) -> List[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(a).ravel()]


def compare_profiles(light_h5, gui_h5) -> Dict[str, Any]:
    """Compare the light and the GUI file. Numbers are compared byte for byte.

    `mean|d| == 0` misses signed zeros and NaN, so `tobytes()` is compared.
    """
    import h5py
    problems: List[str] = []
    checked: List[str] = []
    with h5py.File(str(light_h5), "r") as a, h5py.File(str(gui_h5), "r") as b:
        for name in IDENTICAL_DATASETS:
            key = name.lstrip("/")
            if key not in a or key not in b:
                problems.append(f"{name}: missing (light={key in a}, gui={key in b})")
                continue
            x, y = a[key][()], b[key][()]
            if x.shape != y.shape:
                problems.append(f"{name}: shape {x.shape} vs {y.shape}")
            elif x.dtype == object:
                if _decode(x) != _decode(y):
                    problems.append(f"{name}: string values differ")
            elif np.asarray(x).tobytes() != np.asarray(y).tobytes():
                d = float(np.nanmax(np.abs(np.asarray(x, float) - np.asarray(y, float))))
                problems.append(f"{name}: bytes differ (max|d|={d:.6g})")
            checked.append(name)
        for name in VALUE_ONLY_DATASETS:
            key = name.lstrip("/")
            if key not in a or key not in b:
                problems.append(f"{name}: missing")
                continue
            if _decode(a[key][()]) != _decode(b[key][()]):
                problems.append(f"{name}: values differ")
            checked.append(name)
        gui_only = sorted({"/" + k for k in _walk(b)} - {"/" + k for k in _walk(a)})
    return {"ok": not problems, "problems": problems,
            "checkedIdentical": checked, "guiOnlyDatasets": gui_only,
            "declaredDifferences": DECLARED_DIFFERENCES}


def _walk(f) -> List[str]:
    import h5py
    out: List[str] = []
    f.visititems(lambda n, o: out.append(n) if isinstance(o, h5py.Dataset) else None)
    return out


def readresult_field_names(depthprofiler_m) -> List[str]:
    """The names under `/ezdeprof` that `ReadResult_Callback` refers to. Taken again from the source."""
    src = Path(depthprofiler_m).read_text(errors="replace")
    i = src.index("function ReadResult_Callback")
    j = src.index("\nfunction ", i + 10)
    return sorted(set(re.findall(r"ezdeprof\.([A-Za-z_]\w*)", src[i:j])))


#: What `loadFullEzdeprofFromH5` reads with a rank-2 start/count (equivalent to (1,nelm)).
_BOUNDED_ROW = ("elementsymbol", "orbital", "stringcharge", "element", "prpties", "compound",
                "charge", "elementnumber", "orbitalnumber", "photonenergy", "bindingenergy",
                "valencecharge", "density", "molecularweight", "bandgap")
#: Likewise, what it reads as (nelm,3).
_BOUNDED_COL3 = ("Col",)
#: Experiment only; not written in simulation (MATLAB does not write it either).
EXPERIMENT_ONLY = ("Ismp", "Xsmp", "samplesize", "tsmp")


def readresult_preflight(h5_path, depthprofiler_m) -> Dict[str, Any]:
    """Mimic the h5read calls of `loadFullEzdeprofFromH5` and report where it would fail, in advance.

    **Passing this does not mean "it opened in the GUI"**, since this is not the real MATLAB.
    The check only runs one way: failing here means it will certainly fail there.
    """
    import h5py
    need = readresult_field_names(depthprofiler_m)
    missing: List[str] = []
    rank_issues: List[str] = []
    present: List[str] = []
    with h5py.File(str(h5_path), "r") as f:
        if "ezdeprof" not in f:
            return {"ok": False, "problems": ["/ezdeprof group is absent"]}
        g = f["ezdeprof"]
        sim = bool(np.asarray(g["SimulationOrExperiment"][()]).ravel()[0]) if "SimulationOrExperiment" in g else None
        nelm = 0
        for k in ("subst", "free", "const", "linked"):
            if k in g:
                v = np.asarray(g[k][()]).ravel().astype(int)
                nelm += int((v > 0).sum()) if k != "subst" else int(v.size)
        for name in need:
            if name in EXPERIMENT_ONLY:
                if sim:
                    continue                      # not read in simulation
            if name not in g:
                missing.append(name)
                continue
            d = g[name]
            present.append(name)
            if name in _BOUNDED_ROW and d.ndim != 2:
                rank_issues.append(f"{name}: h5py ndim={d.ndim}; MATLAB reads it with "
                                   f"start=[1 1] count=[1 {nelm}] and needs rank 2")
            if name in _BOUNDED_ROW and d.ndim == 2 and d.shape[0] < nelm:
                rank_issues.append(f"{name}: shape {d.shape}; MATLAB reads {nelm} entries")
            if name in _BOUNDED_COL3 and (d.ndim != 2 or d.shape != (3, nelm)):
                rank_issues.append(f"{name}: shape {d.shape}; expected (3, {nelm}) in h5py "
                                   f"(MATLAB ({nelm}, 3))")
        root_needed = ["Dstk", "Xstk", "NSstk", "Drlt", "Dtrn", "Xrlt", "FV0stk", "FVstk",
                       "FVrlt", "FVtrn", "SVstk", "SVrlt", "SVtrn", "Cstk", "Crlt", "Ctrn",
                       "SPSstk", "SPSrlt", "SPStrn", "Qstk", "Qrlt", "Qtrn", "degree", "alpha"]
        root_missing = [r for r in root_needed if r not in f]
    orient = check_lambdabasic_orientation(h5_path)
    problems = ([f"/ezdeprof/{m} is absent" for m in missing]
                + [f"/ezdeprof/{r}" for r in rank_issues]
                + [f"/{r} is absent (needed by the display path, not by ReadResult itself)"
                   for r in root_missing]
                + ([] if orient["ok"] else [f"/ezdeprof/lambdabasic: {orient['verdict']}"]))
    return {"ok": not problems, "problems": problems, "lambdabasicOrientation": orient,
            "elementCount": nelm, "simulation": sim,
            "ezdeprofPresent": len(present), "ezdeprofRequired": len(need),
            "experimentOnlySkipped": list(EXPERIMENT_ONLY),
            "scope": "structural only: presence, rank and element-count bounds. It does NOT "
                     "run MATLAB and says nothing about whether the GUI displays correctly."}


#: What is known not to agree when regenerated from the same cfg. **Named explicitly.**
#: Every other difference disqualifies.
NON_DETERMINISTIC: Dict[str, str] = {
    "/ezdeprof/iterationtime":
        "Measured seconds taken to build the pre-analysis state. On the MATLAB side it is also the Set time, so it changes per run."
        "The value itself is recorded in preAnalysisState.iterationtime of the manifest.",
}


def compare_generations(a_h5, b_h5) -> Dict[str, Any]:
    """Compare two generations from the same cfg (determinism check).

    Only the names listed in `NON_DETERMINISTIC` may differ; everything else must match byte for byte.
    The return value shows that the differences are "only the allowed names", so pass/fail is judged by `ok`.
    """
    import h5py
    problems: List[str] = []
    allowed: List[str] = []
    with h5py.File(str(a_h5), "r") as fa, h5py.File(str(b_h5), "r") as fb:
        na, nb = set(_walk(fa)), set(_walk(fb))
        if na != nb:
            problems.append(f"dataset name sets differ: only-a={sorted(na-nb)} only-b={sorted(nb-na)}")
        for key in sorted(na & nb):
            x, y = fa[key][()], fb[key][()]
            if getattr(x, "dtype", None) is not None and x.dtype == object:
                same = _decode(x) == _decode(y)
            else:
                same = (np.asarray(x).shape == np.asarray(y).shape
                        and np.asarray(x).tobytes() == np.asarray(y).tobytes())
            if same:
                continue
            name = "/" + key
            if name in NON_DETERMINISTIC:
                allowed.append(f"{name}: {np.ravel(x)[:1]} vs {np.ravel(y)[:1]}")
            else:
                problems.append(f"{name}: differs and is not declared non-deterministic")
    return {"ok": not problems, "problems": problems,
            "allowedDifferences": allowed, "declared": NON_DETERMINISTIC,
            "checked": len(_walk_names(a_h5))}


def _walk_names(path) -> List[str]:
    import h5py
    with h5py.File(str(path), "r") as f:
        return _walk(f)


def check_lambdabasic_orientation(h5_path, manifest_path=None) -> Dict[str, Any]:
    """Check the axis orientation of the stored `/ezdeprof/lambdabasic` **without a reference file**.

    MATLAB reads `lambdabasic(a, i, k)` (a = material, i = emitting element); h5py has the dimensions
    reversed, so the correct order is `h5[0][i][a]`. The IMFP is rebuilt from the material constants
    and binding energies in the file, and compared with both orientations.

    **A light profile does not carry the material constants (`valencecharge`/`density`/`molecularweight`/`bandgap`)**,
    so it cannot be rebuilt from the H5 alone. In that case they are taken from `resolved.forward`
    of `manifest.json` in the same folder (where they are recorded). Without a manifest either,
    `ok=None` is returned to **state explicitly that the check could not be made** (not treated as PASS).

    **The power to tell the orientation depends on the matrix.** If `L` is symmetric or has one
    component, transposing gives the same values and this check cannot tell them apart. Whether it
    can is reported separately in `orientationDiscriminating` (it does not claim "a reversal always shows").
    """
    import h5py
    from deppro.imfp import IMFP
    path = Path(str(h5_path))
    with h5py.File(str(path), "r") as f:
        g = f["ezdeprof"]
        lb = np.asarray(g["lambdabasic"][()], dtype=float)
        stored = lb[0] if lb.ndim == 3 else lb
        be = np.asarray(g["bindingenergy"][()], dtype=float).ravel()
        ph = np.asarray(g["photonenergy"][()], dtype=float).ravel()
        wf = float(np.asarray(g["workfunction"][()], dtype=float).ravel()[0])
        ver = int(np.asarray(g["TPPver"][()]).ravel()[0])
        keys = ("valencecharge", "density", "molecularweight", "bandgap")
        if all(k in g for k in keys):
            nv, rho, mw, eg = (np.asarray(g[k][()], dtype=float).ravel() for k in keys)
            source = "h5 (/ezdeprof material constants)"
        else:
            nv = rho = mw = eg = None
            source = None
    if nv is None:
        mpath = Path(manifest_path) if manifest_path else path.parent / "manifest.json"
        if not mpath.is_file():
            return {"ok": None, "verdict": "cannot check (material constants not in the H5 and no manifest)",
                    "constantsSource": None, "missing": list(keys),
                    "note": "A light profile does not write the material constants. Put manifest.json in the same folder or"
                            " pass manifest_path. **Not treated as PASS.**"}
        fw = (json.loads(mpath.read_text(encoding="utf-8")).get("resolved") or {}).get("forward") or {}
        try:
            nv = np.asarray(fw["valenceCharge"], dtype=float).ravel()
            rho = np.asarray(fw["density"], dtype=float).ravel()
            mw = np.asarray(fw["molecularWeight"], dtype=float).ravel()
            eg = np.asarray(fw["bandGap"], dtype=float).ravel()
        except KeyError as exc:
            return {"ok": None, "verdict": f"cannot check (manifest lacks {exc})",
                    "constantsSource": str(mpath), "note": "Not treated as PASS."}
        source = f"manifest {mpath.name} (resolved.forward)"

    ke = ph - be - wf
    ne = be.size
    want = np.empty((ne, ne), dtype=float)            # want[a, i]
    for a in range(ne):
        for i in range(ne):
            want[a, i] = IMFP.calculate(ke[i], nv[a], rho[a], mw[a], eg[a], ver)
    as_written = float(np.abs(stored - want.T).max())   # correct: h5[0][i][a] == L[a,i]
    transposed = float(np.abs(stored - want).max())     # wrong: h5[0][a][i]
    scale = float(np.abs(want).max())
    asym = float(np.abs(want - want.T).max())
    discriminating = asym > max(1e-6 * scale, 1e-9)
    ok = as_written <= max(1e-4 * scale, 1e-3)
    return {"ok": ok,
            "constantsSource": source,
            "maxAbsDiff_correctOrientation": as_written,
            "maxAbsDiff_transposed": transposed,
            "orientationDiscriminating": bool(discriminating),
            "asymmetry": asym,
            "verdict": "h5[0][i][a] == L[a,i] (same orientation as MATLAB)" if ok else
                       ("**written transposed** (h5[0][a][i]). MATLAB's Set/Calculation and "
                        "the analysis-side problem.py read transposed values"
                        if transposed < as_written else "matches neither orientation"),
            "note": "Self-check without a reference file. The IMFP is rebuilt from the material constants "
                    "and binding energies and compared. When `orientationDiscriminating` is False, only "
                    "numerical consistency is shown and the orientation is not determined."}
