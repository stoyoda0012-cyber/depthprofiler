"""GUI profile の検査。**MATLAB を起動する前に Python 側で落とせるものを落とす。**

2 つある:

* `compare_profiles` — 軽量 profile と GUI profile で、GT・座標・役割・forward 条件が
  同一かを確かめる。export が生成物を書き換えていないことの根拠にする。
  差が出てよいものは `DECLARED_DIFFERENCES` に名前で挙げ、それ以外の差は失格とする。
* `readresult_preflight` — `+deppro/+io/DataStore.m` の `loadFullEzdeprofFromH5` が
  出す `h5read` 呼び出しを**そのまま真似る**。MATLAB 側は try/catch で
  「Please read a "*.h5" file of depth-profiling results!」の 1 行に潰れてしまい、
  どのフィールドで落ちたか分からないので、ここで名前・型・shape まで出す。

どちらも合否は戻り値の `ok` で判定する。標準出力の見た目で判断しない。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

#: GUI profile で意図的に変えるもの。理由まで込みで宣言する。
DECLARED_DIFFERENCES: Dict[str, str] = {
    "/ezdeprof/TransMatObs":
        "light は T(D of sample 1)、GUI は T(Dini)。MATLAB は解析前に T(Dopt)=T(Dini) を持つ。"
        "生成は画素ごとに T を作り直すので、保存値を変えても Dstk/Xstk は変わらない。",
    "/ezdeprof/elementsymbol": "shape のみ (ne,) -> (ne,1)。値と順序は同じ。",
    "/ezdeprof/orbital": "shape のみ (ne,) -> (ne,1)。値と順序は同じ。",
    "/ezdeprof/compound": "shape のみ (ne,) -> (ne,1)。値と順序は同じ。",
    "/ezdeprof/prpties": "shape のみ (ne,) -> (ne,1)。値と順序は同じ。",
}

#: 一致していなければならないもの。GT・座標・役割・forward 条件。
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

#: 値の一致だけを見るもの（shape は宣言済みの差がある）。
VALUE_ONLY_DATASETS: Sequence[str] = (
    "/ezdeprof/elementsymbol", "/ezdeprof/orbital", "/ezdeprof/compound", "/ezdeprof/prpties",
)


def _decode(a: np.ndarray) -> List[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(a).ravel()]


def compare_profiles(light_h5, gui_h5) -> Dict[str, Any]:
    """軽量 / GUI の 2 ファイルを比べる。数値はバイト一致で見る。

    `mean|d| == 0` では署名付きゼロや NaN を見逃すので、`tobytes()` で比べる。
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
    """`ReadResult_Callback` が参照する `/ezdeprof` の名前。ソースから取り直す。"""
    src = Path(depthprofiler_m).read_text(errors="replace")
    i = src.index("function ReadResult_Callback")
    j = src.index("\nfunction ", i + 10)
    return sorted(set(re.findall(r"ezdeprof\.([A-Za-z_]\w*)", src[i:j])))


#: `loadFullEzdeprofFromH5` が rank 2 の start/count 付きで読むもの（(1,nelm) 相当）。
_BOUNDED_ROW = ("elementsymbol", "orbital", "stringcharge", "element", "prpties", "compound",
                "charge", "elementnumber", "orbitalnumber", "photonenergy", "bindingenergy",
                "valencecharge", "density", "molecularweight", "bandgap")
#: 同じく (nelm,3) で読むもの。
_BOUNDED_COL3 = ("Col",)
#: simulation では書かない（MATLAB も書かない）experiment 専用。
EXPERIMENT_ONLY = ("Ismp", "Xsmp", "samplesize", "tsmp")


def readresult_preflight(h5_path, depthprofiler_m) -> Dict[str, Any]:
    """`loadFullEzdeprofFromH5` の h5read を真似て、落ちる場所を先に出す。

    MATLAB の実物ではないので**これが通っても「GUI で開けた」とは言わない**。
    通らなければ確実に落ちる、という向きだけの検査。
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
                    continue                      # simulation では読まれない
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


#: 同じ cfg で作り直しても一致しないと分かっているもの。**名前で挙げる。**
#: これ以外の差はすべて失格。
NON_DETERMINISTIC: Dict[str, str] = {
    "/ezdeprof/iterationtime":
        "解析前の状態を組むのに要した実測秒。MATLAB 側も Set の所要時間なので実行ごとに変わる。"
        "値そのものは manifest の preAnalysisState.iterationtime に記録される。",
}


def compare_generations(a_h5, b_h5) -> Dict[str, Any]:
    """同じ cfg の 2 回の生成を突き合わせる（決定性の検査）。

    `NON_DETERMINISTIC` に挙げた名前だけ差を許し、それ以外はバイト一致を要求する。
    差が「許した名前だけ」であることを戻り値で示すので、合否は `ok` で判定する。
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
    """保存した `/ezdeprof/lambdabasic` の軸の向きを、**参照ファイル無しで**確かめる。

    MATLAB は `lambdabasic(a, i, k)`（a=物質, i=放出元素）で読み、h5py は次元が逆なので
    正しい並びは `h5[0][i][a]`。ファイル内の材料定数と結合エネルギーから IMFP を組み直し、
    2 つの向きのどちらに一致するかを見る。

    **軽量 profile には材料定数（`valencecharge`/`density`/`molecularweight`/`bandgap`）が
    書かれていない**ので、H5 だけでは組み直せない。その場合は同じフォルダの
    `manifest.json` の `resolved.forward` から取る（記録されている）。manifest も無ければ
    `ok=None` を返して**「検査できなかった」ことを明示する**（PASS 扱いにはしない）。

    **向きの識別力は行列に依る。** `L` が対称、あるいは 1 成分だと、転置しても同じ値に
    なるのでこの検査は向きを見分けられない。見分けられるかどうかを
    `orientationDiscriminating` で別に報告する（「逆なら必ず差が出る」とは書かない）。
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
            return {"ok": None, "verdict": "検査できない（材料定数が H5 に無く、manifest も無い）",
                    "constantsSource": None, "missing": list(keys),
                    "note": "軽量 profile は材料定数を書かない。manifest.json を同じフォルダに置くか、"
                            "manifest_path を渡すこと。**PASS 扱いにはしない。**"}
        fw = (json.loads(mpath.read_text(encoding="utf-8")).get("resolved") or {}).get("forward") or {}
        try:
            nv = np.asarray(fw["valenceCharge"], dtype=float).ravel()
            rho = np.asarray(fw["density"], dtype=float).ravel()
            mw = np.asarray(fw["molecularWeight"], dtype=float).ravel()
            eg = np.asarray(fw["bandGap"], dtype=float).ravel()
        except KeyError as exc:
            return {"ok": None, "verdict": f"検査できない（manifest に {exc} が無い）",
                    "constantsSource": str(mpath), "note": "PASS 扱いにはしない。"}
        source = f"manifest {mpath.name} (resolved.forward)"

    ke = ph - be - wf
    ne = be.size
    want = np.empty((ne, ne), dtype=float)            # want[a, i]
    for a in range(ne):
        for i in range(ne):
            want[a, i] = IMFP.calculate(ke[i], nv[a], rho[a], mw[a], eg[a], ver)
    as_written = float(np.abs(stored - want.T).max())   # 正しい: h5[0][i][a] == L[a,i]
    transposed = float(np.abs(stored - want).max())     # 誤り: h5[0][a][i]
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
            "verdict": "h5[0][i][a] == L[a,i]（MATLAB と同じ向き）" if ok else
                       ("**転置して書かれている**（h5[0][a][i]）。MATLAB の Set/Calculation と "
                        "解析側 problem.py は転置した値を読む"
                        if transposed < as_written else "どちらの向きとも一致しない"),
            "note": "参照ファイルを使わない自己検査。材料定数と結合エネルギーから IMFP を "
                    "組み直して比べている。`orientationDiscriminating` が False のときは "
                    "数値整合だけを示し、向きは判別していない。"}
