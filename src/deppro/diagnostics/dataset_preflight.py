"""4D 深さプロファイル データセットの fast preflight（読み取り専用・秒単位・範囲限定）。

MATLAB 側（DepthProfiler GUI / dpRun）が書いた
``{Standard,Checkered}/{Nx}x{Ny}x{Nt}_{N}elem[_variant]/`` フォルダを対象に

  1. 完成性・構造  — ファイル構成、必須データセットの有無、shape の相互整合
  2. 意味の一致    — 元素表と役割インデックスの分割、深さ格子、角度
  3. 物理的閉包    — 生成式 (producer-declared/1, 2026-09-08 確定) で GT から X を作り格納 X と比べる。
                      noise = 0 のデータセットは **PASS/FAIL**（float32 格納精度基準の許容差）、noise > 0 は記録。
                      保存 ``TransMatObs`` は単一 stack の T で GT のどの画素の作用素とも一致しないので記録のみ

を、**検査した画素範囲を明記して**判定し、JSON 台帳に書く。全 chunk 走査・full hash・
サイズ間一貫性は qualification（別トラック）であり、ここでは行わない。
fast pass は全画素の数値正確性を意味しない。

数値の合否は構造的な恒等式にだけ置く（Σ_elem = 1 の float32 丸め、値域 [0, 1]）。
物理閉包は ``deppro.app.contract.metrics.closure_gate`` に従う（生成式と許容差は生成器の仕様）。

状態:
  PASS / FAIL   構造・恒等式の合否
  WARN          由来が分かっている不一致（例: GUI 確認用コピーの derived_view、fencepost 格子）
  UNRESOLVED    この検査では決められない（同居 manifest 不在、役割宣言なし、noise 版）
  RECORDED      値を記録するだけ
  NA            適用外
  ERROR         ツール側の例外（データの合否ではない）

使い方::

    python -m deppro.diagnostics.dataset_preflight /path/to/Checkered/10x10x200_7elem_noisy
    python -m deppro.diagnostics.dataset_preflight --all                 # $DEPPRO_SIMULATION_ROOT
    python -m deppro.diagnostics.dataset_preflight --all --root /path/to/Simulation/DepthProfiler

終了コード: 0 = 走査した全データセットに FAIL なし / 1 = FAIL あり / 2 = ERROR あり。
元ファイルは ``h5py.File(path, "r")`` でしか開かない。出力は ``--out-dir``（既定は
``deppro.paths.ensure_outputs_dir("preflight")`` = リポジトリ直下 ``outputs/``、ignore 済み）。
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import platform
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np

from deppro.paths import ensure_outputs_dir, simulation_root

TOOL = "deppro.diagnostics.dataset_preflight"
# 0.3.0: dxdydt(3) = TimeSize (was misdiagnosed as a stale default); stored T = T(Dopt)
SCHEMA_VERSION = "0.4.0"   # 0.4.0: structure.layout (complete / partial last frame / undeclared); sampling walks
                           #        frames of nx*ny even when stack is not a multiple; scope.nt / last_frame_count

PASS, FAIL, WARN, UNRESOLVED, RECORDED, NA, ERROR = (
    "PASS", "FAIL", "WARN", "UNRESOLVED", "RECORDED", "NA", "ERROR")

from deppro.app.contract.metrics import (  # 生成式 (producer-declared/1) と閉包ゲート
    CLOSURE_TOL_ABS_NOISEFREE, REFORWARD_FORMULA_VERSION, composition_transfer_matrix, forward as _forward,
)
from deppro.app.contract.dataset_files import (  # 共通部品（規約・主 H5 の特定・decode）
    FAMILIES, FOLDER_RE, PRODUCER_NAME_RE, SPECTRA_RE, FolderSpec, decode_strings,
    discover_files, jsonable, matlab_index_list, parse_dataset_folder,
)

SUM_TOL = 1e-3       # Σ_elem = 1 の float32 丸め余裕（7 成分の和）。恒等式にのみ使う
RANGE_TOL = 1e-6     # 値域 [0, 1] の余裕
CLOSURE_MASK = 0.02  # 相対残差を取る X の下限
DXDYDT_T_IS_TIMESIZE = "DXDYDT_T_IS_TIMESIZE"   # dxdydt(3) は TimeSize（producer 2026-09-08）。nt ではない
PARTIAL_LAST_FRAME = "PARTIAL_LAST_FRAME"                       # 末尾フレームが欠ける（manifest layout が宣言、中身と一致）
STACK_NOT_MULTIPLE_OF_GRID_UNDECLARED = "STACK_NOT_MULTIPLE_OF_GRID_UNDECLARED"   # 欠けるのに宣言が無い
LAYOUT_DECLARATION_MISMATCH = "LAYOUT_DECLARATION_MISMATCH"   # manifest layout の nx/ny/nt/lastFrameCount が中身と違う
FLAT_SAMPLES = "FLAT_SAMPLES"                                 # 格子なし（実験データ: stack は測定点数）。manifest が宣言
DEPTH_GRID_FENCEPOST_LEGACY = "DEPTH_GRID_FENCEPOST_LEGACY"  # totaldepth = (Z-1)*step の旧規約
GEN_PROGRESS_MISMATCH = "GEN_PROGRESS_MISMATCH"              # currentstack != stack
NOISE_VARIANT_CONTRADICTION = "NOISE_VARIANT_CONTRADICTION"  # フォルダ名 noisy と ezdeprof/noise が矛盾
DSTK_SUM_EXCEEDS_1_LOCALIZED = "DSTK_SUM_EXCEEDS_1_LOCALIZED"  # Σ_elem D > 1 が少数の深さに局在
DSTK_SUM_NOT_1 = "DSTK_SUM_NOT_1"

REQUIRED_DATASETS = ("Dstk", "Xstk")
REQUIRED_META = ("dxdydt", "stack", "depth", "stepsize", "thetaobs", "noise",
                 "elementsymbol", "charge", "free", "linked", "const", "subst",
                 "TransMatObs")
OPTIONAL_META = ("totaldepth", "slice", "gammaeff", "stringcharge", "orbital",
                 "photonenergy", "SimulationOrExperiment", "calculationinitialized",
                 "currentstack", "transfermatrixupdatevalue")


# --------------------------------------------------------------------------
# data classes
# --------------------------------------------------------------------------
@dataclass
class Item:
    id: str
    status: str
    detail: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


_decode = decode_strings
_idx_list = matlab_index_list
_jsonable = jsonable


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def read_meta(f: h5py.File) -> Dict[str, Any]:
    g = f.get("ezdeprof")
    meta: Dict[str, Any] = {}
    if g is None:
        return meta
    for k in REQUIRED_META + OPTIONAL_META:
        if k in g:
            meta[k] = decode_strings(g[k][()])
    return meta


def _layout_mismatch(layout: Dict[str, Any], want: Dict[str, int]) -> Optional[str]:
    """manifest layout の宣言値と中身から決めた値の食い違い（宣言に無いキーは比べない）。"""
    bad = []
    for k, v in want.items():
        if k in layout:
            try:
                same = int(layout[k]) == int(v)
            except (TypeError, ValueError):
                same = False
            if not same:
                bad.append(f"{k}={layout[k]!r} but file implies {v}")
    return ", ".join(bad) or None


def _sample_frames(nt: int, n_frames: int, seed: int) -> List[int]:
    """先頭・中央・末尾は必ず含め、残りを seed 固定で無作為に足す。"""
    base = {0, nt // 2, nt - 1}
    rest = [i for i in range(nt) if i not in base]
    k = max(0, min(n_frames - len(base), len(rest)))
    if k:
        rng = np.random.default_rng(seed)
        base |= {int(i) for i in rng.choice(rest, size=k, replace=False)}
    return sorted(base)


# --------------------------------------------------------------------------
# the preflight
# --------------------------------------------------------------------------
def run_preflight(folder: Path, *, n_frames: int = 5, max_px: int = 20000,
                  seed: int = 0, root: Optional[Path] = None, closure_px: int = 400) -> Dict[str, Any]:
    """1 データセットフォルダの fast preflight。例外は ERROR 項目として報告に閉じ込める。"""
    t0 = time.perf_counter()
    folder = Path(folder).expanduser().resolve()
    items: List[Item] = []
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "tool": TOOL,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "folder": str(folder),
        "folder_rel": (str(folder.relative_to(Path(root).expanduser().resolve()))
                       if root and folder.is_relative_to(Path(root).expanduser().resolve())
                       else None),
        "spec": None, "files": {}, "scope": {}, "meta": {}, "items": [],
        "verdict": {}, "environment": _environment(), "elapsed_s": None,
    }
    try:
        _run_checks(folder, items, report, n_frames=n_frames, max_px=max_px, seed=seed, closure_px=closure_px)
    except Exception as exc:  # noqa: BLE001  — ツール側の例外はデータの合否と区別して残す
        items.append(Item("tool.error", ERROR,
                          {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc(limit=6)}))
    report["items"] = [dataclasses.asdict(i) for i in items]
    report["verdict"] = _verdict(items)
    report["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return jsonable(report)


def _run_checks(folder: Path, items: List[Item], report: Dict[str, Any], *,
                n_frames: int, max_px: int, seed: int, closure_px: int = 400) -> None:
    # ---- 1. naming --------------------------------------------------------
    spec = parse_dataset_folder(folder)
    if spec is None:
        items.append(Item("naming.parse", WARN, {"folder": folder.name},
                          "規約 {Nx}x{Ny}x{Nt}_{N}elem[_variant] に一致しない。"
                          "nx/ny/nt の期待値なしで続行"))
    else:
        report["spec"] = dataclasses.asdict(spec)
        items.append(Item("naming.parse", PASS if spec.family else WARN,
                          dataclasses.asdict(spec),
                          "" if spec.family else "親フォルダが Standard/Checkered でない"))

    # ---- 1. files ---------------------------------------------------------
    files = discover_files(folder)
    report["files"] = {
        "main": [p.name for p in files["main"]],
        "derived": [p.name for p in files["derived"]],
        "spectra": [p.name for p in files["spectra"]],
        "prj": [p.name for p in files["prj"]],
        "main_rule": files["rule"],
        "manifest_json": files["manifest"].exists(),
        "other_json": [p.name for p in files["other_json"]],
    }
    all_h5 = files["main"] + files["derived"] + files["spectra"]
    bad_sig = [p.name for p in all_h5 if not h5py.is_hdf5(str(p))]
    items.append(Item("files.hdf5_signature", FAIL if bad_sig else PASS,
                      {"n_h5": len(all_h5), "invalid": bad_sig},
                      "拡張子 .h5 だが HDF5 署名がない（途中書き・破損・別形式）" if bad_sig else ""))
    if any(p.name in bad_sig for p in files["main"]):
        _manifest_items(files, items, report)
        return
    # dpIngest が GUI 検証のために作る派生コピー。生成データセットではない
    dv = files["derived_view"]
    if dv.exists() and not files["manifest"].exists():
        try:
            dvd = json.loads(dv.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            dvd = {}
        parent = dvd.get("parent") or {}
        report["dataset_kind"] = "derived_view"
        items.append(Item("files.derived_view", WARN,
                          {"parent_folder": parent.get("folder"), "parent_generationId": parent.get("generationId"),
                           "purpose": dvd.get("purpose")},
                          "GUI 検証用の派生コピー（derived_view.json）。生成データセットではなく、ベンチ対象・"
                          "ループ入口にしない。値の正本は parent.folder"))
    else:
        report["dataset_kind"] = "dataset"
    if len(files["main"]) != 1:
        items.append(Item("files.main_h5_unique", FAIL,
                          {"n_main": len(files["main"]), "rule": files["rule"],
                           "candidates": [p.name for p in files["main"] + files["derived"]]},
                          "主 H5 を 1 本に特定できない（prj 同名なし、または複数）"))
        _manifest_items(files, items, report)
        return
    items.append(Item("files.main_h5_unique", PASS,
                      {"main": files["main"][0].name, "rule": files["rule"],
                       "derived": [p.name for p in files["derived"]]},
                      "派生 H5 は独立標本として数えない" if files["derived"] else ""))
    items.append(Item("files.prj_present", PASS if files["prj"] else WARN,
                      {"prj": [p.name for p in files["prj"]]}))
    main = files["main"][0]
    manifest = _manifest_items(files, items, report, main_name=main.name)
    with h5py.File(main, "r") as f:
        # ---- 1. datasets / meta present ----------------------------------
        missing_ds = [k for k in REQUIRED_DATASETS if k not in f]
        g = f.get("ezdeprof")
        missing_meta = ([k for k in REQUIRED_META if g is None or k not in g])
        if missing_ds or missing_meta:
            items.append(Item("structure.datasets_present", FAIL,
                              {"missing_datasets": missing_ds, "missing_meta": missing_meta}))
            return
        items.append(Item("structure.datasets_present", PASS,
                          {"datasets": REQUIRED_DATASETS, "meta": REQUIRED_META}))

        meta = read_meta(f)
        D, X = f["Dstk"], f["Xstk"]
        T = np.asarray(f["ezdeprof/TransMatObs"][()], dtype=np.float64)
        P, E, Z = D.shape
        P2, E2, A = X.shape
        depth = np.asarray(meta["depth"], dtype=np.float64).ravel()
        theta = np.asarray(meta["thetaobs"], dtype=np.float64).ravel()
        symbols = list(meta["elementsymbol"])
        charge = np.asarray(meta["charge"], dtype=np.float64).ravel()
        stack = int(np.asarray(meta["stack"]).ravel()[0])
        dxdydt = [int(v) for v in np.asarray(meta["dxdydt"]).ravel()]
        report["meta"] = {
            "shapes": {"Dstk": list(D.shape), "Xstk": list(X.shape),
                       "TransMatObs": list(T.shape), "Dstk_chunks": D.chunks,
                       "Xstk_chunks": X.chunks, "Dstk_dtype": str(D.dtype),
                       "Xstk_dtype": str(X.dtype)},
            "dxdydt": dxdydt, "stack": stack,
            "depth_A": depth.tolist(), "stepsize_A": float(np.asarray(meta["stepsize"]).ravel()[0]),
            "thetaobs_deg": theta.tolist(),
            "noise": float(np.asarray(meta["noise"]).ravel()[0]),
            "elementsymbol": symbols, "charge": charge.tolist(),
            "free_1based": matlab_index_list(meta["free"]), "linked_1based": matlab_index_list(meta["linked"]),
            "const_1based": matlab_index_list(meta["const"]), "subst_1based": matlab_index_list(meta["subst"]),
            "optional": {k: jsonable(meta[k]) for k in OPTIONAL_META if k in meta},
        }

        # ---- 1. shapes ----------------------------------------------------
        conds = {
            "Dstk.px == Xstk.px": P == P2,
            "Dstk.elem == Xstk.elem": E == E2,
            "Dstk.depth == len(depth)": Z == depth.size,
            "Xstk.angle == len(thetaobs)": A == theta.size,
            "elem == len(elementsymbol)": E == len(symbols),
            "elem == len(charge)": E == charge.size,
            "stack == Dstk.px": stack == P,
            "TransMatObs.shape == (elem, depth, angle)": tuple(T.shape) == (E, Z, A),
        }
        items.append(Item("structure.shapes", PASS if all(conds.values()) else FAIL,
                          {"conditions": conds, "P": P, "E": E, "Z": Z, "A": A}))

        # ---- 1. layout: 格子は完全か、末尾フレームが欠けるか（producer の 3 状態 layout と同じ規約）--
        # stack が nx*ny の倍数でない（素数 stack など）データは生成側では正当。解析側はそれを
        # 「全体を 1 フレーム」と黙って読み替えず、manifest の宣言と突き合わせて記録する。
        layout = (manifest or {}).get("layout") if isinstance(manifest, dict) else None
        layout = layout if isinstance(layout, dict) else None
        declared_type = layout.get("type") if layout else None
        g_nx, g_ny = ((spec.nx, spec.ny) if spec is not None
                      else ((int(dxdydt[0]), int(dxdydt[1])) if len(dxdydt) == 3 else (0, 0)))
        ppf_grid = g_nx * g_ny
        n_full, rem = divmod(P, ppf_grid) if ppf_grid > 0 else (0, P)
        nt_content = n_full + (1 if rem else 0)          # 欠けた末尾フレームも数える（layout.nt と同じ）
        last_frame_count = rem if rem else ppf_grid
        sim_or_exp = (int(np.asarray(meta["SimulationOrExperiment"]).ravel()[0])
                      if "SimulationOrExperiment" in meta else None)
        lay_detail = {"declared": layout is not None, "declared_type": declared_type,
                      "nx": g_nx, "ny": g_ny, "nt": nt_content, "n_full_frames": n_full,
                      "last_frame_count": last_frame_count, "n_px": P, "code": None,
                      "SimulationOrExperiment": sim_or_exp}
        partial_ok = False
        flat_declared = declared_type == "flat_samples"
        if flat_declared:
            # 格子は宣言されていない。stack は測定点数で、フォルダ名・dxdydt の格子は無関係。
            # 標本抽出は全体を 1 フレームとして扱う（これは表現であって格子の主張ではない）
            items.append(Item("structure.layout", RECORDED, {**lay_detail, "code": FLAT_SAMPLES, "nt": 1,
                                                             "n_full_frames": 1, "last_frame_count": P},
                              "格子なし（flat_samples）。stack は flat な標本列として全部有効。格子の検査は適用しない"))
        elif ppf_grid <= 0:
            items.append(Item("structure.layout", NA, lay_detail, "格子が決まらない（フォルダ名が規約外で dxdydt も不正）"))
        elif rem == 0:
            bad = (_layout_mismatch(layout, {"nx": g_nx, "ny": g_ny, "nt": n_full})
                   if declared_type == "rectangular_grid" else None)
            if bad:
                items.append(Item("structure.layout", FAIL, {**lay_detail, "code": LAYOUT_DECLARATION_MISMATCH, "mismatch": bad},
                                  "manifest の layout が中身と合わない"))
            else:
                items.append(Item("structure.layout", PASS, lay_detail,
                                  "完全な矩形格子（stack = nx*ny*nt）" + ("" if layout else "。manifest 宣言なし、中身から")))
        elif declared_type == "rectangular_grid_partial":
            bad = _layout_mismatch(layout, {"nx": g_nx, "ny": g_ny, "nt": n_full + 1, "lastFrameCount": rem})
            if bad:
                items.append(Item("structure.layout", FAIL, {**lay_detail, "code": LAYOUT_DECLARATION_MISMATCH, "mismatch": bad},
                                  "部分格子の宣言が中身と合わない"))
            else:
                partial_ok = True
                items.append(Item("structure.layout", RECORDED, {**lay_detail, "code": PARTIAL_LAST_FRAME},
                                  f"末尾フレームが {rem}/{ppf_grid} 画素で欠ける部分格子（manifest が宣言、中身と一致）。"
                                  "sample_id は flat な標本列として全部有効"))
        else:
            items.append(Item("structure.layout", FAIL, {**lay_detail, "code": STACK_NOT_MULTIPLE_OF_GRID_UNDECLARED},
                              f"stack {P} が nx*ny={ppf_grid} の倍数でなく（余り {rem}）、manifest に部分格子の宣言が無い。"
                              "切り詰めか誤った格子か決められない"
                              + ("。SimulationOrExperiment=0 なので実験データ（格子なし）の可能性: manifest に "
                                 "layout flat_samples の宣言が要る" if sim_or_exp == 0 else "")))

        # ---- 1. folder vs h5 ----------------------------------------------
        if flat_declared:
            items.append(Item("structure.folder_vs_h5", RECORDED,
                              {"h5_px": P, "folder_n_px": spec.n_px if spec else None,
                               "folder_ncomp": spec.ncomp if spec else None, "h5_elem": E, "code": FLAT_SAMPLES},
                              "格子なし（flat_samples）: フォルダ名の格子は無関係。元素数だけが照合対象"
                              + ("" if spec is None or spec.ncomp == E else "（元素数が違う）")))
        elif spec is not None:
            conds2 = {"nx*ny*nt == Dstk.px": spec.n_px == P, "ncomp == elem": spec.ncomp == E}
            # 部分格子ではフォルダ名の nt は要求値（フル フレーム数）でも、切れたフレーム込みでもよい
            nt_partial_ok = partial_ok and spec.nt in (n_full, nt_content) and spec.ncomp == E
            status = PASS if all(conds2.values()) else (RECORDED if nt_partial_ok else FAIL)
            items.append(Item("structure.folder_vs_h5", status,
                              {"conditions": conds2, "folder_n_px": spec.n_px, "h5_px": P,
                               "folder_nt": spec.nt, "h5_nt_from_content": nt_content,
                               "n_full_frames": n_full, "last_frame_count": last_frame_count,
                               "folder_ncomp": spec.ncomp, "h5_elem": E,
                               "code": PARTIAL_LAST_FRAME if status == RECORDED else None},
                              "フォルダ名の nt は要求値。末尾フレームが欠ける分だけ画素数が違う（layout 参照）"
                              if status == RECORDED else ""))
        else:
            items.append(Item("structure.folder_vs_h5", NA, {}, "フォルダ名が規約外"))

        # ---- 1. dxdydt = [nx, ny, TimeSize] -----------------------------------------
        # 第 3 成分は nt ではなく TimeSize（producer 2026-09-08。8 つの manifest で axes.timeSize と一致）。
        # nt は中身から stack/(nx*ny) で決める（folder_vs_h5 で別に見る）。09-06〜08 の版がこれを
        # 「既定値 [1 1 100] の残留」と判定していたのは誤診で、生成側への「nt に同期」依頼も撤回。
        nx_h5, ny_h5, t3 = dxdydt if len(dxdydt) == 3 else (0, 0, 0)
        if spec is not None:
            xy_ok = (nx_h5, ny_h5) == (spec.nx, spec.ny)
        else:
            xy_ok = nx_h5 > 0 and ny_h5 > 0          # 倍数かどうかは structure.layout が見る
        time_size = (((manifest or {}).get("resolved") or {}).get("axes") or {}).get("timeSize")
        if flat_declared:
            items.append(Item("structure.dxdydt", RECORDED,
                              {"dxdydt": dxdydt, "manifest_timeSize": time_size, "code": FLAT_SAMPLES},
                              "格子なし（flat_samples）なので dxdydt の格子は照合しない。記録のみ"))
        elif not xy_ok:
            items.append(Item("structure.dxdydt", FAIL,
                              {"dxdydt": dxdydt, "expected_nx_ny": [getattr(spec, "nx", None),
                                                                     getattr(spec, "ny", None)]},
                              "第 1・2 成分が格子と合わない"))
        elif time_size is not None:
            same = int(t3) == int(time_size)
            items.append(Item("structure.dxdydt", PASS if same else FAIL,
                              {"dxdydt": dxdydt, "manifest_timeSize": time_size, "nt_from_content": nt_content,
                               "code": DXDYDT_T_IS_TIMESIZE},
                              "" if same else "第 3 成分 (TimeSize) が manifest の axes.timeSize と違う"))
        else:
            items.append(Item("structure.dxdydt", RECORDED,
                              {"dxdydt": dxdydt, "timeSize_from_h5": t3, "nt_from_content": nt_content,
                               "code": DXDYDT_T_IS_TIMESIZE},
                              "第 3 成分は TimeSize（nt ではない）。manifest が無いので照合先なし。nt は中身から"))

        # ---- 1. generation progress marker (semantics unconfirmed: recorded) --
        if "currentstack" in meta:
            cs = int(np.asarray(meta["currentstack"]).ravel()[0])
            # 実データでは完成済みファイルでも currentstack != stack が大半（GUI のカーソル等の
            # 可能性）。意味が producer 契約で確定するまで記録のみ。判定に使わない
            items.append(Item("meta.generation_progress", RECORDED,
                              {"currentstack": cs, "stack": stack, "equal": cs == stack,
                               "code": None if cs == stack else GEN_PROGRESS_MISMATCH},
                              "ezdeprof/currentstack の意味は未確定。記録のみ"))

        # ---- 2. folder variant vs stored noise ----------------------------
        noise_val = report["meta"]["noise"]
        if spec is not None:
            says_noisy = "noisy" in (spec.variant or "")
            contradict = (says_noisy and noise_val == 0) or (not says_noisy and noise_val != 0)
            items.append(Item("meta.noise_vs_variant", WARN if contradict else PASS,
                              {"variant": spec.variant, "noise": noise_val,
                               "code": NOISE_VARIANT_CONTRADICTION if contradict else None},
                              "フォルダ名の noisy と ezdeprof/noise が矛盾" if contradict else ""))

        # ---- 2. element roles ---------------------------------------------
        roles = {"free": matlab_index_list(meta["free"]), "linked": matlab_index_list(meta["linked"]),
                 "const": matlab_index_list(meta["const"]), "subst": matlab_index_list(meta["subst"])}
        all_idx = [i for v in roles.values() for i in v]
        in_range = all(1 <= i <= E for i in all_idx)
        disjoint = len(all_idx) == len(set(all_idx))
        covering = set(all_idx) == set(range(1, E + 1))
        table = []
        for i in range(1, E + 1):
            role = next((r for r, v in roles.items() if i in v), None)
            table.append({"index_1based": i, "symbol": symbols[i - 1] if i <= len(symbols) else None,
                          "charge": float(charge[i - 1]) if i <= charge.size else None, "role": role})
        items.append(Item("meta.element_roles",
                          PASS if (in_range and disjoint and covering) else FAIL,
                          {"in_range": in_range, "disjoint": disjoint, "covering": covering,
                           "components": table, "n_distinct_symbols": len(set(symbols))}))
        n_spec = len(files["spectra"])
        items.append(Item("files.spectra_vs_symbols",
                          PASS if n_spec == len(set(symbols)) else WARN,
                          {"n_spectra_files": n_spec, "n_distinct_symbols": len(set(symbols)),
                           "spectra": [p.name for p in files["spectra"]]},
                          "" if n_spec == len(set(symbols)) else
                          "元素別スペクトル本数が distinct symbol 数と違う（記録のみ）"))

        # ---- 2. depth grid ------------------------------------------------
        step = float(np.asarray(meta["stepsize"]).ravel()[0])
        expected = np.arange(Z) * step
        dev = float(np.max(np.abs(depth - expected))) if depth.size == Z else float("inf")
        tol = 1e-3 * max(step, 1.0)
        dconds = {"depth == i*stepsize (boundary-start)": dev <= tol,
                  "monotone": bool(np.all(np.diff(depth) > 0))}
        if "slice" in meta:
            dconds["slice == len(depth)"] = int(np.asarray(meta["slice"]).ravel()[0]) == Z
        convention = None
        if "totaldepth" in meta:
            td = float(np.asarray(meta["totaldepth"]).ravel()[0])
            if abs(td - Z * step) <= tol:
                convention = "totaldepth = Z*step (current writer)"
            elif abs(td - (Z - 1) * step) <= tol:
                convention = "totaldepth = (Z-1)*step (fencepost, legacy writer)"
            dconds["totaldepth matches a known convention"] = convention is not None
        if not all(dconds.values()):
            st, code = FAIL, None
        elif convention and convention.startswith("totaldepth = (Z-1)"):
            st, code = WARN, DEPTH_GRID_FENCEPOST_LEGACY
        else:
            st, code = PASS, None
        items.append(Item("meta.depth_grid", st,
                          {"conditions": dconds, "stepsize_A": step, "n_depth": Z, "max_dev_A": dev,
                           "depth_start": "boundary (depth[0]=0, MATLAB)",
                           "totaldepth_convention": convention, "code": code},
                          "旧 writer の (Z-1)*step 規約。世代間で刻みが違う（producer_drift 候補）"
                          if code else
                          "Python 側の中央式との Δz/2 差は既知（未解決）"))

        # ---- 2. angles ----------------------------------------------------
        aconds = {"monotone_increasing": bool(np.all(np.diff(theta) > 0)),
                  "within_0_90_deg": bool(np.all((theta >= 0) & (theta <= 90)))}
        items.append(Item("meta.angles", PASS if all(aconds.values()) else FAIL,
                          {"conditions": aconds, "n_angles": A,
                           "range_deg": [float(theta.min()), float(theta.max())]}))
        items.append(Item("meta.angle_reference", UNRESOLVED,
                          {"assumed": "degrees from surface normal"},
                          "角度の基準（法線/表面）と単位は manifest の宣言待ち"))

        # ---- 1. bounded sample: finite / range / Σ=1 / zero pixels -------
        # フレームは格子 nx*ny で切る。stack が倍数でなくても「全体を 1 フレーム」と読み替えない
        # （そうすると先頭 max_px 画素しか見ないことになる）。末尾フレームは実画素数で切る。
        px_per_frame = P if flat_declared else (ppf_grid if ppf_grid > 0 else P)
        nt_eff = -(-P // px_per_frame) if px_per_frame else 1
        frames = _sample_frames(nt_eff, n_frames, seed)
        ranges = []
        finite = True
        dmin = xmin = float("inf")
        dmax = xmax = -float("inf")
        max_dev_D = max_dev_X = 0.0
        n_zero_D = n_zero_X = 0
        n_checked = 0
        dev_by_depth = np.zeros(Z)
        n_cells_dev = n_cells = 0
        dsum_min, dsum_max = float("inf"), -float("inf")
        for fr in frames:
            start = fr * px_per_frame
            cnt = min(px_per_frame, max_px, P - start)          # 末尾フレームは欠けうる
            Db = np.asarray(D[start:start + cnt], dtype=np.float64)
            Xb = np.asarray(X[start:start + cnt], dtype=np.float64)
            ranges.append([int(start), int(start + cnt)])
            n_checked += cnt
            finite &= bool(np.isfinite(Db).all() and np.isfinite(Xb).all())
            if not finite:
                continue
            dmin, dmax = min(dmin, float(Db.min())), max(dmax, float(Db.max()))
            xmin, xmax = min(xmin, float(Xb.min())), max(xmax, float(Xb.max()))
            zD = np.all(Db == 0, axis=(1, 2))
            zX = np.all(Xb == 0, axis=(1, 2))
            n_zero_D += int(zD.sum())
            n_zero_X += int(zX.sum())
            if (~zD).any():
                ds = Db[~zD].sum(axis=1)                            # (n, Z)
                max_dev_D = max(max_dev_D, float(np.abs(ds - 1.0).max()))
                dev_by_depth = np.maximum(dev_by_depth, np.abs(ds - 1.0).max(axis=0))
                n_cells_dev += int((np.abs(ds - 1.0) > SUM_TOL).sum())
                n_cells += int(ds.size)
                dsum_min, dsum_max = min(dsum_min, float(ds.min())), max(dsum_max, float(ds.max()))
            if (~zX).any():
                max_dev_X = max(max_dev_X, float(np.abs(Xb[~zX].sum(axis=1) - 1.0).max()))
        report["scope"] = {
            "frames_checked": frames, "px_per_frame": px_per_frame, "nt": nt_eff,
            "last_frame_count": P if flat_declared else last_frame_count,
            "layout": ("flat_samples" if flat_declared else
                       ("rectangular_grid_partial" if rem else "rectangular_grid")),
            "px_checked_per_frame": min(px_per_frame, max_px),
            "pixel_ranges": ranges, "n_px_checked": n_checked, "n_px_total": P,
            "fraction_checked": n_checked / P if P else None, "seed": seed,
            "note": "bounded sample; fast pass does not certify every pixel",
        }
        items.append(Item("sample.finite", PASS if finite else FAIL,
                          {"n_px_checked": n_checked, "frames": frames}))
        if finite:
            rconds = {"Dstk in [0,1]": dmin >= -RANGE_TOL and dmax <= 1 + RANGE_TOL,
                      "Xstk in [0,1]": xmin >= -RANGE_TOL and xmax <= 1 + RANGE_TOL}
            items.append(Item("sample.value_range", PASS if all(rconds.values()) else FAIL,
                              {"conditions": rconds, "Dstk_min_max": [dmin, dmax],
                               "Xstk_min_max": [xmin, xmax], "tolerance": RANGE_TOL}))
            items.append(Item("sample.xstk_sum_to_one", PASS if max_dev_X <= SUM_TOL else FAIL,
                              {"max_dev": max_dev_X, "tolerance": SUM_TOL,
                               "excluded_all_zero_px": n_zero_X},
                              "観測 X は角度ごとに元素総和 1（total 正規化）の恒等式"))
            dev_depths = [int(i) for i in np.nonzero(dev_by_depth > SUM_TOL)[0]]
            if max_dev_D <= SUM_TOL:
                st, code = PASS, None
            else:
                st = UNRESOLVED
                localized = dsum_min >= 1.0 - SUM_TOL and len(dev_depths) <= max(3, Z // 4)
                code = DSTK_SUM_EXCEEDS_1_LOCALIZED if localized else DSTK_SUM_NOT_1
            items.append(Item("sample.dstk_composition_sum", st, {
                "max_dev": max_dev_D, "tolerance": SUM_TOL, "sum_min": dsum_min, "sum_max": dsum_max,
                "depth_indices_deviating": dev_depths,
                "depth_A_deviating": [float(depth[i]) for i in dev_depths if i < depth.size],
                "fraction_cells_deviating": (n_cells_dev / n_cells) if n_cells else None,
                "excluded_all_zero_px": n_zero_D, "code": code},
                "" if st == PASS else
                "GT の元素総和が 1 でない深さがある。組成の正規化規則は producer 契約待ちで"
                "合否を付けない（SemiFree 表現は Σ flc·C = 1 で Σ_elem は恒等式でない）"))
            if n_checked and n_zero_D == n_checked and n_zero_X < n_checked:
                items.append(Item("sample.gt_all_zero", FAIL,
                                  {"all_zero_Dstk": n_zero_D, "all_zero_Xstk": n_zero_X,
                                   "n_px_checked": n_checked},
                                  "検査範囲の Dstk が全て 0 で Xstk は非ゼロ。GT が書かれていない"
                                  "（完成性の欠落。物理的ゼロの解釈は成り立たない）"))
            items.append(Item("sample.zero_pixels",
                              PASS if (n_zero_D == 0 and n_zero_X == 0) else UNRESOLVED,
                              {"all_zero_Dstk": n_zero_D, "all_zero_Xstk": n_zero_X,
                               "n_px_checked": n_checked},
                              "" if (n_zero_D == 0 and n_zero_X == 0) else
                              "全ゼロ画素あり。物理的ゼロか初期化漏れかはここでは決めない"))

        # ---- 3a. closure gate with the producer-declared forward (bounded subset) ----------
        if finite and "lambdabasic" in g and "gammaeff" in g:
            lam_b = np.asarray(g["lambdabasic"][()], dtype=np.float64).reshape(E, E)
            gam = float(np.asarray(g["gammaeff"][()]).ravel()[0])
            gate_ranges = [[r0, min(r0 + closure_px, r1)] for r0, r1 in ranges]
            Dg = np.concatenate([np.transpose(np.asarray(D[a:b], dtype=np.float64), (0, 2, 1)) for a, b in gate_ranges])
            Xg = np.concatenate([np.transpose(np.asarray(X[a:b], dtype=np.float64), (0, 2, 1)) for a, b in gate_ranges])
            Tg = composition_transfer_matrix(Dg, lam_b, theta, step, gam)
            Xp = _forward(Tg, Dg)
            diff = np.abs(Xp - Xg); mk = Xg > CLOSURE_MASK
            noise_val = report["meta"]["noise"]
            per_px = diff.reshape(Dg.shape[0], -1).max(axis=1)
            detail = {"formula_version": REFORWARD_FORMULA_VERSION, "n_px": int(Dg.shape[0]),
                      "pixel_ranges": gate_ranges, "noise": noise_val,
                      "max_abs": float(diff.max()), "median_abs": float(np.median(diff)),
                      "rel_residual_median": float(np.median(diff[mk] / Xg[mk])) if mk.any() else None,
                      "tol_abs": CLOSURE_TOL_ABS_NOISEFREE,
                      "float32_floor_abs": float(np.finfo(np.float32).eps * Xg.max()),
                      "n_px_over_tol": int((per_px > CLOSURE_TOL_ABS_NOISEFREE).sum()), "gt_renormalised": False}
            if noise_val == 0:
                ok = diff.max() <= CLOSURE_TOL_ABS_NOISEFREE
                items.append(Item("closure.per_pixel_T", PASS if ok else FAIL, detail,
                                  "noise 0: 生成式で GT から X が float32 精度で再現される" if ok else
                                  "noise 0 なのに生成式で X を再現できない。生成器の版か forward が producer-declared/1 と違う"
                                  "（GT を再正規化して合わせない）"))
            else:
                items.append(Item("closure.per_pixel_T", RECORDED, detail,
                                  "noise > 0: 残差にノイズ実現が乗るので合否なし（noise モデルの版は契約待ち）"))
        else:
            items.append(Item("closure.per_pixel_T", NA, {}, "lambdabasic / gammaeff が無いか非有限"))

        # ---- 3b. stored single T (record only; it is NOT any GT pixel's operator) -------
        if finite and tuple(T.shape) == (E, Z, A):
            D0 = np.asarray(D[0], dtype=np.float64)
            X0 = np.asarray(X[0], dtype=np.float64)
            Xp = np.einsum("eza,ez->ea", T, D0)
            s = Xp.sum(axis=0, keepdims=True)
            Xp = np.divide(Xp, s, out=np.zeros_like(Xp), where=s > 0)
            m = X0 > CLOSURE_MASK
            rel = np.abs(Xp - X0)[m] / X0[m] if m.any() else np.array([])
            rel_full = np.where(m, np.abs(Xp - X0) / np.where(m, X0, 1.0), 0.0)
            # 残差が元素ごとの定数因子（感度係数の欠落）で説明できるか: 元素別スケールを当てて再評価
            raw = np.einsum("eza,ez->ea", T, D0)
            scale = np.ones(E)
            for e in range(E):
                if m[e].any() and np.all(raw[e][m[e]] > 0):
                    scale[e] = float(np.median(X0[e][m[e]] / raw[e][m[e]]))
            Xs = raw * scale[:, None]
            ss = Xs.sum(axis=0, keepdims=True)
            Xs = np.divide(Xs, ss, out=np.zeros_like(Xs), where=ss > 0)
            rel_s = np.abs(Xs - X0)[m] / X0[m] if m.any() else np.array([])
            # 保存 T が第 1 画素のものかの同定: ezdeprof/Xobs, Dobs と Xstk[0], Dstk[0] の一致
            g = f["ezdeprof"]
            obs = {}
            if "Xobs" in g and "Dobs" in g:
                obs = {"Xobs_equals_Xstk0": bool(np.array_equal(
                           np.asarray(g["Xobs"][()], dtype=np.float32), np.asarray(X[0], dtype=np.float32))),
                       "Dobs_equals_Dstk0": bool(np.array_equal(
                           np.asarray(g["Dobs"][()], dtype=np.float32), np.asarray(D[0], dtype=np.float32)))}
            items.append(Item("closure.px0_stored_T", RECORDED, {
                "pixel": 0, "mask_X_gt": CLOSURE_MASK, "n_entries": int(m.sum()),
                "rel_residual_median": float(np.median(rel)) if rel.size else None,
                "rel_residual_p95": float(np.percentile(rel, 95)) if rel.size else None,
                "rel_residual_max": float(rel.max()) if rel.size else None,
                "abs_residual_max": float(np.abs(Xp - X0).max()),
                "rel_residual_by_angle_max": rel_full.max(axis=0).tolist(),
                "rel_residual_by_element_max": dict(zip(symbols, rel_full.max(axis=1).tolist())),
                "after_per_element_scale": {
                    "scale": dict(zip(symbols, scale.tolist())),
                    "rel_residual_median": float(np.median(rel_s)) if rel_s.size else None,
                    "rel_residual_max": float(rel_s.max()) if rel_s.size else None},
                "obs_pixel_identity": obs,
                "noise": report["meta"]["noise"],
            }, "保存 TransMatObs = T(Dopt)（解析側の単一プロファイル。Dopt = Dini = Dmod）。GT のどの画素の"
               "作用素でもないので、個々の画素の X は再現しない（λ の組成依存による）。記録のみ。"
               "合否は closure.per_pixel_T"))
        else:
            items.append(Item("closure.px0_stored_T", NA, {}, "shape 不整合か非有限のため未実施"))

        # ---- roles: producer の宣言があればそれと仮定を照合、無ければ UNRESOLVED ----------
        decl = ((manifest or {}).get("resolved") or {}).get("roleDeclarations") or {}
        gt_arr = (decl.get("groundTruth") or {}).get("array") if isinstance(decl, dict) else None
        ds_roles = {d.get("name"): d.get("role") for d in ((manifest or {}).get("resolved") or {}).get("datasets") or []
                    if isinstance(d, dict)}
        assumed = {"Dstk": "ground truth D(z)", "Xstk": "observed X(theta)", "Drlt": "existing reconstruction (if any)"}
        if gt_arr:
            st = PASS if gt_arr == "/Dstk" else WARN
            items.append(Item("roles.declared", st,
                              {"producer_groundTruth_array": gt_arr, "declaredBy": decl.get("declaredBy"),
                               "stillUndeclared": decl.get("stillUndeclared"),
                               "producer_roles": {k: ds_roles.get(k) for k in ("Dstk", "Xstk", "Drlt", "NSstk")},
                               "assumed_here": assumed},
                              "producer が manifest.resolved.roleDeclarations で GT を宣言し、この検査器の仮定と一致"
                              if st == PASS else "producer の GT 宣言がこの検査器の仮定 (/Dstk) と違う"))
        else:
            items.append(Item("roles.declared", UNRESOLVED, {"assumed": assumed},
                              "役割は名前からの仮定。producer manifest の宣言で確定する"))
        if "Drlt" in f and f["Drlt"].shape == D.shape:
            blk = np.asarray(f["Drlt"][ranges[0][0]:ranges[0][1]]) if ranges else np.zeros(0)
            items.append(Item("results.Drlt_present", RECORDED,
                              {"shape": list(f["Drlt"].shape),
                               "first_block_all_zero": bool(blk.size and not np.any(blk)),
                               "block": ranges[0] if ranges else None},
                              "既存再構成の有無。独立標本として数えない"))
        else:
            items.append(Item("results.Drlt_present", RECORDED, {"present": "Drlt" in f}))

        # ---- noise version --------------------------------------------------
        items.append(Item("noise.version", UNRESOLVED,
                          {"noise_value": report["meta"]["noise"]},
                          "noise モデル・版・実効値は manifest 待ち。同一ノイズ条件の比較は保留"))


def _manifest_items(files: Dict[str, Any], items: List[Item], report: Dict[str, Any],
                    main_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """同居 manifest.json の有無・同一性（generationId / schemaVersion / mainFile）を記録する。"""
    mf: Path = files["manifest"]
    if not mf.exists():
        items.append(Item("manifest.present", UNRESOLVED,
                          {"other_json": [p.name for p in files["other_json"]]},
                          "同居 manifest.json なし = legacy。由来 unknown（差分表 優先 1）"))
        return None
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        items.append(Item("manifest.present", FAIL, {"error": str(exc)}, "manifest.json が読めない"))
        return None
    if not isinstance(data, dict):
        items.append(Item("manifest.present", FAIL, {"type": type(data).__name__}, "最上位が object でない"))
        return None
    schema = data.get("schemaVersion", data.get("schema_version"))
    gen = data.get("generationId")
    mfile = data.get("mainFile")
    report["files"]["manifest_generationId"] = gen
    report["files"]["manifest_schemaVersion"] = schema
    items.append(Item("manifest.present", PASS,
                      {"keys": sorted(data)[:40], "schemaVersion": schema, "generationId": gen,
                       "mainFile": Path(mfile).name if isinstance(mfile, str) else None,
                       "confirmed": (data.get("confirmed") or {}).get("status") if isinstance(data.get("confirmed"), dict) else None}))
    if main_name is not None and isinstance(mfile, str) and mfile:
        same = Path(mfile).name == main_name
        items.append(Item("manifest.main_matches", PASS if same else FAIL,
                          {"manifest_mainFile": Path(mfile).name, "discovered_main": main_name},
                          "" if same else "manifest の mainFile とフォルダで特定した主 H5 が違う"))
    return data


def _verdict(items: Sequence[Item]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for it in items:
        counts[it.status] = counts.get(it.status, 0) + 1
    fails = [it.id for it in items if it.status == FAIL]
    errors = [it.id for it in items if it.status == ERROR]
    overall = ERROR if errors else (FAIL if fails else PASS)
    return {"overall": overall, "counts": counts, "fail_ids": fails, "error_ids": errors,
            "warn_ids": [it.id for it in items if it.status == WARN],
            "unresolved_ids": [it.id for it in items if it.status == UNRESOLVED]}


def _environment() -> Dict[str, Any]:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "h5py": h5py.__version__, "machine": platform.machine(),
            "system": platform.system()}


# --------------------------------------------------------------------------
# batch + ledger
# --------------------------------------------------------------------------
def scan_root(root: Path, families: Sequence[str] = FAMILIES, **kw) -> Dict[str, Any]:
    """``root/{family}/*`` の直下を走査。規約外フォルダは not_scanned に記録して飛ばす。"""
    root = Path(root).expanduser().resolve()
    reports: List[Dict[str, Any]] = []
    not_scanned: List[Dict[str, str]] = []
    for fam in families:
        fdir = root / fam
        if not fdir.is_dir():
            not_scanned.append({"folder": str(fdir), "reason": "family dir missing"})
            continue
        for child in sorted(p for p in fdir.iterdir() if p.is_dir()):
            if parse_dataset_folder(child) is None:
                not_scanned.append({"folder": f"{fam}/{child.name}",
                                    "reason": "name not {Nx}x{Ny}x{Nt}_{N}elem[_variant]"})
                continue
            reports.append(run_preflight(child, root=root, **kw))
    return {"schema_version": SCHEMA_VERSION, "tool": TOOL, "root": str(root),
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "reports": reports, "not_scanned": not_scanned}


def _closure_median(rep: Dict[str, Any]) -> Optional[float]:
    for it in rep["items"]:
        if it["id"] == "closure.px0_stored_T":
            return it["detail"].get("rel_residual_median")
    return None


def _item_status(rep: Dict[str, Any], iid: str) -> str:
    for it in rep["items"]:
        if it["id"] == iid:
            return it["status"]
    return "-"


def ledger_markdown(reports: Sequence[Dict[str, Any]],
                    not_scanned: Sequence[Dict[str, str]] = ()) -> str:
    rows = ["| folder | kind | gen | grid | ncomp | variant | noise | manifest | dxdydt | ΣD | verdict | FAIL ids | "
            "WARN ids | #unres | closure gate (max_abs) | px checked | s |",
            "|---|---|---|---|---|---|---:|---|---|---|---|---|---|---:|---|---:|---:|"]
    for r in reports:
        sp = r.get("spec") or {}
        grid = f"{sp.get('nx')}x{sp.get('ny')}x{sp.get('nt')}" if sp else "?"
        v = r["verdict"]
        cg = next((i for i in r["items"] if i["id"] == "closure.per_pixel_T"), None)
        cm = f"{cg['status']} ({cg['detail'].get('max_abs', float('nan')):.1e})" if cg and cg["detail"] else (cg["status"] if cg else "")
        sc = r.get("scope") or {}
        noise = (r.get("meta") or {}).get("noise", "")
        rows.append("| {f} | {kind} | {gen} | {g} | {n} | {var} | {nz} | {mf} | {dx} | {sd} | **{ov}** | {fi} | {w} | {u} | {cm} | {px} | {s} |".format(
            f=r.get("folder_rel") or Path(r["folder"]).name, kind=r.get("dataset_kind", ""),
            gen=(r.get("files", {}).get("manifest_generationId") or "")[:15], g=grid, n=sp.get("ncomp", "?"),
            var=sp.get("variant") or "", nz=noise, mf=_item_status(r, "manifest.present"),
            dx=_item_status(r, "structure.dxdydt"), sd=_item_status(r, "sample.dstk_composition_sum"),
            ov=v["overall"], fi=", ".join(v["fail_ids"] + v["error_ids"]) or "",
            w=", ".join(i.split(".")[-1] for i in v["warn_ids"]) or "",
            u=len(v["unresolved_ids"]), cm=cm,
            px=sc.get("n_px_checked", ""), s=r.get("elapsed_s", "")))
    out = ["# dataset preflight ledger", "",
           f"tool {TOOL} schema {SCHEMA_VERSION}. Bounded read-only checks; "
           "PASS does not certify every pixel. closure gate = producer-declared forward, PASS/FAIL at noise 0 only "
           "(tol 1e-6 abs, float32 storage), RECORDED at noise > 0.", "",
           *rows]
    if not_scanned:
        out += ["", "## not scanned (name outside the convention)", ""]
        out += [f"- `{d['folder']}` — {d['reason']}" for d in not_scanned]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog=f"python -m {TOOL}", description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="*", help="dataset folders to check")
    ap.add_argument("--all", action="store_true", help="scan {root}/{Standard,Checkered}/* ")
    ap.add_argument("--root", default=None,
                    help="simulation root (default: $DEPPRO_SIMULATION_ROOT)")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--frames", type=int, default=5, help="frames sampled per dataset")
    ap.add_argument("--max-px", type=int, default=20000, help="pixels read per sampled frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="default: deppro.paths outputs/preflight")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root).expanduser() if a.root else simulation_root()
    kw = dict(n_frames=a.frames, max_px=a.max_px, seed=a.seed)
    if a.all:
        if not root.is_dir():
            print(f"[preflight] root not found: {root} (set DEPPRO_SIMULATION_ROOT or --root)",
                  file=sys.stderr)
            return 2
        batch = scan_root(root, [s for s in a.families.split(",") if s], **kw)
    elif a.folders:
        batch = {"schema_version": SCHEMA_VERSION, "tool": TOOL, "root": str(root),
                 "generated": _dt.datetime.now().isoformat(timespec="seconds"),
                 "reports": [run_preflight(Path(p), root=root, **kw) for p in a.folders],
                 "not_scanned": []}
    else:
        ap.error("give dataset folders or --all")

    out_dir = Path(a.out_dir).expanduser() if a.out_dir else ensure_outputs_dir("preflight")
    out_dir.mkdir(parents=True, exist_ok=True)
    for rep in batch["reports"]:
        name = (rep.get("folder_rel") or Path(rep["folder"]).name).replace("/", "__")
        (out_dir / f"preflight_{name}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "preflight_ledger.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ledger_markdown(batch["reports"], batch["not_scanned"])
    (out_dir / "preflight_ledger.md").write_text(md, encoding="utf-8")
    if not a.quiet:
        print(md)
        print(f"[preflight] wrote {out_dir}")

    overall = [r["verdict"]["overall"] for r in batch["reports"]]
    if ERROR in overall:
        return 2
    return 1 if FAIL in overall else 0


if __name__ == "__main__":
    sys.exit(main())
