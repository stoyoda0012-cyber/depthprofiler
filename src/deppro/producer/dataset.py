"""cfg -> HDF5 + manifest。Python だけで完結する生成の入口。

**場面の定義（生成規則）と実行方式を別項目として記録する**（設計側の指示 2026-09-08）:

* `manifest.resolved.generationRule` — SV の初期値・更新式・更新/リセットの範囲・演算精度・版
* `manifest.resolved.execution`      — 逐次/並列、worker 数

同じ生成規則なら実行方式を変えても結果が変わらない設計にする。今は逐次互換のみ実装。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from deppro.producer import binding as be_mod
from deppro.producer.forward import ForwardSpec, observations
from deppro.producer.materials import lambda_basic, read_compound_table
from deppro.producer.profile import (
    COMPAT_IDENTITY, CHARGE_ITERATIONS, ProfileSpec,
    depth_profiles, depth_profiles_sequential, support_vector_from_profile,
)
from deppro.producer.gui_export import (
    GuiSettings, INITIAL_GUESS_ITERATIONS, INITIAL_GUESS_RULE, NINETY_DEGREE_POLICY,
    OBJECTIVE_MIN_VAL, PRE_ANALYSIS_STATE, SPARSITY_THRESHOLD, TRANSFER_MATRIX_RULE,
    build_pre_analysis_state, write_gui_datasets,
)
from deppro.producer.scene import SceneFlags, feature_vector_for_stacks, scene_timing

SCHEMA_VERSION = "1.1.0"
PRODUCER = "python"
PRODUCER_VERSION = "0.1.0"

#: SV の更新規則。実行方式ではなく **生成規則** の一部なので identity に入れる。
SV_RULE_SEQUENTIAL = "sequential-feedback/1"


class GenerationError(RuntimeError):
    pass


@dataclass
class ProducerConfig:
    """生成設定。参照 H5 を読まずにこれだけで生成できることを要件とする。"""

    grid_x: int
    grid_y: int
    time_steps: int
    time_size: int
    stack: Optional[int] = None          # 省略時 grid_x*grid_y*time_steps。素数も可
    depth_bins: int = 20
    total_depth: float = 600.0
    noise: float = 0.0

    element_symbols: Sequence[str] = ()
    orbitals: Sequence[str] = ()
    compounds: Sequence[str] = ()
    charges: Sequence[float] = ()
    properties: Sequence[str] = ()        # "Subst." | "Free" | "Const" | "Linked"
    linked_const: Sequence[float] = ()

    feature_vector_source: Sequence[float] = ()
    initial_support_vector: Sequence[float] = ()
    initial_support_vector_provenance: str = "unspecified"
    #: 初期 SV の出どころ。**この 2 つは別物なので混ぜない**。
    #:   "cfg-explicit"      — cfg に書いた上書き値をそのまま生成の種にする
    #:   "derived-from-Dini" — その生成の Dini から p/w/h を導いて種にする（MATLAB の既定）
    #: 切り替えると生成結果そのものが変わる。過去の GT/X 一致結果は流用できない。
    initial_support_vector_mode: str = "cfg-explicit"

    #: "light" = 解析側 reader 向け（既定、従来どおり）／"gui" = MATLAB GUI が開ける形
    export_profile: str = "light"
    gui: Optional[GuiSettings] = None

    binding_energy: Optional[Sequence[Optional[float]]] = None
    photon_energy: float = 8000.0
    work_function: float = 5.0
    gamma_eff: float = 0.8
    tpp_version: int = 3

    theta_obs: Sequence[float] = ()
    top_layer: str = ""
    bottom_layer: str = ""
    interf_redox: str = ""
    etch_grow: str = ""
    distribution: str = "Gaussian"

    chunk: int = 4096
    compound_table: Optional[str] = None
    compound_be_table: Optional[str] = None
    element_be_table: Optional[str] = None

    def n_stack(self) -> int:
        return int(self.stack) if self.stack else self.grid_x * self.grid_y * self.time_steps

    def role_indices(self) -> Dict[str, List[int]]:
        out = {"subst": [], "free": [], "const": [], "linked": []}
        for i, p in enumerate(self.properties):
            key = {"Subst.": "subst", "Free": "free", "Const": "const", "Linked": "linked"}.get(p)
            if key is None:
                raise GenerationError(f"unknown property {p!r} at element {i}")
            out[key].append(i)
        return out


def anti_index(ne: int, subst: Sequence[int]) -> List[int]:
    """MATLAB `antisubst`: subst 以外の全元素（0-based）。"""
    return [i for i in range(int(ne)) if i not in set(subst)]


def _sha256(path: Path) -> str:
    """書き終えたファイルの内容ハッシュ。

    name+bytes だけでは「どのファイルを検証したか」が言えない、という解析検証側の指摘。
    とくに MATLAB の AutoSave は **同じサイズのまま**中身を書き換える経路があるので、
    サイズと mtime による指紋では検出できない。
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _generation_id() -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
    return f"{stamp}-{os.getpid() & 0xFFFFFF:06x}"


def _layout(cfg: ProducerConfig, ns: int) -> Dict[str, Any]:
    """3 状態。矩形が成り立たないものを黙って矩形にしない（PR #9 と同じ契約）。"""
    ppf = cfg.grid_x * cfg.grid_y
    if ppf <= 0:
        return {"type": "flat_samples", "note": "no grid declared", "complete": False}
    nt_full, rem = divmod(ns, ppf)
    if rem == 0:
        return {"type": "rectangular_grid", "nx": cfg.grid_x, "ny": cfg.grid_y, "nt": nt_full,
                "complete": True,
                "completeNote": "STRUCTURAL ONLY: a sample exists at every declared lattice point. "
                                "It says nothing about physical validity."}
    return {"type": "rectangular_grid_partial", "nx": cfg.grid_x, "ny": cfg.grid_y,
            "nt": nt_full + 1, "complete": False, "lastFrameCount": rem,
            "completeNote": f"the last frame holds {rem} of {ppf} lattice points"}


def _stack_xyt(nx: int, ny: int, ns: int) -> np.ndarray:
    s = np.arange(1, ns + 1, dtype=np.int64)
    return np.stack([(s - 1) % nx + 1, ((s - 1) % (nx * ny)) // nx + 1, (s - 1) // (nx * ny) + 1]).astype("f4")


IF_EXISTS_VALUES = ("error", "clean")


def validate_config(cfg: ProducerConfig) -> SceneFlags:
    """cfg だけで分かる不備を**副作用の前に**全部出す。戻り値は検証済みの場面フラグ。

    2026-09-08 の独立点検の指摘: 以前は未移植分岐の拒否が材料表の読み込みと lambda 計算の
    後、`noise>0` の拒否は GT と forward を作り終えた後だった。ファイルを消す前・重い計算の
    前に落ちるよう、ここへまとめた。
    """
    if not cfg.element_symbols:
        raise GenerationError("element_symbols が空")
    ne = len(cfg.element_symbols)
    for name, seq in (("orbitals", cfg.orbitals), ("compounds", cfg.compounds),
                      ("charges", cfg.charges), ("properties", cfg.properties)):
        if len(seq) != ne:
            raise GenerationError(f"{name} の長さ {len(seq)} が element_symbols の {ne} と違う")
    roles = cfg.role_indices()                      # 未知の property はここで落ちる
    if not roles["subst"]:
        raise GenerationError("subst が 1 つも無い")
    if cfg.depth_bins <= 0 or cfg.total_depth <= 0:
        raise GenerationError(f"depth_bins={cfg.depth_bins} total_depth={cfg.total_depth} は正の値")
    if not len(cfg.theta_obs):
        raise GenerationError("theta_obs が空")
    if cfg.n_stack() <= 0:
        raise GenerationError(f"stack 数が {cfg.n_stack()}")
    if cfg.initial_support_vector_mode not in ("cfg-explicit", "derived-from-Dini"):
        raise GenerationError(
            f"initial_support_vector_mode={cfg.initial_support_vector_mode!r} は未定義")
    if cfg.initial_support_vector_mode == "derived-from-Dini":
        raise NotImplementedError(
            "derived-from-Dini は未実装。切り替えると生成の種が変わるので、"
            "小標本で GT/X を取り直してから入れる（過去の一致結果は流用しない）。")
    if cfg.export_profile not in ("light", "gui"):
        raise GenerationError(f"export_profile={cfg.export_profile!r} は 'light' か 'gui'")
    if cfg.noise:
        raise NotImplementedError(
            f"noise={cfg.noise} は未実装。Python 側の分布・スケール・適用段階・seed を "
            "決めてから入れる。noise=0 で先に確かめること。")
    flags = SceneFlags(bottom_layer=cfg.bottom_layer or None, top_layer=cfg.top_layer or None,
                       interf_redox=cfg.interf_redox or None, etch_grow=cfg.etch_grow or None)
    flags.validate()                                # 未移植の時間分岐
    if cfg.distribution != "Gaussian":
        raise NotImplementedError(
            f"distribution={cfg.distribution!r} は未移植（実装は 'Gaussian' のみ）")
    return flags


def _prepare_out_dir(out: Path, if_exists: str) -> None:
    """出力先の保護。**cfg の検査を通してから呼ぶこと**（消してから落ちない）。

    以前は `*.h5` があるときしか効かず、manifest.json だけが残った出力先では
    `if_exists='error'` でも素通りして manifest を上書きしていた（独立点検 P1）。
    いまは**中身が何かあれば**拒否する。
    """
    if if_exists not in IF_EXISTS_VALUES:
        raise GenerationError(
            f"if_exists={if_exists!r} は未定義（{IF_EXISTS_VALUES} のいずれか）")
    if out.exists():
        existing = sorted(p.name for p in out.iterdir() if not p.name.startswith("."))
        if existing:
            if if_exists == "error":
                raise GenerationError(
                    f"{out} は空ではない (if_exists='error'): {existing[:8]}"
                    + (" ..." if len(existing) > 8 else ""))
            for f in list(out.glob("*.h5")) + list(out.glob("*.json")):
                f.unlink()
    out.mkdir(parents=True, exist_ok=True)


def generate(cfg: ProducerConfig, out_dir: Path | str, *, if_exists: str = "error") -> Dict[str, Any]:
    """生成して HDF5 と manifest を書く。戻り値は manifest の内容。"""
    flags = validate_config(cfg)          # 副作用の前に全部検査する
    t0 = time.time()
    out = Path(out_dir).expanduser()
    _prepare_out_dir(out, if_exists)

    ne = len(cfg.element_symbols)
    ns = cfg.n_stack()
    roles = cfg.role_indices()
    depth = np.arange(cfg.depth_bins, dtype=float) * (cfg.total_depth / cfg.depth_bins)
    step = float(cfg.total_depth / cfg.depth_bins)
    theta = np.asarray(cfg.theta_obs, dtype=float)

    mats = read_compound_table(cfg.compounds, cfg.compound_table)
    be = be_mod.resolve(cfg.element_symbols, cfg.orbitals, cfg.compounds,
                        explicit=cfg.binding_energy,
                        compound_table=cfg.compound_be_table, element_table=cfg.element_be_table)
    L = lambda_basic(mats, be.values, cfg.photon_energy, cfg.work_function, cfg.tpp_version)

    fv = feature_vector_for_stacks(np.asarray(cfg.feature_vector_source, float),
                                   np.arange(1, ns + 1), cfg.grid_x, cfg.grid_y, cfg.time_size, flags)
    spec = ProfileSpec(depth=depth, charge=np.asarray(cfg.charges, float),
                       subst_index=roles["subst"], free_index=roles["free"],
                       const_index=roles["const"], linked_index=roles["linked"],
                       support_vector=np.asarray(cfg.initial_support_vector, float),
                       distribution=cfg.distribution)
    # 逐次再帰。SV は前 sample の D から作って次へ送る。近似も chunk ごとの reset もしない。
    D = depth_profiles_sequential(fv, spec)
    fspec = ForwardSpec(depth=depth, angles_deg=theta, step_size=step,
                        gamma_eff=cfg.gamma_eff, lambda_basic=L)
    intensity, X = observations(D, fspec)

    # 保存 T の由来を宣言する。GT のどの画素の forward でもない。
    from deppro.producer.forward import transfer_and_intensity
    t_ref, _, _ = transfer_and_intensity(D[0], fspec)
    trans_mat_provenance = (
        "T(D of sample 1) — a single matrix kept for readers that require the dataset to "
        "carry one. It is NOT the forward of the whole ground truth; rebuild T per sample "
        "from its own D.")

    gui = cfg.gui or GuiSettings()
    state = None
    if cfg.export_profile == "gui":
        # **生成の中で 1 度だけ組む**。export のときに作り直さない。
        state = build_pre_analysis_state(
            depth=depth, charge=np.asarray(cfg.charges, float),
            subst_index=roles["subst"], anti_index=anti_index(ne, roles["subst"]),
            free_index=roles["free"], const_index=roles["const"], linked_index=roles["linked"],
            thck_ini=gui.thck_ini, abrp_ini=gui.abrp_ini, subs_ini=gui.subs_ini,
            thck_mod=gui.thck_mod, abrp_mod=gui.abrp_mod, subs_mod=gui.subs_mod,
            lambda_basic=L, gamma_eff=cfg.gamma_eff, step_size=step,
            theta_obs=theta, theta_opt=gui.theta_opt, profiles=D, x_obs_all=X,
            feature_vectors=fv,
            support_vector_set=np.asarray(cfg.initial_support_vector, float),
            optimization_method_value=gui.optimization_method_value)
        # MATLAB は解析前に TransMatObs = T(Dopt) = T(Dini) を持つ。GT の forward ではない。
        t_ref = state.trans_mat_obs
        trans_mat_provenance = (
            "T(Dini) — the pre-analysis transfer matrix MATLAB stores (handles.Dopt == Dini "
            "before any Calculation). NOT the forward of any ground-truth pixel. The light "
            "profile stores T(D of sample 1) instead; this is a declared difference.")

    gid = _generation_id()
    main = out / f"py_{gid}.h5"
    _write_h5(main, cfg, D, X, L, depth, theta, step, be.values, roles, t_ref, ns,
              state=state, gui=gui, materials=mats, feature_vectors=fv)

    # 閉じてから、別に開き直して構造を確かめる
    checks = _reread(main, ns, ne, cfg.depth_bins, theta.size, cfg.grid_x, cfg.grid_y,
                     lambda_basic=L, step_size=step, gamma_eff=cfg.gamma_eff, angles=theta)

    files = [{"name": main.name, "bytes": main.stat().st_size, "sha256": _sha256(main)}]
    manifest: Dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "ok",
        "generationId": gid,
        "producer": {"name": PRODUCER, "version": PRODUCER_VERSION,
                     "implementation": "deppro.producer", "matlabUsed": False},
        "mainFile": str(main.resolve()),
        "outputFolder": str(out.resolve()),
        "startedAt": datetime.fromtimestamp(t0, timezone.utc).astimezone().isoformat(timespec="seconds"),
        "elapsedSeconds": round(time.time() - t0, 3),
        "layout": _layout(cfg, ns),
        "sampleId": {"rule": "<generationId>#<stackIndex>, stackIndex 1-based, as originally written",
                     "generationId": gid},
        "resolved": {
            "axes": {"nx": cfg.grid_x, "ny": cfg.grid_y,
                     "nt": ns // (cfg.grid_x * cfg.grid_y) if cfg.grid_x * cfg.grid_y else None,
                     "timeSize": cfg.time_size, "stack": ns,
                     "featureVectorWidth": int(np.asarray(cfg.feature_vector_source).size)},
            "files": files,
            "requestedConfig": asdict(cfg),
            "simulation": {"topLayer": cfg.top_layer, "bottomLayer": cfg.bottom_layer,
                           "interfRedox": cfg.interf_redox, "etchGrow": cfg.etch_grow,
                           "distribution": cfg.distribution},
            "elements": {"symbols": list(cfg.element_symbols), "orbitals": list(cfg.orbitals),
                         "compounds": list(cfg.compounds), "charges": list(map(float, cfg.charges)),
                         "properties": list(cfg.properties),
                         "substIndex": [i + 1 for i in roles["subst"]],
                         "freeIndex": [i + 1 for i in roles["free"]],
                         "constIndex": [i + 1 for i in roles["const"]],
                         "linkedIndex": [i + 1 for i in roles["linked"]]},
            "depth": {"bins": cfg.depth_bins, "totalDepth": cfg.total_depth, "stepSize": step,
                      "axis": depth.tolist()},
            "angles": {"thetaObs": theta.tolist()},
            "forward": {
                "photon": cfg.photon_energy, "workFunction": cfg.work_function,
                "gammaEffective": cfg.gamma_eff, "tppVersion": cfg.tpp_version,
                "bindingEnergy": be.values.tolist(),
                "bindingEnergyProvenance": be.provenance,
                "bindingEnergyDetail": be.detail,
                "valenceCharge": mats.valence_charge.tolist(),
                "density": mats.density.tolist(),
                "molecularWeight": mats.molecular_weight.tolist(),
                "bandGap": mats.band_gap.tolist(),
                "lambdaBasic": L.ravel(order="F").tolist(),
                "lambdaBasicOrientation": "L[a, i] = IMFP of element i's photoelectrons in material a",
                "referenceTables": {
                    "CompoundTable": {"path": mats.source_path, "bytes": mats.source_bytes,
                                      "sha256": mats.source_sha256},
                    **be.sources,
                },
            },
            # ---- 生成規則（場面の定義）。実行方式とは別項目 ----
            "generationRule": {
                "identity": COMPAT_IDENTITY,
                "supportVector": {
                    "rule": SV_RULE_SEQUENTIAL,
                    "initial": list(map(float, cfg.initial_support_vector)),
                    # cfg 明示値か Dini 由来かを機械可読で残す。文章の provenance とは別項目。
                    "initialMode": cfg.initial_support_vector_mode,
                    "initialModeMeaning": {
                        "cfg-explicit": "the generation seed is the cfg value, written verbatim; "
                                        "MATLAB instead derives SupportVector_Set from its own Dini",
                        "derived-from-Dini": "the generation seed is p/w/h computed from this run's Dini",
                    }[cfg.initial_support_vector_mode],
                    "initialProvenance": cfg.initial_support_vector_provenance,
                    "update": "SV_s = [p,w,h] over subst then linked, computed from D_s only "
                              "(p = centroid, h = 100*max, w = 2*sqrt(log2/pi)*trapz(100*D)/h)",
                    "updateScope": "every sample, in stack order; never reset within a run",
                    "usesForward": False,
                    "precision": "float64 throughout; stored as float32",
                },
                "chargeLoop": {"iterations": CHARGE_ITERATIONS,
                               "note": "fixed count, not a convergence test. The fixed point is sum=1 "
                                       "but 30 iterations leave up to +4.7% at the interface. NOT corrected."},
                "transMatObsProvenance": trans_mat_provenance,
            },
            "roleDeclarations": _role_declarations(cfg),
            "datasets": _dataset_roles(cfg),
            "exportProfile": cfg.export_profile,
            # 各変調の反転（規則）と合成後の山谷（実測）は別項目として残す。
            "sceneTiming": scene_timing(fv, cfg.grid_x, cfg.grid_y, ns, cfg.time_size),
            "preAnalysisState": _pre_analysis_record(cfg, gui, state),
            # ---- 実行方式。生成規則を変えないこと ----
            "execution": {
                "mode": "sequential",
                "workers": 1,
                "chunk": None,
                "chunkNote": "NOT IMPLEMENTED. cfg.chunk used to be threaded into "
                             "depth_profiles_sequential() and observations(), but in both places the "
                             "parameter was declared and never used -- one had an outer loop that "
                             "still stepped one pixel at a time, the other ignored it entirely. "
                             "Removed 2026-09-09 rather than left as a field that describes nothing. "
                             "Pixels are independent in the forward step and the SV recursion is "
                             "strictly sequential, so no chunking can change the result anyway.",
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        },
        "verify": checks,
        "unknown": {
            "noiseModel": "noise=0 only; no Python noise model is declared yet",
            "matlabParallelEquivalence": "MATLAB's parallel path broadcasts the initial SV instead of "
                                         "feeding it forward. Its semantics are UNDETERMINED; this "
                                         "dataset reproduces the SEQUENTIAL rule only.",
            "physicalCorrectness": "sequential compatibility is not a claim of physical correctness",
        },
    }
    manifest["confirmed"] = {
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "by": "deppro.producer._reread",
        "process": "same process, after closing the file",
        "generationId": gid,
        "status": "ok",
        "scope": "structural: shapes, coordinate formula, finiteness on sampled rows",
        "mainFile": main.name,
        "mainFileSha256": files[0]["sha256"],
        "shaScope": "the bytes as written by this run. A consumer that reads a different hash is "
                    "not looking at what was confirmed here -- MATLAB's AutoSave rewrites the file "
                    "in place, sometimes at the same byte size.",
        **checks,
    }
    _atomic_json(out / "manifest.json", manifest)
    return manifest


def _role_declarations(cfg: ProducerConfig) -> Dict[str, Any]:
    """この producer が**自分で言えることだけ**を宣言する。

    MATLAB 側 (`resolved.roleDeclarations`) はプロジェクト所有者による広範な宣言で、
    生の測定形式や Toyomacro の事情まで含む。**それを写さない**。ここに書くのは
    この producer が実際に書いたデータセットの役割だけ。

    解析側 `dataset_preflight` の `roles.declared` は `groundTruth.array` を見る。
    """
    return {
        "declaredBy": f"deppro.producer {PRODUCER_VERSION} (generation time)",
        "note": "Declared by the Python producer for the datasets it wrote. This is NOT a copy of "
                "the MATLAB producer's declaration; anything absent here is undeclared by this "
                "producer, and a role must not be inferred from a dataset name.",
        "groundTruth": {
            "array": "/Dstk",
            "chain": "Dstk is the ground truth. Xstk is its forward observation. Everything with "
                     "an rlt/trn suffix is a result slot, initialised to the initial guess and "
                     "NOT ground truth.",
        },
        "transferMatrix": {
            "stored": "/ezdeprof/TransMatObs",
            "meaning": ("gui" if cfg.export_profile == "gui" else "light") and (
                "T(Dini) -- the transfer matrix of the INITIAL GUESS, which is what MATLAB holds "
                "before any Calculation" if cfg.export_profile == "gui" else
                "T(D of sample 1) -- one matrix kept for readers that require the dataset to carry "
                "one"),
            "notTheInverseOperator":
                "This single matrix is NOT the operator of the inverse problem and NOT the forward "
                "operator of any other pixel. The generator rebuilds lambda and T from each pixel's "
                "own D. Reusing the stored T across pixels leaves an angle-dependent residual "
                "in X. "
                "A solver that shares one T across pixels carries that as model error, not as "
                "reconstruction error.",
            "rebuildFrom": "gammaeff, lambdabasic, stepsize, depth, thetaobs and the pixel's own D",
            "rule": TRANSFER_MATRIX_RULE,
            "ninetyDegreePolicy": NINETY_DEGREE_POLICY,
        },
        "groundTruthQuirks": {
            "compositionSumNotOne":
                "sum_e D can reach 1.047 at the film/substrate interface (the 30-iteration charge "
                "loop of the compatibility mode). NOT corrected. A solver constrained to sum-to-one "
                "cannot reach these layers; that is a floor on reconstruction accuracy, not a "
                "solver failure.",
            "chargeNeutrality": "sum_e D*q holds to ~4e-05.",
        },
        "stillUndeclared": [
            "noise model (noise>0 is not implemented)",
            "the 6 scene branches this producer has not ported",
        ],
    }


def _dataset_roles(cfg: ProducerConfig) -> List[Dict[str, str]]:
    """`resolved.datasets` — 名前と役割の対応。preflight が名前からの推測を止めるために読む。"""
    rows = [
        {"name": "Dstk", "role": "ground truth D(z) per sample"},
        {"name": "Xstk", "role": "observed X(theta) per sample, the forward of Dstk"},
        {"name": "NSstk", "role": "noise level per sample (all zero while noise>0 is unimplemented)"},
        {"name": "stackxyt", "role": "the (x, y, t) each 1-based stack index stands for"},
    ]
    if cfg.export_profile == "gui":
        rows += [
            {"name": "Drlt", "role": "result slot; initialised to Dini replicated, NOT ground truth"},
            {"name": "Dtrn", "role": "training slot; initialised to Dini replicated, NOT ground truth"},
            {"name": "Xrlt", "role": "result slot; initialised to Xini replicated"},
            {"name": "FV0stk", "role": "FeatureVector_Source per sample (generation input)"},
            {"name": "FVstk", "role": "FeatureVector_Set per sample, measured back from Dstk"},
            {"name": "SVstk", "role": "SupportVector per sample, measured back from Dstk"},
            {"name": "Cstk", "role": "concentration per sample, from Dstk"},
            {"name": "SPSstk", "role": "sparsity count per sample, from Dstk"},
            {"name": "Qstk", "role": "objective at the initial guess per sample; Qrlt and Qtrn are "
                                     "the same array before any Calculation"},
        ]
    return rows


def _pre_analysis_record(cfg, gui, state) -> Optional[Dict[str, Any]]:
    """解析前の状態の由来。**export で作り直していないこと**を言えるようにする。"""
    if state is None:
        return None
    cfg_sv = np.asarray(cfg.initial_support_vector, dtype=float).ravel()
    der = np.asarray(state.support_vector_set_derived, dtype=float).ravel()
    return {
        "identity": state.rule,
        "initialGuessRule": state.initial_guess_rule,
        "initialGuessRuleDetail": {
            "source": "+deppro/+core/InitializeInitialGuess_Parallel.m",
            "chargeIterations": INITIAL_GUESS_ITERATIONS,
            "boost": "1 + 1/k",
            "substShare": "1/len(subst)",
            "note": "生成側 CalculateDistribution（30 回・1+100*exp(-k/2)）とは別の規則。",
        },
        "sparsityThreshold": SPARSITY_THRESHOLD,
        "objectiveMinVal": OBJECTIVE_MIN_VAL,
        "computedOnce": "built inside generate(); the writer only serialises it. "
                        "No initial-state quantity is recomputed at export time.",
        "supportVectorSetWritten": {
            "values": cfg_sv.tolist(),
            "mode": cfg.initial_support_vector_mode,
            "note": "written value == the generation seed. MATLAB writes the Dini-derived value "
                    "here instead; the two are recorded separately and NOT reconciled.",
        },
        "supportVectorSetDerivedFromDini": {
            "values": der.tolist(),
            "maxAbsDifferenceFromWritten": float(np.abs(der - cfg_sv).max())
            if der.size == cfg_sv.size else None,
            "note": "recorded for comparison only; it is not written to the file and was not "
                    "used to generate anything.",
        },
        "featureVectorSet": {
            "values": np.asarray(state.feature_vector_set, float).tolist(),
            "source": "derived from this run's Dini (MATLAB DepthProfiler.m:228)",
        },
        "resultFieldsInitialisation": {
            "Dini/Dmod/Dopt": "the same initial-guess profile",
            "Xini/Xmod/Xopt": "its angle profile on thetaopt (theta=90 deg gives NaN, as in MATLAB)",
            "Drlt/Dtrn": "Dini replicated over every sample",
            "Xrlt": "Xini replicated over every sample",
            "Crlt/Ctrn/FVrlt/FVtrn/SVrlt/SVtrn/SPSrlt/SPStrn": "the corresponding _Set value replicated",
            "Qstk/Qrlt/Qtrn": "per-sample objective at the initial guess (all three identical)",
            "Qopt": "NaN — the only NaN-by-design field",
            "degree/alpha": "1",
            "groundTruthReuse": "none: no GT array is copied into a result field. Dobs/Xobs are the "
                                "observation panel for the current sample, as in MATLAB.",
        },
        "iterationtime": {
            "value": state.elapsed_seconds,
            "meaning": "wall-clock seconds spent building this initial state. It is NOT an "
                       "analysis time; nothing has been optimised.",
            "reproducibility": "THE ONLY dataset in the file that does not reproduce byte-for-byte "
                               "across runs of the same cfg (every other dataset is "
                               "byte-identical between runs). MATLAB's own "
                               "iterationtime is likewise a Set-time measurement. See "
                               "deppro.producer.gui_check.NON_DETERMINISTIC.",
        },
        "notVerified": "GUI compatibility is not evidence about reconstruction accuracy. "
                       "Nothing here has been read back by MATLAB yet.",
    }


def _write_h5(main, cfg, D, X, L, depth, theta, step, be_values, roles, t_ref, ns,
              *, state=None, gui=None, materials=None, feature_vectors=None) -> None:
    import h5py
    ne = len(cfg.element_symbols)
    with h5py.File(main, "w") as f:
        f.create_dataset("Dstk", data=np.transpose(D, (0, 2, 1)).astype("f4"))     # (P, E, Z)
        f.create_dataset("Xstk", data=np.transpose(X, (0, 2, 1)).astype("f4"))     # (P, E, A)
        f.create_dataset("NSstk", data=np.full((1, ns), float(cfg.noise), dtype="f4"))
        f.create_dataset("stackxyt", data=_stack_xyt(cfg.grid_x, cfg.grid_y, ns))  # (3, P)
        g = f.create_group("ezdeprof")
        s = h5py.string_dtype("utf-8")
        # MATLAB の `h5read(...,[1 1],[1 nelm])` は rank 2 を要求するので、GUI profile では
        # (ne,1) で書く（軽量 profile は従来どおり 1 次元。値も順序も同じ）。
        def _strs(values):
            a = np.array(list(values), dtype=object)
            return a.reshape(-1, 1) if state is not None else a
        g.create_dataset("elementsymbol", data=_strs(cfg.element_symbols), dtype=s)
        g.create_dataset("orbital", data=_strs(cfg.orbitals), dtype=s)
        g.create_dataset("compound", data=_strs(cfg.compounds), dtype=s)
        g.create_dataset("prpties", data=_strs(cfg.properties), dtype=s)
        g.create_dataset("charge", data=np.asarray(cfg.charges, dtype="f4").reshape(ne, 1))
        g.create_dataset("depth", data=depth.astype("f4").reshape(1, -1))
        g.create_dataset("thetaobs", data=theta.astype("f4").reshape(1, -1))
        g.create_dataset("stepsize", data=np.array([step], dtype="f4"))
        g.create_dataset("totaldepth", data=np.array([cfg.total_depth], dtype="f4"))
        g.create_dataset("noise", data=np.array([cfg.noise], dtype="f4"))
        g.create_dataset("stack", data=np.array([ns], dtype="u8"))
        g.create_dataset("dxdydt", data=np.array([cfg.grid_x, cfg.grid_y, cfg.time_size], dtype="u2").reshape(3, 1))
        g.create_dataset("m", data=np.array([cfg.depth_bins], dtype="u2"))
        for key, idx in (("subst", roles["subst"]), ("free", roles["free"]),
                         ("const", roles["const"]), ("linked", roles["linked"])):
            g.create_dataset(key, data=np.array([i + 1 for i in idx] or [0], dtype="u1").reshape(-1, 1))
        g.create_dataset("linkedconst", data=np.asarray(cfg.linked_const or [0], dtype="f4").reshape(-1, 1))
        g.create_dataset("bindingenergy", data=np.asarray(be_values, dtype="f4").reshape(ne, 1))
        g.create_dataset("photonenergy", data=np.full((ne, 1), cfg.photon_energy, dtype="f4"))
        g.create_dataset("workfunction", data=np.array([cfg.work_function], dtype="f4"))
        g.create_dataset("gammaeff", data=np.array([cfg.gamma_eff], dtype="f4"))
        g.create_dataset("TPPver", data=np.array([cfg.tpp_version], dtype="u1"))
        # MATLAB は `lambdabasic(a, i, k)`（a=物質, i=放出元素）で読む。h5py は次元が逆なので
        # **`[0][i][a]` の並びで書く** = `L.T`。`L` は `L[a, i]`。
        # 転置を忘れても生成そのもの（Dstk/Xstk）は内部で正しい向きを使うので気づきにくいが、
        # 保存値を読む側（MATLAB の Set/Calculation、解析側の `problem.py`）は転置した値を受け取る。
        g.create_dataset("lambdabasic", data=np.ascontiguousarray(L.T).astype("f4").reshape(1, ne, ne))
        g.create_dataset("TransMatObs", data=np.transpose(t_ref, (2, 1, 0)).astype("f4"))
        g.create_dataset("FeatureVector_Source",
                         data=np.asarray(cfg.feature_vector_source, dtype="f4").reshape(-1, 1))
        g.create_dataset("SupportVector_Set",
                         data=np.asarray(cfg.initial_support_vector, dtype="f4").reshape(-1, 1))
        g.create_dataset("SimulationOrExperiment", data=np.array([1], dtype="u1"))
        if state is not None:
            write_gui_datasets(
                f, state=state, gui=gui, ne=ne, nz=cfg.depth_bins, n_stack=ns,
                symbols=cfg.element_symbols, charges=cfg.charges, orbitals=cfg.orbitals,
                subst_index=roles["subst"], anti_index=anti_index(ne, roles["subst"]),
                free_index=roles["free"], const_index=roles["const"],
                linked_index=roles["linked"], materials=materials, theta_obs=theta,
                photon_energy=cfg.photon_energy, feature_vectors=feature_vectors)


def _reread(main, ns, ne, nz, nth, nx, ny, *, lambda_basic=None, step_size=None,
            gamma_eff=None, angles=None) -> Dict[str, Any]:
    """閉じたファイルを開き直して構造を確かめる。書いた側の記憶を信用しない。

    `lambda_basic` を渡すと、保存した `/ezdeprof/lambdabasic` の**軸の向き**と、
    保存 D・保存 λ から組み直した X が保存 X と合うかも見る。**軽量 profile にも効く**
    共通のゲート（2026-09-08 の独立点検 P2 の指摘。以前は GUI 用の事前検査にしか無く、
    転置バグが軽量 profile 3 件に残った）。"""
    import h5py
    problems: List[str] = []
    with h5py.File(main, "r") as f:
        for name, want in (("Dstk", (ns, ne, nz)), ("Xstk", (ns, ne, nth)), ("stackxyt", (3, ns))):
            got = f[name].shape
            if tuple(got) != want:
                problems.append(f"{name} has shape {tuple(got)}, expected {want}")
        if not problems:
            xyt = np.asarray(f["stackxyt"][()]).T.astype(np.int64)
            s = np.arange(1, ns + 1, dtype=np.int64)
            want = np.stack([(s - 1) % nx + 1, ((s - 1) % (nx * ny)) // nx + 1, (s - 1) // (nx * ny) + 1], axis=1)
            if not np.array_equal(xyt, want):
                problems.append("stackxyt disagrees with the declared formula")
            probe = np.unique(np.linspace(0, ns - 1, min(ns, 8)).astype(int))
            for p in probe:
                d = f["Dstk"][int(p)]
                if not np.all(np.isfinite(d)) or float(np.abs(d).max()) == 0.0:
                    problems.append(f"Dstk row {p} is non-finite or identically zero")
    extra: Dict[str, Any] = {}
    if lambda_basic is not None and not problems:
        extra = _lambda_and_forward_check(main, lambda_basic, step_size, gamma_eff, angles, ns)
        problems.extend(extra.pop("problems"))
    if problems:
        raise GenerationError("re-read of the closed file failed:\n  " + "\n  ".join(problems))
    return {"reread": "ok", "sampledRows": int(min(ns, 8)), "stack": int(ns), "elements": int(ne),
            **extra}


def _lambda_and_forward_check(main, L, step_size, gamma_eff, angles, ns) -> Dict[str, Any]:
    """保存 λ の向きと、保存 D・保存 λ からの再 forward を確かめる。

    MATLAB は `lambdabasic(a, i, k)` で読み、h5py は次元が逆なので正しい並びは `h5[0][i][a]`。
    書いた側の `L[a, i]` を転置せずに書くと、読む側が逆向きの値を受け取る。

    **向きの識別力は行列に依る。** `L` が対称、あるいは 1 成分だと、転置しても同じ値になるので
    この検査は向きを見分けられない。見分けられるかどうかを `orientationDiscriminating` で
    別に報告する（「逆なら必ず差が出る」とは書かない）。
    """
    import h5py
    from deppro.transfer_matrix import TransferMatrix
    L = np.asarray(L, dtype=float)
    problems: List[str] = []
    sym = float(np.abs(L - L.T).max())
    scale = float(np.abs(L).max())
    discriminating = sym > max(1e-6 * scale, 1e-9)
    with h5py.File(main, "r") as f:
        stored = np.asarray(f["ezdeprof"]["lambdabasic"][()], dtype=float)
        stored = stored[0] if stored.ndim == 3 else stored
        as_written = float(np.abs(stored - L.T).max())
        transposed = float(np.abs(stored - L).max())
        if as_written > max(1e-4 * scale, 1e-3):
            problems.append(
                f"/ezdeprof/lambdabasic: stored value is not L.T (max|d|={as_written:.6g}); "
                + (f"it matches the TRANSPOSED layout instead (max|d|={transposed:.6g}). "
                   "MATLAB reads lambdabasic(a,i,k), so h5[0][i][a] must hold L[a,i]."
                   if transposed < as_written else "it matches neither layout."))
        # 保存値だけから X を組み直す（生成器の forward 関数は使わない）
        probe = np.unique(np.linspace(0, ns - 1, min(ns, 8)).astype(int))
        tm = TransferMatrix(depth=np.asarray(f["ezdeprof"]["depth"][()], dtype=float).ravel(),
                            step_size=float(step_size),
                            angles=np.asarray(angles, dtype=float), gamma_eff=float(gamma_eff))
        lb_read = stored.T                      # h5[0][i][a] -> L[a, i]
        worst = 0.0
        for pidx in probe:
            d = np.asarray(f["Dstk"][int(pidx)], dtype=float).T      # (nz, ne)
            lam = tm.calculate_lambda_obs(d, lb_read.T)
            t = tm.calculate_trans_mat(lam)
            i1 = np.einsum("tze,ze->te", t, d)
            x = i1 / i1.sum(axis=1, keepdims=True)
            worst = max(worst, float(np.abs(x.T - np.asarray(f["Xstk"][int(pidx)], dtype=float)).max()))
        if worst > 1e-5:
            problems.append(f"re-forward from the STORED D and lambda misses the stored X by {worst:.6g}")
    return {"problems": problems,
            "lambdabasicOrientation": {"maxAbsDiffFromLT": as_written,
                                       "maxAbsDiffFromTransposed": transposed,
                                       "orientationDiscriminating": bool(discriminating),
                                       "asymmetry": sym,
                                       "note": "L が対称、または 1 成分だと向きは識別できない。"
                                               "そのときこの検査は数値整合だけを示す。"},
            "reForwardFromStored": {"maxAbsDiffX": worst, "sampledRows": int(probe.size),
                                    "scope": "sampled rows only; not every stack"}}


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
