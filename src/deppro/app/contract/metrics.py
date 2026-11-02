"""P1 契約: 既存のベンチマークで使ってきた指標（RMSE・W1・膜厚）の再実装（定義は変えない）と、
**記録のみ**の観測残差。

主指標は総合 RMSE（全画素・全深さ・全元素の平均）と
正準 W1（``depth_profile.wasserstein_distance_1d`` の (px, elem) ベクトル化。等価性はテストで固定）。
膜厚は基板 0.5 交差の線形補間（``deppro.io.dstk.estimate_thickness`` と同じ規則）。
NaN は JSON では null にし、valid 数を併記する。第 3 の定義は導入しない。

観測残差（再 forward）。生成側（``CalculateDistribution.m`` / ``TransformationMatrix.m``）が
宣言する生成式:

    λ(z,i) = γ_eff / Σ_a D(z,a)/λ_basic(a,i)      （局所組成の調和平均。T は画素ごとに違う）
    T(θ,z,i) = cumprod_z(exp(−Δz/(λ(z,i) cosθ))) / exp(·)   （z より上の層だけ）
    I(θ,i) = Σ_z T(θ,z,i)·D(z,i)、ノイズは I に載る
    X(θ,i) = I(θ,i) / Σ_j I(θ,j)                  （角度ごとに全元素で規格化）

``composition_transfer_matrix`` + ``forward`` はこの式そのもの。
ファイルの ``TransMatObs`` は**単一の stack の T**（保存時点のもの）なので、それを全画素に流用すると
組成に応じた残差が出る。これは T の取り違えによるもので、モデルの床ではない。
**合否**は ``closure_gate`` が付ける: noise = 0 のデータセットに対してだけ PASS/FAIL（許容差は
float32 格納精度に対する余裕を持たせた絶対値）、noise > 0 では RECORDED（noise モデルの版と実効値は
producer 契約待ち）。GT を総和 1 に正規化してはならない（``Dstk`` は格納値のまま比較する）。
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence

import numpy as np
from ..._compat import trapezoid

from .dataset_files import jsonable
from .problem import GroundTruth, ProblemSpec

METRIC_VERSION = "deppro-contract-metrics/1"
REFORWARD_FORMULA_VERSION = (
    "producer-declared/1 (2026-09-08): lambda(z,i)=gamma/sum_a D(z,a)/lambda_basic(a,i); "
    "T=cumprod_z(exp(-dz/(lambda cos theta)))/exp(.); I=sum_z T*D; noise on I; "
    "X=I/sum_j I per angle. "
    "Stored TransMatObs is a single stack's T, so a shared-T residual is expected.")
# noise = 0 の閉包ゲート許容差。float32 格納精度 (eps32·max X ≈ 9e-8) に対して約 10 倍、
# ノイズゼロの生成データで観測された最大残差 3e-7 に対して約 3 倍の余裕。
CLOSURE_TOL_ABS_NOISEFREE = 1e-6
GT_EMPTY_THRESHOLD = 1e-6          # depth_profile.wasserstein_distance_1d と同じ


# --------------------------------------------------------------------------
def w1_matrix(D: np.ndarray, Dgt: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """(px, elem) の正準 W1 [Å]。GT の質量が閾値未満なら NaN（deppro 慣例）。

    ``wasserstein_distance_1d(depth, gt, est)`` を各 (px, elem) に適用したものと一致する
    （テストで固定）。
    """
    D = np.asarray(D, dtype=np.float64)
    Dgt = np.asarray(Dgt, dtype=np.float64)
    dz = np.gradient(depth)
    ig = trapezoid(Dgt, depth, axis=1)                          # (px, elem)
    ir = trapezoid(D, depth, axis=1)
    cg = np.cumsum(Dgt / np.maximum(ig, 1e-10)[:, None, :], axis=1) * dz[None, :, None]
    cr = np.cumsum(D / np.maximum(ir, 1e-10)[:, None, :], axis=1) * dz[None, :, None]
    cg = cg / np.maximum(cg[:, -1:, :], 1e-10)
    cr = cr / np.maximum(cr[:, -1:, :], 1e-10)
    w1 = trapezoid(np.abs(cg - cr), depth, axis=1)
    w1[trapezoid(np.abs(Dgt), depth, axis=1) < GT_EMPTY_THRESHOLD] = np.nan
    return w1


def thickness_half_crossing(D_px: np.ndarray, depth: np.ndarray, substrate_idx: int) -> float:
    """基板濃度が 0.5 を初めて超える位置を線形補間（deppro.io.dstk.estimate_thickness と同じ規則）。"""
    D_sub = np.asarray(D_px)[:, substrate_idx]
    crossings = np.where(D_sub >= 0.5)[0]
    if len(crossings) == 0:
        return float(depth[-1])
    idx = int(crossings[0])
    if idx > 0:
        z0, z1 = depth[idx - 1], depth[idx]
        d0, d1 = D_sub[idx - 1], D_sub[idx]
        denom = d1 - d0
        if abs(denom) < 1e-12:
            return float((z0 + z1) / 2)
        return float(z0 + (0.5 - d0) / denom * (z1 - z0))
    return float(depth[0])


def thickness_array(D: np.ndarray, depth: np.ndarray, substrate_idx: int) -> np.ndarray:
    return np.array([thickness_half_crossing(D[q], depth, substrate_idx) for q in range(D.shape[0])])


# --------------------------------------------------------------------------
def composition_transfer_matrix(D: np.ndarray, lambda_basic: np.ndarray, angles_deg: np.ndarray,
                                step: float, gamma_eff: float) -> np.ndarray:
    """組成依存の転送行列 T(D)。``deppro.transfer_matrix`` の一様格子カーネルと同じ式。

    D: (n, depth, elem) → T: (n, angle, depth, elem)。
    λ_obs(z,e) = γ / Σ_i D(z,i)/λ_basic(e,i)（Σ=0 は 1e-10）、
    T(a,z,e) = exp(Σ_{z'<z} −step / (λ_obs(z',e) cos a)) = exp(cumsum(t1) − t1)。
    """
    D = np.asarray(D, dtype=np.float64)
    inv = np.einsum("nzi,ei->nze", D, 1.0 / np.asarray(lambda_basic, dtype=np.float64))
    inv = np.where(inv == 0.0, 1e-10, inv)
    lam = gamma_eff / inv                                          # (n, Z, E)
    cos = np.cos(np.deg2rad(np.asarray(angles_deg, dtype=np.float64)))
    t1 = -step / lam[:, None, :, :] / cos[None, :, None, None]    # (n, A, Z, E)
    return np.exp(np.cumsum(t1, axis=2) - t1)


def forward(T: np.ndarray, D: np.ndarray) -> np.ndarray:
    """X̂ = normalise_angle(Σ_z T·D)。T は共有 (A,Z,E) か画素別 (n,A,Z,E)。暫定式（docstring 参照）。"""
    D = np.asarray(D, dtype=np.float64)
    if T.ndim == 3:
        Xp = np.einsum("aze,nze->nae", T, D)
    else:
        Xp = np.einsum("naze,nze->nae", T, D)
    s = Xp.sum(axis=2, keepdims=True)
    return np.divide(Xp, s, out=np.zeros_like(Xp), where=s > 0)


# --------------------------------------------------------------------------
def _nan_stats(w1: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # All-NaN slice → NaN は仕様
        return np.nanmean(w1, axis=0), np.nanmedian(w1, axis=0), np.sum(~np.isnan(w1), axis=0)


def _metrics_block(D: np.ndarray, Dgt: np.ndarray, depth: np.ndarray, labels: Sequence[str],
                   free: Sequence[int], substrate_idx: int) -> Dict[str, Any]:
    diff = D - Dgt
    rmse_e = np.sqrt(np.mean(diff ** 2, axis=(0, 1)))
    w1 = w1_matrix(D, Dgt, depth)
    w1_mean, w1_med, w1_valid = _nan_stats(w1)
    thk = np.abs(thickness_array(D, depth, substrate_idx) - thickness_array(Dgt, depth, substrate_idx))
    free = list(free)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {
            "n_pixels": int(D.shape[0]),
            "rmse_total": float(np.sqrt(np.mean(diff ** 2))),
            "rmse_free_mean": float(rmse_e[free].mean()) if free else None,
            "rmse_per_element": dict(zip(labels, map(float, rmse_e))),
            "w1_meanpx_per_element_A": dict(zip(labels, map(float, w1_mean))),
            "w1_medianpx_per_element_A": dict(zip(labels, map(float, w1_med))),
            "w1_valid_px_per_element": dict(zip(labels, map(int, w1_valid))),
            "w1_meanpx_all_A": float(np.nanmean(w1_mean)),
            "w1_meanpx_free_A": float(np.nanmean(w1_mean[free])) if free else None,
            "w1_medianpx_all_A": float(np.nanmean(w1_med)),      # 元素ごとの画素中央値を元素平均
            "w1_medianpx_free_A": float(np.nanmean(w1_med[free])) if free else None,
            "thickness_mae_A": float(thk.mean()),
            "thickness_err_median_A": float(np.median(thk)),
        }
    if len(free) == 5:                                           # 秋の表と同じキー名を併記
        out["rmse_free5_mean"] = out["rmse_free_mean"]
        out["w1_meanpx_free5_A"] = out["w1_meanpx_free_A"]
        out["w1_medianpx_free5_A"] = out["w1_medianpx_free_A"]
    return out


def evaluate(D_est: np.ndarray, gt: GroundTruth, spec: ProblemSpec, *,
             free: Optional[Sequence[int]] = None, substrate_idx: Optional[int] = None,
             per_frame: bool = False) -> Dict[str, Any]:
    """既存指標をそのまま計算する。GT は格納値のまま（再正規化しない）。NaN は JSON で null。"""
    D_est = np.asarray(D_est, dtype=np.float64)
    if D_est.shape != gt.D.shape:
        raise ValueError(f"D_est {D_est.shape} vs GT {gt.D.shape}")
    if not np.array_equal(np.asarray(gt.pixel_ids), np.asarray(spec.pixel_ids)):
        raise ValueError("GroundTruth and ProblemSpec pixel_ids differ")
    free = list(spec.elements.free if free is None else free)
    sub = spec.elements.subst[0] if substrate_idx is None else int(substrate_idx)
    if substrate_idx is None and not spec.elements.subst:
        raise ValueError("substrate index unknown: pass substrate_idx")
    labels = spec.elements.labels
    depth = spec.axes.depth_A
    out = {"metric_version": METRIC_VERSION, "gt_source": gt.source,
           "gt_renormalised": False, "free_indices": free, "substrate_index": sub}
    out.update(_metrics_block(D_est, gt.D, depth, labels, free, sub))
    if per_frame:
        frames = spec.frame_of()
        out["per_frame"] = {}
        for fr in np.unique(frames):
            m = frames == fr
            out["per_frame"][str(int(fr))] = _metrics_block(D_est[m], gt.D[m], depth, labels, free, sub)
    return jsonable(out)


def observation_residual(spec: ProblemSpec, D: np.ndarray, *, mask_X_gt: float = 0.02,
                         mode: str = "stored_T") -> Dict[str, Any]:
    """再 forward の残差を**記録**する（合否は ``closure_gate``）。

    mode: ``stored_T`` = 保存 TransMatObs（単一 stack の作用素）を全画素に流用 — 組成依存の分だけ
    残差が出る（説明済み）。``composition_T`` = 各画素の D から T(D) を組み直す生成式そのもの
    （lambda_basic と gamma_eff が要る）。
    """
    D = np.asarray(D, dtype=np.float64)
    if mode == "stored_T":
        T = spec.stored_T
    elif mode == "composition_T":
        if spec.lambda_basic is None or spec.gamma_eff is None:
            return {"status": "NA", "mode": mode, "reason": "lambda_basic/gamma_eff missing"}
        T = composition_transfer_matrix(D, spec.lambda_basic, spec.axes.angles_deg,
                                        spec.axes.step_A, spec.gamma_eff)
    else:
        raise ValueError(mode)
    Xp = forward(T, D)
    X = spec.X
    m = X > mask_X_gt
    rel_full = np.where(m, np.abs(Xp - X) / np.where(m, X, 1.0), 0.0)
    rel = rel_full[m]
    per_px = np.array([np.median(rel_full[i][m[i]]) if m[i].any() else np.nan for i in range(X.shape[0])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {
            "status": "RECORDED", "mode": mode, "formula_version": REFORWARD_FORMULA_VERSION,
            "n_px": int(X.shape[0]), "mask_X_gt": mask_X_gt, "n_entries": int(m.sum()),
            "rel_residual_median": float(np.median(rel)) if rel.size else None,
            "rel_residual_p95": float(np.percentile(rel, 95)) if rel.size else None,
            "rel_residual_max": float(rel.max()) if rel.size else None,
            "abs_residual_max": float(np.abs(Xp - X).max()),
            "rel_residual_by_angle_max": rel_full.max(axis=(0, 2)).tolist(),
            "per_px_median_rel": {"median": float(np.nanmedian(per_px)),
                                  "p95": float(np.nanpercentile(per_px, 95)),
                                  "max": float(np.nanmax(per_px))} if per_px.size else None,
            "note": ("stored_T: single-stack T shared across pixels, residual explained by the "
                     "composition dependence of lambda; composition_T: the producer's forward itself. "
                     "Pass/fail is decided by closure_gate (noise = 0 only)."
                     if mode == "stored_T" else
                     "the producer-declared forward; at noise 0 the residual is float32 storage error; "
                     "at noise > 0 it is the noise realisation (noise model version pending)"),
        }
    return jsonable(out)


def closure_gate(spec: ProblemSpec, D_gt: np.ndarray, *, tol_abs: float = CLOSURE_TOL_ABS_NOISEFREE,
                 mask_X_gt: float = 0.02) -> Dict[str, Any]:
    """GT → 観測の閉包ゲート。生成式（producer-declared/1）で GT から X を作り、格納 X と比べる。

    * noise == 0: max|X̂−X| ≤ tol_abs なら PASS、超えれば FAIL。許容差は float32 格納精度基準。
    * noise > 0 : RECORDED（ノイズ実現分が乗るので合否は付けない。noise モデルの版は契約待ち）。
    * lambda_basic / gamma_eff が無い: NA。
    GT は格納値のまま使う（再正規化しない）。
    """
    D_gt = np.asarray(D_gt, dtype=np.float64)
    if spec.lambda_basic is None or spec.gamma_eff is None:
        return {"status": "NA", "reason": "lambda_basic/gamma_eff missing", "formula_version": REFORWARD_FORMULA_VERSION}
    T = composition_transfer_matrix(D_gt, spec.lambda_basic, spec.axes.angles_deg, spec.axes.step_A, spec.gamma_eff)
    Xp = forward(T, D_gt)
    X = spec.X
    diff = np.abs(Xp - X)
    m = X > mask_X_gt
    rel = diff[m] / X[m] if m.any() else np.array([])
    per_px_max = diff.reshape(X.shape[0], -1).max(axis=1)
    out: Dict[str, Any] = {
        "formula_version": REFORWARD_FORMULA_VERSION, "n_px": int(X.shape[0]), "noise": spec.noise,
        "max_abs": float(diff.max()), "median_abs": float(np.median(diff)),
        "rel_residual_median": float(np.median(rel)) if rel.size else None,
        "rel_residual_max": float(rel.max()) if rel.size else None,
        "n_px_over_tol": int((per_px_max > tol_abs).sum()), "tol_abs": tol_abs,
        "float32_floor_abs": float(np.finfo(np.float32).eps * X.max()),
        "gt_renormalised": False,
    }
    if spec.noise is None:
        out.update(status="UNRESOLVED", reason="noise level unknown; cannot decide exact closure")
    elif spec.noise == 0:
        if diff.max() <= tol_abs:
            out.update(status="PASS", note="exact closure of the producer-declared forward at noise 0")
        else:
            out.update(status="FAIL",
                       note=f"noise 0 but max|X_hat-X| exceeds tol {tol_abs:g}: generator version or "
                            "forward differs from producer-declared/1 (do not renormalise GT to fit)")
            # 診断補助（合否は変えない）: 保存 lambdabasic の向きが MATLAB dialect と逆（転置）なら
            # 再現するか。2026-09-09 の Python producer の実例（値は manifest と一致、H5 の向きだけ逆）。
            if spec.lambda_basic.shape[0] == spec.lambda_basic.shape[1]:
                T_t = composition_transfer_matrix(D_gt, spec.lambda_basic.T, spec.axes.angles_deg,
                                                  spec.axes.step_A, spec.gamma_eff)
                d_t = float(np.abs(forward(T_t, D_gt) - X).max())
                out["max_abs_with_transposed_lambda"] = d_t
                out["transposed_lambda_would_pass"] = bool(d_t <= tol_abs)
                if d_t <= tol_abs:
                    out["note"] += (f". With lambda_basic transposed the closure holds (max {d_t:.2e}): "
                                    "the stored ezdeprof/lambdabasic is probably written in the wrong "
                                    "orientation for the MATLAB HDF5 dialect (producer writer), not a "
                                    "different forward")
    else:
        out.update(status="RECORDED",
                   note="noise > 0: residual includes the noise realisation; noise model version pending")
    return jsonable(out)
