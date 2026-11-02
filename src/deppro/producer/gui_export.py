"""MATLAB GUI が期待する「解析前の状態」を作る export profile。

軽量 profile（解析側 reader 向け）は `dataset.py` のまま。ここはそれに **GUI が
`ReadResult_Callback` で参照する `/ezdeprof` フィールド**と、結果欄の初期化状態を足す。

## 未計算の表し方は**フィールドごとに違う**

MATLAB の「生成済み・解析未実行」の小標本を読んで、初期化規則を確かめた
（他データの解析結果はコピーしていない）:

| フィールド | 解析前の値 | 根拠 |
|---|---|---|
| `Conc_Calc` / `FeatureVector_Calc` / `Sprs_Calc` / `SupportVector_Calc` | 対応する `_Set` の**コピー** | 参照で `_Calc == _Set` がすべて True |
| `Dini` / `Dmod` / `Dopt` | 初期推定プロファイル（3 つとも同じ） | 参照で 3 つがビット一致、かつ GT (`Dobs`) とは max 0.90 違う |
| `Drlt` / `Dtrn` | `Dini` を全 sample へ複製 | 参照で「相異なる行 = 1」「全行が Dini の複製」を確認 |
| `Qini` | 計算値 | 参照で有限 |
| `Qopt` | **NaN** | 参照が NaN。**この 1 つだけ**が NaN |
| `currentloop` / `currentstack` | 1 | 参照 |
| `iterationtotal` | sample 数 | 参照 |
| `calculationinitialized` | `[1, 1, sample 数]` | 参照 |
| `verificationstatus` | 0 | 参照 |

**`Qopt` が NaN だからといって他の結果欄も NaN でよい、とはしない。** 実際 `_Calc` 系は
NaN ではなくコピーだった。

## 初期推定プロファイルは cfg から作る。GT からは作らない

`Dini` は `thckini` / `abrpini` / `subsini` と Homogeneous 分岐（電荷中性を満たす一様組成）
だけで決まる。参照と照合して基板列が erf に一致することを確かめてある。
**生成 GT から導出して解析の初期値へ渡していない。**
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import erf


#: 解析前の状態の識別子。中身を変えたらここも上げる。
PRE_ANALYSIS_STATE = "matlab-Set-initialised/1"

#: InitializeInitialGuess_Parallel の反復回数。生成側の 30 回とは別。
INITIAL_GUESS_ITERATIONS = 1000

#: GUI の既定角度（非一様）。cfg で上書きできる。
THETA_OPT_DEFAULT = [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0,
                     55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]

ATOMIC_NUMBER = {"H":1,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Na":11,"Mg":12,
                 "Al":13,"Si":14,"P":15,"S":16,"Cl":17,"K":19,"Ca":20,"Ti":22,"V":23,
                 "Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,
                 "Ge":32,"As":33,"Se":34,"Zr":40,"Nb":41,"Mo":42,"Ag":47,"In":49,
                 "Sn":50,"Sb":51,"Te":52,"Hf":72,"Ta":73,"W":74,"Pt":78,"Au":79,"Pb":82}
ORBITAL_NUMBER = {"1s":1,"2s":2,"2p":2,"3s":3,"3p":3,"3d":3,"4s":4,"4p":4,"4d":4,"4f":4}


@dataclass(frozen=True)
class GuiSettings:
    """GUI が読む設定。**cfg で明示する**。未計算のものを計算済みに見せない。"""

    theta_opt: Sequence[float] = tuple(THETA_OPT_DEFAULT)
    emission_angle: Optional[float] = None                # None -> max(theta_obs)
    delta_photon: float = 1000.0
    # [value, min, max]。使うのは第 1 要素だけだが、GUI が 3 要素で保持するので合わせる。
    thck_ini: Sequence[float] = (210.0, 0.0, 600.0)
    abrp_ini: Sequence[float] = (20.0, 20.0, 20.0)
    subs_ini: Sequence[float] = (1.0, 1.0, 1.0)
    thck_mod: Sequence[float] = (210.0, 0.0, 600.0)
    abrp_mod: Sequence[float] = (20.0, 20.0, 20.0)
    subs_mod: Sequence[float] = (1.0, 1.0, 1.0)
    colours: Optional[Sequence[Sequence[float]]] = None   # (ne, 3) RGB。None -> 0.9*lines(ne)
    # --- 解析設定（生成側の既定であって、解析を縛るものではない） ---
    loop: int = 10
    kfold: int = 2
    holdout: int = 1
    np_set: Optional[int] = None                          # None -> NP_Max
    np_max: Optional[int] = None                          # None -> ceil(ne/2)*2
    np_calc: int = 2
    nearest_neighbour: int = 1
    optimization_method_value: int = 3      # L1-norm regularization
    initial_guess_value: int = 2            # Iterative updating
    prior_model_value: int = 4              # None
    verification_scheme_value: int = 4      # Nearest-neighbor
    verification_method: int = 2
    transfer_matrix_update_value: int = 2
    multi_depth_profiler: int = 1
    number_of_modeling: int = 1
    alpha_control: int = 0
    degree_control: int = 0
    cross_sec_approx_ord: int = 3
    cross_sec_approx_view: int = 0
    photoionization_cross_section: int = 1
    thickness_variation: int = 65535
    abruptness_variation: int = 0
    subst_conc_variation: int = 0


def _fwhm_sigma(v: float) -> float:
    return v / (2.0 * np.sqrt(np.log(2.0)))


def initial_guess_profile(depth, charges, subst_index, anti_index, thck_ini, abrp_ini,
                          subs_ini=None, const_index=(), linked_index=(), d_const=None):
    """`Dini` / `Dmod` を **cfg だけから** 作る。

    `+deppro/+core/InitializeInitialGuess_Parallel.m` の写し。生成側の
    `CalculateDistribution` とは**別の初期化規則**である点に注意:

    * 基板の取り分は `1/len(subst)`（SupportVector からではない）
    * 膜は電荷中性を満たす一様組成（Dcatani）
    * 電荷ループは **1000 回**、係数 `(1 + 1/k)`
      （生成側は 30 回・`1 + 100*exp(-k/2)`。混同すると Σ_e D が 1.047 まで膨らむ）

    生成した GT は使わない。`d_const` は const 成分がある場合の固定プロファイルで、
    これも解析側の入力であって GT ではない。
    """
    z = np.asarray(depth, dtype=float)
    q = np.asarray(charges, dtype=float)
    ne = q.size
    subst = list(subst_index)
    anti = [e for e in anti_index if e not in set(const_index)]
    d = np.zeros((z.size, ne), dtype=float)

    subs = np.full(len(subst), 1.0 / len(subst))
    if len(subst) == 2:
        qs = q[subst]
        if qs[0] * qs[1] < 0:
            subs = np.linalg.solve(np.array([[1.0, 1.0], qs]), [1.0, 0.0])
        else:
            subs = np.full(2, 0.5)
    elif len(subst) > 2:
        qs = q[subst]
        for _ in range(100):
            if (subs * qs).sum() >= 0:
                subs[qs < 0] *= 1.1
            else:
                subs[qs >= 0] *= 1.1
            subs = subs / subs.sum()
    ramp = 0.5 * np.abs(1.0 + erf((z - float(thck_ini[0])) / _fwhm_sigma(float(abrp_ini[0]))))
    for j, e in enumerate(subst):
        d[:, e] = subs[j] * ramp

    qa = q[anti]
    pos = qa >= 0
    neg = qa < 0
    if pos.any() and neg.any():
        m = np.array([[pos.sum(), neg.sum()], [qa[pos].sum(), qa[neg].sum()]], dtype=float)
        cat, ani = np.linalg.solve(m, [1.0, 0.0])
    else:
        cat = 1.0 / max(pos.sum(), 1)
        ani = 1.0 / max(neg.sum(), 1)

    fix = np.zeros(z.size)
    fix_lock = np.zeros((z.size, 2))
    if len(const_index):
        if d_const is None:
            raise ValueError("const components need their fixed profile (analysis input, not GT)")
        d[:, list(const_index)] = d_const
        if len(linked_index):
            clc = 1.0 - q[list(const_index)] / np.mean(q[list(linked_index)])
            lnk = 0 if np.all(q[list(linked_index)] >= 0) else 1
            fix_lock[:, lnk] = (d_const * (clc - 1.0)).sum(axis=1) / len(linked_index)
        else:
            clc = np.ones(len(const_index))
        fix = (d_const * clc).sum(axis=1)

    rest = np.abs(1.0 - (d[:, subst].sum(axis=1) + fix))
    for k, e in enumerate(anti):
        d[:, e] = (fix_lock[:, 0] + cat * rest) if qa[k] >= 0 else (fix_lock[:, 1] + ani * rest)

    # **1000 回・(1+1/k)**。生成側の 30 回ループとは別。
    pos_anti = [anti[i] for i in np.flatnonzero(pos)]
    neg_anti = [anti[i] for i in np.flatnonzero(neg)]
    for k in range(1, INITIAL_GUESS_ITERATIONS + 1):
        boost = 1.0 + 1.0 / k
        net = d @ q
        if pos_anti:
            d[:, pos_anti] = np.where((net < 0)[:, None], d[:, pos_anti] * boost, d[:, pos_anti])
        if neg_anti:
            d[:, neg_anti] = np.where((net > 0)[:, None], d[:, neg_anti] * boost, d[:, neg_anti])
        d[:, anti] = d[:, anti] / d.sum(axis=1)[:, None]
    return d


def peak_parameters(d: np.ndarray, depth: np.ndarray):
    """p (重心) / w / h。`SimulationUpdater.computeSimulationObservables` と同じ式。"""
    z = np.asarray(depth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (d * z[:, None]).sum(axis=0) / d.sum(axis=0)
        h = 100.0 * d.max(axis=0)
        w = 2.0 * np.sqrt(np.log(2.0) / np.pi) * np.trapezoid(100.0 * d, z, axis=0) / h
    return p, w, h


def default_colours(ne: int) -> np.ndarray:
    """(3, ne) の決定的な配色。GUI 表示用で、数値の意味は持たない。"""
    idx = np.arange(ne, dtype=float)
    return np.stack([(0.37 * idx + 0.11) % 1.0,
                     (0.61 * idx + 0.29) % 1.0,
                     (0.83 * idx + 0.53) % 1.0])


# ---------------------------------------------------------------------------
# 版の固定（2026-09-08）
#
# ここから下の規則は**版付きで凍結する**。数値規則（反復回数・boost 係数・
# 微小値の閾値）を変えるときは、値ではなく版 ID を上げ、差を測って記録する。
# ---------------------------------------------------------------------------

#: 初期推定プロファイルの構成規則。`initial_guess_profile()` の中身と 1:1。
#: 1000 回ループ / boost = 1 + 1/k / 基板の取り分 = 1/len(subst) / 膜は Dcatani。
INITIAL_GUESS_RULE = "InitializeInitialGuess_Parallel/port/1"

#: `Sprs_Set` の「ほぼゼロ」判定。MATLAB: `length(find(Dini(:,free)<1e-5))`。
SPARSITY_THRESHOLD = 1e-5

#: 目的関数を組むときの下限クリップ。MATLAB Set: `minVal = 1e-6`。
OBJECTIVE_MIN_VAL = 1e-6

#: root の結果配列の chunk 行数。小さいほど chunk cache の設定に鈍感になる。
#: 大きな chunk は既定 cache を素通りして速く見えるだけで、cache を広げると大きく落ちる。
ROOT_CHUNK_ROWS = 128

#: MATLAB `0.9*lines(7)` の 7 色（既定の ColorOrder）。8 色目以降は巡回する。
_LINES7 = np.array([
    [0.0000, 0.4470, 0.7410],
    [0.8500, 0.3250, 0.0980],
    [0.9290, 0.6940, 0.1250],
    [0.4940, 0.1840, 0.5560],
    [0.4660, 0.6740, 0.1880],
    [0.3010, 0.7450, 0.9330],
    [0.6350, 0.0780, 0.1840],
])


def matlab_lines_colours(ne: int) -> np.ndarray:
    """`0.9*lines(ne)` を (ne, 3) で返す。GUI の配色検出が Lines を選ぶ値。

    参照 h5 の `Col` と一致することを実測で確かめてある（Checkered-7）。
    数値の意味は持たない表示専用の値。
    """
    return 0.9 * _LINES7[np.arange(int(ne)) % 7]


def string_charges(charges: Sequence[float]) -> List[str]:
    """`DeterminingStringChargeAndElement` の stringcharge。整数電荷のみ対応。"""
    out = []
    for q in charges:
        q = float(q)
        if abs(q - round(q)) > 1e-3:
            raise NotImplementedError(
                f"non-integer charge {q!r}: MATLAB writes rat(charge,1e-3) as 'n/d'; "
                "that branch is not ported")
        n = int(round(q))
        out.append(f"+{n}" if n > 0 else str(n))
    return out


def element_labels(symbols: Sequence[str], charges: Sequence[float]) -> List[str]:
    """`element` = 元素記号 + "(" + stringcharge + ")"。"""
    return [f"{s}({c})" for s, c in zip(symbols, string_charges(charges))]


def lambda_from_profile(d: np.ndarray, lambda_basic: np.ndarray, gamma_eff: float) -> np.ndarray:
    """`lambda(z,i) = gamma_eff / sum_a D(z,a)/L(a,i)`。`L` は `L[a, i]`（MATLAB の向き）。"""
    d = np.asarray(d, dtype=float)                      # (nz, ne_a)
    inv = 1.0 / np.asarray(lambda_basic, dtype=float)   # (a, i)
    return float(gamma_eff) / (d @ inv)                 # (nz, ne_i)


#: T の作り方の版。**式を変えたら値ではなく版 ID を上げる。**
#: 90 度列の扱いが式に依存するので、式そのものを契約にする（下の docstring）。
TRANSFER_MATRIX_RULE = "cumprod-over-exp/matlab-TransformationMatrix/1"

#: 90 度（cos = 0）の列の扱い。`cumprod(e)/e` は 0/0 で NaN になる。
NINETY_DEGREE_POLICY = ("NaN. theta=90 deg gives cos=0, t1=-inf, exp(t1)=0 and cumprod/exp = 0/0. "
                        "The equivalent-looking exp(cumsum(t1)-t1) does NOT produce NaN, because "
                        "numpy's cos(pi/2) is 6.1e-17 rather than 0 -- so the NaN pattern is a "
                        "property of the FORMULA, not of the angle grid. This producer pins the "
                        "cumprod form to match MATLAB's +deppro/+core/TransformationMatrix.m, and "
                        "the reference file's NaN positions were checked to agree exactly. "
                        "Consumers must compare NaN positions, not just values.")


def transfer_matrix(lam: np.ndarray, angles_deg: np.ndarray, step_size: float) -> np.ndarray:
    """`T(th,z,i) = cumprod_z(exp t1)/exp t1`, `t1 = -dz/(lambda cos th)`。(nth, nz, ne)。

    式は `TRANSFER_MATRIX_RULE` で版付け。**90 度列が NaN になるかどうかは式に依存する**
    ので、`NINETY_DEGREE_POLICY` を読むこと。`cumprod(e)/e`（この実装、MATLAB と同形）は
    NaN を出し、`exp(cumsum(t1)-t1)` は出さない。**握りつぶさない。**
    """
    lam = np.asarray(lam, dtype=float)
    th = np.deg2rad(np.asarray(angles_deg, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-float(step_size) / lam[None, :, :]) / np.cos(th)[:, None, None]
        e = np.exp(t1)
        return np.cumprod(e, axis=1) / e


def lambda_cosine_theta(lam: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """`mean_z(lambda) .* cos(theta)`。(nth, ne)。"""
    th = np.deg2rad(np.asarray(angles_deg, dtype=float))
    return np.asarray(lam, dtype=float).mean(axis=0)[None, :] * np.cos(th)[:, None]


def angle_profile(d: np.ndarray, trans_mat: np.ndarray) -> np.ndarray:
    """`X(th,i) = I/sum_j I`,  `I(th,i) = sum_z T(th,z,i) D(z,i)`。(nth, ne)。"""
    with np.errstate(invalid="ignore"):
        i1 = np.einsum("tze,ze->te", np.asarray(trans_mat, float), np.asarray(d, float))
        return i1 / i1.sum(axis=1, keepdims=True)


def concentration(d: np.ndarray, anti_index: Sequence[int]) -> np.ndarray:
    """`Conc = 100*sum(D)/sum(D(:,antisubst),'all')`。基板が 100 を超えるのは正常。"""
    d = np.asarray(d, dtype=float)
    return 100.0 * d.sum(axis=0) / d[:, list(anti_index)].sum()


def sparsity(d: np.ndarray, free_index: Sequence[int]) -> np.ndarray:
    """`Sprs = [count(D(:,free) < 1e-5), nz*nfree]`。閾値は凍結（SPARSITY_THRESHOLD）。"""
    d = np.asarray(d, dtype=float)
    free = list(free_index)
    return np.array([float((d[:, free] < SPARSITY_THRESHOLD).sum()), float(d.shape[0] * len(free))])


def feature_vector_from_profile(d, depth, thck_abrp, freeconst_index) -> np.ndarray:
    """`FV_Set = [thck abrp p/w/h(freeconst)]`（`SimulationUpdater` と `Set` で同一）。"""
    p, w, h = peak_parameters(np.asarray(d, float), np.asarray(depth, float))
    out = [float(thck_abrp[0]), float(thck_abrp[1])]
    for e in freeconst_index:
        out.extend([p[e], w[e], h[e]])
    return np.asarray(out, dtype=float)


def support_vector_from_profile(d, depth, subst_index, linked_index) -> np.ndarray:
    """`SV = [p/w/h(subst) p/w/h(linked)]`。`profile.support_vector_from_profile` と同式。"""
    p, w, h = peak_parameters(np.asarray(d, float), np.asarray(depth, float))
    out: List[float] = []
    for e in list(subst_index) + list(linked_index):
        out.extend([p[e], w[e], h[e]])
    return np.asarray(out, dtype=float)


def _set_model_profile(*, d_ini, depth, charge, subst, free, const, linked,
                       dep_ind, thck, abrp, subs_h, x_free, min_val):
    """`ObjectFcn_Set_Parallel` の内側 `CalcDistrib` が組む `Dopt`。stack に依らない。"""
    z = np.asarray(depth, dtype=float)
    q = np.asarray(charge, dtype=float)
    if linked:
        flc = 1.0 - q[free] / q[linked].mean()
        clc = 1.0 - q[const] / q[linked].mean() if const else np.zeros(0)
        flc = np.where(flc < 1.0, 1.0, flc)
        clc = np.where(clc < 1.0, 1.0, clc)
    else:
        flc = np.ones(len(free)); clc = np.ones(len(const))
    d = np.array(d_ini, dtype=float, copy=True)
    ramp = 0.5 * np.abs(1.0 + erf((z[dep_ind] - thck) / (abrp / (2.0 * np.sqrt(np.log(2.0))))))
    d[np.ix_(dep_ind, subst)] = np.asarray(subs_h, float)[None, :] * ramp[:, None]
    d[np.ix_(dep_ind, free)] = x_free * (1.0 - d[np.ix_(dep_ind, subst)].sum(axis=1))[:, None]
    if linked:
        cols = free + const
        w = np.concatenate([flc - 1.0, clc - 1.0]) if const else (flc - 1.0)
        d[np.ix_(dep_ind, linked)] = (d[np.ix_(dep_ind, cols)] * w[None, :]).sum(axis=1)[:, None]
    d[d > 1.0] = 1.0
    d[d < min_val] = min_val
    return d


def initial_objective_series(*, x_obs_all, d_ini, d_mod, depth, charge, subst, antisubst,
                             free, const, linked, trans_mat_obs, ang_ind, dep_ind,
                             thck_ini, abrp_ini, subs_ini, thck_mod, abrp_mod, subs_mod,
                             optimization_method_value, degree=1.0, alpha=1.0,
                             min_val=OBJECTIVE_MIN_VAL):
    """全 stack の `Qini = [0.5*Chi2, -alpha*S, ObjFnc]` を (P, 3) で返す。

    `DepthProfiler.m` の Set 分岐と `+deppro/+core/ObjectFcn_Set_Parallel.m` の写し。
    模擬側 `Xsimu` は **stack に依らない**（初期推定だけから決まる）ので 1 度だけ
    組み、stack ごとに変わる `Xdata`/`Xreg` だけを回す。

    `x0` の末尾は **[thck, abrp, subst 高さ] のスカラー**である（`thckini` などの
    3 要素ベクトル全体ではない）。これは推測ではなく参照 `Qstk` との照合で決めた:
    3 要素ベクトルを丸ごと連結する読みでは Q(1) が参照と合わず、スカラーの読みでは
    全 stack で参照と一致した（float32 の精度の範囲）。

    `optimizationmethodvalue` は 1 (LLS) と 3 (L1) のみ。基板が 2 成分以上のときの
    `subsini` の割り当ては参照から確かめていないので拒否する。
    """
    if optimization_method_value not in (1, 3):
        raise NotImplementedError(
            f"optimizationmethodvalue={optimization_method_value} は未移植（1=LLS と 3=L1 のみ）")
    subst = list(subst); antisubst = list(antisubst)
    free = list(free); const = list(const); linked = list(linked)
    if len(subst) != 1:
        raise NotImplementedError(
            "subst が 2 成分以上のときの subsini -> 基板高さの割り当ては参照未確認")
    ang_ind = np.asarray(ang_ind, dtype=int); dep_ind = np.asarray(dep_ind, dtype=int)
    m = dep_ind.size; nf = len(free)

    def _tail(t, a, sb):
        return np.concatenate([[float(np.ravel(t)[0]), float(np.ravel(a)[0])],
                               np.ravel(np.asarray(sb, float))[:len(subst)]])

    x = np.concatenate([np.asarray(d_ini, float)[np.ix_(dep_ind, free)].reshape(m * nf, order="F"),
                        _tail(thck_ini, abrp_ini, subs_ini)])
    m0 = np.concatenate([np.asarray(d_mod, float)[np.ix_(dep_ind, free)].reshape(m * nf, order="F"),
                         _tail(thck_mod, abrp_mod, subs_mod)])
    if optimization_method_value == 3:
        m0 = np.zeros_like(m0)
    x = np.where(x < min_val, min_val, x)
    m0 = np.where(m0 < min_val, min_val, m0)

    dopt = _set_model_profile(d_ini=d_ini, depth=depth, charge=charge, subst=subst, free=free,
                              const=const, linked=linked, dep_ind=dep_ind, thck=x[m * nf],
                              abrp=x[m * nf + 1], subs_h=x[m * nf + 2: m * nf + 2 + len(subst)],
                              x_free=x[:m * nf].reshape(m, nf, order="F"), min_val=min_val)

    if optimization_method_value == 1:
        s_reg = 0.0
    else:
        s0 = abs(float((x[:m * nf] - m0[:m * nf]).sum()))
        s1 = abs(float(x[m * nf] - m0[m * nf])) / 100.0
        s_reg = -(s0 + s1)

    xo = np.atleast_3d(np.asarray(x_obs_all, dtype=float))     # (P, nth, ne)
    if xo.ndim == 2:
        xo = xo[None, :, :]
    out = np.empty((xo.shape[0], 3), dtype=float)
    tm = np.asarray(trans_mat_obs, float)
    for s in range(xo.shape[0]):
        xd_full = xo[s]
        xdata = xd_full[np.ix_(ang_ind, antisubst)] / xd_full[np.ix_(ang_ind, subst)].sum(axis=1, keepdims=True)
        ok = np.isfinite(xdata).all(axis=1)
        ang = ang_ind[ok]; xdata = xdata[ok]
        xreg = np.repeat(xdata.mean(axis=0)[None, :], xdata.shape[0], axis=0)
        xreg[xreg == 0] = 1.0
        i1 = np.einsum("tze,ze->te", tm[ang], dopt)
        xall = i1 / i1.sum(axis=1, keepdims=True)
        xsimu = xall[:, antisubst] / xall[:, subst].sum(axis=1, keepdims=True)
        chi2 = float(((xdata - xsimu) ** 2 / xreg ** 2).sum() * degree)
        obj = 0.5 * chi2 if optimization_method_value == 1 else 0.5 * chi2 - alpha * s_reg
        out[s] = (0.5 * chi2, -alpha * s_reg, obj)
    return out


def initial_objective(*, x_obs, **kw):
    """1 stack ぶんの `Qini`。`initial_objective_series` の薄い包み。"""
    return initial_objective_series(x_obs_all=np.asarray(x_obs, float)[None, :, :], **kw)[0]


# ---------------------------------------------------------------------------
# 生成時に確定させる「解析前の状態」
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreAnalysisState:
    """生成の中で 1 度だけ組む解析前の状態。**export では再計算しない。**

    `support_vector_set` は生成に実際に使った初期 SV（cfg 明示値）をそのまま持つ。
    `support_vector_set_derived` は同じ状態の `Dini` から導いた値で、**記録用**。
    MATLAB は後者を `SupportVector_Set` に書くので、両者の差を manifest に残す。
    """

    rule: str
    initial_guess_rule: str
    d_ini: np.ndarray          # (nz, ne)
    d_mod: np.ndarray
    d_opt: np.ndarray
    x_ini: np.ndarray          # (n_opt, ne)
    x_mod: np.ndarray
    x_opt: np.ndarray
    lambda_obs: np.ndarray     # (nz, ne)
    lambda_opt: np.ndarray
    trans_mat_obs: np.ndarray  # (n_obs, nz, ne)
    trans_mat_opt: np.ndarray  # (n_opt, nz, ne)
    lambda_cos_obs: np.ndarray # (n_obs, ne)
    lambda_cos_opt: np.ndarray
    feature_vector_set: np.ndarray
    support_vector_set: np.ndarray
    support_vector_set_derived: np.ndarray
    conc_set: np.ndarray
    sprs_set: np.ndarray
    q_ini: np.ndarray          # (3,)
    q_stack: np.ndarray        # (P, 3)
    fv_stack: np.ndarray       # (P, 2+3*nfreeconst)
    sv_stack: np.ndarray       # (P, 3*(nsubst+nlinked))
    c_stack: np.ndarray        # (P, ne)
    sps_stack: np.ndarray      # (P, 2)
    elapsed_seconds: float


def _peaks_series(d: np.ndarray, depth: np.ndarray):
    """(P, nz, ne) -> p, w, h それぞれ (P, ne)。`peak_parameters` と同式。"""
    z = np.asarray(depth, dtype=float)
    d = np.asarray(d, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (d * z[None, :, None]).sum(axis=1) / d.sum(axis=1)
        h = 100.0 * d.max(axis=1)
        w = 2.0 * np.sqrt(np.log(2.0) / np.pi) * np.trapezoid(100.0 * d, z, axis=1) / h
    return p, w, h


def build_pre_analysis_state(*, depth, charge, subst_index, anti_index, free_index,
                             const_index, linked_index, thck_ini, abrp_ini, subs_ini,
                             thck_mod, abrp_mod, subs_mod, lambda_basic, gamma_eff,
                             step_size, theta_obs, theta_opt, profiles, x_obs_all,
                             feature_vectors, support_vector_set,
                             optimization_method_value, current_stack=1,
                             degree=1.0, alpha=1.0, d_const=None) -> PreAnalysisState:
    """cfg と生成済み GT から、解析前の状態を 1 度だけ組む。

    `Dini` は **cfg だけ**から作る（GT は使わない）。GT を使うのは MATLAB と同じ
    範囲に限る: `Dobs`/`Xobs`（観測パネル）と `*stk` 系（生成側の観測量）。
    """
    t0 = time.time()
    z = np.asarray(depth, dtype=float)
    q = np.asarray(charge, dtype=float)
    subst = list(subst_index); anti = list(anti_index)
    free = list(free_index); const = list(const_index); linked = list(linked_index)
    freeconst = sorted(free + const)
    tho = np.asarray(theta_obs, dtype=float); thp = np.asarray(theta_opt, dtype=float)
    d = np.asarray(profiles, dtype=float)               # (P, nz, ne)
    nz, ne = z.size, q.size

    d_ini = initial_guess_profile(z, q, subst, anti, thck_ini, abrp_ini,
                                  const_index=const, linked_index=linked, d_const=d_const)
    d_mod = initial_guess_profile(z, q, subst, anti, thck_mod, abrp_mod,
                                  const_index=const, linked_index=linked, d_const=d_const)
    lam = lambda_from_profile(d_ini, lambda_basic, gamma_eff)
    t_obs = transfer_matrix(lam, tho, step_size)
    t_opt = transfer_matrix(lam, thp, step_size)
    x_ini = angle_profile(d_ini, t_opt)

    fv_set = feature_vector_from_profile(d_ini, z, (np.ravel(thck_ini)[0], np.ravel(abrp_ini)[0]),
                                         freeconst)
    sv_derived = support_vector_from_profile(d_ini, z, subst, linked)
    sv_set = np.asarray(support_vector_set, dtype=float).ravel()

    p, w, h = _peaks_series(d, z)
    fv = np.atleast_2d(np.asarray(feature_vectors, dtype=float))
    fv_stack = np.concatenate(
        [fv[:, :2]] + [np.stack([p[:, e], w[:, e], h[:, e]], axis=1) for e in freeconst], axis=1)
    sv_stack = np.concatenate(
        [np.stack([p[:, e], w[:, e], h[:, e]], axis=1) for e in subst + linked], axis=1)
    c_stack = 100.0 * d.sum(axis=1) / d[:, :, anti].sum(axis=(1, 2))[:, None]
    sps_stack = np.stack([(d[:, :, free] < SPARSITY_THRESHOLD).sum(axis=(1, 2)).astype(float),
                          np.full(d.shape[0], float(nz * len(free)))], axis=1)

    q_stack = initial_objective_series(
        x_obs_all=np.asarray(x_obs_all, dtype=float), d_ini=d_ini, d_mod=d_mod, depth=z,
        charge=q, subst=subst, antisubst=anti, free=free, const=const, linked=linked,
        trans_mat_obs=t_obs, ang_ind=np.arange(tho.size), dep_ind=np.arange(nz),
        thck_ini=thck_ini, abrp_ini=abrp_ini, subs_ini=subs_ini,
        thck_mod=thck_mod, abrp_mod=abrp_mod, subs_mod=subs_mod,
        optimization_method_value=optimization_method_value, degree=degree, alpha=alpha)

    return PreAnalysisState(
        rule=PRE_ANALYSIS_STATE, initial_guess_rule=INITIAL_GUESS_RULE,
        d_ini=d_ini, d_mod=d_mod, d_opt=d_ini.copy(),
        x_ini=x_ini, x_mod=x_ini.copy(), x_opt=x_ini.copy(),
        lambda_obs=lam, lambda_opt=lam.copy(),
        trans_mat_obs=t_obs, trans_mat_opt=t_opt,
        lambda_cos_obs=lambda_cosine_theta(lam, tho),
        lambda_cos_opt=lambda_cosine_theta(lam, thp),
        feature_vector_set=fv_set, support_vector_set=sv_set,
        support_vector_set_derived=sv_derived,
        conc_set=concentration(d_ini, anti), sprs_set=sparsity(d_ini, free),
        q_ini=q_stack[int(current_stack) - 1].copy(), q_stack=q_stack,
        fv_stack=fv_stack, sv_stack=sv_stack, c_stack=c_stack, sps_stack=sps_stack,
        elapsed_seconds=round(time.time() - t0, 4))


# ---------------------------------------------------------------------------
# 書き出し（MATLAB h5write の向き: HDF5 の次元は MATLAB の逆順）
# ---------------------------------------------------------------------------

def _pad_cols(a: np.ndarray, width: int) -> np.ndarray:
    """(P, k) を (P, width) へ 0 詰め。MATLAB は `*stk` を全元素幅で確保する。"""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    if a.shape[1] > width:
        raise ValueError(f"width {a.shape[1]} > {width}")
    out = np.zeros((a.shape[0], width), dtype=float)
    out[:, :a.shape[1]] = a
    return out


def write_gui_datasets(f, *, state: PreAnalysisState, gui: GuiSettings, ne: int, nz: int,
                       n_stack: int, symbols, charges, orbitals, subst_index, anti_index,
                       free_index, const_index, linked_index, materials, theta_obs,
                       photon_energy, feature_vectors, current_stack: int = 1) -> Dict[str, Any]:
    """開いてある h5py.File に GUI 用フィールドを足す。戻り値は書いた名前の一覧。

    軽量 profile が既に書いたもの（`Dstk`/`Xstk`/`stackxyt`/役割/forward 条件）は
    **触らない**。ここで足すのは解析前の状態と、その結果欄の初期化だけ。
    """
    import h5py
    g = f["ezdeprof"]
    s = h5py.string_dtype("utf-8")
    written: List[str] = []

    def ez(name, data, dtype=None):
        g.create_dataset(name, data=np.asarray(data, dtype=dtype) if dtype else data)
        written.append("/ezdeprof/" + name)

    def ez_str(name, values):
        g.create_dataset(name, data=np.array(list(values), dtype=object).reshape(-1, 1), dtype=s)
        written.append("/ezdeprof/" + name)

    def root(name, data, dtype, stack_axis=0):
        """stack 軸だけ伸長可能にして書く。chunk は **stack 軸だけを切り、行数を小さく**。

        経緯:
        1. `maxshape` だけ渡すと h5py が chunk を自動で選び、元素軸・深さ軸まで切ることがある。
           1 画素読むのに多数の chunk を触るので、ランダムな 1 行の読み出しが遅い。
        2. 行方向に大きな chunk にすると速く見えるが、これは **chunk が h5py 既定の
           chunk cache（1 MiB）に入らず、キャッシュを素通りしていた偶然**。
           `rdcc_nbytes` を広げると大きく落ちる。
        3. いまは **128 行**。cache 設定に依らず安定する範囲。
           連続レイアウトが最速だが、MATLAB 側が伸長可能を前提にする経路があるので
           伸長可能のまま chunk を小さくする方を採った。
        """
        a = np.asarray(data, dtype=dtype)
        ms = list(a.shape); ms[stack_axis] = None
        ch = list(a.shape)
        ch[stack_axis] = max(1, min(a.shape[stack_axis], ROOT_CHUNK_ROWS))
        f.create_dataset(name, data=a, maxshape=tuple(ms), chunks=tuple(ch))
        written.append("/" + name)

    tho = np.asarray(theta_obs, dtype=float)
    thp = np.asarray(gui.theta_opt, dtype=float)
    np_max = gui.np_max if gui.np_max is not None else int(np.ceil(ne / 2) * 2)
    np_set = gui.np_set if gui.np_set is not None else np_max
    angle = float(gui.emission_angle) if gui.emission_angle is not None else float(tho.max())
    col = np.asarray(gui.colours, dtype=float) if gui.colours is not None else matlab_lines_colours(ne)

    # --- 元素の見出しと材料定数 -------------------------------------------------
    ez_str("element", element_labels(symbols, charges))
    ez_str("stringcharge", string_charges(charges))
    ez("Col", col.T, "f4")                                       # MATLAB (ne,3)
    ez("elementnumber", np.array([ATOMIC_NUMBER[x] for x in symbols], dtype="u1").reshape(-1, 1))
    ez("orbitalnumber", np.array([ORBITAL_NUMBER[o] for o in orbitals], dtype="u1").reshape(-1, 1))
    ez("antisubst", np.asarray([i + 1 for i in anti_index], dtype="u1").reshape(-1, 1))
    for name, arr in (("valencecharge", materials.valence_charge), ("density", materials.density),
                      ("molecularweight", materials.molecular_weight), ("bandgap", materials.band_gap)):
        ez(name, np.asarray(arr, dtype="f4").reshape(ne, 1))

    # --- 測定条件 ---------------------------------------------------------------
    ez("datapoints", np.array([tho.size, 1, thp.size], dtype="u2").reshape(3, 1))
    ez("angle", np.array([[angle]], dtype="f4"))
    ez("photon", np.array([[float(photon_energy)]], dtype="f4"))
    ez("deltaphoton", np.array([gui.delta_photon], dtype="f4"))
    ez("thetaopt", thp.astype("f4").reshape(1, -1))
    ez("slice", np.array([nz], dtype="f4"))

    # --- 場面の初期パラメータ（cfg から。GT からではない） -----------------------
    for name, v in (("thckini", gui.thck_ini), ("abrpini", gui.abrp_ini), ("subsini", gui.subs_ini),
                    ("thckmod", gui.thck_mod), ("abrpmod", gui.abrp_mod), ("subsmod", gui.subs_mod)):
        ez(name, np.asarray(v, dtype="f4").reshape(-1, 1))

    # --- 解析前の状態 -----------------------------------------------------------
    ez("lambdaobs", state.lambda_obs.T.astype("f4").reshape(1, ne, nz))
    ez("lambdaopt", state.lambda_opt.T.astype("f4").reshape(1, ne, nz))
    ez("TransMatOpt", np.transpose(state.trans_mat_opt, (2, 1, 0)).astype("f4"))
    ez("lambdacosinethetaobs", state.lambda_cos_obs.T.astype("f4"))
    ez("lambdacosinethetaopt", state.lambda_cos_opt.T.astype("f4"))
    ez("Dini", state.d_ini.T.astype("f4"))
    ez("Dmod", state.d_mod.T.astype("f4"))
    ez("Dopt", state.d_opt.T.astype("f4"))
    ez("Xini", state.x_ini.T.astype("f4"))
    ez("Xmod", state.x_mod.T.astype("f4"))
    ez("Xopt", state.x_opt.T.astype("f4"))
    ez("Dobs", np.asarray(f["Dstk"][current_stack - 1], dtype="f4"))
    ez("Xobs", np.asarray(f["Xstk"][current_stack - 1], dtype="f4"))
    ez("FeatureVector_Set", state.feature_vector_set.astype("f4").reshape(-1, 1))
    ez("FeatureVector_Calc", state.feature_vector_set.astype("f4").reshape(-1, 1))
    ez("SupportVector_Calc", state.support_vector_set.astype("f4").reshape(-1, 1))
    ez("Conc_Set", state.conc_set.astype("f4").reshape(-1, 1))
    ez("Conc_Calc", state.conc_set.astype("f4").reshape(-1, 1))
    ez("Sprs_Set", state.sprs_set.astype("f4").reshape(-1, 1))
    ez("Sprs_Calc", state.sprs_set.astype("f4").reshape(-1, 1))
    ez("Qini", state.q_ini.astype("f4").reshape(-1, 1))
    ez("Qopt", np.full((3, 1), np.nan, dtype="f4"))              # 参照でも Qopt だけが NaN

    # --- 進捗（未解析） ---------------------------------------------------------
    ez("currentloop", np.array([1], dtype="u2"))
    ez("currentstack", np.array([current_stack], dtype="u8"))
    ez("iterationinterval", np.array([1], dtype="u8"))
    ez("iterationtotal", np.array([n_stack], dtype="u8"))
    ez("iterationtime", np.array([state.elapsed_seconds], dtype="f4"))
    ez("calculationinitialized", np.array([1, 1, n_stack], dtype="u8").reshape(3, 1))
    ez("verificationstatus", np.array([0], dtype="u1"))

    # --- GUI 設定（cfg で明示。解析を縛るものではない） --------------------------
    for name, v in (("NP_Set", np_set), ("NP_Calc", gui.np_calc), ("NP_Max", np_max),
                    ("crossecapproximord", gui.cross_sec_approx_ord),
                    ("crossecapproximview", gui.cross_sec_approx_view),
                    ("optimizationmethodvalue", gui.optimization_method_value),
                    ("initialguessvalue", gui.initial_guess_value),
                    ("priormodelvalue", gui.prior_model_value),
                    ("transfermatrixupdatevalue", gui.transfer_matrix_update_value),
                    ("photoionizationcrosssection", gui.photoionization_cross_section),
                    ("verificationschemevalue", gui.verification_scheme_value),
                    ("verificationmethod", gui.verification_method),
                    ("degreecontrol", gui.degree_control), ("alphacontrol", gui.alpha_control),
                    ("multidepthprofiler", gui.multi_depth_profiler)):
        ez(name, np.array([v], dtype="u1"))
    for name, v in (("loop", gui.loop), ("Kfold", gui.kfold),
                    ("NumberofNearestNeighbor", gui.nearest_neighbour),
                    ("thicknessvariation", gui.thickness_variation),
                    ("abruptnessvariation", gui.abruptness_variation),
                    ("substconcvariation", gui.subst_conc_variation)):
        ez(name, np.array([v], dtype="u2"))
    ez("holdout", np.array([gui.holdout], dtype="u8"))
    ez("numberofmodeling", np.array([gui.number_of_modeling], dtype="u8"))

    # --- root の結果欄（解析前の初期化） -----------------------------------------
    fvw = 2 + 3 * ne
    svw = 3 * ne
    p = n_stack
    root("Drlt", np.repeat(state.d_ini.T[None, :, :], p, axis=0), "f4")
    root("Dtrn", np.repeat(state.d_ini.T[None, :, :], p, axis=0), "f4")
    root("Xrlt", np.repeat(state.x_ini.T[None, :, :], p, axis=0), "f4")
    root("FV0stk", _pad_cols(feature_vectors, fvw).T, "f4", stack_axis=1)
    root("FVstk", _pad_cols(state.fv_stack, fvw).T, "f4", stack_axis=1)
    fv_rlt = _pad_cols(np.repeat(state.feature_vector_set[None, :], p, axis=0), fvw).T
    root("FVrlt", fv_rlt, "f4", stack_axis=1)
    root("FVtrn", fv_rlt, "f4", stack_axis=1)
    root("SVstk", _pad_cols(state.sv_stack, svw).T, "f4", stack_axis=1)
    sv_rlt = _pad_cols(np.repeat(state.support_vector_set[None, :], p, axis=0), svw).T
    root("SVrlt", sv_rlt, "f4", stack_axis=1)
    root("SVtrn", sv_rlt, "f4", stack_axis=1)
    root("Cstk", state.c_stack.T, "f4", stack_axis=1)
    c_rlt = np.repeat(state.conc_set[:, None], p, axis=1)
    root("Crlt", c_rlt, "f4", stack_axis=1)
    root("Ctrn", c_rlt, "f4", stack_axis=1)
    root("SPSstk", state.sps_stack.T, "f4", stack_axis=1)
    sps_rlt = np.repeat(state.sprs_set[:, None], p, axis=1)
    root("SPSrlt", sps_rlt, "f4", stack_axis=1)
    root("SPStrn", sps_rlt, "f4", stack_axis=1)
    for name in ("Qstk", "Qrlt", "Qtrn"):
        root(name, state.q_stack.T, "f8", stack_axis=1)
    for name in ("degree", "alpha"):
        root(name, np.ones((3, p)), "f8", stack_axis=1)
    return {"written": written, "count": len(written)}
