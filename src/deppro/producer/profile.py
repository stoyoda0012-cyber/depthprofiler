"""FeatureVector -> 深さプロファイル D。

MATLAB 側の対応（`+deppro/+core/CalculateDistribution.m`、`'Simulation'` 分岐）。
互換モードなので **式の版と反復回数を identity に持たせ、挙動を勝手に直さない**。

特に、末尾の 30 回固定ループは:

    for k = 1:30
        psuni = sum(D.*q,2)<0;  nguni = sum(D.*q,2)>0
        D(psuni, antisubst(q_anti>=0)) *= (1+100*exp(-k/2))
        D(nguni, antisubst(q_anti< 0)) *= (1+100*exp(-k/2))
        D(:,antisubst) /= sum(D,2)            % 非基板列だけを、基板込みの和で割る
    end

不動点は総和 1 だが収束は調和的で、30 回では界面（基板が 1 に近く膜が残る所）で
最大 +4.7% の超過が残る。**これを直すと既存 GT と別物になる**ので直さない。
収束改善は別モード・別 identity。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import erf

#: 互換モードの identity。式の版と反復回数を含める。
COMPAT_IDENTITY = "matlab-CalculateDistribution/Simulation/Gaussian/iter30"

CHARGE_ITERATIONS = 30


@dataclass(frozen=True)
class ProfileSpec:
    """D を組むのに要るもの（stack に依らない部分）。"""

    depth: np.ndarray                 # (nz,) Å
    charge: np.ndarray                # (ne,)
    subst_index: Sequence[int]        # 0-based
    free_index: Sequence[int]         # 0-based
    const_index: Sequence[int]        # 0-based
    linked_index: Sequence[int]       # 0-based
    support_vector: np.ndarray        # SupportVector_Set (1-D)
    distribution: str = "Gaussian"


def _gaussian_sigma(width: np.ndarray) -> np.ndarray:
    return width / (2.0 * np.sqrt(np.log(2.0)))


def depth_profiles(fv: np.ndarray, spec: ProfileSpec) -> np.ndarray:
    """``D[s, z, e]`` を返す。`fv` は (nstack, 2+3*nfree)。"""
    if spec.distribution != "Gaussian":
        raise NotImplementedError(
            f"distribution={spec.distribution!r} is not implemented "
            "(MATLAB has Homogeneous/Error fcn./Exponential/Gaussian/Double Gaussian/Rectangle/Triangle)"
        )
    fv = np.atleast_2d(np.asarray(fv, dtype=float))
    z = np.asarray(spec.depth, dtype=float)
    q = np.asarray(spec.charge, dtype=float)
    ns, nz, ne = fv.shape[0], z.size, q.size
    subst = list(spec.subst_index)
    fcl = sorted(list(spec.free_index) + list(spec.const_index) + list(spec.linked_index))
    anti = [i for i in range(ne) if i not in subst]

    # x = [FeatureVector , SupportVector の linked 部分]  (MATLAB: x = [x y])
    sv = np.asarray(spec.support_vector, dtype=float).ravel()
    subs_height = sv.reshape(-1, 3)[: len(subst), 2] / 100.0     # 3 行目 = height
    x = np.concatenate([fv, np.repeat(sv[3 * len(subst):][None, :], ns, axis=0)], axis=1)

    thck = fv[:, 0]
    abrp = _gaussian_sigma(fv[:, 1])

    D = np.zeros((ns, nz, ne), dtype=float)
    # 基板: erf の立ち上がり
    ramp = 0.5 * np.abs(1.0 + erf((z[None, :] - thck[:, None]) / abrp[:, None]))   # (ns, nz)
    for j, e in enumerate(subst):
        D[:, :, e] = subs_height[j] * ramp
    remainder = np.abs(1.0 - D[:, :, subst].sum(axis=2))                            # (ns, nz)

    # 膜成分: Gaussian 分岐（MATLAB は 1-based で x(3k), x(3k+1), x(3k+2)）
    for k, e in enumerate(fcl, start=1):
        pos = x[:, 3 * k - 1]
        sig = _gaussian_sigma(x[:, 3 * k])
        hgt = x[:, 3 * k + 1] / 100.0
        with np.errstate(divide="ignore", invalid="ignore"):
            shape = np.exp(-((z[None, :] - pos[:, None]) ** 2) / sig[:, None] ** 2)
        D[:, :, e] = hgt[:, None] * shape * remainder

    # MATLAB: D(D<0)=0; D(D>1)=1; D(isnan(D))=1;  ── NaN は 1 になる（0 ではない）
    D[D < 0] = 0.0
    D[D > 1] = 1.0
    D[np.isnan(D)] = 1.0

    q_anti = q[anti]
    pos_anti = [anti[i] for i in np.flatnonzero(q_anti >= 0)]
    neg_anti = [anti[i] for i in np.flatnonzero(q_anti < 0)]
    for k in range(1, CHARGE_ITERATIONS + 1):
        boost = 1.0 + 100.0 * np.exp(-k / 2.0)
        net = np.einsum("sze,e->sz", D, q)
        if pos_anti:
            idx = net < 0
            D[:, :, pos_anti] = np.where(idx[:, :, None], D[:, :, pos_anti] * boost, D[:, :, pos_anti])
        if neg_anti:
            idx = net > 0
            D[:, :, neg_anti] = np.where(idx[:, :, None], D[:, :, neg_anti] * boost, D[:, :, neg_anti])
        total = D.sum(axis=2)[:, :, None]
        D[:, :, anti] = D[:, :, anti] / total
    return D


def support_vector_from_profile(
    d: np.ndarray, depth: np.ndarray, subst_index: Sequence[int], linked_index: Sequence[int]
) -> np.ndarray:
    """D から SupportVector_Set を作る（`SimulationUpdater.computeSimulationObservables`）。

        p = sum(D .* depth) ./ sum(D)                  重心
        h = 100 * max(D)
        w = 2*sqrt(log(2)/pi) * trapz(depth, 100*D) ./ h

    並びは [p,w,h] を subst について、続けて linked について。
    """
    d = np.asarray(d, dtype=float)
    z = np.asarray(depth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (d * z[:, None]).sum(axis=0) / d.sum(axis=0)
        h = 100.0 * d.max(axis=0)
        w = 2.0 * np.sqrt(np.log(2.0) / np.pi) * np.trapezoid(100.0 * d, z, axis=0) / h
    out = []
    for e in list(subst_index) + list(linked_index):
        out.extend([p[e], w[e], h[e]])
    return np.asarray(out, dtype=float)


def depth_profiles_sequential(fv: np.ndarray, spec: ProfileSpec) -> np.ndarray:
    """stack 順に D を作る。**SupportVector は前 stack の出力が次の入力になる。**

    これは実測で確かめた挙動で（前 stack の SV を使う方が一致がよい）、
    その結果 **生成は stack 順に依存する**。stack を並べ替えたり並列に飛ばしたりすると
    別の値になる。`spec.support_vector` は stack 1 の入力（初期値）として使う。

    2026-09-09: `chunk` 引数を外した（宣言されていたが本体で使われていなかった）。

    ただし **この再帰は stack を重ねると初期値を忘れる**（初期 SV を振っても、後の stack
    では差が消える）。保存済みの D から SV を復元すれば frame 単位の並列再生成ができ、
    逐次結果と float32 保存の精度の範囲で一致する。**規則は逐次だが、再現は checkpoint
    並列可**。確かめたのは一部の場面だけで、一般の場面については確かめていない。
    """
    fv = np.atleast_2d(np.asarray(fv, dtype=float))
    ns = fv.shape[0]
    out = np.empty((ns, spec.depth.size, spec.charge.size), dtype=float)
    sv = np.asarray(spec.support_vector, dtype=float).ravel()
    for s in range(ns):
        step = ProfileSpec(depth=spec.depth, charge=spec.charge, subst_index=spec.subst_index,
                           free_index=spec.free_index, const_index=spec.const_index,
                           linked_index=spec.linked_index, support_vector=sv,
                           distribution=spec.distribution)
        d = depth_profiles(fv[s:s + 1], step)[0]
        out[s] = d
        sv = support_vector_from_profile(d, spec.depth, spec.subst_index, spec.linked_index)
    return out
