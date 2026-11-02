"""D -> X。既存カーネル `deppro.transfer_matrix` を再利用する（4つ目の数式実装を作らない）。

MATLAB 側（`+deppro/+core/TransformationMatrix.m` と `CalculateDistribution.m`）:

    lambda(z,i) = gamma_eff / sum_a D(z,a) / lambda_basic(a,i)
    t1(th,z,i)  = -dz / (lambda(z,i) * cos th)
    T(th,z,i)   = cumprod_z(exp t1) / exp t1        # z より上の層だけ（z 自身を含まない）
    I(th,i)     = sum_z T(th,z,i) * D(z,i)
    [noise on I]
    X(th,i)     = I(th,i) / sum_j I(th,j)

**軸の向きに注意**: MATLAB の `lambdabasic(a,i)` は a=物質・i=放出元素。
`TransferMatrix.calculate_lambda_obs` は `lambda_basic[e, i]` を e=放出元素・i=組成の
順で読むので、**転置して渡す**。この向きはノイズゼロの生成データとの照合で決めた（転置すると値がずれる）。

SV の更新は D だけを使うので（`SimulationUpdater.computeSimulationObservables` で
p/h/w はすべて Dobs から作られる）、**forward は逐次再帰の外**にある。D が確定したあとで
まとめて回してよい。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deppro.transfer_matrix import TransferMatrix


@dataclass(frozen=True)
class ForwardSpec:
    depth: np.ndarray          # (nz,) Å
    angles_deg: np.ndarray     # (nth,)
    step_size: float           # Å
    gamma_eff: float
    lambda_basic: np.ndarray   # L[a, i]  (MATLAB の向き)


def _kernel(spec: ForwardSpec) -> TransferMatrix:
    return TransferMatrix(
        depth=np.asarray(spec.depth, dtype=float),
        step_size=float(spec.step_size),
        angles=np.asarray(spec.angles_deg, dtype=float),
        gamma_eff=float(spec.gamma_eff),
    )


def transfer_and_intensity(d: np.ndarray, spec: ForwardSpec):
    """1 画素ぶんの ``(T, I, X)``。T は **その画素の D から**作り直す。

    保存されている `/ezdeprof/TransMatObs` は T(Dopt) であって GT のどの画素の
    T でもない。使い回すと角度依存の残差が出る。
    """
    d = np.asarray(d, dtype=float)
    tm = _kernel(spec)
    lam = tm.calculate_lambda_obs(d, np.asarray(spec.lambda_basic, dtype=float).T)
    t = tm.calculate_trans_mat(lam)
    i = np.einsum("tze,ze->te", t, d)
    x = i / i.sum(axis=1, keepdims=True)
    return t, i, x


def observations(profiles: np.ndarray, spec: ForwardSpec):
    """``D[s,z,e]`` -> ``(I[s,th,e], X[s,th,e])``。

    画素どうしは独立なので、どう分割しても結果は変わらない。

    2026-09-09: `chunk` 引数を外した。**外側のループを 1 本増やしていただけで、内側は
    結局 1 画素ずつ回しており、値にも速度にも効いていなかった**（解析検証側の指摘）。
    分割が要るなら呼び出し側で `profiles` を切って渡す。
    """
    profiles = np.asarray(profiles, dtype=float)
    ns, nz, ne = profiles.shape
    nth = np.asarray(spec.angles_deg).size
    tm = _kernel(spec)
    lb_t = np.asarray(spec.lambda_basic, dtype=float).T
    inten = np.empty((ns, nth, ne), dtype=float)
    frac = np.empty((ns, nth, ne), dtype=float)
    for s in range(ns):
        d = profiles[s]
        lam = tm.calculate_lambda_obs(d, lb_t)
        t = tm.calculate_trans_mat(lam)
        inten[s] = np.einsum("tze,ze->te", t, d)
        frac[s] = inten[s] / inten[s].sum(axis=1, keepdims=True)
    return inten, frac
