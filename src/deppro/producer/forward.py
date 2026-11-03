"""D -> X. Reuses the existing kernel `deppro.transfer_matrix` (no fourth implementation of the formula).

MATLAB side (`+deppro/+core/TransformationMatrix.m` and `CalculateDistribution.m`):

    lambda(z,i) = gamma_eff / sum_a D(z,a) / lambda_basic(a,i)
    t1(th,z,i)  = -dz / (lambda(z,i) * cos th)
    T(th,z,i)   = cumprod_z(exp t1) / exp t1        # only the layers above z (not z itself)
    I(th,i)     = sum_z T(th,z,i) * D(z,i)
    [noise on I]
    X(th,i)     = I(th,i) / sum_j I(th,j)

**Mind the axis orientation**: MATLAB's `lambdabasic(a,i)` has a = material, i = emitting element.
`TransferMatrix.calculate_lambda_obs` reads `lambda_basic[e, i]` as e = emitting element,
i = composition, so **it is passed transposed**. The orientation was settled against noise-free generated data (transposing shifts the values).

The SV update uses only D (in `SimulationUpdater.computeSimulationObservables` p/h/w are
all made from Dobs), so **forward is outside the sequential recursion**: it may be run in
one go once D is final.
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
    lambda_basic: np.ndarray   # L[a, i]  (MATLAB orientation)


def _kernel(spec: ForwardSpec) -> TransferMatrix:
    return TransferMatrix(
        depth=np.asarray(spec.depth, dtype=float),
        step_size=float(spec.step_size),
        angles=np.asarray(spec.angles_deg, dtype=float),
        gamma_eff=float(spec.gamma_eff),
    )


def transfer_and_intensity(d: np.ndarray, spec: ForwardSpec):
    """``(T, I, X)`` of one pixel. T is rebuilt **from that pixel's D**.

    The stored `/ezdeprof/TransMatObs` is T(Dopt), not the T of any GT pixel.
    Reusing it leaves an angle-dependent residual.
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

    Pixels are independent, so the result does not depend on how they are split.

    2026-09-09: the `chunk` argument was removed. **It only added an outer loop; the inner
    loop still went one pixel at a time, so it affected neither values nor speed** (noted by the analysis-side review).
    If splitting is needed, slice `profiles` on the caller's side.
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
