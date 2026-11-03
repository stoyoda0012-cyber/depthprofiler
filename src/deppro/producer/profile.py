"""FeatureVector -> depth profile D.

MATLAB counterpart (`+deppro/+core/CalculateDistribution.m`, the `'Simulation'` branch).
This is a compatibility mode, so **the formula version and the iteration count are part of the identity, and the behavior is not "fixed"**.

In particular, the fixed 30-iteration loop at the end:

    for k = 1:30
        psuni = sum(D.*q,2)<0;  nguni = sum(D.*q,2)>0
        D(psuni, antisubst(q_anti>=0)) *= (1+100*exp(-k/2))
        D(nguni, antisubst(q_anti< 0)) *= (1+100*exp(-k/2))
        D(:,antisubst) /= sum(D,2)            % only the non-substrate columns, divided by the sum including the substrate
    end

The fixed point has sum 1, but convergence is harmonic: after 30 iterations an excess of up to
+4.7% remains at the interface (where the substrate is near 1 and film remains). **Fixing this would
make it a different thing from the existing GT**, so it is not fixed. Better convergence is a separate mode with a separate identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import erf

#: Identity of the compatibility mode. Includes the formula version and the iteration count.
COMPAT_IDENTITY = "matlab-CalculateDistribution/Simulation/Gaussian/iter30"

CHARGE_ITERATIONS = 30


@dataclass(frozen=True)
class ProfileSpec:
    """What is needed to build D (the part that does not depend on the stack)."""

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
    """Return ``D[s, z, e]``. `fv` is (nstack, 2+3*nfree)."""
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

    # x = [FeatureVector, the linked part of SupportVector]  (MATLAB: x = [x y])
    sv = np.asarray(spec.support_vector, dtype=float).ravel()
    subs_height = sv.reshape(-1, 3)[: len(subst), 2] / 100.0     # third row = height
    x = np.concatenate([fv, np.repeat(sv[3 * len(subst):][None, :], ns, axis=0)], axis=1)

    thck = fv[:, 0]
    abrp = _gaussian_sigma(fv[:, 1])

    D = np.zeros((ns, nz, ne), dtype=float)
    # Substrate: the erf rise
    ramp = 0.5 * np.abs(1.0 + erf((z[None, :] - thck[:, None]) / abrp[:, None]))   # (ns, nz)
    for j, e in enumerate(subst):
        D[:, :, e] = subs_height[j] * ramp
    remainder = np.abs(1.0 - D[:, :, subst].sum(axis=2))                            # (ns, nz)

    # Film components: the Gaussian branch (MATLAB is 1-based: x(3k), x(3k+1), x(3k+2))
    for k, e in enumerate(fcl, start=1):
        pos = x[:, 3 * k - 1]
        sig = _gaussian_sigma(x[:, 3 * k])
        hgt = x[:, 3 * k + 1] / 100.0
        with np.errstate(divide="ignore", invalid="ignore"):
            shape = np.exp(-((z[None, :] - pos[:, None]) ** 2) / sig[:, None] ** 2)
        D[:, :, e] = hgt[:, None] * shape * remainder

    # MATLAB: D(D<0)=0; D(D>1)=1; D(isnan(D))=1;  ── NaN becomes 1 (not 0)
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
    """Build SupportVector_Set from D (`SimulationUpdater.computeSimulationObservables`).

        p = sum(D .* depth) ./ sum(D)                  centroid
        h = 100 * max(D)
        w = 2*sqrt(log(2)/pi) * trapz(depth, 100*D) ./ h

    The order is [p,w,h] for each subst, then for each linked.
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
    """Build D in stack order. **The SupportVector output of one stack is the input of the next.**

    This behavior was confirmed by measurement (using the previous stack's SV agrees better),
    and as a result **generation depends on the stack order**: reordering stacks or running them
    in parallel gives different values. `spec.support_vector` is the input (initial value) of stack 1.

    2026-09-09: the `chunk` argument was removed (declared but unused in the body).

    However, **this recursion forgets its initial value as stacks accumulate** (perturbing the initial
    SV makes no difference in later stacks). Restoring the SV from a saved D allows frame-wise parallel
    regeneration that agrees with the sequential result within float32 storage precision. **The rule is
    sequential, but reproduction can be checkpoint-parallel.** Verified only on some scenes, not in general.
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
