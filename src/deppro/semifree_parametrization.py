"""The SemiFree ``x -> D`` map, extracted from the solver instance as pure functions.

Stage -1 extraction. The aim is **to build D from a feasible x without constructing a
`SemiFreeSolver`** (to bound the distance from the GT). **Behavior is unchanged.**
`SemiFreeSolver._build_substrate_profile` / `_build_full_profile` delegate here.

Scope of the extraction (deliberately minimal)
--------------------------
* in: the substrate erf profile, `normalize_free_conc`, the body of charge balance (both
    the numba and the NumPy path), the final clip.
* out: gradient mode, stoichiometric mode, bounds, constraints. These stay on the
    `SemiFreeSolver` side (no duplicate). Bounds and constraints are read on the experiment
    side by capturing the arguments of `scipy.optimize.minimize`.

Known path differences found during extraction (**not fixed here**; fixing them would change behavior)
---------------------------------------------------------------------
* F1: ``numba_kernels.build_full_profile_numba`` has no ``substrate_cutoff``.
    The NumPy fallback has one (``film_scale = where(substrate_sum >= cutoff, 0, film_scale)``).
    So with ``substrate_cutoff`` set, **the two paths return different D**.
    This is not rounding. The default is ``None``, so it does not occur on the normal path.
    This function mirrors the current behavior, so the numba path ignores the cutoff.
* F2: both kernels are ``@jit(..., fastmath=True)``, and erf is numba=``math.erf`` /
    NumPy=``scipy.special.erf``. **Bit-for-bit agreement between the paths is not expected** (compared within a tolerance).

Path selection (fix F6, 2026-09-09)
-----------------------------------
``SemiFreeSolver.__init__`` forces ``use_numba`` to False when ``normalize_free_conc=True``,
and also checks availability with ``use_numba and NUMBA_AVAILABLE``. The same rule is placed at
the extracted entry as ``resolve_use_numba``, so **path selection agrees even for calls that bypass the solver**.
Calls through the solver pass already-coerced values, so the rule is a no-op there and
**behavior does not change**. The availability flag is ``numba_kernels.NUMBA_AVAILABLE`` **as is**
(deriving it from this module's import success would be True even without numba, because of
the fallback decorator on the kernel side, and would disagree with the solver in the opposite direction).

**What this rule actually prevents**: the solver-side comment says
"the gradient and forward assume unnormalized variables", but that concerns the gradient and
forward; ``build_full_profile_numba`` itself does not normalize internally. So for the x→D map
alone, with ``substrate_cutoff=None`` the two paths differ only at **rounding level** (measured ~1e-16).
**Real harm occurs with ``normalize_free_conc=True`` and a non-None ``substrate_cutoff``**, where
D differs because the numba kernel has no cutoff (**the size of the difference depends on the conditions**: over the 54
conditions seed 0-5 × cutoff {0.3, 0.5, 0.7} × t {150, 210, 300}, min 0.0001 / median 0.2187 / max 0.3074,
and 0.2532 under the regression-test conditions). The rule prevents
(1) path inconsistency with the solver and (2) the discrepancy in that combination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.special import erf

try:  # numba is an optional dependency
    # NOTE: `numba_kernels` falls back to an identity decorator without numba, so
    # **this import does not fail**. Availability is known only from the kernel-side flag
    # (this used to set True here, which disagreed with the solver
    # in the opposite direction when numba was absent).
    from .numba_kernels import (
        NUMBA_AVAILABLE as NUMBA_KERNELS_AVAILABLE,
        build_full_profile_numba,
        build_substrate_profile_numba,
    )
except ImportError:  # pragma: no cover - environment dependent
    NUMBA_KERNELS_AVAILABLE = False


__all__ = [
    "SemiFreeGeometry",
    "build_substrate_profile",
    "build_full_profile_core",
    "resolve_use_numba",
    "NUMBA_KERNELS_AVAILABLE",
]


# ---------------------------------------------------------------------------
def resolve_use_numba(normalize_free_conc: bool, use_numba: bool) -> bool:
    """The path-selection rule. Identical to ``SemiFreeSolver.__init__`` (F6).

    The solver side (``__init__``) takes ``use_numba and NUMBA_AVAILABLE`` and then drops to
    ``use_numba = False`` when ``normalize_free_conc=True``. Calling the extracted entry without
    going through the solver would select a different path without this guard, so the rule lives
    here in one place. ``NUMBA_KERNELS_AVAILABLE`` is ``numba_kernels.NUMBA_AVAILABLE``
    itself (deriving it from import success inside this module would always be True because
    of the fallback decorator, and disagree with the solver).

    For what it prevents, see the "Path selection" section of the module docstring.
    With ``substrate_cutoff=None`` it prevents only rounding-level differences; combined with a
    non-None cutoff it prevents a condition-dependent discrepancy (median 0.2187 / max 0.3074 over 54 conditions, 0.2532 under the test conditions).
    """
    if normalize_free_conc:
        return False
    return bool(use_numba) and NUMBA_KERNELS_AVAILABLE


# ---------------------------------------------------------------------------
def build_substrate_profile(
    depth: np.ndarray,
    thickness: float,
    abruptness: float,
    substrate_ratios: np.ndarray,
    *,
    use_numba: bool = False,
) -> np.ndarray:
    """``D_sub(z, e) = ratio_e * 0.5 * (1 + erf((z - thickness) / sigma))``、``sigma = FWHM/(2 sqrt(ln2))``。

    The body of `SemiFreeSolver._build_substrate_profile`, moved as is.
    """
    if use_numba:
        return build_substrate_profile_numba(depth, thickness, abruptness, substrate_ratios)

    # NumPy fallback
    # Convert FWHM to sigma (same as MATLAB InitializePriorModel.m line 2)
    # MATLAB: abrp = handles.abrpmod(1)/(2*sqrt(log(2)));
    sigma = abruptness / (2 * np.sqrt(np.log(2)))
    erf_profile = 0.5 * (1 + erf((depth - thickness) / sigma))
    return np.outer(erf_profile, substrate_ratios)


# ---------------------------------------------------------------------------
def build_full_profile_core(
    free_conc: np.ndarray,
    thickness: float,
    abruptness: float,
    substrate_ratios: np.ndarray,
    *,
    depth: np.ndarray,
    n_elements: int,
    free_idx: Sequence[int],
    sub_idx: Sequence[int],
    linked_idx: Sequence[int],
    free_linked_const: np.ndarray,
    min_conc: float,
    max_conc: float,
    substrate_cutoff: Optional[float] = None,
    normalize_free_conc: bool = False,
    free_idx_array: Optional[np.ndarray] = None,
    sub_idx_array: Optional[np.ndarray] = None,
    linked_idx_array: Optional[np.ndarray] = None,
    use_numba: bool = False,
) -> np.ndarray:
    """``x -> D`` of the charge-balance mode (gradient / stoichiometric not included).

    The relevant part of `SemiFreeSolver._build_full_profile`, moved as is.
    `substrate_cutoff` acts only on the NumPy path (F1).
    """
    use_numba = resolve_use_numba(normalize_free_conc, use_numba)   # F6

    if normalize_free_conc:
        weighted_sum = np.sum(free_conc * free_linked_const, axis=1, keepdims=True)
        weighted_sum = np.maximum(weighted_sum, min_conc)
        free_conc = free_conc / weighted_sum

    if use_numba:
        return build_full_profile_numba(
            free_conc,
            thickness,
            abruptness,
            substrate_ratios,
            depth,
            n_elements,
            free_idx_array if free_idx_array is not None else np.asarray(free_idx, dtype=np.int64),
            sub_idx_array if sub_idx_array is not None else np.asarray(sub_idx, dtype=np.int64),
            linked_idx_array if linked_idx_array is not None else np.asarray(linked_idx, dtype=np.int64),
            free_linked_const,
            min_conc,
            max_conc,
        )

    # NumPy fallback (charge_balance mode)
    n_depth = len(depth)
    D = np.zeros((n_depth, n_elements))

    # 1. Substrate profile
    D_sub = build_substrate_profile(depth, thickness, abruptness, substrate_ratios, use_numba=False)
    D[:, list(sub_idx)] = D_sub

    # 2. Free element concentrations scaled by (1 - substrate_sum)
    substrate_sum = D_sub.sum(axis=1, keepdims=True)
    film_scale = np.maximum(1.0 - substrate_sum, min_conc)
    if substrate_cutoff is not None:
        film_scale = np.where(substrate_sum >= substrate_cutoff, 0.0, film_scale)
    D[:, list(free_idx)] = free_conc * film_scale

    # 3. Linked element concentrations from charge balance
    linked_idx = list(linked_idx)
    if linked_idx:
        D_free = D[:, list(free_idx)]                 # (n_depth, n_free)
        coeff = np.asarray(free_linked_const) - 1     # (n_free,)
        D_linked = (D_free * coeff).sum(axis=1)       # (n_depth,)
        n_linked = len(linked_idx)
        for linked_e in linked_idx:
            D[:, linked_e] = D_linked / n_linked

    return np.clip(D, min_conc, max_conc)


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SemiFreeGeometry:
    """The fixed arguments of `build_full_profile_core` bundled (for use outside the solver).

    `from_solver` is **read-only** and does not modify the solver.
    """

    depth: np.ndarray
    n_elements: int
    free_idx: tuple
    sub_idx: tuple
    linked_idx: tuple
    free_linked_const: np.ndarray
    min_conc: float = 1e-6
    max_conc: float = 1.0
    substrate_cutoff: Optional[float] = None
    normalize_free_conc: bool = False
    use_numba: bool = False

    @classmethod
    def from_solver(cls, solver, *, use_numba: Optional[bool] = None) -> "SemiFreeGeometry":
        return cls(
            depth=np.asarray(solver.depth, dtype=float),
            n_elements=int(solver.n_elements),
            free_idx=tuple(solver.free_idx),
            sub_idx=tuple(solver.sub_idx),
            linked_idx=tuple(solver.linked_idx),
            free_linked_const=np.asarray(solver.free_linked_const, dtype=float),
            min_conc=float(solver.config.min_conc),
            max_conc=float(solver.config.max_conc),
            substrate_cutoff=solver.semi_free_config.substrate_cutoff,
            normalize_free_conc=bool(solver.semi_free_config.normalize_free_conc),
            use_numba=bool(solver.use_numba) if use_numba is None else bool(use_numba),
        )

    @property
    def effective_use_numba(self) -> bool:
        """The path after applying rule F6. Look at this, not at the raw ``use_numba``."""
        return resolve_use_numba(self.normalize_free_conc, self.use_numba)

    @property
    def n_depth(self) -> int:
        return int(self.depth.size)

    @property
    def n_free(self) -> int:
        return len(self.free_idx)

    @property
    def n_sub(self) -> int:
        return len(self.sub_idx)

    def build(self, free_conc, thickness, abruptness, substrate_ratios,
              *, use_numba: Optional[bool] = None) -> np.ndarray:
        return build_full_profile_core(
            np.asarray(free_conc, dtype=float), float(thickness), float(abruptness),
            np.asarray(substrate_ratios, dtype=float),
            depth=self.depth, n_elements=self.n_elements,
            free_idx=self.free_idx, sub_idx=self.sub_idx, linked_idx=self.linked_idx,
            free_linked_const=self.free_linked_const,
            min_conc=self.min_conc, max_conc=self.max_conc,
            substrate_cutoff=self.substrate_cutoff,
            normalize_free_conc=self.normalize_free_conc,
            use_numba=self.use_numba if use_numba is None else use_numba,
        )
