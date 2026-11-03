"""Shared scene-generation kernel (shipped).

Moved from `benchmarks/generate_checkered.py` **into the shipped package**.
`deppro/simulate/d2s2d.py` imported these at module level while `pyproject.toml`
excluded `deppro.benchmarks*` from the distribution, so **in an installed wheel `deppro`
could not be imported at all** (even `import deppro.imfp` failed). That only held together
in a development checkout, so the dependency direction is fixed to benchmarks -> simulate.

The code was only moved; neither formulas nor defaults changed. `generate_checkered`
re-exports from here, so the existing import paths keep working.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .._compat import trapezoid

__all__ = [
    "calculate_distribution",
    "erf_safe",
    "calculate_distribution_gaussian",
    "distrib_bottom_layer_linear",
    "distrib_bottom_layer_exponential",
    "distrib_bottom_layer_trigonometric",
    "distrib_top_layer_checkered",
    "phenom_interf_redox_linear",
    "phenom_interf_redox_exponential",
    "phenom_interf_redox_trigonometric",
    "phenom_etch_grow_linear",
    "phenom_etch_grow_exponential",
    "phenom_etch_grow_trigonometric",
    "phenom_int_dif_sep_linear",
    "phenom_int_dif_sep_exponential",
    "phenom_int_dif_sep_trigonometric",
    "update_feature_vector",
    "fv_to_profiles",
    "STANDARD_6ELEM",
    "SimulationConfig",
    "PRESET_6ELEM_STANDARD",
    "CHECKERED_7ELEM",
    "PRESET_7ELEM_CHECKERED",
]


def calculate_distribution(
    depth: np.ndarray,
    thickness: float,
    abruptness_fwhm: float,
    free_profiles: list[dict],
    linked_profiles: list[dict],
    charges: np.ndarray,
    subst_idx: list[int],
    free_idx: list[int],
    linked_idx: list[int],
) -> np.ndarray:
    """Generate depth profile D(z) using Error function model.

    Exact replication of MATLAB CalculateDistribution.m 'Error fcn.' case.

    In MATLAB, freeconstlinked = sort([free, const, linked]) and all of them
    get Error function profiles. The profiles are ordered by element index,
    and the x-array is [FeatureVector_Source, linked_params_from_SupportVector].

    The neutral (charge=0) elements are computed first using film_space = 1 - substrate.
    Then charged elements use film_space = 1 - substrate - sum(neutral).
    Direction: k < len(freeconstlinked)/2 + 1 → rising erf, else → falling erf.
    Finally, charge balancing iteration normalizes D to sum=1.

    Args:
        depth: Depth array [n_depth] in Å
        thickness: Substrate interface position (Å)
        abruptness_fwhm: Interface FWHM (Å)
        free_profiles: List of {pos, width, height} for each free element
        linked_profiles: List of {pos, width, height} for each linked element
        charges: Oxidation states for ALL elements [n_elements]
        subst_idx: 0-based substrate element indices
        free_idx: 0-based free element indices
        linked_idx: 0-based linked element indices

    Returns:
        D: Depth profile [n_depth, n_elements]
    """
    n_depth = len(depth)
    n_elements = len(charges)
    D = np.zeros((n_depth, n_elements))

    # Convert FWHM to Gaussian sigma: σ = FWHM / (2√(ln2))
    abrp_sigma = abruptness_fwhm / (2.0 * np.sqrt(np.log(2.0)))

    # Substrate: erf rising from 0 (surface) to 1 (bulk)
    subs_frac = 1.0  # Single substrate = 100%
    for s in subst_idx:
        D[:, s] = subs_frac * 0.5 * np.abs(
            1.0 + erf_safe((depth - thickness) / abrp_sigma)
        )

    # Build freeconstlinked order (sorted by element index, like MATLAB)
    # Each entry: (element_index, profile_dict)
    fcl_entries = []
    for i, fi in enumerate(free_idx):
        fcl_entries.append((fi, free_profiles[i]))
    for i, li in enumerate(linked_idx):
        fcl_entries.append((li, linked_profiles[i]))
    fcl_entries.sort(key=lambda x: x[0])

    n_fcl = len(fcl_entries)
    fcl_charges = np.array([charges[e[0]] for e in fcl_entries])

    # Separate neutral (p) and charged (q) by index in fcl
    p_indices = [k for k in range(n_fcl) if fcl_charges[k] == 0]
    q_indices = [k for k in range(n_fcl) if fcl_charges[k] != 0]

    D0 = np.zeros((n_depth, n_fcl))

    # Neutral elements first (MATLAB: for k = p)
    for k in p_indices:
        _, prof = fcl_entries[k]
        sigma_k = prof["width"] / (2.0 * np.sqrt(np.log(2.0)))
        height_frac = prof["height"] / 100.0
        film_space = np.abs(1.0 - np.sum(D[:, subst_idx], axis=1))
        D0[:, k] = height_frac * 0.5 * np.abs(
            1.0 + erf_safe(-(depth - prof["pos"]) / sigma_k)
        ) * film_space

    # Charged elements (MATLAB: for k = q)
    for k in q_indices:
        _, prof = fcl_entries[k]
        sigma_k = prof["width"] / (2.0 * np.sqrt(np.log(2.0)))
        height_frac = prof["height"] / 100.0
        film_space = np.abs(
            1.0 - np.sum(D[:, subst_idx], axis=1)
            - np.sum(D0[:, p_indices], axis=1)
        )
        # MATLAB: k < length(freeconstlinked)/2 + 1 → rising erf
        # k is 1-based in MATLAB, so threshold = n_fcl/2 + 1
        # In 0-based: k+1 < n_fcl/2 + 1 → k < n_fcl/2
        if (k + 1) < n_fcl / 2.0 + 1:
            # Rising erf (deeper layer, positive charge cations)
            D0[:, k] = height_frac * 0.5 * np.abs(
                1.0 + erf_safe((depth - prof["pos"]) / sigma_k)
            ) * film_space
        else:
            # Falling erf (anion / surface-side)
            D0[:, k] = height_frac * 0.5 * np.abs(
                1.0 + erf_safe(-(depth - prof["pos"]) / sigma_k)
            ) * film_space

    # Assign D0 to D
    for k in range(n_fcl):
        elem_idx = fcl_entries[k][0]
        D[:, elem_idx] = D0[:, k]

    # Clip
    D = np.clip(D, 0.0, 1.0)

    # Charge balancing iteration (MATLAB lines 99-105)
    antisubst = sorted(free_idx + linked_idx)
    antisubst_charges = charges[antisubst]
    pos_mask = antisubst_charges >= 0
    neg_mask = antisubst_charges < 0

    for iteration in range(1, 31):  # MATLAB: k=1:30
        charge_sum = np.sum(D * charges, axis=1)  # (n_depth,)
        psuni = charge_sum < 0  # Need more positive
        nguni = charge_sum > 0  # Need more negative

        factor = 1.0 + 100.0 * np.exp(-iteration / 2.0)

        for i, ai in enumerate(antisubst):
            if psuni.any() and pos_mask[i]:
                D[psuni, ai] *= factor
            if nguni.any() and neg_mask[i]:
                D[nguni, ai] *= factor

        # Renormalize antisubst columns so total D sums to 1
        row_sums = np.sum(D, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        for ai in antisubst:
            D[:, ai] /= row_sums.squeeze()

    D = np.clip(D, 0.0, 1.0)

    return D


def erf_safe(x: np.ndarray) -> np.ndarray:
    """Error function with safe handling."""
    from scipy.special import erf
    return erf(x)


def calculate_distribution_gaussian(
    depth: np.ndarray,
    thickness: float,
    abruptness_fwhm: float,
    free_profiles: list[dict],
    linked_profiles: list[dict],
    charges: np.ndarray,
    subst_idx: list[int],
    free_idx: list[int],
    linked_idx: list[int],
) -> np.ndarray:
    """Generate depth profile D(z) using Gaussian model.

    Exact replication of MATLAB CalculateDistribution.m 'Gaussian' case.

    Each free/linked element gets a Gaussian peak:
        D0(k) = (height/100) * exp(-(z - pos)^2 / S^2) * |1 - substrate|
    where S = FWHM / (2*sqrt(ln2)).

    All elements use the same film_space = 1 - substrate (no neutral/charged
    distinction, unlike the 'Error fcn.' mode).

    Args:
        depth: Depth array [n_depth] in Å
        thickness: Substrate interface position (Å)
        abruptness_fwhm: Interface FWHM (Å)
        free_profiles: List of {pos, width, height} for each free element
        linked_profiles: List of {pos, width, height} for each linked element
        charges: Oxidation states for ALL elements [n_elements]
        subst_idx: 0-based substrate element indices
        free_idx: 0-based free element indices
        linked_idx: 0-based linked element indices

    Returns:
        D: Depth profile [n_depth, n_elements]
    """
    n_depth = len(depth)
    n_elements = len(charges)
    D = np.zeros((n_depth, n_elements))

    # Convert FWHM to Gaussian sigma: S = FWHM / (2√(ln2))
    abrp_sigma = abruptness_fwhm / (2.0 * np.sqrt(np.log(2.0)))

    # Substrate: erf rising from 0 (surface) to 1 (bulk)
    subs_frac = 1.0
    for s in subst_idx:
        D[:, s] = subs_frac * 0.5 * np.abs(
            1.0 + erf_safe((depth - thickness) / abrp_sigma)
        )

    # Film space: 1 - substrate (same for ALL elements in Gaussian mode)
    film_space = np.abs(1.0 - np.sum(D[:, subst_idx], axis=1))

    # Build freeconstlinked order (sorted by element index)
    fcl_entries = []
    for i, fi in enumerate(free_idx):
        fcl_entries.append((fi, free_profiles[i]))
    for i, li in enumerate(linked_idx):
        fcl_entries.append((li, linked_profiles[i]))
    fcl_entries.sort(key=lambda x: x[0])

    n_fcl = len(fcl_entries)
    D0 = np.zeros((n_depth, n_fcl))

    # Gaussian peak for each element
    for k in range(n_fcl):
        _, prof = fcl_entries[k]
        S_k = prof["width"] / (2.0 * np.sqrt(np.log(2.0)))
        height_frac = prof["height"] / 100.0
        D0[:, k] = height_frac * np.exp(
            -(depth - prof["pos"]) ** 2 / S_k ** 2
        ) * film_space

    # Assign D0 to D
    for k in range(n_fcl):
        elem_idx = fcl_entries[k][0]
        D[:, elem_idx] = D0[:, k]

    # Clip
    D = np.clip(D, 0.0, 1.0)
    D[np.isnan(D)] = 1.0

    # Charge balancing iteration (same as Error fcn. mode)
    antisubst = sorted(free_idx + linked_idx)
    antisubst_charges = charges[antisubst]
    pos_mask = antisubst_charges >= 0
    neg_mask = antisubst_charges < 0

    for iteration in range(1, 31):
        charge_sum = np.sum(D * charges, axis=1)
        psuni = charge_sum < 0
        nguni = charge_sum > 0

        factor = 1.0 + 100.0 * np.exp(-iteration / 2.0)

        for i, ai in enumerate(antisubst):
            if psuni.any() and pos_mask[i]:
                D[psuni, ai] *= factor
            if nguni.any() and neg_mask[i]:
                D[nguni, ai] *= factor

        row_sums = np.sum(D, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        for ai in antisubst:
            D[:, ai] /= row_sums.squeeze()

    D = np.clip(D, 0.0, 1.0)

    return D


def distrib_bottom_layer_linear(FV, sk, dx, dy):
    """MATLAB DistribBottomLayerLinear.m — 4-quadrant linear spatial modulation."""
    FV = FV.copy()
    fvid1 = [0, 2]  # thck, pos_SiO2
    fvid2 = 3        # w_SiO2
    scale_x = dx / 6.0
    scale_y = dy / 6.0
    id1 = not (int(np.floor((sk - 1) / scale_x)) % 2)
    id2 = not (int(np.floor(np.floor((sk - 1) / dy) / scale_y)) % 2)

    # Local x and y within quadrant
    x_local = (sk - 1) - np.floor((sk - 1) / scale_x) * scale_x
    y_local = np.floor((sk - 1) / dy) - np.floor(np.floor((sk - 1) / dy) / scale_y) * scale_y

    if id1 and id2:
        for idx in fvid1:
            FV[idx] += x_local * (25.0 / scale_x) + y_local * (25.0 / scale_y)
        FV[fvid2] += 1.5 * x_local * (25.0 / scale_x) + 1.5 * y_local * (25.0 / scale_y)
    elif id1 and not id2:
        for idx in fvid1:
            FV[idx] += x_local * (25.0 / scale_x) + 25.0 + (-y_local) * (25.0 / scale_x)
        FV[fvid2] += 1.5 * x_local * (25.0 / scale_x) + 37.5 + 1.5 * (-y_local) * (25.0 / scale_x)
    elif not id1 and id2:
        for idx in fvid1:
            FV[idx] += 25.0 + (-x_local) * (25.0 / scale_x) + y_local * (25.0 / scale_y)
        FV[fvid2] += 37.5 + 1.5 * (-x_local) * (25.0 / scale_x) + 1.5 * y_local * (25.0 / scale_y)
    else:  # not id1 and not id2
        for idx in fvid1:
            FV[idx] += 25.0 + (-x_local) * (25.0 / scale_x) + 25.0 + (-y_local) * (25.0 / scale_x)
        FV[fvid2] += 37.5 + 1.5 * (-x_local) * (25.0 / scale_x) + 37.5 + 1.5 * (-y_local) * (25.0 / scale_x)
    return FV


def distrib_bottom_layer_exponential(FV, sk, dx, dy):
    """MATLAB DistribBottomLayerExponential.m — 4-quadrant exponential spatial modulation."""
    FV = FV.copy()
    fvid1 = [0, 2]
    fvid2 = 3
    scale_x = dx / 6.0
    scale_y = dy / 6.0
    tau = dx / (6.0 * 5.0)
    id1 = not (int(np.floor((sk - 1) / scale_x)) % 2)
    id2 = not (int(np.floor(np.floor((sk - 1) / dy) / scale_y)) % 2)

    x_local = -sk + 1 + np.floor((sk - 1) / scale_x) * scale_x
    y_local = -np.floor((sk - 1) / dy) + np.floor(np.floor((sk - 1) / dy) / scale_y) * scale_y

    exp_x_rise = 1.0 - np.exp(x_local / tau)    # (1-exp(-x/tau)) with x_local negative
    exp_x_fall = np.exp(x_local / tau)
    exp_y_rise = 1.0 - np.exp(y_local / tau)
    exp_y_fall = np.exp(y_local / tau)

    if id1 and id2:
        for idx in fvid1:
            FV[idx] += 25.0 * exp_x_rise + 25.0 * exp_y_rise
        FV[fvid2] += 37.5 * exp_x_rise + 37.5 * exp_y_rise
    elif id1 and not id2:
        for idx in fvid1:
            FV[idx] += 25.0 * exp_x_rise + 25.0 * exp_y_rise
        FV[fvid2] += 37.5 * exp_x_rise + 37.5 * exp_y_rise
    elif not id1 and id2:
        for idx in fvid1:
            FV[idx] += 25.0 * exp_x_fall + 25.0 * exp_y_rise
        FV[fvid2] += 37.5 * exp_x_fall + 37.5 * exp_y_rise
    else:  # not id1 and not id2
        for idx in fvid1:
            FV[idx] += 25.0 * exp_x_fall + 25.0 * exp_y_fall
        FV[fvid2] += 37.5 * exp_x_fall + 37.5 * exp_y_fall
    return FV


def distrib_bottom_layer_trigonometric(FV, sk, dx, dy):
    """MATLAB DistribBottomLayerTrigonometric.m — cosine spatial modulation."""
    FV = FV.copy()
    fvid1 = [0, 2]
    fvid2 = 3
    mod_val = (2.0
               - np.cos(2.0 * np.pi * (sk - 1) / (dx / 3.0))
               - np.cos(2.0 * np.pi * np.floor((sk - 1) / dy) / (dy / 3.0))) / 4.0
    for idx in fvid1:
        FV[idx] += 50.0 * mod_val
    FV[fvid2] += 75.0 * mod_val
    return FV


def distrib_top_layer_checkered(FV, sk, dx, dy):
    """MATLAB DistribTopLayerCheckered.m — FV-based checkered pattern.

    Swaps the height values of the last two free elements.
    fvid1 = 3*(SizeofFree-1)+2 (second-to-last free element height)
    fvid2 = 3*SizeofFree+2      (last free element height, only for >4 free elements)
    """
    FV = FV.copy()
    n_free = (len(FV) - 2) // 3
    # MATLAB: fvid1 = 3*(SizeofFree-1)+2, fvid2 = 3*SizeofFree+2 (1-based)
    # Python 0-based: fvid1 = 3*(n_free-1)+1+2-1 = 3*(n_free-1)+2, but:
    # MATLAB col index 3*(SizeofFree-1)+2: for SizeofFree=4 → col 11 (1-based) → index 10
    # height of element k is at FV[2 + 3*k + 2] = FV[3*k+4] (0-based) where k=0..n_free-1
    # Actually: FV = [thck(0), abrp(1), pos0(2), w0(3), h0(4), pos1(5), w1(6), h1(7), ...]
    # height of free element k: index 4 + 3*k (for k=0), or 2 + 3*(k+1) - 1 = 3*k + 4
    # height of second-to-last free elem: k=n_free-2 → index 3*(n_free-2) + 4
    # height of last free elem: k=n_free-1 → index 3*(n_free-1) + 4
    idx_h1 = 3 * (n_free - 2) + 4  # second-to-last free elem height
    idx_h2 = 3 * (n_free - 1) + 4  # last free elem height

    id1 = not (int(np.ceil(6.0 * sk / dx)) % 2)
    id2 = not (int(np.ceil(6.0 * sk / (dx * dy))) % 2)

    if (id1 and id2) or (not id1 and not id2):
        FV[idx_h1] = 0.0
        FV[idx_h2] = 100.0
    else:
        FV[idx_h1] = 100.0
        FV[idx_h2] = 0.0
    return FV


def _phenom_phase_progress(sk, dxdy, dxdydt):
    """Common phase calculation for temporal phenomena (Linear/Exp)."""
    phase_block = int(np.floor((sk - 1) / dxdydt))
    is_forward = not (phase_block % 2)
    sk_local = sk - 1 - phase_block * dxdydt
    t_step = int(np.floor(sk_local / dxdy))
    return is_forward, t_step, sk_local, phase_block


def phenom_interf_redox_linear(FV, sk, dxdy, dxdydt):
    """MATLAB PhenomInterfRedoxLinear.m — linear interface position shift."""
    FV = FV.copy()
    fvid1 = [0, 2]  # thck, pos_SiO2
    fvid2 = 3        # w_SiO2
    is_forward, t_step, sk_local, phase_block = _phenom_phase_progress(sk, dxdy, dxdydt)

    if is_forward:
        for idx in fvid1:
            FV[idx] += t_step
        FV[fvid2] += 1.5 * t_step
    else:
        back_t = int(np.ceil((-sk + 1 + phase_block * dxdydt) / dxdy))
        for idx in fvid1:
            FV[idx] += 100 + back_t
        FV[fvid2] += 150 + 1.5 * back_t
    return FV


def phenom_interf_redox_exponential(FV, sk, dxdy, dxdydt):
    """MATLAB PhenomInterfRedoxExponential.m — exponential interface penetration."""
    FV = FV.copy()
    fvid1 = [0, 2]
    fvid2 = 3
    is_forward, t_step, sk_local, phase_block = _phenom_phase_progress(sk, dxdy, dxdydt)
    tau = dxdydt / 5.0

    if is_forward:
        prog = 1.0 - np.exp(-t_step * dxdy / tau)
        for idx in fvid1:
            FV[idx] += 100.0 * prog
        FV[fvid2] += 150.0 * prog
    else:
        # MATLAB: ceil((-sk+1+phase_block*dxdydt)/dxdy)*dxdy
        progress = sk - 1 - phase_block * dxdydt
        back_quant = int(np.ceil(-progress / dxdy)) * dxdy
        prog = np.exp(back_quant / tau)
        for idx in fvid1:
            FV[idx] += 100.0 * prog
        FV[fvid2] += 150.0 * prog
    return FV


def phenom_interf_redox_trigonometric(FV, sk, dxdy, dt):
    """MATLAB PhenomInterfRedoxTrigonometric.m — cosine interface modulation."""
    FV = FV.copy()
    fvid1 = [0, 2]
    fvid2 = 3
    t_step = int(np.floor((sk - 1) / dxdy))
    mod_val = (1.0 - np.cos(np.pi * t_step / dt)) / 2.0
    for idx in fvid1:
        FV[idx] += 100.0 * mod_val
    FV[fvid2] += 150.0 * mod_val
    return FV


def _etchgrow_fvid1(n_fv):
    """Dynamic fvid1 for EtchGrow: thck + all free element positions."""
    n_free = (n_fv - 2) // 3
    # MATLAB: [1, 3*(1:SizeofFree)]  (1-based)
    # = [1, 3, 6, 9, ...] for SizeofFree free elements
    # Python 0-based: [0, 2, 5, 8, ...]
    # Position of free element k: index 2 + 3*k  (k=0..n_free-1)
    return [0] + [2 + 3 * k for k in range(n_free)]


def phenom_etch_grow_linear(FV, sk, dxdy, dxdydt):
    """MATLAB PhenomEtchGrowLinear.m — linear film thickness change."""
    FV = FV.copy()
    fvid1 = _etchgrow_fvid1(len(FV))
    is_forward, t_step, sk_local, phase_block = _phenom_phase_progress(sk, dxdy, dxdydt)

    if is_forward:
        shift = -int(np.floor(sk_local / (2 * dxdy)))
        for idx in fvid1:
            FV[idx] += shift
    else:
        back_val = int(np.ceil((-sk + 1 + phase_block * dxdydt) / (2 * dxdy)))
        for idx in fvid1:
            FV[idx] += -50 - back_val
    return FV


def phenom_etch_grow_exponential(FV, sk, dxdy, dxdydt):
    """MATLAB PhenomEtchGrowExponential.m — exponential thickness change."""
    FV = FV.copy()
    fvid1 = _etchgrow_fvid1(len(FV))
    is_forward, t_step, sk_local, phase_block = _phenom_phase_progress(sk, dxdy, dxdydt)
    tau = dxdydt / 5.0

    if is_forward:
        prog = 1.0 - np.exp(-t_step * dxdy / tau)
        for idx in fvid1:
            FV[idx] -= 50.0 * prog
    else:
        back_local = -sk + 1 + phase_block * dxdydt
        prog = np.exp(back_local / tau)
        for idx in fvid1:
            FV[idx] -= 50.0 * prog
    return FV


def phenom_etch_grow_trigonometric(FV, sk, dxdy, dt):
    """MATLAB PhenomEtchGrowTrigonometric.m — cosine thickness modulation."""
    FV = FV.copy()
    fvid1 = _etchgrow_fvid1(len(FV))
    t_step = int(np.floor((sk - 1) / dxdy))
    mod_val = (1.0 - np.cos(np.pi * t_step / dt)) / 2.0
    for idx in fvid1:
        FV[idx] -= 50.0 * mod_val
    return FV


def _intdifsep_indices(n_fv):
    """Dynamic fvid1/fvid2 for IntDifSep, depends on SizeofFree."""
    n_free = (n_fv - 2) // 3
    # MATLAB 1-based column indices; Python 0-based conversion below.
    # FV position of free element k: col_pos(k) = 2 + 3*k (0-based)
    # MATLAB fvid1/fvid2 refer to columns in the full FV array.
    #
    # MATLAB: if SizeofFree <= 4:
    #   fvid1 = 3*(2:SizeofFree-2)  → for n_free=4: 3*[2] = [6]  (1-based col)
    #   fvid2 = 3*(SizeofFree-1):3*(SizeofFree-1)+1  → 3*3 : 3*3+1 = [9,10] (1-based)
    # if SizeofFree > 4:
    #   fvid1 = 3*(2:SizeofFree-3)  → for n_free=5: 3*[2,3] = [6,9]
    #   fvid2 = 3*(SizeofFree-2):3*(SizeofFree-2)+1  → 3*3 : 3*3+1 = [9,10]
    #
    # Convert MATLAB 1-based to Python 0-based: subtract 1
    if n_free <= 4:
        fvid1_m = [3 * k for k in range(2, n_free - 1)]
        last_m = 3 * (n_free - 1)
        fvid2_m = [last_m, last_m + 1]
    else:
        fvid1_m = [3 * k for k in range(2, n_free - 2)]
        last_m = 3 * (n_free - 2)
        fvid2_m = [last_m, last_m + 1]
    # Convert to 0-based
    fvid1 = [m - 1 for m in fvid1_m]
    fvid2 = [m - 1 for m in fvid2_m]
    return fvid1, fvid2


def phenom_int_dif_sep_linear(FV, sk, dxdy, dxdydt):
    """MATLAB PhenomIntDifSepLinear.m — linear diffusion/separation."""
    FV = FV.copy()
    fvid1, fvid2 = _intdifsep_indices(len(FV))
    is_forward, t_step, sk_local, phase_block = _phenom_phase_progress(sk, dxdy, dxdydt)

    if is_forward:
        shift = 0.8 * int(np.floor(sk_local / (2 * dxdy)))
        for idx in fvid1:
            FV[idx] -= shift
        for idx in fvid2:
            FV[idx] += shift
    else:
        back_val = 0.8 * int(np.ceil((-sk + 1 + phase_block * dxdydt) / (2 * dxdy)))
        # MATLAB: fvid1 = FV0 - 40 - 0.8*ceil(...), fvid2 = FV0 + 40 + 0.8*ceil(...)
        for idx in fvid1:
            FV[idx] -= 40 + back_val
        for idx in fvid2:
            FV[idx] += 40 + back_val
    return FV


def phenom_int_dif_sep_exponential(FV, sk, dxdy, dxdydt):
    """MATLAB PhenomIntDifSepExponential.m — exponential diffusion."""
    FV = FV.copy()
    fvid1, fvid2 = _intdifsep_indices(len(FV))
    is_forward, t_step, sk_local, phase_block = _phenom_phase_progress(sk, dxdy, dxdydt)
    tau = dxdydt / 5.0

    if is_forward:
        prog = 1.0 - np.exp(-t_step * dxdy / tau)
        for idx in fvid1:
            FV[idx] -= 40.0 * prog
        for idx in fvid2:
            FV[idx] += 40.0 * prog
    else:
        back_local = -sk + 1 + phase_block * dxdydt
        prog = np.exp(back_local / tau)
        for idx in fvid1:
            FV[idx] -= 40.0 * prog
        for idx in fvid2:
            FV[idx] += 40.0 * prog
    return FV


def phenom_int_dif_sep_trigonometric(FV, sk, dxdy, dt):
    """MATLAB PhenomIntDifSepTrigonometric.m — cosine diffusion modulation."""
    FV = FV.copy()
    fvid1, fvid2 = _intdifsep_indices(len(FV))
    t_step = int(np.floor((sk - 1) / dxdy))
    mod_val = (1.0 - np.cos(np.pi * t_step / dt)) / 2.0
    for idx in fvid1:
        FV[idx] -= 40.0 * mod_val
    for idx in fvid2:
        FV[idx] += 40.0 * mod_val
    return FV


def update_feature_vector(FV, sk, dx, dy, dt, flags):
    """MATLAB SimulationUpdater.updateFeatureVector — full port.

    Applies spatial distribution + temporal phenomena to FeatureVector.
    Mutually exclusive groups: DistribBottom (3), InterfRedox (3),
    EtchGrow (3), IntDifSep (3).  Checkered can combine with DistribBottom.

    Args:
        FV: FeatureVector array (1D, float64)
        sk: Stack index (1-based, MATLAB convention)
        dx, dy: Spatial grid dimensions
        dt: Time steps per phase
        flags: dict of boolean flags

    Returns:
        Updated FV (copy)
    """
    dxdy = dx * dy
    dxdydt = dx * dy * dt

    # DistribBottom (mutually exclusive)
    if flags.get('DistribBottomLayerLinear'):
        FV = distrib_bottom_layer_linear(FV, sk, dx, dy)
    elif flags.get('DistribBottomLayerExponential'):
        FV = distrib_bottom_layer_exponential(FV, sk, dx, dy)
    elif flags.get('DistribBottomLayerTrigonometric'):
        FV = distrib_bottom_layer_trigonometric(FV, sk, dx, dy)

    # Checkered (can combine with DistribBottom)
    if flags.get('DistribTopLayerCheckered'):
        FV = distrib_top_layer_checkered(FV, sk, dx, dy)

    # InterfRedox (mutually exclusive)
    if flags.get('PhenomInterfRedoxLinear'):
        FV = phenom_interf_redox_linear(FV, sk, dxdy, dxdydt)
    elif flags.get('PhenomInterfRedoxExponential'):
        FV = phenom_interf_redox_exponential(FV, sk, dxdy, dxdydt)
    elif flags.get('PhenomInterfRedoxTrigonometric'):
        FV = phenom_interf_redox_trigonometric(FV, sk, dxdy, dt)

    # EtchGrow (mutually exclusive)
    if flags.get('PhenomEtchGrowLinear'):
        FV = phenom_etch_grow_linear(FV, sk, dxdy, dxdydt)
    elif flags.get('PhenomEtchGrowExponential'):
        FV = phenom_etch_grow_exponential(FV, sk, dxdy, dxdydt)
    elif flags.get('PhenomEtchGrowTrigonometric'):
        FV = phenom_etch_grow_trigonometric(FV, sk, dxdy, dt)

    # IntDifSep (mutually exclusive)
    if flags.get('PhenomIntDifSepLinear'):
        FV = phenom_int_dif_sep_linear(FV, sk, dxdy, dxdydt)
    elif flags.get('PhenomIntDifSepExponential'):
        FV = phenom_int_dif_sep_exponential(FV, sk, dxdy, dxdydt)
    elif flags.get('PhenomIntDifSepTrigonometric'):
        FV = phenom_int_dif_sep_trigonometric(FV, sk, dxdy, dt)

    return FV


def fv_to_profiles(FV, n_free):
    """Convert FV array to list of profile dicts for calculate_distribution.

    Args:
        FV: [thck, abrp, pos0, w0, h0, pos1, w1, h1, ...]
        n_free: Number of free elements

    Returns:
        thickness, abruptness, free_profiles list
    """
    thickness = FV[0]
    abruptness = FV[1]
    free_profiles = []
    for k in range(n_free):
        base = 2 + 3 * k
        free_profiles.append({
            "pos": FV[base],
            "width": FV[base + 1],
            "height": FV[base + 2],
        })
    return thickness, abruptness, free_profiles


# Standard 6-element system: Si(0), Si(+4), Ti(+4), Al(+3), C(0), O(-2)
STANDARD_6ELEM = [
    {"symbol": "Si", "orbital": "1s", "compound": "Si",
     "BE": 1839.1, "charge": 0, "density": 2.329, "Aw": 28.086, "Nv": 4, "Eg": 1.12},
    {"symbol": "Si", "orbital": "1s", "compound": "SiO2",
     "BE": 1843.8, "charge": 4, "density": 2.2, "Aw": 60.08, "Nv": 16, "Eg": 8.9},
    {"symbol": "Ti", "orbital": "1s", "compound": "TiO2",
     "BE": 4968.7, "charge": 4, "density": 4.23, "Aw": 79.866, "Nv": 16, "Eg": 3.1},
    {"symbol": "Al", "orbital": "1s", "compound": "Al2O3",
     "BE": 1561.8, "charge": 3, "density": 3.95, "Aw": 101.96, "Nv": 24, "Eg": 7.6},
    {"symbol": "C",  "orbital": "1s", "compound": "C",
     "BE": 284.4,  "charge": 0, "density": 2.26, "Aw": 12.011, "Nv": 4, "Eg": 0.0},
    {"symbol": "O",  "orbital": "1s", "compound": "Oxide",
     "BE": 531.0,  "charge": -2, "density": 2.2, "Aw": 16.0, "Nv": 6, "Eg": 6.0},
]


@dataclass
class SimulationConfig:
    """Configuration for time-varying simulation with spatial + temporal patterns."""
    # Spatial grid
    nx: int = 5
    ny: int = 5
    dt: int = 100         # Time steps per phase
    n_phases: int = 1     # 1=forward only, 2=forward+backward

    # Spatial correlation parameters (H5 metadata dxdydt[0:2]).
    # Controls modulation rate. None → fallback to nx/ny.
    dx: int = None        # type: ignore[assignment]
    dy: int = None        # type: ignore[assignment]

    # Depth grid
    n_depth: int = 20
    step_size: float = 30.0
    total_depth: float = 570.0

    # Observation angles
    n_angles: int = 14
    angle_min: float = 0.0
    angle_max: float = 60.0

    # Physical parameters
    photon_energy: float = 8000.0
    work_function: float = 5.0
    gamma_eff: float = 0.80

    # Base FeatureVector profiles
    thickness: float = 210.0
    abruptness: float = 20.0

    # Free element base profiles (Error fcn): pos, width, height
    free_profiles: list = field(default_factory=lambda: [
        {"pos": 210.0, "width": 30.0, "height": 33.333},   # Si(+4)
        {"pos": 150.0, "width": 80.0, "height": 33.333},   # Ti(+4)
        {"pos":  80.0, "width": 40.0, "height": 40.0},      # Al(+3)
        {"pos":   0.0, "width": 70.0, "height": 100.0},     # C(0)
    ])

    # Element definitions
    elements: list = field(default_factory=lambda: STANDARD_6ELEM)

    # Index arrays (0-based)
    subst_idx: list = field(default_factory=lambda: [0])
    free_idx: list = field(default_factory=lambda: [1, 2, 3, 4])
    linked_idx: list = field(default_factory=lambda: [5])
    linked_const: list = field(default_factory=lambda: [3.0, 3.0, 2.5, 1.0])

    # Linked element profiles
    linked_profiles: list = field(default_factory=lambda: [
        {"pos": 98.025, "width": 197.279, "height": 28.954},
    ])

    # Noise
    noise_level: float = 0.0

    # Distribution function type: 'Error fcn.' or 'Gaussian'
    dist_type: str = 'Error fcn.'

    # Simulation flags
    flags: dict = field(default_factory=lambda: {
        'DistribBottomLayerTrigonometric': True,
        'PhenomInterfRedoxExponential': True,
        'PhenomEtchGrowLinear': True,
    })

    @property
    def eff_dx(self):
        """Effective dx for modulation formulas (H5 metadata or fallback to nx)."""
        return self.dx if self.dx is not None else self.nx

    @property
    def eff_dy(self):
        """Effective dy for modulation formulas (H5 metadata or fallback to ny)."""
        return self.dy if self.dy is not None else self.ny

    @property
    def n_pixels(self):
        return self.nx * self.ny * self.dt * self.n_phases

    def base_fv(self):
        """Build base FeatureVector from config."""
        fv = [self.thickness, self.abruptness]
        for prof in self.free_profiles:
            fv.extend([prof["pos"], prof["width"], prof["height"]])
        return np.array(fv, dtype=np.float64)


# Preset: 6-element Standard (matches MATLAB 5x5x100 GT)
PRESET_6ELEM_STANDARD = SimulationConfig(
    nx=5, ny=5, dt=100, n_phases=1, dx=1, dy=1,
    elements=STANDARD_6ELEM,
    subst_idx=[0], free_idx=[1, 2, 3, 4], linked_idx=[5],
    linked_const=[3.0, 3.0, 2.5, 1.0],
    free_profiles=[
        {"pos": 210.0, "width": 30.0, "height": 33.333},
        {"pos": 150.0, "width": 80.0, "height": 33.333},
        {"pos":  80.0, "width": 40.0, "height": 40.0},
        {"pos":   0.0, "width": 70.0, "height": 100.0},
    ],
    linked_profiles=[{"pos": 98.025, "width": 197.279, "height": 28.954}],
    dist_type='Gaussian',
    flags={
        'DistribBottomLayerTrigonometric': True,
        'PhenomInterfRedoxExponential': True,
        'PhenomEtchGrowLinear': True,
    },
)


# 7-element Checkered system: Si(0), Si(+4), Ti(+4), Al(+3), C(0), B(0), O(-2)
CHECKERED_7ELEM = [
    {"symbol": "Si", "orbital": "1s", "compound": "Si",
     "BE": 1839.1, "charge": 0, "density": 2.329, "Aw": 28.086, "Nv": 4, "Eg": 1.12},
    {"symbol": "Si", "orbital": "1s", "compound": "SiO2",
     "BE": 1843.8, "charge": 4, "density": 2.2, "Aw": 60.08, "Nv": 16, "Eg": 8.9},
    {"symbol": "Ti", "orbital": "1s", "compound": "TiO2",
     "BE": 4968.7, "charge": 4, "density": 4.23, "Aw": 79.866, "Nv": 16, "Eg": 3.1},
    {"symbol": "Al", "orbital": "1s", "compound": "Al2O3",
     "BE": 1561.8, "charge": 3, "density": 3.95, "Aw": 101.96, "Nv": 24, "Eg": 7.6},
    {"symbol": "C",  "orbital": "1s", "compound": "C",
     "BE": 284.4,  "charge": 0, "density": 2.26, "Aw": 12.011, "Nv": 4, "Eg": 0.0},
    {"symbol": "B",  "orbital": "1s", "compound": "B",
     "BE": 188.0,  "charge": 0, "density": 2.08, "Aw": 10.812, "Nv": 3, "Eg": 0.0},
    {"symbol": "O",  "orbital": "1s", "compound": "Oxide",
     "BE": 531.0,  "charge": -2, "density": 2.2, "Aw": 16.0, "Nv": 6, "Eg": 6.0},
]


# Preset: 7-element Checkered (matches MATLAB 10x10x200 GT)
PRESET_7ELEM_CHECKERED = SimulationConfig(
    nx=10, ny=10, dt=100, n_phases=1, dx=10, dy=10,
    elements=CHECKERED_7ELEM,
    subst_idx=[0], free_idx=[1, 2, 3, 4, 5], linked_idx=[6],
    linked_const=[3.0, 3.0, 2.5, 1.0, 1.0],
    free_profiles=[
        {"pos": 210.0, "width": 30.0, "height": 33.333},   # Si(+4)
        {"pos": 150.0, "width": 80.0, "height": 33.333},   # Ti(+4)
        {"pos":  80.0, "width": 40.0, "height": 40.0},      # Al(+3)
        {"pos":   0.0, "width": 70.0, "height": 100.0},     # C(0)
        {"pos":   0.0, "width": 70.0, "height": 100.0},     # B(0)
    ],
    linked_profiles=[{"pos": 98.025, "width": 197.279, "height": 28.954}],
    dist_type='Gaussian',
    flags={
        'DistribBottomLayerTrigonometric': True,
        'DistribTopLayerCheckered': True,
        'PhenomInterfRedoxExponential': True,
        'PhenomEtchGrowLinear': True,
    },
)


