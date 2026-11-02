#!/usr/bin/env python3
"""
Universal Roundtrip Pipeline — H5フリーの D→I→D' 統合パイプライン
================================================================

ProblemScenario (任意材料系) から:
  1. 合成データ生成 (D_true → I_clean via TransferMatrix)
  2. ノイズ注入 (Poisson / relative)
  3. ソルバー逆解析 (Free / SemiFree / GenericLayer)
  4. 評価 (RMSE, W1, thickness)
  5. NN訓練データエクスポート (I, D ペア → NPZ)

Usage:
    # Verify (5 scenarios × 2 noise × 3 solvers)
    python -m deppro.benchmarks.universal_roundtrip --verify

    # Full roundtrip benchmark
    python -m deppro.benchmarks.universal_roundtrip --roundtrip --n-scenarios 20

    # Export training data for NN
    python -m deppro.benchmarks.universal_roundtrip --export --n-scenarios 50 --n-aug 1000
"""

import argparse
import os
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from deppro.benchmarks.unified_benchmark import (
    AggregatedMetrics,
    ProblemScenario,
    SolverRunner,
    UnifiedMetrics,
    aggregate_metrics,
    extract_metrics,
    make_random_grid,
    LAYER_LIBRARY,
)
from deppro.depth_profile import ElementConfig
from deppro.transfer_matrix import TransferMatrix


# =========================================================================
# RoundtripContext — シナリオのクリーンデータ + 転送行列を保持
# =========================================================================

@dataclass
class RoundtripContext:
    """シナリオから生成されたクリーンデータと転送行列."""

    scenario: ProblemScenario
    I_clean: np.ndarray          # (n_angles, n_elem) normalized clean intensity
    D_true: np.ndarray           # (n_depth, n_elem) ground truth profile
    depth: np.ndarray            # (n_depth,)
    elements: List[ElementConfig]
    elem_names: List[str]
    metadata: dict
    tm: TransferMatrix
    step_size: float
    substrate_idx: int              # primary substrate element index
    substrate_indices: List[int]    # all substrate element indices
    angles: List[float]


def prepare_context(scenario: ProblemScenario, seed: int = 42) -> RoundtripContext:
    """シナリオからクリーンデータを生成し、RoundtripContext を構築.

    noise_level=0 で generate() を呼び、転送行列と基板インデックスを保持する。
    """
    # Clean generation (noise_level=0)
    clean_scenario = replace(scenario, noise_level=0.0)
    I_clean, D_true, depth, elements, metadata = clean_scenario.generate(seed=seed)

    elem_names = metadata['element_names']
    layer_elem_indices = metadata['layer_elem_indices']

    # Substrate indices (may be multi-element, e.g. GaAs, SiC)
    sub_stoich = LAYER_LIBRARY[scenario.substrate]['stoich']
    substrate_indices = []
    for sub_name in sub_stoich.keys():
        canonical = sub_name.split('_')[0] if sub_name.startswith('O_') else sub_name
        if canonical in elem_names:
            substrate_indices.append(elem_names.index(canonical))
        elif sub_name in elem_names:
            substrate_indices.append(elem_names.index(sub_name))
    substrate_idx = substrate_indices[0]  # primary (for backward compat)

    # Transfer matrix
    step_size = depth[1] - depth[0] if len(depth) > 1 else 1.0
    angles = np.linspace(0, 60, scenario.n_angles).tolist()
    tm = TransferMatrix(
        depth=depth, step_size=step_size,
        angles=angles, photon_energy=scenario.photon_energy,
    )
    tm.compute(D_true, elements)

    return RoundtripContext(
        scenario=scenario,
        I_clean=I_clean,
        D_true=D_true,
        depth=depth,
        elements=elements,
        elem_names=elem_names,
        metadata=metadata,
        tm=tm,
        step_size=step_size,
        substrate_idx=substrate_idx,
        substrate_indices=substrate_indices,
        angles=angles,
    )


# =========================================================================
# Noise injection — add_noise() wrapper for 2D arrays
# =========================================================================

def apply_noise(
    I_clean: np.ndarray,
    noise_level: float,
    noise_type: str = 'relative',
    seed: int = 42,
) -> np.ndarray:
    """I_clean (n_angles, n_elem) にノイズを付加.

    d2s2d.add_noise() は 3D (n_px, n_angles, n_elem) を想定するため、
    2D の場合は expand → squeeze する。
    """
    if noise_level <= 0:
        return I_clean.copy()

    from deppro.simulate.d2s2d import add_noise

    was_2d = I_clean.ndim == 2
    if was_2d:
        I_3d = I_clean[np.newaxis]
    else:
        I_3d = I_clean

    I_noisy = add_noise(I_3d, noise_level, noise_type=noise_type, seed=seed)

    # Re-normalize per angle (sum over elements = 1)
    I_sum = I_noisy.sum(axis=-1, keepdims=True)
    I_sum[I_sum == 0] = 1
    I_noisy /= I_sum

    if was_2d:
        return I_noisy[0]
    return I_noisy


# =========================================================================
# Solver dispatch — SolverRunner パターンを流用
# =========================================================================

def _estimate_thickness_from_iobs(
    I_obs: np.ndarray,
    ctx: RoundtripContext,
) -> float:
    """I_obs から膜厚を推定 (基板元素の角度依存性から).

    基板元素の強度が大きい → 薄膜 (or 基板むき出し)
    基板元素の強度がゼロ → 厚膜 (IMFP の数倍以上)

    Returns
    -------
    thk_est : Å  推定膜厚
    """
    I_sub = I_obs[:, ctx.substrate_idx]
    I_sub_mean = float(np.mean(I_sub))

    # Estimate mean IMFP from transfer matrix
    T = ctx.tm.trans_mat  # (n_angles, n_depth, n_elem)
    T_sub = T[0, :, ctx.substrate_idx]  # normal emission
    if T_sub[0] > 0 and T_sub[1] > 0:
        ratio = T_sub[1] / T_sub[0]
        if 0 < ratio < 1:
            imfp = -ctx.step_size / np.log(ratio)
        else:
            imfp = 30.0
    else:
        imfp = 30.0

    # Beer-Lambert: I_sub ~ C_sub_bulk * exp(-d/λ)
    # For normalized intensity: I_sub ~ (1/n_elem) * exp(-d/λ) approximately
    # Correction: erf transition means effective signal is higher than step model
    # Factor 1.5 compensates for erf vs step-function difference
    n_elem = len(ctx.elem_names)
    correction = 1.5

    if I_sub_mean > 0.01:
        # Detectable substrate: Beer-Lambert estimate
        est = -imfp * np.log(max(I_sub_mean * n_elem, 0.01)) * correction
        return max(est, 5.0)
    elif I_sub_mean > 1e-5:
        # Weak but nonzero: medium thickness
        est = -imfp * np.log(max(I_sub_mean * n_elem, 1e-5)) * correction
        return min(est, 8.0 * imfp)  # cap at 8λ
    else:
        # Substrate invisible: at information limit
        return 5.0 * imfp


def _classify_elements(ctx: RoundtripContext):
    """元素をfree/linked/substrate に分類.

    Linked (charge-balance) mode is only valid when a single compound
    type is present.  When multiple cation species with different
    oxidation states coexist (e.g. ZrO2 + Al2O3 + TiO2), a single
    freelinkedconst set cannot satisfy sum=1 at every depth because
    each compound has a different cation:anion ratio.

    Rule: use linked only when all free cations share the same
    oxidation state (single-compound regime).  Otherwise treat O/N
    as free elements too.
    """
    anion_names = {'O', 'N'}
    sub_set = set(ctx.substrate_indices)
    linked_candidates = []
    free_indices = []
    for i, name in enumerate(ctx.elem_names):
        if i in sub_set:
            continue
        elif name in anion_names:
            linked_candidates.append(i)
        else:
            free_indices.append(i)

    if not free_indices and linked_candidates:
        # All film elements are anions — treat as free
        return linked_candidates, []

    if not linked_candidates:
        # No anions — all metals, no linked needed
        return free_indices, []

    # Separate oxide cations (ox > 0) from pure metals (ox = 0 or None)
    oxide_cations = []
    metal_indices = []
    for fi in free_indices:
        elem = ctx.elements[fi]
        ox = getattr(elem, 'oxidation_state', None)
        if ox is not None and ox > 0:
            oxide_cations.append(fi)
        else:
            metal_indices.append(fi)

    if not oxide_cations:
        # No oxide cations — all pure metals, no linked
        return free_indices + linked_candidates, []

    # Check if all oxide cations share the same oxidation state
    oxide_charges = set()
    for fi in oxide_cations:
        ox = getattr(ctx.elements[fi], 'oxidation_state', 0)
        oxide_charges.add(ox)

    # Charge balance Σ(charge_i × conc_i) = 0 works for ANY mix of
    # oxide charges — O²⁻ balances all cations regardless of individual
    # oxidation states (e.g. ZrO₂ + Al₂O₃ + SrTiO₃ all balance with O).
    #
    # However, charge balance is ONE equation — so only ONE anion species
    # can be linked.  When multiple anions exist (e.g. O and N from
    # ZrO₂+Si₃N₄), pick the one that participates in more compounds
    # (= more cations share its charge type).  The other anion becomes free.
    if len(linked_candidates) > 1:
        # Count how many oxide cations each anion "serves"
        # O²⁻ serves all oxide cations (they form oxides)
        # N³⁻ serves nitride cations (Si in Si₃N₄)
        # Heuristic: O is almost always the dominant anion in mixed systems
        best_anion = None
        best_count = -1
        for ai in linked_candidates:
            aname = ctx.elem_names[ai]
            if aname == 'O':
                # O serves all oxide cations
                count = len(oxide_cations)
            elif aname == 'N':
                # N serves nitride cations — those with '_n' suffix
                count = sum(1 for fi in oxide_cations
                            if ctx.elem_names[fi].endswith('_n'))
            else:
                count = 0
            if count > best_count:
                best_count = count
                best_anion = ai
        linked_final = [best_anion] if best_anion is not None else []
        extra_free = [ai for ai in linked_candidates if ai != best_anion]
        return oxide_cations + metal_indices + extra_free, linked_final

    return oxide_cations + metal_indices, linked_candidates


def _solve_semifree_multistart(
    ctx: RoundtripContext,
    I_obs: np.ndarray,
    solver_config,
) -> object:
    """Multi-start SemiFree with known thickness (±20%) and fixed interface width.

    Given information:
    - Substrate element (exact)
    - Total thickness ±20% as initial guess
    - Interface width (exact, fixed during optimization)
    - Element classification (free/linked/substrate)

    NOT given:
    - Layer count (single erf model)
    - Layer compositions
    """
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    free_indices, linked_indices = _classify_elements(ctx)
    scenario = ctx.scenario

    # True thickness with ±20% perturbation for multi-start
    thk_true = scenario.total_thickness
    thk_inits = [
        thk_true * 0.8,   # -20%
        thk_true,          # exact
        thk_true * 1.2,    # +20%
    ]

    # True interface width — use mean of layer interface widths
    abrp_true = np.mean([l.interface_width for l in scenario.layers])

    # Bounds: ±3x from true
    thk_bounds = (max(1.0, thk_true * 0.2), thk_true * 5.0)

    # For multi-element substrates: fix stoichiometric ratio from LAYER_LIBRARY
    n_sub = len(ctx.substrate_indices)
    if n_sub > 1:
        sub_stoich = LAYER_LIBRARY[scenario.substrate]['stoich']
        # Build ratio vector matching ctx.substrate_indices order
        sub_ratios = []
        for si in ctx.substrate_indices:
            elem_name = ctx.elem_names[si]
            # Find matching stoich key
            matched = False
            for skey, sval in sub_stoich.items():
                canonical = skey.split('_')[0] if skey.startswith('O_') else skey
                if canonical == elem_name or skey == elem_name:
                    sub_ratios.append(sval)
                    matched = True
                    break
            if not matched:
                sub_ratios.append(1.0)
        sub_ratios = np.array(sub_ratios, dtype=np.float64)
        sub_ratios /= sub_ratios.sum()  # normalize to sum=1
        optimize_substrate_ratio = False
    else:
        sub_ratios = None  # let solver use default (1.0 for single)
        optimize_substrate_ratio = True

    candidates = []

    def _run_starts(intensity_norm):
        """Run multistart with given normalization mode."""
        results = []
        for method in ['LLS', 'L1']:
            alpha = 0.0 if method == 'LLS' else 0.01
            sc_try = SolverConfig(
                method=method, alpha=alpha,
                scipy_method='SLSQP', enable_fallback=True,
            )
            for thk_init in thk_inits:
                use_charge = bool(linked_indices)
                semi_config = SemiFreeSolverConfig(
                    thickness_init=thk_init,
                    abruptness_init=abrp_true,
                    optimize_abruptness=False,
                    optimize_substrate_ratio=optimize_substrate_ratio,
                    free_indices=free_indices,
                    linked_indices=linked_indices if linked_indices else None,
                    substrate_indices=ctx.substrate_indices,
                    thickness_bounds=thk_bounds,
                    use_charge_constraint=use_charge,
                    intensity_normalization=intensity_norm,
                )
                try:
                    solver = SemiFreeSolver(ctx.tm, sc_try, semi_config, elements=ctx.elements)
                    result = solver.solve(I_obs, verbose=False)
                    results.append(result)
                except Exception:
                    continue
        return results

    def _is_stuck(D_opt, sub_indices):
        """Detect if profile is stuck at uniform initial values."""
        film_mask = [i for i in range(D_opt.shape[1]) if i not in sub_indices]
        if len(film_mask) <= 1:
            return False
        uniform = 1.0 / len(film_mask)
        n_stuck = sum(1 for fi in film_mask
                      if abs(D_opt[0, fi] - uniform) < 0.01)
        return n_stuck == len(film_mask)

    def _select_valid(results, prefer_native=True):
        """Select best valid (sum≈1, not stuck) candidate.

        When prefer_native=True and rmse values are nearly identical
        (within 1%), prefer SLSQP-native over trust-constr fallback.
        When prefer_native=False, just pick lowest rmse.
        """
        valid = [r for r in results
                 if (np.max(np.abs(r.D_opt.sum(axis=1) - 1.0)) < 0.05
                     and not _is_stuck(r.D_opt, ctx.substrate_indices))]
        if not valid:
            return None, False
        best_rmse = min(r.rmse for r in valid)
        if prefer_native:
            tied = [r for r in valid if r.rmse < best_rmse * 1.01]
            non_fb = [r for r in tied
                      if not getattr(r, 'fallback_used', False)]
            pool = non_fb if non_fb else tied
        else:
            pool = valid
        return min(pool, key=lambda r: r.rmse), True

    def _fit_erf_to_substrate(D_free):
        """Fit erf model to FreeSolver substrate profile → (thickness, abruptness).

        D_sub(z) = 0.5 * (1 + erf((z - thk) / sigma))
        where sigma = abruptness / (2*sqrt(ln2))
        """
        from scipy.optimize import curve_fit
        from scipy.special import erf as scipy_erf

        depth = ctx.depth
        # Sum substrate columns
        sub_profile = D_free[:, ctx.substrate_indices].sum(axis=1)

        def erf_model(z, thk, abrp):
            sigma = max(abrp, 0.1) / (2 * np.sqrt(np.log(2)))
            return 0.5 * (1 + scipy_erf((z - thk) / sigma))

        try:
            p0 = [thk_true, abrp_true]
            popt, _ = curve_fit(erf_model, depth, sub_profile,
                                p0=p0,
                                bounds=([thk_bounds[0], 0.1],
                                        [thk_bounds[1], 50.0]))
            return float(popt[0]), float(popt[1])
        except Exception:
            return thk_true, abrp_true

    def _run_free_init_starts(D_free, intensity_norm):
        """Run SF multistart using FreeSolver result as D_prior.

        Also optimizes abruptness since erf-fit from Free profile
        tends to overestimate it (Free profiles are smoother).
        """
        thk_fit, abrp_fit = _fit_erf_to_substrate(D_free)
        results = []
        # Broader thickness grid (erf-fit tends to be inaccurate for
        # weak substrates) + two abrp values (the erf-fit abruptness can be off)
        thk_mults = [0.8, 1.0, 1.2]
        abrp_inits = [abrp_fit]
        for method in ['LLS', 'L1']:
            alpha = 0.0 if method == 'LLS' else 0.01
            for thk_mult in thk_mults:
                thk_init = np.clip(thk_fit * thk_mult,
                                   thk_bounds[0], thk_bounds[1])
                for abrp_init in abrp_inits:
                    use_charge = bool(linked_indices)
                    sc_try = SolverConfig(
                        method=method, alpha=alpha,
                        scipy_method='SLSQP', enable_fallback=True,
                    )
                    semi_config = SemiFreeSolverConfig(
                        thickness_init=thk_init,
                        abruptness_init=abrp_init,
                        optimize_abruptness=True,
                        optimize_substrate_ratio=optimize_substrate_ratio,
                        free_indices=free_indices,
                        linked_indices=linked_indices if linked_indices else None,
                        substrate_indices=ctx.substrate_indices,
                        thickness_bounds=thk_bounds,
                        use_charge_constraint=use_charge,
                        intensity_normalization=intensity_norm,
                    )
                    try:
                        solver = SemiFreeSolver(
                            ctx.tm, sc_try, semi_config,
                            elements=ctx.elements, D_prior=D_free,
                        )
                        result = solver.solve(I_obs, verbose=False)
                        results.append(result)
                    except Exception:
                        continue
        return results

    # Strategy: always run substrate-norm.  When substrate signal is weak
    # (max I_sub < 0.02 in total-normalized data), also run Free-initialized
    # SF with scalar_substrate norm and pick the better result.
    # Free-init is used only for a weak substrate signal, because an
    # inaccurate Free profile can steer SF away from a good start.
    sub_candidates = _run_starts('substrate')
    best_sub, has_valid_sub = _select_valid(sub_candidates)

    # Detect weak substrate signal
    I_sub_all = I_obs[:, ctx.substrate_indices].sum(axis=1)
    _weak_substrate = np.max(I_sub_all) < 0.02

    free_init_candidates = []
    if _weak_substrate:
        try:
            free_result = _solve_free_iterative(ctx, I_obs, n_outer=5)
            D_free = free_result.D_opt
            free_init_candidates = _run_free_init_starts(D_free, 'scalar_substrate')
        except Exception:
            pass

    best_fi, has_valid_fi = _select_valid(free_init_candidates)

    # Choose: when substrate is weak, prefer free-init (it has better
    # initial values and scalar normalization avoids blow-up)
    if has_valid_sub and has_valid_fi:
        best_result = best_fi if _weak_substrate else best_sub
    elif has_valid_sub:
        best_result = best_sub
    elif has_valid_fi:
        best_result = best_fi
    else:
        all_cands = sub_candidates + free_init_candidates
        best_result = min(all_cands, key=lambda r: r.rmse) if all_cands else None

    if best_result is None:
        use_charge_fb = bool(linked_indices)
        semi_config = SemiFreeSolverConfig(
            thickness_init=thk_true,
            abruptness_init=abrp_true,
            optimize_abruptness=False,
            free_indices=free_indices,
            linked_indices=linked_indices if linked_indices else None,
            substrate_indices=ctx.substrate_indices,
            thickness_bounds=(1.0, 500.0),
            use_charge_constraint=use_charge_fb,
        )
        solver = SemiFreeSolver(ctx.tm, solver_config, semi_config, elements=ctx.elements)
        best_result = solver.solve(I_obs, verbose=False)

    return best_result


def _solve_free_iterative(ctx: RoundtripContext, I_obs: np.ndarray, n_outer: int = 5):
    """FreeSolver with Livesey (1994) exponential prior + iterative L1.

    Uses known substrate information to create a physically motivated prior:
    - Overlayer elements decay exponentially from the surface
    - Substrate element fills the remainder (approaches 1.0 at depth)
    - d_penetration = 0.1 × total_thickness (sharper prior)

    Iterative: each loop uses previous result as prior, allowing the
    optimizer to progressively refine while anchoring unobservable depths.

    Settings used here: d_pen_factor 0.1, L1 with α=0.01, n_outer 5.

    Reference:
        A.K. Livesey and G.C. Smith, J. Electron Spectrosc. Relat. Phenom.
        67 (1994) 439-461, Equation 19.
    """
    from deppro.inverse_solver import FreeSolver, SolverConfig

    n_depth, n_elem = ctx.D_true.shape
    depth = ctx.tm.depth
    sub_indices = ctx.substrate_indices

    # --- Build Livesey exponential prior ---
    # d_penetration: 0.1 × total_thickness (a sharper decay concentrates the
    # film information near the surface)
    # Floor at 3.0 Å to avoid degenerate priors for ultra-thin films (<30Å)
    d_penetration = max(3.0, ctx.scenario.total_thickness * 0.1)

    # Surface concentrations: equal among overlayer elements, 0 for substrate
    surface_conc = np.ones(n_elem)
    for si in sub_indices:
        surface_conc[si] = 0.0
    n_overlayer = n_elem - len(sub_indices)
    if n_overlayer > 0:
        surface_conc /= n_overlayer
    for si in sub_indices:
        surface_conc[si] = 0.0

    # Build prior: overlayer decays, substrate fills remainder
    D_prior = np.zeros((n_depth, n_elem))
    for j in range(n_elem):
        if j not in sub_indices:
            D_prior[:, j] = surface_conc[j] * np.exp(-depth / max(d_penetration, 1.0))

    # Substrate: distribute remainder equally among substrate elements
    remainder = 1.0 - D_prior.sum(axis=1)
    remainder = np.maximum(remainder, 1e-6)
    n_sub = len(sub_indices)
    sub_stoich = LAYER_LIBRARY[ctx.scenario.substrate]['stoich']
    sub_total = sum(sub_stoich.values())
    for si in sub_indices:
        # Use stoichiometric ratio for multi-element substrates
        elem_name = ctx.elem_names[si]
        ratio = 1.0 / n_sub  # default: equal
        for skey, sval in sub_stoich.items():
            canonical = skey.split('_')[0] if skey.startswith('O_') else skey
            if canonical == elem_name or skey == elem_name:
                ratio = sval / sub_total
                break
        D_prior[:, si] = remainder * ratio

    # Normalize
    D_prior = D_prior / D_prior.sum(axis=1, keepdims=True)
    D_current = D_prior.copy()

    # The result is the average of an MEM run and an L1 run (an ensemble)
    # rather than the run with the lower intensity residual, because the
    # intensity residual need not track the depth error in this ill-posed
    # problem.
    D_opts = []
    last_result = None
    for method, alpha in [('MEM', 1e-4), ('L1', 0.01)]:
        D_run = D_prior.copy()
        result = None
        for outer in range(n_outer):
            cfg = SolverConfig(
                method=method, alpha=alpha,
                scipy_method='SLSQP', enable_fallback=True,
                n_loops=1, max_iter=200,
                degree=max(1.0, 10.0 * 0.7 ** outer),
            )
            solver = FreeSolver(ctx.tm, cfg, D_prior=D_run.copy(), elements=ctx.elements)
            result = solver.solve(I_obs, D_init=D_run.copy(),
                                  substrate_indices=None,
                                  verbose=False)
            D_run = result.D_opt.copy()
        if result is not None:
            D_opts.append(result.D_opt.copy())
            last_result = result

    # Ensemble: average D_opt from MEM and L1, then re-normalize
    if len(D_opts) == 2:
        D_avg = (D_opts[0] + D_opts[1]) / 2.0
        row_sums = D_avg.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        D_avg /= row_sums
        last_result.D_opt = D_avg

    return last_result


def _solve_genericlayer_multistart(
    ctx: RoundtripContext,
    I_obs: np.ndarray,
    solver_config,
) -> object:
    """Multi-start GenericLayer with thickness multipliers.

    Runs GL solver with 6 different initial thickness multipliers
    (×0.5, ×0.8, ×1.0, ×1.2, ×1.5, ×2.0) and selects the best
    candidate by ismp RMSE (intensity fit quality).

    Trying several starting thicknesses guards against a single start
    stopping in a local minimum.
    """
    from deppro.inverse_solver import (
        GenericLayerSolver, GenericLayerConfig, LayerDefinition,
    )

    scenario = ctx.scenario
    layer_elem_indices = ctx.metadata['layer_elem_indices']

    # Build layer and substrate definitions (shared across starts)
    layers = []
    for i, layer_spec in enumerate(scenario.layers):
        stoich = LAYER_LIBRARY[layer_spec.material]['stoich']
        idx_map = layer_elem_indices[i]
        stoich_by_idx = {idx_map[name]: ratio for name, ratio in stoich.items()}
        layers.append(LayerDefinition(
            name=layer_spec.material,
            element_indices=list(stoich_by_idx.keys()),
            stoichiometry=stoich_by_idx,
        ))

    sub_stoich = LAYER_LIBRARY[scenario.substrate]['stoich']
    sub_idx_map = layer_elem_indices[-1]
    sub_stoich_by_idx = {sub_idx_map[name]: ratio for name, ratio in sub_stoich.items()}
    substrate_def = LayerDefinition(
        name=scenario.substrate,
        element_indices=list(sub_stoich_by_idx.keys()),
        stoichiometry=sub_stoich_by_idx,
    )

    init_widths = [l.interface_width for l in scenario.layers]
    thk_bounds = (0.5, max(500.0, scenario.total_thickness * 5))

    # Multi-start: sweep thickness multipliers
    thk_multipliers = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    candidates = []

    for mult in thk_multipliers:
        init_thk = [l.thickness * mult for l in scenario.layers]
        layer_config = GenericLayerConfig(
            layers=layers,
            substrate=substrate_def,
            initial_thicknesses=init_thk,
            initial_widths=init_widths,
            thickness_bounds=thk_bounds,
        )
        try:
            solver = GenericLayerSolver(
                ctx.tm, solver_config, layer_config, elements=ctx.elements,
            )
            result = solver.solve(I_obs, verbose=False)
            candidates.append(result)
        except Exception:
            continue

    # Select best by ismp RMSE
    best_result = None
    best_rmse = np.inf
    for result in candidates:
        if result.rmse < best_rmse:
            best_rmse = result.rmse
            best_result = result

    if best_result is None:
        # Fallback: single run with ×1.0
        init_thk = [l.thickness for l in scenario.layers]
        layer_config = GenericLayerConfig(
            layers=layers,
            substrate=substrate_def,
            initial_thicknesses=init_thk,
            initial_widths=init_widths,
            thickness_bounds=thk_bounds,
        )
        solver = GenericLayerSolver(
            ctx.tm, solver_config, layer_config, elements=ctx.elements,
        )
        best_result = solver.solve(I_obs, verbose=False)

    return best_result


def _solve_single(
    ctx: RoundtripContext,
    I_obs: np.ndarray,
    solver_name: str,
    method: str = 'LLS',
    alpha: float = 0.0,
) -> UnifiedMetrics:
    """1つのソルバーで逆解析し、メトリクスを返す."""
    from deppro.inverse_solver import (
        FreeSolver, SemiFreeSolver, SemiFreeSolverConfig,
        GenericLayerSolver, GenericLayerConfig, LayerDefinition,
        SolverConfig,
    )

    solver_config = SolverConfig(
        method=method, alpha=alpha,
        scipy_method='SLSQP', enable_fallback=True,
    )
    scenario = ctx.scenario
    layer_elem_indices = ctx.metadata['layer_elem_indices']

    t0 = time.perf_counter()

    if solver_name == 'Free':
        result = _solve_free_iterative(ctx, I_obs)

    elif solver_name == 'SemiFree':
        result = _solve_semifree_multistart(ctx, I_obs, solver_config)

    elif solver_name == 'GenericLayer':
        result = _solve_genericlayer_multistart(ctx, I_obs, solver_config)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    elapsed = time.perf_counter() - t0
    result.elapsed_time = elapsed

    return extract_metrics(
        result, ctx.D_true, ctx.depth,
        thk_true=scenario.total_thickness,
        solver_name=solver_name,
        substrate_idx=ctx.substrate_idx,
    )


# =========================================================================
# run_scenario_roundtrip — メインエントリポイント
# =========================================================================

def run_scenario_roundtrip(
    scenarios: List[ProblemScenario],
    noise_levels: List[float] = [0.0, 0.05, 0.1],
    solvers: List[str] = ['Free', 'SemiFree', 'GenericLayer'],
    n_trials: int = 5,
    method: str = 'LLS',
    alpha: float = 0.0,
    noise_type: str = 'relative',
    verbose: bool = True,
) -> List[dict]:
    """Universal roundtrip: generate → noise → solve → evaluate.

    Returns
    -------
    records : List[dict]
        Each dict: {scenario, noise_level, solver, trial, ...metrics}
    """
    records = []

    for si, scenario in enumerate(scenarios):
        if verbose:
            print(f"\n[{si+1}/{len(scenarios)}] {scenario.name} "
                  f"({scenario.n_layers}L, {len(scenario._build_elements()[0])}elem, "
                  f"{scenario.n_angles}ang)")

        ctx = prepare_context(scenario, seed=42)

        for nl in noise_levels:
            for trial in range(n_trials):
                seed = 10000 * si + 1000 * trial + int(nl * 10000) + 7
                I_obs = apply_noise(ctx.I_clean, nl, noise_type=noise_type, seed=seed)

                for solver_name in solvers:
                    try:
                        m = _solve_single(ctx, I_obs, solver_name, method, alpha)
                        records.append({
                            'scenario': scenario.name,
                            'n_layers': scenario.n_layers,
                            'n_elem': len(ctx.elem_names),
                            'n_angles': scenario.n_angles,
                            'noise_level': nl,
                            'trial': trial,
                            'solver': solver_name,
                            'depth_rmse': m.depth_rmse,
                            'w1_mean': m.w1_mean,
                            'thickness_err': m.thickness_err,
                            'ismp_rmse': m.ismp_rmse,
                            'elapsed': m.elapsed_time,
                            'success': m.success,
                        })
                    except Exception as e:
                        if verbose:
                            print(f"    {solver_name} noise={nl:.2f} t{trial}: FAIL ({e})")
                        records.append({
                            'scenario': scenario.name,
                            'n_layers': scenario.n_layers,
                            'n_elem': len(ctx.elem_names),
                            'n_angles': scenario.n_angles,
                            'noise_level': nl,
                            'trial': trial,
                            'solver': solver_name,
                            'depth_rmse': np.nan,
                            'w1_mean': np.nan,
                            'thickness_err': np.nan,
                            'ismp_rmse': np.nan,
                            'elapsed': 0.0,
                            'success': False,
                        })

            if verbose:
                # Print summary for this noise level
                for solver_name in solvers:
                    solver_recs = [r for r in records
                                   if r['scenario'] == scenario.name
                                   and r['noise_level'] == nl
                                   and r['solver'] == solver_name
                                   and r['success']]
                    if solver_recs:
                        rmses = [r['depth_rmse'] for r in solver_recs]
                        w1s = [r['w1_mean'] for r in solver_recs]
                        print(f"  noise={nl:.2f} {solver_name:14s}: "
                              f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}  "
                              f"W1={np.mean(w1s):.1f}±{np.std(w1s):.1f}Å")

    return records


# =========================================================================
# export_training_data — NN訓練データ生成
# =========================================================================

def _batch_forward(D_batch, T):
    """Vectorized forward model: I = D @ T^T per element.

    Parameters
    ----------
    D_batch : (n_batch, n_depth, n_elem)
    T : (n_angles, n_depth, n_elem)  — TransferMatrix.trans_mat

    Returns
    -------
    I_batch : (n_batch, n_angles, n_elem) normalized
    """
    n_batch, n_depth, n_elem = D_batch.shape
    n_angles = T.shape[0]
    I_batch = np.zeros((n_batch, n_angles, n_elem), dtype=np.float32)
    for e in range(n_elem):
        # D_batch[:, :, e] @ T[:, :, e].T → (n_batch, n_angles)
        I_batch[:, :, e] = D_batch[:, :, e] @ T[:, :, e].T

    # Normalize per angle (sum over elements)
    I_sum = I_batch.sum(axis=2, keepdims=True)
    I_sum[I_sum == 0] = 1
    I_batch /= I_sum
    return I_batch


def export_training_data(
    scenarios: List[ProblemScenario],
    n_aug_per_scenario: int = 1000,
    noise_augment: float = 0.3,
    val_fraction: float = 0.1,
    output_dir: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[Tuple[int, int, int], str]:
    """NN訓練データを (n_angles, n_elem, n_depth) グループごとに NPZ 出力.

    Parameters
    ----------
    scenarios : ProblemScenario のリスト
    n_aug_per_scenario : シナリオあたりの拡張データ数
    noise_augment : ノイズ付加する割合 (0.3 = 30%)
    val_fraction : validation split 割合
    output_dir : 出力ディレクトリ
    seed : ランダムシード

    Returns
    -------
    output_paths : {(n_angles, n_elem, n_depth): npz_path}
    """
    from deppro.inference.train import _augment_profiles

    rng = np.random.default_rng(seed)
    if output_dir is None:
        from deppro.paths import outputs_dir as _od
        output_dir = str(_od() / 'training_data')
    os.makedirs(output_dir, exist_ok=True)

    # Group scenarios by (n_angles, n_elem, n_depth)
    groups = defaultdict(list)
    for scenario in scenarios:
        ctx = prepare_context(scenario, seed=42)
        n_elem = len(ctx.elem_names)
        key = (scenario.n_angles, n_elem, scenario.n_depth)
        groups[key].append(ctx)

    if verbose:
        print(f"Scenarios: {len(scenarios)} → {len(groups)} groups")

    output_paths = {}

    for (n_angles, n_elem, n_depth), ctxs in groups.items():
        if verbose:
            print(f"\nGroup ({n_angles}ang, {n_elem}elem, {n_depth}depth): "
                  f"{len(ctxs)} scenarios")

        all_I = []
        all_D = []

        for ci, ctx in enumerate(ctxs):
            # Collect D_true (1, n_depth, n_elem) → augment
            D_base = ctx.D_true[np.newaxis].astype(np.float32)
            D_aug = _augment_profiles(D_base, n_aug_per_scenario, rng)
            D_all = np.concatenate([D_base, D_aug], axis=0)  # (1+n_aug, n_depth, n_elem)

            # Forward model using this scenario's TransferMatrix
            T = ctx.tm.trans_mat  # (n_angles, n_depth, n_elem)
            I_all = _batch_forward(D_all, T)

            all_I.append(I_all)
            all_D.append(D_all)

            if verbose and (ci + 1) % 10 == 0:
                print(f"  {ci+1}/{len(ctxs)} scenarios processed")

        # Concatenate all scenarios in this group
        I_cat = np.concatenate(all_I, axis=0)  # (N, n_angles, n_elem)
        D_cat = np.concatenate(all_D, axis=0)  # (N, n_depth, n_elem)
        n_total = I_cat.shape[0]

        # Noise augmentation on a fraction of samples
        if noise_augment > 0:
            n_noisy = int(n_total * noise_augment)
            noise_idx = rng.choice(n_total, n_noisy, replace=False)
            noise_levels = rng.uniform(0.01, 0.05, size=n_noisy).astype(np.float32)
            for i, idx in enumerate(noise_idx):
                sigma = noise_levels[i]
                noise = rng.normal(0, sigma, size=I_cat[idx].shape).astype(np.float32)
                I_cat[idx] = np.maximum(I_cat[idx] + noise, 0)
                row_sum = I_cat[idx].sum(axis=1, keepdims=True)
                row_sum[row_sum == 0] = 1
                I_cat[idx] /= row_sum

        # Flatten
        I_flat = I_cat.reshape(n_total, -1)  # (N, n_angles * n_elem)
        D_flat = D_cat.reshape(n_total, -1)  # (N, n_depth * n_elem)

        # Train/val split
        n_val = max(1, int(n_total * val_fraction))
        perm = rng.permutation(n_total)

        # Collect element names from first context (all same n_elem in group)
        elem_names = ctxs[0].elem_names
        step_size = ctxs[0].step_size
        depth = ctxs[0].depth

        fname = f"train_{n_angles}ang_{n_elem}elem_{n_depth}depth.npz"
        fpath = os.path.join(output_dir, fname)

        np.savez_compressed(
            fpath,
            I_train=I_flat[perm[:-n_val]],
            D_train=D_flat[perm[:-n_val]],
            I_val=I_flat[perm[-n_val:]],
            D_val=D_flat[perm[-n_val:]],
            depth=depth,
            step_size=step_size,
            elem_names=np.array(elem_names),
            n_angles=n_angles,
            n_elem=n_elem,
            n_depth=n_depth,
            n_scenarios=len(ctxs),
            scenario_names=np.array([c.scenario.name for c in ctxs]),
        )

        output_paths[(n_angles, n_elem, n_depth)] = fpath

        if verbose:
            print(f"  Saved: {fpath}")
            print(f"  Train: {n_total - n_val:,}  Val: {n_val:,}  "
                  f"I: ({n_angles}×{n_elem})={n_angles*n_elem}  "
                  f"D: ({n_depth}×{n_elem})={n_depth*n_elem}")

    return output_paths


# =========================================================================
# Print helpers
# =========================================================================

def print_summary_table(records: List[dict]):
    """records を集約して solver × noise テーブルを表示."""
    import pandas as pd

    df = pd.DataFrame(records)
    if df.empty:
        print("No results.")
        return

    # Aggregate: mean over trials and scenarios
    agg = df.groupby(['noise_level', 'solver']).agg(
        depth_rmse_mean=('depth_rmse', 'mean'),
        depth_rmse_std=('depth_rmse', 'std'),
        w1_mean_mean=('w1_mean', 'mean'),
        w1_mean_std=('w1_mean', 'std'),
        thk_err_mean=('thickness_err', 'mean'),
        elapsed_mean=('elapsed', 'mean'),
        success_rate=('success', 'mean'),
        n=('success', 'count'),
    ).reset_index()

    print("\n" + "=" * 90)
    print("ROUNDTRIP SUMMARY")
    print("=" * 90)
    print(f"{'Noise':>6s} {'Solver':>14s} | {'RMSE':>12s} | {'W1 (Å)':>12s} | "
          f"{'Thk err':>8s} | {'Time':>6s} | {'OK%':>4s}")
    print("-" * 90)

    for _, row in agg.iterrows():
        print(f"{row['noise_level']:6.2f} {row['solver']:>14s} | "
              f"{row['depth_rmse_mean']:5.4f}±{row['depth_rmse_std']:5.4f} | "
              f"{row['w1_mean_mean']:5.1f}±{row['w1_mean_std']:5.1f}     | "
              f"{row['thk_err_mean']:7.1f}Å | "
              f"{row['elapsed_mean']:5.2f}s | "
              f"{row['success_rate']*100:3.0f}%")
    print("=" * 90)

    return df


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Universal Roundtrip Pipeline (H5-free)',
    )
    parser.add_argument('--verify', action='store_true',
                        help='Quick verify: 5 scenarios × 2 noise × 3 solvers')
    parser.add_argument('--roundtrip', action='store_true',
                        help='Full roundtrip benchmark')
    parser.add_argument('--export', action='store_true',
                        help='Export training data for NN')
    parser.add_argument('--n-scenarios', type=int, default=20)
    parser.add_argument('--n-trials', type=int, default=5)
    parser.add_argument('--n-aug', type=int, default=1000,
                        help='Augmentations per scenario (for --export)')
    parser.add_argument('--noise', nargs='+', type=float, default=None,
                        help='Noise levels (default: [0.0, 0.05, 0.1])')
    parser.add_argument('--solvers', nargs='+', default=None,
                        help='Solvers (default: Free SemiFree GenericLayer)')
    parser.add_argument('--method', default='LLS')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--output-dir', default=None,
                        help='default: $DEPPRO_OUTPUTS/training_data (deppro.paths.outputs_dir)')

    args = parser.parse_args()
    if args.output_dir is None:
        from deppro.paths import outputs_dir as _od
        args.output_dir = str(_od() / 'training_data')

    noise_levels = args.noise or [0.0, 0.05, 0.1]
    solvers = args.solvers or ['Free', 'SemiFree', 'GenericLayer']

    if args.verify:
        print("=== VERIFY MODE (5 scenarios, 2 noise, 3 solvers, 3 trials) ===")
        scenarios = make_random_grid(n_scenarios=5, seed=args.seed)
        records = run_scenario_roundtrip(
            scenarios,
            noise_levels=[0.0, 0.05],
            solvers=['Free', 'SemiFree', 'GenericLayer'],
            n_trials=3,
            method=args.method,
            verbose=True,
        )
        print_summary_table(records)

    elif args.roundtrip:
        print(f"=== ROUNDTRIP ({args.n_scenarios} scenarios) ===")
        scenarios = make_random_grid(n_scenarios=args.n_scenarios, seed=args.seed)
        records = run_scenario_roundtrip(
            scenarios,
            noise_levels=noise_levels,
            solvers=solvers,
            n_trials=args.n_trials,
            method=args.method,
            verbose=True,
        )
        df = print_summary_table(records)

        # Save raw results
        out = os.path.join(args.output_dir, 'roundtrip_results.npz')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        import pandas as pd
        df_full = pd.DataFrame(records)
        df_full.to_csv(out.replace('.npz', '.csv'), index=False)
        print(f"\nSaved: {out.replace('.npz', '.csv')}")

    elif args.export:
        print(f"=== EXPORT ({args.n_scenarios} scenarios, {args.n_aug} aug/scenario) ===")
        scenarios = make_random_grid(n_scenarios=args.n_scenarios, seed=args.seed)
        paths = export_training_data(
            scenarios,
            n_aug_per_scenario=args.n_aug,
            output_dir=args.output_dir,
            seed=args.seed,
            verbose=True,
        )
        print(f"\nExported {len(paths)} group(s):")
        for key, path in paths.items():
            print(f"  {key} → {path}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
