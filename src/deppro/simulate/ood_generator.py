"""
OOD (Out-of-Distribution) Data Generator
========================================

Generates synthetic ARXPS benchmark data that lies outside the training
distribution of a 7-element NN depth model. Three OOD axes are supported:

    Axis 1: Multi-layer  — N_layers ∈ {2, 3, 4, 5}
    Axis 2: Element pool — swap Ti→Zr, Al→Ga, or Si→Ge at the compound level
    Axis 3: Thickness    — sample from [10,50], [500,1000], or mixed

The NN input/output topology (14 angles × 7 elements in, 20 depths × 7 elements
out) is preserved so the pretrained MLP is directly callable.

All generators return a dict with:
    'Dstk'  : (n_px, n_depth, 7)       ground-truth depth profile
    'Xstk'  : (n_px, n_angles, 7)      clean per-angle-normalized intensities
    'Xstk_noisy' : same shape, with Poisson noise added
    'depth' : (n_depth,)
    'angles': (n_angles,)
    'element_configs' : list[ElementConfig]
    'charges' : (n_elem,)
    'metadata' : dict of scenario parameters
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import erf

from deppro import TransferMatrix, ElementConfig, CompoundDatabase
# 配布対象の共有カーネルから取る（benchmarks は pyproject の exclude 対象なので、
# wheel からは import できない）。d2s2d と同じ向きに揃える。
from deppro.simulate.scene_kernel import CHECKERED_7ELEM
from deppro.simulate.d2s2d import add_noise


# ------------------------------------------------------------------
# Element property templates (for axis 2 substitutions)
# ------------------------------------------------------------------

# Template entries mirror STANDARD_6ELEM / CHECKERED_7ELEM shape.
# BE values: Ti 1s ≈ 4969, Zr 1s ≈ 17998, Al 1s ≈ 1562, Ga 1s ≈ 10367,
# Si 1s ≈ 1843, Ge 1s ≈ 11103. Pushing some of these above photon_energy
# would make the kinetic energy negative; substitutions are chosen so
# KE stays > 0 at 8 keV.
#
# For the 7-elem training set, photon_energy=8000 eV. Substituting Ti(+4)
# with Zr(+4) would require KE = 8000 - 17998 = negative → instead use
# Zr 3d (BE≈182 eV) at +4 oxidation. This keeps the NN-compatible layout
# but probes "different element with different IMFP / cross-section".

_SUBST_TEMPLATES: Dict[str, Dict[str, dict]] = {
    'Ti_to_Zr': {
        'target_index': 2,  # Ti(+4)
        'new_element': {
            "symbol": "Zr", "orbital": "3d", "compound": "ZrO2",
            "BE": 182.0, "charge": 4,
            "density": 5.68, "Aw": 123.22, "Nv": 16, "Eg": 5.8,
        },
    },
    'Al_to_Ga': {
        'target_index': 3,  # Al(+3)
        'new_element': {
            # Ga 2p3/2 ≈ 1117 eV BE; use Ga(+3) in Ga2O3.
            # Ga2O3 is not in CompoundTable, fall back to Ga metal density/Aw.
            "symbol": "Ga", "orbital": "2p", "compound": "Ga",
            "BE": 1117.0, "charge": 3,
            "density": 5.91, "Aw": 69.7231, "Nv": 3, "Eg": 0.0,
        },
    },
    'Si_to_Ge': {
        'target_index': 0,  # Si(0) substrate
        'new_element': {
            # Ge 2p3/2 ≈ 1217 eV BE; use Ge(0) metal.
            "symbol": "Ge", "orbital": "2p", "compound": "Ge",
            "BE": 1217.0, "charge": 0,
            "density": 5.323, "Aw": 72.631, "Nv": 4, "Eg": 0.67,
        },
    },
}


@dataclass
class OODScenario:
    """Metadata describing one OOD condition."""
    axis: str                 # 'layer_count', 'element_sub', 'thickness'
    label: str                # e.g. 'N=3', 'Ti->Zr', 'thin_10-50'
    params: dict
    n_pixels: int
    seed: int
    noise_level: float


def _erf_step(z: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """Smooth unit step rising from 0 (surface side) to 1 (bulk side)."""
    return 0.5 * (1.0 + erf((z - center) / max(sigma, 1e-3)))


def _fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / (2.0 * np.sqrt(np.log(2.0)))


# ------------------------------------------------------------------
# Axis 1: Multi-layer D profiles
# ------------------------------------------------------------------

# Layer recipes: each recipe specifies, for N layers (counting the
# substrate as the last layer), which of the 7 elements dominates.
# Element indices follow CHECKERED_7ELEM order:
#   0: Si(0)    substrate
#   1: Si(+4)   film
#   2: Ti(+4)   film
#   3: Al(+3)   film
#   4: C(0)     film
#   5: B(0)     film
#   6: O(-2)    linked
#
# Dominant element stoichiometry per layer is charge-balanced against O.

_LAYER_RECIPES = {
    # (layer_index -> dominant elem indices)  — substrate always index 0 at bottom.
    2: [
        {'name': 'SiO2',  'elem': 1, 'oxy_stoich': 2.0},  # Si(+4)+2O
        {'name': 'Si',    'elem': 0, 'oxy_stoich': 0.0},  # bulk Si
    ],
    3: [
        {'name': 'C',     'elem': 4, 'oxy_stoich': 0.0},
        {'name': 'Al2O3', 'elem': 3, 'oxy_stoich': 1.5},  # 2Al+3O / 2 = 1.5 per Al
        {'name': 'Si',    'elem': 0, 'oxy_stoich': 0.0},
    ],
    4: [
        {'name': 'C',     'elem': 4, 'oxy_stoich': 0.0},
        {'name': 'Al2O3', 'elem': 3, 'oxy_stoich': 1.5},
        {'name': 'TiO2',  'elem': 2, 'oxy_stoich': 2.0},
        {'name': 'Si',    'elem': 0, 'oxy_stoich': 0.0},
    ],
    5: [
        {'name': 'C',     'elem': 4, 'oxy_stoich': 0.0},
        {'name': 'B',     'elem': 5, 'oxy_stoich': 0.0},
        {'name': 'Al2O3', 'elem': 3, 'oxy_stoich': 1.5},
        {'name': 'TiO2',  'elem': 2, 'oxy_stoich': 2.0},
        {'name': 'Si',    'elem': 0, 'oxy_stoich': 0.0},
    ],
}


def _build_multilayer_dstk(
    depth: np.ndarray,
    interface_depths: np.ndarray,   # (N-1,) sorted, ascending
    abruptness_fwhm: float,
    recipe: list[dict],             # len = N
) -> np.ndarray:
    """Build a (n_depth, 7) profile for an N-layer stack.

    Layer k occupies (interface[k-1], interface[k]) with interface[0]=-∞,
    interface[N]=+∞. Membership is a product of erf-transition ribbons.
    Within layer k, dominant element has conc = 1/(1+oxy_stoich), O has
    oxy_stoich/(1+oxy_stoich). Charge neutrality is automatic.
    """
    n_depth = len(depth)
    n_layers = len(recipe)
    assert len(interface_depths) == n_layers - 1

    sigma = _fwhm_to_sigma(abruptness_fwhm)

    # membership[k] = fraction of layer k at each depth
    membership = np.zeros((n_depth, n_layers))
    # Build cumulative "have we entered layer k yet" = erf_step(z, interface_{k-1})
    cum_entered = np.ones((n_depth, n_layers + 1))  # cum[0]=1 everywhere, cum[N]=0
    for k in range(1, n_layers):
        cum_entered[:, k] = _erf_step(depth, interface_depths[k - 1], sigma)
    cum_entered[:, n_layers] = 0.0
    # membership[k] = cum[k] - cum[k+1]  but we want membership in layer k =
    # (1 - entered layer k+1) - (1 - entered layer k) = entered k - entered k+1.
    # With our convention cum[0]=1 (entered layer 0 everywhere) and
    # cum[k] = prob of having reached layer boundary k-1.  So:
    for k in range(n_layers):
        membership[:, k] = cum_entered[:, k] - cum_entered[:, k + 1]

    # Clip and renormalize (numerical safety at fast transitions)
    membership = np.clip(membership, 0.0, 1.0)
    row_sum = membership.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    membership /= row_sum

    # Now distribute into 7 elements
    D = np.zeros((n_depth, 7))
    for k, layer in enumerate(recipe):
        dom = layer['elem']
        s = layer['oxy_stoich']
        # fraction of dominant element = 1 / (1 + s), O fraction = s / (1 + s)
        frac_dom = 1.0 / (1.0 + s)
        frac_O = s / (1.0 + s)
        D[:, dom] += membership[:, k] * frac_dom
        D[:, 6] += membership[:, k] * frac_O

    # Row-normalize to sum=1 (absorbs numerical drift)
    row_sum = D.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    D /= row_sum
    return D


def generate_ood_layer_count(
    n_layers: int,
    n_pixels: int = 128,
    noise_level: float = 0.02,
    seed: int = 42,
    photon_energy: float = 8000.0,
    total_depth: float = 600.0,
    n_depth: int = 20,
    n_angles: int = 14,
    angle_range: Tuple[float, float] = (0.0, 60.0),
    verbose: bool = True,
) -> dict:
    """Axis 1: multi-layer OOD.

    Generates `n_pixels` random realizations of an N-layer stack. Interface
    depths and abruptnesses are drawn uniformly; recipe (element per layer)
    is fixed per N so reconstruction targets are well-posed.
    """
    if n_layers not in _LAYER_RECIPES:
        raise ValueError(f"n_layers={n_layers} not supported. Choices: {list(_LAYER_RECIPES)}")

    rng = np.random.default_rng(seed)
    depth = np.arange(n_depth, dtype=np.float64) * (total_depth / n_depth)
    angles = np.linspace(angle_range[0], angle_range[1], n_angles)

    # Build ElementConfig list from CHECKERED_7ELEM (fixed pool for axis 1)
    element_configs = [
        ElementConfig(
            symbol=e["symbol"], orbital=e["orbital"], compound=e["compound"],
            binding_energy=e["BE"], oxidation_state=e["charge"],
        )
        for e in CHECKERED_7ELEM
    ]
    charges = np.array([e["charge"] for e in CHECKERED_7ELEM], dtype=np.float64)

    # Pre-build TM (IMFP is pixel-independent for axis 1)
    tm = TransferMatrix(
        depth=depth, step_size=float(depth[1] - depth[0]),
        angles=angles, photon_energy=photon_energy,
        work_function=5.0, gamma_eff=0.80,
    )
    compound_db = CompoundDatabase()
    lambda_basic = tm.calculate_lambda_basic(element_configs, compound_db)
    tm.lambda_basic = lambda_basic
    tm.n_elements = 7

    recipe = _LAYER_RECIPES[n_layers]
    Dstk = np.empty((n_pixels, n_depth, 7), dtype=np.float64)
    Xstk = np.empty((n_pixels, n_angles, 7), dtype=np.float64)

    # Sample interfaces: split (40, total_depth-40) into N layers at random
    low, high = 40.0, total_depth - 40.0
    for px in range(n_pixels):
        # Sample N-1 sorted interface positions with a minimum gap of 30 Å
        interfaces = np.sort(rng.uniform(low, high, size=n_layers - 1))
        for k in range(1, n_layers - 1):
            if interfaces[k] - interfaces[k - 1] < 30.0:
                interfaces[k] = interfaces[k - 1] + 30.0
        interfaces = np.clip(interfaces, low, high)
        abrp = rng.uniform(10.0, 40.0)

        D = _build_multilayer_dstk(depth, interfaces, abrp, recipe)
        Dstk[px] = D
        tm.lambda_obs = tm.calculate_lambda_obs(D, lambda_basic)
        tm.trans_mat = tm.calculate_trans_mat(tm.lambda_obs)
        Xstk[px] = tm.forward(D, normalize=True)

    Xstk_noisy = add_noise(Xstk, noise_level, noise_type='poisson', seed=seed + 1)

    if verbose:
        print(f"  [axis=layer_count N={n_layers}] n_pixels={n_pixels} "
              f"noise={noise_level} seed={seed}")

    return {
        'Dstk': Dstk, 'Xstk': Xstk, 'Xstk_noisy': Xstk_noisy,
        'depth': depth, 'angles': angles,
        'element_configs': element_configs, 'charges': charges,
        'metadata': {
            'axis': 'layer_count', 'n_layers': n_layers,
            'recipe': [l['name'] for l in recipe],
            'photon_energy': photon_energy,
            'noise_level': noise_level, 'seed': seed,
        },
    }


# ------------------------------------------------------------------
# Axis 2: Element substitution
# ------------------------------------------------------------------

def _apply_element_substitution(substitution: str) -> list[dict]:
    """Return a modified copy of CHECKERED_7ELEM with one element replaced."""
    if substitution == 'baseline':
        return [dict(e) for e in CHECKERED_7ELEM]
    if substitution not in _SUBST_TEMPLATES:
        raise ValueError(f"Unknown substitution '{substitution}'. "
                         f"Choices: baseline, {list(_SUBST_TEMPLATES)}")
    tmpl = _SUBST_TEMPLATES[substitution]
    new_elems = [dict(e) for e in CHECKERED_7ELEM]
    new_elems[tmpl['target_index']] = dict(tmpl['new_element'])
    return new_elems


def generate_ood_element_sub(
    substitution: str,          # 'Ti_to_Zr', 'Al_to_Ga', 'Si_to_Ge', or 'baseline'
    n_pixels: int = 128,
    noise_level: float = 0.02,
    seed: int = 42,
    photon_energy: float = 8000.0,
    total_depth: float = 600.0,
    n_depth: int = 20,
    n_angles: int = 14,
    angle_range: Tuple[float, float] = (0.0, 60.0),
    verbose: bool = True,
) -> dict:
    """Axis 2: one element in the 7-pool is replaced with an OOD element.

    D profiles follow the baseline 2-layer recipe (film + substrate) and are
    sampled with varied thickness/abruptness. The NN sees the same 7-channel
    input/output but physics (IMFP + BE → KE) differs for one channel.
    """
    elements_defs = _apply_element_substitution(substitution)

    rng = np.random.default_rng(seed)
    depth = np.arange(n_depth, dtype=np.float64) * (total_depth / n_depth)
    angles = np.linspace(angle_range[0], angle_range[1], n_angles)

    element_configs = [
        ElementConfig(
            symbol=e["symbol"], orbital=e["orbital"], compound=e["compound"],
            binding_energy=e["BE"], oxidation_state=e["charge"],
        )
        for e in elements_defs
    ]
    charges = np.array([e["charge"] for e in elements_defs], dtype=np.float64)

    tm = TransferMatrix(
        depth=depth, step_size=float(depth[1] - depth[0]),
        angles=angles, photon_energy=photon_energy,
        work_function=5.0, gamma_eff=0.80,
    )
    compound_db = CompoundDatabase()
    lambda_basic = tm.calculate_lambda_basic(element_configs, compound_db)
    tm.lambda_basic = lambda_basic
    tm.n_elements = 7

    # Use a 4-layer recipe (C/Al2O3/TiO2/Si) so Ti, Al, C, Si are all
    # non-zero — this makes substitution effects observable.
    recipe = _LAYER_RECIPES[4]
    Dstk = np.empty((n_pixels, n_depth, 7), dtype=np.float64)
    Xstk = np.empty((n_pixels, n_angles, 7), dtype=np.float64)
    for px in range(n_pixels):
        # Sample 3 interfaces in (40, total_depth-40) with minimum 30 Å gap
        low, high = 40.0, total_depth - 40.0
        interfaces = np.sort(rng.uniform(low, high, size=3))
        for k in range(1, 3):
            if interfaces[k] - interfaces[k - 1] < 30.0:
                interfaces[k] = interfaces[k - 1] + 30.0
        interfaces = np.clip(interfaces, low, high)
        abrp = rng.uniform(15.0, 45.0)
        D = _build_multilayer_dstk(depth, interfaces, abrp, recipe)
        Dstk[px] = D
        tm.lambda_obs = tm.calculate_lambda_obs(D, lambda_basic)
        tm.trans_mat = tm.calculate_trans_mat(tm.lambda_obs)
        Xstk[px] = tm.forward(D, normalize=True)

    Xstk_noisy = add_noise(Xstk, noise_level, noise_type='poisson', seed=seed + 1)

    if verbose:
        print(f"  [axis=element_sub sub={substitution}] n_pixels={n_pixels} "
              f"noise={noise_level} seed={seed}")

    return {
        'Dstk': Dstk, 'Xstk': Xstk, 'Xstk_noisy': Xstk_noisy,
        'depth': depth, 'angles': angles,
        'element_configs': element_configs, 'charges': charges,
        'metadata': {
            'axis': 'element_sub', 'substitution': substitution,
            'element_symbols': [e['symbol'] for e in elements_defs],
            'photon_energy': photon_energy,
            'noise_level': noise_level, 'seed': seed,
        },
    }


# ------------------------------------------------------------------
# Axis 3: Thickness extrapolation
# ------------------------------------------------------------------

_THICKNESS_RANGES = {
    'baseline':  (80.0, 360.0),
    'thin':      (10.0, 50.0),
    'thick':     (500.0, 1000.0),
    'mixed':     None,       # handled specially: 50:50 split
}


def generate_ood_thickness(
    thickness_scenario: str,   # 'thin', 'thick', 'mixed', or 'baseline'
    n_pixels: int = 128,
    noise_level: float = 0.02,
    seed: int = 42,
    photon_energy: float = 8000.0,
    total_depth: float = 600.0,
    n_depth: int = 20,
    n_angles: int = 14,
    angle_range: Tuple[float, float] = (0.0, 60.0),
    verbose: bool = True,
) -> dict:
    """Axis 3: thickness distribution sweep.

    'thin':  [10, 50] Å (below the training range)
    'thick': [500, 1000] Å (above the training range; with totaldepth=600
             the substrate interface may lie *beyond* the depth grid)
    'mixed': half thin + half thick
    """
    if thickness_scenario not in _THICKNESS_RANGES:
        raise ValueError(f"Unknown thickness scenario '{thickness_scenario}'")

    rng = np.random.default_rng(seed)
    depth = np.arange(n_depth, dtype=np.float64) * (total_depth / n_depth)
    angles = np.linspace(angle_range[0], angle_range[1], n_angles)

    element_configs = [
        ElementConfig(
            symbol=e["symbol"], orbital=e["orbital"], compound=e["compound"],
            binding_energy=e["BE"], oxidation_state=e["charge"],
        )
        for e in CHECKERED_7ELEM
    ]
    charges = np.array([e["charge"] for e in CHECKERED_7ELEM], dtype=np.float64)

    tm = TransferMatrix(
        depth=depth, step_size=float(depth[1] - depth[0]),
        angles=angles, photon_energy=photon_energy,
        work_function=5.0, gamma_eff=0.80,
    )
    compound_db = CompoundDatabase()
    lambda_basic = tm.calculate_lambda_basic(element_configs, compound_db)
    tm.lambda_basic = lambda_basic
    tm.n_elements = 7

    recipe = _LAYER_RECIPES[2]
    Dstk = np.empty((n_pixels, n_depth, 7), dtype=np.float64)
    Xstk = np.empty((n_pixels, n_angles, 7), dtype=np.float64)

    thicknesses = np.empty(n_pixels)
    if thickness_scenario == 'mixed':
        half = n_pixels // 2
        thicknesses[:half] = rng.uniform(*_THICKNESS_RANGES['thin'], size=half)
        thicknesses[half:] = rng.uniform(*_THICKNESS_RANGES['thick'], size=n_pixels - half)
        rng.shuffle(thicknesses)
    else:
        lo, hi = _THICKNESS_RANGES[thickness_scenario]
        thicknesses = rng.uniform(lo, hi, size=n_pixels)

    for px in range(n_pixels):
        t = thicknesses[px]
        abrp = rng.uniform(15.0, 45.0)
        D = _build_multilayer_dstk(depth, np.array([t]), abrp, recipe)
        Dstk[px] = D
        tm.lambda_obs = tm.calculate_lambda_obs(D, lambda_basic)
        tm.trans_mat = tm.calculate_trans_mat(tm.lambda_obs)
        Xstk[px] = tm.forward(D, normalize=True)

    Xstk_noisy = add_noise(Xstk, noise_level, noise_type='poisson', seed=seed + 1)

    if verbose:
        print(f"  [axis=thickness scen={thickness_scenario}] n_pixels={n_pixels} "
              f"thk range=({thicknesses.min():.1f},{thicknesses.max():.1f}) noise={noise_level}")

    return {
        'Dstk': Dstk, 'Xstk': Xstk, 'Xstk_noisy': Xstk_noisy,
        'depth': depth, 'angles': angles,
        'element_configs': element_configs, 'charges': charges,
        'metadata': {
            'axis': 'thickness', 'thickness_scenario': thickness_scenario,
            'thickness_range': (float(thicknesses.min()), float(thicknesses.max())),
            'photon_energy': photon_energy,
            'noise_level': noise_level, 'seed': seed,
        },
    }


# ------------------------------------------------------------------
# Registry helper
# ------------------------------------------------------------------

def get_all_scenarios(
    n_pixels: int = 128,
    noise_level: float = 0.02,
    seed: int = 42,
) -> list[OODScenario]:
    """List of all OOD conditions for the benchmark."""
    scenarios = [
        # Axis 1: layer count (baseline=2, OOD=3,4,5)
        OODScenario('layer_count', 'N=2 (baseline)', {'n_layers': 2},
                    n_pixels, seed, noise_level),
        OODScenario('layer_count', 'N=3', {'n_layers': 3},
                    n_pixels, seed, noise_level),
        OODScenario('layer_count', 'N=4', {'n_layers': 4},
                    n_pixels, seed, noise_level),
        OODScenario('layer_count', 'N=5', {'n_layers': 5},
                    n_pixels, seed, noise_level),
        # Axis 2: element substitution
        OODScenario('element_sub', 'baseline', {'substitution': 'baseline'},
                    n_pixels, seed, noise_level),
        OODScenario('element_sub', 'Ti->Zr', {'substitution': 'Ti_to_Zr'},
                    n_pixels, seed, noise_level),
        OODScenario('element_sub', 'Al->Ga', {'substitution': 'Al_to_Ga'},
                    n_pixels, seed, noise_level),
        OODScenario('element_sub', 'Si->Ge', {'substitution': 'Si_to_Ge'},
                    n_pixels, seed, noise_level),
        # Axis 3: thickness
        OODScenario('thickness', 'baseline (80-360)', {'thickness_scenario': 'baseline'},
                    n_pixels, seed, noise_level),
        OODScenario('thickness', 'thin (10-50)', {'thickness_scenario': 'thin'},
                    n_pixels, seed, noise_level),
        OODScenario('thickness', 'thick (500-1000)', {'thickness_scenario': 'thick'},
                    n_pixels, seed, noise_level),
        OODScenario('thickness', 'mixed (thin+thick)', {'thickness_scenario': 'mixed'},
                    n_pixels, seed, noise_level),
    ]
    return scenarios


def run_scenario(sc: OODScenario, verbose: bool = False) -> dict:
    """Dispatch helper."""
    if sc.axis == 'layer_count':
        return generate_ood_layer_count(
            sc.params['n_layers'], n_pixels=sc.n_pixels,
            noise_level=sc.noise_level, seed=sc.seed, verbose=verbose)
    if sc.axis == 'element_sub':
        return generate_ood_element_sub(
            sc.params['substitution'], n_pixels=sc.n_pixels,
            noise_level=sc.noise_level, seed=sc.seed, verbose=verbose)
    if sc.axis == 'thickness':
        return generate_ood_thickness(
            sc.params['thickness_scenario'], n_pixels=sc.n_pixels,
            noise_level=sc.noise_level, seed=sc.seed, verbose=verbose)
    raise ValueError(f"Unknown axis '{sc.axis}'")
