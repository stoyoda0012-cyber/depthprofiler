#!/usr/bin/env python3
"""
Unified Solver Benchmark — 3ソルバーシナジー評価フレームワーク
============================================================

FreeSolver / SemiFreeSolver / GenericLayerSolver を同一データで比較し、
4つの統一メトリクスで評価する。

メトリクス:
    1. ismp_rmse  — 強度空間の再構成誤差
    2. depth_rmse — 深さ空間の濃度誤差 (元素別)
    3. sigma_dthk — 膜厚推定の標準偏差 (multi-trial)
    4. w1_mean    — W1距離の元素平均 (空間位置精度)

Usage:
    from deppro.benchmarks.unified_benchmark import SolverRunner, ProblemScenario
    runner = SolverRunner()
    results = runner.run_all(scenario, n_trials=10)
"""

import time
import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import erf

from deppro.depth_profile import wasserstein_distance_1d, ElementConfig
from deppro.transfer_matrix import TransferMatrix


# =========================================================================
# UnifiedMetrics — 3ソルバー共通の評価データ
# =========================================================================

@dataclass
class UnifiedMetrics:
    """3ソルバー共通の評価メトリクス (1 trial の結果)."""

    # Intensity space
    ismp_rmse: float          # √(chi2 / n)

    # Depth space (requires ground truth)
    depth_rmse: float         # per-element mean RMSE
    depth_rmse_per_elem: np.ndarray  # (n_elem,)

    # Structural
    thickness_est: float      # 推定膜厚 (total, Å)
    thickness_err: float      # |thk_est - thk_true| (Å)

    # Spatial
    w1_mean: float            # W1距離の元素平均 (Å)
    w1_per_elem: np.ndarray   # (n_elem,)

    # Meta
    solver_name: str
    elapsed_time: float
    success: bool
    n_dof: int                # 自由度


@dataclass
class AggregatedMetrics:
    """Multi-trial の集約結果."""

    solver_name: str
    n_trials: int

    # Mean ± std over trials
    ismp_rmse_mean: float
    ismp_rmse_std: float

    depth_rmse_mean: float
    depth_rmse_std: float

    w1_mean_mean: float
    w1_mean_std: float

    sigma_dthk: float         # σ(thickness) over trials
    thk_bias: float           # mean(thk_est - thk_true)

    elapsed_mean: float
    success_rate: float
    n_dof: int

    # Per-element detail
    depth_rmse_per_elem: np.ndarray  # (n_elem,) mean
    w1_per_elem: np.ndarray          # (n_elem,) mean


# =========================================================================
# Thickness estimation from free-form profiles
# =========================================================================

def estimate_thickness_from_profile(
    depth: np.ndarray,
    D: np.ndarray,
    substrate_idx: int,
) -> float:
    """D_opt から膜厚を推定 (基板プロファイルの50%クロッシング).

    Parameters
    ----------
    depth : (n_depth,)
    D : (n_depth, n_elem)
    substrate_idx : 基板元素のインデックス

    Returns
    -------
    thickness : Å (50%クロッシング点)
    """
    sub = D[:, substrate_idx]
    sub_max = sub.max()
    if sub_max < 1e-6:
        return depth[-1]
    # Find 50% crossing
    half = sub_max * 0.5
    idx = np.argmin(np.abs(sub - half))
    return float(depth[idx])


# =========================================================================
# extract_metrics — 任意の Result から UnifiedMetrics を抽出
# =========================================================================

def extract_metrics(
    result,
    D_true: np.ndarray,
    depth: np.ndarray,
    thk_true: float,
    solver_name: str,
    substrate_idx: int = -1,
) -> UnifiedMetrics:
    """任意のソルバー Result から統一メトリクスを抽出.

    Parameters
    ----------
    result : SolverResult | SemiFreeSolverResult | GenericLayerResult
    D_true : (n_depth, n_elem) ground truth
    depth : (n_depth,)
    thk_true : total true thickness (Å)
    solver_name : 'Free', 'SemiFree', 'GenericLayer'
    substrate_idx : 基板元素のインデックス (default: last element)
    """
    D_opt = result.D_opt
    n_elem = D_true.shape[1]

    # Ismp RMSE
    ismp_rmse = result.rmse

    # Depth RMSE per element
    rmse_per_elem = np.zeros(n_elem)
    for e in range(n_elem):
        rmse_per_elem[e] = np.sqrt(np.mean((D_opt[:, e] - D_true[:, e]) ** 2))
    depth_rmse = float(np.mean(rmse_per_elem))

    # W1 per element
    w1_per_elem = np.zeros(n_elem)
    for e in range(n_elem):
        w1 = wasserstein_distance_1d(depth, D_true[:, e], D_opt[:, e])
        w1_per_elem[e] = w1 if not np.isnan(w1) else 0.0
    w1_mean = float(np.mean(w1_per_elem))

    # Thickness estimation
    if hasattr(result, 'layer_thicknesses') and result.layer_thicknesses is not None:
        # GenericLayerSolver
        thk_est = float(np.sum(result.layer_thicknesses))
    elif hasattr(result, 'thickness') and result.thickness is not None:
        # SemiFreeSolver
        thk_est = float(result.thickness)
    else:
        # FreeSolver: estimate from profile
        thk_est = estimate_thickness_from_profile(depth, D_opt, substrate_idx)

    thickness_err = abs(thk_est - thk_true)

    # DOF estimation
    if hasattr(result, 'layer_thicknesses') and result.layer_thicknesses is not None:
        n_dof = len(result.layer_thicknesses) * 2  # thicknesses + widths
    elif hasattr(result, 'thickness'):
        n_free = D_opt.shape[0] * max(1, n_elem - 2) + 3  # free conc + thk + abrp + sub_ratio
        n_dof = n_free
    else:
        n_dof = D_opt.shape[0] * n_elem  # full DOF

    return UnifiedMetrics(
        ismp_rmse=ismp_rmse,
        depth_rmse=depth_rmse,
        depth_rmse_per_elem=rmse_per_elem,
        thickness_est=thk_est,
        thickness_err=thickness_err,
        w1_mean=w1_mean,
        w1_per_elem=w1_per_elem,
        solver_name=solver_name,
        elapsed_time=result.elapsed_time,
        success=result.success,
        n_dof=n_dof,
    )


def aggregate_metrics(
    metrics_list: List[UnifiedMetrics],
    thk_true: float,
) -> AggregatedMetrics:
    """Multi-trial の UnifiedMetrics を集約.

    Note: success=False でも有限な RMSE がある結果は「有効」として扱う。
    scipy の収束判定が厳しく success=False でも実用的な解が得られることが多い。
    """
    successful = [m for m in metrics_list if not np.isnan(m.depth_rmse)]
    if not successful:
        n_elem = metrics_list[0].depth_rmse_per_elem.shape[0] if metrics_list else 0
        return AggregatedMetrics(
            solver_name=metrics_list[0].solver_name if metrics_list else 'unknown',
            n_trials=len(metrics_list),
            ismp_rmse_mean=np.nan, ismp_rmse_std=np.nan,
            depth_rmse_mean=np.nan, depth_rmse_std=np.nan,
            w1_mean_mean=np.nan, w1_mean_std=np.nan,
            sigma_dthk=np.nan, thk_bias=np.nan,
            elapsed_mean=np.nan, success_rate=0.0, n_dof=0,
            depth_rmse_per_elem=np.full(n_elem, np.nan),
            w1_per_elem=np.full(n_elem, np.nan),
        )

    ismp = [m.ismp_rmse for m in successful]
    drmse = [m.depth_rmse for m in successful]
    w1m = [m.w1_mean for m in successful]
    thks = [m.thickness_est for m in successful]

    return AggregatedMetrics(
        solver_name=successful[0].solver_name,
        n_trials=len(metrics_list),
        ismp_rmse_mean=float(np.mean(ismp)),
        ismp_rmse_std=float(np.std(ismp)),
        depth_rmse_mean=float(np.mean(drmse)),
        depth_rmse_std=float(np.std(drmse)),
        w1_mean_mean=float(np.mean(w1m)),
        w1_mean_std=float(np.std(w1m)),
        sigma_dthk=float(np.std(thks)),
        thk_bias=float(np.mean(thks) - thk_true),
        elapsed_mean=float(np.mean([m.elapsed_time for m in successful])),
        success_rate=sum(1 for m in metrics_list if m.success) / len(metrics_list),
        n_dof=successful[0].n_dof,
        depth_rmse_per_elem=np.mean([m.depth_rmse_per_elem for m in successful], axis=0),
        w1_per_elem=np.mean([m.w1_per_elem for m in successful], axis=0),
    )


# =========================================================================
# ProblemScenario — 問題定義 + 合成データ生成
# =========================================================================

# Material library: (name, element_indices, stoichiometry)
# Index convention: elements are assigned sequentially in order of layer definition
LAYER_LIBRARY = {
    # Single elements
    'C':      {'stoich': {'C': 1.0}, 'compound': 'C'},
    'Si':     {'stoich': {'Si': 1.0}, 'compound': 'Si'},
    'Ge':     {'stoich': {'Ge': 1.0}, 'compound': 'Ge'},
    'Cu':     {'stoich': {'Cu': 1.0}, 'compound': 'Cu'},
    'Au':     {'stoich': {'Au': 1.0}, 'compound': 'Au'},
    'Ni':     {'stoich': {'Ni': 1.0}, 'compound': 'Ni'},
    'Fe':     {'stoich': {'Fe': 1.0}, 'compound': 'Fe'},
    'Cr':     {'stoich': {'Cr': 1.0}, 'compound': 'Cr'},
    'Ti_metal': {'stoich': {'Ti_m': 1.0}, 'compound': 'Ti'},
    'W':      {'stoich': {'W': 1.0}, 'compound': 'W'},
    'Ta_metal': {'stoich': {'Ta_m': 1.0}, 'compound': 'Ta'},
    # Binary oxides (cation + O)
    'Al2O3':  {'stoich': {'Al': 2.0, 'O_Al2O3': 3.0}, 'compound': 'Al2O3'},
    'TiO2':   {'stoich': {'Ti': 1.0, 'O_TiO2': 2.0}, 'compound': 'TiO2'},
    'SiO2':   {'stoich': {'Si_ox': 1.0, 'O_SiO2': 2.0}, 'compound': 'SiO2'},
    'ZrO2':   {'stoich': {'Zr': 1.0, 'O_ZrO2': 2.0}, 'compound': 'ZrO2'},
    'Ta2O5':  {'stoich': {'Ta': 2.0, 'O_Ta2O5': 5.0}, 'compound': 'Ta2O5'},
    'GeO2':   {'stoich': {'Ge_ox': 1.0, 'O_GeO2': 2.0}, 'compound': 'GeO2'},
    # Nitride (compound lookup uses aSi3N4 since the CompoundTable split Si3N4 into a/b)
    'Si3N4':  {'stoich': {'Si_n': 3.0, 'N': 4.0}, 'compound': 'aSi3N4'},
    # Ternary oxide
    'SrTiO3': {'stoich': {'Sr': 1.0, 'Ti_sto': 1.0, 'O_STO': 3.0}, 'compound': 'SrTiO3'},
    # Carbide
    'SiC':    {'stoich': {'Si_c': 1.0, 'C_sic': 1.0}, 'compound': 'SiC'},
    # III-V
    'GaAs':   {'stoich': {'Ga': 1.0, 'As': 1.0}, 'compound': 'GaAs'},
}

# Element name → (symbol, orbital, compound for IMFP, oxidation_state)
# Keys: unique internal names; allows same element symbol in different compounds
# oxidation_state: physical charge for charge-balance constraint
#   cations: +N, anions: -N, metals/substrates: 0
ELEM_CONFIG_MAP = {
    # Light elements
    'C':       ('C',  '1s', 'C',      0),     # elemental C
    'C_sic':   ('C',  '1s', 'SiC',    0),     # SiC substrate (covalent)
    'N':       ('N',  '1s', 'aSi3N4', -3),    # N³⁻ in nitrides
    # Group 14
    'Si':      ('Si', '2p', 'Si',      0),     # elemental Si
    'Si_ox':   ('Si', '2p', 'SiO2',   +4),    # Si⁴⁺ in SiO2
    'Si_n':    ('Si', '2p', 'aSi3N4',  +4),   # Si⁴⁺ in Si3N4 (charge balance: 3×4 + 4×(-3) = 0)
    'Si_c':    ('Si', '2p', 'SiC',     0),     # SiC substrate (covalent)
    'Ge':      ('Ge', '3d', 'Ge',      0),     # elemental Ge
    'Ge_ox':   ('Ge', '3d', 'GeO2',   +4),    # Ge⁴⁺ in GeO2
    # 3d metals
    'Cr':      ('Cr', '2p', 'Cr',      0),     # elemental Cr
    'Fe':      ('Fe', '2p', 'Fe',      0),     # elemental Fe
    'Ni':      ('Ni', '2p', 'Ni',      0),     # elemental Ni
    'Cu':      ('Cu', '2p', 'Cu',      0),     # elemental Cu
    'Ti':      ('Ti', '2p', 'TiO2',   +4),    # Ti⁴⁺ in TiO2
    'Ti_m':    ('Ti', '2p', 'Ti',      0),     # elemental Ti
    'Ti_sto':  ('Ti', '2p', 'SrTiO3', +4),    # Ti⁴⁺ in SrTiO3
    # 4d/5d metals
    'Zr':      ('Zr', '3d', 'ZrO2',   +4),    # Zr⁴⁺ in ZrO2
    'Sr':      ('Sr', '3d', 'SrTiO3', +2),    # Sr²⁺ in SrTiO3
    'Ta':      ('Ta', '4f', 'Ta2O5',  +5),    # Ta⁵⁺ in Ta2O5
    'Ta_m':    ('Ta', '4f', 'Ta',      0),     # elemental Ta
    'W':       ('W',  '4f', 'W',       0),     # elemental W
    'Au':      ('Au', '4f', 'Au',      0),     # elemental Au
    # Group 13/15
    'Al':      ('Al', '2p', 'Al2O3',  +3),    # Al³⁺ in Al2O3
    'Ga':      ('Ga', '3d', 'GaAs',    0),     # GaAs substrate (covalent)
    'As':      ('As', '3d', 'GaAs',    0),     # GaAs substrate (covalent)
    # Oxygen variants (different IMFP per compound)
    'O_Al2O3': ('O',  '1s', 'Al2O3', -2),
    'O_TiO2':  ('O',  '1s', 'TiO2',  -2),
    'O_SiO2':  ('O',  '1s', 'SiO2',  -2),
    'O_ZrO2':  ('O',  '1s', 'ZrO2',  -2),
    'O_Ta2O5': ('O',  '1s', 'Ta2O5', -2),
    'O_GeO2':  ('O',  '1s', 'GeO2',  -2),
    'O_STO':   ('O',  '1s', 'SrTiO3',-2),
}

# =========================================================================
# Substrate candidates (materials that can serve as substrate)
# =========================================================================
SUBSTRATE_MATERIALS = ['Si', 'Ge', 'GaAs', 'SiC', 'Cu', 'Au', 'Ni', 'Fe', 'W']

# Film layer candidates (materials that can be stacked as films)
FILM_MATERIALS = [
    'C', 'Al2O3', 'TiO2', 'SiO2', 'ZrO2', 'Ta2O5',
    'Si3N4', 'GeO2', 'SrTiO3', 'Ni', 'Cu', 'Cr', 'Fe',
    'Ti_metal', 'Ta_metal', 'Au', 'W',
]


# =========================================================================
# Random scenario generator
# =========================================================================

def make_random_grid(
    n_scenarios: int = 100,
    seed: int = 12345,
    n_layers_range: Tuple[int, int] = (1, 5),
    noise_range: Tuple[float, float] = (0.0, 0.15),
    thickness_range: Tuple[float, float] = (5.0, 300.0),
    interface_range: Tuple[float, float] = (1.0, 20.0),
    n_angles_choices: List[int] = None,
) -> List['ProblemScenario']:
    """ランダムな元素数・元素種のシナリオを生成.

    Parameters
    ----------
    n_scenarios : 生成するシナリオ数
    seed : 乱数シード
    n_layers_range : 層数の範囲 (min, max)
    noise_range : ノイズレベルの範囲
    thickness_range : 層厚の範囲 (per layer, Å)
    interface_range : 界面幅の範囲 (Å)
    n_angles_choices : 角度数の候補リスト

    Returns
    -------
    List[ProblemScenario]
    """
    if n_angles_choices is None:
        n_angles_choices = [5, 7, 9, 14]

    rng = np.random.RandomState(seed)
    scenarios = []

    for i in range(n_scenarios):
        n_layers = rng.randint(n_layers_range[0], n_layers_range[1] + 1)

        # Pick substrate
        substrate = rng.choice(SUBSTRATE_MATERIALS)

        # Pick film layers (avoid duplicates, avoid substrate material)
        available_films = [m for m in FILM_MATERIALS if m != substrate]
        # Ensure substrate element doesn't conflict with film elements
        sub_elems = set(LAYER_LIBRARY[substrate]['stoich'].keys())

        # Filter films that share cation with substrate (oxide of same element OK)
        safe_films = []
        for m in available_films:
            film_elems = set(LAYER_LIBRARY[m]['stoich'].keys())
            # Allow if no raw element name overlap (O_ variants are fine)
            cation_overlap = film_elems & sub_elems
            # Remove O_ variants from overlap check
            real_overlap = {e for e in cation_overlap if not e.startswith('O_')}
            if not real_overlap:
                safe_films.append(m)

        if len(safe_films) < n_layers:
            safe_films = available_films  # fallback

        n_pick = min(n_layers, len(safe_films))
        film_materials = list(rng.choice(safe_films, size=n_pick, replace=False))

        # Random parameters per layer
        layers = []
        for mat in film_materials:
            thk = rng.uniform(*thickness_range)
            iface = rng.uniform(*interface_range)
            layers.append(LayerSpec(mat, thk, iface))

        noise = rng.uniform(*noise_range)
        n_angles = rng.choice(n_angles_choices)

        mat_str = '+'.join(film_materials)
        name = f"rand{i:03d}_{n_pick}L_{mat_str}_{substrate}_n{noise:.2f}_a{n_angles}"

        scenarios.append(ProblemScenario(
            name=name,
            layers=layers,
            substrate=substrate,
            noise_level=noise,
            n_angles=n_angles,
        ))

    return scenarios


@dataclass
class LayerSpec:
    """層の定義."""
    material: str            # LAYER_LIBRARY のキー
    thickness: float         # Å
    interface_width: float   # Å (FWHM)


@dataclass
class ProblemScenario:
    """ベンチマーク問題の定義."""
    name: str
    layers: List[LayerSpec]         # surface → bulk (基板除く)
    substrate: str                   # 基板材料 (LAYER_LIBRARY key)
    noise_level: float              # 0, 0.01, 0.05, 0.1
    n_angles: int                   # 角度数
    n_depth: int = 25               # 深さ点数
    photon_energy: float = 1486.6   # eV (Al Kα default)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    def generate(
        self, seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ElementConfig], dict]:
        """合成データを生成.

        Returns
        -------
        I_obs : (n_angles, n_elem) 観測強度
        D_true : (n_depth, n_elem) 真の深さプロファイル
        depth : (n_depth,) 深さグリッド
        elements : List[ElementConfig]
        metadata : dict (element_names, layer_thicknesses, etc.)
        """
        rng = np.random.RandomState(seed)

        # Determine unique elements and build element list
        elem_names, elem_configs, layer_elem_indices = self._build_elements()
        n_elem = len(elem_names)

        # Depth grid: extend beyond total thickness
        max_depth = self.total_thickness * 3.0
        step_size = max_depth / (self.n_depth - 1)
        depth = np.linspace(0, max_depth, self.n_depth)

        # Build D_true using erf profiles
        D_true = self._build_profile(depth, elem_names, layer_elem_indices)

        # Angles
        angles = np.linspace(0, 60, self.n_angles).tolist()

        # Transfer matrix and forward model
        tm = TransferMatrix(
            depth=depth, step_size=step_size,
            angles=angles, photon_energy=self.photon_energy,
        )
        tm.compute(D_true, elem_configs)
        I_true = tm.forward(D_true, normalize=True)

        # Add noise
        if self.noise_level > 0:
            noise = self.noise_level * rng.randn(*I_true.shape)
            I_obs = I_true + noise * I_true
            I_obs = np.clip(I_obs, 1e-6, None)
            I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
        else:
            I_obs = I_true.copy()

        metadata = {
            'element_names': elem_names,
            'layer_thicknesses': [l.thickness for l in self.layers],
            'layer_materials': [l.material for l in self.layers],
            'layer_elem_indices': layer_elem_indices,
            'total_thickness': self.total_thickness,
        }

        return I_obs, D_true, depth, elem_configs, metadata

    def _build_elements(self):
        """元素リストとインデックスマッピングを構築.

        O atoms from different compounds (O_Al2O3, O_TiO2, O_SiO2) are merged
        into a single 'O' element (same IMFP, same signal in XPS).
        """
        elem_names = []       # unique element names for the profile
        elem_name_to_idx = {}

        # Merge rule: O_* variants → single 'O' element
        def _canonical(name: str) -> str:
            if name.startswith('O_'):
                return 'O'
            return name

        all_layers = self.layers + [LayerSpec(self.substrate, 0, 0)]
        layer_elem_indices = []  # per-layer: dict[raw_name -> global_idx]

        for layer_spec in all_layers:
            stoich = LAYER_LIBRARY[layer_spec.material]['stoich']
            idx_map = {}
            for raw_name in stoich:
                canonical = _canonical(raw_name)
                if canonical not in elem_name_to_idx:
                    elem_name_to_idx[canonical] = len(elem_names)
                    elem_names.append(canonical)
                idx_map[raw_name] = elem_name_to_idx[canonical]
            layer_elem_indices.append(idx_map)

        # Build ElementConfig list using ELEM_CONFIG_MAP
        elem_configs = []
        for name in elem_names:
            if name in ELEM_CONFIG_MAP:
                symbol, orbital, compound, ox_state = ELEM_CONFIG_MAP[name]
            elif name == 'O':
                # Merged O: use SiO2 as representative compound for IMFP
                symbol, orbital, compound, ox_state = 'O', '1s', 'SiO2', -2
            else:
                symbol, orbital, compound, ox_state = name, '1s', name, 0
            elem_configs.append(
                ElementConfig(symbol, orbital, compound, oxidation_state=ox_state)
            )

        return elem_names, elem_configs, layer_elem_indices

    def _build_profile(self, depth, elem_names, layer_elem_indices):
        """erf プロファイルを構築."""
        n_depth = len(depth)
        n_elem = len(elem_names)
        D = np.zeros((n_depth, n_elem))

        # Compute cumulative interface positions
        z_boundaries = [0.0]
        for layer in self.layers:
            z_boundaries.append(z_boundaries[-1] + layer.thickness)

        # Film layers
        for i, layer_spec in enumerate(self.layers):
            sigma = layer_spec.interface_width / (2.0 * np.sqrt(np.log(2))) if layer_spec.interface_width > 0 else 1e-6

            # Layer occupancy: erf envelope between boundaries
            if i == 0:
                # Surface layer: 1 at top, drops at z_boundaries[1]
                envelope = 0.5 * (1.0 - erf((depth - z_boundaries[1]) / sigma))
            else:
                sigma_prev = self.layers[i-1].interface_width / (2.0 * np.sqrt(np.log(2))) if self.layers[i-1].interface_width > 0 else 1e-6
                envelope = (
                    0.5 * (1.0 + erf((depth - z_boundaries[i]) / sigma_prev))
                    * 0.5 * (1.0 - erf((depth - z_boundaries[i+1]) / sigma))
                )

            # Distribute according to stoichiometry
            stoich = LAYER_LIBRARY[layer_spec.material]['stoich']
            total_stoich = sum(stoich.values())
            idx_map = layer_elem_indices[i]
            for elem_name, ratio in stoich.items():
                D[:, idx_map[elem_name]] += envelope * (ratio / total_stoich)

        # Substrate
        sigma_last = self.layers[-1].interface_width / (2.0 * np.sqrt(np.log(2))) if self.layers else 1e-6
        sub_envelope = 0.5 * (1.0 + erf((depth - z_boundaries[-1]) / sigma_last))
        sub_stoich = LAYER_LIBRARY[self.substrate]['stoich']
        total_sub = sum(sub_stoich.values())
        sub_idx_map = layer_elem_indices[-1]
        for elem_name, ratio in sub_stoich.items():
            D[:, sub_idx_map[elem_name]] += sub_envelope * (ratio / total_sub)

        # Normalize: ensure sum=1, clip
        D = np.clip(D, 1e-6, None)
        D = D / D.sum(axis=1, keepdims=True)
        return D


# =========================================================================
# Standard problem grid
# =========================================================================

def make_standard_grid() -> List[ProblemScenario]:
    """162シナリオのフルグリッドを生成."""
    scenarios = []

    # Layer configurations
    layer_configs = {
        '1layer': [LayerSpec('SiO2', None, None)],          # thickness/width filled below
        '2layer': [LayerSpec('C', None, None), LayerSpec('SiO2', None, None)],
        '4layer': [
            LayerSpec('C', None, None),
            LayerSpec('Al2O3', None, None),
            LayerSpec('TiO2', None, None),
            LayerSpec('SiO2', None, None),
        ],
    }

    # Thickness distribution per n_layers (total → per-layer)
    thickness_split = {
        '1layer': lambda t: [t],
        '2layer': lambda t: [t * 0.2, t * 0.8],
        '4layer': lambda t: [t * 0.06, t * 0.18, t * 0.42, t * 0.34],
    }

    noise_levels = [0.0, 0.02, 0.1]
    thicknesses = [10.0, 50.0, 200.0]
    interfaces = [2.0, 10.0]
    n_angles_list = [5, 9, 14]

    for n_lay_key, noise, total_thk, iface, n_ang in product(
        layer_configs.keys(), noise_levels, thicknesses, interfaces, n_angles_list
    ):
        splits = thickness_split[n_lay_key](total_thk)
        layers = []
        for template, thk in zip(layer_configs[n_lay_key], splits):
            layers.append(LayerSpec(template.material, thk, iface))

        name = f"{n_lay_key}_thk{total_thk:.0f}_noise{noise:.2f}_iface{iface:.0f}_ang{n_ang}"
        scenarios.append(ProblemScenario(
            name=name,
            layers=layers,
            substrate='Si',
            noise_level=noise,
            n_angles=n_ang,
        ))

    return scenarios


def make_small_grid() -> List[ProblemScenario]:
    """テスト用小グリッド (18シナリオ)."""
    scenarios = []

    configs = {
        '1layer': ([LayerSpec('SiO2', 30.0, 3.0)],),
        '2layer': ([LayerSpec('C', 5.0, 2.0), LayerSpec('SiO2', 25.0, 3.0)],),
        '4layer': ([
            LayerSpec('C', 5.0, 2.0),
            LayerSpec('Al2O3', 15.0, 3.0),
            LayerSpec('TiO2', 35.0, 4.0),
            LayerSpec('SiO2', 30.0, 5.0),
        ],),
    }

    for n_lay_key, noise in product(configs.keys(), [0.0, 0.02]):
        layers = configs[n_lay_key][0]
        for n_ang in [5, 9, 14]:
            name = f"{n_lay_key}_noise{noise:.2f}_ang{n_ang}"
            scenarios.append(ProblemScenario(
                name=name,
                layers=[LayerSpec(l.material, l.thickness, l.interface_width) for l in layers],
                substrate='Si',
                noise_level=noise,
                n_angles=n_ang,
            ))

    return scenarios


# =========================================================================
# SolverRunner — 3ソルバー統一実行
# =========================================================================

class SolverRunner:
    """1つのProblemScenarioに対して3ソルバーを実行."""

    def __init__(self, method: str = 'LLS', alpha: float = 0.0, verbose: bool = False):
        self.method = method
        self.alpha = alpha
        self.verbose = verbose

    def run_all(
        self,
        scenario: ProblemScenario,
        n_trials: int = 10,
    ) -> Dict[str, AggregatedMetrics]:
        """3ソルバーを実行し、集約メトリクスを返す."""
        results = {}
        for solver_name in ['Free', 'SemiFree', 'GenericLayer']:
            metrics_list = []
            for trial in range(n_trials):
                seed = 42 + trial
                try:
                    m = self._run_single(scenario, solver_name, seed)
                    metrics_list.append(m)
                except Exception as e:
                    if self.verbose:
                        print(f"  {solver_name} trial {trial}: FAILED ({e})")
                    # Append a failed metric
                    n_elem = len(scenario._build_elements()[0])
                    metrics_list.append(UnifiedMetrics(
                        ismp_rmse=np.nan, depth_rmse=np.nan,
                        depth_rmse_per_elem=np.full(n_elem, np.nan),
                        thickness_est=np.nan, thickness_err=np.nan,
                        w1_mean=np.nan, w1_per_elem=np.full(n_elem, np.nan),
                        solver_name=solver_name, elapsed_time=0.0,
                        success=False, n_dof=0,
                    ))

            results[solver_name] = aggregate_metrics(
                metrics_list, scenario.total_thickness,
            )

        return results

    def _run_single(
        self, scenario: ProblemScenario, solver_name: str, seed: int,
    ) -> UnifiedMetrics:
        """1 trial を実行."""
        from deppro.inverse_solver import (
            FreeSolver, SemiFreeSolver, SemiFreeSolverConfig,
            GenericLayerSolver, GenericLayerConfig, LayerDefinition,
            SolverConfig,
        )

        I_obs, D_true, depth, elements, meta = scenario.generate(seed=seed)
        n_elem = D_true.shape[1]
        elem_names = meta['element_names']
        layer_elem_indices = meta['layer_elem_indices']

        # Find substrate element index
        sub_stoich = LAYER_LIBRARY[scenario.substrate]['stoich']
        sub_elem_name = list(sub_stoich.keys())[0]
        substrate_idx = elem_names.index(sub_elem_name)

        # Transfer matrix (already built in generate(), rebuild for solver)
        step_size = depth[1] - depth[0] if len(depth) > 1 else 1.0
        angles = np.linspace(0, 60, scenario.n_angles).tolist()
        tm = TransferMatrix(
            depth=depth, step_size=step_size,
            angles=angles, photon_energy=scenario.photon_energy,
        )
        tm.compute(D_true, elements)

        solver_config = SolverConfig(
            method=self.method,
            alpha=self.alpha,
            scipy_method='SLSQP',
            enable_fallback=True,
        )

        if solver_name == 'Free':
            result = self._run_free(
                I_obs, tm, elements, solver_config, D_true, substrate_idx,
            )
        elif solver_name == 'SemiFree':
            result = self._run_semifree(
                I_obs, tm, elements, solver_config,
                elem_names, layer_elem_indices, substrate_idx, scenario,
            )
        elif solver_name == 'GenericLayer':
            result = self._run_generic_layer(
                I_obs, tm, elements, solver_config,
                elem_names, layer_elem_indices, substrate_idx, scenario,
            )
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        return extract_metrics(
            result, D_true, depth,
            thk_true=scenario.total_thickness,
            solver_name=solver_name,
            substrate_idx=substrate_idx,
        )

    def _run_free(self, I_obs, tm, elements, config, D_true, substrate_idx):
        """FreeSolver: 事前知識なし."""
        from deppro.inverse_solver import FreeSolver

        # Uniform prior
        n_depth, n_elem = D_true.shape
        D_prior = np.full((n_depth, n_elem), 1.0 / n_elem)

        solver = FreeSolver(tm, config, D_prior, elements=elements)
        return solver.solve(I_obs, substrate_indices=[substrate_idx], verbose=False)

    def _run_semifree(
        self, I_obs, tm, elements, config,
        elem_names, layer_elem_indices, substrate_idx, scenario,
    ):
        """SemiFreeSolver: 基板元素のみ指定."""
        from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig

        # Determine free/linked/substrate indices
        # substrate: substrate element(s) — from scenario substrate
        # linked: anions (O, N) if present — charge-balanced
        # free: everything else (cations, metals)
        anion_names = {'O', 'N'}  # charge-balanced candidates
        linked_indices = []
        free_indices = []
        for i, name in enumerate(elem_names):
            if i == substrate_idx:
                continue
            elif name in anion_names:
                linked_indices.append(i)
            else:
                free_indices.append(i)

        # If no free indices (all are substrate + anions), move anions to free
        if not free_indices and linked_indices:
            free_indices = linked_indices
            linked_indices = []

        semi_config = SemiFreeSolverConfig(
            thickness_init=scenario.total_thickness * 1.5,  # overestimate
            abruptness_init=5.0,
            free_indices=free_indices,
            linked_indices=linked_indices if linked_indices else None,
            substrate_indices=[substrate_idx],
            thickness_bounds=(1.0, max(50.0, scenario.total_thickness * 5)),
        )

        solver = SemiFreeSolver(tm, config, semi_config, elements=elements)
        return solver.solve(I_obs, verbose=False)

    def _run_generic_layer(
        self, I_obs, tm, elements, config,
        elem_names, layer_elem_indices, substrate_idx, scenario,
    ):
        """GenericLayerSolver: 正しい層構造を提供."""
        from deppro.inverse_solver import (
            GenericLayerSolver, GenericLayerConfig, LayerDefinition,
        )

        layers = []
        for i, layer_spec in enumerate(scenario.layers):
            stoich = LAYER_LIBRARY[layer_spec.material]['stoich']
            idx_map = layer_elem_indices[i]
            stoich_by_idx = {idx_map[name]: ratio for name, ratio in stoich.items()}
            elem_indices = list(stoich_by_idx.keys())
            layers.append(LayerDefinition(
                name=layer_spec.material,
                element_indices=elem_indices,
                stoichiometry=stoich_by_idx,
            ))

        # Substrate
        sub_stoich = LAYER_LIBRARY[scenario.substrate]['stoich']
        sub_idx_map = layer_elem_indices[-1]
        sub_stoich_by_idx = {sub_idx_map[name]: ratio for name, ratio in sub_stoich.items()}
        substrate_def = LayerDefinition(
            name=scenario.substrate,
            element_indices=list(sub_stoich_by_idx.keys()),
            stoichiometry=sub_stoich_by_idx,
        )

        # Initial: overestimate by 50%
        init_thk = [l.thickness * 1.5 for l in scenario.layers]
        init_widths = [l.interface_width for l in scenario.layers]

        layer_config = GenericLayerConfig(
            layers=layers,
            substrate=substrate_def,
            initial_thicknesses=init_thk,
            initial_widths=init_widths,
            thickness_bounds=(0.5, max(500.0, scenario.total_thickness * 5)),
        )

        solver = GenericLayerSolver(tm, config, layer_config, elements=elements)
        return solver.solve(I_obs, verbose=False)


# =========================================================================
# Full benchmark runner (synthetic scenarios)
# =========================================================================

def run_benchmark(
    scenarios: List[ProblemScenario],
    n_trials: int = 10,
    method: str = 'LLS',
    alpha: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Dict[str, AggregatedMetrics]]:
    """全シナリオで3ソルバーベンチマークを実行.

    Returns
    -------
    results : {scenario_name: {solver_name: AggregatedMetrics}}
    """
    runner = SolverRunner(method=method, alpha=alpha, verbose=verbose)
    results = {}

    for i, scenario in enumerate(scenarios):
        if verbose:
            print(f"[{i+1}/{len(scenarios)}] {scenario.name} "
                  f"({scenario.n_layers}L, noise={scenario.noise_level}, "
                  f"thk={scenario.total_thickness:.0f}Å, ang={scenario.n_angles})")
        try:
            results[scenario.name] = runner.run_all(scenario, n_trials=n_trials)
        except Exception as e:
            if verbose:
                print(f"  SCENARIO FAILED: {e}")
            results[scenario.name] = {}

    return results


# =========================================================================
# H5-based benchmark — 同一 Dstk→Ismp で 5手法比較
# =========================================================================

ALL_SOLVER_NAMES = ['Free', 'SemiFree', 'GenericLayer', 'NN', 'Hybrid']


def run_h5_benchmark(
    h5_path: str,
    nn_model_path: Optional[str] = None,
    solvers: Optional[List[str]] = None,
    max_pixels: int = 100,
    method: str = 'LLS',
    alpha: float = 0.0,
    verbose: bool = True,
) -> Dict[str, AggregatedMetrics]:
    """H5ベンチマークデータで全ソルバーを比較.

    Parameters
    ----------
    h5_path : H5ベンチマークファイル (Dstk + Xstk + Drlt)
    nn_model_path : NN モデル (.npz). None の場合 NN/Hybrid をスキップ.
    solvers : 比較するソルバー名のリスト (default: 全5手法)
    max_pixels : 評価ピクセル数 (Free/SemiFree/GenericLayerは遅いので制限)
    method : 正則化手法
    alpha : 正則化強度

    Returns
    -------
    {solver_name: AggregatedMetrics}
    """
    from deppro.benchmarks.benchmark_dstk import load_dstk_data, estimate_thickness

    if solvers is None:
        solvers = list(ALL_SOLVER_NAMES)

    # Remove NN/Hybrid if no model
    if nn_model_path is None:
        solvers = [s for s in solvers if s not in ('NN', 'Hybrid')]

    data = load_dstk_data(h5_path)
    n_pixels = min(max_pixels, data.n_pixels)

    if verbose:
        print(f"H5 Benchmark: {h5_path}")
        print(f"  {data.n_pixels} pixels ({n_pixels} evaluated)")
        print(f"  {data.n_elements} elements: {data.element_names}")
        print(f"  depth: {data.n_depth} pts, angles: {data.n_angles}")
        print(f"  Solvers: {solvers}")

    # Load NN model if needed
    nn_model = None
    if nn_model_path and any(s in solvers for s in ('NN', 'Hybrid')):
        nn_model = _load_nn_model(nn_model_path)

    # Build transfer matrix once
    tm = TransferMatrix(
        depth=data.depth, step_size=data.step_size,
        angles=data.angles.tolist(),
        photon_energy=data.photon_energy,
    )
    tm.compute(data.Dstk[0], data.elements)
    # Use pre-computed transfer matrix from H5
    tm.T = data.trans_mat  # (n_angles, n_depth, n_elements)

    results = {}
    for solver_name in solvers:
        if verbose:
            print(f"\n  Running {solver_name}...", end=' ', flush=True)

        metrics_list = []
        t0 = time.time()

        for px in range(n_pixels):
            I_obs = data.Xstk[px]       # (n_angles, n_elem)
            D_true = data.Dstk[px]      # (n_depth, n_elem)
            thk_true = estimate_thickness(D_true, data.depth)

            try:
                m = _run_h5_single(
                    solver_name, I_obs, D_true, data, tm,
                    thk_true, nn_model, method, alpha,
                )
                metrics_list.append(m)
            except Exception as e:
                if verbose and px == 0:
                    print(f"pixel {px} FAILED: {e}")
                metrics_list.append(UnifiedMetrics(
                    ismp_rmse=np.nan, depth_rmse=np.nan,
                    depth_rmse_per_elem=np.full(data.n_elements, np.nan),
                    thickness_est=np.nan, thickness_err=np.nan,
                    w1_mean=np.nan, w1_per_elem=np.full(data.n_elements, np.nan),
                    solver_name=solver_name, elapsed_time=0.0,
                    success=False, n_dof=0,
                ))

        elapsed = time.time() - t0
        if verbose:
            ok = sum(1 for m in metrics_list if m.success)
            print(f"{ok}/{n_pixels} ok, {elapsed:.1f}s")

        mean_thk = np.nanmean([
            estimate_thickness(data.Dstk[px], data.depth) for px in range(n_pixels)
        ])
        results[solver_name] = aggregate_metrics(metrics_list, mean_thk)

    return results


def _run_h5_single(
    solver_name, I_obs, D_true, data, tm,
    thk_true, nn_model, method, alpha,
) -> UnifiedMetrics:
    """H5データの1ピクセルを1ソルバーで解析."""
    from deppro.inverse_solver import (
        FreeSolver, SemiFreeSolver, SemiFreeSolverConfig,
        SolverConfig,
    )
    from deppro.benchmarks.benchmark_dstk import estimate_thickness

    n_depth, n_elem = D_true.shape
    substrate_idx = data.substrate_indices[0]

    config = SolverConfig(
        method=method, alpha=alpha,
        scipy_method='SLSQP', enable_fallback=True,
    )

    t0 = time.time()

    if solver_name == 'Free':
        D_prior = np.full((n_depth, n_elem), 1.0 / n_elem)
        solver = FreeSolver(tm, config, D_prior, elements=data.elements)
        result = solver.solve(I_obs, substrate_indices=data.substrate_indices, verbose=False)

    elif solver_name == 'SemiFree':
        semi_config = SemiFreeSolverConfig(
            thickness_init=thk_true * 1.5,
            abruptness_init=5.0,
            free_indices=data.free_indices,
            linked_indices=None,  # H5 data handles linking internally
            substrate_indices=data.substrate_indices,
            thickness_bounds=(5.0, thk_true * 5),
        )
        solver = SemiFreeSolver(tm, config, semi_config, elements=data.elements)
        result = solver.solve(I_obs, verbose=False)

    elif solver_name == 'NN':
        result = _run_nn(I_obs, D_true, data, nn_model, tm)

    elif solver_name == 'Hybrid':
        result = _run_hybrid(I_obs, D_true, data, nn_model, tm, config)

    else:
        raise ValueError(f"Unknown solver: {solver_name} (GenericLayer requires layer structure info)")

    elapsed = time.time() - t0

    return extract_metrics(
        result, D_true, data.depth,
        thk_true=thk_true,
        solver_name=solver_name,
        substrate_idx=substrate_idx,
    )


def _load_nn_model(model_path: str):
    """NN モデルをロード (既存の load_model を使用)."""
    from deppro.inference.train import load_model
    return load_model(model_path)


class _NNResult:
    """NN結果を SolverResult 互換にするアダプタ."""

    def __init__(self, D_opt, I_opt, chi2, elapsed_time, success, thickness=None):
        self.D_opt = D_opt
        self.I_opt = I_opt
        self.chi2 = chi2
        self.elapsed_time = elapsed_time
        self.success = success
        self.thickness = thickness
        self.n_iterations = 0
        self.regularization = 0.0
        self.objective = chi2

    @property
    def rmse(self):
        return np.sqrt(self.chi2 / max(self.I_opt.size, 1))


def _run_nn(I_obs, D_true, data, nn_model, tm):
    """NN推論 (バッチサイズ1)."""
    import mlx.core as mx
    from deppro.inference.postprocess import enforce_constraints

    t0 = time.time()

    # Flatten input
    I_flat = mx.array(I_obs.reshape(1, -1).astype(np.float32))
    D_flat = np.array(nn_model(I_flat))[0]
    mx.eval(nn_model.parameters())

    # Post-process
    n_depth, n_elem = D_true.shape
    D_flat = enforce_constraints(D_flat[np.newaxis, :], n_depth, n_elem)[0]
    D_opt = D_flat.reshape(n_depth, n_elem)

    # Forward model for chi2
    I_pred = tm.forward(D_opt, normalize=True)
    chi2 = float(np.sum((I_obs - I_pred) ** 2))

    elapsed = time.time() - t0
    return _NNResult(D_opt, I_pred, chi2, elapsed, True)


def _run_hybrid(I_obs, D_true, data, nn_model, tm, config):
    """Hybrid: NN → SemiFree warm start."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig
    from deppro.hybrid_solver import solve_hybrid

    semi_config = SemiFreeSolverConfig(
        thickness_init=100.0,
        abruptness_init=5.0,
        free_indices=data.free_indices,
        linked_indices=None,
        substrate_indices=data.substrate_indices,
        thickness_bounds=(5.0, 500.0),
    )

    solver = SemiFreeSolver(tm, config, semi_config, elements=data.elements)
    result = solve_hybrid(I_obs, solver, nn_model, maxiter=10, mode='hybrid')

    # Adapt HybridResult → extract_metrics compatible
    I_pred = tm.forward(result.D_opt, normalize=True)
    chi2 = float(np.sum((I_obs - I_pred) ** 2))

    return _NNResult(
        D_opt=result.D_opt, I_opt=I_pred, chi2=chi2,
        elapsed_time=result.time_nn_s + result.time_slsqp_s,
        success=result.success, thickness=result.thickness,
    )
