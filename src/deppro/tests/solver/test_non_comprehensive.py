#!/usr/bin/env python3
"""
NON系多層構造の包括的検証

目的:
- 5種類のX線源（800eV〜Ga-Kα）での系統的比較
- ノイズ耐性（2%, 5%, 10%）
- 濃度変化（N/O混合相）
- 非周期構造
- 普遍的な知見の抽出

X線源:
- Synchrotron 800eV: IMFP ~15Å (超表面敏感)
- Al-Kα 1486.6eV: IMFP ~35Å (標準XPS)
- Ag-Lα 2984.2eV: IMFP ~60Å (中間)
- Cr-Kα 5414.7eV: IMFP ~105Å (HAXPES)
- Ga-Kα 9251.7eV: IMFP ~166Å (HAXPES)
"""

import pytest
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 物性パラメータ (TPP-2M)
# =============================================================================

def imfp_tpp2m(KE: float, Nv: int, density: float, Aw: float, Eg: float) -> float:
    """TPP-2M formula for IMFP calculation"""
    if KE <= 50:
        return 5.0  # Minimum IMFP
    gamma = 0.191 * density ** (-0.50)
    Ep = 28.8 * (density * Nv / Aw) ** 0.5
    beta = -0.10 + 0.944 / (Ep**2 + Eg**2) ** 0.5 + 0.069 * density ** 0.1
    U = Ep**2 / 829.4
    C = 1.97 - 0.91 * U
    D = 53.4 - 20.8 * U
    return KE / (Ep**2 * (beta * np.log(gamma * KE) - C / KE + D / KE**2))


# Material parameters: (Nv, density, Aw, Eg)
MATERIALS = {
    'SiO2':   (16, 2.2, 60.0843, 8.9),
    'Si3N4': (32, 3.2, 140.28346, 5.6),
    'Si':     (4, 2.33, 28.09, 1.12),
}

# X-ray sources (5 sources spanning wide IMFP range)
XRAY_SOURCES = {
    'Sync-800eV': 800.0,      # Synchrotron (ultra surface sensitive)
    'Al-Kα': 1486.6,          # Standard XPS
    'Ag-Lα': 2984.2,          # Intermediate
    'Cr-Kα': 5414.7,          # HAXPES
    'Ga-Kα': 9251.7,          # HAXPES (bulk sensitive)
}

# Core level binding energies
BINDING_ENERGIES = {
    'Si_2p': 103,
    'Si_2s': 154,
    'Si_1s': 1839,
    'O_1s': 532,
    'N_1s': 398,
}


def get_imfp(KE: float, material: str) -> float:
    """Get IMFP for a specific material"""
    params = MATERIALS[material]
    return imfp_tpp2m(KE, *params)


def get_imfp_mixed(KE: float, x_O: float) -> float:
    """Get IMFP for SiO2(x)/Si3N4(1-x) mixture"""
    imfp_sio2 = get_imfp(KE, 'SiO2')
    imfp_si3n4 = get_imfp(KE, 'Si3N4')
    return imfp_sio2 * x_O + imfp_si3n4 * (1 - x_O)


def get_available_core_levels(hv: float) -> Dict[str, float]:
    """Get available core levels for given photon energy"""
    available = {}
    for name, BE in BINDING_ENERGIES.items():
        KE = hv - BE
        if KE > 50:  # Minimum usable KE
            available[name] = KE
    return available


# =============================================================================
# 多層膜モデル（拡張版：非周期対応）
# =============================================================================

@dataclass
class Layer:
    """単一層の定義"""
    thickness: float  # Å
    x_O: float  # O fraction (1.0=SiO2, 0.0=Si3N4)

    @property
    def composition_str(self) -> str:
        if self.x_O >= 0.95:
            return 'O'
        elif self.x_O <= 0.05:
            return 'N'
        else:
            return f'M{int(self.x_O*100)}'  # Mixed


@dataclass
class MultiLayerStructure:
    """多層膜構造（非周期対応）"""
    layers: List[Layer] = field(default_factory=list)

    @classmethod
    def from_pattern(cls, pattern: str, layer_thickness: float) -> 'MultiLayerStructure':
        """Create from pattern string (e.g., 'NONONON')"""
        layers = []
        for char in pattern:
            x_O = 1.0 if char == 'O' else 0.0
            layers.append(Layer(layer_thickness, x_O))
        return cls(layers)

    @classmethod
    def from_thicknesses(cls, pattern: str, thicknesses: List[float]) -> 'MultiLayerStructure':
        """Create non-periodic structure"""
        layers = []
        for char, d in zip(pattern, thicknesses):
            x_O = 1.0 if char == 'O' else 0.0
            layers.append(Layer(d, x_O))
        return cls(layers)

    @classmethod
    def from_mixed(cls, pattern: str, layer_thickness: float,
                   x_O_values: Dict[str, float]) -> 'MultiLayerStructure':
        """Create with mixed compositions"""
        layers = []
        for char in pattern:
            x_O = x_O_values.get(char, 1.0 if char == 'O' else 0.0)
            layers.append(Layer(layer_thickness, x_O))
        return cls(layers)

    def get_layers_tuples(self) -> List[Tuple[float, float]]:
        return [(l.thickness, l.x_O) for l in self.layers]

    def total_thickness(self) -> float:
        return sum(l.thickness for l in self.layers)

    def n_layers(self) -> int:
        return len(self.layers)

    def pattern_str(self) -> str:
        return ''.join(l.composition_str for l in self.layers)

    def avg_thickness(self) -> float:
        return self.total_thickness() / self.n_layers() if self.layers else 0


# =============================================================================
# 順問題（ARXPS シミュレーション）
# =============================================================================

def simulate_arxps(layers: List[Tuple[float, float]],
                   angles: np.ndarray,
                   hv: float,
                   include_substrate: bool = True) -> Dict[str, np.ndarray]:
    """
    Simulate ARXPS for NON structure (fast version).

    Returns dict with element intensities.
    """
    # Determine available core levels
    available = get_available_core_levels(hv)

    # Must have at least O_1s or N_1s
    if 'O_1s' not in available and 'N_1s' not in available:
        return None

    # Select Si core level
    if 'Si_2p' in available:
        KE_Si = available['Si_2p']
        si_core = 'Si_2p'
    elif 'Si_2s' in available:
        KE_Si = available['Si_2s']
        si_core = 'Si_2s'
    else:
        return None  # Cannot measure Si

    KE_O = available.get('O_1s', 0)
    KE_N = available.get('N_1s', 0)

    n_angles = len(angles)
    I_Si = np.zeros(n_angles)
    I_O = np.zeros(n_angles)
    I_N = np.zeros(n_angles)
    I_Si_bulk = np.zeros(n_angles)

    # Precompute layer boundaries
    z_starts = [0]
    for d, _ in layers:
        z_starts.append(z_starts[-1] + d)

    for i_ang, theta in enumerate(angles):
        cos_theta = np.cos(np.radians(theta))
        if cos_theta < 0.05:
            continue

        # Integrate through each layer (5 points per layer)
        for i_layer, (d, x_O) in enumerate(layers):
            z_start = z_starts[i_layer]

            imfp_Si = get_imfp_mixed(KE_Si, x_O) if KE_Si > 0 else 1e10
            imfp_O = get_imfp_mixed(KE_O, x_O) if KE_O > 0 else 1e10
            imfp_N = get_imfp_mixed(KE_N, x_O) if KE_N > 0 else 1e10

            for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                z = z_start + d * frac

                # Attenuation
                atten_Si = atten_O = atten_N = 0
                z_temp = 0

                for j, (d2, x_O2) in enumerate(layers):
                    if z_temp >= z:
                        break
                    d_eff = min(d2, z - z_temp)

                    if KE_Si > 0:
                        atten_Si += d_eff / (get_imfp_mixed(KE_Si, x_O2) * cos_theta)
                    if KE_O > 0:
                        atten_O += d_eff / (get_imfp_mixed(KE_O, x_O2) * cos_theta)
                    if KE_N > 0:
                        atten_N += d_eff / (get_imfp_mixed(KE_N, x_O2) * cos_theta)
                    z_temp += d2

                dz = d / 5
                # Stoichiometry: SiO2 = Si + 2O, Si3N4 = 3Si + 4N
                # Interpolate for mixed compositions
                si_coeff = 1.0 * x_O + 3.0 * (1 - x_O)  # Si per formula unit
                o_coeff = 2.0 * x_O
                n_coeff = 4.0 * (1 - x_O)

                I_Si[i_ang] += si_coeff * np.exp(-atten_Si) * dz
                if KE_O > 0:
                    I_O[i_ang] += o_coeff * np.exp(-atten_O) * dz
                if KE_N > 0:
                    I_N[i_ang] += n_coeff * np.exp(-atten_N) * dz

        # Substrate
        if include_substrate and KE_Si > 0:
            imfp_Si_bulk = get_imfp(KE_Si, 'Si')
            atten_total = sum(d / (get_imfp_mixed(KE_Si, x_O) * cos_theta)
                             for d, x_O in layers)
            I_Si_bulk[i_ang] = imfp_Si_bulk * np.exp(-atten_total) * cos_theta

    return {
        'Si': I_Si,
        'O': I_O if KE_O > 0 else None,
        'N': I_N if KE_N > 0 else None,
        'Si_bulk': I_Si_bulk,
    }


def add_noise(signal: np.ndarray, noise_level: float, seed: int = None) -> np.ndarray:
    """Add Gaussian noise"""
    if seed is not None:
        np.random.seed(seed)
    if signal is None or np.max(signal) == 0:
        return signal
    return signal + np.random.normal(0, noise_level * np.max(signal), signal.shape)


# =============================================================================
# フィッティング
# =============================================================================

def fit_periodic_structure(I_obs: Dict[str, np.ndarray],
                          pattern: str,
                          angles: np.ndarray,
                          hv: float,
                          d_bounds: Tuple[float, float] = (3.0, 100.0)) -> Tuple[float, float]:
    """Fit uniform layer thickness for periodic structure"""

    def objective(params):
        d = params[0]
        if d < d_bounds[0] or d > d_bounds[1]:
            return 1e10

        structure = MultiLayerStructure.from_pattern(pattern, d)
        I_calc = simulate_arxps(structure.get_layers_tuples(), angles, hv)

        if I_calc is None:
            return 1e10

        residual = 0
        n_terms = 0

        for elem in ['Si', 'O', 'N', 'Si_bulk']:
            if elem in I_obs and I_obs[elem] is not None and elem in I_calc and I_calc[elem] is not None:
                obs = I_obs[elem]
                calc = I_calc[elem]
                if np.max(obs) > 0 and np.max(calc) > 0:
                    obs_n = obs / np.max(obs)
                    calc_n = calc / np.max(calc)
                    residual += np.sum((obs_n - calc_n)**2)
                    n_terms += 1

        return residual / max(n_terms, 1)

    best_result = None
    best_fun = 1e10

    for d0 in np.linspace(d_bounds[0], d_bounds[1], 5):
        try:
            result = minimize(objective, [d0], method='L-BFGS-B',
                            bounds=[d_bounds], options={'maxiter': 50})
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
        except:
            pass

    if best_result is None:
        return np.nan, 1e10

    return best_result.x[0], best_result.fun


def fit_variable_structure(I_obs: Dict[str, np.ndarray],
                          pattern: str,
                          angles: np.ndarray,
                          hv: float,
                          d_init: List[float],
                          d_bounds: Tuple[float, float] = (3.0, 100.0)) -> Tuple[List[float], float]:
    """Fit individual layer thicknesses for non-periodic structure"""

    n_layers = len(pattern)

    def objective(params):
        thicknesses = list(params)
        if any(d < d_bounds[0] or d > d_bounds[1] for d in thicknesses):
            return 1e10

        structure = MultiLayerStructure.from_thicknesses(pattern, thicknesses)
        I_calc = simulate_arxps(structure.get_layers_tuples(), angles, hv)

        if I_calc is None:
            return 1e10

        residual = 0
        n_terms = 0

        for elem in ['Si', 'O', 'N', 'Si_bulk']:
            if elem in I_obs and I_obs[elem] is not None and elem in I_calc and I_calc[elem] is not None:
                obs = I_obs[elem]
                calc = I_calc[elem]
                if np.max(obs) > 0 and np.max(calc) > 0:
                    obs_n = obs / np.max(obs)
                    calc_n = calc / np.max(calc)
                    residual += np.sum((obs_n - calc_n)**2)
                    n_terms += 1

        return residual / max(n_terms, 1)

    bounds = [d_bounds] * n_layers

    best_result = None
    best_fun = 1e10

    # Multi-start
    for scale in [0.7, 1.0, 1.3]:
        x0 = [d * scale for d in d_init]
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                            bounds=bounds, options={'maxiter': 100})
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
        except:
            pass

    if best_result is None:
        return [np.nan] * n_layers, 1e10

    return list(best_result.x), best_result.fun


# =============================================================================
# テスト関数群
# =============================================================================

def run_xray_sources(structure: MultiLayerStructure,
                      noise: float = 0.02,
                      n_trials: int = 10,
                      use_etch: bool = False) -> Dict:
    """Compare X-ray sources for a given structure"""

    angles = np.array([0, 15, 30, 45, 55, 65, 75])
    pattern = structure.pattern_str()
    true_d = structure.avg_thickness()
    layers = structure.get_layers_tuples()

    results = {}

    for source_name, hv in XRAY_SOURCES.items():
        # Check if core levels are available
        available = get_available_core_levels(hv)
        if 'O_1s' not in available:
            continue  # Skip if O 1s not measurable

        KE_ref = available.get('Si_2p', available.get('Si_2s', 0))
        imfp = get_imfp_mixed(KE_ref, 0.5) if KE_ref > 0 else 0

        d_errors = []
        chi2_values = []

        for trial in range(n_trials):
            if use_etch:
                # Etch + ARXPS
                etch_step = true_d
                etch_depths = np.arange(0, structure.total_thickness(), etch_step)

                all_I_obs = []
                for etch_depth in etch_depths:
                    remaining = []
                    accumulated = 0.0
                    for d, x_O in layers:
                        if accumulated + d <= etch_depth:
                            accumulated += d
                            continue
                        elif accumulated < etch_depth:
                            d_remaining = d - (etch_depth - accumulated)
                            remaining.append((d_remaining, x_O))
                            accumulated = etch_depth
                        else:
                            remaining.append((d, x_O))
                        accumulated += d

                    if not remaining:
                        continue

                    I_sim = simulate_arxps(remaining, angles, hv)
                    if I_sim is None:
                        continue

                    I_obs = {k: add_noise(v, noise, seed=trial*1000+int(etch_depth))
                            for k, v in I_sim.items()}
                    all_I_obs.append((etch_depth, I_obs, remaining))

                # Global fit with etch data
                def objective_etch(params):
                    d = params[0]
                    if d < 3.0 or d > 100.0:
                        return 1e10

                    total_resid = 0
                    n_terms = 0

                    for etch_depth, I_obs, _ in all_I_obs:
                        # Reconstruct remaining layers
                        test_structure = MultiLayerStructure.from_pattern(pattern, d)
                        remaining = []
                        accumulated = 0.0
                        for d_layer, x_O in test_structure.get_layers_tuples():
                            if accumulated + d_layer <= etch_depth:
                                accumulated += d_layer
                                continue
                            elif accumulated < etch_depth:
                                d_remaining = d_layer - (etch_depth - accumulated)
                                remaining.append((d_remaining, x_O))
                                accumulated = etch_depth
                            else:
                                remaining.append((d_layer, x_O))
                            accumulated += d_layer

                        if not remaining:
                            continue

                        I_calc = simulate_arxps(remaining, angles, hv)
                        if I_calc is None:
                            continue

                        for elem in ['Si', 'O', 'N', 'Si_bulk']:
                            if elem in I_obs and I_obs[elem] is not None:
                                if elem in I_calc and I_calc[elem] is not None:
                                    obs = I_obs[elem]
                                    calc = I_calc[elem]
                                    if np.max(obs) > 0 and np.max(calc) > 0:
                                        obs_n = obs / np.max(obs)
                                        calc_n = calc / np.max(calc)
                                        total_resid += np.sum((obs_n - calc_n)**2)
                                        n_terms += 1

                    return total_resid / max(n_terms, 1)

                best_fun = 1e10
                best_d = np.nan
                for d0 in [true_d * 0.5, true_d, true_d * 1.5]:
                    try:
                        result = minimize(objective_etch, [d0], method='L-BFGS-B',
                                        bounds=[(3.0, 100.0)], options={'maxiter': 50})
                        if result.fun < best_fun:
                            best_fun = result.fun
                            best_d = result.x[0]
                    except:
                        pass

                d_fit = best_d
                chi2 = best_fun

            else:
                # Single ARXPS
                I_sim = simulate_arxps(layers, angles, hv)
                if I_sim is None:
                    continue

                I_obs = {k: add_noise(v, noise, seed=trial) for k, v in I_sim.items()}
                d_fit, chi2 = fit_periodic_structure(I_obs, pattern, angles, hv)

            if not np.isnan(d_fit):
                d_errors.append(abs(d_fit - true_d))
                chi2_values.append(chi2)

        if d_errors:
            results[source_name] = {
                'd_error_mean': np.mean(d_errors),
                'd_error_std': np.std(d_errors),
                'd_error_max': np.max(d_errors),
                'chi2_mean': np.mean(chi2_values),
                'imfp': imfp,
                'imfp_ratio': structure.total_thickness() / imfp if imfp > 0 else np.inf,
                'n_success': len(d_errors),
            }

    return results


def run_noise_sweep(structure: MultiLayerStructure,
                    noise_levels: List[float] = [0.02, 0.05, 0.10],
                    n_trials: int = 20) -> Dict:
    """Test noise robustness"""

    results = {}

    for noise in noise_levels:
        results[f'{int(noise*100)}%'] = run_xray_sources(
            structure, noise=noise, n_trials=n_trials, use_etch=False
        )

    return results


def run_non_periodic(pattern: str,
                     thicknesses_true: List[float],
                     noise: float = 0.02,
                     n_trials: int = 10) -> Dict:
    """Test non-periodic structure recovery"""

    angles = np.array([0, 15, 30, 45, 55, 65, 75])
    structure = MultiLayerStructure.from_thicknesses(pattern, thicknesses_true)
    layers = structure.get_layers_tuples()

    results = {}

    for source_name, hv in XRAY_SOURCES.items():
        available = get_available_core_levels(hv)
        if 'O_1s' not in available:
            continue

        all_errors = []

        for trial in range(n_trials):
            I_sim = simulate_arxps(layers, angles, hv)
            if I_sim is None:
                continue

            I_obs = {k: add_noise(v, noise, seed=trial) for k, v in I_sim.items()}

            # Fit variable thicknesses
            d_init = thicknesses_true.copy()
            d_fit, chi2 = fit_variable_structure(I_obs, pattern, angles, hv, d_init)

            if not any(np.isnan(d_fit)):
                errors = [abs(d_f - d_t) for d_f, d_t in zip(d_fit, thicknesses_true)]
                all_errors.append(errors)

        if all_errors:
            all_errors = np.array(all_errors)
            results[source_name] = {
                'layer_errors_mean': np.mean(all_errors, axis=0).tolist(),
                'layer_errors_std': np.std(all_errors, axis=0).tolist(),
                'total_error_mean': np.mean(np.sum(all_errors, axis=1)),
                'rmse': np.sqrt(np.mean(all_errors**2)),
            }

    return results


def run_mixed_composition(pattern: str,
                          layer_thickness: float,
                          x_O_N: float = 0.1,  # O content in "N" layers
                          x_N_O: float = 0.1,  # N content in "O" layers
                          noise: float = 0.02,
                          n_trials: int = 10) -> Dict:
    """Test with mixed O/N compositions (reduced contrast)"""

    angles = np.array([0, 15, 30, 45, 55, 65, 75])

    # Create mixed structure
    x_O_values = {'O': 1.0 - x_N_O, 'N': x_O_N}  # O fraction for each type
    structure = MultiLayerStructure.from_mixed(pattern, layer_thickness, x_O_values)
    layers = structure.get_layers_tuples()

    # Calculate contrast
    contrast = abs((1.0 - x_N_O) - x_O_N)  # 0 = no contrast, 1 = full contrast

    results = {'contrast': contrast, 'sources': {}}

    for source_name, hv in XRAY_SOURCES.items():
        available = get_available_core_levels(hv)
        if 'O_1s' not in available:
            continue

        d_errors = []

        for trial in range(n_trials):
            I_sim = simulate_arxps(layers, angles, hv)
            if I_sim is None:
                continue

            I_obs = {k: add_noise(v, noise, seed=trial) for k, v in I_sim.items()}

            # Fit assuming pure N/O pattern (model mismatch)
            d_fit, chi2 = fit_periodic_structure(I_obs, pattern, angles, hv)

            if not np.isnan(d_fit):
                d_errors.append(abs(d_fit - layer_thickness))

        if d_errors:
            results['sources'][source_name] = {
                'd_error_mean': np.mean(d_errors),
                'd_error_std': np.std(d_errors),
            }

    return results


# =============================================================================
# メイン実行
# =============================================================================

def main():
    print("="*80)
    print("NON系多層構造：包括的検証")
    print("="*80)

    # Show X-ray source info
    print("\n### X-ray Sources ###")
    print(f"{'Source':<15} {'hν (eV)':>10} {'IMFP@NON (Å)':>15}")
    print("-"*45)
    for name, hv in XRAY_SOURCES.items():
        available = get_available_core_levels(hv)
        KE_ref = available.get('Si_2p', available.get('Si_2s', 0))
        imfp = get_imfp_mixed(KE_ref, 0.5) if KE_ref > 0 else 0
        o1s_ok = 'O_1s' in available
        print(f"{name:<15} {hv:>10.1f} {imfp:>15.1f}   {'(O 1s OK)' if o1s_ok else '(O 1s NG)'}")

    # =========================================================================
    # Test 1: X-ray source comparison (periodic, 2% noise)
    # =========================================================================
    print("\n\n" + "="*80)
    print("TEST 1: X-ray Source Comparison (periodic, 2% noise)")
    print("="*80)

    test1_results = {}

    for pattern in ['NONONON']:
        for d in [10.0, 20.0, 50.0]:
            structure = MultiLayerStructure.from_pattern(pattern, d)
            key = f"{pattern}_{d}A"

            print(f"\n### {key} (Total: {structure.total_thickness()}Å) ###")

            # Single ARXPS
            single = run_xray_sources(structure, noise=0.02, n_trials=10, use_etch=False)

            # Etch + ARXPS
            etch = run_xray_sources(structure, noise=0.02, n_trials=10, use_etch=True)

            test1_results[key] = {'single': single, 'etch': etch}

            print(f"\n{'Source':<15} {'Single d_err':>15} {'Etch d_err':>15} {'IMFP ratio':>12}")
            print("-"*60)
            for source in XRAY_SOURCES.keys():
                if source in single:
                    s_err = f"{single[source]['d_error_mean']:.2f}±{single[source]['d_error_std']:.2f}"
                    e_err = f"{etch[source]['d_error_mean']:.2f}±{etch[source]['d_error_std']:.2f}" if source in etch else "---"
                    ratio = f"{single[source]['imfp_ratio']:.1f}"
                    print(f"{source:<15} {s_err:>15} {e_err:>15} {ratio:>12}")

    # =========================================================================
    # Test 2: Noise sweep
    # =========================================================================
    print("\n\n" + "="*80)
    print("TEST 2: Noise Robustness (NONONON, 20Å)")
    print("="*80)

    structure = MultiLayerStructure.from_pattern('NONONON', 20.0)
    noise_results = run_noise_sweep(structure, [0.02, 0.05, 0.10], n_trials=20)

    print(f"\n{'Source':<15} {'2% noise':>12} {'5% noise':>12} {'10% noise':>12}")
    print("-"*55)
    for source in XRAY_SOURCES.keys():
        row = f"{source:<15}"
        for noise_key in ['2%', '5%', '10%']:
            if noise_key in noise_results and source in noise_results[noise_key]:
                err = noise_results[noise_key][source]['d_error_mean']
                row += f" {err:>11.2f}Å"
            else:
                row += f" {'---':>11}"
        print(row)

    # =========================================================================
    # Test 3: Non-periodic structure
    # =========================================================================
    print("\n\n" + "="*80)
    print("TEST 3: Non-periodic Structure")
    print("="*80)

    # Asymmetric thicknesses
    pattern = 'NONONON'
    thicknesses = [8.0, 15.0, 12.0, 25.0, 10.0, 20.0, 18.0]  # Varying thicknesses

    print(f"\nPattern: {pattern}")
    print(f"True thicknesses: {thicknesses}")
    print(f"Total: {sum(thicknesses)}Å")

    nonperiodic_results = run_non_periodic(pattern, thicknesses, noise=0.02, n_trials=10)

    print(f"\n{'Source':<15} {'Layer-wise RMSE':>18} {'Total RMSE':>12}")
    print("-"*50)
    for source, res in nonperiodic_results.items():
        layer_rmse = np.sqrt(np.mean(np.array(res['layer_errors_mean'])**2))
        print(f"{source:<15} {layer_rmse:>18.2f}Å {res['rmse']:>12.2f}Å")

    # =========================================================================
    # Test 4: Mixed composition (reduced contrast)
    # =========================================================================
    print("\n\n" + "="*80)
    print("TEST 4: Mixed Composition (Reduced Contrast)")
    print("="*80)

    pattern = 'NONONON'
    d = 20.0

    mix_levels = [
        (0.0, 0.0),   # Pure (100% contrast)
        (0.1, 0.1),   # 10% mixing (80% contrast)
        (0.2, 0.2),   # 20% mixing (60% contrast)
        (0.3, 0.3),   # 30% mixing (40% contrast)
    ]

    mix_results = {}

    for x_O_N, x_N_O in mix_levels:
        key = f"mix_{int((1-x_O_N-x_N_O)*100)}%"
        mix_results[key] = run_mixed_composition(pattern, d, x_O_N, x_N_O, noise=0.02, n_trials=10)

    print(f"\n{'Contrast':<12} ", end="")
    for source in XRAY_SOURCES.keys():
        print(f"{source:>12}", end="")
    print()
    print("-"*75)

    for key, res in mix_results.items():
        contrast = res['contrast']
        row = f"{contrast*100:.0f}%       "
        for source in XRAY_SOURCES.keys():
            if source in res['sources']:
                err = res['sources'][source]['d_error_mean']
                row += f" {err:>11.2f}"
            else:
                row += f" {'---':>11}"
        print(row)


    # Save plot
    from deppro.paths import ensure_outputs_dir
    output_dir = str(ensure_outputs_dir())
    os.makedirs(output_dir, exist_ok=True)

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Source comparison for different thicknesses
    ax = axes[0, 0]
    sources = [s for s in XRAY_SOURCES.keys() if s != 'Sync-800eV']
    thicknesses = [10.0, 20.0, 50.0]
    x_pos = np.arange(len(thicknesses))
    width = 0.15

    for i, source in enumerate(sources):
        errors = []
        for d in thicknesses:
            key = f"NONONON_{d}A"
            if key in test1_results and source in test1_results[key]['single']:
                errors.append(test1_results[key]['single'][source]['d_error_mean'])
            else:
                errors.append(np.nan)
        ax.bar(x_pos + i*width, errors, width, label=source, alpha=0.7)

    ax.set_xlabel('Layer Thickness (Å)')
    ax.set_ylabel('Thickness Error (Å)')
    ax.set_title('Single ARXPS: Source Comparison')
    ax.set_xticks(x_pos + width*1.5)
    ax.set_xticklabels([f'{d}Å' for d in thicknesses])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 2)

    # Plot 2: Etch vs Single improvement
    ax = axes[0, 1]
    for i, source in enumerate(sources):
        single_errs = []
        etch_errs = []
        for d in thicknesses:
            key = f"NONONON_{d}A"
            if key in test1_results:
                if source in test1_results[key]['single']:
                    single_errs.append(test1_results[key]['single'][source]['d_error_mean'])
                else:
                    single_errs.append(np.nan)
                if source in test1_results[key]['etch']:
                    etch_errs.append(test1_results[key]['etch'][source]['d_error_mean'])
                else:
                    etch_errs.append(np.nan)

        ax.plot(thicknesses, single_errs, 'o--', label=f'{source} (single)', alpha=0.5)
        ax.plot(thicknesses, etch_errs, 's-', label=f'{source} (etch)', alpha=0.8)

    ax.set_xlabel('Layer Thickness (Å)')
    ax.set_ylabel('Thickness Error (Å)')
    ax.set_title('Single vs Etch+ARXPS')
    ax.legend(fontsize=6, loc='upper left')
    ax.set_ylim(0, 2)

    # Plot 3: Noise sensitivity
    ax = axes[0, 2]
    noise_levels = [2, 5, 10]
    for source in sources:
        errors = []
        for n in noise_levels:
            key = f'{n}%'
            if key in noise_results and source in noise_results[key]:
                errors.append(noise_results[key][source]['d_error_mean'])
            else:
                errors.append(np.nan)
        ax.plot(noise_levels, errors, 'o-', label=source)

    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('Thickness Error (Å)')
    ax.set_title('Noise Sensitivity (NONONON 20Å)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 12)

    # Plot 4: IMFP ratio vs error
    ax = axes[1, 0]
    for d in thicknesses:
        key = f"NONONON_{d}A"
        if key not in test1_results:
            continue
        ratios = []
        errors = []
        labels = []
        for source in sources:
            if source in test1_results[key]['single']:
                ratios.append(test1_results[key]['single'][source]['imfp_ratio'])
                errors.append(test1_results[key]['single'][source]['d_error_mean'])
                labels.append(source)
        ax.scatter(ratios, errors, label=f'{d}Å', s=50)

    ax.set_xlabel('Total Thickness / IMFP')
    ax.set_ylabel('Thickness Error (Å)')
    ax.set_title('IMFP Ratio vs Error')
    ax.legend()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2)

    # Plot 5: Contrast dependence
    ax = axes[1, 1]
    contrasts = []
    for key, res in mix_results.items():
        contrasts.append(res['contrast'] * 100)

    for source in sources:
        errors = []
        for key in mix_results.keys():
            if source in mix_results[key]['sources']:
                errors.append(mix_results[key]['sources'][source]['d_error_mean'])
            else:
                errors.append(np.nan)
        ax.plot(contrasts, errors, 'o-', label=source)

    ax.set_xlabel('Contrast (%)')
    ax.set_ylabel('Thickness Error (Å)')
    ax.set_title('Contrast Dependence (NONONON 20Å)')
    ax.legend(fontsize=8)
    ax.set_xlim(30, 105)

    # Plot 6: Non-periodic layer-wise errors
    ax = axes[1, 2]
    n_nonperiodic_layers = 7  # NONONON
    layer_positions = np.arange(n_nonperiodic_layers)
    for source, res in nonperiodic_results.items():
        if 'layer_errors_mean' in res:
            ax.plot(layer_positions, res['layer_errors_mean'], 'o-', label=source)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Thickness Error (Å)')
    ax.set_title('Non-periodic: Layer-wise Errors')
    ax.legend(fontsize=8)
    ax.set_xticks(layer_positions)
    ax.set_xticklabels([f'L{i+1}' for i in layer_positions])

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'test_non_comprehensive.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close()

    return {
        'test1': test1_results,
        'noise': noise_results,
        'nonperiodic': nonperiodic_results,
        'mixed': mix_results,
    }


# =============================================================================
# Phase Diagram用パラメータスイープ（拡張版）
# =============================================================================

@pytest.mark.slow  # 実測 735s (2026-07-29)
def test_failure_threshold_sweep(n_trials: int = 5, quick: bool = True):
    """
    Phase diagram用のパラメータスイープ

    Parameters
    ----------
    n_trials : int
        各条件での試行回数
    quick : bool
        True: 高速版（パラメータ削減）、False: フル版

    Returns
    -------
    dict
        results[(contrast, noise, thickness, source)] = {
            'd_error_mean', 'd_error_std', 'chi2_mean', 'success_rate'
        }
    """
    print("="*80)
    print("FAILURE THRESHOLD SWEEP - Phase Diagram Generation")
    print("="*80)

    # パラメータ定義
    if quick:
        contrasts = [100, 60, 40, 25]  # %
        noises = [0.02, 0.05, 0.10]
        thicknesses = [10, 20, 50]  # Å
    else:
        contrasts = [100, 80, 60, 50, 40, 30, 25, 20]
        noises = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
        thicknesses = [5, 10, 15, 20, 30, 50, 80, 100]

    sources = ['Al-Kα', 'Ag-Lα', 'Cr-Kα', 'Ga-Kα']
    pattern = 'NONONON'
    angles = np.array([0, 15, 30, 45, 55, 65, 75])

    total_cases = len(contrasts) * len(noises) * len(thicknesses) * len(sources)
    print(f"\nTotal cases: {total_cases} x {n_trials} trials")

    results = {}
    case_count = 0

    for contrast in contrasts:
        for noise in noises:
            for thickness in thicknesses:
                for source in sources:
                    case_count += 1

                    # Skip Sync-800eV (O 1s not available)
                    hv = XRAY_SOURCES.get(source, 0)
                    if hv == 0:
                        continue
                    available = get_available_core_levels(hv)
                    if 'O_1s' not in available:
                        continue

                    # Create mixed structure for contrast
                    # contrast 100% = pure, 40% = 30% mixing
                    x_mix = (100 - contrast) / 200  # 100% -> 0, 40% -> 0.3
                    x_O_values = {'O': 1.0 - x_mix, 'N': x_mix}
                    structure = MultiLayerStructure.from_mixed(pattern, thickness, x_O_values)
                    layers = structure.get_layers_tuples()

                    d_errors = []
                    chi2_values = []

                    for trial in range(n_trials):
                        I_sim = simulate_arxps(layers, angles, hv)
                        if I_sim is None:
                            continue

                        I_obs = {k: add_noise(v, noise, seed=trial) for k, v in I_sim.items()}

                        # Fit assuming pure pattern
                        d_fit, chi2 = fit_periodic_structure(I_obs, pattern, angles, hv)

                        if not np.isnan(d_fit):
                            d_errors.append(abs(d_fit - thickness))
                            chi2_values.append(chi2)

                    if d_errors:
                        results[(contrast, noise, thickness, source)] = {
                            'd_error_mean': np.mean(d_errors),
                            'd_error_std': np.std(d_errors),
                            'chi2_mean': np.mean(chi2_values),
                            'success_rate': len(d_errors) / n_trials,
                        }

                    # Progress
                    if case_count % 10 == 0:
                        print(f"  Progress: {case_count}/{total_cases} ({100*case_count/total_cases:.0f}%)")

    print(f"\nCompleted {len(results)} valid cases")
    return results


def extract_failure_thresholds(sweep_results: Dict, error_threshold: float = 3.0):
    """
    スイープ結果から破綻閾値を抽出

    Parameters
    ----------
    sweep_results : dict
        test_failure_threshold_sweep()の結果
    error_threshold : float
        破綻と判定する膜厚誤差 [Å]

    Returns
    -------
    dict
        各X線源の限界条件
    """
    print("\n" + "="*80)
    print(f"FAILURE THRESHOLD EXTRACTION (error > {error_threshold} Å = FAIL)")
    print("="*80)

    sources = ['Al-Kα', 'Ag-Lα', 'Cr-Kα', 'Ga-Kα']

    # Initialize thresholds
    thresholds = {source: {
        'min_contrast_at_2pct_noise': None,
        'max_noise_at_100pct_contrast': None,
        'max_thickness_at_2pct_noise': None,
        'failure_rate': 0,
        'success_rate': 0,
    } for source in sources}

    # Analyze results
    for source in sources:
        source_results = {k: v for k, v in sweep_results.items() if k[3] == source}

        if not source_results:
            continue

        total = len(source_results)
        failures = sum(1 for v in source_results.values() if v['d_error_mean'] > error_threshold)
        thresholds[source]['failure_rate'] = failures / total if total > 0 else 0
        thresholds[source]['success_rate'] = 1 - thresholds[source]['failure_rate']

        # Find min contrast at 2% noise (thickness=20Å)
        for contrast in [100, 80, 60, 50, 40, 30, 25, 20]:
            key = (contrast, 0.02, 20, source)
            if key in source_results:
                if source_results[key]['d_error_mean'] <= error_threshold:
                    thresholds[source]['min_contrast_at_2pct_noise'] = contrast
                else:
                    break

        # Find max noise at 100% contrast (thickness=20Å)
        for noise in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
            key = (100, noise, 20, source)
            if key in source_results:
                if source_results[key]['d_error_mean'] <= error_threshold:
                    thresholds[source]['max_noise_at_100pct_contrast'] = noise * 100
                else:
                    break

        # Find max thickness at 2% noise, 100% contrast
        for thickness in [5, 10, 15, 20, 30, 50, 80, 100]:
            key = (100, 0.02, thickness, source)
            if key in source_results:
                if source_results[key]['d_error_mean'] <= error_threshold:
                    thresholds[source]['max_thickness_at_2pct_noise'] = thickness

    # Print results
    print("\n### Table 1: X-ray Source Limits ###")
    print(f"{'Condition':<35} {'Al-Kα':>10} {'Ag-Lα':>10} {'Cr-Kα':>10} {'Ga-Kα':>10}")
    print("-"*80)

    print(f"{'Min contrast (2% noise, 20Å)':<35}", end="")
    for source in sources:
        val = thresholds[source]['min_contrast_at_2pct_noise']
        print(f" {val if val else '---':>9}%", end="")
    print()

    print(f"{'Max noise (100% contrast, 20Å)':<35}", end="")
    for source in sources:
        val = thresholds[source]['max_noise_at_100pct_contrast']
        print(f" {val if val else '---':>9}%", end="")
    print()

    print(f"{'Max thickness (2% noise, 100% cont)':<35}", end="")
    for source in sources:
        val = thresholds[source]['max_thickness_at_2pct_noise']
        print(f" {val if val else '---':>9}Å", end="")
    print()

    print(f"{'Success rate':<35}", end="")
    for source in sources:
        val = thresholds[source]['success_rate']
        print(f" {val*100:>9.0f}%", end="")
    print()

    return thresholds


def create_phase_diagrams(sweep_results: Dict, output_dir: str = None):
    """
    フェーズダイアグラムを作成

    Parameters
    ----------
    sweep_results : dict
        test_failure_threshold_sweep()の結果
    output_dir : str
        出力ディレクトリ
    """
    if output_dir is None:
        from deppro.paths import ensure_outputs_dir
        output_dir = str(ensure_outputs_dir())
    os.makedirs(output_dir, exist_ok=True)

    sources = ['Al-Kα', 'Ag-Lα', 'Cr-Kα', 'Ga-Kα']

    # Extract unique parameters
    contrasts = sorted(set(k[0] for k in sweep_results.keys()))
    noises = sorted(set(k[1] for k in sweep_results.keys()))
    thicknesses = sorted(set(k[2] for k in sweep_results.keys()))

    # Fig 1: Contrast vs Noise heatmap (thickness=20Å)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Fig 1: Contrast vs Noise (thickness=20Å)', fontsize=14)

    for idx, source in enumerate(sources):
        ax = axes[idx // 2, idx % 2]

        # Build matrix
        matrix = np.full((len(contrasts), len(noises)), np.nan)
        for i, contrast in enumerate(contrasts):
            for j, noise in enumerate(noises):
                key = (contrast, noise, 20, source)
                if key in sweep_results:
                    matrix[i, j] = sweep_results[key]['d_error_mean']

        im = ax.imshow(matrix, aspect='auto', origin='lower',
                       extent=[min(noises)*100, max(noises)*100, min(contrasts), max(contrasts)],
                       cmap='RdYlGn_r', vmin=0, vmax=5)
        ax.set_xlabel('Noise (%)')
        ax.set_ylabel('Contrast (%)')
        ax.set_title(source)
        plt.colorbar(im, ax=ax, label='Error (Å)')

        # Add failure boundary (3Å line)
        ax.contour(noises, contrasts, matrix, levels=[3.0], colors='red', linewidths=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_diagram_contrast_noise.png'), dpi=150)
    plt.close()
    print(f"Saved: phase_diagram_contrast_noise.png")

    # Fig 2: Thickness vs Noise heatmap (contrast=100%)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Fig 2: Thickness vs Noise (contrast=100%)', fontsize=14)

    for idx, source in enumerate(sources):
        ax = axes[idx // 2, idx % 2]

        matrix = np.full((len(thicknesses), len(noises)), np.nan)
        for i, thickness in enumerate(thicknesses):
            for j, noise in enumerate(noises):
                key = (100, noise, thickness, source)
                if key in sweep_results:
                    matrix[i, j] = sweep_results[key]['d_error_mean']

        im = ax.imshow(matrix, aspect='auto', origin='lower',
                       extent=[min(noises)*100, max(noises)*100, min(thicknesses), max(thicknesses)],
                       cmap='RdYlGn_r', vmin=0, vmax=5)
        ax.set_xlabel('Noise (%)')
        ax.set_ylabel('Thickness (Å)')
        ax.set_title(source)
        plt.colorbar(im, ax=ax, label='Error (Å)')

        ax.contour(noises, thicknesses, matrix, levels=[3.0], colors='red', linewidths=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_diagram_thickness_noise.png'), dpi=150)
    plt.close()
    print(f"Saved: phase_diagram_thickness_noise.png")

    # Fig 3: Summary bar chart (contrast thresholds)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(sources))
    width = 0.25

    # Collect data for different noise levels
    errors_2pct = []
    errors_5pct = []
    errors_10pct = []

    for source in sources:
        key_2 = (40, 0.02, 20, source)  # 40% contrast, 2% noise
        key_5 = (40, 0.05, 20, source)
        key_10 = (40, 0.10, 20, source)

        errors_2pct.append(sweep_results.get(key_2, {}).get('d_error_mean', np.nan))
        errors_5pct.append(sweep_results.get(key_5, {}).get('d_error_mean', np.nan))
        errors_10pct.append(sweep_results.get(key_10, {}).get('d_error_mean', np.nan))

    ax.bar(x - width, errors_2pct, width, label='2% noise', color='green', alpha=0.7)
    ax.bar(x, errors_5pct, width, label='5% noise', color='orange', alpha=0.7)
    ax.bar(x + width, errors_10pct, width, label='10% noise', color='red', alpha=0.7)

    ax.axhline(3.0, color='black', linestyle='--', label='Failure threshold')
    ax.set_xlabel('X-ray Source')
    ax.set_ylabel('Thickness Error (Å)')
    ax.set_title('Error at 40% Contrast (SiGe-like)')
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.legend()
    ax.set_ylim(0, 10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_diagram_summary.png'), dpi=150)
    plt.close()
    print(f"Saved: phase_diagram_summary.png")


def run_phase_diagram_analysis(quick: bool = True):
    """
    Phase diagram解析を実行

    Parameters
    ----------
    quick : bool
        True: 高速版（約5分）、False: フル版（約30分）
    """
    print("\n" + "="*80)
    print("PHASE DIAGRAM ANALYSIS")
    print("="*80)

    # Run sweep
    sweep_results = test_failure_threshold_sweep(n_trials=5, quick=quick)

    # Extract thresholds
    thresholds = extract_failure_thresholds(sweep_results)

    # Create diagrams
    create_phase_diagrams(sweep_results)

    return sweep_results, thresholds


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--phase-diagram':
        # Phase diagram mode
        quick = '--full' not in sys.argv
        results, thresholds = run_phase_diagram_analysis(quick=quick)
    else:
        # Original comprehensive test
        results = main()
