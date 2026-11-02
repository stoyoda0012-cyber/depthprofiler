"""
Multi-layer Poisson Noise Benchmark: C/Al2O3/TiO2/SiO2/Si
==========================================================

Tests solver accuracy on a realistic multi-layer structure with various noise types.

Structure:
    C (10Å) / Al2O3 (15Å) / TiO2 (20Å) / SiO2 (25Å) / Si (substrate)

Elements:
    - C (adventitious carbon)
    - Al, O (from Al2O3)
    - Ti, O (from TiO2)
    - Si, O (from SiO2)
    - Si (substrate)

Usage:
    python -m deppro.tests.benchmark.test_multilayer_poisson
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Literal, Tuple, List
from scipy.stats import wasserstein_distance
from scipy.special import erf

# Internal imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from deppro.depth_profile import ElementConfig
from deppro.transfer_matrix import TransferMatrix
from deppro.inverse_solver import (
    SemiFreeSolver,
    SemiFreeSolverConfig,
    SolverConfig,
)


NoiseType = Literal["gaussian", "poisson", "uniform"]


@dataclass
class NoiseConfig:
    """Noise configuration for benchmark."""
    noise_type: NoiseType
    noise_level: float  # relative noise level (e.g., 0.02 = 2%)


@dataclass
class LayerConfig:
    """Configuration for a single layer."""
    name: str
    thickness: float  # Å
    elements: List[str]  # element symbols
    stoichiometry: List[float]  # relative amounts


def create_multilayer_profile(
    depth: np.ndarray,
    layers: List[LayerConfig],
    interface_width: float = 5.0,
) -> Tuple[np.ndarray, List[ElementConfig]]:
    """
    Create a multi-layer depth profile.

    Args:
        depth: Depth array (Å)
        layers: List of layer configurations (from surface to substrate)
        interface_width: Interface width (FWHM) in Å

    Returns:
        D: Concentration profile (n_depth, n_elements)
        elements: List of ElementConfig
    """
    # Collect unique elements
    element_set = set()
    for layer in layers:
        for elem in layer.elements:
            element_set.add(elem)

    # Create ElementConfig list
    # Order: C, Al, Ti, Si_oxide, O, Si_substrate
    element_order = ['C', 'Al', 'Ti', 'Si_ox', 'O', 'Si_sub']
    elements = []
    elem_to_idx = {}

    for i, elem in enumerate(element_order):
        if elem == 'C':
            elements.append(ElementConfig('C', '1s', 'C'))
        elif elem == 'Al':
            elements.append(ElementConfig('Al', '2p', 'Al2O3'))
        elif elem == 'Ti':
            elements.append(ElementConfig('Ti', '2p', 'TiO2'))
        elif elem == 'Si_ox':
            elements.append(ElementConfig('Si', '2p', 'SiO2', oxidation_state=4))
        elif elem == 'O':
            elements.append(ElementConfig('O', '1s', 'SiO2'))  # Use SiO2 as reference
        elif elem == 'Si_sub':
            elements.append(ElementConfig('Si', '2p', 'Si', oxidation_state=0))
        elem_to_idx[elem] = i

    n_elements = len(elements)
    n_depth = len(depth)
    D = np.zeros((n_depth, n_elements))

    # Sigma from FWHM
    sigma = interface_width / (2.0 * np.sqrt(np.log(2.0)))

    # Cumulative thickness
    z_interface = 0.0

    for i, layer in enumerate(layers):
        z_start = z_interface
        z_end = z_interface + layer.thickness

        # Interface transition functions
        if i == 0:
            # First layer: only bottom interface
            f_layer = 0.5 * (1 - erf((depth - z_end) / sigma))
        elif i == len(layers) - 1:
            # Last layer (substrate): only top interface
            f_layer = 0.5 * (1 + erf((depth - z_start) / sigma))
        else:
            # Middle layers: both interfaces
            f_top = 0.5 * (1 + erf((depth - z_start) / sigma))
            f_bottom = 0.5 * (1 - erf((depth - z_end) / sigma))
            f_layer = f_top * f_bottom

        # Distribute elements according to stoichiometry
        total_stoich = sum(layer.stoichiometry)
        for elem, stoich in zip(layer.elements, layer.stoichiometry):
            idx = elem_to_idx[elem]
            D[:, idx] += f_layer * (stoich / total_stoich)

        z_interface = z_end

    # Normalize each depth to sum to 1
    row_sums = D.sum(axis=1, keepdims=True)
    D = D / np.maximum(row_sums, 1e-10)

    return D, elements


def add_noise(I_true: np.ndarray, config: NoiseConfig, seed: int = 42) -> np.ndarray:
    """Add noise to true intensities."""
    np.random.seed(seed)

    if config.noise_type == "gaussian":
        noise = config.noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true * (1 + noise)
    elif config.noise_type == "poisson":
        mean_counts = 1.0 / (config.noise_level ** 2)
        counts = I_true * mean_counts
        noisy_counts = np.random.poisson(counts.astype(np.float64))
        I_obs = noisy_counts / mean_counts
    elif config.noise_type == "uniform":
        noise = config.noise_level * (2 * np.random.rand(*I_true.shape) - 1)
        I_obs = I_true * (1 + noise)
    else:
        raise ValueError(f"Unknown noise type: {config.noise_type}")

    return np.clip(I_obs, 1e-10, None)


def compute_w1_distance(D_true: np.ndarray, D_opt: np.ndarray, depth: np.ndarray) -> float:
    """Compute average Wasserstein-1 distance across all elements."""
    w1_sum = 0.0
    n_elements = D_true.shape[1]
    for i in range(n_elements):
        w1_sum += wasserstein_distance(depth, depth, D_true[:, i], D_opt[:, i])
    return w1_sum / n_elements


def run_multilayer_benchmark(
    noise_config: NoiseConfig,
    seed: int = 42,
    use_wls: bool = False,
) -> Tuple[float, float, float, dict]:
    """
    Run benchmark on C/Al2O3/TiO2/SiO2/Si multilayer.

    Returns:
        total_thickness_error: Error in total film thickness
        w1_distance: Average W1 distance
        elapsed_time: Solve time (seconds)
        layer_errors: Dict of per-layer thickness errors
    """
    # Define layer structure
    layers = [
        LayerConfig('C', 10.0, ['C'], [1.0]),
        LayerConfig('Al2O3', 15.0, ['Al', 'O'], [2.0, 3.0]),
        LayerConfig('TiO2', 20.0, ['Ti', 'O'], [1.0, 2.0]),
        LayerConfig('SiO2', 25.0, ['Si_ox', 'O'], [1.0, 2.0]),
        LayerConfig('Si', 100.0, ['Si_sub'], [1.0]),  # Substrate
    ]

    total_film_thickness = sum(l.thickness for l in layers[:-1])  # Exclude substrate

    # Depth array
    depth = np.arange(0, 200, 1.0)
    angles = np.array([15, 30, 45, 60, 75])

    # Create true profile
    D_true, elements = create_multilayer_profile(depth, layers, interface_width=5.0)

    # Transfer matrix
    tm = TransferMatrix(
        depth=depth,
        step_size=1.0,
        angles=angles,
        photon_energy=1486.6,
    )
    tm.compute(D_true, elements)

    # True intensities
    I_true = tm.forward(D_true, normalize=True)

    # Add noise
    I_obs = add_noise(I_true, noise_config, seed)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    # Solver config
    config = SolverConfig(
        method='L1',
        alpha=1.0,
        use_wls=use_wls,
        scipy_method='SLSQP',
        max_iter=200,
    )

    # For multi-layer, we use total thickness as single parameter
    # free: C, Al, Ti, Si_ox, O  (indices 0-4)
    # substrate: Si_sub (index 5)
    semi_config = SemiFreeSolverConfig(
        thickness_init=total_film_thickness + 10.0,  # offset from true
        abruptness_init=5.0,
        thickness_bounds=(10.0, 150.0),
        abruptness_bounds=(1.0, 20.0),
        free_indices=[0, 1, 2, 3],  # C, Al, Ti, Si_ox
        linked_indices=[4],  # O (linked to oxides)
        substrate_indices=[5],  # Si_sub
    )

    # Create solver
    solver = SemiFreeSolver(
        transfer_matrix=tm,
        config=config,
        semi_free_config=semi_config,
        elements=elements,
    )

    # Solve
    start_time = time.time()
    result = solver.solve(I_obs)
    elapsed_time = time.time() - start_time

    # Compute errors
    thickness_error = abs(result.thickness - total_film_thickness)
    w1_distance = compute_w1_distance(D_true, result.D_opt, depth)

    return thickness_error, w1_distance, elapsed_time, {}


def run_noise_comparison(
    noise_type: NoiseType,
    noise_level: float = 0.02,
    n_trials: int = 10,
) -> dict:
    """Compare OLS vs WLS for multilayer structure."""
    print(f"\n{'='*70}")
    print(f"C/Al2O3/TiO2/SiO2/Si Multilayer - {noise_type.upper()} ({noise_level*100:.0f}%)")
    print(f"Structure: C(10Å) / Al2O3(15Å) / TiO2(20Å) / SiO2(25Å) / Si")
    print(f"Total film thickness: 70 Å")
    print(f"Trials: {n_trials}")
    print('='*70)

    noise_config = NoiseConfig(noise_type=noise_type, noise_level=noise_level)

    results = {}

    for use_wls, method_name in [(False, "OLS"), (True, "WLS")]:
        thickness_errors = []
        w1_distances = []
        times = []

        for trial in range(n_trials):
            try:
                t_err, w1, elapsed, _ = run_multilayer_benchmark(
                    noise_config=noise_config,
                    seed=42 + trial,
                    use_wls=use_wls,
                )
                thickness_errors.append(t_err)
                w1_distances.append(w1)
                times.append(elapsed)
            except Exception as e:
                print(f"  Trial {trial} failed: {e}")

        if thickness_errors:
            results[method_name] = {
                'thickness_error': np.mean(thickness_errors),
                'thickness_std': np.std(thickness_errors),
                'w1': np.mean(w1_distances),
                'w1_std': np.std(w1_distances),
                'time_ms': np.mean(times) * 1000,
            }
            print(f"\n{method_name}:")
            print(f"  Thickness error: {np.mean(thickness_errors):.2f} ± {np.std(thickness_errors):.2f} Å")
            print(f"  W1 distance:     {np.mean(w1_distances):.2f} ± {np.std(w1_distances):.2f} Å")
            print(f"  Solve time:      {np.mean(times)*1000:.1f} ± {np.std(times)*1000:.1f} ms")

    return results


def main():
    """Run multilayer benchmark."""
    print("="*70)
    print("Multilayer Benchmark: C/Al2O3/TiO2/SiO2/Si")
    print("="*70)
    print("\nComparing simple SiO2/Si (2 layers) vs complex 5-layer structure")

    all_results = {}

    # Test with different noise types
    all_results["gaussian"] = run_noise_comparison("gaussian", noise_level=0.02, n_trials=10)
    all_results["poisson"] = run_noise_comparison("poisson", noise_level=0.02, n_trials=10)
    all_results["uniform"] = run_noise_comparison("uniform", noise_level=0.02, n_trials=10)

    # Summary table
    print("\n" + "="*70)
    print("Summary Table: C/Al2O3/TiO2/SiO2/Si (70Å total)")
    print("="*70)
    print(f"\n{'Noise Type':<12} {'OLS W1 (Å)':<14} {'WLS W1 (Å)':<14} {'OLS Δt (Å)':<12} {'WLS Δt (Å)':<12}")
    print("-" * 64)
    for noise_type, results in all_results.items():
        ols_w1 = results.get('OLS', {}).get('w1', float('nan'))
        wls_w1 = results.get('WLS', {}).get('w1', float('nan'))
        ols_t = results.get('OLS', {}).get('thickness_error', float('nan'))
        wls_t = results.get('WLS', {}).get('thickness_error', float('nan'))
        print(f"{noise_type:<12} {ols_w1:<14.2f} {wls_w1:<14.2f} {ols_t:<12.2f} {wls_t:<12.2f}")

    print("\n" + "="*70)
    print("Comparison with SiO2/Si")
    print("="*70)
    print("Run test_wls_poisson.py under the same noise settings to compare.")
    print("\nA multilayer can give a larger W1 than SiO2/Si because of:")
    print("  - More parameters to estimate")
    print("  - More complex interference between layers")
    print("  - Overlapping O signal from multiple oxides")


if __name__ == "__main__":
    main()
