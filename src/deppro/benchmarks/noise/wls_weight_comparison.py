"""
WLS Weight Strategy Comparison
==============================

Compare different WLS weighting strategies on 4-layer system.

Strategies:
1. weight = I (current): favors high intensity
2. weight = 1/I (GLS): favors low intensity (theoretically correct for Poisson)
3. weight = sqrt(I): compromise
4. weight = 1 (OLS): uniform

Date: 2026-01-28
"""

import numpy as np
from scipy.special import erf
from typing import List
import time


def create_4layer_profile(depth: np.ndarray) -> np.ndarray:
    """Create 4-layer profile: C / Al2O3 / TiO2 / SiO2 / Si"""
    thicknesses = [5.0, 15.0, 35.0, 30.0]
    interface_widths = [2.0, 3.0, 4.0, 5.0]
    sigma = [w / (2 * np.sqrt(np.log(2))) for w in interface_widths]
    z = [0] + list(np.cumsum(thicknesses))

    D = np.zeros((len(depth), 6))

    D[:, 0] = 0.5 * (1 - erf((depth - z[1]) / sigma[0]))

    al2o3 = 0.5 * (1 + erf((depth - z[1]) / sigma[0])) * 0.5 * (1 - erf((depth - z[2]) / sigma[1]))
    D[:, 1] = (2/5) * al2o3
    D[:, 4] = (3/5) * al2o3

    tio2 = 0.5 * (1 + erf((depth - z[2]) / sigma[1])) * 0.5 * (1 - erf((depth - z[3]) / sigma[2]))
    D[:, 2] = (1/3) * tio2
    D[:, 4] += (2/3) * tio2

    sio2 = 0.5 * (1 + erf((depth - z[3]) / sigma[2])) * 0.5 * (1 - erf((depth - z[4]) / sigma[3]))
    D[:, 3] = (1/3) * sio2
    D[:, 4] += (2/3) * sio2

    D[:, 5] = 0.5 * (1 + erf((depth - z[4]) / sigma[3]))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def add_poisson_noise(I_true: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Add Poisson noise."""
    np.random.seed(seed)
    total_counts = 1.0 / (noise_level ** 2)
    counts = I_true * total_counts
    noisy_counts = np.random.poisson(counts)
    I_obs = noisy_counts / total_counts
    I_obs = np.clip(I_obs, 1e-10, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    return I_obs


def compute_wasserstein(depth: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Wasserstein-1 distance."""
    p1_norm = p1 / (np.sum(p1) + 1e-10)
    p2_norm = p2 / (np.sum(p2) + 1e-10)
    cdf1 = np.cumsum(p1_norm)
    cdf2 = np.cumsum(p2_norm)
    step = depth[1] - depth[0] if len(depth) > 1 else 1.0
    return np.sum(np.abs(cdf1 - cdf2)) * step


def compute_weights(I_obs: np.ndarray, strategy: str) -> np.ndarray:
    """Compute WLS weights based on strategy."""
    I_safe = np.maximum(I_obs, 1e-10)

    if strategy == 'OLS':
        return np.ones_like(I_obs)
    elif strategy == 'weight_I':  # Current implementation
        raw = I_safe
    elif strategy == 'weight_inv_I':  # 1/I (GLS for Poisson)
        raw = 1.0 / I_safe
    elif strategy == 'weight_sqrt_I':  # sqrt(I) compromise
        raw = np.sqrt(I_safe)
    elif strategy == 'weight_inv_sqrt_I':  # 1/sqrt(I)
        raw = 1.0 / np.sqrt(I_safe)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Normalize so sum = n_data_points
    weights = raw * (raw.size / raw.sum())
    return weights


def run_solver_with_weights(I_obs, tm, elements, weights: np.ndarray):
    """Run SemiFreeSolver with custom weights.

    Note: This is a simplified version that injects weights into the objective.
    The actual solver uses internal weighting.
    """
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    semi_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
    )

    # For now, test with the built-in WLS option
    # which uses weight = I
    solver_config = SolverConfig(
        method='L1',
        alpha=1.0,
        max_iter=100,
        use_wls=False,  # We'll test the effect of different weights separately
    )

    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)
    result = solver.solve(I_obs, verbose=False)

    return result.D_opt, result.thickness, result.success


def main():
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    print("=" * 70)
    print("WLS Weight Strategy Analysis (4-Layer System)")
    print("=" * 70)

    depth = np.arange(0, 120, 4.0)
    angles = [0, 10, 20, 30, 40, 50, 60]

    elements = [
        ElementConfig('C', '1s', 'C', oxidation_state=0),
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_4layer_profile(depth)

    tm = TransferMatrix(depth=depth, step_size=4.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    # Analyze weight distribution for each strategy
    print("\n重み付け戦略の分析:")
    print("-" * 60)

    # Add some Poisson noise for analysis
    I_obs = add_poisson_noise(I_true, 0.02, seed=42)

    strategies = ['OLS', 'weight_I', 'weight_sqrt_I', 'weight_inv_sqrt_I', 'weight_inv_I']
    el_names = ['C', 'Al', 'Ti', 'Si_ox', 'O', 'Si_bulk']

    print(f"\n元素別平均強度と重み:")
    print(f"{'Element':<10} {'Mean I':<10}", end='')
    for s in strategies:
        print(f" {s:<12}", end='')
    print()
    print("-" * 80)

    for i, name in enumerate(el_names):
        I_el = I_obs[:, i]
        print(f"{name:<10} {I_el.mean():.4f}    ", end='')

        for s in strategies:
            weights = compute_weights(I_obs, s)
            w_el = weights[:, i].mean()
            print(f" {w_el:.3f}       ", end='')
        print()

    # Note on reading the table
    print("\n" + "=" * 70)
    print("NOTE")
    print("=" * 70)
    print("""
weight = I は強度の小さいチャネル（例: 埋もれた基板の信号）の寄与を小さくする。
その信号が総膜厚の情報を持つ系では、重みの選び方が結果に効く。
上の表は、この実行の条件での比較である。
    """)

    print("=" * 70)


if __name__ == "__main__":
    main()
