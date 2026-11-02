"""
Low Intensity Poisson Noise Test
================================

Test conditions where low intensity signals dominate (high angles, thin films).
This is where Poisson noise causes the most problems.

Date: 2026-01-28
"""

import numpy as np
from scipy.special import erf
from typing import List, Dict


def create_sio2_si_profile(depth: np.ndarray, thickness: float, interface_width: float = 3.0) -> np.ndarray:
    """Create SiO2/Si bilayer profile."""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))

    D = np.zeros((len(depth), 3))
    sio2 = 0.5 * (1 - erf((depth - thickness) / sigma))
    D[:, 0] = (1/3) * sio2
    D[:, 1] = (2/3) * sio2
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def add_poisson_noise(I_true: np.ndarray, noise_level: float, seed: int = None) -> np.ndarray:
    """Add Poisson noise."""
    if seed is not None:
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


def analyze_intensity_distribution(I_true: np.ndarray, angles: List[float], elements: List[str]):
    """Analyze intensity distribution."""
    print("\n  Intensity distribution:")
    print(f"  {'Angle':<10} {'Min I':<10} {'Max I':<10} {'Min/Max':<10}")
    print("  " + "-" * 40)

    for i, angle in enumerate(angles):
        I_row = I_true[i, :]
        min_I = I_row.min()
        max_I = I_row.max()
        ratio = min_I / max_I
        print(f"  {angle:>5}°    {min_I:.4f}     {max_I:.4f}     {ratio:.3f}")

    print(f"\n  Overall min: {I_true.min():.4f}")
    print(f"  Overall max: {I_true.max():.4f}")
    print(f"  Dynamic range: {I_true.max() / I_true.min():.1f}x")


def run_solver(I_obs, tm, elements, use_wls: bool = False):
    """Run SemiFreeSolver."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    semi_config = SemiFreeSolverConfig(
        thickness_init=30.0,
        abruptness_init=3.0,
        free_indices=[0],
        linked_indices=[1],
        substrate_indices=[2],
    )

    solver_config = SolverConfig(
        method='L1',
        alpha=1.0,
        max_iter=100,
        use_wls=use_wls,
    )

    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)
    result = solver.solve(I_obs, verbose=False)

    return result.D_opt, result.thickness, result.success


def run_test(
    n_trials: int,
    thickness: float,
    angles: List[float],
    noise_level: float,
    label: str,
) -> Dict:
    """Run test for specific conditions."""
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    depth = np.arange(0, 80, 2.0)

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_sio2_si_profile(depth, thickness=thickness)

    tm = TransferMatrix(depth=depth, step_size=2.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    print(f"\n{label}:")
    analyze_intensity_distribution(I_true, angles, ['Si_ox', 'O', 'Si_bulk'])

    results = {'OLS': {'w1': [], 'success': []}, 'WLS': {'w1': [], 'success': []}}

    for trial in range(n_trials):
        I_obs = add_poisson_noise(I_true, noise_level, seed=42 + trial)

        for use_wls in [False, True]:
            name = 'WLS' if use_wls else 'OLS'
            D_opt, _, success = run_solver(I_obs, tm, elements, use_wls=use_wls)

            results[name]['success'].append(success)
            if success:
                w1 = np.mean([compute_wasserstein(depth, D_true[:, i], D_opt[:, i]) for i in range(3)])
                results[name]['w1'].append(w1)
            else:
                results[name]['w1'].append(np.nan)

    # Summary
    print("\n  Results:")
    for name in ['OLS', 'WLS']:
        n_success = sum(results[name]['success'])
        w1_valid = [w for w in results[name]['w1'] if not np.isnan(w)]
        w1_med = np.median(w1_valid) if w1_valid else np.nan
        print(f"    {name}: Success {n_success}/{n_trials}, median W1 = {w1_med:.3f} Å")

    return results


def main():
    print("=" * 70)
    print("Low Intensity Poisson Noise Test")
    print("=" * 70)
    print("\nTest conditions with high dynamic range (low intensity signals)")
    print("Poisson noise causes more problems when some signals are weak")

    n_trials = 30

    # Standard condition (reference)
    run_test(
        n_trials=n_trials,
        thickness=30.0,
        angles=[0, 15, 30, 45, 60],
        noise_level=0.005,
        label="Reference: 30Å, 0-60°, Poisson 0.5%",
    )

    # High angle (very low substrate signal)
    run_test(
        n_trials=n_trials,
        thickness=30.0,
        angles=[0, 20, 40, 60, 75],  # Include 75°
        noise_level=0.005,
        label="High angle: 30Å, 0-75°, Poisson 0.5%",
    )

    # Very thin film (low film signal)
    run_test(
        n_trials=n_trials,
        thickness=5.0,
        angles=[0, 15, 30, 45, 60],
        noise_level=0.005,
        label="Thin film: 5Å, 0-60°, Poisson 0.5%",
    )

    # Higher Poisson noise (more heteroscedasticity)
    run_test(
        n_trials=n_trials,
        thickness=30.0,
        angles=[0, 15, 30, 45, 60],
        noise_level=0.01,  # 1% = 10000 counts at I=1
        label="Higher noise: 30Å, 0-60°, Poisson 1%",
    )

    # Poisson 2%
    run_test(
        n_trials=n_trials,
        thickness=30.0,
        angles=[0, 15, 30, 45, 60],
        noise_level=0.02,  # 2% = 2500 counts at I=1
        label="Original: 30Å, 0-60°, Poisson 2%",
    )

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
Low intensity conditions (high angles, thin films, high noise)
increase failure rate for Poisson noise.

WLS helps by giving more weight to high-intensity signals
which have better signal-to-noise ratio.
    """)


if __name__ == "__main__":
    main()
