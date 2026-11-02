"""
SemiFreeSolver Nitride Pattern Tests

Test NO/ON/NON patterns with SemiFreeSolver.
Compares to GenericLayerSolver results.

Usage:
    python -m deppro.tests.solver.test_semifree_nitride_patterns
"""

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import (
    SemiFreeSolver, SemiFreeSolverConfig, SolverConfig
)


def create_no_pattern(depth, n_thickness, o_thickness, interface_width=5.0):
    """
    Create NO pattern: Si3N4 / SiO2 / Si (N on surface).

    Elements: [Si_n, N, Si_o, O, Si_bulk] = [0, 1, 2, 3, 4]
    """
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = n_thickness
    z2 = n_thickness + o_thickness

    D = np.zeros((len(depth), 5))

    # Si3N4 layer (surface)
    sin_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    D[:, 0] = (3/7) * sin_profile  # Si in Si3N4
    D[:, 1] = (4/7) * sin_profile  # N

    # SiO2 layer (interface)
    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    D[:, 2] = (1/3) * sio2_profile  # Si in SiO2
    D[:, 3] = (2/3) * sio2_profile  # O

    # Si substrate
    D[:, 4] = 0.5 * (1 + erf((depth - z2) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def create_on_pattern(depth, o_thickness, n_thickness, interface_width=5.0):
    """
    Create ON pattern: SiO2 / Si3N4 / Si (N at interface).

    Elements: [Si_o, O, Si_n, N, Si_bulk] = [0, 1, 2, 3, 4]
    """
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = o_thickness
    z2 = o_thickness + n_thickness

    D = np.zeros((len(depth), 5))

    # SiO2 layer (surface)
    sio2_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    D[:, 0] = (1/3) * sio2_profile  # Si in SiO2
    D[:, 1] = (2/3) * sio2_profile  # O

    # Si3N4 layer (interface)
    sin_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    D[:, 2] = (3/7) * sin_profile  # Si in Si3N4
    D[:, 3] = (4/7) * sin_profile  # N

    # Si substrate
    D[:, 4] = 0.5 * (1 + erf((depth - z2) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def create_non_pattern(depth, n1_thick, o_thick, n2_thick, interface_width=5.0):
    """
    Create NON pattern: Si3N4 / SiO2 / Si3N4 / Si (N at surface AND interface).

    When N1 and N2 are chemically distinguishable:
    Elements: [Si_n1, N1, Si_o, O, Si_n2, N2, Si_bulk] = [0, 1, 2, 3, 4, 5, 6]

    When N1 and N2 are merged (same chemical state):
    Elements: [Si_n, N, Si_o, O, Si_bulk] = [0, 1, 2, 3, 4]
    """
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = n1_thick
    z2 = n1_thick + o_thick
    z3 = n1_thick + o_thick + n2_thick

    # Distinguishable case (7 elements)
    D7 = np.zeros((len(depth), 7))

    # Surface Si3N4
    sin1_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    D7[:, 0] = (3/7) * sin1_profile  # Si in Si3N4 (surface)
    D7[:, 1] = (4/7) * sin1_profile  # N1 (surface)

    # SiO2 layer
    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    D7[:, 2] = (1/3) * sio2_profile  # Si in SiO2
    D7[:, 3] = (2/3) * sio2_profile  # O

    # Interface Si3N4
    sin2_profile = 0.5 * (1 + erf((depth - z2) / sigma)) * 0.5 * (1 - erf((depth - z3) / sigma))
    D7[:, 4] = (3/7) * sin2_profile  # Si in Si3N4 (interface)
    D7[:, 5] = (4/7) * sin2_profile  # N2 (interface)

    # Si substrate
    D7[:, 6] = 0.5 * (1 + erf((depth - z3) / sigma))

    D7 = np.clip(D7, 1e-6, None)
    D7 = D7 / D7.sum(axis=1, keepdims=True)

    # Merged case (5 elements) - N1 and N2 are same signal
    D5 = np.zeros((len(depth), 5))
    D5[:, 0] = D7[:, 0] + D7[:, 4]  # Si_n = Si_n1 + Si_n2
    D5[:, 1] = D7[:, 1] + D7[:, 5]  # N = N1 + N2
    D5[:, 2] = D7[:, 2]  # Si_o
    D5[:, 3] = D7[:, 3]  # O
    D5[:, 4] = D7[:, 6]  # Si_bulk

    D5 = np.clip(D5, 1e-6, None)
    D5 = D5 / D5.sum(axis=1, keepdims=True)

    return D7, D5


def test_no_pattern_semifree():
    """Test NO pattern with SemiFreeSolver."""
    print("\n" + "=" * 70)
    print("Test: NO Pattern with SemiFreeSolver")
    print("Structure: Si3N4 (5A) / SiO2 (20A) / Si")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]
    true_n_thick = 5.0
    true_o_thick = 20.0

    elements = [
        ElementConfig('Si', '2p', 'Si3N4', oxidation_state=4),  # Si in Si3N4
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),  # N
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),   # Si in SiO2
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),   # O
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),     # Si bulk
    ]

    D_true = create_no_pattern(depth, true_n_thick, true_o_thick)

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)
    I_true = tm.forward(D_true, normalize=True)

    # Add noise
    np.random.seed(42)
    noise = 0.02 * np.random.randn(*I_true.shape)
    I_obs = I_true * (1 + noise)
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    # SemiFreeSolver setup
    # film elements: Si_n(0), N(1), Si_o(2), O(3)
    # substrate: Si_bulk(4)
    semi_config = SemiFreeSolverConfig(
        free_indices=[0, 2],  # Si in nitride and oxide (free)
        linked_indices=[1, 3],  # N and O (linked by charge)
        substrate_indices=[4],  # Si bulk
        thickness_init=30.0,
        abruptness_init=5.0,
        thickness_bounds=(5.0, 50.0),
        optimize_abruptness=False,
    )

    for method in ['LLS', 'L1', 'L2', 'MEM']:
        alpha = 0.0 if method == 'LLS' else 0.01
        config = SolverConfig(method=method, alpha=alpha, max_iter=300)

        solver = SemiFreeSolver(tm, config, semi_config, elements=elements)
        result = solver.solve(I_obs)

        print(f"\n{method}: thickness={result.thickness:.1f}A, chi2={result.chi2:.6f}, success={result.success}")

        # Check N profile
        N_profile = result.D_opt[:, 1]
        N_peak_depth = depth[np.argmax(N_profile)]
        N_peak_val = np.max(N_profile)
        print(f"  N peak: depth={N_peak_depth:.1f}A, value={N_peak_val:.3f}")


def test_on_pattern_semifree():
    """Test ON pattern with SemiFreeSolver."""
    print("\n" + "=" * 70)
    print("Test: ON Pattern with SemiFreeSolver")
    print("Structure: SiO2 (20A) / Si3N4 (5A) / Si")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]
    true_o_thick = 20.0
    true_n_thick = 5.0

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),   # Si in SiO2
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),   # O
        ElementConfig('Si', '2p', 'Si3N4', oxidation_state=4),  # Si in Si3N4
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),  # N
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),     # Si bulk
    ]

    D_true = create_on_pattern(depth, true_o_thick, true_n_thick)

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)
    I_true = tm.forward(D_true, normalize=True)

    np.random.seed(42)
    noise = 0.02 * np.random.randn(*I_true.shape)
    I_obs = I_true * (1 + noise)
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    semi_config = SemiFreeSolverConfig(
        free_indices=[0, 2],  # Si in oxide and nitride
        linked_indices=[1, 3],  # O and N
        substrate_indices=[4],  # Si bulk
        thickness_init=30.0,
        abruptness_init=5.0,
        thickness_bounds=(5.0, 50.0),
        optimize_abruptness=False,
    )

    for method in ['LLS', 'L1', 'L2', 'MEM']:
        alpha = 0.0 if method == 'LLS' else 0.01
        config = SolverConfig(method=method, alpha=alpha, max_iter=300)

        solver = SemiFreeSolver(tm, config, semi_config, elements=elements)
        result = solver.solve(I_obs)

        print(f"\n{method}: thickness={result.thickness:.1f}A, chi2={result.chi2:.6f}, success={result.success}")

        # Check N profile - should peak at interface
        N_profile = result.D_opt[:, 3]
        N_peak_depth = depth[np.argmax(N_profile)]
        N_peak_val = np.max(N_profile)
        print(f"  N peak: depth={N_peak_depth:.1f}A, value={N_peak_val:.3f}")


def test_non_pattern_merged_semifree():
    """Test NON pattern (merged N) with SemiFreeSolver and degree adjustment."""
    print("\n" + "=" * 70)
    print("Test: NON Pattern (MERGED N) with SemiFreeSolver + degree adjustment")
    print("Structure: Si3N4 (5A) / SiO2 (15A) / Si3N4 (5A) / Si")
    print("N at surface and interface CANNOT be chemically distinguished")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]
    true_n1 = 5.0
    true_o = 15.0
    true_n2 = 5.0

    # Create merged N profile (5 elements, not 7)
    _, D_true = create_non_pattern(depth, true_n1, true_o, true_n2)

    elements = [
        ElementConfig('Si', '2p', 'Si3N4', oxidation_state=4),  # Si in Si3N4 (both layers)
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),  # N (merged)
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),   # Si in SiO2
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),   # O
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),     # Si bulk
    ]

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)
    I_true = tm.forward(D_true, normalize=True)

    np.random.seed(42)
    noise = 0.02 * np.random.randn(*I_true.shape)
    I_obs = I_true * (1 + noise)
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    print(f"\nTrue N profile: bimodal at z={true_n1/2:.1f}A and z={true_n1 + true_o + true_n2/2:.1f}A")

    # Test different degree values
    semi_config = SemiFreeSolverConfig(
        free_indices=[0, 2],  # Si_n, Si_o
        linked_indices=[1, 3],  # N, O
        substrate_indices=[4],  # Si bulk
        thickness_init=30.0,
        abruptness_init=5.0,
        thickness_bounds=(10.0, 40.0),
        optimize_abruptness=False,
    )

    results = {}
    degree_values = [1.0, 2.0, 5.0, 10.0, 20.0]

    for degree in degree_values:
        config = SolverConfig(
            method='LLS',
            alpha=0.0,
            degree=degree,
            max_iter=500
        )

        solver = SemiFreeSolver(tm, config, semi_config, elements=elements)
        result = solver.solve(I_obs)

        # Analyze N profile
        N_profile = result.D_opt[:, 1]
        N_max = np.max(N_profile)
        N_peaks = []

        # Find local maxima
        for i in range(1, len(N_profile) - 1):
            if N_profile[i] > N_profile[i-1] and N_profile[i] > N_profile[i+1]:
                if N_profile[i] > 0.1 * N_max:  # Significant peaks only
                    N_peaks.append((depth[i], N_profile[i]))

        n_peaks = len(N_peaks)
        is_bimodal = n_peaks >= 2

        results[degree] = {
            'thickness': result.thickness,
            'chi2': result.chi2,
            'success': result.success,
            'n_peaks': n_peaks,
            'peaks': N_peaks,
            'is_bimodal': is_bimodal,
            'D_opt': result.D_opt,
        }

        peak_str = ", ".join([f"{z:.0f}A({v:.2f})" for z, v in N_peaks[:3]])
        bimodal_mark = "BIMODAL!" if is_bimodal else "unimodal"
        print(f"\ndegree={degree:4.1f}: thickness={result.thickness:.1f}A, chi2={result.chi2:.6f}")
        print(f"  N peaks ({n_peaks}): {peak_str} -> {bimodal_mark}")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # True profile
    ax = axes[0, 0]
    ax.plot(depth, D_true[:, 1], 'b-', linewidth=2, label='N (merged)')
    ax.plot(depth, D_true[:, 3], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_true[:, 4], 'k--', linewidth=1, label='Si bulk')
    ax.axvline(true_n1, color='b', linestyle=':', alpha=0.5)
    ax.axvline(true_n1 + true_o, color='b', linestyle=':', alpha=0.5)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('TRUE Profile (NON)')
    ax.legend()

    # Plot each degree result
    for i, degree in enumerate(degree_values[:5]):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row, col]

        D_opt = results[degree]['D_opt']
        ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
        ax.plot(depth, D_opt[:, 3], 'r-', linewidth=2, label='O')
        ax.plot(depth, D_opt[:, 4], 'k--', linewidth=1, label='Si bulk')
        ax.axvline(true_n1, color='b', linestyle=':', alpha=0.5)
        ax.axvline(true_n1 + true_o, color='b', linestyle=':', alpha=0.5)
        ax.set_xlim(0, 50)
        ax.set_xlabel('Depth (A)')
        ax.set_ylabel('Concentration')
        bimodal = "BIMODAL" if results[degree]['is_bimodal'] else "unimodal"
        ax.set_title(f'degree={degree}: {bimodal}')
        ax.legend()

    plt.suptitle('NON Pattern Recovery with SemiFreeSolver + degree adjustment', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/semifree_non_pattern_degree.png', dpi=150)
    print("\nSaved: outputs/semifree_non_pattern_degree.png")
    plt.close()

    return results


if __name__ == "__main__":
    test_no_pattern_semifree()
    test_on_pattern_semifree()
    test_non_pattern_merged_semifree()

    print("\n" + "=" * 70)
    print("All SemiFreeSolver nitride pattern tests completed!")
    print("=" * 70)
