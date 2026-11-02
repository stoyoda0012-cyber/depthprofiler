"""
FreeSolver with Bimodal Prior for NON Pattern

Test if FreeSolver + bimodal prior + degree adjustment can detect
bimodal N distribution when N1 and N2 cannot be chemically distinguished.

Key finding from previous session:
- Bimodal prior + degree=5 successfully detects bimodal distribution

Usage:
    python -m deppro.tests.solver.test_freesolver_bimodal_prior
"""

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import FreeSolver, SolverConfig
from deppro.paths import ensure_outputs_dir


def create_non_merged_profile(depth, n1_thick, o_thick, n2_thick, interface_width=5.0):
    """
    Create NON pattern with merged N (5 elements).

    Structure: Si3N4 / SiO2 / Si3N4 / Si
    N1 (surface) and N2 (interface) are chemically indistinguishable.

    Elements: [Si_n, N, Si_o, O, Si_bulk] = [0, 1, 2, 3, 4]
    """
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = n1_thick
    z2 = n1_thick + o_thick
    z3 = n1_thick + o_thick + n2_thick

    D = np.zeros((len(depth), 5))

    # Surface Si3N4
    sin1_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    D[:, 0] += (3/7) * sin1_profile  # Si_n
    D[:, 1] += (4/7) * sin1_profile  # N (surface)

    # SiO2 layer
    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    D[:, 2] = (1/3) * sio2_profile  # Si_o
    D[:, 3] = (2/3) * sio2_profile  # O

    # Interface Si3N4
    sin2_profile = 0.5 * (1 + erf((depth - z2) / sigma)) * 0.5 * (1 - erf((depth - z3) / sigma))
    D[:, 0] += (3/7) * sin2_profile  # Si_n (added to surface)
    D[:, 1] += (4/7) * sin2_profile  # N (interface, merged with surface)

    # Si substrate
    D[:, 4] = 0.5 * (1 + erf((depth - z3) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def create_exponential_prior(depth, n_elements, substrate_idx, d_penetration):
    """Create exponential prior (Livesey 1994)."""
    D_prior = np.zeros((len(depth), n_elements))

    # Substrate: exponential increase
    D_prior[:, substrate_idx] = 1 - np.exp(-depth / d_penetration)

    # Film elements: exponential decrease
    for i in range(n_elements):
        if i != substrate_idx:
            D_prior[:, i] = np.exp(-depth / d_penetration)

    D_prior = np.clip(D_prior, 1e-6, None)
    D_prior = D_prior / D_prior.sum(axis=1, keepdims=True)
    return D_prior


def create_bimodal_prior(depth, n_elements, N_idx, Si_n_idx, substrate_idx,
                         z_surface, z_interface, d_penetration, width=5.0):
    """
    Create bimodal prior for NON pattern.

    N has peaks at surface and interface.
    """
    sigma = width / (2 * np.sqrt(np.log(2)))
    D_prior = np.zeros((len(depth), n_elements))

    # Substrate: exponential increase from interface
    D_prior[:, substrate_idx] = 1 - np.exp(-depth / d_penetration)

    # N: bimodal (surface + interface peaks)
    surface_peak = np.exp(-0.5 * ((depth - z_surface) / sigma) ** 2)
    interface_peak = np.exp(-0.5 * ((depth - z_interface) / sigma) ** 2)
    D_prior[:, N_idx] = 0.5 * (surface_peak + interface_peak)

    # Si_n follows N
    D_prior[:, Si_n_idx] = D_prior[:, N_idx] * (3/4)  # Si3N4 ratio

    # Other film elements: fill remaining
    film_idx = [i for i in range(n_elements) if i not in [N_idx, Si_n_idx, substrate_idx]]
    remaining = 1.0 - D_prior[:, substrate_idx] - D_prior[:, N_idx] - D_prior[:, Si_n_idx]
    remaining = np.clip(remaining, 0, None)
    for i in film_idx:
        D_prior[:, i] = remaining / len(film_idx)

    D_prior = np.clip(D_prior, 1e-6, None)
    D_prior = D_prior / D_prior.sum(axis=1, keepdims=True)
    return D_prior


def analyze_n_profile(depth, N_profile, true_n1, true_o, true_n2):
    """Analyze N profile for bimodal detection."""
    N_max = np.max(N_profile)
    peaks = []

    # Find local maxima
    for i in range(1, len(N_profile) - 1):
        if N_profile[i] > N_profile[i-1] and N_profile[i] > N_profile[i+1]:
            if N_profile[i] > 0.1 * N_max:  # Significant peaks only
                peaks.append((depth[i], N_profile[i]))

    n_peaks = len(peaks)
    is_bimodal = n_peaks >= 2

    # Expected peak positions
    expected_surface = true_n1 / 2
    expected_interface = true_n1 + true_o + true_n2 / 2

    return {
        'n_peaks': n_peaks,
        'peaks': peaks,
        'is_bimodal': is_bimodal,
        'expected_surface': expected_surface,
        'expected_interface': expected_interface,
    }


def test_bimodal_prior():
    """Test FreeSolver with bimodal prior for NON pattern."""
    print("=" * 70)
    print("Test: FreeSolver + Bimodal Prior for NON Pattern")
    print("Structure: Si3N4 (5A) / SiO2 (15A) / Si3N4 (5A) / Si")
    print("N at surface and interface CANNOT be chemically distinguished")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]
    true_n1 = 5.0
    true_o = 15.0
    true_n2 = 5.0

    elements = [
        ElementConfig('Si', '2p', 'Si3N4', oxidation_state=4),  # Si_n (0)
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),  # N (1) - MERGED
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),   # Si_o (2)
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),   # O (3)
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),     # Si_bulk (4)
    ]

    D_true = create_non_merged_profile(depth, true_n1, true_o, true_n2)

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)
    I_true = tm.forward(D_true, normalize=True)

    np.random.seed(42)
    noise = 0.02 * np.random.randn(*I_true.shape)
    I_obs = I_true * (1 + noise)
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    d_penetration = 30.0  # Approximate IMFP

    print(f"\nTrue N profile: bimodal at z={true_n1/2:.1f}A (surface) and z={true_n1 + true_o + true_n2/2:.1f}A (interface)")

    results = {}

    # Test 1: Exponential prior (baseline)
    print("\n--- Exponential Prior (baseline) ---")
    D_prior_exp = create_exponential_prior(depth, 5, substrate_idx=4, d_penetration=d_penetration)

    for degree in [1.0, 5.0, 10.0]:
        config = SolverConfig(method='MEM', alpha=0.0003, degree=degree, max_iter=500)
        solver = FreeSolver(tm, config, D_prior_exp, elements=elements)
        result = solver.solve(I_obs)

        analysis = analyze_n_profile(depth, result.D_opt[:, 1], true_n1, true_o, true_n2)
        bimodal_str = "BIMODAL!" if analysis['is_bimodal'] else "unimodal"

        peak_str = ", ".join([f"{z:.0f}A({v:.2f})" for z, v in analysis['peaks'][:3]])
        print(f"  Exp deg={degree}: Chi²={result.chi2:.4f}, peaks={analysis['n_peaks']} [{peak_str}] -> {bimodal_str}")

        results[f'exp_deg{degree:.1f}'] = {'result': result, 'analysis': analysis}

    # Test 2: Bimodal prior
    print("\n--- Bimodal Prior ---")
    D_prior_bimodal = create_bimodal_prior(
        depth, n_elements=5, N_idx=1, Si_n_idx=0, substrate_idx=4,
        z_surface=2.5, z_interface=22.5, d_penetration=d_penetration
    )

    for degree in [1.0, 5.0, 10.0]:
        config = SolverConfig(method='MEM', alpha=0.0003, degree=degree, max_iter=500)
        solver = FreeSolver(tm, config, D_prior_bimodal, elements=elements)
        result = solver.solve(I_obs)

        analysis = analyze_n_profile(depth, result.D_opt[:, 1], true_n1, true_o, true_n2)
        bimodal_str = "BIMODAL!" if analysis['is_bimodal'] else "unimodal"

        peak_str = ", ".join([f"{z:.0f}A({v:.2f})" for z, v in analysis['peaks'][:3]])
        print(f"  Bimodal deg={degree}: Chi²={result.chi2:.4f}, peaks={analysis['n_peaks']} [{peak_str}] -> {bimodal_str}")

        results[f'bimodal_deg{degree:.1f}'] = {'result': result, 'analysis': analysis}

    # Test 3: Bimodal prior with L2
    print("\n--- Bimodal Prior + L2 ---")
    for degree in [1.0, 5.0]:
        config = SolverConfig(method='L2', alpha=0.01, degree=degree, max_iter=500)
        solver = FreeSolver(tm, config, D_prior_bimodal, elements=elements)
        result = solver.solve(I_obs)

        analysis = analyze_n_profile(depth, result.D_opt[:, 1], true_n1, true_o, true_n2)
        bimodal_str = "BIMODAL!" if analysis['is_bimodal'] else "unimodal"

        peak_str = ", ".join([f"{z:.0f}A({v:.2f})" for z, v in analysis['peaks'][:3]])
        print(f"  L2 Bimodal deg={degree}: Chi²={result.chi2:.4f}, peaks={analysis['n_peaks']} [{peak_str}] -> {bimodal_str}")

        results[f'l2_bimodal_deg{degree:.1f}'] = {'result': result, 'analysis': analysis}

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # True profile
    ax = axes[0, 0]
    ax.plot(depth, D_true[:, 1], 'b-', linewidth=2, label='N (merged)')
    ax.plot(depth, D_true[:, 3], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_true[:, 4], 'k--', linewidth=1, label='Si bulk')
    ax.axvline(true_n1/2, color='b', linestyle=':', alpha=0.5, label='N peak expected')
    ax.axvline(true_n1 + true_o + true_n2/2, color='b', linestyle=':', alpha=0.5)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('TRUE Profile (NON)')
    ax.legend(loc='upper right')

    # Best exponential result
    ax = axes[0, 1]
    D_opt = results['exp_deg5.0']['result'].D_opt
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_opt[:, 3], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_opt[:, 4], 'k--', linewidth=1, label='Si bulk')
    ax.axvline(true_n1/2, color='b', linestyle=':', alpha=0.5)
    ax.axvline(true_n1 + true_o + true_n2/2, color='b', linestyle=':', alpha=0.5)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    bimodal = "BIMODAL" if results['exp_deg5.0']['analysis']['is_bimodal'] else "unimodal"
    ax.set_title(f'Exponential Prior (deg=5): {bimodal}')
    ax.legend(loc='upper right')

    # Best bimodal result
    ax = axes[0, 2]
    D_opt = results['bimodal_deg10.0']['result'].D_opt  # deg=10 shows best bimodal
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_opt[:, 3], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_opt[:, 4], 'k--', linewidth=1, label='Si bulk')
    ax.axvline(true_n1/2, color='b', linestyle=':', alpha=0.5)
    ax.axvline(true_n1 + true_o + true_n2/2, color='b', linestyle=':', alpha=0.5)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    bimodal = "BIMODAL" if results['bimodal_deg10.0']['analysis']['is_bimodal'] else "unimodal"
    ax.set_title(f'Bimodal Prior (deg=10): {bimodal}')
    ax.legend(loc='upper right')

    # Prior comparison
    ax = axes[1, 0]
    ax.plot(depth, D_prior_exp[:, 1], 'g--', linewidth=1.5, label='Exp Prior (N)')
    ax.plot(depth, D_prior_bimodal[:, 1], 'm--', linewidth=1.5, label='Bimodal Prior (N)')
    ax.plot(depth, D_true[:, 1], 'b-', linewidth=2, alpha=0.5, label='True N')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('Prior Comparison')
    ax.legend(loc='upper right')

    # Degree comparison for bimodal
    ax = axes[1, 1]
    for deg in [1.0, 5.0, 10.0]:
        D_opt = results[f'bimodal_deg{deg:.1f}']['result'].D_opt
        ax.plot(depth, D_opt[:, 1], linewidth=1.5, label=f'deg={deg}')
    ax.plot(depth, D_true[:, 1], 'k--', linewidth=2, alpha=0.5, label='True')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('N Concentration')
    ax.set_title('Bimodal Prior: degree comparison')
    ax.legend(loc='upper right')

    # Chi2 comparison
    ax = axes[1, 2]
    labels = ['Exp d=1', 'Exp d=5', 'Exp d=10', 'Bi d=1', 'Bi d=5', 'Bi d=10']
    chi2_vals = [
        results['exp_deg1.0']['result'].chi2,
        results['exp_deg5.0']['result'].chi2,
        results['exp_deg10.0']['result'].chi2,
        results['bimodal_deg1.0']['result'].chi2,
        results['bimodal_deg5.0']['result'].chi2,
        results['bimodal_deg10.0']['result'].chi2,
    ]
    colors = ['lightblue'] * 3 + ['lightgreen'] * 3
    ax.bar(labels, chi2_vals, color=colors)
    ax.set_ylabel('Chi²')
    ax.set_title('Chi² Comparison')
    ax.tick_params(axis='x', rotation=45)

    plt.suptitle('FreeSolver: Exponential vs Bimodal Prior for NON Pattern', fontsize=14)
    plt.tight_layout()
    output_path = str(ensure_outputs_dir() / 'freesolver_bimodal_prior_non.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Expected N peaks: surface @ {true_n1/2:.1f}A, interface @ {true_n1 + true_o + true_n2/2:.1f}A")

    best_exp = results['exp_deg5.0']
    best_bimodal = results['bimodal_deg10.0']  # deg=10 shows best bimodal

    print(f"\nExponential Prior (deg=5):")
    print(f"  Chi² = {best_exp['result'].chi2:.4f}")
    print(f"  Bimodal: {best_exp['analysis']['is_bimodal']}")
    if best_exp['analysis']['peaks']:
        print(f"  N peaks: {', '.join([f'{z:.0f}A' for z, _ in best_exp['analysis']['peaks'][:3]])}")

    print(f"\nBimodal Prior (deg=10):")
    print(f"  Chi² = {best_bimodal['result'].chi2:.4f}")
    print(f"  Bimodal: {best_bimodal['analysis']['is_bimodal']}")
    if best_bimodal['analysis']['peaks']:
        print(f"  N peaks: {', '.join([f'{z:.0f}A' for z, _ in best_bimodal['analysis']['peaks'][:3]])}")

    return results


if __name__ == "__main__":
    results = test_bimodal_prior()

    print("\n" + "=" * 70)
    print("FreeSolver Bimodal Prior test completed!")
    print("=" * 70)
