"""
Stoichiometry Solver - Extreme Degree Test

Test extreme degree values for ON and NON patterns.

Usage:
    python -m deppro.tests.solver.test_stoichiometry_degree_extreme
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig


def create_on_pattern(depth, o_thickness, n_thickness, interface_width=5.0):
    """ON: SiO2 / Si3N4 / Si"""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = o_thickness
    z2 = o_thickness + n_thickness

    D = np.zeros((len(depth), 4))

    sio2_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    si_o = (1/3) * sio2_profile
    D[:, 2] = (2/3) * sio2_profile

    sin_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    si_n = (3/7) * sin_profile
    D[:, 1] = (4/7) * sin_profile

    D[:, 0] = si_n + si_o
    D[:, 3] = 0.5 * (1 + erf((depth - z2) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def create_non_pattern(depth, n1_thickness, o_thickness, n2_thickness, interface_width=5.0):
    """NON: Si3N4 / SiO2 / Si3N4 / Si"""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = n1_thickness
    z2 = n1_thickness + o_thickness
    z3 = n1_thickness + o_thickness + n2_thickness

    D = np.zeros((len(depth), 4))

    sin1_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    si_n1 = (3/7) * sin1_profile
    n1 = (4/7) * sin1_profile

    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    si_o = (1/3) * sio2_profile
    D[:, 2] = (2/3) * sio2_profile

    sin2_profile = 0.5 * (1 + erf((depth - z2) / sigma)) * 0.5 * (1 - erf((depth - z3) / sigma))
    si_n2 = (3/7) * sin2_profile
    n2 = (4/7) * sin2_profile

    D[:, 1] = n1 + n2
    D[:, 0] = si_n1 + si_o + si_n2
    D[:, 3] = 0.5 * (1 + erf((depth - z3) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


class StoichiometrySolver:
    """Solver with stoichiometry constraint and degree adjustment."""

    def __init__(self, transfer_matrix, N_idx, O_idx, Si4_idx, substrate_idx,
                 Si_N_ratio=3/4, Si_O_ratio=1/2, degree=1.0):
        self.tm = transfer_matrix
        self.depth = transfer_matrix.depth
        self.n_depth = len(self.depth)
        self.n_elements = transfer_matrix.n_elements
        self.N_idx = N_idx
        self.O_idx = O_idx
        self.Si4_idx = Si4_idx
        self.substrate_idx = substrate_idx
        self.Si_N_ratio = Si_N_ratio
        self.Si_O_ratio = Si_O_ratio
        self.degree = degree

    def _build_substrate_profile(self, thickness, abruptness):
        sigma = abruptness / (2 * np.sqrt(np.log(2)))
        return 0.5 * (1 + erf((self.depth - thickness) / sigma))

    def _build_profile(self, N_conc, O_conc, thickness, abruptness):
        D = np.zeros((self.n_depth, self.n_elements))
        sub_profile = self._build_substrate_profile(thickness, abruptness)
        D[:, self.substrate_idx] = sub_profile
        film_scale = np.maximum(1.0 - sub_profile, 1e-6)

        N_scaled = N_conc * film_scale
        O_scaled = O_conc * film_scale
        Si4_scaled = N_scaled * self.Si_N_ratio + O_scaled * self.Si_O_ratio

        D[:, self.N_idx] = N_scaled
        D[:, self.O_idx] = O_scaled
        D[:, self.Si4_idx] = Si4_scaled

        D = np.clip(D, 1e-6, 1.0)
        D = D / np.maximum(D.sum(axis=1, keepdims=True), 1e-6)
        return D

    def _objective(self, x, I_obs, thickness, abruptness):
        N_conc = x[:self.n_depth]
        O_conc = x[self.n_depth:2*self.n_depth]
        D = self._build_profile(N_conc, O_conc, thickness, abruptness)
        I_pred = self.tm.forward(D, normalize=True)
        chi2 = np.sum((I_pred - I_obs) ** 2)
        return self.degree * chi2

    def solve(self, I_obs, thickness_init=25.0, abruptness=5.0, max_iter=500):
        x0 = np.concatenate([np.ones(self.n_depth) * 0.3, np.ones(self.n_depth) * 0.3])
        bounds = Bounds(lb=np.zeros(2 * self.n_depth), ub=np.ones(2 * self.n_depth))
        result = minimize(self._objective, x0, args=(I_obs, thickness_init, abruptness),
                          method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})

        N_conc = result.x[:self.n_depth]
        O_conc = result.x[self.n_depth:2*self.n_depth]
        D_opt = self._build_profile(N_conc, O_conc, thickness_init, abruptness)
        I_opt = self.tm.forward(D_opt, normalize=True)
        chi2 = np.sum((I_opt - I_obs) ** 2)
        return {'D_opt': D_opt, 'chi2': chi2, 'success': result.success}


def test_extreme_degree():
    """Test extreme degree values."""
    print("=" * 70)
    print("Stoichiometry Solver - EXTREME Degree Test")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    patterns = {
        'ON': {
            'D_true': create_on_pattern(depth, 20.0, 5.0),
            'thickness': 25.0,
        },
        'NON': {
            'D_true': create_non_pattern(depth, 5.0, 15.0, 5.0),
            'thickness': 25.0,
        },
    }

    # Extreme degree values
    degree_values = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    all_results = {}

    for name, pattern in patterns.items():
        print(f"\n{'=' * 70}")
        print(f"Pattern: {name}")
        print(f"{'=' * 70}")

        D_true = pattern['D_true']
        tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
        tm.compute(D_true, elements)
        I_true = tm.forward(D_true, normalize=True)

        np.random.seed(42)
        I_obs = I_true * (1 + 0.02 * np.random.randn(*I_true.shape))
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

        results = {}
        for degree in degree_values:
            solver = StoichiometrySolver(
                tm, N_idx=1, O_idx=2, Si4_idx=0, substrate_idx=3, degree=degree
            )
            result = solver.solve(I_obs, thickness_init=pattern['thickness'], abruptness=5.0, max_iter=1000)
            D_opt = result['D_opt']

            # Calculate RMSE
            rmse_n = np.sqrt(np.mean((D_opt[:, 1] - D_true[:, 1]) ** 2))
            rmse_o = np.sqrt(np.mean((D_opt[:, 2] - D_true[:, 2]) ** 2))

            # Find N peaks
            N_peaks = []
            N_max = np.max(D_opt[:, 1])
            for i in range(1, len(depth) - 1):
                if D_opt[i, 1] > D_opt[i-1, 1] and D_opt[i, 1] > D_opt[i+1, 1]:
                    if D_opt[i, 1] > 0.1 * N_max:
                        N_peaks.append((depth[i], D_opt[i, 1]))

            is_bimodal = len(N_peaks) >= 2

            results[degree] = {
                'D_opt': D_opt,
                'chi2': result['chi2'],
                'rmse_n': rmse_n,
                'rmse_o': rmse_o,
                'N_peaks': N_peaks,
                'is_bimodal': is_bimodal,
            }

            peak_str = ", ".join([f"{z:.0f}Å({v:.2f})" for z, v in N_peaks[:3]]) if N_peaks else "none"
            bimodal_str = "BIMODAL!" if is_bimodal else ""
            print(f"  deg={degree:7.1f}: Chi²={result['chi2']:.6f}, RMSE(N)={rmse_n:.4f}, peaks=[{peak_str}] {bimodal_str}")

        all_results[name] = {'D_true': D_true, 'results': results}

    # Plot NON results in detail
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    D_true = all_results['NON']['D_true']
    results = all_results['NON']['results']

    # True profile
    ax = axes[0, 0]
    ax.plot(depth, D_true[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_true[:, 2], 'r-', linewidth=2, label='O')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title('NON TRUE')
    ax.legend()

    # Selected degrees
    selected_degrees = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

    for idx, degree in enumerate(selected_degrees):
        if degree not in results:
            continue
        row = (idx + 1) // 4
        col = (idx + 1) % 4
        ax = axes[row, col]

        D_opt = results[degree]['D_opt']
        ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
        ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O')
        ax.plot(depth, D_true[:, 1], 'b--', alpha=0.3)
        ax.plot(depth, D_true[:, 2], 'r--', alpha=0.3)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 0.8)
        ax.set_xlabel('Depth (Å)')

        bimodal = "BIMODAL" if results[degree]['is_bimodal'] else ""
        ax.set_title(f'deg={degree} {bimodal}')
        ax.legend(fontsize=8)

    # Last subplot: N profile comparison
    ax = axes[2, 3]
    ax.plot(depth, D_true[:, 1], 'k-', linewidth=2, label='True')
    for degree in [1.0, 10.0, 100.0, 1000.0]:
        if degree in results:
            ax.plot(depth, results[degree]['D_opt'][:, 1], linewidth=1.5, label=f'deg={degree}')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('N Concentration')
    ax.set_title('N Profile Comparison')
    ax.legend(fontsize=8)

    plt.suptitle('NON Pattern: Extreme Degree Sweep', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/stoichiometry_degree_extreme.png', dpi=150)
    print(f"\nSaved: outputs/stoichiometry_degree_extreme.png")
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: NON Pattern Bimodal Detection")
    print("=" * 70)
    print(f"{'Degree':>10} | {'Chi²':>12} | {'RMSE(N)':>10} | {'Bimodal':>8} | N Peaks")
    print("-" * 70)
    for degree in degree_values:
        res = all_results['NON']['results'][degree]
        bimodal = "Yes" if res['is_bimodal'] else "No"
        peaks = ", ".join([f"{z:.0f}Å" for z, _ in res['N_peaks'][:3]]) if res['N_peaks'] else "-"
        print(f"{degree:>10.1f} | {res['chi2']:>12.6f} | {res['rmse_n']:>10.4f} | {bimodal:>8} | {peaks}")

    return all_results


if __name__ == "__main__":
    test_extreme_degree()
