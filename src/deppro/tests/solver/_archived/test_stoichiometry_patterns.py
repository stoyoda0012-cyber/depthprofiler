"""
Stoichiometry Solver - Multiple Patterns Test

Test patterns:
1. NO: Si3N4 / SiO2 / Si (N on surface)
2. ON: SiO2 / Si3N4 / Si (N at interface)
3. NON: Si3N4 / SiO2 / Si3N4 / Si (N at surface AND interface)

All with merged Si4+ (cannot distinguish Si_n and Si_o chemically)

Usage:
    python -m deppro.tests.solver.test_stoichiometry_patterns
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig


def create_no_pattern(depth, n_thickness, o_thickness, interface_width=5.0):
    """NO: Si3N4 / SiO2 / Si (N on surface)"""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = n_thickness
    z2 = n_thickness + o_thickness

    D = np.zeros((len(depth), 4))

    # Si3N4 (surface)
    sin_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    si_n = (3/7) * sin_profile
    D[:, 1] = (4/7) * sin_profile  # N

    # SiO2
    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    si_o = (1/3) * sio2_profile
    D[:, 2] = (2/3) * sio2_profile  # O

    D[:, 0] = si_n + si_o  # Si4+
    D[:, 3] = 0.5 * (1 + erf((depth - z2) / sigma))  # Si_bulk

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def create_on_pattern(depth, o_thickness, n_thickness, interface_width=5.0):
    """ON: SiO2 / Si3N4 / Si (N at interface)"""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = o_thickness
    z2 = o_thickness + n_thickness

    D = np.zeros((len(depth), 4))

    # SiO2 (surface)
    sio2_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    si_o = (1/3) * sio2_profile
    D[:, 2] = (2/3) * sio2_profile  # O

    # Si3N4 (interface)
    sin_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    si_n = (3/7) * sin_profile
    D[:, 1] = (4/7) * sin_profile  # N

    D[:, 0] = si_n + si_o  # Si4+
    D[:, 3] = 0.5 * (1 + erf((depth - z2) / sigma))  # Si_bulk

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def create_non_pattern(depth, n1_thickness, o_thickness, n2_thickness, interface_width=5.0):
    """NON: Si3N4 / SiO2 / Si3N4 / Si (N at surface AND interface)"""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = n1_thickness
    z2 = n1_thickness + o_thickness
    z3 = n1_thickness + o_thickness + n2_thickness

    D = np.zeros((len(depth), 4))

    # Surface Si3N4
    sin1_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    si_n1 = (3/7) * sin1_profile
    n1 = (4/7) * sin1_profile

    # SiO2
    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    si_o = (1/3) * sio2_profile
    D[:, 2] = (2/3) * sio2_profile  # O

    # Interface Si3N4
    sin2_profile = 0.5 * (1 + erf((depth - z2) / sigma)) * 0.5 * (1 - erf((depth - z3) / sigma))
    si_n2 = (3/7) * sin2_profile
    n2 = (4/7) * sin2_profile

    D[:, 1] = n1 + n2  # N (merged)
    D[:, 0] = si_n1 + si_o + si_n2  # Si4+
    D[:, 3] = 0.5 * (1 + erf((depth - z3) / sigma))  # Si_bulk

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


class StoichiometrySolver:
    """Solver with stoichiometry constraint: Si4+ = N×(3/4) + O×(1/2)"""

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
        return {'D_opt': D_opt, 'I_opt': I_opt, 'chi2': chi2, 'success': result.success}


def analyze_profile(depth, D_opt, D_true, N_idx, O_idx):
    """Analyze N and O profiles."""
    # Find peaks
    N_opt = D_opt[:, N_idx]
    O_opt = D_opt[:, O_idx]
    N_true = D_true[:, N_idx]
    O_true = D_true[:, O_idx]

    # N peak positions
    N_peaks_opt = []
    N_peaks_true = []
    N_max = max(np.max(N_opt), np.max(N_true))

    for i in range(1, len(depth) - 1):
        if N_opt[i] > N_opt[i-1] and N_opt[i] > N_opt[i+1] and N_opt[i] > 0.1 * N_max:
            N_peaks_opt.append(depth[i])
        if N_true[i] > N_true[i-1] and N_true[i] > N_true[i+1] and N_true[i] > 0.1 * N_max:
            N_peaks_true.append(depth[i])

    # RMSE
    rmse_n = np.sqrt(np.mean((N_opt - N_true) ** 2))
    rmse_o = np.sqrt(np.mean((O_opt - O_true) ** 2))

    return {
        'N_peaks_opt': N_peaks_opt,
        'N_peaks_true': N_peaks_true,
        'rmse_n': rmse_n,
        'rmse_o': rmse_o,
        'is_bimodal_opt': len(N_peaks_opt) >= 2,
        'is_bimodal_true': len(N_peaks_true) >= 2,
    }


def test_all_patterns():
    """Test NO, ON, NON patterns."""
    print("=" * 70)
    print("Stoichiometry Solver - Multiple Patterns Test")
    print("Constraint: Si4+ = N × (3/4) + O × (1/2)")
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
        'NO': {
            'desc': 'Si3N4 (5Å) / SiO2 (20Å) / Si',
            'D_true': create_no_pattern(depth, 5.0, 20.0),
            'total_thickness': 25.0,
        },
        'ON': {
            'desc': 'SiO2 (20Å) / Si3N4 (5Å) / Si',
            'D_true': create_on_pattern(depth, 20.0, 5.0),
            'total_thickness': 25.0,
        },
        'NON': {
            'desc': 'Si3N4 (5Å) / SiO2 (15Å) / Si3N4 (5Å) / Si',
            'D_true': create_non_pattern(depth, 5.0, 15.0, 5.0),
            'total_thickness': 25.0,
        },
    }

    results = {}

    for name, pattern in patterns.items():
        print(f"\n{'=' * 70}")
        print(f"Pattern: {name} - {pattern['desc']}")
        print(f"{'=' * 70}")

        D_true = pattern['D_true']
        tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
        tm.compute(D_true, elements)
        I_true = tm.forward(D_true, normalize=True)

        np.random.seed(42)
        I_obs = I_true * (1 + 0.02 * np.random.randn(*I_true.shape))
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

        solver = StoichiometrySolver(
            tm, N_idx=1, O_idx=2, Si4_idx=0, substrate_idx=3, degree=1.0
        )

        result = solver.solve(I_obs, thickness_init=pattern['total_thickness'], abruptness=5.0)
        D_opt = result['D_opt']

        analysis = analyze_profile(depth, D_opt, D_true, N_idx=1, O_idx=2)

        print(f"\nChi² = {result['chi2']:.6f}")
        print(f"RMSE(N) = {analysis['rmse_n']:.4f}, RMSE(O) = {analysis['rmse_o']:.4f}")
        print(f"N peaks (true): {analysis['N_peaks_true']}")
        print(f"N peaks (opt):  {analysis['N_peaks_opt']}")

        if name == 'NON':
            print(f"Bimodal detection: true={analysis['is_bimodal_true']}, opt={analysis['is_bimodal_opt']}")

        # Detailed comparison
        print(f"\n{'Depth':>6} | {'N opt':>8} | {'N true':>8} | {'O opt':>8} | {'O true':>8}")
        print("-" * 50)
        for z in [0, 5, 10, 15, 20, 25]:
            print(f"{z:>6} | {D_opt[z, 1]:>8.4f} | {D_true[z, 1]:>8.4f} | {D_opt[z, 2]:>8.4f} | {D_true[z, 2]:>8.4f}")

        results[name] = {
            'D_true': D_true,
            'D_opt': D_opt,
            'chi2': result['chi2'],
            'analysis': analysis,
        }

    # Plot all patterns
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for i, (name, res) in enumerate(results.items()):
        D_true = res['D_true']
        D_opt = res['D_opt']

        # True
        ax = axes[i, 0]
        ax.plot(depth, D_true[:, 1], 'b-', linewidth=2, label='N')
        ax.plot(depth, D_true[:, 2], 'r-', linewidth=2, label='O')
        ax.plot(depth, D_true[:, 0], 'g-', linewidth=2, label='Si4+')
        ax.plot(depth, D_true[:, 3], 'k--', linewidth=1, label='Si_bulk')
        ax.set_xlim(0, 50)
        ax.set_xlabel('Depth (Å)')
        ax.set_ylabel('Concentration')
        ax.set_title(f'{name} TRUE')
        ax.legend(loc='upper right', fontsize=8)

        # Optimized
        ax = axes[i, 1]
        ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
        ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O')
        ax.plot(depth, D_opt[:, 0], 'g-', linewidth=2, label='Si4+')
        ax.plot(depth, D_opt[:, 3], 'k--', linewidth=1, label='Si_bulk')
        ax.set_xlim(0, 50)
        ax.set_xlabel('Depth (Å)')
        ax.set_ylabel('Concentration')
        ax.set_title(f'{name} OPT (Chi²={res["chi2"]:.5f})')
        ax.legend(loc='upper right', fontsize=8)

        # Comparison
        ax = axes[i, 2]
        ax.plot(depth, D_true[:, 1], 'b--', linewidth=1.5, alpha=0.7, label='N (true)')
        ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N (opt)')
        ax.plot(depth, D_true[:, 2], 'r--', linewidth=1.5, alpha=0.7, label='O (true)')
        ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O (opt)')
        ax.set_xlim(0, 50)
        ax.set_xlabel('Depth (Å)')
        ax.set_ylabel('Concentration')
        ax.set_title(f'{name} Comparison')
        ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Stoichiometry Solver: NO / ON / NON Patterns (Si4+ merged)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/stoichiometry_all_patterns.png', dpi=150)
    print(f"\nSaved: outputs/stoichiometry_all_patterns.png")
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Pattern':<10} | {'Chi²':>12} | {'RMSE(N)':>10} | {'RMSE(O)':>10} | {'Bimodal':>10}")
    print("-" * 60)
    for name, res in results.items():
        bimodal = "Yes" if res['analysis']['is_bimodal_opt'] else "No"
        if name != 'NON':
            bimodal = "N/A"
        print(f"{name:<10} | {res['chi2']:>12.6f} | {res['analysis']['rmse_n']:>10.4f} | {res['analysis']['rmse_o']:>10.4f} | {bimodal:>10}")

    return results


if __name__ == "__main__":
    results = test_all_patterns()
