"""
Stoichiometry Constraint Solver (v3)

Simplified version: No ratio objective needed.
Just use stoichiometry constraint: Si4+ = N × (3/4) + O × (1/2)

Usage:
    python -m deppro.tests.solver.test_ratio_objective_v3
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig


def create_no_pattern_4elem(depth, n_thickness, o_thickness, interface_width=5.0):
    """Create NO pattern with 4 elements (Si4+ merged)."""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = n_thickness
    z2 = n_thickness + o_thickness

    D = np.zeros((len(depth), 4))

    sin_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    si_n = (3/7) * sin_profile
    D[:, 1] = (4/7) * sin_profile  # N

    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    si_o = (1/3) * sio2_profile
    D[:, 2] = (2/3) * sio2_profile  # O

    D[:, 0] = si_n + si_o  # Si4+
    D[:, 3] = 0.5 * (1 + erf((depth - z2) / sigma))  # Si_bulk

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


class StoichiometrySolver:
    """
    Simplified solver with stoichiometry constraint only.

    Si4+ = N × (3/4) + O × (1/2)
    No ratio objective needed!
    """

    def __init__(
        self,
        transfer_matrix,
        N_idx, O_idx, Si4_idx, substrate_idx,
        Si_N_ratio=3/4,
        Si_O_ratio=1/2,
        degree=1.0,
    ):
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
        """Build profile with stoichiometry constraint."""
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
        """Simple Chi² objective only."""
        N_conc = x[:self.n_depth]
        O_conc = x[self.n_depth:2*self.n_depth]

        D = self._build_profile(N_conc, O_conc, thickness, abruptness)
        I_pred = self.tm.forward(D, normalize=True)

        chi2 = np.sum((I_pred - I_obs) ** 2)
        return self.degree * chi2

    def solve(self, I_obs, thickness_init=25.0, abruptness=5.0, max_iter=500):
        x0 = np.concatenate([
            np.ones(self.n_depth) * 0.3,
            np.ones(self.n_depth) * 0.3,
        ])

        bounds = Bounds(
            lb=np.zeros(2 * self.n_depth),
            ub=np.ones(2 * self.n_depth),
        )

        result = minimize(
            self._objective, x0,
            args=(I_obs, thickness_init, abruptness),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter},
        )

        N_conc = result.x[:self.n_depth]
        O_conc = result.x[self.n_depth:2*self.n_depth]
        D_opt = self._build_profile(N_conc, O_conc, thickness_init, abruptness)
        I_opt = self.tm.forward(D_opt, normalize=True)
        chi2 = np.sum((I_opt - I_obs) ** 2)

        return {
            'D_opt': D_opt,
            'I_opt': I_opt,
            'chi2': chi2,
            'success': result.success,
        }


def test_stoichiometry_solver():
    """Test simplified stoichiometry solver."""
    print("=" * 70)
    print("Test: Stoichiometry Solver (No λ_ratio needed!)")
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

    D_true = create_no_pattern_4elem(depth, 5.0, 20.0)

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)
    I_true = tm.forward(D_true, normalize=True)

    np.random.seed(42)
    I_obs = I_true * (1 + 0.02 * np.random.randn(*I_true.shape))
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    solver = StoichiometrySolver(
        tm, N_idx=1, O_idx=2, Si4_idx=0, substrate_idx=3,
        degree=1.0,
    )

    result = solver.solve(I_obs, thickness_init=25.0, abruptness=5.0, max_iter=500)
    D_opt = result['D_opt']

    print(f"\nChi² = {result['chi2']:.6f}")
    print(f"\n{'Depth':>6} | {'Si4+ opt':>10} | {'Si4+ true':>10} | {'N opt':>10} | {'N true':>10} | {'O opt':>10} | {'O true':>10}")
    print("-" * 90)
    for z in [0, 5, 10, 15, 20, 25]:
        print(f"{z:>6} | {D_opt[z, 0]:>10.4f} | {D_true[z, 0]:>10.4f} | {D_opt[z, 1]:>10.4f} | {D_true[z, 1]:>10.4f} | {D_opt[z, 2]:>10.4f} | {D_true[z, 2]:>10.4f}")

    # Error analysis
    print("\n--- Error Analysis ---")
    for z in [0, 10]:
        n_err = abs(D_opt[z, 1] - D_true[z, 1])
        o_err = abs(D_opt[z, 2] - D_true[z, 2])
        si_err = abs(D_opt[z, 0] - D_true[z, 0])
        print(f"z={z}A: N err={n_err:.4f}, O err={o_err:.4f}, Si4+ err={si_err:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(depth, D_true[:, 0], 'g--', linewidth=2, alpha=0.7, label='Si4+ (true)')
    ax.plot(depth, D_true[:, 1], 'b--', linewidth=2, alpha=0.7, label='N (true)')
    ax.plot(depth, D_true[:, 2], 'r--', linewidth=2, alpha=0.7, label='O (true)')
    ax.plot(depth, D_opt[:, 0], 'g-', linewidth=2, label='Si4+ (opt)')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N (opt)')
    ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O (opt)')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title(f'Stoichiometry Solver: Chi²={result["chi2"]:.5f}')
    ax.legend()

    ax = axes[1]
    ax.plot(depth, D_true[:, 1], 'b--', linewidth=2, label='N (true)')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N (opt)')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('N Concentration')
    ax.set_title('N Profile')
    ax.legend()

    ax = axes[2]
    ax.plot(depth, D_true[:, 2], 'r--', linewidth=2, label='O (true)')
    ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O (opt)')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('O Concentration')
    ax.set_title('O Profile')
    ax.legend()

    plt.suptitle('Stoichiometry Constraint Only (No λ_ratio)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/stoichiometry_solver.png', dpi=150)
    print(f"\nSaved: outputs/stoichiometry_solver.png")
    plt.close()

    return result


if __name__ == "__main__":
    test_stoichiometry_solver()
