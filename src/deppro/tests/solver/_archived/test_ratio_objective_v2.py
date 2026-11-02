"""
I(N)/I(O) Ratio Objective Solver v2

Key improvement: Si4+ is computed from N and O via stoichiometry
- Si4+ = Si_n + Si_o = N × (3/4) + O × (1/2)
- This ensures physical consistency

Usage:
    python -m deppro.tests.solver.test_ratio_objective_v2
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig


def create_no_pattern_4elem(depth, n_thickness, o_thickness, interface_width=5.0):
    """
    Create NO pattern with 4 elements (Si4+ merged).
    Elements: [Si4+, N, O, Si_bulk] = [0, 1, 2, 3]
    """
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = n_thickness
    z2 = n_thickness + o_thickness

    D = np.zeros((len(depth), 4))

    # Si3N4 layer (surface)
    sin_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    si_n = (3/7) * sin_profile
    D[:, 1] = (4/7) * sin_profile  # N

    # SiO2 layer
    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    si_o = (1/3) * sio2_profile
    D[:, 2] = (2/3) * sio2_profile  # O

    # Si4+ = Si_n + Si_o (merged)
    D[:, 0] = si_n + si_o

    # Si substrate
    D[:, 3] = 0.5 * (1 + erf((depth - z2) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


class RatioObjectiveSolverV2:
    """
    Solver v2: Si4+ computed from N and O via stoichiometry.

    Key constraint:
    - Si4+ = N × (3/4) + O × (1/2)
    - Si3N4: Si:N = 3:4 → Si_n = N × (3/4)
    - SiO2:  Si:O = 1:2 → Si_o = O × (1/2)
    """

    def __init__(
        self,
        transfer_matrix,
        N_idx,
        O_idx,
        Si4_idx,
        substrate_idx,
        Si_N_ratio=3/4,  # Si/N ratio in Si3N4
        Si_O_ratio=1/2,  # Si/O ratio in SiO2
        lambda_ratio=1.0,
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
        self.lambda_ratio = lambda_ratio
        self.degree = degree

    def _build_substrate_profile(self, thickness, abruptness):
        sigma = abruptness / (2 * np.sqrt(np.log(2)))
        erf_profile = 0.5 * (1 + erf((self.depth - thickness) / sigma))
        return erf_profile

    def _build_profile(self, N_conc, O_conc, thickness, abruptness):
        """
        Build depth profile with Si4+ computed from stoichiometry.

        Si4+ = N × (3/4) + O × (1/2)
        """
        D = np.zeros((self.n_depth, self.n_elements))

        # Substrate profile
        sub_profile = self._build_substrate_profile(thickness, abruptness)
        D[:, self.substrate_idx] = sub_profile

        # Film region scale
        film_scale = np.maximum(1.0 - sub_profile, 1e-6)

        # N and O concentrations (scaled to film region)
        N_scaled = N_conc * film_scale
        O_scaled = O_conc * film_scale

        # Si4+ from stoichiometry: Si4+ = Si_n + Si_o = N×(3/4) + O×(1/2)
        Si4_scaled = N_scaled * self.Si_N_ratio + O_scaled * self.Si_O_ratio

        D[:, self.N_idx] = N_scaled
        D[:, self.O_idx] = O_scaled
        D[:, self.Si4_idx] = Si4_scaled

        # Normalize (this may adjust values slightly)
        D = np.clip(D, 1e-6, 1.0)
        D_sum = D.sum(axis=1, keepdims=True)
        D = D / np.maximum(D_sum, 1e-6)

        return D

    def _objective(self, x, I_obs, thickness, abruptness):
        """Combined objective: Chi² + λ × ratio_loss"""
        N_conc = x[:self.n_depth]
        O_conc = x[self.n_depth:2*self.n_depth]

        D = self._build_profile(N_conc, O_conc, thickness, abruptness)
        I_pred = self.tm.forward(D, normalize=True)

        # Chi² loss
        chi2 = np.sum((I_pred - I_obs) ** 2)

        # Ratio loss
        eps = 1e-8
        ratio_pred = I_pred[:, self.N_idx] / np.maximum(I_pred[:, self.O_idx], eps)
        ratio_obs = I_obs[:, self.N_idx] / np.maximum(I_obs[:, self.O_idx], eps)
        ratio_loss = np.sum((ratio_pred - ratio_obs) ** 2)

        return self.degree * (chi2 + self.lambda_ratio * ratio_loss)

    def solve(self, I_obs, thickness_init=25.0, abruptness=5.0, max_iter=500):
        """Solve with stoichiometry constraint."""
        x0 = np.concatenate([
            np.ones(self.n_depth) * 0.3,  # N_conc
            np.ones(self.n_depth) * 0.3,  # O_conc
        ])

        bounds = Bounds(
            lb=np.zeros(2 * self.n_depth),
            ub=np.ones(2 * self.n_depth),
        )

        result = minimize(
            self._objective,
            x0,
            args=(I_obs, thickness_init, abruptness),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter},
        )

        N_conc = result.x[:self.n_depth]
        O_conc = result.x[self.n_depth:2*self.n_depth]
        D_opt = self._build_profile(N_conc, O_conc, thickness_init, abruptness)
        I_opt = self.tm.forward(D_opt, normalize=True)
        chi2 = np.sum((I_opt - I_obs) ** 2)

        eps = 1e-8
        ratio_pred = I_opt[:, self.N_idx] / np.maximum(I_opt[:, self.O_idx], eps)
        ratio_obs = I_obs[:, self.N_idx] / np.maximum(I_obs[:, self.O_idx], eps)
        ratio_loss = np.sum((ratio_pred - ratio_obs) ** 2)

        return {
            'D_opt': D_opt,
            'I_opt': I_opt,
            'chi2': chi2,
            'ratio_loss': ratio_loss,
            'thickness': thickness_init,
            'success': result.success,
            'N_conc': N_conc,
            'O_conc': O_conc,
        }


def test_ratio_objective_v2():
    """Test ratio objective solver v2 with stoichiometry constraint."""
    print("=" * 70)
    print("Test: Ratio Objective Solver v2 (Si4+ from stoichiometry)")
    print("Structure: Si3N4 (5A) / SiO2 (20A) / Si")
    print("Constraint: Si4+ = N × (3/4) + O × (1/2)")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]
    true_n_thick = 5.0
    true_o_thick = 20.0

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),   # Si4+
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),  # N
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),   # O
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),     # Si_bulk
    ]

    D_true = create_no_pattern_4elem(depth, true_n_thick, true_o_thick)

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)
    I_true = tm.forward(D_true, normalize=True)

    np.random.seed(42)
    I_obs = I_true * (1 + 0.02 * np.random.randn(*I_true.shape))
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    print("\n--- TRUE Profile ---")
    print(f"{'Depth':>6} | {'Si4+':>8} | {'N':>8} | {'O':>8} | {'Si_bulk':>8} | Si4+_calc")
    print("-" * 70)
    for z in [0, 5, 10, 15, 20, 25]:
        si4_calc = D_true[z, 1] * (3/4) + D_true[z, 2] * (1/2)
        print(f"{z:>6} | {D_true[z, 0]:>8.4f} | {D_true[z, 1]:>8.4f} | {D_true[z, 2]:>8.4f} | {D_true[z, 3]:>8.4f} | {si4_calc:.4f}")

    # Test different λ_ratio values
    print("\n--- Testing Solver v2 ---")
    results = {}

    for lambda_ratio in [0.0, 0.1, 1.0, 10.0]:
        solver = RatioObjectiveSolverV2(
            tm,
            N_idx=1, O_idx=2, Si4_idx=0, substrate_idx=3,
            lambda_ratio=lambda_ratio,
            degree=1.0,
        )

        result = solver.solve(I_obs, thickness_init=25.0, abruptness=5.0, max_iter=500)
        D_opt = result['D_opt']

        print(f"\nλ_ratio={lambda_ratio:5.1f}: Chi²={result['chi2']:.6f}, Ratio_loss={result['ratio_loss']:.6f}")
        print(f"  z=0:  Si4+={D_opt[0, 0]:.3f}, N={D_opt[0, 1]:.3f}, O={D_opt[0, 2]:.3f}")
        print(f"        (true: Si4+={D_true[0, 0]:.3f}, N={D_true[0, 1]:.3f}, O={D_true[0, 2]:.3f})")
        print(f"  z=10: Si4+={D_opt[10, 0]:.3f}, N={D_opt[10, 1]:.3f}, O={D_opt[10, 2]:.3f}")
        print(f"        (true: Si4+={D_true[10, 0]:.3f}, N={D_true[10, 1]:.3f}, O={D_true[10, 2]:.3f})")

        # Check stoichiometry
        si4_from_stoich = D_opt[0, 1] * (3/4) + D_opt[0, 2] * (1/2)
        print(f"  Stoich check at z=0: Si4+={D_opt[0, 0]:.3f} vs calc={si4_from_stoich:.3f}")

        results[lambda_ratio] = result

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # True profile
    ax = axes[0, 0]
    ax.plot(depth, D_true[:, 0], 'g-', linewidth=2, label='Si4+')
    ax.plot(depth, D_true[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_true[:, 2], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_true[:, 3], 'k--', linewidth=2, label='Si_bulk')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title('TRUE Profile')
    ax.legend()

    # λ=0
    ax = axes[0, 1]
    D_opt = results[0.0]['D_opt']
    ax.plot(depth, D_opt[:, 0], 'g-', linewidth=2, label='Si4+')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_opt[:, 3], 'k--', linewidth=2, label='Si_bulk')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title(f'v2 λ=0: Chi²={results[0.0]["chi2"]:.4f}')
    ax.legend()

    # λ=1.0
    ax = axes[0, 2]
    D_opt = results[1.0]['D_opt']
    ax.plot(depth, D_opt[:, 0], 'g-', linewidth=2, label='Si4+')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_opt[:, 3], 'k--', linewidth=2, label='Si_bulk')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title(f'v2 λ=1.0: Chi²={results[1.0]["chi2"]:.4f}')
    ax.legend()

    # λ=10
    ax = axes[1, 0]
    D_opt = results[10.0]['D_opt']
    ax.plot(depth, D_opt[:, 0], 'g-', linewidth=2, label='Si4+')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_opt[:, 3], 'k--', linewidth=2, label='Si_bulk')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title(f'v2 λ=10: Chi²={results[10.0]["chi2"]:.4f}')
    ax.legend()

    # N comparison
    ax = axes[1, 1]
    ax.plot(depth, D_true[:, 1], 'b--', linewidth=2, alpha=0.7, label='N (true)')
    for lam in [0.0, 1.0, 10.0]:
        ax.plot(depth, results[lam]['D_opt'][:, 1], linewidth=1.5, label=f'N (λ={lam})')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('N Concentration')
    ax.set_title('N Profile Comparison')
    ax.legend()

    # O comparison
    ax = axes[1, 2]
    ax.plot(depth, D_true[:, 2], 'r--', linewidth=2, alpha=0.7, label='O (true)')
    for lam in [0.0, 1.0, 10.0]:
        ax.plot(depth, results[lam]['D_opt'][:, 2], linewidth=1.5, label=f'O (λ={lam})')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('O Concentration')
    ax.set_title('O Profile Comparison')
    ax.legend()

    plt.suptitle('Ratio Objective v2: Si4+ = N×(3/4) + O×(1/2)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/ratio_objective_v2.png', dpi=150)
    print(f"\nSaved: outputs/ratio_objective_v2.png")
    plt.close()

    return results


if __name__ == "__main__":
    results = test_ratio_objective_v2()

    print("\n" + "=" * 70)
    print("Ratio Objective v2 test completed!")
    print("=" * 70)
