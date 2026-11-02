"""
I(N)/I(O) Ratio Objective Solver

When Si4+ cannot be chemically separated into Si_n and Si_o,
use the N/O intensity ratio to constrain the optimization.

Key idea:
- Standard objective: L = ||I_pred - I_obs||²
- Ratio objective: L_ratio = Σ_θ (I_pred[N]/I_pred[O] - I_obs[N]/I_obs[O])²
- Combined: L_total = L + λ × L_ratio

This allows N and O to be optimized based on their relative intensities
without requiring Si4+ to be split.

Usage:
    python -m deppro.tests.solver.test_ratio_objective
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


class RatioObjectiveSolver:
    """
    Solver using I(N)/I(O) ratio objective.

    Optimizes N and O profiles directly based on:
    1. Standard Chi² fit to all intensities
    2. N/O ratio matching at each angle

    Parameters:
    - N_idx, O_idx: indices for N and O elements
    - lambda_ratio: weight for ratio objective (default 1.0)
    """

    def __init__(
        self,
        transfer_matrix,
        N_idx,
        O_idx,
        substrate_idx,
        Si4_idx=None,  # Optional: if provided, Si4+ is also optimized
        lambda_ratio=1.0,
        degree=1.0,
    ):
        self.tm = transfer_matrix
        self.depth = transfer_matrix.depth
        self.n_depth = len(self.depth)
        self.n_elements = transfer_matrix.n_elements

        self.N_idx = N_idx
        self.O_idx = O_idx
        self.substrate_idx = substrate_idx
        self.Si4_idx = Si4_idx
        self.lambda_ratio = lambda_ratio
        self.degree = degree

    def _build_substrate_profile(self, thickness, abruptness):
        sigma = abruptness / (2 * np.sqrt(np.log(2)))
        erf_profile = 0.5 * (1 + erf((self.depth - thickness) / sigma))
        return erf_profile

    def _build_profile(self, N_conc, O_conc, thickness, abruptness):
        """
        Build depth profile from N and O concentrations.

        N_conc, O_conc: (n_depth,) - concentration at each depth
        Si4+ is computed as: 1 - N - O - Si_bulk (to satisfy sum=1)
        """
        D = np.zeros((self.n_depth, self.n_elements))

        # Substrate profile
        sub_profile = self._build_substrate_profile(thickness, abruptness)
        D[:, self.substrate_idx] = sub_profile

        # Film region scale
        film_scale = np.maximum(1.0 - sub_profile, 1e-6)

        # N and O concentrations (scaled to film region)
        D[:, self.N_idx] = N_conc * film_scale
        D[:, self.O_idx] = O_conc * film_scale

        # Si4+ fills the rest (if provided)
        if self.Si4_idx is not None:
            D[:, self.Si4_idx] = np.maximum(
                film_scale - D[:, self.N_idx] - D[:, self.O_idx],
                1e-6
            )

        # Normalize
        D = np.clip(D, 1e-6, 1.0)
        D_sum = D.sum(axis=1, keepdims=True)
        D = D / np.maximum(D_sum, 1e-6)

        return D

    def _objective(self, x, I_obs, thickness, abruptness):
        """
        Combined objective: Chi² + λ × ratio_loss
        """
        # Unpack parameters
        N_conc = x[:self.n_depth]
        O_conc = x[self.n_depth:2*self.n_depth]

        # Build profile and predict intensities
        D = self._build_profile(N_conc, O_conc, thickness, abruptness)
        I_pred = self.tm.forward(D, normalize=True)

        # Standard Chi² loss
        chi2 = np.sum((I_pred - I_obs) ** 2)

        # Ratio loss: (I_pred[N]/I_pred[O] - I_obs[N]/I_obs[O])² at each angle
        # Avoid division by zero
        eps = 1e-8
        ratio_pred = I_pred[:, self.N_idx] / np.maximum(I_pred[:, self.O_idx], eps)
        ratio_obs = I_obs[:, self.N_idx] / np.maximum(I_obs[:, self.O_idx], eps)
        ratio_loss = np.sum((ratio_pred - ratio_obs) ** 2)

        # Combined objective
        total = self.degree * (chi2 + self.lambda_ratio * ratio_loss)

        return total

    def solve(self, I_obs, thickness_init=25.0, abruptness=5.0, max_iter=500):
        """Solve with ratio objective."""
        # Initial guess: uniform N and O
        x0 = np.concatenate([
            np.ones(self.n_depth) * 0.3,  # N_conc
            np.ones(self.n_depth) * 0.3,  # O_conc
        ])

        # Bounds: concentrations between 0 and 1
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

        # Extract results
        N_conc = result.x[:self.n_depth]
        O_conc = result.x[self.n_depth:2*self.n_depth]
        D_opt = self._build_profile(N_conc, O_conc, thickness_init, abruptness)
        I_opt = self.tm.forward(D_opt, normalize=True)
        chi2 = np.sum((I_opt - I_obs) ** 2)

        # Compute final ratio loss
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


def test_ratio_objective():
    """Test ratio objective solver on merged Si4+ case."""
    print("=" * 70)
    print("Test: I(N)/I(O) Ratio Objective Solver")
    print("Structure: Si3N4 (5A) / SiO2 (20A) / Si")
    print("Si4+ is MERGED (cannot distinguish Si_n and Si_o)")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]
    true_n_thick = 5.0
    true_o_thick = 20.0

    # 4-element case: Si4+, N, O, Si_bulk
    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),   # Si4+ (merged)
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
    print(f"{'Depth':>6} | {'Si4+':>8} | {'N':>8} | {'O':>8} | {'Si_bulk':>8}")
    print("-" * 50)
    for z in [0, 5, 10, 15, 20, 25]:
        print(f"{z:>6} | {D_true[z, 0]:>8.4f} | {D_true[z, 1]:>8.4f} | {D_true[z, 2]:>8.4f} | {D_true[z, 3]:>8.4f}")

    # Observed N/O ratios at each angle
    print("\n--- Observed I(N)/I(O) ratios ---")
    for i, angle in enumerate(angles):
        ratio = I_obs[i, 1] / I_obs[i, 2]
        print(f"  θ={angle:2d}°: I(N)/I(O) = {ratio:.3f}")

    # Test different lambda_ratio values
    print("\n--- Testing different λ_ratio values ---")
    results = {}

    for lambda_ratio in [0.0, 0.1, 1.0, 10.0, 100.0]:
        solver = RatioObjectiveSolver(
            tm,
            N_idx=1, O_idx=2, substrate_idx=3, Si4_idx=0,
            lambda_ratio=lambda_ratio,
            degree=1.0,
        )

        result = solver.solve(I_obs, thickness_init=25.0, abruptness=5.0, max_iter=500)
        D_opt = result['D_opt']

        print(f"\nλ_ratio={lambda_ratio:5.1f}: Chi²={result['chi2']:.6f}, Ratio_loss={result['ratio_loss']:.6f}")
        print(f"  At z=0: N={D_opt[0, 1]:.3f}, O={D_opt[0, 2]:.3f} (true: N={D_true[0, 1]:.3f}, O={D_true[0, 2]:.3f})")
        print(f"  At z=10: N={D_opt[10, 1]:.3f}, O={D_opt[10, 2]:.3f} (true: N={D_true[10, 1]:.3f}, O={D_true[10, 2]:.3f})")

        results[lambda_ratio] = result

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # True profile
    ax = axes[0, 0]
    ax.plot(depth, D_true[:, 0], 'g-', linewidth=2, label='Si4+ (merged)')
    ax.plot(depth, D_true[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_true[:, 2], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_true[:, 3], 'k--', linewidth=2, label='Si_bulk')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title('TRUE Profile')
    ax.legend()

    # λ=0 (no ratio constraint)
    ax = axes[0, 1]
    D_opt = results[0.0]['D_opt']
    ax.plot(depth, D_opt[:, 0], 'g-', linewidth=2, label='Si4+')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_opt[:, 3], 'k--', linewidth=2, label='Si_bulk')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title(f'λ=0 (Chi² only): Chi²={results[0.0]["chi2"]:.4f}')
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
    ax.set_title(f'λ=1.0: Chi²={results[1.0]["chi2"]:.4f}')
    ax.legend()

    # λ=10.0
    ax = axes[1, 0]
    D_opt = results[10.0]['D_opt']
    ax.plot(depth, D_opt[:, 0], 'g-', linewidth=2, label='Si4+')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_opt[:, 3], 'k--', linewidth=2, label='Si_bulk')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title(f'λ=10: Chi²={results[10.0]["chi2"]:.4f}')
    ax.legend()

    # λ=100.0
    ax = axes[1, 1]
    D_opt = results[100.0]['D_opt']
    ax.plot(depth, D_opt[:, 0], 'g-', linewidth=2, label='Si4+')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_opt[:, 3], 'k--', linewidth=2, label='Si_bulk')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title(f'λ=100: Chi²={results[100.0]["chi2"]:.4f}')
    ax.legend()

    # N and O comparison (best vs true)
    ax = axes[1, 2]
    ax.plot(depth, D_true[:, 1], 'b--', linewidth=1.5, alpha=0.7, label='N (true)')
    ax.plot(depth, D_true[:, 2], 'r--', linewidth=1.5, alpha=0.7, label='O (true)')

    # Find best λ (lowest total error)
    best_lambda = min(results.keys(), key=lambda k: results[k]['chi2'] + results[k]['ratio_loss'])
    D_best = results[best_lambda]['D_opt']
    ax.plot(depth, D_best[:, 1], 'b-', linewidth=2, label=f'N (λ={best_lambda})')
    ax.plot(depth, D_best[:, 2], 'r-', linewidth=2, label=f'O (λ={best_lambda})')

    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title(f'Best result: λ={best_lambda}')
    ax.legend()

    plt.suptitle('I(N)/I(O) Ratio Objective: Merged Si4+ Case', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/ratio_objective_solver.png', dpi=150)
    print(f"\nSaved: outputs/ratio_objective_solver.png")
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best λ_ratio: {best_lambda}")
    print(f"Chi²: {results[best_lambda]['chi2']:.6f}")
    print(f"Ratio loss: {results[best_lambda]['ratio_loss']:.6f}")

    return results


if __name__ == "__main__":
    results = test_ratio_objective()

    print("\n" + "=" * 70)
    print("Ratio Objective test completed!")
    print("=" * 70)
