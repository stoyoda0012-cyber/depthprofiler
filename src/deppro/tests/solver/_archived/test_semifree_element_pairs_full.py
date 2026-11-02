"""
ElementPairSolver - Full Element Display

Show all 5 elements including Si4+ (Si_n, Si_o) to verify they are being optimized.

Usage:
    python -m deppro.tests.solver.test_semifree_element_pairs_full
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig


def create_no_pattern(depth, n_thickness, o_thickness, interface_width=5.0):
    """Create NO pattern: Si3N4 / SiO2 / Si"""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = n_thickness
    z2 = n_thickness + o_thickness

    D = np.zeros((len(depth), 5))

    # Si3N4 layer (surface)
    sin_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    D[:, 0] = (3/7) * sin_profile  # Si_n
    D[:, 1] = (4/7) * sin_profile  # N

    # SiO2 layer
    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    D[:, 2] = (1/3) * sio2_profile  # Si_o
    D[:, 3] = (2/3) * sio2_profile  # O

    # Si substrate
    D[:, 4] = 0.5 * (1 + erf((depth - z2) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


class ElementPairSolver:
    """Prototype solver with element-pair linking."""

    def __init__(self, transfer_matrix, free_indices, linked_pairs,
                 substrate_indices, elements, degree=1.0):
        self.tm = transfer_matrix
        self.depth = transfer_matrix.depth
        self.n_depth = len(self.depth)
        self.n_elements = transfer_matrix.n_elements

        self.free_idx = free_indices
        self.linked_pairs = linked_pairs
        self.sub_idx = substrate_indices
        self.elements = elements
        self.degree = degree

    def _build_substrate_profile(self, thickness, abruptness):
        sigma = abruptness / (2 * np.sqrt(np.log(2)))
        erf_profile = 0.5 * (1 + erf((self.depth - thickness) / sigma))
        return erf_profile

    def _build_full_profile(self, free_conc, thickness, abruptness):
        D = np.zeros((self.n_depth, self.n_elements))

        # 1. Substrate profile
        sub_profile = self._build_substrate_profile(thickness, abruptness)
        for sub_e in self.sub_idx:
            D[:, sub_e] = sub_profile

        # 2. Free element concentrations (scaled by film region)
        film_scale = np.maximum(1.0 - sub_profile, 1e-6)

        for i, free_e in enumerate(self.free_idx):
            D[:, free_e] = free_conc[:, i] * film_scale

            # 3. Linked element from this free element
            if free_e in self.linked_pairs:
                linked_e, ratio = self.linked_pairs[free_e]
                D[:, linked_e] = D[:, free_e] * ratio

        # Normalize
        D = np.clip(D, 1e-6, 1.0)
        D_sum = D.sum(axis=1, keepdims=True)
        D = D / np.maximum(D_sum, 1e-6)

        return D

    def _objective(self, x, I_obs, thickness, abruptness):
        n_free = len(self.free_idx)
        free_conc = x.reshape(self.n_depth, n_free)

        D = self._build_full_profile(free_conc, thickness, abruptness)
        I_pred = self.tm.forward(D, normalize=True)

        chi2 = np.sum((I_pred - I_obs) ** 2)
        return self.degree * chi2

    def solve(self, I_obs, thickness_init=30.0, abruptness=5.0, max_iter=300):
        n_free = len(self.free_idx)
        x0 = np.ones(self.n_depth * n_free) * 0.2

        bounds = Bounds(
            lb=np.zeros(self.n_depth * n_free),
            ub=np.ones(self.n_depth * n_free),
        )

        result = minimize(
            self._objective,
            x0,
            args=(I_obs, thickness_init, abruptness),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter},
        )

        free_conc = result.x.reshape(self.n_depth, n_free)
        D_opt = self._build_full_profile(free_conc, thickness_init, abruptness)
        I_opt = self.tm.forward(D_opt, normalize=True)
        chi2 = np.sum((I_opt - I_obs) ** 2)

        return {
            'D_opt': D_opt,
            'I_opt': I_opt,
            'chi2': chi2,
            'thickness': thickness_init,
            'success': result.success,
            'free_conc': free_conc,
        }


def test_full_elements():
    """Test with all 5 elements displayed."""
    print("=" * 70)
    print("ElementPairSolver - Full Element Analysis")
    print("Structure: Si3N4 (5A) / SiO2 (20A) / Si")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]
    true_n_thick = 5.0
    true_o_thick = 20.0

    elements = [
        ElementConfig('Si', '2p', 'Si3N4', oxidation_state=4),  # Si_n (0) - Si4+ in nitride
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),  # N (1)
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),   # Si_o (2) - Si4+ in oxide
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),   # O (3)
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),     # Si_bulk (4) - Si0
    ]

    D_true = create_no_pattern(depth, true_n_thick, true_o_thick)

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)
    I_true = tm.forward(D_true, normalize=True)

    np.random.seed(42)
    I_obs = I_true * (1 + 0.02 * np.random.randn(*I_true.shape))
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    # Show true profile
    print("\n--- TRUE Profile ---")
    print(f"{'Depth':>6} | {'Si_n':>8} | {'N':>8} | {'Si_o':>8} | {'O':>8} | {'Si_bulk':>8}")
    print("-" * 60)
    for z in [0, 5, 10, 15, 20, 25, 30]:
        print(f"{z:>6} | {D_true[z, 0]:>8.4f} | {D_true[z, 1]:>8.4f} | {D_true[z, 2]:>8.4f} | {D_true[z, 3]:>8.4f} | {D_true[z, 4]:>8.4f}")

    # Element-pair linking
    linked_pairs = {
        0: (1, 4/3),  # Si_n → N (Si3N4: Si:N = 3:4)
        2: (3, 2.0),  # Si_o → O (SiO2: Si:O = 1:2)
    }

    solver = ElementPairSolver(
        tm, free_indices=[0, 2], linked_pairs=linked_pairs,
        substrate_indices=[4], elements=elements, degree=5.0
    )

    result = solver.solve(I_obs, thickness_init=25.0, abruptness=5.0, max_iter=500)
    D_opt = result['D_opt']

    print("\n--- OPTIMIZED Profile ---")
    print(f"Chi² = {result['chi2']:.6f}")
    print(f"{'Depth':>6} | {'Si_n':>8} | {'N':>8} | {'Si_o':>8} | {'O':>8} | {'Si_bulk':>8}")
    print("-" * 60)
    for z in [0, 5, 10, 15, 20, 25, 30]:
        print(f"{z:>6} | {D_opt[z, 0]:>8.4f} | {D_opt[z, 1]:>8.4f} | {D_opt[z, 2]:>8.4f} | {D_opt[z, 3]:>8.4f} | {D_opt[z, 4]:>8.4f}")

    # Check stoichiometry
    print("\n--- Stoichiometry Check ---")
    print(f"At z=0 (surface):")
    print(f"  N/Si_n = {D_opt[0, 1] / max(D_opt[0, 0], 1e-6):.2f} (expected: 1.33)")
    print(f"  O/Si_o = {D_opt[0, 3] / max(D_opt[0, 2], 1e-6):.2f} (expected: 2.00)")

    print(f"\nAt z=10 (in SiO2 region):")
    print(f"  N/Si_n = {D_opt[10, 1] / max(D_opt[10, 0], 1e-6):.2f}")
    print(f"  O/Si_o = {D_opt[10, 3] / max(D_opt[10, 2], 1e-6):.2f} (expected: 2.00)")

    # Plot all 5 elements
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # True profile - all elements
    ax = axes[0, 0]
    ax.plot(depth, D_true[:, 0], 'c-', linewidth=2, label='Si_n (Si4+ in Si3N4)')
    ax.plot(depth, D_true[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_true[:, 2], 'm-', linewidth=2, label='Si_o (Si4+ in SiO2)')
    ax.plot(depth, D_true[:, 3], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_true[:, 4], 'k--', linewidth=2, label='Si_bulk (Si0)')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('TRUE Profile - All 5 Elements')
    ax.legend(loc='upper right')

    # Optimized profile - all elements
    ax = axes[0, 1]
    ax.plot(depth, D_opt[:, 0], 'c-', linewidth=2, label='Si_n (opt)')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N (opt)')
    ax.plot(depth, D_opt[:, 2], 'm-', linewidth=2, label='Si_o (opt)')
    ax.plot(depth, D_opt[:, 3], 'r-', linewidth=2, label='O (opt)')
    ax.plot(depth, D_opt[:, 4], 'k--', linewidth=2, label='Si_bulk (opt)')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title(f'OPTIMIZED Profile (Chi²={result["chi2"]:.4f})')
    ax.legend(loc='upper right')

    # Comparison: N and O only
    ax = axes[1, 0]
    ax.plot(depth, D_true[:, 1], 'b--', linewidth=1.5, alpha=0.7, label='N (true)')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N (opt)')
    ax.plot(depth, D_true[:, 3], 'r--', linewidth=1.5, alpha=0.7, label='O (true)')
    ax.plot(depth, D_opt[:, 3], 'r-', linewidth=2, label='O (opt)')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('N and O Comparison')
    ax.legend(loc='upper right')

    # Comparison: Si4+ (Si_n + Si_o) only
    ax = axes[1, 1]
    ax.plot(depth, D_true[:, 0], 'c--', linewidth=1.5, alpha=0.7, label='Si_n (true)')
    ax.plot(depth, D_opt[:, 0], 'c-', linewidth=2, label='Si_n (opt)')
    ax.plot(depth, D_true[:, 2], 'm--', linewidth=1.5, alpha=0.7, label='Si_o (true)')
    ax.plot(depth, D_opt[:, 2], 'm-', linewidth=2, label='Si_o (opt)')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('Si4+ Comparison (Si_n in Si3N4, Si_o in SiO2)')
    ax.legend(loc='upper right')

    plt.suptitle('ElementPairSolver: All Elements Including Si4+', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/element_pair_solver_full.png', dpi=150)
    print(f"\nSaved: outputs/element_pair_solver_full.png")
    plt.close()

    # Verify Si4+ is being optimized
    print("\n--- Verification: Si4+ Optimization ---")
    free_conc = result['free_conc']
    print(f"Free concentration shape: {free_conc.shape}")
    print(f"  Column 0 = Si_n (optimized)")
    print(f"  Column 1 = Si_o (optimized)")
    print(f"\nSi_n at z=0: {free_conc[0, 0]:.4f} -> D[0, Si_n]={D_opt[0, 0]:.4f}")
    print(f"Si_o at z=0: {free_conc[0, 1]:.4f} -> D[0, Si_o]={D_opt[0, 2]:.4f}")
    print(f"\nSi_n at z=10: {free_conc[10, 0]:.4f} -> D[10, Si_n]={D_opt[10, 0]:.4f}")
    print(f"Si_o at z=10: {free_conc[10, 1]:.4f} -> D[10, Si_o]={D_opt[10, 2]:.4f}")

    return result


if __name__ == "__main__":
    test_full_elements()

    print("\n" + "=" * 70)
    print("Full element analysis completed!")
    print("=" * 70)
