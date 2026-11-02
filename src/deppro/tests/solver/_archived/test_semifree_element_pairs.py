"""
SemiFreeSolver with Element-Pair Linking

Prototype for separate charge constraints per compound.

Instead of:
  linked_indices = [N, O]  (all linked elements in one group)

Use:
  linked_pairs = {
      Si_n_idx: (N_idx, ratio_N_to_Si_n),  # N = ratio × Si_n
      Si_o_idx: (O_idx, ratio_O_to_Si_o),  # O = ratio × Si_o
  }

Usage:
    python -m deppro.tests.solver.test_semifree_element_pairs
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
    """
    Prototype solver with element-pair linking.

    Key difference from SemiFreeSolver:
    - Each free element has its OWN linked element with a specific ratio
    - linked_pairs = {free_idx: (linked_idx, ratio)}

    For Si3N4/SiO2/Si:
    - linked_pairs = {0: (1, 4/3), 2: (3, 2.0)}
    - Si_n (0) → N (1) with ratio 4/3 (from Si3N4 stoichiometry)
    - Si_o (2) → O (3) with ratio 2.0 (from SiO2 stoichiometry)
    """

    def __init__(
        self,
        transfer_matrix,
        free_indices,
        linked_pairs,  # {free_idx: (linked_idx, ratio)}
        substrate_indices,
        elements,
        degree=1.0,
    ):
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
        """Build substrate erf profile."""
        sigma = abruptness / (2 * np.sqrt(np.log(2)))
        erf_profile = 0.5 * (1 + erf((self.depth - thickness) / sigma))
        return erf_profile

    def _build_full_profile(self, free_conc, thickness, abruptness):
        """
        Build full profile with element-pair linking.

        free_conc: (n_depth, n_free)
        """
        D = np.zeros((self.n_depth, self.n_elements))

        # 1. Substrate profile
        sub_profile = self._build_substrate_profile(thickness, abruptness)
        for i, sub_e in enumerate(self.sub_idx):
            D[:, sub_e] = sub_profile  # Single substrate element

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
        """Objective function: Chi² (or degree × Chi²)."""
        n_free = len(self.free_idx)
        free_conc = x.reshape(self.n_depth, n_free)

        D = self._build_full_profile(free_conc, thickness, abruptness)
        I_pred = self.tm.forward(D, normalize=True)

        chi2 = np.sum((I_pred - I_obs) ** 2)
        return self.degree * chi2

    def solve(self, I_obs, thickness_init=30.0, abruptness=5.0, max_iter=300):
        """Solve inverse problem."""
        n_free = len(self.free_idx)

        # Initial guess: uniform free concentrations
        x0 = np.ones(self.n_depth * n_free) * 0.2

        # Bounds
        bounds = Bounds(
            lb=np.zeros(self.n_depth * n_free),
            ub=np.ones(self.n_depth * n_free),
        )

        # Optimize with fixed thickness (simplified)
        result = minimize(
            self._objective,
            x0,
            args=(I_obs, thickness_init, abruptness),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter},
        )

        # Extract results
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
        }


def test_element_pair_solver():
    """Test ElementPairSolver on NO pattern."""
    print("=" * 70)
    print("Test: ElementPairSolver with Separate Charge Constraints")
    print("Structure: Si3N4 (5A) / SiO2 (20A) / Si")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]
    true_n_thick = 5.0
    true_o_thick = 20.0

    elements = [
        ElementConfig('Si', '2p', 'Si3N4', oxidation_state=4),  # Si_n (0)
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),  # N (1)
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),   # Si_o (2)
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),   # O (3)
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),     # Si_bulk (4)
    ]

    D_true = create_no_pattern(depth, true_n_thick, true_o_thick)

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)
    I_true = tm.forward(D_true, normalize=True)

    np.random.seed(42)
    noise = 0.02 * np.random.randn(*I_true.shape)
    I_obs = I_true * (1 + noise)
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    print(f"\nTrue profile at z=0:")
    print(f"  Si_n: {D_true[0, 0]:.4f}")
    print(f"  N:    {D_true[0, 1]:.4f}")
    print(f"  Si_o: {D_true[0, 2]:.4f}")
    print(f"  O:    {D_true[0, 3]:.4f}")
    print(f"  N/Si_n ratio: {D_true[0, 1] / max(D_true[0, 0], 1e-6):.2f} (expected: 4/3 = 1.33)")

    # Element-pair linking
    # Si_n (0) → N (1) with ratio 4/3 (from Si3N4: Si:N = 3:4)
    # Si_o (2) → O (3) with ratio 2.0 (from SiO2: Si:O = 1:2)
    linked_pairs = {
        0: (1, 4/3),  # Si_n → N
        2: (3, 2.0),  # Si_o → O
    }

    print(f"\nLinked pairs:")
    print(f"  Si_n (0) → N (1) with ratio 4/3")
    print(f"  Si_o (2) → O (3) with ratio 2.0")

    # Test with different degree values
    for degree in [1.0, 5.0, 10.0]:
        solver = ElementPairSolver(
            tm,
            free_indices=[0, 2],
            linked_pairs=linked_pairs,
            substrate_indices=[4],
            elements=elements,
            degree=degree,
        )

        result = solver.solve(I_obs, thickness_init=25.0, abruptness=5.0, max_iter=500)

        D_opt = result['D_opt']
        print(f"\ndegree={degree}:")
        print(f"  Chi²: {result['chi2']:.6f}")
        print(f"  D at z=0: Si_n={D_opt[0, 0]:.3f}, N={D_opt[0, 1]:.3f}, Si_o={D_opt[0, 2]:.3f}, O={D_opt[0, 3]:.3f}")
        print(f"  N/Si_n ratio: {D_opt[0, 1] / max(D_opt[0, 0], 1e-6):.2f}")
        print(f"  O/Si_o ratio: {D_opt[0, 2] and D_opt[0, 3] / max(D_opt[0, 2], 1e-6):.2f}")

        # Check if N and O are different now
        if abs(D_opt[0, 1] - D_opt[0, 3]) > 0.1:
            print(f"  --> N and O are now different! (good)")
        else:
            print(f"  --> N and O still similar")

    # Plot best result
    solver = ElementPairSolver(
        tm, free_indices=[0, 2], linked_pairs=linked_pairs,
        substrate_indices=[4], elements=elements, degree=5.0
    )
    result = solver.solve(I_obs, thickness_init=25.0, abruptness=5.0, max_iter=500)
    D_opt = result['D_opt']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(depth, D_true[:, 1], 'b-', linewidth=2, label='N (true)')
    ax.plot(depth, D_true[:, 3], 'r-', linewidth=2, label='O (true)')
    ax.plot(depth, D_true[:, 4], 'k--', linewidth=1, label='Si_bulk (true)')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('TRUE Profile')
    ax.legend()

    ax = axes[1]
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N (opt)')
    ax.plot(depth, D_opt[:, 3], 'r-', linewidth=2, label='O (opt)')
    ax.plot(depth, D_opt[:, 4], 'k--', linewidth=1, label='Si_bulk (opt)')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title(f'ElementPairSolver (degree=5, Chi²={result["chi2"]:.4f})')
    ax.legend()

    plt.suptitle('Element-Pair Linking: N/O now independent', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/element_pair_solver_no.png', dpi=150)
    print(f"\nSaved: outputs/element_pair_solver_no.png")
    plt.close()


if __name__ == "__main__":
    test_element_pair_solver()

    print("\n" + "=" * 70)
    print("Element-Pair Solver test completed!")
    print("=" * 70)
