"""
ElementPairSolver - Merged Si4+ Case

What happens when Si_n and Si_o cannot be chemically distinguished?
(i.e., Si4+ appears as a single peak in XPS)

Case 1: Si_n, N, Si_o, O, Si_bulk (5 elements) - distinguishable
Case 2: Si4+, N, O, Si_bulk (4 elements) - Si_n and Si_o merged

Usage:
    python -m deppro.tests.solver.test_semifree_merged_si4
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig


def create_no_pattern_5elem(depth, n_thickness, o_thickness, interface_width=5.0):
    """Create NO pattern with 5 elements (Si_n and Si_o distinguishable)."""
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


def create_no_pattern_4elem(depth, n_thickness, o_thickness, interface_width=5.0):
    """
    Create NO pattern with 4 elements (Si4+ merged).

    Elements: [Si4+, N, O, Si_bulk] = [0, 1, 2, 3]
    Si4+ = Si_n + Si_o (cannot be distinguished)
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


def test_merged_si4():
    """Test when Si4+ cannot be chemically distinguished."""
    print("=" * 70)
    print("Test: Merged Si4+ (Si_n and Si_o indistinguishable)")
    print("Structure: Si3N4 (5A) / SiO2 (20A) / Si")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]
    true_n_thick = 5.0
    true_o_thick = 20.0

    # 4-element case: Si4+, N, O, Si_bulk
    elements_4 = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),   # Si4+ (merged)
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),  # N
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),   # O
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),     # Si_bulk
    ]

    D_true_4 = create_no_pattern_4elem(depth, true_n_thick, true_o_thick)

    tm_4 = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm_4.compute(D_true_4, elements_4)
    I_true_4 = tm_4.forward(D_true_4, normalize=True)

    np.random.seed(42)
    I_obs_4 = I_true_4 * (1 + 0.02 * np.random.randn(*I_true_4.shape))
    I_obs_4 = np.clip(I_obs_4, 1e-6, None)
    I_obs_4 = I_obs_4 / I_obs_4.sum(axis=1, keepdims=True)

    print("\n--- TRUE Profile (4 elements: Si4+ merged) ---")
    print(f"{'Depth':>6} | {'Si4+':>8} | {'N':>8} | {'O':>8} | {'Si_bulk':>8}")
    print("-" * 50)
    for z in [0, 5, 10, 15, 20, 25]:
        print(f"{z:>6} | {D_true_4[z, 0]:>8.4f} | {D_true_4[z, 1]:>8.4f} | {D_true_4[z, 2]:>8.4f} | {D_true_4[z, 3]:>8.4f}")

    # Problem: How to link Si4+ to BOTH N and O?
    # Option 1: Si4+ -> N only (ignores O constraint)
    # Option 2: Si4+ -> O only (ignores N constraint)
    # Option 3: Cannot use element-pair linking properly

    print("\n" + "=" * 70)
    print("PROBLEM: Si4+ is linked to BOTH N and O with DIFFERENT ratios!")
    print("  - In Si3N4: Si:N = 3:4 → N = (4/3) × Si")
    print("  - In SiO2:  Si:O = 1:2 → O = 2 × Si")
    print("  - But we only have ONE Si4+ signal!")
    print("=" * 70)

    # Try different linking strategies
    strategies = [
        ("Si4+ → N only (ratio 4/3)", {0: (1, 4/3)}),
        ("Si4+ → O only (ratio 2.0)", {0: (2, 2.0)}),
        ("Si4+ → N (ratio 1.0, wrong)", {0: (1, 1.0)}),
    ]

    results = {}
    for name, linked_pairs in strategies:
        print(f"\n--- Strategy: {name} ---")

        solver = ElementPairSolver(
            tm_4, free_indices=[0], linked_pairs=linked_pairs,
            substrate_indices=[3], elements=elements_4, degree=5.0
        )

        result = solver.solve(I_obs_4, thickness_init=25.0, abruptness=5.0, max_iter=500)
        D_opt = result['D_opt']

        print(f"  Chi² = {result['chi2']:.6f}")
        print(f"  At z=0: Si4+={D_opt[0, 0]:.3f}, N={D_opt[0, 1]:.3f}, O={D_opt[0, 2]:.3f}")
        print(f"  At z=10: Si4+={D_opt[10, 0]:.3f}, N={D_opt[10, 1]:.3f}, O={D_opt[10, 2]:.3f}")

        results[name] = {'result': result, 'D_opt': D_opt}

    # Alternative: Use TWO free parameters that share Si4+ signal
    print("\n" + "=" * 70)
    print("Alternative: Split Si4+ into virtual Si_n and Si_o")
    print("  - Constraint: Si_n + Si_o = observed Si4+ intensity")
    print("  - This requires a different solver architecture")
    print("=" * 70)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # True profile
    ax = axes[0, 0]
    ax.plot(depth, D_true_4[:, 0], 'g-', linewidth=2, label='Si4+ (merged)')
    ax.plot(depth, D_true_4[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_true_4[:, 2], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_true_4[:, 3], 'k--', linewidth=2, label='Si_bulk')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('TRUE Profile (Si4+ merged)')
    ax.legend()

    # Strategy 1: Si4+ -> N
    ax = axes[0, 1]
    D_opt = results["Si4+ → N only (ratio 4/3)"]['D_opt']
    ax.plot(depth, D_opt[:, 0], 'g-', linewidth=2, label='Si4+')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_opt[:, 3], 'k--', linewidth=2, label='Si_bulk')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    chi2 = results["Si4+ → N only (ratio 4/3)"]['result']['chi2']
    ax.set_title(f'Si4+ → N only (Chi²={chi2:.4f})')
    ax.legend()

    # Strategy 2: Si4+ -> O
    ax = axes[1, 0]
    D_opt = results["Si4+ → O only (ratio 2.0)"]['D_opt']
    ax.plot(depth, D_opt[:, 0], 'g-', linewidth=2, label='Si4+')
    ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N')
    ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_opt[:, 3], 'k--', linewidth=2, label='Si_bulk')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    chi2 = results["Si4+ → O only (ratio 2.0)"]['result']['chi2']
    ax.set_title(f'Si4+ → O only (Chi²={chi2:.4f})')
    ax.legend()

    # N/O comparison
    ax = axes[1, 1]
    ax.plot(depth, D_true_4[:, 1], 'b--', linewidth=1.5, alpha=0.7, label='N (true)')
    ax.plot(depth, D_true_4[:, 2], 'r--', linewidth=1.5, alpha=0.7, label='O (true)')

    D_opt_n = results["Si4+ → N only (ratio 4/3)"]['D_opt']
    ax.plot(depth, D_opt_n[:, 1], 'b-', linewidth=2, label='N (Si4+→N)')

    D_opt_o = results["Si4+ → O only (ratio 2.0)"]['D_opt']
    ax.plot(depth, D_opt_o[:, 2], 'r-', linewidth=2, label='O (Si4+→O)')

    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('N and O: True vs Different Strategies')
    ax.legend()

    plt.suptitle('Merged Si4+ Problem: Cannot link to both N and O', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/merged_si4_problem.png', dpi=150)
    print(f"\nSaved: outputs/merged_si4_problem.png")
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Merged Si4+ Case")
    print("=" * 70)
    print("""
When Si4+ (Si_n + Si_o) cannot be chemically distinguished:

1. Element-pair linking FAILS because:
   - One Si4+ signal must determine BOTH N and O
   - But N/Si and O/Si ratios are DIFFERENT (4/3 vs 2.0)
   - Cannot satisfy both constraints simultaneously

2. This is similar to the NON problem:
   - Same element at different depths with different chemistry
   - ARXPS fundamentally cannot separate them

3. Possible solutions:
   a) Use I(N)/I(O) ratio objective (user suggestion)
      - Optimize N/O ratio directly from observed intensities
      - Does not require separating Si4+ into Si_n and Si_o

   b) Virtual splitting with constraint:
      - Introduce virtual Si_n and Si_o parameters
      - Constraint: I(Si_n) + I(Si_o) = I(Si4+_observed)
      - More complex implementation

   c) GenericLayerSolver (if structure is known):
      - Define layers explicitly with stoichiometry
      - Works if layer order is known
""")

    return results


if __name__ == "__main__":
    test_merged_si4()

    print("\n" + "=" * 70)
    print("Merged Si4+ test completed!")
    print("=" * 70)
