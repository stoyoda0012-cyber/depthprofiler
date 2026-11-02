"""
SemiFreeSolver Diagnosis

Investigate why N and O are not moving independently.

Key issues identified:
1. linked elements are distributed equally (line 1538-1540)
2. Multiple linked elements with different stoichiometries not handled
3. No separate charge balance for Si3N4 and SiO2 regions

Usage:
    python -m deppro.tests.solver.test_semifree_diagnosis
"""

import numpy as np
from scipy.special import erf

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import (
    SemiFreeSolver, SemiFreeSolverConfig, SolverConfig
)


def create_simple_no_pattern(depth, n_thickness, o_thickness, interface_width=5.0):
    """Create simple NO pattern for diagnosis."""
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


def diagnose_charge_coefficients():
    """Diagnose the charge balance coefficients."""
    print("=" * 70)
    print("Diagnosis 1: Charge Balance Coefficients")
    print("=" * 70)

    elements = [
        ElementConfig('Si', '2p', 'Si3N4', oxidation_state=4),  # Si_n (0)
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),  # N (1)
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),   # Si_o (2)
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),   # O (3)
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),     # Si_bulk (4)
    ]

    print("\nElement charges:")
    for i, e in enumerate(elements):
        print(f"  [{i}] {e.symbol} {e.orbital} ({e.compound}): {e.oxidation_state}")

    # Current SemiFreeSolver setup
    free_indices = [0, 2]  # Si_n, Si_o
    linked_indices = [1, 3]  # N, O
    substrate_indices = [4]

    charges = np.array([e.oxidation_state for e in elements])

    print(f"\nCurrent setup:")
    print(f"  free_indices = {free_indices} (Si_n, Si_o)")
    print(f"  linked_indices = {linked_indices} (N, O)")
    print(f"  substrate_indices = {substrate_indices}")

    # Calculate freelinkedconst (current implementation)
    linked_charges = charges[linked_indices]
    mean_linked_charge = np.mean(linked_charges)
    print(f"\n  mean_linked_charge = mean({linked_charges}) = {mean_linked_charge}")

    freelinkedconst = []
    for free_e in free_indices:
        flc = 1 - charges[free_e] / mean_linked_charge
        freelinkedconst.append(flc)
        print(f"  freelinkedconst[{free_e}] = 1 - {charges[free_e]} / {mean_linked_charge} = {flc}")

    print(f"\n  Problem: D_linked = sum(D_free × (flc - 1))")
    print(f"  D_linked is then split equally among ALL linked elements (N and O)")
    print(f"  This forces N and O to have THE SAME profile!")

    print("\n" + "-" * 70)
    print("Expected behavior:")
    print("  Si3N4: Si_n : N = 3 : 4 (stoichiometry)")
    print("  SiO2:  Si_o : O = 1 : 2 (stoichiometry)")
    print("  Each pair should maintain its own charge balance")


def diagnose_profile_building():
    """Diagnose how the profile is built."""
    print("\n" + "=" * 70)
    print("Diagnosis 2: Profile Building")
    print("=" * 70)

    depth = np.arange(0, 50, 1.0)
    angles = [0, 30, 60]

    elements = [
        ElementConfig('Si', '2p', 'Si3N4', oxidation_state=4),
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_simple_no_pattern(depth, 5.0, 20.0)

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    semi_config = SemiFreeSolverConfig(
        free_indices=[0, 2],
        linked_indices=[1, 3],
        substrate_indices=[4],
        thickness_init=25.0,
        abruptness_init=5.0,
    )
    config = SolverConfig(method='LLS', alpha=0.0, max_iter=1)  # Just 1 iteration

    solver = SemiFreeSolver(tm, config, semi_config, elements=elements)

    print(f"\nCharge coefficients computed by solver:")
    print(f"  free_linked_const = {solver.free_linked_const}")

    # Check initial profile
    print(f"\nInitial D_prior shape: {solver.D_prior.shape}")
    print(f"D_prior at z=0:")
    for i, e in enumerate(elements):
        print(f"  [{i}] {e.symbol}: {solver.D_prior[0, i]:.4f}")

    # Try building a profile manually
    n_depth = len(depth)
    n_free = len(solver.free_idx)

    # Initial free concentrations (uniform)
    free_conc = np.ones((n_depth, n_free)) * 0.3

    # Build profile
    D = solver._build_full_profile(free_conc, 25.0, 5.0, np.array([1.0]))

    print(f"\nBuilt profile at z=0 (thickness=25A):")
    for i, e in enumerate(elements):
        print(f"  [{i}] {e.symbol}: {D[0, i]:.4f}")

    print(f"\nBuilt profile at z=10A:")
    for i, e in enumerate(elements):
        print(f"  [{i}] {e.symbol}: {D[10, i]:.4f}")

    # Key check: N and O should be different in NO pattern
    print(f"\nKey observation:")
    print(f"  D[0, N] = {D[0, 1]:.4f}")
    print(f"  D[0, O] = {D[0, 3]:.4f}")
    if abs(D[0, 1] - D[0, 3]) < 0.01:
        print(f"  --> N and O are almost identical! This is the problem.")
    else:
        print(f"  --> N and O differ (good)")


def suggest_alternative_approaches():
    """Suggest alternative approaches for NO/ON/NON patterns."""
    print("\n" + "=" * 70)
    print("Alternative Approaches")
    print("=" * 70)

    print("""
1. SEPARATE CHARGE CONSTRAINTS per compound
   - Si3N4 region: Si_n × 4 + N × (-3) = 0 → N = (4/3) × Si_n
   - SiO2 region:  Si_o × 4 + O × (-2) = 0 → O = 2 × Si_o

2. I(N1s)/I(O1s) RATIO OBJECTIVE (user suggestion)
   - Instead of optimizing all intensities:
     L = ||I_pred - I_obs||² + λ × ||I_pred[N]/I_pred[O] - I_obs[N]/I_obs[O]||²
   - This isolates N vs O competition
   - Implementation: modify objective function

3. ELEMENT-PAIR LINKING
   - link N only to Si_n (not to Si_o)
   - link O only to Si_o (not to Si_n)
   - Requires refactoring SemiFreeSolverConfig

4. USE GenericLayerSolver (recommended for NO/ON/NON)
   - Already handles stoichiometry per layer
   - NO: [Si3N4 layer] + [SiO2 layer] + [Si substrate]
   - Works well when structure is known

5. USE FreeSolver + custom prior (for NON with merged N)
   - Bimodal prior for N
   - Already demonstrated to work
""")


def run_comparison_with_correct_setup():
    """Test SemiFreeSolver with a single-compound setup."""
    print("\n" + "=" * 70)
    print("Comparison: Single compound (SiO2/Si) vs Multi-compound (NO pattern)")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]

    # Test 1: Simple SiO2/Si (should work)
    print("\n--- Test 1: SiO2/Si (single compound) ---")
    elements_sio2 = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    # True SiO2/Si profile
    sigma = 5.0 / (2 * np.sqrt(np.log(2)))
    D_true_sio2 = np.zeros((len(depth), 3))
    sio2_profile = 0.5 * (1 - erf((depth - 20.0) / sigma))
    D_true_sio2[:, 0] = (1/3) * sio2_profile  # Si_ox
    D_true_sio2[:, 1] = (2/3) * sio2_profile  # O
    D_true_sio2[:, 2] = 0.5 * (1 + erf((depth - 20.0) / sigma))  # Si_bulk
    D_true_sio2 = np.clip(D_true_sio2, 1e-6, None)
    D_true_sio2 = D_true_sio2 / D_true_sio2.sum(axis=1, keepdims=True)

    tm_sio2 = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm_sio2.compute(D_true_sio2, elements_sio2)
    I_true_sio2 = tm_sio2.forward(D_true_sio2, normalize=True)

    np.random.seed(42)
    I_obs_sio2 = I_true_sio2 * (1 + 0.02 * np.random.randn(*I_true_sio2.shape))
    I_obs_sio2 = np.clip(I_obs_sio2, 1e-6, None)
    I_obs_sio2 = I_obs_sio2 / I_obs_sio2.sum(axis=1, keepdims=True)

    semi_config_sio2 = SemiFreeSolverConfig(
        free_indices=[0],  # Si_ox only
        linked_indices=[1],  # O
        substrate_indices=[2],
        thickness_init=30.0,
        abruptness_init=5.0,
    )

    solver_sio2 = SemiFreeSolver(
        tm_sio2, SolverConfig(method='LLS', max_iter=200),
        semi_config_sio2, elements=elements_sio2
    )
    result_sio2 = solver_sio2.solve(I_obs_sio2)

    print(f"  True thickness: 20.0 A")
    print(f"  Result thickness: {result_sio2.thickness:.1f} A")
    print(f"  Chi²: {result_sio2.chi2:.6f}")
    print(f"  O/Si_ox ratio at z=0: {result_sio2.D_opt[0, 1] / result_sio2.D_opt[0, 0]:.2f} (expected: 2.0)")

    # Test 2: NO pattern (multi-compound) - current implementation
    print("\n--- Test 2: NO pattern (current implementation) ---")
    elements_no = [
        ElementConfig('Si', '2p', 'Si3N4', oxidation_state=4),
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true_no = create_simple_no_pattern(depth, 5.0, 20.0)

    tm_no = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm_no.compute(D_true_no, elements_no)
    I_true_no = tm_no.forward(D_true_no, normalize=True)

    np.random.seed(42)
    I_obs_no = I_true_no * (1 + 0.02 * np.random.randn(*I_true_no.shape))
    I_obs_no = np.clip(I_obs_no, 1e-6, None)
    I_obs_no = I_obs_no / I_obs_no.sum(axis=1, keepdims=True)

    semi_config_no = SemiFreeSolverConfig(
        free_indices=[0, 2],  # Si_n, Si_o
        linked_indices=[1, 3],  # N, O
        substrate_indices=[4],
        thickness_init=30.0,
        abruptness_init=5.0,
    )

    solver_no = SemiFreeSolver(
        tm_no, SolverConfig(method='LLS', max_iter=200),
        semi_config_no, elements=elements_no
    )
    result_no = solver_no.solve(I_obs_no)

    print(f"  True thickness: 25.0 A (N=5 + O=20)")
    print(f"  Result thickness: {result_no.thickness:.1f} A")
    print(f"  Chi²: {result_no.chi2:.6f}")
    print(f"  N at z=0: {result_no.D_opt[0, 1]:.4f} (expected: ~0.57)")
    print(f"  O at z=0: {result_no.D_opt[0, 3]:.4f} (expected: ~0)")
    print(f"  N/O ratio at z=0: {result_no.D_opt[0, 1] / max(result_no.D_opt[0, 3], 1e-6):.2f}")

    if abs(result_no.D_opt[0, 1] - result_no.D_opt[0, 3]) < 0.1:
        print("\n  *** PROBLEM CONFIRMED: N and O have similar values! ***")
        print("      This is because linked elements are distributed equally.")


if __name__ == "__main__":
    diagnose_charge_coefficients()
    diagnose_profile_building()
    suggest_alternative_approaches()
    run_comparison_with_correct_setup()

    print("\n" + "=" * 70)
    print("Diagnosis complete!")
    print("=" * 70)
