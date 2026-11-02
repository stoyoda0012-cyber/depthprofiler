"""
NON Pattern Reconstruction using 3-Stage Optimization

Based on MEM_ver17_3_SiON algorithm:
- Stage 1: Optimize Si species (Si4+_SiO2, Si4+_SiON, Si_bulk)
- Stage 2: Optimize O and N_total with stoichiometry constraint
- Stage 3: Separate N into N1 (surface) and N2 (interface)

Key insight: N1 + N2 = N_total constraint allows bimodal detection

Usage:
    python -m deppro.tests.solver.test_non_three_stage
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.special import erf
import matplotlib.pyplot as plt
from deppro.paths import ensure_outputs_dir


def create_non_profile(depth, n1_thick, o_thick, n2_thick, interface_width=5.0):
    """
    Create NON pattern (Si3N4 / SiO2 / Si3N4 / Si).

    Elements:
    - A: Si4+ in SiO2 (index 0)
    - B: Si4+ in Si3N4 (index 1)
    - C: Si bulk (index 2)
    - D: O (index 3)
    - E: N total (index 4)

    For Stage 3, E is split into:
    - N1: surface Si3N4
    - N2: interface Si3N4
    """
    m = len(depth)
    sigma = interface_width / (2 * np.sqrt(np.log(2)))

    z1 = n1_thick
    z2 = n1_thick + o_thick
    z3 = n1_thick + o_thick + n2_thick

    D = np.zeros((m, 5))

    # Surface Si3N4 layer (0 to z1)
    sin1_profile = 0.5 * (1 - erf((depth - z1) / sigma))

    # SiO2 layer (z1 to z2)
    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))

    # Interface Si3N4 layer (z2 to z3)
    sin2_profile = 0.5 * (1 + erf((depth - z2) / sigma)) * 0.5 * (1 - erf((depth - z3) / sigma))

    # Si substrate (beyond z3)
    si_profile = 0.5 * (1 + erf((depth - z3) / sigma))

    # Element assignments based on stoichiometry
    # SiO2: Si:O = 1:2 -> Si = 1/3, O = 2/3
    # Si3N4: Si:N = 3:4 -> Si = 3/7, N = 4/7

    D[:, 0] = (1/3) * sio2_profile                        # A: Si4+ in SiO2
    D[:, 1] = (3/7) * (sin1_profile + sin2_profile)       # B: Si4+ in Si3N4
    D[:, 2] = si_profile                                   # C: Si bulk
    D[:, 3] = (2/3) * sio2_profile                        # D: O
    D[:, 4] = (4/7) * (sin1_profile + sin2_profile)       # E: N total

    # Normalize
    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    # Also return N1 and N2 separately for ground truth
    N1_true = (4/7) * sin1_profile
    N2_true = (4/7) * sin2_profile

    return D, N1_true, N2_true


def build_transfer_matrix(depth, angles, lambda_values):
    """Build transfer matrices for each element."""
    m = len(depth)
    n_angles = len(angles)
    step_size = depth[1] - depth[0] if len(depth) > 1 else 1.0

    T = {}
    for name, lam in lambda_values.items():
        T[name] = np.zeros((n_angles, m))
        for ia, theta in enumerate(angles):
            cos_theta = np.cos(np.radians(theta))
            lambda_eff = lam * cos_theta
            T[name][ia, :] = np.exp(-depth / lambda_eff) * step_size / lambda_eff

    return T


def stage1_objective(x, T, m, I_obs_Si, m0, alpha=0.0005):
    """
    Stage 1: MEM objective for Si species.

    Elements: A (Si4+ in SiO2), B (Si4+ in Si3N4), C (Si bulk)
    Q = 0.5 * Chi2 - alpha * S
    """
    # Reconstruct from log-space
    D_A = m0[:m] * np.exp(x[:m])
    D_B = m0[m:2*m] * np.exp(x[m:2*m])
    D_C = m0[2*m:3*m] * np.exp(x[2*m:3*m])

    # Forward model
    I_A = T['A'] @ D_A
    I_B = T['B'] @ D_B
    I_C = T['C'] @ D_C
    I_total = I_A + I_B + I_C

    X_A = I_A / I_total
    X_B = I_B / I_total
    X_C = I_C / I_total

    X_pred = np.concatenate([X_A, X_B, X_C])

    # Chi2
    chi2 = np.sum((X_pred - I_obs_Si) ** 2)

    # Entropy (MEM regularization)
    S = np.sum(m0 * (np.exp(x) - 1 - x * np.exp(x)))

    return 0.5 * chi2 - alpha * S


def stage1_constraint(x, m, m0):
    """
    Stage 1 constraint: stoichiometry sum = 1.

    From mycon1.m: (13/5)*A + (9/5)*B + C = 1

    This comes from:
    - SiO2: Si = 1/3 of total, O = 2/3 -> need 5/13 factor?
    - Actually looking at model.m: A has 5/13, B has 5/9 initial
    - The constraint ensures sum of all = 1 at each depth

    Let's use: A*coeff_A + B*coeff_B + C = 1
    where coeff reflects stoichiometry expansion
    """
    D_A = m0[:m] * np.exp(x[:m])
    D_B = m0[m:2*m] * np.exp(x[m:2*m])
    D_C = m0[2*m:3*m] * np.exp(x[2*m:3*m])

    # Stoichiometry: SiO2 expands to 3x (Si+2O), Si3N4 expands to 7/3x
    # (13/5)*A + (9/5)*B + C = 1
    # Actually this seems to be: each compound's total contribution sums to 1
    return (13/5) * D_A + (9/5) * D_B + D_C - 1


def stage2_objective(x, T, m, I_obs_NO, D_C):
    """
    Stage 2: Optimize O and N_total.

    Elements: D (O), E (N total)
    Constraint: (3/2)*D + (7/4)*E = 1 - C (non-substrate = oxide+nitride)
    """
    D_D = x[:m]  # O
    D_E = x[m:2*m]  # N total

    # Forward model with relative sensitivity correction
    I_D = T['D'] @ D_D
    I_E = T['E'] @ D_E

    # Sum for normalization
    R_D = T['D'].sum(axis=1)
    R_E = T['E'].sum(axis=1)

    # Relative sensitivity corrected
    X_D = (I_D / R_D) / ((I_D / R_D) + (I_E / R_E))
    X_E = (I_E / R_E) / ((I_D / R_D) + (I_E / R_E))

    X_pred = np.column_stack([X_D, X_E])

    return np.sum((X_pred - I_obs_NO) ** 2)


def stage3_objective(x, T, m, I_obs_N_resolved):
    """
    Stage 3: Separate N into N1 and N2.

    Constraint: N1 + N2 = E (N total from Stage 2)
    """
    D_N1 = x[:m]
    D_N2 = x[m:2*m]

    I_N1 = T['F'] @ D_N1
    I_N2 = T['F'] @ D_N2
    I_total = I_N1 + I_N2

    X_N1 = I_N1 / I_total
    X_N2 = I_N2 / I_total

    X_pred = np.concatenate([X_N1, X_N2])

    return np.sum((X_pred - I_obs_N_resolved) ** 2)


def run_three_stage_optimization(depth, angles, D_true, N1_true, N2_true, noise_level=0.02):
    """Run the 3-stage optimization algorithm."""
    m = len(depth)

    # IMFP values (Angstrom) - matching MEM_ver17_3
    lambda_values = {
        'A': 18,  # Si4+ in SiO2
        'B': 18,  # Si4+ in Si3N4
        'C': 15,  # Si bulk
        'D': 11,  # O
        'E': 13,  # N
        'F': 10,  # N (different KE for resolved)
    }

    # Build transfer matrices
    T = build_transfer_matrix(depth, angles, lambda_values)

    # Generate synthetic observations
    # Stage 1: Si species
    I_A = T['A'] @ D_true[:, 0]
    I_B = T['B'] @ D_true[:, 1]
    I_C = T['C'] @ D_true[:, 2]
    I_Si_total = I_A + I_B + I_C

    X_obs_A = I_A / I_Si_total
    X_obs_B = I_B / I_Si_total
    X_obs_C = I_C / I_Si_total

    # Stage 2: O/N
    I_D = T['D'] @ D_true[:, 3]
    I_E = T['E'] @ D_true[:, 4]
    R_D = T['D'].sum(axis=1)
    R_E = T['E'].sum(axis=1)
    rs = 2.9  # Relative sensitivity N/O

    X_obs_D = (I_D / R_D) / ((I_D / R_D) + rs * (I_E / R_E))
    X_obs_E = rs * (I_E / R_E) / ((I_D / R_D) + rs * (I_E / R_E))

    # Stage 3: N resolved (using ground truth N1, N2)
    I_N1 = T['F'] @ N1_true
    I_N2 = T['F'] @ N2_true
    I_N_total = I_N1 + I_N2

    X_obs_N1 = I_N1 / I_N_total
    X_obs_N2 = I_N2 / I_N_total

    # Add noise
    np.random.seed(42)
    I_obs_Si = np.concatenate([X_obs_A, X_obs_B, X_obs_C])
    I_obs_Si = I_obs_Si * (1 + noise_level * np.random.randn(len(I_obs_Si)))

    I_obs_NO = np.column_stack([X_obs_D, X_obs_E])
    I_obs_NO = I_obs_NO * (1 + noise_level * np.random.randn(*I_obs_NO.shape))
    I_obs_NO = np.clip(I_obs_NO, 1e-6, None)
    I_obs_NO = I_obs_NO / I_obs_NO.sum(axis=1, keepdims=True)

    I_obs_N_resolved = np.concatenate([X_obs_N1, X_obs_N2])
    I_obs_N_resolved = I_obs_N_resolved * (1 + noise_level * np.random.randn(len(I_obs_N_resolved)))

    results = {}

    # =========== Stage 1 ===========
    print("\n--- Stage 1: Si species optimization ---")

    # Initial model (exponential decay prior)
    model_lambda = 7
    step_size = depth[1] - depth[0]
    m0_A = (5/13) * np.exp(-depth / model_lambda)
    m0_B = (5/9) * np.exp(-depth / model_lambda)
    m0_C = 1 - np.exp(-depth / model_lambda)
    m0_C[0] = 1e-10  # Avoid log(0)
    m0 = np.concatenate([m0_A, m0_B, m0_C])
    m0 = np.clip(m0, 1e-10, None)

    x0 = np.log(m0 / m0)  # = zeros

    # Constraint: (13/5)*A + (9/5)*B + C = 1
    def constraint_stage1(x):
        D_A = m0[:m] * np.exp(x[:m])
        D_B = m0[m:2*m] * np.exp(x[m:2*m])
        D_C = m0[2*m:3*m] * np.exp(x[2*m:3*m])
        return (13/5) * D_A + (9/5) * D_B + D_C - 1

    # Bounds: x >= -log(m0) ensures D >= 0
    lb = -np.log(m0)

    res1 = minimize(
        lambda x: stage1_objective(x, T, m, I_obs_Si, m0, alpha=0.0005),
        x0,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': constraint_stage1},
        bounds=[(lb[i], None) for i in range(len(x0))],
        options={'maxiter': 500, 'disp': False}
    )

    D_opt_A = m0[:m] * np.exp(res1.x[:m])
    D_opt_B = m0[m:2*m] * np.exp(res1.x[m:2*m])
    D_opt_C = m0[2*m:3*m] * np.exp(res1.x[2*m:3*m])

    print(f"  Stage 1 converged: {res1.success}, iterations: {res1.nit}")

    # Fit sigmoid to Si_bulk for thickness estimation
    from scipy.optimize import curve_fit
    def sigmoid(z, t, delta):
        return 1 / (1 + np.exp(-(z - t) / delta))

    try:
        popt, _ = curve_fit(sigmoid, depth, D_opt_C, p0=[25, 1], maxfev=5000)
        Tox = popt[0]
        delta_Tox = popt[1]
        D_opt_C = sigmoid(depth, Tox, delta_Tox)
        print(f"  Estimated thickness: {Tox:.1f} A, interface width: {delta_Tox:.1f} A")
    except:
        Tox = 25
        delta_Tox = 5

    results['stage1'] = {
        'D_A': D_opt_A,
        'D_B': D_opt_B,
        'D_C': D_opt_C,
        'Tox': Tox,
        'delta_Tox': delta_Tox
    }

    # =========== Stage 2 ===========
    print("\n--- Stage 2: O/N total optimization ---")

    # Initial guess from Stage 1 stoichiometry
    # SiO2: Si4+ = 1/3, O = 2/3 -> O = 2*Si_A
    # Si3N4: Si = 3/7, N = 4/7 -> N = (4/3)*Si_B
    D_init_D = D_opt_A * 2  # O from SiO2 stoichiometry
    D_init_E = D_opt_B * (4/3)  # N from Si3N4 stoichiometry

    x0_stage2 = np.concatenate([D_init_D, D_init_E])
    x0_stage2 = np.clip(x0_stage2, 1e-6, None)

    # Constraint: (3/2)*D + (7/4)*E = 1 - C
    # This ensures O + N fill the non-substrate region with correct stoichiometry
    A_eq = np.zeros((m, 2*m))
    A_eq[:, :m] = (3/2) * np.eye(m)  # O coefficient
    A_eq[:, m:] = (7/4) * np.eye(m)  # N coefficient
    b_eq = 1 - D_opt_C

    constraint_stage2 = LinearConstraint(A_eq, b_eq, b_eq)

    res2 = minimize(
        lambda x: stage2_objective(x, T, m, I_obs_NO, D_opt_C),
        x0_stage2,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq},
        bounds=[(0, None) for _ in range(2*m)],
        options={'maxiter': 500, 'disp': False}
    )

    D_opt_D = res2.x[:m]  # O
    D_opt_E = res2.x[m:2*m]  # N total

    print(f"  Stage 2 converged: {res2.success}, iterations: {res2.nit}")

    # Recalculate Si_A from O stoichiometry
    D_opt_A_recalc = D_opt_D / 2 + D_opt_E * (3/4)

    results['stage2'] = {
        'D_D': D_opt_D,  # O
        'D_E': D_opt_E,  # N total
        'D_A_recalc': D_opt_A_recalc
    }

    # =========== Stage 3 ===========
    print("\n--- Stage 3: N separation (N1 + N2 = N_total) ---")

    # Initial guess: split N equally
    D_init_N1 = D_opt_E / 2
    D_init_N2 = D_opt_E / 2

    x0_stage3 = np.concatenate([D_init_N1, D_init_N2])

    # Constraint: N1 + N2 = E
    A_eq_n = np.zeros((m, 2*m))
    A_eq_n[:, :m] = np.eye(m)  # N1
    A_eq_n[:, m:] = np.eye(m)  # N2
    b_eq_n = D_opt_E

    res3 = minimize(
        lambda x: stage3_objective(x, T, m, I_obs_N_resolved),
        x0_stage3,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': lambda x: A_eq_n @ x - b_eq_n},
        bounds=[(0, None) for _ in range(2*m)],
        options={'maxiter': 500, 'disp': False}
    )

    D_opt_N1 = res3.x[:m]
    D_opt_N2 = res3.x[m:2*m]

    print(f"  Stage 3 converged: {res3.success}, iterations: {res3.nit}")

    results['stage3'] = {
        'D_N1': D_opt_N1,
        'D_N2': D_opt_N2
    }

    return results


def analyze_and_plot(depth, D_true, N1_true, N2_true, results):
    """Analyze results and create plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # True profiles
    ax = axes[0, 0]
    ax.plot(depth, D_true[:, 0], 'b-', linewidth=2, label='Si4+ (SiO2)')
    ax.plot(depth, D_true[:, 1], 'g-', linewidth=2, label='Si4+ (Si3N4)')
    ax.plot(depth, D_true[:, 2], 'k--', linewidth=1, label='Si bulk')
    ax.plot(depth, D_true[:, 3], 'r-', linewidth=2, label='O')
    ax.plot(depth, D_true[:, 4], 'm-', linewidth=2, label='N total')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('TRUE Profile')
    ax.legend(loc='upper right', fontsize=8)

    # True N1, N2
    ax = axes[0, 1]
    ax.plot(depth, N1_true, 'c-', linewidth=2, label='N1 (surface)')
    ax.plot(depth, N2_true, 'y-', linewidth=2, label='N2 (interface)')
    ax.plot(depth, N1_true + N2_true, 'm--', linewidth=1, label='N total')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('TRUE N Distribution (Bimodal)')
    ax.legend(loc='upper right')

    # Stage 1 result
    ax = axes[0, 2]
    ax.plot(depth, results['stage1']['D_A'], 'b-', linewidth=2, label='Si4+ (SiO2) opt')
    ax.plot(depth, results['stage1']['D_B'], 'g-', linewidth=2, label='Si4+ (Si3N4) opt')
    ax.plot(depth, results['stage1']['D_C'], 'k--', linewidth=1, label='Si bulk opt')
    ax.plot(depth, D_true[:, 0], 'b:', alpha=0.5, label='true')
    ax.plot(depth, D_true[:, 1], 'g:', alpha=0.5)
    ax.plot(depth, D_true[:, 2], 'k:', alpha=0.5)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title(f"Stage 1: Si species (Tox={results['stage1']['Tox']:.1f}A)")
    ax.legend(loc='upper right', fontsize=8)

    # Stage 2 result
    ax = axes[1, 0]
    ax.plot(depth, results['stage2']['D_D'], 'r-', linewidth=2, label='O opt')
    ax.plot(depth, results['stage2']['D_E'], 'm-', linewidth=2, label='N total opt')
    ax.plot(depth, D_true[:, 3], 'r:', alpha=0.5, label='O true')
    ax.plot(depth, D_true[:, 4], 'm:', alpha=0.5, label='N true')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('Stage 2: O/N total')
    ax.legend(loc='upper right')

    # Stage 3 result - THE KEY RESULT
    ax = axes[1, 1]
    ax.plot(depth, results['stage3']['D_N1'], 'c-', linewidth=2, label='N1 opt (surface)')
    ax.plot(depth, results['stage3']['D_N2'], 'y-', linewidth=2, label='N2 opt (interface)')
    ax.plot(depth, N1_true, 'c:', alpha=0.5, label='N1 true')
    ax.plot(depth, N2_true, 'y:', alpha=0.5, label='N2 true')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('Stage 3: N Separation (BIMODAL)')
    ax.legend(loc='upper right')

    # Comparison: N1+N2 vs N_total
    ax = axes[1, 2]
    N_sum = results['stage3']['D_N1'] + results['stage3']['D_N2']
    ax.plot(depth, N_sum, 'k-', linewidth=2, label='N1+N2 (reconstructed)')
    ax.plot(depth, results['stage2']['D_E'], 'm--', linewidth=1, label='N total (Stage 2)')
    ax.plot(depth, D_true[:, 4], 'm:', alpha=0.5, label='N true')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Depth (A)')
    ax.set_ylabel('Concentration')
    ax.set_title('N Conservation Check')
    ax.legend(loc='upper right')

    plt.suptitle('3-Stage Optimization for NON Pattern', fontsize=14)
    plt.tight_layout()
    output_path = str(ensure_outputs_dir() / 'three_stage_non_pattern.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.close()

    # Analyze bimodality
    N1_opt = results['stage3']['D_N1']
    N2_opt = results['stage3']['D_N2']

    N1_peak_idx = np.argmax(N1_opt)
    N2_peak_idx = np.argmax(N2_opt)

    print(f"\n=== Bimodal Analysis ===")
    print(f"N1 peak position: {depth[N1_peak_idx]:.1f} A (expected: ~2.5 A)")
    print(f"N2 peak position: {depth[N2_peak_idx]:.1f} A (expected: ~22.5 A)")
    print(f"Peak separation: {abs(depth[N2_peak_idx] - depth[N1_peak_idx]):.1f} A")

    # Check if bimodal
    is_bimodal = abs(depth[N2_peak_idx] - depth[N1_peak_idx]) > 5
    print(f"Bimodal detection: {'SUCCESS' if is_bimodal else 'FAILED'}")

    return is_bimodal


def run_matlab_style_optimization(depth, angles, D_true, N1_true, N2_true, noise_level=0.02):
    """
    Run optimization matching MATLAB MEM_ver17_3 more closely.

    Key difference: Stage 3 requires RESOLVED N spectra as input.
    If N1 and N2 are chemically indistinguishable, Stage 3 cannot separate them.

    This test shows:
    1. What MEM_ver17_3 can do when N is resolved (e.g., different binding energies)
    2. Why it fails when N is NOT resolved (pure NON without chemical shift)
    """
    m = len(depth)
    step_size = depth[1] - depth[0]

    # Parameters from MEM_ver17_3
    lambda_A = 18  # Si4+ in oxide
    lambda_B = 18  # Si4+ in nitride
    lambda_C = 15  # Si bulk
    lambda_D = 11  # O
    lambda_E = 13  # N
    lambda_F = 10  # N at different KE (for resolved spectra)
    rs = 2.9       # Relative sensitivity N/O

    # Build transfer matrices (matching MATLAB's transfer.m)
    n_angles = len(angles)
    T = {}
    for name, lam in [('A', lambda_A), ('B', lambda_B), ('C', lambda_C),
                       ('D', lambda_D), ('E', lambda_E), ('F', lambda_F)]:
        T[name] = np.zeros((n_angles, m))
        for ia, theta in enumerate(angles):
            cos_theta = np.cos(np.radians(theta))
            for j in range(m):
                T[name][ia, j] = np.exp(-j * step_size / lam / cos_theta)

    # Calculate R (sum of transfer matrix, for normalization)
    R = {name: T[name].sum(axis=1) for name in T}

    # ===== Generate synthetic observations =====
    # Si species (Stage 1 input)
    I_A = T['A'] @ D_true[:, 0]
    I_B = T['B'] @ D_true[:, 1]
    I_C = T['C'] @ D_true[:, 2]
    I_Si_sum = I_A + I_B + I_C
    X_obs_A = I_A / I_Si_sum
    X_obs_B = I_B / I_Si_sum
    X_obs_C = I_C / I_Si_sum

    # O/N (Stage 2 input)
    I_D = T['D'] @ D_true[:, 3]
    I_E = T['E'] @ D_true[:, 4]
    # XobsNO=[IobsD./(rs*IobsE+IobsD) rs*IobsE./(rs*IobsE+IobsD)]
    X_obs_D = I_D / (rs * I_E + I_D)
    X_obs_E = rs * I_E / (rs * I_E + I_D)
    X_obs_NO = np.column_stack([X_obs_D, X_obs_E])

    # N resolved (Stage 3 input) - THIS IS THE KEY
    I_N1 = T['F'] @ N1_true
    I_N2 = T['F'] @ N2_true
    I_N_sum = I_N1 + I_N2
    X_obs_N1 = I_N1 / I_N_sum
    X_obs_N2 = I_N2 / I_N_sum

    # Add noise
    np.random.seed(42)
    noise_mult = 1 + noise_level * np.random.randn(n_angles)

    X_obs = np.column_stack([X_obs_A, X_obs_B, X_obs_C])
    X_obs = X_obs * noise_mult[:, np.newaxis]
    X_obs = X_obs / X_obs.sum(axis=1, keepdims=True)
    xXobs = X_obs.T.flatten()

    X_obs_NO = X_obs_NO * (1 + noise_level * np.random.randn(*X_obs_NO.shape))
    X_obs_NO = np.clip(X_obs_NO, 1e-6, None)
    X_obs_NO = X_obs_NO / X_obs_NO.sum(axis=1, keepdims=True)

    xXobs_N_resolved = np.concatenate([
        X_obs_N1 * (1 + noise_level * np.random.randn(n_angles)),
        X_obs_N2 * (1 + noise_level * np.random.randn(n_angles))
    ])

    results = {}

    # ===== Stage 1: Si species =====
    print("\n--- Stage 1: Si species (MEM) ---")

    # Initial model (model.m)
    model_lambda = 7
    m0 = np.zeros(3 * m)
    for j in range(m):
        m0[j] = (5/13) * np.exp(-j * step_size / model_lambda)
        m0[m + j] = (5/9) * np.exp(-j * step_size / model_lambda)
        m0[2*m + j] = 1 - np.exp(-j * step_size / model_lambda)
    m0[2*m] = 1e-10  # Avoid log(0)
    m0 = np.clip(m0, 1e-10, None)

    # Initial guess in log space
    x0 = np.zeros(3 * m)

    def myfun1(x):
        D_A = m0[:m] * np.exp(x[:m])
        D_B = m0[m:2*m] * np.exp(x[m:2*m])
        D_C = m0[2*m:3*m] * np.exp(x[2*m:3*m])

        I_A = T['A'] @ D_A
        I_B = T['B'] @ D_B
        I_C = T['C'] @ D_C
        I_sum = I_A + I_B + I_C

        X_A = I_A / I_sum
        X_B = I_B / I_sum
        X_C = I_C / I_sum

        chi2 = np.sum((np.concatenate([X_A, X_B, X_C]) - xXobs) ** 2)
        S = np.sum(m0 * (np.exp(x) - 1 - x * np.exp(x)))
        return 0.5 * chi2 - 0.0005 * S

    def mycon1(x):
        D_A = m0[:m] * np.exp(x[:m])
        D_B = m0[m:2*m] * np.exp(x[m:2*m])
        D_C = m0[2*m:3*m] * np.exp(x[2*m:3*m])
        return (13/5) * D_A + (9/5) * D_B + D_C - 1

    # Bounds: x >= -log(m0)
    lb = -np.log(m0)

    res1 = minimize(
        myfun1, x0,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': mycon1},
        bounds=[(lb[i], None) for i in range(3*m)],
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    D_opt_A = m0[:m] * np.exp(res1.x[:m])
    D_opt_B = m0[m:2*m] * np.exp(res1.x[m:2*m])
    D_opt_C = m0[2*m:3*m] * np.exp(res1.x[2*m:3*m])

    print(f"  Converged: {res1.success}, iterations: {res1.nit}, fval: {res1.fun:.6f}")

    # Sigmoid fit for thickness
    from scipy.optimize import curve_fit
    def sigmoid(z, t, delta):
        return 1 / (1 + np.exp(-(z - t) / delta))

    try:
        popt, _ = curve_fit(sigmoid, depth, D_opt_C, p0=[25, 2], bounds=([0, 0.1], [100, 20]))
        Tox, delta_Tox = popt
        D_opt_C_fit = sigmoid(depth, Tox, delta_Tox)
        print(f"  Thickness: {Tox:.1f} A, interface: {delta_Tox:.1f} A")
    except Exception as e:
        print(f"  Sigmoid fit failed: {e}")
        Tox, delta_Tox = 25, 5
        D_opt_C_fit = D_opt_C

    results['stage1'] = {
        'D_A': D_opt_A, 'D_B': D_opt_B, 'D_C': D_opt_C_fit,
        'Tox': Tox, 'delta_Tox': delta_Tox
    }

    # ===== Stage 2: O/N total =====
    print("\n--- Stage 2: O/N total ---")

    # Initial guess from stoichiometry
    D_init_D = D_opt_A * (4/5) + D_opt_B * (2/5)  # O estimate
    D_init_E = D_opt_A * (4/5) + D_opt_B * (2/5)  # N estimate (same as O initially)

    x0_2 = np.concatenate([D_init_D, D_init_E])
    x0_2 = np.clip(x0_2, 1e-6, None)

    # Constraint: (3/2)*D + (7/4)*E = 1 - C
    def constraint_2(x):
        D_D = x[:m]
        D_E = x[m:2*m]
        return (3/2) * D_D + (7/4) * D_E - (1 - D_opt_C_fit)

    def myfun2(x):
        D_D = x[:m]
        D_E = x[m:2*m]

        I_D = T['D'] @ D_D
        I_E = T['E'] @ D_E

        # Matching MEM_ver17_3 normalization
        X_D = (I_D / R['D']) / ((I_D / R['D']) + (I_E / R['E']))
        X_E = (I_E / R['E']) / ((I_D / R['D']) + (I_E / R['E']))

        return np.sum((np.column_stack([X_D, X_E]) - X_obs_NO) ** 2)

    res2 = minimize(
        myfun2, x0_2,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': constraint_2},
        bounds=[(0, None) for _ in range(2*m)],
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    D_opt_D = res2.x[:m]
    D_opt_E = res2.x[m:2*m]

    print(f"  Converged: {res2.success}, iterations: {res2.nit}, fval: {res2.fun:.6f}")

    results['stage2'] = {'D_D': D_opt_D, 'D_E': D_opt_E}

    # ===== Stage 3: N separation (REQUIRES RESOLVED SPECTRA) =====
    print("\n--- Stage 3: N separation ---")
    print("  NOTE: This requires chemically resolved N1/N2 spectra!")

    x0_3 = np.concatenate([D_opt_E / 2, D_opt_E / 2])

    # Constraint: N1 + N2 = E
    def constraint_3(x):
        return x[:m] + x[m:2*m] - D_opt_E

    def myfun3(x):
        D_N1 = x[:m]
        D_N2 = x[m:2*m]

        I_N1 = T['F'] @ D_N1
        I_N2 = T['F'] @ D_N2
        I_sum = I_N1 + I_N2

        X_N1 = I_N1 / I_sum
        X_N2 = I_N2 / I_sum

        return np.sum((np.concatenate([X_N1, X_N2]) - xXobs_N_resolved) ** 2)

    res3 = minimize(
        myfun3, x0_3,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': constraint_3},
        bounds=[(0, None) for _ in range(2*m)],
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    D_opt_N1 = res3.x[:m]
    D_opt_N2 = res3.x[m:2*m]

    print(f"  Converged: {res3.success}, iterations: {res3.nit}, fval: {res3.fun:.6f}")

    results['stage3'] = {'D_N1': D_opt_N1, 'D_N2': D_opt_N2}

    return results


def test_three_stage():
    """Main test function."""
    print("=" * 70)
    print("3-Stage Optimization for NON Pattern")
    print("Based on MEM_ver17_3_SiON algorithm")
    print("=" * 70)

    # Setup (matching MEM_ver17_3 parameters)
    tA = 27.515  # Angstrom per unit
    xA = 10
    dt = tA / xA
    ttotal = 70
    m = int(round(ttotal / dt))

    depth = np.arange(m) * dt
    angles = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 57.5, 60]

    # True structure (in the original scale)
    n1_thick = 5.0   # Surface Si3N4
    o_thick = 15.0   # SiO2
    n2_thick = 5.0   # Interface Si3N4

    print(f"\nStructure: Si3N4 ({n1_thick}A) / SiO2 ({o_thick}A) / Si3N4 ({n2_thick}A) / Si")
    print(f"Depth points: {m}, step size: {dt:.2f} A")

    # Create true profile
    D_true, N1_true, N2_true = create_non_profile(depth, n1_thick, o_thick, n2_thick)

    # Run MATLAB-style optimization
    results = run_matlab_style_optimization(depth, angles, D_true, N1_true, N2_true, noise_level=0.02)

    # Analyze and plot
    is_bimodal = analyze_and_plot(depth, D_true, N1_true, N2_true, results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Stage 1: Si species separation")
    print(f"Stage 2: O/N total optimization")
    print(f"Stage 3: N bimodal separation - {'SUCCESS' if is_bimodal else 'FAILED'}")
    print("")
    print("IMPORTANT: Stage 3 requires chemically RESOLVED N spectra.")
    print("If N1 and N2 have the same binding energy, they CANNOT be separated!")
    print("=" * 70)

    return results, is_bimodal


if __name__ == "__main__":
    results, is_bimodal = test_three_stage()
