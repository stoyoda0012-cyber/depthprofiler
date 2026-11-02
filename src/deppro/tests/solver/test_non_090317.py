"""
Test NON pattern using 090317 approach (SemiFreeSolver-like).

This approach:
- 4 components: Si-B (bulk), Si-O (oxide), O, N
- O and N are FREE variables (independent)
- Si-O is computed from O, N via stoichiometry
- Constraint: (3/2)*O + (7/4)*N = 1 - Si-B

Key difference from Run_DepthProfiling.m:
- O and N are optimized INDEPENDENTLY (not linked)
- This allows distinguishing NON from ONO patterns

Based on: python/NON/090317/Run_SQP.m
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from deppro.paths import ensure_outputs_dir


def imfp_tpp2m(KE: float, Nv: int, density: float, Aw: float, Eg: float) -> float:
    """TPP-2M formula for IMFP."""
    gamma = 0.191 * density ** (-0.50)
    Ep = 28.8 * (density * Nv / Aw) ** 0.5
    beta = -0.10 + 0.944 / (Ep**2 + Eg**2) ** 0.5 + 0.069 * density ** 0.1
    U = Ep**2 / 829.4
    C = 1.97 - 0.91 * U
    D = 53.4 - 20.8 * U
    return KE / (Ep**2 * (beta * np.log(gamma * KE) - C / KE + D / KE**2))


def build_transfer_matrix(depth: np.ndarray, angles: np.ndarray,
                          lambda_val: float) -> np.ndarray:
    """Build transfer matrix T[angle, depth]."""
    m = len(depth)
    n_angles = len(angles)
    step_size = depth[1] - depth[0] if len(depth) > 1 else 1.0

    T = np.zeros((n_angles, m))
    for ia, theta in enumerate(angles):
        cos_theta = np.cos(np.radians(theta))
        for j in range(m):
            # Cumulative attenuation from surface to depth j
            T[ia, j] = np.exp(-j * step_size / lambda_val / cos_theta)

    return T


def create_non_profile(depth: np.ndarray, n1_thick: float, o_thick: float,
                       n2_thick: float) -> np.ndarray:
    """
    Create NON (Si3N4/SiO2/Si3N4/Si) depth profile.

    Returns D[depth, 4]: Si-B, Si-O, O, N
    """
    m = len(depth)
    D = np.zeros((m, 4))

    for j, z in enumerate(depth):
        if z < n1_thick:
            # Surface Si3N4 layer
            # Si3N4: Si=3/7, N=4/7, O=0
            D[j, 0] = 0.0      # Si-B
            D[j, 1] = 3/7      # Si-O (actually Si in nitride)
            D[j, 2] = 0.0      # O
            D[j, 3] = 4/7      # N
        elif z < n1_thick + o_thick:
            # SiO2 layer
            # SiO2: Si=1/3, O=2/3, N=0
            D[j, 0] = 0.0      # Si-B
            D[j, 1] = 1/3      # Si-O
            D[j, 2] = 2/3      # O
            D[j, 3] = 0.0      # N
        elif z < n1_thick + o_thick + n2_thick:
            # Interface Si3N4 layer
            D[j, 0] = 0.0      # Si-B
            D[j, 1] = 3/7      # Si-O
            D[j, 2] = 0.0      # O
            D[j, 3] = 4/7      # N
        else:
            # Si substrate
            D[j, 0] = 1.0      # Si-B
            D[j, 1] = 0.0      # Si-O
            D[j, 2] = 0.0      # O
            D[j, 3] = 0.0      # N

    return D


def create_ono_profile(depth: np.ndarray, o1_thick: float, n_thick: float,
                       o2_thick: float) -> np.ndarray:
    """
    Create ONO (SiO2/Si3N4/SiO2/Si) depth profile.
    """
    m = len(depth)
    D = np.zeros((m, 4))

    for j, z in enumerate(depth):
        if z < o1_thick:
            # Surface SiO2 layer
            D[j, 0] = 0.0
            D[j, 1] = 1/3
            D[j, 2] = 2/3
            D[j, 3] = 0.0
        elif z < o1_thick + n_thick:
            # Si3N4 layer
            D[j, 0] = 0.0
            D[j, 1] = 3/7
            D[j, 2] = 0.0
            D[j, 3] = 4/7
        elif z < o1_thick + n_thick + o2_thick:
            # Interface SiO2 layer
            D[j, 0] = 0.0
            D[j, 1] = 1/3
            D[j, 2] = 2/3
            D[j, 3] = 0.0
        else:
            # Si substrate
            D[j, 0] = 1.0
            D[j, 1] = 0.0
            D[j, 2] = 0.0
            D[j, 3] = 0.0

    return D


def run_090317_optimization(depth: np.ndarray, angles: np.ndarray,
                            D_true: np.ndarray, noise_level: float = 0.02,
                            method: str = 'LLS', alpha: float = 0.0005):
    """
    Run 090317-style optimization.

    Parameters
    ----------
    depth : array
        Depth points
    angles : array
        Measurement angles
    D_true : array
        True depth profile [m, 4]: Si-B, Si-O, O, N
    noise_level : float
        Relative noise level
    method : str
        'LLS' or 'MEM'
    alpha : float
        MEM regularization parameter

    Returns
    -------
    D_opt : array
        Optimized depth profile
    """
    m = len(depth)
    n_angles = len(angles)
    step_size = depth[1] - depth[0]

    # IMFP values (approximate, from simulator.m)
    # Using typical values for SiON system at ~1000 eV
    lambda_Si_B = 25.0   # Si bulk
    lambda_Si_O = 22.0   # Si in oxide
    lambda_O = 18.0      # O
    lambda_N = 20.0      # N

    # Build transfer matrices
    T = [
        build_transfer_matrix(depth, angles, lambda_Si_B),  # Si-B
        build_transfer_matrix(depth, angles, lambda_Si_O),  # Si-O
        build_transfer_matrix(depth, angles, lambda_O),     # O
        build_transfer_matrix(depth, angles, lambda_N),     # N
    ]

    # Generate synthetic observations
    I_true = np.zeros((n_angles, 4))
    for k in range(4):
        I_true[:, k] = T[k] @ D_true[:, k]

    # Add noise
    np.random.seed(42)
    I_obs = I_true * (1 + noise_level * np.random.randn(*I_true.shape))

    # Convert to atomic fractions
    X_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    # Find oxide region (where Si-B < 1)
    oxide_mask = D_true[:, 0] < 0.99
    m_oxide = oxide_mask.sum()
    dep_ind = np.where(oxide_mask)[0]

    print(f"Oxide region: {m_oxide} points (0 to {depth[dep_ind[-1]] if len(dep_ind) > 0 else 0:.1f} Å)")

    # Get Si-B profile (fixed from Si analysis)
    D_SiB = D_true[:, 0].copy()  # In real case, this comes from Si 2p analysis

    # Initial guess: uniform distribution
    if method == 'LLS':
        # Direct O, N concentrations
        x0_O = np.ones(m_oxide) * 0.3
        x0_N = np.ones(m_oxide) * 0.2
        x0 = np.concatenate([x0_O, x0_N])
    else:  # MEM
        # Prior model (exponential decay)
        m0_O = 0.3 * np.exp(-np.arange(m_oxide) * step_size / 15)
        m0_N = 0.2 * np.exp(-np.arange(m_oxide) * step_size / 15)
        m0 = np.concatenate([m0_O, m0_N])
        m0 = np.clip(m0, 1e-10, None)
        x0 = np.zeros(2 * m_oxide)  # log(D/m0) = 0 initially

    def objective_lls(x):
        """LLS objective: minimize Chi2."""
        D_O = x[:m_oxide]
        D_N = x[m_oxide:2*m_oxide]

        # Compute Si-O from stoichiometry
        D_SiO = (1/2) * D_O + (3/4) * D_N

        # Build full profile
        D_opt = np.zeros((m, 4))
        D_opt[:, 0] = D_SiB
        D_opt[dep_ind, 1] = D_SiO
        D_opt[dep_ind, 2] = D_O
        D_opt[dep_ind, 3] = D_N

        # Calculate intensities
        I_calc = np.zeros((n_angles, 4))
        for k in range(4):
            I_calc[:, k] = T[k] @ D_opt[:, k]

        # Atomic fractions
        X_calc = I_calc / I_calc.sum(axis=1, keepdims=True)

        # Chi2 on O, N fractions only (more sensitive)
        X_ON_calc = X_calc[:, 2:4] / X_calc[:, 2:4].sum(axis=1, keepdims=True)
        X_ON_obs = X_obs[:, 2:4] / X_obs[:, 2:4].sum(axis=1, keepdims=True)

        chi2 = np.sum((X_ON_calc - X_ON_obs) ** 2)
        return 0.5 * chi2

    def objective_mem(x):
        """MEM objective: Chi2 - alpha*S."""
        D_O = m0[:m_oxide] * np.exp(x[:m_oxide])
        D_N = m0[m_oxide:2*m_oxide] * np.exp(x[m_oxide:2*m_oxide])

        D_SiO = (1/2) * D_O + (3/4) * D_N

        D_opt = np.zeros((m, 4))
        D_opt[:, 0] = D_SiB
        D_opt[dep_ind, 1] = D_SiO
        D_opt[dep_ind, 2] = D_O
        D_opt[dep_ind, 3] = D_N

        I_calc = np.zeros((n_angles, 4))
        for k in range(4):
            I_calc[:, k] = T[k] @ D_opt[:, k]

        X_calc = I_calc / I_calc.sum(axis=1, keepdims=True)

        X_ON_calc = X_calc[:, 2:4] / X_calc[:, 2:4].sum(axis=1, keepdims=True)
        X_ON_obs = X_obs[:, 2:4] / X_obs[:, 2:4].sum(axis=1, keepdims=True)

        chi2 = np.sum((X_ON_calc - X_ON_obs) ** 2)
        S = np.sum(m0 * (np.exp(x) - 1 - x * np.exp(x)))

        return 0.5 * chi2 - alpha * S

    def constraint_stoichiometry(x):
        """Stoichiometry constraint: (3/2)*O + (7/4)*N = 1 - Si-B."""
        if method == 'LLS':
            D_O = x[:m_oxide]
            D_N = x[m_oxide:2*m_oxide]
        else:
            D_O = m0[:m_oxide] * np.exp(x[:m_oxide])
            D_N = m0[m_oxide:2*m_oxide] * np.exp(x[m_oxide:2*m_oxide])

        return (3/2) * D_O + (7/4) * D_N - (1 - D_SiB[dep_ind])

    # Bounds
    if method == 'LLS':
        bounds = [(1e-6, 1.0)] * (2 * m_oxide)
        # Linear equality constraint
        Aeq = np.zeros((m_oxide, 2 * m_oxide))
        Aeq[:, :m_oxide] = (3/2) * np.eye(m_oxide)
        Aeq[:, m_oxide:] = (7/4) * np.eye(m_oxide)
        beq = 1 - D_SiB[dep_ind]

        result = minimize(
            objective_lls, x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': constraint_stoichiometry},
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
    else:  # MEM
        bounds = [(None, -np.log(m0[i])) for i in range(2 * m_oxide)]
        result = minimize(
            objective_mem, x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': constraint_stoichiometry},
            options={'maxiter': 1000, 'ftol': 1e-12}
        )

    print(f"Optimization: success={result.success}, iterations={result.nit}, fval={result.fun:.6f}")

    # Extract results
    if method == 'LLS':
        D_O_opt = result.x[:m_oxide]
        D_N_opt = result.x[m_oxide:2*m_oxide]
    else:
        D_O_opt = m0[:m_oxide] * np.exp(result.x[:m_oxide])
        D_N_opt = m0[m_oxide:2*m_oxide] * np.exp(result.x[m_oxide:2*m_oxide])

    D_SiO_opt = (1/2) * D_O_opt + (3/4) * D_N_opt

    D_opt = np.zeros((m, 4))
    D_opt[:, 0] = D_SiB
    D_opt[dep_ind, 1] = D_SiO_opt
    D_opt[dep_ind, 2] = D_O_opt
    D_opt[dep_ind, 3] = D_N_opt

    return D_opt, result


def analyze_pattern(depth: np.ndarray, D_opt: np.ndarray) -> str:
    """Analyze the reconstructed profile to determine pattern type."""
    # Find where O and N are significant
    O_profile = D_opt[:, 2]
    N_profile = D_opt[:, 3]

    # Find peaks
    from scipy.signal import find_peaks

    O_peaks, _ = find_peaks(O_profile, height=0.1)
    N_peaks, _ = find_peaks(N_profile, height=0.1)

    # Alternative: find center of mass
    if O_profile.sum() > 0:
        O_center = np.average(depth, weights=O_profile)
    else:
        O_center = 0

    if N_profile.sum() > 0:
        N_center = np.average(depth, weights=N_profile)
    else:
        N_center = 0

    print(f"\nProfile analysis:")
    print(f"  O center of mass: {O_center:.1f} Å")
    print(f"  N center of mass: {N_center:.1f} Å")

    # Check for bimodal N distribution
    N_threshold = 0.1 * N_profile.max() if N_profile.max() > 0 else 0
    N_regions = np.where(N_profile > N_threshold)[0]

    if len(N_regions) > 0:
        # Check if N has two separated regions
        gaps = np.diff(N_regions)
        large_gaps = np.where(gaps > 3)[0]

        if len(large_gaps) > 0:
            pattern = "NON (bimodal N)"
            print(f"  N has {len(large_gaps)+1} separated regions")
        elif N_center < O_center:
            pattern = "NO (N above O)"
        else:
            pattern = "ON (O above N)"
    else:
        if O_profile.sum() > 0:
            pattern = "O only"
        else:
            pattern = "Unknown"

    print(f"  Detected pattern: {pattern}")
    return pattern


def test_non_pattern():
    """Test NON pattern reconstruction."""
    print("=" * 70)
    print("090317 Method Test: NON Pattern")
    print("Si3N4 (5Å) / SiO2 (15Å) / Si3N4 (5Å) / Si")
    print("=" * 70)

    # Setup
    step_size = 1.0
    depth = np.arange(0, 50, step_size)
    angles = np.array([0, 20, 35, 45, 55, 65, 75])

    # True profile
    D_true = create_non_profile(depth, n1_thick=5.0, o_thick=15.0, n2_thick=5.0)

    # Run optimization
    print("\n--- LLS Method ---")
    D_opt_lls, result_lls = run_090317_optimization(
        depth, angles, D_true, noise_level=0.02, method='LLS'
    )
    pattern_lls = analyze_pattern(depth, D_opt_lls)

    print("\n--- MEM Method ---")
    D_opt_mem, result_mem = run_090317_optimization(
        depth, angles, D_true, noise_level=0.02, method='MEM', alpha=0.0005
    )
    pattern_mem = analyze_pattern(depth, D_opt_mem)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # True profile
    ax = axes[0, 0]
    ax.plot(depth, D_true[:, 2] * 100, 'r-', linewidth=2, label='O (true)')
    ax.plot(depth, D_true[:, 3] * 100, 'b-', linewidth=2, label='N (true)')
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration (%)')
    ax.set_title('True Profile: NON')
    ax.legend()
    ax.set_xlim(0, 40)

    # LLS result
    ax = axes[0, 1]
    ax.plot(depth, D_true[:, 2] * 100, 'r--', linewidth=1, alpha=0.5, label='O (true)')
    ax.plot(depth, D_true[:, 3] * 100, 'b--', linewidth=1, alpha=0.5, label='N (true)')
    ax.plot(depth, D_opt_lls[:, 2] * 100, 'r-', linewidth=2, label='O (LLS)')
    ax.plot(depth, D_opt_lls[:, 3] * 100, 'b-', linewidth=2, label='N (LLS)')
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration (%)')
    ax.set_title(f'LLS Result: {pattern_lls}')
    ax.legend()
    ax.set_xlim(0, 40)

    # MEM result
    ax = axes[1, 0]
    ax.plot(depth, D_true[:, 2] * 100, 'r--', linewidth=1, alpha=0.5, label='O (true)')
    ax.plot(depth, D_true[:, 3] * 100, 'b--', linewidth=1, alpha=0.5, label='N (true)')
    ax.plot(depth, D_opt_mem[:, 2] * 100, 'r-', linewidth=2, label='O (MEM)')
    ax.plot(depth, D_opt_mem[:, 3] * 100, 'b-', linewidth=2, label='N (MEM)')
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration (%)')
    ax.set_title(f'MEM Result: {pattern_mem}')
    ax.legend()
    ax.set_xlim(0, 40)

    # Comparison
    ax = axes[1, 1]
    ax.plot(depth, (D_opt_lls[:, 3] - D_opt_lls[:, 2]) * 100, 'g-', linewidth=2, label='N-O (LLS)')
    ax.plot(depth, (D_opt_mem[:, 3] - D_opt_mem[:, 2]) * 100, 'm-', linewidth=2, label='N-O (MEM)')
    ax.plot(depth, (D_true[:, 3] - D_true[:, 2]) * 100, 'k--', linewidth=1, label='N-O (true)')
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('N - O (%)')
    ax.set_title('N-O Difference (NON should be positive at surface and interface)')
    ax.legend()
    ax.set_xlim(0, 40)

    plt.tight_layout()
    output_path = str(ensure_outputs_dir() / 'test_non_090317.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.show()

    return D_opt_lls, D_opt_mem


def test_ono_pattern():
    """Test ONO pattern reconstruction for comparison."""
    print("\n" + "=" * 70)
    print("090317 Method Test: ONO Pattern")
    print("SiO2 (5Å) / Si3N4 (15Å) / SiO2 (5Å) / Si")
    print("=" * 70)

    step_size = 1.0
    depth = np.arange(0, 50, step_size)
    angles = np.array([0, 20, 35, 45, 55, 65, 75])

    D_true = create_ono_profile(depth, o1_thick=5.0, n_thick=15.0, o2_thick=5.0)

    print("\n--- LLS Method ---")
    D_opt_lls, _ = run_090317_optimization(
        depth, angles, D_true, noise_level=0.02, method='LLS'
    )
    pattern_lls = analyze_pattern(depth, D_opt_lls)

    return D_opt_lls


if __name__ == "__main__":
    D_lls, D_mem = test_non_pattern()
    D_ono = test_ono_pattern()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The 090317 approach optimizes O and N INDEPENDENTLY,")
    print("which allows distinguishing NON from ONO patterns.")
    print("This is different from Run_DepthProfiling.m where O and N are LINKED.")
