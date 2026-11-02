"""
Numba-accelerated kernels for depth profile reconstruction.
============================================================

These functions extract the computational hotspots from inverse_solver.py
and compile them with Numba JIT for significant performance improvements.

The hot path is objective() together with _build_full_profile() and
_build_substrate_profile(). These kernels avoid Python overhead in the
inner loop, where minimize() calls objective() many times per solve.

Date: 2026-01-24
"""

import numpy as np
from math import erf, sqrt, log, exp

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: identity decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ============================================================================
# Variance-Stabilizing Transforms for Poisson Noise
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def anscombe_transform_numba(I: np.ndarray) -> np.ndarray:
    """
    Apply Anscombe variance-stabilizing transform for Poisson data.

    Transform: Y = 2 * sqrt(I + 3/8)

    This transforms Poisson-distributed data to approximately Gaussian
    with variance ≈ 1, enabling standard least-squares fitting.

    Args:
        I: Intensity array (any shape)

    Returns:
        Transformed array with stabilized variance
    """
    result = np.empty_like(I)
    for i in range(I.size):
        result.flat[i] = 2.0 * sqrt(I.flat[i] + 0.375)
    return result


@jit(nopython=True, cache=True, fastmath=True)
def anscombe_inverse_numba(Y: np.ndarray) -> np.ndarray:
    """
    Inverse Anscombe transform.

    Inverse: I = (Y/2)^2 - 3/8

    Args:
        Y: Transformed intensity array

    Returns:
        Original-scale intensity array
    """
    result = np.empty_like(Y)
    for i in range(Y.size):
        val = Y.flat[i] / 2.0
        result.flat[i] = val * val - 0.375
    return result


def anscombe_transform(I: np.ndarray) -> np.ndarray:
    """NumPy version of Anscombe transform."""
    return 2.0 * np.sqrt(I + 0.375)


def anscombe_inverse(Y: np.ndarray) -> np.ndarray:
    """NumPy version of inverse Anscombe transform."""
    return (Y / 2.0) ** 2 - 0.375


# ============================================================================
# Core Numba Kernels
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def build_substrate_profile_numba(
    depth: np.ndarray,
    thickness: float,
    abruptness: float,
    substrate_ratios: np.ndarray,
) -> np.ndarray:
    """
    Build substrate profile using error function.

    D_sub(z, e) = ratio_e × 0.5 × (1 + erf((z - thickness) / sigma))

    Args:
        depth: Depth array (n_depth,)
        thickness: Film thickness (Å)
        abruptness: Interface abruptness FWHM (Å)
        substrate_ratios: Concentration ratios for substrate elements (n_sub,)

    Returns:
        Substrate profile (n_depth, n_sub)
    """
    n_depth = len(depth)
    n_sub = len(substrate_ratios)

    # Convert FWHM to sigma
    sigma = abruptness / (2.0 * sqrt(log(2.0)))

    # Output array
    D_sub = np.zeros((n_depth, n_sub))

    # Compute erf profile and apply ratios
    for i in range(n_depth):
        erf_val = 0.5 * (1.0 + erf((depth[i] - thickness) / sigma))
        for j in range(n_sub):
            D_sub[i, j] = erf_val * substrate_ratios[j]

    return D_sub


@jit(nopython=True, cache=True, fastmath=True)
def build_full_profile_numba(
    free_conc: np.ndarray,
    thickness: float,
    abruptness: float,
    substrate_ratios: np.ndarray,
    depth: np.ndarray,
    n_elements: int,
    free_idx: np.ndarray,
    sub_idx: np.ndarray,
    linked_idx: np.ndarray,
    free_linked_const: np.ndarray,
    min_conc: float,
    max_conc: float,
) -> np.ndarray:
    """
    Build full depth profile from optimized parameters.

    1. D_substrate = ratio × erf(...)
    2. D_free = normalized_conc × (1 - Σ D_substrate)
    3. D_linked = Σ (D_free × (freelinkedconst - 1))

    Args:
        free_conc: Free element concentrations (n_depth, n_free)
        thickness: Film thickness (Å)
        abruptness: Interface abruptness (Å)
        substrate_ratios: Substrate element ratios (n_sub,)
        depth: Depth array (n_depth,)
        n_elements: Total number of elements
        free_idx: Indices of free elements
        sub_idx: Indices of substrate elements
        linked_idx: Indices of linked elements
        free_linked_const: Charge-linked constraints
        min_conc: Minimum concentration
        max_conc: Maximum concentration

    Returns:
        Full depth profile (n_depth, n_elements)
    """
    n_depth = len(depth)
    n_free = len(free_idx)
    n_sub = len(sub_idx)
    n_linked = len(linked_idx)

    D = np.zeros((n_depth, n_elements))

    # 1. Substrate profile
    D_sub = build_substrate_profile_numba(depth, thickness, abruptness, substrate_ratios)

    # Copy substrate to D
    for i in range(n_depth):
        for j in range(n_sub):
            D[i, sub_idx[j]] = D_sub[i, j]

    # 2. Free element concentrations scaled by (1 - substrate_sum)
    for i in range(n_depth):
        substrate_sum = 0.0
        for j in range(n_sub):
            substrate_sum += D_sub[i, j]

        film_scale = max(1.0 - substrate_sum, min_conc)

        for j in range(n_free):
            D[i, free_idx[j]] = free_conc[i, j] * film_scale

    # 3. Linked element concentrations from charge balance
    if n_linked > 0:
        for i in range(n_depth):
            D_linked = 0.0
            for j in range(n_free):
                coeff = free_linked_const[j] - 1.0
                D_linked += D[i, free_idx[j]] * coeff

            # Distribute equally among linked elements
            for k in range(n_linked):
                D[i, linked_idx[k]] = D_linked / n_linked

    # Clip to valid range
    for i in range(n_depth):
        for j in range(n_elements):
            if D[i, j] < min_conc:
                D[i, j] = min_conc
            elif D[i, j] > max_conc:
                D[i, j] = max_conc

    return D


@jit(nopython=True, cache=True, fastmath=True)
def compute_forward_model_numba(
    D: np.ndarray,
    T_matrices: np.ndarray,  # (n_elements, n_angles, n_depth)
    n_angles: int,
    n_elements: int,
) -> np.ndarray:
    """
    Compute forward model: I_pred = T @ D for each element.

    Args:
        D: Depth profile (n_depth, n_elements)
        T_matrices: Transfer matrices stacked (n_elements, n_angles, n_depth)
        n_angles: Number of angles
        n_elements: Number of elements

    Returns:
        Predicted intensities (n_angles, n_elements)
    """
    n_depth = D.shape[0]
    I_pred = np.zeros((n_angles, n_elements))

    for e in range(n_elements):
        for a in range(n_angles):
            val = 0.0
            for z in range(n_depth):
                val += T_matrices[e, a, z] * D[z, e]
            I_pred[a, e] = val

    return I_pred


@jit(nopython=True, cache=True, fastmath=True)
def compute_chi2_numba(
    I_obs_norm: np.ndarray,
    I_pred_norm: np.ndarray,
    antisubst_idx: np.ndarray,
    degree: float,
) -> float:
    """
    Compute chi-squared error (ordinary least squares, uniform weights).

    Args:
        I_obs_norm: Normalized observed intensities (n_angles, n_elements)
        I_pred_norm: Normalized predicted intensities (n_angles, n_elements)
        antisubst_idx: Indices of free + linked elements
        degree: Degree of freedom scaling

    Returns:
        Chi-squared value
    """
    n_angles = I_obs_norm.shape[0]
    n_antisubst = len(antisubst_idx)

    chi2 = 0.0
    for a in range(n_angles):
        for i in range(n_antisubst):
            idx = antisubst_idx[i]
            diff = I_obs_norm[a, idx] - I_pred_norm[a, idx]
            chi2 += diff * diff

    return degree * chi2


@jit(nopython=True, cache=True, fastmath=True)
def compute_chi2_wls_numba(
    I_obs_norm: np.ndarray,
    I_pred_norm: np.ndarray,
    antisubst_idx: np.ndarray,
    degree: float,
    weights: np.ndarray,
) -> float:
    """
    Compute weighted chi-squared error for WLS (Weighted Least Squares).

    Args:
        I_obs_norm: Normalized observed intensities (n_angles, n_elements)
        I_pred_norm: Normalized predicted intensities (n_angles, n_elements)
        antisubst_idx: Indices of free + linked elements
        degree: Degree of freedom scaling
        weights: Weight matrix (n_angles, n_elements) for WLS.
                 For Poisson noise: weight = 1 / I_obs (variance-inverse weighting)

    Returns:
        Weighted chi-squared value
    """
    n_angles = I_obs_norm.shape[0]
    n_antisubst = len(antisubst_idx)

    chi2 = 0.0
    for a in range(n_angles):
        for i in range(n_antisubst):
            idx = antisubst_idx[i]
            diff = I_obs_norm[a, idx] - I_pred_norm[a, idx]
            chi2 += weights[a, idx] * diff * diff

    return degree * chi2


@jit(nopython=True, cache=True, fastmath=True)
def compute_poisson_nll_numba(
    I_obs_norm: np.ndarray,
    I_pred_norm: np.ndarray,
    antisubst_idx: np.ndarray,
    degree: float,
) -> float:
    """
    Compute Poisson deviance (scaled negative log-likelihood).

    The Poisson deviance is:
        D = 2 * Σ [I_obs * log(I_obs/I_pred) - (I_obs - I_pred)]

    This is better scaled than raw NLL and approaches Chi² when I_pred ≈ I_obs.
    The deviance is on the same scale as Chi² (squared errors).

    Args:
        I_obs_norm: Normalized observed intensities (n_angles, n_elements)
        I_pred_norm: Normalized predicted intensities (n_angles, n_elements)
        antisubst_idx: Indices of free + linked elements
        degree: Scaling factor

    Returns:
        Poisson deviance value (scaled like Chi²)
    """
    n_angles = I_obs_norm.shape[0]
    n_antisubst = len(antisubst_idx)

    deviance = 0.0
    for a in range(n_angles):
        for i in range(n_antisubst):
            idx = antisubst_idx[i]
            I_obs = I_obs_norm[a, idx]
            I_pred = I_pred_norm[a, idx]

            # Ensure positive values for log
            I_obs_safe = max(I_obs, 1e-10)
            I_pred_safe = max(I_pred, 1e-10)

            # Poisson deviance: 2 * [I_obs * log(I_obs/I_pred) - (I_obs - I_pred)]
            deviance += 2.0 * (I_obs_safe * log(I_obs_safe / I_pred_safe) - (I_obs - I_pred))

    return degree * deviance


@jit(nopython=True, cache=True, fastmath=True)
def compute_regularization_l2_numba(
    x: np.ndarray,
    m0: np.ndarray,
    film_size: int,
    thickness: float,
    abruptness: float,
    m0_thickness: float,
    m0_abruptness: float,
) -> float:
    """Compute L2 regularization term."""
    reg = 0.0
    for i in range(film_size):
        diff = x[i] - m0[i]
        reg += diff * diff

    # Thickness and abruptness regularization
    reg += 100.0 * ((thickness - m0_thickness) / 100.0) ** 2
    reg += 100.0 * ((abruptness - m0_abruptness) / 10.0) ** 2

    return reg


@jit(nopython=True, cache=True, fastmath=True)
def compute_regularization_l1_numba(
    x: np.ndarray,
    m0: np.ndarray,
    film_size: int,
    thickness: float,
    m0_thickness: float,
) -> float:
    """Compute L1 regularization term."""
    reg = 0.0
    for i in range(film_size):
        reg += abs(x[i] - m0[i])

    reg += abs(thickness - m0_thickness) / 100.0

    return reg


@jit(nopython=True, cache=True, fastmath=True)
def compute_regularization_l1_logcosh_numba(
    x: np.ndarray,
    m0: np.ndarray,
    film_size: int,
    thickness: float,
    m0_thickness: float,
    beta: float = 0.01,
) -> float:
    """Compute log-cosh smooth L1 regularization: β·Σlog(cosh((x-m0)/β))."""
    reg = 0.0
    for i in range(film_size):
        d = (x[i] - m0[i]) / beta
        # Numerically stable log(cosh(d)) = |d| + log(1 + exp(-2|d|)) - log(2)
        ad = abs(d)
        if ad > 20.0:
            reg += beta * ad  # Asymptotic: β·|d| = |x-m0|
        else:
            reg += beta * (ad + np.log1p(np.exp(-2.0 * ad)) - 0.6931471805599453)

    d_t = (thickness - m0_thickness) / (100.0 * beta)
    ad_t = abs(d_t)
    if ad_t > 20.0:
        reg += beta * ad_t
    else:
        reg += beta * (ad_t + np.log1p(np.exp(-2.0 * ad_t)) - 0.6931471805599453)

    return reg


@jit(nopython=True, cache=True, fastmath=True)
def compute_regularization_mem_numba(
    x: np.ndarray,
    m0: np.ndarray,
    film_size: int,
    thickness: float,
    m0_thickness: float,
) -> float:
    """Compute Maximum Entropy regularization (Bregman divergence).

    D_φ(x‖m0) = Σ(x·log(x/m0) - x + m0) ≥ 0 always, φ(x)=x·log(x).
    MATLAB equivalent: -S where S = Σ m0·(eˣ-1-x·eˣ) ≤ 0.
    """
    reg = 0.0
    for i in range(film_size):
        x_safe = max(x[i], 1e-10)
        m0_safe = max(m0[i], 1e-10)
        reg += x_safe * log(x_safe / m0_safe) - x_safe + m0_safe

    # Thickness regularization
    t_safe = max(thickness, 1e-10)
    m0_t_safe = max(m0_thickness, 1e-10)
    reg += (t_safe * log(t_safe / m0_t_safe) - t_safe + m0_t_safe) / 10.0

    return reg


@jit(nopython=True, cache=True, fastmath=True)
def compute_sum_penalty_numba(D: np.ndarray) -> float:
    """Compute concentration sum penalty."""
    n_depth = D.shape[0]
    n_elements = D.shape[1]

    penalty = 0.0
    for i in range(n_depth):
        row_sum = 0.0
        for j in range(n_elements):
            row_sum += D[i, j]
        deviation = row_sum - 1.0
        penalty += deviation * deviation

    return 100.0 * penalty


@jit(nopython=True, cache=True, fastmath=True)
def normalize_by_substrate_numba(
    I: np.ndarray,
    sub_idx: np.ndarray,
) -> np.ndarray:
    """Normalize intensities by substrate sum."""
    n_angles = I.shape[0]
    n_elements = I.shape[1]
    n_sub = len(sub_idx)

    I_norm = np.zeros_like(I)

    for a in range(n_angles):
        # Compute substrate sum
        sub_sum = 0.0
        for j in range(n_sub):
            sub_sum += I[a, sub_idx[j]]

        sub_sum = max(sub_sum, 1e-10)

        # Normalize
        for e in range(n_elements):
            I_norm[a, e] = I[a, e] / sub_sum

    return I_norm


# ============================================================================
# Combined Objective Function Kernel
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def objective_kernel_numba(
    x: np.ndarray,
    m0: np.ndarray,
    I_obs_norm: np.ndarray,
    T_matrices: np.ndarray,
    depth: np.ndarray,
    n_elements: int,
    n_angles: int,
    free_idx: np.ndarray,
    sub_idx: np.ndarray,
    linked_idx: np.ndarray,
    free_linked_const: np.ndarray,
    antisubst_idx: np.ndarray,
    min_conc: float,
    max_conc: float,
    degree: float,
    alpha: float,
    method: int,  # 0=LLS, 1=L2, 2=L1, 3=MEM
    use_sum_constraint: bool,
) -> float:
    """
    Complete objective function kernel (standard OLS).

    This is the main bottleneck function, called many times per solve.
    By JIT-compiling the entire objective, we avoid Python overhead.
    """
    n_depth = len(depth)
    n_free = len(free_idx)
    n_sub = len(sub_idx)

    # Unpack parameters
    film_size = n_depth * n_free

    # Film concentrations (reshape from flat)
    free_conc = np.zeros((n_depth, n_free))
    for i in range(n_depth):
        for j in range(n_free):
            free_conc[i, j] = x[i * n_free + j]

    # Structure parameters
    thickness = x[film_size]
    abruptness = x[film_size + 1]

    # Substrate ratios
    substrate_ratios = np.zeros(n_sub)
    for j in range(n_sub):
        substrate_ratios[j] = x[film_size + 2 + j]

    # Build full profile
    D = build_full_profile_numba(
        free_conc, thickness, abruptness, substrate_ratios,
        depth, n_elements, free_idx, sub_idx, linked_idx,
        free_linked_const, min_conc, max_conc
    )

    # Forward model
    I_pred = compute_forward_model_numba(D, T_matrices, n_angles, n_elements)

    # Normalize by substrate
    I_pred_norm = normalize_by_substrate_numba(I_pred, sub_idx)

    # Chi-squared (standard OLS)
    chi2 = compute_chi2_numba(I_obs_norm, I_pred_norm, antisubst_idx, degree)

    # Regularization
    m0_thickness = m0[film_size]
    m0_abruptness = m0[film_size + 1]

    if method == 1:  # L2
        reg = compute_regularization_l2_numba(
            x, m0, film_size, thickness, abruptness, m0_thickness, m0_abruptness
        )
    elif method == 2:  # L1
        reg = compute_regularization_l1_numba(x, m0, film_size, thickness, m0_thickness)
    elif method == 3:  # MEM
        reg = compute_regularization_mem_numba(x, m0, film_size, thickness, m0_thickness)
    elif method == 4:  # L1 log-cosh
        reg = compute_regularization_l1_logcosh_numba(x, m0, film_size, thickness, m0_thickness)
    else:  # LLS
        reg = 0.0

    # Sum constraint penalty
    if use_sum_constraint:
        sum_penalty = compute_sum_penalty_numba(D)
    else:
        sum_penalty = 0.0

    return 0.5 * chi2 + alpha * reg + sum_penalty


@jit(nopython=True, cache=True, fastmath=True)
def objective_kernel_wls_numba(
    x: np.ndarray,
    m0: np.ndarray,
    I_obs_norm: np.ndarray,
    T_matrices: np.ndarray,
    depth: np.ndarray,
    n_elements: int,
    n_angles: int,
    free_idx: np.ndarray,
    sub_idx: np.ndarray,
    linked_idx: np.ndarray,
    free_linked_const: np.ndarray,
    antisubst_idx: np.ndarray,
    min_conc: float,
    max_conc: float,
    degree: float,
    alpha: float,
    method: int,  # 0=LLS, 1=L2, 2=L1, 3=MEM
    use_sum_constraint: bool,
    weights: np.ndarray,
) -> float:
    """
    Complete objective function kernel with WLS (Weighted Least Squares).

    Same as objective_kernel_numba but uses weighted chi-squared for Poisson noise.

    Args:
        weights: Weight matrix (n_angles, n_elements) for WLS.
                 For Poisson noise: weight = 1 / I_obs
    """
    n_depth = len(depth)
    n_free = len(free_idx)
    n_sub = len(sub_idx)

    # Unpack parameters
    film_size = n_depth * n_free

    # Film concentrations (reshape from flat)
    free_conc = np.zeros((n_depth, n_free))
    for i in range(n_depth):
        for j in range(n_free):
            free_conc[i, j] = x[i * n_free + j]

    # Structure parameters
    thickness = x[film_size]
    abruptness = x[film_size + 1]

    # Substrate ratios
    substrate_ratios = np.zeros(n_sub)
    for j in range(n_sub):
        substrate_ratios[j] = x[film_size + 2 + j]

    # Build full profile
    D = build_full_profile_numba(
        free_conc, thickness, abruptness, substrate_ratios,
        depth, n_elements, free_idx, sub_idx, linked_idx,
        free_linked_const, min_conc, max_conc
    )

    # Forward model
    I_pred = compute_forward_model_numba(D, T_matrices, n_angles, n_elements)

    # Normalize by substrate
    I_pred_norm = normalize_by_substrate_numba(I_pred, sub_idx)

    # Chi-squared (WLS - weighted)
    chi2 = compute_chi2_wls_numba(I_obs_norm, I_pred_norm, antisubst_idx, degree, weights)

    # Regularization
    m0_thickness = m0[film_size]
    m0_abruptness = m0[film_size + 1]

    if method == 1:  # L2
        reg = compute_regularization_l2_numba(
            x, m0, film_size, thickness, abruptness, m0_thickness, m0_abruptness
        )
    elif method == 2:  # L1
        reg = compute_regularization_l1_numba(x, m0, film_size, thickness, m0_thickness)
    elif method == 3:  # MEM
        reg = compute_regularization_mem_numba(x, m0, film_size, thickness, m0_thickness)
    elif method == 4:  # L1 log-cosh
        reg = compute_regularization_l1_logcosh_numba(x, m0, film_size, thickness, m0_thickness)
    else:  # LLS
        reg = 0.0

    # Sum constraint penalty
    if use_sum_constraint:
        sum_penalty = compute_sum_penalty_numba(D)
    else:
        sum_penalty = 0.0

    return 0.5 * chi2 + alpha * reg + sum_penalty


# ============================================================================
# Utility Functions
# ============================================================================

def is_numba_available() -> bool:
    """Check if Numba is available."""
    return NUMBA_AVAILABLE


# ============================================================================
# Analytical Jacobian (Gradient) Kernels
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def compute_d_erf_d_thickness_numba(
    depth: np.ndarray,
    thickness: float,
    sigma: float,
) -> np.ndarray:
    """
    Compute derivative of erf profile with respect to thickness.

    erf_val = 0.5 * (1 + erf((z - t) / sigma))
    d(erf_val)/dt = -1/sigma * 1/sqrt(pi) * exp(-((z-t)/sigma)^2)
    """
    n_depth = len(depth)
    d_erf = np.zeros(n_depth)

    inv_sqrt_pi = 1.0 / sqrt(3.141592653589793)
    inv_sigma = 1.0 / sigma

    for i in range(n_depth):
        arg = (depth[i] - thickness) / sigma
        d_erf[i] = -inv_sigma * inv_sqrt_pi * exp(-arg * arg)

    return d_erf


@jit(nopython=True, cache=True, fastmath=True)
def compute_d_erf_d_abruptness_numba(
    depth: np.ndarray,
    thickness: float,
    abruptness: float,
) -> np.ndarray:
    """
    Compute derivative of erf profile with respect to abruptness.

    sigma = abruptness / (2 * sqrt(ln2))
    d_sigma/d_abruptness = 1 / (2 * sqrt(ln2))

    erf_val = 0.5 * (1 + erf((z - t) / sigma))
    d(erf_val)/d_sigma = -(z-t)/sigma^2 * 1/sqrt(pi) * exp(-((z-t)/sigma)^2)
    """
    n_depth = len(depth)
    d_erf = np.zeros(n_depth)

    sigma = abruptness / (2.0 * sqrt(log(2.0)))
    d_sigma_d_abrupt = 1.0 / (2.0 * sqrt(log(2.0)))

    inv_sqrt_pi = 1.0 / sqrt(3.141592653589793)
    sigma_sq = sigma * sigma

    for i in range(n_depth):
        arg = (depth[i] - thickness) / sigma
        d_erf_d_sigma = -(depth[i] - thickness) / sigma_sq * inv_sqrt_pi * exp(-arg * arg)
        d_erf[i] = d_erf_d_sigma * d_sigma_d_abrupt

    return d_erf


@jit(nopython=True, cache=True, fastmath=True)
def gradient_kernel_numba(
    x: np.ndarray,
    m0: np.ndarray,
    I_obs_norm: np.ndarray,
    T_matrices: np.ndarray,
    depth: np.ndarray,
    n_elements: int,
    n_angles: int,
    free_idx: np.ndarray,
    sub_idx: np.ndarray,
    linked_idx: np.ndarray,
    free_linked_const: np.ndarray,
    antisubst_idx: np.ndarray,
    min_conc: float,
    max_conc: float,
    degree: float,
    alpha: float,
    method: int,  # 0=LLS, 1=L2, 2=L1, 3=MEM
    use_sum_constraint: bool,
    xreg_weights: np.ndarray = np.empty(0, dtype=np.float64),
) -> np.ndarray:
    """
    Compute analytical gradient of the objective function (standard OLS).

    This provides exact gradients instead of finite differences, which
    saves objective evaluations in the SLSQP optimizer.

    Args:
        Same as objective_kernel_numba

    Returns:
        Gradient vector (same size as x)
    """
    n_depth = len(depth)
    n_free = len(free_idx)
    n_sub = len(sub_idx)
    n_linked = len(linked_idx)
    n_antisubst = len(antisubst_idx)

    # Unpack parameters
    film_size = n_depth * n_free
    total_size = film_size + 2 + n_sub

    grad = np.zeros(total_size)

    # Film concentrations (reshape from flat)
    free_conc = np.zeros((n_depth, n_free))
    for i in range(n_depth):
        for j in range(n_free):
            free_conc[i, j] = x[i * n_free + j]

    # Structure parameters
    thickness = x[film_size]
    abruptness = x[film_size + 1]

    # Substrate ratios
    substrate_ratios = np.zeros(n_sub)
    for j in range(n_sub):
        substrate_ratios[j] = x[film_size + 2 + j]

    # Build full profile (forward pass)
    sigma = abruptness / (2.0 * sqrt(log(2.0)))

    # Substrate profile and its erf values
    erf_vals = np.zeros(n_depth)
    for i in range(n_depth):
        erf_vals[i] = 0.5 * (1.0 + erf((depth[i] - thickness) / sigma))

    # D_sub[i, j] = erf_vals[i] * substrate_ratios[j]
    D_sub = np.zeros((n_depth, n_sub))
    for i in range(n_depth):
        for j in range(n_sub):
            D_sub[i, j] = erf_vals[i] * substrate_ratios[j]

    # Film scale (1 - substrate_sum) with clipping
    # Track whether film_scale is clipped for gradient computation
    film_scale = np.zeros(n_depth)
    film_scale_clipped = np.zeros(n_depth, dtype=np.int64)  # 1 if clipped, 0 otherwise
    for i in range(n_depth):
        substrate_sum = 0.0
        for j in range(n_sub):
            substrate_sum += D_sub[i, j]
        unclipped = 1.0 - substrate_sum
        if unclipped < min_conc:
            film_scale[i] = min_conc
            film_scale_clipped[i] = 1
        else:
            film_scale[i] = unclipped
            film_scale_clipped[i] = 0

    # Build full D matrix
    D = np.zeros((n_depth, n_elements))

    # Substrate elements
    for i in range(n_depth):
        for j in range(n_sub):
            D[i, sub_idx[j]] = D_sub[i, j]

    # Free elements
    for i in range(n_depth):
        for j in range(n_free):
            D[i, free_idx[j]] = free_conc[i, j] * film_scale[i]

    # Linked elements (if any)
    if n_linked > 0:
        for i in range(n_depth):
            D_linked_val = 0.0
            for j in range(n_free):
                coeff = free_linked_const[j] - 1.0
                D_linked_val += D[i, free_idx[j]] * coeff
            for k in range(n_linked):
                D[i, linked_idx[k]] = D_linked_val / n_linked

    # Forward model: I_pred = T @ D
    I_pred = np.zeros((n_angles, n_elements))
    for e in range(n_elements):
        for a in range(n_angles):
            val = 0.0
            for z in range(n_depth):
                val += T_matrices[e, a, z] * D[z, e]
            I_pred[a, e] = val

    # Normalize by substrate sum
    I_sub_sum = np.zeros(n_angles)
    for a in range(n_angles):
        for j in range(n_sub):
            I_sub_sum[a] += I_pred[a, sub_idx[j]]
        I_sub_sum[a] = max(I_sub_sum[a], 1e-10)

    I_pred_norm = np.zeros((n_angles, n_elements))
    for a in range(n_angles):
        for e in range(n_elements):
            I_pred_norm[a, e] = I_pred[a, e] / I_sub_sum[a]

    # ========================================
    # Backward pass: compute gradients
    # ========================================

    # Loss = 0.5 * chi² where chi² = degree * Σ_{a,e in antisubst} r²
    # r[a,e] = I_obs_norm[a,e] - I_pred_norm[a,e]
    #
    # I_pred_norm[a,e] = I_pred[a,e] / S  where S = Σ_{j in sub} I_pred[a, sub_j]
    #
    # Gradient via chain rule:
    # dL/dI_pred[a,k] = Σ_e dL/dI_pred_norm[a,e] * dI_pred_norm[a,e]/dI_pred[a,k]
    #
    # For the quotient I_pred_norm[a,e] = I_pred[a,e] / S:
    # - If k is NOT a substrate element:
    #   dI_pred_norm[a,e]/dI_pred[a,k] = δ(e,k) / S
    # - If k IS a substrate element:
    #   dI_pred_norm[a,e]/dI_pred[a,k] = δ(e,k) / S - I_pred[a,e] / S²
    #     (since S depends on I_pred[a,k] when k is substrate)

    # Gradient w.r.t. I_pred (before normalization)
    dL_dI_pred = np.zeros((n_angles, n_elements))

    # Precompute Xreg weight² per antisubst element
    has_xreg = xreg_weights.size > 0
    w2 = np.ones(n_antisubst)
    if has_xreg:
        for i in range(n_antisubst):
            w2[i] = xreg_weights[i] * xreg_weights[i]

    for a in range(n_angles):
        S = I_sub_sum[a]
        S_sq = S * S

        # dL/dI_pred_norm[a,e] = -degree * r * w²[e]
        # With Xreg: Chi² = degree * Σ (r * w)², so dChi²/dr = 2 * degree * r * w²
        # Loss = 0.5 * Chi², so dL/dr = degree * r * w²
        for i in range(n_antisubst):
            e = antisubst_idx[i]
            r = I_obs_norm[a, e] - I_pred_norm[a, e]
            dL_dI_norm_e = -degree * r * w2[i]

            # Direct contribution: dI_pred_norm[a,e]/dI_pred[a,e] = 1/S
            dL_dI_pred[a, e] += dL_dI_norm_e / S

        # Indirect contribution from ALL antisubst elements via substrate normalization
        # When I_pred[a, sub_k] changes, ALL I_pred_norm[a,*] change via S
        for j in range(n_sub):
            sub_k = sub_idx[j]
            grad_sub_k = 0.0

            for i in range(n_antisubst):
                e = antisubst_idx[i]
                r = I_obs_norm[a, e] - I_pred_norm[a, e]
                dL_dI_norm_e = -degree * r * w2[i]

                # dI_pred_norm[a,e]/dI_pred[a,sub_k] = -I_pred[a,e] / S²
                grad_sub_k += dL_dI_norm_e * (-I_pred[a, e] / S_sq)

            dL_dI_pred[a, sub_k] += grad_sub_k

    # Gradient w.r.t. D
    # I_pred[a,e] = Σ_z T[e,a,z] * D[z,e]
    # dI_pred[a,e]/dD[z,e] = T[e,a,z]
    dL_dD = np.zeros((n_depth, n_elements))
    for e in range(n_elements):
        for z in range(n_depth):
            val = 0.0
            for a in range(n_angles):
                val += dL_dI_pred[a, e] * T_matrices[e, a, z]
            dL_dD[z, e] = val

    # ========================================
    # Gradient w.r.t. optimization variables
    # ========================================

    # 1. Gradient w.r.t. free_conc (film concentrations)
    # D[z, free_e] = free_conc[z, j] * film_scale[z]
    # Also linked elements depend on free elements

    for j in range(n_free):
        free_e = free_idx[j]
        coeff = free_linked_const[j] - 1.0

        for z in range(n_depth):
            # Direct contribution
            grad_val = dL_dD[z, free_e] * film_scale[z]

            # Linked element contribution
            if n_linked > 0:
                for k in range(n_linked):
                    linked_e = linked_idx[k]
                    # D[z, linked_e] = Σ D[z, free_e] * coeff / n_linked
                    grad_val += dL_dD[z, linked_e] * film_scale[z] * coeff / n_linked

            # Pack into gradient vector (row-major order, like NumPy reshape)
            # Note: the 0.5 factor from f = 0.5 * chi² is already in dL_dI_norm
            grad[z * n_free + j] = grad_val

    # 2. Gradient w.r.t. thickness
    # D_sub[z, j] = erf_vals[z] * substrate_ratios[j]
    # film_scale[z] = max(1 - Σ D_sub[z, :], min_conc)
    # When film_scale is clipped, d(film_scale)/d(thickness) = 0

    d_erf_d_t = compute_d_erf_d_thickness_numba(depth, thickness, sigma)

    grad_thickness = 0.0
    for z in range(n_depth):
        # d(film_scale)/d(thickness) = -Σ_j substrate_ratios[j] * d_erf/d_t
        # BUT only if film_scale is not clipped
        if film_scale_clipped[z] == 0:
            d_film_scale = 0.0
            for j in range(n_sub):
                d_film_scale -= substrate_ratios[j] * d_erf_d_t[z]

            # Contribution from free elements via film_scale
            for jf in range(n_free):
                free_e = free_idx[jf]
                grad_thickness += dL_dD[z, free_e] * free_conc[z, jf] * d_film_scale

                # Linked contribution
                if n_linked > 0:
                    coeff = free_linked_const[jf] - 1.0
                    for k in range(n_linked):
                        linked_e = linked_idx[k]
                        grad_thickness += dL_dD[z, linked_e] * free_conc[z, jf] * d_film_scale * coeff / n_linked

        # Contribution from substrate elements (always applies)
        for j in range(n_sub):
            sub_e = sub_idx[j]
            grad_thickness += dL_dD[z, sub_e] * substrate_ratios[j] * d_erf_d_t[z]

    grad[film_size] = grad_thickness

    # 3. Gradient w.r.t. abruptness
    d_erf_d_a = compute_d_erf_d_abruptness_numba(depth, thickness, abruptness)

    grad_abruptness = 0.0
    for z in range(n_depth):
        # Only apply film_scale gradient if not clipped
        if film_scale_clipped[z] == 0:
            d_film_scale = 0.0
            for j in range(n_sub):
                d_film_scale -= substrate_ratios[j] * d_erf_d_a[z]

            for jf in range(n_free):
                free_e = free_idx[jf]
                grad_abruptness += dL_dD[z, free_e] * free_conc[z, jf] * d_film_scale

                if n_linked > 0:
                    coeff = free_linked_const[jf] - 1.0
                    for k in range(n_linked):
                        linked_e = linked_idx[k]
                        grad_abruptness += dL_dD[z, linked_e] * free_conc[z, jf] * d_film_scale * coeff / n_linked

        for j in range(n_sub):
            sub_e = sub_idx[j]
            grad_abruptness += dL_dD[z, sub_e] * substrate_ratios[j] * d_erf_d_a[z]

    grad[film_size + 1] = grad_abruptness

    # 4. Gradient w.r.t. substrate_ratios
    for j in range(n_sub):
        sub_e = sub_idx[j]
        grad_ratio = 0.0

        for z in range(n_depth):
            # D_sub[z, j] = erf_vals[z] * substrate_ratios[j]
            # Direct contribution (always applies)
            grad_ratio += dL_dD[z, sub_e] * erf_vals[z]

            # Film_scale contribution (only if not clipped)
            if film_scale_clipped[z] == 0:
                d_film_scale = -erf_vals[z]

                for jf in range(n_free):
                    free_e = free_idx[jf]
                    grad_ratio += dL_dD[z, free_e] * free_conc[z, jf] * d_film_scale

                    if n_linked > 0:
                        coeff = free_linked_const[jf] - 1.0
                        for k in range(n_linked):
                            linked_e = linked_idx[k]
                            grad_ratio += dL_dD[z, linked_e] * free_conc[z, jf] * d_film_scale * coeff / n_linked

        grad[film_size + 2 + j] = grad_ratio

    # ========================================
    # Regularization gradients
    # ========================================

    m0_thickness = m0[film_size]
    m0_abruptness = m0[film_size + 1]

    if method == 1:  # L2
        # d/dx_i [ Σ (x_i - m0_i)² ] = 2(x_i - m0_i)
        for i in range(film_size):
            grad[i] += alpha * 2.0 * (x[i] - m0[i])

        # Thickness regularization: 100 * ((t - m0_t) / 100)²
        # d/dt = 2 * 100 * (t - m0_t) / 100² = 0.02 * (t - m0_t)
        grad[film_size] += alpha * 0.02 * (thickness - m0_thickness)

        # Abruptness regularization: 100 * ((a - m0_a) / 10)²
        # d/da = 2 * 100 * (a - m0_a) / 10² = 2 * (a - m0_a)
        grad[film_size + 1] += alpha * 2.0 * (abruptness - m0_abruptness)

    elif method == 2:  # L1 — smoothed subgradient (Huber-like)
        # d/dx_i |x_i - m0_i| ≈ (x_i - m0_i) / max(|x_i - m0_i|, delta)
        # Avoids sharp kinks that confuse SLSQP BFGS Hessian approximation
        _l1_delta = 1e-3
        for i in range(film_size):
            diff = x[i] - m0[i]
            ad = abs(diff)
            grad[i] += alpha * diff / max(ad, _l1_delta)

        diff_t = thickness - m0_thickness
        ad_t = abs(diff_t)
        grad[film_size] += alpha * 0.01 * diff_t / max(ad_t, _l1_delta)

    elif method == 3:  # MEM — d/dx[Σ(x·log(x/m0) - x + m0)] = log(x/m0)
        for i in range(film_size):
            x_safe = max(x[i], 1e-10)
            m0_safe = max(m0[i], 1e-10)
            grad[i] += alpha * log(x_safe / m0_safe)

        t_safe = max(thickness, 1e-10)
        m0_t_safe = max(m0_thickness, 1e-10)
        grad[film_size] += alpha * log(t_safe / m0_t_safe) / 10.0

    elif method == 4:  # L1 log-cosh: grad = tanh((x-m0)/β)
        _l1_beta = 0.01
        for i in range(film_size):
            d = (x[i] - m0[i]) / _l1_beta
            grad[i] += alpha * np.tanh(d)
        d_t = (thickness - m0_thickness) / (100.0 * _l1_beta)
        grad[film_size] += alpha * np.tanh(d_t) * 0.01

    # ========================================
    # Sum constraint penalty gradient
    # ========================================

    if use_sum_constraint:
        # penalty = 100 * Σ_z (D.sum(axis=1)[z] - 1)²
        # d(penalty)/d(D[z,e]) = 200 * (D.sum(axis=1)[z] - 1) for each e
        #
        # D[z, free_e] = free_conc[z, jf] * film_scale[z]
        # D[z, sub_e] = erf_vals[z] * substrate_ratios[j]
        # D[z, linked_e] = Σ_jf D[z, free_e] * (coeff[jf] - 1) / n_linked
        #
        # The chain rule gives us:
        # d(penalty)/d(x) = Σ_z,e d(penalty)/d(D[z,e]) * d(D[z,e])/d(x)

        # Pre-compute d(sum)/d(erf) for each depth
        # This is used for thickness/abruptness gradients
        d_erf_t = compute_d_erf_d_thickness_numba(depth, thickness, sigma)
        d_erf_a = compute_d_erf_d_abruptness_numba(depth, thickness, abruptness)

        grad_pen_thickness = 0.0
        grad_pen_abruptness = 0.0
        grad_pen_sub_ratios = np.zeros(n_sub)

        for z in range(n_depth):
            row_sum = 0.0
            for e in range(n_elements):
                row_sum += D[z, e]
            deviation = row_sum - 1.0
            d_penalty = 200.0 * deviation

            # 1. Propagate to free_conc (film concentrations)
            for jf in range(n_free):
                # d(sum)/d(free_conc) = d(D_free)/d(free_conc) + d(D_linked)/d(free_conc)
                #                     = film_scale + (coeff - 1) * film_scale / n_linked * n_linked
                #                     = film_scale * coeff
                coeff = free_linked_const[jf]
                grad[z * n_free + jf] += d_penalty * film_scale[z] * coeff

            # 2. Propagate to thickness via erf profile
            # D[z, sub_e] = erf_vals[z] * substrate_ratios[j]
            # film_scale[z] = 1 - Σ D[z, sub_e] (if not clipped)
            if film_scale_clipped[z] == 0:
                # d(film_scale)/d(thickness) = -Σ_j substrate_ratios[j] * d_erf/d_t
                d_film_scale_t = 0.0
                for j in range(n_sub):
                    d_film_scale_t -= substrate_ratios[j] * d_erf_t[z]

                # d(D_free)/d(thickness) + d(D_linked)/d(thickness) via film_scale
                for jf in range(n_free):
                    coeff = free_linked_const[jf]
                    grad_pen_thickness += d_penalty * free_conc[z, jf] * d_film_scale_t * coeff

            # d(D_sub)/d(thickness) = substrate_ratios * d_erf/d_t
            for j in range(n_sub):
                grad_pen_thickness += d_penalty * substrate_ratios[j] * d_erf_t[z]

            # 3. Propagate to abruptness via erf profile
            if film_scale_clipped[z] == 0:
                d_film_scale_a = 0.0
                for j in range(n_sub):
                    d_film_scale_a -= substrate_ratios[j] * d_erf_a[z]

                for jf in range(n_free):
                    coeff = free_linked_const[jf]
                    grad_pen_abruptness += d_penalty * free_conc[z, jf] * d_film_scale_a * coeff

            for j in range(n_sub):
                grad_pen_abruptness += d_penalty * substrate_ratios[j] * d_erf_a[z]

            # 4. Propagate to substrate_ratios
            for j in range(n_sub):
                # Direct: d(D_sub)/d(substrate_ratios) = erf_vals[z]
                grad_pen_sub_ratios[j] += d_penalty * erf_vals[z]

                # Indirect via film_scale (if not clipped)
                if film_scale_clipped[z] == 0:
                    d_film_scale_sr = -erf_vals[z]
                    for jf in range(n_free):
                        coeff = free_linked_const[jf]
                        grad_pen_sub_ratios[j] += d_penalty * free_conc[z, jf] * d_film_scale_sr * coeff

        grad[film_size] += grad_pen_thickness
        grad[film_size + 1] += grad_pen_abruptness
        for j in range(n_sub):
            grad[film_size + 2 + j] += grad_pen_sub_ratios[j]

    return grad


@jit(nopython=True, cache=True, fastmath=True)
def gradient_kernel_wls_numba(
    x: np.ndarray,
    m0: np.ndarray,
    I_obs_norm: np.ndarray,
    T_matrices: np.ndarray,
    depth: np.ndarray,
    n_elements: int,
    n_angles: int,
    free_idx: np.ndarray,
    sub_idx: np.ndarray,
    linked_idx: np.ndarray,
    free_linked_const: np.ndarray,
    antisubst_idx: np.ndarray,
    min_conc: float,
    max_conc: float,
    degree: float,
    alpha: float,
    method: int,  # 0=LLS, 1=L2, 2=L1, 3=MEM
    use_sum_constraint: bool,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute analytical gradient of the objective function (WLS version).

    Same as gradient_kernel_numba but uses weighted residuals for Poisson noise.

    Args:
        weights: Weight matrix (n_angles, n_elements) for WLS.
                 For Poisson noise: weight = 1 / I_obs

    Returns:
        Gradient vector (same size as x)
    """
    n_depth = len(depth)
    n_free = len(free_idx)
    n_sub = len(sub_idx)
    n_linked = len(linked_idx)
    n_antisubst = len(antisubst_idx)

    # Unpack parameters
    film_size = n_depth * n_free
    total_size = film_size + 2 + n_sub

    grad = np.zeros(total_size)

    # Film concentrations (reshape from flat)
    free_conc = np.zeros((n_depth, n_free))
    for i in range(n_depth):
        for j in range(n_free):
            free_conc[i, j] = x[i * n_free + j]

    # Structure parameters
    thickness = x[film_size]
    abruptness = x[film_size + 1]

    # Substrate ratios
    substrate_ratios = np.zeros(n_sub)
    for j in range(n_sub):
        substrate_ratios[j] = x[film_size + 2 + j]

    # Build full profile (forward pass)
    sigma = abruptness / (2.0 * sqrt(log(2.0)))

    # Substrate profile and its erf values
    erf_vals = np.zeros(n_depth)
    for i in range(n_depth):
        erf_vals[i] = 0.5 * (1.0 + erf((depth[i] - thickness) / sigma))

    # D_sub[i, j] = erf_vals[i] * substrate_ratios[j]
    D_sub = np.zeros((n_depth, n_sub))
    for i in range(n_depth):
        for j in range(n_sub):
            D_sub[i, j] = erf_vals[i] * substrate_ratios[j]

    # Film scale (1 - substrate_sum) with clipping
    film_scale = np.zeros(n_depth)
    film_scale_clipped = np.zeros(n_depth, dtype=np.int64)
    for i in range(n_depth):
        substrate_sum = 0.0
        for j in range(n_sub):
            substrate_sum += D_sub[i, j]
        unclipped = 1.0 - substrate_sum
        if unclipped < min_conc:
            film_scale[i] = min_conc
            film_scale_clipped[i] = 1
        else:
            film_scale[i] = unclipped
            film_scale_clipped[i] = 0

    # Build full D matrix
    D = np.zeros((n_depth, n_elements))

    # Substrate elements
    for i in range(n_depth):
        for j in range(n_sub):
            D[i, sub_idx[j]] = D_sub[i, j]

    # Free elements
    for i in range(n_depth):
        for j in range(n_free):
            D[i, free_idx[j]] = free_conc[i, j] * film_scale[i]

    # Linked elements (if any)
    if n_linked > 0:
        for i in range(n_depth):
            D_linked_val = 0.0
            for j in range(n_free):
                coeff = free_linked_const[j] - 1.0
                D_linked_val += D[i, free_idx[j]] * coeff
            for k in range(n_linked):
                D[i, linked_idx[k]] = D_linked_val / n_linked

    # Forward model: I_pred = T @ D
    I_pred = np.zeros((n_angles, n_elements))
    for e in range(n_elements):
        for a in range(n_angles):
            val = 0.0
            for z in range(n_depth):
                val += T_matrices[e, a, z] * D[z, e]
            I_pred[a, e] = val

    # Normalize by substrate sum
    I_sub_sum = np.zeros(n_angles)
    for a in range(n_angles):
        for j in range(n_sub):
            I_sub_sum[a] += I_pred[a, sub_idx[j]]
        I_sub_sum[a] = max(I_sub_sum[a], 1e-10)

    I_pred_norm = np.zeros((n_angles, n_elements))
    for a in range(n_angles):
        for e in range(n_elements):
            I_pred_norm[a, e] = I_pred[a, e] / I_sub_sum[a]

    # ========================================
    # Backward pass: compute gradients (WLS)
    # ========================================

    # Gradient w.r.t. I_pred (before normalization)
    # WLS: weight the residuals by weights[a, e]
    dL_dI_pred = np.zeros((n_angles, n_elements))

    for a in range(n_angles):
        S = I_sub_sum[a]
        S_sq = S * S

        # dL/dI_pred_norm[a,e] = -degree * w[a,e] * (I_obs - I_pred_norm)
        for i in range(n_antisubst):
            e = antisubst_idx[i]
            r = I_obs_norm[a, e] - I_pred_norm[a, e]
            w = weights[a, e]
            dL_dI_norm_e = -degree * w * r

            dL_dI_pred[a, e] += dL_dI_norm_e / S

        # Indirect contribution from substrate normalization
        for j in range(n_sub):
            sub_k = sub_idx[j]
            grad_sub_k = 0.0

            for i in range(n_antisubst):
                e = antisubst_idx[i]
                r = I_obs_norm[a, e] - I_pred_norm[a, e]
                w = weights[a, e]
                dL_dI_norm_e = -degree * w * r
                grad_sub_k += dL_dI_norm_e * (-I_pred[a, e] / S_sq)

            dL_dI_pred[a, sub_k] += grad_sub_k

    # Gradient w.r.t. D
    dL_dD = np.zeros((n_depth, n_elements))
    for e in range(n_elements):
        for z in range(n_depth):
            val = 0.0
            for a in range(n_angles):
                val += dL_dI_pred[a, e] * T_matrices[e, a, z]
            dL_dD[z, e] = val

    # ========================================
    # Gradient w.r.t. optimization variables
    # ========================================

    # 1. Gradient w.r.t. free_conc
    for j in range(n_free):
        free_e = free_idx[j]
        coeff = free_linked_const[j] - 1.0

        for z in range(n_depth):
            grad_val = dL_dD[z, free_e] * film_scale[z]

            if n_linked > 0:
                for k in range(n_linked):
                    linked_e = linked_idx[k]
                    grad_val += dL_dD[z, linked_e] * film_scale[z] * coeff / n_linked

            grad[z * n_free + j] = grad_val

    # 2. Gradient w.r.t. thickness
    d_erf_d_t = compute_d_erf_d_thickness_numba(depth, thickness, sigma)

    grad_thickness = 0.0
    for z in range(n_depth):
        if film_scale_clipped[z] == 0:
            d_film_scale = 0.0
            for j in range(n_sub):
                d_film_scale -= substrate_ratios[j] * d_erf_d_t[z]

            for jf in range(n_free):
                free_e = free_idx[jf]
                grad_thickness += dL_dD[z, free_e] * free_conc[z, jf] * d_film_scale

                if n_linked > 0:
                    coeff = free_linked_const[jf] - 1.0
                    for k in range(n_linked):
                        linked_e = linked_idx[k]
                        grad_thickness += dL_dD[z, linked_e] * free_conc[z, jf] * d_film_scale * coeff / n_linked

        for j in range(n_sub):
            sub_e = sub_idx[j]
            grad_thickness += dL_dD[z, sub_e] * substrate_ratios[j] * d_erf_d_t[z]

    grad[film_size] = grad_thickness

    # 3. Gradient w.r.t. abruptness
    d_erf_d_a = compute_d_erf_d_abruptness_numba(depth, thickness, abruptness)

    grad_abruptness = 0.0
    for z in range(n_depth):
        if film_scale_clipped[z] == 0:
            d_film_scale = 0.0
            for j in range(n_sub):
                d_film_scale -= substrate_ratios[j] * d_erf_d_a[z]

            for jf in range(n_free):
                free_e = free_idx[jf]
                grad_abruptness += dL_dD[z, free_e] * free_conc[z, jf] * d_film_scale

                if n_linked > 0:
                    coeff = free_linked_const[jf] - 1.0
                    for k in range(n_linked):
                        linked_e = linked_idx[k]
                        grad_abruptness += dL_dD[z, linked_e] * free_conc[z, jf] * d_film_scale * coeff / n_linked

        for j in range(n_sub):
            sub_e = sub_idx[j]
            grad_abruptness += dL_dD[z, sub_e] * substrate_ratios[j] * d_erf_d_a[z]

    grad[film_size + 1] = grad_abruptness

    # 4. Gradient w.r.t. substrate_ratios
    for j in range(n_sub):
        sub_e = sub_idx[j]
        grad_ratio = 0.0

        for z in range(n_depth):
            grad_ratio += dL_dD[z, sub_e] * erf_vals[z]

            if film_scale_clipped[z] == 0:
                d_film_scale = -erf_vals[z]

                for jf in range(n_free):
                    free_e = free_idx[jf]
                    grad_ratio += dL_dD[z, free_e] * free_conc[z, jf] * d_film_scale

                    if n_linked > 0:
                        coeff = free_linked_const[jf] - 1.0
                        for k in range(n_linked):
                            linked_e = linked_idx[k]
                            grad_ratio += dL_dD[z, linked_e] * free_conc[z, jf] * d_film_scale * coeff / n_linked

        grad[film_size + 2 + j] = grad_ratio

    # ========================================
    # Regularization gradients
    # ========================================

    m0_thickness = m0[film_size]
    m0_abruptness = m0[film_size + 1]

    if method == 1:  # L2
        for i in range(film_size):
            grad[i] += alpha * 2.0 * (x[i] - m0[i])
        grad[film_size] += alpha * 0.02 * (thickness - m0_thickness)
        grad[film_size + 1] += alpha * 2.0 * (abruptness - m0_abruptness)

    elif method == 2:  # L1 — smoothed subgradient (Huber-like)
        _l1_delta = 1e-3
        for i in range(film_size):
            diff = x[i] - m0[i]
            ad = abs(diff)
            grad[i] += alpha * diff / max(ad, _l1_delta)

        diff_t = thickness - m0_thickness
        ad_t = abs(diff_t)
        grad[film_size] += alpha * 0.01 * diff_t / max(ad_t, _l1_delta)

    elif method == 3:  # MEM — d/dx[Σ(x·log(x/m0) - x + m0)] = log(x/m0)
        for i in range(film_size):
            x_safe = max(x[i], 1e-10)
            m0_safe = max(m0[i], 1e-10)
            grad[i] += alpha * log(x_safe / m0_safe)

        t_safe = max(thickness, 1e-10)
        m0_t_safe = max(m0_thickness, 1e-10)
        grad[film_size] += alpha * log(t_safe / m0_t_safe) / 10.0

    elif method == 4:  # L1 log-cosh: grad = tanh((x-m0)/β)
        _l1_beta = 0.01
        for i in range(film_size):
            d = (x[i] - m0[i]) / _l1_beta
            grad[i] += alpha * np.tanh(d)
        d_t = (thickness - m0_thickness) / (100.0 * _l1_beta)
        grad[film_size] += alpha * np.tanh(d_t) * 0.01

    # ========================================
    # Sum constraint penalty gradient
    # ========================================

    if use_sum_constraint:
        d_erf_t = compute_d_erf_d_thickness_numba(depth, thickness, sigma)
        d_erf_a = compute_d_erf_d_abruptness_numba(depth, thickness, abruptness)

        grad_pen_thickness = 0.0
        grad_pen_abruptness = 0.0
        grad_pen_sub_ratios = np.zeros(n_sub)

        for z in range(n_depth):
            row_sum = 0.0
            for e in range(n_elements):
                row_sum += D[z, e]
            deviation = row_sum - 1.0
            d_penalty = 200.0 * deviation

            for jf in range(n_free):
                coeff = free_linked_const[jf]
                grad[z * n_free + jf] += d_penalty * film_scale[z] * coeff

            if film_scale_clipped[z] == 0:
                d_film_scale_t = 0.0
                for j in range(n_sub):
                    d_film_scale_t -= substrate_ratios[j] * d_erf_t[z]

                for jf in range(n_free):
                    coeff = free_linked_const[jf]
                    grad_pen_thickness += d_penalty * free_conc[z, jf] * d_film_scale_t * coeff

            for j in range(n_sub):
                grad_pen_thickness += d_penalty * substrate_ratios[j] * d_erf_t[z]

            if film_scale_clipped[z] == 0:
                d_film_scale_a = 0.0
                for j in range(n_sub):
                    d_film_scale_a -= substrate_ratios[j] * d_erf_a[z]

                for jf in range(n_free):
                    coeff = free_linked_const[jf]
                    grad_pen_abruptness += d_penalty * free_conc[z, jf] * d_film_scale_a * coeff

            for j in range(n_sub):
                grad_pen_abruptness += d_penalty * substrate_ratios[j] * d_erf_a[z]

            for j in range(n_sub):
                grad_pen_sub_ratios[j] += d_penalty * erf_vals[z]

                if film_scale_clipped[z] == 0:
                    d_film_scale_sr = -erf_vals[z]
                    for jf in range(n_free):
                        coeff = free_linked_const[jf]
                        grad_pen_sub_ratios[j] += d_penalty * free_conc[z, jf] * d_film_scale_sr * coeff

        grad[film_size] += grad_pen_thickness
        grad[film_size + 1] += grad_pen_abruptness
        for j in range(n_sub):
            grad[film_size + 2 + j] += grad_pen_sub_ratios[j]

    return grad


@jit(nopython=True, cache=True, fastmath=True)
def gradient_kernel_poisson_numba(
    x: np.ndarray,
    m0: np.ndarray,
    I_obs_norm: np.ndarray,
    T_matrices: np.ndarray,
    depth: np.ndarray,
    n_elements: int,
    n_angles: int,
    free_idx: np.ndarray,
    sub_idx: np.ndarray,
    linked_idx: np.ndarray,
    free_linked_const: np.ndarray,
    antisubst_idx: np.ndarray,
    min_conc: float,
    max_conc: float,
    degree: float,
    alpha: float,
    method: int,  # 0=LLS, 1=L2, 2=L1, 3=MEM
    use_sum_constraint: bool,
) -> np.ndarray:
    """
    Compute analytical gradient for Poisson negative log-likelihood objective.

    For Poisson NLL: L = Σ [I_pred - I_obs * log(I_pred)]
    Gradient: dL/dI_pred = 1 - I_obs/I_pred

    This is the correct gradient for Poisson-distributed data.

    Args:
        Same as gradient_kernel_numba

    Returns:
        Gradient vector (same size as x)
    """
    n_depth = len(depth)
    n_free = len(free_idx)
    n_sub = len(sub_idx)
    n_linked = len(linked_idx)
    n_antisubst = len(antisubst_idx)

    # Unpack parameters
    film_size = n_depth * n_free
    total_size = film_size + 2 + n_sub

    grad = np.zeros(total_size)

    # Film concentrations (reshape from flat)
    free_conc = np.zeros((n_depth, n_free))
    for i in range(n_depth):
        for j in range(n_free):
            free_conc[i, j] = x[i * n_free + j]

    # Structure parameters
    thickness = x[film_size]
    abruptness = x[film_size + 1]

    # Substrate ratios
    substrate_ratios = np.zeros(n_sub)
    for j in range(n_sub):
        substrate_ratios[j] = x[film_size + 2 + j]

    # Build full profile (forward pass)
    sigma = abruptness / (2.0 * sqrt(log(2.0)))

    # Substrate profile and its erf values
    erf_vals = np.zeros(n_depth)
    for i in range(n_depth):
        erf_vals[i] = 0.5 * (1.0 + erf((depth[i] - thickness) / sigma))

    # D_sub[i, j] = erf_vals[i] * substrate_ratios[j]
    D_sub = np.zeros((n_depth, n_sub))
    for i in range(n_depth):
        for j in range(n_sub):
            D_sub[i, j] = erf_vals[i] * substrate_ratios[j]

    # Film scale (1 - substrate_sum) with clipping
    film_scale = np.zeros(n_depth)
    film_scale_clipped = np.zeros(n_depth, dtype=np.int64)
    for i in range(n_depth):
        substrate_sum = 0.0
        for j in range(n_sub):
            substrate_sum += D_sub[i, j]
        unclipped = 1.0 - substrate_sum
        if unclipped < min_conc:
            film_scale[i] = min_conc
            film_scale_clipped[i] = 1
        else:
            film_scale[i] = unclipped
            film_scale_clipped[i] = 0

    # Build full D matrix
    D = np.zeros((n_depth, n_elements))

    # Substrate elements
    for i in range(n_depth):
        for j in range(n_sub):
            D[i, sub_idx[j]] = D_sub[i, j]

    # Free elements
    for i in range(n_depth):
        for j in range(n_free):
            D[i, free_idx[j]] = free_conc[i, j] * film_scale[i]

    # Linked elements (if any)
    if n_linked > 0:
        for i in range(n_depth):
            D_linked_val = 0.0
            for j in range(n_free):
                coeff = free_linked_const[j] - 1.0
                D_linked_val += D[i, free_idx[j]] * coeff
            for k in range(n_linked):
                D[i, linked_idx[k]] = D_linked_val / n_linked

    # Forward model: I_pred = T @ D
    I_pred = np.zeros((n_angles, n_elements))
    for e in range(n_elements):
        for a in range(n_angles):
            val = 0.0
            for z in range(n_depth):
                val += T_matrices[e, a, z] * D[z, e]
            I_pred[a, e] = val

    # Normalize by substrate sum
    I_sub_sum = np.zeros(n_angles)
    for a in range(n_angles):
        for j in range(n_sub):
            I_sub_sum[a] += I_pred[a, sub_idx[j]]
        I_sub_sum[a] = max(I_sub_sum[a], 1e-10)

    I_pred_norm = np.zeros((n_angles, n_elements))
    for a in range(n_angles):
        for e in range(n_elements):
            I_pred_norm[a, e] = I_pred[a, e] / I_sub_sum[a]

    # ========================================
    # Backward pass: compute gradients (Poisson NLL)
    # ========================================

    # Gradient w.r.t. I_pred_norm (before normalization backprop)
    # Poisson NLL: dL/dI_pred_norm = degree * (1 - I_obs/I_pred_norm)
    dL_dI_pred = np.zeros((n_angles, n_elements))

    for a in range(n_angles):
        S = I_sub_sum[a]
        S_sq = S * S

        # Direct gradient for antisubstrate elements
        for i in range(n_antisubst):
            e = antisubst_idx[i]
            I_pred_safe = max(I_pred_norm[a, e], 1e-10)
            I_obs = I_obs_norm[a, e]

            # Poisson deviance gradient: dD/dI_pred_norm = 2 * (1 - I_obs/I_pred)
            dL_dI_norm_e = degree * 2.0 * (1.0 - I_obs / I_pred_safe)

            dL_dI_pred[a, e] += dL_dI_norm_e / S

        # Indirect contribution from substrate normalization
        for j in range(n_sub):
            sub_k = sub_idx[j]
            grad_sub_k = 0.0

            for i in range(n_antisubst):
                e = antisubst_idx[i]
                I_pred_safe = max(I_pred_norm[a, e], 1e-10)
                I_obs = I_obs_norm[a, e]

                # Poisson deviance gradient with factor 2
                dL_dI_norm_e = degree * 2.0 * (1.0 - I_obs / I_pred_safe)
                grad_sub_k += dL_dI_norm_e * (-I_pred[a, e] / S_sq)

            dL_dI_pred[a, sub_k] += grad_sub_k

    # Gradient w.r.t. D
    dL_dD = np.zeros((n_depth, n_elements))
    for e in range(n_elements):
        for z in range(n_depth):
            val = 0.0
            for a in range(n_angles):
                val += dL_dI_pred[a, e] * T_matrices[e, a, z]
            dL_dD[z, e] = val

    # ========================================
    # Gradient w.r.t. optimization variables
    # ========================================

    # 1. Gradient w.r.t. free_conc
    for j in range(n_free):
        free_e = free_idx[j]
        coeff = free_linked_const[j] - 1.0

        for z in range(n_depth):
            grad_val = dL_dD[z, free_e] * film_scale[z]

            if n_linked > 0:
                for k in range(n_linked):
                    linked_e = linked_idx[k]
                    grad_val += dL_dD[z, linked_e] * film_scale[z] * coeff / n_linked

            grad[z * n_free + j] = grad_val

    # 2. Gradient w.r.t. thickness
    d_erf_d_t = compute_d_erf_d_thickness_numba(depth, thickness, sigma)

    grad_thickness = 0.0
    for z in range(n_depth):
        if film_scale_clipped[z] == 0:
            d_film_scale = 0.0
            for j in range(n_sub):
                d_film_scale -= substrate_ratios[j] * d_erf_d_t[z]

            for jf in range(n_free):
                free_e = free_idx[jf]
                grad_thickness += dL_dD[z, free_e] * free_conc[z, jf] * d_film_scale

                if n_linked > 0:
                    coeff = free_linked_const[jf] - 1.0
                    for k in range(n_linked):
                        linked_e = linked_idx[k]
                        grad_thickness += dL_dD[z, linked_e] * free_conc[z, jf] * d_film_scale * coeff / n_linked

        for j in range(n_sub):
            sub_e = sub_idx[j]
            grad_thickness += dL_dD[z, sub_e] * substrate_ratios[j] * d_erf_d_t[z]

    grad[film_size] = grad_thickness

    # 3. Gradient w.r.t. abruptness
    d_erf_d_a = compute_d_erf_d_abruptness_numba(depth, thickness, abruptness)

    grad_abruptness = 0.0
    for z in range(n_depth):
        if film_scale_clipped[z] == 0:
            d_film_scale = 0.0
            for j in range(n_sub):
                d_film_scale -= substrate_ratios[j] * d_erf_d_a[z]

            for jf in range(n_free):
                free_e = free_idx[jf]
                grad_abruptness += dL_dD[z, free_e] * free_conc[z, jf] * d_film_scale

                if n_linked > 0:
                    coeff = free_linked_const[jf] - 1.0
                    for k in range(n_linked):
                        linked_e = linked_idx[k]
                        grad_abruptness += dL_dD[z, linked_e] * free_conc[z, jf] * d_film_scale * coeff / n_linked

        for j in range(n_sub):
            sub_e = sub_idx[j]
            grad_abruptness += dL_dD[z, sub_e] * substrate_ratios[j] * d_erf_d_a[z]

    grad[film_size + 1] = grad_abruptness

    # 4. Gradient w.r.t. substrate_ratios
    for j in range(n_sub):
        sub_e = sub_idx[j]
        grad_ratio = 0.0

        for z in range(n_depth):
            grad_ratio += dL_dD[z, sub_e] * erf_vals[z]

            if film_scale_clipped[z] == 0:
                d_film_scale = -erf_vals[z]

                for jf in range(n_free):
                    free_e = free_idx[jf]
                    grad_ratio += dL_dD[z, free_e] * free_conc[z, jf] * d_film_scale

                    if n_linked > 0:
                        coeff = free_linked_const[jf] - 1.0
                        for k in range(n_linked):
                            linked_e = linked_idx[k]
                            grad_ratio += dL_dD[z, linked_e] * free_conc[z, jf] * d_film_scale * coeff / n_linked

        grad[film_size + 2 + j] = grad_ratio

    # ========================================
    # Regularization gradients
    # ========================================

    m0_thickness = m0[film_size]
    m0_abruptness = m0[film_size + 1]

    if method == 1:  # L2
        for i in range(film_size):
            grad[i] += alpha * 2.0 * (x[i] - m0[i])
        grad[film_size] += alpha * 0.02 * (thickness - m0_thickness)
        grad[film_size + 1] += alpha * 2.0 * (abruptness - m0_abruptness)

    elif method == 2:  # L1 — smoothed subgradient (Huber-like)
        _l1_delta = 1e-3
        for i in range(film_size):
            diff = x[i] - m0[i]
            ad = abs(diff)
            grad[i] += alpha * diff / max(ad, _l1_delta)

        diff_t = thickness - m0_thickness
        ad_t = abs(diff_t)
        grad[film_size] += alpha * 0.01 * diff_t / max(ad_t, _l1_delta)

    elif method == 3:  # MEM — d/dx[Σ(x·log(x/m0) - x + m0)] = log(x/m0)
        for i in range(film_size):
            x_safe = max(x[i], 1e-10)
            m0_safe = max(m0[i], 1e-10)
            grad[i] += alpha * log(x_safe / m0_safe)

        t_safe = max(thickness, 1e-10)
        m0_t_safe = max(m0_thickness, 1e-10)
        grad[film_size] += alpha * log(t_safe / m0_t_safe) / 10.0

    elif method == 4:  # L1 log-cosh: grad = tanh((x-m0)/β)
        _l1_beta = 0.01
        for i in range(film_size):
            d = (x[i] - m0[i]) / _l1_beta
            grad[i] += alpha * np.tanh(d)
        d_t = (thickness - m0_thickness) / (100.0 * _l1_beta)
        grad[film_size] += alpha * np.tanh(d_t) * 0.01

    # ========================================
    # Sum constraint penalty gradient
    # ========================================

    if use_sum_constraint:
        d_erf_t = compute_d_erf_d_thickness_numba(depth, thickness, sigma)
        d_erf_a = compute_d_erf_d_abruptness_numba(depth, thickness, abruptness)

        grad_pen_thickness = 0.0
        grad_pen_abruptness = 0.0
        grad_pen_sub_ratios = np.zeros(n_sub)

        for z in range(n_depth):
            row_sum = 0.0
            for e in range(n_elements):
                row_sum += D[z, e]
            deviation = row_sum - 1.0
            d_penalty = 200.0 * deviation

            for jf in range(n_free):
                coeff = free_linked_const[jf]
                grad[z * n_free + jf] += d_penalty * film_scale[z] * coeff

            if film_scale_clipped[z] == 0:
                d_film_scale_t = 0.0
                for j in range(n_sub):
                    d_film_scale_t -= substrate_ratios[j] * d_erf_t[z]

                for jf in range(n_free):
                    coeff = free_linked_const[jf]
                    grad_pen_thickness += d_penalty * free_conc[z, jf] * d_film_scale_t * coeff

            for j in range(n_sub):
                grad_pen_thickness += d_penalty * substrate_ratios[j] * d_erf_t[z]

            if film_scale_clipped[z] == 0:
                d_film_scale_a = 0.0
                for j in range(n_sub):
                    d_film_scale_a -= substrate_ratios[j] * d_erf_a[z]

                for jf in range(n_free):
                    coeff = free_linked_const[jf]
                    grad_pen_abruptness += d_penalty * free_conc[z, jf] * d_film_scale_a * coeff

            for j in range(n_sub):
                grad_pen_abruptness += d_penalty * substrate_ratios[j] * d_erf_a[z]

            for j in range(n_sub):
                grad_pen_sub_ratios[j] += d_penalty * erf_vals[z]

                if film_scale_clipped[z] == 0:
                    d_film_scale_sr = -erf_vals[z]
                    for jf in range(n_free):
                        coeff = free_linked_const[jf]
                        grad_pen_sub_ratios[j] += d_penalty * free_conc[z, jf] * d_film_scale_sr * coeff

        grad[film_size] += grad_pen_thickness
        grad[film_size + 1] += grad_pen_abruptness
        for j in range(n_sub):
            grad[film_size + 2 + j] += grad_pen_sub_ratios[j]

    return grad


@jit(nopython=True, cache=True, fastmath=True)
def gradient_backward_numba(
    x: np.ndarray,
    m0: np.ndarray,
    dL_dD: np.ndarray,
    film_conc: np.ndarray,
    thickness: float,
    abruptness: float,
    substrate_ratios: np.ndarray,
    depth: np.ndarray,
    n_elements: int,
    free_idx: np.ndarray,
    sub_idx: np.ndarray,
    linked_idx: np.ndarray,
    free_linked_const: np.ndarray,
    min_conc: float,
    degree: float,
    alpha: float,
    method: int,  # 0=LLS, 1=L2, 2=L1, 3=MEM
) -> np.ndarray:
    """
    Backward pass only: compute gradient from pre-computed dL/dD.

    This kernel implements the chain rule from dL/dD to dL/dx (the optimization
    variable vector), including regularization gradients. It is designed to be
    used with forward pass results cached by the objective function, avoiding
    the forward pass mismatch that exists in gradient_kernel_numba.

    The forward pass (x → D → I_pred → I_pred_norm → dL/dI_pred → dL/dD)
    is computed in Python/NumPy using the same code path as the objective,
    ensuring exact consistency. Only this backward pass (dL/dD → dL/dx)
    runs in Numba for speed.

    Args:
        x: Optimization variable vector
        m0: Prior vector (same layout as x)
        dL_dD: Gradient of loss w.r.t. D, shape (n_depth, n_elements)
        film_conc: Free element concentrations (n_depth, n_free) from _unpack
        thickness: Film thickness from _unpack
        abruptness: Interface abruptness from _unpack
        substrate_ratios: Substrate ratios from _unpack
        depth: Depth array (n_depth,)
        n_elements: Total number of elements
        free_idx: Indices of free elements
        sub_idx: Indices of substrate elements
        linked_idx: Indices of linked elements
        free_linked_const: Charge-linked constraints
        min_conc: Minimum concentration
        degree: Weight for data fit term
        alpha: Regularization strength
        method: 0=LLS, 1=L2, 2=L1, 3=MEM

    Returns:
        Gradient vector (same size as x)
    """
    n_depth = len(depth)
    n_free = len(free_idx)
    n_sub = len(sub_idx)
    n_linked = len(linked_idx)

    film_size = n_depth * n_free
    total_size = film_size + 2 + n_sub
    grad = np.zeros(total_size)

    # Recompute erf and film_scale (cheap, avoids passing extra arrays)
    sigma = abruptness / (2.0 * sqrt(log(2.0)))

    erf_vals = np.zeros(n_depth)
    for i in range(n_depth):
        erf_vals[i] = 0.5 * (1.0 + erf((depth[i] - thickness) / sigma))

    film_scale = np.zeros(n_depth)
    film_scale_clipped = np.zeros(n_depth, dtype=np.int64)
    for i in range(n_depth):
        substrate_sum = 0.0
        for j in range(n_sub):
            substrate_sum += erf_vals[i] * substrate_ratios[j]
        unclipped = 1.0 - substrate_sum
        if unclipped < min_conc:
            film_scale[i] = min_conc
            film_scale_clipped[i] = 1
        else:
            film_scale[i] = unclipped
            film_scale_clipped[i] = 0

    # ========================================
    # 1. Gradient w.r.t. free_conc
    # ========================================
    # D[z, free_e] = free_conc[z, j] * film_scale[z]
    # D[z, linked_e] = Σ_j D[z, free_e] * (coeff[j] - 1) / n_linked
    for z in range(n_depth):
        for j in range(n_free):
            free_e = free_idx[j]
            grad_val = dL_dD[z, free_e] * film_scale[z]

            if n_linked > 0:
                coeff = free_linked_const[j] - 1.0
                for k in range(n_linked):
                    linked_e = linked_idx[k]
                    grad_val += dL_dD[z, linked_e] * film_scale[z] * coeff / n_linked

            grad[z * n_free + j] = grad_val

    # ========================================
    # 2. Gradient w.r.t. thickness
    # ========================================
    d_erf_d_t = compute_d_erf_d_thickness_numba(depth, thickness, sigma)

    grad_thickness = 0.0
    for z in range(n_depth):
        if film_scale_clipped[z] == 0:
            d_film_scale = 0.0
            for j in range(n_sub):
                d_film_scale -= substrate_ratios[j] * d_erf_d_t[z]

            for jf in range(n_free):
                free_e = free_idx[jf]
                grad_thickness += dL_dD[z, free_e] * film_conc[z, jf] * d_film_scale

                if n_linked > 0:
                    coeff = free_linked_const[jf] - 1.0
                    for k in range(n_linked):
                        linked_e = linked_idx[k]
                        grad_thickness += dL_dD[z, linked_e] * film_conc[z, jf] * d_film_scale * coeff / n_linked

        for j in range(n_sub):
            sub_e = sub_idx[j]
            grad_thickness += dL_dD[z, sub_e] * substrate_ratios[j] * d_erf_d_t[z]

    grad[film_size] = grad_thickness

    # ========================================
    # 3. Gradient w.r.t. abruptness
    # ========================================
    d_erf_d_a = compute_d_erf_d_abruptness_numba(depth, thickness, abruptness)

    grad_abruptness = 0.0
    for z in range(n_depth):
        if film_scale_clipped[z] == 0:
            d_film_scale = 0.0
            for j in range(n_sub):
                d_film_scale -= substrate_ratios[j] * d_erf_d_a[z]

            for jf in range(n_free):
                free_e = free_idx[jf]
                grad_abruptness += dL_dD[z, free_e] * film_conc[z, jf] * d_film_scale

                if n_linked > 0:
                    coeff = free_linked_const[jf] - 1.0
                    for k in range(n_linked):
                        linked_e = linked_idx[k]
                        grad_abruptness += dL_dD[z, linked_e] * film_conc[z, jf] * d_film_scale * coeff / n_linked

        for j in range(n_sub):
            sub_e = sub_idx[j]
            grad_abruptness += dL_dD[z, sub_e] * substrate_ratios[j] * d_erf_d_a[z]

    grad[film_size + 1] = grad_abruptness

    # ========================================
    # 4. Gradient w.r.t. substrate_ratios
    # ========================================
    for j in range(n_sub):
        sub_e = sub_idx[j]
        grad_ratio = 0.0

        for z in range(n_depth):
            grad_ratio += dL_dD[z, sub_e] * erf_vals[z]

            if film_scale_clipped[z] == 0:
                d_film_scale = -erf_vals[z]

                for jf in range(n_free):
                    free_e = free_idx[jf]
                    grad_ratio += dL_dD[z, free_e] * film_conc[z, jf] * d_film_scale

                    if n_linked > 0:
                        coeff = free_linked_const[jf] - 1.0
                        for k in range(n_linked):
                            linked_e = linked_idx[k]
                            grad_ratio += dL_dD[z, linked_e] * film_conc[z, jf] * d_film_scale * coeff / n_linked

        grad[film_size + 2 + j] = grad_ratio

    # ========================================
    # 5. Regularization gradients
    # ========================================
    m0_thickness = m0[film_size]
    m0_abruptness = m0[film_size + 1]

    if method == 1:  # L2
        for i in range(film_size):
            grad[i] += alpha * 2.0 * (x[i] - m0[i])
        grad[film_size] += alpha * 0.02 * (thickness - m0_thickness)
        grad[film_size + 1] += alpha * 2.0 * (abruptness - m0_abruptness)

    elif method == 2:  # L1 — smoothed subgradient (Huber-like)
        _l1_delta = 1e-3
        for i in range(film_size):
            diff = x[i] - m0[i]
            ad = abs(diff)
            grad[i] += alpha * diff / max(ad, _l1_delta)
        diff_t = thickness - m0_thickness
        ad_t = abs(diff_t)
        grad[film_size] += alpha * 0.01 * diff_t / max(ad_t, _l1_delta)

    elif method == 3:  # MEM — d/dx[Σ(x·log(x/m0) - x + m0)] = log(x/m0)
        for i in range(film_size):
            x_safe = max(x[i], 1e-10)
            m0_safe = max(m0[i], 1e-10)
            grad[i] += alpha * log(x_safe / m0_safe)
        t_safe = max(thickness, 1e-10)
        m0_t_safe = max(m0_thickness, 1e-10)
        grad[film_size] += alpha * log(t_safe / m0_t_safe) / 10.0

    elif method == 4:  # L1 log-cosh: grad = tanh((x-m0)/β)
        _l1_beta = 0.01  # Must match SolverConfig.l1_beta
        for i in range(film_size):
            d = (x[i] - m0[i]) / _l1_beta
            grad[i] += alpha * np.tanh(d)
        d_t = (thickness - m0_thickness) / (100.0 * _l1_beta)
        grad[film_size] += alpha * np.tanh(d_t) * 0.01

    return grad


def warmup_jit():
    """
    Warmup JIT compilation by running each kernel once with small inputs.
    This avoids compilation overhead during actual solving.
    """
    if not NUMBA_AVAILABLE:
        return

    # Small test data
    depth = np.linspace(0, 100, 10)
    substrate_ratios = np.array([0.5, 0.5])
    free_conc = np.random.rand(10, 2)

    # Warmup calls
    _ = build_substrate_profile_numba(depth, 50.0, 10.0, substrate_ratios)

    free_idx = np.array([0, 1], dtype=np.int64)
    sub_idx = np.array([2, 3], dtype=np.int64)
    linked_idx = np.array([4], dtype=np.int64)
    free_linked_const = np.array([1.0, 1.0])

    _ = build_full_profile_numba(
        free_conc, 50.0, 10.0, substrate_ratios,
        depth, 5, free_idx, sub_idx, linked_idx,
        free_linked_const, 1e-6, 1.0
    )

    D = np.random.rand(10, 5)
    T = np.random.rand(5, 5, 10)
    _ = compute_forward_model_numba(D, T, 5, 5)

    # Warmup gradient kernel
    x = np.random.rand(10 * 2 + 2 + 2)  # film_size + 2 + n_sub
    m0 = np.random.rand(10 * 2 + 2 + 2)
    I_obs_norm = np.random.rand(5, 5)
    antisubst_idx = np.array([0, 1, 4], dtype=np.int64)

    _ = gradient_kernel_numba(
        x, m0, I_obs_norm, T, depth, 5, 5,
        free_idx, sub_idx, linked_idx, free_linked_const,
        antisubst_idx, 1e-6, 1.0, 1.0, 0.1, 1, False
    )

    # Warmup backward-only gradient kernel
    dL_dD = np.random.rand(10, 5)
    film_conc = np.random.rand(10, 2)
    _ = gradient_backward_numba(
        x, m0, dL_dD, film_conc, 50.0, 10.0, substrate_ratios,
        depth, 5, free_idx, sub_idx, linked_idx, free_linked_const,
        1e-6, 1.0, 0.1, 1
    )


# Auto-warmup on import: type-specialize all kernels so first solve has zero JIT overhead
warmup_jit()
