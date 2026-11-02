"""
Alpha Selection Benchmark: L-curve vs GCV
==========================================

Compares automatic regularization parameter selection methods:
1. L-curve: Curvature-based selection
2. GCV: Generalized Cross-Validation

Test conditions:
- Model: 4-layer C / Al2O3 / TiO2 / SiO2 / Si
- Solver: SemiFreeSolver with L1 regularization
- Noise: 2% Gaussian
- Metric: Wasserstein distance (W1)
"""

import numpy as np
from deppro._compat import trapezoid
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy.special import erf

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import (
    SemiFreeSolver, SemiFreeSolverConfig, SolverConfig
)


# =============================================================================
# Alpha Selection Methods
# =============================================================================

@dataclass
class AlphaSelectionResult:
    """Result from alpha selection."""
    alpha_optimal: float
    alphas_tested: np.ndarray
    scores: np.ndarray  # Residuals for L-curve, GCV scores for GCV
    reg_norms: Optional[np.ndarray] = None  # For L-curve only
    curvatures: Optional[np.ndarray] = None  # For L-curve only
    method: str = ''


def l_curve_selection(
    I_obs: np.ndarray,
    tm: TransferMatrix,
    semi_free_config: SemiFreeSolverConfig,
    alphas: np.ndarray,
    verbose: bool = False,
) -> AlphaSelectionResult:
    """
    L-curve method for alpha selection.

    Uses the "corner" criterion: find the point that maximizes
    the distance from the line connecting the endpoints.
    This is more robust than curvature-based methods when the
    L-curve is not well-defined.

    Parameters
    ----------
    I_obs : ndarray
        Observed intensities (n_angles, n_elements)
    tm : TransferMatrix
        Transfer matrix
    semi_free_config : SemiFreeSolverConfig
        Semi-free solver configuration
    alphas : ndarray
        Array of alpha values to test
    verbose : bool
        Print progress

    Returns
    -------
    AlphaSelectionResult
    """
    residuals = []
    reg_norms = []

    for i, alpha in enumerate(alphas):
        config = SolverConfig(method='L1', alpha=alpha, max_iter=100)
        solver = SemiFreeSolver(tm, config, semi_free_config)
        result = solver.solve(I_obs)

        # Residual norm: ||I_pred - I_obs||
        residual = np.linalg.norm(result.I_opt - I_obs / I_obs.sum(axis=1, keepdims=True))
        residuals.append(residual)

        # Regularization norm: ||D|| or ||D - D_prior||
        # For L1 with D_prior=0, this is just sum(|D|)
        reg_norm = np.sum(np.abs(result.D_opt))
        reg_norms.append(reg_norm)

        if verbose:
            print(f"  α={alpha:.4f}: residual={residual:.6f}, reg_norm={reg_norm:.4f}")

    residuals = np.array(residuals)
    reg_norms = np.array(reg_norms)

    # Convert to log scale
    log_res = np.log10(residuals + 1e-10)
    log_reg = np.log10(reg_norms + 1e-10)

    # Normalize to [0, 1] range
    log_res_norm = (log_res - log_res.min()) / (log_res.max() - log_res.min() + 1e-10)
    log_reg_norm = (log_reg - log_reg.min()) / (log_reg.max() - log_reg.min() + 1e-10)

    # Method 1: Maximum distance from diagonal line
    # The "corner" is the point farthest from the line connecting endpoints
    # Line from (log_res[0], log_reg[0]) to (log_res[-1], log_reg[-1])
    x1, y1 = log_res_norm[0], log_reg_norm[0]
    x2, y2 = log_res_norm[-1], log_reg_norm[-1]

    # Distance from point (x, y) to line ax + by + c = 0
    # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1

    distances = np.abs(a * log_res_norm + b * log_reg_norm + c) / np.sqrt(a**2 + b**2 + 1e-10)

    # Find maximum distance (the corner)
    best_idx = np.argmax(distances)

    # Also compute curvature for reference
    curvatures = compute_curvature(log_res, log_reg)

    alpha_optimal = alphas[best_idx]

    if verbose:
        print(f"\nL-curve optimal: α = {alpha_optimal:.4f} (index {best_idx})")
        print(f"  (max distance from diagonal = {distances[best_idx]:.4f})")

    return AlphaSelectionResult(
        alpha_optimal=alpha_optimal,
        alphas_tested=alphas,
        scores=residuals,
        reg_norms=reg_norms,
        curvatures=distances,  # Using distances instead of curvatures
        method='L-curve',
    )


def compute_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute curvature of a parametric curve (x(t), y(t)).

    Curvature κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)

    For L-curve, we want POSITIVE curvature (convex corner).
    The L-curve has a characteristic shape where both x and y decrease
    as we increase alpha. The "corner" has high curvature and is convex.

    Parameters
    ----------
    x, y : ndarray
        Curve coordinates (log scale)

    Returns
    -------
    curvatures : ndarray
        Signed curvature at each point (positive = convex corner)
    """
    n = len(x)
    if n < 3:
        return np.zeros(n)

    # Use spline interpolation for smoother derivatives
    from scipy.interpolate import UnivariateSpline

    try:
        # Parameter t = 0, 1, 2, ...
        t = np.arange(n)

        # Smooth splines
        spline_x = UnivariateSpline(t, x, s=0.01, k=4)
        spline_y = UnivariateSpline(t, y, s=0.01, k=4)

        # First derivatives
        dx = spline_x.derivative(1)(t)
        dy = spline_y.derivative(1)(t)

        # Second derivatives
        ddx = spline_x.derivative(2)(t)
        ddy = spline_y.derivative(2)(t)

    except Exception:
        # Fallback to finite differences
        dx = np.zeros(n)
        dy = np.zeros(n)
        dx[1:-1] = (x[2:] - x[:-2]) / 2
        dy[1:-1] = (y[2:] - y[:-2]) / 2
        dx[0] = x[1] - x[0]
        dx[-1] = x[-1] - x[-2]
        dy[0] = y[1] - y[0]
        dy[-1] = y[-1] - y[-2]

        ddx = np.zeros(n)
        ddy = np.zeros(n)
        ddx[1:-1] = x[2:] - 2*x[1:-1] + x[:-2]
        ddy[1:-1] = y[2:] - 2*y[1:-1] + y[:-2]

    # Signed curvature: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**(1.5) + 1e-10

    curvature = numerator / denominator

    # For L-curve, we want the point where curvature changes sign
    # or where there's a local maximum in |curvature|
    # Actually, the corner of L-curve has maximum positive curvature

    return curvature


def gcv_selection(
    I_obs: np.ndarray,
    tm: TransferMatrix,
    semi_free_config: SemiFreeSolverConfig,
    alphas: np.ndarray,
    verbose: bool = False,
) -> AlphaSelectionResult:
    """
    Generalized Cross-Validation for alpha selection.

    GCV(α) = ||Ax - b||² / (1 - trace(A_α)/n)²

    For nonlinear problems, we use an approximation based on
    effective degrees of freedom.

    Parameters
    ----------
    I_obs : ndarray
        Observed intensities (n_angles, n_elements)
    tm : TransferMatrix
        Transfer matrix
    semi_free_config : SemiFreeSolverConfig
        Semi-free solver configuration
    alphas : ndarray
        Array of alpha values to test
    verbose : bool
        Print progress

    Returns
    -------
    AlphaSelectionResult
    """
    n_obs = I_obs.size
    gcv_scores = []

    for i, alpha in enumerate(alphas):
        config = SolverConfig(method='L1', alpha=alpha, max_iter=100)
        solver = SemiFreeSolver(tm, config, semi_free_config)
        result = solver.solve(I_obs)

        # Residual sum of squares
        I_norm = I_obs / I_obs.sum(axis=1, keepdims=True)
        rss = np.sum((result.I_opt - I_norm)**2)

        # Effective degrees of freedom approximation
        # For L1, df ≈ number of non-zero elements
        # Higher alpha → more sparsity → fewer df
        D_flat = result.D_opt.flatten()
        n_params = len(D_flat)

        # Approximate df: count "active" parameters (above threshold)
        threshold = 0.01 * np.max(np.abs(D_flat))
        n_active = np.sum(np.abs(D_flat) > threshold)

        # Trace approximation: df / n_obs
        # For regularized problems: trace ≈ df / (1 + α * something)
        trace_approx = n_active / n_obs

        # GCV score
        denominator = (1 - trace_approx)**2
        if denominator < 1e-10:
            gcv_score = np.inf
        else:
            gcv_score = rss / denominator

        gcv_scores.append(gcv_score)

        if verbose:
            print(f"  α={alpha:.4f}: RSS={rss:.6f}, df≈{n_active}, GCV={gcv_score:.6f}")

    gcv_scores = np.array(gcv_scores)

    # Find minimum GCV score
    best_idx = np.argmin(gcv_scores)
    alpha_optimal = alphas[best_idx]

    if verbose:
        print(f"\nGCV optimal: α = {alpha_optimal:.4f} (index {best_idx})")

    return AlphaSelectionResult(
        alpha_optimal=alpha_optimal,
        alphas_tested=alphas,
        scores=gcv_scores,
        method='GCV',
    )


# =============================================================================
# Test Infrastructure (from existing benchmarks)
# =============================================================================

def create_4layer_profile(
    depth: np.ndarray,
    thicknesses: List[float] = [5.0, 15.0, 35.0, 30.0],
    interface_widths: List[float] = [2.0, 3.0, 4.0, 5.0],
) -> np.ndarray:
    """
    Create 4-layer model: C / Al2O3 / TiO2 / SiO2 / Si

    Elements: [C, Al, Ti, Si_ox, O, Si_bulk]
    """
    sigma = [w / (2 * np.sqrt(np.log(2))) for w in interface_widths]
    z = [0] + list(np.cumsum(thicknesses))

    D = np.zeros((len(depth), 6))

    # C layer
    D[:, 0] = 0.5 * (1 - erf((depth - z[1]) / sigma[0]))

    # Al2O3 layer
    al2o3 = 0.5 * (1 + erf((depth - z[1]) / sigma[0])) * 0.5 * (1 - erf((depth - z[2]) / sigma[1]))
    D[:, 1] = (2/5) * al2o3
    D[:, 4] = (3/5) * al2o3

    # TiO2 layer
    tio2 = 0.5 * (1 + erf((depth - z[2]) / sigma[1])) * 0.5 * (1 - erf((depth - z[3]) / sigma[2]))
    D[:, 2] = (1/3) * tio2
    D[:, 4] += (2/3) * tio2

    # SiO2 layer
    sio2 = 0.5 * (1 + erf((depth - z[3]) / sigma[2])) * 0.5 * (1 - erf((depth - z[4]) / sigma[3]))
    D[:, 3] = (1/3) * sio2
    D[:, 4] += (2/3) * sio2

    # Si substrate
    D[:, 5] = 0.5 * (1 + erf((depth - z[4]) / sigma[3]))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def create_test_data(
    thicknesses: List[float] = [5.0, 15.0, 35.0, 30.0],
    interface_widths: List[float] = [2.0, 3.0, 4.0, 5.0],
    noise_level: float = 0.02,
    seed: int = 42,
):
    """Create test data with transfer matrix."""
    np.random.seed(seed)
    depth = np.arange(0, 100, 4.0)
    angles = np.array([0, 10, 20, 30, 40, 50, 60])

    elements = [
        ElementConfig('C', '1s', 'C', oxidation_state=0),
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_4layer_profile(depth, thicknesses, interface_widths)

    tm = TransferMatrix(depth=depth, step_size=4.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    if noise_level > 0:
        noise = noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true + noise * I_true
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    else:
        I_obs = I_true.copy()

    return I_obs, tm, elements, D_true, depth


def wasserstein_distance(D_true: np.ndarray, D_opt: np.ndarray, depth: np.ndarray) -> float:
    """
    Compute Wasserstein-1 distance between true and optimized profiles.

    Average over all elements.
    """
    dz = np.mean(np.diff(depth))
    n_elements = D_true.shape[1]

    w1_total = 0.0
    for e in range(n_elements):
        # Normalize to distributions
        p = D_true[:, e] / (np.sum(D_true[:, e]) * dz + 1e-10)
        q = D_opt[:, e] / (np.sum(D_opt[:, e]) * dz + 1e-10)

        # CDF
        cdf_p = np.cumsum(p) * dz
        cdf_q = np.cumsum(q) * dz

        # W1 = integral |CDF_p - CDF_q|
        w1 = trapezoid(np.abs(cdf_p - cdf_q), depth)
        w1_total += w1

    return w1_total / n_elements


# =============================================================================
# Main Benchmark
# =============================================================================

def run_alpha_selection_benchmark(
    n_trials: int = 5,
    noise_level: float = 0.02,
    verbose: bool = True,
):
    """
    Run benchmark comparing L-curve and GCV alpha selection.

    Parameters
    ----------
    n_trials : int
        Number of random trials
    noise_level : float
        Gaussian noise level
    verbose : bool
        Print details
    """
    print("=" * 70)
    print("Alpha Selection Benchmark: L-curve vs GCV")
    print("=" * 70)
    print(f"Model: 4-layer C / Al2O3 / TiO2 / SiO2 / Si")
    print(f"Noise: {noise_level*100:.1f}% Gaussian")
    print(f"Trials: {n_trials}")
    print()

    # Semi-free config
    semi_free_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        thickness_bounds=(10.0, 90.0),
        free_indices=[0, 1, 2, 3],  # C, Al, Ti, Si_ox
        linked_indices=[4],  # O
        substrate_indices=[5],  # Si_bulk
    )

    # Alpha range to test
    alphas = np.logspace(-2, 2, 15)  # 0.01 to 100

    # Reference: manual optimal alpha (from previous tests)
    alpha_reference = 1.0

    # Results storage
    results = {
        'L-curve': {'alphas': [], 'w1': [], 'time': []},
        'GCV': {'alphas': [], 'w1': [], 'time': []},
        'Reference': {'alphas': [], 'w1': [], 'time': []},
    }

    for trial in range(n_trials):
        if verbose:
            print(f"\n--- Trial {trial + 1}/{n_trials} ---")

        # Create test data with new seed each trial
        I_obs, tm, elements, D_true, depth = create_test_data(
            noise_level=noise_level,
            seed=42 + trial,
        )

        # L-curve selection
        if verbose:
            print("\nL-curve method:")
        t0 = time.time()
        lcurve_result = l_curve_selection(
            I_obs, tm, semi_free_config, alphas, verbose=verbose
        )
        lcurve_time = time.time() - t0

        # Solve with L-curve alpha
        config_lc = SolverConfig(method='L1', alpha=lcurve_result.alpha_optimal, max_iter=100)
        solver_lc = SemiFreeSolver(tm, config_lc, semi_free_config, elements=elements)
        result_lc = solver_lc.solve(I_obs)
        w1_lc = wasserstein_distance(D_true, result_lc.D_opt, depth)

        results['L-curve']['alphas'].append(lcurve_result.alpha_optimal)
        results['L-curve']['w1'].append(w1_lc)
        results['L-curve']['time'].append(lcurve_time)

        # GCV selection
        if verbose:
            print("\nGCV method:")
        t0 = time.time()
        gcv_result = gcv_selection(
            I_obs, tm, semi_free_config, alphas, verbose=verbose
        )
        gcv_time = time.time() - t0

        # Solve with GCV alpha
        config_gcv = SolverConfig(method='L1', alpha=gcv_result.alpha_optimal, max_iter=100)
        solver_gcv = SemiFreeSolver(tm, config_gcv, semi_free_config, elements=elements)
        result_gcv = solver_gcv.solve(I_obs)
        w1_gcv = wasserstein_distance(D_true, result_gcv.D_opt, depth)

        results['GCV']['alphas'].append(gcv_result.alpha_optimal)
        results['GCV']['w1'].append(w1_gcv)
        results['GCV']['time'].append(gcv_time)

        # Reference (fixed alpha)
        t0 = time.time()
        config_ref = SolverConfig(method='L1', alpha=alpha_reference, max_iter=100)
        solver_ref = SemiFreeSolver(tm, config_ref, semi_free_config, elements=elements)
        result_ref = solver_ref.solve(I_obs)
        ref_time = time.time() - t0
        w1_ref = wasserstein_distance(D_true, result_ref.D_opt, depth)

        results['Reference']['alphas'].append(alpha_reference)
        results['Reference']['w1'].append(w1_ref)
        results['Reference']['time'].append(ref_time)

        if verbose:
            print(f"\nTrial {trial + 1} Results:")
            print(f"  L-curve: α={lcurve_result.alpha_optimal:.4f}, W1={w1_lc:.4f} Å, time={lcurve_time:.2f}s")
            print(f"  GCV:     α={gcv_result.alpha_optimal:.4f}, W1={w1_gcv:.4f} Å, time={gcv_time:.2f}s")
            print(f"  Ref:     α={alpha_reference:.4f}, W1={w1_ref:.4f} Å, time={ref_time:.2f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<12} {'Mean α':>10} {'Mean W1 (Å)':>12} {'Std W1':>10} {'Time (s)':>10}")
    print("-" * 54)

    for method in ['L-curve', 'GCV', 'Reference']:
        mean_alpha = np.mean(results[method]['alphas'])
        mean_w1 = np.mean(results[method]['w1'])
        std_w1 = np.std(results[method]['w1'])
        mean_time = np.mean(results[method]['time'])
        print(f"{method:<12} {mean_alpha:>10.4f} {mean_w1:>12.4f} {std_w1:>10.4f} {mean_time:>10.2f}")

    # Improvement analysis
    print("\n" + "-" * 54)
    ref_w1 = np.mean(results['Reference']['w1'])
    for method in ['L-curve', 'GCV']:
        method_w1 = np.mean(results[method]['w1'])
        improvement = (ref_w1 - method_w1) / ref_w1 * 100
        print(f"{method} vs Reference: {improvement:+.1f}% improvement")

    return results


def test_alpha_selection():
    """Run the alpha selection benchmark test."""
    results = run_alpha_selection_benchmark(n_trials=3, verbose=True)

    # Basic assertions
    assert 'L-curve' in results
    assert 'GCV' in results
    assert len(results['L-curve']['w1']) == 3
    assert len(results['GCV']['w1']) == 3

    print("\n[PASS] Alpha selection benchmark completed successfully")


# =============================================================================
# Parallel Benchmark
# =============================================================================

def run_single_trial(args):
    """
    Run a single trial (for parallel execution).

    Args:
        args: (trial_idx, noise_level, alphas, semi_free_config_dict, alpha_reference)

    Returns:
        dict with results for this trial
    """
    trial_idx, noise_level, alphas, semi_free_config_dict, alpha_reference = args

    # Recreate config (can't pickle SemiFreeSolverConfig directly in some cases)
    semi_free_config = SemiFreeSolverConfig(**semi_free_config_dict)

    # Create test data
    I_obs, tm, elements, D_true, depth = create_test_data(
        noise_level=noise_level,
        seed=42 + trial_idx,
    )

    results = {}

    # L-curve selection
    t0 = time.time()
    lcurve_result = l_curve_selection(I_obs, tm, semi_free_config, alphas, verbose=False)
    lcurve_time = time.time() - t0

    config_lc = SolverConfig(method='L1', alpha=lcurve_result.alpha_optimal, max_iter=100)
    solver_lc = SemiFreeSolver(tm, config_lc, semi_free_config, elements=elements)
    result_lc = solver_lc.solve(I_obs)
    w1_lc = wasserstein_distance(D_true, result_lc.D_opt, depth)

    results['L-curve'] = {
        'alpha': lcurve_result.alpha_optimal,
        'w1': w1_lc,
        'time': lcurve_time,
    }

    # GCV selection
    t0 = time.time()
    gcv_result = gcv_selection(I_obs, tm, semi_free_config, alphas, verbose=False)
    gcv_time = time.time() - t0

    config_gcv = SolverConfig(method='L1', alpha=gcv_result.alpha_optimal, max_iter=100)
    solver_gcv = SemiFreeSolver(tm, config_gcv, semi_free_config, elements=elements)
    result_gcv = solver_gcv.solve(I_obs)
    w1_gcv = wasserstein_distance(D_true, result_gcv.D_opt, depth)

    results['GCV'] = {
        'alpha': gcv_result.alpha_optimal,
        'w1': w1_gcv,
        'time': gcv_time,
    }

    # Reference (fixed alpha)
    t0 = time.time()
    config_ref = SolverConfig(method='L1', alpha=alpha_reference, max_iter=100)
    solver_ref = SemiFreeSolver(tm, config_ref, semi_free_config, elements=elements)
    result_ref = solver_ref.solve(I_obs)
    ref_time = time.time() - t0
    w1_ref = wasserstein_distance(D_true, result_ref.D_opt, depth)

    results['Reference'] = {
        'alpha': alpha_reference,
        'w1': w1_ref,
        'time': ref_time,
    }

    return trial_idx, results


def run_parallel_benchmark(
    n_trials: int = 20,
    noise_level: float = 0.02,
    n_workers: Optional[int] = None,
    use_threads: bool = True,
):
    """
    Run benchmark with parallel execution.

    Parameters
    ----------
    n_trials : int
        Number of random trials
    noise_level : float
        Gaussian noise level
    n_workers : int, optional
        Number of parallel workers (default: CPU count)
    use_threads : bool
        Use ThreadPoolExecutor (default) or ProcessPoolExecutor

    Returns
    -------
    results : dict
        Aggregated results
    elapsed : float
        Total elapsed time
    """
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    import multiprocessing

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), n_trials)

    executor_type = "ThreadPool" if use_threads else "ProcessPool"

    print("=" * 70)
    print(f"Alpha Selection Benchmark (PARALLEL - {executor_type})")
    print("=" * 70)
    print(f"Model: 4-layer C / Al2O3 / TiO2 / SiO2 / Si")
    print(f"Noise: {noise_level*100:.1f}% Gaussian")
    print(f"Trials: {n_trials}")
    print(f"Workers: {n_workers}")
    print()

    # Config as dict (for pickling)
    semi_free_config_dict = {
        'thickness_init': 70.0,
        'abruptness_init': 5.0,
        'thickness_bounds': (10.0, 90.0),
        'free_indices': [0, 1, 2, 3],
        'linked_indices': [4],
        'substrate_indices': [5],
    }

    alphas = np.logspace(-2, 2, 15)
    alpha_reference = 1.0

    # Prepare arguments
    args_list = [
        (i, noise_level, alphas, semi_free_config_dict, alpha_reference)
        for i in range(n_trials)
    ]

    # Run in parallel
    t_start = time.time()

    all_results = {}
    PoolExecutor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with PoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_trial, args): args[0] for args in args_list}

        completed = 0
        for future in as_completed(futures):
            trial_idx, trial_results = future.result()
            all_results[trial_idx] = trial_results
            completed += 1
            print(f"  Completed trial {trial_idx + 1}/{n_trials} ({completed}/{n_trials})")

    elapsed = time.time() - t_start

    # Aggregate results
    results = {
        'L-curve': {'alphas': [], 'w1': [], 'time': []},
        'GCV': {'alphas': [], 'w1': [], 'time': []},
        'Reference': {'alphas': [], 'w1': [], 'time': []},
    }

    for i in range(n_trials):
        for method in ['L-curve', 'GCV', 'Reference']:
            results[method]['alphas'].append(all_results[i][method]['alpha'])
            results[method]['w1'].append(all_results[i][method]['w1'])
            results[method]['time'].append(all_results[i][method]['time'])

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {elapsed:.2f}s (parallel with {n_workers} workers)")
    print(f"Time per trial (effective): {elapsed/n_trials:.2f}s")
    print()
    print(f"{'Method':<12} {'Mean α':>10} {'Mean W1 (Å)':>12} {'Std W1':>10} {'95% CI':>14}")
    print("-" * 60)

    for method in ['L-curve', 'GCV', 'Reference']:
        mean_alpha = np.mean(results[method]['alphas'])
        mean_w1 = np.mean(results[method]['w1'])
        std_w1 = np.std(results[method]['w1'])
        # 95% confidence interval
        ci_95 = 1.96 * std_w1 / np.sqrt(n_trials)
        print(f"{method:<12} {mean_alpha:>10.4f} {mean_w1:>12.4f} {std_w1:>10.4f} {mean_w1:.3f}±{ci_95:.3f}")

    # Improvement analysis
    print("\n" + "-" * 60)
    ref_w1 = np.mean(results['Reference']['w1'])
    for method in ['L-curve', 'GCV']:
        method_w1 = np.mean(results[method]['w1'])
        improvement = (ref_w1 - method_w1) / ref_w1 * 100
        print(f"{method} vs Reference: {improvement:+.1f}% improvement")

    return results, elapsed


def benchmark_scaling():
    """
    Benchmark scaling: compare sequential vs parallel for different n_trials.
    """
    print("=" * 70)
    print("SCALING BENCHMARK: Sequential vs Parallel")
    print("=" * 70)
    print()

    trial_counts = [5, 10, 20]

    print(f"{'n_trials':<10} {'Sequential (s)':>15} {'Parallel (s)':>15} {'Speedup':>10}")
    print("-" * 50)

    for n_trials in trial_counts:
        # Sequential
        t0 = time.time()
        run_alpha_selection_benchmark(n_trials=n_trials, verbose=False)
        seq_time = time.time() - t0

        # Parallel
        _, par_time = run_parallel_benchmark(n_trials=n_trials, noise_level=0.02)

        speedup = seq_time / par_time
        print(f"{n_trials:<10} {seq_time:>15.2f} {par_time:>15.2f} {speedup:>10.2f}x")

    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Alpha Selection Benchmark')
    parser.add_argument('--parallel', '-p', action='store_true', help='Use parallel execution')
    parser.add_argument('--process', action='store_true', help='Use ProcessPoolExecutor (default: ThreadPool)')
    parser.add_argument('--trials', '-n', type=int, default=5, help='Number of trials')
    parser.add_argument('--scaling', '-s', action='store_true', help='Run scaling benchmark')
    args = parser.parse_args()

    if args.scaling:
        benchmark_scaling()
    elif args.parallel:
        use_threads = not args.process
        run_parallel_benchmark(n_trials=args.trials, use_threads=use_threads)
    else:
        run_alpha_selection_benchmark(n_trials=args.trials, verbose=True)
