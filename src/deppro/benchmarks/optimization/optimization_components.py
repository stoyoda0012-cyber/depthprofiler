"""
Optimization Components Benchmark
=================================

Measures the individual and combined effects of:
  1. Numba JIT compilation
  2. Analytical Jacobian

4 configurations tested:
  - NumPy + Numerical Jacobian (baseline)
  - NumPy + Analytical Jacobian (N/A - requires Numba)
  - Numba + Numerical Jacobian
  - Numba + Analytical Jacobian (full optimization)

Metrics:
  - Throughput (distributions/second)
  - Accuracy (W1 distance)
  - Convergence (iterations, success rate)
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.special import erf
from scipy.stats import wasserstein_distance
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import (
    SemiFreeSolver, SemiFreeSolverConfig, SolverConfig
)

# Check Numba availability
try:
    from deppro.numba_kernels import NUMBA_AVAILABLE, warmup_jit
except ImportError:
    NUMBA_AVAILABLE = False
    warmup_jit = lambda: None


@dataclass
class BenchmarkResult:
    """Result of a single benchmark configuration."""
    config_name: str
    use_numba: bool
    use_analytical_jac: bool
    n_trials: int
    n_success: int
    avg_time: float
    std_time: float
    avg_iterations: float
    avg_w1: float
    std_w1: float
    throughput: float  # distributions/second


def create_test_problem(noise_level: float = 0.02, seed: int = 42):
    """
    Create a 4-layer C/Al2O3/TiO2/SiO2/Si test problem.

    Returns:
        I_obs, tm, elements, D_true, depth
    """
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

    # Create 4-layer profile: C(5Å) / Al2O3(15Å) / TiO2(35Å) / SiO2(30Å) / Si
    thicknesses = [5.0, 15.0, 35.0, 30.0]
    interface_widths = [2.0, 3.0, 4.0, 5.0]
    sigma = [w / (2 * np.sqrt(np.log(2))) for w in interface_widths]
    z = [0] + list(np.cumsum(thicknesses))

    D_true = np.zeros((len(depth), 6))

    # C layer
    D_true[:, 0] = 0.5 * (1 - erf((depth - z[1]) / sigma[0]))

    # Al2O3 layer
    al2o3 = 0.5 * (1 + erf((depth - z[1]) / sigma[0])) * 0.5 * (1 - erf((depth - z[2]) / sigma[1]))
    D_true[:, 1] = (2/5) * al2o3
    D_true[:, 4] = (3/5) * al2o3

    # TiO2 layer
    tio2 = 0.5 * (1 + erf((depth - z[2]) / sigma[1])) * 0.5 * (1 - erf((depth - z[3]) / sigma[2]))
    D_true[:, 2] = (1/3) * tio2
    D_true[:, 4] += (2/3) * tio2

    # SiO2 layer
    sio2 = 0.5 * (1 + erf((depth - z[3]) / sigma[2])) * 0.5 * (1 - erf((depth - z[4]) / sigma[3]))
    D_true[:, 3] = (1/3) * sio2
    D_true[:, 4] += (2/3) * sio2

    # Si substrate
    D_true[:, 5] = 0.5 * (1 + erf((depth - z[4]) / sigma[3]))

    # Normalize
    D_true = np.clip(D_true, 1e-6, None)
    D_true = D_true / D_true.sum(axis=1, keepdims=True)

    # Transfer matrix
    tm = TransferMatrix(depth=depth, step_size=4.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    # Observed intensity with noise
    I_true = tm.forward(D_true, normalize=True)

    if noise_level > 0:
        noise = noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true + noise * I_true
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    else:
        I_obs = I_true.copy()

    return I_obs, tm, elements, D_true, depth


def compute_w1_distance(D_true: np.ndarray, D_est: np.ndarray, depth: np.ndarray) -> float:
    """Compute Wasserstein-1 distance between two depth profiles."""
    w1_sum = 0.0
    for i in range(D_true.shape[1]):
        w1_sum += wasserstein_distance(depth, depth, D_true[:, i], D_est[:, i])
    return w1_sum / D_true.shape[1]


def run_single_trial(
    use_numba: bool,
    use_analytical_jac: bool,
    seed: int,
    noise_level: float = 0.02,
):
    """
    Run a single trial with specified configuration.

    Args:
        use_numba: Whether to use Numba JIT
        use_analytical_jac: Whether to use analytical Jacobian
        seed: Random seed
        noise_level: Noise level

    Returns:
        (elapsed_time, iterations, success, w1_distance or None)
    """
    # Create test problem
    I_obs, tm, elements, D_true, depth = create_test_problem(
        noise_level=noise_level, seed=seed
    )

    # Solver config
    config = SolverConfig(method='L1', alpha=1.0, max_iter=100)
    semi_free_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        thickness_bounds=(10.0, 90.0),
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
    )

    # Create solver
    solver = SemiFreeSolver(
        tm, config, semi_free_config,
        elements=elements,
        use_numba=use_numba,
    )

    # If we want Numba kernels but numerical Jacobian, we need to patch
    if use_numba and not use_analytical_jac:
        # Temporarily make solver not use analytical Jacobian
        original_use_numba = solver.use_numba

        def patched_solve(I_obs_inner):
            """Solve with Numba kernels but numerical Jacobian."""
            # The solve method checks self.use_numba for jac=gradient
            # We need to intercept the minimize call

            import deppro.inverse_solver as solver_module

            original_minimize = solver_module.minimize

            def patched_minimize(fun, x0, method=None, jac=None, **kwargs):
                # Force numerical Jacobian
                return original_minimize(fun, x0, method=method, jac=None, **kwargs)

            solver_module.minimize = patched_minimize
            try:
                result = solver.solve.__wrapped__(solver, I_obs_inner) if hasattr(solver.solve, '__wrapped__') else type(solver).solve(solver, I_obs_inner)
            except:
                # Fallback: directly call parent solve
                from scipy.optimize import minimize as scipy_minimize
                solver_module.minimize = scipy_minimize
                # We need a different approach - modify the solver's internal flag
                pass
            finally:
                solver_module.minimize = original_minimize

            return result

        # Actually, let's use a simpler approach: subclass with override
        pass

    # Simple approach: just run solve with the given use_numba setting
    # For Numba + Numerical Jacobian case, we'll use monkey patching

    t0 = time.perf_counter()

    if use_numba and not use_analytical_jac:
        # Patch to disable analytical Jacobian while keeping Numba kernels
        import deppro.inverse_solver as solver_module
        from scipy.optimize import minimize as original_minimize

        call_count = [0]
        original_jac_arg = [None]

        def minimize_no_jac(fun, x0, method=None, jac=None, **kwargs):
            call_count[0] += 1
            original_jac_arg[0] = jac
            # Call with jac=None to force numerical differentiation
            return original_minimize(fun, x0, method=method, jac=None, **kwargs)

        # Patch minimize in the solver module
        old_minimize = solver_module.minimize
        solver_module.minimize = minimize_no_jac
        try:
            result = solver.solve(I_obs)
        finally:
            solver_module.minimize = old_minimize
    else:
        result = solver.solve(I_obs)

    elapsed = time.perf_counter() - t0

    if result.success:
        w1 = compute_w1_distance(D_true, result.D_opt, depth)
    else:
        w1 = None

    return elapsed, result.n_iterations, result.success, w1


def run_benchmark_config(
    config_name: str,
    use_numba: bool,
    use_analytical_jac: bool,
    n_trials: int = 50,
    noise_level: float = 0.02,
) -> BenchmarkResult:
    """
    Run benchmark for a specific configuration.

    Args:
        config_name: Name of this configuration
        use_numba: Whether to use Numba JIT
        use_analytical_jac: Whether to use analytical Jacobian
        n_trials: Number of trials
        noise_level: Noise level for synthetic data

    Returns:
        BenchmarkResult with metrics
    """
    times = []
    iterations = []
    w1_distances = []
    successes = 0

    for trial in range(n_trials):
        elapsed, n_iter, success, w1 = run_single_trial(
            use_numba=use_numba,
            use_analytical_jac=use_analytical_jac,
            seed=42 + trial,
            noise_level=noise_level,
        )

        times.append(elapsed)
        iterations.append(n_iter)

        if success:
            successes += 1
            if w1 is not None:
                w1_distances.append(w1)

    # Calculate metrics
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_iterations = np.mean(iterations)
    avg_w1 = np.mean(w1_distances) if w1_distances else np.nan
    std_w1 = np.std(w1_distances) if w1_distances else np.nan
    throughput = n_trials / sum(times)

    return BenchmarkResult(
        config_name=config_name,
        use_numba=use_numba,
        use_analytical_jac=use_analytical_jac,
        n_trials=n_trials,
        n_success=successes,
        avg_time=avg_time,
        std_time=std_time,
        avg_iterations=avg_iterations,
        avg_w1=avg_w1,
        std_w1=std_w1,
        throughput=throughput,
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print results as a formatted table."""
    print("\n" + "=" * 95)
    print("OPTIMIZATION COMPONENTS BENCHMARK RESULTS")
    print("=" * 95)

    # Header
    print(f"\n{'Configuration':<30} {'Numba':<8} {'Anal.Jac':<10} {'Throughput':<14} {'W1 (Å)':<14} {'Success':<10} {'Iter':<8}")
    print("-" * 95)

    for r in results:
        numba_str = "Yes" if r.use_numba else "No"
        jac_str = "Yes" if r.use_analytical_jac else "No"
        w1_str = f"{r.avg_w1:.2f} +/- {r.std_w1:.2f}" if not np.isnan(r.avg_w1) else "N/A"
        success_str = f"{r.n_success}/{r.n_trials}"

        print(f"{r.config_name:<30} {numba_str:<8} {jac_str:<10} {r.throughput:>8.1f} d/s   {w1_str:<14} {success_str:<10} {r.avg_iterations:>5.1f}")

    print("-" * 95)


def print_speedup_analysis(results: List[BenchmarkResult]):
    """Print speedup analysis relative to baseline."""
    print("\n" + "=" * 65)
    print("SPEEDUP ANALYSIS (relative to NumPy + Numerical Jacobian)")
    print("=" * 65)

    baseline = results[0]  # NumPy + Numerical

    print(f"\nBaseline: {baseline.config_name}")
    print(f"  Throughput: {baseline.throughput:.2f} dist/s")
    print(f"  W1 distance: {baseline.avg_w1:.3f} Å")

    for r in results[1:]:
        speedup = r.throughput / baseline.throughput
        w1_diff = r.avg_w1 - baseline.avg_w1
        w1_rel = (r.avg_w1 / baseline.avg_w1 - 1) * 100 if not np.isnan(baseline.avg_w1) else 0

        print(f"\n{r.config_name}:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  W1 diff: {w1_diff:+.3f} Å ({w1_rel:+.1f}%)")


def print_component_effects(results: List[BenchmarkResult]):
    """Analyze individual component effects."""
    print("\n" + "=" * 65)
    print("COMPONENT EFFECTS ANALYSIS")
    print("=" * 65)

    # results order: [NumPy+Num, Numba+Num, Numba+Anal]
    if len(results) < 3:
        print("Not enough results for component analysis")
        return

    numpy_num = results[0]
    numba_num = results[1]
    numba_anal = results[2]

    # Numba effect (keeping numerical Jacobian)
    numba_speedup = numba_num.throughput / numpy_num.throughput
    numba_w1_diff = numba_num.avg_w1 - numpy_num.avg_w1

    print(f"\n1. Numba JIT Effect (objective + kernels only):")
    print(f"   Speedup: {numba_speedup:.2f}x")
    print(f"   W1 change: {numba_w1_diff:+.3f} Å")
    print(f"   -> Numba accelerates objective/gradient computation")

    # Analytical Jacobian effect (with Numba)
    anal_speedup = numba_anal.throughput / numba_num.throughput
    anal_w1_diff = numba_anal.avg_w1 - numba_num.avg_w1

    print(f"\n2. Analytical Jacobian Effect (with Numba):")
    print(f"   Speedup: {anal_speedup:.2f}x")
    print(f"   W1 change: {anal_w1_diff:+.3f} Å")
    print(f"   -> Eliminates finite-difference gradient calls")

    # Combined effect
    combined_speedup = numba_anal.throughput / numpy_num.throughput
    combined_w1_diff = numba_anal.avg_w1 - numpy_num.avg_w1

    # Expected vs actual
    expected_speedup = numba_speedup * anal_speedup
    synergy = combined_speedup / expected_speedup if expected_speedup > 0 else 1.0

    print(f"\n3. Combined Effect (Numba + Analytical Jacobian):")
    print(f"   Total speedup: {combined_speedup:.2f}x")
    print(f"   W1 change: {combined_w1_diff:+.3f} Å")
    print(f"   Expected (multiplicative): {expected_speedup:.2f}x")
    print(f"   Synergy factor: {synergy:.2f}")


def run_all_benchmarks(n_trials: int = 50):
    """Run complete benchmark suite."""
    print("=" * 75)
    print("OPTIMIZATION COMPONENTS BENCHMARK")
    print("Numba JIT vs Analytical Jacobian Effects")
    print("=" * 75)

    if NUMBA_AVAILABLE:
        print("\nNumba Status: AVAILABLE")
        import numba
        print(f"Numba version: {numba.__version__}")
        print("Warming up JIT...")
        warmup_jit()
    else:
        print("\nNumba Status: NOT INSTALLED")
        print("Only NumPy configuration will be tested.")

    print(f"\nTrials per configuration: {n_trials}")
    print("Noise level: 2%")
    print("Model: 4-layer C/Al2O3/TiO2/SiO2/Si")

    results = []

    # Configuration 1: NumPy + Numerical Jacobian (baseline)
    print("\n[1/3] Testing: NumPy + Numerical Jacobian (baseline)...")
    r1 = run_benchmark_config(
        "NumPy + Numerical Jac",
        use_numba=False,
        use_analytical_jac=False,
        n_trials=n_trials,
    )
    results.append(r1)
    print(f"      Done. {r1.throughput:.1f} dist/s, W1={r1.avg_w1:.2f} Å, Success={r1.n_success}/{r1.n_trials}")

    if NUMBA_AVAILABLE:
        # Configuration 2: Numba + Numerical Jacobian
        print("\n[2/3] Testing: Numba + Numerical Jacobian...")
        r2 = run_benchmark_config(
            "Numba + Numerical Jac",
            use_numba=True,
            use_analytical_jac=False,
            n_trials=n_trials,
        )
        results.append(r2)
        print(f"      Done. {r2.throughput:.1f} dist/s, W1={r2.avg_w1:.2f} Å, Success={r2.n_success}/{r2.n_trials}")

        # Configuration 3: Numba + Analytical Jacobian (full optimization)
        print("\n[3/3] Testing: Numba + Analytical Jacobian (full opt)...")
        r3 = run_benchmark_config(
            "Numba + Analytical Jac",
            use_numba=True,
            use_analytical_jac=True,
            n_trials=n_trials,
        )
        results.append(r3)
        print(f"      Done. {r3.throughput:.1f} dist/s, W1={r3.avg_w1:.2f} Å, Success={r3.n_success}/{r3.n_trials}")
    else:
        print("\n[2/3] Skipped: Numba not available")
        print("[3/3] Skipped: Numba not available")

    # Print results
    print_results_table(results)

    if len(results) >= 3:
        print_speedup_analysis(results)
        print_component_effects(results)

    # Summary table (Markdown format)
    print("\n" + "=" * 65)
    print("SUMMARY TABLE (Markdown)")
    print("=" * 65)

    print("\n| Configuration | Numba | Anal.Jac | Throughput | W1 (Å) | Speedup |")
    print("|---------------|-------|----------|------------|--------|---------|")

    baseline_tp = results[0].throughput
    for r in results:
        numba = "Yes" if r.use_numba else "No"
        jac = "Yes" if r.use_analytical_jac else "No"
        speedup = r.throughput / baseline_tp
        print(f"| {r.config_name} | {numba} | {jac} | {r.throughput:.1f} d/s | {r.avg_w1:.2f} | {speedup:.1f}x |")

    print("\n" + "=" * 75)
    print("BENCHMARK COMPLETE")
    print("=" * 75)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimization Components Benchmark")
    parser.add_argument("-n", "--trials", type=int, default=50, help="Number of trials per config")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 trials")
    args = parser.parse_args()

    n_trials = 10 if args.quick else args.trials
    results = run_all_benchmarks(n_trials=n_trials)
