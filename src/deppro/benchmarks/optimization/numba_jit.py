"""
Numba JIT Benchmark
===================

Measures the performance improvement from Numba JIT compilation
of the inverse solver's hot path functions.

The hot path is objective() together with _build_full_profile() and
_build_substrate_profile(); objective() is called many times per solve.
"""

import numpy as np
import time
from scipy.special import erf as scipy_erf
from math import sqrt, log

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import (
    SemiFreeSolver, SemiFreeSolverConfig, SolverConfig
)

# Import Numba kernels
try:
    from deppro.numba_kernels import (
        build_substrate_profile_numba,
        build_full_profile_numba,
        compute_forward_model_numba,
        normalize_by_substrate_numba,
        compute_chi2_numba,
        objective_kernel_numba,
        is_numba_available,
        warmup_jit,
        NUMBA_AVAILABLE,
    )
except ImportError as e:
    print(f"Warning: Could not import numba_kernels: {e}")
    NUMBA_AVAILABLE = False


# =============================================================================
# Pure NumPy Reference Implementations
# =============================================================================

def build_substrate_profile_numpy(
    depth: np.ndarray,
    thickness: float,
    abruptness: float,
    substrate_ratios: np.ndarray,
) -> np.ndarray:
    """NumPy reference implementation."""
    sigma = abruptness / (2.0 * np.sqrt(np.log(2.0)))
    erf_profile = 0.5 * (1.0 + scipy_erf((depth - thickness) / sigma))
    return np.outer(erf_profile, substrate_ratios)


def build_full_profile_numpy(
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
    """NumPy reference implementation."""
    n_depth = len(depth)
    D = np.zeros((n_depth, n_elements))

    # 1. Substrate profile
    D_sub = build_substrate_profile_numpy(depth, thickness, abruptness, substrate_ratios)
    D[:, sub_idx] = D_sub

    # 2. Free elements
    substrate_sum = D_sub.sum(axis=1, keepdims=True)
    film_scale = np.maximum(1.0 - substrate_sum, min_conc)
    D[:, free_idx] = free_conc * film_scale

    # 3. Linked elements
    if len(linked_idx) > 0:
        D_free = D[:, free_idx]
        coeff = free_linked_const - 1
        D_linked = (D_free * coeff).sum(axis=1)
        for linked_e in linked_idx:
            D[:, linked_e] = D_linked / len(linked_idx)

    return np.clip(D, min_conc, max_conc)


def compute_forward_model_numpy(
    D: np.ndarray,
    T_matrices: np.ndarray,  # (n_elements, n_angles, n_depth)
) -> np.ndarray:
    """NumPy reference implementation."""
    n_angles = T_matrices.shape[1]
    n_elements = D.shape[1]
    I_pred = np.zeros((n_angles, n_elements))

    for e in range(n_elements):
        I_pred[:, e] = T_matrices[e] @ D[:, e]

    return I_pred


# =============================================================================
# Test Data Creation
# =============================================================================

def create_test_data(noise_level: float = 0.02, seed: int = 42):
    """Create test data for benchmark."""
    from scipy.special import erf

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

    # Create 4-layer profile
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

    D_true = np.clip(D_true, 1e-6, None)
    D_true = D_true / D_true.sum(axis=1, keepdims=True)

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


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_substrate_profile(n_iterations: int = 10000):
    """Benchmark substrate profile computation."""
    print("\n" + "=" * 60)
    print("Benchmark: build_substrate_profile")
    print("=" * 60)

    # Test data
    n_depth = 100
    depth = np.linspace(0, 100, n_depth)
    thickness = 50.0
    abruptness = 5.0
    substrate_ratios = np.array([0.333, 0.667])

    # Warmup
    if NUMBA_AVAILABLE:
        warmup_jit()

    # NumPy reference
    print(f"\nIterations: {n_iterations}")

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = build_substrate_profile_numpy(depth, thickness, abruptness, substrate_ratios)
    numpy_time = time.perf_counter() - t0
    print(f"  NumPy:    {numpy_time:.4f}s ({n_iterations / numpy_time:.0f} iter/s)")

    if not NUMBA_AVAILABLE:
        print("  Numba:    [NOT AVAILABLE]")
        return

    # Numba JIT
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = build_substrate_profile_numba(depth, thickness, abruptness, substrate_ratios)
    numba_time = time.perf_counter() - t0
    print(f"  Numba:    {numba_time:.4f}s ({n_iterations / numba_time:.0f} iter/s)")

    speedup = numpy_time / numba_time
    print(f"\n  Speedup:  {speedup:.1f}x")

    # Verify correctness
    result_numpy = build_substrate_profile_numpy(depth, thickness, abruptness, substrate_ratios)
    result_numba = build_substrate_profile_numba(depth, thickness, abruptness, substrate_ratios)
    max_diff = np.abs(result_numpy - result_numba).max()
    print(f"  Max diff: {max_diff:.2e} (should be ~0)")


def benchmark_full_profile(n_iterations: int = 10000):
    """Benchmark full profile computation."""
    print("\n" + "=" * 60)
    print("Benchmark: build_full_profile")
    print("=" * 60)

    # Test data
    n_depth = 100
    n_elements = 6
    n_free = 4
    n_sub = 2

    depth = np.linspace(0, 100, n_depth)
    free_conc = np.random.rand(n_depth, n_free) * 0.5
    thickness = 50.0
    abruptness = 5.0
    substrate_ratios = np.array([0.333, 0.667])

    free_idx = np.array([0, 1, 2, 3], dtype=np.int64)
    sub_idx = np.array([4, 5], dtype=np.int64)
    linked_idx = np.array([], dtype=np.int64)
    free_linked_const = np.array([1.0, 1.0, 1.0, 1.0])
    min_conc = 1e-6
    max_conc = 1.0

    # Warmup
    if NUMBA_AVAILABLE:
        warmup_jit()

    print(f"\nIterations: {n_iterations}")

    # NumPy reference
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = build_full_profile_numpy(
            free_conc, thickness, abruptness, substrate_ratios,
            depth, n_elements, free_idx, sub_idx, linked_idx,
            free_linked_const, min_conc, max_conc
        )
    numpy_time = time.perf_counter() - t0
    print(f"  NumPy:    {numpy_time:.4f}s ({n_iterations / numpy_time:.0f} iter/s)")

    if not NUMBA_AVAILABLE:
        print("  Numba:    [NOT AVAILABLE]")
        return

    # Numba JIT
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = build_full_profile_numba(
            free_conc, thickness, abruptness, substrate_ratios,
            depth, n_elements, free_idx, sub_idx, linked_idx,
            free_linked_const, min_conc, max_conc
        )
    numba_time = time.perf_counter() - t0
    print(f"  Numba:    {numba_time:.4f}s ({n_iterations / numba_time:.0f} iter/s)")

    speedup = numpy_time / numba_time
    print(f"\n  Speedup:  {speedup:.1f}x")

    # Verify correctness
    result_numpy = build_full_profile_numpy(
        free_conc, thickness, abruptness, substrate_ratios,
        depth, n_elements, free_idx, sub_idx, linked_idx,
        free_linked_const, min_conc, max_conc
    )
    result_numba = build_full_profile_numba(
        free_conc, thickness, abruptness, substrate_ratios,
        depth, n_elements, free_idx, sub_idx, linked_idx,
        free_linked_const, min_conc, max_conc
    )
    max_diff = np.abs(result_numpy - result_numba).max()
    print(f"  Max diff: {max_diff:.2e} (should be ~0)")


def benchmark_forward_model(n_iterations: int = 10000):
    """Benchmark forward model computation."""
    print("\n" + "=" * 60)
    print("Benchmark: compute_forward_model")
    print("=" * 60)

    # Test data
    n_depth = 100
    n_angles = 10
    n_elements = 6

    D = np.random.rand(n_depth, n_elements)
    # T_matrices shape: (n_elements, n_angles, n_depth)
    T_matrices = np.random.rand(n_elements, n_angles, n_depth)

    # Warmup
    if NUMBA_AVAILABLE:
        warmup_jit()

    print(f"\nIterations: {n_iterations}")

    # NumPy reference
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = compute_forward_model_numpy(D, T_matrices)
    numpy_time = time.perf_counter() - t0
    print(f"  NumPy:    {numpy_time:.4f}s ({n_iterations / numpy_time:.0f} iter/s)")

    if not NUMBA_AVAILABLE:
        print("  Numba:    [NOT AVAILABLE]")
        return

    # Numba JIT
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = compute_forward_model_numba(D, T_matrices, n_angles, n_elements)
    numba_time = time.perf_counter() - t0
    print(f"  Numba:    {numba_time:.4f}s ({n_iterations / numba_time:.0f} iter/s)")

    speedup = numpy_time / numba_time
    print(f"\n  Speedup:  {speedup:.1f}x")

    # Verify correctness
    result_numpy = compute_forward_model_numpy(D, T_matrices)
    result_numba = compute_forward_model_numba(D, T_matrices, n_angles, n_elements)
    max_diff = np.abs(result_numpy - result_numba).max()
    print(f"  Max diff: {max_diff:.2e} (should be ~0)")


def benchmark_full_solve():
    """
    Benchmark complete solve with and without Numba optimization.

    This measures the end-to-end performance that includes:
    - Profile building (called many times per solve)
    - Forward model computation
    - Chi-squared calculation
    - Regularization
    """
    print("\n" + "=" * 60)
    print("Benchmark: Full SemiFreeSolver.solve()")
    print("=" * 60)

    # Create test data
    print("\nCreating test problem...")
    I_obs, tm, elements, D_true, depth = create_test_data()

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

    # Warmup
    if NUMBA_AVAILABLE:
        print("Warming up Numba JIT...")
        warmup_jit()

    # Benchmark
    n_solves = 10
    print(f"\nRunning {n_solves} solves...")

    solver = SemiFreeSolver(tm, config, semi_free_config, elements=elements)

    times = []
    for i in range(n_solves):
        t0 = time.perf_counter()
        result = solver.solve(I_obs)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if i == 0:
            print(f"  Solve 1: {elapsed:.3f}s (iterations: {result.n_iterations})")

    avg_time = np.mean(times[1:])  # Exclude first (might include JIT overhead)
    std_time = np.std(times[1:])

    print(f"\n  Average: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  Rate:    {1/avg_time:.2f} distributions/second")


def benchmark_parallel_numba(n_distributions: int = 100):
    """
    Benchmark ProcessPool parallelization combined with Numba JIT.

    This simulates processing multiple distributions in parallel,
    which is the target use case for large-scale ARXPS analysis.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print("\n" + "=" * 60)
    print("Benchmark: ProcessPool + Numba (Parallel Distributions)")
    print("=" * 60)

    n_workers = min(16, os.cpu_count() or 4)

    print(f"\nDistributions: {n_distributions}")
    print(f"Workers: {n_workers}")

    # Time for parallel processing
    print("\nRunning parallel benchmark...")
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_distributions):
            # Each distribution has a slightly different noise seed
            future = executor.submit(solve_single_distribution, i)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            results.append(future.result())

    parallel_time = time.perf_counter() - t0

    # Calculate metrics
    total_solve_time = sum(r[1] for r in results)
    avg_solve_time = total_solve_time / n_distributions
    parallel_rate = n_distributions / parallel_time
    efficiency = (total_solve_time / parallel_time) / n_workers * 100

    print(f"\n  Total time:        {parallel_time:.2f}s")
    print(f"  Distributions:     {n_distributions}")
    print(f"  Rate:              {parallel_rate:.2f} dist/s")
    print(f"  Avg solve time:    {avg_solve_time:.3f}s")
    print(f"  Parallel efficiency: {efficiency:.1f}%")


def solve_single_distribution(seed: int):
    """Solve a single distribution (for parallel execution)."""
    I_obs, tm, elements, D_true, depth = create_test_data(noise_level=0.02, seed=42 + seed)

    config = SolverConfig(method='L1', alpha=1.0, max_iter=100)
    semi_free_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        thickness_bounds=(10.0, 90.0),
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
    )

    solver = SemiFreeSolver(tm, config, semi_free_config, elements=elements, use_numba=True)

    t0 = time.perf_counter()
    result = solver.solve(I_obs)
    elapsed = time.perf_counter() - t0

    return seed, elapsed, result.success


def run_all_benchmarks():
    """Run all Numba JIT benchmarks."""
    print("=" * 70)
    print("NUMBA JIT BENCHMARK SUITE")
    print("=" * 70)

    if NUMBA_AVAILABLE:
        print("\nNumba Status: AVAILABLE ✓")
        import numba
        print(f"Numba version: {numba.__version__}")
    else:
        print("\nNumba Status: NOT INSTALLED ✗")
        print("Install with: pip install numba")

    # Individual kernel benchmarks
    benchmark_substrate_profile(n_iterations=10000)
    benchmark_full_profile(n_iterations=10000)
    benchmark_forward_model(n_iterations=10000)

    # Full solver benchmark
    benchmark_full_solve()

    # Parallel + Numba benchmark - multiple scales
    for n_dist in [100, 500, 1000, 2000]:
        benchmark_parallel_numba(n_distributions=n_dist)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
