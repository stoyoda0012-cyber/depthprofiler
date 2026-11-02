"""
Batch Processor for ARXPS Analysis
===================================

Parallel processing of multiple samples or time-series data.

Features:
    - Multiprocessing for CPU-bound tasks
    - Optional GPU acceleration (CuPy)
    - Progress tracking with callbacks
    - Chunked processing for large datasets

Usage:
    from deppro.batch_processor import BatchProcessor, ParallelConfig

    processor = BatchProcessor(
        config=ParallelConfig(n_workers=4, use_gpu=False)
    )

    # Process multiple samples
    results = processor.process_batch(samples, solver_config)

    # Process with progress callback
    results = processor.process_batch(
        samples, solver_config,
        progress_callback=lambda i, n: print(f"{i}/{n}")
    )

Date: 2026-01-23

.. warning::
   The solving path (``process_batch`` and everything built on it:
   ``monte_carlo_batch``, ``bootstrap_batch``, ``parallel_monte_carlo``,
   ``parallel_bootstrap``) was written for ``MultiLayerFilmSolver``, which has
   been removed from ``deppro.inverse_solver``. These calls raise
   ``NotImplementedError`` until they are ported to a current solver. Before
   2026-10 they did not raise: every sample failed inside the worker and was
   only counted in ``BatchResult.n_failed``. ``GPUAccelerator`` is unaffected.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Union, Any, Tuple
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings

# Optional imports
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@dataclass
class ParallelConfig:
    """Configuration for parallel processing.

    Attributes:
        n_workers: Number of worker processes (None = auto)
        use_gpu: Whether to use GPU acceleration
        gpu_device: GPU device ID to use
        chunk_size: Number of items per chunk
        timeout: Timeout per task in seconds
        progress_interval: Interval for progress updates
    """
    n_workers: Optional[int] = None
    use_gpu: bool = False
    gpu_device: int = 0
    chunk_size: int = 100
    timeout: Optional[float] = None
    progress_interval: int = 10

    def __post_init__(self):
        if self.n_workers is None:
            self.n_workers = max(1, mp.cpu_count() - 1)

        if self.use_gpu and not HAS_CUPY:
            warnings.warn("CuPy not available, falling back to CPU")
            self.use_gpu = False


@dataclass
class BatchResult:
    """Results from batch processing.

    Attributes:
        results: List of individual results
        timing: Processing time per item
        errors: List of (index, error) tuples for failed items
        metadata: Additional metadata
    """
    results: List[Any] = field(default_factory=list)
    timing: List[float] = field(default_factory=list)
    errors: List[Tuple[int, Exception]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_success(self) -> int:
        return len(self.results)

    @property
    def n_failed(self) -> int:
        return len(self.errors)

    @property
    def success_rate(self) -> float:
        total = self.n_success + self.n_failed
        return self.n_success / total if total > 0 else 0.0

    @property
    def mean_time(self) -> float:
        return np.mean(self.timing) if self.timing else 0.0

    @property
    def total_time(self) -> float:
        return sum(self.timing)


SOLVER_UNAVAILABLE = (
    "deppro.batch_processor solves with MultiLayerFilmSolver, which has been removed "
    "from deppro.inverse_solver. process_batch / monte_carlo_batch / bootstrap_batch / "
    "parallel_monte_carlo / parallel_bootstrap are unavailable until they are ported to a "
    "current solver (for a single sample, use SemiFreeSolver as `deppro solve` does)."
)


def _solve_single(args: Tuple) -> Tuple[int, Any, float, Optional[Exception]]:
    """Worker function for parallel solving.

    Returns:
        (index, result, time, error)
    """
    # The body below targets the removed MultiLayerFilmSolver; kept for porting.
    raise NotImplementedError(SOLVER_UNAVAILABLE)
    import time
    idx, intensity_data, solver_config, transfer_matrix_builder = args

    t0 = time.time()
    error = None
    result = None

    try:
        # Build transfer matrix for this data
        tm = transfer_matrix_builder(intensity_data)

        # Import here to avoid pickling issues
        from .inverse_solver import MultiLayerFilmSolver

        solver = MultiLayerFilmSolver(
            transfer_matrix=tm,
            config=solver_config,
        )
        result = solver.solve(intensity_data.intensities)

    except Exception as e:
        error = e

    elapsed = time.time() - t0
    return idx, result, elapsed, error


class BatchProcessor:
    """Batch processor for ARXPS analysis.

    Supports parallel processing of multiple samples or Monte Carlo simulations.
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize batch processor.

        Args:
            config: Parallel processing configuration
        """
        self.config = config or ParallelConfig()

    def process_batch(
        self,
        data_list: List['IntensityData'],
        solver_config: 'MultiLayerFilmConfig',
        transfer_matrix_builder: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """Process multiple samples in parallel.

        Args:
            data_list: List of IntensityData objects
            solver_config: Solver configuration
            transfer_matrix_builder: Function to build transfer matrix from data
            progress_callback: Optional callback(current, total)

        Returns:
            BatchResult with all results

        Raises:
            NotImplementedError: always (see the module warning). The body is
            kept for porting to a current solver.
        """
        raise NotImplementedError(SOLVER_UNAVAILABLE)
        n_samples = len(data_list)
        batch_result = BatchResult()
        batch_result.metadata['n_samples'] = n_samples

        # Prepare arguments
        args_list = [
            (i, data, solver_config, transfer_matrix_builder)
            for i, data in enumerate(data_list)
        ]

        # Process in parallel
        results_dict = {}

        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {
                executor.submit(_solve_single, args): args[0]
                for args in args_list
            }

            completed = 0
            for future in as_completed(futures, timeout=self.config.timeout):
                idx, result, elapsed, error = future.result()

                if error is None:
                    results_dict[idx] = result
                    batch_result.timing.append(elapsed)
                else:
                    batch_result.errors.append((idx, error))

                completed += 1
                if progress_callback and completed % self.config.progress_interval == 0:
                    progress_callback(completed, n_samples)

        # Sort results by index
        batch_result.results = [
            results_dict[i] for i in sorted(results_dict.keys())
        ]

        if progress_callback:
            progress_callback(n_samples, n_samples)

        return batch_result

    def monte_carlo_batch(
        self,
        base_data: 'IntensityData',
        solver_config: 'MultiLayerFilmConfig',
        transfer_matrix_builder: Callable,
        noise_level: float,
        n_trials: int = 100,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        seed: Optional[int] = None,
    ) -> BatchResult:
        """Run Monte Carlo simulation with noise perturbation.

        Args:
            base_data: Base intensity data
            solver_config: Solver configuration
            transfer_matrix_builder: Function to build transfer matrix
            noise_level: Relative noise level (0.02 = 2%)
            n_trials: Number of Monte Carlo trials
            progress_callback: Optional callback(current, total)
            seed: Random seed for reproducibility

        Returns:
            BatchResult with Monte Carlo results
        """
        from .data_io import IntensityData

        if seed is not None:
            np.random.seed(seed)

        # Generate noisy samples
        noisy_data_list = []
        for _ in range(n_trials):
            noise = np.random.randn(*base_data.intensities.shape) * noise_level
            noisy_intensities = base_data.intensities * (1 + noise)
            noisy_intensities = np.maximum(noisy_intensities, 0)  # Ensure positive

            noisy_data = IntensityData(
                angles=base_data.angles.copy(),
                element_labels=base_data.element_labels.copy(),
                intensities=noisy_intensities,
                kinetic_energies=base_data.kinetic_energies.copy() if base_data.kinetic_energies is not None else None,
            )
            noisy_data_list.append(noisy_data)

        # Process batch
        result = self.process_batch(
            noisy_data_list,
            solver_config,
            transfer_matrix_builder,
            progress_callback,
        )

        result.metadata['noise_level'] = noise_level
        result.metadata['n_trials'] = n_trials
        result.metadata['seed'] = seed

        return result

    def bootstrap_batch(
        self,
        base_data: 'IntensityData',
        solver_config: 'MultiLayerFilmConfig',
        transfer_matrix_builder: Callable,
        n_bootstrap: int = 100,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        seed: Optional[int] = None,
    ) -> BatchResult:
        """Run bootstrap resampling for error estimation.

        Resamples angles with replacement.

        Args:
            base_data: Base intensity data
            solver_config: Solver configuration
            transfer_matrix_builder: Function to build transfer matrix
            n_bootstrap: Number of bootstrap samples
            progress_callback: Optional callback(current, total)
            seed: Random seed for reproducibility

        Returns:
            BatchResult with bootstrap results
        """
        from .data_io import IntensityData

        if seed is not None:
            np.random.seed(seed)

        n_angles = len(base_data.angles)

        # Generate bootstrap samples
        bootstrap_data_list = []
        for _ in range(n_bootstrap):
            # Resample angles with replacement
            indices = np.random.choice(n_angles, size=n_angles, replace=True)

            bootstrap_data = IntensityData(
                angles=base_data.angles[indices],
                element_labels=base_data.element_labels.copy(),
                intensities=base_data.intensities[indices],
                kinetic_energies=base_data.kinetic_energies.copy() if base_data.kinetic_energies is not None else None,
            )
            bootstrap_data_list.append(bootstrap_data)

        # Process batch
        result = self.process_batch(
            bootstrap_data_list,
            solver_config,
            transfer_matrix_builder,
            progress_callback,
        )

        result.metadata['n_bootstrap'] = n_bootstrap
        result.metadata['seed'] = seed

        return result


# ============================================================================
# GPU-Accelerated Operations
# ============================================================================

class GPUAccelerator:
    """GPU-accelerated matrix operations for transfer matrix calculations.

    Uses CuPy for GPU acceleration when available.
    """

    def __init__(self, device: int = 0):
        """Initialize GPU accelerator.

        Args:
            device: GPU device ID
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy is required for GPU acceleration")

        self.device = device
        cp.cuda.Device(device).use()

    def transfer_to_gpu(self, arr: np.ndarray) -> 'cp.ndarray':
        """Transfer array to GPU."""
        return cp.asarray(arr)

    def transfer_to_cpu(self, arr: 'cp.ndarray') -> np.ndarray:
        """Transfer array to CPU."""
        return cp.asnumpy(arr)

    def batch_matrix_multiply(
        self,
        A: np.ndarray,
        B_batch: np.ndarray,
    ) -> np.ndarray:
        """Batch matrix multiplication on GPU.

        Args:
            A: Matrix (m, n)
            B_batch: Batch of matrices (batch, n, k)

        Returns:
            Result (batch, m, k)
        """
        A_gpu = self.transfer_to_gpu(A)
        B_gpu = self.transfer_to_gpu(B_batch)

        # Batch matmul: (m, n) @ (batch, n, k) -> (batch, m, k)
        result_gpu = cp.einsum('mn,bnk->bmk', A_gpu, B_gpu)

        return self.transfer_to_cpu(result_gpu)

    def parallel_solve(
        self,
        transfer_matrices: np.ndarray,
        intensities_batch: np.ndarray,
    ) -> np.ndarray:
        """Solve multiple linear systems in parallel on GPU.

        Args:
            transfer_matrices: Batch of transfer matrices (batch, m, n)
            intensities_batch: Batch of intensity vectors (batch, m, k)

        Returns:
            Solutions (batch, n, k)
        """
        T_gpu = self.transfer_to_gpu(transfer_matrices)
        I_gpu = self.transfer_to_gpu(intensities_batch)

        # Use pseudo-inverse for overdetermined systems
        batch_size = T_gpu.shape[0]
        results = []

        for i in range(batch_size):
            # Compute pseudo-inverse and solve
            T_pinv = cp.linalg.pinv(T_gpu[i])
            x = T_pinv @ I_gpu[i]
            results.append(x)

        result_gpu = cp.stack(results, axis=0)
        return self.transfer_to_cpu(result_gpu)


# ============================================================================
# Convenience Functions
# ============================================================================

def parallel_monte_carlo(
    base_data: 'IntensityData',
    solver_config: 'MultiLayerFilmConfig',
    transfer_matrix_builder: Callable,
    noise_level: float,
    n_trials: int = 100,
    n_workers: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Convenience function for parallel Monte Carlo simulation.

    Args:
        base_data: Base intensity data
        solver_config: Solver configuration
        transfer_matrix_builder: Function to build transfer matrix
        noise_level: Relative noise level
        n_trials: Number of trials
        n_workers: Number of workers (None = auto)
        seed: Random seed

    Returns:
        Dict with 'thicknesses', 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    config = ParallelConfig(n_workers=n_workers)
    processor = BatchProcessor(config)

    result = processor.monte_carlo_batch(
        base_data,
        solver_config,
        transfer_matrix_builder,
        noise_level,
        n_trials,
        seed=seed,
    )

    # Extract thickness results
    thicknesses = np.array([
        r.thicknesses for r in result.results if r is not None
    ])

    if len(thicknesses) == 0:
        return {
            'thicknesses': np.array([]),
            'mean': np.array([]),
            'std': np.array([]),
            'ci_lower': np.array([]),
            'ci_upper': np.array([]),
            'success_rate': 0.0,
        }

    mean = np.mean(thicknesses, axis=0)
    std = np.std(thicknesses, axis=0)
    ci_lower = np.percentile(thicknesses, 2.5, axis=0)
    ci_upper = np.percentile(thicknesses, 97.5, axis=0)

    return {
        'thicknesses': thicknesses,
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'success_rate': result.success_rate,
    }


def parallel_bootstrap(
    base_data: 'IntensityData',
    solver_config: 'MultiLayerFilmConfig',
    transfer_matrix_builder: Callable,
    n_bootstrap: int = 100,
    n_workers: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Convenience function for parallel bootstrap estimation.

    Args:
        base_data: Base intensity data
        solver_config: Solver configuration
        transfer_matrix_builder: Function to build transfer matrix
        n_bootstrap: Number of bootstrap samples
        n_workers: Number of workers (None = auto)
        seed: Random seed

    Returns:
        Dict with 'thicknesses', 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    config = ParallelConfig(n_workers=n_workers)
    processor = BatchProcessor(config)

    result = processor.bootstrap_batch(
        base_data,
        solver_config,
        transfer_matrix_builder,
        n_bootstrap,
        seed=seed,
    )

    # Extract thickness results
    thicknesses = np.array([
        r.thicknesses for r in result.results if r is not None
    ])

    if len(thicknesses) == 0:
        return {
            'thicknesses': np.array([]),
            'mean': np.array([]),
            'std': np.array([]),
            'ci_lower': np.array([]),
            'ci_upper': np.array([]),
            'success_rate': 0.0,
        }

    mean = np.mean(thicknesses, axis=0)
    std = np.std(thicknesses, axis=0)
    ci_lower = np.percentile(thicknesses, 2.5, axis=0)
    ci_upper = np.percentile(thicknesses, 97.5, axis=0)

    return {
        'thicknesses': thicknesses,
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'success_rate': result.success_rate,
    }


if __name__ == '__main__':
    print("Batch Processor Module")
    print("=" * 40)
    print(f"\nCPU cores available: {mp.cpu_count()}")
    print(f"GPU acceleration (CuPy): {'Available' if HAS_CUPY else 'Not available'}")
    print("\nUsage:")
    print("  from deppro.batch_processor import BatchProcessor, ParallelConfig")
    print("  processor = BatchProcessor(ParallelConfig(n_workers=4))")
    print("  results = processor.monte_carlo_batch(data, config, builder, 0.02, 100)")
