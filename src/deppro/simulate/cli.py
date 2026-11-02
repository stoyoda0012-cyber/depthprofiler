"""
Dstk Simulator CLI
==================

Command-line interface for generating simulated ARXPS depth profiles.

Usage:
    # NN training data (Parquet)
    python -m deppro.simulate --n-samples 10000000 --preset layered_6elem \\
        --output train.parquet --workers 0

    # HAXPES configuration
    python -m deppro.simulate --n-samples 100000 --preset layered_6elem \\
        --photon-energy 8000 \\
        --angles 0,15,30,45,55,60,65,70,73,75,78,80,82,85 \\
        --depth-end 600 --step-size 30 --output haxpes_train.parquet

    # Benchmark H5 (MATLAB compatible)
    python -m deppro.simulate --n-samples 2500 --preset layered_6elem \\
        --output benchmark.h5 --format h5

    # With noise
    python -m deppro.simulate --n-samples 1000000 --noise 0.02 \\
        --output noisy.parquet

    # List available presets
    python -m deppro.simulate --list-presets
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .configs import SimulationConfig
from .presets import ElementSystemPreset, get_preset, list_presets


def _worker_generate(args: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Worker function for parallel generation.

    Each worker creates its own RNG, Sampler, and TransferMatrix
    to avoid pickle issues with complex objects.

    Returns
    -------
    D_batch : (N, n_depth, n_elem) float32
    I_batch : (N, n_angles, n_elem) float32
    thickness : (N,) float32
    abruptness : (N,) float32
    n_layers : (N,) int8
    """
    (worker_id, n_samples, seed, preset_name,
     depth_end, step_size, photon_energy, work_function, gamma_eff,
     angles, noise_level, noise_type,
     sampler_kwargs) = args

    # Local imports to avoid pickle issues
    from .presets import get_preset
    from .sampler import RandomProfileSampler
    from ..transfer_matrix import TransferMatrix
    from ..imfp import CompoundDatabase

    preset = get_preset(preset_name)
    depth = np.arange(0, depth_end + step_size, step_size)
    angles_arr = np.array(angles, dtype=np.float64)

    # Create per-worker RNG (deterministic from seed + worker_id)
    rng_seed = seed + worker_id * 1_000_000
    sampler = RandomProfileSampler(
        preset=preset,
        depth=depth,
        seed=rng_seed,
        **sampler_kwargs,
    )

    # Create TransferMatrix and compute lambda_basic once
    tm = TransferMatrix(
        depth=depth,
        step_size=step_size,
        angles=angles_arr,
        photon_energy=photon_energy,
        work_function=work_function,
        gamma_eff=gamma_eff,
    )
    compound_db = CompoundDatabase()
    lambda_basic = tm.calculate_lambda_basic(preset.elements, compound_db)

    # Use fused generate_batch for maximum throughput
    return sampler.generate_batch(
        n_samples, lambda_basic, angles_arr, step_size,
        gamma_eff=gamma_eff,
        noise_level=noise_level,
        noise_type=noise_type,
    )


def run_simulation(config: SimulationConfig, preset: ElementSystemPreset,
                   n_workers: int = 0, verbose: bool = True) -> None:
    """Run the full simulation pipeline.

    Parameters
    ----------
    config : SimulationConfig
    preset : ElementSystemPreset
    n_workers : int
        0 = auto (cpu_count), 1 = serial, N = N workers.
    verbose : bool
        Print progress messages.
    """
    depth = np.arange(0, config.depth_end + config.step_size, config.step_size)
    n_depth = len(depth)
    n_angles = len(config.angles)
    n_elem = preset.n_elements

    if verbose:
        print(f"Dstk Simulator")
        print(f"  Preset:        {preset.name} ({n_elem} elements)")
        print(f"  Samples:       {config.n_samples:,}")
        print(f"  Depth:         0-{config.depth_end} A, step={config.step_size} A "
              f"({n_depth} points)")
        print(f"  Angles:        {n_angles} ({config.angles})")
        print(f"  Photon energy: {config.photon_energy} eV")
        print(f"  Noise:         {config.noise_level} ({config.noise_type})"
              if config.noise_level > 0 else "  Noise:         none")
        print(f"  Output:        {config.output_path} ({config.output_format})")

    # Determine parallelism
    if n_workers == 0:
        n_workers = max(1, mp.cpu_count())
    if config.n_samples < 1000:
        n_workers = 1  # Not worth parallelizing small jobs

    if verbose:
        print(f"  Workers:       {n_workers}")
        print()

    t0 = time.time()

    # Sampler kwargs (exclude seed which is per-worker)
    sampler_kwargs = {
        'n_layers_range': config.n_layers_range,
        'thickness_range': config.thickness_range,
        'abruptness_range': config.abruptness_range,
        'max_abruptness_ratio': config.max_abruptness_ratio,
        'min_conc': config.min_conc,
        'surface_contamination_prob': config.surface_contamination_prob,
        'contamination_decay_range': config.contamination_decay_range,
    }

    # Split work across workers
    samples_per_worker = [config.n_samples // n_workers] * n_workers
    for i in range(config.n_samples % n_workers):
        samples_per_worker[i] += 1

    worker_args = [
        (
            worker_id, n_samples, config.seed, preset.name,
            config.depth_end, config.step_size,
            config.photon_energy, config.work_function, config.gamma_eff,
            config.angles, config.noise_level, config.noise_type,
            sampler_kwargs,
        )
        for worker_id, n_samples in enumerate(samples_per_worker)
        if n_samples > 0
    ]

    # Generate
    if n_workers == 1:
        results = [_worker_generate(worker_args[0])]
    else:
        with mp.Pool(n_workers) as pool:
            results = pool.map(_worker_generate, worker_args)

    t_gen = time.time() - t0

    # Concatenate results
    D_all = np.concatenate([r[0] for r in results], axis=0)
    I_all = np.concatenate([r[1] for r in results], axis=0)
    thickness_all = np.concatenate([r[2] for r in results])
    abruptness_all = np.concatenate([r[3] for r in results])
    n_layers_all = np.concatenate([r[4] for r in results])

    if verbose:
        rate = config.n_samples / t_gen
        print(f"Generation: {t_gen:.1f}s ({rate:,.0f} samples/s)")

    # Write output
    t_write = time.time()

    if config.output_format in ('parquet', 'both'):
        parquet_path = config.output_path
        if config.output_format == 'both' and not parquet_path.endswith('.parquet'):
            parquet_path = str(Path(config.output_path).with_suffix('.parquet'))

        from .writer import ParquetWriter
        with ParquetWriter(
            parquet_path, n_angles, n_depth, n_elem,
            batch_size=config.batch_size,
        ) as pw:
            # Write in batches
            for start in range(0, config.n_samples, config.batch_size):
                end = min(start + config.batch_size, config.n_samples)
                pw.write_batch(
                    I_all[start:end],
                    D_all[start:end],
                    thickness_all[start:end],
                    abruptness_all[start:end],
                    n_layers_all[start:end],
                )
        if verbose:
            size_mb = Path(parquet_path).stat().st_size / 1e6
            print(f"Parquet: {parquet_path} ({size_mb:.1f} MB, "
                  f"{pw.total_written:,} samples)")

    if config.output_format in ('h5', 'both'):
        h5_path = config.output_path
        if config.output_format == 'both' and not h5_path.endswith('.h5'):
            h5_path = str(Path(config.output_path).with_suffix('.h5'))

        from .writer import H5Writer
        hw = H5Writer(h5_path, preset)
        # Write in batches
        for start in range(0, config.n_samples, config.batch_size):
            end = min(start + config.batch_size, config.n_samples)
            hw.add_batch(D_all[start:end], I_all[start:end])
        hw.finalize(
            depth=depth,
            angles=np.array(config.angles, dtype=np.float64),
            step_size=config.step_size,
            photon_energy=config.photon_energy,
            work_function=config.work_function,
            gamma_eff=config.gamma_eff,
        )
        if verbose:
            size_mb = Path(h5_path).stat().st_size / 1e6
            print(f"H5:      {h5_path} ({size_mb:.1f} MB)")

    t_write_elapsed = time.time() - t_write
    t_total = time.time() - t0

    if verbose:
        print(f"Write:   {t_write_elapsed:.1f}s")
        print(f"Total:   {t_total:.1f}s")

        # Summary statistics
        print(f"\nProfile statistics:")
        print(f"  Thickness:  {thickness_all.mean():.1f} +/- "
              f"{thickness_all.std():.1f} A "
              f"(range {thickness_all.min():.1f}-{thickness_all.max():.1f})")
        print(f"  Abruptness: {abruptness_all.mean():.1f} +/- "
              f"{abruptness_all.std():.1f} A")
        print(f"  Layers:     {n_layers_all.mean():.1f} +/- "
              f"{n_layers_all.std():.1f} "
              f"(range {n_layers_all.min()}-{n_layers_all.max()})")
        print(f"  D sum:      {D_all.sum(axis=2).mean():.6f} "
              f"(should be 1.000000)")
        i_sum = I_all.sum(axis=2).mean()
        i_nan = np.isnan(I_all).any()
        print(f"  I_obs sum:  {i_sum:.6f} "
              f"(should be 1.000000)"
              f"{' *** HAS NaN ***' if i_nan else ''}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Dstk Simulator - Generate ARXPS depth profile training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (100 samples, SiO2/Si)
  python -m deppro.simulate --n-samples 100 --preset sio2_si --output test.parquet

  # NN training data (1M samples)
  python -m deppro.simulate --n-samples 1000000 --preset layered_6elem --output train.parquet

  # MATLAB-compatible benchmark
  python -m deppro.simulate --n-samples 2500 --preset layered_6elem --output bench.h5 --format h5

  # HAXPES with 14 angles
  python -m deppro.simulate --n-samples 100000 --preset layered_6elem \\
      --photon-energy 8000 --angles 0,15,30,45,55,60,65,70,73,75,78,80,82,85

  # With noise injection
  python -m deppro.simulate --n-samples 100000 --noise 0.02 --noise-type relative
""",
    )

    # Required
    parser.add_argument('--n-samples', type=int, default=10_000,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--preset', default='layered_6elem',
                        help='Element system preset (default: layered_6elem)')
    parser.add_argument('--output', default='train.parquet',
                        help='Output file path')

    # Output format
    parser.add_argument('--format', choices=['parquet', 'h5', 'both'],
                        default='parquet', help='Output format (default: parquet)')

    # Depth grid
    parser.add_argument('--depth-end', type=float, default=100.0,
                        help='Maximum depth in Angstroms (default: 100)')
    parser.add_argument('--step-size', type=float, default=5.0,
                        help='Depth step in Angstroms (default: 5)')

    # Physics
    parser.add_argument('--photon-energy', type=float, default=1486.6,
                        help='X-ray photon energy in eV (default: 1486.6)')
    parser.add_argument('--angles', type=str, default='0,15,30,45,60,75',
                        help='Observation angles in degrees, comma-separated')
    parser.add_argument('--work-function', type=float, default=4.5,
                        help='Work function in eV (default: 4.5)')

    # Sampling
    parser.add_argument('--thickness-range', type=str, default='10,500',
                        help='Thickness range in Angstroms (default: 10,500)')
    parser.add_argument('--abruptness-range', type=str, default='1,50',
                        help='Abruptness FWHM range in Angstroms (default: 1,50)')
    parser.add_argument('--n-layers-range', type=str, default='1,4',
                        help='Number of layers range (default: 1,4)')

    # Noise
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise level (default: 0, no noise)')
    parser.add_argument('--noise-type', choices=['relative', 'poisson'],
                        default='relative', help='Noise type (default: relative)')

    # Parallelism
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of workers (0=auto, default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--batch-size', type=int, default=10_000,
                        help='Batch size for writing (default: 10000)')

    # Utility
    parser.add_argument('--list-presets', action='store_true',
                        help='List available presets and exit')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args(argv)

    # List presets mode
    if args.list_presets:
        print("Available presets:")
        for name in list_presets():
            p = get_preset(name)
            print(f"  {name:15s}  {p.n_elements} elements  {p.description}")
        return

    # Parse comma-separated arguments
    angles = tuple(float(a) for a in args.angles.split(','))
    thickness_range = tuple(float(x) for x in args.thickness_range.split(','))
    abruptness_range = tuple(float(x) for x in args.abruptness_range.split(','))
    n_layers_range = tuple(int(x) for x in args.n_layers_range.split(','))

    # Auto-detect format from extension
    output_format = args.format
    if output_format == 'parquet' and args.output.endswith('.h5'):
        output_format = 'h5'
    elif output_format == 'parquet' and args.output.endswith('.parquet'):
        output_format = 'parquet'

    # Build config
    config = SimulationConfig(
        n_samples=args.n_samples,
        seed=args.seed,
        depth_end=args.depth_end,
        step_size=args.step_size,
        photon_energy=args.photon_energy,
        angles=angles,
        work_function=args.work_function,
        n_layers_range=n_layers_range,
        thickness_range=thickness_range,
        abruptness_range=abruptness_range,
        noise_level=args.noise,
        noise_type=args.noise_type,
        output_path=args.output,
        output_format=output_format,
        batch_size=args.batch_size,
    )

    # Get preset
    preset = get_preset(args.preset)

    # Validate photon energy: auto-upgrade if preset requires higher energy
    if config.photon_energy < preset.min_photon_energy:
        auto_energy = max(8000.0, preset.min_photon_energy)
        if not args.quiet:
            print(f"WARNING: {preset.name} requires photon energy >= "
                  f"{preset.min_photon_energy:.0f} eV (1s orbitals).")
            print(f"  Auto-upgrading: {config.photon_energy:.1f} -> "
                  f"{auto_energy:.1f} eV (HAXPES)")
        config.photon_energy = auto_energy

    # Run
    run_simulation(config, preset, n_workers=args.workers,
                   verbose=not args.quiet)
