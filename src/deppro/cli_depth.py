#!/usr/bin/env python3
"""
CLI for Depth Profile ↔ Spectra Conversion
==========================================

depthprofile2spectra: Generate XPS spectra from depth profiles
spectra2depthprofile: Reconstruct depth profiles from angle-resolved spectra

Usage:
    # Solve inverse problem: intensity data -> depth profile
    python -m deppro.cli_depth solve \
        --input intensity.csv \
        --config model.yaml \
        --output results/

    # Solve with default multilayer model
    python -m deppro.cli_depth solve \
        --input intensity.csv \
        --model multilayer

    # Generate spectra from depth profile
    python -m deppro.cli_depth generate \
        --profile profile.json \
        --angles 0,30,45,60,75 \
        --output spectra/ \
        --noise-level 100

    # Generate with five-layer test profile
    python -m deppro.cli_depth generate \
        --five-layer \
        --angles 0,30,45,60,75 \
        --output spectra/

    # Show depth profile info
    python -m deppro.cli_depth info --profile profile.json

    # Create five-layer test profile
    python -m deppro.cli_depth create-five-layer --output five_layer_profile.json

    # Create sample configuration file
    python -m deppro.cli_depth init-config --model multilayer --output config.yaml
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional
import time


def parse_angles(angles_str: str) -> np.ndarray:
    """Parse comma-separated angles string to array."""
    try:
        return np.array([float(a.strip()) for a in angles_str.split(',')])
    except ValueError as e:
        raise ValueError(f"Invalid angles format: {angles_str}. Use comma-separated numbers.") from e


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate spectra from depth profile."""
    from .depth_profile import DepthProfile, create_five_layer_profile
    from .depth_generator import DepthSpectraGenerator, DepthGeneratorConfig

    print("=" * 60)
    print("depthprofile2spectra - Depth Profile to Spectra Generator")
    print("=" * 60)

    # Load or create profile
    if args.five_layer:
        print("\nUsing five-layer test profile (Si/SiO2/TiO2/Al2O3/C multi-layer)...")
        profile = create_five_layer_profile(depth_end=args.depth_end, step_size=args.step_size)
    elif args.profile:
        print(f"\nLoading profile from: {args.profile}")
        profile = DepthProfile.load(args.profile)
    else:
        print("Error: Must specify --profile or --five-layer")
        return 1

    print(f"  Elements: {profile.element_labels}")
    print(f"  Depth: {profile.depth[0]:.1f} - {profile.depth[-1]:.1f} Å ({profile.n_depth} points)")

    # Parse angles
    angles = parse_angles(args.angles)
    print(f"\nObservation angles: {angles}")

    # Configure generator
    config = DepthGeneratorConfig(
        photon_energy=args.photon_energy,
        noise_level=args.noise_level,
        noise_type=args.noise_type,
        energy_edge=args.energy_edge,
        energy_step=args.energy_step,
    )

    print(f"\nConfiguration:")
    print(f"  Photon energy: {config.photon_energy} eV")
    print(f"  Noise level: {config.noise_level}")
    print(f"  Noise type: {config.noise_type}")
    print(f"  Energy edge: {config.energy_edge} eV")
    print(f"  Energy step: {config.energy_step} eV")

    # Create generator
    generator = DepthSpectraGenerator(profile, angles, config)

    # Print angle-resolved intensities
    print(f"\nAngle-resolved intensities (normalized):")
    header = f"{'Angle':>6} | " + " | ".join(f"{l:>6}" for l in profile.element_labels)
    print(header)
    print("-" * len(header))
    for i, theta in enumerate(angles):
        vals = " | ".join(f"{v:>6.3f}" for v in generator.intensities[i])
        print(f"{theta:>6.0f} | {vals}")

    # Generate spectra
    output_dir = Path(args.output)
    project_name = args.project or "depth_profile"

    print(f"\nGenerating spectra to: {output_dir}/")
    start_time = time.time()

    files = generator.generate(
        output_dir,
        noise_level=args.noise_level,
        project_name=project_name,
    )

    elapsed = time.time() - start_time
    print(f"\nGenerated {len(files)} files in {elapsed:.2f}s")

    # List output structure
    print(f"\nOutput structure:")
    for angle in angles:
        angle_dir = output_dir / f"angle_{int(angle):02d}deg"
        print(f"  {angle_dir.name}/")
        for elem in profile.element_labels:
            print(f"    {elem}_{project_name}.h5")

    print(f"  {project_name}.h5 (metadata)")
    print(f"  {project_name}_profile.json (depth profile)")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show depth profile info."""
    from .depth_profile import DepthProfile

    print("=" * 60)
    print("Depth Profile Info")
    print("=" * 60)

    profile = DepthProfile.load(args.profile)

    print(f"\nFile: {args.profile}")
    print(f"\nDepth:")
    print(f"  Range: {profile.depth[0]:.1f} - {profile.depth[-1]:.1f} Å")
    print(f"  Points: {profile.n_depth}")
    print(f"  Step size: {profile.step_size} Å")

    print(f"\nElements ({profile.n_elements}):")
    for i, elem in enumerate(profile.elements):
        print(f"  {i+1}. {elem.label}")
        print(f"     Symbol: {elem.symbol}, Orbital: {elem.orbital}")
        print(f"     Compound: {elem.compound}")
        print(f"     Binding energy: {elem.binding_energy} eV")
        print(f"     Gaussian FWHM: {elem.gaussian_fwhm} eV")
        print(f"     Lorentzian FWHM: {elem.lorentzian_fwhm} eV")

    print(f"\nConcentration summary:")
    for i, label in enumerate(profile.element_labels):
        conc = profile.concentrations[:, i]
        print(f"  {label}: min={conc.min():.3f}, max={conc.max():.3f}, mean={conc.mean():.3f}")

    return 0


def cmd_create_five_layer(args: argparse.Namespace) -> int:
    """Create five-layer test profile."""
    from .depth_profile import create_five_layer_profile

    print("=" * 60)
    print("Create Five-layer Test Profile")
    print("=" * 60)

    profile = create_five_layer_profile(
        depth_end=args.depth_end,
        step_size=args.step_size,
    )

    output = Path(args.output)
    if output.suffix == '.json':
        profile.save_json(output)
    elif output.suffix in ('.h5', '.hdf5'):
        profile.save_h5(output)
    else:
        # Default to JSON
        output = output.with_suffix('.json')
        profile.save_json(output)

    print(f"\nCreated five-layer profile: {output}")
    print(f"  Elements: {profile.element_labels}")
    print(f"  Depth: {profile.depth[0]:.1f} - {profile.depth[-1]:.1f} Å")
    print(f"\nLayer structure:")
    print("  1. Surface contamination (C): 0-5 Å")
    print("  2. Al2O3 layer: 5-15 Å")
    print("  3. TiO2 layer: 15-35 Å")
    print("  4. SiO2 layer: 35-60 Å")
    print("  5. Si substrate: 60+ Å")

    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Plot depth profile."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting")
        return 1

    from .depth_profile import DepthProfile

    profile = DepthProfile.load(args.profile)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, label in enumerate(profile.element_labels):
        ax.plot(profile.depth, profile.concentrations[:, i], label=label, linewidth=2)

    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('Concentration')
    ax.set_title(f'Depth Profile: {args.profile}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(profile.depth[0], profile.depth[-1])
    ax.set_ylim(0, 1.05)

    output = args.output or f"{Path(args.profile).stem}_plot.png"
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved plot to: {output}")

    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    """Solve inverse problem: intensity data -> depth profile.

    Uses SemiFreeSolver (MATLAB GUI compatible):
    - Substrate follows erf profile
    - Film elements freely optimized at each depth
    - Linked elements determined by charge balance
    """
    from .data_io import (
        load_intensity_csv,
        load_config_yaml,
        create_default_config,
        save_profile_csv,
    )
    from .transfer_matrix import TransferMatrix
    from .inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig
    from .depth_profile import ElementConfig

    print("=" * 60)
    print("deppro solve - Depth Profile Reconstruction (SemiFreeSolver)")
    print("=" * 60)

    # Load configuration
    if args.config:
        print(f"\nLoading configuration: {args.config}")
        config = load_config_yaml(args.config)
    elif args.model:
        print(f"\nUsing default model: {args.model}")
        config = create_default_config(args.model)
    else:
        print("\nUsing default multilayer model")
        config = create_default_config('multilayer')

    # Load input data
    input_path = args.input or config.input_file
    if not input_path:
        print("Error: No input file specified. Use --input or set input_file in config.")
        return 1

    print(f"Loading intensity data: {input_path}")
    try:
        intensity_data = load_intensity_csv(input_path)
    except Exception as e:
        print(f"Error loading input: {e}")
        return 1

    print(f"  Angles: {intensity_data.angles}")
    print(f"  Elements: {intensity_data.element_labels}")
    print(f"  Shape: {intensity_data.intensities.shape}")

    # Build ElementConfig list from config or labels (needed before cross-section correction)
    elements = []
    if config.elements:
        for elem_def in config.elements:
            elements.append(ElementConfig(
                symbol=elem_def.symbol,
                orbital=elem_def.orbital,
                compound=elem_def.compound or elem_def.symbol,
                binding_energy=elem_def.binding_energy,
                oxidation_state=elem_def.oxidation_state,
            ))
    else:
        # Fallback: use kinetic energies from CSV and generic compound
        for i, label in enumerate(intensity_data.element_labels):
            symbol = label.split('_')[0] if '_' in label else label
            be = 0.0
            if intensity_data.kinetic_energies is not None:
                be = config.model.photon_energy - intensity_data.kinetic_energies[i]
            elements.append(ElementConfig(
                symbol=symbol, orbital='1s', compound=symbol,
                binding_energy=be,
            ))

    print(f"  Elements: {[f'{e.symbol}({e.compound})' for e in elements]}")

    # Cross-section correction (like MATLAB: Xsmpsk = Ismp ./ repmat(crosssection, nAngles, 1))
    if config.model.cross_section_correction:
        from .haxpes.cross_section_loader import CrossSectionDatabase
        from .data_io import IntensityData
        db = CrossSectionDatabase()
        cross_sections = np.array([
            db.get_cross_section(
                el.symbol, el.orbital, config.model.photon_energy,
                source=config.model.cross_section_source
            ).cross_section_Mb
            for el in elements
        ])
        intensity_data = IntensityData(
            angles=intensity_data.angles,
            element_labels=intensity_data.element_labels,
            intensities=intensity_data.intensities / cross_sections[np.newaxis, :],
            kinetic_energies=intensity_data.kinetic_energies,
        )
        print(f"  Cross-section correction applied ({config.model.cross_section_source}):")
        for el, cs in zip(elements, cross_sections):
            print(f"    {el.symbol}({el.orbital}): {cs:.4e} Mb")

    # Normalize if requested
    if config.normalize_input:
        intensity_data = intensity_data.normalize(config.normalization_mode)
        print(f"  Normalized: {config.normalization_mode}")

    # Set up depth grid
    depth = np.arange(
        config.model.depth_start,
        config.model.depth_end + config.model.depth_step,
        config.model.depth_step,
    )
    n_depth = len(depth)
    n_elements = intensity_data.n_elements
    step_size = config.model.depth_step
    print(f"\nDepth grid: {depth[0]:.1f} - {depth[-1]:.1f} Å ({n_depth} points)")

    # Create transfer matrix with compound-specific IMFP
    tm = TransferMatrix(
        depth=depth,
        step_size=step_size,
        angles=intensity_data.angles,
        photon_energy=config.model.photon_energy,
    )

    # Initialize with uniform profile for transfer matrix computation
    D_init = np.full((n_depth, n_elements), 1.0 / n_elements)
    tm.compute(D_init, elements)

    print(f"  Transfer matrix: {tm.trans_mat.shape}")
    print(f"  Lambda basic (diagonal):")
    for i, el in enumerate(elements):
        print(f"    {el.symbol}({el.compound}): {tm.lambda_basic[i, i]:.2f} Å")

    # Configure solver
    solver_config = SolverConfig(
        method=config.model.regularization_method,
        alpha=config.model.regularization_alpha,
        max_iter=config.model.max_iterations,
        tol=config.model.tolerance,
        scipy_method=config.model.solver_method,
        use_xreg_weighting=getattr(config.model, 'xreg_weighting', False),
        use_sum_constraint=getattr(config.model, 'use_sum_constraint', True),
        use_charge_constraint=getattr(config.model, 'use_charge_constraint', True),
    )

    # Determine initial thickness (interface depth where substrate begins)
    if config.model.thickness_init is not None:
        thickness_init = config.model.thickness_init
    elif config.model.layer_thicknesses:
        # Sum all layer thicknesses except the last (substrate) as initial guess
        thickness_init = sum(config.model.layer_thicknesses[:-1]) if len(config.model.layer_thicknesses) > 1 else config.model.layer_thicknesses[0]
    else:
        thickness_init = depth[-1] / 3
    thickness_max = depth[-1] * 0.9

    semi_free_config = SemiFreeSolverConfig(
        free_indices=config.model.free_indices,
        linked_indices=config.model.linked_indices,
        substrate_indices=config.model.substrate_indices,
        thickness_init=thickness_init,
        abruptness_init=config.model.abruptness_init if config.model.abruptness_init is not None else (config.model.interface_widths[0] if config.model.interface_widths else 30.0),
        optimize_thickness=True,
        optimize_abruptness=False,
        thickness_bounds=(config.model.thickness_bounds[0], min(config.model.thickness_bounds[1], thickness_max)),
        use_charge_constraint=True,
        use_sum_constraint=True,
    )

    print(f"\nSolver: SemiFreeSolver ({config.model.solver_method})")
    print(f"Regularization: {config.model.regularization_method} (alpha={config.model.regularization_alpha})")
    if solver_config.use_xreg_weighting:
        print(f"Weighting: Xreg (MATLAB-compatible, emphasize trace elements)")
    print(f"Initial thickness: {thickness_init:.1f} Å")
    print(f"Free: {config.model.free_indices}, Linked: {config.model.linked_indices}, Substrate: {config.model.substrate_indices}")

    # Solve (multi-loop: rebuild transfer matrix + decay alpha, like MATLAB GUI)
    n_loops = getattr(args, 'loops', 1)
    alpha_current = config.model.regularization_alpha
    alpha_decay = 0.8  # MATLAB uses degree = degree * 0.8 per loop

    print(f"\nOptimizing ({n_loops} loop(s), alpha={alpha_current}, decay={alpha_decay})...")

    result = None
    for loop_i in range(n_loops):
        if loop_i > 0:
            # Rebuild transfer matrix from current D_opt
            print(f"\n--- Loop {loop_i + 1}/{n_loops}: rebuilding transfer matrix ---")
            tm.compute(result.D_opt, elements)

            # Decay alpha
            alpha_current *= alpha_decay
            solver_config = SolverConfig(
                method=solver_config.method,
                alpha=alpha_current,
                max_iter=solver_config.max_iter,
                tol=solver_config.tol,
                scipy_method=solver_config.scipy_method,
                use_xreg_weighting=solver_config.use_xreg_weighting,
                use_sum_constraint=solver_config.use_sum_constraint,
                use_charge_constraint=solver_config.use_charge_constraint,
            )
            print(f"  Alpha: {alpha_current:.4f}")

        solver = SemiFreeSolver(
            transfer_matrix=tm,
            config=solver_config,
            semi_free_config=semi_free_config,
            D_prior=None,  # Zero prior for L1 sparsity
            elements=elements,
            use_numba=True,
        )

        result = solver.solve(intensity_data.intensities, verbose=config.verbose)
        print(f"  Loop {loop_i + 1}: thickness={result.thickness:.2f} Å, chi²={result.chi2:.4f}")

    # Results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Thickness: {result.thickness:.2f} Å")
    print(f"Chi²: {result.chi2:.6f}")

    # Save results
    output_dir = Path(args.output or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.save_profile_csv:
        csv_path = output_dir / 'profile.csv'
        save_profile_csv(depth, result.D_opt, intensity_data.element_labels, csv_path)
        print(f"\nSaved: {csv_path}")

    if config.save_profile_json:
        json_path = output_dir / 'result.json'
        # SemiFreeSolverResult has no layer fields, so the JSON is written here
        # (save_result_json is for layer-model results)
        import json
        result_dict = {
            'success': result.success,
            'thickness': float(result.thickness),
            'chi2': float(result.chi2),
            'depth': depth.tolist(),
            'element_labels': intensity_data.element_labels,
            'D_opt': result.D_opt.tolist(),
            'solver': 'SemiFreeSolver',
            'method': config.model.regularization_method,
            'alpha': config.model.regularization_alpha,
        }
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"Saved: {json_path}")

    if config.save_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Depth profile
            for i, label in enumerate(intensity_data.element_labels):
                ax1.plot(depth, result.D_opt[:, i], label=label, linewidth=2)

            ax1.set_xlabel('Depth (Å)')
            ax1.set_ylabel('Concentration')
            ax1.set_title('Reconstructed Depth Profile (SemiFreeSolver)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, depth[-1])
            ax1.set_ylim(0, 1.05)

            # Mark thickness
            ax1.axvline(result.thickness, color='gray', linestyle='--', alpha=0.5,
                       label=f'thickness={result.thickness:.1f} Å')

            # Fit quality: observed vs simulated
            I_sim = np.zeros_like(intensity_data.intensities)
            for e in range(n_elements):
                I_sim[:, e] = tm.trans_mat[:, :, e] @ result.D_opt[:, e]
            # Normalize
            for a in range(I_sim.shape[0]):
                row_sum = I_sim[a].sum()
                if row_sum > 0:
                    I_sim[a] /= row_sum

            for i, label in enumerate(intensity_data.element_labels):
                ax2.scatter(intensity_data.angles, intensity_data.intensities[:, i],
                           label=f'{label} obs', alpha=0.7)
                ax2.plot(intensity_data.angles, I_sim[:, i],
                        label=f'{label} fit', linestyle='--')

            ax2.set_xlabel('Angle (deg)')
            ax2.set_ylabel('Normalized Intensity')
            ax2.set_title(f'Fit Quality (Chi²={result.chi2:.4f})')
            ax2.legend(ncol=2, fontsize=7)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = output_dir / 'result.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved: {plot_path}")

        except ImportError:
            print("Warning: matplotlib not available, skipping plot")

    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    """Create sample configuration file."""
    from .data_io import create_default_config, save_config_yaml

    print("=" * 60)
    print("Create Configuration File")
    print("=" * 60)

    model = args.model or 'multilayer'
    output = Path(args.output or f'{model}_config.yaml')

    config = create_default_config(model)
    save_config_yaml(config, output)

    print(f"\nCreated: {output}")
    print(f"Model type: {model}")
    print(f"\nEdit this file to customize:")
    print("  - Layer thicknesses (initial guess)")
    print("  - Interface widths (recommend keeping fixed)")
    print("  - Solver parameters")
    print(f"\nThen run: python -m deppro.cli_depth solve --config {output}")

    return 0


def _batch_solver_unavailable(command: str) -> int:
    from .batch_processor import SOLVER_UNAVAILABLE
    print(f"deppro {command}: unavailable. {SOLVER_UNAVAILABLE}", file=sys.stderr)
    return 1


def cmd_batch(args: argparse.Namespace) -> int:
    """Process multiple files in batch (unavailable: see batch_processor)."""
    return _batch_solver_unavailable("batch")
    # The body below targets the removed MultiLayerFilmSolver; kept for porting.
    from .data_io import (
        load_intensity_csv,
        load_config_yaml,
        create_default_config,
        save_result_json,
    )
    from .batch_processor import BatchProcessor, ParallelConfig

    print("=" * 60)
    print("deppro batch - Batch Processing")
    print("=" * 60)

    # Find input files
    input_dir = Path(args.input_dir)
    pattern = args.pattern or '*.csv'
    files = sorted(input_dir.glob(pattern))

    if not files:
        print(f"No files found matching {input_dir}/{pattern}")
        return 1

    print(f"\nFound {len(files)} input files")

    # Load configuration
    if args.config:
        config = load_config_yaml(args.config)
    else:
        config = create_default_config('multilayer')

    # Setup parallel processing
    parallel_config = ParallelConfig(
        n_workers=args.workers,
        chunk_size=args.chunk_size,
    )
    processor = BatchProcessor(parallel_config)

    print(f"\nParallel config:")
    print(f"  Workers: {parallel_config.n_workers}")
    print(f"  Chunk size: {parallel_config.chunk_size}")

    # Load all data
    print("\nLoading data files...")
    data_list = []
    for f in files:
        try:
            data = load_intensity_csv(f)
            data_list.append(data)
            print(f"  {f.name}: {data.n_angles} angles, {data.n_elements} elements")
        except Exception as e:
            print(f"  {f.name}: ERROR - {e}")

    if not data_list:
        print("No valid data files loaded")
        return 1

    # Define transfer matrix builder (simplified for batch processing)
    def build_transfer_matrix(intensity_data):
        from .transfer_matrix import TransferMatrix
        from .imfp import IMFP

        depth = np.arange(
            config.model.depth_start,
            config.model.depth_end + config.model.depth_step,
            config.model.depth_step,
        )

        tm = TransferMatrix(
            depth=depth,
            step_size=config.model.depth_step,
            angles=intensity_data.angles,
            photon_energy=config.model.photon_energy,
        )
        tm.n_elements = intensity_data.n_elements

        # Build transfer matrices
        tm.trans_mat = np.zeros((tm.n_angles, tm.n_depth, tm.n_elements))
        for e in range(tm.n_elements):
            ke = config.model.photon_energy - 100  # Simplified
            lambda_e = IMFP.tpp2m(ke, Nv=4, density=3.0, Aw=28.0, Eg=5.0)

            for a, angle in enumerate(intensity_data.angles):
                cos_theta = np.cos(np.radians(angle))
                lambda_eff = lambda_e * cos_theta
                for d in range(tm.n_depth):
                    tm.trans_mat[a, d, e] = np.exp(-depth[d] / lambda_eff) * config.model.depth_step

        return tm

    # Setup solver config
    from .inverse_solver import MultiLayerFilmConfig, SolverConfig

    ml_config = MultiLayerFilmConfig(
        n_layers=config.model.n_layers,
        layer_thicknesses=config.model.layer_thicknesses,
        interface_widths=config.model.interface_widths,
        optimize_thicknesses=config.model.optimize_thicknesses,
        optimize_abruptnesses=config.model.optimize_widths,
    )

    solver_config = SolverConfig(
        method=config.model.regularization_method,
        alpha=config.model.regularization_alpha,
        max_iter=config.model.max_iterations,
    )

    # Process batch
    print(f"\nProcessing {len(data_list)} files...")
    start_time = time.time()

    def progress_cb(current, total):
        print(f"  Progress: {current}/{total}")

    result = processor.process_batch(
        data_list,
        solver_config,
        build_transfer_matrix,
        progress_callback=progress_cb,
    )

    elapsed = time.time() - start_time

    # Report results
    print("\n" + "=" * 60)
    print("Batch Results")
    print("=" * 60)
    print(f"Success: {result.n_success}/{len(data_list)}")
    print(f"Failed: {result.n_failed}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average time per file: {result.mean_time*1000:.1f} ms")

    # Save results
    output_dir = Path(args.output or './batch_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / 'batch_summary.json'
    import json
    summary = {
        'n_files': len(files),
        'n_success': result.n_success,
        'n_failed': result.n_failed,
        'total_time': elapsed,
        'mean_time': result.mean_time,
        'files': [str(f) for f in files],
        'errors': [(i, str(e)) for i, e in result.errors],
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary: {summary_path}")

    return 0


def cmd_monte_carlo(args: argparse.Namespace) -> int:
    """Run Monte Carlo error estimation (unavailable: see batch_processor)."""
    return _batch_solver_unavailable("monte-carlo")
    # The body below targets the removed MultiLayerFilmSolver; kept for porting.
    from .data_io import load_intensity_csv, load_config_yaml, create_default_config
    from .batch_processor import parallel_monte_carlo

    print("=" * 60)
    print("deppro monte-carlo - Error Estimation")
    print("=" * 60)

    # Load data
    print(f"\nLoading data: {args.input}")
    intensity_data = load_intensity_csv(args.input)
    print(f"  Angles: {intensity_data.angles}")
    print(f"  Elements: {intensity_data.element_labels}")

    # Load configuration
    if args.config:
        config = load_config_yaml(args.config)
    else:
        config = create_default_config('multilayer')

    # Define transfer matrix builder
    def build_transfer_matrix(data):
        from .transfer_matrix import TransferMatrix
        from .imfp import IMFP

        depth = np.arange(
            config.model.depth_start,
            config.model.depth_end + config.model.depth_step,
            config.model.depth_step,
        )

        tm = TransferMatrix(
            depth=depth,
            step_size=config.model.depth_step,
            angles=data.angles,
            photon_energy=config.model.photon_energy,
        )
        tm.n_elements = data.n_elements

        tm.trans_mat = np.zeros((tm.n_angles, tm.n_depth, tm.n_elements))
        for e in range(tm.n_elements):
            ke = config.model.photon_energy - 100
            lambda_e = IMFP.tpp2m(ke, Nv=4, density=3.0, Aw=28.0, Eg=5.0)

            for a, angle in enumerate(data.angles):
                cos_theta = np.cos(np.radians(angle))
                lambda_eff = lambda_e * cos_theta
                for d in range(tm.n_depth):
                    tm.trans_mat[a, d, e] = np.exp(-depth[d] / lambda_eff) * config.model.depth_step

        return tm

    # Setup solver config
    from .inverse_solver import MultiLayerFilmConfig, SolverConfig

    ml_config = MultiLayerFilmConfig(
        n_layers=config.model.n_layers,
        layer_thicknesses=config.model.layer_thicknesses,
        interface_widths=config.model.interface_widths,
    )

    solver_config = SolverConfig(
        method=config.model.regularization_method,
        alpha=config.model.regularization_alpha,
        max_iter=config.model.max_iterations,
    )

    # Run Monte Carlo
    noise_level = args.noise / 100.0  # Convert % to fraction
    n_trials = args.trials

    print(f"\nMonte Carlo parameters:")
    print(f"  Noise level: {args.noise}%")
    print(f"  Trials: {n_trials}")
    print(f"  Workers: {args.workers or 'auto'}")
    print(f"  Seed: {args.seed or 'random'}")

    print("\nRunning Monte Carlo simulation...")
    start_time = time.time()

    mc_result = parallel_monte_carlo(
        intensity_data,
        solver_config,
        build_transfer_matrix,
        noise_level,
        n_trials,
        n_workers=args.workers,
        seed=args.seed,
    )

    elapsed = time.time() - start_time

    # Results
    print("\n" + "=" * 60)
    print("Monte Carlo Results")
    print("=" * 60)
    print(f"Success rate: {mc_result['success_rate']*100:.1f}%")
    print(f"Total time: {elapsed:.2f}s")

    if len(mc_result['mean']) > 0:
        print(f"\nLayer thickness estimates:")
        for i in range(len(mc_result['mean'])):
            mean = mc_result['mean'][i]
            std = mc_result['std'][i]
            ci_low = mc_result['ci_lower'][i]
            ci_high = mc_result['ci_upper'][i]
            print(f"  Layer {i+1}: {mean:.2f} ± {std:.2f} Å (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")

        # Save results
        output_dir = Path(args.output or './monte_carlo')
        output_dir.mkdir(parents=True, exist_ok=True)

        import json
        result_path = output_dir / 'monte_carlo_result.json'
        result_data = {
            'noise_level_percent': args.noise,
            'n_trials': n_trials,
            'success_rate': float(mc_result['success_rate']),
            'mean_thickness': mc_result['mean'].tolist(),
            'std_thickness': mc_result['std'].tolist(),
            'ci_lower': mc_result['ci_lower'].tolist(),
            'ci_upper': mc_result['ci_upper'].tolist(),
        }
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\nSaved: {result_path}")

        # Plot if requested
        if args.plot:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, len(mc_result['mean']), figsize=(4*len(mc_result['mean']), 4))
                if len(mc_result['mean']) == 1:
                    axes = [axes]

                for i, ax in enumerate(axes):
                    ax.hist(mc_result['thicknesses'][:, i], bins=30, edgecolor='black', alpha=0.7)
                    ax.axvline(mc_result['mean'][i], color='red', linestyle='--', label=f'Mean: {mc_result["mean"][i]:.2f}')
                    ax.axvline(mc_result['ci_lower'][i], color='gray', linestyle=':', alpha=0.7)
                    ax.axvline(mc_result['ci_upper'][i], color='gray', linestyle=':', alpha=0.7)
                    ax.set_xlabel('Thickness (Å)')
                    ax.set_ylabel('Count')
                    ax.set_title(f'Layer {i+1}')
                    ax.legend()

                plt.tight_layout()
                plot_path = output_dir / 'monte_carlo_histogram.png'
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"Saved: {plot_path}")

            except ImportError:
                print("Warning: matplotlib not available, skipping plot")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Depth Profile ↔ Spectra Conversion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve inverse problem (intensity -> depth profile)
  python -m deppro.cli_depth solve --input intensity.csv --config model.yaml

  # Solve with default multilayer model
  python -m deppro.cli_depth solve --input intensity.csv --model multilayer

  # Create configuration file
  python -m deppro.cli_depth init-config --model multilayer --output config.yaml

  # Generate spectra from five-layer test profile
  python -m deppro.cli_depth generate --five-layer --angles 0,30,45,60,75 --output spectra/

  # Generate from custom profile
  python -m deppro.cli_depth generate --profile my_profile.json --angles 0,30,60 --output spectra/

  # Create five-layer test profile
  python -m deppro.cli_depth create-five-layer --output five_layer.json

  # Show profile info
  python -m deppro.cli_depth info --profile five_layer.json

  # Plot depth profile
  python -m deppro.cli_depth plot --profile five_layer.json --output five_layer_plot.png
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve inverse problem')
    solve_parser.add_argument('--input', '-i', type=str,
                             help='Input intensity data (CSV)')
    solve_parser.add_argument('--config', '-c', type=str,
                             help='Configuration file (YAML)')
    solve_parser.add_argument('--model', '-m', type=str,
                             choices=['multilayer', 'sio2_si'],
                             help='Use default model configuration')
    solve_parser.add_argument('--output', '-o', type=str,
                             help='Output directory (default: ./results)')
    solve_parser.add_argument('--loops', '-l', type=int, default=1,
                             help='Number of optimization loops (default: 1). '
                                  'Each loop rebuilds transfer matrix from current D_opt '
                                  'and decays alpha by 0.8x (like MATLAB GUI).')
    solve_parser.set_defaults(func=cmd_solve)

    # Init-config command
    init_parser = subparsers.add_parser('init-config', help='Create configuration file')
    init_parser.add_argument('--model', '-m', type=str,
                            choices=['multilayer', 'sio2_si'],
                            default='multilayer',
                            help='Model type (default: multilayer)')
    init_parser.add_argument('--output', '-o', type=str,
                            help='Output file path')
    init_parser.set_defaults(func=cmd_init_config)

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate spectra from depth profile')
    gen_parser.add_argument('--profile', '-p', type=str, help='Path to depth profile (JSON or H5)')
    gen_parser.add_argument('--five-layer', action='store_true', help='Use five-layer test profile')
    gen_parser.add_argument('--angles', '-a', type=str, default='0,30,45,60,75',
                           help='Observation angles (comma-separated, default: 0,30,45,60,75)')
    gen_parser.add_argument('--output', '-o', type=str, default='./spectra',
                           help='Output directory (default: ./spectra)')
    gen_parser.add_argument('--project', type=str, help='Project name for output files')
    gen_parser.add_argument('--noise-level', '-n', type=float, default=0.0,
                           help='Noise level (0=none, 100=slight, 1000=visible)')
    gen_parser.add_argument('--noise-type', type=str, default='poisson',
                           choices=['poisson', 'gaussian', 'none'],
                           help='Noise type (default: poisson)')
    gen_parser.add_argument('--photon-energy', type=float, default=1486.6,
                           help='Photon energy in eV (default: 1486.6 for Al Kα)')
    gen_parser.add_argument('--energy-edge', type=float, default=2.5,
                           help='Energy edge width in eV (default: 2.5)')
    gen_parser.add_argument('--energy-step', type=float, default=0.05,
                           help='Energy step in eV (default: 0.05)')
    gen_parser.add_argument('--depth-end', type=float, default=100.0,
                           help='Maximum depth in Å (default: 100)')
    gen_parser.add_argument('--step-size', type=float, default=1.0,
                           help='Depth step size in Å (default: 1.0)')
    gen_parser.set_defaults(func=cmd_generate)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show depth profile info')
    info_parser.add_argument('--profile', '-p', type=str, required=True,
                            help='Path to depth profile')
    info_parser.set_defaults(func=cmd_info)

    # Create five-layer command
    five_layer_parser = subparsers.add_parser('create-five-layer', help='Create five-layer test profile')
    five_layer_parser.add_argument('--output', '-o', type=str, default='five_layer_profile.json',
                            help='Output file path (default: five_layer_profile.json)')
    five_layer_parser.add_argument('--depth-end', type=float, default=100.0,
                            help='Maximum depth in Å (default: 100)')
    five_layer_parser.add_argument('--step-size', type=float, default=1.0,
                            help='Depth step size in Å (default: 1.0)')
    five_layer_parser.set_defaults(func=cmd_create_five_layer)

    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Plot depth profile')
    plot_parser.add_argument('--profile', '-p', type=str, required=True,
                            help='Path to depth profile')
    plot_parser.add_argument('--output', '-o', type=str,
                            help='Output image path (default: {profile}_plot.png)')
    plot_parser.set_defaults(func=cmd_plot)

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='(unavailable) Process multiple files in batch')
    batch_parser.add_argument('--input-dir', '-i', type=str, required=True,
                             help='Input directory containing CSV files')
    batch_parser.add_argument('--pattern', '-p', type=str, default='*.csv',
                             help='File pattern to match (default: *.csv)')
    batch_parser.add_argument('--config', '-c', type=str,
                             help='Configuration file (YAML)')
    batch_parser.add_argument('--output', '-o', type=str,
                             help='Output directory (default: ./batch_results)')
    batch_parser.add_argument('--workers', '-w', type=int, default=None,
                             help='Number of parallel workers (default: auto)')
    batch_parser.add_argument('--chunk-size', type=int, default=100,
                             help='Chunk size for batch processing')
    batch_parser.set_defaults(func=cmd_batch)

    # Monte Carlo command
    mc_parser = subparsers.add_parser('monte-carlo', help='(unavailable) Run Monte Carlo error estimation')
    mc_parser.add_argument('--input', '-i', type=str, required=True,
                          help='Input intensity data (CSV)')
    mc_parser.add_argument('--config', '-c', type=str,
                          help='Configuration file (YAML)')
    mc_parser.add_argument('--noise', '-n', type=float, default=2.0,
                          help='Noise level in percent (default: 2.0)')
    mc_parser.add_argument('--trials', '-t', type=int, default=100,
                          help='Number of Monte Carlo trials (default: 100)')
    mc_parser.add_argument('--workers', '-w', type=int, default=None,
                          help='Number of parallel workers (default: auto)')
    mc_parser.add_argument('--seed', '-s', type=int, default=None,
                          help='Random seed for reproducibility')
    mc_parser.add_argument('--output', '-o', type=str,
                          help='Output directory (default: ./monte_carlo)')
    mc_parser.add_argument('--plot', action='store_true',
                          help='Generate histogram plots')
    mc_parser.set_defaults(func=cmd_monte_carlo)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
