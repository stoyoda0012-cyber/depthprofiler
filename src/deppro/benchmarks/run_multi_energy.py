"""
Multi-Energy CRLB Analysis CLI
===============================

Convenient CLI for testing different photon energy configurations and
their impact on ARXPS depth profiling precision (CRLB).

Modes:
    sweep     - Lambda ratio sweep (single-energy CRLB at each ratio)
    combine   - Test a specific energy combination (Fisher additivity)
    sources   - Compare preset photon sources (eV-specified, IMFP recalculation)
    allocate  - Optimize count allocation between two energies
    optimize  - Find best energy pair from candidates
    angles    - Angle count sweep at a given energy
    summary   - Generate 6-panel summary plot

Usage:
    cd <repo>/src

    # Lambda ratio sweep
    python -m deppro.benchmarks.run_multi_energy --mode sweep \\
        --ratios 1 1.5 2 3 5 10

    # Combine energies (Fisher additivity)
    python -m deppro.benchmarks.run_multi_energy --mode combine \\
        --energies "Conv:1.0:1e6" "SXPS(1/2):2.0:1e6"

    # eV-specified energies (full IMFP recalculation)
    python -m deppro.benchmarks.run_multi_energy --mode combine \\
        --energies-eV "AlKa:1486.6:1e6" "HAXPES:8000:1e6"

    # Compare 7 X-ray sources from SESSAAnalyser database
    python -m deppro.benchmarks.run_multi_energy --mode sources --preset 7sources

    # Compare lab sources (Al Kα, Ag Lα, Cr Kα, Ga Kα)
    python -m deppro.benchmarks.run_multi_energy --mode sources --preset lab

    # Compare specific sources by name
    python -m deppro.benchmarks.run_multi_energy --mode sources \\
        --source-names Al_Ka Cr_Ka Ga_Ka

    # Count allocation optimization
    python -m deppro.benchmarks.run_multi_energy --mode allocate \\
        --ratios 1.0 2.0 --total-counts 2e6

    # Best energy pair search
    python -m deppro.benchmarks.run_multi_energy --mode optimize \\
        --candidates 1.0 1.5 2.0 3.0 5.0 10.0

    # 6-panel summary
    python -m deppro.benchmarks.run_multi_energy --mode summary
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from deppro.benchmarks.paths import GT_6ELEM

DEFAULT_H5 = GT_6ELEM
from deppro.paths import outputs_dir as _outputs_dir  # noqa: E402
OUTPUT_DIR = _outputs_dir()


def parse_energy_spec(spec: str):
    """Parse "label:ratio:counts" -> EnergyConfig."""
    from deppro.multi_energy import EnergyConfig
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Energy spec must be 'label:ratio:counts', got '{spec}'")
    return EnergyConfig(
        label=parts[0],
        lambda_ratio=float(parts[1]),
        n_counts=float(parts[2]),
    )


def parse_energy_eV_spec(spec: str):
    """Parse "label:eV:counts" -> EnergyConfig with photon_energy_eV."""
    from deppro.multi_energy import EnergyConfig
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Energy-eV spec must be 'label:eV:counts', got '{spec}'")
    return EnergyConfig(
        label=parts[0],
        lambda_ratio=0.0,  # Not used when photon_energy_eV is set
        n_counts=float(parts[2]),
        photon_energy_eV=float(parts[1]),
    )


def _load_data_and_create_me(args):
    """Shared setup: load H5, create FisherInformation and MultiEnergyCRLB."""
    from deppro.benchmarks.benchmark_dstk import load_dstk_data, create_tm_from_h5
    from deppro.fisher_information import FisherInformation
    from deppro.multi_energy import MultiEnergyCRLB

    print(f"Loading: {args.h5}")
    data = load_dstk_data(args.h5)
    tm = create_tm_from_h5(data)
    print(f"  {data.n_depth} depths, {data.n_elements} elements, "
          f"{data.n_angles} angles")
    print(f"  Elements: {data.element_names}")

    fi = FisherInformation(tm, data, n_counts=args.n_counts)
    me = MultiEnergyCRLB(fi, pixel_idx=args.pixel)
    print(f"  Params: {me.n_params} ({me.film_size} film + 2 struct + "
          f"{fi.n_sub} sub)")
    print(f"  Film region: {me.film_mask.sum()}/{me.n_depth} depths")
    print()
    return data, fi, me


# =====================================================================
# Mode: sweep
# =====================================================================

def run_sweep(args):
    """Lambda ratio sweep."""
    data, fi, me = _load_data_and_create_me(args)
    ratios = args.ratios or [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    print(f"=== Lambda Ratio Sweep (N={args.n_counts:.0e}) ===\n")
    t0 = time.perf_counter()
    result = me.sweep_lambda_ratios(ratios, n_counts=args.n_counts)
    dt = time.perf_counter() - t0

    # Table
    print(f"  {'Ratio':>8s}  {'Rank':>6s}  {'Film CRLB (%)':>14s}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*14}")
    for r, rank, crlb in zip(result['ratios'], result['ranks'], result['crlb_pcts']):
        print(f"  {r:8.2f}  {rank:6d}  {crlb:14.4f}")
    print(f"\n  Computed in {dt:.1f}s")

    if not args.no_plot:
        _plot_sweep(result, args)


def _plot_sweep(result, args):
    """Plot lambda ratio sweep: rank and CRLB twin axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    ax.plot(result['ratios'], result['ranks'], 'bo-', markersize=7, label='Rank')
    ax2.semilogy(result['ratios'], result['crlb_pcts'], 'rs-', markersize=7,
                 label='Film CRLB (%)')

    ax.set_xlabel(r"$\lambda_{conv} / \lambda$ (lambda ratio)")
    ax.set_ylabel("Effective rank", color='b')
    ax2.set_ylabel("Film CRLB (%)", color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title("Multi-Energy Sweep: Rank vs CRLB")
    ax.grid(True, alpha=0.3)

    outpath = OUTPUT_DIR / (args.output or "multi_energy_sweep.png")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(outpath), dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outpath}")
    plt.close()


# =====================================================================
# Mode: combine
# =====================================================================

def run_combine(args):
    """Test a specific energy combination."""
    from deppro.multi_energy import EnergyConfig

    data, fi, me = _load_data_and_create_me(args)

    # Parse energy specifications
    energies = []
    if args.energies_eV:
        for spec in args.energies_eV:
            energies.append(parse_energy_eV_spec(spec))
    elif args.energies:
        for spec in args.energies:
            energies.append(parse_energy_spec(spec))
    else:
        # Default: Conv + SXPS(1/2)
        energies = [
            EnergyConfig("Conv", 1.0, args.n_counts),
            EnergyConfig("SXPS(1/2)", 2.0, args.n_counts),
        ]

    print(f"=== Energy Combination Analysis (N_total={sum(e.n_counts for e in energies):.0e}) ===\n")
    t0 = time.perf_counter()
    result = me.analyze_combination(energies)
    dt = time.perf_counter() - t0

    # Per-energy table
    print("Per-energy results:")
    print(f"  {'Label':>16s}  {'λ ratio':>8s}  {'eV':>10s}  {'N':>10s}  "
          f"{'Rank':>6s}  {'CRLB (%)':>10s}")
    print(f"  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*10}")
    for e, rank, crlb in zip(result.energies, result.per_energy_ranks,
                              result.per_energy_crlbs):
        eV_str = f"{e.photon_energy_eV:.0f}" if e.photon_energy_eV else "—"
        print(f"  {e.label:>16s}  {e.lambda_ratio:8.2f}  {eV_str:>10s}  "
              f"{e.n_counts:10.0e}  {rank:6d}  {crlb:10.4f}")

    # Combined
    print(f"\nCombined Fisher (F = sum F_i):")
    print(f"  Rank: {result.combined_rank}")
    print(f"  Film CRLB: {result.mean_crlb_film_pct:.4f}%")
    print(f"  Improvement vs Conv: {result.improvement_factor:.1f}x")

    # Per-element table
    print(f"\nPer-element CRLB (%):")
    header = f"  {'Element':>12s}"
    for e in result.energies:
        header += f"  {e.label:>10s}"
    header += f"  {'Combined':>10s}  {'Improvement':>12s}"
    print(header)
    print(f"  {'─'*12}" + f"  {'─'*10}" * (len(result.energies) + 1) + f"  {'─'*12}")

    # Compute per-element per-energy CRLB
    per_energy_per_elem = []
    for F in result.fisher_matrices:
        _, cf, _, _ = me.extract_crlb(
            F, me.film_size, me.n_depth, me.n_free,
            film_mask=me.film_mask,
        )
        if me.film_mask.any():
            per_elem = np.mean(cf[me.film_mask, :], axis=0) * 100
        else:
            per_elem = np.mean(cf, axis=0) * 100
        per_energy_per_elem.append(per_elem)

    for j, name in enumerate(result.element_names):
        row = f"  {name:>12s}"
        for pe in per_energy_per_elem:
            row += f"  {pe[j]:10.4f}"
        row += f"  {result.combined_crlb_per_element[j]:10.4f}"
        ref_val = per_energy_per_elem[0][j] if per_energy_per_elem else 0
        if result.combined_crlb_per_element[j] > 0:
            imp = ref_val / result.combined_crlb_per_element[j]
            row += f"  {imp:10.1f}x"
        else:
            row += f"  {'inf':>10s}x"
        print(row)

    print(f"\n  Computed in {dt:.1f}s")

    if not args.no_plot:
        _plot_combine(result, me, per_energy_per_elem, args)


def _plot_combine(result, me, per_energy_per_elem, args):
    """Plot energy combination: per-element bar chart + depth CRLB."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-element CRLB bar chart
    n_elem = len(result.element_names)
    n_bars = len(result.energies) + 1  # per-energy + combined
    width = 0.8 / n_bars
    x_pos = np.arange(n_elem)

    colors_e = plt.cm.Set2(np.linspace(0, 0.8, len(result.energies)))
    for i, (e, pe) in enumerate(zip(result.energies, per_energy_per_elem)):
        ax1.bar(x_pos + i * width, pe, width, label=e.label,
                color=colors_e[i], alpha=0.8)
    ax1.bar(x_pos + len(result.energies) * width,
            result.combined_crlb_per_element, width,
            label='Combined', color='forestgreen', alpha=0.9)

    ax1.set_xticks(x_pos + width * n_bars / 2)
    ax1.set_xticklabels(result.element_names, fontsize=8)
    ax1.set_ylabel("CRLB (%)")
    ax1.set_title("Per-Element CRLB Comparison")
    ax1.legend(fontsize=8)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: depth-resolved combined CRLB
    depth = me.fi.tm.depth
    crlb_mean = np.mean(result.combined_crlb_film, axis=1) * 100
    ax2.semilogy(depth, np.maximum(crlb_mean, 1e-6), 'g-', linewidth=2,
                 label='Combined')

    # Also show individual energy depth profiles
    for i, (e, F_i) in enumerate(zip(result.energies, result.fisher_matrices)):
        _, cf_i, _, _ = me.extract_crlb(
            F_i, me.film_size, me.n_depth, me.n_free,
            film_mask=me.film_mask,
        )
        crlb_i_mean = np.mean(cf_i, axis=1) * 100
        ax2.semilogy(depth, np.maximum(crlb_i_mean, 1e-6), '--', linewidth=1.5,
                     label=e.label, color=colors_e[i])

    # Film region
    film_depths = depth[me.film_mask]
    if len(film_depths) > 0:
        ax2.axvspan(film_depths[0], film_depths[-1], alpha=0.08, color='yellow')

    ax2.set_xlabel("Depth (Angstrom)")
    ax2.set_ylabel("CRLB (%)")
    ax2.set_title("Depth-Resolved CRLB")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Multi-Energy CRLB: "
                 f"Combined CRLB={result.mean_crlb_film_pct:.4f}% "
                 f"({result.improvement_factor:.1f}x improvement)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    outpath = OUTPUT_DIR / (args.output or "multi_energy_combine.png")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outpath), dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outpath}")
    plt.close()


# =====================================================================
# Mode: sources
# =====================================================================

def run_sources(args):
    """Compare preset photon sources with full IMFP recalculation."""
    from deppro.multi_energy import (
        EnergyConfig, PHOTON_SOURCES, ENERGY_PRESETS,
        make_energies_from_preset,
    )

    data, fi, me = _load_data_and_create_me(args)

    # Build energy list from --preset, --source-names, or default
    if args.source_names:
        energies = []
        for name in args.source_names:
            if name not in PHOTON_SOURCES:
                avail = ", ".join(PHOTON_SOURCES.keys())
                print(f"ERROR: Unknown source '{name}'. Available: {avail}")
                sys.exit(1)
            src = PHOTON_SOURCES[name]
            energies.append(EnergyConfig(
                label=src["label"],
                lambda_ratio=0.0,
                n_counts=args.n_counts,
                photon_energy_eV=src["energy_eV"],
            ))
        preset_desc = f"Custom: {', '.join(args.source_names)}"
    else:
        preset_name = args.preset or "lab"
        if preset_name not in ENERGY_PRESETS:
            avail = ", ".join(ENERGY_PRESETS.keys())
            print(f"ERROR: Unknown preset '{preset_name}'. Available: {avail}")
            sys.exit(1)
        energies = make_energies_from_preset(preset_name, n_counts=args.n_counts)
        preset_desc = f"Preset '{preset_name}': {ENERGY_PRESETS[preset_name]['description']}"

    print(f"=== Photon Source Comparison ===")
    print(f"  {preset_desc}")
    print(f"  N_counts per source: {args.n_counts:.0e}\n")

    # --- Part 1: Individual source analysis ---
    t0 = time.perf_counter()
    individual = []
    for energy in energies:
        r = me.analyze_single(energy)
        individual.append(r)

    print(f"{'Source':>14s}  {'eV':>8s}  {'Rank':>6s}  {'CRLB (%)':>12s}")
    print(f"{'─'*14}  {'─'*8}  {'─'*6}  {'─'*12}")
    for energy, r in zip(energies, individual):
        eV = energy.photon_energy_eV or 0
        print(f"{energy.label:>14s}  {eV:8.1f}  {r['rank']:6d}  {r['crlb_pct']:12.4f}")

    # --- Part 2: Pairwise combinations (if >=2 sources) ---
    if len(energies) >= 2:
        print(f"\nPairwise Combinations (N={args.n_counts:.0e} each):")
        print(f"  {'Pair':>30s}  {'Rank':>6s}  {'CRLB (%)':>12s}  {'Improvement':>12s}")
        print(f"  {'─'*30}  {'─'*6}  {'─'*12}  {'─'*12}")

        pair_results = []
        from itertools import combinations as itercombs
        for i, j in itercombs(range(len(energies)), 2):
            pair = [energies[i], energies[j]]
            result = me.analyze_combination(pair)
            pair_results.append((i, j, result))

        # Sort by CRLB
        pair_results.sort(key=lambda x: x[2].mean_crlb_film_pct)
        for idx, (i, j, result) in enumerate(pair_results):
            label = f"{energies[i].label} + {energies[j].label}"
            star = " ★" if idx == 0 else ""
            print(f"  {label:>30s}  {result.combined_rank:6d}  "
                  f"{result.mean_crlb_film_pct:12.4f}  "
                  f"{result.improvement_factor:10.1f}x{star}")

        # Best pair summary
        best_i, best_j, best_r = pair_results[0]
        print(f"\n  Best pair: {energies[best_i].label} + {energies[best_j].label}")
        print(f"  CRLB: {best_r.mean_crlb_film_pct:.4f}%, "
              f"Improvement: {best_r.improvement_factor:.1f}x")

    # --- Part 3: All combined (if >=3 sources) ---
    if len(energies) >= 3:
        result_all = me.analyze_combination(energies)
        print(f"\nAll {len(energies)} sources combined:")
        print(f"  Rank: {result_all.combined_rank}, "
              f"CRLB: {result_all.mean_crlb_film_pct:.4f}%, "
              f"Improvement: {result_all.improvement_factor:.1f}x")

    dt = time.perf_counter() - t0
    print(f"\n  Computed in {dt:.1f}s")

    if not args.no_plot:
        _plot_sources(energies, individual, pair_results if len(energies) >= 2 else [],
                      me, args, preset_desc)


def _plot_sources(energies, individual, pair_results, me, args, title_desc):
    """Plot source comparison: individual bars + pairwise heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deppro.multi_energy import PHOTON_SOURCES

    n_src = len(energies)
    fig, axes = plt.subplots(1, 2 if n_src >= 2 else 1,
                              figsize=(14 if n_src >= 2 else 7, 6))
    if n_src < 2:
        axes = [axes]

    # Panel 1: Individual CRLB bars
    ax = axes[0]
    labels = [e.label for e in energies]
    crlbs = [r['crlb_pct'] for r in individual]
    ranks = [r['rank'] for r in individual]

    # Color by source
    colors = []
    for e in energies:
        found = False
        for src_info in PHOTON_SOURCES.values():
            if src_info["label"] == e.label:
                colors.append(src_info["color"])
                found = True
                break
        if not found:
            colors.append("gray")

    bars = ax.bar(range(n_src), crlbs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(n_src))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Film CRLB (%)")
    ax.set_title("Individual Source CRLB")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate with rank
    for i, (bar, rank) in enumerate(zip(bars, ranks)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"r={rank}", ha='center', va='bottom', fontsize=8)

    # Panel 2: Pairwise CRLB heatmap
    if n_src >= 2 and pair_results:
        ax2 = axes[1]
        crlb_matrix = np.full((n_src, n_src), np.nan)
        for i, r in enumerate(individual):
            crlb_matrix[i, i] = r['crlb_pct']
        for pi, pj, pr in pair_results:
            crlb_matrix[pi, pj] = pr.mean_crlb_film_pct
            crlb_matrix[pj, pi] = pr.mean_crlb_film_pct

        im = ax2.imshow(np.log10(np.maximum(crlb_matrix, 1e-6)),
                        cmap='RdYlGn_r', aspect='equal')
        ax2.set_xticks(range(n_src))
        ax2.set_yticks(range(n_src))
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_title("Pairwise CRLB (log₁₀ %)")

        # Annotate cells
        for i in range(n_src):
            for j in range(n_src):
                if not np.isnan(crlb_matrix[i, j]):
                    val = crlb_matrix[i, j]
                    if val < 0.01:
                        txt = f"{val:.4f}"
                    elif val < 1:
                        txt = f"{val:.3f}"
                    else:
                        txt = f"{val:.1f}"
                    ax2.text(j, i, txt, ha='center', va='center', fontsize=7)

        plt.colorbar(im, ax=ax2, shrink=0.8, label='log₁₀ CRLB (%)')

    plt.suptitle(f"Photon Source Comparison — {title_desc}",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()

    outpath = OUTPUT_DIR / (args.output or "multi_energy_sources.png")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outpath), dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outpath}")
    plt.close()


# =====================================================================
# Mode: allocate
# =====================================================================

def run_allocate(args):
    """Count allocation optimization between two energies."""
    data, fi, me = _load_data_and_create_me(args)
    ratios = args.ratios or [1.0, 2.0]
    if len(ratios) != 2:
        print("ERROR: --mode allocate requires exactly 2 ratios (--ratios A B)")
        sys.exit(1)

    total = args.total_counts
    print(f"=== Count Allocation: ratio_a={ratios[0]}, ratio_b={ratios[1]}, "
          f"N_total={total:.0e} ===\n")

    t0 = time.perf_counter()
    result = me.sweep_count_allocation(ratios[0], ratios[1], total, n_steps=21)
    dt = time.perf_counter() - t0

    print(f"  {'Frac(A)':>8s}  {'N(A)':>10s}  {'N(B)':>10s}  "
          f"{'Rank':>6s}  {'CRLB (%)':>10s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*10}")
    for f, r, c in zip(result['fractions'], result['ranks'], result['crlb_pcts']):
        print(f"  {f:8.2f}  {total*f:10.0e}  {total*(1-f):10.0e}  "
              f"{r:6d}  {c:10.4f}")

    print(f"\n  Optimal: f={result['optimal_fraction']:.2f}, "
          f"CRLB={result['optimal_crlb']:.4f}%")
    print(f"  Computed in {dt:.1f}s")

    if not args.no_plot:
        _plot_allocate(result, args)


def _plot_allocate(result, args):
    """Plot count allocation sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(result['fractions'], result['crlb_pcts'], 'go-', markersize=5)
    ax.axvline(result['optimal_fraction'], color='r', ls='--', alpha=0.5,
               label=f"Optimal f={result['optimal_fraction']:.2f}")
    ax.set_xlabel(f"Fraction allocated to ratio={result['ratio_a']}")
    ax.set_ylabel("Film CRLB (%)")
    ax.set_title(f"Count Allocation: ratio {result['ratio_a']} vs {result['ratio_b']}\n"
                 f"Total N={result['total_counts']:.0e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    outpath = OUTPUT_DIR / (args.output or "multi_energy_allocate.png")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(outpath), dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outpath}")
    plt.close()


# =====================================================================
# Mode: optimize
# =====================================================================

def run_optimize(args):
    """Find best energy combination from candidates."""
    data, fi, me = _load_data_and_create_me(args)
    candidates = args.candidates or [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    print(f"=== Energy Optimization: {len(candidates)} candidates, "
          f"choose {args.n_energies}, N_total={args.total_counts:.0e} ===\n")
    print(f"  Candidates: {candidates}")

    t0 = time.perf_counter()
    result = me.optimize_energy_set(candidates, args.n_energies, args.total_counts)
    dt = time.perf_counter() - t0

    print(f"\n  {'#':>4s}  {'Combination':>30s}  {'Rank':>6s}  {'CRLB (%)':>10s}")
    print(f"  {'─'*4}  {'─'*30}  {'─'*6}  {'─'*10}")
    for i, r in enumerate(result['all_results'][:10]):
        combo_str = " + ".join(f"{c:.1f}" for c in r['combo'])
        marker = " ★" if i == 0 else ""
        print(f"  {i+1:4d}  {combo_str:>30s}  {r['rank']:6d}  "
              f"{r['crlb_pct']:10.4f}{marker}")

    if len(result['all_results']) > 10:
        print(f"  ... ({len(result['all_results'])} total combinations)")

    print(f"\n  Best: {result['best_combo']} → "
          f"rank={result['best_rank']}, CRLB={result['best_crlb']:.4f}%")
    print(f"  Computed in {dt:.1f}s")


# =====================================================================
# Mode: angles
# =====================================================================

def run_angles(args):
    """Angle count sweep."""
    data, fi, me = _load_data_and_create_me(args)
    n_angles = args.n_angles_list or [2, 3, 4, 5, 6, 7, 8, 10, 12, 14]
    lr = args.lambda_ratio if hasattr(args, 'lambda_ratio') and args.lambda_ratio else 1.0

    print(f"=== Angle Count Sweep (λ_ratio={lr:.2f}, N={args.n_counts:.0e}) ===\n")
    t0 = time.perf_counter()
    result = me.sweep_n_angles(n_angles, lambda_ratio=lr, n_counts=args.n_counts)
    dt = time.perf_counter() - t0

    print(f"  {'N_angles':>8s}  {'Rank':>6s}  {'Film CRLB (%)':>14s}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*14}")
    for n, r, c in zip(result['n_angles'], result['ranks'], result['crlb_pcts']):
        print(f"  {n:8d}  {r:6d}  {c:14.4f}")
    print(f"\n  Computed in {dt:.1f}s")

    if not args.no_plot:
        _plot_angles(result, lr, args)


def _plot_angles(result, lambda_ratio, args):
    """Plot angle count sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()
    ax.plot(result['n_angles'], result['ranks'], 'bo-', markersize=7, label='Rank')
    ax2.plot(result['n_angles'], result['crlb_pcts'], 'rs-', markersize=7,
             label='CRLB (%)')
    ax.set_xlabel("Number of angles")
    ax.set_ylabel("Effective rank", color='b')
    ax2.set_ylabel("Film CRLB (%)", color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title(f"Angle Count Sweep (lambda_ratio={lambda_ratio:.2f})")
    ax.grid(True, alpha=0.3)

    outpath = OUTPUT_DIR / (args.output or "multi_energy_angles.png")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(outpath), dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outpath}")
    plt.close()


# =====================================================================
# Mode: summary
# =====================================================================

def run_summary(args):
    """Generate 6-panel summary plot."""
    from deppro.multi_energy import EnergyConfig

    data, fi, me = _load_data_and_create_me(args)
    print("=== Multi-Energy Summary (6-panel) ===\n")
    t0 = time.perf_counter()

    # Panel 1: SXPS rank paradox
    print("  Panel 1: SXPS rank vs lambda ratio...")
    sweep_ratios = [1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
    sweep = me.sweep_lambda_ratios(sweep_ratios, n_counts=args.n_counts)

    # Panel 2: Combined Fisher improvement
    print("  Panel 2: Combined Fisher configurations...")
    configs = [
        ([EnergyConfig("Conv", 1.0, args.n_counts)], "Conv"),
        ([EnergyConfig("S(1/2)", 2.0, args.n_counts)], "S(1/2)"),
        ([EnergyConfig("S(1/3)", 3.0, args.n_counts)], "S(1/3)"),
        ([EnergyConfig("S(1/5)", 5.0, args.n_counts)], "S(1/5)"),
        ([EnergyConfig("Conv", 1.0, args.n_counts),
          EnergyConfig("S(1/2)", 2.0, args.n_counts)], "C+S(1/2)"),
        ([EnergyConfig("Conv", 1.0, args.n_counts),
          EnergyConfig("S(1/3)", 3.0, args.n_counts)], "C+S(1/3)"),
        ([EnergyConfig("Conv", 1.0, args.n_counts),
          EnergyConfig("S(1/2)", 2.0, args.n_counts),
          EnergyConfig("S(1/3)", 3.0, args.n_counts)], "C+S2+S3"),
        ([EnergyConfig("Conv", 1.0, args.n_counts),
          EnergyConfig("S(1/2)", 2.0, args.n_counts),
          EnergyConfig("S(1/3)", 3.0, args.n_counts),
          EnergyConfig("S(1/5)", 5.0, args.n_counts)], "C+3S"),
    ]
    combo_results = []
    for energies, label in configs:
        r = me.analyze_combination(energies)
        combo_results.append((label, r.combined_rank, r.mean_crlb_film_pct))
        print(f"    {label:12s} rank={r.combined_rank:2d}  CRLB={r.mean_crlb_film_pct:.4f}%")

    # Panel 3: Angle count sweep
    print("  Panel 3: Angle count sweep...")
    angles = me.sweep_n_angles([2, 3, 4, 5, 6, 7, 8, 10, 12, 14])

    # Panel 4: Count regime (Conv vs Conv+SXPS)
    print("  Panel 4: Count regime...")
    J_s2, X_s2 = me.compute_jacobian_at_energy(2.0)
    count_range = np.logspace(3, 8, 20)
    crlb_conv_arr = []
    crlb_comb_arr = []
    for n_c in count_range:
        F_c = me.build_fisher_matrix(me._J_conv, me._X_conv, n_c)
        _, _, pct_c, _ = me.extract_crlb(
            F_c, me.film_size, me.n_depth, me.n_free, film_mask=me.film_mask)
        crlb_conv_arr.append(pct_c)

        F_half = (me.build_fisher_matrix(me._J_conv, me._X_conv, n_c / 2) +
                  me.build_fisher_matrix(J_s2, X_s2, n_c / 2))
        _, _, pct_h, _ = me.extract_crlb(
            F_half, me.film_size, me.n_depth, me.n_free, film_mask=me.film_mask)
        crlb_comb_arr.append(pct_h)

    # Panel 5: Allocation sweep (Conv vs SXPS(1/2))
    print("  Panel 5: Count allocation...")
    alloc = me.sweep_count_allocation(1.0, 2.0, 2e6, n_steps=21)

    # Panel 6: Combined depth CRLB
    print("  Panel 6: Depth-resolved CRLB...")
    # Already have data from combo_results

    dt = time.perf_counter() - t0
    print(f"\n  All panels computed in {dt:.1f}s")

    if not args.no_plot:
        _plot_summary(
            sweep, combo_results, angles, count_range,
            crlb_conv_arr, crlb_comb_arr, alloc, me, args,
        )


def _plot_summary(sweep, combo_results, angles, count_range,
                  crlb_conv_arr, crlb_comb_arr, alloc, me, args):
    """Generate 6-panel summary plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Multi-Energy CRLB Summary\n"
                 "Information-Theoretic Measurement Strategy Optimization",
                 fontsize=14, fontweight='bold')

    # Panel 1: Lambda ratio sweep
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.plot(sweep['ratios'], sweep['ranks'], 'bo-', markersize=6, label='Rank')
    ax2.semilogy(sweep['ratios'], sweep['crlb_pcts'], 'rs-', markersize=6,
                 label='CRLB (%)')
    ax.set_xlabel(r"$\lambda_{conv} / \lambda$")
    ax.set_ylabel("Effective rank", color='b')
    ax2.set_ylabel("Film CRLB (%)", color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title("SXPS Paradox: Rank down, CRLB down")
    ax.grid(True, alpha=0.3)

    # Panel 2: Combined Fisher
    ax = axes[0, 1]
    labels = [c[0] for c in combo_results]
    ranks = [c[1] for c in combo_results]
    crlbs = [c[2] for c in combo_results]
    bar_colors = ['steelblue', 'coral', 'coral', 'coral',
                  'forestgreen', 'forestgreen', 'forestgreen', 'gold']
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, ranks, color=bar_colors[:len(labels)], alpha=0.8,
           edgecolor='black', linewidth=0.5)
    for i, (r, c) in enumerate(zip(ranks, crlbs)):
        ax.text(i, r + 0.3, f'{c:.2f}%' if c >= 0.01 else f'{c:.3f}%',
                ha='center', fontsize=6, rotation=45)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel("Effective rank")
    ax.set_title("Fisher Additivity: F = F1 + F2")
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Angle count
    ax = axes[0, 2]
    ax2 = ax.twinx()
    ax.plot(angles['n_angles'], angles['ranks'], 'bo-', markersize=6)
    ax2.plot(angles['n_angles'], angles['crlb_pcts'], 'rs-', markersize=6)
    ax.set_xlabel("Number of angles")
    ax.set_ylabel("Effective rank", color='b')
    ax2.set_ylabel("Film CRLB (%)", color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title("Angle Count: Diminishing Returns")
    ax.grid(True, alpha=0.3)

    # Panel 4: Count regime
    ax = axes[1, 0]
    ax.loglog(count_range, crlb_conv_arr, 'b-', linewidth=2, label='Conv only')
    ax.loglog(count_range, crlb_comb_arr, 'g-', linewidth=2,
              label='Conv(N/2) + SXPS(N/2)')
    ax.set_xlabel("Total counts N")
    ax.set_ylabel("Film CRLB (%)")
    ax.set_title("Practical Regime: CRLB ~ 1/sqrt(N)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.01, 1000)

    # Panel 5: Allocation sweep
    ax = axes[1, 1]
    ax.semilogy(alloc['fractions'], alloc['crlb_pcts'], 'go-', markersize=4)
    ax.axvline(alloc['optimal_fraction'], color='r', ls='--', alpha=0.5)
    ax.text(alloc['optimal_fraction'] + 0.02,
            alloc['optimal_crlb'] * 1.3,
            f"Optimal f={alloc['optimal_fraction']:.2f}",
            fontsize=8, color='r')
    ax.set_xlabel("Fraction to Conv (rest to SXPS(1/2))")
    ax.set_ylabel("Film CRLB (%)")
    ax.set_title("Count Allocation Optimization")
    ax.grid(True, alpha=0.3)

    # Panel 6: depth CRLB comparison
    ax = axes[1, 2]
    depth = me.fi.tm.depth
    # Conv
    F_conv = me.build_fisher_matrix(me._J_conv, me._X_conv, me.fi.n_counts)
    _, cf_conv, _, _ = me.extract_crlb(
        F_conv, me.film_size, me.n_depth, me.n_free, film_mask=me.film_mask)
    # SXPS(1/3)
    J_s3, X_s3 = me.compute_jacobian_at_energy(3.0)
    F_s3 = me.build_fisher_matrix(J_s3, X_s3, me.fi.n_counts)
    _, cf_s3, _, _ = me.extract_crlb(
        F_s3, me.film_size, me.n_depth, me.n_free, film_mask=me.film_mask)
    # Combined
    J_s2, X_s2 = me.compute_jacobian_at_energy(2.0)
    F_cs2 = F_conv + me.build_fisher_matrix(J_s2, X_s2, me.fi.n_counts)
    _, cf_cs2, _, _ = me.extract_crlb(
        F_cs2, me.film_size, me.n_depth, me.n_free, film_mask=me.film_mask)

    ax.semilogy(depth, np.maximum(np.mean(cf_conv, axis=1) * 100, 1e-6),
                'b-', linewidth=2, label='Conv')
    ax.semilogy(depth, np.maximum(np.mean(cf_s3, axis=1) * 100, 1e-6),
                'r-', linewidth=2, label='SXPS(1/3)')
    ax.semilogy(depth, np.maximum(np.mean(cf_cs2, axis=1) * 100, 1e-6),
                'g-', linewidth=2, label='Conv+SXPS(1/2)')
    film_depths = depth[me.film_mask]
    if len(film_depths) > 0:
        ax.axvspan(film_depths[0], film_depths[-1], alpha=0.08, color='yellow')
    ax.set_xlabel("Depth (Angstrom)")
    ax.set_ylabel("CRLB (%)")
    ax.set_title("Depth-Resolved CRLB Comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-4, 100)

    plt.tight_layout()
    outpath = OUTPUT_DIR / (args.output or "multi_energy_summary.png")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outpath), dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {outpath}")
    plt.close()


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Energy CRLB Analysis for ARXPS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--h5", default=DEFAULT_H5, help="Benchmark H5 file")
    parser.add_argument("--mode", default="sweep",
                        choices=["sweep", "combine", "sources", "allocate",
                                 "optimize", "angles", "summary"],
                        help="Analysis mode (default: sweep)")
    parser.add_argument("--n-counts", type=float, default=1e6,
                        help="Count number per energy (default: 1e6)")
    parser.add_argument("--pixel", type=int, default=0,
                        help="Pixel index (default: 0)")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: auto)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")

    # Mode: sweep
    parser.add_argument("--ratios", type=float, nargs="+",
                        help="Lambda ratios for sweep/allocate")

    # Mode: combine
    parser.add_argument("--energies", nargs="+",
                        help="Energy specs: 'label:ratio:counts' ...")
    parser.add_argument("--energies-eV", nargs="+",
                        help="Energy specs by eV: 'label:eV:counts' ...")

    # Mode: sources
    parser.add_argument("--preset", default=None,
                        help="Photon source preset: 7sources/lab/haxpes/dual_anode/soft_hard")
    parser.add_argument("--source-names", nargs="+",
                        help="Specific source names: SR_620 Al_Ka Cr_Ka Ga_Ka ...")

    # Mode: allocate
    parser.add_argument("--total-counts", type=float, default=2e6,
                        help="Total counts for allocation/optimize (default: 2e6)")

    # Mode: optimize
    parser.add_argument("--candidates", type=float, nargs="+",
                        help="Candidate lambda ratios for optimization")
    parser.add_argument("--n-energies", type=int, default=2,
                        help="Number of energies to combine (default: 2)")

    # Mode: angles
    parser.add_argument("--n-angles-list", type=int, nargs="+",
                        help="Angle counts to test")
    parser.add_argument("--lambda-ratio", type=float, default=1.0,
                        help="Lambda ratio for angle sweep (default: 1.0)")

    args = parser.parse_args()

    mode_dispatch = {
        "sweep": run_sweep,
        "combine": run_combine,
        "sources": run_sources,
        "allocate": run_allocate,
        "optimize": run_optimize,
        "angles": run_angles,
        "summary": run_summary,
    }

    mode_dispatch[args.mode](args)


if __name__ == "__main__":
    main()
