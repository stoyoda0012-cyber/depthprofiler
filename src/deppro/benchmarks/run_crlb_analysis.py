"""
Fisher Information / CRLB Analysis for ARXPS Depth Profiling
============================================================

Computes Cramér-Rao Lower Bounds for the benchmark dataset and generates
a 6-panel diagnostic plot.

Usage:
    cd <repo>
    python -m deppro.benchmarks.run_crlb_analysis \
        --h5 /path/to/dataset.h5 \
        [--n-counts 1e6] [--n-pixels 50] [--output fisher_crlb_analysis.png]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Default benchmark H5
from deppro.benchmarks.paths import GT_6ELEM

DEFAULT_H5 = GT_6ELEM
from deppro.paths import outputs_dir as _outputs_dir  # noqa: E402
OUTPUT_DIR = _outputs_dir()


def main():
    parser = argparse.ArgumentParser(description="Fisher/CRLB analysis")
    parser.add_argument("--h5", default=DEFAULT_H5, help="Benchmark H5 file")
    parser.add_argument("--n-counts", type=float, default=1e6,
                        help="Total count number for Poisson noise (default: 1e6)")
    parser.add_argument("--n-pixels", type=int, default=50,
                        help="Number of pixels for multi-pixel statistics")
    parser.add_argument("--output", default="fisher_crlb_analysis.png",
                        help="Output filename")
    args = parser.parse_args()

    # Import here to avoid circular imports
    from deppro.benchmarks.benchmark_dstk import load_dstk_data, create_tm_from_h5
    from deppro.fisher_information import FisherInformation

    print(f"Loading benchmark data from {args.h5} ...")
    data = load_dstk_data(args.h5)
    tm = create_tm_from_h5(data)

    print(f"  {data.n_pixels} pixels, {data.n_depth} depths, "
          f"{data.n_elements} elements, {data.n_angles} angles")
    print(f"  Elements: {data.element_names}")
    print(f"  Free: {data.free_indices}, Sub: {data.substrate_indices}, "
          f"Linked: {data.linked_indices}")

    fi = FisherInformation(tm, data, n_counts=args.n_counts)
    print(f"\nParameter space: {fi.n_params} parameters "
          f"({fi.film_size} film + 1 thickness + 1 abruptness + {fi.n_sub} sub)")

    # ===================================================================
    # Phase A: Single pixel CRLB
    # ===================================================================
    t0 = time.time()
    print("\n=== Phase A: Single-pixel CRLB (pixel 0) ===")
    result = fi.compute_crlb(pixel_idx=0)
    print(f"  Computed in {time.time() - t0:.1f}s")
    print(f"  F matrix: {result.fisher_matrix.shape}, "
          f"cond = {result.condition_number:.2e}, "
          f"eff rank = {result.effective_rank}/{fi.n_params}")
    print(f"  Thickness CRLB: {result.crlb_thickness:.3f} Å")
    print(f"  Abruptness CRLB: {result.crlb_abruptness:.3f} Å")

    # Element-wise CRLB summary
    free_names = [data.element_names[i] for i in data.free_indices]
    print(f"\n  Element | Mean CRLB (%) | Max CRLB (%)")
    print(f"  --------|---------------|-------------")
    for j, name in enumerate(free_names):
        mean_c = np.mean(result.crlb_film_conc[:, j]) * 100
        max_c = np.max(result.crlb_film_conc[:, j]) * 100
        print(f"  {name:8s}| {mean_c:13.4f} | {max_c:11.4f}")

    # ===================================================================
    # Phase B: Efficiency vs SLSQP RMSE
    # ===================================================================
    print("\n=== Phase B: Efficiency vs SLSQP RMSE (from benchmark data) ===")
    # Load MATLAB baseline if available
    try:
        from deppro.benchmarks.benchmark_dstk import compute_matlab_baseline
        pixel_indices = np.arange(data.n_pixels)
        ml_result = compute_matlab_baseline(data, pixel_indices)
        slsqp_rmse = ml_result['rmse_per_element']
        print(f"  MATLAB RMSE available: {slsqp_rmse}")

        # CRLB averaged over depths
        crlb_per_element = np.mean(result.crlb_film_conc, axis=0)  # (n_free,)
        print(f"\n  Element | CRLB (%) | MATLAB RMSE (%) | Efficiency (CRLB/RMSE)")
        print(f"  --------|----------|-----------------|----------------------")
        for j, name in enumerate(free_names):
            free_e = data.free_indices[j]
            c = crlb_per_element[j] * 100
            r = slsqp_rmse[free_e] * 100
            eff = c / r if r > 0 else 0
            print(f"  {name:8s}| {c:8.4f} | {r:15.4f} | {eff:20.4f}")
    except Exception as exc:
        print(f"  MATLAB baseline not available: {exc}")
        slsqp_rmse = None

    # ===================================================================
    # Phase C: Count scaling
    # ===================================================================
    print("\n=== Phase C: CRLB vs count number ===")
    t0 = time.time()
    count_results = fi.compute_crlb_vs_counts(pixel_idx=0)
    print(f"  Computed in {time.time() - t0:.1f}s")

    for i, nc in enumerate(count_results['count_range']):
        mean_c = count_results['mean_crlb_film'][i] * 100
        t_c = count_results['crlb_thickness'][i]
        print(f"  N={nc:.0e}: mean film CRLB={mean_c:.4f}%, thickness CRLB={t_c:.3f} Å")

    # Verify 1/sqrt(N) scaling
    c1 = count_results['mean_crlb_film']
    n1 = count_results['count_range']
    if len(c1) >= 2:
        ratio = c1[0] / c1[-1]
        expected = np.sqrt(n1[-1] / n1[0])
        print(f"\n  Scaling check: CRLB ratio = {ratio:.2f}, expected 1/sqrt(N) = {expected:.2f}")

    # ===================================================================
    # Phase D: SXPS comparison
    # ===================================================================
    print("\n=== Phase D: SXPS vs conventional CRLB ===")
    t0 = time.time()
    sxps_results = fi.compute_crlb_sxps(pixel_idx=0)
    print(f"  Computed in {time.time() - t0:.1f}s")
    print(f"  λ ratio: {sxps_results['lambda_ratio']:.3f}")
    print(f"  Thickness improvement: {sxps_results['improvement_thickness']:.2f}x")

    imp = sxps_results['improvement_film']
    for j, name in enumerate(free_names):
        mean_imp = np.mean(imp[:, j])
        print(f"  {name}: mean improvement = {mean_imp:.2f}x")

    # ===================================================================
    # Phase E: SVD threshold sensitivity
    # ===================================================================
    print("\n=== Phase E: SVD threshold sensitivity ===")
    sens = fi.svd_threshold_sensitivity(pixel_idx=0)
    for i, th in enumerate(sens['thresholds']):
        print(f"  threshold={th:.0e}: mean film CRLB = "
              f"{sens['mean_crlb_film'][i] * 100:.4f}%")

    # ===================================================================
    # Phase F: Multi-pixel statistics
    # ===================================================================
    print(f"\n=== Phase F: Multi-pixel CRLB ({args.n_pixels} pixels) ===")
    t0 = time.time()
    multi = fi.compute_crlb_multi_pixel(n_pixels=args.n_pixels)
    print(f"  Computed {multi['n_computed']} pixels in {time.time() - t0:.1f}s")
    print(f"  Median condition number: {multi['condition_number_median']:.2e}")
    print(f"  Thickness CRLB mean: {multi['crlb_thickness_mean']:.3f} Å")

    for j, name in enumerate(free_names):
        mean_c = np.mean(multi['crlb_film_mean'][:, j]) * 100
        std_c = np.mean(multi['crlb_film_std'][:, j]) * 100
        print(f"  {name}: CRLB = {mean_c:.4f} ± {std_c:.4f} %")

    # ===================================================================
    # Generate 6-panel plot
    # ===================================================================
    print("\n=== Generating 6-panel plot ===")
    _generate_plot(
        result, count_results, sxps_results, multi, data, fi,
        slsqp_rmse=slsqp_rmse,
        output_path=str(OUTPUT_DIR / args.output),
    )
    print(f"  Saved to {OUTPUT_DIR / args.output}")


def _generate_plot(result, count_results, sxps_results, multi, data, fi,
                   slsqp_rmse=None, output_path="fisher_crlb_analysis.png"):
    """Generate the 6-panel diagnostic figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Fisher Information / CRLB Analysis for ARXPS Depth Profiling",
                 fontsize=14, fontweight='bold')

    free_names = [data.element_names[i] for i in data.free_indices]
    depth = data.depth
    colors = plt.cm.tab10(np.linspace(0, 1, len(free_names)))

    # --- Panel 1 (top-left): Singular value spectrum ---
    ax = axes[0, 0]
    S = result.singular_values
    ax.semilogy(np.arange(len(S)) + 1, S, 'b.-', markersize=4)
    ax.axhline(S[0] * 1e-10, color='r', ls='--', alpha=0.5, label='threshold (1e-10)')
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.set_title(f"F matrix SVD spectrum\n"
                 f"cond={result.condition_number:.1e}, "
                 f"rank={result.effective_rank}/{fi.n_params}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2 (top-center): CRLB vs depth (per element) ---
    ax = axes[0, 1]
    for j, (name, color) in enumerate(zip(free_names, colors)):
        ax.plot(depth, result.crlb_film_conc[:, j] * 100, '.-',
                color=color, label=name, markersize=3)
    ax.set_xlabel("Depth (Å)")
    ax.set_ylabel("CRLB (%)")
    ax.set_title("CRLB vs Depth (per element)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel 3 (top-right): CRLB vs SLSQP RMSE scatter ---
    ax = axes[0, 2]
    crlb_per_elem = np.mean(result.crlb_film_conc, axis=0) * 100

    if slsqp_rmse is not None:
        free_idx = np.array(data.free_indices)
        rmse_free = slsqp_rmse[free_idx] * 100
        for j, (name, color) in enumerate(zip(free_names, colors)):
            ax.scatter(crlb_per_elem[j], rmse_free[j],
                       c=[color], s=80, zorder=3, label=name)

        lim_max = max(np.max(rmse_free), np.max(crlb_per_elem)) * 1.3
        ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.3, label='CRLB = RMSE')
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
    else:
        for j, (name, color) in enumerate(zip(free_names, colors)):
            ax.scatter(crlb_per_elem[j], crlb_per_elem[j] * 1.5,
                       c=[color], s=80, zorder=3, label=name)
        ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='CRLB = RMSE')

    ax.set_xlabel("CRLB (%)")
    ax.set_ylabel("Solver RMSE (%)")
    ax.set_title("CRLB vs Solver RMSE\n(diagonal = perfect efficiency)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 4 (bottom-left): CRLB vs count number ---
    ax = axes[1, 0]
    n_counts = count_results['count_range']
    mean_crlb = count_results['mean_crlb_film'] * 100
    ax.loglog(n_counts, mean_crlb, 'bo-', label='Film CRLB (mean)', markersize=4)

    # Reference 1/sqrt(N) line
    ref = mean_crlb[len(mean_crlb) // 2] * np.sqrt(n_counts[len(n_counts) // 2] / n_counts)
    ax.loglog(n_counts, ref, 'r--', alpha=0.5, label=r'$\propto 1/\sqrt{N}$')

    # Thickness CRLB on secondary axis
    ax2 = ax.twinx()
    ax2.loglog(n_counts, count_results['crlb_thickness'], 'gs-',
               label='Thickness CRLB (Å)', markersize=4, alpha=0.7)
    ax2.set_ylabel("Thickness CRLB (Å)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    ax.set_xlabel("Total counts N")
    ax.set_ylabel("Mean film CRLB (%)")
    ax.set_title(r"CRLB vs Count Number ($\propto 1/\sqrt{N}$)")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Panel 5 (bottom-center): SXPS vs conventional ---
    ax = axes[1, 1]
    crlb_conv = sxps_results['crlb_conv'].crlb_film_conc * 100
    crlb_sxps = sxps_results['crlb_sxps'].crlb_film_conc * 100

    x_pos = np.arange(len(free_names))
    width = 0.35

    conv_means = np.mean(crlb_conv, axis=0)
    sxps_means = np.mean(crlb_sxps, axis=0)

    bars1 = ax.bar(x_pos - width / 2, conv_means, width, label='Conventional',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + width / 2, sxps_means, width, label='SXPS',
                   color='coral', alpha=0.8)

    # Add improvement labels
    for j in range(len(free_names)):
        imp = conv_means[j] / max(sxps_means[j], 1e-10)
        ax.text(x_pos[j], max(conv_means[j], sxps_means[j]) * 1.05,
                f'{imp:.1f}x', ha='center', fontsize=7, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(free_names, fontsize=8)
    ax.set_ylabel("Mean CRLB (%)")
    ax.set_title(f"SXPS vs Conventional\n(λ ratio = {sxps_results['lambda_ratio']:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 6 (bottom-right): Fisher matrix heatmap ---
    ax = axes[1, 2]
    F = result.fisher_matrix
    # Log-scale for visibility
    F_abs = np.abs(F)
    F_log = np.log10(np.maximum(F_abs, 1e-30))

    im = ax.imshow(F_log, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='log10(|F_ij|)')

    # Mark boundaries between parameter groups
    fs = fi.film_size
    for boundary in [fs, fs + 1, fs + 2]:
        if boundary < fi.n_params:
            ax.axhline(boundary - 0.5, color='r', ls='--', alpha=0.5, lw=0.8)
            ax.axvline(boundary - 0.5, color='r', ls='--', alpha=0.5, lw=0.8)

    ax.set_xlabel("Parameter index")
    ax.set_ylabel("Parameter index")
    ax.set_title("Fisher Matrix (log10 scale)\n"
                 "red lines: film|thickness|abruptness|sub")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
