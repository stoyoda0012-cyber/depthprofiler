"""
Comparison with the theoretical literature
================
Compares the theoretical predictions of Cumpson (1995) with the present calculations

Key reference:
P.J. Cumpson, "Angle-resolved XPS and AES: Depth-resolution limits
and a general comparison of properties of depth-profile reconstruction methods"
J. Electron Spectros. Relat. Phenom. 73 (1995) 25-52

Cumpson's main conclusions:
1. Depth resolution: Δz/z ≈ 0.81 (relative resolution of about 81%)
2. At most 3 parameters per element can be obtained from ARXPS data
3. The S/N ratio, not the number of angles, limits the resolution
4. Optimal angles: 0°, 40°, 55°, 63°, 70° (weak elastic scattering)

Usage:
    cd <repo>
    python src/deppro/examples/theory/theory_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Output directory
from deppro.paths import ensure_outputs_dir
import textwrap


def main():
    OUTPUT_DIR = ensure_outputs_dir()

    # Physical parameters
    LAMBDA_OX = 2.96  # nm
    R_FACTOR = 0.87

    def calculate_thickness(I_ox, I_si):
        return LAMBDA_OX * np.log(1 + R_FACTOR * I_ox / I_si)

    def arxps_forward_model(theta_deg, d_ox, lambda_ox=LAMBDA_OX):
        theta_rad = np.radians(theta_deg)
        cos_theta = np.cos(theta_rad)
        lambda_eff = lambda_ox * cos_theta
        atten = np.exp(-d_ox / lambda_eff)
        ratio = (1 - atten) / (R_FACTOR * atten)
        return ratio

    def mc_depth_resolution(theta_array, d_true, total_counts, n_trials=500):
        """Depth resolution by Monte Carlo"""
        from scipy.optimize import curve_fit
        d_fits = []

        for _ in range(n_trials):
            ratios_noisy = []
            for theta in theta_array:
                cos_theta = np.cos(np.radians(theta))
                counts = total_counts * cos_theta
                ratio_true = arxps_forward_model(np.array([theta]), d_true)[0]
                I_si = counts / (1 + ratio_true)
                I_ox = counts - I_si
                I_ox_noisy = max(1, np.random.poisson(I_ox))
                I_si_noisy = max(1, np.random.poisson(I_si))
                ratios_noisy.append(I_ox_noisy / I_si_noisy)

            try:
                popt, _ = curve_fit(arxps_forward_model, theta_array, ratios_noisy,
                                   p0=[d_true], bounds=(0.01, 20))
                d_fits.append(popt[0])
            except:
                pass

        return np.mean(d_fits), np.std(d_fits)

    print("=" * 70)
    print("Comparison with the theoretical literature: Cumpson (1995) versus the present calculation")
    print("=" * 70)

    # ===== Cumpson's main predictions =====
    print(textwrap.dedent("""
    [Cumpson's theoretical predictions (1995)]
    ─────────────────────────────────────────────────────────────────────
    1. Relative depth resolution: Δz/z ≈ 0.81 (81%)
          → at depth z the resolution Δz is about 0.81z
          → this is an "information-theoretic limit" and depends on the S/N ratio

    2. Information content of ARXPS:
          - at most 3 parameters can be obtained from the ARXPS data of one element
          - 5 basis functions can represent every ARXPS signal
          - up to about 10 resolvable depth zones

    3. Limiting factor:
          - "the S/N ratio, not the number of angles, determines the resolution"
          - typical measurement precision: 2-3%

    4. Optimal measurement angles:
          - weak elastic scattering: 0°, 40°, 55°, 63°, 70°
          - strong elastic scattering: 0°, 33°, 45°, 54°, 60°
    ─────────────────────────────────────────────────────────────────────
    """))

    # ===== Comparison with the present calculation =====
    print("\n[Comparison with the present calculation]")
    print("=" * 70)

    # 1. Relative resolution
    print("\n■ Relative depth resolution Δz/z")
    print("-" * 70)

    theta_optimal = np.array([0, 40, 55, 63, 70])  # Cumpson's recommended angles
    theta_standard = np.array([0, 20, 40, 55, 65, 75, 80])  # angles used here

    d_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    counts_values = [2500, 10000, 50000]  # noise equivalent to 2%, 1%, 0.45%

    print(f"\nAngles used: Cumpson optimal {list(map(int, theta_optimal))}")
    print(f"{'d (nm)':>8} | {'2500 cnt':>12} | {'10000 cnt':>12} | {'50000 cnt':>12} | Cumpson prediction")
    print("-" * 70)

    results_comparison = []
    for d in d_values:
        row = f"{d:>8.1f} |"
        for counts in counts_values:
            d_mean, d_std = mc_depth_resolution(theta_optimal, d, counts, n_trials=300)
            rel_res = d_std / d
            row += f" {rel_res:>11.3f} |"
            results_comparison.append({'d': d, 'counts': counts, 'rel_res': rel_res, 'sigma': d_std})
        # Cumpson prediction (Δz/z ≈ 0.81 scaled by the noise)
        # In fact an "information-theoretic limit", approached under perfect conditions
        print(row + f" ~0.81 (theoretical limit)")

    # 2. Noise dependence
    print("\n\n■ Dependence on the S/N ratio")
    print("-" * 70)
    print("Cumpson: \"the depth resolution is limited by the S/N ratio, not by the number of angles\"")
    print()

    d_test = 2.0
    noise_levels = [10, 5, 2, 1, 0.5, 0.2]  # %
    equiv_counts = [(100/n)**2 for n in noise_levels]

    print(f"{'Noise %':>10} | {'Equiv counts':>12} | {'σ_d (nm)':>12} | {'Δz/z':>10}")
    print("-" * 60)

    noise_rows = []
    for noise, counts in zip(noise_levels, equiv_counts):
        if counts > 100:
            _, sigma = mc_depth_resolution(theta_optimal, d_test, int(counts), n_trials=200)
            noise_rows.append((noise, sigma / d_test))
            print(f"{noise:>10.1f} | {int(counts):>12,} | {sigma:>12.4f} | {sigma/d_test:>10.3f}")

    # 3. Effect of the number of angles (against Cumpson's claim)
    print("\n\n■ Effect of the number of angles (checking Cumpson's claim)")
    print("-" * 70)
    print("Cumpson: \"the resolution is set by the S/N ratio, not by the number of angles\"")
    print()

    angle_sets = [
        ([0, 55, 70], "3 angles"),
        ([0, 40, 55, 70], "4 angles"),
        ([0, 40, 55, 63, 70], "5 angles (Cumpson optimal)"),
        ([0, 30, 40, 50, 55, 63, 70], "7 angles"),
        ([0, 20, 30, 40, 50, 55, 60, 65, 70], "9 angles"),
    ]

    counts_fixed = 10000
    print(f"Fixed counts: {counts_fixed:,}")
    print(f"{'Number of angles':>25} | {'σ_d (nm)':>12} | {'Δz/z':>10} | improvement")
    print("-" * 65)

    base_sigma = None
    angle_sigma = {}
    for angles, name in angle_sets:
        _, sigma = mc_depth_resolution(np.array(angles), d_test, counts_fixed, n_trials=300)
        angle_sigma[len(angles)] = sigma
        if base_sigma is None:
            base_sigma = sigma
            improvement = "-"
        else:
            improvement = f"{base_sigma/sigma:.2f}x"
        print(f"{name:>25} | {sigma:>12.4f} | {sigma/d_test:>10.3f} | {improvement:>8}")

    # ===== Summary of the values of this run (no fixed conclusion text) =====
    # Only values computed in this run are printed. The random seed is not fixed, so
    # the values vary from run to run.
    noise_min, dz_min = noise_rows[-1]
    d2 = [(r['counts'], r['rel_res']) for r in results_comparison if r['d'] == d_test]
    slope = float(np.polyfit(np.log([c for c, _ in d2]), np.log([v for _, v in d2]), 1)[0])
    ratio_3_5 = angle_sigma[3] / angle_sigma[5]
    ratio_5_9 = angle_sigma[5] / angle_sigma[9]

    print("\n" + "=" * 70)
    print("[Summary of this run] (Cumpson's claims are summaries of the literature; the right side is the present value)")
    print("=" * 70)
    print(f"  Cumpson: Δz/z ≈ 0.81 for an arbitrary profile  | uniform film, 1 parameter, noise {noise_min:g}%: Δz/z = {dz_min:.3f}")
    print(f"  Cumpson: resolution is set by the S/N          | d = {d_test} nm: Δz/z ∝ counts^{slope:.2f} (-0.50 if 1/√N)")
    print(f"  Cumpson: the number of angles matters little    | ratio of σ_d: 3→5 angles {ratio_3_5:.2f}x, 5→9 angles {ratio_5_9:.2f}x")
    print()
    print("  Premise: the calculation here is a fit to a known model (uniform film, one parameter),")
    print("  a different question from Cumpson's setting of reconstructing an arbitrary unknown profile.")

    # ===== Visualization =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Relative resolution versus counts
    ax1 = axes[0, 0]
    for d in [1.0, 2.0, 3.0]:
        counts_range = [500, 1000, 2500, 5000, 10000, 25000, 50000]
        rel_res_list = []
        for c in counts_range:
            _, sigma = mc_depth_resolution(theta_optimal, d, c, n_trials=200)
            rel_res_list.append(sigma / d)
        ax1.loglog(counts_range, rel_res_list, 'o-', label=f'd = {d} nm')

    ax1.axhline(0.81, color='red', linestyle='--', linewidth=2, label='Cumpson limit (0.81)')
    ax1.axhline(0.1, color='green', linestyle=':', label='10% resolution')
    ax1.set_xlabel('Total Counts', fontsize=11)
    ax1.set_ylabel('Relative Resolution Δz/z', fontsize=11)
    ax1.set_title('Relative Depth Resolution vs Counts\n(Cumpson optimal angles)', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3, which='both')
    ax1.set_ylim(0.001, 1)

    # 2. Effect of the number of angles
    ax2 = axes[0, 1]
    n_angles = [3, 4, 5, 7, 9]
    # Draw the values of the same run as the table (no separate Monte Carlo)
    sigmas = [angle_sigma[len(angles)] for angles, _ in angle_sets]

    ax2.plot(n_angles, sigmas, 'o-', color='#e74c3c', lw=2, ms=10)
    ax2.set_xlabel('Number of Angles', fontsize=11)
    ax2.set_ylabel('Depth Resolution σ_d (nm)', fontsize=11)
    ax2.set_title('Effect of Angle Number\n(d=2nm, 10k counts)', fontsize=12)
    ax2.grid(alpha=0.3)

    # Add the theoretical √n scaling line
    n_theory = np.array([3, 4, 5, 7, 9])
    sigma_theory = sigmas[0] * np.sqrt(3) / np.sqrt(n_theory)
    ax2.plot(n_theory, sigma_theory, '--', color='gray', label='√n scaling (theory)')
    ax2.legend()

    # 3. Conceptual figure: parametric versus non-parametric
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.95, 'Depth Resolution Regimes', fontsize=14, fontweight='bold',
             ha='center', va='top', transform=ax3.transAxes)

    # Conceptual figure: draw only Cumpson's limit (literature value) and the values of this run
    ax3.fill_between([0, 5], [0.81, 0.81], [1, 1], alpha=0.3, color='red', label='Non-parametric (Cumpson)')
    ax3.axhline(0.81, color='red', linestyle='--', lw=2)
    ax3.text(2.5, 0.85, 'Cumpson limit: Δz/z = 0.81', fontsize=10, ha='center')
    ax3.axhline(dz_min, color='green', linestyle='-', lw=2,
                label=f'This run: uniform film, d = {d_test} nm, {noise_min:g}% noise')
    ax3.text(2.5, dz_min + 0.03, f'This run (d = {d_test} nm): Δz/z = {dz_min:.3f}', fontsize=10, ha='center')

    ax3.set_xlabel('Depth z (nm)', fontsize=11)
    ax3.set_ylabel('Relative Resolution Δz/z', fontsize=11)
    ax3.set_xlim(0, 5)
    ax3.set_ylim(0, 1)
    ax3.legend(loc='center right')
    ax3.set_title('Resolution Regimes in ARXPS', fontsize=12)

    # 4. Summary of conclusions
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = textwrap.dedent(f"""
    Summary: Theory vs Calculation
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Cumpson (1995) Theory:
    • Δz/z ≈ 0.81 for arbitrary profiles
    • Max 3 parameters per element
    • S/N ratio limits resolution

    This run (Monte Carlo, uniform film,
    one parameter; values vary run to run):
    • Δz/z = {dz_min:.3f} at {noise_min:g}% noise
    • Δz/z ∝ counts^{slope:.2f} (d = {d_test} nm)
    • σ_d ratio 3→5 angles: {ratio_3_5:.2f}x
    • σ_d ratio 5→9 angles: {ratio_5_9:.2f}x
    """)
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'theory_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved the figure: {output_path}")


if __name__ == "__main__":
    main()
