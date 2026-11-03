"""
XPS SiO2/Si thickness calculation - integrated analysis of the effect of noise
==========================================
Comparison of Gaussian noise (given as relative %) and Poisson noise (counting statistics)

Usage:
    cd <repo>
    python src/deppro/examples/theory/xps_noise_summary.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Output directory
from deppro.paths import ensure_outputs_dir
import textwrap


def main():
    OUTPUT_DIR = ensure_outputs_dir()

    # Physical parameters
    LAMBDA_OX = 2.96  # IMFP of Si 2p in SiO2 (nm)
    R_FACTOR = 0.87   # sensitivity-factor ratio

    def calculate_thickness(I_ox, I_si):
        """Strohmeier formula: d = λ·ln(1 + R·I_ox/I_si)"""
        return LAMBDA_OX * np.log(1 + R_FACTOR * I_ox / I_si)

    def mc_gaussian(I_ox, I_si, noise_pct, n=10000):
        """Monte Carlo with Gaussian noise"""
        d_list = []
        for _ in range(n):
            Iox = max(1, I_ox + np.random.normal(0, I_ox * noise_pct / 100))
            Isi = max(1, I_si + np.random.normal(0, I_si * noise_pct / 100))
            d_list.append(calculate_thickness(Iox, Isi))
        return np.array(d_list)

    def mc_poisson(I_ox, I_si, n=10000):
        """Monte Carlo with Poisson noise"""
        d_list = []
        for _ in range(n):
            Iox = max(1, np.random.poisson(I_ox))
            Isi = max(1, np.random.poisson(I_si))
            d_list.append(calculate_thickness(Iox, Isi))
        return np.array(d_list)

    def analytical_error(I_ox, I_si, sigma_ox, sigma_si):
        """Analytical error propagation"""
        denom = I_si + R_FACTOR * I_ox
        dd_dIox = LAMBDA_OX * R_FACTOR / denom
        dd_dIsi = -LAMBDA_OX * R_FACTOR * I_ox / (I_si * denom)
        return np.sqrt((dd_dIox * sigma_ox)**2 + (dd_dIsi * sigma_si)**2)

    # ===== Integrated analysis =====
    print("=" * 70)
    print("XPS SiO2/Si thickness calculation - integrated summary of the effect of noise")
    print("=" * 70)
    print(f"\n[Thickness formula] d = λ·ln(1 + R·I_ox/I_si)")
    print(f"  λ(SiO2) = {LAMBDA_OX} nm,  R = {R_FACTOR}")

    # Representative thickness cases
    cases = [
        {"name": "Ultra-thin", "d_target": 0.5},
        {"name": "Thin", "d_target": 1.0},
        {"name": "Medium", "d_target": 2.0},
        {"name": "Thick", "d_target": 3.0},
        {"name": "Very thick", "d_target": 4.0},
    ]

    # Thickness back from the intensity ratio
    for c in cases:
        c['ratio'] = (np.exp(c['d_target'] / LAMBDA_OX) - 1) / R_FACTOR

    # Compare at a fixed total count
    total_counts = 10000

    print(f"\n[Comparison at total counts = {total_counts:,}]")
    print("-" * 70)
    print(f"{'Thickness':>10} | {'Poisson σ':>12} | {'Gauss 2% σ':>12} | {'Poisson':>10} | {'Gauss 2%':>10}")
    print(f"{'(nm)':>10} | {'(nm)':>12} | {'(nm)':>12} | {'Rel.Err':>10} | {'Rel.Err':>10}")
    print("-" * 70)

    results = []
    for c in cases:
        ratio = c['ratio']
        I_si = int(total_counts / (1 + ratio))
        I_ox = total_counts - I_si
        d_true = calculate_thickness(I_ox, I_si)

        # Poisson
        mc_p = mc_poisson(I_ox, I_si)
        sigma_p = np.std(mc_p)

        # Gaussian 2%
        mc_g = mc_gaussian(I_ox, I_si, 2.0)
        sigma_g = np.std(mc_g)

        results.append({
            'name': c['name'], 'd_true': d_true, 'ratio': ratio,
            'I_ox': I_ox, 'I_si': I_si,
            'sigma_p': sigma_p, 'sigma_g': sigma_g,
            'rel_p': sigma_p/d_true*100, 'rel_g': sigma_g/d_true*100,
            'mc_p': mc_p, 'mc_g': mc_g
        })

        print(f"{d_true:>10.2f} | {sigma_p:>12.4f} | {sigma_g:>12.4f} | {sigma_p/d_true*100:>9.2f}% | {sigma_g/d_true*100:>9.2f}%")

    # Noise-level correspondence table
    print("\n" + "=" * 70)
    print("[Gaussian noise % ↔ Poisson counts]")
    print("-" * 70)
    print("  Gaussian noise: σ = I × (noise%/100) for each peak")
    print("  Poisson noise: σ = √N")
    print("  Equivalence: noise% = 100/√N  ⇔  N = (100/noise%)²")
    print("-" * 70)
    print(f"{'Gauss Noise':>12} | {'Equiv. Counts':>15} | {'Notes':>30}")
    print("-" * 70)
    equiv_table = [
        (0.5, "Ultra-high precision"),
        (1.0, "High precision"),
        (2.0, "Standard measurement"),
        (3.0, "Quick scan"),
        (5.0, "Survey scan"),
        (10.0, "Very noisy"),
    ]
    for noise, note in equiv_table:
        counts = int((100/noise)**2)
        print(f"{noise:>11.1f}% | {counts:>15,} | {note:>30}")

    # Required counts
    print("\n" + "=" * 70)
    print("[Requirements for a target precision]")
    print("-" * 70)

    def required_counts_for_precision(d_target, sigma_target):
        """Counts needed for a target precision"""
        ratio = (np.exp(d_target / LAMBDA_OX) - 1) / R_FACTOR
        # Bisection
        low, high = 100, 1e7
        while high - low > 100:
            mid = (low + high) / 2
            I_si = mid / (1 + ratio)
            I_ox = mid - I_si
            sigma = analytical_error(I_ox, I_si, np.sqrt(I_ox), np.sqrt(I_si))
            if sigma > sigma_target:
                low = mid
            else:
                high = mid
        return int(high)

    print(f"\nFor a 1 nm film:")
    for target in [0.1, 0.05, 0.02, 0.01]:
        N = required_counts_for_precision(1.0, target)
        equiv_noise = 100 / np.sqrt(N)
        print(f"  σ ≤ {target:.2f} nm ({target/1.0*100:.0f}%): {N:>10,} counts (≈ {equiv_noise:.1f}% Gauss)")

    print(f"\nFor a 3 nm film:")
    for target in [0.1, 0.05, 0.02, 0.01]:
        N = required_counts_for_precision(3.0, target)
        equiv_noise = 100 / np.sqrt(N)
        print(f"  σ ≤ {target:.2f} nm ({target/3.0*100:.1f}%): {N:>10,} counts (≈ {equiv_noise:.1f}% Gauss)")

    # ===== Visualization =====
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8])

    # 1. Gaussian versus Poisson distributions (representative case)
    ax1 = fig.add_subplot(gs[0, 0])
    idx = 2  # Medium thickness
    r = results[idx]
    bins = np.linspace(min(r['mc_p'].min(), r['mc_g'].min()),
                       max(r['mc_p'].max(), r['mc_g'].max()), 50)
    ax1.hist(r['mc_p'], bins=bins, alpha=0.6, color='#e74c3c', label='Poisson', density=True)
    ax1.hist(r['mc_g'], bins=bins, alpha=0.6, color='#3498db', label='Gaussian 2%', density=True)
    ax1.axvline(r['d_true'], color='black', linestyle='--', lw=2, label=f"True: {r['d_true']:.2f} nm")
    ax1.set_xlabel('Thickness (nm)', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title(f"Distribution Comparison (d={r['d_true']:.1f} nm, 10k counts)", fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Thickness versus absolute error
    ax2 = fig.add_subplot(gs[0, 1])
    d_vals = [r['d_true'] for r in results]
    sigma_p = [r['sigma_p'] for r in results]
    sigma_g = [r['sigma_g'] for r in results]

    x = np.arange(len(d_vals))
    width = 0.35
    ax2.bar(x - width/2, sigma_p, width, label='Poisson (10k counts)', color='#e74c3c', alpha=0.8)
    ax2.bar(x + width/2, sigma_g, width, label='Gaussian 2%', color='#3498db', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{d:.1f}" for d in d_vals])
    ax2.set_xlabel('Thickness (nm)', fontsize=11)
    ax2.set_ylabel('Thickness Error σ (nm)', fontsize=11)
    ax2.set_title('Absolute Error: Poisson vs Gaussian', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    # 3. Thickness versus relative error
    ax3 = fig.add_subplot(gs[1, 0])
    rel_p = [r['rel_p'] for r in results]
    rel_g = [r['rel_g'] for r in results]

    ax3.plot(d_vals, rel_p, 'o-', color='#e74c3c', lw=2, ms=8, label='Poisson (10k counts)')
    ax3.plot(d_vals, rel_g, 's--', color='#3498db', lw=2, ms=8, label='Gaussian 2%')
    ax3.set_xlabel('Thickness (nm)', fontsize=11)
    ax3.set_ylabel('Relative Error (%)', fontsize=11)
    ax3.set_title('Relative Error vs Thickness', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Counts versus error
    ax4 = fig.add_subplot(gs[1, 1])
    count_range = np.logspace(2.5, 5.5, 30)
    d_test = 2.0
    ratio_test = (np.exp(d_test / LAMBDA_OX) - 1) / R_FACTOR

    errors_p = []
    errors_g = []
    for N in count_range:
        I_si = N / (1 + ratio_test)
        I_ox = N - I_si
        # Poisson analytical solution
        sigma_p_val = analytical_error(I_ox, I_si, np.sqrt(I_ox), np.sqrt(I_si))
        # Gaussian 2% analytical solution
        sigma_g_val = analytical_error(I_ox, I_si, I_ox*0.02, I_si*0.02)
        errors_p.append(sigma_p_val)
        errors_g.append(sigma_g_val)

    ax4.loglog(count_range, errors_p, '-', color='#e74c3c', lw=2, label='Poisson')
    ax4.loglog(count_range, errors_g, '--', color='#3498db', lw=2, label='Gaussian 2%')
    ax4.axhline(0.1, color='green', ls=':', lw=1.5, label='0.1 nm target')
    ax4.axhline(0.05, color='orange', ls=':', lw=1.5, label='0.05 nm target')
    ax4.set_xlabel('Total Counts', fontsize=11)
    ax4.set_ylabel('Thickness Error (nm)', fontsize=11)
    ax4.set_title(f'Error vs Counts (d={d_test} nm)', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.grid(alpha=0.3, which='both')
    ax4.set_xlim(300, 3e5)
    ax4.set_ylim(0.005, 0.5)

    # 5. Summary table (text)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    summary_text = textwrap.dedent("""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                  SUMMARY: XPS SiO₂/Si Thickness Error Analysis
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      Formula:  d = λ·ln(1 + R·I_ox/I_si)     λ = 2.96 nm (IMFP),  R = 0.87

      ┌─────────────────────────────────────────────────────────────────────────────────┐
      │  Noise Type Comparison (10,000 total counts)                                    │
      ├─────────────┬──────────────────────────────────────────────────────────────────┤
      │  Poisson    │  σ = √N  →  ~1% noise at 10k counts                              │
      │  Gaussian   │  σ = I × 2%  (user-specified percentage)                         │
      └─────────────┴──────────────────────────────────────────────────────────────────┘

      ┌─────────────────────────────────────────────────────────────────────────────────┐
      │  Noise % ↔ Count Equivalence:   N = (100/noise%)²                              │
      ├─────────────┬────────────────┬──────────────────────────────────────────────────┤
      │  0.5%       │   40,000       │  High-resolution measurement                     │
      │  1.0%       │   10,000       │  Standard measurement                            │
      │  2.0%       │    2,500       │  Quick measurement                               │
      │  5.0%       │      400       │  Survey scan                                     │
      └─────────────┴────────────────┴──────────────────────────────────────────────────┘

      ┌─────────────────────────────────────────────────────────────────────────────────┐
      │  Relations used above                                                           │
      ├─────────────────────────────────────────────────────────────────────────────────┤
      │  • Thinner films → Higher relative error (ln function sensitivity)             │
      │  • Counts × 4  →  Error ÷ 2  (√N scaling)                                       │
      │  • Required counts for a target precision: read from the panels (this run)     │
      └─────────────────────────────────────────────────────────────────────────────────┘
    """)

    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'xps_noise_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print("\n" + "=" * 70)
    print("[Conclusions]")
    print("=" * 70)
    print(textwrap.dedent("""
    ■ Gaussian versus Poisson:
        - equivalence: Gauss noise% = 100/√N  ⇔  N = (100/noise%)²
        - 2% Gaussian noise ≈ Poisson noise at 2,500 counts
        - measurements combine both (Poisson plus readout noise and the like)

    ■ Error propagation into the thickness:
        - thin films: large relative error (sensitivity of the logarithm)
        - thick films: large absolute error but small relative error

    ■ Guidance for experiment planning:
        - read the counts needed for a target precision from the figure panel (computed in this run).
    """))

    print(f"\nSaved the integrated summary figure: {output_path}")


if __name__ == "__main__":
    main()
