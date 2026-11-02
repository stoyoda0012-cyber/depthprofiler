"""
理論文献との比較
================
Cumpson (1995) の理論予測と今回の計算結果を比較

Key reference:
P.J. Cumpson, "Angle-resolved XPS and AES: Depth-resolution limits
and a general comparison of properties of depth-profile reconstruction methods"
J. Electron Spectros. Relat. Phenom. 73 (1995) 25-52

Cumpsonの主要な結論:
1. 深さ分解能: Δz/z ≈ 0.81 (相対分解能約81%)
2. ARXPSデータから得られるパラメータ数は最大3個/元素
3. 測定角度数ではなくS/N比が分解能を制限
4. 最適角度: 0°, 40°, 55°, 63°, 70° (弱い弾性散乱の場合)

Usage:
    cd <repo>
    python src/deppro/examples/theory/theory_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 出力ディレクトリ
from deppro.paths import ensure_outputs_dir
import textwrap


def main():
    OUTPUT_DIR = ensure_outputs_dir()

    # 物理パラメータ
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
        """モンテカルロで深さ分解能を計算"""
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
    print("理論文献との比較: Cumpson (1995) vs 今回の計算")
    print("=" * 70)

    # ===== Cumpsonの主要予測 =====
    print(textwrap.dedent("""
    【Cumpsonの理論予測 (1995)】
    ─────────────────────────────────────────────────────────────────────
    1. 相対深さ分解能: Δz/z ≈ 0.81 (81%)
       → 深さzの位置で、分解能Δzは約0.81z
       → これは「情報理論的な限界」であり、S/N比に依存

    2. ARXPSの情報量:
       - 1元素のARXPSデータから得られるパラメータ数は最大3個
       - 5つの基底関数で全てのARXPS信号を表現可能
       - 分解可能な深さゾーンは約10層まで

    3. 制限要因:
       - 「角度数ではなくS/N比が分解能を決める」
       - 典型的な測定精度: 2-3%

    4. 最適測定角度:
       - 弱い弾性散乱: 0°, 40°, 55°, 63°, 70°
       - 強い弾性散乱: 0°, 33°, 45°, 54°, 60°
    ─────────────────────────────────────────────────────────────────────
    """))

    # ===== 今回の計算結果との比較 =====
    print("\n【今回の計算結果との比較】")
    print("=" * 70)

    # 1. 相対分解能の検証
    print("\n■ 相対深さ分解能 Δz/z の検証")
    print("-" * 70)

    theta_optimal = np.array([0, 40, 55, 63, 70])  # Cumpsonの推奨角度
    theta_standard = np.array([0, 20, 40, 55, 65, 75, 80])  # 今回使用した角度

    d_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    counts_values = [2500, 10000, 50000]  # 2%, 1%, 0.45%相当のノイズ

    print(f"\n使用角度: Cumpson最適 {list(map(int, theta_optimal))}")
    print(f"{'d (nm)':>8} | {'2500 cnt':>12} | {'10000 cnt':>12} | {'50000 cnt':>12} | Cumpson予測")
    print("-" * 70)

    results_comparison = []
    for d in d_values:
        row = f"{d:>8.1f} |"
        for counts in counts_values:
            d_mean, d_std = mc_depth_resolution(theta_optimal, d, counts, n_trials=300)
            rel_res = d_std / d
            row += f" {rel_res:>11.3f} |"
            results_comparison.append({'d': d, 'counts': counts, 'rel_res': rel_res, 'sigma': d_std})
        # Cumpson予測 (Δz/z ≈ 0.81 をノイズでスケール)
        # 実際は「情報理論的限界」なので、完璧な条件で0.81に近づく
        print(row + f" ~0.81 (理論限界)")

    # 2. ノイズ依存性の確認
    print("\n\n■ S/N比依存性の確認")
    print("-" * 70)
    print("Cumpson: 「深さ分解能はS/N比で制限され、角度数ではない」")
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

    # 3. 角度数の影響（Cumpsonの主張との比較）
    print("\n\n■ 角度数の影響（Cumpsonの主張検証）")
    print("-" * 70)
    print("Cumpson: 「分解能は角度数ではなくS/N比で決まる」")
    print()

    angle_sets = [
        ([0, 55, 70], "3 angles"),
        ([0, 40, 55, 70], "4 angles"),
        ([0, 40, 55, 63, 70], "5 angles (Cumpson optimal)"),
        ([0, 30, 40, 50, 55, 63, 70], "7 angles"),
        ([0, 20, 30, 40, 50, 55, 60, 65, 70], "9 angles"),
    ]

    counts_fixed = 10000
    print(f"カウント数固定: {counts_fixed:,}")
    print(f"{'角度数':>25} | {'σ_d (nm)':>12} | {'Δz/z':>10} | 改善率")
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

    # ===== この実行で得た値の要約（固定の結論文は持たない） =====
    # 印字するのは、この実行で計算した値だけ。乱数は固定していないので、
    # 値は実行ごとに揺れる。
    noise_min, dz_min = noise_rows[-1]
    d2 = [(r['counts'], r['rel_res']) for r in results_comparison if r['d'] == d_test]
    slope = float(np.polyfit(np.log([c for c, _ in d2]), np.log([v for _, v in d2]), 1)[0])
    ratio_3_5 = angle_sigma[3] / angle_sigma[5]
    ratio_5_9 = angle_sigma[5] / angle_sigma[9]

    print("\n" + "=" * 70)
    print("【この実行の要約】（Cumpson の主張は文献の要約、右は今回の計算値）")
    print("=" * 70)
    print(f"  Cumpson: 任意プロファイルの Δz/z ≈ 0.81  | 均一膜 1 パラメータ、ノイズ {noise_min:g}%: Δz/z = {dz_min:.3f}")
    print(f"  Cumpson: 分解能は S/N で決まる          | d = {d_test} nm で Δz/z ∝ counts^{slope:.2f}（1/√N なら -0.50）")
    print(f"  Cumpson: 角度数の影響は小さい            | σ_d の比: 3→5 角度 {ratio_3_5:.2f}x、5→9 角度 {ratio_5_9:.2f}x")
    print()
    print("  前提: ここでの計算は既知モデル（均一膜、パラメータ 1 個）へのフィットで、")
    print("  任意の未知プロファイルを再構成する Cumpson の設定とは問いが異なる。")

    # ===== 可視化 =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 相対分解能 vs カウント数
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

    # 2. 角度数の効果
    ax2 = axes[0, 1]
    n_angles = [3, 4, 5, 7, 9]
    # 表と同じ実行の値を描く（別のモンテカルロで計算し直さない）
    sigmas = [angle_sigma[len(angles)] for angles, _ in angle_sets]

    ax2.plot(n_angles, sigmas, 'o-', color='#e74c3c', lw=2, ms=10)
    ax2.set_xlabel('Number of Angles', fontsize=11)
    ax2.set_ylabel('Depth Resolution σ_d (nm)', fontsize=11)
    ax2.set_title('Effect of Angle Number\n(d=2nm, 10k counts)', fontsize=12)
    ax2.grid(alpha=0.3)

    # 理論的なスケーリング √n の線を追加
    n_theory = np.array([3, 4, 5, 7, 9])
    sigma_theory = sigmas[0] * np.sqrt(3) / np.sqrt(n_theory)
    ax2.plot(n_theory, sigma_theory, '--', color='gray', label='√n scaling (theory)')
    ax2.legend()

    # 3. パラメトリック vs ノンパラメトリックの概念図
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.95, 'Depth Resolution Regimes', fontsize=14, fontweight='bold',
             ha='center', va='top', transform=ax3.transAxes)

    # 概念図: Cumpson の限界（文献値）と、この実行で得た値だけを描く
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

    # 4. 結論サマリー
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

    print(f"\n図を保存: {output_path}")


if __name__ == "__main__":
    main()
