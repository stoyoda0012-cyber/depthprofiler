"""
Quick test of gradient mode parameter handling.
Focus on internal mechanics, not full optimization.

Date: 2026-02-03
"""

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # <repo>/src

from deppro.inverse_solver import SemiFreeSolverConfig, SemiFreeSolverResult
from deppro.paths import ensure_outputs_dir


def test_gradient_config():
    """Test gradient configuration."""
    print("Test 1: Configuration")
    print("-" * 40)

    # Default (no gradient)
    cfg1 = SemiFreeSolverConfig()
    assert cfg1.gradient_mode == False
    print("  Default gradient_mode: False ✓")

    # With gradient
    cfg2 = SemiFreeSolverConfig(
        gradient_mode=True,
        gradient_type='linear',
        gradient_element_indices=[0, 1],
        gradient_bounds={0: (-0.01, 0.01), 1: (-0.02, 0.02)},
    )
    assert cfg2.gradient_mode == True
    assert cfg2.gradient_element_indices == [0, 1]
    print("  Gradient enabled: ✓")
    print("  Multiple elements: ✓")
    print()


def test_result_properties():
    """Test SemiFreeSolverResult gradient properties."""
    print("Test 2: Result Properties")
    print("-" * 40)

    # Result without gradient
    result1 = SemiFreeSolverResult(
        D_opt=np.zeros((20, 3)),
        I_opt=np.zeros((5, 3)),
        thickness=25.0,
        abruptness=5.0,
        substrate_ratios=np.array([1.0]),
        chi2=0.01,
        objective=0.02,
        n_iterations=50,
        elapsed_time=1.0,
        success=True,
    )
    assert result1.has_gradient == False
    assert result1.get_gradient_slopes() == {}
    print("  No gradient result: ✓")

    # Result with gradient
    result2 = SemiFreeSolverResult(
        D_opt=np.zeros((20, 3)),
        I_opt=np.zeros((5, 3)),
        thickness=25.0,
        abruptness=5.0,
        substrate_ratios=np.array([1.0]),
        chi2=0.005,
        objective=0.015,
        n_iterations=80,
        elapsed_time=2.0,
        success=True,
        gradient_params=np.array([0.003, -0.001]),
        gradient_element_indices=[0, 2],
    )
    assert result2.has_gradient == True
    slopes = result2.get_gradient_slopes()
    assert slopes == {0: 0.003, 2: -0.001}
    print("  Gradient result: ✓")
    print(f"  Slopes: {slopes}")
    print()


def test_gradient_profile_effect():
    """Visualize the effect of gradient on profiles."""
    print("Test 3: Gradient Profile Visualization")
    print("-" * 40)

    depth = np.arange(0, 50, 1.0)
    thickness = 25.0

    # Create base profile (film + substrate)
    sigma = 5.0 / (2 * np.sqrt(np.log(2)))
    substrate = 0.5 * (1 + erf((depth - thickness) / sigma))
    film_base = 1.0 - substrate

    # Different gradients
    slopes = [-0.01, 0, 0.01, 0.02]
    colors = ['blue', 'gray', 'green', 'red']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Full profiles
    ax1 = axes[0]
    for slope, color in zip(slopes, colors):
        # Apply gradient to base concentration
        base_conc = 0.3
        conc = base_conc + slope * depth
        conc = np.clip(conc, 0.05, 0.95)
        film = conc * film_base
        ax1.plot(depth, film, color=color, lw=2, label=f'slope={slope:.2f}')

    ax1.plot(depth, substrate, 'k--', lw=2, label='Substrate')
    ax1.axvline(thickness, color='gray', ls=':', alpha=0.5)
    ax1.set_xlabel('Depth (Å)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('Film Concentration with Different Gradients')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)

    # Right: Concentration at interface region
    ax2 = axes[1]
    z_interface = depth[(depth > 15) & (depth < 35)]

    for slope, color in zip(slopes, colors):
        conc = 0.3 + slope * z_interface
        ax2.plot(z_interface, conc, color=color, lw=2, marker='o', ms=4,
                label=f'slope={slope:.2f}')

    ax2.axvline(thickness, color='gray', ls='--', alpha=0.5, label='Interface')
    ax2.set_xlabel('Depth (Å)')
    ax2.set_ylabel('Base Concentration')
    ax2.set_title('Concentration Change Near Interface')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = str(ensure_outputs_dir() / 'gradient_profile_effect.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    print()

    return fig


def test_arxps_sensitivity():
    """Show how ARXPS signal changes with gradient (theoretical)."""
    print("Test 4: ARXPS Sensitivity to Gradient")
    print("-" * 40)

    depth = np.arange(0, 60, 1.0)
    angles = np.array([0, 15, 30, 45, 60])
    thickness = 25.0
    imfp = 20.0

    # Create profiles with different gradients
    sigma = 5.0 / (2 * np.sqrt(np.log(2)))
    substrate = 0.5 * (1 + erf((depth - thickness) / sigma))
    film_base = 1.0 - substrate

    slopes = [0, 0.005, 0.01]
    step = depth[1] - depth[0]

    fig, ax = plt.subplots(figsize=(8, 6))

    for slope in slopes:
        conc = 0.3 + slope * depth
        conc = np.clip(conc, 0.1, 0.9)
        film = conc * film_base

        # Compute ARXPS signal
        I = np.zeros(len(angles))
        for i, angle in enumerate(angles):
            cos_theta = np.cos(np.radians(angle))
            for j, z in enumerate(depth):
                I[i] += film[j] * np.exp(-z / (imfp * cos_theta)) * step

        # Normalize
        I = I / I[0]
        ax.plot(angles, I, 'o-', lw=2, ms=8, label=f'slope={slope:.3f}')

    ax.set_xlabel('Angle (°)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('ARXPS Signal Dependence on Gradient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = str(ensure_outputs_dir() / 'gradient_arxps_sensitivity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    # Calculate signal difference
    print()
    print("  Signal change (normalized to 0° angle):")
    for slope in slopes:
        conc = 0.3 + slope * depth
        conc = np.clip(conc, 0.1, 0.9)
        film = conc * film_base
        I_0 = sum(film * np.exp(-depth / imfp) * step)
        I_60 = sum(film * np.exp(-depth / (imfp * np.cos(np.radians(60)))) * step)
        ratio = I_60 / I_0
        print(f"    slope={slope:.3f}: I(60°)/I(0°) = {ratio:.4f}")

    print()

    return fig


def main():
    print("=" * 60)
    print("Gradient Mode Quick Tests")
    print("=" * 60)
    print()

    test_gradient_config()
    test_result_properties()
    test_gradient_profile_effect()
    test_arxps_sensitivity()

    print("=" * 60)
    print("All Quick Tests Passed!")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
