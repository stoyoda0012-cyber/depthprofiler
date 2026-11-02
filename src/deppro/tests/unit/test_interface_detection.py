"""
Test Suite for Model-Free Interface Detection
==============================================

Tests gradient, 50% crossing, and inflection point methods
for detecting interface positions without model assumptions.

Usage:
    python -m deppro.test_interface_detection
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from deppro.depth_profile import (
    create_multilayer_profile,
    detect_interfaces_gradient,
    detect_interfaces_50percent,
    detect_interfaces_inflection,
    detect_interfaces,
)


def test_gradient_detection():
    """Test gradient-based interface detection."""
    print("=" * 60)
    print("Test 1: Gradient-Based Interface Detection")
    print("=" * 60)

    # Create known profile
    true_thicknesses = [5.0, 10.0, 20.0, 30.0]
    true_positions = np.cumsum(true_thicknesses)  # [5, 15, 35, 65]

    profile = create_multilayer_profile(
        layer_thicknesses=true_thicknesses,
        interface_widths=[2.0, 3.0, 4.0, 5.0],
    )

    # Detect interfaces
    detected = detect_interfaces_gradient(profile.depth, profile.concentrations)

    print(f"  True interface positions: {true_positions}")
    print(f"  Detected positions: {detected}")

    # Check that we detected reasonable number of interfaces
    print(f"  Number detected: {len(detected)}")

    # Check accuracy for detected interfaces
    errors = []
    for true_pos in true_positions:
        if len(detected) > 0:
            closest = detected[np.argmin(np.abs(detected - true_pos))]
            error = abs(closest - true_pos)
            errors.append(error)
            print(f"    True {true_pos:.1f} Å -> Detected {closest:.1f} Å (error: {error:.1f} Å)")

    print("Test 1 passed\n")
    return detected, true_positions


def test_50percent_detection():
    """Test 50% crossing interface detection."""
    print("=" * 60)
    print("Test 2: 50% Crossing Interface Detection")
    print("=" * 60)

    # Create known profile
    true_thicknesses = [5.0, 10.0, 20.0, 30.0]
    true_positions = np.cumsum(true_thicknesses)

    profile = create_multilayer_profile(
        layer_thicknesses=true_thicknesses,
        interface_widths=[2.0, 3.0, 4.0, 5.0],
    )

    # Test on different elements
    for elem_idx, label in enumerate(profile.element_labels):
        detected = detect_interfaces_50percent(
            profile.depth,
            profile.concentrations,
            element_index=elem_idx,
            direction='both'
        )
        print(f"  {label}: {len(detected)} interfaces at {detected}")

    print("Test 2 passed\n")


def test_inflection_detection():
    """Test inflection point interface detection."""
    print("=" * 60)
    print("Test 3: Inflection Point Interface Detection")
    print("=" * 60)

    true_thicknesses = [5.0, 10.0, 20.0, 30.0]
    true_positions = np.cumsum(true_thicknesses)

    profile = create_multilayer_profile(
        layer_thicknesses=true_thicknesses,
        interface_widths=[2.0, 3.0, 4.0, 5.0],
    )

    detected = detect_interfaces_inflection(profile.depth, profile.concentrations)

    print(f"  True interface positions: {true_positions}")
    print(f"  Detected positions: {detected}")

    print("Test 3 passed\n")


def test_consensus_detection():
    """Test combined consensus detection."""
    print("=" * 60)
    print("Test 4: Consensus Interface Detection")
    print("=" * 60)

    true_thicknesses = [5.0, 10.0, 20.0, 30.0]
    true_positions = np.cumsum(true_thicknesses)  # [5, 15, 35, 65]

    profile = create_multilayer_profile(
        layer_thicknesses=true_thicknesses,
        interface_widths=[2.0, 3.0, 4.0, 5.0],
    )

    result = detect_interfaces(
        profile.depth,
        profile.concentrations,
        n_interfaces=4,
        labels=profile.element_labels,
    )

    print(f"  True positions: {true_positions}")
    print(f"  Gradient method: {result.gradient_positions}")
    print(f"  50% method: {result.percent50_positions}")
    print(f"  Inflection method: {result.inflection_positions}")
    print(f"\n  Consensus positions: {result.consensus_positions}")
    print(f"  Consensus std: {result.consensus_std}")

    # Compare consensus with true positions
    print("\n  Comparison with true positions:")
    for i, true_pos in enumerate(true_positions):
        if i < len(result.consensus_positions):
            detected = result.consensus_positions[i]
            error = abs(detected - true_pos)
            print(f"    Interface {i+1}: True {true_pos:.1f} Å, Detected {detected:.1f} Å, Error {error:.1f} Å")

    print("Test 4 passed\n")
    return result, true_positions, profile


def create_visualization(result, true_positions, profile, output_path):
    """Create visualization of interface detection results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    depth = profile.depth
    colors = ['gray', 'green', 'blue', 'red', 'orange', 'purple']

    # 1. Profile with all detected interfaces
    ax1 = axes[0, 0]
    for i, label in enumerate(profile.element_labels):
        ax1.plot(depth, profile.concentrations[:, i],
                 label=label, color=colors[i], linewidth=2, alpha=0.7)

    # Mark true interfaces
    for pos in true_positions:
        ax1.axvline(pos, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

    # Mark consensus interfaces
    for pos in result.consensus_positions:
        ax1.axvline(pos, color='red', linestyle='-', alpha=0.8, linewidth=2)

    ax1.set_xlabel('Depth (Å)')
    ax1.set_ylabel('Concentration')
    ax1.set_title('Profile with Detected Interfaces\n(black dashed=true, red=detected)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)

    # 2. Gradient profile
    ax2 = axes[0, 1]
    grad_sum = np.zeros_like(depth)
    for i in range(profile.concentrations.shape[1]):
        grad = np.abs(np.gradient(profile.concentrations[:, i], depth))
        grad_sum += grad
        ax2.plot(depth, grad, label=profile.element_labels[i], alpha=0.5)
    ax2.plot(depth, grad_sum, 'k-', label='Sum', linewidth=2)

    for pos in result.gradient_positions:
        ax2.axvline(pos, color='red', linestyle='--', alpha=0.7)

    ax2.set_xlabel('Depth (Å)')
    ax2.set_ylabel('|∂c/∂z|')
    ax2.set_title('Gradient Profile\n(red lines = detected interfaces)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)

    # 3. Method comparison
    ax3 = axes[1, 0]
    methods = ['Gradient', '50%', 'Inflection', 'Consensus']
    all_results = [
        result.gradient_positions,
        result.percent50_positions,
        result.inflection_positions,
        result.consensus_positions,
    ]

    # Create offset for visualization
    for i, (method, positions) in enumerate(zip(methods, all_results)):
        y = np.ones(len(positions)) * (3 - i)
        ax3.scatter(positions, y, s=100, label=method)

    # Add true positions
    ax3.scatter(true_positions, np.ones(len(true_positions)) * 4,
                s=150, marker='v', color='black', label='True')

    ax3.set_xlabel('Depth (Å)')
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.set_yticklabels(['Consensus', 'Inflection', '50%', 'Gradient', 'True'])
    ax3.set_title('Interface Detection Comparison')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(-0.5, 4.5)

    # 4. Error summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Compute errors for each method
    def compute_errors(detected, true):
        if len(detected) == 0:
            return None
        errors = []
        for t in true:
            if len(detected) > 0:
                closest = detected[np.argmin(np.abs(detected - t))]
                errors.append(abs(closest - t))
        return np.array(errors)

    grad_errors = compute_errors(result.gradient_positions, true_positions)
    p50_errors = compute_errors(result.percent50_positions, true_positions)
    inflect_errors = compute_errors(result.inflection_positions, true_positions)
    consensus_errors = compute_errors(result.consensus_positions, true_positions)

    summary = f"""
    Interface Detection Summary
    ===========================

    True interface positions: {true_positions.tolist()} Å

    Gradient Method:
      Detected: {result.gradient_positions.tolist()}
      Mean error: {np.mean(grad_errors):.2f} Å (if matched)

    50% Crossing Method:
      Detected: {result.percent50_positions.tolist()}
      Mean error: {np.mean(p50_errors) if p50_errors is not None and len(p50_errors) > 0 else 'N/A'} Å

    Inflection Method:
      Detected: {result.inflection_positions.tolist()}
      Mean error: {np.mean(inflect_errors) if inflect_errors is not None and len(inflect_errors) > 0 else 'N/A'} Å

    Consensus (All Methods Combined):
      Detected: {[f'{p:.1f}' for p in result.consensus_positions]}
      Std dev: {[f'{s:.1f}' for s in result.consensus_std]}
      Mean error: {np.mean(consensus_errors) if consensus_errors is not None and len(consensus_errors) > 0 else 'N/A'} Å
    """

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to {output_path}")


def run_all_tests():
    """Run all interface detection tests."""
    print("=" * 60)
    print("Model-Free Interface Detection Test Suite")
    print("=" * 60)
    print()

    test_gradient_detection()
    test_50percent_detection()
    test_inflection_detection()
    result, true_positions, profile = test_consensus_detection()

    # Create visualization
    output_path = Path('/tmp/interface_detection.png')
    create_visualization(result, true_positions, profile, output_path)

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
