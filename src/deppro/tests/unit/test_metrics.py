"""
Test Suite for Profile Comparison Metrics
==========================================

Tests Wasserstein distance, centroid, and other metrics for
depth profile comparison.

Usage:
    pytest tests/unit/test_metrics.py -v
    python -m deppro.tests.unit.test_metrics
"""

import numpy as np


def test_wasserstein_basic():
    """Test W1 distance with simple distributions."""
    from deppro.depth_profile import wasserstein_distance_1d

    depth = np.linspace(0, 100, 101)

    # Two identical Gaussians
    sigma = 5.0
    g1 = np.exp(-(depth - 30)**2 / (2 * sigma**2))
    g2 = np.exp(-(depth - 30)**2 / (2 * sigma**2))
    w1_same = wasserstein_distance_1d(depth, g1, g2)
    assert w1_same < 0.1, f"W1 for identical should be ~0, got {w1_same}"

    # Shifted Gaussians (10 Å apart)
    g3 = np.exp(-(depth - 40)**2 / (2 * sigma**2))
    w1_shifted = wasserstein_distance_1d(depth, g1, g3)
    assert 8 < w1_shifted < 12, f"W1 for 10Å shift should be ~10, got {w1_shifted}"

    # Shifted Gaussians (20 Å apart)
    g4 = np.exp(-(depth - 50)**2 / (2 * sigma**2))
    w1_shifted_20 = wasserstein_distance_1d(depth, g1, g4)
    assert 18 < w1_shifted_20 < 22, f"W1 for 20Å shift should be ~20, got {w1_shifted_20}"


def test_centroid():
    """Test centroid calculation."""
    from deppro.depth_profile import centroid

    depth = np.linspace(0, 100, 101)

    # Gaussian centered at 30 Å
    g1 = np.exp(-(depth - 30)**2 / (2 * 5**2))
    c1 = centroid(depth, g1)
    assert abs(c1 - 30) < 0.5, f"Centroid should be ~30, got {c1}"

    # Gaussian centered at 60 Å
    g2 = np.exp(-(depth - 60)**2 / (2 * 5**2))
    c2 = centroid(depth, g2)
    assert abs(c2 - 60) < 0.5, f"Centroid should be ~60, got {c2}"

    # Bimodal distribution
    g3 = g1 + g2
    c3 = centroid(depth, g3)
    assert abs(c3 - 45) < 2, f"Centroid should be ~45, got {c3}"


def test_moments():
    """Test moment calculation."""
    from deppro.depth_profile import profile_moments

    depth = np.linspace(0, 100, 201)

    # Gaussian with known sigma
    true_center = 50.0
    true_sigma = 8.0
    g = np.exp(-(depth - true_center)**2 / (2 * true_sigma**2))

    moments = profile_moments(depth, g)
    assert abs(moments['m1'] - true_center) < 0.5
    assert abs(moments['sigma'] - true_sigma) < 0.5
    assert abs(moments['skewness']) < 0.1


def test_multilayer_comparison():
    """Test comparison on multilayer profiles."""
    from deppro.depth_profile import create_multilayer_profile, compare_with_array

    # Create true profile
    true_thicknesses = [5.0, 10.0, 20.0, 30.0]
    true_widths = [2.0, 3.0, 4.0, 5.0]

    profile_true = create_multilayer_profile(
        layer_thicknesses=true_thicknesses,
        interface_widths=true_widths,
    )

    # Create perturbed profile (slightly different thicknesses)
    perturbed_thicknesses = [6.0, 12.0, 22.0, 32.0]  # +1, +2, +2, +2 Å
    profile_perturbed = create_multilayer_profile(
        layer_thicknesses=perturbed_thicknesses,
        interface_widths=true_widths,
    )

    # Compare
    result = compare_with_array(
        profile_true.depth,
        profile_true.concentrations,
        profile_perturbed.concentrations,
        profile_true.element_labels,
    )

    # W1 should reflect the ~2 Å average shift
    assert result.w1_mean > 0.5, "W1 should detect the perturbation"
    assert result.w1_mean < 10.0, "W1 should be reasonable"


def run_all_tests():
    """Run all tests (standalone mode)."""
    print("=" * 60)
    print("Profile Comparison Metrics Test Suite")
    print("=" * 60)
    print()

    test_wasserstein_basic()
    print("Test 1: Wasserstein basic ... PASSED")

    test_centroid()
    print("Test 2: Centroid ... PASSED")

    test_moments()
    print("Test 3: Moments ... PASSED")

    test_multilayer_comparison()
    print("Test 4: Multilayer comparison ... PASSED")

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
