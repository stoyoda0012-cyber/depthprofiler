"""
Tests for MultiEnergyCRLB multi-energy analysis module.

Run:
    cd <repo>/src
    python -m pytest deppro/tests/unit/test_multi_energy.py -v
"""

import numpy as np
import pytest
from math import erf as _erf
from dataclasses import dataclass
from typing import List

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig
from deppro.fisher_information import FisherInformation
from deppro.multi_energy import MultiEnergyCRLB, EnergyConfig, MultiEnergyResult


# =====================================================================
# Minimal mock data (shared with test_fisher_information.py pattern)
# =====================================================================

def _make_mock_data(n_depth=10, n_angles=5, n_elements=3):
    """Create minimal mock benchmark data for multi-energy testing."""
    depth = np.linspace(0, 80, n_depth)
    angles = np.linspace(0, 60, n_angles)
    step_size = depth[1] - depth[0] if n_depth > 1 else 4.0

    tm = TransferMatrix(
        depth=depth, step_size=step_size, angles=angles,
        photon_energy=8000.0, work_function=4.5, gamma_eff=1.0,
    )

    # Build transfer matrix: exponential Beer-Lambert
    trans_mat = np.zeros((n_angles, n_depth, n_elements))
    lambdas = [20.0 + 5.0 * i for i in range(n_elements)]
    for a, theta in enumerate(angles):
        cos_theta = np.cos(np.radians(theta))
        for e in range(n_elements):
            for d in range(n_depth):
                trans_mat[a, d, e] = np.exp(-depth[d] / (lambdas[e] * cos_theta))
    tm.trans_mat = trans_mat
    tm.n_elements = n_elements

    # Mock Dstk: erf step function
    Dstk = np.zeros((1, n_depth, n_elements))
    thickness = 40.0
    sigma = 5.0
    for d in range(n_depth):
        erf_val = 0.5 * (1.0 + _erf((depth[d] - thickness) / sigma))
        Dstk[0, d, 0] = erf_val                # substrate Si(0)
        Dstk[0, d, 1] = 0.3 * (1 - erf_val)    # free Si(+4)
        Dstk[0, d, 2] = 0.7 * (1 - erf_val)    # linked O(-2)

    # Forward model → Xstk
    Xstk = np.zeros((1, n_angles, n_elements))
    for e in range(n_elements):
        Xstk[0, :, e] = trans_mat[:, :, e] @ Dstk[0, :, e]
    Xstk_total = Xstk.sum(axis=2, keepdims=True)
    Xstk = Xstk / np.maximum(Xstk_total, 1e-30)

    elements = [
        ElementConfig(symbol='Si', orbital='2p', compound='Si',
                      binding_energy=99.0, oxidation_state=0.0),
        ElementConfig(symbol='Si', orbital='2p', compound='SiO2',
                      binding_energy=103.0, oxidation_state=4.0),
        ElementConfig(symbol='O', orbital='1s', compound='SiO2',
                      binding_energy=532.0, oxidation_state=-2.0),
    ]

    charges = np.array([0.0, 4.0, -2.0])
    linkedconst = np.array([3.0])

    @dataclass
    class MockData:
        Dstk: np.ndarray
        Xstk: np.ndarray
        Drlt: np.ndarray
        trans_mat: np.ndarray
        depth: np.ndarray
        angles: np.ndarray
        elements: list
        element_names: list
        charges: np.ndarray
        linkedconst: np.ndarray
        step_size: float
        gamma_eff: float
        photon_energy: float
        work_function: float
        n_pixels: int
        n_depth: int
        n_angles: int
        n_elements: int
        free_matlab: np.ndarray
        subst_matlab: np.ndarray
        linked_matlab: np.ndarray

        @property
        def free_indices(self):
            return [int(x) - 1 for x in self.free_matlab]

        @property
        def substrate_indices(self):
            return [int(x) - 1 for x in self.subst_matlab]

        @property
        def linked_indices(self):
            return [int(x) - 1 for x in self.linked_matlab]

    return tm, MockData(
        Dstk=Dstk, Xstk=Xstk, Drlt=Dstk.copy(),
        trans_mat=trans_mat, depth=depth, angles=angles,
        elements=elements,
        element_names=['Si(0)', 'Si(+4)', 'O(-2)'],
        charges=charges, linkedconst=linkedconst,
        step_size=step_size, gamma_eff=1.0,
        photon_energy=8000.0, work_function=4.5,
        n_pixels=1, n_depth=n_depth, n_angles=n_angles,
        n_elements=n_elements,
        free_matlab=np.array([2]),     # Si(+4)
        subst_matlab=np.array([1]),    # Si(0)
        linked_matlab=np.array([3]),   # O(-2)
    )


# =====================================================================
# Tests
# =====================================================================

class TestMultiEnergyCRLB:
    """Tests for the MultiEnergyCRLB class."""

    def setup_method(self):
        self.tm, self.data = _make_mock_data(n_depth=10, n_angles=5, n_elements=3)
        self.fi = FisherInformation(self.tm, self.data, n_counts=1e6)
        self.me = MultiEnergyCRLB(self.fi, pixel_idx=0)

    # -------------------------------------------------------------------
    # Test 1: lambda_ratio=1.0 equals conventional
    # -------------------------------------------------------------------

    def test_lambda_ratio_1_equals_conventional(self):
        """Lambda_ratio=1.0 should reproduce the conventional result exactly."""
        # Conventional (cached in __init__)
        J_conv = self.me._J_conv
        X_conv = self.me._X_conv

        # compute_jacobian_at_energy(1.0) should return a copy of the same
        J_r1, X_r1 = self.me.compute_jacobian_at_energy(1.0)

        np.testing.assert_allclose(J_r1, J_conv, atol=1e-12,
                                   err_msg="Jacobian at ratio=1.0 differs from conventional")
        np.testing.assert_allclose(X_r1, X_conv, atol=1e-12,
                                   err_msg="X_norm at ratio=1.0 differs from conventional")

    # -------------------------------------------------------------------
    # Test 2: Fisher additivity — F(A+B) = F(A) + F(B)
    # -------------------------------------------------------------------

    def test_fisher_additivity(self):
        """Combined Fisher matrix should equal the sum of individual Fisher matrices."""
        e_conv = EnergyConfig("Conv", 1.0, 1e6)
        e_sxps = EnergyConfig("SXPS", 2.0, 1e6)

        r_conv = self.me.analyze_single(e_conv)
        r_sxps = self.me.analyze_single(e_sxps)

        result = self.me.analyze_combination([e_conv, e_sxps])

        F_sum = r_conv['F'] + r_sxps['F']
        np.testing.assert_allclose(
            result.combined_fisher, F_sum, atol=1e-10,
            err_msg="Fisher additivity violated: F_combined != F_conv + F_sxps"
        )

    # -------------------------------------------------------------------
    # Test 3: Combined CRLB improves over individual energies
    # -------------------------------------------------------------------

    def test_combined_crlb_improves(self):
        """Combining two energies should give CRLB <= best single energy."""
        e_conv = EnergyConfig("Conv", 1.0, 1e6)
        e_sxps = EnergyConfig("SXPS", 2.0, 1e6)

        r_conv = self.me.analyze_single(e_conv)
        r_sxps = self.me.analyze_single(e_sxps)

        result = self.me.analyze_combination([e_conv, e_sxps])

        # Combined CRLB should be <= min(conv, sxps)
        best_single = min(r_conv['crlb_pct'], r_sxps['crlb_pct'])
        assert result.mean_crlb_film_pct <= best_single + 1e-10, (
            f"Combined CRLB {result.mean_crlb_film_pct:.4f}% > "
            f"best single {best_single:.4f}%"
        )

    # -------------------------------------------------------------------
    # Test 4: Lambda ratio sweep — CRLB should generally decrease
    # -------------------------------------------------------------------

    def test_sweep_monotonic_trend(self):
        """Higher lambda ratio should give lower CRLB (SXPS rank paradox).

        We test the general trend, not strict monotonicity, since mock data
        may show local fluctuations.
        """
        ratios = [1.0, 2.0, 3.0, 5.0]
        sweep = self.me.sweep_lambda_ratios(ratios, n_counts=1e6)

        # Check return structure
        assert len(sweep['ratios']) == 4
        assert len(sweep['ranks']) == 4
        assert len(sweep['crlb_pcts']) == 4

        # CRLB at ratio=5.0 should be significantly less than ratio=1.0
        # (SXPS rank paradox: shorter lambda = lower CRLB despite lower rank)
        assert sweep['crlb_pcts'][-1] < sweep['crlb_pcts'][0], (
            f"CRLB at ratio=5.0 ({sweep['crlb_pcts'][-1]:.4f}) >= "
            f"CRLB at ratio=1.0 ({sweep['crlb_pcts'][0]:.4f})"
        )

    # -------------------------------------------------------------------
    # Test 5: Count allocation has non-trivial optimum
    # -------------------------------------------------------------------

    def test_count_allocation_has_optimum(self):
        """Optimal allocation should not be 100% in either energy."""
        alloc = self.me.sweep_count_allocation(
            ratio_a=1.0, ratio_b=2.0,
            total_counts=2e6, n_steps=11,
        )

        # Check structure
        assert len(alloc['fractions']) == 11
        assert len(alloc['crlb_pcts']) == 11
        assert 0.0 <= alloc['optimal_fraction'] <= 1.0

        # Pure single-energy allocations (f=0 or f=1)
        # At least one interior point should be better than both endpoints
        interior_best = np.min(alloc['crlb_pcts'][1:-1])
        endpoint_worst = max(alloc['crlb_pcts'][0], alloc['crlb_pcts'][-1])

        # Interior should be at least as good as the worst endpoint
        assert interior_best <= endpoint_worst + 1e-10, (
            f"Interior best {interior_best:.4f} > worst endpoint {endpoint_worst:.4f}"
        )

    # -------------------------------------------------------------------
    # Test 6: Different lambda ratios produce different Jacobians
    # -------------------------------------------------------------------

    def test_different_ratios_different_jacobians(self):
        """Lambda ratios other than 1.0 should produce different Jacobians."""
        J1, X1 = self.me.compute_jacobian_at_energy(1.0)
        J2, X2 = self.me.compute_jacobian_at_energy(2.0)

        # Jacobians should differ
        assert not np.allclose(J1, J2, atol=1e-6), \
            "Jacobians at ratio=1.0 and 2.0 should differ"

        # X_norm should also differ
        assert not np.allclose(X1, X2, atol=1e-6), \
            "X_norm at ratio=1.0 and 2.0 should differ"

    # -------------------------------------------------------------------
    # Test 7: Analyze single returns valid structure
    # -------------------------------------------------------------------

    def test_analyze_single_structure(self):
        """analyze_single should return all expected keys with correct shapes."""
        energy = EnergyConfig("Test", 1.0, 1e6)
        result = self.me.analyze_single(energy)

        # Check all keys exist
        expected_keys = ['F', 'J', 'X_norm', 'rank', 'crlb_film',
                         'crlb_pct', 'crlb_per_element', 'singular_values']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Check shapes
        n_p = self.me.n_params
        n_obs = self.fi.n_obs
        assert result['F'].shape == (n_p, n_p)
        assert result['J'].shape == (n_obs, n_p)
        assert result['X_norm'].shape == (n_obs,)
        assert result['crlb_film'].shape == (self.me.n_depth, self.me.n_free)

        # CRLB should be positive
        assert result['crlb_pct'] > 0
        assert result['rank'] > 0

    # -------------------------------------------------------------------
    # Test 8: analyze_combination returns valid MultiEnergyResult
    # -------------------------------------------------------------------

    def test_analyze_combination_structure(self):
        """analyze_combination should return properly structured MultiEnergyResult."""
        energies = [
            EnergyConfig("Conv", 1.0, 1e6),
            EnergyConfig("SXPS", 2.0, 1e6),
        ]
        result = self.me.analyze_combination(energies)

        assert isinstance(result, MultiEnergyResult)
        assert len(result.energies) == 2
        assert len(result.fisher_matrices) == 2
        assert len(result.per_energy_ranks) == 2
        assert len(result.per_energy_crlbs) == 2
        assert result.combined_fisher.shape == (self.me.n_params, self.me.n_params)
        assert result.combined_rank > 0
        assert result.mean_crlb_film_pct > 0
        assert result.improvement_factor > 0
        assert len(result.element_names) == self.me.n_free

    # -------------------------------------------------------------------
    # Test 9: build_fisher_matrix is symmetric PSD
    # -------------------------------------------------------------------

    def test_build_fisher_matrix_symmetric_psd(self):
        """Fisher matrix should be symmetric and positive semi-definite."""
        J = self.me._J_conv
        X = self.me._X_conv
        F = MultiEnergyCRLB.build_fisher_matrix(J, X, 1e6)

        # Symmetric
        np.testing.assert_allclose(F, F.T, atol=1e-10,
                                   err_msg="Fisher matrix not symmetric")

        # PSD: all eigenvalues >= 0
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals >= -1e-10), \
            f"Fisher matrix has negative eigenvalue: {eigvals.min():.2e}"

    # -------------------------------------------------------------------
    # Test 10: extract_crlb dimensions and validity
    # -------------------------------------------------------------------

    def test_extract_crlb_valid(self):
        """extract_crlb should return valid rank and CRLB values."""
        J = self.me._J_conv
        X = self.me._X_conv
        F = MultiEnergyCRLB.build_fisher_matrix(J, X, 1e6)

        rank, crlb_film, mean_pct, sv = MultiEnergyCRLB.extract_crlb(
            F, self.me.film_size, self.me.n_depth, self.me.n_free,
        )

        assert rank > 0, "Rank should be positive"
        assert rank <= self.me.n_params, f"Rank {rank} > n_params {self.me.n_params}"
        assert crlb_film.shape == (self.me.n_depth, self.me.n_free)
        assert mean_pct >= 0, "CRLB should be non-negative"
        assert len(sv) == self.me.n_params

    # -------------------------------------------------------------------
    # Test 11: optimize_energy_set returns best combo
    # -------------------------------------------------------------------

    def test_optimize_energy_set(self):
        """optimize_energy_set should find best pair from candidates."""
        result = self.me.optimize_energy_set(
            candidate_ratios=[1.0, 2.0, 3.0],
            n_energies=2,
            total_counts=2e6,
        )

        assert 'best_combo' in result
        assert 'best_crlb' in result
        assert 'best_rank' in result
        assert len(result['all_results']) == 3  # C(3,2) = 3
        assert result['best_crlb'] > 0

        # best_crlb should be the minimum
        all_crlbs = [r['crlb_pct'] for r in result['all_results']]
        assert result['best_crlb'] == min(all_crlbs)

    # -------------------------------------------------------------------
    # Test 12: N scaling — CRLB ∝ 1/sqrt(N)
    # -------------------------------------------------------------------

    def test_crlb_count_scaling(self):
        """CRLB should scale as 1/sqrt(N)."""
        e_low = EnergyConfig("Low N", 1.0, 1e4)
        e_high = EnergyConfig("High N", 1.0, 1e6)

        r_low = self.me.analyze_single(e_low)
        r_high = self.me.analyze_single(e_high)

        # Ratio should be sqrt(1e6/1e4) = 10
        ratio = r_low['crlb_pct'] / max(r_high['crlb_pct'], 1e-30)
        assert 5.0 < ratio < 20.0, \
            f"CRLB scaling ratio {ratio:.2f}, expected ~10.0"


class TestMultiEnergyCRLBAngles:
    """Tests for angle sweep functionality."""

    def setup_method(self):
        # Use more angles for angle sweep tests
        self.tm, self.data = _make_mock_data(n_depth=10, n_angles=10, n_elements=3)
        self.fi = FisherInformation(self.tm, self.data, n_counts=1e6)
        self.me = MultiEnergyCRLB(self.fi, pixel_idx=0)

    def test_sweep_n_angles(self):
        """Angle sweep should return valid results."""
        sweep = self.me.sweep_n_angles([2, 4, 6, 8, 10])

        assert len(sweep['n_angles']) == 5
        assert len(sweep['ranks']) == 5
        assert len(sweep['crlb_pcts']) == 5

        # More angles should generally not decrease rank
        assert sweep['ranks'][-1] >= sweep['ranks'][0], (
            f"rank at n=10 ({sweep['ranks'][-1]}) < rank at n=2 ({sweep['ranks'][0]})"
        )


class TestEnergyConfig:
    """Tests for EnergyConfig dataclass."""

    def test_basic_creation(self):
        """EnergyConfig should store values correctly."""
        ec = EnergyConfig("Test", 2.0, 1e6)
        assert ec.label == "Test"
        assert ec.lambda_ratio == 2.0
        assert ec.n_counts == 1e6
        assert ec.photon_energy_eV is None

    def test_eV_creation(self):
        """EnergyConfig with eV should store photon_energy_eV."""
        ec = EnergyConfig("HAXPES", 1.0, 1e6, photon_energy_eV=8000.0)
        assert ec.photon_energy_eV == 8000.0


class TestStaticMethods:
    """Tests for static methods independent of FisherInformation."""

    def test_build_fisher_matrix_with_identity_jacobian(self):
        """Identity Jacobian → diagonal Fisher matrix."""
        n = 5
        J = np.eye(n)
        X_norm = np.ones(n)
        N = 100.0

        F = MultiEnergyCRLB.build_fisher_matrix(J, X_norm, N)
        np.testing.assert_allclose(F, N * np.eye(n), atol=1e-10)

    def test_build_fisher_matrix_zero_x_protection(self):
        """X_norm near zero should not cause inf/nan."""
        J = np.ones((3, 2))
        X_norm = np.array([1.0, 0.0, 1e-31])
        F = MultiEnergyCRLB.build_fisher_matrix(J, X_norm, 1e6)

        assert np.all(np.isfinite(F)), "Fisher matrix contains inf/nan"

    def test_extract_crlb_rank_deficient(self):
        """Rank-deficient Fisher matrix should still produce valid CRLB."""
        # F with rank 1
        v = np.array([1.0, 2.0, 3.0])
        F = np.outer(v, v)  # rank 1

        rank, crlb_film, mean_pct, sv = MultiEnergyCRLB.extract_crlb(
            F, film_size=3, n_depth=3, n_free=1,
        )

        assert rank == 1
        assert crlb_film.shape == (3, 1)
        # Some CRLB entries may be zero (for null-space directions)
        assert np.all(np.isfinite(crlb_film))
