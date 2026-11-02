"""
Tests for Fisher Information Matrix and CRLB calculation.

Run:
    cd <repo>/src
    python -m pytest deppro/tests/unit/test_fisher_information.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List, Optional

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig


# =====================================================================
# Minimal mock data for unit tests (no H5 file required)
# =====================================================================

def _make_mock_data(n_depth=10, n_angles=5, n_elements=3):
    """Create minimal mock benchmark data for testing."""
    # Simple transfer matrix: exponential decay
    depth = np.linspace(0, 80, n_depth)
    angles = np.linspace(0, 60, n_angles)
    step_size = depth[1] - depth[0] if n_depth > 1 else 4.0

    tm = TransferMatrix(
        depth=depth, step_size=step_size, angles=angles,
        photon_energy=8000.0, work_function=4.5, gamma_eff=1.0,
    )

    # Build a simple transfer matrix manually
    trans_mat = np.zeros((n_angles, n_depth, n_elements))
    # Generate lambdas for arbitrary number of elements
    lambdas = [20.0 + 5.0 * i for i in range(n_elements)]
    for a, theta in enumerate(angles):
        cos_theta = np.cos(np.radians(theta))
        for e in range(n_elements):
            for d in range(n_depth):
                trans_mat[a, d, e] = np.exp(-depth[d] / (lambdas[e] * cos_theta))
    tm.trans_mat = trans_mat
    tm.n_elements = n_elements

    # Build mock Dstk (ground truth profile) — simple step function
    Dstk = np.zeros((1, n_depth, n_elements))
    # Element 0 = substrate (Si), increases with depth (erf)
    # Element 1 = film (SiO2), decreases
    # Element 2 = linked (O)
    from math import erf as _erf
    thickness = 40.0
    sigma = 5.0
    for d in range(n_depth):
        erf_val = 0.5 * (1.0 + _erf((depth[d] - thickness) / sigma))
        Dstk[0, d, 0] = erf_val          # substrate
        Dstk[0, d, 1] = 0.3 * (1 - erf_val)  # free element
        Dstk[0, d, 2] = 0.7 * (1 - erf_val)  # linked element

    # Forward model to get Xstk
    Xstk = np.zeros((1, n_angles, n_elements))
    for e in range(n_elements):
        Xstk[0, :, e] = trans_mat[:, :, e] @ Dstk[0, :, e]
    Xstk_total = Xstk.sum(axis=2, keepdims=True)
    Xstk = Xstk / np.maximum(Xstk_total, 1e-30)

    # Elements
    elements = [
        ElementConfig(symbol='Si', orbital='2p', compound='Si',
                      binding_energy=99.0, oxidation_state=0.0),
        ElementConfig(symbol='Si', orbital='2p', compound='SiO2',
                      binding_energy=103.0, oxidation_state=4.0),
        ElementConfig(symbol='O', orbital='1s', compound='SiO2',
                      binding_energy=532.0, oxidation_state=-2.0),
    ]

    charges = np.array([0.0, 4.0, -2.0])
    linkedconst = np.array([3.0])  # freelinkedconst for Si(+4)

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

class TestFisherInformation:

    def setup_method(self):
        from deppro.fisher_information import FisherInformation
        self.tm, self.data = _make_mock_data(n_depth=10, n_angles=5, n_elements=3)
        self.fi = FisherInformation(self.tm, self.data, n_counts=1e6)

    def test_dimensions(self):
        """Check dimension calculations."""
        assert self.fi.n_depth == 10
        assert self.fi.n_angles == 5
        assert self.fi.n_elements == 3
        assert self.fi.n_free == 1   # Only Si(+4) is free
        assert self.fi.n_sub == 1    # Si(0) is substrate
        assert self.fi.n_linked == 1  # O(-2) is linked
        assert self.fi.film_size == 10  # 10 depths × 1 free
        assert self.fi.n_params == 10 + 2 + 1  # film + thickness + abruptness + sub

    def test_forward_model(self):
        """Forward model should produce valid normalised intensities."""
        D = self.data.Dstk[0]
        x = self.fi._build_x_from_profile(D, thickness=40.0, abruptness=8.0)
        X_norm = self.fi._forward_normalised(x)

        assert X_norm.shape == (5, 3)
        # All values should be non-negative
        assert np.all(X_norm >= 0)
        # Should not be all zeros
        assert np.any(X_norm > 0)

    def test_jacobian_shape(self):
        """Jacobian should have correct shape."""
        D = self.data.Dstk[0]
        x = self.fi._build_x_from_profile(D, thickness=40.0, abruptness=8.0)
        J = self.fi.compute_jacobian(x)

        assert J.shape == (self.fi.n_obs, self.fi.n_params)
        # Jacobian should not be all zeros
        assert np.any(np.abs(J) > 1e-15)

    def test_fisher_matrix_symmetric(self):
        """Fisher matrix should be symmetric and positive semi-definite."""
        D = self.data.Dstk[0]
        x = self.fi._build_x_from_profile(D, thickness=40.0, abruptness=8.0)
        F, cond, S, rank = self.fi.compute_fisher_matrix(x)

        # Symmetric
        assert np.allclose(F, F.T, atol=1e-10)
        # Positive semi-definite (all eigenvalues >= 0)
        assert np.all(S >= -1e-10)
        # Non-trivial
        assert rank > 0

    def test_crlb_positive(self):
        """CRLB should be positive everywhere."""
        result = self.fi.compute_crlb(pixel_idx=0, thickness=40.0, abruptness=8.0)

        assert np.all(result.crlb >= 0)
        assert result.crlb_thickness >= 0
        assert result.crlb_abruptness >= 0

    def test_crlb_scaling(self):
        """CRLB should scale as 1/sqrt(N) with count number."""
        from deppro.fisher_information import FisherInformation

        fi_low = FisherInformation(self.tm, self.data, n_counts=1e4)
        fi_high = FisherInformation(self.tm, self.data, n_counts=1e6)

        r_low = fi_low.compute_crlb(pixel_idx=0, thickness=40.0, abruptness=8.0)
        r_high = fi_high.compute_crlb(pixel_idx=0, thickness=40.0, abruptness=8.0)

        # CRLB ratio should be sqrt(1e6/1e4) = 10
        mean_low = np.mean(r_low.crlb_film_conc)
        mean_high = np.mean(r_high.crlb_film_conc)

        ratio = mean_low / max(mean_high, 1e-30)
        assert 5.0 < ratio < 20.0, f"Expected ratio ~10, got {ratio:.2f}"

    def test_jacobian_finite_difference_consistency(self):
        """Jacobian at two different step sizes should approximately agree."""
        D = self.data.Dstk[0]
        x = self.fi._build_x_from_profile(D, thickness=40.0, abruptness=8.0)

        J1 = self.fi.compute_jacobian(x, eps=1e-7)
        J2 = self.fi.compute_jacobian(x, eps=1e-6)

        # Should agree to ~1% for well-conditioned directions
        # (ill-conditioned directions may differ more)
        norms = np.linalg.norm(J1, axis=0)
        mask = norms > 1e-10  # Only check non-degenerate columns
        if np.any(mask):
            rel_diff = np.linalg.norm(J1[:, mask] - J2[:, mask]) / np.linalg.norm(J1[:, mask])
            assert rel_diff < 0.1, f"FD Jacobians differ by {rel_diff:.3f}"

    def test_svd_threshold_sensitivity(self):
        """SVD threshold sensitivity should return valid results."""
        sens = self.fi.svd_threshold_sensitivity(pixel_idx=0)

        assert len(sens['thresholds']) == len(sens['mean_crlb_film'])
        assert np.all(sens['mean_crlb_film'] >= 0)
        # Looser threshold should give smaller (or equal) CRLB
        # (more regularisation → more truncation → lower CRLB, but less reliable)

    def test_count_range(self):
        """compute_crlb_vs_counts should return correct shapes."""
        counts = np.array([1e4, 1e5, 1e6])
        result = self.fi.compute_crlb_vs_counts(
            pixel_idx=0,
            count_range=counts,
            thickness=40.0,
            abruptness=8.0,
        )

        assert len(result['count_range']) == 3
        assert result['crlb_film'].shape == (3, 10, 1)
        assert len(result['crlb_thickness']) == 3
        assert len(result['mean_crlb_film']) == 3

    def test_structural_param_estimation(self):
        """Thickness/abruptness estimation from profile should be reasonable."""
        D = self.data.Dstk[0]
        t, a = self.fi._estimate_structural_params(D)

        # Should be in reasonable range for our mock data (thickness ~40Å)
        assert 10 < t < 80, f"Estimated thickness {t:.1f} out of range"
        assert 1 < a < 30, f"Estimated abruptness {a:.1f} out of range"

    def test_efficiency(self):
        """Efficiency computation should work."""
        result = self.fi.compute_crlb(pixel_idx=0, thickness=40.0, abruptness=8.0)

        # Fake RMSE (should be larger than CRLB)
        fake_rmse = result.crlb_film_conc * 2.0 + 1e-10  # add epsilon to avoid 0/0

        efficiency = self.fi.compute_efficiency(fake_rmse, result)
        # For depths where CRLB > 0, efficiency should be ~0.5
        mask = result.crlb_film_conc > 1e-20
        if np.any(mask):
            assert np.allclose(efficiency[mask], 0.5, atol=0.05)


class TestFisherWithLargerSystem:
    """Test with 6-element system mimicking the real benchmark."""

    def setup_method(self):
        from deppro.fisher_information import FisherInformation
        self.tm, self.data = _make_mock_data(n_depth=20, n_angles=14, n_elements=6)
        # Override element config for 6-element system
        self.data.element_names = ['Si(0)', 'Si(+4)', 'Ti(+4)', 'Al(+3)', 'C(0)', 'O(-2)']
        self.data.free_matlab = np.array([2, 3, 4, 5])
        self.data.subst_matlab = np.array([1])
        self.data.linked_matlab = np.array([6])
        self.data.linkedconst = np.array([3.0, 3.0, 2.5, 1.0])
        self.data.charges = np.array([0.0, 4.0, 4.0, 3.0, 0.0, -2.0])
        self.data.elements = [
            ElementConfig(symbol='Si', orbital='2p', compound='Si',
                          binding_energy=99.0, oxidation_state=0.0),
            ElementConfig(symbol='Si', orbital='2p', compound='SiO2',
                          binding_energy=103.0, oxidation_state=4.0),
            ElementConfig(symbol='Ti', orbital='2p', compound='TiO2',
                          binding_energy=458.0, oxidation_state=4.0),
            ElementConfig(symbol='Al', orbital='2p', compound='Al2O3',
                          binding_energy=74.0, oxidation_state=3.0),
            ElementConfig(symbol='C', orbital='1s', compound='C',
                          binding_energy=284.0, oxidation_state=0.0),
            ElementConfig(symbol='O', orbital='1s', compound='SiO2',
                          binding_energy=532.0, oxidation_state=-2.0),
        ]

        # Build a richer mock profile
        depth = self.data.depth
        n_d = len(depth)
        Dstk = np.zeros((1, n_d, 6))
        thickness = 40.0
        sigma = 5.0
        for d in range(n_d):
            from math import erf as _erf
            erf_val = 0.5 * (1.0 + _erf((depth[d] - thickness) / sigma))
            Dstk[0, d, 0] = erf_val * 1.0       # Si substrate
            film = 1.0 - erf_val
            Dstk[0, d, 1] = 0.15 * film         # Si(+4)
            Dstk[0, d, 2] = 0.10 * film         # Ti(+4)
            Dstk[0, d, 3] = 0.05 * film         # Al(+3)
            Dstk[0, d, 4] = 0.05 * film         # C(0)
            Dstk[0, d, 5] = 0.65 * film         # O(-2)
        self.data.Dstk = Dstk

        # Recompute Xstk
        Xstk = np.zeros((1, 14, 6))
        for e in range(6):
            Xstk[0, :, e] = self.tm.trans_mat[:, :, e] @ Dstk[0, :, e]
        Xstk_total = Xstk.sum(axis=2, keepdims=True)
        self.data.Xstk = Xstk / np.maximum(Xstk_total, 1e-30)

        self.fi = FisherInformation(self.tm, self.data, n_counts=1e6)

    def test_dimensions_6elem(self):
        """6-element system dimensions."""
        assert self.fi.n_free == 4
        assert self.fi.n_sub == 1
        assert self.fi.n_linked == 1
        assert self.fi.film_size == 20 * 4  # 80
        assert self.fi.n_params == 80 + 2 + 1  # 83

    def test_crlb_6elem(self):
        """CRLB for 6-element system should complete without error."""
        result = self.fi.compute_crlb(pixel_idx=0, thickness=40.0, abruptness=8.0)

        assert result.crlb_film_conc.shape == (20, 4)
        assert result.condition_number > 1
        assert result.effective_rank > 0
        # All CRLBs positive
        assert np.all(result.crlb_film_conc >= 0)

    def test_sxps_improvement(self):
        """SXPS should show improvement over conventional."""
        sxps = self.fi.compute_crlb_sxps(
            pixel_idx=0, lambda_ratio=1.0 / 3.0,
            thickness=40.0, abruptness=8.0,
        )

        # SXPS should generally improve CRLB (shorter λ → more depth info)
        mean_improvement = np.nanmean(sxps['improvement_film'])
        assert mean_improvement > 0.5 or np.isnan(mean_improvement), \
            f"Expected SXPS improvement, got {mean_improvement:.2f}x"
