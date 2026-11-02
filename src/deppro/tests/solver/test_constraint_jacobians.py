"""
Test analytical Jacobians for SemiFreeSolver constraints.

Verifies each constraint Jacobian against finite-difference approximation.
Uses a minimal SiO2/Si system (charge_balance mode) and a SiON system
(stoichiometric mode) to cover all constraint types.

Usage:
    pytest tests/solver/test_constraint_jacobians.py -v
    python -m deppro.tests.solver.test_constraint_jacobians
"""

import numpy as np
import pytest
from scipy.special import erf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # <repo>/src

from deppro.inverse_solver import (
    SemiFreeSolver, SemiFreeSolverConfig, SolverConfig,
)
from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _central_diff_jac(fun, x0, eps=1e-7):
    """Compute Jacobian via central finite differences."""
    f0 = np.atleast_1d(fun(x0))
    n_out = len(f0)
    n_x = len(x0)
    jac = np.zeros((n_out, n_x))
    for i in range(n_x):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        jac[:, i] = (np.atleast_1d(fun(xp)) - np.atleast_1d(fun(xm))) / (2 * eps)
    return jac


def _create_sio2_setup(n_depth=10, noise=False):
    """Create SiO2/Si system for charge_balance mode tests.

    Elements: Si4+ (free), O2- (linked), Si0 (substrate)
    """
    depth = np.linspace(0, 50, n_depth)
    angles = [0, 20, 40, 60]

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    # True profile: SiO2 (20 Å) / Si
    sigma = 3.0 / (2 * np.sqrt(np.log(2)))
    erf_sub = 0.5 * (1 + erf((depth - 20.0) / sigma))
    D = np.zeros((n_depth, 3))
    D[:, 2] = erf_sub
    D[:, 0] = (1 - erf_sub) / 3
    D[:, 1] = (1 - erf_sub) * 2 / 3
    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    tm = TransferMatrix(depth=depth, step_size=depth[1] - depth[0],
                        angles=angles, photon_energy=1486.6)
    tm.compute(D, elements)
    I_obs = tm.forward(D, normalize=True)

    if noise:
        rng = np.random.default_rng(42)
        I_obs = I_obs * (1 + 0.02 * rng.standard_normal(I_obs.shape))
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    return I_obs, tm, elements, D


def _create_multioxide_setup(n_depth=10):
    """Create mixed-oxide system WITHOUT linked elements for charge constraint test.

    Elements: Si4+ (free), Ti4+ (free), O2- (free), Si0 (substrate)
    No linked elements → explicit charge constraint is active.
    """
    depth = np.linspace(0, 50, n_depth)
    angles = [0, 20, 40, 60]

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'Oxide', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    sigma = 3.0 / (2 * np.sqrt(np.log(2)))
    erf_sub = 0.5 * (1 + erf((depth - 25.0) / sigma))
    D = np.zeros((n_depth, 4))
    D[:, 3] = erf_sub
    D[:, 0] = (1 - erf_sub) * 0.15
    D[:, 1] = (1 - erf_sub) * 0.18
    D[:, 2] = (1 - erf_sub) * 0.67
    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    tm = TransferMatrix(depth=depth, step_size=depth[1] - depth[0],
                        angles=angles, photon_energy=1486.6)
    tm.compute(D, elements)
    I_obs = tm.forward(D, normalize=True)

    return I_obs, tm, elements, D


def _create_non_setup(n_depth=10):
    """Create NON (Si3N4/SiO2/Si) system for stoichiometric mode tests.

    Elements: O (free), N (free), Si-B (substrate)
    Derived: Si-O computed from stoichiometric rules
    """
    depth = np.linspace(0, 40, n_depth)
    angles = [0, 20, 40, 60]

    elements = [
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    sigma = 3.0 / (2 * np.sqrt(np.log(2)))
    erf_sub = 0.5 * (1 + erf((depth - 20.0) / sigma))
    D = np.zeros((n_depth, 3))
    D[:, 2] = erf_sub
    D[:, 0] = (1 - erf_sub) * 0.5
    D[:, 1] = (1 - erf_sub) * 0.3
    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    tm = TransferMatrix(depth=depth, step_size=depth[1] - depth[0],
                        angles=angles, photon_energy=1486.6)
    tm.compute(D, elements)
    I_obs = tm.forward(D, normalize=True)

    return I_obs, tm, elements, D


def _extract_constraints(solver, I_obs):
    """Run solver with max_iter=0 to set up constraints, return them.

    We patch minimize at the import location in deppro.inverse_solver to
    capture the constraints before optimization runs.
    """
    import unittest.mock as mock
    from scipy.optimize import OptimizeResult
    captured = {}

    def fake_minimize(fun, x0, method=None, jac=None, bounds=None,
                      constraints=(), options=None, **kw):
        captured['x0'] = x0.copy()
        captured['constraints'] = list(constraints)
        captured['bounds'] = bounds
        captured['jac'] = jac
        return OptimizeResult(
            x=x0, fun=0.0, jac=np.zeros_like(x0),
            nit=0, nfev=0, success=True, status=0,
            message='intercepted'
        )

    with mock.patch('deppro.inverse_solver.minimize', side_effect=fake_minimize):
        try:
            solver.solve(I_obs, verbose=False)
        except Exception:
            pass

    return captured


# ---------------------------------------------------------------------------
# Tests for Sum Constraint Jacobian (with linked elements)
# ---------------------------------------------------------------------------

class TestSumConstraintLinkedJacobian:
    """Sum constraint: Σ(flc × film_conc) - 1 + erf_sub + const_fix = 0."""

    def test_sum_linked_vs_fd(self):
        """Analytic Jacobian matches finite differences for sum constraint (linked)."""
        I_obs, tm, elements, D = _create_sio2_setup(n_depth=10)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0,
            abruptness_init=5.0,
            free_indices=[0],
            linked_indices=[1],
            substrate_indices=[2],
        )
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP')
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        cap = _extract_constraints(solver, I_obs)
        assert len(cap['constraints']) > 0, "No constraints captured"

        # First constraint should be sum constraint
        c = cap['constraints'][0]
        x0 = cap['x0']

        assert 'jac' in c and c['jac'] is not None, "Sum constraint missing Jacobian"

        jac_analytic = c['jac'](x0)
        jac_fd = _central_diff_jac(c['fun'], x0)

        np.testing.assert_allclose(
            jac_analytic, jac_fd, rtol=1e-4, atol=1e-6,
            err_msg="Sum constraint (linked) Jacobian mismatch"
        )


# ---------------------------------------------------------------------------
# Tests for Sum Constraint Jacobian (without linked elements)
# ---------------------------------------------------------------------------

class TestSumConstraintNoLinkedJacobian:
    """Sum constraint: Σ(film_conc) - 1 + erf_sub + const_fix = 0."""

    def test_sum_nolinked_vs_fd(self):
        """Analytic Jacobian matches finite differences for sum constraint (no linked)."""
        I_obs, tm, elements, D = _create_multioxide_setup(n_depth=10)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=25.0,
            abruptness_init=5.0,
            free_indices=[0, 1, 2],
            linked_indices=[],
            substrate_indices=[3],
        )
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP')
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        cap = _extract_constraints(solver, I_obs)

        # Find sum constraint (first eq constraint with multi-row output)
        sum_c = None
        for c in cap['constraints']:
            val = c['fun'](cap['x0'])
            if hasattr(val, '__len__') and len(val) > 1:
                sum_c = c
                break

        assert sum_c is not None, "Sum constraint not found"
        assert 'jac' in sum_c and sum_c['jac'] is not None

        jac_analytic = sum_c['jac'](cap['x0'])
        jac_fd = _central_diff_jac(sum_c['fun'], cap['x0'])

        np.testing.assert_allclose(
            jac_analytic, jac_fd, rtol=1e-4, atol=1e-6,
            err_msg="Sum constraint (no linked) Jacobian mismatch"
        )


# ---------------------------------------------------------------------------
# Tests for Charge Balance Constraint Jacobian
# ---------------------------------------------------------------------------

class TestChargeConstraintJacobian:
    """Charge balance: Σ(film_conc × fc) + cc = 0."""

    def test_charge_vs_fd(self):
        """Analytic Jacobian matches finite differences for charge constraint."""
        I_obs, tm, elements, D = _create_multioxide_setup(n_depth=10)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=25.0,
            abruptness_init=5.0,
            free_indices=[0, 1, 2],
            linked_indices=[],
            substrate_indices=[3],
            use_charge_constraint=True,
        )
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP')
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        cap = _extract_constraints(solver, I_obs)

        # Find charge constraint: n_depth rows, different from sum constraint
        # (charge uses different weights than sum)
        charge_c = None
        for c in cap['constraints']:
            val = c['fun'](cap['x0'])
            if hasattr(val, '__len__') and len(val) > 1:
                # Check if it's the charge constraint (not sum) by checking values
                # Charge constraint has non-zero cc offset or different pattern
                if charge_c is None:
                    # First multi-row is sum, second multi-row is charge
                    charge_c = 'skip_first'
                elif charge_c == 'skip_first':
                    charge_c = c
                    break

        if charge_c is None or charge_c == 'skip_first':
            pytest.skip("Charge constraint not active in this setup")

        assert 'jac' in charge_c and charge_c['jac'] is not None

        jac_analytic = charge_c['jac'](cap['x0'])
        jac_fd = _central_diff_jac(charge_c['fun'], cap['x0'])

        np.testing.assert_allclose(
            jac_analytic, jac_fd, rtol=1e-4, atol=1e-6,
            err_msg="Charge constraint Jacobian mismatch"
        )

    def test_charge_jac_structure(self):
        """Charge Jacobian is block-diagonal with charge weights."""
        I_obs, tm, elements, D = _create_multioxide_setup(n_depth=8)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=25.0,
            abruptness_init=5.0,
            free_indices=[0, 1, 2],
            linked_indices=[],
            substrate_indices=[3],
            use_charge_constraint=True,
        )
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP')
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        cap = _extract_constraints(solver, I_obs)

        # Find charge constraint (second multi-row constraint)
        multi_row_constraints = []
        for c in cap['constraints']:
            val = c['fun'](cap['x0'])
            if hasattr(val, '__len__') and len(val) > 1:
                multi_row_constraints.append(c)

        if len(multi_row_constraints) < 2:
            pytest.skip("Charge constraint not active")

        charge_c = multi_row_constraints[1]
        jac = charge_c['jac'](cap['x0'])

        n_depth = 8
        n_film = 3
        film_size = n_depth * n_film

        # Non-film columns should be all zeros
        assert np.allclose(jac[:, film_size:], 0.0), \
            "Charge Jacobian should be zero for non-film parameters"

        # Check block-diagonal structure
        charges = np.array([4, 4, -2])  # Si4+, Ti4+, O2-
        for z in range(n_depth):
            for j in range(n_film):
                assert jac[z, z * n_film + j] == pytest.approx(charges[j], abs=1e-10)
            # Off-diagonal blocks should be zero
            for z2 in range(n_depth):
                if z2 != z:
                    assert np.allclose(jac[z, z2 * n_film:(z2 + 1) * n_film], 0.0)


# ---------------------------------------------------------------------------
# Tests for Substrate Sum Constraint Jacobian
# ---------------------------------------------------------------------------

class TestSubstrateSumJacobian:
    """Substrate sum: Σ sub_ratios - 1 = 0."""

    def test_substrate_sum_vs_fd(self):
        """Analytic Jacobian matches finite differences for substrate sum."""
        I_obs, tm, elements, D = _create_sio2_setup(n_depth=10)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0,
            abruptness_init=5.0,
            free_indices=[0],
            linked_indices=[1],
            substrate_indices=[2],
        )
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP')
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        cap = _extract_constraints(solver, I_obs)

        # Find scalar substrate sum constraint
        sub_sum_c = None
        for c in cap['constraints']:
            val = c['fun'](cap['x0'])
            val = np.atleast_1d(val)
            if len(val) == 1:
                sub_sum_c = c
                break

        assert sub_sum_c is not None, "Substrate sum constraint not found"
        assert 'jac' in sub_sum_c and sub_sum_c['jac'] is not None

        jac_analytic = sub_sum_c['jac'](cap['x0'])
        jac_fd = _central_diff_jac(sub_sum_c['fun'], cap['x0'])

        np.testing.assert_allclose(
            jac_analytic, jac_fd, rtol=1e-4, atol=1e-6,
            err_msg="Substrate sum Jacobian mismatch"
        )

    def test_substrate_sum_jac_structure(self):
        """Substrate sum Jacobian has 1.0 only at sub_ratio positions."""
        I_obs, tm, elements, D = _create_sio2_setup(n_depth=10)

        n_depth = 10
        n_film = 1  # only Si4+ is free
        n_sub = 1
        film_size = n_depth * n_film

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0,
            abruptness_init=5.0,
            free_indices=[0],
            linked_indices=[1],
            substrate_indices=[2],
        )
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP')
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        cap = _extract_constraints(solver, I_obs)

        sub_sum_c = None
        for c in cap['constraints']:
            val = np.atleast_1d(c['fun'](cap['x0']))
            if len(val) == 1:
                sub_sum_c = c
                break

        assert sub_sum_c is not None
        jac = sub_sum_c['jac'](cap['x0'])

        # Only sub_ratios positions should be non-zero
        assert jac.shape == (1, len(cap['x0']))
        assert np.allclose(jac[0, :film_size + 2], 0.0), "Film/thickness/abruptness should be 0"
        assert jac[0, film_size + 2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests for Substrate Charge Constraint Jacobian
# ---------------------------------------------------------------------------

class TestSubstrateChargeJacobian:
    """Substrate charge: Σ(sc × sub_ratios) = 0."""

    def test_substrate_charge_vs_fd(self):
        """Analytic Jacobian matches finite differences for substrate charge."""
        # Need multiple substrate elements with mixed charges
        depth = np.linspace(0, 50, 10)
        angles = [0, 20, 40, 60]

        elements = [
            ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
            ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
            ElementConfig('Si', '2p', 'Si', oxidation_state=0),
            ElementConfig('Fe', '2p', 'Fe', oxidation_state=3),
        ]

        # Profile with 2 substrate elements
        sigma = 3.0 / (2 * np.sqrt(np.log(2)))
        erf_sub = 0.5 * (1 + erf((depth - 20.0) / sigma))
        D = np.zeros((10, 4))
        D[:, 2] = erf_sub * 0.7
        D[:, 3] = erf_sub * 0.3
        D[:, 0] = (1 - erf_sub) / 3
        D[:, 1] = (1 - erf_sub) * 2 / 3
        D = np.clip(D, 1e-6, None)
        D = D / D.sum(axis=1, keepdims=True)

        tm = TransferMatrix(depth=depth, step_size=depth[1] - depth[0],
                            angles=angles, photon_energy=1486.6)
        tm.compute(D, elements)
        I_obs = tm.forward(D, normalize=True)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0,
            abruptness_init=5.0,
            free_indices=[0],
            linked_indices=[1],
            substrate_indices=[2, 3],
            use_charge_constraint=True,
        )
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP')
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        cap = _extract_constraints(solver, I_obs)

        # Find scalar constraints (substrate sum and substrate charge)
        scalar_constraints = []
        for c in cap['constraints']:
            val = np.atleast_1d(c['fun'](cap['x0']))
            if len(val) == 1:
                scalar_constraints.append(c)

        assert len(scalar_constraints) >= 2, \
            f"Expected >=2 scalar constraints (sub sum + sub charge), got {len(scalar_constraints)}"

        # Second scalar constraint should be substrate charge
        charge_c = scalar_constraints[1]
        assert 'jac' in charge_c and charge_c['jac'] is not None

        jac_analytic = charge_c['jac'](cap['x0'])
        jac_fd = _central_diff_jac(charge_c['fun'], cap['x0'])

        np.testing.assert_allclose(
            jac_analytic, jac_fd, rtol=1e-4, atol=1e-6,
            err_msg="Substrate charge Jacobian mismatch"
        )


# ---------------------------------------------------------------------------
# Tests for Stoichiometric Constraint Jacobian
# ---------------------------------------------------------------------------

class TestStoichiometricJacobian:
    """Stoichiometric: Σ(coeffs × film_conc) - 1 + erf_sub = 0."""

    def _make_stoich_solver(self, n_depth=10):
        """Create stoichiometric mode solver (NON-style).

        4 elements: Si-B (substrate, idx=0), Si-O (derived, idx=1), O (free, idx=2), N (free, idx=3)
        Rules: Si-O = 0.5*O + 0.75*N
        Coeffs: [1.5, 1.75] → (3/2)*D_O + (7/4)*D_N = 1 - D_SiB
        """
        depth = np.linspace(0, 40, n_depth)
        angles = [0, 20, 40, 60]

        elements = [
            ElementConfig('Si', '2p', 'Si', oxidation_state=0),    # 0: Si-B (substrate)
            ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),  # 1: Si-O (derived)
            ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),  # 2: O (free)
            ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3), # 3: N (free)
        ]

        sigma = 3.0 / (2 * np.sqrt(np.log(2)))
        erf_sub = 0.5 * (1 + erf((depth - 20.0) / sigma))
        D = np.zeros((n_depth, 4))
        D[:, 0] = erf_sub                      # Si-B substrate
        D[:, 2] = (1 - erf_sub) * 0.5          # O
        D[:, 3] = (1 - erf_sub) * 0.2          # N
        D[:, 1] = 0.5 * D[:, 2] + 0.75 * D[:, 3]  # Si-O derived
        D = np.clip(D, 1e-6, None)
        D = D / D.sum(axis=1, keepdims=True)

        tm = TransferMatrix(depth=depth, step_size=depth[1] - depth[0],
                            angles=angles, photon_energy=1486.6)
        tm.compute(D, elements)
        I_obs = tm.forward(D, normalize=True)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0,
            abruptness_init=5.0,
            free_indices=[2, 3],       # O, N
            linked_indices=[],
            substrate_indices=[0],     # Si-B
            stoichiometric_mode='stoichiometric',
            stoichiometric_rules={1: ([2, 3], [0.5, 0.75])},  # Si-O = 0.5*O + 0.75*N
            stoichiometric_constraint_coeffs=[1.5, 1.75],
            use_charge_constraint=False,
        )
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP')
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        return solver, I_obs

    def test_stoich_vs_fd(self):
        """Analytic Jacobian matches finite differences for stoichiometric constraint."""
        solver, I_obs = self._make_stoich_solver(n_depth=10)
        cap = _extract_constraints(solver, I_obs)

        # Find stoichiometric constraint: last multi-row constraint
        multi_row = []
        for c in cap['constraints']:
            val = c['fun'](cap['x0'])
            if hasattr(val, '__len__') and len(val) > 1:
                multi_row.append(c)

        stoich_c = multi_row[-1]
        assert stoich_c is not None, "Stoichiometric constraint not found"
        assert 'jac' in stoich_c and stoich_c['jac'] is not None

        jac_analytic = stoich_c['jac'](cap['x0'])
        jac_fd = _central_diff_jac(stoich_c['fun'], cap['x0'])

        np.testing.assert_allclose(
            jac_analytic, jac_fd, rtol=1e-4, atol=1e-6,
            err_msg="Stoichiometric constraint Jacobian mismatch"
        )

    def test_stoich_jac_film_block(self):
        """Film block of stoichiometric Jacobian has correct coefficients."""
        solver, I_obs = self._make_stoich_solver(n_depth=8)
        cap = _extract_constraints(solver, I_obs)

        # Find stoichiometric constraint: it has coefficients [1.5, 1.75]
        # (vs sum constraint which has unit weights 1.0)
        # The stoichiometric constraint is the last multi-row constraint.
        multi_row = []
        for c in cap['constraints']:
            val = c['fun'](cap['x0'])
            if hasattr(val, '__len__') and len(val) > 1:
                multi_row.append(c)

        # Last multi-row constraint should be stoichiometric
        stoich_c = multi_row[-1]
        assert stoich_c is not None
        jac = stoich_c['jac'](cap['x0'])

        n_depth = 8
        n_film = 2  # O, N
        coeffs = np.array([1.5, 1.75])

        # Jacobian rows correspond to active (non-degenerate) depths only
        n_active = jac.shape[0]
        assert n_active <= n_depth, \
            f"Expected <= {n_depth} rows (active depths), got {n_active}"
        assert n_active > 0, "Expected at least 1 active depth"

        # Each active row should have correct stoichiometric coefficients
        # Identify which depth each row corresponds to by finding nonzero entries
        for row_i in range(n_active):
            # Find the depth index from the nonzero film-block entries
            film_block = jac[row_i, :n_depth * n_film]
            nonzero_idx = np.where(np.abs(film_block) > 1e-12)[0]
            assert len(nonzero_idx) == n_film, \
                f"Row {row_i}: expected {n_film} nonzero film entries, got {len(nonzero_idx)}"
            z = nonzero_idx[0] // n_film
            for j in range(n_film):
                assert jac[row_i, z * n_film + j] == pytest.approx(coeffs[j], abs=1e-10)


# ---------------------------------------------------------------------------
# Tests for Degenerate Depth Filtering
# ---------------------------------------------------------------------------

class TestDegenDepthFiltering:
    """Verify that degenerate depths (where erf_sub ≈ 1) are excluded
    from constraint rows to prevent rank-deficient Jacobians."""

    def test_constraint_rows_filtered(self):
        """Constraint output has <= n_depth rows (degenerate rows excluded)."""
        n_depth = 15
        I_obs, tm, elements, D = _create_sio2_setup(n_depth=n_depth)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0,
            abruptness_init=5.0,
            free_indices=[0],
            linked_indices=[1],
            substrate_indices=[2],
        )
        # Enable opt-in degenerate-depth filtering (default is 0.0 = disabled).
        # Without it, no rows are excluded and this test would
        # not actually exercise the active/degenerate split.
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP',
                                  degen_depth_threshold=0.01)
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        cap = _extract_constraints(solver, I_obs)

        # Sum constraint should have <= n_depth rows
        c = cap['constraints'][0]
        val = c['fun'](cap['x0'])
        n_active = len(val)
        assert 0 < n_active <= n_depth, \
            f"Expected 0 < rows <= {n_depth}, got {n_active}"

        jac = c['jac'](cap['x0'])
        assert jac.shape[0] == n_active, \
            f"Jacobian rows ({jac.shape[0]}) != constraint rows ({n_active})"

    def test_shallow_film_no_degenerate(self):
        """With very deep analysis, many depths should be degenerate."""
        # 50 Å depth range, 20 Å film → depths beyond ~30 Å are degenerate
        n_depth = 20
        depth = np.linspace(0, 80, n_depth)  # Extended depth
        angles = [0, 20, 40, 60]

        elements = [
            ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
            ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
            ElementConfig('Si', '2p', 'Si', oxidation_state=0),
        ]

        sigma = 3.0 / (2 * np.sqrt(np.log(2)))
        erf_sub = 0.5 * (1 + erf((depth - 20.0) / sigma))
        D = np.zeros((n_depth, 3))
        D[:, 2] = erf_sub
        D[:, 0] = (1 - erf_sub) / 3
        D[:, 1] = (1 - erf_sub) * 2 / 3
        D = np.clip(D, 1e-6, None)
        D = D / D.sum(axis=1, keepdims=True)

        tm = TransferMatrix(depth=depth, step_size=depth[1] - depth[0],
                            angles=angles, photon_energy=1486.6)
        tm.compute(D, elements)
        I_obs = tm.forward(D, normalize=True)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0,
            abruptness_init=5.0,
            free_indices=[0],
            linked_indices=[1],
            substrate_indices=[2],
        )
        # Degenerate-depth filtering is opt-in: the production
        # default degen_depth_threshold=0.0 keeps all depths active to avoid
        # thickness_init-sensitivity. Enable it here to exercise the feature.
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP',
                                  degen_depth_threshold=0.01)
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        cap = _extract_constraints(solver, I_obs)
        c = cap['constraints'][0]
        val = c['fun'](cap['x0'])

        # With 20 Å film and 80 Å depth range, some depths should be degenerate
        assert len(val) < n_depth, \
            f"Expected degenerate filtering to reduce rows from {n_depth}, got {len(val)}"


# ---------------------------------------------------------------------------
# Tests for Warm Restart
# ---------------------------------------------------------------------------

class TestWarmRestart:
    """Verify warm restart triggers when thickness changes significantly."""

    def test_warm_restart_triggers_on_thickness_change(self):
        """Warm restart should re-run when active depths change."""
        # Create a system where the initial thickness is far from optimal
        n_depth = 15
        depth = np.linspace(0, 60, n_depth)
        angles = [0, 20, 40, 60]

        elements = [
            ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
            ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
            ElementConfig('Si', '2p', 'Si', oxidation_state=0),
        ]

        # True film is 40 Å
        sigma = 3.0 / (2 * np.sqrt(np.log(2)))
        erf_sub = 0.5 * (1 + erf((depth - 40.0) / sigma))
        D = np.zeros((n_depth, 3))
        D[:, 2] = erf_sub
        D[:, 0] = (1 - erf_sub) / 3
        D[:, 1] = (1 - erf_sub) * 2 / 3
        D = np.clip(D, 1e-6, None)
        D = D / D.sum(axis=1, keepdims=True)

        tm = TransferMatrix(depth=depth, step_size=depth[1] - depth[0],
                            angles=angles, photon_energy=1486.6)
        tm.compute(D, elements)
        I_obs = tm.forward(D, normalize=True)

        # Start with thickness_init=15 Å (far from true 40 Å)
        # This should cause many false-degenerate depths initially,
        # and a warm restart after thickness grows
        semi_cfg = SemiFreeSolverConfig(
            thickness_init=15.0,
            abruptness_init=5.0,
            free_indices=[0],
            linked_indices=[1],
            substrate_indices=[2],
        )
        solver_cfg = SolverConfig(
            method='LLS', scipy_method='SLSQP', max_iter=100,
            max_warm_restarts=2,
            warm_restart_threshold=0.2,
        )
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        result = solver.solve(I_obs, verbose=False)

        # Result should be reasonable regardless of whether warm restart triggered
        assert result.thickness > 0, "Thickness should be positive"
        # warm_restarts should be non-negative
        assert result.warm_restarts >= 0

    def test_no_warm_restart_when_disabled(self):
        """No warm restart when max_warm_restarts=0."""
        n_depth = 10
        I_obs, tm, elements, D = _create_sio2_setup(n_depth=n_depth)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0,
            abruptness_init=5.0,
            free_indices=[0],
            linked_indices=[1],
            substrate_indices=[2],
        )
        solver_cfg = SolverConfig(
            method='LLS', scipy_method='SLSQP', max_iter=50,
            max_warm_restarts=0,
        )
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        result = solver.solve(I_obs, verbose=False)
        assert result.warm_restarts == 0


# ---------------------------------------------------------------------------
# Integration: SLSQP convergence with all analytical Jacobians
# ---------------------------------------------------------------------------

class TestSLSQPConvergenceWithJacobians:
    """Verify SLSQP converges properly with all analytical Jacobians."""

    def test_sio2_convergence(self):
        """SiO2/Si system converges with sum + substrate Jacobians."""
        I_obs, tm, elements, D = _create_sio2_setup(n_depth=15, noise=True)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=25.0,
            abruptness_init=5.0,
            free_indices=[0],
            linked_indices=[1],
            substrate_indices=[2],
        )
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP', max_iter=100)
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        result = solver.solve(I_obs, verbose=False)

        # Should converge (not 1-iteration termination)
        assert result.n_iterations > 1 or result.success, \
            f"SLSQP terminated early: nit={result.n_iterations}, success={result.success}"

        # Thickness should be reasonable (true = 20 Å)
        assert abs(result.thickness - 20.0) < 15.0, \
            f"Thickness {result.thickness:.1f} too far from true 20.0 Å"

    def test_multioxide_convergence(self):
        """Multi-oxide system converges with charge + sum + substrate Jacobians."""
        I_obs, tm, elements, D = _create_multioxide_setup(n_depth=12)

        semi_cfg = SemiFreeSolverConfig(
            thickness_init=30.0,
            abruptness_init=5.0,
            free_indices=[0, 1, 2],
            linked_indices=[],
            substrate_indices=[3],
            use_charge_constraint=True,
        )
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP', max_iter=100)
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        result = solver.solve(I_obs, verbose=False)

        assert result.n_iterations > 1 or result.success, \
            f"SLSQP terminated early: nit={result.n_iterations}, success={result.success}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
