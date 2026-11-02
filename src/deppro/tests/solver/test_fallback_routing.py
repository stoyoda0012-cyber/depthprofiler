"""
Test condition-based fallback routing for SemiFreeSolver.

Verifies that SLSQP → trust-constr fallback activates correctly on failure,
respects config flags, and produces better results than failed SLSQP alone.

Usage:
    pytest tests/solver/test_fallback_routing.py -v
    python -m deppro.tests.solver.test_fallback_routing
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

def _create_sio2_setup(n_depth=10, noise=False):
    """SiO2/Si system: Si4+ (free), O2- (linked), Si0 (substrate)."""
    depth = np.linspace(0, 50, n_depth)
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
    if noise:
        rng = np.random.default_rng(42)
        I_obs = I_obs * (1 + 0.02 * rng.standard_normal(I_obs.shape))
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    return I_obs, tm, elements, D


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFallbackConfig:
    """Verify that fallback config parameters are wired correctly."""

    def test_default_config_enables_fallback(self):
        """Default SolverConfig has fallback enabled."""
        cfg = SolverConfig()
        assert cfg.enable_fallback is True
        assert cfg.fallback_method == 'trust-constr'
        assert cfg.fallback_condition_threshold == 1e12

    def test_disable_fallback(self):
        """enable_fallback=False prevents fallback."""
        cfg = SolverConfig(enable_fallback=False)
        assert cfg.enable_fallback is False

    def test_result_has_fallback_used_field(self):
        """SemiFreeSolverResult contains fallback_used attribute."""
        I_obs, tm, elements, D = _create_sio2_setup()
        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0, abruptness_init=5.0,
            free_indices=[0], linked_indices=[1], substrate_indices=[2],
        )
        solver_cfg = SolverConfig(method='LLS', scipy_method='SLSQP', max_iter=50)
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)
        result = solver.solve(I_obs, verbose=False)
        assert hasattr(result, 'fallback_used')
        assert isinstance(result.fallback_used, bool)


class TestFallbackRouting:
    """Verify fallback logic triggers correctly."""

    def test_successful_slsqp_no_fallback(self):
        """When SLSQP succeeds, fallback is not triggered."""
        I_obs, tm, elements, D = _create_sio2_setup(n_depth=10)
        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0, abruptness_init=5.0,
            free_indices=[0], linked_indices=[1], substrate_indices=[2],
        )
        solver_cfg = SolverConfig(
            method='LLS', scipy_method='SLSQP',
            max_iter=100, enable_fallback=True,
        )
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)
        result = solver.solve(I_obs, verbose=False)

        # On clean data, SLSQP should succeed without fallback
        if result.success:
            assert result.fallback_used is False

    def test_fallback_disabled_does_not_trigger(self):
        """When enable_fallback=False, never falls back even on failure."""
        import unittest.mock as mock
        from scipy.optimize import OptimizeResult

        I_obs, tm, elements, D = _create_sio2_setup(n_depth=10)
        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0, abruptness_init=5.0,
            free_indices=[0], linked_indices=[1], substrate_indices=[2],
        )
        solver_cfg = SolverConfig(
            method='LLS', scipy_method='SLSQP',
            max_iter=100, enable_fallback=False,
        )
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        call_count = [0]
        original_minimize = None

        def counting_minimize(*args, **kwargs):
            call_count[0] += 1
            # First call: simulate SLSQP failure
            if call_count[0] == 1:
                x0 = args[1] if len(args) > 1 else kwargs.get('x0', np.zeros(1))
                return OptimizeResult(
                    x=x0, fun=1e10, jac=np.zeros_like(x0),
                    nit=1, nfev=1, success=False, status=8,
                    message='Positive directional derivative'
                )
            # Should never get here if fallback is disabled
            raise AssertionError("Fallback minimize should not be called")

        with mock.patch('deppro.inverse_solver.minimize', side_effect=counting_minimize):
            result = solver.solve(I_obs, verbose=False)

        assert call_count[0] == 1, "Only one minimize call should happen when fallback disabled"
        assert result.fallback_used is False

    def test_fallback_triggered_on_status8(self):
        """Fallback triggers when SLSQP returns status=8 (positive dir derivative)."""
        import unittest.mock as mock
        from scipy.optimize import OptimizeResult

        I_obs, tm, elements, D = _create_sio2_setup(n_depth=10)
        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0, abruptness_init=5.0,
            free_indices=[0], linked_indices=[1], substrate_indices=[2],
        )
        solver_cfg = SolverConfig(
            method='LLS', scipy_method='SLSQP',
            max_iter=100, enable_fallback=True,
        )
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        call_count = [0]

        def mock_minimize(fun, x0, method=None, jac=None, bounds=None,
                          constraints=(), options=None, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: simulate SLSQP failure with status=8
                return OptimizeResult(
                    x=x0.copy(), fun=1e10, jac=np.zeros_like(x0),
                    nit=1, nfev=1, success=False, status=8,
                    message='Positive directional derivative'
                )
            else:
                # Second call: trust-constr succeeds with lower objective
                assert method == 'trust-constr', f"Expected trust-constr, got {method}"
                return OptimizeResult(
                    x=x0.copy(), fun=0.5, jac=np.zeros_like(x0),
                    nit=10, nfev=50, success=True, status=1,
                    message='Optimization converged'
                )

        with mock.patch('deppro.inverse_solver.minimize', side_effect=mock_minimize):
            result = solver.solve(I_obs, verbose=False)

        assert call_count[0] == 2, "Two minimize calls expected (SLSQP + fallback)"
        assert result.fallback_used is True

    def test_fallback_not_accepted_if_worse(self):
        """Fallback result is rejected if worse than SLSQP result."""
        import unittest.mock as mock
        from scipy.optimize import OptimizeResult

        I_obs, tm, elements, D = _create_sio2_setup(n_depth=10)
        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0, abruptness_init=5.0,
            free_indices=[0], linked_indices=[1], substrate_indices=[2],
        )
        solver_cfg = SolverConfig(
            method='LLS', scipy_method='SLSQP',
            max_iter=100, enable_fallback=True,
        )
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        call_count = [0]

        def mock_minimize(fun, x0, method=None, jac=None, bounds=None,
                          constraints=(), options=None, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return OptimizeResult(
                    x=x0.copy(), fun=0.5, jac=np.zeros_like(x0),
                    nit=1, nfev=1, success=False, status=8,
                    message='Positive directional derivative'
                )
            else:
                # Fallback gives worse result
                return OptimizeResult(
                    x=x0.copy(), fun=1.0, jac=np.zeros_like(x0),
                    nit=10, nfev=50, success=False, status=2,
                    message='Did not converge'
                )

        with mock.patch('deppro.inverse_solver.minimize', side_effect=mock_minimize):
            result = solver.solve(I_obs, verbose=False)

        assert call_count[0] == 2, "Two calls expected"
        # Fallback worse → not accepted
        assert result.fallback_used is False

    def test_non_slsqp_method_no_fallback(self):
        """Fallback only triggers for SLSQP method."""
        I_obs, tm, elements, D = _create_sio2_setup(n_depth=10)
        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0, abruptness_init=5.0,
            free_indices=[0], linked_indices=[1], substrate_indices=[2],
        )
        solver_cfg = SolverConfig(
            method='LLS', scipy_method='trust-constr',
            max_iter=50, enable_fallback=True,
        )
        solver = SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

        call_count = [0]
        import unittest.mock as mock
        from scipy.optimize import OptimizeResult

        def mock_minimize(fun, x0, method=None, jac=None, bounds=None,
                          constraints=(), options=None, **kw):
            call_count[0] += 1
            return OptimizeResult(
                x=x0.copy(), fun=0.1, jac=np.zeros_like(x0),
                nit=10, nfev=50, success=True, status=1,
                message='OK'
            )

        with mock.patch('deppro.inverse_solver.minimize', side_effect=mock_minimize):
            result = solver.solve(I_obs, verbose=False)

        assert call_count[0] == 1, "Only one minimize call for non-SLSQP"
        assert result.fallback_used is False


class TestNoProgressRouting:
    """SLSQP "success" without any step is routed to the fallback.

    SciPy's SLSQP can exit with mode 0 at the unchanged x0 after several
    iterations: when its QP subproblem returns a non-descent step it resets
    the BFGS matrix five times and then passes its relaxed test trivially
    (no step was taken, so |f - f0| = 0). Observed with badly scaled
    gradients (see the comment in SemiFreeSolver.solve). The detector only
    looks at "success, more than one iteration, x unchanged"; a one-
    iteration success at x0 is outside its condition and is left alone.
    The reroute is kept only if it improves, so a misfire on a point that
    already is optimal changes nothing.
    """

    @staticmethod
    def _solver():
        I_obs, tm, elements, D = _create_sio2_setup(n_depth=10)
        semi_cfg = SemiFreeSolverConfig(
            thickness_init=20.0, abruptness_init=5.0,
            free_indices=[0], linked_indices=[1], substrate_indices=[2],
        )
        solver_cfg = SolverConfig(
            method='LLS', scipy_method='SLSQP',
            max_iter=100, enable_fallback=True, max_warm_restarts=0,
        )
        return I_obs, SemiFreeSolver(tm, solver_cfg, semi_cfg, elements=elements)

    @staticmethod
    def _stalled(x0, bounds, nit):
        """What SLSQP returns on this path: x0 clipped to the bounds (the
        SciPy driver clips before iterating), mode 0, one gradient."""
        from scipy.optimize import OptimizeResult
        return OptimizeResult(
            x=np.clip(x0, bounds.lb, bounds.ub), fun=1e5, jac=np.zeros_like(x0),
            nit=nit, nfev=1 + x0.size, njev=1, success=True, status=0,
            message='Optimization terminated successfully',
        )

    def test_success_without_step_triggers_fallback(self):
        import unittest.mock as mock
        from scipy.optimize import OptimizeResult

        I_obs, solver = self._solver()
        calls = []

        def mock_minimize(fun, x0, method=None, jac=None, bounds=None,
                          constraints=(), options=None, **kw):
            calls.append(method)
            if len(calls) == 1:
                return self._stalled(x0, bounds, nit=5)
            # trust-constr fallback, then the SLSQP polish from its point
            return OptimizeResult(
                x=x0.copy(), fun=1.0 / len(calls), jac=np.zeros_like(x0),
                nit=10, nfev=50, success=True, status=1,
                message='Optimization converged',
            )

        with mock.patch('deppro.inverse_solver.minimize', side_effect=mock_minimize):
            result = solver.solve(I_obs, verbose=False)

        assert calls == ['SLSQP', 'trust-constr', 'SLSQP'], calls
        assert result.fallback_used is True
        assert result.objective == 1.0 / 3  # the polished result was taken

    def test_one_iteration_success_at_x0_is_not_rerouted(self):
        """nit == 1 at x0 is outside the detector's condition."""
        import unittest.mock as mock

        I_obs, solver = self._solver()
        calls = []

        def mock_minimize(fun, x0, method=None, jac=None, bounds=None,
                          constraints=(), options=None, **kw):
            calls.append(method)
            return self._stalled(x0, bounds, nit=1)

        with mock.patch('deppro.inverse_solver.minimize', side_effect=mock_minimize):
            result = solver.solve(I_obs, verbose=False)

        assert calls == ['SLSQP'], calls
        assert result.fallback_used is False

    def test_success_after_moving_is_not_rerouted(self):
        """A success that did move (x != x0) is accepted as before."""
        import unittest.mock as mock

        I_obs, solver = self._solver()
        calls = []

        def mock_minimize(fun, x0, method=None, jac=None, bounds=None,
                          constraints=(), options=None, **kw):
            calls.append(method)
            r = self._stalled(x0, bounds, nit=5)
            moved = r.x.copy()
            moved[solver.n_depth * len(solver.free_idx)] += 1.0  # thickness
            r.x = moved
            return r

        with mock.patch('deppro.inverse_solver.minimize', side_effect=mock_minimize):
            result = solver.solve(I_obs, verbose=False)

        assert calls == ['SLSQP'], calls
        assert result.fallback_used is False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
