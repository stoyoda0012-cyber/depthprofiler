"""SemiFreeSolver keeps its equality constraints whatever ``scipy_method`` says.

Until 2026-10-20 ``SolverConfig.scipy_method`` defaulted to 'L-BFGS-B'.
``scipy.optimize.minimize`` ignores ``constraints`` for that method (it only
warns), so a SemiFreeSolver built with the default dropped its stoichiometric
equality constraints without notice. The default is now 'SLSQP', and a method
that cannot handle constraints is replaced by SLSQP with a warning.

The tests wrap ``minimize`` inside ``deppro.inverse_solver`` to see which
method the first call actually uses, and evaluate the equality constraints it
was given at the point it returned.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import deppro.inverse_solver as inv
from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig
from deppro.tests.solver.test_semifree_solver import create_test_data

SWITCH_MESSAGE = "cannot handle the equality constraints"


@pytest.fixture
def first_minimize_call(monkeypatch):
    """Record method, constraints and result of the first ``minimize`` call."""
    calls = []
    real = inv.minimize

    def spy(fun, x0, *args, **kwargs):
        res = real(fun, x0, *args, **kwargs)
        calls.append({"method": kwargs.get("method"),
                      "constraints": kwargs.get("constraints") or [],
                      "x": np.asarray(res.x, dtype=float)})
        return res

    monkeypatch.setattr(inv, "minimize", spy)
    return calls


def _solve(solver_config):
    I_obs, tm, elements, _, _ = create_test_data(noise_level=0.0)
    semi = SemiFreeSolverConfig(
        thickness_init=70.0, abruptness_init=5.0,
        free_indices=[0, 1, 2, 3], linked_indices=[4], substrate_indices=[5],
    )
    return SemiFreeSolver(tm, solver_config, semi, elements=elements).solve(I_obs, verbose=False)


def _max_eq_residual(call):
    eqs = [c for c in call["constraints"] if c.get("type") == "eq"]
    assert eqs, "the problem has no equality constraints; the test would be empty"
    return max(float(np.max(np.abs(np.atleast_1d(c["fun"](call["x"]))))) for c in eqs)


def _switch_warnings(record):
    return [w for w in record if SWITCH_MESSAGE in str(w.message)]


def test_default_scipy_method_is_slsqp():
    assert SolverConfig().scipy_method == "SLSQP"


def test_default_config_solves_with_slsqp_and_keeps_the_constraints(first_minimize_call):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _solve(SolverConfig(method="LLS", alpha=0.0))
    first = first_minimize_call[0]
    assert first["method"] == "SLSQP"
    assert _max_eq_residual(first) < 1e-6
    assert not _switch_warnings(record)


def test_explicit_lbfgsb_is_replaced_by_slsqp_with_a_warning(first_minimize_call):
    with pytest.warns(UserWarning, match=SWITCH_MESSAGE) as record:
        _solve(SolverConfig(method="LLS", alpha=0.0, scipy_method="L-BFGS-B"))
    assert any("'L-BFGS-B'" in str(w.message) for w in _switch_warnings(record))
    first = first_minimize_call[0]
    assert first["method"] == "SLSQP"
    assert _max_eq_residual(first) < 1e-6


def test_explicit_trust_constr_is_kept(first_minimize_call):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _solve(SolverConfig(method="LLS", alpha=0.0, scipy_method="trust-constr",
                            max_iter=30, enable_fallback=False))
    assert first_minimize_call[0]["method"] == "trust-constr"
    assert not _switch_warnings(record)


@pytest.mark.parametrize("given, used", [("slsqp", "SLSQP"), ("TRUST-CONSTR", "trust-constr")])
def test_method_names_are_normalised(first_minimize_call, given, used):
    """Other spellings reach minimize in the spelling the solver compares
    against: 'slsqp' keeps the SLSQP fallback routing, 'TRUST-CONSTR' gets the
    trust-constr options (it used to receive ftol and fail)."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _solve(SolverConfig(method="LLS", alpha=0.0, scipy_method=given,
                            max_iter=30, enable_fallback=False))
    assert first_minimize_call[0]["method"] == used
    assert not _switch_warnings(record)


def test_a_fallback_that_raises_keeps_the_slsqp_result(monkeypatch):
    """If the fallback optimizer raises inside scipy, solve() returns the SLSQP
    result with a RuntimeWarning instead of propagating the error.

    trust-constr raises a dimension mismatch for SemiFreeSolver's gradient_mode
    (scipy 1.16 and 1.18). With SLSQP as the default, a failed SLSQP run routes
    there; on Linux CI this aborted test_gradient_recovery."""
    real = inv.minimize

    def fake(fun, x0, *args, **kwargs):
        if kwargs.get("method") == "trust-constr":
            raise ValueError("Dimension mismatch (simulated)")
        res = real(fun, x0, *args, **kwargs)
        res.success, res.status = False, 8   # force the fallback route
        return res

    monkeypatch.setattr(inv, "minimize", fake)
    with pytest.warns(RuntimeWarning, match="keeping the SLSQP result"):
        result = _solve(SolverConfig(method="LLS", alpha=0.0, max_iter=30))
    assert result is not None and np.isfinite(result.thickness)


def test_an_error_in_the_polish_step_is_not_swallowed(monkeypatch):
    """Only the fallback call is guarded: on the no-progress route the SLSQP
    polish after the fallback must still raise if it fails."""
    real = inv.minimize
    calls = {"slsqp": 0}

    def fake(fun, x0, *args, **kwargs):
        method = kwargs.get("method")
        if method == "SLSQP":
            calls["slsqp"] += 1
            if calls["slsqp"] == 1:
                # first SLSQP run: report "success" without having moved (no-progress route)
                res = real(fun, x0, *args, **kwargs)
                b = kwargs["bounds"]
                res.x = np.clip(np.asarray(x0, dtype=float), b.lb, b.ub)
                res.success, res.status, res.nit = True, 0, 5
                return res
            raise ValueError("polish failed (simulated)")
        return real(fun, x0, *args, **kwargs)

    monkeypatch.setattr(inv, "minimize", fake)
    with pytest.raises(ValueError, match="polish failed"):
        _solve(SolverConfig(method="LLS", alpha=0.0, max_iter=30))
