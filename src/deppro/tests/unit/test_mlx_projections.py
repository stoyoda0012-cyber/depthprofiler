"""Unit tests for solvers_mlx.projections (Stage A)."""
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')

from deppro.solvers_mlx.projections import project_capped_simplex  # noqa: E402


def _scipy_reference(v, lo, hi, target, w, m):
    """Float64 QP reference via scipy (trust-constr)."""
    from scipy.optimize import minimize, LinearConstraint, Bounds
    n = len(v)
    res = minimize(
        lambda x: 0.5 * np.sum(m * (x - v) ** 2),
        np.clip(v, lo, hi),
        jac=lambda x: m * (x - v),
        method='trust-constr',
        bounds=Bounds(np.full(n, lo), np.full(n, hi)),
        constraints=[LinearConstraint(w[None, :], target, target)],
        options={'gtol': 1e-12, 'xtol': 1e-14},
    )
    return res.x


def test_simplex_no_active_bounds():
    """Interior case has the closed form x = v - (sum(v)-1)/n."""
    rng = np.random.default_rng(0)
    v = rng.uniform(0.3, 0.7, size=(5, 4))          # projection stays interior
    v = v / v.sum(axis=1, keepdims=True) + rng.normal(0, 0.01, (5, 4))
    expected = v - (v.sum(axis=1, keepdims=True) - 1.0) / 4
    out = np.array(project_capped_simplex(mx.array(v, dtype=mx.float32),
                                          lo=0.0, hi=1.0))
    np.testing.assert_allclose(out, expected, atol=5e-6)


def test_feasibility_exact():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 2.0, size=(200, 6))
    out = np.array(project_capped_simplex(mx.array(v, dtype=mx.float32),
                                          lo=1e-6, hi=1.0))
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=5e-6)
    assert out.min() >= 1e-6 - 1e-7 and out.max() <= 1.0 + 1e-7


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_matches_scipy_qp_with_active_bounds(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.2, 0.8, size=6)
    lo, hi = 0.01, 0.6                               # forces active bounds
    ref = _scipy_reference(v, lo, hi, 1.0, np.ones(6), np.ones(6))
    out = np.array(project_capped_simplex(
        mx.array(v[None], dtype=mx.float32), lo=lo, hi=hi))[0]
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_weighted_and_metric():
    """General form: sum(w*x)=target under diagonal metric m."""
    rng = np.random.default_rng(3)
    v = rng.normal(0.3, 0.5, size=6)
    w = rng.uniform(0.5, 2.0, size=6)
    m = rng.uniform(0.5, 4.0, size=6)
    ref = _scipy_reference(v, 0.0, 1.0, 1.0, w, m)
    out = np.array(project_capped_simplex(
        mx.array(v[None], dtype=mx.float32), lo=0.0, hi=1.0, target=1.0,
        weights=mx.array(w, dtype=mx.float32),
        metric=mx.array(m, dtype=mx.float32)))[0]
    np.testing.assert_allclose(out, ref, atol=2e-5)


def test_idempotent():
    rng = np.random.default_rng(4)
    v = mx.array(rng.normal(0, 1, size=(50, 6)), dtype=mx.float32)
    p1 = project_capped_simplex(v, lo=1e-6, hi=1.0)
    p2 = project_capped_simplex(p1, lo=1e-6, hi=1.0)
    np.testing.assert_allclose(np.array(p1), np.array(p2), atol=5e-6)
