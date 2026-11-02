"""
FreeSolver analytical-gradient consistency tests
================================================

Regression tests for the normalization-derivative bug found during the
MLX solver migration (Stage A):

``FreeSolver._objective`` normalizes I_pred per angle (I/ΣI) before the
Chi² term, but ``FreeSolver._gradient`` used to ignore the derivative of
that normalization ("simplified"). The inconsistent gradient/merit pair
made SLSQP stall 2-9% above the true objective minimum even with
maxiter=3000 / tol=1e-14.

These tests pin the fix:
  1. The analytical gradient matches central finite differences of
     ``_objective`` at random feasible points (L2 / LLS / MEM).
  2. SLSQP driven by the analytical gradient reaches the same final f as
     SLSQP with jac=None (scipy forward-difference), rel diff <= 1e-6.

Usage:
    pytest tests/solver/test_freesolver_gradient_consistency.py -v
"""

import numpy as np
import pytest


def _build_problem(seed=0):
    """Small five-layer-profile problem shared by all tests."""
    from deppro.depth_profile import create_five_layer_profile
    from deppro.transfer_matrix import TransferMatrix

    profile = create_five_layer_profile(depth_end=100, step_size=4.0)
    angles = np.array([0, 20, 40, 60])
    step_size = profile.depth[1] - profile.depth[0]

    tm = TransferMatrix(
        depth=profile.depth,
        step_size=step_size,
        angles=angles,
        photon_energy=1486.6,
    )
    tm.compute(profile.concentrations, profile.elements)

    D_true = profile.concentrations
    I_true = tm.forward(D_true, normalize=True)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.01, I_true.shape) * I_true
    I_obs = np.maximum(I_true + noise, 0)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    return profile, tm, I_obs


def _objective_args(solver, I_obs, degree):
    """Replicate the (args) tuple FreeSolver.solve passes to minimize."""
    n_depth, n_elements = solver.D_prior.shape
    I_normalized = I_obs / np.maximum(I_obs.sum(axis=1, keepdims=True), 1e-10)
    T_matrices = [solver.tm.get_matrix(e) for e in range(n_elements)]
    return I_normalized, T_matrices, n_depth, n_elements, degree


def _central_diff_gradient(f, x, args, eps=1e-7):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (f(xp, *args) - f(xm, *args)) / (2 * eps)
    return grad


@pytest.mark.parametrize('method', ['L2', 'LLS', 'MEM'])
def test_gradient_matches_finite_differences(method):
    """Analytical gradient == central FD of the (normalized) objective."""
    from deppro.inverse_solver import FreeSolver, SolverConfig

    profile, tm, I_obs = _build_problem()
    n_depth, n_elements = profile.concentrations.shape

    config = SolverConfig(method=method, alpha=0.1, degree=1.0)
    D_prior = np.ones((n_depth, n_elements)) / n_elements
    solver = FreeSolver(tm, config, D_prior)

    args = _objective_args(solver, I_obs, degree=config.degree)

    rng = np.random.default_rng(123)
    for _ in range(3):
        # Random feasible point away from bounds (keeps FD well-defined
        # and MEM's log term smooth)
        D = rng.uniform(0.05, 0.95, (n_depth, n_elements))
        D = D / D.sum(axis=1, keepdims=True)
        x = D.flatten()

        g_ana = solver._gradient(x, *args)
        g_fd = _central_diff_gradient(solver._objective, x, args)

        scale = max(np.linalg.norm(g_fd), 1e-12)
        rel_err = np.linalg.norm(g_ana - g_fd) / scale
        assert rel_err < 1e-6, (
            f"{method}: analytical vs central-FD gradient rel error "
            f"{rel_err:.3e} (normalization chain rule broken?)"
        )


@pytest.mark.parametrize('method', ['L2', 'LLS', 'MEM'])
def test_slsqp_analytic_jac_matches_numerical_jac(method):
    """SLSQP(jac=analytic) agrees with SLSQP(jac=None) at rel 1e-6.

    Two checks (before the fix the simplified gradient stalled 2-9% above
    the FD result, failing both):

    1. Direct: both runs from D_prior reach the same f. **L2 and MEM
       only.** For LLS the objective is chi2 alone (alpha·R = 0), the
       problem is ill-posed and the valley floor is flat, so the two runs
       stop at *different* points and this check does not measure gradient
       consistency. Measured on the five-layer problem below (values and
       conditions in the commit that removed the LLS arm): the two stall points differ visibly in
       max|dD|, each is KKT-stationary, neither can be lowered by polishing
       with the other gradient (check 2 in both directions returns exactly
       0), and the sign of f_ana - f_fd depends on the BLAS thread count,
       i.e. on floating-point reduction order. The earlier relaxed form
       "analytic <= FD" therefore asserted an ordering with no meaning and
       failed or passed by rounding; it is dropped, not loosened.
    2. Polish: restarting SLSQP(jac=None) from the analytic solution must
       not lower f by more than rel 1e-6 — i.e. the analytic solution is a
       true minimizer as seen by finite differences. This is the
       discriminating check for all three methods (pre-fix the FD polish
       descended 2-9%), and the only one applied to LLS.
    """
    from deppro.inverse_solver import FreeSolver, SolverConfig

    profile, tm, I_obs = _build_problem()
    n_depth, n_elements = profile.concentrations.shape
    D_prior = np.ones((n_depth, n_elements)) / n_elements

    kwargs = dict(
        method=method,
        alpha=0.1,
        max_iter=3000,
        tol=1e-14,
        scipy_method='SLSQP',
    )
    solver_ana = FreeSolver(tm, SolverConfig(**kwargs), D_prior)
    solver_fd = FreeSolver(
        tm, SolverConfig(**kwargs, use_numerical_gradient=True), D_prior
    )

    r_ana = solver_ana.solve(I_obs)
    r_fd = solver_fd.solve(I_obs)
    f_ana, f_fd = r_ana.objective, r_fd.objective

    # Check 1: direct agreement (L2 / MEM — unique minimizer). Skipped for
    # LLS: two valid stall points on the flat chi2 floor, see docstring.
    rel = abs(f_ana - f_fd) / max(abs(f_fd), 1e-300)
    if method != 'LLS':
        assert rel <= 1e-6, (
            f"{method}: SLSQP final objective differs between analytical jac "
            f"({f_ana:.12e}) and numerical jac ({f_fd:.12e}), rel={rel:.3e}"
        )

    # Check 2: FD polish from the analytic solution must not descend
    r_polish = solver_fd.solve(I_obs, D_init=r_ana.D_opt)
    improvement = (f_ana - r_polish.objective) / max(abs(f_ana), 1e-300)
    assert improvement <= 1e-6, (
        f"{method}: numerical-jac SLSQP improved on the analytic solution by "
        f"rel {improvement:.3e} — analytic gradient inconsistent with the "
        f"objective"
    )


def test_old_simplified_gradient_is_inconsistent():
    """Guard: the pre-fix 'simplified' gradient really does mismatch FD.

    Documents the failure mode so the FD-match test above is known to be
    discriminating (it would have failed before the fix).
    """
    from deppro.inverse_solver import FreeSolver, SolverConfig

    profile, tm, I_obs = _build_problem()
    n_depth, n_elements = profile.concentrations.shape

    config = SolverConfig(method='LLS', alpha=0.0, degree=1.0)
    D_prior = np.ones((n_depth, n_elements)) / n_elements
    solver = FreeSolver(tm, config, D_prior)
    args = _objective_args(solver, I_obs, degree=1.0)

    rng = np.random.default_rng(7)
    D = rng.uniform(0.05, 0.95, (n_depth, n_elements))
    D = D / D.sum(axis=1, keepdims=True)
    x = D.flatten()

    # Pre-fix gradient: degree * T.T @ residual, no normalization chain rule
    T_matrices = args[1]
    I_pred = np.column_stack([T_matrices[e] @ D[:, e] for e in range(n_elements)])
    I_pred_norm = I_pred / np.maximum(I_pred.sum(axis=1, keepdims=True), 1e-10)
    residual = I_pred_norm - args[0]
    g_old = np.column_stack(
        [T_matrices[e].T @ residual[:, e] for e in range(n_elements)]
    ).flatten()

    g_fd = _central_diff_gradient(solver._objective, x, args)
    rel_err = np.linalg.norm(g_old - g_fd) / max(np.linalg.norm(g_fd), 1e-12)
    assert rel_err > 1e-3, (
        "Simplified gradient unexpectedly matches FD — this guard (and the "
        "tests above) may no longer be discriminating"
    )


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
