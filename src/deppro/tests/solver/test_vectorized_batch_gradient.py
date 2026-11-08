"""VectorizedBatchSolver: the data-term gradient must be the gradient of the
chi2 that ``solve_batch`` reports (per-angle-normalized intensities).

Until 2026-11-07 ``_gradient_batch`` used ``residual @ T``, the gradient of the
*unnormalized* least squares, while the residual and chi2 were computed on
``I_pred / sum_e I_pred``; scipy's ``check_grad`` gave ~80 % relative error.
"""
import numpy as np
import pytest
from scipy.optimize import check_grad

from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import SolverConfig, VectorizedBatchSolver
from deppro.transfer_matrix import TransferMatrix


def _problem():
    depth = np.arange(0, 60, 5.0)
    angles = np.array([0.0, 30.0, 45.0, 60.0, 70.0])
    elements = [
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
    ]
    tm = TransferMatrix(depth=depth, step_size=5.0, angles=angles, photon_energy=1486.6)
    rng = np.random.default_rng(0)
    D0 = rng.uniform(0.05, 1.0, size=(len(depth), len(elements)))
    D0 /= D0.sum(axis=1, keepdims=True)
    tm.compute(D0, elements)
    return tm, elements, rng


@pytest.mark.parametrize('method', ['LLS', 'L2', 'MEM'])
def test_gradient_batch_matches_normalized_chi2(method):
    tm, elements, rng = _problem()
    config = SolverConfig(method=method, alpha=0.3, degree=1.0)
    solver = VectorizedBatchSolver(tm, config, D_prior=None)
    n_depth, n_elem = tm.depth.size, len(elements)

    I_obs = rng.uniform(0.1, 1.0, size=(1, tm.angles.size, n_elem))
    I_norm = I_obs / I_obs.sum(axis=2, keepdims=True)

    def objective(x):
        D = x.reshape(1, n_depth, n_elem)
        I_pred = solver._forward_batch(D)
        I_pred_norm = I_pred / np.maximum(I_pred.sum(axis=2, keepdims=True), 1e-10)
        data = 0.5 * config.degree * np.sum((I_pred_norm - I_norm) ** 2)
        if method == 'L2':
            reg = config.alpha * np.sum((D - solver.D_prior) ** 2)
        elif method == 'MEM':
            m0 = np.maximum(solver.D_prior, 1e-10)
            Ds = np.maximum(D, 1e-10)
            reg = config.alpha * np.sum(Ds * np.log(Ds / m0) - Ds + m0)
        else:
            reg = 0.0
        return data + reg

    def gradient(x):
        D = x.reshape(1, n_depth, n_elem)
        I_pred = solver._forward_batch(D)
        I_pred_sum = I_pred.sum(axis=2, keepdims=True)
        I_pred_norm = I_pred / np.maximum(I_pred_sum, 1e-10)
        residual = I_pred_norm - I_norm
        g = solver._gradient_batch(D, residual, config.degree, config.alpha,
                                   I_pred_norm=I_pred_norm, I_pred_sum=I_pred_sum)
        return g.ravel()

    x0 = rng.uniform(0.05, 1.0, size=n_depth * n_elem)
    err = check_grad(objective, gradient, x0, epsilon=1e-6)
    scale = np.linalg.norm(gradient(x0))
    assert err / scale < 1e-4, f"relative gradient error {err / scale:.2e}"


def test_legacy_call_without_normalization_still_works():
    """The two new keyword arguments are optional (old behavior kept)."""
    tm, elements, rng = _problem()
    solver = VectorizedBatchSolver(tm, SolverConfig(method='LLS'), D_prior=None)
    D = np.full((2, tm.depth.size, len(elements)), 1 / 3)
    residual = rng.normal(size=(2, tm.angles.size, len(elements)))
    g = solver._gradient_batch(D, residual, 1.0, 0.0)
    assert g.shape == D.shape and np.all(np.isfinite(g))
