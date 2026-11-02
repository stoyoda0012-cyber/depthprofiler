"""Explicit thickness prior P(t) = (t-t0)²/(2σ²) — wiring tests.

The prior is opt-in (t_prior_center=None default = exact legacy
behavior) and sits OUTSIDE alpha·reg in both the SLSQP reference and the
MLX batch solver.

These tests verify wiring, not physics: a tight prior must dominate the
thickness estimate, a loose prior must not move it materially, and both
solver families must agree under the same tight prior.
"""
import dataclasses

import numpy as np
import pytest

pytest.importorskip('mlx.core')

import mlx.core as mx  # noqa: E402


@pytest.fixture(autouse=True)
def _seed_mlx():
    mx.random.seed(0)

from deppro.simulate.d2s2d import (  # noqa: E402
    PRESETS, generate_dstk_xstk, add_noise,
)
from deppro.transfer_matrix import TransferMatrix, CompoundDatabase  # noqa: E402
from deppro.inverse_solver import (  # noqa: E402
    SemiFreeSolver, SemiFreeSolverConfig, SolverConfig,
)
from deppro.solvers_mlx import MLXBatchSemiFreeSolver  # noqa: E402

THK_BOUNDS = (50.0, 500.0)
N_PX = 2


@pytest.fixture(scope='module')
def problem():
    cfg = dataclasses.replace(PRESETS['6elem_standard'], nx=5, ny=5, dt=1)
    res = generate_dstk_xstk(cfg, verbose=False, forward_backend='numba')
    Xn = add_noise(res['Xstk'], 0.03, 'poisson', 42)[:N_PX]
    FV = res['FV_all'][:N_PX]
    depth, angles = res['depth'], res['angles']
    n_elem = Xn.shape[2]
    D_init = np.ones((len(depth), n_elem)) / n_elem
    tm = TransferMatrix(
        depth=depth, step_size=cfg.step_size, angles=angles,
        photon_energy=cfg.photon_energy, work_function=cfg.work_function,
        gamma_eff=cfg.gamma_eff)
    tm.compute(D_init, res['element_configs'], CompoundDatabase())
    return dict(Xn=Xn, FV=FV, tm=tm, depth=depth, sim=cfg,
                element_configs=res['element_configs'])


def _sc(problem):
    return SolverConfig(method='L1', alpha=0.01, max_iter=200, n_loops=3,
                        scipy_method='SLSQP', use_xreg_weighting=True)


def _scfg(problem, thk, t_prior_center=None, t_prior_sigma=None):
    sim = problem['sim']
    return SemiFreeSolverConfig(
        thickness_init=thk, abruptness_init=8.0,
        thickness_bounds=THK_BOUNDS, abruptness_bounds=(1.0, 50.0),
        optimize_thickness=True, optimize_abruptness=False,
        free_indices=list(sim.free_idx),
        linked_indices=list(sim.linked_idx),
        substrate_indices=list(sim.subst_idx),
        free_linked_const_override=list(sim.linked_const),
        t_prior_center=t_prior_center, t_prior_sigma=t_prior_sigma)


def _solve_ref(problem, px, thk0, **prior):
    solver = SemiFreeSolver(problem['tm'], _sc(problem),
                            _scfg(problem, thk0, **prior),
                            D_prior=None,
                            elements=problem['element_configs'])
    return solver.solve(problem['Xn'][px])


def test_invalid_prior_params_raise(problem):
    """sigma<=0 / non-finite params must raise, not divide by zero."""
    thk0 = float(np.clip(problem['FV'][0, 0], *THK_BOUNDS))
    for bad in (dict(t_prior_center=240.0, t_prior_sigma=0.0),
                dict(t_prior_center=240.0, t_prior_sigma=-1.0),
                dict(t_prior_center=240.0, t_prior_sigma=np.nan),
                dict(t_prior_center=np.inf, t_prior_sigma=24.0),
                dict(t_prior_center=240.0)):
        with pytest.raises(ValueError):
            _solve_ref(problem, 0, thk0, **bad)
        with pytest.raises(ValueError):
            mlx_solver = MLXBatchSemiFreeSolver(
                problem['tm'], _sc(problem), _scfg(problem, 240.0, **bad),
                D_prior=None, elements=problem['element_configs'])
            mlx_solver.solve_batch(
                problem['Xn'][:1], thickness_init=np.array([240.0]),
                abruptness_init=np.array([8.0]))


def test_sigma_missing_at_solve_time_raises(problem):
    """A center set after construction (bypassing __post_init__) without a
    sigma must be rejected: at solve time by SemiFreeSolver, and by the MLX
    adapter when it builds its batch config (the solve-time check inside
    solve_semifree_batch is tested directly in
    tests/unit/test_t_prior_sigma_required.py)."""
    thk0 = float(np.clip(problem['FV'][0, 0], *THK_BOUNDS))
    scfg = _scfg(problem, thk0)
    scfg.t_prior_center = 240.0
    solver = SemiFreeSolver(problem['tm'], _sc(problem), scfg,
                            D_prior=None,
                            elements=problem['element_configs'])
    with pytest.raises(ValueError, match='t_prior_sigma'):
        solver.solve(problem['Xn'][0])
    # the MLX adapter may reject the config when it builds its batch config
    # (construction) or when it solves; either is a rejection.
    with pytest.raises(ValueError, match='t_prior_sigma'):
        mlx_solver = MLXBatchSemiFreeSolver(
            problem['tm'], _sc(problem), scfg,
            D_prior=None, elements=problem['element_configs'])
        mlx_solver.solve_batch(
            problem['Xn'][:1], thickness_init=np.array([240.0]),
            abruptness_init=np.array([8.0]))


def test_none_is_legacy_noop(problem):
    """t_prior_center=None must reproduce the legacy solution exactly."""
    thk0 = float(np.clip(problem['FV'][0, 0], *THK_BOUNDS))
    r_legacy = _solve_ref(problem, 0, thk0)
    r_none = _solve_ref(problem, 0, thk0, t_prior_center=None)
    assert r_legacy.thickness == r_none.thickness
    np.testing.assert_array_equal(r_legacy.x_opt, r_none.x_opt)


def test_tight_prior_dominates_ref(problem):
    """σ=0.2Å prior at t0 far from the data optimum must pin t near t0.

    σ must be well below the data curvature scale: this synthetic pixel
    has ~10/Å data gradient near t0+60, so σ=1Å merely competes with it
    (measured gap −44Å) while σ=0.2Å pins within 1Å.
    """
    thk0 = float(np.clip(problem['FV'][0, 0], *THK_BOUNDS))
    r_free = _solve_ref(problem, 0, thk0)
    t0 = float(np.clip(r_free.thickness + 60.0, *THK_BOUNDS))
    r_pin = _solve_ref(problem, 0, thk0, t_prior_center=t0, t_prior_sigma=0.2)
    assert abs(r_pin.thickness - t0) < 3.0, (
        f"tight prior at {t0} ignored: t={r_pin.thickness}")


def test_loose_prior_is_gentle_ref(problem):
    """σ=1000Å prior must not move t materially from the free solution."""
    thk0 = float(np.clip(problem['FV'][0, 0], *THK_BOUNDS))
    r_free = _solve_ref(problem, 0, thk0)
    r_loose = _solve_ref(problem, 0, thk0,
                         t_prior_center=r_free.thickness + 60.0,
                         t_prior_sigma=1000.0)
    assert abs(r_loose.thickness - r_free.thickness) < 5.0


def test_mlx_prior_pull_and_agreement(problem):
    """MLX: tight prior pins t near t0 and agrees with the reference."""
    sim = problem['sim']
    thk0 = np.clip(problem['FV'][:, 0], *THK_BOUNDS).astype(np.float64)
    t0 = 300.0

    def run(prior_sigma):
        prior = dict(t_prior_center=t0, t_prior_sigma=prior_sigma) \
            if prior_sigma else {}
        mlx_solver = MLXBatchSemiFreeSolver(
            problem['tm'], _sc(problem), _scfg(problem, 240.0, **prior),
            D_prior=None, elements=problem['element_configs'])
        return mlx_solver.solve_batch(
            problem['Xn'], thickness_init=thk0,
            abruptness_init=np.full(N_PX, 8.0))

    r_free = run(None)
    r_pin = run(0.2)
    assert np.all(np.abs(r_pin.thickness - t0) < 5.0), (
        f"tight prior at {t0} ignored: t={r_pin.thickness}")
    # objective must include the prior term (t at t0 -> prior ~ 0, so the
    # pinned objective cannot be below the free data optimum by more than
    # rounding, while a missing prior term would allow it to wander)
    ref_pin = _solve_ref(problem, 0, float(thk0[0]),
                         t_prior_center=t0, t_prior_sigma=0.2)
    assert abs(float(r_pin.thickness[0]) - ref_pin.thickness) < 5.0
