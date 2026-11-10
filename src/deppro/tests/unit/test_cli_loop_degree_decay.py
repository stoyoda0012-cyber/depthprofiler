"""`deppro solve --loops N` decays the data weight, not alpha.

The objective is ``0.5 * degree * chi2 + alpha * R``. The MATLAB original
(Run_DepthProfiling.m) multiplies ``degree`` by 0.8 on every loop after the
first when DegreeControl is on, which makes the regularization relatively
stronger; alpha is left alone. FreeSolver.solve, solver_workers, the
streaming pipeline and the MLX batch solver all do the same.

Until 2026-11-10 the CLI loop multiplied ``alpha`` by 0.8 instead (the
opposite direction: regularization got weaker), and it rebuilt SolverConfig
from a hand-picked list of fields on every loop, so any field outside that
list fell back to its default from loop 2 on. These tests pin the corrected
schedule and the carry-over of the remaining fields.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from deppro.cli_depth import _solve_semifree_loops
from deppro.inverse_solver import (
    SemiFreeSolver, SemiFreeSolverConfig, SolverConfig,
)


class _RecordingTM:
    def __init__(self):
        self.compute_calls = []

    def compute(self, D, elements):
        self.compute_calls.append(np.array(D, copy=True))


def _recording_solver(seen):
    """A stand-in for SemiFreeSolver that records the config of each loop."""

    class _Solver:
        def __init__(self, transfer_matrix, config, semi_free_config,
                     D_prior=None, elements=None, use_numba=True):
            self.config = config

        def solve(self, intensities, verbose=False):
            seen.append(self.config)
            k = len(seen)
            return SimpleNamespace(D_opt=np.full((3, 2), float(k)),
                                   thickness=10.0 * k, chi2=1.0 / k,
                                   success=True)

    return _Solver


def _run(config, n_loops):
    seen, tm = [], _RecordingTM()
    result, final = _solve_semifree_loops(
        tm, elements=['a', 'b'], intensities=np.ones((4, 2)),
        solver_config=config, semi_free_config=None, n_loops=n_loops,
        solver_cls=_recording_solver(seen))
    return seen, tm, result, final


def test_loops_decay_degree_and_keep_alpha():
    cfg = SolverConfig(method='MEM', alpha=0.003, degree=2.0)
    seen, _, _, final = _run(cfg, 4)

    assert [c.alpha for c in seen] == [0.003] * 4
    np.testing.assert_allclose([c.degree for c in seen],
                               [2.0 * 0.8 ** k for k in range(4)])
    # Relative regularization strength alpha/degree must grow, never shrink.
    ratio = [c.alpha / c.degree for c in seen]
    assert all(b > a for a, b in zip(ratio, ratio[1:]))
    assert final.degree == pytest.approx(2.0 * 0.8 ** 3)
    # The caller's config object is not mutated.
    assert cfg.degree == 2.0 and cfg.alpha == 0.003


def test_degree_decay_comes_from_the_config():
    seen, _, _, _ = _run(SolverConfig(degree_decay=0.5), 3)
    np.testing.assert_allclose([c.degree for c in seen], [1.0, 0.5, 0.25])

    seen, _, _, _ = _run(SolverConfig(degree_decay=1.0), 3)
    assert [c.degree for c in seen] == [1.0, 1.0, 1.0]


def test_single_loop_leaves_the_config_untouched():
    cfg = SolverConfig(alpha=0.01)
    seen, tm, _, final = _run(cfg, 1)
    assert len(seen) == 1 and seen[0] == cfg and final == cfg
    assert tm.compute_calls == []


def test_every_other_field_survives_the_loops():
    """No SolverConfig field may fall back to its default from loop 2 on."""
    cfg = SolverConfig(
        method='MEM', alpha=0.02, degree=3.0, degree_decay=0.9,
        max_iter=17, tol=1e-9, min_conc=1e-4, max_conc=0.7,
        scipy_method='trust-constr', mem_space='log',
        use_xreg_weighting=True, xreg_max_weight=50.0,
        use_sum_constraint=False, use_charge_constraint=False,
        use_wls=True, l1_smoothing='log_cosh', l1_beta=0.05,
        max_warm_restarts=0, enable_fallback=False,
    )
    seen, _, _, _ = _run(cfg, 3)
    for k, c in enumerate(seen):
        expected = dataclasses.replace(cfg, degree=3.0 * 0.9 ** k)
        got = dataclasses.replace(c, degree=expected.degree)
        assert got == expected, f"loop {k + 1} lost a field"
        assert c.degree == pytest.approx(expected.degree)


def test_transfer_matrix_is_rebuilt_from_the_previous_profile():
    _, tm, result, _ = _run(SolverConfig(), 3)
    assert len(tm.compute_calls) == 2
    np.testing.assert_array_equal(tm.compute_calls[0], np.full((3, 2), 1.0))
    np.testing.assert_array_equal(tm.compute_calls[1], np.full((3, 2), 2.0))
    np.testing.assert_array_equal(result.D_opt, np.full((3, 2), 3.0))


def test_real_solver_second_loop_runs_with_decayed_degree():
    """End to end with SemiFreeSolver: loop 2 of the CLI helper equals a
    hand-built second solve with degree*0.8 and the same alpha, and differs
    from the pre-fix schedule (alpha*0.8, degree unchanged)."""
    from deppro.simulate.d2s2d import PRESETS, generate_dstk_xstk
    from deppro.transfer_matrix import TransferMatrix, CompoundDatabase

    sim = dataclasses.replace(PRESETS['6elem_standard'], nx=2, ny=2, dt=1)
    res = generate_dstk_xstk(sim, verbose=False, forward_backend='numba')
    X = res['Xstk'][0]
    depth, elements = res['depth'], res['element_configs']
    n_elem = X.shape[1]

    def fresh_tm():
        tm = TransferMatrix(
            depth=depth, step_size=sim.step_size, angles=res['angles'],
            photon_energy=sim.photon_energy, work_function=sim.work_function,
            gamma_eff=sim.gamma_eff)
        tm.compute(np.ones((len(depth), n_elem)) / n_elem, elements,
                   CompoundDatabase())
        return tm

    sc = SolverConfig(method='L2', alpha=0.05, max_iter=60,
                      scipy_method='SLSQP', use_xreg_weighting=True)
    scfg = SemiFreeSolverConfig(
        thickness_init=float(np.clip(res['FV_all'][0, 0], 50.0, 500.0)),
        abruptness_init=8.0, thickness_bounds=(50.0, 500.0),
        abruptness_bounds=(1.0, 50.0),
        optimize_thickness=True, optimize_abruptness=False,
        free_indices=list(sim.free_idx), linked_indices=list(sim.linked_idx),
        substrate_indices=list(sim.subst_idx),
        free_linked_const_override=list(sim.linked_const))

    result, final = _solve_semifree_loops(fresh_tm(), elements, X, sc, scfg, 2)
    assert final.degree == pytest.approx(0.8) and final.alpha == 0.05

    def second_loop(config):
        tm = fresh_tm()
        first = SemiFreeSolver(tm, sc, scfg, D_prior=None,
                               elements=elements).solve(X)
        tm.compute(first.D_opt, elements)
        return SemiFreeSolver(tm, config, scfg, D_prior=None,
                              elements=elements).solve(X)

    expected = second_loop(dataclasses.replace(sc, degree=0.8))
    np.testing.assert_allclose(result.D_opt, expected.D_opt, rtol=0, atol=1e-12)
    assert result.thickness == pytest.approx(expected.thickness, abs=1e-12)

    old = second_loop(dataclasses.replace(sc, alpha=0.05 * 0.8))
    assert not np.allclose(result.D_opt, old.D_opt, atol=1e-6)
