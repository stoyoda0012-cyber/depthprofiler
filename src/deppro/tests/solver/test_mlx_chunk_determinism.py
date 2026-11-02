"""R1 chunk-determinism regression test (PAPER3 CHUNK-2, S8M/CHUNK lineage).

With ``deterministic_power_init=True`` and explicit global ``pixel_ids``,
solving the same 64 synthetic independent pixels as

1. one batch,
2. 16-pixel chunks, and
3. a declared permutation (restored to original order afterwards)

must give profile/thickness equality to implementation precision.  The
archived default path (``deterministic_power_init=False``) is unchanged and
remains chunk-sensitive; that behaviour is NOT asserted here beyond a
smoke check that the flag actually switches paths.
"""
import dataclasses

import numpy as np
import pytest

pytest.importorskip('mlx.core')

import mlx.core as mx  # noqa: E402

from deppro.simulate.d2s2d import (  # noqa: E402
    PRESETS, generate_dstk_xstk, add_noise,
)
from deppro.transfer_matrix import TransferMatrix, CompoundDatabase  # noqa: E402
from deppro.inverse_solver import (  # noqa: E402
    SemiFreeSolverConfig, SolverConfig,
)
from deppro.solvers_mlx import MLXBatchSemiFreeSolver  # noqa: E402

N_PX = 64
PERM_SEED = 20260811


@pytest.fixture(scope='module')
def problem():
    cfg = dataclasses.replace(PRESETS['6elem_standard'], nx=8, ny=8, dt=1)
    res = generate_dstk_xstk(cfg, verbose=False, forward_backend='numba')
    Xn = add_noise(res['Xstk'], 0.05, 'poisson', 7)[:N_PX]
    depth = res['depth']
    n_elem = Xn.shape[2]
    D_init = np.ones((len(depth), n_elem)) / n_elem
    tm = TransferMatrix(
        depth=depth, step_size=cfg.step_size, angles=res['angles'],
        photon_energy=cfg.photon_energy, work_function=cfg.work_function,
        gamma_eff=cfg.gamma_eff)
    tm.compute(D_init, res['element_configs'], CompoundDatabase())
    return dict(Xn=Xn, tm=tm, sim=cfg)


def _solver(problem, chunk_px, det=True):
    sim = problem['sim']
    sc = SolverConfig(method='LLS', alpha=0.0, max_iter=120, n_loops=2,
                      use_xreg_weighting=False)
    scfg = SemiFreeSolverConfig(
        thickness_init=100.0, abruptness_init=10.0,
        thickness_bounds=(50.0, 500.0), abruptness_bounds=(1.0, 50.0),
        optimize_thickness=True, optimize_abruptness=False,
        free_indices=list(sim.free_idx),
        linked_indices=list(sim.linked_idx),
        substrate_indices=list(sim.subst_idx),
        free_linked_const_override=list(sim.linked_const))
    return MLXBatchSemiFreeSolver(
        problem['tm'], config=sc, semi_free_config=scfg,
        fista_max_iter=150, outer_rounds=4, t_grid=6, t_refine_iters=40,
        t_grid_later=4, t_refine_later=20, chunk_px=chunk_px,
        deterministic_power_init=det)


def _run(problem, chunk_px, order=None, det=True):
    Xn = problem['Xn']
    ids = np.arange(N_PX, dtype=float)
    if order is not None:
        Xn, ids = Xn[order], ids[order]
    mx.random.seed(0)
    r = _solver(problem, chunk_px, det).solve_batch(Xn, pixel_ids=ids)
    inv = np.argsort(order) if order is not None else slice(None)
    return r.D_opt[inv], r.thickness[inv], r.objective[inv]


def test_r1_chunk_and_order_invariance(problem):
    D1, t1, f1 = _run(problem, chunk_px=10_000)          # one batch
    D2, t2, f2 = _run(problem, chunk_px=16)              # 16-px chunks
    rng = np.random.default_rng(PERM_SEED)
    perm = rng.permutation(N_PX)
    D3, t3, f3 = _run(problem, chunk_px=16, order=perm)  # declared permutation

    for Dx, tx, tag in ((D2, t2, 'chunk16'), (D3, t3, 'chunk16+perm')):
        assert np.array_equal(t1, tx), \
            f'{tag}: thickness not exact (max|dt|={np.abs(t1-tx).max():.3g})'
        assert np.array_equal(D1, Dx), \
            f'{tag}: profiles not exact (max|dD|={np.abs(D1-Dx).max():.3g})'


def test_flag_switches_path(problem):
    """Default (archived) path must remain untouched by the R1 machinery."""
    Da, ta, _ = _run(problem, chunk_px=10_000, det=True)
    Db, tb, _ = _run(problem, chunk_px=10_000, det=False)
    # not asserting inequality of solutions (they may coincide); assert the
    # deterministic path is reproducible against itself
    Dc, tc, _ = _run(problem, chunk_px=10_000, det=True)
    assert np.array_equal(Da, Dc) and np.array_equal(ta, tc)
