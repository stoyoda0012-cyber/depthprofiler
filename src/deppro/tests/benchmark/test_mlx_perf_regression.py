"""Performance regression guard for the MLX batched Free solver.

Fails if the B=1000 reduced benchmark exceeds 1.5x the recorded baseline.
Baseline measured 2026-07-05 on M3 Max, mlx 0.30.3, exact-breakpoint
projection: solver wall ~0.074 s (13.5K px/s). BASELINE_S is set with
machine-variance headroom; update it deliberately, recording the new
measurement and its conditions, if the algorithm legitimately changes.
"""
import dataclasses

import numpy as np
import pytest

pytest.importorskip('mlx.core')

BASELINE_S = 0.12
FAIL_FACTOR = 1.5
B = 1000


@pytest.mark.timeout(300)
def test_free_batch_b1000_regression():
    from deppro.simulate.d2s2d import PRESETS, generate_dstk_xstk, add_noise
    from deppro.transfer_matrix import TransferMatrix, CompoundDatabase
    from deppro.inverse_solver import SolverConfig
    from deppro.solvers_mlx import MLXBatchFreeSolver

    cfg = dataclasses.replace(PRESETS['6elem_standard'], nx=5, ny=5, dt=1)
    res = generate_dstk_xstk(cfg, verbose=False, forward_backend='numba')
    reps = int(np.ceil(B / len(res['Xstk'])))
    Xn = add_noise(np.tile(res['Xstk'], (reps, 1, 1))[:B], 0.03, 'poisson', 42)
    depth, angles = res['depth'], res['angles']
    n_elem = Xn.shape[2]
    D_init = np.ones((len(depth), n_elem)) / n_elem
    tm = TransferMatrix(
        depth=depth, step_size=cfg.step_size, angles=angles,
        photon_energy=cfg.photon_energy, work_function=cfg.work_function,
        gamma_eff=cfg.gamma_eff)
    tm.compute(D_init, res['element_configs'], CompoundDatabase())

    solver = MLXBatchFreeSolver(
        tm, SolverConfig(method='L2', alpha=0.01, n_loops=1),
        fista_max_iter=350, conv_tol=1e-3)
    solver.solve_batch(Xn)               # warmup (JIT/compile)
    r = solver.solve_batch(Xn)           # steady state

    assert r.elapsed_time <= BASELINE_S * FAIL_FACTOR, (
        f"perf regression: {r.elapsed_time:.3f}s > "
        f"{BASELINE_S * FAIL_FACTOR:.3f}s (baseline {BASELINE_S}s x{FAIL_FACTOR})")
    assert r.converged.all()
