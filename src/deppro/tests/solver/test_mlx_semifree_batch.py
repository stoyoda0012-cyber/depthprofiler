"""Stage C acceptance tests: MLX batched SemiFree solver vs SLSQP reference.

Equivalence tiers (adapted to the SemiFree class):

1. One-sided objective vs the production SemiFreeSolver (SLSQP,
   analytical/FD gradient per method): both solutions re-evaluated with
   the same float64 objective. The C subproblem inherits the LLS-type
   O(1/k) tail documented in Stage B, so the layer-1 margin is the
   one-sided bound from Stage B (criterion 1 below).

2. Physical parameter: |thickness_mlx - thickness_ref| bounded.

3. D2S2D ground-truth metrics: MLX must not be materially worse than the
   reference on TOTAL RMSE / mean W1 (layer-3 thresholds).

Margins were set from the authors' Stage C benchmark (N=25, gt_fv init).
"""
import dataclasses

import numpy as np
import pytest

pytest.importorskip('mlx.core')
# the fixtures build the transfer matrix through CompoundDatabase(), whose default
# source is toyomacro (optional extra `fit`); without it the module is unverified, not green
pytest.importorskip('toyomacro')

import mlx.core as mx  # noqa: E402


@pytest.fixture(autouse=True)
def _seed_mlx():
    """Deterministic power-iteration vectors: the LLS O(1/k)
    tail sits at 1.3-2.7e-2 depending on the global RNG state
    (order-dependent flake otherwise)."""
    mx.random.seed(0)

from deppro.simulate.d2s2d import (  # noqa: E402
    PRESETS, generate_dstk_xstk, add_noise,
)
from deppro.transfer_matrix import TransferMatrix, CompoundDatabase  # noqa: E402
from deppro.inverse_solver import (  # noqa: E402
    SemiFreeSolver, SemiFreeSolverConfig, SolverConfig,
)
from deppro.solvers_mlx import MLXBatchSemiFreeSolver  # noqa: E402

N_PX = 8  # test subset of the N=25 benchmark
THK_BOUNDS = (50.0, 500.0)


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
                D_gt=res['Dstk'][:N_PX],
                element_configs=res['element_configs'])


def _configs(problem, method):
    sim = problem['sim']
    sc = SolverConfig(method=method, alpha=0.01, max_iter=200, n_loops=3,
                      scipy_method='SLSQP', use_xreg_weighting=True,
                      enable_fallback=True, fallback_method='trust-constr')
    def scfg(thk, abrp):
        return SemiFreeSolverConfig(
            thickness_init=thk, abruptness_init=abrp,
            thickness_bounds=THK_BOUNDS, abruptness_bounds=(1.0, 50.0),
            optimize_thickness=True, optimize_abruptness=False,
            free_indices=list(sim.free_idx),
            linked_indices=list(sim.linked_idx),
            substrate_indices=list(sim.subst_idx),
            free_linked_const_override=list(sim.linked_const))
    return sc, scfg


def _f64_objective(problem, method, I_obs, C, t, a, s, m0_t):
    """Reference objective, float64 (SemiFreeSolver.solve() conventions)."""
    sim, tm = problem['sim'], problem['tm']
    free_idx = list(sim.free_idx)
    anti = free_idx + list(sim.linked_idx)
    sub_idx = list(sim.subst_idx)
    n_elem = I_obs.shape[1]
    sc, scfg = _configs(problem, method)
    ref = SemiFreeSolver(tm, sc, scfg(m0_t, a), D_prior=None,
                         elements=None, use_numba=False)
    D = ref._build_full_profile(C, t, a, np.atleast_1d(s))
    I_pred = np.stack([tm.get_matrix(e) @ D[:, e] for e in range(n_elem)],
                      axis=-1)
    Ipn = I_pred / np.maximum(I_pred[:, sub_idx].sum(1, keepdims=True), 1e-10)
    Ion = I_obs / np.maximum(I_obs[:, sub_idx].sum(1, keepdims=True), 1e-10)
    xreg = np.mean(Ion[:, anti], axis=0)
    xreg[xreg == 0] = 1.0
    chi2 = np.sum(((Ion[:, anti] - Ipn[:, anti]) / xreg) ** 2)
    m0C = ref.D_prior[:, free_idx]
    if method == 'L2':
        reg = np.sum((C - m0C) ** 2) + 100 * ((t - m0_t) / 100) ** 2
    elif method == 'MEM':
        Cs, m0s = np.maximum(C, 1e-10), np.maximum(m0C, 1e-10)
        reg = np.sum(Cs * np.log(Cs / m0s) - Cs + m0s)
        reg += (t * np.log(t / m0_t) - t + m0_t) / 10
    elif method == 'L1':
        reg = np.sum(np.abs(C - m0C)) + abs(t - m0_t) / 100
    else:
        reg = 0.0
    return 0.5 * chi2 + 0.01 * reg


# Converged-equivalence budget (cf. the free_batch LLS test's 4000 iters:
# enough to demonstrate the O(1/k) block-coordinate tail is exhausted, not
# the production default). Measured N=8: pulls every same-basin pixel below
# the 3e-2 bound of criterion 1 (~5s/method).
_TEST_KW = dict(outer_rounds=12, t_grid_later=8, t_refine_later=120,
                fista_max_iter=500)


def _run_pair(problem, method):
    sim = problem['sim']
    sc, scfg = _configs(problem, method)
    thk0 = np.clip(problem['FV'][:, 0], *THK_BOUNDS)
    abrp0 = problem['FV'][:, 1].astype(np.float64)

    f_ref, thk_ref, D_ref = [], [], []
    n_film = len(sim.free_idx)
    for px in range(N_PX):
        solver = SemiFreeSolver(problem['tm'], sc,
                                scfg(float(thk0[px]), float(abrp0[px])),
                                D_prior=None,
                                elements=problem['element_configs'])
        r = solver.solve(problem['Xn'][px])
        C = r.x_opt[:len(problem['depth']) * n_film].reshape(-1, n_film)
        f_ref.append(_f64_objective(problem, method, problem['Xn'][px], C,
                                    r.thickness, r.abruptness,
                                    r.substrate_ratios, float(thk0[px])))
        thk_ref.append(r.thickness)
        D_ref.append(r.D_opt)

    mlx = MLXBatchSemiFreeSolver(
        problem['tm'], sc, scfg(float(np.median(thk0)),
                                float(np.median(abrp0))),
        D_prior=None, elements=problem['element_configs'], **_TEST_KW)
    b = mlx.solve_batch(problem['Xn'], thickness_init=thk0,
                        abruptness_init=abrp0)
    f_mlx = np.array([
        _f64_objective(problem, method, problem['Xn'][px], b.film_conc[px],
                       b.thickness[px], b.abruptness[px], b.sub_ratios[px],
                       float(thk0[px]))
        for px in range(N_PX)])
    return (np.array(f_ref), np.array(thk_ref), np.stack(D_ref),
            f_mlx, b)


@pytest.mark.parametrize('method', ['L1', 'LLS'])
def test_one_sided_objective_and_thickness(problem, method):
    """Layers 1-2 under criteria 1 and 6.

    Criterion 1 (LLS/L1 one-sided 3e-2): same-basin pixels — those whose
    thickness agrees with the reference to <= 1 A — must sit within the
    C-block O(1/k) tail bound. Criterion 6 (dthk > 1 A requires f_mlx <=
    f_ref): a thickness disagreement is acceptable ONLY when MLX's
    objective is no worse than the reference's, i.e. the two settled in
    genuinely equivalent (or MLX found a better) basins — not when MLX
    simply converged to a worse local minimum. This f-arbitration
    replaces the single-basin |dt| <= 0.5 A criterion used earlier.
    """
    f_ref, thk_ref, _, f_mlx, b = _run_pair(problem, method)
    rel = (f_mlx - f_ref) / np.abs(f_ref)
    dthk = np.abs(b.thickness - thk_ref)
    same_basin = dthk <= 1.0

    # Criterion 1: same-basin pixels within the one-sided C-block tail bound.
    if np.any(same_basin):
        assert np.max(rel[same_basin]) <= 3e-2, \
            f"same-basin rel_f (criterion 1): {rel[same_basin]}"
    # Criterion 6: a thickness move > 1 A is allowed only if it does not
    # worsen the objective (flat-valley alternate basin or genuine
    # improvement). A worse f at a moved thickness is a real regression.
    moved = ~same_basin
    if np.any(moved):
        assert np.all(f_mlx[moved] <= f_ref[moved] * (1 + 1e-6)), \
            (f"dthk>1A pixels must not worsen f (criterion 6): "
             f"dthk={dthk[moved]}, rel_f={rel[moved]}")
    # median thickness agreement stays tight regardless.
    assert np.median(dthk) <= 1.0, f"median dthk {np.median(dthk)}"


def test_constraints_satisfied(problem):
    """Weighted sum-to-one on the film concentrations at active depths."""
    method = 'L1'
    _, _, _, _, b = _run_pair(problem, method)
    flc = np.maximum(np.asarray(problem['sim'].linked_const), 1.0)
    # active depths: film fraction meaningfully > 0 at the solved t
    from scipy.special import erf
    depth = problem['depth']
    for px in range(N_PX):
        sigma = b.abruptness[px] / (2 * np.sqrt(np.log(2)))
        erf_sub = 0.5 * (1 + erf((depth - b.thickness[px]) / sigma))
        active = (1.0 - erf_sub) >= 1e-3
        rows = (b.film_conc[px] * flc).sum(axis=1)
        assert np.max(np.abs(rows[active] - 1.0)) < 1e-4


def test_gt_metrics_not_materially_worse(problem):
    """Layer 3: |dRMSE| <= 0.002 and |dW1| <= 0.3 A vs the
    reference on the same pixels."""
    from deppro.depth_profile import wasserstein_distance_1d
    _, _, D_ref, _, b = _run_pair(problem, 'L1')
    D_gt, depth = problem['D_gt'], problem['depth']

    def metrics(D):
        rmse = float(np.sqrt(np.mean((D - D_gt) ** 2)))
        w1 = float(np.nanmean([
            wasserstein_distance_1d(depth, D_gt[p, :, e], D[p, :, e])
            for p in range(N_PX) for e in range(D_gt.shape[2])
            if D_gt[p, :, e].sum() > 1e-6]))
        return rmse, w1

    rmse_r, w1_r = metrics(D_ref)
    rmse_m, w1_m = metrics(b.D_opt)
    assert rmse_m - rmse_r <= 0.002, f"RMSE {rmse_r:.4f} -> {rmse_m:.4f}"
    assert w1_m - w1_r <= 0.3, f"W1 {w1_r:.2f} -> {w1_m:.2f}"


def test_unsupported_features_raise(problem):
    sc, scfg = _configs(problem, 'L1')
    cfg = scfg(200.0, 20.0)
    cfg.use_nonlinear_sum_constraint = True
    with pytest.raises(NotImplementedError):
        MLXBatchSemiFreeSolver(problem['tm'], sc, cfg,
                               elements=problem['element_configs'])
    sc2 = SolverConfig(method='MEM', mem_space='log')
    with pytest.raises(NotImplementedError):
        MLXBatchSemiFreeSolver(problem['tm'], sc2, scfg(200.0, 20.0),
                               elements=problem['element_configs'])
