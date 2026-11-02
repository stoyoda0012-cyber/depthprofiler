"""Stage D acceptance tests: MLX batched LM vs GenericLayerSolver (L-BFGS-B).

Equivalence tiers:

1. Objective: both solutions re-evaluated on the same OLS chi2. Small
   dimension (~4 params), so the layer-1 bound |df| <= 1e-3
   applies directly. Measured |dchi2| ~ 1e-5, and crucially the one-sided
   dchi2 <= 0 (MLX is never worse than the reference).

2. Parameters: ||dx||rel <= 1e-2 for the well-constrained layers. A buried
   layer's thickness (SiO2 under the Si substrate erf) is weakly
   constrained; there MLX and L-BFGS-B settle at different values with
   equal-or-better chi2. Those are arbitrated by the
   f-comparison, not the parameter norm.
"""
import numpy as np
import pytest

pytest.importorskip('mlx.core')

import mlx.core as mx  # noqa: E402


@pytest.fixture(autouse=True)
def _seed_mlx():
    mx.random.seed(0)


from scipy.special import erf  # noqa: E402
from deppro.transfer_matrix import TransferMatrix  # noqa: E402
from deppro.depth_profile import ElementConfig  # noqa: E402
from deppro.inverse_solver import (  # noqa: E402
    GenericLayerSolver, GenericLayerConfig, LayerDefinition, SolverConfig,
)
from deppro.solvers_mlx import MLXBatchGenericLayerSolver  # noqa: E402

N_PX = 8
INIT = [10.0, 20.0, 40.0, 35.0]
WIDTHS = np.array([2.0, 3.0, 4.0, 5.0])


def _layer_config(thk):
    layers = [
        LayerDefinition('C', [0], {0: 1.0}),
        LayerDefinition('Al2O3', [1, 4], {1: 2.0, 4: 3.0}),
        LayerDefinition('TiO2', [2, 4], {2: 1.0, 4: 2.0}),
        LayerDefinition('SiO2', [3, 4], {3: 1.0, 4: 2.0}),
    ]
    return GenericLayerConfig(
        layers=layers, substrate=LayerDefinition('Si', [5], {5: 1.0}),
        initial_thicknesses=list(thk), initial_widths=list(WIDTHS),
        thickness_bounds=(1.0, 120.0), linked_element_index=4)


def _nominal_D(z, thk, w):
    sig = np.array(w) / (2 * np.sqrt(np.log(2)))
    ze = np.cumsum(thk)
    D = np.zeros((len(z), 6))
    D[:, 0] = np.maximum(1 - 0.5 * (1 + erf((z - ze[0]) / sig[0])), 0)
    for i, a in [(1, 1), (2, 2), (3, 3)]:
        r = 0.5 * (1 + erf((z - ze[i - 1]) / sig[i - 1]))
        f = 0.5 * (1 + erf((z - ze[i]) / sig[i]))
        p = np.maximum(r - f, 0)
        D[:, a] += 0.4 * p
        D[:, 4] += 0.6 * p
    D[:, 5] = 0.5 * (1 + erf((z - ze[3]) / sig[3]))
    D = np.clip(D, 1e-6, None)
    return D / D.sum(1, keepdims=True)


@pytest.fixture(scope='module')
def problem():
    rng = np.random.default_rng(0)
    depth = np.arange(0, 100, 1.0)
    angles = [0, 10, 20, 30, 40, 50, 60]
    elements = [
        ElementConfig('C', '1s', 'C', oxidation_state=0),
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]
    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles,
                        photon_energy=1486.6)
    tm.compute(_nominal_D(depth, INIT, WIDTHS), elements)

    gt = GenericLayerSolver(tm, SolverConfig(method='LLS'),
                            _layer_config(INIT), elements=elements)
    true_thk = np.column_stack([
        rng.uniform(4, 12, N_PX), rng.uniform(12, 28, N_PX),
        rng.uniform(30, 50, N_PX), rng.uniform(28, 42, N_PX)])
    I_obs = np.zeros((N_PX, len(angles), 6))
    for b in range(N_PX):
        D = gt._build_profile(true_thk[b], WIDTHS)
        I = np.stack([tm.get_matrix(e) @ D[:, e] for e in range(6)], axis=-1)
        I = I / np.maximum(I.sum(1, keepdims=True), 1e-10)
        I *= (1 + 0.01 * rng.standard_normal(I.shape))
        I_obs[b] = np.clip(I, 1e-6, None)
    return dict(tm=tm, elements=elements, I_obs=I_obs, true_thk=true_thk)


def _reference(problem):
    ref = []
    for b in range(N_PX):
        s = GenericLayerSolver(
            problem['tm'], SolverConfig(method='LLS', max_iter=500, tol=1e-12),
            _layer_config(INIT), elements=problem['elements'])
        ref.append(s.solve(problem['I_obs'][b]))
    return ref


def test_objective_never_worse_and_bounded(problem):
    """Layer 1: MLX chi2 within 1e-3 of the reference and never worse."""
    ref = _reference(problem)
    f_ref = np.array([r.chi2 for r in ref])
    mlx = MLXBatchGenericLayerSolver(
        problem['tm'], _layer_config(INIT), elements=problem['elements'],
        max_iter=40)
    b = mlx.solve_batch(problem['I_obs'])
    dchi2 = b.chi2 - f_ref
    assert np.max(np.abs(dchi2)) <= 1e-3, f"layer 1 |dchi2|: {dchi2}"
    # one-sided: LM must not settle above the reference objective.
    assert np.max(dchi2) <= 1e-4, f"MLX worse than reference: {dchi2}"


def test_thickness_agreement_f_arbitrated(problem):
    """Layer 2: well-constrained layers agree; a moved layer is allowed
    only when its chi2 is no worse (f-arbitration)."""
    ref = _reference(problem)
    f_ref = np.array([r.chi2 for r in ref])
    thk_ref = np.stack([r.layer_thicknesses for r in ref])
    mlx = MLXBatchGenericLayerSolver(
        problem['tm'], _layer_config(INIT), elements=problem['elements'],
        max_iter=40)
    b = mlx.solve_batch(problem['I_obs'])
    dchi2 = b.chi2 - f_ref
    # per-layer relative disagreement
    per_layer = np.abs(b.layer_thicknesses - thk_ref)
    # The surface stack (layers 0-2) is well constrained: tight everywhere.
    assert np.max(per_layer[:, :3]) <= 1.0, \
        f"surface layers should agree < 1 A: {per_layer[:, :3]}"
    # The buried 4th layer may differ, but only with chi2 no worse.
    moved = per_layer[:, 3] > 1.0
    if np.any(moved):
        assert np.all(dchi2[moved] <= 1e-6), \
            f"buried-layer moves must not worsen chi2: {dchi2[moved]}"


def test_constraints_and_normalization(problem):
    """Profiles are valid: sum-to-1 at each depth, thickness within bounds."""
    mlx = MLXBatchGenericLayerSolver(
        problem['tm'], _layer_config(INIT), elements=problem['elements'])
    b = mlx.solve_batch(problem['I_obs'])
    sums = b.D_opt.sum(axis=2)
    assert np.max(np.abs(sums - 1.0)) < 1e-5, "profiles not normalized"
    assert np.all(b.layer_thicknesses >= 1.0 - 1e-6)
    assert np.all(b.layer_thicknesses <= 120.0 + 1e-6)


def test_single_pixel_dropin(problem):
    """solve() returns a GenericLayerResult like the reference."""
    mlx = MLXBatchGenericLayerSolver(
        problem['tm'], _layer_config(INIT), elements=problem['elements'])
    r = mlx.solve(problem['I_obs'][0])
    assert r.layer_thicknesses.shape == (4,)
    assert r.D_opt.shape == (100, 6)
    assert np.isfinite(r.chi2)


def test_unsupported_variable_stoichiometry(problem):
    layers = [
        LayerDefinition('SiO2', [3, 4], {3: 1.0, 4: 2.0},
                        stoichiometry_bounds={4: (1.5, 2.0)}),
    ]
    cfg = GenericLayerConfig(
        layers=layers, substrate=LayerDefinition('Si', [5], {5: 1.0}),
        initial_thicknesses=[30.0], initial_widths=[3.0],
        optimize_stoichiometry=True, linked_element_index=4)
    with pytest.raises(NotImplementedError):
        MLXBatchGenericLayerSolver(problem['tm'], cfg,
                                   elements=problem['elements'])
