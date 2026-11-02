"""t_prior_sigma has no default: setting t_prior_center without it must fail.

The sigma of the explicit thickness prior is a penalty width chosen for a
specific objective convention and specific external knowledge, so the
configs do not supply one. Leaving the prior off must keep working with
sigma unset.
"""
import pytest

from deppro.inverse_solver import SemiFreeSolverConfig


def test_semifree_config_requires_sigma_with_center():
    with pytest.raises(ValueError, match='t_prior_sigma'):
        SemiFreeSolverConfig(t_prior_center=240.0)


def test_semifree_config_prior_off_needs_no_sigma():
    cfg = SemiFreeSolverConfig()
    assert cfg.t_prior_center is None
    assert cfg.t_prior_sigma is None


def test_semifree_config_accepts_explicit_sigma():
    cfg = SemiFreeSolverConfig(t_prior_center=240.0, t_prior_sigma=24.0)
    assert cfg.t_prior_sigma == 24.0


def test_mlx_batch_config_requires_sigma_with_center():
    pytest.importorskip('mlx.core')
    from deppro.solvers_mlx.semifree_batch import SemiFreeBatchConfig
    with pytest.raises(ValueError, match='t_prior_sigma'):
        SemiFreeBatchConfig(t_prior_center=240.0)
    assert SemiFreeBatchConfig().t_prior_sigma is None
    assert SemiFreeBatchConfig(t_prior_center=240.0,
                               t_prior_sigma=24.0).t_prior_sigma == 24.0


def test_mlx_solve_rejects_center_without_sigma_at_solve_time():
    """A config whose center is set after construction (bypassing
    __post_init__) must still be rejected by solve_semifree_batch itself."""
    pytest.importorskip('mlx.core')
    import numpy as np
    from deppro.solvers_mlx.semifree_batch import (
        SemiFreeBatchConfig, solve_semifree_batch)
    cfg = SemiFreeBatchConfig()
    cfg.t_prior_center = 240.0
    n_angles, n_depth, B = 2, 4, 1
    with pytest.raises(ValueError, match='t_prior_sigma'):
        solve_semifree_batch(
            T=np.ones((n_angles, n_depth, 2)),
            I_obs_norm=np.ones((B, n_angles, 2)),
            C_init=np.ones((B, n_depth, 1)),
            t_init=np.array([10.0]), a_init=np.array([2.0]),
            s_init=np.ones((B, 1)),
            depth=np.arange(n_depth) * 5.0,
            free_idx=[0], linked_idx=[], sub_idx=[1],
            free_linked_const=np.ones(1),
            m0_C=np.ones((B, n_depth, 1)), m0_t=np.array([10.0]),
            config=cfg)
