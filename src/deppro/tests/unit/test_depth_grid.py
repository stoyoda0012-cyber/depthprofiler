"""Depth grid builders, the make_grid dispatcher and register_grid_type."""
import numpy as np
import pytest

from deppro import depth_grid
from deppro.depth_grid import (
    GRID_TYPES, make_grid, make_uniform_grid, register_grid_type,
)


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(depth_grid, 'GRID_TYPES', dict(depth_grid.GRID_TYPES))
    monkeypatch.setattr(depth_grid, '_REGISTERED_BUILDERS',
                        dict(depth_grid._REGISTERED_BUILDERS))


def test_builtin_types_dispatch():
    np.testing.assert_array_equal(make_grid('uniform', 5, 100.0),
                                  make_uniform_grid(5, 100.0))
    g = make_grid('geometric', 6, 100.0, ratio=1.5)
    assert g.shape == (6,) and np.all(np.diff(g) > 0) and g[-1] == pytest.approx(100.0)
    g = make_grid('log', 6, 100.0, z_min=1.0)
    assert g[0] == pytest.approx(1.0) and g[-1] == pytest.approx(100.0)
    for name in ('uniform', 'geometric', 'log'):
        assert name in GRID_TYPES


def test_unknown_type_raises():
    with pytest.raises(ValueError, match='Unknown grid type'):
        make_grid('no-such-grid', 5, 100.0)


def test_register_grid_type_dispatches_with_arguments(clean_registry):
    seen = {}

    def builder(n_depth, z_max, lambda_eff=None, angles_deg=None, **kwargs):
        seen.update(n_depth=n_depth, z_max=z_max, lambda_eff=lambda_eff,
                    angles_deg=angles_deg, kwargs=kwargs)
        return np.linspace(1.0, z_max, n_depth)

    register_grid_type('custom-test', 'test grid', builder)
    assert depth_grid.GRID_TYPES['custom-test'] == 'test grid'
    g = depth_grid.make_grid('custom-test', 4, 80.0, lambda_eff=30.0,
                             angles_deg=np.array([0.0, 45.0]), extra=1)
    np.testing.assert_array_equal(g, np.linspace(1.0, 80.0, 4))
    assert seen['n_depth'] == 4 and seen['z_max'] == 80.0
    assert seen['lambda_eff'] == 30.0
    np.testing.assert_array_equal(seen['angles_deg'], [0.0, 45.0])
    assert seen['kwargs'] == {'extra': 1}


def test_builtin_types_cannot_be_replaced(clean_registry):
    for name in ('uniform', 'geometric', 'log'):
        with pytest.raises(ValueError, match='built in'):
            register_grid_type(name, 'x', lambda *a, **k: None)
