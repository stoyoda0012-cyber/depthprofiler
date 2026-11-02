"""register_architecture: create_model and load_model for added architectures."""
import numpy as np
import pytest

pytest.importorskip('mlx.core')

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from deppro.inference import model as model_mod  # noqa: E402
from deppro.inference.model import create_model, register_architecture  # noqa: E402
from deppro.inference.train import load_model  # noqa: E402


class _TinyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.hidden_dim, self.n_hidden = hidden_dim, n_hidden
        self.zz = [nn.Linear(input_dim, hidden_dim),
                   nn.Linear(hidden_dim, output_dim)]

    def __call__(self, x):
        return self.zz[1](nn.relu(self.zz[0](x)))


def _dims(w):
    return (w['zz.0.weight'].shape[1], w['zz.1.weight'].shape[0],
            w['zz.0.weight'].shape[0], 1)


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(model_mod, 'MODEL_REGISTRY', dict(model_mod.MODEL_REGISTRY))
    monkeypatch.setattr(model_mod, '_REGISTERED_LOADERS',
                        list(model_mod._REGISTERED_LOADERS))


def test_registered_architecture_creates_and_loads(clean_registry, tmp_path):
    register_architecture('tiny-test', _TinyNet, 'zz.', _dims)
    m = model_mod.create_model('tiny-test', input_dim=6, output_dim=3,
                               hidden_dim=8, n_hidden=1)
    assert isinstance(m, _TinyNet)
    path = tmp_path / 'tiny.npz'
    m.save_weights(str(path))
    loaded = load_model(str(path))
    assert isinstance(loaded, _TinyNet)
    assert (loaded.input_dim, loaded.output_dim, loaded.hidden_dim) == (6, 3, 8)
    x = mx.array(np.random.default_rng(0).random((4, 6)).astype(np.float32))
    np.testing.assert_array_equal(np.array(m(x)), np.array(loaded(x)))


def test_builtin_architectures_cannot_be_replaced(clean_registry):
    for name in ('mlp', 'resmlp', 'conv1d', 'elemnet', 'transformer'):
        with pytest.raises(ValueError, match='built in'):
            register_architecture(name, _TinyNet, 'zz.', _dims)


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match='Unknown arch'):
        create_model('no-such-arch')
