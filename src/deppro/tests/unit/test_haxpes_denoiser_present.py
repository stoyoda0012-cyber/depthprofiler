"""With ``haxpes/denoiser.py`` present, ``deppro.haxpes`` exports the class,
and the class reaches the network through the installed ``dnndenoiser``
package — not through a sibling checkout on ``sys.path``.

Split from test_haxpes_optional_denoiser.py (same check) when the module was
still private; it is public now. The end-to-end checks need torch and
dnndenoiser; where they are not installed those tests are skipped, and a
skip is **not** a pass.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from .test_haxpes_optional_denoiser import _run


def test_with_the_module_present_the_class_is_exported():
    r = _run("import deppro.haxpes as h; print(h.HaxpesDenoiser.__name__)")
    assert r.returncode == 0 and r.stdout.strip() == "HaxpesDenoiser", r.stderr


def test_importing_the_module_touches_neither_sys_path_nor_torch():
    code = (
        "import sys\n"
        "before = list(sys.path)\n"
        "import deppro.haxpes.denoiser\n"
        "assert sys.path == before, 'sys.path changed'\n"
        "assert 'torch' not in sys.modules, 'torch imported at module import'\n"
        "assert 'dnndenoiser' not in sys.modules, 'dnndenoiser imported at module import'\n"
        "print('ok')"
    )
    r = _run(code)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_without_dnndenoiser_construction_names_the_package():
    # torch may be absent too (core-only install); stub it so the message
    # under test is the one about dnndenoiser, not the one about torch.
    code = (
        "import sys, types\n"
        "sys.modules.setdefault('torch', types.ModuleType('torch'))\n"
        "for m in ('dnndenoiser', 'dnndenoiser.models', 'dnndenoiser.models.network'):\n"
        "    sys.modules[m] = None\n"
        "from deppro.haxpes.denoiser import HaxpesDenoiser\n"
        "try:\n"
        "    HaxpesDenoiser('/nonexistent')\n"
        "except ImportError as e:\n"
        "    assert 'dnndenoiser' in str(e) and 'deppro[nn]' in str(e), e\n"
        "    print('ok')\n"
    )
    r = _run(code)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_without_torch_construction_names_torch_and_the_extra():
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from deppro.haxpes.denoiser import HaxpesDenoiser\n"
        "try:\n"
        "    HaxpesDenoiser('/nonexistent')\n"
        "except ImportError as e:\n"
        "    assert 'torch' in str(e) and 'deppro[nn]' in str(e), e\n"
        "    print('ok')\n"
    )
    r = _run(code)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


# ---------------------------------------------------------------------------
# end to end with the installed packages (random weights, no data)
# ---------------------------------------------------------------------------

def _network_or_skip():
    torch = pytest.importorskip("torch")
    pytest.importorskip("dnndenoiser")
    from dnndenoiser.models.network import DenoisingNetwork
    return torch, DenoisingNetwork


def test_the_network_is_the_papers_resnet_fcnn():
    """658,177 parameters: the count the deposit README and dnndenoiser's P1
    record both state for the papers' ResNet-FCNN at this shape."""
    _, DenoisingNetwork = _network_or_skip()
    from deppro.haxpes.denoiser import TARGET_LENGTH
    net = DenoisingNetwork(num_features=TARGET_LENGTH, num_hidden_units=100,
                           layer_type='ResNet-FCNN', encoder_output_dim=64)
    assert sum(p.numel() for p in net.parameters()) == 658_177


def test_bridge_loads_weights_and_denoises_with_the_recorded_normalisation(tmp_path):
    torch, DenoisingNetwork = _network_or_skip()
    from deppro.haxpes.denoiser import TARGET_LENGTH, HaxpesDenoiser

    torch.manual_seed(0)
    net = DenoisingNetwork(num_features=TARGET_LENGTH, num_hidden_units=100,
                           layer_type='ResNet-FCNN', encoder_output_dim=64)
    torch.save(net.state_dict(), tmp_path / "model_Si1s_fold0.pt")
    (tmp_path / "norm_Si1s.json").write_text(json.dumps({"global_min": 5.0, "global_max": 387.0}))

    den = HaxpesDenoiser(tmp_path, device="cpu", fold=0)
    rng = np.random.default_rng(1)
    spec = rng.poisson(200.0, size=(301, 8)).astype(np.float64)   # (n_energy, n_angles)
    out = den.denoise_spectrum(spec, "Si")
    assert out.shape == spec.shape and np.all(np.isfinite(out))

    # The bridge must apply exactly: resample -> (x - min)/(max - min) -> net
    # -> * (max - min) + min -> resample back. Reproduce that by hand.
    from deppro.haxpes.denoiser import _resample
    net.load_state_dict(torch.load(tmp_path / "model_Si1s_fold0.pt", weights_only=True))
    net.eval()
    x = _resample(spec.T, TARGET_LENGTH)
    xn = ((x - 5.0) / (387.0 - 5.0)).astype(np.float32)
    with torch.no_grad():
        yn = net(torch.tensor(xn))[0].numpy()
    expected = _resample(yn * (387.0 - 5.0) + 5.0, spec.shape[0]).T
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-4)

    # a 1-D spectrum round-trips its shape too
    assert den.denoise_spectrum(spec[:, 0], "Si").shape == (301,)


def test_denoise_pixel_passes_a_missing_element_through_and_warns(tmp_path, caplog):
    torch, DenoisingNetwork = _network_or_skip()
    from deppro.haxpes.denoiser import TARGET_LENGTH, HaxpesDenoiser

    net = DenoisingNetwork(num_features=TARGET_LENGTH, num_hidden_units=100,
                           layer_type='ResNet-FCNN', encoder_output_dim=64)
    torch.save(net.state_dict(), tmp_path / "model_Si1s_fold0.pt")
    (tmp_path / "norm_Si1s.json").write_text(json.dumps({"global_min": 0.0, "global_max": 1.0}))
    den = HaxpesDenoiser(tmp_path)
    pixel = {"Si": np.ones((50, 3)), "Ti": np.full((50, 3), 2.0)}
    with caplog.at_level("WARNING", logger="deppro.haxpes.denoiser"):
        out = den.denoise_pixel(pixel)
    assert set(out) == {"Si", "Ti"}
    assert out["Ti"] is pixel["Ti"]                 # untouched, same object
    assert any("Ti" in rec.message and "without denoising" in rec.message for rec in caplog.records)
