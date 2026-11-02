"""Calls that referred to the removed MultiLayerFilmSolver classes.

``LayerModel.to_solver_config()`` returned a ``MultiLayerFilmConfig`` for the
removed solver; it now raises ``NotImplementedError`` naming it.
``save_result_json()`` only imported the removed ``MultiLayerResult`` for its
annotation; without that import it writes a ``GenericLayerResult``, and a
result without the layer fields stops with a ``TypeError`` naming them.
The usage shown in ``examples/custom_structure.yaml`` is checked as well.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deppro.data_io import save_result_json
from deppro.inverse_solver import GenericLayerResult
from deppro.layer_model import LayerModel

# relative to this file (tests/unit/ → src/deppro/examples/): the examples are
# not packaged, so they are taken from the source tree the tests come from
_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "custom_structure.yaml"


def test_to_solver_config_raises_with_the_reason():
    model = LayerModel.from_yaml(_EXAMPLE)
    with pytest.raises(NotImplementedError, match="MultiLayerFilmSolver"):
        model.to_solver_config()


def test_save_result_json_writes_a_generic_layer_result(tmp_path):
    depth = np.linspace(0.0, 20.0, 5)
    D = np.full((5, 2), 0.5)
    result = GenericLayerResult(D_opt=D, I_opt=np.ones((3, 2)), layer_thicknesses=np.array([10.0]),
                                interface_widths=np.array([2.0]), chi2=0.06, objective=0.06,
                                n_iterations=7, elapsed_time=0.5, success=True)
    save_result_json(result, depth, ["Si", "O"], tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text())
    assert set(data) == {"depth", "concentrations", "chi2", "rmse", "layer_thicknesses",
                         "interface_widths", "n_iterations", "elapsed_time", "success"}
    assert data["concentrations"]["O"] == D[:, 1].tolist()
    assert data["layer_thicknesses"] == [10.0] and data["n_iterations"] == 7
    assert data["rmse"] == pytest.approx(np.sqrt(0.06 / 6))


def test_save_result_json_names_missing_fields(tmp_path):
    class NoLayers:  # no layer fields and no rmse (SemiFreeSolverResult lacks the layer fields)
        D_opt = np.zeros((3, 1)); chi2 = 0.0; n_iterations = 1; elapsed_time = 0.0; success = True

    with pytest.raises(TypeError, match="rmse, layer_thicknesses, interface_widths"):
        save_result_json(NoLayers(), np.zeros(3), ["Si"], tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_example_yaml_usage_builds_a_profile():
    model = LayerModel.from_yaml(_EXAMPLE)
    depth = np.arange(model.depth_start, model.depth_end + model.depth_step, model.depth_step)
    D = model.build_depth_profile(np.array(model.layer_thicknesses),
                                  np.array(model.interface_widths), depth)
    assert D.shape == (len(depth), model.n_elements)
    np.testing.assert_allclose(D.sum(axis=1), 1.0, atol=1e-9)
    assert "to_solver_config" not in _EXAMPLE.read_text(encoding="utf-8")
