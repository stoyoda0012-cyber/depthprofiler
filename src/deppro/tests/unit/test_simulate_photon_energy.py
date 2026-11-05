"""``deppro.simulate.run_simulation`` refuses a photon energy that leaves an
orbital with a negative kinetic energy, instead of failing deep in the
compiled kernel.

Before this check, the default ``SimulationConfig`` (1486.6 eV) with a
1s-orbital preset (minimum 5100 eV) produced NaN IMFPs from TPP-2M, and the
fused Numba kernel ended in a bare ``ZeroDivisionError`` with no hint of
the cause. Two layers now refuse early and by name: ``run_simulation``
compares the photon energy with ``preset.min_photon_energy``, and
``RandomProfileSampler.generate_batch`` rejects a ``lambda_basic`` with
non-finite or non-positive entries.
"""
from __future__ import annotations

import numpy as np
import pytest

from deppro.simulate import LAYERED_6ELEM, SimulationConfig, run_simulation
from deppro.simulate.sampler import RandomProfileSampler

ANGLES = np.array([0.0, 30.0, 60.0])


def test_run_simulation_refuses_a_photon_energy_below_the_preset_minimum(tmp_path):
    cfg = SimulationConfig(n_samples=4, output_path=str(tmp_path / "x.h5"),
                           output_format="h5", seed=0)          # default 1486.6 eV
    assert cfg.photon_energy < LAYERED_6ELEM.min_photon_energy  # the premise of this test
    with pytest.raises(ValueError, match="min") as ei:
        run_simulation(cfg, LAYERED_6ELEM, verbose=False)
    msg = str(ei.value)
    assert "1486.6" in msg and "5100" in msg and LAYERED_6ELEM.name in msg
    assert not (tmp_path / "x.h5").exists()                     # refused before any work


@pytest.mark.parametrize("bad_value", [np.nan, 0.0, -1.0, np.inf])
def test_generate_batch_refuses_a_bad_imfp_matrix_by_element(bad_value):
    depth = np.arange(0.0, 100.0, 5.0)
    sampler = RandomProfileSampler(LAYERED_6ELEM, depth, seed=0)
    n = LAYERED_6ELEM.n_elements
    L = np.full((n, n), 25.0)
    L[2, :] = bad_value                                          # row 2 = Ti 1s (emitting element)
    with pytest.raises(ValueError, match="Ti 1s"):
        sampler.generate_batch(2, L, ANGLES, 5.0)


def test_generate_batch_refuses_a_wrongly_shaped_imfp_matrix():
    depth = np.arange(0.0, 100.0, 5.0)
    sampler = RandomProfileSampler(LAYERED_6ELEM, depth, seed=0)
    with pytest.raises(ValueError, match="shape"):
        sampler.generate_batch(2, np.full((2, 2), 25.0), ANGLES, 5.0)


def test_run_simulation_writes_finite_arrays_above_the_minimum(tmp_path):
    """The same call with a HAXPES photon energy runs to the end and writes finite data."""
    h5py = pytest.importorskip("h5py")
    cfg = SimulationConfig(n_samples=6, output_path=str(tmp_path / "ok.h5"),
                           output_format="h5", seed=0, photon_energy=8000.0)
    run_simulation(cfg, LAYERED_6ELEM, verbose=False)
    with h5py.File(tmp_path / "ok.h5", "r") as f:
        D, X = f["Dstk"][:], f["Xstk"][:]
    assert D.shape[0] == 6 and X.shape[0] == 6
    assert np.isfinite(D).all() and np.isfinite(X).all()
    assert D.min() >= 0.0
