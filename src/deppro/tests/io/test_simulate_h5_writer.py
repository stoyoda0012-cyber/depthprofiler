"""``deppro.simulate.writer.H5Writer`` writes ``ezdeprof/element`` and
``ezdeprof/dxdydt``, so a file it produces (dialect D3 of
``docs/data_format.md``) is readable by ``deppro.visualize_4d.load_gt_4d``,
which reads both datasets without an existence check.

The loader import needs none of the optional rendering backends (pyvista,
imageio), so nothing here is skipped.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from deppro.io.h5_writer import format_element_with_charge
from deppro.simulate import CHECKER_7ELEM, LAYERED_6ELEM, H5Writer
from deppro.simulate.writer import _element_label
from deppro.visualize_4d import _STANDARD_ELEMENT_COLORS, load_gt_4d

ANGLES = np.array([10.0, 30.0, 50.0])
DEPTH = np.arange(0.0, 50.0, 5.0)


def _write(path, preset, n_px, rng, **grid):
    n_elem = preset.n_elements
    D = rng.random((n_px, len(DEPTH), n_elem))
    X = rng.random((n_px, len(ANGLES), n_elem))
    w = H5Writer(path, preset)
    w.add_batch(D[: n_px // 2], X[: n_px // 2])
    w.add_batch(D[n_px // 2:], X[n_px // 2:])
    w.finalize(depth=DEPTH, angles=ANGLES, step_size=5.0, photon_energy=1486.6, **grid)
    return D


def test_element_labels_follow_the_matlab_charge_convention(tmp_path):
    p = tmp_path / "layered.h5"
    _write(p, LAYERED_6ELEM, 4, np.random.default_rng(0))
    with h5py.File(p, "r") as f:
        raw = f["ezdeprof/element"][:].flatten()
        labels = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
    assert labels == ["C(0)", "Al(+3)", "Ti(+4)", "Si(+4)", "O(-2)", "Si(0)"]
    # every label is one the renderer's standard palette knows
    assert all(name in _STANDARD_ELEMENT_COLORS for name in labels)


def test_label_helper_matches_the_legacy_io_writer():
    for sym, c in [("Si", 0), ("Si", 4), ("O", -2), ("Al", 3.0)]:
        assert _element_label(sym, c) == format_element_with_charge(sym, c)


def test_dxdydt_records_the_given_grid(tmp_path):
    p = tmp_path / "grid.h5"
    _write(p, CHECKER_7ELEM, 2 * 3 * 4, np.random.default_rng(1), nx=2, ny=3)
    with h5py.File(p, "r") as f:
        d = f["ezdeprof/dxdydt"]
        assert d.shape == (1, 3) and d.dtype == np.float64
        assert d[:].flatten().tolist() == [2.0, 3.0, 4.0]


def test_dxdydt_defaults_to_one_row_of_samples(tmp_path):
    p = tmp_path / "row.h5"
    _write(p, CHECKER_7ELEM, 6, np.random.default_rng(2))
    with h5py.File(p, "r") as f:
        assert f["ezdeprof/dxdydt"][:].flatten().tolist() == [6.0, 1.0, 1.0]


@pytest.mark.parametrize("grid", [dict(nx=4, ny=4), dict(nx=3), dict(nx=0, ny=2)])
def test_a_grid_that_does_not_fit_is_refused_before_writing(tmp_path, grid):
    p = tmp_path / "bad.h5"
    with pytest.raises(ValueError):
        _write(p, CHECKER_7ELEM, 6, np.random.default_rng(3), **grid)
    assert not p.exists()


def test_visualize_4d_loader_reads_the_file_and_recovers_the_pixels(tmp_path):
    nx, ny, nt = 3, 2, 2
    p = tmp_path / "gt.h5"
    D = _write(p, CHECKER_7ELEM, nx * ny * nt, np.random.default_rng(4), nx=nx, ny=ny)

    data = load_gt_4d(str(p))

    assert (data.nx, data.ny, data.nt) == (nx, ny, nt)
    assert data.dxdydt == (nx, ny, nt)
    assert data.element_names == ["Si(0)", "Si(+4)", "Ti(+4)", "Al(+3)", "C(0)", "B(0)", "O(-2)"]
    assert data.data.shape == (nt, nx, ny, len(DEPTH), CHECKER_7ELEM.n_elements)
    assert data.step_size == 5.0
    np.testing.assert_allclose(data.depth, DEPTH)
    # pixel (ix, iy) of frame t is sample t*nx*ny + ix + iy*nx (MATLAB column-major),
    # stored as float32 by the writer
    for t in range(nt):
        for ix in range(nx):
            for iy in range(ny):
                px = t * nx * ny + ix + iy * nx
                np.testing.assert_allclose(data.data[t, ix, iy], D[px], rtol=1e-6)
