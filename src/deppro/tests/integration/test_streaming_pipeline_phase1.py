"""End-to-end Phase 1 of the streaming per-pixel pipeline on a tiny synthetic map.

Regression test for the 8-vs-9 tuple mismatch between
``streaming_pipeline.run_phase1`` and ``solver_workers.worker_phase1`` that
made ``python -m deppro.streaming_pipeline`` fail on its first pixel
(ValueError: too many values to unpack) until 2026-11-07.
"""
import numpy as np
import pytest
from scipy.special import erf

h5py = pytest.importorskip("h5py")

from deppro.simulate import H5Writer, get_preset
from deppro.streaming_pipeline import StreamingDepthPipeline, StreamingPipelineConfig
from deppro.transfer_matrix import TransferMatrix


def _write_map(path):
    preset = get_preset('sio2_si')          # O(-2), Si(+4), Si(0)
    depth = np.arange(1, 13) * 50.0          # 50 ... 600 A
    angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
    tm = TransferMatrix(depth=depth, step_size=50.0, angles=angles, photon_energy=1486.6)
    tm.compute(np.full((depth.size, 3), 1 / 3), preset.elements)

    thicknesses = [150.0, 200.0, 250.0, 300.0]   # one SiO2 film per pixel, 2 x 2 map
    sigma = 20.0 / (2 * np.sqrt(np.log(2)))
    D = np.zeros((4, depth.size, 3))
    X = np.zeros((4, angles.size, 3))
    for i, t in enumerate(thicknesses):
        sub = 0.5 * (1 + erf((depth - t) / sigma))
        film = 1.0 - sub
        D[i, :, 0] = 2 * film / 3       # O
        D[i, :, 1] = film / 3           # Si(+4)
        D[i, :, 2] = sub                # Si(0)
        X[i] = tm.forward(D[i], normalize=False)

    writer = H5Writer(str(path), preset)
    writer.add_batch(D, X)
    writer.finalize(depth=depth, angles=angles, step_size=50.0, photon_energy=1486.6,
                    trans_mat=tm.trans_mat, nx=2, ny=2)
    return thicknesses


def test_phase1_completes_and_writes_dtrn(tmp_path):
    h5_in = tmp_path / 'map.h5'
    h5_out = tmp_path / 'map_streaming.h5'
    thicknesses = _write_map(h5_in)

    config = StreamingPipelineConfig(
        method='LLS', n_loops=1, n_workers=1, nx=2, ny=2,
        thickness_init=210.0, abruptness_init=20.0,
        solver_max_iter=50, chunk_pixels=4,
        compression=None, full_gui_compat=False,
    )
    StreamingDepthPipeline(str(h5_in), str(h5_out), config).run()

    with h5py.File(h5_out, 'r') as f:
        Dtrn = f['Dtrn'][:]          # (n_pixels, n_elements, n_depth)
        Qtrn = f['Qtrn'][:]          # (3, n_pixels): chi2, thickness, abruptness
    assert Dtrn.shape == (4, 3, 12)
    assert np.all(np.isfinite(Dtrn))
    # Every pixel was solved: a failed pixel is written as all zeros.
    assert np.all(Dtrn.sum(axis=(1, 2)) > 0)
    thk = Qtrn[1] if Qtrn.shape[0] == 3 else Qtrn[:, 1]
    assert np.all((thk >= 50.0) & (thk <= 500.0))
    # The fitted film thicknesses follow the true ordering of the four pixels.
    assert np.all(np.diff(thk[np.argsort(thicknesses)]) > 0)
