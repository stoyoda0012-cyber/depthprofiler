"""Tests for pipeline_bridge: 橋B (read_ezdeprof) + 橋C (batch_solve) + run_pipeline."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from deppro.app import DEPPRO_AVAILABLE
from deppro.app.pipeline_bridge import batch_solve, read_ezdeprof, run_pipeline

# ============================================================================
# Fixtures / Helpers
# ============================================================================


def _make_ezdeprof_h5(
    path: str,
    *,
    n_px: int = 10,
    n_depth: int = 50,
    n_angles: int = 7,
    n_elem: int = 3,
    step_size: float = 4.0,
    include_transmat: bool = True,
) -> str:
    """Create a Dstk H5 with /ezdeprof/ and /Xstk for testing."""
    depth = np.arange(n_depth, dtype=np.float64) * step_size
    angles = np.linspace(0, 65, n_angles, dtype=np.float64)

    symbols = ["Si", "O", "C"][:n_elem]
    orbitals = ["2p", "1s", "1s"][:n_elem]
    compounds = ["Si", "SiO2", "C"][:n_elem]
    element_names = ["Si2p", "O1s", "C1s"][:n_elem]
    binding_energies = np.array([99.5, 532.0, 285.0], dtype=np.float64)[:n_elem]
    charges = np.array([0.0, -2.0, 0.0], dtype=np.float64)[:n_elem]

    # free/subst/linked (MATLAB 1-based)
    free_matlab = np.array([1], dtype=np.float64)       # Si → 0
    subst_matlab = np.array([3], dtype=np.float64)[:min(1, n_elem)]  # C → 2
    linked_matlab = np.array([2], dtype=np.float64)     # O → 1
    linkedconst = np.array([2.0], dtype=np.float64)

    dt = h5py.string_dtype()

    with h5py.File(path, "w") as f:
        ez = f.create_group("ezdeprof")
        ez.create_dataset("depth", data=depth.reshape(-1, 1))
        ez.create_dataset("stepsize", data=np.array([[step_size]]))
        ez.create_dataset("thetaobs", data=angles.reshape(1, -1))
        ez.create_dataset("photonenergy", data=np.array([[1486.6]]))
        ez.create_dataset("workfunction", data=np.array([[4.5]]))
        ez.create_dataset("gammaeff", data=np.array([[1.0]]))

        ez.create_dataset("element", data=element_names, dtype=dt)
        ez.create_dataset("elementsymbol", data=symbols, dtype=dt)
        ez.create_dataset("orbital", data=orbitals, dtype=dt)
        ez.create_dataset("compound", data=compounds, dtype=dt)
        ez.create_dataset("bindingenergy", data=binding_energies)
        ez.create_dataset("charge", data=charges)

        ez.create_dataset("free", data=free_matlab)
        ez.create_dataset("subst", data=subst_matlab)
        ez.create_dataset("linked", data=linked_matlab)
        ez.create_dataset("linkedconst", data=linkedconst)

        ez.create_dataset("dxdydt", data=np.array([[5, 2, 1]], dtype=np.float64))

        if include_transmat:
            # MATLAB layout: (n_elem, n_depth, n_angles)
            tm_matlab = np.random.RandomState(42).rand(
                n_elem, n_depth, n_angles
            ).astype(np.float64)
            ez.create_dataset("TransMatObs", data=tm_matlab)

        # Xstk: MATLAB layout (n_px, n_elem, n_angles)
        xstk = np.random.RandomState(99).rand(
            n_px, n_elem, n_angles
        ).astype(np.float32)
        f.create_dataset("Xstk", data=xstk, compression="gzip")

        # Dstk: MATLAB layout (n_px, n_elem, n_depth)
        dstk = np.random.RandomState(7).rand(
            n_px, n_elem, n_depth
        ).astype(np.float32)
        f.create_dataset("Dstk", data=dstk, compression="gzip")

    return path


_ELEMENT_BE = {"Si2p": 99.5, "O1s": 532.0, "C1s": 285.0}


def _make_fitpara_h5_for_pipeline(
    path: str,
    *,
    n_angles: int = 7,
    n_y: int = 2,
    n_x: int = 5,
    n_comp: int = 1,
    element: str = "Si2p",
) -> str:
    """Create a minimal fitpara H5 for pipeline integration tests."""
    from toyomacro.io.schema import ToyomacroSchema

    n_spectra = n_x * n_y * n_angles
    fitpara = np.zeros((n_spectra, n_comp, 9), dtype=np.float32)
    FP = ToyomacroSchema.FitparaIndex
    center_be = _ELEMENT_BE.get(element, 99.5)

    for i in range(n_spectra):
        angle_idx = i % n_angles
        fitpara[i, 0, int(FP.INTEGRATED_AREA)] = 100.0 * (1 + angle_idx / 10.0)
        fitpara[i, 0, int(FP.CENTER)] = center_be

    xytdata = np.zeros((3, n_spectra), dtype=np.float32)
    angles_deg = np.linspace(0, 60, n_angles)
    for i in range(n_spectra):
        xytdata[0, i] = angles_deg[i % n_angles]

    with h5py.File(path, "w") as f:
        f.create_dataset("fitpara", data=fitpara)
        f.create_dataset("xytdata", data=xytdata)
        misc = f.create_group("misc")
        misc.create_dataset("dim_shape", data=np.array([n_angles, n_y, 1]))
        dt = h5py.string_dtype()
        misc.create_dataset("xdata_str", data=element, dtype=dt)

    return path


# ============================================================================
# 橋B: read_ezdeprof (h5py-only, no deppro needed)
# ============================================================================


class TestReadEzdeprof:
    """Tests for read_ezdeprof — runs without deppro."""

    def test_basic_shape(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"))
        result = read_ezdeprof(h5)

        assert "tm_params" in result
        assert "trans_mat" in result
        assert "elements" in result

        tm_p = result["tm_params"]
        assert tm_p["depth"].shape == (50,)
        assert tm_p["angles"].shape == (7,)
        assert tm_p["step_size"] == 4.0
        assert tm_p["photon_energy"] == 1486.6
        assert tm_p["work_function"] == 4.5
        assert tm_p["gamma_eff"] == 1.0

    def test_trans_mat_shape(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"))
        result = read_ezdeprof(h5)

        tm = result["trans_mat"]
        assert tm is not None
        # (n_angles, n_depth, n_elem)
        assert tm.shape == (7, 50, 3)
        assert tm.dtype == np.float64

    def test_trans_mat_none_when_missing(self, tmp_path):
        h5 = _make_ezdeprof_h5(
            str(tmp_path / "test.h5"), include_transmat=False
        )
        result = read_ezdeprof(h5)
        assert result["trans_mat"] is None

    def test_element_names(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"))
        result = read_ezdeprof(h5)

        assert result["element_names"] == ["Si2p", "O1s", "C1s"]

    def test_elements_list(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"))
        result = read_ezdeprof(h5)

        elems = result["elements"]
        assert len(elems) == 3
        # Check first element (works for both ElementConfig and dict fallback)
        e0 = elems[0]
        if isinstance(e0, dict):
            assert e0["symbol"] == "Si"
            assert e0["orbital"] == "2p"
        else:
            assert e0.symbol == "Si"
            assert e0.orbital == "2p"

    def test_index_conversion(self, tmp_path):
        """MATLAB 1-based → Python 0-based."""
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"))
        result = read_ezdeprof(h5)

        assert result["free_indices"] == [0]       # MATLAB 1 → Python 0
        assert result["linked_indices"] == [1]     # MATLAB 2 → Python 1
        assert result["substrate_indices"] == [2]  # MATLAB 3 → Python 2

    def test_charges(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"))
        result = read_ezdeprof(h5)

        np.testing.assert_array_equal(result["charges"], [0.0, -2.0, 0.0])

    def test_linkedconst(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"))
        result = read_ezdeprof(h5)

        np.testing.assert_array_equal(result["linkedconst"], [2.0])

    def test_depth_axis(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"), step_size=2.0, n_depth=25)
        result = read_ezdeprof(h5)

        expected = np.arange(25) * 2.0
        np.testing.assert_allclose(result["depth_axis"], expected)

    def test_default_physics_params(self, tmp_path):
        """Test default values when optional scalars are missing."""
        h5_path = str(tmp_path / "minimal.h5")

        with h5py.File(h5_path, "w") as f:
            ez = f.create_group("ezdeprof")
            ez.create_dataset("depth", data=np.arange(10, dtype=np.float64).reshape(-1, 1))
            ez.create_dataset("stepsize", data=np.array([[4.0]]))
            ez.create_dataset("thetaobs", data=np.linspace(0, 65, 5).reshape(1, -1))
            # NO photonenergy, workfunction, gammaeff

            dt = h5py.string_dtype()
            ez.create_dataset("element", data=["Si2p"], dtype=dt)
            ez.create_dataset("elementsymbol", data=["Si"], dtype=dt)
            ez.create_dataset("orbital", data=["2p"], dtype=dt)
            ez.create_dataset("compound", data=["Si"], dtype=dt)
            ez.create_dataset("bindingenergy", data=np.array([99.5]))
            ez.create_dataset("charge", data=np.array([0.0]))

        result = read_ezdeprof(h5_path)
        assert result["tm_params"]["photon_energy"] == 1486.6
        assert result["tm_params"]["work_function"] == 4.5
        assert result["tm_params"]["gamma_eff"] == 1.0


# ============================================================================
# 橋C: batch_solve (requires deppro)
# ============================================================================


@pytest.mark.skipif(not DEPPRO_AVAILABLE, reason="deppro not available")
class TestBatchSolveFree:
    """Tests for batch_solve with solver='free'."""

    def test_shape(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"), n_px=5, n_depth=20, n_angles=7, n_elem=3)
        ezdeprof = read_ezdeprof(h5)

        I_elem = np.random.RandomState(0).rand(5, 7, 3).astype(np.float32)
        depth_map = batch_solve(I_elem, ezdeprof, solver="free", method="LLS")

        assert depth_map.shape == (5, 20, 3)
        assert depth_map.dtype == np.float32

    def test_nonnegative(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"), n_px=3, n_depth=20, n_angles=7, n_elem=3)
        ezdeprof = read_ezdeprof(h5)

        I_elem = np.abs(np.random.RandomState(1).rand(3, 7, 3).astype(np.float32))
        depth_map = batch_solve(I_elem, ezdeprof, solver="free", method="LLS")

        assert np.all(depth_map >= -1e-6), "Depth map should be non-negative"


@pytest.mark.skipif(not DEPPRO_AVAILABLE, reason="deppro not available")
class TestBatchSolveSemiFree:
    """Tests for batch_solve with solver='semifree'.

    Note: SemiFreeSolver requires physically realistic data (proper
    TransMatObs, consistent element classification).  Random test data
    may trigger internal index errors in the SLSQP objective.
    """

    def test_shape(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"), n_px=2, n_depth=20, n_angles=7, n_elem=3)
        ezdeprof = read_ezdeprof(h5)

        I_elem = np.random.RandomState(0).rand(2, 7, 3).astype(np.float32)
        depth_map = batch_solve(I_elem, ezdeprof, solver="semifree", method="L2")

        assert depth_map.shape == (2, 20, 3)
        assert depth_map.dtype == np.float32


# ============================================================================
# NN / Hybrid (skip if no model)
# ============================================================================


@pytest.mark.skipif(not DEPPRO_AVAILABLE, reason="deppro not available")
class TestBatchSolveNN:
    """NN solver requires a model file — skip if not available."""

    def test_no_model_raises(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"))
        ezdeprof = read_ezdeprof(h5)
        I_elem = np.random.RandomState(0).rand(5, 7, 3).astype(np.float32)

        with pytest.raises(ValueError, match="nn_model_path required"):
            batch_solve(I_elem, ezdeprof, solver="nn")

    def test_hybrid_no_model_raises(self, tmp_path):
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"))
        ezdeprof = read_ezdeprof(h5)
        I_elem = np.random.RandomState(0).rand(5, 7, 3).astype(np.float32)

        with pytest.raises(ValueError, match="nn_model_path required"):
            batch_solve(I_elem, ezdeprof, solver="hybrid")


# ============================================================================
# run_pipeline end-to-end
# ============================================================================


@pytest.mark.skipif(not DEPPRO_AVAILABLE, reason="deppro not available")
class TestRunPipeline:
    """Integration test: fitpara → run_pipeline → output H5."""

    def test_roundtrip(self, tmp_path):
        # Create 3 fitpara files (one per element)
        n_angles = 7
        n_x, n_y = 5, 2
        n_px = n_x * n_y
        n_elem = 3

        fitpara_paths = []
        for elem in ["Si2p", "O1s", "C1s"]:
            fp_path = str(tmp_path / f"{elem}.h5")
            _make_fitpara_h5_for_pipeline(
                fp_path, n_angles=n_angles, n_x=n_x, n_y=n_y, element=elem
            )
            fitpara_paths.append(fp_path)

        # Create Dstk H5 with matching geometry
        dstk_path = _make_ezdeprof_h5(
            str(tmp_path / "dstk.h5"),
            n_px=n_px, n_depth=20, n_angles=n_angles, n_elem=n_elem,
        )

        # Run pipeline
        output = run_pipeline(
            fitpara_paths, dstk_path,
            solver="free", method="LLS",
            output_dir=str(tmp_path / "output"),
        )

        assert Path(output).exists()
        assert "depth_est_FREE_" in Path(output).name

        with h5py.File(output, "r") as f:
            # D_est structure (estimate, not GT)
            assert "D_est" in f
            assert "D_est/data" in f
            assert "D_est/solver" in f
            assert "D_est/timestamp" in f
            assert "Xstk" in f
            assert "ezdeprof" in f
            # No GT datasets
            assert "Dstk" not in f
            assert "Py_Drlt" not in f

            # D_est/data: MATLAB layout (n_px, n_elem, n_depth)
            d_est = f["D_est/data"][:]
            assert d_est.shape == (n_px, n_elem, 20)


# ============================================================================
# Error handling
# ============================================================================


class TestErrorHandling:
    """Edge cases and error paths."""

    def test_unknown_solver(self, tmp_path):
        if not DEPPRO_AVAILABLE:
            pytest.skip("deppro not available")

        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"))
        ezdeprof = read_ezdeprof(h5)
        I_elem = np.random.RandomState(0).rand(5, 7, 3).astype(np.float32)

        with pytest.raises(ValueError, match="Unknown solver"):
            batch_solve(I_elem, ezdeprof, solver="bogus")

    def test_batch_solve_no_deppro(self, monkeypatch, tmp_path):
        """batch_solve raises ImportError when deppro unavailable."""
        h5 = _make_ezdeprof_h5(str(tmp_path / "test.h5"))
        ezdeprof = read_ezdeprof(h5)
        I_elem = np.random.RandomState(0).rand(5, 7, 3).astype(np.float32)

        import deppro.app as depth_mod
        import deppro.app.pipeline_bridge as pb

        original = depth_mod.DEPPRO_AVAILABLE
        monkeypatch.setattr(depth_mod, "DEPPRO_AVAILABLE", False)
        try:
            with pytest.raises(ImportError, match="deppro package required"):
                batch_solve(I_elem, ezdeprof, solver="free")
        finally:
            monkeypatch.setattr(depth_mod, "DEPPRO_AVAILABLE", original)
