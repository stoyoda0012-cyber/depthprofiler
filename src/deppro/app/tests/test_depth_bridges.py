"""Tests for depth pipeline bridges: fitpara_reader (橋A) and depth_result_writer (橋D)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from deppro.app.depth_result_writer import write_depth_estimate

# 橋A は兄弟パッケージ toyomacro（optional extra ``fit``）を必要とする。無い環境では
# 収集エラーでなく skip にする（skip を「通った」と読まないこと）。
ToyomacroSchema = pytest.importorskip("toyomacro.io.schema").ToyomacroSchema
from deppro.app.fitpara_reader import read_fitpara_areas  # noqa: E402

# ============================================================================
# Fixtures / Helpers
# ============================================================================

FP = ToyomacroSchema.FitparaIndex


def _make_fitpara_h5(
    path: str,
    *,
    n_x: int = 5,
    n_y: int = 4,
    n_angles: int = 7,
    n_comp: int = 2,
    element: str = "Si2p",
    area_value: float = 100.0,
) -> str:
    """Create a minimal Toyomacro fitpara H5 with known area values.

    Index layout: idx = angle + n_angles * y + n_angles * n_y * x
    """
    n_spectra = n_x * n_y * n_angles
    fitpara = np.zeros((n_spectra, n_comp, 9), dtype=np.float32)

    # Fill area column with predictable pattern: area = area_value * (1 + angle/10)
    for i in range(n_spectra):
        angle_idx = i % n_angles
        for c in range(n_comp):
            fitpara[i, c, int(FP.INTEGRATED_AREA)] = area_value * (1 + angle_idx / 10.0) / n_comp
            fitpara[i, c, int(FP.CENTER)] = 99.5 + c * 0.5
            fitpara[i, c, int(FP.FWHM_GAUSSIAN)] = 0.8

    # xytdata
    xytdata = np.zeros((3, n_spectra), dtype=np.float32)
    angles_deg = np.linspace(0, 60, n_angles)
    for i in range(n_spectra):
        angle_idx = i % n_angles
        remainder = i // n_angles
        y = remainder % n_y
        x = remainder // n_y
        xytdata[0, i] = angles_deg[angle_idx]
        xytdata[1, i] = y
        xytdata[2, i] = 0

    with h5py.File(path, "w") as f:
        f.create_dataset("fitpara", data=fitpara)
        f.create_dataset("xytdata", data=xytdata)
        f.create_dataset("otherpara", data=np.zeros((10, n_spectra), dtype=np.float32))

        # Minimal specdata (energy axis in row 0)
        energy = np.linspace(95.0, 105.0, 256)
        specdata = np.zeros((n_spectra + 1, 256), dtype=np.float32)
        specdata[0, :] = energy
        f.create_dataset("specdata", data=specdata)

        # misc metadata
        misc = f.create_group("misc")
        misc.create_dataset("dim_shape", data=np.array([n_angles, n_y, 1], dtype=np.int32))
        misc.create_dataset("dim0_is_angle", data=np.array([1], dtype=np.int32))
        misc.create_dataset("maxcomp", data=np.array([n_comp], dtype=np.int32))
        dt = h5py.string_dtype()
        misc.create_dataset("xdata_str", data=element, dtype=dt)

    return path


def _make_dstk_h5(
    path: str,
    *,
    n_px: int = 20,
    n_elem: int = 3,
    n_angles: int = 7,
    n_depth: int = 25,
    n_x: int = 5,
    n_y: int = 4,
) -> str:
    """Create a minimal Dstk-format H5 with /Xstk."""
    rng = np.random.RandomState(42)

    # Xstk: MATLAB layout (n_px, n_elem, n_angles)
    xstk = rng.rand(n_px, n_elem, n_angles).astype(np.float32)
    # Dstk: MATLAB layout (n_px, n_elem, n_depth)
    dstk = rng.rand(n_px, n_elem, n_depth).astype(np.float32)

    elements = ["Si(0)", "Si(+4)", "O(-2)"][:n_elem]
    angles = np.linspace(0, 60, n_angles)
    depth = np.arange(n_depth, dtype=np.float64) * 4.0

    with h5py.File(path, "w") as f:
        f.create_dataset("Xstk", data=xstk)
        f.create_dataset("Dstk", data=dstk)
        f.create_dataset("thetaobs", data=angles.reshape(1, -1))

        dt = h5py.string_dtype()
        f.create_dataset("StrData", data=elements, dtype=dt)

        ez = f.create_group("ezdeprof")
        ez.create_dataset("depth", data=depth.reshape(-1, 1))
        ez.create_dataset("stepsize", data=np.array([[4.0]]))
        ez.create_dataset("element", data=elements, dtype=dt)
        ez.create_dataset("dxdydt", data=np.array([[n_x, n_y, 1]], dtype=np.float64))
        ez.create_dataset("thetaobs", data=angles.reshape(1, -1))

    return path


# ============================================================================
# 橋A: fitpara_reader tests
# ============================================================================


class TestFitparaReaderElementFiles:
    """Test element-per-file reading path."""

    def test_single_element_shape(self, tmp_path):
        h5 = _make_fitpara_h5(str(tmp_path / "Si2p.h5"), n_x=5, n_y=4, n_angles=7)
        I_elem, meta = read_fitpara_areas([h5])
        # Default n_comp=2 → 2 components per file
        assert I_elem.shape == (20, 7, 2)  # 5*4=20 pixels, 7 angles, 2 comp

    def test_single_element_dtype(self, tmp_path):
        h5 = _make_fitpara_h5(str(tmp_path / "Si2p.h5"))
        I_elem, _ = read_fitpara_areas([h5])
        assert I_elem.dtype == np.float32

    def test_multi_element_shape(self, tmp_path):
        h5_si = _make_fitpara_h5(str(tmp_path / "Si2p.h5"), element="Si2p")
        h5_o = _make_fitpara_h5(str(tmp_path / "O1s.h5"), element="O1s")
        h5_c = _make_fitpara_h5(str(tmp_path / "C1s.h5"), element="C1s")
        I_elem, meta = read_fitpara_areas([h5_si, h5_o, h5_c])
        # Each file has n_comp=2 → 6 total components
        assert I_elem.shape == (20, 7, 6)
        assert meta["elements"] == ["Si2p", "Si2p", "O1s", "O1s", "C1s", "C1s"]

    def test_area_values_per_component(self, tmp_path):
        """Each component area should be read individually (not summed)."""
        area_val = 200.0
        h5 = _make_fitpara_h5(
            str(tmp_path / "Si2p.h5"), n_x=2, n_y=2, n_angles=3,
            n_comp=2, area_value=area_val,
        )
        I_elem, _ = read_fitpara_areas([h5])
        # 2 components → 2 columns, each has area_value*(1+angle/10)/n_comp
        assert I_elem.shape == (4, 3, 2)
        per_comp = area_val / 2.0  # 100.0 per component
        # pixel (0,0), angle 0, comp 0 → per_comp * 1.0 = 100.0
        np.testing.assert_allclose(I_elem[0, 0, 0], per_comp, rtol=1e-5)
        # angle 2 → per_comp * (1 + 2/10) = 120.0
        np.testing.assert_allclose(I_elem[0, 2, 0], per_comp * 1.2, rtol=1e-5)

    def test_metadata_angles(self, tmp_path):
        h5 = _make_fitpara_h5(str(tmp_path / "test.h5"), n_angles=7)
        _, meta = read_fitpara_areas([h5])
        assert meta["n_angles"] == 7
        assert len(meta["angles_deg"]) == 7
        np.testing.assert_allclose(meta["angles_deg"][0], 0.0)
        np.testing.assert_allclose(meta["angles_deg"][-1], 60.0)

    def test_pixel_shape_override(self, tmp_path):
        h5 = _make_fitpara_h5(str(tmp_path / "Si.h5"), n_x=5, n_y=4, n_angles=7)
        I_elem, meta = read_fitpara_areas([h5], pixel_shape=(5, 4))
        assert meta["pixel_shape"] == (5, 4)
        assert I_elem.shape[0] == 20

    def test_nonzero_all_pixels(self, tmp_path):
        """Every pixel-angle should have a nonzero area."""
        h5 = _make_fitpara_h5(str(tmp_path / "Si.h5"), n_x=3, n_y=3, n_angles=5)
        I_elem, _ = read_fitpara_areas([h5])
        assert np.all(I_elem > 0)

    def test_empty_paths_raises(self):
        with pytest.raises(ValueError, match="empty"):
            read_fitpara_areas([])


class TestFitparaReaderDstk:
    """Test Dstk single-file reading path."""

    def test_dstk_auto_detect(self, tmp_path):
        h5 = _make_dstk_h5(str(tmp_path / "dstk.h5"), n_px=20, n_elem=3, n_angles=7)
        I_elem, meta = read_fitpara_areas([h5])
        assert I_elem.shape == (20, 7, 3)

    def test_dstk_elements(self, tmp_path):
        h5 = _make_dstk_h5(str(tmp_path / "dstk.h5"), n_elem=3)
        _, meta = read_fitpara_areas([h5])
        assert meta["elements"] == ["Si(0)", "Si(+4)", "O(-2)"]

    def test_dstk_pixel_shape(self, tmp_path):
        h5 = _make_dstk_h5(str(tmp_path / "dstk.h5"), n_x=5, n_y=4)
        _, meta = read_fitpara_areas([h5])
        assert meta["pixel_shape"] == (5, 4)

    def test_dstk_values_match_xstk(self, tmp_path):
        """I_elem should match Xstk transposed."""
        h5 = _make_dstk_h5(str(tmp_path / "dstk.h5"), n_px=20, n_elem=3, n_angles=7)
        I_elem, _ = read_fitpara_areas([h5])

        with h5py.File(h5, "r") as f:
            xstk = np.asarray(f["Xstk"])  # (n_px, n_elem, n_angles)
            expected = np.transpose(xstk, (0, 2, 1))  # (n_px, n_angles, n_elem)

        np.testing.assert_array_almost_equal(I_elem, expected)


# ============================================================================
# 橋D: depth_result_writer tests (推定値、GT ではない)
# ============================================================================


class TestDepthResultWriter:
    """Test write_depth_estimate — writes inverse-problem estimates (not GT)."""

    def _dummy_d_est(self, n_px=20, n_depth=25, n_elem=6, seed=42):
        rng = np.random.RandomState(seed)
        dm = rng.rand(n_px, n_depth, n_elem).astype(np.float64)
        dm /= dm.sum(axis=2, keepdims=True)
        return dm

    def _dummy_metadata(self, n_x=5, n_y=4, n_elem=6, n_depth=25):
        return {
            "pixel_shape": (n_x, n_y),
            "elements": [f"elem_{i}" for i in range(n_elem)],
            "depth_axis": np.arange(n_depth) * 4.0,
            "step_size": 4.0,
            "angles_deg": np.linspace(0, 60, 7),
        }

    def test_write_creates_file(self, tmp_path):
        dm = self._dummy_d_est()
        meta = self._dummy_metadata()
        path = write_depth_estimate(dm, meta, output_dir=str(tmp_path))
        assert Path(path).exists()

    def test_write_auto_naming(self, tmp_path):
        dm = self._dummy_d_est()
        meta = self._dummy_metadata()
        path = write_depth_estimate(dm, meta, output_dir=str(tmp_path), solver="SLSQP")
        assert "depth_est_SLSQP_" in Path(path).name
        assert path.endswith(".h5")

    def test_d_est_structure(self, tmp_path):
        """H5 has /D_est/data, /D_est/solver, /D_est/timestamp — no /Dstk or /Py_Drlt."""
        dm = self._dummy_d_est(n_px=20, n_depth=25, n_elem=6)
        meta = self._dummy_metadata()
        path = write_depth_estimate(dm, meta, output_dir=str(tmp_path))

        with h5py.File(path, "r") as f:
            # Estimate group
            assert "D_est" in f
            assert "D_est/data" in f
            assert "D_est/solver" in f
            assert "D_est/timestamp" in f
            # MATLAB layout: (n_px, n_elem, n_depth)
            assert f["D_est/data"].shape == (20, 6, 25)
            # No GT datasets
            assert "Dstk" not in f
            assert "Py_Drlt" not in f

    def test_d_est_values_roundtrip(self, tmp_path):
        """Values survive Python→MATLAB→Python transpose."""
        dm = self._dummy_d_est(n_px=20, n_depth=25, n_elem=6)
        meta = self._dummy_metadata()
        path = write_depth_estimate(dm, meta, output_dir=str(tmp_path))

        with h5py.File(path, "r") as f:
            d_est_matlab = np.asarray(f["D_est/data"])  # (n_px, n_elem, n_depth)
            recovered = np.transpose(d_est_matlab, (0, 2, 1))

        np.testing.assert_array_almost_equal(recovered, dm.astype(np.float32), decimal=5)

    def test_ezdeprof_metadata(self, tmp_path):
        dm = self._dummy_d_est()
        meta = self._dummy_metadata()
        path = write_depth_estimate(dm, meta, output_dir=str(tmp_path))

        with h5py.File(path, "r") as f:
            assert "ezdeprof" in f
            assert "depth" in f["ezdeprof"]
            assert "stepsize" in f["ezdeprof"]
            assert "element" in f["ezdeprof"]
            assert "dxdydt" in f["ezdeprof"]

            depth = f["ezdeprof/depth"][:].flatten()
            assert len(depth) == 25
            np.testing.assert_allclose(depth[1] - depth[0], 4.0)

            ss = f["ezdeprof/stepsize"][:].flatten()
            assert float(ss[0]) == 4.0

            dxdydt = f["ezdeprof/dxdydt"][:].flatten()
            assert int(dxdydt[0]) == 5
            assert int(dxdydt[1]) == 4

    def test_xstk_written_when_provided(self, tmp_path):
        dm = self._dummy_d_est(n_px=20, n_depth=25, n_elem=3)
        I_obs = np.random.rand(20, 7, 3).astype(np.float32)
        meta = self._dummy_metadata(n_elem=3)
        path = write_depth_estimate(dm, meta, output_dir=str(tmp_path), I_obs=I_obs)

        with h5py.File(path, "r") as f:
            assert "Xstk" in f
            assert f["Xstk"].shape == (20, 3, 7)
            xstk = np.asarray(f["Xstk"])
            recovered = np.transpose(xstk, (0, 2, 1))
            np.testing.assert_array_almost_equal(recovered, I_obs, decimal=5)

    def test_thetaobs_written(self, tmp_path):
        dm = self._dummy_d_est()
        meta = self._dummy_metadata()
        path = write_depth_estimate(dm, meta, output_dir=str(tmp_path))

        with h5py.File(path, "r") as f:
            theta = f["ezdeprof/thetaobs"][:].flatten()
            assert len(theta) == 7
            np.testing.assert_allclose(theta[0], 0.0)
            np.testing.assert_allclose(theta[-1], 60.0)

    def test_element_mismatch_raises(self, tmp_path):
        dm = self._dummy_d_est(n_elem=3)
        meta = self._dummy_metadata(n_elem=6)  # mismatch!
        with pytest.raises(ValueError, match="elements length"):
            write_depth_estimate(dm, meta, output_dir=str(tmp_path))

    def test_solver_tag_stored(self, tmp_path):
        dm = self._dummy_d_est()
        meta = self._dummy_metadata()
        path = write_depth_estimate(dm, meta, output_dir=str(tmp_path), solver="LLS")

        with h5py.File(path, "r") as f:
            solver_val = f["D_est/solver"][()]
            if isinstance(solver_val, bytes):
                solver_val = solver_val.decode()
            assert solver_val == "LLS"


# ============================================================================
# Round-trip: 橋D estimate → read back Xstk via 橋A
# ============================================================================


class TestBridgeRoundTrip:
    """write_depth_estimate writes /Xstk → read_fitpara_areas reads it back."""

    def test_xstk_roundtrip(self, tmp_path):
        """write_depth_estimate → read_fitpara_areas (Xstk path)."""
        n_px, n_depth, n_elem, n_angles = 20, 25, 3, 7
        rng = np.random.RandomState(123)
        dm = rng.rand(n_px, n_depth, n_elem).astype(np.float64)
        I_obs = rng.rand(n_px, n_angles, n_elem).astype(np.float32)

        meta = {
            "pixel_shape": (5, 4),
            "elements": ["Si(0)", "Si(+4)", "O(-2)"],
            "depth_axis": np.arange(n_depth) * 4.0,
            "step_size": 4.0,
            "angles_deg": np.linspace(0, 60, n_angles),
        }

        path = write_depth_estimate(dm, meta, output_dir=str(tmp_path), I_obs=I_obs)

        # Read back via bridge A (reads /Xstk)
        I_elem, read_meta = read_fitpara_areas([path])

        assert I_elem.shape == (n_px, n_angles, n_elem)
        np.testing.assert_array_almost_equal(I_elem, I_obs, decimal=5)
        assert read_meta["elements"] == ["Si(0)", "Si(+4)", "O(-2)"]
        assert read_meta["pixel_shape"] == (5, 4)
