"""Read batch-fit results from Toyomacro HDF5 files into I_elem arrays.

Supports two input formats:
  1. **Element-per-file** — one H5 per element (Si2p.h5, O1s.h5, ...),
     each containing ``/fitpara (n_spectra, maxcomp, 9)`` with the
     4D flat-index layout ``idx = angle + n_angles*Y + n_angles*n_y*X + ...``.
  2. **Dstk single-file** — one H5 with ``/Xstk (n_px, n_elem, n_angles)``
     already assembled (DepthProfiler convention).

The reader auto-detects which format to use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray


def _schema():
    """``toyomacro.io.schema.ToyomacroSchema`` or a named ImportError.

    toyomacro is the optional extra ``fit``; it is imported here, at the point
    of use, from the installed package only, so that importing this module and
    ``deppro.app`` works without it.
    """
    try:
        from toyomacro.io.schema import ToyomacroSchema
    except ImportError as exc:
        raise ImportError(
            "deppro.app.fitpara_reader needs toyomacro.io to read fitpara HDF5 files: "
            "pip install 'deppro[fit]' (toyomacro, the optional extra `fit`). "
            f"Underlying import error: {exc}"
        ) from exc
    return ToyomacroSchema


def _area_col() -> int:
    return int(_schema().FitparaIndex.INTEGRATED_AREA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_fitpara_areas(
    h5_paths: list[str],
    pixel_shape: tuple[int, int] | None = None,
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    """Read peak areas from fitpara HDF5 files and assemble *I_elem*.

    Parameters
    ----------
    h5_paths : list[str]
        Element-per-file paths **or** a single Dstk-format path.
    pixel_shape : (n_x, n_y) | None
        Spatial grid.  Auto-detected from H5 metadata when *None*.

    Returns
    -------
    I_elem : ndarray, shape (n_pixels, n_angles, n_elements), float32
        ``n_pixels = n_x * n_y`` (spatial flat, angle dimension separated).
    metadata : dict
        ``pixel_shape``, ``n_angles``, ``angles_deg``, ``elements``, ``h5_paths``.
    """
    _schema()  # fail here, by name, when toyomacro is absent (before any file is opened)
    if not h5_paths:
        raise ValueError("h5_paths must not be empty")

    # Auto-detect: single file with /Xstk → Dstk format
    if len(h5_paths) == 1:
        with h5py.File(h5_paths[0], "r") as f:
            if "Xstk" in f:
                return _read_dstk(h5_paths[0], pixel_shape)

    return _read_element_files(h5_paths, pixel_shape)


# ---------------------------------------------------------------------------
# Element-per-file reader
# ---------------------------------------------------------------------------


def _read_element_files(
    h5_paths: list[str],
    pixel_shape: tuple[int, int] | None,
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    """Read element-per-file fitpara HDF5s.

    Each component in a multi-component fit becomes a separate column in
    I_elem so that chemical states (e.g. Si(0), Si(+4)) are resolved.
    """
    _CENTER_COL = int(_schema().FitparaIndex.CENTER)
    _AREA_COL = _area_col()

    # First pass: determine dimensions and count total components
    first_path = h5_paths[0]
    with h5py.File(first_path, "r") as f:
        fitpara0 = _load_fitpara(f)
        n_spectra = fitpara0.shape[0]
        n_angles, n_y, n_x = _read_dims(f, n_spectra)
        angles_deg = _read_angles(f, n_angles)

    if pixel_shape is not None:
        n_x, n_y = pixel_shape
    n_pixels = n_x * n_y

    # Collect per-component info from all files
    comp_infos: list[dict] = []  # {path, elem_name, comp_idx, center_be}
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            fitpara = _load_fitpara(f)  # (n_spectra, n_comp, 9)
            elem_name = _read_element_name(f, path)
            n_comp = fitpara.shape[1]
            # Average center BE over non-zero spectra for each component
            centers = fitpara[:, :, _CENTER_COL]  # (n_spectra, n_comp)
            for c in range(n_comp):
                col = centers[:, c]
                mask = col > 0
                avg_be = float(np.mean(col[mask])) if np.any(mask) else 0.0
                if avg_be > 0:  # skip empty/zero components
                    comp_infos.append({
                        "path": path,
                        "elem_name": elem_name,
                        "comp_idx": c,
                        "center_be": avg_be,
                    })

    n_total_comp = len(comp_infos)
    print(f"[fitpara_reader] {len(h5_paths)} files → {n_total_comp} components")

    # Build I_elem with one column per component
    I_elem = np.zeros((n_pixels, n_angles, n_total_comp), dtype=np.float32)
    comp_labels: list[str] = []
    comp_centers: list[float] = []

    for col_idx, info in enumerate(comp_infos):
        with h5py.File(info["path"], "r") as f:
            fitpara = _load_fitpara(f)
            areas = fitpara[:, info["comp_idx"], _AREA_COL]  # (n_spectra,)

        comp_labels.append(info["elem_name"])
        comp_centers.append(info["center_be"])

        # Scatter into (pixel, angle) grid
        for spec_idx in range(len(areas)):
            angle_idx = spec_idx % n_angles
            remainder = spec_idx // n_angles
            y = remainder % n_y
            x = (remainder // n_y) % n_x
            pixel_flat = x * n_y + y
            if pixel_flat < n_pixels and angle_idx < n_angles:
                I_elem[pixel_flat, angle_idx, col_idx] = areas[spec_idx]

    metadata = {
        "pixel_shape": (n_x, n_y),
        "n_angles": n_angles,
        "angles_deg": angles_deg,
        "elements": comp_labels,
        "component_centers_be": comp_centers,
        "h5_paths": h5_paths,
    }
    return I_elem, metadata


# ---------------------------------------------------------------------------
# Dstk single-file reader
# ---------------------------------------------------------------------------


def _read_dstk(
    h5_path: str,
    pixel_shape: tuple[int, int] | None,
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    """Read Dstk-format H5 with /Xstk."""
    with h5py.File(h5_path, "r") as f:
        # Xstk: MATLAB layout (n_px, n_elem, n_angles)
        xstk_raw = np.asarray(f["Xstk"], dtype=np.float32)
        n_px, n_elem, n_angles = xstk_raw.shape

        # Transpose to (n_px, n_angles, n_elem)
        I_elem = np.transpose(xstk_raw, (0, 2, 1)).copy()

        # Angles
        if "thetaobs" in f:
            angles_deg = np.asarray(f["thetaobs"], dtype=np.float64).flatten()
        elif "ezdeprof" in f and "thetaobs" in f["ezdeprof"]:
            angles_deg = np.asarray(f["ezdeprof/thetaobs"], dtype=np.float64).flatten()
        else:
            angles_deg = np.linspace(0, 65, n_angles)

        # Elements
        if "ezdeprof" in f and "element" in f["ezdeprof"]:
            raw = f["ezdeprof/element"][:].flatten()
            elements = [e.decode() if isinstance(e, bytes) else str(e) for e in raw]
        elif "StrData" in f:
            raw = f["StrData"][:].flatten()
            elements = [e.decode() if isinstance(e, bytes) else str(e) for e in raw]
        else:
            elements = [f"elem_{i}" for i in range(n_elem)]

        # Grid
        if pixel_shape is not None:
            n_x, n_y = pixel_shape
        elif "ezdeprof" in f and "dxdydt" in f["ezdeprof"]:
            dxdydt = np.asarray(f["ezdeprof/dxdydt"], dtype=np.float64).flatten()
            n_x, n_y = int(dxdydt[0]), int(dxdydt[1])
        else:
            n_x = int(np.sqrt(n_px))
            n_y = n_px // n_x

    metadata = {
        "pixel_shape": (n_x, n_y),
        "n_angles": n_angles,
        "angles_deg": angles_deg,
        "elements": elements,
        "h5_paths": [h5_path],
    }
    return I_elem, metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fitpara(f: h5py.File) -> NDArray[np.float32]:
    """Load fitpara array from H5, supporting both raw and compressed formats.

    Returns (n_spectra, n_components, 9) float32 array.
    """
    fp_key = _schema().PATH_FITPARA  # "/fitpara"

    if fp_key in f:
        return np.asarray(f[fp_key], dtype=np.float32)
    elif "fitpara_compressed" in f:
        try:
            from toyomacro.io.compression import decode_fitpara_from_h5
        except ImportError as exc:  # same named error as _schema(), for callers of this helper
            raise ImportError(
                "deppro.app.fitpara_reader needs toyomacro.io to decode compressed fitpara: "
                f"pip install 'deppro[fit]'. Underlying import error: {exc}"
            ) from exc

        return decode_fitpara_from_h5(f)
    else:
        raise KeyError(f"Neither {fp_key} nor fitpara_compressed found in {f.filename}")


def _read_total_areas(f: h5py.File) -> NDArray[np.floating]:
    """Read fitpara col 8 and sum over components → (n_spectra,)."""
    fitpara = _load_fitpara(f)
    areas = fitpara[:, :, _area_col()]  # (n_spectra, maxcomp)
    return np.nansum(areas, axis=1).astype(np.float32)


def _read_dims(
    f: h5py.File, n_spectra: int
) -> tuple[int, int, int]:
    """Return (n_angles, n_y, n_x) from /misc/dim_shape metadata.

    dim_shape follows hdf5_cache convention: [n_angles, n_y, n_t].
    n_x is inferred: n_x = n_spectra // (n_angles * n_y * n_t).
    """
    misc = f.get("misc")
    if misc is not None and "dim_shape" in misc:
        dim_shape = list(misc["dim_shape"][:])
        if len(dim_shape) >= 3:
            n_a, n_y, n_t = int(dim_shape[0]), int(dim_shape[1]), int(dim_shape[2])
            n_t = max(1, n_t)
            n_x = max(1, n_spectra // (n_a * n_y * n_t))
            return n_a, n_y, n_x
        elif len(dim_shape) == 2:
            n_a, n_y = int(dim_shape[0]), int(dim_shape[1])
            n_x = max(1, n_spectra // (n_a * n_y))
            return n_a, n_y, n_x
        elif len(dim_shape) == 1:
            n_a = int(dim_shape[0])
            n_y = max(1, n_spectra // n_a) if n_a > 0 else 1
            return n_a, n_y, 1

    # Fallback: try xytdata
    if _schema().PATH_XYTDATA in f:
        xyt = np.asarray(f[_schema().PATH_XYTDATA], dtype=np.float32)
        # row 0 = angle values — find period
        angles_raw = xyt[0, :]
        if len(angles_raw) > 1:
            diffs = np.diff(angles_raw)
            wraps = np.where(diffs < 0)[0]
            if len(wraps) > 0:
                n_angles = int(wraps[0]) + 1
            else:
                n_angles = len(np.unique(angles_raw))
        else:
            n_angles = 1
        n_y = max(1, n_spectra // n_angles)
        return n_angles, n_y, 1

    raise ValueError("Cannot determine dimensions: no dim_shape or xytdata")


def _read_angles(f: h5py.File, n_angles: int) -> NDArray[np.float64]:
    """Read angle values in degrees from xytdata or generate defaults."""
    if _schema().PATH_XYTDATA in f:
        xyt = np.asarray(f[_schema().PATH_XYTDATA], dtype=np.float64)
        return xyt[0, :n_angles].copy()
    return np.linspace(0, 65, n_angles)


def _read_element_name(f: h5py.File, path: str) -> str:
    """Read element label from /misc or infer from filename.

    Priority: /misc/element > filename stem > /misc/xdata_str.
    ``xdata_str`` often contains axis labels (e.g. "Emission Angle (degree)")
    rather than element names, so it is only used as a last resort.
    """
    misc = f.get("misc")
    if misc is not None:
        # Try /misc/element first — most reliable element name
        if "element" in misc:
            val = misc["element"]
            if hasattr(val, "asstr"):
                return str(val.asstr()[()])
            raw = val[()]
            if isinstance(raw, bytes):
                return raw.decode()
            return str(raw)

    # Filename stem (e.g. "Si2p" from "Si2p.h5") — reliable for element-per-file
    return Path(path).stem
