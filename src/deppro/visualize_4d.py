"""
4D Depth Profile Animation — EzDeP4D.m Python port
====================================================

3D isosurface time-lapse video from GT H5 files (Dstk data).
Replicates MATLAB EzDeP4D.m using PyVista for rendering.

Usage
-----
::

    python -m deppro.visualize_4d \\
        --h5 path/to/dataset.h5 \\
        --nx 10 --ny 10 \\
        --output ezdeP4D_7elem.mp4

Date: 2026-03-02

Isocaps are clip_scalar-based smooth triangulated isocaps (matching
MATLAB's isocaps() behaviour): boundary edges are interpolated triangles,
not axis-aligned stairs.
"""

from __future__ import annotations

import argparse
import gc
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# The module must import without optional dependencies and
# fail with a clear error only at the point of use.  pyvista below already
# follows this pattern; imageio (needed only to write animations) gets the
# same treatment.  An unconditional import here broke collection of
# test_triptych.py on CI, where imageio is not installed.
try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    imageio = None

try:
    from scipy.ndimage import zoom as _ndimage_zoom

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class DepthProfile4DData:
    """5D depth-profile array + metadata for 4D animation.

    Attributes
    ----------
    data : ndarray or None, shape (nt, nx, ny, n_depth, n_elem)
        Concentration fraction (0-1).  ``None`` in lazy (out-of-core) mode;
        use :meth:`get_frame` to access individual time frames.
    depth : ndarray, shape (n_depth,)
        Depth grid in Angstroms.
    element_names : list[str]
        Element labels, e.g. ``['O(-2)', 'B(0)', ...]``.
    colors : ndarray, shape (n_elem, 3)
        RGB colours per element (0-1 range).
    step_size : float
        Depth spacing in Angstroms.
    nx, ny, nt : int
        Spatial and temporal grid dimensions.
    n_depth, n_elem : int
        Depth points and element count.
    source_path : str
        Path of the source H5 file.
    dxdydt : tuple[int, int, int]
        ``(dx, dy, dt)`` metadata from H5.
    _lazy_h5_path : str
        Path to temporary H5 file (lazy mode only).
    """

    data: Optional[np.ndarray]  # (nt, nx, ny, n_depth, n_elem) or None
    depth: np.ndarray
    element_names: List[str]
    colors: np.ndarray
    step_size: float
    nx: int
    ny: int
    nt: int
    n_depth: int
    n_elem: int
    source_path: str = ""
    dxdydt: Tuple[int, int, int] = (1, 1, 1)
    _lazy_h5_path: str = ""

    # --- Runtime state (not constructor args) ---
    def __post_init__(self):
        self._lazy_h5_file: Optional["h5py.File"] = None
        self._lazy_dset: Optional["h5py.Dataset"] = None
        self._frame_cache: Optional[Tuple[int, np.ndarray]] = None

    @property
    def is_lazy(self) -> bool:
        """True if data is stored out-of-core (temp H5)."""
        return self.data is None and self._lazy_h5_path != ""

    def get_frame(self, t: int) -> np.ndarray:
        """Get frame (nx, ny, n_depth, n_elem) for time *t*.

        In-memory mode: direct indexing into ``self.data``.
        Lazy mode: reads one chunk from temp H5, with single-frame cache
        so X/Y/Z slider changes (same *t*) hit cache.
        """
        # In-memory fast path
        if self.data is not None:
            return self.data[t]

        # Lazy: check cache
        if self._frame_cache is not None and self._frame_cache[0] == t:
            return self._frame_cache[1]

        # Lazy: open H5 on first access
        if self._lazy_h5_file is None:
            self._lazy_h5_file = h5py.File(self._lazy_h5_path, "r")
            self._lazy_dset = self._lazy_h5_file["data"]

        frame = np.array(self._lazy_dset[t])  # read one chunk
        self._frame_cache = (t, frame)
        return frame

    def close(self) -> None:
        """Close lazy H5 handle and delete temp file."""
        if self._lazy_h5_file is not None:
            try:
                self._lazy_h5_file.close()
            except Exception:
                pass
            self._lazy_h5_file = None
            self._lazy_dset = None
        if self._lazy_h5_path:
            try:
                Path(self._lazy_h5_path).unlink(missing_ok=True)
            except Exception:
                pass
            self._lazy_h5_path = ""
        self._frame_cache = None


@dataclass
class AnimationConfig:
    """Configuration for 4D animation rendering.

    All numeric parameters have sensible defaults matching MATLAB's
    ``EzDeP4D.m`` (``sp=2`` mode).  Override via constructor kwargs
    or CLI flags.
    """

    # Isosurface
    isovalue_degree: float = 0.5  # threshold = max(V) * degree
    face_alpha: float = 0.75  # body surface opacity (0-1)

    # Composite isocap
    cap_threshold: float = 0.02  # min concentration to render on boundary caps

    # Material
    specular: float = 0.3  # specular reflection strength (0-1)

    # Render window (divisible by 16 for ffmpeg macro_block)
    window_size: Tuple[int, int] = (1280, 960)
    background_color: str = "white"

    # Camera
    azimuth: float = -105.0
    elevation: float = 5.0
    camera_zoom: float = 1.1

    # MATLAB-compatible light — lightangle(-115, 25) in EzDeP4D.m
    # (used by render_mode="matlab")
    light_azimuth: float = -115.0
    light_elevation: float = 25.0

    # MATLAB axes default (DataAspectRatio auto): stretch each axis to
    # fill the plot box.  Applied at the display (actor-scale) level;
    # mesh data keeps physical coordinates (render_mode="matlab" only).
    matlab_stretch_to_fill: bool = True

    # Plot-box side ratios for the stretch above.  ``None`` means a cube,
    # which is what MATLAB's auto stretch-to-fill produces for a *square*
    # axes (measured, R2023b — see MatlabIsosurfaceRenderer).  Pass an
    # explicit ``(x, y, z)`` to reproduce a non-square axes.
    plot_box_aspect: Optional[Tuple[float, float, float]] = None

    # MATLAB resolves transparency by sorting objects back-to-front
    # (axes SortMethod='depth').  VTK draws translucent geometry unsorted,
    # which turns stacked layers into mud.  Depth peeling is the VTK route
    # to the same result (render_mode="matlab", face_alpha < 4/3 only).
    matlab_depth_peeling: bool = True

    # MATLAB patch material defaults, measured on R2023b:
    #   AmbientStrength 0.3, DiffuseStrength 0.6,
    #   SpecularStrength 0.9, SpecularExponent 10
    # (VTK's own specular power default is 100, i.e. a much tighter
    # highlight, which is why the two renderers disagreed per-face.)
    # These apply to render_mode="matlab"; `specular` above stays the
    # generic knob for the composite isosurface mode.
    matlab_ambient: float = 0.3
    matlab_diffuse: float = 0.6
    matlab_specular: float = 0.9
    matlab_specular_power: float = 10.0

    # Depth axis
    z_reversed: bool = True  # depth increases downward
    lighting: bool = True

    # Render mode: 'isosurface' (sp=2, 3D bodies + composite caps),
    # 'slice' (sp=1, 6-face textures), or 'matlab' (sp=2 faithful:
    # per-element interpolated isocaps, physical coords, fixed camera/light)
    render_mode: str = "isosurface"

    # Depth profile subplot
    depth_profile: str = "average"  # 'none' | 'average'

    # Animation
    fps: int = 10
    forward_backward: bool = True  # play forward then backward

    # Element visibility
    invisible_elements: List[int] = field(default_factory=list)

    # Output
    output_path: str = "ezdeP4D.mp4"
    video_quality: int = 8  # imageio/ffmpeg quality (1-10, lower = better)


# ============================================================================
# Data loading
# ============================================================================


# Standard MATLAB DepthProfiler colour palette (MATLAB default line colors)
_STANDARD_ELEMENT_COLORS: dict = {
    "Si(0)":  [0.00, 0.40, 0.67],  # blue  (substrate)
    "Si(+4)": [0.76, 0.29, 0.09],  # rust/orange
    "Ti(+4)": [0.84, 0.62, 0.11],  # golden
    "Al(+3)": [0.44, 0.17, 0.50],  # purple
    "C(0)":   [0.42, 0.61, 0.17],  # olive green
    "B(0)":   [0.27, 0.67, 0.84],  # light blue/cyan
    "O(-2)":  [0.57, 0.07, 0.17],  # dark red
    "O(0)":   [0.57, 0.07, 0.17],  # dark red (alias)
}


def _standard_colors(
    element_names: List[str],
    palette: Optional[Dict[str, List[float]]] = None,
) -> Optional[np.ndarray]:
    """Return palette colours if all elements are matched, else *None*.

    Parameters
    ----------
    element_names : list[str]
        Element labels from the H5 file.
    palette : dict, optional
        ``{element_name: [r, g, b]}``.  Falls back to
        ``_STANDARD_ELEMENT_COLORS`` when *None*.
    """
    lut = palette if palette is not None else _STANDARD_ELEMENT_COLORS
    colors = []
    for name in element_names:
        if name in lut:
            colors.append(lut[name])
        else:
            return None  # unknown element → fall back
    return np.array(colors, dtype=np.float64)


def _hsv_fallback_colors(n: int) -> np.ndarray:
    """Generate HSV-based colours matching MATLAB default."""
    try:
        import matplotlib.cm as cm

        cmap = cm.hsv(np.linspace(0, 1, n, endpoint=False))[:, :3]
    except ImportError:
        # Pure numpy fallback
        hues = np.linspace(0, 1, n, endpoint=False)
        cmap = np.stack(
            [
                np.clip(np.abs(hues * 6 - 3) - 1, 0, 1),
                np.clip(2 - np.abs(hues * 6 - 2), 0, 1),
                np.clip(2 - np.abs(hues * 6 - 4), 0, 1),
            ],
            axis=1,
        )
    return (cmap * 0.9).astype(np.float64)


def _infer_grid_from_path(h5_path: str) -> Optional[Tuple[int, int]]:
    """Try to extract (nx, ny) from parent folder name like ``100x100x200_7elem``.

    Returns None if no match.
    """
    import re

    parent_name = Path(h5_path).parent.name
    m = re.match(r"(\d+)x(\d+)x\d+", parent_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _resolve_colors(
    element_names: List[str],
    h5_col: Optional[np.ndarray],
    n_elem: int,
    color_source: str = "auto",
    palette: Optional[Dict[str, List[float]]] = None,
) -> np.ndarray:
    """Resolve element colours from available sources.

    Parameters
    ----------
    element_names : list[str]
        Element labels from the H5 file.
    h5_col : ndarray or None
        ``ezdeprof/Col`` data from H5 (may need reshape).
    n_elem : int
        Number of elements.
    color_source : str
        ``"auto"`` (H5 → standard → HSV), ``"h5"``, ``"standard"``,
        or ``"hsv"``.
    palette : dict, optional
        Custom ``{element_name: [r, g, b]}`` palette.  When *None* the
        built-in ``_STANDARD_ELEMENT_COLORS`` is used for ``"standard"``
        look-ups.

    Returns
    -------
    ndarray, shape (n_elem, 3)
    """

    def _from_h5() -> Optional[np.ndarray]:
        if h5_col is None:
            return None
        c = np.asarray(h5_col, dtype=np.float64)
        if c.ndim == 1:
            c = c.reshape(-1, 3)
        if c.shape[0] == 3 and c.shape[1] != 3:
            c = c.T
        return c

    def _from_std() -> Optional[np.ndarray]:
        return _standard_colors(element_names, palette=palette)

    def _from_hsv() -> np.ndarray:
        return _hsv_fallback_colors(n_elem)

    def _first(*funcs):
        """Return first non-None result."""
        for fn in funcs:
            result = fn()
            if result is not None:
                return result
        return _from_hsv()  # ultimate fallback

    if color_source == "h5":
        return _first(_from_h5)
    if color_source == "standard":
        return _first(_from_std)
    if color_source == "hsv":
        return _from_hsv()
    # auto: H5 → standard → HSV
    return _first(_from_h5, _from_std)


def load_gt_4d(
    h5_path: str,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    color_source: str = "auto",
    palette: Optional[Dict[str, List[float]]] = None,
) -> DepthProfile4DData:
    """Load ground-truth Dstk from H5 and reshape to 5D.

    Parameters
    ----------
    h5_path : str
        Path to a GT H5 file (a MATLAB DepthProfiler dataset H5).
    nx, ny : int, optional
        Spatial grid dimensions.  If *None*, derived from
        ``/ezdeprof/dxdydt`` metadata ``[dx, dy, dt]``.
    color_source : str
        ``"auto"`` (H5 → standard → HSV), ``"h5"``, ``"standard"``,
        or ``"hsv"``.
    palette : dict, optional
        Custom ``{element_name: [r, g, b]}`` colour map.  Used when
        *color_source* is ``"auto"`` or ``"standard"``.

    Returns
    -------
    DepthProfile4DData
        Ready-to-render 5D data.
    """
    h5_path = str(h5_path)

    with h5py.File(h5_path, "r") as f:
        # --- Dstk: MATLAB layout (n_px, n_elem, n_depth) ---
        Dstk_raw = np.array(f["Dstk"], dtype=np.float64)
        n_px, n_elem_h5, n_depth = Dstk_raw.shape
        # Transpose to Python layout: (n_px, n_depth, n_elem)
        Dstk = np.transpose(Dstk_raw, (0, 2, 1))

        # --- Metadata ---
        depth = np.array(f["ezdeprof/depth"], dtype=np.float64).flatten()
        step_size = float(
            np.array(f["ezdeprof/stepsize"], dtype=np.float64).flatten()[0]
        )

        # Element names
        raw_elem = f["ezdeprof/element"][:].flatten()
        element_names = [
            e.decode() if isinstance(e, bytes) else str(e) for e in raw_elem
        ]

        # dxdydt
        dxdydt_arr = np.array(f["ezdeprof/dxdydt"], dtype=np.float64).flatten()
        dx_h5 = int(dxdydt_arr[0])
        dy_h5 = int(dxdydt_arr[1])
        dt_h5 = int(dxdydt_arr[2])

        # Colours: configurable source priority
        h5_col = (
            np.array(f["ezdeprof/Col"], dtype=np.float64)
            if "ezdeprof/Col" in f
            else None
        )
        colors = _resolve_colors(
            element_names, h5_col, n_elem_h5,
            color_source=color_source, palette=palette,
        )

    # --- Spatial grid ---
    if nx is None:
        nx = dx_h5
    if ny is None:
        ny = dy_h5
    dxdy = nx * ny

    if n_px % dxdy != 0:
        # Attempt to infer from parent folder name (e.g. "100x100x200_7elem")
        inferred = _infer_grid_from_path(h5_path)
        if inferred is not None:
            nx, ny = inferred
            dxdy = nx * ny
            print(
                f"H5 dxdydt ({dx_h5},{dy_h5}) incompatible with n_px={n_px}; "
                f"inferred nx={nx}, ny={ny} from folder name"
            )
        if n_px % dxdy != 0:
            raise ValueError(
                f"n_px={n_px} is not divisible by nx*ny={dxdy}. "
                f"Check --nx/--ny or H5 dxdydt metadata."
            )
    nt = n_px // dxdy

    # --- Reshape to 5D: (nt, nx, ny, n_depth, n_elem) ---
    # MATLAB column-major: x varies fastest within a time frame.
    # Vectorised via Fortran-order index array (no Python loops).
    Dstk_4d = Dstk.reshape(nt, dxdy, n_depth, n_elem_h5)
    idx_F = np.arange(dxdy).reshape(nx, ny, order="F")  # Fortran pixel map
    data_5d = Dstk_4d[:, idx_F, :, :]  # (nt, nx, ny, n_depth, n_elem)

    print(
        f"Loaded {Path(h5_path).name}: "
        f"({nt}, {nx}, {ny}, {n_depth}, {n_elem_h5}) = "
        f"{data_5d.nbytes / 1e6:.1f} MB"
    )

    return DepthProfile4DData(
        data=data_5d,
        depth=depth,
        element_names=element_names,
        colors=colors,
        step_size=step_size,
        nx=nx,
        ny=ny,
        nt=nt,
        n_depth=n_depth,
        n_elem=n_elem_h5,
        source_path=h5_path,
        dxdydt=(dx_h5, dy_h5, dt_h5),
    )


def load_gt_4d_lazy(
    h5_path: str,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    color_source: str = "auto",
    palette: Optional[Dict[str, List[float]]] = None,
    compression: str = "gzip",
) -> DepthProfile4DData:
    """Load GT Dstk out-of-core: metadata in memory, frames in temp H5.

    Creates a temporary HDF5 file with dataset ``"data"`` of shape
    ``(nt, nx, ny, n_depth, n_elem)`` chunked by time frame.
    Only one frame needs to fit in memory at a time.

    Parameters
    ----------
    h5_path : str
        Path to a GT H5 file.
    nx, ny : int, optional
        Spatial grid (same as :func:`load_gt_4d`).
    color_source, palette
        Colour options (same as :func:`load_gt_4d`).
    compression : str
        HDF5 compression for temp file (``"gzip"``, ``"lzf"``, or ``None``).

    Returns
    -------
    DepthProfile4DData
        ``data`` is ``None``; use :meth:`get_frame` to read frames.
        Call :meth:`close` to clean up the temp file.
    """
    h5_path = str(h5_path)

    with h5py.File(h5_path, "r") as f:
        dset = f["Dstk"]
        n_px, n_elem_h5, n_depth = dset.shape

        # Metadata (small)
        depth = np.array(f["ezdeprof/depth"], dtype=np.float64).flatten()
        step_size = float(
            np.array(f["ezdeprof/stepsize"], dtype=np.float64).flatten()[0]
        )
        raw_elem = f["ezdeprof/element"][:].flatten()
        element_names = [
            e.decode() if isinstance(e, bytes) else str(e) for e in raw_elem
        ]
        dxdydt_arr = np.array(f["ezdeprof/dxdydt"], dtype=np.float64).flatten()
        dx_h5, dy_h5, dt_h5 = int(dxdydt_arr[0]), int(dxdydt_arr[1]), int(dxdydt_arr[2])
        h5_col = (
            np.array(f["ezdeprof/Col"], dtype=np.float64)
            if "ezdeprof/Col" in f
            else None
        )
        colors = _resolve_colors(
            element_names, h5_col, n_elem_h5,
            color_source=color_source, palette=palette,
        )

        # Grid dimensions
        if nx is None:
            nx = dx_h5
        if ny is None:
            ny = dy_h5
        dxdy = nx * ny
        if n_px % dxdy != 0:
            inferred = _infer_grid_from_path(h5_path)
            if inferred is not None:
                nx, ny = inferred
                dxdy = nx * ny
        if n_px % dxdy != 0:
            raise ValueError(
                f"n_px={n_px} not divisible by nx*ny={dxdy}"
            )
        nt = n_px // dxdy

        # Fortran-order pixel map
        idx_F = np.arange(dxdy).reshape(nx, ny, order="F")

        # Create temp H5 with time-chunked 5D layout
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="dp4d_lazy_")
        import os
        os.close(tmp_fd)

        frame_shape = (nx, ny, n_depth, n_elem_h5)
        print(
            f"Lazy load: creating temp H5 at {tmp_path}\n"
            f"  Shape: ({nt}, {nx}, {ny}, {n_depth}, {n_elem_h5}), "
            f"compression={compression}"
        )

        with h5py.File(tmp_path, "w") as tmp_f:
            tmp_dset = tmp_f.create_dataset(
                "data",
                shape=(nt, nx, ny, n_depth, n_elem_h5),
                dtype=np.float32,
                chunks=(1, *frame_shape),
                compression=compression,
                compression_opts=1 if compression == "gzip" else None,
            )

            # Write frame-by-frame (only 1 frame in memory at a time)
            for t in range(nt):
                px_start = t * dxdy
                px_end = px_start + dxdy
                # Read pixels for this frame: (dxdy, n_elem, n_depth)
                raw = np.array(dset[px_start:px_end], dtype=np.float32)
                # Transpose to (dxdy, n_depth, n_elem)
                frame_flat = np.transpose(raw, (0, 2, 1))
                # Reshape via Fortran index map: (nx, ny, n_depth, n_elem)
                frame_5d = frame_flat[idx_F]
                tmp_dset[t] = frame_5d

                if (t + 1) % 50 == 0 or (t + 1) == nt:
                    print(f"  Written {t + 1}/{nt} frames")

    per_frame_mb = nx * ny * n_depth * n_elem_h5 * 4 / 1e6
    tmp_size_mb = Path(tmp_path).stat().st_size / 1e6
    full_mem_gb = n_px * n_depth * n_elem_h5 * 8 / 1e9
    print(
        f"Lazy load complete: {tmp_size_mb:.0f} MB on disk "
        f"(vs {full_mem_gb:.1f} GB in-memory), "
        f"{per_frame_mb:.1f} MB/frame"
    )

    return DepthProfile4DData(
        data=None,
        depth=depth,
        element_names=element_names,
        colors=colors,
        step_size=step_size,
        nx=nx,
        ny=ny,
        nt=nt,
        n_depth=n_depth,
        n_elem=n_elem_h5,
        source_path=h5_path,
        dxdydt=(dx_h5, dy_h5, dt_h5),
        _lazy_h5_path=tmp_path,
    )


def load_depth_estimate(
    h5_path: str,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    color_source: str = "auto",
    palette: Optional[Dict[str, List[float]]] = None,
) -> DepthProfile4DData:
    """Load inverse-problem estimate from ``/D_est/data`` and reshape to 5D.

    Same interface and return type as :func:`load_gt_4d`, but reads
    ``/D_est/data`` (estimate) instead of ``/Dstk`` (ground truth).

    Parameters
    ----------
    h5_path : str
        Path to an estimate H5 written by
        ``deppro.app.write_depth_estimate()``.
    nx, ny : int, optional
        Spatial grid.  Derived from ``/ezdeprof/dxdydt`` if *None*.
    color_source, palette
        See :func:`load_gt_4d`.
    """
    h5_path = str(h5_path)

    with h5py.File(h5_path, "r") as f:
        # --- D_est: MATLAB layout (n_px, n_elem, n_depth) ---
        D_est_raw = np.array(f["D_est/data"], dtype=np.float64)
        n_px, n_elem_h5, n_depth = D_est_raw.shape
        # Transpose to Python layout: (n_px, n_depth, n_elem)
        D_est = np.transpose(D_est_raw, (0, 2, 1))

        # --- Metadata ---
        depth = np.array(f["ezdeprof/depth"], dtype=np.float64).flatten()
        step_size = float(
            np.array(f["ezdeprof/stepsize"], dtype=np.float64).flatten()[0]
        )

        # Element names
        raw_elem = f["ezdeprof/element"][:].flatten()
        element_names = [
            e.decode() if isinstance(e, bytes) else str(e) for e in raw_elem
        ]

        # dxdydt
        dxdydt_arr = np.array(f["ezdeprof/dxdydt"], dtype=np.float64).flatten()
        dx_h5 = int(dxdydt_arr[0])
        dy_h5 = int(dxdydt_arr[1])
        dt_h5 = int(dxdydt_arr[2])

        # Colours
        h5_col = (
            np.array(f["ezdeprof/Col"], dtype=np.float64)
            if "ezdeprof/Col" in f
            else None
        )
        colors = _resolve_colors(
            element_names, h5_col, n_elem_h5,
            color_source=color_source, palette=palette,
        )

    # --- Spatial grid ---
    if nx is None:
        nx = dx_h5
    if ny is None:
        ny = dy_h5
    dxdy = nx * ny

    if n_px % dxdy != 0:
        inferred = _infer_grid_from_path(h5_path)
        if inferred is not None:
            nx, ny = inferred
            dxdy = nx * ny
        if n_px % dxdy != 0:
            raise ValueError(
                f"n_px={n_px} is not divisible by nx*ny={dxdy}. "
                f"Check --nx/--ny or H5 dxdydt metadata."
            )
    nt = n_px // dxdy

    # --- Reshape to 5D: (nt, nx, ny, n_depth, n_elem) ---
    D_est_4d = D_est.reshape(nt, dxdy, n_depth, n_elem_h5)
    idx_F = np.arange(dxdy).reshape(nx, ny, order="F")
    data_5d = D_est_4d[:, idx_F, :, :]

    print(
        f"Loaded estimate {Path(h5_path).name}: "
        f"({nt}, {nx}, {ny}, {n_depth}, {n_elem_h5}) = "
        f"{data_5d.nbytes / 1e6:.1f} MB"
    )

    return DepthProfile4DData(
        data=data_5d,
        depth=depth,
        element_names=element_names,
        colors=colors,
        step_size=step_size,
        nx=nx,
        ny=ny,
        nt=nt,
        n_depth=n_depth,
        n_elem=n_elem_h5,
        source_path=h5_path,
        dxdydt=(dx_h5, dy_h5, dt_h5),
    )


def load_depth_estimate_lazy(
    h5_path: str,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    color_source: str = "auto",
    palette: Optional[Dict[str, List[float]]] = None,
    compression: str = "gzip",
) -> DepthProfile4DData:
    """Load estimate ``/D_est/data`` out-of-core via temp H5.

    Lazy counterpart to :func:`load_depth_estimate`.
    Same interface as :func:`load_gt_4d_lazy` but reads ``/D_est/data``.

    Returns
    -------
    DepthProfile4DData
        ``data`` is ``None``; use :meth:`get_frame` to access frames.
        Call :meth:`close` to clean up the temp file.
    """
    h5_path = str(h5_path)

    with h5py.File(h5_path, "r") as f:
        dset = f["D_est/data"]
        n_px, n_elem_h5, n_depth = dset.shape

        # Metadata (small)
        depth = np.array(f["ezdeprof/depth"], dtype=np.float64).flatten()
        step_size = float(
            np.array(f["ezdeprof/stepsize"], dtype=np.float64).flatten()[0]
        )
        raw_elem = f["ezdeprof/element"][:].flatten()
        element_names = [
            e.decode() if isinstance(e, bytes) else str(e) for e in raw_elem
        ]
        dxdydt_arr = np.array(f["ezdeprof/dxdydt"], dtype=np.float64).flatten()
        dx_h5, dy_h5, dt_h5 = int(dxdydt_arr[0]), int(dxdydt_arr[1]), int(dxdydt_arr[2])
        h5_col = (
            np.array(f["ezdeprof/Col"], dtype=np.float64)
            if "ezdeprof/Col" in f
            else None
        )
        colors = _resolve_colors(
            element_names, h5_col, n_elem_h5,
            color_source=color_source, palette=palette,
        )

        # Grid dimensions
        if nx is None:
            nx = dx_h5
        if ny is None:
            ny = dy_h5
        dxdy = nx * ny
        if n_px % dxdy != 0:
            inferred = _infer_grid_from_path(h5_path)
            if inferred is not None:
                nx, ny = inferred
                dxdy = nx * ny
        if n_px % dxdy != 0:
            raise ValueError(
                f"n_px={n_px} not divisible by nx*ny={dxdy}"
            )
        nt = n_px // dxdy

        # Fortran-order pixel map
        idx_F = np.arange(dxdy).reshape(nx, ny, order="F")

        # Create temp H5
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="dp4d_est_lazy_")
        import os
        os.close(tmp_fd)

        frame_shape = (nx, ny, n_depth, n_elem_h5)
        print(
            f"Lazy load (estimate): creating temp H5 at {tmp_path}\n"
            f"  Shape: ({nt}, {nx}, {ny}, {n_depth}, {n_elem_h5}), "
            f"compression={compression}"
        )

        with h5py.File(tmp_path, "w") as tmp_f:
            tmp_dset = tmp_f.create_dataset(
                "data",
                shape=(nt, nx, ny, n_depth, n_elem_h5),
                dtype=np.float32,
                chunks=(1, *frame_shape),
                compression=compression,
                compression_opts=1 if compression == "gzip" else None,
            )

            for t in range(nt):
                px_start = t * dxdy
                px_end = px_start + dxdy
                raw = np.array(dset[px_start:px_end], dtype=np.float32)
                frame_flat = np.transpose(raw, (0, 2, 1))
                frame_5d = frame_flat[idx_F]
                tmp_dset[t] = frame_5d

                if (t + 1) % 50 == 0 or (t + 1) == nt:
                    print(f"  Written {t + 1}/{nt} frames")

    per_frame_mb = nx * ny * n_depth * n_elem_h5 * 4 / 1e6
    tmp_size_mb = Path(tmp_path).stat().st_size / 1e6
    full_mem_gb = n_px * n_depth * n_elem_h5 * 8 / 1e9
    print(
        f"Lazy load (estimate) complete: {tmp_size_mb:.0f} MB on disk "
        f"(vs {full_mem_gb:.1f} GB in-memory), "
        f"{per_frame_mb:.1f} MB/frame"
    )

    return DepthProfile4DData(
        data=None,
        depth=depth,
        element_names=element_names,
        colors=colors,
        step_size=step_size,
        nx=nx,
        ny=ny,
        nt=nt,
        n_depth=n_depth,
        n_elem=n_elem_h5,
        source_path=h5_path,
        dxdydt=(dx_h5, dy_h5, dt_h5),
        _lazy_h5_path=tmp_path,
    )


# ============================================================================
# Isosurface Renderer
# ============================================================================


class IsosurfaceRenderer:
    """MATLAB EzDeP4D ``sp=2`` style renderer — per-element solid
    isosurfaces with isocaps (``isosurface`` + ``isocaps``).

    Uses PyVista ``threshold`` + ``extract_surface`` to produce filled
    blocks equivalent to MATLAB's ``isosurface`` + ``isocaps`` pair.
    Each element is rendered as a separate coloured solid body.
    """

    def __init__(self, data: DepthProfile4DData, config: AnimationConfig):
        if not HAS_PYVISTA:
            raise ImportError("PyVista required: pip install pyvista")
        self.data = data
        self.config = config

        # Visible element indices
        all_idx = set(range(data.n_elem))
        hidden = set(config.invisible_elements)
        self.visible = sorted(all_idx - hidden)

        # Clip ranges (mutable — interactive sliders update these)
        # None = full range; (lo, hi) clips to [lo, hi] inclusive
        self.clip_x: Optional[Tuple[int, int]] = None  # 0..nx-1
        self.clip_y: Optional[Tuple[int, int]] = None  # 0..ny-1
        self.clip_z: Optional[Tuple[int, int]] = None  # 0..n_depth-1

    @staticmethod
    def _auto_upscale(nx: int, ny: int, nz: int) -> Tuple[int, int]:
        """Choose anisotropic upscale factors ``(scale_xy, scale_z)``.

        The depth dimension (z) typically has far fewer points (e.g. 20)
        than the spatial grid, so it gets a higher upscale factor to
        produce smooth layer boundaries.
        """
        n = max(nx, ny)
        if n <= 20:
            scale_xy = 6
        elif n <= 50:
            scale_xy = 3
        else:
            scale_xy = 2
        # Z always needs generous upscaling for smooth depth boundaries
        scale_z = max(scale_xy, 4)
        return scale_xy, scale_z

    @staticmethod
    def _smooth_isocap(
        sl: np.ndarray,
        threshold: float,
        axis: int,
        pos: float,
        sp: Tuple[float, float],
        up: Tuple[float, float] = (1, 1),
    ) -> Optional["pv.PolyData"]:
        """Create a smooth triangulated isocap via ``clip_scalar``.

        Equivalent to MATLAB's ``isocaps()`` — uses point-data interpolation
        to produce smooth triangulated boundary edges, not axis-aligned
        staircase faces.

        Parameters
        ----------
        sl : ndarray, shape (na, nb)
            2D concentration slice on the boundary face (already masked by
            dominant-element filter).
        threshold : float
            Iso-value threshold.
        axis : int
            Normal axis of the face (0=X, 1=Y, 2=Z).
        pos : float
            Position along ``axis``.
        sp : tuple of float
            ``(spacing_a, spacing_b)`` in display coordinates for the two
            in-plane axes.
        up : tuple of float
            ``(upscale_a, upscale_b)`` factors for the 2D slice.

        Returns
        -------
        pv.PolyData or None
            Flat triangulated cap mesh, or *None* if nothing to render.
        """
        if sl.max() < threshold:
            return None
        if sl.shape[0] < 2 or sl.shape[1] < 2:
            return None

        # --- Upscale for smooth boundary resolution ---
        if HAS_SCIPY and (up[0] > 1 or up[1] > 1):
            sl_up = _ndimage_zoom(sl, up, order=1)
        else:
            sl_up = sl

        na, nb = sl_up.shape

        # Spacings map upscaled indices → original display coordinates
        ext_a = sp[0] * (sl.shape[0] - 1)
        ext_b = sp[1] * (sl.shape[1] - 1)
        sp_a = ext_a / max(na - 1, 1)
        sp_b = ext_b / max(nb - 1, 1)
        eps = min(sp_a, sp_b) * 0.01  # very thin slab

        # --- Build thin-slab ImageData with identical values on both
        #     normal-axis layers (point_data for interpolated clipping) ---
        if axis == 2:  # Z-face: in-plane = (X, Y)
            slab = np.stack([sl_up, sl_up], axis=2)  # (na, nb, 2)
            dims = (na, nb, 2)
            spacing = (sp_a, sp_b, eps)
            org = (0.0, 0.0, pos - eps * 0.5)
        elif axis == 0:  # X-face: in-plane = (Y, Z)
            slab = np.zeros((2, na, nb))
            slab[0] = sl_up
            slab[1] = sl_up
            dims = (2, na, nb)
            spacing = (eps, sp_a, sp_b)
            org = (pos - eps * 0.5, 0.0, 0.0)
        elif axis == 1:  # Y-face: in-plane = (X, Z)
            slab = np.zeros((na, 2, nb))
            slab[:, 0, :] = sl_up
            slab[:, 1, :] = sl_up
            dims = (na, 2, nb)
            spacing = (sp_a, eps, sp_b)
            org = (0.0, pos - eps * 0.5, 0.0)
        else:
            return None

        grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=org)
        grid.point_data["c"] = slab.ravel(order="F")

        try:
            # clip_scalar keeps region where scalar >= value (invert=True default)
            clipped = grid.clip_scalar(scalars="c", value=threshold)
            if clipped.n_cells > 0:
                surf = clipped.extract_surface()
                # Project onto the exact boundary plane
                surf.points[:, axis] = pos
                surf = surf.clean(tolerance=1e-8)
                if surf.n_points > 0:
                    return surf
        except Exception:
            pass
        return None

    def _composite_isocap(
        self,
        frame: np.ndarray,
        axis: int,
        face_idx: int,
        pos: float,
        sp: Tuple[float, float],
        up: Tuple[int, int] = (1, 1),
    ) -> Optional["pv.PolyData"]:
        """Opaque isocap with concentration-weighted element colours.

        Instead of overlapping semi-transparent per-element caps (which
        blend into a washed-out pink — "桜色" artifact), produce a
        single mesh with per-cell RGB colour = weighted average of
        visible element colours by local concentration.

        Parameters
        ----------
        frame : ndarray, shape (nx, ny, n_depth, n_elem)
            Concentration data for one time frame.
        axis : int
            Normal axis of the face (0=X, 1=Y, 2=Z).
        face_idx : int
            Index along *axis* (0 or max-1).
        pos : float
            Display-coordinate position along *axis*.
        sp : tuple of float
            ``(spacing_a, spacing_b)`` in display coordinates.
        up : tuple of int
            ``(upscale_a, upscale_b)`` factors.

        Returns
        -------
        pv.PolyData or None
            Flat quad mesh with per-cell ``"RGB"`` data, or *None*.
        """
        vis = self.visible
        colors_vis = self.data.colors[vis]  # (n_vis, 3)

        # Extract 2D concentration slice for visible elements
        if axis == 2:
            conc = frame[:, :, face_idx][:, :, vis]
        elif axis == 0:
            conc = frame[face_idx, :, :][:, :, vis]
        elif axis == 1:
            conc = frame[:, face_idx, :][:, :, vis]
        else:
            return None

        na, nb, nv = conc.shape
        if na < 2 or nb < 2:
            return None

        # Upscale each element channel for smooth colour transitions
        if HAS_SCIPY and (up[0] > 1 or up[1] > 1):
            conc_up = np.stack(
                [_ndimage_zoom(conc[:, :, i], up, order=1) for i in range(nv)],
                axis=2,
            )
        else:
            conc_up = conc

        na_up, nb_up = conc_up.shape[:2]

        # Cell-centre concentrations (average of 4 corner points)
        c00 = conc_up[:-1, :-1, :]
        c10 = conc_up[1:, :-1, :]
        c01 = conc_up[:-1, 1:, :]
        c11 = conc_up[1:, 1:, :]
        cc = (c00 + c10 + c01 + c11) * 0.25  # (nc_a, nc_b, nv)

        # Weighted-average RGB per cell
        total = cc.sum(axis=2, keepdims=True).clip(min=1e-10)
        weights = cc / total
        rgb = np.einsum("ije,ec->ijc", weights, colors_vis)
        rgb = np.clip(rgb, 0, 1)

        # Mask: skip cells where no element exceeds threshold
        max_conc = cc.max(axis=2)
        mask = max_conc > self.config.cap_threshold
        if not mask.any():
            return None

        nc_a, nc_b = cc.shape[:2]
        ext_a = sp[0] * (na - 1)
        ext_b = sp[1] * (nb - 1)
        a_vals = np.linspace(0, ext_a, nc_a + 1)
        b_vals = np.linspace(0, ext_b, nc_b + 1)
        A, B = np.meshgrid(a_vals, b_vals, indexing="ij")

        # Build 3D point array on the face plane
        if axis == 2:
            pts = np.column_stack(
                [A.ravel(), B.ravel(), np.full(A.size, pos)]
            )
        elif axis == 0:
            pts = np.column_stack(
                [np.full(A.size, pos), A.ravel(), B.ravel()]
            )
        elif axis == 1:
            pts = np.column_stack(
                [A.ravel(), np.full(A.size, pos), B.ravel()]
            )
        else:
            return None

        # Vectorised quad face construction
        stride = nc_b + 1
        i_idx = np.repeat(np.arange(nc_a), nc_b)
        j_idx = np.tile(np.arange(nc_b), nc_a)
        p0 = i_idx * stride + j_idx
        p1 = p0 + 1
        p2 = (i_idx + 1) * stride + j_idx + 1
        p3 = (i_idx + 1) * stride + j_idx

        # Filter to cells passing the mask
        keep = np.where(mask.ravel())[0]
        if len(keep) == 0:
            return None

        n_keep = len(keep)
        faces = np.empty((n_keep, 5), dtype=np.int64)
        faces[:, 0] = 4
        faces[:, 1] = p0[keep]
        faces[:, 2] = p1[keep]
        faces[:, 3] = p2[keep]
        faces[:, 4] = p3[keep]

        mesh = pv.PolyData(pts, faces=faces.ravel())
        rgb_keep = rgb.reshape(-1, 3)[keep]
        mesh.cell_data["RGB"] = (rgb_keep * 255).astype(np.uint8)

        return mesh

    def render_frame(self, t: int, plotter: pv.Plotter) -> None:
        """Render a single time frame — isosurface + isocaps (MATLAB style).

        Uses upscaled volume + marching cubes ``contour`` for smooth outer
        surface, plus 6-face boundary caps (``isocaps``) for solid fill.

        Parameters
        ----------
        t : int
            Time frame index (0-based).
        plotter : pv.Plotter
            PyVista plotter (already cleared by caller).
        """
        d = self.data
        cfg = self.config

        # Lighting FIRST — must be set before meshes for off-screen render
        if cfg.lighting:
            plotter.enable_3_lights()

        # Z-spacing for display: normalise so depth ≈ spatial scale
        spatial_extent = max(d.nx, d.ny)
        if d.n_depth > 1:
            z_display_spacing = spatial_extent / (d.n_depth - 1)
        else:
            z_display_spacing = 1.0

        # --- Apply clip ranges ---
        frame_full = d.get_frame(t)  # (nx, ny, n_depth, n_elem)
        x0 = self.clip_x[0] if self.clip_x else 0
        x1 = self.clip_x[1] if self.clip_x else d.nx - 1
        y0 = self.clip_y[0] if self.clip_y else 0
        y1 = self.clip_y[1] if self.clip_y else d.ny - 1
        z0 = self.clip_z[0] if self.clip_z else 0
        z1 = self.clip_z[1] if self.clip_z else d.n_depth - 1
        frame = frame_full[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1, :]
        cnx, cny, cnz = frame.shape[:3]  # clipped dimensions

        if HAS_SCIPY:
            scale_xy, scale_z = self._auto_upscale(cnx, cny, cnz)
        else:
            scale_xy, scale_z = 1, 1

        # Bounding box in display coords (origin at clip start)
        x_max = float(cnx - 1)
        y_max = float(cny - 1)
        z_max = float(cnz - 1) * z_display_spacing
        pad_x = 1.0            # padding = 1 original cell
        pad_z = z_display_spacing

        # --- Per-element isosurface BODY (marching cubes contour) ---
        # Each element gets its own semi-transparent isosurface.
        # Boundary caps are rendered separately as composite colour-blended
        # meshes (see Phase 2 below) to avoid pink/washed-out overlap.

        # Need at least 2 voxels in each dimension for isosurface extraction
        can_isosurface = cnx >= 2 and cny >= 2 and cnz >= 2

        for elem_idx in self.visible:
            V = frame[:, :, :, elem_idx]  # (nx, ny, n_depth)
            v_max = V.max()
            if v_max < 1e-10:
                continue
            threshold = v_max * cfg.isovalue_degree
            color = d.colors[elem_idx].tolist()

            meshes = []

            if not can_isosurface:
                # Too few voxels for contour — skip body, caps only
                pass
            elif scale_xy > 1 or scale_z > 1:
                # Pad + anisotropic upscale for smooth isosurface
                V_pad = np.pad(V, pad_width=1, mode="edge")
                V_hi = _ndimage_zoom(
                    V_pad, (scale_xy, scale_xy, scale_z), order=1
                )
                nx_hi, ny_hi, nz_hi = V_hi.shape
                sp_x = 1.0 / scale_xy
                sp_z = z_display_spacing / scale_z

                # --- 1. Smooth isosurface (marching cubes) ---
                grid_pt = pv.ImageData(
                    dimensions=(nx_hi, ny_hi, nz_hi),
                    spacing=(sp_x, sp_x, sp_z),
                    origin=(-pad_x, -pad_x, -pad_z),
                )
                grid_pt.point_data["c"] = V_hi.ravel(order="F")
                try:
                    iso = grid_pt.contour([threshold], scalars="c")
                    if iso.n_points > 0:
                        meshes.append(iso)
                except Exception:
                    pass

                # (Isocaps rendered as composite blend — see Phase 2)

            else:
                # No upscale: threshold for each element independently
                grid = pv.ImageData(
                    dimensions=(cnx + 1, cny + 1, cnz + 1),
                    spacing=(1.0, 1.0, z_display_spacing),
                    origin=(0, 0, 0),
                )
                grid.cell_data["c"] = V.ravel(order="F")
                try:
                    solid = grid.threshold(threshold, scalars="c")
                    if solid.n_cells > 0:
                        meshes.append(solid.extract_surface())
                except Exception:
                    pass

            if not meshes:
                continue

            # Merge all surfaces + clip to original bounding box
            combined = meshes[0]
            for m in meshes[1:]:
                combined = combined.merge(m)
            try:
                combined = combined.clip_box(
                    [0, x_max, 0, y_max, 0, z_max], invert=False
                )
            except Exception:
                continue
            if combined.n_points == 0:
                continue

            # Z-axis reversal
            if cfg.z_reversed:
                combined.points[:, 2] = -combined.points[:, 2]

            plotter.add_mesh(
                combined,
                color=color,
                opacity=cfg.face_alpha,
                smooth_shading=True,
                specular=cfg.specular,
                label=d.element_names[elem_idx],
            )

        # --- Phase 2: Composite isocaps (concentration-weighted colour) ---
        # Single opaque mesh per boundary face.  Eliminates the pink
        # "桜色" artifact from overlapping per-element semi-transparent caps.
        cap_defs = [
            # (axis, [(face_idx, pos), ...], spacing, upscale)
            (2, [(0, 0.0), (cnz - 1, z_max)],
             (1.0, 1.0), (scale_xy, scale_xy)),
            (0, [(0, 0.0), (cnx - 1, x_max)],
             (1.0, z_display_spacing), (scale_xy, scale_z)),
            (1, [(0, 0.0), (cny - 1, y_max)],
             (1.0, z_display_spacing), (scale_xy, scale_z)),
        ]
        for axis, pairs, sp_cap, up_cap in cap_defs:
            for fi, pos in pairs:
                cap = self._composite_isocap(
                    frame, axis, fi, pos, sp_cap, up_cap,
                )
                if cap is not None and cap.n_points > 0:
                    if cfg.z_reversed:
                        cap.points[:, 2] = -cap.points[:, 2]
                    plotter.add_mesh(
                        cap, scalars="RGB", rgb=True,
                        opacity=1.0, smooth_shading=False,
                    )

        # --- Axes and title ---
        plotter.add_axes(
            xlabel="X", ylabel="Y", zlabel="Depth (A)", line_width=2
        )
        name = Path(d.source_path).stem if d.source_path else "GT"
        clip_info = ""
        if self.clip_x or self.clip_y or self.clip_z:
            clip_info = f" | clip [{cnx}x{cny}x{cnz}]"
        plotter.add_title(
            f"3D Depth Profile | t = {t + 1}/{d.nt}{clip_info} | {name}",
            font_size=10,
        )

        # Camera
        plotter.camera_position = "iso"
        plotter.camera.elevation = cfg.elevation
        plotter.camera.azimuth = cfg.azimuth
        plotter.camera.zoom(cfg.camera_zoom)

        # Legend
        legend_entries = [
            [d.element_names[i], d.colors[i].tolist()] for i in self.visible
        ]
        plotter.add_legend(
            legend_entries, bcolor="white", face="circle", size=(0.15, 0.22)
        )

    def render_frame_with_depth_profile(
        self, t: int, plotter: pv.Plotter
    ) -> None:
        """Render 3D isosurface + 2D depth profile chart."""
        self.render_frame(t, plotter)

        if self.config.depth_profile == "none":
            return

        d = self.data
        # Use clipped spatial/depth ranges for the chart
        x0 = self.clip_x[0] if self.clip_x else 0
        x1 = self.clip_x[1] if self.clip_x else d.nx - 1
        y0 = self.clip_y[0] if self.clip_y else 0
        y1 = self.clip_y[1] if self.clip_y else d.ny - 1
        z0 = self.clip_z[0] if self.clip_z else 0
        z1 = self.clip_z[1] if self.clip_z else d.n_depth - 1
        clipped_frame = d.get_frame(t)[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1, :]
        avg = clipped_frame.mean(axis=(0, 1))  # (cnz, n_elem)
        depth_clipped = d.depth[z0:z1 + 1]

        try:
            chart = pv.Chart2D(size=(0.35, 0.35), loc=(0.62, 0.02))
        except (AttributeError, TypeError):
            return

        for elem_idx in self.visible:
            c = tuple(int(v * 255) for v in d.colors[elem_idx][:3])
            chart.line(
                depth_clipped,
                avg[:, elem_idx] * 100,
                color=c,
                width=2.0,
                label=d.element_names[elem_idx],
            )

        chart.x_label = "Depth (A)"
        chart.y_label = "Conc. (%)"
        chart.x_range = [float(depth_clipped[0]), float(depth_clipped[-1])]
        chart.y_range = [0, 101]
        chart.grid = True
        chart.title = f"Avg. Depth Profile (t={t + 1})"
        plotter.add_chart(chart)


# ============================================================================
# MATLAB-compatible Isosurface Renderer (sp=2 faithful)
# ============================================================================


class MatlabIsosurfaceRenderer:
    """MATLAB EzDeP4D ``SurfacePatch == 2`` faithful renderer.

    Replicates the geometry and appearance of the MATLAB ``sp = 2``
    branch, unlike :class:`IsosurfaceRenderer` whose composite opaque
    isocaps trade fidelity for artifact suppression:

    - ``meshgrid(y, x, z)``: world X carries the data *y* pixel index
      (1-based), world Y carries *x*, world Z carries physical depth in
      Angstroms.  Mesh data is never normalised; the MATLAB axes
      default stretch-to-fill (``DataAspectRatio`` auto — EzDeP4D sets
      no ``daspect``/``pbaspect``) is replicated at the display level
      via the renderer's per-axis actor scale.
    - Per frame, per element: threshold = ``max(V) * isovalue_degree``
      (``isovn`` in MATLAB), ``isosurface`` -> marching-cubes contour,
      ``isocaps`` -> interpolated cap mesh on the six boundary faces.
    - Surface and caps share the element colour with
      ``FaceAlpha = 0.75 * face_alpha``, no edges, Gouraud shading.
    - ``ZDir = 'reverse'``: implemented by negating world Z (surface at
      0, deeper = more negative) when ``config.z_reversed``.
    - Fixed orthographic camera equivalent to ``view([-105 5])`` and a
      single directional light equivalent to ``lightangle(-115, 25)``,
      both set once in :meth:`setup` and never per frame.
    - Actors and their mappers are created once and reused: each frame
      only replaces the mesh data via ``copy_from`` (MATLAB updates
      ``.Faces`` / ``.Vertices`` on the same patch handles).

    MATLAB's ``divx/divy = 50`` block decomposition (a ``parfor``
    optimisation whose internal cap faces are deleted afterwards) is
    not replicated; computing on the full volume is visually
    equivalent.

    Measured against MATLAB R2023b
    ------------------------------
    Queried on an axes built exactly as ``EzDeP4D.m`` builds it
    (``ZDir='reverse'``, ``view([-105 5])``, both aspect-ratio modes
    left on ``auto``):

    - ``Projection = orthographic`` — matches the parallel projection
      set here.
    - ``SortMethod = depth`` — MATLAB resolves transparency by sorting
      objects back-to-front, *not* by depth peeling (which MATLAB has
      no equivalent of).  VTK does not sort, so ``matlab_depth_peeling``
      buys back the same order-independent result by another route.
    - ``PlotBoxAspectRatio`` depends only on the axes rectangle's aspect
      and the view direction, never on the data ranges (those are
      absorbed by ``DataAspectRatio``).  For a **square** axes it is
      ``[1 1 1]``, which is what the cube normalisation here produces.
      For a wide axes it is not: e.g. 327x220 px at ``view([-105 5])``
      gives ``[0.713 1 0.685]``, and 600x200 px gives ``[0.426 1 0.344]``.
      That non-square stretch is **not** derived automatically — set
      ``plot_box_aspect`` explicitly to reproduce a given tile.
    - ``lightangle.m`` composes its direction in *data* space and
      pre-multiplies by ``DataAspectRatio``; through the stretch this
      leaves the view-formula direction with **z negated** by
      ``ZDir='reverse'``.  Solved from flat-face calibration cubes
      (gray for ambient+diffuse, black for the pure specular term):
      effective display direction (-0.85, 0.36, -0.39) vs the formula's
      (-0.82, 0.38, -0.42); the ~2% residual is EzDeP4D calling
      ``lightangle`` *before* it sets the axis limits, so the
      DataAspectRatio it bakes in is the auto-limit one.  That quirk is
      not replicated.
    - MATLAB patch material (measured): AmbientStrength 0.3,
      DiffuseStrength 0.6, SpecularStrength 0.9, SpecularExponent 10,
      SpecularColorReflectance 1 (white highlight).  The specular lobe
      behaves Phong-like (``(R.V)^n``): the black-cube front face reads
      15/255, matching ``(R.V)^10`` with the flipped light, and VTK's
      highlight responds the same way once the direction is fixed.

    A frame-for-frame check against MATLAB's own ``isosurface`` /
    ``isocaps`` output (same data, same frame, square axes,
    ``FaceAlpha 0.75``) matches in cube proportions, layer boundaries
    and colour; depth peeling is what closes the remaining gap.
    """

    # animate() keeps one plotter for the whole run instead of one per frame
    persistent_plotter = True

    def __init__(self, data: DepthProfile4DData, config: AnimationConfig):
        if not HAS_PYVISTA:
            raise ImportError("PyVista required: pip install pyvista")
        self.data = data
        self.config = config

        # Visible element indices
        all_idx = set(range(data.n_elem))
        hidden = set(config.invisible_elements)
        self.visible = sorted(all_idx - hidden)

        # Clip ranges (interactive sliders; MATLAB itself never clips)
        self.clip_x: Optional[Tuple[int, int]] = None
        self.clip_y: Optional[Tuple[int, int]] = None
        self.clip_z: Optional[Tuple[int, int]] = None

        # MATLAB physical coordinates: x = 1:nx, y = 1:ny, z = depth (A)
        self._xv = np.arange(1, data.nx + 1, dtype=np.float64)
        self._yv = np.arange(1, data.ny + 1, dtype=np.float64)
        self._zv = np.asarray(data.depth, dtype=np.float64).ravel()
        if self._zv.size != data.n_depth:
            self._zv = np.arange(data.n_depth, dtype=np.float64) * data.step_size

        # Persistent render state (populated by setup)
        self._plotter: Optional["pv.Plotter"] = None
        self._placeholders: Dict[Tuple[int, str], "pv.PolyData"] = {}
        self._actor_names: Dict[Tuple[int, str], str] = {}
        self._title_actor = None
        self._chart = None

    # ------------------------------------------------------------------
    # Mesh computation (isosurface + isocaps)
    # ------------------------------------------------------------------

    @staticmethod
    def _degenerate_mesh() -> "pv.PolyData":
        """Zero-area stand-in so actors always have valid (invisible) input."""
        pts = np.zeros((3, 3))
        return pv.PolyData(pts, faces=np.array([3, 0, 1, 2]))

    @staticmethod
    def _isosurface(
        A: np.ndarray,
        wx: np.ndarray,
        wy: np.ndarray,
        wz: np.ndarray,
        threshold: float,
    ) -> Optional["pv.PolyData"]:
        """MATLAB ``isosurface`` equivalent — marching-cubes contour.

        Parameters
        ----------
        A : ndarray, shape (len(wx), len(wy), len(wz))
            Volume in world index order.
        wx, wy, wz : ndarray
            World coordinate vectors (physical units).
        threshold : float
            Iso-value.
        """
        if A.shape[0] < 2 or A.shape[1] < 2 or A.shape[2] < 2:
            return None
        grid = pv.RectilinearGrid(wx, wy, wz)
        grid.point_data["c"] = A.ravel(order="F")
        try:
            surf = grid.contour(
                [threshold], scalars="c", compute_normals=True
            )
        except Exception:
            return None
        return surf if surf.n_points else None

    @staticmethod
    def _isocap_face(
        S: np.ndarray,
        c1: np.ndarray,
        c2: np.ndarray,
        axis: int,
        pos: float,
        threshold: float,
    ) -> Optional["pv.PolyData"]:
        """Interpolated iso-cap on one boundary face (MATLAB ``isocaps``).

        Clips the boundary slice at the iso-value so the cap edge is
        interpolated between grid points, exactly like ``isocaps``
        (``'enclose' = 'above'``: the region with ``S >= threshold``
        is kept).

        Parameters
        ----------
        S : ndarray, shape (len(c1), len(c2))
            Concentration slice on the face.
        c1, c2 : ndarray
            In-plane world coordinate vectors.
        axis : int
            World normal axis of the face (0=X, 1=Y, 2=Z).
        pos : float
            World position of the face along *axis*.
        threshold : float
            Iso-value.
        """
        if S.shape[0] < 2 or S.shape[1] < 2:
            return None
        if not np.isfinite(S).any() or np.nanmax(S) < threshold:
            return None
        C1, C2 = np.meshgrid(c1, c2, indexing="ij")
        P = np.full(C1.shape, pos)
        if axis == 0:
            sg = pv.StructuredGrid(P, C1, C2)
        elif axis == 1:
            sg = pv.StructuredGrid(C1, P, C2)
        elif axis == 2:
            sg = pv.StructuredGrid(C1, C2, P)
        else:
            return None
        sg.point_data["c"] = S.ravel(order="F")
        try:
            # invert=False keeps the region where c >= threshold
            clipped = sg.clip_scalar(
                scalars="c", value=threshold, invert=False
            )
            if clipped.n_cells == 0:
                return None
            cap = clipped.extract_surface()
        except Exception:
            return None
        return cap if cap.n_points else None

    def _isocaps(
        self,
        A: np.ndarray,
        wx: np.ndarray,
        wy: np.ndarray,
        wz: np.ndarray,
        threshold: float,
    ) -> Optional["pv.PolyData"]:
        """Caps on all six boundary faces, merged into one mesh
        (MATLAB has a single ``isocaps`` patch per element)."""
        n0, n1, n2 = A.shape
        caps: List["pv.PolyData"] = []
        for axis, n in ((0, n0), (1, n1), (2, n2)):
            for idx in sorted({0, n - 1}):
                if axis == 0:
                    S, c1, c2 = A[idx, :, :], wy, wz
                    pos = float(wx[idx])
                elif axis == 1:
                    S, c1, c2 = A[:, idx, :], wx, wz
                    pos = float(wy[idx])
                else:
                    S, c1, c2 = A[:, :, idx], wx, wy
                    pos = float(wz[idx])
                cap = self._isocap_face(S, c1, c2, axis, pos, threshold)
                if cap is not None:
                    caps.append(cap)
        if not caps:
            return None
        merged = caps[0]
        for c in caps[1:]:
            merged = merged.merge(c)
        return merged

    def _clip_bounds(self) -> Tuple[int, int, int, int, int, int]:
        d = self.data
        x0 = self.clip_x[0] if self.clip_x else 0
        x1 = self.clip_x[1] if self.clip_x else d.nx - 1
        y0 = self.clip_y[0] if self.clip_y else 0
        y1 = self.clip_y[1] if self.clip_y else d.ny - 1
        z0 = self.clip_z[0] if self.clip_z else 0
        z1 = self.clip_z[1] if self.clip_z else d.n_depth - 1
        return x0, x1, y0, y1, z0, z1

    def compute_frame_meshes(
        self, t: int
    ) -> Dict[int, Tuple[Optional["pv.PolyData"], Optional["pv.PolyData"]]]:
        """Compute per-element ``(surface, caps)`` meshes for frame *t*.

        Returns
        -------
        dict
            Keyed by element index (visible elements only).  Meshes are
            in display coordinates: world X = data y index, world Y =
            data x index, world Z = physical depth in Angstroms
            (negated when ``config.z_reversed``).  Either entry may be
            *None* when the element has nothing to render.
        """
        d = self.data
        cfg = self.config
        x0, x1, y0, y1, z0, z1 = self._clip_bounds()
        frame = d.get_frame(t)[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1, :]
        xv = self._xv[x0:x1 + 1]
        yv = self._yv[y0:y1 + 1]
        zv = self._zv[z0:z1 + 1]

        out: Dict[int, Tuple[Optional["pv.PolyData"], Optional["pv.PolyData"]]] = {}
        for elem_idx in self.visible:
            V = np.nan_to_num(
                np.asarray(frame[:, :, :, elem_idx], dtype=np.float64),
                nan=0.0,
            )
            v_max = float(V.max()) if V.size else 0.0
            if v_max <= 0.0:
                out[elem_idx] = (None, None)
                continue
            # MATLAB isovn: per-frame, per-element max
            threshold = v_max * cfg.isovalue_degree

            # meshgrid(y, x, z): world X carries y, world Y carries x
            A = V.swapaxes(0, 1)  # (ny, nx, nz) in world (X, Y, Z) order
            surf = self._isosurface(A, yv, xv, zv, threshold)
            caps = self._isocaps(A, yv, xv, zv, threshold)

            if cfg.z_reversed:
                for mesh in (surf, caps):
                    if mesh is None or mesh.n_points == 0:
                        continue
                    mesh.points[:, 2] *= -1.0
                    if "Normals" in mesh.point_data:
                        mesh.point_data["Normals"][:, 2] *= -1.0
            out[elem_idx] = (surf, caps)
        return out

    # ------------------------------------------------------------------
    # Persistent scene (actors, camera, light created once)
    # ------------------------------------------------------------------

    @staticmethod
    def _direction_from_angles(az_deg: float, el_deg: float) -> np.ndarray:
        """MATLAB view/lightangle convention: azimuth about +Z measured
        from the -Y axis, elevation above the X-Y plane."""
        az = np.radians(az_deg)
        el = np.radians(el_deg)
        return np.array([
            np.sin(az) * np.cos(el),
            -np.cos(az) * np.cos(el),
            np.sin(el),
        ])

    def _display_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Full-volume display bounds — MATLAB fixes X/Y/ZLim before
        the animation loop, independent of interactive clips."""
        cfg = self.config
        wx0, wx1 = float(self._yv[0]), float(self._yv[-1])
        wy0, wy1 = float(self._xv[0]), float(self._xv[-1])
        z_min, z_max = float(self._zv.min()), float(self._zv.max())
        if cfg.z_reversed:
            wz0, wz1 = -z_max, -z_min
        else:
            wz0, wz1 = z_min, z_max
        return wx0, wx1, wy0, wy1, wz0, wz1

    def _is_initialized(self, plotter: "pv.Plotter") -> bool:
        return (
            self._plotter is plotter
            and bool(self._actor_names)
            and all(n in plotter.actors for n in self._actor_names.values())
        )

    def setup(self, plotter: "pv.Plotter") -> None:
        """One-time scene construction: actors, axes, camera, light."""
        d = self.data
        cfg = self.config

        wx0, wx1, wy0, wy1, wz0, wz1 = self._display_bounds()

        # --- Per-element actors (surface + caps), reused every frame ---
        self._placeholders = {}
        self._actor_names = {}
        alpha = 0.75 * cfg.face_alpha
        for elem_idx in self.visible:
            color = d.colors[elem_idx].tolist()
            for kind in ("surf", "cap"):
                placeholder = self._degenerate_mesh()
                name = f"matlab_iso_{kind}_{elem_idx}"
                actor = plotter.add_mesh(
                    placeholder,
                    color=color,
                    opacity=alpha,
                    show_edges=False,
                    ambient=cfg.matlab_ambient,
                    diffuse=cfg.matlab_diffuse,
                    specular=cfg.matlab_specular,
                    specular_power=cfg.matlab_specular_power,
                    name=name,
                    reset_camera=False,
                )
                # MATLAB `lighting gouraud`
                actor.prop.SetInterpolationToGouraud()
                # MATLAB SpecularColorReflectance = 1: the highlight takes
                # the *light's* colour, not the material's.  PyVista derives
                # every material colour from `color=`, which tints the
                # highlight and loses that white lift.
                actor.prop.specular_color = (1.0, 1.0, 1.0)
                if not cfg.lighting:
                    actor.prop.lighting = False
                self._placeholders[(elem_idx, kind)] = placeholder
                self._actor_names[(elem_idx, kind)] = name

        # --- Axes + title (MATLAB labels its X axis with the data y
        #     name and vice versa, matching meshgrid(y, x, z)) ---
        plotter.add_axes(
            xlabel="Y", ylabel="X", zlabel="Depth (A)", line_width=2
        )
        name_str = Path(d.source_path).stem if d.source_path else "GT"
        # Tuple position + viewport gives a vtkTextActor (SetInput works
        # for per-frame updates); a string position would give a
        # CornerAnnotation, which has no SetInput
        self._title_actor = plotter.add_text(
            f"3D Profile | t = 1/{d.nt} | {name_str}",
            position=(0.02, 0.95), viewport=True,
            font_size=10, name="matlab_iso_title",
        )

        # --- Transparency: match MATLAB's SortMethod='depth' ---
        # VTK's default translucent pass is unsorted, so stacked layers
        # composite in arbitrary order and go muddy.  Any opacity below
        # 1.0 triggers it — 0.95 looks no better than 0.75.  Depth peeling
        # makes the result order-independent, which is what MATLAB's
        # sorted patches give.  Skipped when fully opaque (no cost).
        if cfg.matlab_depth_peeling and alpha < 1.0:
            try:
                plotter.enable_depth_peeling(
                    number_of_peels=12, occlusion_ratio=0.0,
                )
            except Exception:
                pass

        # --- MATLAB stretch-to-fill (DataAspectRatio auto): normalise
        #     each axis at the actor level; mesh data stays physical ---
        if cfg.matlab_stretch_to_fill:
            box = cfg.plot_box_aspect or (1.0, 1.0, 1.0)
            sx = box[0] / (wx1 - wx0) if wx1 > wx0 else box[0]
            sy = box[1] / (wy1 - wy0) if wy1 > wy0 else box[1]
            sz = box[2] / (wz1 - wz0) if wz1 > wz0 else box[2]
        else:
            sx = sy = sz = 1.0
        self._scale = (sx, sy, sz)
        if (sx, sy, sz) != (1.0, 1.0, 1.0):
            plotter.set_scale(
                xscale=sx, yscale=sy, zscale=sz,
                reset_camera=False, render=False,
            )

        # Display-space bounds after the stretch (camera and light live
        # in display space, like MATLAB's normalised plot box)
        bounds_disp = (
            wx0 * sx, wx1 * sx, wy0 * sy, wy1 * sy, wz0 * sz, wz1 * sz,
        )
        center = np.array([
            (bounds_disp[0] + bounds_disp[1]) * 0.5,
            (bounds_disp[2] + bounds_disp[3]) * 0.5,
            (bounds_disp[4] + bounds_disp[5]) * 0.5,
        ])
        diag = float(np.linalg.norm([
            bounds_disp[1] - bounds_disp[0],
            bounds_disp[3] - bounds_disp[2],
            bounds_disp[5] - bounds_disp[4],
        ])) or 1.0

        # --- Camera: view([-105 5]), orthographic, fixed thereafter ---
        view_dir = self._direction_from_angles(cfg.azimuth, cfg.elevation)
        plotter.camera.focal_point = tuple(center)
        plotter.camera.position = tuple(center + view_dir * diag * 3.0)
        plotter.camera.up = (0.0, 0.0, 1.0)
        plotter.camera.enable_parallel_projection()
        plotter.reset_camera(render=False, bounds=bounds_disp)
        plotter.camera.zoom(cfg.camera_zoom)
        # Clipping planes sized once to the fixed display bounds —
        # off-screen renders do not auto-reset them for a manually
        # placed camera, and a stale near plane clips the front corner
        plotter.renderer.ResetCameraClippingRange(*bounds_disp)

        # --- Single fixed light: lightangle(-115, 25) ---
        if cfg.lighting:
            # lightangle.m builds its direction in MATLAB *data* space
            # (+z = depth) and pre-multiplies by DataAspectRatio, which
            # the stretch-to-fill cancels again.  What survives into
            # display space is the view-convention direction with z
            # negated by ZDir='reverse' (weighted by the plot box when
            # non-cubic).  Verified on R2023b flat-face calibration
            # cubes: without the z flip the top face receives spurious
            # diffuse light and the front faces lose their Phong
            # highlight (the light sits slightly *below* the display
            # horizon, exactly as MATLAB renders it).
            u = self._direction_from_angles(
                cfg.light_azimuth, cfg.light_elevation
            )
            box = cfg.plot_box_aspect or (1.0, 1.0, 1.0)
            z_sign = -1.0 if cfg.z_reversed else 1.0
            light_dir = np.array([
                u[0] * box[0], u[1] * box[1], z_sign * u[2] * box[2],
            ])
            light_dir /= np.linalg.norm(light_dir)
            plotter.remove_all_lights()
            light = pv.Light(
                position=tuple(center + light_dir * diag * 10.0),
                focal_point=tuple(center),
                light_type="scene light",
            )
            light.positional = False
            plotter.add_light(light)

        self._plotter = plotter

    @staticmethod
    def _update_placeholder(
        placeholder: "pv.PolyData", mesh: Optional["pv.PolyData"]
    ) -> None:
        """Replace the actor's mesh data in place (MATLAB's
        ``.Faces`` / ``.Vertices`` update on an existing patch)."""
        if mesh is None or mesh.n_points == 0:
            mesh = MatlabIsosurfaceRenderer._degenerate_mesh()
        try:
            placeholder.copy_from(mesh)
        except AttributeError:  # pyvista < 0.37
            placeholder.deep_copy(mesh)
        placeholder.Modified()

    def update_frame(self, t: int) -> None:
        """Update mesh data + title for frame *t*.  Camera and light
        are never touched here."""
        d = self.data
        meshes = self.compute_frame_meshes(t)
        for elem_idx in self.visible:
            surf, caps = meshes.get(elem_idx, (None, None))
            self._update_placeholder(
                self._placeholders[(elem_idx, "surf")], surf
            )
            self._update_placeholder(
                self._placeholders[(elem_idx, "cap")], caps
            )
        if self._title_actor is not None:
            name_str = Path(d.source_path).stem if d.source_path else "GT"
            try:
                self._title_actor.SetInput(
                    f"3D Profile | t = {t + 1}/{d.nt} | {name_str}"
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Renderer API (animate / interactive compatibility)
    # ------------------------------------------------------------------

    def render_frame(self, t: int, plotter: "pv.Plotter") -> None:
        """Render frame *t*, building the persistent scene on first use."""
        if not self._is_initialized(plotter):
            self.setup(plotter)
        self.update_frame(t)

    def can_fast_update(self) -> bool:
        """Time-only updates always reuse the existing actors."""
        return self._plotter is not None and self._is_initialized(self._plotter)

    def update_time_fast(self, t: int, plotter: "pv.Plotter") -> None:
        self.render_frame(t, plotter)
        plotter.render()

    def render_frame_with_depth_profile(
        self, t: int, plotter: "pv.Plotter"
    ) -> None:
        """Render 3D scene + 2D depth profile chart."""
        self.render_frame(t, plotter)

        if self.config.depth_profile == "none":
            return

        if self._chart is not None:
            try:
                plotter.remove_chart(self._chart)
            except Exception:
                pass
            self._chart = None

        d = self.data
        x0, x1, y0, y1, z0, z1 = self._clip_bounds()
        clipped_frame = d.get_frame(t)[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1, :]
        avg = clipped_frame.mean(axis=(0, 1))  # (cnz, n_elem)
        depth_clipped = d.depth[z0:z1 + 1]

        try:
            chart = pv.Chart2D(size=(0.35, 0.35), loc=(0.62, 0.02))
        except (AttributeError, TypeError):
            return

        for elem_idx in self.visible:
            c = tuple(int(v * 255) for v in d.colors[elem_idx][:3])
            chart.line(
                depth_clipped,
                avg[:, elem_idx] * 100,
                color=c,
                width=2.0,
                label=d.element_names[elem_idx],
            )
        chart.x_label = "Depth (A)"
        chart.y_label = "Conc. (%)"
        chart.x_range = [float(depth_clipped[0]), float(depth_clipped[-1])]
        chart.y_range = [0, 101]
        chart.grid = True
        chart.title = f"Avg. Depth Profile (t={t + 1})"
        plotter.add_chart(chart)
        self._chart = chart


# ============================================================================
# Slice Renderer (sp=1)
# ============================================================================


class SliceRenderer:
    """MATLAB EzDeP4D ``sp=1`` style renderer — 6-face boundary textures.

    Instead of 3D isosurfaces, renders the bounding box faces as flat
    textured quads with concentration-weighted RGB colour per cell.
    Much faster than isosurface extraction and ideal for interactive use.
    """

    def __init__(self, data: DepthProfile4DData, config: AnimationConfig):
        if not HAS_PYVISTA:
            raise ImportError("PyVista required: pip install pyvista")
        self.data = data
        self.config = config

        # Visible element indices
        all_idx = set(range(data.n_elem))
        hidden = set(config.invisible_elements)
        self.visible = sorted(all_idx - hidden)

        # Clip ranges (same interface as IsosurfaceRenderer)
        self.clip_x: Optional[Tuple[int, int]] = None
        self.clip_y: Optional[Tuple[int, int]] = None
        self.clip_z: Optional[Tuple[int, int]] = None

        # Cache for in-place updates (populated by render_frame)
        self._cached_meshes: List["pv.PolyData"] = []
        self._cached_face_defs: List[Tuple] = []  # (axis, face_idx)
        self._cached_clip: Optional[Tuple] = None  # clip state snapshot
        self._cached_title = None  # vtkTextActor

    def _conc_to_rgb(self, conc_2d: np.ndarray) -> np.ndarray:
        """Convert (na, nb, n_vis) concentration to (na, nb, 3) RGB.

        Each cell's colour is a concentration-weighted average of the
        visible element colours.
        """
        colors_vis = self.data.colors[self.visible]  # (n_vis, 3)
        total = conc_2d.sum(axis=2, keepdims=True).clip(min=1e-10)
        weights = conc_2d / total  # (na, nb, n_vis)
        return np.einsum("ije,ec->ijc", weights, colors_vis)  # (na, nb, 3)

    @staticmethod
    def _make_face_mesh(
        rgb: np.ndarray,
        axis: int,
        pos: float,
        sp: Tuple[float, float],
        shape: Tuple[int, int],
        threshold: float = 0.01,
    ) -> Optional["pv.PolyData"]:
        """Build a flat quad mesh for one boundary face.

        Parameters
        ----------
        rgb : ndarray, shape (na, nb, 3)
            Per-cell RGB colour (0-1).
        axis : int
            Normal axis (0=X, 1=Y, 2=Z).
        pos : float
            Position along the normal axis.
        sp : tuple[float, float]
            Cell spacing ``(sa, sb)`` for the two in-plane axes.
        shape : tuple[int, int]
            Grid shape ``(na, nb)`` (number of cells in each in-plane axis).
        threshold : float
            Minimum total concentration to show a cell.
        """
        na, nb = shape
        if na < 1 or nb < 1:
            return None

        # Cell-centre total concentration for masking
        total = rgb.sum(axis=2)  # proxy: bright = filled
        mask = total > threshold  # (na, nb)
        n_cells = int(mask.sum())
        if n_cells == 0:
            return None

        sa, sb = sp
        ia, ib = np.where(mask)  # (n_cells,) each

        # Vectorised quad construction — no Python loop
        a0 = ia.astype(np.float32) * sa
        a1 = (ia + 1).astype(np.float32) * sa
        b0 = ib.astype(np.float32) * sb
        b1 = (ib + 1).astype(np.float32) * sb
        p = np.full(n_cells, pos, dtype=np.float32)

        # 4 corners per cell: shape (n_cells, 4, 3)
        if axis == 2:    # Z-face: in-plane = (X, Y)
            pts = np.stack([
                np.column_stack([a0, b0, p]),
                np.column_stack([a1, b0, p]),
                np.column_stack([a1, b1, p]),
                np.column_stack([a0, b1, p]),
            ], axis=1)
        elif axis == 0:  # X-face: in-plane = (Y, Z)
            pts = np.stack([
                np.column_stack([p, a0, b0]),
                np.column_stack([p, a1, b0]),
                np.column_stack([p, a1, b1]),
                np.column_stack([p, a0, b1]),
            ], axis=1)
        elif axis == 1:  # Y-face: in-plane = (X, Z)
            pts = np.stack([
                np.column_stack([a0, p, b0]),
                np.column_stack([a1, p, b0]),
                np.column_stack([a1, p, b1]),
                np.column_stack([a0, p, b1]),
            ], axis=1)
        else:
            return None

        pts_flat = pts.reshape(-1, 3)  # (n_cells*4, 3)

        # Face connectivity: [4, v0, v1, v2, v3] per cell
        base = np.arange(n_cells, dtype=np.int32) * 4
        faces = np.column_stack([
            np.full(n_cells, 4, dtype=np.int32),
            base, base + 1, base + 2, base + 3,
        ])

        mesh = pv.PolyData(pts_flat, faces=faces.ravel())
        mesh.cell_data["RGB"] = (rgb[ia, ib] * 255).astype(np.uint8)
        return mesh

    @staticmethod
    def _make_full_face_mesh(
        axis: int,
        pos: float,
        shape: Tuple[int, int],
        sp: Tuple[float, float],
    ) -> Optional["pv.PolyData"]:
        """Build a flat quad mesh covering ALL cells (no threshold filter).

        Same geometry layout as ``_make_face_mesh`` but includes every
        cell in the grid — designed for the pre-allocated RGBA mode
        where visibility is controlled by the alpha channel rather than
        cell omission.
        """
        na, nb = shape
        if na < 1 or nb < 1:
            return None

        sa, sb = sp
        n_cells = na * nb

        # Grid indices for all cells (C-order ravel: ia varies slowest)
        ia_grid, ib_grid = np.mgrid[0:na, 0:nb]
        ia = ia_grid.ravel().astype(np.float32)
        ib = ib_grid.ravel().astype(np.float32)

        a0 = ia * sa
        a1 = (ia + 1) * sa
        b0 = ib * sb
        b1 = (ib + 1) * sb
        p = np.full(n_cells, pos, dtype=np.float32)

        if axis == 2:    # Z-face: in-plane = (X, Y)
            pts = np.stack([
                np.column_stack([a0, b0, p]),
                np.column_stack([a1, b0, p]),
                np.column_stack([a1, b1, p]),
                np.column_stack([a0, b1, p]),
            ], axis=1)
        elif axis == 0:  # X-face: in-plane = (Y, Z)
            pts = np.stack([
                np.column_stack([p, a0, b0]),
                np.column_stack([p, a1, b0]),
                np.column_stack([p, a1, b1]),
                np.column_stack([p, a0, b1]),
            ], axis=1)
        elif axis == 1:  # Y-face: in-plane = (X, Z)
            pts = np.stack([
                np.column_stack([a0, p, b0]),
                np.column_stack([a1, p, b0]),
                np.column_stack([a1, p, b1]),
                np.column_stack([a0, p, b1]),
            ], axis=1)
        else:
            return None

        pts_flat = pts.reshape(-1, 3)  # (n_cells*4, 3)
        base = np.arange(n_cells, dtype=np.int32) * 4
        faces = np.column_stack([
            np.full(n_cells, 4, dtype=np.int32),
            base, base + 1, base + 2, base + 3,
        ])

        return pv.PolyData(pts_flat, faces=faces.ravel())

    def render_frame(self, t: int, plotter: pv.Plotter) -> None:
        """Render a single frame as 6-face boundary slices (MATLAB sp=1 style).

        Parameters
        ----------
        t : int
            Time frame index.
        plotter : pv.Plotter
            PyVista plotter (already cleared by caller).
        """
        d = self.data
        cfg = self.config

        if cfg.lighting:
            plotter.enable_3_lights()

        # Z-spacing for display
        spatial_extent = max(d.nx, d.ny)
        if d.n_depth > 1:
            z_sp = spatial_extent / (d.n_depth - 1)
        else:
            z_sp = 1.0

        # Apply clip ranges
        frame_full = d.get_frame(t)
        x0 = self.clip_x[0] if self.clip_x else 0
        x1 = self.clip_x[1] if self.clip_x else d.nx - 1
        y0 = self.clip_y[0] if self.clip_y else 0
        y1 = self.clip_y[1] if self.clip_y else d.ny - 1
        z0 = self.clip_z[0] if self.clip_z else 0
        z1 = self.clip_z[1] if self.clip_z else d.n_depth - 1
        frame = frame_full[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1, :]
        cnx, cny, cnz = frame.shape[:3]

        x_max = float(cnx - 1)
        y_max = float(cny - 1)
        z_max = float(cnz - 1) * z_sp

        # Cell sizes: n data points span [0, max], so cell_size = max / n
        # (ensures n cells exactly fill the bounding box with no overshoot)
        cx = x_max / cnx if cnx > 1 else 1.0
        cy = y_max / cny if cny > 1 else 1.0
        cz = z_max / cnz if cnz > 1 else z_sp

        # Concentration for visible elements only
        vis_frame = frame[:, :, :, self.visible]  # (cnx, cny, cnz, n_vis)

        # --- 6 boundary faces ---
        face_defs = [
            # (axis, face_idx, pos, in-plane shape, spacing)
            # Z-faces (top/bottom): in-plane = (X, Y)
            (2, 0, 0.0, (cnx, cny), (cx, cy)),
            (2, cnz - 1, z_max, (cnx, cny), (cx, cy)),
            # X-faces (left/right): in-plane = (Y, Z)
            (0, 0, 0.0, (cny, cnz), (cy, cz)),
            (0, cnx - 1, x_max, (cny, cnz), (cy, cz)),
            # Y-faces (front/back): in-plane = (X, Z)
            (1, 0, 0.0, (cnx, cnz), (cx, cz)),
            (1, cny - 1, y_max, (cnx, cnz), (cx, cz)),
        ]

        self._cached_meshes.clear()
        self._cached_face_defs.clear()
        # Use threshold=-1 to include ALL cells (stable topology for in-place updates)
        for axis, fi, pos, shape, sp in face_defs:
            # Extract 2D slice
            if axis == 2:
                sl = vis_frame[:, :, fi, :]  # (cnx, cny, n_vis)
            elif axis == 0:
                sl = vis_frame[fi, :, :, :]  # (cny, cnz, n_vis)
            elif axis == 1:
                sl = vis_frame[:, fi, :, :]  # (cnx, cnz, n_vis)
            else:
                continue

            rgb = self._conc_to_rgb(sl)

            mesh = self._make_face_mesh(
                rgb, axis, pos, sp, shape, threshold=-1,
            )
            if mesh is not None and mesh.n_points > 0:
                if cfg.z_reversed:
                    mesh.points[:, 2] = -mesh.points[:, 2]
                plotter.add_mesh(
                    mesh, scalars="RGB", rgb=True,
                    opacity=1.0, smooth_shading=False,
                )
                self._cached_meshes.append(mesh)
                self._cached_face_defs.append((axis, fi))

        # --- Wireframe bounding box for spatial reference ---
        if cnx >= 2 and cny >= 2 and cnz >= 2:
            z_top = 0.0
            z_bot = z_max
            if cfg.z_reversed:
                z_top, z_bot = -z_top, -z_bot
            box = pv.Box(bounds=[0, x_max, 0, y_max, z_bot, z_top])
            plotter.add_mesh(
                box, style="wireframe", color="grey",
                line_width=1, opacity=0.3,
            )

        # --- Axes and title ---
        plotter.add_axes(
            xlabel="X", ylabel="Y", zlabel="Depth (A)", line_width=2,
        )
        name = Path(d.source_path).stem if d.source_path else "GT"
        clip_info = ""
        if self.clip_x or self.clip_y or self.clip_z:
            clip_info = f" | clip [{cnx}x{cny}x{cnz}]"
        self._cached_title = plotter.add_title(
            f"Slice View | t = {t + 1}/{d.nt}{clip_info} | {name}",
            font_size=10,
        )
        self._cached_clip = (self.clip_x, self.clip_y, self.clip_z)

        # Camera
        plotter.camera_position = "iso"
        plotter.camera.elevation = cfg.elevation
        plotter.camera.azimuth = cfg.azimuth
        plotter.camera.zoom(cfg.camera_zoom)

        # Legend
        legend_entries = [
            [d.element_names[i], d.colors[i].tolist()] for i in self.visible
        ]
        plotter.add_legend(
            legend_entries, bcolor="white", face="circle", size=(0.15, 0.22),
        )

    def can_fast_update(self) -> bool:
        """Check if fast in-place update is possible (same clip state)."""
        return (
            len(self._cached_meshes) > 0
            and self._cached_clip == (self.clip_x, self.clip_y, self.clip_z)
        )

    def update_time_fast(self, t: int, plotter: "pv.Plotter") -> None:
        """Fast path: update only RGB colours on cached meshes (no clear).

        Only valid when clip ranges haven't changed since the last
        ``render_frame`` call.  Check ``can_fast_update()`` first.
        """
        d = self.data
        x0 = self.clip_x[0] if self.clip_x else 0
        x1 = self.clip_x[1] if self.clip_x else d.nx - 1
        y0 = self.clip_y[0] if self.clip_y else 0
        y1 = self.clip_y[1] if self.clip_y else d.ny - 1
        z0 = self.clip_z[0] if self.clip_z else 0
        z1 = self.clip_z[1] if self.clip_z else d.n_depth - 1
        frame = d.get_frame(t)[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1, :]
        vis_frame = frame[:, :, :, self.visible]

        for mesh, (axis, fi) in zip(self._cached_meshes, self._cached_face_defs):
            if axis == 2:
                sl = vis_frame[:, :, fi, :]
            elif axis == 0:
                sl = vis_frame[fi, :, :, :]
            elif axis == 1:
                sl = vis_frame[:, fi, :, :]
            else:
                continue
            rgb = self._conc_to_rgb(sl)
            mesh.cell_data["RGB"] = (rgb.reshape(-1, 3) * 255).astype(np.uint8)
            # Mark VTK pipeline as dirty so the mapper picks up the change
            mesh.Modified()

        # Update title text
        if self._cached_title is not None:
            name = Path(d.source_path).stem if d.source_path else "GT"
            clip_info = ""
            if self.clip_x or self.clip_y or self.clip_z:
                cnx, cny, cnz = frame.shape[:3]
                clip_info = f" | clip [{cnx}x{cny}x{cnz}]"
            try:
                # vtkCornerAnnotation uses SetText(corner, text)
                self._cached_title.SetText(
                    2, f"Slice View | t = {t + 1}/{d.nt}{clip_info} | {name}"
                )
            except Exception:
                pass

        plotter.render()

    def render_frame_with_depth_profile(
        self, t: int, plotter: pv.Plotter,
    ) -> None:
        """Render slice view + 2D depth profile chart."""
        self.render_frame(t, plotter)

        if self.config.depth_profile == "none":
            return

        d = self.data
        x0 = self.clip_x[0] if self.clip_x else 0
        x1 = self.clip_x[1] if self.clip_x else d.nx - 1
        y0 = self.clip_y[0] if self.clip_y else 0
        y1 = self.clip_y[1] if self.clip_y else d.ny - 1
        z0 = self.clip_z[0] if self.clip_z else 0
        z1 = self.clip_z[1] if self.clip_z else d.n_depth - 1
        clipped_frame = d.get_frame(t)[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1, :]
        avg = clipped_frame.mean(axis=(0, 1))
        depth_clipped = d.depth[z0:z1 + 1]

        try:
            chart = pv.Chart2D(size=(0.35, 0.35), loc=(0.62, 0.02))
        except (AttributeError, TypeError):
            return

        for elem_idx in self.visible:
            c = tuple(int(v * 255) for v in d.colors[elem_idx][:3])
            chart.line(
                depth_clipped,
                avg[:, elem_idx] * 100,
                color=c, width=2.0,
                label=d.element_names[elem_idx],
            )
        chart.x_label = "Depth (A)"
        chart.y_label = "Conc. (%)"
        chart.x_range = [float(depth_clipped[0]), float(depth_clipped[-1])]
        chart.y_range = [0, 101]
        chart.grid = True
        chart.title = f"Avg. Depth Profile (t={t + 1})"
        plotter.add_chart(chart)

    # ------------------------------------------------------------------
    # Pre-allocated RGBA mode (for interactive viewer)
    # ------------------------------------------------------------------

    def init_prealloc(self, plotter: "pv.Plotter") -> None:
        """Pre-allocate 6 full-size face meshes for fast in-place updates.

        Call once after creating the plotter.  All subsequent updates use
        ``update_all_fast()`` which only modifies RGBA cell data and
        vertex positions — no mesh creation or deletion.

        Each face covers the full data grid; visibility is controlled by
        the alpha channel (0 = hidden, 255 = visible).  Movable faces
        (Z-top, X-hi, Y-hi) have their vertex positions updated to
        reflect the current clip state.
        """
        d = self.data
        cfg = self.config

        if cfg.lighting:
            plotter.enable_3_lights()

        spatial_extent = max(d.nx, d.ny)
        z_sp = spatial_extent / (d.n_depth - 1) if d.n_depth > 1 else 1.0

        x_max = float(d.nx - 1)
        y_max = float(d.ny - 1)
        z_max = float(d.n_depth - 1) * z_sp

        cx = x_max / d.nx if d.nx > 1 else 1.0
        cy = y_max / d.ny if d.ny > 1 else 1.0
        cz = z_max / d.n_depth if d.n_depth > 1 else z_sp

        # Store display parameters for update_all_fast
        self._pa_z_sp = z_sp
        self._pa_cx, self._pa_cy, self._pa_cz = cx, cy, cz
        self._pa_x_max, self._pa_y_max, self._pa_z_max = x_max, y_max, z_max

        # 6 face specs: (name, axis, initial_pos, grid_shape, cell_spacing)
        face_specs = [
            ("z_lo", 2, 0.0, (d.nx, d.ny), (cx, cy)),
            ("z_hi", 2, z_max, (d.nx, d.ny), (cx, cy)),
            ("x_lo", 0, 0.0, (d.ny, d.n_depth), (cy, cz)),
            ("x_hi", 0, x_max, (d.ny, d.n_depth), (cy, cz)),
            ("y_lo", 1, 0.0, (d.nx, d.n_depth), (cx, cz)),
            ("y_hi", 1, y_max, (d.nx, d.n_depth), (cx, cz)),
        ]

        self._pa_faces: List[Tuple[str, int, "pv.PolyData", Tuple[int, int]]] = []

        for name, axis, pos, shape, sp in face_specs:
            mesh = self._make_full_face_mesh(axis, pos, shape, sp)
            if mesh is None:
                continue

            # Initialise fully transparent
            n_cells = mesh.n_cells
            rgba = np.zeros((n_cells, 4), dtype=np.uint8)
            mesh.cell_data["RGBA"] = rgba

            if cfg.z_reversed:
                mesh.points[:, 2] = -mesh.points[:, 2]

            plotter.add_mesh(
                mesh, scalars="RGBA", rgba=True,
                opacity=1.0, smooth_shading=False,
            )
            self._pa_faces.append((name, axis, mesh, shape))

        # Wireframe bounding box
        z_top = 0.0
        z_bot = z_max
        if cfg.z_reversed:
            z_top, z_bot = -z_top, -z_bot
        z_lo_b, z_hi_b = min(z_top, z_bot), max(z_top, z_bot)
        self._pa_box = pv.Box(bounds=[0, x_max, 0, y_max, z_lo_b, z_hi_b])
        plotter.add_mesh(
            self._pa_box, style="wireframe", color="grey",
            line_width=1, opacity=0.3,
        )

        # Axes
        plotter.add_axes(
            xlabel="X", ylabel="Y", zlabel="Depth (A)", line_width=2,
        )

        # Title (stored for in-place text updates via SetInput)
        name_str = Path(d.source_path).stem if d.source_path else "GT"
        self._pa_title = plotter.add_text(
            f"Slice View | t = 1/{d.nt} | {name_str}",
            position=(0.02, 0.01), font_size=10, name="_pa_title",
        )

        # Camera
        plotter.camera_position = "iso"
        plotter.camera.elevation = cfg.elevation
        plotter.camera.azimuth = cfg.azimuth
        plotter.camera.zoom(cfg.camera_zoom)

        # Legend
        legend_entries = [
            [d.element_names[i], d.colors[i].tolist()] for i in self.visible
        ]
        plotter.add_legend(
            legend_entries, bcolor="white", face="circle", size=(0.15, 0.22),
        )

    def update_all_fast(
        self,
        t: int,
        x_lo: int,
        x_hi: int,
        y_lo: int,
        y_hi: int,
        z_peel: int,
        plotter: "pv.Plotter",
    ) -> None:
        """Fast in-place update of all pre-allocated faces.

        Updates RGBA colours (concentration + alpha mask) and vertex
        positions for movable faces.  No mesh creation or deletion —
        all changes are data-only writes + ``Modified()`` calls.

        Parameters
        ----------
        t : int
            Time frame index.
        x_lo, x_hi : int
            X clip range (0-based pixel indices, inclusive).
        y_lo, y_hi : int
            Y clip range (0-based pixel indices, inclusive).
        z_peel : int
            Z peel position (0-based depth index; 0 = full depth).
        plotter : pv.Plotter
            The plotter instance (for ``render()`` call).
        """
        d = self.data
        cfg = self.config
        frame = d.get_frame(t)  # (nx, ny, n_depth, n_elem)
        vis_frame = frame[:, :, :, self.visible]

        cx = self._pa_cx
        cy = self._pa_cy
        cz = self._pa_cz
        z_rev = cfg.z_reversed

        for name, axis, mesh, shape in self._pa_faces:
            na, nb = shape

            # --- Extract 2D concentration slice ---
            if name == "z_lo":
                z_idx = min(z_peel, d.n_depth - 1)
                sl = vis_frame[:, :, z_idx, :]
            elif name == "z_hi":
                sl = vis_frame[:, :, -1, :]
            elif name == "x_lo":
                xi = min(x_lo, d.nx - 1)
                sl = vis_frame[xi, :, :, :]
            elif name == "x_hi":
                xi = min(x_hi, d.nx - 1)
                sl = vis_frame[xi, :, :, :]
            elif name == "y_lo":
                yi = min(y_lo, d.ny - 1)
                sl = vis_frame[:, yi, :, :]
            elif name == "y_hi":
                yi = min(y_hi, d.ny - 1)
                sl = vis_frame[:, yi, :, :]
            else:
                continue

            # --- RGB from concentration ---
            rgb = self._conc_to_rgb(sl)  # (na, nb, 3)

            # --- Alpha mask (vectorised boolean arrays) ---
            idx_a = np.arange(na)
            idx_b = np.arange(nb)

            if name in ("z_lo", "z_hi"):
                # Z-face grid = (nx, ny): mask by X and Y range
                mask_a = (idx_a >= x_lo) & (idx_a <= x_hi)
                mask_b = (idx_b >= y_lo) & (idx_b <= y_hi)
            elif name in ("x_lo", "x_hi"):
                # X-face grid = (ny, n_depth): mask by Y range and Z peel
                mask_a = (idx_a >= y_lo) & (idx_a <= y_hi)
                mask_b = idx_b >= z_peel
            elif name in ("y_lo", "y_hi"):
                # Y-face grid = (nx, n_depth): mask by X range and Z peel
                mask_a = (idx_a >= x_lo) & (idx_a <= x_hi)
                mask_b = idx_b >= z_peel
            else:
                mask_a = np.ones(na, dtype=bool)
                mask_b = np.ones(nb, dtype=bool)

            alpha = mask_a[:, None] & mask_b[None, :]  # (na, nb)

            # --- Write RGBA ---
            rgba = np.empty((na * nb, 4), dtype=np.uint8)
            rgba[:, :3] = (rgb.reshape(-1, 3) * 255).astype(np.uint8)
            rgba[:, 3] = alpha.ravel().astype(np.uint8) * 255
            mesh.cell_data["RGBA"] = rgba

            # --- Move vertices for movable faces ---
            if name == "z_lo":
                new_z = z_peel * cz
                mesh.points[:, 2] = -new_z if z_rev else new_z
            elif name == "x_lo":
                new_x = x_lo * cx
                mesh.points[:, 0] = new_x
            elif name == "x_hi":
                new_x = min((x_hi + 1) * cx, self._pa_x_max)
                mesh.points[:, 0] = new_x
            elif name == "y_lo":
                new_y = y_lo * cy
                mesh.points[:, 1] = new_y
            elif name == "y_hi":
                new_y = min((y_hi + 1) * cy, self._pa_y_max)
                mesh.points[:, 1] = new_y

            mesh.Modified()

        # --- Update wireframe box bounds ---
        if self._pa_box is not None:
            x_lo_pos = x_lo * cx
            x_hi_pos = min((x_hi + 1) * cx, self._pa_x_max)
            y_lo_pos = y_lo * cy
            y_hi_pos = min((y_hi + 1) * cy, self._pa_y_max)
            z_lo_pos = z_peel * cz
            z_hi_pos = self._pa_z_max

            if z_rev:
                zb_lo, zb_hi = -z_hi_pos, -z_lo_pos
            else:
                zb_lo, zb_hi = z_lo_pos, z_hi_pos

            try:
                new_box = pv.Box(bounds=[
                    x_lo_pos, x_hi_pos, y_lo_pos, y_hi_pos,
                    min(zb_lo, zb_hi), max(zb_lo, zb_hi),
                ])
                if self._pa_box.n_points == new_box.n_points:
                    self._pa_box.points[:] = new_box.points
                    self._pa_box.Modified()
            except Exception:
                pass

        # --- Update title ---
        cnx = x_hi - x_lo + 1
        cny = y_hi - y_lo + 1
        cnz = d.n_depth - z_peel
        name_str = Path(d.source_path).stem if d.source_path else "GT"
        clip_info = ""
        is_clipped = (
            x_lo > 0 or x_hi < d.nx - 1
            or y_lo > 0 or y_hi < d.ny - 1
            or z_peel > 0
        )
        if is_clipped:
            clip_info = f" | clip [{cnx}x{cny}x{cnz}]"
        try:
            self._pa_title.SetInput(
                f"Slice View | t = {t + 1}/{d.nt}{clip_info} | {name_str}"
            )
        except Exception:
            pass

        plotter.render()


# ============================================================================
# Renderer factory
# ============================================================================


def _make_renderer(
    data: DepthProfile4DData, config: AnimationConfig,
) -> "IsosurfaceRenderer | SliceRenderer | MatlabIsosurfaceRenderer":
    """Create the appropriate renderer based on ``config.render_mode``."""
    if config.render_mode == "slice":
        return SliceRenderer(data, config)
    if config.render_mode == "matlab":
        return MatlabIsosurfaceRenderer(data, config)
    return IsosurfaceRenderer(data, config)


# ============================================================================
# Animation
# ============================================================================


def animate(
    data: DepthProfile4DData,
    config: Optional[AnimationConfig] = None,
) -> str:
    """Generate 4D animation as MP4 video.

    Parameters
    ----------
    data : DepthProfile4DData
        Loaded 5D data.
    config : AnimationConfig, optional
        Rendering configuration.  Defaults to sensible values.

    Returns
    -------
    str
        Path to the output MP4 file.
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista required: pip install pyvista")

    if config is None:
        config = AnimationConfig()

    renderer = _make_renderer(data, config)

    # Time sequence: forward, then backward
    t_forward = list(range(data.nt))
    if config.forward_backward:
        t_sequence = t_forward + t_forward[-2:0:-1]
    else:
        t_sequence = t_forward

    total_frames = len(t_sequence)
    if not HAS_IMAGEIO:
        raise ImportError(
            "imageio is required to write animations "
            "(pip install imageio imageio-ffmpeg); deppro.visualize_4d "
            "imports without it by design")

    output = str(Path(config.output_path).resolve())

    print(
        f"Rendering {total_frames} frames "
        f"({data.nt} time steps, "
        f"{'forward+backward' if config.forward_backward else 'forward'}) "
        f"→ {output}"
    )

    # Incremental write: one frame at a time to avoid OOM on large datasets
    writer = imageio.get_writer(
        output,
        fps=config.fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=config.video_quality,
    )

    if getattr(renderer, "persistent_plotter", False):
        # MATLAB-style single plotter: actors, camera and light are set
        # up once; each frame only replaces the mesh data.
        plotter = pv.Plotter(
            off_screen=True,
            window_size=config.window_size,
        )
        plotter.set_background(config.background_color)

        for i, t in enumerate(t_sequence):
            if config.depth_profile != "none":
                renderer.render_frame_with_depth_profile(t, plotter)
            else:
                renderer.render_frame(t, plotter)

            # screenshot() does not re-render an already-rendered
            # plotter, so flush the in-place mesh updates explicitly
            plotter.render()
            img = plotter.screenshot(return_img=True)
            writer.append_data(img)
            del img

            if (i + 1) % 50 == 0 or (i + 1) == total_frames:
                print(f"  Frame {i + 1}/{total_frames} (t={t})")

        plotter.close()
    else:
        for i, t in enumerate(t_sequence):
            plotter = pv.Plotter(
                off_screen=True,
                window_size=config.window_size,
            )
            plotter.set_background(config.background_color)
            # Depth peeling disabled: composite opaque isocaps eliminate the
            # transparency-overlap artifact that depth peeling was meant to fix,
            # and depth peeling can run out of GPU memory over long animations.

            if config.depth_profile != "none":
                renderer.render_frame_with_depth_profile(t, plotter)
            else:
                renderer.render_frame(t, plotter)

            img = plotter.screenshot(return_img=True)
            writer.append_data(img)
            plotter.close()
            del plotter, img

            # Periodic GC to help release VTK/Metal GPU resources
            if (i + 1) % 20 == 0:
                gc.collect()

            if (i + 1) % 50 == 0 or (i + 1) == total_frames:
                print(f"  Frame {i + 1}/{total_frames} (t={t})")

    writer.close()
    data.close()  # Clean up temp H5 if lazy mode
    print(f"Saved: {output}  ({total_frames} frames)")
    return output


# ============================================================================
# Interactive viewer
# ============================================================================


def interactive(
    data: DepthProfile4DData,
    config: Optional[AnimationConfig] = None,
) -> None:
    """Open interactive 3D viewer with time + X/Y/Z clip sliders.

    Parameters
    ----------
    data : DepthProfile4DData
        Loaded 5D data.
    config : AnimationConfig, optional
        Rendering configuration.

    Notes
    -----
    Requires a display (not off-screen).  Frame rebuild time grows with
    the grid size.

    Sliders
    -------
    - **Time** (top): select time frame
    - **X cut** (bottom-left): clip X range ``[0, value]``
    - **Y cut** (bottom-centre): clip Y range ``[0, value]``
    - **Z cut** (bottom-right): clip Z (depth) range ``[0, value]``

    Moving any clip slider to the maximum shows the full range (no clip).
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista required: pip install pyvista")

    if config is None:
        config = AnimationConfig()

    renderer = _make_renderer(data, config)

    plotter = pv.Plotter(window_size=config.window_size)
    plotter.set_background(config.background_color)

    # Pre-allocated fast path (SliceRenderer only)
    use_prealloc = hasattr(renderer, "init_prealloc")

    state = {
        "t": 0,
        "x_lo": 0,
        "x_hi": data.nx - 1,
        "y_lo": 0,
        "y_hi": data.ny - 1,
        "z_cut": 0,  # 0 = full depth, >0 = peel surface layers
        "actor_keys": [],  # keys in plotter.actors (non-prealloc path only)
    }

    if use_prealloc:
        # ==== Pre-allocated path: all 6 faces created once, RGBA in-place ====
        renderer.init_prealloc(plotter)

        def _pa_update() -> None:
            """Dispatch current state to the fast renderer."""
            renderer.update_all_fast(
                state["t"],
                state["x_lo"], state["x_hi"],
                state["y_lo"], state["y_hi"],
                state["z_cut"], plotter,
            )

        _pa_update()  # initial render

        def _on_time_change(value: float) -> None:
            t = int(round(value))
            if t == state["t"]:
                return
            state["t"] = t
            _pa_update()

        def _on_x_lo(value: float) -> None:
            v = int(round(value))
            if v == state["x_lo"]:
                return
            state["x_lo"] = v
            _pa_update()

        def _on_x_hi(value: float) -> None:
            v = int(round(value))
            if v == state["x_hi"]:
                return
            state["x_hi"] = v
            _pa_update()

        def _on_y_lo(value: float) -> None:
            v = int(round(value))
            if v == state["y_lo"]:
                return
            state["y_lo"] = v
            _pa_update()

        def _on_y_hi(value: float) -> None:
            v = int(round(value))
            if v == state["y_hi"]:
                return
            state["y_hi"] = v
            _pa_update()

        def _on_z_cut(value: float) -> None:
            v = int(round(value))
            if v == state["z_cut"]:
                return
            state["z_cut"] = v
            _pa_update()

    else:
        # ==== Original path: rebuild meshes on clip change ====
        def _apply_clips() -> None:
            """Update renderer clip ranges from current state."""
            renderer.clip_x = (
                (state["x_lo"], state["x_hi"])
                if state["x_lo"] > 0 or state["x_hi"] < data.nx - 1
                else None
            )
            renderer.clip_y = (
                (state["y_lo"], state["y_hi"])
                if state["y_lo"] > 0 or state["y_hi"] < data.ny - 1
                else None
            )
            renderer.clip_z = (
                (state["z_cut"], data.n_depth - 1)
                if state["z_cut"] > 0 else None
            )

        def _remove_meshes() -> None:
            """Remove only mesh/axes/title actors — leave widgets."""
            for key in state["actor_keys"]:
                try:
                    plotter.remove_actor(key)
                except Exception:
                    pass
            state["actor_keys"].clear()

        def _render_meshes(t: int) -> None:
            """Render frame and track the actor keys it added."""
            before = set(plotter.actors.keys())
            _apply_clips()
            if config.depth_profile != "none":
                renderer.render_frame_with_depth_profile(t, plotter)
            else:
                renderer.render_frame(t, plotter)
            after = set(plotter.actors.keys())
            state["actor_keys"] = list(after - before)

        def _rebuild() -> None:
            """Remove mesh actors and re-render (sliders stay)."""
            cam_pos = plotter.camera_position
            _remove_meshes()
            _render_meshes(state["t"])
            plotter.camera_position = cam_pos
            plotter.render()

        # --- Slider callbacks (original path) ---
        def _on_time_change(value: float) -> None:
            t = int(round(value))
            if t == state["t"]:
                return
            state["t"] = t
            _apply_clips()
            if (hasattr(renderer, "can_fast_update")
                    and renderer.can_fast_update()):
                renderer.update_time_fast(t, plotter)
            else:
                _rebuild()

        def _on_x_lo(value: float) -> None:
            v = int(round(value))
            if v == state["x_lo"]:
                return
            state["x_lo"] = v
            if hasattr(renderer, "_cached_meshes"):
                renderer._cached_meshes.clear()
            _rebuild()

        def _on_x_hi(value: float) -> None:
            v = int(round(value))
            if v == state["x_hi"]:
                return
            state["x_hi"] = v
            if hasattr(renderer, "_cached_meshes"):
                renderer._cached_meshes.clear()
            _rebuild()

        def _on_y_lo(value: float) -> None:
            v = int(round(value))
            if v == state["y_lo"]:
                return
            state["y_lo"] = v
            if hasattr(renderer, "_cached_meshes"):
                renderer._cached_meshes.clear()
            _rebuild()

        def _on_y_hi(value: float) -> None:
            v = int(round(value))
            if v == state["y_hi"]:
                return
            state["y_hi"] = v
            if hasattr(renderer, "_cached_meshes"):
                renderer._cached_meshes.clear()
            _rebuild()

        def _on_z_cut(value: float) -> None:
            v = int(round(value))
            if v == state["z_cut"]:
                return
            state["z_cut"] = v
            if hasattr(renderer, "_cached_meshes"):
                renderer._cached_meshes.clear()
            _rebuild()

        # Initial render (non-prealloc path)
        _render_meshes(0)

    # --- Sliders: created ONCE, never recreated ---
    # Layout:  Time (top),  X lo/hi + Y lo/hi (two rows),  Z peel (right)
    plotter.add_slider_widget(
        _on_time_change,
        rng=[0, data.nt - 1],
        value=0,
        title=f"Time (0\u2013{data.nt - 1})",
        pointa=(0.1, 0.92),
        pointb=(0.9, 0.92),
        style="modern",
        interaction_event="always",
    )
    if data.nx > 1:
        plotter.add_slider_widget(
            _on_x_lo,
            rng=[0, data.nx - 1],
            value=0,
            title="X lo",
            pointa=(0.02, 0.14),
            pointb=(0.23, 0.14),
            style="modern",
            interaction_event="always",
        )
        plotter.add_slider_widget(
            _on_x_hi,
            rng=[0, data.nx - 1],
            value=data.nx - 1,
            title="X hi",
            pointa=(0.02, 0.06),
            pointb=(0.23, 0.06),
            style="modern",
            interaction_event="always",
        )
    if data.ny > 1:
        plotter.add_slider_widget(
            _on_y_lo,
            rng=[0, data.ny - 1],
            value=0,
            title="Y lo",
            pointa=(0.28, 0.14),
            pointb=(0.49, 0.14),
            style="modern",
            interaction_event="always",
        )
        plotter.add_slider_widget(
            _on_y_hi,
            rng=[0, data.ny - 1],
            value=data.ny - 1,
            title="Y hi",
            pointa=(0.28, 0.06),
            pointb=(0.49, 0.06),
            style="modern",
            interaction_event="always",
        )
    if data.n_depth > 1:
        plotter.add_slider_widget(
            _on_z_cut,
            rng=[0, data.n_depth - 1],
            value=0,
            title="Z peel",
            pointa=(0.55, 0.06),
            pointb=(0.98, 0.06),
            style="modern",
            interaction_event="always",
        )

    mode = "pre-allocated" if use_prealloc else "rebuild"
    lazy_str = " [lazy/out-of-core]" if data.is_lazy else ""
    print(
        f"Interactive viewer ({mode}{lazy_str}): {data.nx}x{data.ny}x{data.n_depth}, "
        f"{data.nt} time steps, {data.n_elem} elements\n"
        f"  Sliders: Time (top), X lo/hi + Y lo/hi (middle/bottom), Z peel (right)"
    )
    try:
        plotter.show()
    finally:
        data.close()  # Clean up temp H5 if lazy mode


# ============================================================================
# CLI
# ============================================================================


def _auto_output_name(h5_path: str, nx: int, ny: int, nt: int) -> str:
    """Generate descriptive output filename from H5 metadata.

    Pattern: ``ezdeP4D_{nx}x{ny}x{nt}_{stem}.mp4``
    e.g. ``ezdeP4D_20x20x200_dataset.mp4`` for ``dataset.h5``
    """
    stem = Path(h5_path).stem
    return f"ezdeP4D_{nx}x{ny}x{nt}_{stem}.mp4"


def _estimate_memory_gb(nx: int, ny: int, nt: int, n_depth: int, n_elem: int) -> float:
    """Estimate peak memory in GB for the 5D array."""
    bytes_5d = nt * nx * ny * n_depth * n_elem * 8  # float64
    return bytes_5d / 1e9


def _find_oxygen_index(element_names: List[str]) -> Optional[int]:
    """Find index of oxygen element (O(-2) or O) in element list."""
    for i, name in enumerate(element_names):
        if name.startswith("O(") or name == "O":
            return i
    return None


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="EzDeP4D — 4D Depth Profile Animation (PyVista)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Auto output name, auto grid from H5 metadata
  python -m deppro.visualize_4d --h5 dataset.h5

  # Explicit grid + hide oxygen
  python -m deppro.visualize_4d --h5 dataset.h5 --nx 10 --ny 10 --hide-oxygen

  # Custom output path + settings
  python -m deppro.visualize_4d --h5 dataset.h5 -o my_video.mp4 --fps 15

  # Forward-only, no depth profile
  python -m deppro.visualize_4d --h5 dataset.h5 --no-forward-backward --no-depth-profile

  # Interactive 3D viewer with time slider
  python -m deppro.visualize_4d --h5 dataset.h5 -i
""",
    )

    parser.add_argument("--h5", required=True, help="Path to GT H5 file")
    parser.add_argument(
        "--nx", type=int, default=None,
        help="Spatial X grid (default: from H5 metadata dxdydt[0])",
    )
    parser.add_argument(
        "--ny", type=int, default=None,
        help="Spatial Y grid (default: from H5 metadata dxdydt[1])",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output MP4 path (default: auto-generated from H5 name and grid)",
    )
    parser.add_argument(
        "--isovalue", type=float, default=0.5,
        help="Isovalue degree: threshold = max * degree (default: 0.5, MATLAB default)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.75,
        help="Face opacity 0-1 (default: 0.75, MATLAB default)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument(
        "--no-forward-backward", action="store_true",
        help="Disable forward+backward playback (forward only)",
    )
    parser.add_argument("--no-depth-profile", action="store_true", help="Disable depth profile subplot")
    parser.add_argument(
        "--no-lighting", action="store_true",
        help="Flat unlit colors — EzDeP4D with the 3D Lighting toggle off "
        "(the mode its production movies were rendered in)",
    )
    parser.add_argument(
        "--hide", nargs="*", type=int, default=[],
        help="Element indices to hide (0-based)",
    )
    parser.add_argument(
        "--hide-oxygen", action="store_true",
        help="Automatically hide oxygen element (O(-2))",
    )
    parser.add_argument("--azimuth", type=float, default=-105.0, help="Camera azimuth (default: -105)")
    parser.add_argument("--elevation", type=float, default=5.0, help="Camera elevation (default: 5)")
    parser.add_argument("--zoom", type=float, default=1.1, help="Camera zoom factor (default: 1.1)")
    parser.add_argument("--specular", type=float, default=0.3, help="Specular reflection 0-1 (default: 0.3)")
    parser.add_argument("--cap-threshold", type=float, default=0.02, help="Min conc. for isocap cells (default: 0.02)")
    parser.add_argument("--quality", type=int, default=8, help="Video quality 1-10, lower=better (default: 8)")
    parser.add_argument(
        "--color-source", default="auto",
        choices=["auto", "h5", "standard", "hsv"],
        help="Colour source: auto (h5>standard>hsv), h5, standard, hsv (default: auto)",
    )
    parser.add_argument(
        "--render-mode", default="isosurface",
        choices=["isosurface", "slice", "matlab"],
        help="Render mode: isosurface (sp=2, 3D bodies + composite caps), "
        "slice (sp=1, 6-face textures), or matlab (sp=2 MATLAB-faithful: "
        "per-element interpolated isocaps, physical coords, fixed camera/light) "
        "(default: isosurface)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Open interactive 3D viewer with time slider instead of rendering video",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit number of time frames (for quick preview)",
    )
    parser.add_argument(
        "--lazy", action="store_true", default=None,
        help="Out-of-core mode: store 5D data in temp H5, load 1 frame at a time. "
        "Auto-enabled when estimated memory > 4 GB.",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="Force in-memory mode even for large datasets.",
    )
    parser.add_argument(
        "--lazy-threshold", type=float, default=4.0,
        help="Auto-lazy threshold in GB (default: 4.0). "
        "Datasets above this use out-of-core mode unless --no-lazy.",
    )

    args = parser.parse_args()

    # Estimate memory before loading
    # (peek at H5 to get dimensions without loading Dstk)
    with h5py.File(str(args.h5), "r") as _peek:
        _n_px, _n_elem, _n_depth = _peek["Dstk"].shape
        _dxdydt = np.array(_peek["ezdeprof/dxdydt"], dtype=np.float64).flatten()
        _nx = args.nx or int(_dxdydt[0])
        _ny = args.ny or int(_dxdydt[1])
    _dxdy = _nx * _ny
    _nt = _n_px // _dxdy if _n_px % _dxdy == 0 else _n_px
    mem_gb = _estimate_memory_gb(_nx, _ny, _nt, _n_depth, _n_elem)
    print(f"Estimated memory (5D float64): {mem_gb:.2f} GB")

    # Decide lazy vs in-memory
    use_lazy = args.lazy
    if args.no_lazy:
        use_lazy = False
    elif use_lazy is None:
        use_lazy = mem_gb > args.lazy_threshold
        if use_lazy:
            print(
                f"Auto-enabling lazy mode ({mem_gb:.1f} GB > "
                f"{args.lazy_threshold:.1f} GB threshold)"
            )

    # Load data
    if use_lazy:
        data = load_gt_4d_lazy(
            args.h5, nx=args.nx, ny=args.ny,
            color_source=args.color_source,
        )
    else:
        data = load_gt_4d(
            args.h5, nx=args.nx, ny=args.ny,
            color_source=args.color_source,
        )

    # Auto-detect oxygen for --hide-oxygen
    hidden = list(args.hide)
    if args.hide_oxygen:
        oxy_idx = _find_oxygen_index(data.element_names)
        if oxy_idx is not None and oxy_idx not in hidden:
            hidden.append(oxy_idx)
            print(f"Hiding oxygen: {data.element_names[oxy_idx]} (index {oxy_idx})")
        else:
            print("Warning: --hide-oxygen specified but no oxygen element found")

    # Auto output name
    output_path = args.output
    if output_path is None:
        output_path = _auto_output_name(args.h5, data.nx, data.ny, data.nt)
        print(f"Auto output: {output_path}")

    # Optional frame limit (only for in-memory mode)
    effective_data = data
    if args.max_frames is not None and args.max_frames < data.nt:
        if data.data is not None:
            effective_data = DepthProfile4DData(
                data=data.data[: args.max_frames],
                depth=data.depth,
                element_names=data.element_names,
                colors=data.colors,
                step_size=data.step_size,
                nx=data.nx,
                ny=data.ny,
                nt=args.max_frames,
                n_depth=data.n_depth,
                n_elem=data.n_elem,
                source_path=data.source_path,
                dxdydt=data.dxdydt,
            )
        else:
            # Lazy mode: just adjust nt (frames beyond max_frames won't be accessed)
            effective_data = DepthProfile4DData(
                data=None,
                depth=data.depth,
                element_names=data.element_names,
                colors=data.colors,
                step_size=data.step_size,
                nx=data.nx,
                ny=data.ny,
                nt=args.max_frames,
                n_depth=data.n_depth,
                n_elem=data.n_elem,
                source_path=data.source_path,
                dxdydt=data.dxdydt,
                _lazy_h5_path=data._lazy_h5_path,
            )
            # Transfer ownership so original doesn't delete temp file
            data._lazy_h5_path = ""
        print(f"Limiting to {args.max_frames} time frames (of {data.nt})")

    # Build config
    config = AnimationConfig(
        render_mode=args.render_mode,
        isovalue_degree=args.isovalue,
        face_alpha=args.alpha,
        cap_threshold=args.cap_threshold,
        specular=args.specular,
        fps=args.fps,
        forward_backward=not args.no_forward_backward,
        depth_profile="none" if args.no_depth_profile else "average",
        invisible_elements=hidden,
        output_path=output_path,
        azimuth=args.azimuth,
        elevation=args.elevation,
        camera_zoom=args.zoom,
        video_quality=args.quality,
        lighting=not args.no_lighting,
    )

    # Render or view interactively
    if args.interactive:
        interactive(effective_data, config)
    else:
        animate(effective_data, config)


if __name__ == "__main__":
    main()
