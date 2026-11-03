"""Write inverse-problem depth estimates to HDF5.

Output structure:
    /D_est/data      — (n_px, n_elem, n_depth) float32  MATLAB layout
    /D_est/solver    — solver name string
    /D_est/timestamp — ISO 8601 execution timestamp
    /Xstk            — (n_px, n_elem, n_angles) float32  (optional, observed)
    /ezdeprof/       — metadata copied from source Dstk

**Not** written: ``/Dstk`` (ground truth), ``/Py_Drlt`` (legacy).

Compatible with ``deppro.visualize_4d.load_depth_estimate()`` for
4D isosurface visualization.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray


def write_depth_estimate(
    D_est: NDArray[np.floating],
    metadata: dict[str, Any],
    output_dir: str,
    solver: str = "NN",
    I_obs: NDArray[np.floating] | None = None,
) -> str:
    """Write inverse-problem depth estimate to HDF5.

    Parameters
    ----------
    D_est : ndarray, shape (n_pixels, n_depth, n_elements)
        Python layout concentration fractions (0–1).
    metadata : dict
        Must contain ``pixel_shape``, ``elements``.
        Optional: ``depth_axis``, ``step_size``, ``angles_deg``.
    output_dir : str
        Directory for output files.
    solver : str
        Solver tag for filename: ``"NN"``, ``"SLSQP"``, ``"LLS"``, etc.
    I_obs : ndarray | None, shape (n_pixels, n_angles, n_elements)
        If provided, written as ``/Xstk``.

    Returns
    -------
    str
        Path of the saved HDF5 file.
    """
    n_px, n_depth, n_elem = D_est.shape
    n_x, n_y = metadata["pixel_shape"]
    elements: list[str] = metadata["elements"]

    if len(elements) != n_elem:
        raise ValueError(
            f"elements length ({len(elements)}) != D_est n_elem ({n_elem})"
        )

    # Resolve output path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(Path(output_dir) / f"depth_est_{solver}_{ts}.h5")

    # Depth axis
    step_size = metadata.get("step_size", 4.0)
    depth_axis = metadata.get(
        "depth_axis", np.arange(n_depth, dtype=np.float64) * step_size
    )

    # Transpose to MATLAB layout: (n_px, n_elem, n_depth)
    d_est_matlab = np.transpose(D_est, (0, 2, 1)).astype(np.float32)

    with h5py.File(output_path, "w") as f:
        # --- /D_est group ---
        d_grp = f.create_group("D_est")
        d_grp.create_dataset("data", data=d_est_matlab, compression="gzip")

        dt_str = h5py.string_dtype()
        d_grp.create_dataset("solver", data=solver, dtype=dt_str)
        d_grp.create_dataset(
            "timestamp",
            data=datetime.now().isoformat(),
            dtype=dt_str,
        )

        # --- /Xstk (input intensities) ---
        if I_obs is not None:
            # (n_px, n_angles, n_elem) → MATLAB (n_px, n_elem, n_angles)
            xstk_matlab = np.transpose(I_obs, (0, 2, 1)).astype(np.float32)
            f.create_dataset("Xstk", data=xstk_matlab, compression="gzip")

        # --- /ezdeprof metadata group ---
        ez = f.create_group("ezdeprof")

        # depth: (n_depth, 1) — MATLAB column vector
        ez.create_dataset(
            "depth", data=np.asarray(depth_axis, dtype=np.float64).reshape(-1, 1)
        )

        # stepsize: (1, 1) — MATLAB scalar
        ez.create_dataset("stepsize", data=np.array([[step_size]], dtype=np.float64))

        # element: (n_elem,) variable-length strings
        ez.create_dataset("element", data=elements, dtype=dt_str)

        # dxdydt: (1, 3) — spatial grid info
        n_t = max(1, n_px // (n_x * n_y))
        ez.create_dataset(
            "dxdydt", data=np.array([[n_x, n_y, n_t]], dtype=np.float64)
        )

        # thetaobs (if angles available)
        angles = metadata.get("angles_deg")
        if angles is not None:
            ez.create_dataset(
                "thetaobs",
                data=np.asarray(angles, dtype=np.float64).reshape(1, -1),
            )

        # TransMatObs (if available) — for forward simulation / NRMSE
        # Python layout (n_angles, n_depth, n_elem) → MATLAB (n_elem, n_depth, n_angles)
        trans_mat = metadata.get("trans_mat")
        if trans_mat is not None:
            tm_matlab = np.transpose(trans_mat, (2, 1, 0))
            ez.create_dataset(
                "TransMatObs", data=tm_matlab.astype(np.float64), compression="gzip",
            )

    return output_path
