"""
Benchmark Results I/O — unified storage for multi-method benchmark results.

Separates solver results from ground-truth data. All methods' results
are stored in a single ``benchmark_results.h5`` file using HDF5 groups.

Schema
------
::

    /meta/
        depth          (n_depth,)     depth grid [Angstrom]
        elements       (n_elem,)      element names
        n_pixels       scalar
        created        string         ISO timestamp

    /<method_name>/               e.g. "matlab_L1", "slsqp_L1", "nn_w1"
        D              (n_px, n_elem, n_depth)   depth profiles (MATLAB layout)
        time_sec       scalar         total compute time [s]
        loops          scalar         optimisation loops (optional)
        config         string         JSON config dump (optional)

Notes
-----
- ``D`` is stored in **MATLAB layout** ``(n_px, n_elem, n_depth)`` for
  compatibility with existing H5 files.  Reader helpers transpose to
  Python layout ``(n_px, n_depth, n_elem)`` on load.
- Method groups can be freely added / removed / renamed.
- The ``/meta`` group is written once and shared across all methods.

Usage
-----
::

    from deppro.benchmarks.benchmark_results_io import (
        save_result, load_result, list_methods, results_path,
    )

    # Save
    save_result(gt_h5_path, method="slsqp_L1", D=D_opt, time_sec=42.3)

    # Load
    D, meta = load_result(gt_h5_path, method="slsqp_L1")

    # List
    methods = list_methods(gt_h5_path)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def results_path(gt_h5_path: str) -> Path:
    """Return the ``benchmark_results.h5`` path next to a GT H5 file."""
    return Path(gt_h5_path).parent / "benchmark_results.h5"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def save_result(
    gt_h5_path: str,
    method: str,
    D: np.ndarray,
    *,
    time_sec: Optional[float] = None,
    loops: Optional[int] = None,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> Path:
    """Save a single method's depth-profile result.

    Parameters
    ----------
    gt_h5_path : str
        Path to the ground-truth H5 file.  Used to locate the results file
        and to initialise ``/meta`` (depth, elements) on first write.
    method : str
        Method name used as the HDF5 group name (e.g. ``"slsqp_L1"``).
        Must be a valid HDF5 group name (no leading ``/``).
    D : ndarray
        Depth profiles, shape ``(n_px, n_depth, n_elem)`` (Python layout).
        Stored as ``(n_px, n_elem, n_depth)`` (MATLAB layout) in the file.
    time_sec : float, optional
        Total compute time in seconds.
    loops : int, optional
        Number of optimisation loops.
    config : dict, optional
        Arbitrary solver configuration (stored as JSON string).
    overwrite : bool
        If *True*, overwrite existing group for this method.

    Returns
    -------
    Path
        Path to the ``benchmark_results.h5`` file.
    """
    out = results_path(gt_h5_path)

    # Transpose Python (n_px, n_depth, n_elem) → MATLAB (n_px, n_elem, n_depth)
    D_matlab = np.transpose(D, (0, 2, 1)).astype(np.float64)

    with h5py.File(str(out), "a") as f:
        # --- /meta (create once) ---
        if "meta" not in f:
            _init_meta(f, gt_h5_path, D.shape[0])

        # --- method group ---
        if method in f:
            if not overwrite:
                raise ValueError(
                    f"Method '{method}' already exists in {out}. "
                    f"Use overwrite=True to replace."
                )
            del f[method]

        g = f.create_group(method)
        g.create_dataset("D", data=D_matlab, compression="gzip", compression_opts=4)

        if time_sec is not None:
            g.attrs["time_sec"] = float(time_sec)
        if loops is not None:
            g.attrs["loops"] = int(loops)
        if config is not None:
            g.attrs["config"] = json.dumps(config, ensure_ascii=False)

        g.attrs["saved_at"] = datetime.now().isoformat()

    print(f"  Saved '{method}' → {out}  D{D.shape}")
    return out


def _init_meta(f: h5py.File, gt_h5_path: str, n_pixels: int) -> None:
    """Initialise the ``/meta`` group from the GT file."""
    meta = f.create_group("meta")
    meta.attrs["created"] = datetime.now().isoformat()
    meta.attrs["n_pixels"] = int(n_pixels)
    meta.attrs["gt_file"] = str(Path(gt_h5_path).name)

    with h5py.File(gt_h5_path, "r") as gt:
        # depth
        depth = np.array(gt["ezdeprof/depth"], dtype=np.float64).flatten()
        meta.create_dataset("depth", data=depth)

        # elements
        raw = gt["ezdeprof/element"][:].flatten()
        names = [
            e.decode() if isinstance(e, bytes) else str(e) for e in raw
        ]
        meta.create_dataset(
            "elements",
            data=np.array(names, dtype=h5py.string_dtype()),
        )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def load_result(
    gt_h5_path: str,
    method: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a single method's result.

    Returns
    -------
    D : ndarray
        Depth profiles in Python layout ``(n_px, n_depth, n_elem)``.
    meta : dict
        ``depth``, ``elements``, ``time_sec``, ``loops``, ``config`` etc.
    """
    p = results_path(gt_h5_path)
    if not p.exists():
        raise FileNotFoundError(f"No benchmark results file: {p}")

    with h5py.File(str(p), "r") as f:
        if method not in f:
            avail = [k for k in f.keys() if k != "meta"]
            raise KeyError(
                f"Method '{method}' not found. Available: {avail}"
            )

        g = f[method]
        # MATLAB (n_px, n_elem, n_depth) → Python (n_px, n_depth, n_elem)
        D = np.transpose(
            np.array(g["D"], dtype=np.float64), (0, 2, 1)
        )

        meta = _read_meta(f)
        # per-method attrs
        if "time_sec" in g.attrs:
            meta["time_sec"] = float(g.attrs["time_sec"])
        if "loops" in g.attrs:
            meta["loops"] = int(g.attrs["loops"])
        if "config" in g.attrs:
            meta["config"] = json.loads(g.attrs["config"])
        if "saved_at" in g.attrs:
            meta["saved_at"] = str(g.attrs["saved_at"])

    return D, meta


def load_all(
    gt_h5_path: str,
) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    """Load all methods from the results file.

    Returns
    -------
    dict
        ``{method_name: (D, meta)}`` for each stored method.
    """
    p = results_path(gt_h5_path)
    if not p.exists():
        raise FileNotFoundError(f"No benchmark results file: {p}")

    out = {}
    with h5py.File(str(p), "r") as f:
        for method in f.keys():
            if method == "meta":
                continue
            D = np.transpose(
                np.array(f[method]["D"], dtype=np.float64), (0, 2, 1)
            )
            meta = _read_meta(f)
            g = f[method]
            if "time_sec" in g.attrs:
                meta["time_sec"] = float(g.attrs["time_sec"])
            if "loops" in g.attrs:
                meta["loops"] = int(g.attrs["loops"])
            if "config" in g.attrs:
                meta["config"] = json.loads(g.attrs["config"])
            out[method] = (D, meta)

    return out


def _read_meta(f: h5py.File) -> Dict[str, Any]:
    """Extract ``/meta`` as a plain dict."""
    meta: Dict[str, Any] = {}
    if "meta" in f:
        mg = f["meta"]
        if "depth" in mg:
            meta["depth"] = np.array(mg["depth"], dtype=np.float64).flatten()
        if "elements" in mg:
            raw = mg["elements"][:]
            meta["elements"] = [
                e.decode() if isinstance(e, bytes) else str(e) for e in raw
            ]
        for k in ("n_pixels", "gt_file", "created"):
            if k in mg.attrs:
                meta[k] = mg.attrs[k]
    return meta


# ---------------------------------------------------------------------------
# Query / management
# ---------------------------------------------------------------------------

def list_methods(gt_h5_path: str) -> List[str]:
    """Return method names available in the results file."""
    p = results_path(gt_h5_path)
    if not p.exists():
        return []
    with h5py.File(str(p), "r") as f:
        return [k for k in f.keys() if k != "meta"]


def delete_method(gt_h5_path: str, method: str) -> None:
    """Remove a method group from the results file."""
    p = results_path(gt_h5_path)
    with h5py.File(str(p), "a") as f:
        if method in f:
            del f[method]
            print(f"  Deleted '{method}' from {p}")
        else:
            print(f"  '{method}' not found in {p}")


# ---------------------------------------------------------------------------
# Migration helper
# ---------------------------------------------------------------------------

def migrate_from_gt(
    gt_h5_path: str,
    method: str,
    dataset_name: str = "Dtrn",
    *,
    time_sec: Optional[float] = None,
) -> None:
    """Copy a Dtrn/Drlt dataset from the GT H5 into benchmark_results.h5.

    Parameters
    ----------
    gt_h5_path : str
        Path to the GT H5 file containing the dataset.
    method : str
        Method name for the results file (e.g. ``"matlab_L1"``).
    dataset_name : str
        Dataset to read from the GT file (``"Dtrn"`` or ``"Drlt"``).
    time_sec : float, optional
        Compute time if known.
    """
    with h5py.File(gt_h5_path, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"'{dataset_name}' not found in {gt_h5_path}")
        # MATLAB layout (n_px, n_elem, n_depth) → Python (n_px, n_depth, n_elem)
        D_raw = np.array(f[dataset_name], dtype=np.float64)
        D_py = np.transpose(D_raw, (0, 2, 1))

    save_result(
        gt_h5_path,
        method=method,
        D=D_py,
        time_sec=time_sec,
        overwrite=True,
    )
    print(f"  Migrated '{dataset_name}' → '{method}'")
