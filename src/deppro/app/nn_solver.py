"""NN surrogate solver for ARXPS depth profiling.

Wraps deppro.inference DepthMLP for toyomacro's GUI data conventions.
MLX GPU inference — single-process, no ProcessPool needed.

Usage::

    from deppro.app.nn_solver import NNDepthSolver

    solver = NNDepthSolver("path/to/model.npz")
    drlt_3d = solver.solve_batch(xstk_3d, n_depth)     # (n_px, n_elem, n_depth)
    drlt    = solver.solve_single(x_obs, n_depth)       # (n_elem, n_depth)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MLX / deppro availability
# ---------------------------------------------------------------------------

NN_AVAILABLE = False
_nn_import_error: str | None = None

try:
    import mlx.core as mx
    from deppro.inference.model import DepthMLP
    from deppro.inference.postprocess import enforce_constraints
    from deppro.inference.train import load_model

    NN_AVAILABLE = True
except ImportError as _e:
    _nn_import_error = str(_e)


def _check_nn() -> None:
    if not NN_AVAILABLE:
        raise ImportError(
            f"NN solver requires mlx + deppro.inference: {_nn_import_error}"
        )


# ---------------------------------------------------------------------------
# NNDepthSolver
# ---------------------------------------------------------------------------


class NNDepthSolver:
    """MLX-based NN surrogate for SemiFreeSolver.

    Parameters
    ----------
    model_path : str or Path
        Path to .npz weights (auto-detects architecture).
    """

    def __init__(self, model_path: str | Path) -> None:
        _check_nn()
        self.model_path = str(model_path)
        self.model = load_model(self.model_path)
        # Extract dims from model
        self._input_dim: int = self.model.layers[0].weight.shape[1]
        self._output_dim: int = self.model.layers[-1].weight.shape[0]
        log.info(
            "NNDepthSolver: loaded %s (%d -> %d, %d params)",
            self.model_path,
            self._input_dim,
            self._output_dim,
            self.model.n_parameters(),
        )

    def solve_batch(
        self,
        xstk_3d: NDArray[np.floating],
        n_depth: int,
        *,
        progress_callback: Callable | None = None,
    ) -> NDArray[np.float32]:
        """Batch NN inference for all pixels.

        Parameters
        ----------
        xstk_3d : (n_px, n_elem, n_angles) — GUI storage format.
        n_depth : Number of depth points.
        progress_callback : Optional (current, total) callback.

        Returns
        -------
        drlt_3d : (n_px, n_elem, n_depth) float32 — GUI storage format.
        """
        n_px, n_elem, n_angles = xstk_3d.shape

        # Transpose to (n_px, n_angles, n_elem) then flatten
        X = np.ascontiguousarray(
            xstk_3d.transpose(0, 2, 1), dtype=np.float32
        )  # (n_px, n_angles, n_elem)
        X_flat = X.reshape(n_px, -1)  # (n_px, n_angles * n_elem)

        # MLX inference
        t0 = time.perf_counter()
        X_mx = mx.array(X_flat)
        D_mx = self.model(X_mx)
        mx.eval(D_mx)
        D_flat = np.array(D_mx)  # (n_px, n_depth * n_elem)
        t_infer = time.perf_counter() - t0

        # Post-processing: non-neg + sum normalization
        D_flat = enforce_constraints(D_flat, n_depth, n_elem)

        # Reshape to (n_px, n_depth, n_elem) then transpose to GUI format
        D = D_flat.reshape(n_px, n_depth, n_elem)
        drlt_3d = np.ascontiguousarray(
            D.transpose(0, 2, 1), dtype=np.float32
        )  # (n_px, n_elem, n_depth)

        throughput = n_px / t_infer if t_infer > 0 else float("inf")
        log.info(
            "NN batch: %d px in %.4fs (%.0f px/s)",
            n_px, t_infer, throughput,
        )

        if progress_callback:
            progress_callback(n_px, n_px)

        return drlt_3d

    def solve_single(
        self,
        x_obs: NDArray[np.floating],
        n_depth: int,
    ) -> NDArray[np.float32]:
        """Single-pixel NN inference.

        Parameters
        ----------
        x_obs : (n_elem, n_angles) — single pixel, GUI format.
        n_depth : Number of depth points.

        Returns
        -------
        drlt : (n_elem, n_depth) float32 — GUI format.
        """
        n_elem, n_angles = x_obs.shape

        # Transpose + flatten: (n_elem, n_angles) → (1, n_angles * n_elem)
        X_flat = x_obs.T.reshape(1, -1).astype(np.float32)

        # MLX inference
        X_mx = mx.array(X_flat)
        D_mx = self.model(X_mx)
        mx.eval(D_mx)
        D_flat = np.array(D_mx)  # (1, n_depth * n_elem)

        # Post-processing
        D_flat = enforce_constraints(D_flat, n_depth, n_elem)

        # Reshape: (1, n_depth, n_elem) → (n_elem, n_depth)
        D = D_flat.reshape(n_depth, n_elem)
        return D.T.astype(np.float32)
