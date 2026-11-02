"""
Linear-algebra helpers for building model inputs
================================================

``truncated_pinv`` gives a rank-truncated pseudo-inverse of a transfer
matrix, and ``estimate_imfp`` estimates the effective IMFP from the
normal-emission decay of a transfer matrix.
"""


import numpy as np


def truncated_pinv(T: np.ndarray, k: int = 3) -> np.ndarray:
    """Pseudo-inverse of T keeping only the top-k singular components.

    Parameters
    ----------
    T : (n_angles, n_depth) transfer matrix for one element.
    k : number of singular components to retain (regularization strength).

    Returns
    -------
    (n_depth, n_angles) truncated pseudo-inverse.
    """
    U, S, Vt = np.linalg.svd(T, full_matrices=False)
    k = min(k, len(S))
    S_inv = np.zeros_like(S)
    nonzero = S[:k] > 0
    S_inv[:k][nonzero] = 1.0 / S[:k][nonzero]
    return (Vt.T * S_inv[np.newaxis, :]) @ U.T


def estimate_imfp(T: np.ndarray, total_depth: float,
                  default: float = 25.0) -> float:
    """Estimate the effective IMFP from the normal-emission decay of T.

    Uses the ratio of the deepest to the shallowest transfer-matrix entry at
    the first (most normal) angle. Falls back to ``default`` when the decay
    is not well-defined.
    """
    t_normal = T[0, :]
    if t_normal[0] > 0 and t_normal[-1] > 0:
        return -total_depth / np.log(max(t_normal[-1] / t_normal[0], 1e-10))
    return default
