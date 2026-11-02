"""
Post-processing for Neural Network Predictions
===============================================

Enforces physical constraints on predicted depth profiles.
"""

import numpy as np


def enforce_constraints(
    D_flat: np.ndarray,
    n_depth: int,
    n_elem: int,
) -> np.ndarray:
    """Enforce physical constraints on predicted depth profiles.

    1. Non-negativity (should already hold from softplus output)
    2. Sum normalization: Σ_elem D(z) = 1 at each depth

    Parameters
    ----------
    D_flat : ndarray, shape (batch, n_depth * n_elem)
        Raw NN output (flattened).
    n_depth : int
        Number of depth points.
    n_elem : int
        Number of elements.

    Returns
    -------
    ndarray, shape (batch, n_depth * n_elem)
        Constrained depth profiles.
    """
    batch = D_flat.shape[0]
    D = D_flat.reshape(batch, n_depth, n_elem).copy()

    # Non-negativity (safety net)
    np.maximum(D, 0, out=D)

    # Sum normalization at each depth
    total = D.sum(axis=2, keepdims=True)
    total = np.maximum(total, 1e-10)
    D /= total

    return D.reshape(batch, n_depth * n_elem)


def enforce_constraints_mlx(D_flat, n_depth: int, n_elem: int):
    """MLX (on-device) version of :func:`enforce_constraints`.

    Runs the non-negativity + sum-normalization on the GPU *before* the
    single GPU->host transfer, instead of as a numpy/CPU pass afterward.
    The numpy variant is memory-bandwidth bound on the host; doing it
    on-device keeps the constrained result on the GPU and transfers it in
    one copy.

    Parameters
    ----------
    D_flat : mx.array, shape (batch, n_depth * n_elem)
        Raw NN output (flattened), still on the MLX device.
    n_depth, n_elem : int

    Returns
    -------
    mx.array, shape (batch, n_depth * n_elem)
        Constrained depth profiles, still on the MLX device. Call
        ``np.array(...)`` on the result for the (single) host transfer.
    """
    import mlx.core as mx

    batch = D_flat.shape[0]
    D = D_flat.reshape(batch, n_depth, n_elem)
    D = mx.maximum(D, 0.0)
    total = mx.maximum(D.sum(axis=2, keepdims=True), 1e-10)
    D = D / total
    return D.reshape(batch, n_depth * n_elem)


def normalize_intensity_mlx(I_mx):
    """On-device intensity normalization: I / Σ_feature I (per pixel).

    Parameters
    ----------
    I_mx : mx.array, shape (batch, n_features)

    Returns
    -------
    mx.array, same shape, rows summing to 1 (zero rows left at 0).
    """
    import mlx.core as mx
    return I_mx / mx.maximum(I_mx.sum(axis=1, keepdims=True), 1e-30)


def estimate_thickness_mlx(D_mx, depth_mx, sub_idx: int, thickness_init: float):
    """First 50% substrate-crossover depth per pixel, on-device.

    Vectorized equivalent of the streaming thickness estimator, kept in MLX so
    the whole inversion tail (normalize → infer → sum=1 → thickness) runs on the
    GPU with a single host transfer. argmax over the boolean crossover mask
    returns the first True index (verified to match numpy); pixels with no
    crossover keep ``thickness_init``.

    Parameters
    ----------
    D_mx : mx.array, shape (batch, n_depth, n_elem)
        Constrained depth profiles, on device.
    depth_mx : mx.array, shape (n_depth,)
        Depth grid (Å), on device.
    sub_idx : int
        Substrate element index.
    thickness_init : float
        Fallback thickness for pixels with no 50% crossover.

    Returns
    -------
    mx.array, shape (batch,)
        Per-pixel thickness (Å), on device.
    """
    import mlx.core as mx
    above = D_mx[:, :, sub_idx] > 0.5            # (batch, n_depth) bool
    has_cross = mx.any(above, axis=1)            # (batch,)
    first_idx = mx.argmax(above.astype(mx.int32), axis=1)  # first True
    return mx.where(has_cross, mx.take(depth_mx, first_idx),
                    mx.array(thickness_init, dtype=depth_mx.dtype))
