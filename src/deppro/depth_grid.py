"""
Depth Grid Generation
=====================

Uniform and surface-dense depth grids, a dispatcher ``make_grid`` and a
registry through which further grid types can be added with
``register_grid_type``.

Usage:
    from deppro.depth_grid import make_grid, make_uniform_grid

    # Uniform grid
    grid = make_uniform_grid(n_depth=20, z_max=200.0)

    # Geometric grid (dense near the surface)
    grid = make_grid('geometric', n_depth=12, z_max=200.0, ratio=1.3)
"""

import numpy as np
from typing import Callable, Optional


def make_uniform_grid(n_depth: int, z_max: float) -> np.ndarray:
    """Equal spacing grid (excludes z=0).

    Args:
        n_depth: Number of grid points
        z_max: Maximum depth (Angstroms)

    Returns:
        Depth grid (n_depth,) in Angstroms
    """
    return np.linspace(0, z_max, n_depth + 1)[1:]


def make_geometric_grid(
    n_depth: int, z_max: float, ratio: float = 1.3
) -> np.ndarray:
    """Geometric progression: dz_i = dz_0 * r^i.

    Dense near surface, sparse at depth.

    Args:
        n_depth: Number of grid points
        z_max: Maximum depth (Angstroms)
        ratio: Growth ratio for step sizes

    Returns:
        Depth grid (n_depth,) in Angstroms
    """
    steps = np.array([ratio ** i for i in range(n_depth)])
    steps = steps / steps.sum() * z_max
    return np.cumsum(steps)


def make_log_grid(
    n_depth: int, z_max: float, z_min: float = 2.0
) -> np.ndarray:
    """Logarithmic spacing: much denser near surface.

    Args:
        n_depth: Number of grid points
        z_max: Maximum depth (Angstroms)
        z_min: Minimum depth (first grid point, Angstroms)

    Returns:
        Depth grid (n_depth,) in Angstroms
    """
    return np.geomspace(z_min, z_max, n_depth)




# Registry of available grid types: name -> description
GRID_TYPES = {
    'uniform': 'Equal spacing',
    'geometric': 'Geometric progression (dense near surface)',
    'log': 'Logarithmic spacing (much denser near surface)',
}

# Builders added with register_grid_type: name -> callable
_REGISTERED_BUILDERS = {}


def register_grid_type(name: str, description: str,
                       builder: Callable[..., np.ndarray]) -> None:
    """Add a grid type to ``make_grid``.

    ``builder`` is called as ``builder(n_depth, z_max, lambda_eff=...,
    angles_deg=..., **kwargs)`` and must return the depth grid (n_depth,)
    in Angstroms. Built-in names cannot be replaced.

    Args:
        name: Grid type name used with ``make_grid``
        description: One-line description stored in ``GRID_TYPES``
        builder: Callable returning the grid
    """
    if name in ('uniform', 'geometric', 'log'):
        raise ValueError(f"Grid type '{name}' is built in and cannot be replaced")
    GRID_TYPES[name] = description
    _REGISTERED_BUILDERS[name] = builder


def make_grid(
    grid_type: str,
    n_depth: int,
    z_max: float,
    lambda_eff: float = 25.0,
    angles_deg: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Create a depth grid of the specified type.

    Args:
        grid_type: A name in ``GRID_TYPES`` ('uniform', 'geometric', 'log',
                   or a type added with ``register_grid_type``)
        n_depth: Number of grid points
        z_max: Maximum depth (Angstroms)
        lambda_eff: Mean IMFP (Angstroms), passed to registered grid types
        angles_deg: Observation angles (degrees), passed to registered grid
                    types
        **kwargs: Additional arguments passed to the grid function

    Returns:
        Depth grid (n_depth,) in Angstroms
    """
    if grid_type == 'uniform':
        return make_uniform_grid(n_depth, z_max)
    elif grid_type == 'geometric':
        ratio = kwargs.get('ratio', 1.3)
        return make_geometric_grid(n_depth, z_max, ratio=ratio)
    elif grid_type == 'log':
        z_min = kwargs.get('z_min', 2.0)
        return make_log_grid(n_depth, z_max, z_min=z_min)
    elif grid_type in _REGISTERED_BUILDERS:
        return _REGISTERED_BUILDERS[grid_type](
            n_depth, z_max, lambda_eff=lambda_eff, angles_deg=angles_deg,
            **kwargs)
    else:
        raise ValueError(
            f"Unknown grid type '{grid_type}'. "
            f"Available: {list(GRID_TYPES.keys())}"
        )


def interpolate_profiles_to_grid(
    D: np.ndarray,
    old_depth: np.ndarray,
    new_depth: np.ndarray,
) -> np.ndarray:
    """Interpolate depth profiles from old grid to new grid.

    Args:
        D: Depth profiles, shape (n_depth_old, n_elements) or
           (n_pixels, n_depth_old, n_elements)
        old_depth: Original depth grid (n_depth_old,)
        new_depth: Target depth grid (n_depth_new,)

    Returns:
        Interpolated profiles with same shape except n_depth dimension
    """
    if D.ndim == 2:
        # (n_depth, n_elements)
        n_elem = D.shape[1]
        D_new = np.zeros((len(new_depth), n_elem))
        for e in range(n_elem):
            D_new[:, e] = np.interp(new_depth, old_depth, D[:, e])
        return D_new
    elif D.ndim == 3:
        # (n_pixels, n_depth, n_elements)
        n_px, _, n_elem = D.shape
        D_new = np.zeros((n_px, len(new_depth), n_elem))
        for px in range(n_px):
            for e in range(n_elem):
                D_new[px, :, e] = np.interp(new_depth, old_depth, D[px, :, e])
        return D_new
    else:
        raise ValueError(f"D must be 2D or 3D, got {D.ndim}D")
