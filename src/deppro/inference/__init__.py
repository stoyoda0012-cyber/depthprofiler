"""
AI Inference Engine for ARXPS Depth Profiling
=============================================

MLX-based neural network for instant depth profile prediction.
Replaces iterative SLSQP solver with trained MLP.

Usage:
    python -m deppro.inference train --train train.parquet --val val.parquet
    python -m deppro.inference train-h5 --h5 benchmark.h5 --output model.npz
    python -m deppro.inference benchmark --model model.npz --h5 benchmark.h5 --h5-order
"""

from .model import DepthMLP, create_model, register_architecture
from .data import ParquetDataset
from .train import train_model, train_from_h5, load_model
from .features import truncated_pinv
from .postprocess import (
    enforce_constraints,
    enforce_constraints_mlx,
    normalize_intensity_mlx,
    estimate_thickness_mlx,
)

__all__ = [
    'DepthMLP',
    'create_model',
    'register_architecture',
    'ParquetDataset',
    'train_model',
    'train_from_h5',
    'load_model',
    'truncated_pinv',
    'enforce_constraints',
    'enforce_constraints_mlx',
    'normalize_intensity_mlx',
    'estimate_thickness_mlx',
]

# Optional extensions (see deppro/__init__.py): a distribution may ship
# ``deppro.inference._extensions``, which imports modules that register
# further architectures. Its absence is the normal case.
import importlib.util as _importlib_util

if _importlib_util.find_spec(f"{__name__}._extensions") is not None:
    from . import _extensions  # noqa: F401
del _importlib_util
