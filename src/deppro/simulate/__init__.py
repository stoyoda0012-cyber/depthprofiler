"""
deppro.simulate — Dstk Simulator
=================================

Universal ground truth depth profile generator for ARXPS inverse problem
benchmarking and NN training data generation.

Core Components:
    SimulationConfig    - Simulation parameters (depth grid, physics, sampling)
    ElementSystemPreset - Element system definition (elements, constraints)
    RandomProfileSampler - Random profile generation with physical constraints
    ForwardModel        - Protocol for D(z) -> I(theta) transformation
    BeerLambertModel    - Beer-Lambert forward model (wraps TransferMatrix)
    ParquetWriter       - NN training data output (PyTorch compatible)
    H5Writer            - MATLAB DepthProfiler compatible output

Quick Start:
    # Generate 100 samples
    python -m deppro.simulate --n-samples 100 --preset sio2_si --output test.parquet

    # 10M samples for NN training
    python -m deppro.simulate --n-samples 10000000 --preset layered_6elem \\
        --output train.parquet --workers 0

Python API:
    from deppro.simulate import (
        SimulationConfig, get_preset, RandomProfileSampler,
        BeerLambertModel, ParquetWriter, H5Writer,
    )

    preset = get_preset('layered_6elem')
    sampler = RandomProfileSampler(preset, depth=np.arange(0, 105, 5))
    profile = sampler.sample()  # SampledProfile
"""

from .configs import SimulationConfig
from .presets import (
    ElementSystemPreset,
    LAYERED_6ELEM,
    CHECKER_7ELEM,
    SIO2_SI,
    get_preset,
    list_presets,
    register_preset,
)
from .sampler import RandomProfileSampler, SampledProfile
from .forward_model import ForwardModel, BeerLambertModel
from .writer import ParquetWriter, H5Writer
from .cli import run_simulation
from .d2s2d import D2S2DConfig, D2S2DResult, run_d2s2d

__all__ = [
    # Config
    'SimulationConfig',
    # Presets
    'ElementSystemPreset',
    'LAYERED_6ELEM',
    'CHECKER_7ELEM',
    'SIO2_SI',
    'get_preset',
    'list_presets',
    'register_preset',
    # Sampler
    'RandomProfileSampler',
    'SampledProfile',
    # Forward model
    'ForwardModel',
    'BeerLambertModel',
    # Writers
    'ParquetWriter',
    'H5Writer',
    # Runner
    'run_simulation',
    # D2S2D pipeline
    'D2S2DConfig',
    'D2S2DResult',
    'run_d2s2d',
]
