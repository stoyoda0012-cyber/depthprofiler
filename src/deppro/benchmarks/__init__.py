"""
deppro benchmarks package

Roundtrip benchmark for verifying depth profile reconstruction accuracy.
Co-evolution benchmark: MATLAB × Python × NN unified comparison.
"""

from .noise_sweep import compute_metrics_from_profiles

__all__ = ['compute_metrics_from_profiles']
