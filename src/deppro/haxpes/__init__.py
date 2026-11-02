"""
deppro.haxpes - AR-HAXPES Real Data Analysis Pipeline
======================================================

Modules for loading, preprocessing, and analyzing angle-resolved
Hard X-ray Photoelectron Spectroscopy data from Scienta Omicron instruments.

Supported data formats:
    - IBW: Igor Binary Wave (fix mode, 1001 detector pixels)
    - PXT: Scienta format (fix mode, pre-binned 12 angle channels)

Quick start::

    from deppro.haxpes import HAXPESPipeline, FIX_IBW_INSTRUMENT, CALTIOSI_EXPERIMENT

    pipeline = HAXPESPipeline(
        instrument=FIX_IBW_INSTRUMENT,
        experiment=CALTIOSI_EXPERIMENT,
    )
    result = pipeline.run_fix_ibw(FIX_IBW_DIR)
"""
from .config import (
    HAXPESInstrumentConfig,
    HAXPESExperimentConfig,
    HAXPESSolverConfig,
    FIX_IBW_INSTRUMENT,
    FIX_PXT_INSTRUMENT,
    CALTIOSI_EXPERIMENT,
    FIX_IBW_DIR,
    FIX_PXT_DIR,
    MAP_H5,
    OUTPUT_DIR,
)
from .loader import (
    RawElementData,
    load_fix_ibw,
    load_prebinned_pxt,
    load_prebinned_pixel_stacks,
    sum_and_bin,
)
from .preprocessor import (
    detector_to_emission,
    make_uniform_bins,
    bin_detector_data,
    subtract_background,
    integrate_peak_areas,
    apply_cross_section_correction,
    compute_si_normalized_rrmse,
)
from .pipeline import (
    PipelineResult,
    process_single_pixel,
    run_depth_profiling,
    run_3d_map,
    HAXPESPipeline,
)
from .bootstrap import (
    BootstrapResult,
    average_pixels,
    build_perbin_pixels,
    build_frame_pixels,
    frame_bootstrap,
    block_bootstrap,
)
try:
    from .denoiser import HaxpesDenoiser
except ModuleNotFoundError as _e:
    if _e.name != f"{__name__}.denoiser":
        raise
    # Tolerate a checkout that omits the denoiser wrapper. Everything else in
    # deppro.haxpes works without it; callers that need it check for None.
    # (It needs the optional dnndenoiser package only when HaxpesDenoiser
    # is constructed.)
    HaxpesDenoiser = None

__all__ = [
    # Config dataclasses
    'HAXPESInstrumentConfig',
    'HAXPESExperimentConfig',
    'HAXPESSolverConfig',
    # Pre-built configs
    'FIX_IBW_INSTRUMENT',
    'FIX_PXT_INSTRUMENT',
    'CALTIOSI_EXPERIMENT',
    # Path constants
    'FIX_IBW_DIR',
    'FIX_PXT_DIR',
    'MAP_H5',
    'OUTPUT_DIR',
    # Loader
    'RawElementData',
    'load_fix_ibw',
    'load_prebinned_pxt',
    'load_prebinned_pixel_stacks',
    'sum_and_bin',
    # Preprocessor
    'detector_to_emission',
    'make_uniform_bins',
    'bin_detector_data',
    'subtract_background',
    'integrate_peak_areas',
    'apply_cross_section_correction',
    'compute_si_normalized_rrmse',
    # Pipeline
    'PipelineResult',
    'process_single_pixel',
    'run_depth_profiling',
    'run_3d_map',
    'HAXPESPipeline',
    # Bootstrap
    'BootstrapResult',
    'average_pixels',
    'build_perbin_pixels',
    'build_frame_pixels',
    'frame_bootstrap',
    'block_bootstrap',
    # Denoiser
    'HaxpesDenoiser',
]
