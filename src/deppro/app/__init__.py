"""Application pipeline over the deppro solvers.

Depth-profile inference, NN surrogate solving, physical validity checks,
result export, and HDF5 write-back — the layer that turns fitted peak
areas into depth profiles.

Moved here from ``toyomacro.depth`` on 2026-07-28. It lived outside
deppro only for historical reasons; the sys.path bootstrap it used to
carry is gone, since ``deppro`` now resolves as the parent package.
The reverse link still holds in one place: ``fitpara_reader`` imports
``toyomacro.io`` (optional extra ``fit``). It is resolved lazily — see
``__getattr__`` below — so ``import deppro.app`` works without toyomacro
(an optional dependency) and only ``deppro.app.read_fitpara_areas`` needs it.
"""

# Re-export deppro classes for convenient access
try:
    from deppro import (
        # IMFP
        IMFP,
        # Pipeline
        ARXPSPipeline,
        CompoundDatabase,
        CompoundParams,
        DepthProfile,
        DepthProfileBuilder,
        # Core data structures
        ElementConfig,
        # Solvers
        FreeSolver,
        GenericLayerConfig,
        GenericLayerSolver,
        LayerDefinition,
        PipelineConfig,
        PipelineResult,
        SemiFreeSolver,
        SemiFreeSolverConfig,
        SolverConfig,
        SolverRecommendation,
        SolverResult,
        # Structure analysis
        StructureAnalyzer,
        StructureType,
        # Transfer matrix
        TransferMatrix,
        TransferMatrixParams,
        create_multilayer_profile,
        # Profile factories
        create_simple_sio2_si_profile,
        create_sio2_si_with_chemical_states,
        # Convenience functions
        create_solver,
        solve_depth_profile,
    )

    DEPPRO_AVAILABLE = True

except ImportError as e:
    DEPPRO_AVAILABLE = False
    _import_error = str(e)

    def _not_available(*args, **kwargs):
        raise ImportError(
            f"deppro package not available: {_import_error}\n"
            f"Dev fallback path checked: {_deppro_path}\n"
            "Install with: uv pip install -e /path/to/depthprofiler"
        )

    # Placeholder classes
    ElementConfig = _not_available
    DepthProfile = _not_available
    TransferMatrix = _not_available
    FreeSolver = _not_available
    SemiFreeSolver = _not_available
    ARXPSPipeline = _not_available


from deppro.app.depth_result_writer import write_depth_estimate
from deppro.app.nn_solver import NN_AVAILABLE, NNDepthSolver
from deppro.app.pipeline_bridge import batch_solve, read_ezdeprof, run_pipeline


def __getattr__(name: str):
    """Resolve only ``read_fitpara_areas`` lazily.

    ``fitpara_reader`` needs the sibling package toyomacro (optional extra ``fit``).
    Since 2026-10-19 fitpara_reader itself imports toyomacro lazily inside the function,
    so resolving this attribute works without toyomacro and the ImportError (naming
    ``deppro[fit]``) comes when ``read_fitpara_areas(...)`` is called. The lazy resolution
    stays so that ``import deppro.app`` is decoupled from importing fitpara_reader.
    """
    if name == "read_fitpara_areas":
        # fitpara_reader imports toyomacro lazily, so this import no longer fails
        # for want of toyomacro; an ImportError here would be another cause (h5py,
        # a broken install) and is passed through unchanged. The toyomacro error,
        # naming `deppro[fit]`, is raised by read_fitpara_areas() itself.
        from deppro.app.fitpara_reader import read_fitpara_areas
        return read_fitpara_areas
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEPPRO_AVAILABLE",
    "read_fitpara_areas",
    "write_depth_estimate",
    "ElementConfig",
    "DepthProfile",
    "DepthProfileBuilder",
    "create_simple_sio2_si_profile",
    "create_sio2_si_with_chemical_states",
    "create_multilayer_profile",
    "TransferMatrix",
    "TransferMatrixParams",
    "IMFP",
    "CompoundParams",
    "CompoundDatabase",
    "FreeSolver",
    "SemiFreeSolver",
    "SemiFreeSolverConfig",
    "GenericLayerSolver",
    "GenericLayerConfig",
    "LayerDefinition",
    "SolverConfig",
    "SolverResult",
    "ARXPSPipeline",
    "PipelineConfig",
    "PipelineResult",
    "StructureAnalyzer",
    "StructureType",
    "SolverRecommendation",
    "create_solver",
    "solve_depth_profile",
    "read_ezdeprof",
    "batch_solve",
    "run_pipeline",
    "NN_AVAILABLE",
    "NNDepthSolver",
]
