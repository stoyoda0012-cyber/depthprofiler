"""
deppro - Depth Profile Analysis Package

Core modules for XPS depth profile simulation and analysis.

Modules:
    depth_profile: Layer structure and concentration profile
    transfer_matrix: Electron intensity calculation with IMFP
    imfp: Inelastic Mean Free Path calculations (TPP-2M)
    depth_generator: Generate simulated depth profiles
    cli_depth: Command-line interface for depth profile operations

Date: 2026-01-22
"""

from .depth_profile import (
    ElementConfig,
    LayerConfig,
    DepthProfile,
    DistributionFunction,
    DepthProfileBuilder,
    create_five_layer_profile,
    create_simple_sio2_si_profile,
    create_sio2_si_with_chemical_states,
    create_multilayer_profile,
    MultiLayerFilmConfig,
    verify_multilayer_charge_balance,
    verify_multilayer_stoichiometry,
    # Constraint utilities
    COMMON_OXIDATION_STATES,
    parse_compound_formula,
    infer_oxidation_states,
    create_charge_constraint_matrix,
    create_stoichiometry_constraint_matrix,
    create_concentration_sum_constraint_matrix,
    # Profile comparison metrics
    wasserstein_distance_1d,
    centroid,
    profile_moments,
    ProfileComparisonResult,
    compare_profiles,
    compare_with_array,
    # Model-free interface detection
    detect_interfaces_gradient,
    detect_interfaces_50percent,
    detect_interfaces_inflection,
    InterfaceDetectionResult,
    detect_interfaces,
    # Exponential Prior for MEM (Livesey 1994)
    create_exponential_prior,
)
from .transfer_matrix import (
    TransferMatrix,
    TransferMatrixParams,
    AngleResolvedData,
    compute_angle_profile,
    analyze_transfer_matrix_condition,
    suggest_optimal_parameters,
)
from .depth_grid import (
    make_uniform_grid,
    make_geometric_grid,
    make_log_grid,
    make_grid,
    register_grid_type,
    interpolate_profiles_to_grid,
    GRID_TYPES,
)
from .imfp import (
    IMFP,
    CompoundParams,
    CompoundDatabase,
    BindingEnergyDatabase,
    calculate_imfp,
)
# depth_generator is NOT imported here — see the lazy __getattr__ at the end of
# this module. It needs toyomacro, which pyproject declares as the optional
# `fit` extra, and importing it eagerly made that extra mandatory in practice.
from .inverse_solver import (
    # New names (recommended)
    FreeSolver,
    SemiFreeSolver,
    SemiFreeSolverConfig,
    SemiFreeSolverResult,
    # Generic Layer Solver (general version, recommended)
    LayerDefinition,
    GenericLayerConfig,
    GenericLayerResult,
    GenericLayerSolver,
    # Common
    SolverConfig,
    SolverResult,
    BatchInverseSolver,
    VectorizedBatchSolver,
    IterativeMEMSolver,
    reconstruct_depth_profile,
    # Backward compatibility aliases (deprecated)
    InverseSolver,  # → FreeSolver
    FilmStructureSolver,  # → SemiFreeSolver
    FilmStructureConfig,  # → SemiFreeSolverConfig
    FilmStructureResult,  # → SemiFreeSolverResult
)
from .data_io import (
    IntensityData,
    ModelConfig,
    ElementDef,
    AnalysisConfig,
    load_intensity_csv,
    save_intensity_csv,
    load_config_yaml,
    save_config_yaml,
    create_default_config,
    save_result_json,
    save_profile_csv,
    load_vamas,
    load_kratos,
)
from .batch_processor import (
    BatchProcessor,
    ParallelConfig,
    BatchResult,
    parallel_monte_carlo,
    parallel_bootstrap,
)
from .profile_4d import (
    Profile4D,
    Profile4DConfig,
    TimeSeriesAnalyzer,
    create_profile_4d,
    load_profile_4d,
)
from .structure_analyzer import (
    StructureAnalyzer,
    StructureType,
    SolverRecommendation,
    AnalysisResult,
    SubstrateDetectionResult,
)
from .pipeline import (
    ARXPSPipeline,
    PipelineConfig,
    PipelineResult,
    ValidationStatus,
    ValidationResult,
)
from .simulate import (
    SimulationConfig,
    ElementSystemPreset,
    RandomProfileSampler,
    SampledProfile,
    ForwardModel,
    BeerLambertModel,
    get_preset,
    list_presets,
)

# AR-HAXPES real data analysis pipeline (subpackage)
# Usage: from deppro.haxpes import HAXPESPipeline, FIX_IBW_INSTRUMENT, CALTIOSI_EXPERIMENT
# See deppro/haxpes/__init__.py for full API

__version__ = "0.2.0"


# =============================================================================
# Convenience Functions for Default Solver (SemiFreeSolver)
# =============================================================================

def create_solver(
    transfer_matrix: TransferMatrix,
    solver_type: str = 'semi-free',
    method: str = 'LLS',
    alpha: float = 0.0,
    **kwargs,
):
    """
    Factory for solvers (SemiFreeSolver by default)

    Parameters
    ----------
    transfer_matrix : TransferMatrix
        Transfer-matrix object
    solver_type : str, optional
        Solver type: 'semi-free' (default), 'free', 'layer-structure'
    method : str, optional
        Regularization method: 'LLS' (default, recommended), 'L2', 'L1', 'MEM'
    alpha : float, optional
        Regularization parameter (default: 0.0, unused by LLS)
    **kwargs
        Solver-specific parameters

        semi-free:
            thickness_init : float - initial thickness estimate (default: 30.0)
            abruptness_init : float - initial interface width (default: 5.0)
            free_indices : list - indices of the freely optimized elements
            linked_indices : list - indices of the charge-conservation elements
            substrate_indices : list - indices of the substrate elements

        generic-layer:
            layer_config : GenericLayerConfig - layer structure
            elements : list - element configuration

    Returns
    -------
    solver : FreeSolver | SemiFreeSolver | GenericLayerSolver

    Examples
    --------
    >>> # SemiFreeSolver (recommended, default)
    >>> solver = create_solver(tm, free_indices=[0], linked_indices=[1], substrate_indices=[2])
    >>> result = solver.solve(I_obs)

    >>> # FreeSolver (exploratory analysis)
    >>> solver = create_solver(tm, solver_type='free')

    >>> # GenericLayerSolver (precise thickness measurement)
    >>> from deppro import GenericLayerConfig, LayerDefinition
    >>> layer_config = GenericLayerConfig(...)
    >>> solver = create_solver(tm, solver_type='generic-layer', layer_config=layer_config)
    """
    solver_config = SolverConfig(method=method, alpha=alpha)

    if solver_type == 'semi-free':
        # Extract the SemiFreeSolver parameters
        semi_config = SemiFreeSolverConfig(
            thickness_init=kwargs.get('thickness_init', 30.0),
            abruptness_init=kwargs.get('abruptness_init', 5.0),
            free_indices=kwargs.get('free_indices', []),
            linked_indices=kwargs.get('linked_indices', []),
            substrate_indices=kwargs.get('substrate_indices', []),
            thickness_bounds=kwargs.get('thickness_bounds', (1.0, 200.0)),
            abruptness_bounds=kwargs.get('abruptness_bounds', (0.5, 30.0)),
            optimize_thickness=kwargs.get('optimize_thickness', True),
            optimize_abruptness=kwargs.get('optimize_abruptness', False),
        )
        D_prior = kwargs.get('D_prior')
        elements = kwargs.get('elements')
        return SemiFreeSolver(transfer_matrix, solver_config, semi_config, D_prior, elements)

    elif solver_type == 'free':
        D_prior = kwargs.get('D_prior')
        elements = kwargs.get('elements')
        return FreeSolver(transfer_matrix, solver_config, D_prior, elements)

    elif solver_type == 'generic-layer':
        # For GenericLayerSolver (layer_config is passed directly)
        layer_config = kwargs.get('layer_config')
        if layer_config is None:
            raise ValueError("layer_config is required for 'generic-layer' solver_type")
        elements = kwargs.get('elements')
        return GenericLayerSolver(transfer_matrix, solver_config, layer_config, elements)

    else:
        raise ValueError(f"Unknown solver_type: {solver_type}. "
                        f"Use 'semi-free', 'free', or 'generic-layer'")


def solve_depth_profile(
    I_obs,
    transfer_matrix: TransferMatrix,
    elements=None,
    D_prior=None,
    free_indices=None,
    linked_indices=None,
    substrate_indices=None,
    thickness_init: float = 30.0,
    method: str = 'LLS',
    alpha: float = 0.0,
    verbose: bool = False,
):
    """
    Solve the depth-profile inverse problem (SemiFreeSolver by default)

    The simplest API. Solves the inverse problem with SemiFreeSolver.

    Parameters
    ----------
    I_obs : ndarray
        Observed intensities (n_angles, n_elements)
    transfer_matrix : TransferMatrix
        Transfer-matrix object
    elements : list[ElementConfig], optional
        List of element configurations
    D_prior : ndarray, optional
        Prior distribution (n_depth, n_elements)
    free_indices : list[int], optional
        Indices of the freely optimized elements
    linked_indices : list[int], optional
        Indices of the elements determined by charge conservation
    substrate_indices : list[int], optional
        Indices of the substrate elements
    thickness_init : float, optional
        Initial thickness estimate (default: 30.0 Å)
    method : str, optional
        Regularization method (default: 'LLS', recommended)
    alpha : float, optional
        Regularization parameter (default: 0.0, unused by LLS)
    verbose : bool, optional
        Verbose output (default: False)

    Returns
    -------
    result : SemiFreeSolverResult
        Analysis result (includes D_opt, thickness, abruptness and more)

    Examples
    --------
    >>> # The simplest use
    >>> result = solve_depth_profile(
    ...     I_obs, tm,
    ...     free_indices=[0],      # Si_ox
    ...     linked_indices=[1],    # O (charge conservation)
    ...     substrate_indices=[2], # Si_bulk
    ... )
    >>> print(f"Thickness: {result.thickness:.1f} Å")
    """
    import numpy as np

    # Default indices
    if free_indices is None:
        free_indices = []
    if linked_indices is None:
        linked_indices = []
    if substrate_indices is None:
        substrate_indices = []

    # SemiFreeSolver settings
    semi_config = SemiFreeSolverConfig(
        thickness_init=thickness_init,
        abruptness_init=5.0,
        free_indices=free_indices,
        linked_indices=linked_indices,
        substrate_indices=substrate_indices,
        optimize_thickness=True,
        optimize_abruptness=False,  # keep the interface width fixed (recommended)
    )

    solver_config = SolverConfig(method=method, alpha=alpha)

    solver = SemiFreeSolver(
        transfer_matrix, solver_config, semi_config, D_prior, elements
    )

    return solver.solve(I_obs, verbose=verbose)

__all__ = [
    # depth_profile
    "ElementConfig",
    "LayerConfig",
    "DepthProfile",
    "DistributionFunction",
    "DepthProfileBuilder",
    "create_five_layer_profile",
    "create_simple_sio2_si_profile",
    "create_sio2_si_with_chemical_states",
    "create_multilayer_profile",
    "MultiLayerFilmConfig",
    "verify_multilayer_charge_balance",
    "verify_multilayer_stoichiometry",
    # Profile comparison metrics
    "wasserstein_distance_1d",
    "centroid",
    "profile_moments",
    "ProfileComparisonResult",
    "compare_profiles",
    "compare_with_array",
    # Model-free interface detection
    "detect_interfaces_gradient",
    "detect_interfaces_50percent",
    "detect_interfaces_inflection",
    "InterfaceDetectionResult",
    "detect_interfaces",
    # Constraint utilities
    "COMMON_OXIDATION_STATES",
    "parse_compound_formula",
    "infer_oxidation_states",
    "create_charge_constraint_matrix",
    "create_stoichiometry_constraint_matrix",
    "create_concentration_sum_constraint_matrix",
    # Exponential Prior for MEM (Livesey 1994)
    "create_exponential_prior",
    # transfer_matrix
    "TransferMatrix",
    "TransferMatrixParams",
    "AngleResolvedData",
    "compute_angle_profile",
    "analyze_transfer_matrix_condition",
    "suggest_optimal_parameters",
    # depth_grid
    "make_uniform_grid",
    "make_geometric_grid",
    "make_log_grid",
    "make_grid",
    "register_grid_type",
    "interpolate_profiles_to_grid",
    "GRID_TYPES",
    # imfp
    "IMFP",
    "CompoundParams",
    "CompoundDatabase",
    "BindingEnergyDatabase",
    "calculate_imfp",
    # depth_generator
    "DepthSpectraGenerator",
    "DepthGeneratorConfig",
    "generate_from_profile",
    # inverse_solver (new names)
    "FreeSolver",
    "SemiFreeSolver",
    "SemiFreeSolverConfig",
    "SemiFreeSolverResult",
    # Generic Layer Solver (recommended)
    "LayerDefinition",
    "GenericLayerConfig",
    "GenericLayerResult",
    "GenericLayerSolver",
    "SolverConfig",
    "SolverResult",
    "BatchInverseSolver",
    "VectorizedBatchSolver",
    "IterativeMEMSolver",
    "reconstruct_depth_profile",
    # inverse_solver (deprecated aliases)
    "InverseSolver",
    "FilmStructureSolver",
    "FilmStructureConfig",
    "FilmStructureResult",
    # data_io
    "IntensityData",
    "ModelConfig",
    "ElementDef",
    "AnalysisConfig",
    "load_intensity_csv",
    "save_intensity_csv",
    "load_config_yaml",
    "save_config_yaml",
    "create_default_config",
    "save_result_json",
    "save_profile_csv",
    "load_vamas",
    "load_kratos",
    # batch_processor
    "BatchProcessor",
    "ParallelConfig",
    "BatchResult",
    "parallel_monte_carlo",
    "parallel_bootstrap",
    # profile_4d
    "Profile4D",
    "Profile4DConfig",
    "TimeSeriesAnalyzer",
    "create_profile_4d",
    "load_profile_4d",
    # Convenience functions (SemiFreeSolver as default)
    "create_solver",
    "solve_depth_profile",
    # structure_analyzer
    "StructureAnalyzer",
    "StructureType",
    "SolverRecommendation",
    "AnalysisResult",
    "SubstrateDetectionResult",
    # pipeline
    "ARXPSPipeline",
    "PipelineConfig",
    "PipelineResult",
    "ValidationStatus",
    "ValidationResult",
    # simulate
    "SimulationConfig",
    "ElementSystemPreset",
    "RandomProfileSampler",
    "SampledProfile",
    "ForwardModel",
    "BeerLambertModel",
    "get_preset",
    "list_presets",
]


# ---------------------------------------------------------------------------
# Lazily resolved exports (the optional `fit` extra)
#
# depth_generator used to import toyomacro.voigtfit unconditionally while
# pyproject declares toyomacro as the optional extra `fit`. Re-exporting its
# symbols eagerly made the two disagree: `import deppro` raised
# ModuleNotFoundError without the extra, so a plain `pip install deppro`
# produced a package that could not be imported at all, and pytest died while
# loading conftest before any importorskip could take effect (CI, 2026-08-02).
#
# Since 2026-10-19 depth_generator itself imports toyomacro lazily, so the
# names below resolve without the extra and the ImportError (naming
# `deppro[fit]`) is raised when a generator is constructed. The lazy export is
# kept so that `import deppro` stays independent of depth_generator's import
# cost and of any future eager dependency there.
# ---------------------------------------------------------------------------
_LAZY_FIT_EXPORTS = frozenset({
    "DepthSpectraGenerator",
    "DepthGeneratorConfig",
    "generate_from_profile",
})


def __getattr__(name):
    if name in _LAZY_FIT_EXPORTS:
        from . import depth_generator
        return getattr(depth_generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_FIT_EXPORTS)


# ---------------------------------------------------------------------------
# Optional extensions
#
# A distribution may ship ``deppro._extensions``, which imports modules that
# add grid types or model architectures through the public registration
# functions (``register_grid_type``, ``deppro.inference.register_architecture``).
# Its absence is the normal case; errors raised inside it are not hidden.
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util

if _importlib_util.find_spec(f"{__name__}._extensions") is not None:
    from . import _extensions  # noqa: F401
del _importlib_util
