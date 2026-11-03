"""P1 contract layer: ProblemSpec (observations only) / GroundTruth (separate) / the existing Evaluator metrics / provenance / the dataset registry.

The pass/fail of the re-forward is assigned by ``metrics.closure_gate`` (PASS/FAIL only for datasets with noise = 0;
with noise > 0 only the residual is recorded). The core does not import this layer.
"""
from .dataset_files import (  # noqa: F401
    FAMILIES, FolderSpec, discover_files, jsonable, parse_dataset_folder,
)
from .metrics import (  # noqa: F401
    CLOSURE_TOL_ABS_NOISEFREE, METRIC_VERSION, REFORWARD_FORMULA_VERSION, closure_gate,
    composition_transfer_matrix, evaluate, forward, observation_residual, thickness_array,
    thickness_half_crossing, w1_matrix,
)
from .problem import (  # noqa: F401
    Axes, DatasetRef, Elements, GroundTruth, ProblemSpec, load_existing_result, load_problem,
    select_pixel_ids,
)
from .provenance import (  # noqa: F401
    CONTRACT_VERSION, RunManifest, SolverRun, code_revision, config_hash, environment,
    load_run_manifest, new_run_id, save_arrays, write_run_manifest,
)
from .registry import (  # noqa: F401
    DatasetEntry, Registry, benchmark_targets, default_registry, entry, excluded, explain, load_registry,
)
