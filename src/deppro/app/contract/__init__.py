"""P1 契約層: ProblemSpec（観測のみ）/ GroundTruth（別）/ 既存指標 Evaluator / provenance / 対象台帳。

再 forward の合否は ``metrics.closure_gate`` が付ける（noise = 0 のデータセットだけ PASS/FAIL、
noise > 0 では残差を記録するだけ）。core からこの層を import しない。
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
