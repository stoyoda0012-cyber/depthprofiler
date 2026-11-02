"""MLX batched solvers for the ARXPS inverse problem (Apple Silicon GPU).

The solvers were built in stages, and the stage letters appear in the code:
Stage A, batched FISTA for FreeSolver OLS/WLS with L2/LLS; Stage B adds
L1 (exact |.|_1), MEM (I-div) and Poisson-NLL (IRLS); Stage C, the
SemiFreeSolver class (block coordinate: FISTA concentration block +
batched grid + C-refine thickness search); Stage D, the
GenericLayerSolver class (batched Levenberg-Marquardt on layer
thicknesses, sigmoid-reparametrized bounds, CPU-stream normal-equation
solve). Raggedness-free by construction: fixed iteration
rounds, masked accept/reject, closed-form projections.

The SciPy paths remain the reference implementation; select this backend
explicitly via ``MLXBatchFreeSolver`` / ``MLXBatchSemiFreeSolver`` /
``MLXBatchGenericLayerSolver`` or ``inverse_method='mlx_free'`` /
``'mlx_semifree'`` in d2s2d.

The tests in tests/solver/test_mlx_*.py state what is checked here.
"""
from .projections import project_capped_simplex
from .free_batch import FreeBatchConfig, FreeBatchResult, solve_free_batch
from .semifree_batch import (
    SemiFreeBatchConfig, SemiFreeBatchResult, solve_semifree_batch,
)
from .layer_lm import LayerLMConfig, LayerLMResult, solve_layer_lm
from .adapter import (
    MLXBatchFreeSolver, MLXBatchSemiFreeSolver, MLXBatchGenericLayerSolver,
)

__all__ = [
    'project_capped_simplex',
    'FreeBatchConfig',
    'FreeBatchResult',
    'solve_free_batch',
    'SemiFreeBatchConfig',
    'SemiFreeBatchResult',
    'solve_semifree_batch',
    'LayerLMConfig',
    'LayerLMResult',
    'solve_layer_lm',
    'MLXBatchFreeSolver',
    'MLXBatchSemiFreeSolver',
    'MLXBatchGenericLayerSolver',
]
