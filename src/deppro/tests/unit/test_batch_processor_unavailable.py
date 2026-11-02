"""The batch / Monte Carlo path fails loudly instead of failing every sample.

``deppro.batch_processor`` was written for ``MultiLayerFilmSolver``, which has
been removed. Before 2026-10 ``process_batch`` caught the import error inside
each worker, so every sample was only counted as failed, and the CLI commands
``batch`` / ``monte-carlo`` stopped with an ImportError. Until the path is
ported, every entry point raises (or the CLI exits 1) with a message that says
why.
"""
from __future__ import annotations

import numpy as np
import pytest

from deppro import batch_processor as bp
from deppro.cli_depth import main
from deppro.data_io import IntensityData

_DATA = IntensityData(angles=np.array([0.0, 30.0, 60.0]), element_labels=["Si", "O"],
                      intensities=np.full((3, 2), 0.5))


def _builder(data):  # never reached
    raise AssertionError("the transfer-matrix builder must not be called")


@pytest.mark.parametrize("call", [
    lambda: bp.BatchProcessor().process_batch([_DATA], None, _builder),
    lambda: bp.BatchProcessor().monte_carlo_batch(_DATA, None, _builder, 0.02, n_trials=2, seed=0),
    lambda: bp.parallel_monte_carlo(_DATA, None, _builder, 0.02, n_trials=2, n_workers=1, seed=0),
    lambda: bp._solve_single((0, _DATA, None, _builder)),
], ids=["process_batch", "monte_carlo_batch", "parallel_monte_carlo", "_solve_single"])
def test_entry_points_raise_with_the_reason(call):
    with pytest.raises(NotImplementedError, match="MultiLayerFilmSolver"):
        call()


@pytest.mark.parametrize("argv", [
    ["batch", "--input-dir", "."],
    ["monte-carlo", "--input", "nonexistent.csv"],
])
def test_cli_commands_exit_1_with_the_reason(argv, capsys):
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "unavailable" in err and "MultiLayerFilmSolver" in err
