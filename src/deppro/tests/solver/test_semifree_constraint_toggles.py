"""
SemiFreeSolver constraint toggles
=================================

SemiFreeSolverConfig.use_sum_constraint / use_charge_constraint switch the
sum and charge constraints on and off independently. The constraint closures
share helpers (array sizes, erf derivatives, active depths); these were once
defined only inside the sum-constraint branch, so turning the sum constraint
off made solve() fail with a NameError from the substrate-sum constraint.

Each test records the constraints handed to the first scipy minimize call and
checks that exactly the expected ones are there. An unknown
sum_constraint_mode raises instead of being ignored.
"""

import numpy as np
import pytest

from deppro.tests.solver.test_semifree_solver import create_test_data


FILM_CHARGE = 'charge_constraint'
SUB_SUM = 'substrate_sum_constraint'


def _solve_and_record(monkeypatch, *, linked, use_sum, use_charge):
    import deppro.inverse_solver as inv
    from deppro.inverse_solver import (
        SemiFreeSolver, SemiFreeSolverConfig, SolverConfig,
    )

    calls = []
    real_minimize = inv.minimize

    def recording_minimize(*args, **kwargs):
        calls.append([c['fun'].__name__ for c in kwargs.get('constraints', [])])
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(inv, 'minimize', recording_minimize)

    I_obs, tm, elements, _, _ = create_test_data()
    semi_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        free_indices=[0, 1, 2, 3] if linked else [0, 1, 2, 3, 4],
        linked_indices=[4] if linked else [],
        substrate_indices=[5],
        use_sum_constraint=use_sum,
        use_charge_constraint=use_charge,
    )
    solver = SemiFreeSolver(
        tm,
        SolverConfig(method='LLS', alpha=0.0, scipy_method='SLSQP'),
        semi_config,
        elements=elements,
    )
    result = solver.solve(I_obs, verbose=False)

    assert calls, "minimize was not called"
    assert np.isfinite(result.thickness)
    assert np.all(np.isfinite(result.D_opt))
    return calls[0]


@pytest.mark.parametrize('use_sum', [True, False])
@pytest.mark.parametrize('use_charge', [True, False])
def test_constraint_toggles_no_linked(monkeypatch, use_sum, use_charge):
    """Without linked elements the film charge constraint is active.

    Free elements carry mixed-sign oxidation states (O is -2), so the
    charge flag decides whether the film charge constraint is added.
    """
    names = _solve_and_record(
        monkeypatch, linked=False, use_sum=use_sum, use_charge=use_charge)

    expected = []
    if use_sum:
        expected.append('sum_constraint_nolinked')
    if use_charge:
        expected.append(FILM_CHARGE)
    # The substrate ratios always sum to one; the substrate (Si, charge 0)
    # adds no substrate charge constraint.
    expected.append(SUB_SUM)
    assert names == expected


@pytest.mark.parametrize('use_sum', [True, False])
@pytest.mark.parametrize('use_charge', [True, False])
def test_constraint_toggles_linked(monkeypatch, use_sum, use_charge):
    """With O linked, only the sum flag changes the constraint list.

    The film charge constraint is not added here, but this fixture does not
    show why: the remaining free elements (C 0, Al +3, Ti +4, Si +4) have no
    mixed signs, which by itself skips it. The `not linked_idx` guard is not
    exercised by this test.
    """
    names = _solve_and_record(
        monkeypatch, linked=True, use_sum=use_sum, use_charge=use_charge)

    expected = (['sum_constraint_matlab'] if use_sum else []) + [SUB_SUM]
    assert names == expected


def test_unknown_sum_constraint_mode_raises():
    """Only the equality form exists; any other mode must not be ignored."""
    from deppro.inverse_solver import (
        SemiFreeSolver, SemiFreeSolverConfig, SolverConfig,
    )

    I_obs, tm, elements, _, _ = create_test_data()
    semi_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
        sum_constraint_mode='penalty',
    )
    solver = SemiFreeSolver(
        tm,
        SolverConfig(method='LLS', alpha=0.0, scipy_method='SLSQP'),
        semi_config,
        elements=elements,
    )
    with pytest.raises(ValueError, match='sum_constraint_mode'):
        solver.solve(I_obs, verbose=False)
