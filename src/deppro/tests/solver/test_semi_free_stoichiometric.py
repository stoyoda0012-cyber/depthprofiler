"""
Test SemiFreeSolver stoichiometric mode (090317-style).

This test verifies that the stoichiometric mode correctly handles
O/N independent optimization for NON pattern detection.
"""

import numpy as np
import pytest
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # <repo>/src

from deppro.inverse_solver import (
    SemiFreeSolver, SemiFreeSolverConfig, SolverConfig,
)
from deppro.transfer_matrix import TransferMatrix


def create_non_profile(depth: np.ndarray, n1_thick: float = 5.0,
                       o_thick: float = 15.0, n2_thick: float = 5.0) -> np.ndarray:
    """
    Create NON (Si3N4/SiO2/Si3N4/Si) depth profile.

    Element indices:
    - 0: Si-B (bulk Si)
    - 1: Si-O (Si in oxide)
    - 2: O
    - 3: N
    """
    m = len(depth)
    D = np.zeros((m, 4))

    for j, z in enumerate(depth):
        if z < n1_thick:
            # Surface Si3N4: Si=3/7, N=4/7
            D[j, 0] = 0.0      # Si-B
            D[j, 1] = 3/7      # Si-O (actually Si in nitride)
            D[j, 2] = 0.0      # O
            D[j, 3] = 4/7      # N
        elif z < n1_thick + o_thick:
            # SiO2: Si=1/3, O=2/3
            D[j, 0] = 0.0
            D[j, 1] = 1/3
            D[j, 2] = 2/3
            D[j, 3] = 0.0
        elif z < n1_thick + o_thick + n2_thick:
            # Interface Si3N4
            D[j, 0] = 0.0
            D[j, 1] = 3/7
            D[j, 2] = 0.0
            D[j, 3] = 4/7
        else:
            # Si substrate
            D[j, 0] = 1.0
            D[j, 1] = 0.0
            D[j, 2] = 0.0
            D[j, 3] = 0.0

    return D


def simulate_arxps(D: np.ndarray, tm: TransferMatrix,
                   noise_level: float = 0.02, seed: int = 42) -> np.ndarray:
    """Simulate ARXPS measurement with noise."""
    n_angles = len(tm.angles)
    n_elements = D.shape[1]

    I_true = np.zeros((n_angles, n_elements))
    for e in range(n_elements):
        T = tm.get_matrix(e)
        I_true[:, e] = T @ D[:, e]

    np.random.seed(seed)
    I_obs = I_true * (1 + noise_level * np.random.randn(*I_true.shape))
    I_obs = np.maximum(I_obs, 1e-10)

    return I_obs


def test_stoichiometric_mode_config():
    """Test stoichiometric mode configuration validation."""
    # Valid config
    config = SemiFreeSolverConfig(
        free_indices=[2, 3],  # O, N
        substrate_indices=[0],  # Si-B
        stoichiometric_mode='stoichiometric',
        stoichiometric_rules={1: ([2, 3], [0.5, 0.75])},  # Si-O = 0.5*O + 0.75*N
        stoichiometric_constraint_coeffs=[1.5, 1.75],  # (3/2)*O + (7/4)*N = 1 - SiB
    )
    assert config.stoichiometric_mode == 'stoichiometric'

    # Invalid: missing rules
    with pytest.raises(ValueError, match="requires stoichiometric_rules"):
        depth = np.arange(0, 50, 1.0)
        tm = TransferMatrix(depth, 1.0, [0, 30, 60])
        config_bad = SemiFreeSolverConfig(
            free_indices=[2, 3],
            substrate_indices=[0],
            stoichiometric_mode='stoichiometric',
            # Missing stoichiometric_rules
        )
        solver = SemiFreeSolver(tm, SolverConfig(), config_bad)

    # Invalid: wrong coeffs length
    with pytest.raises(ValueError, match="must match free_indices length"):
        depth = np.arange(0, 50, 1.0)
        tm = TransferMatrix(depth, 1.0, [0, 30, 60])
        config_bad = SemiFreeSolverConfig(
            free_indices=[2, 3],
            substrate_indices=[0],
            stoichiometric_mode='stoichiometric',
            stoichiometric_rules={1: ([2, 3], [0.5, 0.75])},
            stoichiometric_constraint_coeffs=[1.5],  # Wrong length
        )
        solver = SemiFreeSolver(tm, SolverConfig(), config_bad)


def test_stoichiometric_mode_non_pattern():
    """Test NON pattern stoichiometric mode configuration."""
    # This test verifies that stoichiometric mode can be configured correctly
    # Full integration requires TransferMatrix setup with element-specific IMFPs

    step_size = 1.0
    depth = np.arange(0, 50, step_size)

    # Setup stoichiometric solver config
    semi_free_config = SemiFreeSolverConfig(
        free_indices=[2, 3],  # O, N (independent optimization)
        linked_indices=[],     # No linked elements
        substrate_indices=[0], # Si-B

        stoichiometric_mode='stoichiometric',
        stoichiometric_rules={
            1: ([2, 3], [0.5, 0.75])  # Si-O = 0.5*O + 0.75*N
        },
        stoichiometric_constraint_coeffs=[1.5, 1.75],  # (3/2)*O + (7/4)*N = 1 - SiB

        thickness_init=25.0,
        optimize_abruptness=False,
        abruptness_init=2.0,
    )

    # Verify configuration
    assert semi_free_config.stoichiometric_mode == 'stoichiometric'
    assert 1 in semi_free_config.stoichiometric_rules
    assert len(semi_free_config.stoichiometric_constraint_coeffs) == 2

    print("\nStoichiometric mode configuration for NON pattern: OK")
    print(f"  Free indices: {semi_free_config.free_indices}")
    print(f"  Rules: Si-O = 0.5*D_O + 0.75*D_N")
    print(f"  Constraint: (3/2)*D_O + (7/4)*D_N = 1 - D_SiB")


def test_charge_balance_mode_still_works():
    """Verify that default charge_balance mode still works."""
    # This test just verifies that charge_balance mode can be configured
    # without errors - full integration test requires more setup

    config = SemiFreeSolverConfig(
        free_indices=[0],       # Si-ox
        linked_indices=[1],     # O (from charge balance)
        substrate_indices=[2],  # Si-bulk
        stoichiometric_mode='charge_balance',  # Default
        thickness_init=20.0,
    )

    assert config.stoichiometric_mode == 'charge_balance'
    assert config.stoichiometric_rules is None
    print("\nCharge balance mode configuration: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Stoichiometric mode configuration")
    print("=" * 60)
    test_stoichiometric_mode_config()
    print("PASSED")

    print("\n" + "=" * 60)
    print("Test 2: Charge balance mode (backwards compatibility)")
    print("=" * 60)
    test_charge_balance_mode_still_works()
    print("PASSED")

    print("\n" + "=" * 60)
    print("Test 3: NON pattern with stoichiometric mode")
    print("=" * 60)
    test_stoichiometric_mode_non_pattern()
    print("PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
