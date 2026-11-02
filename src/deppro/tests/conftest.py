"""
pytest configuration for deppro tests

Shared fixtures and configuration for all test modules.
"""

import pytest
import numpy as np


@pytest.fixture
def seed():
    """Fixed random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def standard_angles():
    """Standard measurement angles."""
    return [0, 15, 30, 45, 60, 75]


@pytest.fixture
def narrow_angles():
    """Narrow angle range (near-ambient XPS)."""
    return [0, 5, 10, 15, 20]


@pytest.fixture
def standard_thicknesses():
    """Standard 4-layer film thicknesses."""
    return [5.0, 15.0, 35.0, 30.0]


@pytest.fixture
def standard_interface_widths():
    """Standard interface widths."""
    return [2.0, 3.0, 4.0, 5.0]
