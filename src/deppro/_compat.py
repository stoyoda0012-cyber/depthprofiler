"""NumPy compatibility shim.

`np.trapz` was deprecated in NumPy 2.0 and **removed in 2.4** (renamed to
`np.trapezoid` to match SciPy). pyproject declares `numpy>=1.24`, and 1.x has no
`trapezoid`, so this resolves whichever exists.

Found on 2026-08-03 when CI (numpy 2.4.6) failed with
`AttributeError: module 'numpy' has no attribute 'trapz'`. The development machine
had numpy 2.3.3, which still had `trapz`,
so it did not reproduce locally.

Numerically this is a rename of the same function; the integration rule is unchanged.
"""

from __future__ import annotations

import numpy as np

# NumPy 2.0+ has trapezoid, 1.x has trapz.
trapezoid = getattr(np, "trapezoid", None)
if trapezoid is None:  # pragma: no cover - only taken on numpy 1.x
    trapezoid = np.trapz

__all__ = ["trapezoid"]
