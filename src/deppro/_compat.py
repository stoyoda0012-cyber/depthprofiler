"""NumPy 互換シム。

`np.trapz` は NumPy 2.0 で非推奨、**2.4 で削除**された（SciPy と名前を
揃えるため `np.trapezoid` に改名）。pyproject は `numpy>=1.24` を宣言して
いて 1.x には `trapezoid` が無いので、どちらでも動くようここで解決する。

2026-08-03 に CI（numpy 2.4.6）が
`AttributeError: module 'numpy' has no attribute 'trapz'` で落ちて発覚した。
開発機の numpy は 2.3.3 でまだ `trapz` を持っていたため、ローカルでは
再現しなかった。

数値としては同一関数の改名であって、積分規則は変わっていない。
"""

from __future__ import annotations

import numpy as np

# NumPy 2.0+ は trapezoid、1.x は trapz。
trapezoid = getattr(np, "trapezoid", None)
if trapezoid is None:  # pragma: no cover - numpy 1.x のみ通る枝
    trapezoid = np.trapz

__all__ = ["trapezoid"]
