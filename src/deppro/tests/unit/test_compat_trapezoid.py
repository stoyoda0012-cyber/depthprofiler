"""`deppro._compat.trapezoid` が台形則として正しいことを確かめる。

なぜ要るか
----------
`np.trapz` は NumPy 2.0 で非推奨、2.4 で削除された。深さプロファイルの
Wasserstein 距離・重心・モーメント（`depth_profile.py`）と面積指標
（`derived_metrics.py`）は全部この積分に乗っているので、シムが黙って
別物にすり替わると、落ちずに数値だけが変わる。

ここは「np と一致すること」ではなく **解析解と一致すること**を見る。
np の別名に張ると、np 側が変わったときに一緒に間違える。
"""

import numpy as np
import pytest

from deppro._compat import trapezoid


def test_trapezoid_matches_analytic_integral_of_a_linear_function():
    """∫₀³ 2x dx = 9。台形則は 1 次多項式に対して厳密。"""
    x = np.linspace(0.0, 3.0, 7)
    assert trapezoid(2.0 * x, x) == pytest.approx(9.0, rel=1e-12)


def test_trapezoid_matches_analytic_integral_of_a_gaussian():
    """∫ exp(-x²/2)/√(2π) dx = 1（±8σ で打ち切り）。

    台形則は 2 次以上では厳密でないので、刻みを細かくして許容差で見る。
    """
    x = np.linspace(-8.0, 8.0, 4001)
    y = np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi)
    assert trapezoid(y, x) == pytest.approx(1.0, abs=1e-9)


def test_trapezoid_handles_non_uniform_spacing():
    """不等間隔でも 1 次関数なら厳密。x 間隔を暗黙に 1 と見なしていないこと。"""
    x = np.array([0.0, 0.5, 2.0, 2.25, 5.0])
    assert trapezoid(3.0 * x, x) == pytest.approx(3.0 * 25.0 / 2.0, rel=1e-12)


def test_trapezoid_axis_argument_is_supported():
    """`derived_metrics.area` が axis= を渡すので、その経路を通しておく。"""
    x = np.linspace(0.0, 1.0, 101)
    y = np.stack([x, 2.0 * x, 3.0 * x], axis=1)  # (101, 3)
    got = trapezoid(y, x, axis=0)
    assert got == pytest.approx(np.array([0.5, 1.0, 1.5]), rel=1e-12)


def test_shim_resolved_to_the_numpy_2_name_when_available():
    """NumPy 2.x では改名後の関数を掴んでいること（1.x では skip）。"""
    if not hasattr(np, "trapezoid"):
        pytest.skip("NumPy 1.x: trapz しか存在しない")
    assert trapezoid is np.trapezoid
