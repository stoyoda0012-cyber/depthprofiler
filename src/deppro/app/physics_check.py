"""Physical validity checks for depth profiles.

Runs automated checks on a DepthResult and returns PhysicsCheck results.
Logic is separated from GUI to enable CLI/API/test usage.
"""

from __future__ import annotations

import numpy as np

from deppro.app.depth_result import DepthResult, PhysicsCheck


def run_physics_checks(result: DepthResult) -> list[PhysicsCheck]:
    """Run all physical validity checks on a depth profile result.

    Checks:
        1. Composition sum ≈ 100% at each depth
        2. Non-negativity of all concentrations
        3. Substrate convergence at deepest layers

    Args:
        result: DepthResult to validate.

    Returns:
        List of PhysicsCheck results (one per check).
    """
    checks: list[PhysicsCheck] = []
    checks.append(_check_composition_sum(result))
    checks.append(_check_non_negativity(result))
    checks.append(_check_substrate_convergence(result))
    return checks


def _check_composition_sum(result: DepthResult) -> PhysicsCheck:
    """Check that element concentrations sum to ~1.0 at every depth."""
    total = result.depth_map.sum(axis=0)  # (n_depth,)
    deviations = np.abs(total - 1.0)
    max_dev = float(deviations.max())
    mean_dev = float(deviations.mean())
    threshold = 0.05  # 5%

    worst_idx = int(np.argmax(deviations))
    worst_depth = float(result.depth_axis[worst_idx])

    return PhysicsCheck(
        name="組成整合",
        passed=max_dev < threshold,
        value=max_dev * 100,
        threshold=threshold * 100,
        message=f"最大偏差 {max_dev * 100:.1f}%",
        detail=(
            f"平均偏差 {mean_dev * 100:.1f}%, "
            f"最大偏差 {max_dev * 100:.1f}% at depth {worst_depth:.0f} Å"
        ),
    )


def _check_non_negativity(result: DepthResult) -> PhysicsCheck:
    """Check that no concentration values are significantly negative."""
    min_val = float(result.depth_map.min())
    threshold = -0.01  # allow tiny numerical noise

    detail_parts = []
    if min_val < threshold:
        for i, el in enumerate(result.elements):
            el_min = float(result.depth_map[i].min())
            if el_min < threshold:
                idx = int(np.argmin(result.depth_map[i]))
                depth = float(result.depth_axis[idx])
                detail_parts.append(
                    f"{el}: {el_min:.4f} at depth {depth:.0f} Å"
                )

    return PhysicsCheck(
        name="非負性",
        passed=min_val >= threshold,
        value=min_val,
        threshold=threshold,
        message=f"最小値 {min_val:.4f}",
        detail="; ".join(detail_parts) if detail_parts else "全値非負",
    )


def _check_substrate_convergence(result: DepthResult) -> PhysicsCheck:
    """Check that the deepest layers converge to substrate composition.

    The substrate element (default: Si) should dominate at the bottom
    of the profile. Uses the last 5 depth points for averaging.
    """
    substrate_el = result.metadata.get("substrate_element", "Si")

    if substrate_el not in result.elements:
        return PhysicsCheck(
            name="基板収束",
            passed=True,
            value=0.0,
            threshold=0.0,
            message=f"{substrate_el} not in elements (skip)",
            detail=f"元素 {substrate_el} はプロファイルに含まれていません",
        )

    idx = result.elements.index(substrate_el)
    n_bottom = min(5, result.n_depth)
    bottom_conc = float(result.depth_map[idx, -n_bottom:].mean())
    threshold = 0.8

    return PhysicsCheck(
        name="基板収束",
        passed=bottom_conc > threshold,
        value=bottom_conc,
        threshold=threshold,
        message=f"{substrate_el} 最深部 {bottom_conc * 100:.0f}%",
        detail=(
            f"{substrate_el} 濃度 (底部{n_bottom}点平均): "
            f"{bottom_conc * 100:.1f}% (閾値: {threshold * 100:.0f}%)"
        ),
    )


def overall_traffic_light(checks: list[PhysicsCheck]) -> str:
    """Determine overall traffic light from a list of checks.

    Returns:
        'green' if all pass, 'yellow' if 1 failure, 'red' if 2+ failures.
    """
    n_fail = sum(not c.passed for c in checks)
    if n_fail == 0:
        return "green"
    elif n_fail == 1:
        return "yellow"
    return "red"
