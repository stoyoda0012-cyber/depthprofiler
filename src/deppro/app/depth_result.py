"""Data models for DepthInference depth profile analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class PhysicsCheck:
    """Result of a single physical validity check on a depth profile.

    Attributes:
        name: Short check name (e.g., "composition sum", "non-negativity")
        passed: Whether the check passed
        value: Measured value
        threshold: Threshold for pass/fail
        message: Human-readable summary
        detail: Extended detail for collapsible panel
    """

    name: str
    passed: bool
    value: float
    threshold: float
    message: str
    detail: str = ""


@dataclass
class DepthResult:
    """Complete result from a depth profile inference.

    Phase 1: single-point ARXPS measurement (1 site, n_angles, n_elements).

    Attributes:
        depth_map: Concentration profiles, shape (n_elem, n_depth), values in [0, 1].
        elements: Element labels matching axis 0 of depth_map.
        depth_axis: Depth grid in Angstroms, shape (n_depth,).
        confidence_lower: Lower bound of confidence band, same shape as depth_map.
        confidence_upper: Upper bound of confidence band, same shape as depth_map.
        reconstruction_nrmse: Normalized RMSE of forward simulation vs observed.
        condition_number: Condition number of the transfer matrix.
        physics_checks: Results of physical validity checks.
        forward_residuals: X_obs - X_model, shape (n_elem, n_theta).
        observed_intensities: Raw angle-resolved intensities, shape (n_elem, n_theta).
        metadata: Extensible metadata dict (lot_id, wafer_id, timestamp, etc.).
        previous: Previous DepthResult for overlay comparison.
    """

    depth_map: NDArray[np.floating]
    elements: list[str]
    depth_axis: NDArray[np.floating]
    confidence_lower: NDArray[np.floating]
    confidence_upper: NDArray[np.floating]
    reconstruction_nrmse: float
    condition_number: float
    physics_checks: list[PhysicsCheck]
    forward_residuals: NDArray[np.floating]
    observed_intensities: NDArray[np.floating]
    metadata: dict = field(default_factory=dict)
    previous: DepthResult | None = None

    def __post_init__(self) -> None:
        n_elem = len(self.elements)
        n_depth = len(self.depth_axis)

        if self.depth_map.shape != (n_elem, n_depth):
            raise ValueError(
                f"depth_map shape {self.depth_map.shape} != "
                f"({n_elem}, {n_depth})"
            )
        if self.confidence_lower.shape != (n_elem, n_depth):
            raise ValueError(
                f"confidence_lower shape {self.confidence_lower.shape} != "
                f"({n_elem}, {n_depth})"
            )
        if self.confidence_upper.shape != (n_elem, n_depth):
            raise ValueError(
                f"confidence_upper shape {self.confidence_upper.shape} != "
                f"({n_elem}, {n_depth})"
            )

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_depth(self) -> int:
        return len(self.depth_axis)

    @property
    def all_checks_passed(self) -> bool:
        return all(c.passed for c in self.physics_checks)

    @property
    def traffic_light(self) -> str:
        """Return traffic-light status: 'green', 'yellow', or 'red'."""
        n_fail = sum(not c.passed for c in self.physics_checks)
        if n_fail == 0:
            return "green"
        elif n_fail == 1:
            return "yellow"
        return "red"

    def interface_depth(self, element_idx: int, threshold: float = 0.5) -> float | None:
        """Find depth where concentration crosses threshold (50% by default).

        Returns depth in Angstroms, or None if threshold not crossed.
        """
        profile = self.depth_map[element_idx]
        crossings = np.where(np.diff(np.sign(profile - threshold)))[0]
        if len(crossings) == 0:
            return None
        # Linear interpolation at first crossing
        i = crossings[0]
        d0, d1 = self.depth_axis[i], self.depth_axis[i + 1]
        v0, v1 = profile[i], profile[i + 1]
        if abs(v1 - v0) < 1e-12:
            return float(d0)
        return float(d0 + (threshold - v0) / (v1 - v0) * (d1 - d0))

    def interface_width(
        self, element_idx: int, low: float = 0.1, high: float = 0.9
    ) -> float | None:
        """Calculate interface width as distance between low% and high% depths."""
        d_low = self.interface_depth(element_idx, threshold=low)
        d_high = self.interface_depth(element_idx, threshold=high)
        if d_low is None or d_high is None:
            return None
        return abs(d_high - d_low)
