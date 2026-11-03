"""Export functions for DepthResult — CSV, SVG, PDF.

CSV: tab-separated with metadata header comments.
SVG/PDF: matplotlib 2-panel publication-quality figure.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from deppro.app.depth_result import DepthResult

# Okabe-Ito palette (the same colors as the archived GUI's profile chart)
_OKABE_ITO = {
    "Al": "#56B4E9",
    "B": "#E69F00",
    "C": "#009E73",
    "O": "#D55E00",
    "Si": "#CC79A7",
    "Ti": "#0072B2",
}
_FALLBACK_COLORS = [
    "#F0E442", "#000000", "#56B4E9", "#E69F00",
    "#009E73", "#D55E00", "#CC79A7", "#0072B2",
]


def _element_color(el: str) -> str:
    return _OKABE_ITO.get(el, _FALLBACK_COLORS[hash(el) % len(_FALLBACK_COLORS)])


# ── CSV export ───────────────────────────────────────────────────────────────


def export_csv(result: DepthResult, path: str) -> None:
    """Export depth profile as CSV with metadata header.

    Format:
        # DepthInference Export
        # date: 2026-03-01
        # h5_path: /path/to/file.h5
        # mode: quick
        # nrmse: <value>
        # cond_k: <value>
        # elements: Si,O,C,Al
        #
        depth_A,Si_at%,O_at%,...,Si_CI_lower,Si_CI_upper,...
        0.0,5.0,80.0,...
    """
    elements = result.elements
    n_depth = result.n_depth

    lines: list[str] = []

    # Header comments
    lines.append("# DepthInference Export")
    lines.append(f"# date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"# h5_path: {result.metadata.get('h5_path', '')}")
    lines.append(f"# mode: {result.metadata.get('mode', 'unknown')}")
    lines.append(f"# nrmse: {result.reconstruction_nrmse:.4f}")
    lines.append(f"# cond_k: {result.condition_number:.1f}")
    lines.append(f"# elements: {','.join(elements)}")
    lines.append("#")

    # Column header
    cols = ["depth_A"]
    for el in elements:
        cols.append(f"{el}_at%")
    for el in elements:
        cols.append(f"{el}_CI_lower")
    for el in elements:
        cols.append(f"{el}_CI_upper")
    lines.append(",".join(cols))

    # Data rows
    for j in range(n_depth):
        row = [f"{result.depth_axis[j]:.2f}"]
        for i in range(len(elements)):
            row.append(f"{result.depth_map[i, j] * 100:.4f}")
        for i in range(len(elements)):
            row.append(f"{result.confidence_lower[i, j] * 100:.4f}")
        for i in range(len(elements)):
            row.append(f"{result.confidence_upper[i, j] * 100:.4f}")
        lines.append(",".join(row))

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Figure export (SVG / PDF) ────────────────────────────────────────────────


def export_figure(result: DepthResult, path: str, fmt: str = "svg") -> None:
    """Export 2-panel publication figure via matplotlib.

    Panel 1 (top, 60%): Depth profile with CI bands and depth limit.
    Panel 2 (bottom, 40%): Angle fitting — X_obs vs X_model + residuals.

    Args:
        result: DepthResult to render.
        path: Output file path (.svg or .pdf).
        fmt: 'svg' or 'pdf'.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(3, 1, height_ratios=[6, 3, 1.5], hspace=0.35)

    # ── Panel 1: Depth Profile ───────────────────────────────────────
    ax_depth = fig.add_subplot(gs[0])
    depth = result.depth_axis
    imfp = result.metadata.get("imfp_angstrom", 25.0)
    max_reliable = 3.0 * imfp

    for i, el in enumerate(result.elements):
        color = _element_color(el)
        # CI band
        ax_depth.fill_between(
            depth,
            result.confidence_lower[i] * 100,
            result.confidence_upper[i] * 100,
            color=color, alpha=0.15,
        )
        # Profile line
        ax_depth.plot(
            depth, result.depth_map[i] * 100,
            color=color, linewidth=1.8, label=el,
        )
        # Previous overlay
        if result.previous is not None and el in result.previous.elements:
            prev_idx = result.previous.elements.index(el)
            ax_depth.plot(
                result.previous.depth_axis,
                result.previous.depth_map[prev_idx] * 100,
                color=color, linewidth=1.0, linestyle="--", alpha=0.4,
            )

    # Depth limit gray-out
    if max_reliable < depth[-1]:
        ax_depth.axvspan(max_reliable, depth[-1], color="gray", alpha=0.1)
        ax_depth.axvline(max_reliable, color="gray", linewidth=0.8, linestyle="--")

    ax_depth.set_xlabel("Depth (Å)", fontsize=10)
    ax_depth.set_ylabel("Concentration (at.%)", fontsize=10)
    ax_depth.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax_depth.set_xlim(0, depth[-1])
    ax_depth.set_ylim(0, 105)
    ax_depth.tick_params(labelsize=9)

    # ── Panel 2: Angle Fitting ───────────────────────────────────────
    ax_fit = fig.add_subplot(gs[1])
    angles = result.metadata.get("angles")
    if angles is None:
        n_theta = result.observed_intensities.shape[1]
        angles = np.linspace(0, 65, n_theta)
    angles = np.asarray(angles)

    x_obs = result.observed_intensities
    x_model = x_obs - result.forward_residuals

    for i, el in enumerate(result.elements):
        color = _element_color(el)
        ax_fit.plot(angles, x_obs[i], "o", color=color, markersize=4, alpha=0.7)
        ax_fit.plot(angles, x_model[i], "-", color=color, linewidth=1.2, label=el)

    ax_fit.set_xlabel("Angle (deg)", fontsize=10)
    ax_fit.set_ylabel("Intensity", fontsize=10)
    ax_fit.legend(fontsize=8, loc="upper right", framealpha=0.8, ncol=2)
    ax_fit.tick_params(labelsize=9)

    # ── Panel 3: Residuals ───────────────────────────────────────────
    ax_resid = fig.add_subplot(gs[2])
    ax_resid.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    for i, el in enumerate(result.elements):
        color = _element_color(el)
        ax_resid.plot(
            angles, result.forward_residuals[i],
            "o-", color=color, markersize=3, linewidth=0.8,
        )

    ax_resid.set_xlabel("Angle (deg)", fontsize=10)
    ax_resid.set_ylabel("Residual", fontsize=10)
    ax_resid.tick_params(labelsize=9)

    # ── Metadata annotation ──────────────────────────────────────────
    mode = result.metadata.get("mode", "")
    nrmse_pct = result.reconstruction_nrmse * 100
    info = f"NRMSE: {nrmse_pct:.1f}%  |  cond(K): {result.condition_number:.0f}  |  mode: {mode}"
    fig.text(0.5, 0.01, info, ha="center", fontsize=8, color="gray")

    dpi = 300 if fmt == "pdf" else 150
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
