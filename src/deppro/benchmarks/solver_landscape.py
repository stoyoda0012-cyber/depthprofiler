#!/usr/bin/env python3
"""
Solver Landscape Visualization
==============================

3種の可視化:
  (A) Decision Boundary Map — どの問題にどのソルバーが最適か
  (B) ヒートマップ + レーダーチャート
  (C) PCA 散布図

Usage:
    from deppro.benchmarks.solver_landscape import plot_all
    plot_all(results, scenarios, output_dir)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec


# Solver colors (consistent across all plots)
SOLVER_COLORS = {
    'Free': '#E74C3C',        # red
    'SemiFree': '#3498DB',    # blue
    'GenericLayer': '#2ECC71', # green
    'NN': '#9B59B6',          # purple
    'Hybrid': '#F39C12',      # orange
    'MATLAB': '#7F8C8D',      # gray (reference)
}
SOLVER_MARKERS = {
    'Free': 'o',
    'SemiFree': 's',
    'GenericLayer': '^',
    'NN': 'D',
    'Hybrid': 'P',
    'MATLAB': 'X',
}
SOLVER_NAMES = ['Free', 'SemiFree', 'GenericLayer', 'NN', 'Hybrid']

# Metric display names
METRIC_LABELS = {
    'ismp_rmse': 'Ismp RMSE',
    'depth_rmse': 'Depth RMSE',
    'w1_mean': 'W1 Mean (Å)',
    'sigma_dthk': 'σ(Δthk) (Å)',
}


def _extract_metric(agg, metric_name: str) -> float:
    """AggregatedMetrics から指定メトリクスを取得."""
    if metric_name == 'ismp_rmse':
        return agg.ismp_rmse_mean
    elif metric_name == 'depth_rmse':
        return agg.depth_rmse_mean
    elif metric_name == 'w1_mean':
        return agg.w1_mean_mean
    elif metric_name == 'sigma_dthk':
        return agg.sigma_dthk
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def _parse_scenario_name(name: str) -> dict:
    """シナリオ名からパラメータを抽出."""
    parts = name.split('_')
    params = {}
    for p in parts:
        if p.startswith('thk'):
            params['thickness'] = float(p[3:])
        elif p.startswith('noise'):
            params['noise'] = float(p[5:])
        elif p.startswith('iface'):
            params['interface'] = float(p[5:])
        elif p.startswith('ang'):
            params['n_angles'] = int(p[3:])
        elif 'layer' in p:
            params['n_layers'] = int(p[0])
    return params


# =========================================================================
# (A) Decision Boundary Map
# =========================================================================

def plot_decision_boundary(
    results: Dict[str, Dict[str, 'AggregatedMetrics']],
    scenarios: list,
    metric: str = 'depth_rmse',
    axis_x: str = 'n_layers',
    axis_y: str = 'noise',
    output_path: Optional[Path] = None,
    fig_size: Tuple[float, float] = (14, 5),
):
    """問題空間でのソルバー選択マップ.

    Parameters
    ----------
    results : {scenario_name: {solver_name: AggregatedMetrics}}
    scenarios : list of ProblemScenario
    metric : 最適化するメトリクス ('depth_rmse', 'w1_mean', 'ismp_rmse', 'sigma_dthk')
    axis_x, axis_y : 問題パラメータの2軸
    """
    metrics_to_plot = ['depth_rmse', 'w1_mean', 'sigma_dthk', 'ismp_rmse']
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=fig_size)
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, met in zip(axes, metrics_to_plot):
        # Collect data points
        x_vals, y_vals, best_solvers = [], [], []

        for scenario in scenarios:
            if scenario.name not in results:
                continue
            solver_results = results[scenario.name]
            if not solver_results:
                continue

            params = _parse_scenario_name(scenario.name)
            if axis_x not in params or axis_y not in params:
                continue

            # Find best solver for this metric
            best_val = np.inf
            best_solver = None
            for sname in SOLVER_NAMES:
                if sname not in solver_results:
                    continue
                val = _extract_metric(solver_results[sname], met)
                if not np.isnan(val) and val < best_val:
                    best_val = val
                    best_solver = sname

            if best_solver is not None:
                x_vals.append(params[axis_x])
                y_vals.append(params[axis_y])
                best_solvers.append(best_solver)

        if not x_vals:
            ax.set_title(f'{METRIC_LABELS.get(met, met)}\n(no data)')
            continue

        # Scatter with solver colors
        for sname in SOLVER_NAMES:
            mask = [s == sname for s in best_solvers]
            if not any(mask):
                continue
            xs = [x_vals[i] for i in range(len(mask)) if mask[i]]
            ys = [y_vals[i] for i in range(len(mask)) if mask[i]]
            ax.scatter(
                xs, ys,
                c=SOLVER_COLORS[sname],
                marker=SOLVER_MARKERS[sname],
                s=120, edgecolors='k', linewidths=0.5,
                label=sname, zorder=5,
            )

        ax.set_xlabel(_axis_label(axis_x))
        ax.set_ylabel(_axis_label(axis_y))
        ax.set_title(f'Best solver by\n{METRIC_LABELS.get(met, met)}')
        ax.legend(fontsize=8, loc='best')

        # Log scale for noise axis
        if axis_y == 'noise':
            ax.set_yscale('symlog', linthresh=0.005)

    fig.suptitle('Decision Boundary: Best Solver per Problem Region', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


def _axis_label(axis_name: str) -> str:
    labels = {
        'n_layers': 'Number of Layers',
        'noise': 'Noise Level',
        'thickness': 'Total Thickness (Å)',
        'interface': 'Interface Width (Å)',
        'n_angles': 'Number of Angles',
    }
    return labels.get(axis_name, axis_name)


# =========================================================================
# (B) ヒートマップ + レーダーチャート
# =========================================================================

def plot_heatmap_radar(
    results: Dict[str, Dict[str, 'AggregatedMetrics']],
    scenarios: list,
    output_path: Optional[Path] = None,
    fig_size: Tuple[float, float] = (18, 12),
):
    """ヒートマップ (全メトリクス) + レーダーチャート (問題タイプ別)."""
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    # --- (B1) Heatmap ---
    ax_heat = fig.add_subplot(gs[0, :])
    _plot_heatmap(ax_heat, results, scenarios)

    # --- (B2) Radar charts by n_layers ---
    for i, n_lay in enumerate([1, 2, 4]):
        if i >= 2:
            break
        ax_radar = fig.add_subplot(gs[1, i], polar=True)
        _plot_radar(ax_radar, results, scenarios, n_layers_filter=n_lay)

    fig.suptitle('Solver Performance Heatmap & Radar', fontsize=14, fontweight='bold', y=1.01)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


def _plot_heatmap(ax, results, scenarios):
    """正規化ヒートマップ: 行=シナリオ, 列=ソルバー×メトリクス."""
    metrics = ['depth_rmse', 'w1_mean', 'sigma_dthk', 'ismp_rmse']
    n_met = len(metrics)

    # Filter scenarios that have results
    valid_scenarios = [s for s in scenarios if s.name in results and results[s.name]]
    if not valid_scenarios:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    # Limit display
    max_rows = 30
    if len(valid_scenarios) > max_rows:
        step = len(valid_scenarios) // max_rows
        valid_scenarios = valid_scenarios[::step][:max_rows]

    n_scenarios = len(valid_scenarios)
    data = np.full((n_scenarios, len(SOLVER_NAMES) * n_met), np.nan)

    for i, scenario in enumerate(valid_scenarios):
        sr = results[scenario.name]
        for j, sname in enumerate(SOLVER_NAMES):
            if sname not in sr:
                continue
            for k, met in enumerate(metrics):
                val = _extract_metric(sr[sname], met)
                data[i, j * n_met + k] = val

    # Normalize per metric column (0=best, 1=worst within each row)
    data_norm = np.full_like(data, np.nan)
    for i in range(n_scenarios):
        for k in range(n_met):
            cols = [j * n_met + k for j in range(len(SOLVER_NAMES))]
            vals = data[i, cols]
            valid = ~np.isnan(vals)
            if valid.sum() > 0:
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                if vmax > vmin:
                    for c, v in zip(cols, vals):
                        if not np.isnan(v):
                            data_norm[i, c] = (v - vmin) / (vmax - vmin)
                else:
                    for c in cols:
                        if not np.isnan(data[i, c]):
                            data_norm[i, c] = 0.0

    # Plot
    im = ax.imshow(data_norm, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)

    # Labels
    col_labels = []
    for sname in SOLVER_NAMES:
        for met in metrics:
            col_labels.append(f"{sname[:4]}\n{METRIC_LABELS.get(met, met)[:8]}")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=6, rotation=45, ha='right')

    row_labels = [s.name[:30] for s in valid_scenarios]
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(row_labels, fontsize=6)

    ax.set_title('Normalized Performance (green=best, red=worst)')
    plt.colorbar(im, ax=ax, shrink=0.6, label='Relative rank (0=best)')


def _plot_radar(ax, results, scenarios, n_layers_filter: int):
    """レーダーチャート: 4メトリクスで3ソルバーを比較."""
    metrics = ['depth_rmse', 'w1_mean', 'sigma_dthk', 'ismp_rmse']
    n_met = len(metrics)

    # Filter by n_layers
    filtered = []
    for s in scenarios:
        params = _parse_scenario_name(s.name)
        if params.get('n_layers') == n_layers_filter and s.name in results:
            filtered.append(s)

    if not filtered:
        ax.set_title(f'{n_layers_filter}-layer\n(no data)')
        return

    # Average metrics across scenarios
    solver_means = {}
    for sname in SOLVER_NAMES:
        vals = {m: [] for m in metrics}
        for s in filtered:
            sr = results.get(s.name, {})
            if sname in sr:
                for m in metrics:
                    v = _extract_metric(sr[sname], m)
                    if not np.isnan(v):
                        vals[m].append(v)
        solver_means[sname] = {m: np.mean(vals[m]) if vals[m] else np.nan for m in metrics}

    # Normalize: 0=best, 1=worst across solvers
    norm_vals = {sname: [] for sname in SOLVER_NAMES}
    for m in metrics:
        all_vals = [solver_means[sname][m] for sname in SOLVER_NAMES]
        valid = [v for v in all_vals if not np.isnan(v)]
        if not valid:
            for sname in SOLVER_NAMES:
                norm_vals[sname].append(0.5)
            continue
        vmin, vmax = min(valid), max(valid)
        for sname in SOLVER_NAMES:
            v = solver_means[sname][m]
            if np.isnan(v) or vmax == vmin:
                norm_vals[sname].append(0.5)
            else:
                # Invert: lower is better → lower value = larger radar area
                norm_vals[sname].append(1.0 - (v - vmin) / (vmax - vmin))

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, n_met, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    for sname in SOLVER_NAMES:
        values = norm_vals[sname] + norm_vals[sname][:1]
        ax.plot(angles, values, 'o-', color=SOLVER_COLORS[sname],
                linewidth=2, label=sname, markersize=4)
        ax.fill(angles, values, color=SOLVER_COLORS[sname], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS.get(m, m)[:10] for m in metrics], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(f'{n_layers_filter}-layer (larger=better)', fontsize=10)
    ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.1))


# =========================================================================
# (C) PCA 散布図
# =========================================================================

def plot_pca_scatter(
    results: Dict[str, Dict[str, 'AggregatedMetrics']],
    scenarios: list,
    output_path: Optional[Path] = None,
    fig_size: Tuple[float, float] = (10, 8),
):
    """4メトリクス空間の PCA 2D projection."""
    metrics = ['depth_rmse', 'w1_mean', 'sigma_dthk', 'ismp_rmse']

    # Collect all data points
    data_points = []  # (metric_vector, solver_name, n_layers)
    for scenario in scenarios:
        if scenario.name not in results:
            continue
        sr = results[scenario.name]
        params = _parse_scenario_name(scenario.name)
        n_lay = params.get('n_layers', 0)

        for sname in SOLVER_NAMES:
            if sname not in sr:
                continue
            vec = [_extract_metric(sr[sname], m) for m in metrics]
            if any(np.isnan(v) for v in vec):
                continue
            data_points.append((vec, sname, n_lay))

    if len(data_points) < 3:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.text(0.5, 0.5, 'Insufficient data for PCA', ha='center', va='center')
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
        return fig

    X = np.array([dp[0] for dp in data_points])
    solvers = [dp[1] for dp in data_points]
    n_layers = [dp[2] for dp in data_points]

    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    # PCA via SVD
    U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)
    PC = X_norm @ Vt[:2].T  # project to 2D

    # Variance explained
    var_explained = S[:2] ** 2 / (S ** 2).sum() * 100

    fig, ax = plt.subplots(figsize=fig_size)

    # Marker shapes by n_layers
    layer_markers = {1: 'o', 2: 's', 4: '^', 0: 'x'}

    for sname in SOLVER_NAMES:
        for n_lay in sorted(set(n_layers)):
            mask = [(s == sname and nl == n_lay) for s, nl in zip(solvers, n_layers)]
            if not any(mask):
                continue
            idx = [i for i, m in enumerate(mask) if m]
            marker = layer_markers.get(n_lay, 'x')
            ax.scatter(
                PC[idx, 0], PC[idx, 1],
                c=SOLVER_COLORS[sname],
                marker=marker,
                s=80, edgecolors='k', linewidths=0.3,
                alpha=0.7,
                label=f'{sname} ({n_lay}L)',
            )

    ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}% variance)')
    ax.set_title('PCA of 4-Metric Space (solver × problem type)')

    # Loading vectors
    loadings = Vt[:2].T  # (4, 2) = metric contributions to PC1, PC2
    scale = max(abs(PC[:, 0].max()), abs(PC[:, 1].max())) * 0.8
    for i, met in enumerate(metrics):
        ax.annotate(
            '', xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
        )
        ax.text(
            loadings[i, 0] * scale * 1.1,
            loadings[i, 1] * scale * 1.1,
            METRIC_LABELS.get(met, met),
            fontsize=8, color='gray', ha='center',
        )

    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.axhline(0, color='lightgray', lw=0.5)
    ax.axvline(0, color='lightgray', lw=0.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


# =========================================================================
# Convenience: plot all three
# =========================================================================

def plot_all(
    results: Dict[str, Dict[str, 'AggregatedMetrics']],
    scenarios: list,
    output_dir: Path,
):
    """3種の可視化を全て生成."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Decision Boundary Map...")
    plot_decision_boundary(
        results, scenarios,
        output_path=output_dir / 'solver_landscape_decision_boundary.png',
    )

    print("Generating Heatmap + Radar...")
    plot_heatmap_radar(
        results, scenarios,
        output_path=output_dir / 'solver_landscape_heatmap_radar.png',
    )

    print("Generating PCA Scatter...")
    plot_pca_scatter(
        results, scenarios,
        output_path=output_dir / 'solver_landscape_pca.png',
    )

    print("All visualizations generated.")
