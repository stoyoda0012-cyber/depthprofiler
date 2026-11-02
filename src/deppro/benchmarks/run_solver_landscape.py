#!/usr/bin/env python3
"""
Solver Landscape CLI
====================

3ソルバー (Free / SemiFree / GenericLayer) をグリッド問題で比較し可視化する。

Usage:
    # 小グリッド (18シナリオ × 3ソルバー × 5 trial = 270 runs。所要時間は機種による)
    python -m deppro.benchmarks.run_solver_landscape --small

    # フルグリッド (162シナリオ × 3ソルバー × 10 trial = 4860 runs。所要時間は機種による)
    python -m deppro.benchmarks.run_solver_landscape --full

    # 結果から可視化のみ再生成
    python -m deppro.benchmarks.run_solver_landscape --plot-only --results results.npz

    # 特定メソッド
    python -m deppro.benchmarks.run_solver_landscape --small --method L1 --alpha 0.001
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Unified Solver Landscape Benchmark',
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--small', action='store_true', help='Small grid (18 scenarios)')
    group.add_argument('--full', action='store_true', help='Full grid (162 scenarios)')
    group.add_argument('--plot-only', action='store_true', help='Re-plot from saved results')

    parser.add_argument('--method', default='LLS', choices=['LLS', 'L1', 'L2', 'MEM'])
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--trials', type=int, default=None, help='Override n_trials (default: 5 small, 10 full)')
    parser.add_argument('--results', type=str, help='Path to saved results .npz (for --plot-only)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    # Output directory
    from deppro.paths import outputs_dir
    output_dir = Path(args.output) if args.output else outputs_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        if not args.results:
            print("ERROR: --results required with --plot-only")
            sys.exit(1)
        _plot_from_saved(Path(args.results), output_dir)
        return

    # Import here to avoid heavy deps on --help
    from deppro.benchmarks.unified_benchmark import (
        make_small_grid, make_standard_grid, run_benchmark,
    )
    from deppro.benchmarks.solver_landscape import plot_all

    if args.small:
        scenarios = make_small_grid()
        n_trials = args.trials or 5
    else:
        scenarios = make_standard_grid()
        n_trials = args.trials or 10

    print(f"=" * 60)
    print(f"Solver Landscape Benchmark")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Solvers: Free, SemiFree, GenericLayer")
    print(f"  Trials: {n_trials}")
    print(f"  Method: {args.method}, alpha={args.alpha}")
    print(f"  Total runs: {len(scenarios) * 3 * n_trials}")
    print(f"  Output: {output_dir}")
    print(f"=" * 60)

    t0 = time.time()
    results = run_benchmark(
        scenarios,
        n_trials=n_trials,
        method=args.method,
        alpha=args.alpha,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nBenchmark completed in {elapsed:.1f}s")

    # Save results
    results_path = output_dir / 'solver_landscape_results.npz'
    _save_results(results, scenarios, results_path)
    print(f"Results saved: {results_path}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_all(results, scenarios, output_dir)

    # Print summary table
    _print_summary(results, scenarios)


def _save_results(results, scenarios, path):
    """結果をnpzに保存."""
    save_dict = {
        'scenario_names': np.array([s.name for s in scenarios]),
    }

    for scenario_name, solver_results in results.items():
        for solver_name, agg in solver_results.items():
            prefix = f"{scenario_name}__{solver_name}"
            save_dict[f"{prefix}__ismp_rmse_mean"] = agg.ismp_rmse_mean
            save_dict[f"{prefix}__depth_rmse_mean"] = agg.depth_rmse_mean
            save_dict[f"{prefix}__w1_mean_mean"] = agg.w1_mean_mean
            save_dict[f"{prefix}__sigma_dthk"] = agg.sigma_dthk
            save_dict[f"{prefix}__thk_bias"] = agg.thk_bias
            save_dict[f"{prefix}__success_rate"] = agg.success_rate
            save_dict[f"{prefix}__elapsed_mean"] = agg.elapsed_mean
            save_dict[f"{prefix}__depth_rmse_per_elem"] = agg.depth_rmse_per_elem
            save_dict[f"{prefix}__w1_per_elem"] = agg.w1_per_elem

    np.savez_compressed(path, **save_dict)


def _plot_from_saved(results_path, output_dir):
    """保存済み結果から可視化を再生成."""
    from deppro.benchmarks.unified_benchmark import (
        AggregatedMetrics, ProblemScenario, make_small_grid, make_standard_grid,
    )
    from deppro.benchmarks.solver_landscape import plot_all

    data = np.load(results_path, allow_pickle=True)
    scenario_names = list(data['scenario_names'])

    # Detect grid size
    if len(scenario_names) <= 20:
        scenarios = make_small_grid()
    else:
        scenarios = make_standard_grid()

    # Reconstruct results
    results = {}
    for scenario_name in scenario_names:
        solver_results = {}
        for solver_name in ['Free', 'SemiFree', 'GenericLayer']:
            prefix = f"{scenario_name}__{solver_name}"
            key = f"{prefix}__ismp_rmse_mean"
            if key not in data:
                continue
            solver_results[solver_name] = AggregatedMetrics(
                solver_name=solver_name,
                n_trials=0,
                ismp_rmse_mean=float(data[f"{prefix}__ismp_rmse_mean"]),
                ismp_rmse_std=0.0,
                depth_rmse_mean=float(data[f"{prefix}__depth_rmse_mean"]),
                depth_rmse_std=0.0,
                w1_mean_mean=float(data[f"{prefix}__w1_mean_mean"]),
                w1_mean_std=0.0,
                sigma_dthk=float(data[f"{prefix}__sigma_dthk"]),
                thk_bias=float(data[f"{prefix}__thk_bias"]),
                elapsed_mean=float(data[f"{prefix}__elapsed_mean"]),
                success_rate=float(data[f"{prefix}__success_rate"]),
                n_dof=0,
                depth_rmse_per_elem=np.array(data[f"{prefix}__depth_rmse_per_elem"]),
                w1_per_elem=np.array(data[f"{prefix}__w1_per_elem"]),
            )
        results[scenario_name] = solver_results

    print(f"Loaded {len(results)} scenarios from {results_path}")
    plot_all(results, scenarios, output_dir)


def _print_summary(results, scenarios):
    """サマリーテーブル表示."""
    print(f"\n{'='*80}")
    print("SUMMARY: Best Solver per Problem Type")
    print(f"{'='*80}")

    # Group by n_layers
    from collections import defaultdict
    groups = defaultdict(list)
    for s in scenarios:
        params = _parse_scenario_name(s.name)
        n_lay = params.get('n_layers', 0)
        groups[n_lay].append(s)

    for n_lay in sorted(groups):
        print(f"\n--- {n_lay}-layer problems ---")
        wins = {'Free': 0, 'SemiFree': 0, 'GenericLayer': 0}
        total = 0

        for s in groups[n_lay]:
            if s.name not in results or not results[s.name]:
                continue
            sr = results[s.name]
            # Best by depth_rmse
            best_solver = None
            best_val = np.inf
            for sname in ['Free', 'SemiFree', 'GenericLayer']:
                if sname in sr:
                    val = sr[sname].depth_rmse_mean
                    if not np.isnan(val) and val < best_val:
                        best_val = val
                        best_solver = sname
            if best_solver:
                wins[best_solver] += 1
                total += 1

        if total > 0:
            for sname in ['Free', 'SemiFree', 'GenericLayer']:
                pct = wins[sname] / total * 100
                bar = '#' * int(pct / 5)
                print(f"  {sname:15s}: {wins[sname]:3d}/{total} ({pct:5.1f}%) {bar}")


def _parse_scenario_name(name):
    """Reuse from solver_landscape."""
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


if __name__ == '__main__':
    main()
