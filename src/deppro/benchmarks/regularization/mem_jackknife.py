"""
MEM解析 with ジャックナイフ法
==============================

SemiFreeSolver L1正則化の結果をPriorとして使用するMEM解析。
ジャックナイフ法: i番目の解析時に、i番目以外のトライアルの平均をPriorに使用。

Date: 2026-01-24
"""

import numpy as np
from scipy.special import erf
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_4layer_profile(
    depth: np.ndarray,
    thicknesses: List[float] = [5.0, 15.0, 35.0, 30.0],
    interface_widths: List[float] = [2.0, 3.0, 4.0, 5.0],
) -> np.ndarray:
    """Create 4-layer profile: C / Al2O3 / TiO2 / SiO2 / Si"""
    sigma = [w / (2 * np.sqrt(np.log(2))) for w in interface_widths]
    z = [0] + list(np.cumsum(thicknesses))

    D = np.zeros((len(depth), 6))

    D[:, 0] = 0.5 * (1 - erf((depth - z[1]) / sigma[0]))

    al2o3 = 0.5 * (1 + erf((depth - z[1]) / sigma[0])) * 0.5 * (1 - erf((depth - z[2]) / sigma[1]))
    D[:, 1] = (2/5) * al2o3
    D[:, 4] = (3/5) * al2o3

    tio2 = 0.5 * (1 + erf((depth - z[2]) / sigma[1])) * 0.5 * (1 - erf((depth - z[3]) / sigma[2]))
    D[:, 2] = (1/3) * tio2
    D[:, 4] += (2/3) * tio2

    sio2 = 0.5 * (1 + erf((depth - z[3]) / sigma[2])) * 0.5 * (1 - erf((depth - z[4]) / sigma[3]))
    D[:, 3] = (1/3) * sio2
    D[:, 4] += (2/3) * sio2

    D[:, 5] = 0.5 * (1 + erf((depth - z[4]) / sigma[3]))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def create_test_data(
    thicknesses: List[float] = [5.0, 15.0, 35.0, 30.0],
    interface_widths: List[float] = [2.0, 3.0, 4.0, 5.0],
    noise_level: float = 0.02,
    seed: int = 42,
):
    """Create test data."""
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    np.random.seed(seed)
    depth = np.arange(0, 100, 4.0)
    angles = [0, 10, 20, 30, 40, 50, 60]

    elements = [
        ElementConfig('C', '1s', 'C', oxidation_state=0),
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_4layer_profile(depth, thicknesses, interface_widths)

    tm = TransferMatrix(depth=depth, step_size=4.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    if noise_level > 0:
        noise = noise_level * np.random.randn(*I_true.shape)
        I_obs = I_true + noise * I_true
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
    else:
        I_obs = I_true.copy()

    return I_obs, tm, elements, D_true, depth


def compute_wasserstein(depth: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Wasserstein-1 distance between two 1D distributions."""
    p1_norm = p1 / (np.sum(p1) + 1e-10)
    p2_norm = p2 / (np.sum(p2) + 1e-10)
    cdf1 = np.cumsum(p1_norm)
    cdf2 = np.cumsum(p2_norm)
    step = depth[1] - depth[0] if len(depth) > 1 else 1.0
    w1 = np.sum(np.abs(cdf1 - cdf2)) * step
    return w1


def run_semifree_solver(I_obs, tm, elements, method: str = 'LLS', alpha: float = 0.0,
                        D_prior: np.ndarray = None):
    """Run SemiFreeSolver with specified method, alpha, and optional D_prior."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    semi_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
    )

    solver_config = SolverConfig(method=method, alpha=alpha)
    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements, D_prior=D_prior)

    result = solver.solve(I_obs, verbose=False)
    return result.D_opt, result.thickness


def run_mem_jackknife_test(n_trials: int = 5, noise_level: float = 0.02):
    """
    Run MEM analysis with jackknife prior estimation.

    Process:
    1. Generate n_trials datasets with different noise
    2. Run L1 regularization (alpha=1.0) on all trials
    3. For each trial i, compute jackknife prior = mean of L1 results excluding trial i
    4. Run MEM with jackknife prior
    5. Compare results
    """

    true_thicknesses = [5.0, 15.0, 35.0, 30.0]
    interface_widths = [2.0, 3.0, 4.0, 5.0]
    element_labels = ['C', 'Al', 'Ti', 'Si_ox', 'O', 'Si_bulk']

    print("=" * 80)
    print("Phase 1: Generate test data and run L1 regularization")
    print("=" * 80)

    # Store all test data and L1 results
    test_data = []
    l1_results = []

    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}: Generating data and running L1...")

        I_obs, tm, elements, D_true, depth = create_test_data(
            thicknesses=true_thicknesses,
            interface_widths=interface_widths,
            noise_level=noise_level,
            seed=42 + trial,
        )

        test_data.append({
            'I_obs': I_obs,
            'tm': tm,
            'elements': elements,
            'D_true': D_true,
            'depth': depth,
        })

        # Run L1 regularization (alpha=1.0, D_prior=0 is now default)
        D_l1, _ = run_semifree_solver(I_obs, tm, elements, method='L1', alpha=1.0)
        l1_results.append(D_l1)

    D_true = test_data[0]['D_true']  # Same for all trials
    depth = test_data[0]['depth']

    print("\n" + "=" * 80)
    print("Phase 2: Run MEM with jackknife priors")
    print("=" * 80)

    # Results storage
    results = {
        'L1': {'w1': [], 'D_opt': []},
        'MEM_jackknife_a0.001': {'w1': [], 'D_opt': []},
        'MEM_jackknife_a0.01': {'w1': [], 'D_opt': []},
        'MEM_jackknife_a0.1': {'w1': [], 'D_opt': []},
    }

    # Store L1 results
    for trial, D_l1 in enumerate(l1_results):
        w1 = [compute_wasserstein(depth, D_true[:, i], D_l1[:, i]) for i in range(6)]
        results['L1']['w1'].append(w1)
        results['L1']['D_opt'].append(D_l1.copy())

    # Run MEM with jackknife priors at different alpha values
    alpha_values = [0.001, 0.01, 0.1]

    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}: Running MEM with jackknife prior...")

        # Compute jackknife prior: mean of L1 results excluding current trial
        other_trials = [i for i in range(n_trials) if i != trial]
        D_prior_jackknife = np.mean([l1_results[i] for i in other_trials], axis=0)

        # Ensure prior is positive
        D_prior_jackknife = np.maximum(D_prior_jackknife, 1e-6)

        # Run MEM with jackknife prior
        I_obs = test_data[trial]['I_obs']
        tm = test_data[trial]['tm']
        elements = test_data[trial]['elements']

        for alpha in alpha_values:
            key = f'MEM_jackknife_a{alpha}'
            try:
                D_mem_jk, _ = run_semifree_solver(
                    I_obs, tm, elements,
                    method='MEM', alpha=alpha,
                    D_prior=D_prior_jackknife
                )
                w1_jk = [compute_wasserstein(depth, D_true[:, i], D_mem_jk[:, i]) for i in range(6)]
                results[key]['w1'].append(w1_jk)
                results[key]['D_opt'].append(D_mem_jk.copy())
            except Exception as e:
                print(f"    Warning: MEM jackknife (α={alpha}) failed: {e}")
                results[key]['w1'].append([np.nan] * 6)
                results[key]['D_opt'].append(np.full_like(D_true, np.nan))

    return results, element_labels, depth, D_true, l1_results


def print_summary(results: Dict, element_labels: List[str]):
    """Print summary of results."""
    print("\n" + "=" * 100)
    print("MEM Jackknife Summary: Mean Wasserstein Distance (Å)")
    print("=" * 100)

    methods = ['L1', 'MEM_jackknife_a0.001', 'MEM_jackknife_a0.01', 'MEM_jackknife_a0.1']
    method_names = {
        'L1': 'L1 (α=1.0, D_prior=0)',
        'MEM_jackknife_a0.001': 'MEM jackknife (α=0.001)',
        'MEM_jackknife_a0.01': 'MEM jackknife (α=0.01)',
        'MEM_jackknife_a0.1': 'MEM jackknife (α=0.1)',
    }

    # Header
    header = f"{'Method':<25}"
    for label in element_labels:
        header += f"{label:<10}"
    header += f"{'MEAN':<10}"
    print(header)
    print("-" * 100)

    for method in methods:
        if method not in results:
            continue
        row = f"{method_names.get(method, method):<25}"
        w1_all = np.array(results[method]['w1'])
        if not np.all(np.isnan(w1_all)):
            for i in range(6):
                w1_mean = np.nanmean(w1_all[:, i])
                row += f"{w1_mean:.3f}     "
            row += f"{np.nanmean(w1_all):.3f}"
        else:
            row += "FAILED"
        print(row)

    print("=" * 100)

    # Improvement comparison
    print("\nImprovement over L1:")
    l1_mean = np.nanmean(results['L1']['w1'])
    for method in methods[1:]:
        if method not in results:
            continue
        mem_mean = np.nanmean(results[method]['w1'])
        improvement = (l1_mean - mem_mean) / l1_mean * 100
        print(f"  {method_names.get(method, method)}: {improvement:+.1f}%")


def visualize_comparison(
    results: Dict,
    element_labels: List[str],
    depth: np.ndarray,
    D_true: np.ndarray,
    output_path: str
):
    """Visualize depth profiles: L1 vs MEM_jackknife at different alphas."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharey=True)
    axes = axes.flatten()

    methods = ['L1', 'MEM_jackknife_a0.001', 'MEM_jackknife_a0.01', 'MEM_jackknife_a0.1']
    titles = [
        'L1 (α=1.0, D_prior=0)',
        'MEM jackknife (α=0.001)',
        'MEM jackknife (α=0.01)',
        'MEM jackknife (α=0.1)',
    ]

    element_colors = ['black', 'blue', 'purple', 'orange', 'red', 'green']

    for ax_idx, (method, title) in enumerate(zip(methods, titles)):
        ax = axes[ax_idx]

        # Plot ground truth
        for el_idx, (label, color) in enumerate(zip(element_labels, element_colors)):
            ax.plot(depth, D_true[:, el_idx] * 100, color=color, linewidth=2.5,
                    linestyle='-', alpha=0.9,
                    label=f'{label} (true)' if ax_idx == 0 else None)

        # Plot all trials
        if method in results:
            D_opt_list = results[method]['D_opt']
            for trial_idx, D_opt in enumerate(D_opt_list):
                if not np.any(np.isnan(D_opt)):
                    for el_idx, color in enumerate(element_colors):
                        ax.plot(depth, D_opt[:, el_idx] * 100, color=color,
                                linewidth=0.8, linestyle='--', alpha=0.5)

        ax.set_xlabel('Depth (Å)', fontsize=12)
        if ax_idx % 2 == 0:
            ax.set_ylabel('Concentration (%)', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=c, linewidth=2.5, label=f'{l} (true)')
        for l, c in zip(element_labels, element_colors)
    ]
    legend_elements.append(
        plt.Line2D([0], [0], color='gray', linewidth=0.8, linestyle='--', alpha=0.5, label='Trials')
    )

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=7,
        fontsize=10
    )

    plt.suptitle(
        'MEM with Jackknife Prior (α tuning) vs L1 Regularization\n'
        'SemiFreeSolver, 4-layer model: C / Al₂O₃ / TiO₂ / SiO₂ / Si, 2% noise',
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {output_path}")

    return fig


def visualize_w1_comparison(
    results: Dict,
    element_labels: List[str],
    output_path: str
):
    """Visualize Wasserstein distance comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))

    methods = ['L1', 'MEM_jackknife_a0.001', 'MEM_jackknife_a0.01', 'MEM_jackknife_a0.1']
    method_names = ['L1 (α=1.0)', 'MEM jk (α=0.001)', 'MEM jk (α=0.01)', 'MEM jk (α=0.1)']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    x = np.arange(len(element_labels))
    width = 0.2

    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        if method not in results:
            continue
        w1_all = np.array(results[method]['w1'])
        w1_mean = np.nanmean(w1_all, axis=0)
        w1_std = np.nanstd(w1_all, axis=0)

        ax.bar(x + i * width, w1_mean, width, yerr=w1_std,
               label=name, color=color, capsize=3)

    ax.set_xlabel('Element', fontsize=12)
    ax.set_ylabel('Wasserstein Distance (Å)', fontsize=12)
    ax.set_title('Wasserstein Distance Comparison: L1 vs MEM with Jackknife Prior', fontsize=14)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(element_labels)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"W1 comparison figure saved: {output_path}")

    return fig


def main():
    """Main function."""
    print("=" * 80)
    print("MEM Analysis with Jackknife Prior")
    print("SemiFreeSolver: L1 → MEM with leave-one-out prior")
    print("=" * 80)

    # Run test
    results, element_labels, depth, D_true, l1_results = run_mem_jackknife_test(
        n_trials=5, noise_level=0.02
    )

    # Print summary
    print_summary(results, element_labels)

    # Visualize
    visualize_comparison(
        results, element_labels, depth, D_true,
        '/tmp/mem_jackknife_depth_profiles.png'
    )

    visualize_w1_comparison(
        results, element_labels,
        '/tmp/mem_jackknife_w1_comparison.png'
    )

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
