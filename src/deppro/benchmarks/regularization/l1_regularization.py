"""
L1正則化テスト: スパース化の効果を検証
=========================================

GenericLayerSolverでL1ノルム正則化を使用し、
alphaパラメータを変えながら深さ分布の変化を観察。

見るもの:
- alpha を大きくするとスパース化が進むか

Date: 2026-01-24
"""

import numpy as np
from scipy.special import erf
from typing import List, Dict
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


def run_generic_layer_solver_with_method(
    I_obs, tm, elements,
    method: str = 'LLS',
    alpha: float = 0.0,
    thicknesses_init: List[float] = None
):
    """Run GenericLayerSolver with specified method and alpha."""
    from deppro.inverse_solver import GenericLayerSolver, GenericLayerConfig, LayerDefinition, SolverConfig

    if thicknesses_init is None:
        thicknesses_init = [10.0, 20.0, 40.0, 35.0]

    layers = [
        LayerDefinition(name='C', element_indices=[0], stoichiometry={0: 1.0}),
        LayerDefinition(name='Al2O3', element_indices=[1, 4], stoichiometry={1: 2.0, 4: 3.0}),
        LayerDefinition(name='TiO2', element_indices=[2, 4], stoichiometry={2: 1.0, 4: 2.0}),
        LayerDefinition(name='SiO2', element_indices=[3, 4], stoichiometry={3: 1.0, 4: 2.0}),
    ]
    substrate = LayerDefinition(name='Si', element_indices=[5], stoichiometry={5: 1.0})

    layer_config = GenericLayerConfig(
        layers=layers,
        substrate=substrate,
        initial_thicknesses=thicknesses_init,
        initial_widths=[2.0, 3.0, 4.0, 5.0],
    )

    solver_config = SolverConfig(method=method, alpha=alpha)
    solver = GenericLayerSolver(tm, solver_config, layer_config, elements=elements)

    result = solver.solve(I_obs, verbose=False)
    return result.D_opt, result.layer_thicknesses


def run_semifree_solver_with_method(
    I_obs, tm, elements,
    method: str = 'LLS',
    alpha: float = 0.0,
    use_zero_prior: bool = False,
):
    """Run SemiFreeSolver with specified method and alpha.

    Args:
        use_zero_prior: If True, use D_prior=0 for pure L1 sparsity (sum|D|)
    """
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    semi_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
    )

    solver_config = SolverConfig(method=method, alpha=alpha)

    # Create D_prior
    if use_zero_prior:
        n_depth = len(tm.depth)
        n_elements = tm.n_elements
        D_prior = np.zeros((n_depth, n_elements)) + 1e-6  # Small value to avoid log(0)
    else:
        D_prior = None  # Use default (uniform)

    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements, D_prior=D_prior)

    result = solver.solve(I_obs, verbose=False)
    return result.D_opt, result.thickness


def run_l1_comparison(n_trials: int = 5, noise_level: float = 0.02):
    """Run L1 regularization comparison with different alpha values using SemiFreeSolver.

    Tests both:
    - L1 with uniform prior: sum|D - D_prior| where D_prior = 1/n_elements
    - L1 with zero prior: sum|D| (pure sparsity)
    """

    true_thicknesses = [5.0, 15.0, 35.0, 30.0]
    interface_widths = [2.0, 3.0, 4.0, 5.0]
    element_labels = ['C', 'Al', 'Ti', 'Si_ox', 'O', 'Si_bulk']

    # Alpha values to test - focus on smaller range for L1 with zero prior
    alpha_values = [0.001, 0.01, 0.1, 1.0]

    results = {}

    # LLS baseline
    results['LLS'] = {'w1': [], 'D_opt': []}

    # L1 with uniform prior (default)
    for alpha in alpha_values:
        results[f'L1_uniform_a{alpha}'] = {'w1': [], 'D_opt': []}

    # L1 with zero prior (sparsity)
    for alpha in alpha_values:
        results[f'L1_zero_a{alpha}'] = {'w1': [], 'D_opt': []}

    D_true_stored = None

    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}...")

        I_obs, tm, elements, D_true, depth = create_test_data(
            thicknesses=true_thicknesses,
            interface_widths=interface_widths,
            noise_level=noise_level,
            seed=42 + trial,
        )

        if D_true_stored is None:
            D_true_stored = D_true

        # Run LLS (baseline) with SemiFreeSolver
        D_lls, _ = run_semifree_solver_with_method(
            I_obs, tm, elements, method='LLS', alpha=0.0
        )
        w1_lls = [compute_wasserstein(depth, D_true[:, i], D_lls[:, i]) for i in range(6)]
        results['LLS']['w1'].append(w1_lls)
        results['LLS']['D_opt'].append(D_lls.copy())

        # Run L1 with uniform prior
        for alpha in alpha_values:
            key = f'L1_uniform_a{alpha}'
            try:
                D_l1, _ = run_semifree_solver_with_method(
                    I_obs, tm, elements, method='L1', alpha=alpha, use_zero_prior=False
                )
                w1_l1 = [compute_wasserstein(depth, D_true[:, i], D_l1[:, i]) for i in range(6)]
                results[key]['w1'].append(w1_l1)
                results[key]['D_opt'].append(D_l1.copy())
            except Exception as e:
                print(f"  Warning: {key} failed: {e}")
                results[key]['w1'].append([np.nan] * 6)
                results[key]['D_opt'].append(np.full_like(D_true, np.nan))

        # Run L1 with zero prior (sparsity)
        for alpha in alpha_values:
            key = f'L1_zero_a{alpha}'
            try:
                D_l1, _ = run_semifree_solver_with_method(
                    I_obs, tm, elements, method='L1', alpha=alpha, use_zero_prior=True
                )
                w1_l1 = [compute_wasserstein(depth, D_true[:, i], D_l1[:, i]) for i in range(6)]
                results[key]['w1'].append(w1_l1)
                results[key]['D_opt'].append(D_l1.copy())
            except Exception as e:
                print(f"  Warning: {key} failed: {e}")
                results[key]['w1'].append([np.nan] * 6)
                results[key]['D_opt'].append(np.full_like(D_true, np.nan))

    return results, element_labels, depth, D_true_stored, alpha_values


def visualize_l1_depth_profiles(
    results: Dict,
    element_labels: List[str],
    depth: np.ndarray,
    D_true: np.ndarray,
    alpha_values: List[float],
    output_path: str
):
    """
    Plot depth profiles comparing LLS vs L1 (zero prior) with different alpha values.
    Focus on sparsity effect.
    """
    # 3x3 grid: LLS + 4 alpha values for zero prior + 4 for uniform prior
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()

    element_colors = ['black', 'blue', 'purple', 'orange', 'red', 'green']

    def plot_panel(ax, key, title):
        # Ground truth
        for el_idx, (label, color) in enumerate(zip(element_labels, element_colors)):
            ax.plot(depth, D_true[:, el_idx] * 100, color=color, linewidth=2.5,
                    linestyle='-', alpha=0.9)

        # Reconstructed (all trials)
        if key in results:
            D_opt_list = results[key]['D_opt']
            for trial_idx, D_opt in enumerate(D_opt_list):
                if not np.any(np.isnan(D_opt)):
                    for el_idx, color in enumerate(element_colors):
                        ax.plot(depth, D_opt[:, el_idx] * 100, color=color,
                                linewidth=0.8, linestyle='--', alpha=0.5)

        ax.set_xlabel('Depth (Å)', fontsize=10)
        ax.set_ylabel('Concentration (%)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.axvspan(5, 20, alpha=0.15, color='blue')  # Al2O3 layer region

    # Plot LLS
    plot_panel(axes[0], 'LLS', 'LLS (no regularization)')

    # Plot L1 with zero prior (sparsity)
    for idx, alpha in enumerate(alpha_values):
        key = f'L1_zero_a{alpha}'
        plot_panel(axes[idx + 1], key, f'L1 zero prior (α={alpha})')

    # Plot L1 with uniform prior
    for idx, alpha in enumerate(alpha_values):
        key = f'L1_uniform_a{alpha}'
        plot_panel(axes[idx + 5], key, f'L1 uniform prior (α={alpha})')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=c, linewidth=2.5, label=f'{l} (true)')
        for l, c in zip(element_labels, element_colors)
    ]
    legend_elements.append(
        plt.Line2D([0], [0], color='gray', linewidth=0.8, linestyle='--', alpha=0.5, label='Trials (1-5)')
    )
    legend_elements.append(
        plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.15, label='Al₂O₃ layer (5-20 Å)')
    )

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=8,
        fontsize=9
    )

    plt.suptitle(
        'L1 Regularization: Zero Prior (sparsity) vs Uniform Prior\n'
        'SemiFreeSolver, 4-layer model: C / Al₂O₃ / TiO₂ / SiO₂ / Si, 2% noise',
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"L1 depth profile figure saved: {output_path}")

    return fig


def visualize_al_profile_comparison(
    results: Dict,
    depth: np.ndarray,
    D_true: np.ndarray,
    alpha_values: List[float],
    output_path: str
):
    """
    Focus plot on Al profile (element 1) to show sparsity effect clearly.
    Compare L1 with zero prior vs uniform prior.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    al_idx = 1  # Al element index

    # Color map for alpha values
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(alpha_values)))

    for ax_idx, (prior_type, title) in enumerate([
        ('zero', 'L1 with Zero Prior (sparsity: sum|D|)'),
        ('uniform', 'L1 with Uniform Prior (sum|D - D_prior|)')
    ]):
        ax = axes[ax_idx]

        # Ground truth
        ax.plot(depth, D_true[:, al_idx] * 100, 'k-', linewidth=3, label='True', zorder=10)

        # LLS baseline
        D_lls_mean = np.mean([D[:, al_idx] for D in results['LLS']['D_opt']], axis=0)
        D_lls_std = np.std([D[:, al_idx] for D in results['LLS']['D_opt']], axis=0)
        ax.plot(depth, D_lls_mean * 100, color='gray', linewidth=2,
                linestyle='--', label='LLS')
        ax.fill_between(depth, (D_lls_mean - D_lls_std) * 100, (D_lls_mean + D_lls_std) * 100,
                        color='gray', alpha=0.2)

        # L1 with different alpha
        for idx, alpha in enumerate(alpha_values):
            key = f'L1_{prior_type}_a{alpha}'
            if key not in results:
                continue

            D_opt_list = [D[:, al_idx] for D in results[key]['D_opt'] if not np.any(np.isnan(D))]
            if len(D_opt_list) > 0:
                D_mean = np.mean(D_opt_list, axis=0)
                D_std = np.std(D_opt_list, axis=0)
                ax.plot(depth, D_mean * 100, color=colors[idx], linewidth=2,
                        linestyle='-', label=f'α={alpha}')
                ax.fill_between(depth, (D_mean - D_std) * 100, (D_mean + D_std) * 100,
                                color=colors[idx], alpha=0.15)

        # Highlight Al layer region
        ax.axvspan(5, 20, alpha=0.1, color='blue')
        ax.axvline(x=5, color='blue', linestyle=':', alpha=0.5)
        ax.axvline(x=20, color='blue', linestyle=':', alpha=0.5)

        ax.set_xlabel('Depth (Å)', fontsize=12)
        ax.set_ylabel('Al Concentration (%)', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 20)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Al Profile: Effect of L1 Prior Choice\n'
        'Blue region = true Al₂O₃ layer (5-20 Å)',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Al profile comparison figure saved: {output_path}")

    return fig


def print_l1_summary(results: Dict, element_labels: List[str], alpha_values: List[float]):
    """Print summary of L1 regularization results."""
    print("\n" + "=" * 100)
    print("L1 Regularization Summary: Mean Wasserstein Distance (Å)")
    print("=" * 100)

    # Header
    header = f"{'Method':<20}"
    for label in element_labels:
        header += f"{label:<10}"
    header += f"{'MEAN':<10}"
    print(header)
    print("-" * 100)

    # LLS baseline
    row = f"{'LLS':<20}"
    w1_all = np.array(results['LLS']['w1'])
    for i in range(6):
        w1_mean = np.mean(w1_all[:, i])
        row += f"{w1_mean:.3f}     "
    row += f"{np.mean(w1_all):.3f}"
    print(row)

    print("-" * 100)
    print("L1 with ZERO prior (sparsity: sum|D|):")

    # L1 with zero prior
    for alpha in alpha_values:
        key = f'L1_zero_a{alpha}'
        row = f"{f'  α={alpha}':<20}"
        if key in results:
            w1_all = np.array(results[key]['w1'])
            if not np.all(np.isnan(w1_all)):
                for i in range(6):
                    w1_mean = np.nanmean(w1_all[:, i])
                    row += f"{w1_mean:.3f}     "
                row += f"{np.nanmean(w1_all):.3f}"
            else:
                row += "FAILED"
        print(row)

    print("-" * 100)
    print("L1 with UNIFORM prior (sum|D - D_prior|):")

    # L1 with uniform prior
    for alpha in alpha_values:
        key = f'L1_uniform_a{alpha}'
        row = f"{f'  α={alpha}':<20}"
        if key in results:
            w1_all = np.array(results[key]['w1'])
            if not np.all(np.isnan(w1_all)):
                for i in range(6):
                    w1_mean = np.nanmean(w1_all[:, i])
                    row += f"{w1_mean:.3f}     "
                row += f"{np.nanmean(w1_all):.3f}"
            else:
                row += "FAILED"
        print(row)

    print("=" * 100)


def main():
    """Main function."""
    print("=" * 80)
    print("L1 Regularization Test")
    print("GenericLayerSolver with different alpha values")
    print("=" * 80)

    # Run comparison
    results, element_labels, depth, D_true, alpha_values = run_l1_comparison(
        n_trials=5, noise_level=0.02
    )

    # Print summary
    print_l1_summary(results, element_labels, alpha_values)

    # Visualize all depth profiles
    profile_output_path = '/tmp/l1_regularization_depth_profiles.png'
    visualize_l1_depth_profiles(
        results, element_labels, depth, D_true, alpha_values, profile_output_path
    )

    # Visualize Al profile comparison
    al_output_path = '/tmp/l1_regularization_al_profile.png'
    visualize_al_profile_comparison(
        results, depth, D_true, alpha_values, al_output_path
    )

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
