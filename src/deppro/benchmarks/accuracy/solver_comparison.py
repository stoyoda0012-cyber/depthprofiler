"""
3ソルバー比較テスト: SemiFreeSolver vs Two-Stage vs GenericLayerSolver
=====================================================================

各元素のWasserstein距離を2Dマップで可視化。
X軸: 元素の真の重心位置 (Å)
Y軸: Wasserstein距離 (Å, ログスケール)

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
    depth = np.arange(0, 100, 4.0)  # ~100 parameters
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
    # Normalize
    p1_norm = p1 / (np.sum(p1) + 1e-10)
    p2_norm = p2 / (np.sum(p2) + 1e-10)

    # CDF
    cdf1 = np.cumsum(p1_norm)
    cdf2 = np.cumsum(p2_norm)

    # W1 = integral of |CDF1 - CDF2|
    step = depth[1] - depth[0] if len(depth) > 1 else 1.0
    w1 = np.sum(np.abs(cdf1 - cdf2)) * step
    return w1


def compute_centroid(depth: np.ndarray, profile: np.ndarray) -> float:
    """Compute centroid of a distribution."""
    total = np.sum(profile)
    if total < 1e-10:
        return 0.0
    return np.sum(depth * profile) / total


def run_semifree_solver(I_obs, tm, elements, method: str = 'LLS', alpha: float = 0.0):
    """Run SemiFreeSolver."""
    from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

    semi_config = SemiFreeSolverConfig(
        thickness_init=70.0,
        abruptness_init=5.0,
        free_indices=[0, 1, 2, 3],
        linked_indices=[4],
        substrate_indices=[5],
    )

    solver_config = SolverConfig(method=method, alpha=alpha)
    solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)

    result = solver.solve(I_obs, verbose=False)
    return result.D_opt, result.thickness


def run_generic_layer_solver(I_obs, tm, elements, thicknesses_init: List[float] = None,
                              method: str = 'LLS', alpha: float = 0.0):
    """Run GenericLayerSolver."""
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


def run_two_stage_solver(I_obs, tm, elements, method: str = 'LLS', alpha: float = 0.0):
    """Run Two-Stage: SemiFreeSolver → GenericLayerSolver."""
    # Stage 1: SemiFreeSolver for rough estimate
    D_semi, thickness_semi = run_semifree_solver(I_obs, tm, elements, method=method, alpha=alpha)

    # Extract layer thicknesses from SemiFreeSolver result
    # Use simple heuristic: distribute total thickness proportionally
    # Better approach: use analytical structure from D_semi
    depth = tm.depth

    # Estimate layer positions from profile crossings
    estimated_thicknesses = estimate_layer_thicknesses_from_profile(depth, D_semi)

    # Stage 2: GenericLayerSolver with refined initial values
    D_opt, layer_thicknesses = run_generic_layer_solver(
        I_obs, tm, elements,
        thicknesses_init=estimated_thicknesses,
        method=method, alpha=alpha
    )

    return D_opt, layer_thicknesses


def estimate_layer_thicknesses_from_profile(depth: np.ndarray, D: np.ndarray) -> List[float]:
    """Estimate layer thicknesses from concentration profile."""
    # Elements: [C, Al, Ti, Si_ox, O, Si_bulk]
    # Find 50% crossings for each layer

    thicknesses = []

    # C layer: where C drops to 50%
    c_profile = D[:, 0]
    idx = np.argmin(np.abs(c_profile - 0.5 * c_profile.max()))
    z1 = depth[idx] if c_profile.max() > 0.1 else 5.0
    thicknesses.append(max(z1, 2.0))

    # Al2O3 layer: from z1 to where Al drops to 50%
    al_profile = D[:, 1]
    if al_profile.max() > 0.05:
        al_peak_idx = np.argmax(al_profile)
        after_peak = al_profile[al_peak_idx:]
        if len(after_peak) > 1:
            idx = al_peak_idx + np.argmin(np.abs(after_peak - 0.5 * al_profile.max()))
            z2 = depth[idx]
        else:
            z2 = z1 + 15.0
    else:
        z2 = z1 + 15.0
    thicknesses.append(max(z2 - z1, 5.0))

    # TiO2 layer
    ti_profile = D[:, 2]
    if ti_profile.max() > 0.05:
        ti_peak_idx = np.argmax(ti_profile)
        after_peak = ti_profile[ti_peak_idx:]
        if len(after_peak) > 1:
            idx = ti_peak_idx + np.argmin(np.abs(after_peak - 0.5 * ti_profile.max()))
            z3 = depth[idx]
        else:
            z3 = z2 + 35.0
    else:
        z3 = z2 + 35.0
    thicknesses.append(max(z3 - z2, 10.0))

    # SiO2 layer: from z3 to where Si_bulk rises to 50%
    si_bulk = D[:, 5]
    idx = np.argmin(np.abs(si_bulk - 0.5))
    z4 = depth[idx] if si_bulk.max() > 0.1 else z3 + 30.0
    thicknesses.append(max(z4 - z3, 10.0))

    return thicknesses


def run_comparison_test(n_trials: int = 5, noise_level: float = 0.02,
                        method: str = 'LLS', alpha: float = 0.0):
    """Run comparison test for all three solvers.

    Args:
        n_trials: Number of trials with different noise realizations
        noise_level: Noise level (fraction)
        method: Optimization method ('LLS', 'L1', 'L2', 'MEM')
        alpha: Regularization strength
    """

    true_thicknesses = [5.0, 15.0, 35.0, 30.0]
    interface_widths = [2.0, 3.0, 4.0, 5.0]
    element_labels = ['C', 'Al', 'Ti', 'Si_ox', 'O', 'Si_bulk']

    results = {
        'SemiFreeSolver': {'w1': [], 'centroids_true': [], 'centroids_est': [], 'D_opt': []},
        'Two-Stage': {'w1': [], 'centroids_true': [], 'centroids_est': [], 'D_opt': []},
        'GenericLayerSolver': {'w1': [], 'centroids_true': [], 'centroids_est': [], 'D_opt': []},
    }

    D_true_stored = None  # Store ground truth (same for all trials)

    for trial in range(n_trials):
        I_obs, tm, elements, D_true, depth = create_test_data(
            thicknesses=true_thicknesses,
            interface_widths=interface_widths,
            noise_level=noise_level,
            seed=42 + trial,
        )

        if D_true_stored is None:
            D_true_stored = D_true  # Ground truth is the same for all trials

        # 1. SemiFreeSolver
        D_semi, _ = run_semifree_solver(I_obs, tm, elements, method=method, alpha=alpha)

        # 2. Two-Stage
        D_two, _ = run_two_stage_solver(I_obs, tm, elements, method=method, alpha=alpha)

        # 3. GenericLayerSolver only (note: GenericLayerSolver doesn't use method/alpha internally)
        D_generic, _ = run_generic_layer_solver(I_obs, tm, elements, method=method, alpha=alpha)

        # Compute metrics for each element
        for solver_name, D_opt in [
            ('SemiFreeSolver', D_semi),
            ('Two-Stage', D_two),
            ('GenericLayerSolver', D_generic),
        ]:
            w1_list = []
            centroids_true = []
            centroids_est = []

            for i in range(6):
                w1 = compute_wasserstein(depth, D_true[:, i], D_opt[:, i])
                c_true = compute_centroid(depth, D_true[:, i])
                c_est = compute_centroid(depth, D_opt[:, i])

                w1_list.append(w1)
                centroids_true.append(c_true)
                centroids_est.append(c_est)

            results[solver_name]['w1'].append(w1_list)
            results[solver_name]['centroids_true'].append(centroids_true)
            results[solver_name]['centroids_est'].append(centroids_est)
            results[solver_name]['D_opt'].append(D_opt.copy())

    return results, element_labels, depth, D_true_stored


def visualize_comparison(results: Dict, element_labels: List[str], output_path: str):
    """Create 2D map visualization: X=True centroid, Y=Wasserstein distance (log scale)."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    solvers = ['SemiFreeSolver', 'Two-Stage', 'GenericLayerSolver']
    colors = ['black', 'blue', 'purple', 'orange', 'red', 'green']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    # Reference lines for 1Å, 0.1Å, 0.01Å
    ref_lines = [1.0, 0.1, 0.01]
    ref_labels = ['1 Å', '0.1 Å', '0.01 Å']

    for ax_idx, solver_name in enumerate(solvers):
        ax = axes[ax_idx]

        w1_all = np.array(results[solver_name]['w1'])  # (n_trials, 6)
        centroids_true = np.array(results[solver_name]['centroids_true'])  # (n_trials, 6)

        # Mean and std across trials
        w1_mean = np.mean(w1_all, axis=0)
        w1_std = np.std(w1_all, axis=0)
        centroids_mean = np.mean(centroids_true, axis=0)

        # Plot each element
        for i, label in enumerate(element_labels):
            ax.errorbar(
                centroids_mean[i], w1_mean[i],
                yerr=w1_std[i],
                fmt=markers[i], color=colors[i],
                markersize=12, capsize=5, capthick=2,
                label=label, linewidth=2
            )

        # Reference lines
        for ref, ref_label in zip(ref_lines, ref_labels):
            ax.axhline(y=ref, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(ax.get_xlim()[1] * 0.95, ref * 1.1, ref_label,
                   ha='right', va='bottom', fontsize=9, color='gray')

        ax.set_xlabel('True Centroid Position (Å)', fontsize=12)
        ax.set_ylabel('Wasserstein Distance (Å)', fontsize=12)
        ax.set_title(solver_name, fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(0.005, 5.0)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='upper right', fontsize=9)

    plt.suptitle(
        'Solver Comparison: Wasserstein Distance vs True Position\n'
        '4-layer model (C/Al₂O₃/TiO₂/SiO₂/Si), 2% noise, 5 trials',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {output_path}")

    return fig


def visualize_depth_profiles(
    results: Dict,
    element_labels: List[str],
    depth: np.ndarray,
    D_true: np.ndarray,
    output_path: str,
    method: str = 'LLS',
    alpha: float = 0.0
):
    """
    Plot depth profiles: 3 solvers side by side.
    Each panel shows all 5 trials overlaid on the ground truth.
    """
    solvers = ['SemiFreeSolver', 'Two-Stage', 'GenericLayerSolver']
    n_elements = len(element_labels)

    # Colors for elements
    element_colors = ['black', 'blue', 'purple', 'orange', 'red', 'green']

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)

    for ax_idx, solver_name in enumerate(solvers):
        ax = axes[ax_idx]

        # Plot ground truth (thick lines)
        for el_idx, (label, color) in enumerate(zip(element_labels, element_colors)):
            ax.plot(
                depth, D_true[:, el_idx] * 100,
                color=color, linewidth=2.5, linestyle='-',
                label=f'{label} (true)' if ax_idx == 0 else None,
                alpha=0.9
            )

        # Plot all 5 trials (thin lines, semi-transparent)
        D_opt_list = results[solver_name]['D_opt']
        n_trials = len(D_opt_list)

        for trial_idx, D_opt in enumerate(D_opt_list):
            for el_idx, color in enumerate(element_colors):
                ax.plot(
                    depth, D_opt[:, el_idx] * 100,
                    color=color, linewidth=0.8, linestyle='--',
                    alpha=0.5,
                    label=f'Trial {trial_idx+1}' if (ax_idx == 0 and el_idx == 0) else None
                )

        ax.set_xlabel('Depth (Å)', fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel('Concentration (%)', fontsize=12)
        ax.set_title(solver_name, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

    # Create legend
    # Elements legend
    legend_elements = [
        plt.Line2D([0], [0], color=c, linewidth=2.5, label=f'{l} (true)')
        for l, c in zip(element_labels, element_colors)
    ]
    legend_elements.append(
        plt.Line2D([0], [0], color='gray', linewidth=0.8, linestyle='--', alpha=0.5, label='Trials (1-5)')
    )

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=7,
        fontsize=10
    )

    # Title with method info
    if method == 'LLS':
        method_str = 'LLS (no regularization)'
    else:
        method_str = f'{method} (α={alpha}, D_prior=0)'

    plt.suptitle(
        f'Depth Profile Comparison: Ground Truth vs 5 Trials ({method_str}, 2% noise)\n'
        '4-layer model: C / Al₂O₃ / TiO₂ / SiO₂ / Si',
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Depth profile figure saved: {output_path}")

    return fig


def print_summary(results: Dict, element_labels: List[str]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("Summary: Mean Wasserstein Distance (Å) per Element")
    print("=" * 80)

    solvers = ['SemiFreeSolver', 'Two-Stage', 'GenericLayerSolver']

    header = f"{'Element':<10}"
    for solver in solvers:
        header += f"{solver:<20}"
    print(header)
    print("-" * 70)

    for i, label in enumerate(element_labels):
        row = f"{label:<10}"
        for solver in solvers:
            w1_all = np.array(results[solver]['w1'])
            w1_mean = np.mean(w1_all[:, i])
            w1_std = np.std(w1_all[:, i])
            row += f"{w1_mean:.3f} ± {w1_std:.3f}       "
        print(row)

    print("-" * 70)

    # Total mean
    row = f"{'MEAN':<10}"
    for solver in solvers:
        w1_all = np.array(results[solver]['w1'])
        total_mean = np.mean(w1_all)
        total_std = np.std(np.mean(w1_all, axis=1))
        row += f"{total_mean:.3f} ± {total_std:.3f}       "
    print(row)
    print("=" * 80)


def main():
    """Main function."""

    # ========================================
    # 1. LLS (baseline, no regularization)
    # ========================================
    print("=" * 80)
    print("3-Solver Comparison Test: LLS (no regularization)")
    print("SemiFreeSolver vs Two-Stage vs GenericLayerSolver")
    print("=" * 80)

    results_lls, element_labels, depth, D_true = run_comparison_test(
        n_trials=5, noise_level=0.02, method='LLS', alpha=0.0
    )

    print_summary(results_lls, element_labels)

    visualize_comparison(results_lls, element_labels, '/tmp/solver_comparison_w1_map_LLS.png')
    visualize_depth_profiles(results_lls, element_labels, depth, D_true,
                             '/tmp/solver_comparison_depth_profiles_LLS.png',
                             method='LLS', alpha=0.0)

    # ========================================
    # 2. L1 regularization (alpha=1.0, D_prior=0)
    # ========================================
    print("\n" + "=" * 80)
    print("3-Solver Comparison Test: L1 (α=1.0, D_prior=0)")
    print("SemiFreeSolver vs Two-Stage vs GenericLayerSolver")
    print("=" * 80)

    results_l1, _, _, _ = run_comparison_test(
        n_trials=5, noise_level=0.02, method='L1', alpha=1.0
    )

    print_summary(results_l1, element_labels)

    visualize_comparison(results_l1, element_labels, '/tmp/solver_comparison_w1_map_L1.png')
    visualize_depth_profiles(results_l1, element_labels, depth, D_true,
                             '/tmp/solver_comparison_depth_profiles_L1.png',
                             method='L1', alpha=1.0)

    # ========================================
    # 3. Summary comparison
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY: LLS vs L1 Comparison")
    print("=" * 80)

    solvers = ['SemiFreeSolver', 'Two-Stage', 'GenericLayerSolver']

    print(f"\n{'Solver':<20} {'LLS Mean W1':<15} {'L1 Mean W1':<15} {'Improvement':<15}")
    print("-" * 65)

    for solver in solvers:
        lls_mean = np.mean(results_lls[solver]['w1'])
        l1_mean = np.mean(results_l1[solver]['w1'])
        improvement = (lls_mean - l1_mean) / lls_mean * 100

        print(f"{solver:<20} {lls_mean:.3f} Å        {l1_mean:.3f} Å        {improvement:+.1f}%")

    print("=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
