"""
Regularization Comparison for Variable Stoichiometry

Compare LLS, L1, L2, MEM methods when optimizing:
- Thickness
- Interface width
- Stoichiometry (O ratio)

This is an information-limited case where regularization may help.

Usage:
    python -m deppro.tests.benchmark.regularization.test_variable_stoich_regularization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig
from deppro.inverse_solver import (
    GenericLayerSolver, GenericLayerConfig, LayerDefinition, SolverConfig
)
from deppro.paths import ensure_outputs_dir


def create_siox_profile(depth, thickness, o_ratio, interface_width=5.0):
    """Create SiOx/Si profile."""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    D = np.zeros((len(depth), 3))
    siox_profile = 0.5 * (1 - erf((depth - thickness) / sigma))
    total_stoich = 1 + o_ratio
    D[:, 0] = (1 / total_stoich) * siox_profile
    D[:, 1] = (o_ratio / total_stoich) * siox_profile
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))
    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def run_comparison():
    print('=' * 70)
    print('Regularization Comparison: LLS vs L1 vs L2 vs MEM')
    print('=' * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]

    true_thickness = 20.0
    true_o_ratio = 1.7
    true_width = 5.0

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_siox_profile(depth, true_thickness, true_o_ratio, true_width)
    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)
    I_true = tm.forward(D_true, normalize=True)

    n_trials = 10
    noise_level = 0.03

    methods = ['LLS', 'L1', 'L2', 'MEM']
    results_all = {m: {'thickness': [], 'o_ratio': [], 'width': [], 'chi2': [], 'success': []} for m in methods}

    print(f'\nTrue values: thickness={true_thickness}A, O_ratio={true_o_ratio}, width={true_width}A')
    print(f'Noise level: {noise_level*100}%, Trials: {n_trials}')
    print('\nRunning trials...')

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        I_obs = I_true * (1 + noise_level * np.random.randn(*I_true.shape))
        I_obs = np.clip(I_obs, 1e-6, None)
        I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

        for method in methods:
            layer_config = GenericLayerConfig(
                layers=[
                    LayerDefinition(
                        'SiOx', [0, 1], {0: 1, 1: 2.0},
                        stoichiometry_bounds={1: (1.3, 2.0)},
                    ),
                ],
                substrate=LayerDefinition('Si', [2], {2: 1}),
                initial_thicknesses=[25.0],
                initial_widths=[8.0],
                optimize_stoichiometry=True,
                optimize_widths=True,
            )

            if method == 'LLS':
                alpha_val = 0.0
            else:
                alpha_val = 0.01

            config = SolverConfig(
                method=method,
                alpha=alpha_val,
                max_iter=500,
            )

            solver = GenericLayerSolver(tm, config, layer_config, elements=elements)
            result = solver.solve(I_obs)

            results_all[method]['thickness'].append(result.layer_thicknesses[0])
            results_all[method]['o_ratio'].append(result.layer_stoichiometries[0][1])
            results_all[method]['width'].append(result.interface_widths[0])
            results_all[method]['chi2'].append(result.chi2)
            results_all[method]['success'].append(result.success)

    # Results
    print('\n' + '=' * 70)
    print('Results Summary (mean +/- std)')
    print('=' * 70)
    print(f'{"Method":8} | {"Thickness":15} | {"O_ratio":15} | {"Width":15} | {"Chi2":10}')
    print('-' * 70)

    for method in methods:
        t = np.array(results_all[method]['thickness'])
        o = np.array(results_all[method]['o_ratio'])
        w = np.array(results_all[method]['width'])
        c = np.array(results_all[method]['chi2'])

        t_err = abs(t - true_thickness)
        o_err = abs(o - true_o_ratio)
        w_err = abs(w - true_width)

        print(f'{method:8} | {t.mean():.2f}+/-{t.std():.2f} (e={t_err.mean():.2f}) | {o.mean():.2f}+/-{o.std():.2f} (e={o_err.mean():.2f}) | {w.mean():.2f}+/-{w.std():.2f} (e={w_err.mean():.2f}) | {c.mean():.6f}')

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    for i, method in enumerate(methods):
        data = results_all[method]['thickness']
        ax.boxplot([data], positions=[i], widths=0.6)
    ax.axhline(true_thickness, color='r', linestyle='--', label=f'True={true_thickness}')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel('Thickness (A)')
    ax.set_title('Thickness Estimation')
    ax.legend()

    ax = axes[0, 1]
    for i, method in enumerate(methods):
        data = results_all[method]['o_ratio']
        ax.boxplot([data], positions=[i], widths=0.6)
    ax.axhline(true_o_ratio, color='r', linestyle='--', label=f'True={true_o_ratio}')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel('O/Si ratio')
    ax.set_title('Oxygen Ratio Estimation')
    ax.legend()

    ax = axes[1, 0]
    for i, method in enumerate(methods):
        data = results_all[method]['width']
        ax.boxplot([data], positions=[i], widths=0.6)
    ax.axhline(true_width, color='r', linestyle='--', label=f'True={true_width}')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel('Interface Width (A)')
    ax.set_title('Interface Width Estimation')
    ax.legend()

    ax = axes[1, 1]
    for i, method in enumerate(methods):
        data = results_all[method]['chi2']
        ax.boxplot([data], positions=[i], widths=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel('Chi2')
    ax.set_title('Fitting Quality (Chi2)')
    ax.set_yscale('log')

    plt.suptitle(f'Regularization Comparison: SiOx with Variable Stoichiometry + Width\n(Noise={noise_level*100}%, {n_trials} trials)', fontsize=12)
    plt.tight_layout()
    output_path = str(ensure_outputs_dir() / 'regularization_comparison.png')
    plt.savefig(output_path, dpi=150)
    print(f'\nSaved: {output_path}')
    plt.close()

    # Correlation analysis
    print('\n' + '=' * 70)
    print('Parameter Correlations (LLS method)')
    print('=' * 70)
    t_lls = np.array(results_all['LLS']['thickness'])
    o_lls = np.array(results_all['LLS']['o_ratio'])
    w_lls = np.array(results_all['LLS']['width'])

    corr_to = np.corrcoef(t_lls, o_lls)[0, 1]
    corr_tw = np.corrcoef(t_lls, w_lls)[0, 1]
    corr_ow = np.corrcoef(o_lls, w_lls)[0, 1]

    print(f'Thickness-O_ratio correlation: {corr_to:.3f}')
    print(f'Thickness-Width correlation:   {corr_tw:.3f}')
    print(f'O_ratio-Width correlation:     {corr_ow:.3f}')
    print('\nNote: High correlation indicates parameter coupling (hard to distinguish)')

    return results_all


if __name__ == "__main__":
    run_comparison()
