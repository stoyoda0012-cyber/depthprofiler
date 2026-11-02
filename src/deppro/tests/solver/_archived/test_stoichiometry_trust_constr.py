"""
Stoichiometry Solver - Trust-Constr vs SLSQP vs L-BFGS-B Comparison

Compare optimization methods for NON pattern reconstruction.

Usage:
    python -m deppro.tests.solver.test_stoichiometry_trust_constr
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import time

from deppro.transfer_matrix import TransferMatrix
from deppro.depth_profile import ElementConfig


def create_non_pattern(depth, n1_thickness, o_thickness, n2_thickness, interface_width=5.0):
    """NON: Si3N4 / SiO2 / Si3N4 / Si"""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    z1 = n1_thickness
    z2 = n1_thickness + o_thickness
    z3 = n1_thickness + o_thickness + n2_thickness

    D = np.zeros((len(depth), 4))

    sin1_profile = 0.5 * (1 - erf((depth - z1) / sigma))
    si_n1 = (3/7) * sin1_profile
    n1 = (4/7) * sin1_profile

    sio2_profile = 0.5 * (1 + erf((depth - z1) / sigma)) * 0.5 * (1 - erf((depth - z2) / sigma))
    si_o = (1/3) * sio2_profile
    D[:, 2] = (2/3) * sio2_profile

    sin2_profile = 0.5 * (1 + erf((depth - z2) / sigma)) * 0.5 * (1 - erf((depth - z3) / sigma))
    si_n2 = (3/7) * sin2_profile
    n2 = (4/7) * sin2_profile

    D[:, 1] = n1 + n2
    D[:, 0] = si_n1 + si_o + si_n2
    D[:, 3] = 0.5 * (1 + erf((depth - z3) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


class StoichiometrySolverMultiMethod:
    """Solver with multiple optimization methods."""

    def __init__(self, transfer_matrix, N_idx, O_idx, Si4_idx, substrate_idx,
                 Si_N_ratio=3/4, Si_O_ratio=1/2, degree=1.0):
        self.tm = transfer_matrix
        self.depth = transfer_matrix.depth
        self.n_depth = len(self.depth)
        self.n_elements = transfer_matrix.n_elements
        self.N_idx = N_idx
        self.O_idx = O_idx
        self.Si4_idx = Si4_idx
        self.substrate_idx = substrate_idx
        self.Si_N_ratio = Si_N_ratio
        self.Si_O_ratio = Si_O_ratio
        self.degree = degree

    def _build_substrate_profile(self, thickness, abruptness):
        sigma = abruptness / (2 * np.sqrt(np.log(2)))
        return 0.5 * (1 + erf((self.depth - thickness) / sigma))

    def _build_profile(self, N_conc, O_conc, thickness, abruptness):
        D = np.zeros((self.n_depth, self.n_elements))
        sub_profile = self._build_substrate_profile(thickness, abruptness)
        D[:, self.substrate_idx] = sub_profile
        film_scale = np.maximum(1.0 - sub_profile, 1e-6)

        N_scaled = N_conc * film_scale
        O_scaled = O_conc * film_scale
        Si4_scaled = N_scaled * self.Si_N_ratio + O_scaled * self.Si_O_ratio

        D[:, self.N_idx] = N_scaled
        D[:, self.O_idx] = O_scaled
        D[:, self.Si4_idx] = Si4_scaled

        D = np.clip(D, 1e-6, 1.0)
        D = D / np.maximum(D.sum(axis=1, keepdims=True), 1e-6)
        return D

    def _objective(self, x, I_obs, thickness, abruptness):
        N_conc = x[:self.n_depth]
        O_conc = x[self.n_depth:2*self.n_depth]
        D = self._build_profile(N_conc, O_conc, thickness, abruptness)
        I_pred = self.tm.forward(D, normalize=True)
        chi2 = np.sum((I_pred - I_obs) ** 2)
        return self.degree * chi2

    def _gradient(self, x, I_obs, thickness, abruptness):
        """Numerical gradient for trust-constr."""
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = self._objective(x, I_obs, thickness, abruptness)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (self._objective(x_plus, I_obs, thickness, abruptness) - f0) / eps
        return grad

    def solve(self, I_obs, thickness_init=25.0, abruptness=5.0, max_iter=500, method='L-BFGS-B'):
        x0 = np.concatenate([np.ones(self.n_depth) * 0.3, np.ones(self.n_depth) * 0.3])
        bounds = Bounds(lb=np.zeros(2 * self.n_depth), ub=np.ones(2 * self.n_depth))

        t0 = time.time()

        if method == 'trust-constr':
            # Trust region constrained method
            from scipy.optimize import BFGS
            result = minimize(
                self._objective, x0,
                args=(I_obs, thickness_init, abruptness),
                method='trust-constr',
                jac='2-point',  # Use numerical gradient
                hess=BFGS(),    # Use BFGS quasi-Newton Hessian approximation
                bounds=bounds,
                options={
                    'maxiter': max_iter,
                    'verbose': 0,
                    'gtol': 1e-8,
                    'xtol': 1e-8,
                }
            )
        elif method == 'SLSQP':
            result = minimize(
                self._objective, x0,
                args=(I_obs, thickness_init, abruptness),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter}
            )
        else:  # L-BFGS-B
            result = minimize(
                self._objective, x0,
                args=(I_obs, thickness_init, abruptness),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter}
            )

        elapsed = time.time() - t0

        N_conc = result.x[:self.n_depth]
        O_conc = result.x[self.n_depth:2*self.n_depth]
        D_opt = self._build_profile(N_conc, O_conc, thickness_init, abruptness)
        I_opt = self.tm.forward(D_opt, normalize=True)
        chi2 = np.sum((I_opt - I_obs) ** 2)

        return {
            'D_opt': D_opt,
            'chi2': chi2,
            'success': result.success,
            'nit': result.nit if hasattr(result, 'nit') else 0,
            'elapsed': elapsed,
        }


def test_methods():
    """Compare optimization methods."""
    print("=" * 70)
    print("Stoichiometry Solver - Method Comparison")
    print("trust-constr vs SLSQP vs L-BFGS-B")
    print("=" * 70)

    depth = np.arange(0, 100, 1.0)
    angles = [0, 15, 30, 45, 60]

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('N', '1s', 'Si3N4', oxidation_state=-3),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    # NON pattern: Si3N4(5Å) / SiO2(15Å) / Si3N4(5Å) / Si
    D_true = create_non_pattern(depth, 5.0, 15.0, 5.0)

    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)
    I_true = tm.forward(D_true, normalize=True)

    np.random.seed(42)
    I_obs = I_true * (1 + 0.02 * np.random.randn(*I_true.shape))
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    methods = ['L-BFGS-B', 'SLSQP', 'trust-constr']
    degree_values = [1.0, 5.0, 10.0, 50.0]

    all_results = {}

    for method in methods:
        print(f"\n{'=' * 70}")
        print(f"Method: {method}")
        print(f"{'=' * 70}")

        results = {}
        for degree in degree_values:
            solver = StoichiometrySolverMultiMethod(
                tm, N_idx=1, O_idx=2, Si4_idx=0, substrate_idx=3, degree=degree
            )
            result = solver.solve(I_obs, thickness_init=25.0, abruptness=5.0,
                                  max_iter=1000, method=method)
            D_opt = result['D_opt']

            # Calculate RMSE
            rmse_n = np.sqrt(np.mean((D_opt[:, 1] - D_true[:, 1]) ** 2))
            rmse_o = np.sqrt(np.mean((D_opt[:, 2] - D_true[:, 2]) ** 2))

            # Find N peaks
            N_peaks = []
            N_max = np.max(D_opt[:, 1])
            for i in range(1, len(depth) - 1):
                if D_opt[i, 1] > D_opt[i-1, 1] and D_opt[i, 1] > D_opt[i+1, 1]:
                    if D_opt[i, 1] > 0.1 * N_max:
                        N_peaks.append((depth[i], D_opt[i, 1]))

            is_bimodal = len(N_peaks) >= 2

            results[degree] = {
                'D_opt': D_opt,
                'chi2': result['chi2'],
                'rmse_n': rmse_n,
                'rmse_o': rmse_o,
                'N_peaks': N_peaks,
                'is_bimodal': is_bimodal,
                'elapsed': result['elapsed'],
                'nit': result['nit'],
            }

            peak_str = ", ".join([f"{z:.0f}Å" for z, _ in N_peaks[:3]]) if N_peaks else "none"
            bimodal_str = "BIMODAL!" if is_bimodal else ""
            print(f"  deg={degree:5.1f}: Chi²={result['chi2']:.6f}, RMSE(N)={rmse_n:.4f}, "
                  f"time={result['elapsed']:.2f}s, peaks=[{peak_str}] {bimodal_str}")

        all_results[method] = results

    # Visualization
    fig, axes = plt.subplots(len(methods) + 1, len(degree_values), figsize=(16, 12))

    # True profile
    for j, degree in enumerate(degree_values):
        ax = axes[0, j]
        ax.plot(depth, D_true[:, 1], 'b-', linewidth=2, label='N')
        ax.plot(depth, D_true[:, 2], 'r-', linewidth=2, label='O')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 0.8)
        if j == 0:
            ax.set_ylabel('TRUE')
        ax.set_title(f'degree={degree}')
        ax.legend(fontsize=8)

    # Results for each method
    for i, method in enumerate(methods):
        for j, degree in enumerate(degree_values):
            ax = axes[i + 1, j]
            D_opt = all_results[method][degree]['D_opt']

            ax.plot(depth, D_opt[:, 1], 'b-', linewidth=2, label='N opt')
            ax.plot(depth, D_opt[:, 2], 'r-', linewidth=2, label='O opt')
            ax.plot(depth, D_true[:, 1], 'b--', alpha=0.3, label='N true')
            ax.plot(depth, D_true[:, 2], 'r--', alpha=0.3, label='O true')
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 0.8)

            if j == 0:
                ax.set_ylabel(method)

            bimodal = "BIMODAL" if all_results[method][degree]['is_bimodal'] else ""
            rmse = all_results[method][degree]['rmse_n']
            ax.set_title(f'{bimodal}\nRMSE(N)={rmse:.3f}', fontsize=9)

            if i == len(methods) - 1:
                ax.set_xlabel('Depth (Å)')

    plt.suptitle('NON Pattern: Method Comparison (trust-constr vs SLSQP vs L-BFGS-B)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/stoichiometry_trust_constr.png', dpi=150)
    print(f"\nSaved: outputs/stoichiometry_trust_constr.png")
    plt.close()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: NON Pattern - Method Comparison")
    print("=" * 70)
    print(f"{'Method':>15} | {'Degree':>8} | {'Chi²':>12} | {'RMSE(N)':>10} | {'Time(s)':>8} | {'Bimodal':>8}")
    print("-" * 80)
    for method in methods:
        for degree in degree_values:
            res = all_results[method][degree]
            bimodal = "Yes" if res['is_bimodal'] else "No"
            print(f"{method:>15} | {degree:>8.1f} | {res['chi2']:>12.6f} | {res['rmse_n']:>10.4f} | "
                  f"{res['elapsed']:>8.2f} | {bimodal:>8}")

    return all_results


if __name__ == "__main__":
    test_methods()
