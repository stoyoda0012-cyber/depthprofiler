"""
FreeSolver with SolverConfig.recommended()
=================================================
Reconstruction of a three-layer film with MEM + exponential prior + constraints

This example reconstructs the depth profile of an Al2O3/SiO2/Si three-layer film.
"""

import numpy as np
from scipy.special import erf

from deppro import (
    TransferMatrix,
    ElementConfig,
    FreeSolver,
    SolverConfig,
    create_exponential_prior,
    IMFP,
)


def create_3layer_profile(depth, d1=15.0, d2=45.0, sigma=3.0):
    """Three-layer profile (Al2O3/SiO2/Si)"""
    n_depth = len(depth)
    D = np.zeros((n_depth, 4))

    layer1 = 0.5 * (1 - erf((depth - d1) / sigma))
    layer2 = 0.5 * (1 + erf((depth - d1) / sigma)) * 0.5 * (1 - erf((depth - d2) / sigma))
    layer3 = 0.5 * (1 + erf((depth - d2) / sigma))

    D[:, 0] = 0.4 * layer1       # Al
    D[:, 3] = 0.6 * layer1       # O from Al2O3
    D[:, 1] = 0.33 * layer2      # Si_ox
    D[:, 3] += 0.67 * layer2     # O from SiO2
    D[:, 2] = layer3             # Si_bulk

    D = np.clip(D, 1e-6, None)
    return D / D.sum(axis=1, keepdims=True)


def main():
    # ========== Settings ==========
    depth = np.arange(0, 100, 2.0)
    angles = np.array([0, 30, 45, 60])
    d1_true, d2_true = 15.0, 45.0
    photon_energy = 1486.6  # Al Kα

    # Element definitions (with oxidation states)
    elements = [
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
    ]

    # IMFP of the substrate
    imfp_si = IMFP.tpp2m(
        KE=photon_energy - 99.5,  # Si 2p BE
        Nv=4, density=2.33, Aw=28.09, Eg=1.1
    )
    print(f"IMFP(Si) = {imfp_si:.1f} Å")

    # ========== True profile and observed data ==========
    D_true = create_3layer_profile(depth, d1_true, d2_true)

    # Transfer matrix
    tm = TransferMatrix(depth=depth, step_size=2.0, angles=angles, photon_energy=photon_energy)
    tm.compute(D_true, elements)

    # Observed data (2 % noise added)
    I_true = tm.forward(D_true, normalize=True)
    np.random.seed(42)
    noise = np.random.randn(*I_true.shape) * 0.02 * I_true
    I_obs = I_true + noise
    I_obs = np.maximum(I_obs, 1e-6)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    # ========== Recommended FreeSolver settings ==========
    # 1. Exponential Prior (d = λ_substrate)
    D_prior = create_exponential_prior(
        depth, n_elements=4, substrate_idx=2,
        d_penetration=imfp_si
    )

    # 2. Use the settings of SolverConfig.recommended()
    config = SolverConfig.recommended(alpha=0.0003)

    # 3. Run the solver
    solver = FreeSolver(tm, config, D_prior, elements=elements)
    result = solver.solve(I_obs, D_init=D_prior.copy(), verbose=True)

    # ========== Evaluation ==========
    D_est = result.D_opt
    rmse = np.sqrt(np.mean((D_est - D_true)**2))

    print("\n" + "=" * 60)
    print("Reconstruction")
    print("=" * 60)
    print(f"RMSE: {rmse:.4f}")
    print(f"chi²: {result.chi2:.2e}")
    print()
    print(f"{'Element':<10} {'True(0Å)':>10} {'Est(0Å)':>10} {'True(80Å)':>10} {'Est(80Å)':>10}")
    print("-" * 60)

    element_names = ['Al³⁺', 'Si⁴⁺', 'Si⁰', 'O²⁻']
    for i, name in enumerate(element_names):
        print(f"{name:<10} {D_true[0, i]:>10.3f} {D_est[0, i]:>10.3f} "
              f"{D_true[40, i]:>10.3f} {D_est[40, i]:>10.3f}")


if __name__ == '__main__':
    main()
