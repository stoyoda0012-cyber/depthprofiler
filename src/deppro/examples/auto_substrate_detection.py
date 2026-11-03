"""
Depth-profile analysis with automatic substrate detection
========================================
Uses the substrate detection of StructureAnalyzer to set substrate_idx
automatically and to build the exponential prior.
"""

import numpy as np
from scipy.special import erf

from deppro import (
    TransferMatrix,
    ElementConfig,
    FreeSolver,
    SolverConfig,
    StructureAnalyzer,
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
    print("=" * 70)
    print("Depth-profile analysis with automatic substrate detection")
    print("=" * 70)

    # ========== Settings ==========
    depth = np.arange(0, 100, 2.0)
    angles = np.array([0, 30, 45, 60])
    photon_energy = 1486.6

    # Element definitions (with oxidation states)
    elements = [
        ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
    ]
    element_labels = ['Al³⁺', 'Si⁴⁺', 'Si⁰', 'O²⁻']

    # ========== True profile and observed data ==========
    D_true = create_3layer_profile(depth)

    tm = TransferMatrix(depth=depth, step_size=2.0, angles=angles, photon_energy=photon_energy)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)
    np.random.seed(42)
    noise = np.random.randn(*I_true.shape) * 0.02 * I_true
    I_obs = I_true + noise
    I_obs = np.maximum(I_obs, 1e-6)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    # ========== Step 1: automatic substrate detection ==========
    print("\n>>> Step 1: substrate detection")
    print("-" * 50)

    analyzer = StructureAnalyzer()
    result = analyzer.analyze_with_substrate_detection(
        I_obs, angles,
        element_labels=element_labels,
        verbose=True
    )

    substrate_detection = result.substrate_detection

    if not substrate_detection.has_substrate:
        print(f"\n⚠️ Substrate detection failed: {substrate_detection.fallback_reason}")
        print("Give substrate_idx by hand")
        return

    # Use the detected substrate index
    substrate_idx = substrate_detection.substrate_indices[0]
    print(f"\nDetected substrate: index={substrate_idx} ({element_labels[substrate_idx]})")

    # ========== Step 2: IMFP ==========
    print("\n>>> Step 2: IMFP")
    print("-" * 50)

    # IMFP of the substrate element (Si)
    imfp_substrate = IMFP.tpp2m(
        KE=photon_energy - 99.5,  # Si 2p BE
        Nv=4, density=2.33, Aw=28.09, Eg=1.1
    )
    print(f"IMFP(substrate) = {imfp_substrate:.1f} Å")

    # ========== Step 3: exponential prior ==========
    print("\n>>> Step 3: exponential prior")
    print("-" * 50)

    D_prior = create_exponential_prior(
        depth,
        n_elements=4,
        substrate_idx=substrate_idx,
        d_penetration=imfp_substrate
    )
    print(f"Prior built: substrate_idx={substrate_idx}, d_penetration={imfp_substrate:.1f} Å")

    # ========== Step 4: FreeSolver ==========
    print("\n>>> Step 4: running FreeSolver")
    print("-" * 50)

    config = SolverConfig.recommended(alpha=0.0003)
    solver = FreeSolver(tm, config, D_prior, elements=elements)
    solve_result = solver.solve(I_obs, D_init=D_prior.copy(), verbose=True)

    # ========== Evaluation ==========
    D_est = solve_result.D_opt
    rmse = np.sqrt(np.mean((D_est - D_true)**2))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"RMSE: {rmse:.4f}")
    print(f"chi²: {solve_result.chi2:.2e}")
    print()
    print(f"{'Element':<10} {'True(0Å)':>10} {'Est(0Å)':>10} {'Error':>10}")
    print("-" * 50)
    for i, name in enumerate(element_labels):
        error = D_est[0, i] - D_true[0, i]
        print(f"{name:<10} {D_true[0, i]:>10.3f} {D_est[0, i]:>10.3f} {error:>+10.3f}")

    print("\nCompleted fully automatically, from substrate detection to the FreeSolver analysis.")


if __name__ == '__main__':
    main()
