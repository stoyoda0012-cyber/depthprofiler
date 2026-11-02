"""
Loss Function Diagnostics
=========================

Investigate why Poisson NLL performs poorly.

Date: 2026-01-28
"""

import numpy as np
from scipy.special import erf


def create_sio2_si_profile(depth: np.ndarray, thickness: float = 30.0, interface_width: float = 3.0) -> np.ndarray:
    """Create SiO2/Si bilayer profile."""
    sigma = interface_width / (2 * np.sqrt(np.log(2)))

    D = np.zeros((len(depth), 3))
    sio2 = 0.5 * (1 - erf((depth - thickness) / sigma))
    D[:, 0] = (1/3) * sio2  # Si in SiO2
    D[:, 1] = (2/3) * sio2  # O in SiO2
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)
    return D


def compute_losses(I_obs: np.ndarray, I_pred: np.ndarray, weights: np.ndarray = None):
    """Compute all three loss functions."""
    # OLS
    diff = I_obs - I_pred
    ols = np.sum(diff ** 2)

    # WLS
    if weights is None:
        weights = I_obs  # Default: weight = intensity
    wls = np.sum(weights * diff ** 2)

    # Poisson NLL (deviance)
    I_obs_safe = np.maximum(I_obs, 1e-10)
    I_pred_safe = np.maximum(I_pred, 1e-10)
    poisson_nll = 2.0 * np.sum(
        I_obs_safe * np.log(I_obs_safe / I_pred_safe) - (I_obs - I_pred_safe)
    )

    return {
        'OLS': ols,
        'WLS': wls,
        'Poisson_NLL': poisson_nll,
    }


def compute_gradients(I_obs: np.ndarray, I_pred: np.ndarray, weights: np.ndarray = None):
    """Compute gradients of loss functions w.r.t. I_pred."""
    if weights is None:
        weights = I_obs

    diff = I_obs - I_pred

    # OLS gradient: d/dI_pred = -2 * (I_obs - I_pred)
    grad_ols = -2 * diff

    # WLS gradient: d/dI_pred = -2 * w * (I_obs - I_pred)
    grad_wls = -2 * weights * diff

    # Poisson NLL gradient: d/dI_pred = 2 * (1 - I_obs / I_pred)
    I_pred_safe = np.maximum(I_pred, 1e-10)
    grad_poisson = 2 * (1 - I_obs / I_pred_safe)

    return {
        'OLS': grad_ols,
        'WLS': grad_wls,
        'Poisson_NLL': grad_poisson,
    }


def main():
    from deppro.transfer_matrix import TransferMatrix
    from deppro.depth_profile import ElementConfig

    print("=" * 70)
    print("Loss Function Diagnostics")
    print("=" * 70)

    # Setup
    depth = np.arange(0, 80, 2.0)
    angles = [0, 15, 30, 45, 60]

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    D_true = create_sio2_si_profile(depth, thickness=30.0)

    tm = TransferMatrix(depth=depth, step_size=2.0, angles=angles, photon_energy=1486.6)
    tm.compute(D_true, elements)

    I_true = tm.forward(D_true, normalize=True)

    print("\n1. True intensity statistics:")
    print(f"   Shape: {I_true.shape}")
    print(f"   Range: [{I_true.min():.4f}, {I_true.max():.4f}]")
    print(f"   Mean: {I_true.mean():.4f}")

    # Check intensity values per element
    print("\n   Per-element ranges:")
    for i, el in enumerate(['Si_ox', 'O', 'Si_bulk']):
        print(f"   {el}: [{I_true[:, i].min():.4f}, {I_true[:, i].max():.4f}]")

    # Test with exact match (no error)
    print("\n2. Loss at exact match (I_pred = I_obs):")
    losses = compute_losses(I_true, I_true)
    for name, val in losses.items():
        print(f"   {name}: {val:.2e}")

    # Test with small perturbation
    print("\n3. Loss with 2% perturbation:")
    I_perturbed = I_true * (1 + 0.02 * np.random.randn(*I_true.shape))
    I_perturbed = np.clip(I_perturbed, 1e-10, None)
    I_perturbed = I_perturbed / I_perturbed.sum(axis=1, keepdims=True)

    losses = compute_losses(I_true, I_perturbed)
    for name, val in losses.items():
        print(f"   {name}: {val:.2e}")

    # Test with systematic bias (thickness error)
    print("\n4. Loss with 5Å thickness error:")
    D_wrong = create_sio2_si_profile(depth, thickness=35.0)  # 5Å too thick
    I_wrong = tm.forward(D_wrong, normalize=True)

    losses = compute_losses(I_true, I_wrong)
    for name, val in losses.items():
        print(f"   {name}: {val:.2e}")

    # Gradient magnitude comparison
    print("\n5. Gradient magnitude (normalized) at 2% perturbation:")
    grads = compute_gradients(I_true, I_perturbed)
    for name, grad in grads.items():
        norm = np.linalg.norm(grad)
        mean_abs = np.mean(np.abs(grad))
        print(f"   {name}: |grad|={norm:.2e}, mean|grad|={mean_abs:.2e}")

    # Key insight: Poisson NLL gradient scaling
    print("\n6. Poisson NLL gradient scaling issue:")
    print("   Poisson NLL: d/dI_pred = 2 * (1 - I_obs / I_pred)")
    print("   At small I_pred, gradient → -2 * I_obs / I_pred (diverges)")
    print("   At large I_pred, gradient → 2 (bounded)")
    print()
    print("   This asymmetry can cause optimization instability")
    print("   when I_pred < I_obs (underestimation)")

    # Test gradient at different error levels
    print("\n7. Gradient behavior at different errors:")
    errors = [0.01, 0.05, 0.10, 0.20, 0.50]
    for err in errors:
        I_under = I_true * (1 - err)  # Underestimate
        I_under = np.clip(I_under, 1e-10, None)
        I_under = I_under / I_under.sum(axis=1, keepdims=True)

        grads = compute_gradients(I_true, I_under)
        print(f"   Error={err*100:.0f}% (under):", end='')
        for name, grad in grads.items():
            print(f" {name}={np.mean(np.abs(grad)):.2e}", end='')
        print()

    # Recommendation
    print("\n" + "=" * 70)
    print("FINDINGS:")
    print("=" * 70)
    print("""
1. Poisson NLL gradient diverges when I_pred << I_obs
   → Optimization can get stuck in local minima

2. Poisson NLL assumes count data, not normalized probabilities
   → Deviance formula is inappropriate for [0,1] range data

3. WLS with weight=I naturally handles heteroscedasticity
   → High intensity = low relative noise = high weight
   → Works well for both Gaussian and Poisson noise

RECOMMENDATION:
- Use WLS (use_wls=True) for Poisson noise handling
- Poisson NLL needs reformulation for normalized data
    """)

    print("=" * 70)


if __name__ == "__main__":
    main()
