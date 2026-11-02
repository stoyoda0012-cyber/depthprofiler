"""
MLX Training Loop for Depth Profile MLP
========================================

Two training modes:
1. train_model() — from Parquet (Dstk simulator output)
2. train_from_h5() — augmented from H5 benchmark (TransMat-based)
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from .model import DepthMLP, create_model
from .data import ParquetDataset


def mse_loss(model: DepthMLP, I_obs: mx.array, D_true: mx.array) -> mx.array:
    """Mean squared error on depth profiles."""
    D_pred = model(I_obs)
    return mx.mean((D_pred - D_true) ** 2)


def physics_loss(model: DepthMLP, I_obs: mx.array, D_true: mx.array,
                 n_depth: int = 20, n_elem: int = 6,
                 lambda_sum: float = 0.1) -> mx.array:
    """MSE + sum constraint penalty."""
    D_pred = model(I_obs)
    mse = mx.mean((D_pred - D_true) ** 2)
    D_reshaped = D_pred.reshape(-1, n_depth, n_elem)
    sum_per_depth = mx.sum(D_reshaped, axis=2)
    violation = mx.mean(mx.maximum(sum_per_depth - 1.0, 0.0) ** 2)
    return mse + lambda_sum * violation


def mse_w1_loss(model: DepthMLP, I_obs: mx.array, D_true: mx.array,
                n_depth: int = 20, n_elem: int = 6,
                lambda_w1: float = 1.0,
                step_size: float = 1.0) -> mx.array:
    """MSE + W1 CDF loss for improved spatial positioning.

    W1 distance = integral |CDF_pred - CDF_true| dz.
    For discrete profiles: W1 = sum(|cumsum(pred - true)| * step_size).

    The CDF loss encourages anti-correlated depth errors (adjacent depths
    cancel in the cumulative sum). With an MSE-only loss, positively
    correlated errors accumulate in the CDF and inflate W1.

    Parameters
    ----------
    lambda_w1 : float
        Weight for W1 CDF loss term (default: 1.0).
    step_size : float
        Depth step size in Angstroms for W1 scaling (default: 1.0).
        For uniform grid with n_depth=20, z_max=200: step_size=10.0.
    """
    D_pred = model(I_obs)
    mse = mx.mean((D_pred - D_true) ** 2)

    # Reshape to (batch, depth, elem) for cumsum along depth axis
    pred_3d = D_pred.reshape(-1, n_depth, n_elem)
    true_3d = D_true.reshape(-1, n_depth, n_elem)

    # CDF difference: cumsum of (pred - true) along depth
    diff_cumsum = mx.cumsum(pred_3d - true_3d, axis=1)

    # W1 = mean |CDF_pred - CDF_true| * step_size
    w1 = mx.mean(mx.abs(diff_cumsum)) * step_size

    return mse + lambda_w1 * w1


def _run_training_loop(
    model: DepthMLP,
    I_train: np.ndarray,
    D_train: np.ndarray,
    I_val: np.ndarray,
    D_val: np.ndarray,
    output_path: str,
    n_epochs: int = 50,
    batch_size: int = 8192,
    lr: float = 1e-3,
    patience: int = 10,
    verbose: bool = True,
    loss_fn=None,
    shuffle_seed: int = 42,
) -> dict:
    """Core training loop shared by both training modes.

    Parameters
    ----------
    loss_fn : callable, optional
        Loss function with signature (model, I_obs, D_true) -> scalar.
        If None, defaults to mse_loss.
    shuffle_seed : int
        Seed for the per-epoch minibatch shuffle order. Vary across ensemble
        members to decorrelate their training trajectories.
    """
    if loss_fn is None:
        loss_fn = mse_loss
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    n_train = len(I_train)
    steps_per_epoch = max(1, n_train // batch_size)
    total_steps = max(1, n_epochs * steps_per_epoch)
    schedule = optim.cosine_decay(lr, total_steps)
    optimizer = optim.Adam(learning_rate=schedule)

    I_val_mx = mx.array(I_val)
    D_val_mx = mx.array(D_val)

    rng = np.random.default_rng(42)
    best_val = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    t_start = time.perf_counter()

    for epoch in range(n_epochs):
        t_ep = time.perf_counter()
        perm = rng.permutation(n_train)
        train_loss_sum, n_batches = 0, 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            loss, grads = loss_and_grad(
                model, mx.array(I_train[idx]), mx.array(D_train[idx]))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            train_loss_sum += loss.item()
            n_batches += 1

        val_loss = loss_fn(model, I_val_mx, D_val_mx)
        mx.eval(val_loss)
        vl = val_loss.item()

        train_loss = train_loss_sum / n_batches
        history['train_loss'].append(train_loss)
        history['val_loss'].append(vl)

        t_ep_elapsed = time.perf_counter() - t_ep

        if verbose:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  "
                  f"train={train_loss:.6f}  val={vl:.6f}  "
                  f"({t_ep_elapsed:.1f}s)", end='')

        if vl < best_val:
            best_val = vl
            best_epoch = epoch + 1
            patience_counter = 0
            weights = dict(mlx.utils.tree_flatten(model.parameters()))
            mx.savez(output_path, **weights)
            if verbose:
                print("  *best*", end='')
        else:
            patience_counter += 1

        if verbose:
            print()

        if patience_counter >= patience:
            if verbose:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(best={best_epoch})")
            break

    t_total = time.perf_counter() - t_start

    # Load best weights
    weights = mx.load(output_path)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val
    history['elapsed'] = t_total
    history['n_parameters'] = model.n_parameters()

    if verbose:
        print(f"\n  Training complete: {t_total:.1f}s")
        print(f"  Best epoch:     {best_epoch}")
        print(f"  Best val loss:  {best_val:.6f}")
        print(f"  Model saved:    {output_path}")

    return history


def train_model(
    train_path: str,
    val_path: str,
    output_path: str = 'depth_mlp.npz',
    input_dim: int = 84,
    output_dim: int = 120,
    hidden_dim: int = 256,
    n_hidden: int = 3,
    n_epochs: int = 50,
    batch_size: int = 8192,
    lr: float = 1e-3,
    patience: int = 5,
    use_physics_loss: bool = True,
    lambda_sum: float = 0.1,
    n_depth: int = 20,
    n_elem: int = 6,
    verbose: bool = True,
) -> dict:
    """Train from Parquet files (Dstk simulator output)."""
    if verbose:
        print(f"DepthMLP Training (Parquet mode)")
        print(f"  Architecture:  {input_dim} -> {hidden_dim}x{n_hidden} -> {output_dim}")
        print()

    train_loader = ParquetDataset(train_path, batch_size=batch_size, shuffle=False)
    val_loader = ParquetDataset(val_path, batch_size=batch_size, shuffle=False)

    assert train_loader.input_dim == input_dim
    assert train_loader.output_dim == output_dim

    model = DepthMLP(input_dim, output_dim, hidden_dim, n_hidden)
    mx.eval(model.parameters())

    if verbose:
        print(f"  Train: {train_loader.n_samples:,}, Val: {val_loader.n_samples:,}")
        print(f"  Parameters: {model.n_parameters():,}")
        print()

    return _run_training_loop(
        model, train_loader.I_obs, train_loader.D_true,
        val_loader.I_obs, val_loader.D_true,
        output_path, n_epochs, batch_size, lr, patience, verbose,
    )


def _augment_profiles(
    Dstk: np.ndarray,
    n_aug: int,
    rng: np.random.Generator,
    sum_perturb: bool = True,
) -> np.ndarray:
    """Augment depth profiles via convex mixtures, noise, shifts, and sharpening.

    Fully vectorized — no Python loops.

    Modes (uniform random per sample):
        0: Convex mixture of two random profiles
        1: Profile + Gaussian noise
        2: Depth-shifted profile (roll by 1–3 bins)
        3: Sharpened profile (power > 1 → step-like interfaces)
        4: Broadened profile (power < 1 → smoother interfaces)

    When sum_perturb=True, ~20% of augmented samples get sum(D)
    perturbed to U[0.95, 1.05] to improve robustness to imperfect
    normalization in real data.

    Note: RNG call order differs from the original loop version, so
    results are statistically equivalent but not bitwise identical.
    """
    n_px, n_depth, n_elem = Dstk.shape
    Dstk_f32 = Dstk.astype(np.float32)

    # Pre-generate mode assignments
    modes = rng.integers(0, 5, size=n_aug)
    mask0 = modes == 0
    mask1 = modes == 1
    mask2 = modes == 2
    mask3 = modes == 3
    mask4 = modes == 4
    n0, n1, n2, n3, n4 = (
        int(mask0.sum()), int(mask1.sum()), int(mask2.sum()),
        int(mask3.sum()), int(mask4.sum()),
    )

    D_aug = np.empty((n_aug, n_depth, n_elem), dtype=np.float32)

    # Mode 0: Convex mixture of two random profiles
    if n0 > 0:
        idx_a = rng.integers(n_px, size=n0)
        idx_b = rng.integers(n_px, size=n0)
        alpha = rng.uniform(0.1, 0.9, size=(n0, 1, 1)).astype(np.float32)
        D_aug[mask0] = alpha * Dstk_f32[idx_a] + (1 - alpha) * Dstk_f32[idx_b]

    # Mode 1: Profile + Gaussian noise
    if n1 > 0:
        idx = rng.integers(n_px, size=n1)
        noise = rng.normal(0, 0.02, size=(n1, n_depth, n_elem)).astype(np.float32)
        D_aug[mask1] = np.maximum(Dstk_f32[idx] + noise, 0)

    # Mode 2: Depth-shifted profile (roll by 1, 2, or 3 bins)
    if n2 > 0:
        idx = rng.integers(n_px, size=n2)
        shifts = rng.integers(1, 4, size=n2)
        base = Dstk_f32[idx]
        where2 = np.where(mask2)[0]
        for s in (1, 2, 3):
            s_mask = shifts == s
            if s_mask.any():
                D_aug[where2[s_mask]] = np.roll(base[s_mask], s, axis=1)

    # Mode 3: Sharpen (power ∈ [1.5, 3.0])
    if n3 > 0:
        idx = rng.integers(n_px, size=n3)
        powers = rng.uniform(1.5, 3.0, size=(n3, 1, 1)).astype(np.float32)
        D_aug[mask3] = np.power(np.maximum(Dstk_f32[idx], 0), powers)

    # Mode 4: Broaden (power ∈ [0.3, 0.8])
    if n4 > 0:
        idx = rng.integers(n_px, size=n4)
        powers = rng.uniform(0.3, 0.8, size=(n4, 1, 1)).astype(np.float32)
        D_aug[mask4] = np.power(np.maximum(Dstk_f32[idx], 0), powers)

    # Normalize: sum(elements) = 1.0 at each depth
    total = D_aug.sum(axis=2, keepdims=True)
    total[total == 0] = 1
    D_aug /= total

    # Charge balance perturbation: ~20% of samples
    if sum_perturb:
        perturb_mask = rng.random(n_aug) < 0.2
        n_perturb = int(perturb_mask.sum())
        if n_perturb > 0:
            scales = rng.uniform(
                0.95, 1.05, size=(n_perturb, 1, 1),
            ).astype(np.float32)
            D_aug[perturb_mask] *= scales

    return D_aug


def train_from_h5(
    h5_path: str,
    output_path: str = 'depth_mlp.npz',
    hidden_dim: int = 256,
    n_hidden: int = 3,
    n_aug: int = 500_000,
    n_epochs: int = 50,
    batch_size: int = 8192,
    lr: float = 1e-3,
    patience: int = 10,
    val_fraction: float = 0.1,
    use_w1_loss: bool = False,
    lambda_w1: float = 1.0,
    noise_augment: float = 0.0,
    arch: str = 'mlp',
    verbose: bool = True,
) -> dict:
    """Train from H5 benchmark data with augmentation.

    Uses H5's TransferMatrix to generate I_obs from augmented D profiles.
    This closes the domain gap between simulator and benchmark data.

    Parameters
    ----------
    h5_path : str
        Path to benchmark H5 (must contain Dstk, Xstk, ezdeprof/TransMatObs).
    n_aug : int
        Number of augmented samples to generate.
    use_w1_loss : bool
        If True, use MSE + W1 CDF loss instead of pure MSE.
    lambda_w1 : float
        Weight for W1 CDF loss term (only used when use_w1_loss=True).
    noise_augment : float
        Fraction of training samples with Poisson-like noise on I_obs (0-1).
        Meant to make the model less sensitive to measurement noise.
    arch : str
        Model architecture: 'mlp', 'resmlp', 'conv1d', 'elemnet'.
    """
    import h5py

    if verbose:
        print("DepthMLP Training (H5 augmentation mode)")
        print(f"  H5: {h5_path}")
        print()

    # Load H5 data
    with h5py.File(h5_path, 'r') as f:
        Dstk = np.transpose(f['Dstk'][:].astype(np.float64), (0, 2, 1))
        tm = np.transpose(
            f['ezdeprof/TransMatObs'][:].astype(np.float64), (2, 1, 0))

    n_pixels, n_depth, n_elem = Dstk.shape
    n_angles = tm.shape[0]
    input_dim = n_angles * n_elem
    output_dim = n_depth * n_elem

    if verbose:
        print(f"  Source pixels: {n_pixels}")
        print(f"  Augmented samples: {n_aug:,}")
        print(f"  Dims: I={input_dim} ({n_angles}×{n_elem}), "
              f"D={output_dim} ({n_depth}×{n_elem})")
        print()

    # Generate augmented profiles
    rng = np.random.default_rng(42)

    if verbose:
        print("  Augmenting profiles...", end='', flush=True)
    t0 = time.perf_counter()
    D_aug = _augment_profiles(Dstk, n_aug, rng)
    t1 = time.perf_counter()
    if verbose:
        print(f" {t1-t0:.1f}s")

    # Combine original + augmented
    D_all = np.concatenate(
        [Dstk.astype(np.float32), D_aug], axis=0)

    # Forward model: I = TransMat @ D
    if verbose:
        print("  Computing I_obs...", end='', flush=True)
    t0 = time.perf_counter()
    I_all = np.zeros(
        (D_all.shape[0], n_angles, n_elem), dtype=np.float32)
    for e in range(n_elem):
        I_all[:, :, e] = D_all[:, :, e] @ tm[:, :, e].T
    I_sum = I_all.sum(axis=2, keepdims=True)
    I_sum[I_sum == 0] = 1
    I_all /= I_sum
    t1 = time.perf_counter()
    if verbose:
        print(f" {t1-t0:.1f}s")

    # Optional noise augmentation on I_obs (e.g. 0.3 = 30% of samples noised)
    if noise_augment > 0:
        n_noisy = int(len(I_all) * noise_augment)
        noise_idx = rng.choice(len(I_all), n_noisy, replace=False)
        noise_levels = rng.uniform(0.01, 0.05, size=n_noisy)
        for i, idx in enumerate(noise_idx):
            sigma = noise_levels[i]
            noise = rng.normal(0, sigma, size=I_all[idx].shape).astype(np.float32)
            I_all[idx] = np.maximum(I_all[idx] + noise, 0)
            # Re-normalize
            row_sum = I_all[idx].sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1
            I_all[idx] /= row_sum
        if verbose:
            print(f"  Noise augment: {noise_augment:.0%} of samples, σ~U(1%,5%)")

    # Flatten
    I_flat = I_all.reshape(D_all.shape[0], -1)
    D_flat = D_all.reshape(D_all.shape[0], -1)

    # Train/val split
    n_total = D_all.shape[0]
    n_val = int(n_total * val_fraction)
    perm = rng.permutation(n_total)
    I_train = I_flat[perm[:-n_val]]
    D_train = D_flat[perm[:-n_val]]
    I_val = I_flat[perm[-n_val:]]
    D_val = D_flat[perm[-n_val:]]

    if verbose:
        print(f"  Train: {len(I_train):,}, Val: {len(I_val):,}")

    # Create model
    extra_kwargs = {}
    if arch == 'conv1d':
        extra_kwargs = dict(n_angles=n_angles, n_elem=n_elem)
    elif arch == 'elemnet':
        extra_kwargs = dict(n_depth=n_depth, n_elem=n_elem)
    elif arch == 'transformer':
        extra_kwargs = dict(n_angles=n_angles, n_elem=n_elem)
    model = create_model(arch, input_dim, output_dim, hidden_dim, n_hidden,
                         **extra_kwargs)
    mx.eval(model.parameters())

    if verbose:
        print(f"  Arch: {arch} ({model.__class__.__name__})")
        print(f"  Parameters: {model.n_parameters():,}")

    # Build loss function
    loss_fn = None
    if use_w1_loss:
        # Compute step_size from H5 depth grid
        import h5py as _h5
        with _h5.File(h5_path, 'r') as _f:
            if 'ezdeprof/depth' in _f:
                _depth = _f['ezdeprof/depth'][:].flatten()
                _step = float(np.mean(np.diff(_depth)))
            else:
                # Fallback: estimate from totaldepth/n_depth
                _td = float(_f['ezdeprof/totaldepth'][()]) if 'ezdeprof/totaldepth' in _f else 200.0
                _step = _td / n_depth

        if verbose:
            print(f"  W1 CDF loss: lambda_w1={lambda_w1}, step_size={_step:.1f}Å")

        def loss_fn(model, I_obs, D_true,
                    _n_depth=n_depth, _n_elem=n_elem,
                    _lambda_w1=lambda_w1, _step=_step):
            return mse_w1_loss(model, I_obs, D_true,
                               n_depth=_n_depth, n_elem=_n_elem,
                               lambda_w1=_lambda_w1, step_size=_step)

    if verbose:
        print()

    return _run_training_loop(
        model, I_train, D_train, I_val, D_val,
        output_path, n_epochs, batch_size, lr, patience, verbose,
        loss_fn=loss_fn,
    )


def _prepare_d2s2d_data(
    n_aug: int = 200_000,
    grid_sizes: tuple = ((50, 50), (100, 100)),
    dt_values: tuple = (1, 5, 20),
    val_fraction: float = 0.1,
    noise_augment: float = 0.0,
    preset: str = '7elem_checkered',
    mix_dist_types: bool = True,
    step_size: float = None,
    total_depth: float = None,
    thickness: float = None,
    verbose: bool = True,
) -> dict:
    """Generate and prepare training data from D2S2D pipeline.

    Returns dict with I_train, D_train, I_val, D_val, and metadata.
    Shared across multiple training runs in auto_tune().

    Parameters
    ----------
    noise_augment : float
        If > 0, add Poisson-like noise to I_obs for noise robustness training.
        The fraction of augmented samples that get noise added.
    preset : str
        Simulation preset name ('7elem_checkered', '6elem_standard', etc.).
        Must match a key in PRESETS dict from d2s2d.py.
    mix_dist_types : bool
        If True, generate training data with both 'Gaussian' and 'Error fcn.'
        profile types (50/50 split). This reduces domain gap when evaluating
        on H5 data that uses erf-based profiles.
    step_size : float or None
        Override depth step size (Å). If None, use preset default (30.0).
        Must match target H5's ``ezdeprof/stepsize`` field (NOT totaldepth/(slice-1)).
    total_depth : float or None
        Override total depth (Å). If None, use preset default (570.0).
    thickness : float or None
        Override base film thickness (Å) in SimulationConfig.
        Adjust to match the thickness distribution of the target data.
    """
    from dataclasses import replace as dc_replace
    from deppro.simulate.d2s2d import generate_dstk_xstk, D2S2DConfig, PRESETS
    from deppro.transfer_matrix import TransferMatrix  # forward model (NOT the classical solver)
    from deppro import CompoundDatabase

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")

    base_sim_config = PRESETS[preset]

    if verbose:
        n_elem = len(base_sim_config.elements)
        print(f"  Preset: {preset} ({n_elem} elements)")
        print(f"  Data generation: grids={grid_sizes}, dt={dt_values}")
        if mix_dist_types:
            print(f"  Profile types: Gaussian + Error fcn. (mixed)")

    # Determine dist_types to use
    dist_types = ['Gaussian', 'Error fcn.'] if mix_dist_types else [base_sim_config.dist_type]

    # Apply overrides if specified
    overrides = {}
    if step_size is not None:
        overrides['step_size'] = step_size
    if total_depth is not None:
        overrides['total_depth'] = total_depth
    if thickness is not None:
        overrides['thickness'] = thickness
    if overrides:
        base_sim_config = dc_replace(base_sim_config, **overrides)
        if verbose:
            if step_size is not None or total_depth is not None:
                print(f"  Depth grid override: step={base_sim_config.step_size:.4f}Å, "
                      f"total={base_sim_config.total_depth:.1f}Å")
            if thickness is not None:
                print(f"  Thickness override: {thickness:.1f}Å")

    # Generate diverse training data from D2S2D
    all_Dstk = []
    all_Xstk = []

    t0 = time.perf_counter()
    for dist_type in dist_types:
        for nx, ny in grid_sizes:
            for dt in dt_values:
                sim_config = dc_replace(
                    base_sim_config, nx=nx, ny=ny, dt=dt, n_phases=1,
                    dist_type=dist_type,
                )
                result = generate_dstk_xstk(sim_config, verbose=False)
                all_Dstk.append(result['Dstk'])
                all_Xstk.append(result['Xstk'])
                if verbose:
                    n = result['Dstk'].shape[0]
                    tag = 'G' if dist_type == 'Gaussian' else 'E'
                    print(f"    {nx}x{ny} dt={dt} [{tag}]: {n} px")

    Dstk = np.concatenate(all_Dstk, axis=0)
    Xstk = np.concatenate(all_Xstk, axis=0)
    t_gen = time.perf_counter() - t0

    n_pixels, n_depth, n_elem = Dstk.shape
    n_angles = Xstk.shape[1]
    input_dim = n_angles * n_elem
    output_dim = n_depth * n_elem
    step_size = float(result['depth'][1] - result['depth'][0])

    if verbose:
        print(f"  Source: {n_pixels:,} px ({t_gen:.1f}s)")

    # Augment
    rng = np.random.default_rng(42)
    D_aug = _augment_profiles(Dstk, n_aug, rng)
    D_all = np.concatenate([Dstk.astype(np.float32), D_aug], axis=0)

    # Forward model: compute I_obs via TransferMatrix
    depth = result['depth']
    angles = result['angles']
    element_configs = result['element_configs']
    compound_db = CompoundDatabase()

    D_uniform = np.ones((n_depth, n_elem)) / n_elem
    tm = TransferMatrix(
        depth=depth, step_size=step_size, angles=angles,
        photon_energy=base_sim_config.photon_energy,
        work_function=base_sim_config.work_function,
        gamma_eff=base_sim_config.gamma_eff,
    )
    tm.compute(D_uniform, element_configs, compound_db)

    T = tm.trans_mat  # (n_angles, n_depth, n_elem)
    I_all = np.zeros((D_all.shape[0], n_angles, n_elem), dtype=np.float32)
    for e in range(n_elem):
        I_all[:, :, e] = D_all[:, :, e] @ T[:, :, e].T
    I_sum = I_all.sum(axis=2, keepdims=True)
    I_sum[I_sum == 0] = 1
    I_all /= I_sum

    # Optional noise augmentation on I_obs
    if noise_augment > 0:
        n_noisy = int(len(I_all) * noise_augment)
        noise_idx = rng.choice(len(I_all), n_noisy, replace=False)
        noise_levels = rng.uniform(0.01, 0.05, size=n_noisy)
        for i, idx in enumerate(noise_idx):
            sigma = noise_levels[i]
            noise = rng.normal(0, sigma, size=I_all[idx].shape).astype(np.float32)
            I_all[idx] = np.maximum(I_all[idx] + noise, 0)
            # Re-normalize
            row_sum = I_all[idx].sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1
            I_all[idx] /= row_sum

    # Flatten + split
    I_flat = I_all.reshape(D_all.shape[0], -1)
    D_flat = D_all.reshape(D_all.shape[0], -1)

    n_total = D_all.shape[0]
    n_val = int(n_total * val_fraction)
    perm = rng.permutation(n_total)

    if verbose:
        print(f"  Total: {n_total:,} (train={n_total-n_val:,}, val={n_val:,})")
        if noise_augment > 0:
            print(f"  Noise augment: {noise_augment:.0%} of samples, σ~U(1%,5%)")

    return {
        'I_train': I_flat[perm[:-n_val]],
        'D_train': D_flat[perm[:-n_val]],
        'I_val': I_flat[perm[-n_val:]],
        'D_val': D_flat[perm[-n_val:]],
        'input_dim': input_dim,
        'output_dim': output_dim,
        'n_depth': n_depth,
        'n_elem': n_elem,
        'step_size': step_size,
        'preset': preset,
    }


def train_from_d2s2d(
    output_path: str = 'depth_mlp.npz',
    hidden_dim: int = 256,
    n_hidden: int = 3,
    n_aug: int = 200_000,
    n_epochs: int = 50,
    batch_size: int = 8192,
    lr: float = 1e-3,
    patience: int = 10,
    val_fraction: float = 0.1,
    use_w1_loss: bool = False,
    lambda_w1: float = 1.0,
    grid_sizes: tuple = ((50, 50), (100, 100)),
    dt_values: tuple = (1, 5, 20),
    preset: str = '7elem_checkered',
    mix_dist_types: bool = True,
    noise_augment: float = 0.0,
    step_size: float = None,
    total_depth: float = None,
    thickness: float = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    prepared_data: Optional[dict] = None,
) -> dict:
    """Train from D2S2D-generated data with augmentation.

    Parameters
    ----------
    preset : str
        Simulation preset ('7elem_checkered', '6elem_standard', etc.).
    mix_dist_types : bool
        If True, generate both Gaussian and erf profile types.
    noise_augment : float
        Fraction of samples with noise augmentation on I_obs.
    thickness : float or None
        Override base film thickness (Å). Adjust to match H5 data distribution.
    seed : int, optional
        Reproducibility / deep-ensemble seed (weight init + shuffle). See
        ``_train_on_prepared_data``. None = legacy global-RNG init.
    prepared_data : dict, optional
        Reuse data from a previous ``_prepare_d2s2d_data`` call (e.g. to train
        an ensemble on shared data, varying only ``seed``). Skips generation.
    """
    if verbose:
        print(f"DepthMLP Training (D2S2D mode, preset={preset})")

    data = prepared_data if prepared_data is not None else _prepare_d2s2d_data(
        n_aug=n_aug, grid_sizes=grid_sizes, dt_values=dt_values,
        val_fraction=val_fraction, noise_augment=noise_augment,
        preset=preset, mix_dist_types=mix_dist_types,
        step_size=step_size, total_depth=total_depth,
        thickness=thickness, verbose=verbose,
    )

    return _train_on_prepared_data(
        data, output_path=output_path,
        hidden_dim=hidden_dim, n_hidden=n_hidden,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        patience=patience, use_w1_loss=use_w1_loss,
        lambda_w1=lambda_w1, verbose=verbose, seed=seed,
    )


def _train_on_prepared_data(
    data: dict,
    output_path: str = 'depth_mlp.npz',
    hidden_dim: int = 256,
    n_hidden: int = 3,
    n_epochs: int = 50,
    batch_size: int = 8192,
    lr: float = 1e-3,
    patience: int = 10,
    use_w1_loss: bool = False,
    lambda_w1: float = 1.0,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> dict:
    """Train on pre-prepared data (from _prepare_d2s2d_data).

    Parameters
    ----------
    seed : int, optional
        If given, seeds MLX (weight init) and the minibatch shuffle so the run
        is reproducible. Vary across deep-ensemble members for decorrelated
        weights + shuffle order. If None, weight init uses the global MLX RNG
        (legacy behavior, unchanged).
    """
    input_dim = data['input_dim']
    output_dim = data['output_dim']
    n_depth = data['n_depth']
    n_elem = data['n_elem']
    step_size = data['step_size']

    if seed is not None:
        mx.random.seed(seed)
    model = DepthMLP(input_dim, output_dim, hidden_dim, n_hidden)
    mx.eval(model.parameters())

    if verbose:
        print(f"  Model: {input_dim}->{hidden_dim}x{n_hidden}->{output_dim} "
              f"({model.n_parameters():,} params)")

    # Build loss function
    loss_fn = None
    if use_w1_loss:
        if verbose:
            print(f"  Loss: MSE + W1 (λ={lambda_w1}, step={step_size:.0f}Å)")

        def loss_fn(model, I_obs, D_true,
                    _n_depth=n_depth, _n_elem=n_elem,
                    _lambda_w1=lambda_w1, _step=step_size):
            return mse_w1_loss(model, I_obs, D_true,
                               n_depth=_n_depth, n_elem=_n_elem,
                               lambda_w1=_lambda_w1, step_size=_step)

    if verbose:
        print()

    return _run_training_loop(
        model, data['I_train'], data['D_train'],
        data['I_val'], data['D_val'],
        output_path, n_epochs, batch_size, lr, patience, verbose,
        loss_fn=loss_fn,
        shuffle_seed=seed if seed is not None else 42,
    )


def auto_tune(
    output_dir: Optional[str] = None,
    n_aug: int = 200_000,
    n_epochs: int = 50,
    batch_size: int = 4096,
    lr: float = 1e-3,
    patience: int = 10,
    eval_nx: int = 10,
    eval_ny: int = 10,
    preset: str = '7elem_checkered',
    mix_dist_types: bool = True,
    verbose: bool = True,
) -> list:
    """Automated hyperparameter sweep: architecture, loss, noise augmentation.

    Sweep axes:
    - 6a: Architecture (hidden_dim × n_hidden)
    - 6b: Loss weighting (lambda_w1)
    - 6c: Noise augmentation (fraction of training data with noise)

    All sweeps share the same base training data (generated once).

    Parameters
    ----------
    preset : str
        Simulation preset ('7elem_checkered', '6elem_standard', etc.).
    mix_dist_types : bool
        If True, generate both Gaussian and erf profile types in training data.

    Returns list of dicts with config + metrics for each experiment.
    """
    from deppro.simulate.d2s2d import run_d2s2d, D2S2DConfig, PRESETS
    from deppro.inference.postprocess import enforce_constraints
    import os
    from dataclasses import replace as dc_replace

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")

    base_sim_config = PRESETS[preset]

    if output_dir is None:
        from deppro.paths import outputs_dir
        output_dir = str(outputs_dir())
    os.makedirs(output_dir, exist_ok=True)

    results = []

    if verbose:
        n_elem = len(base_sim_config.elements)
        print(f"auto_tune: preset={preset} ({n_elem} elements)")

    def _evaluate(model_path, label):
        """Evaluate a model on noise-free and noisy D2S2D data."""
        metrics = {'label': label, 'model_path': model_path}
        eval_sim = dc_replace(base_sim_config, nx=eval_nx, ny=eval_ny, dt=1, n_phases=1)
        for noise, tag in [(0.0, 'nf'), (0.03, 'n03')]:
            cfg = D2S2DConfig(
                sim_config=eval_sim,
                inverse_method='nn', nn_model_path=model_path,
                noise_level=noise)
            r = run_d2s2d(cfg, verbose=False)
            metrics[f'rmse_{tag}'] = r.rmse_total
            metrics[f'w1_{tag}'] = r.w1_mean
        return metrics

    def _print_row(m):
        print(f"  {m['label']:<28} "
              f"{m['rmse_nf']:>7.4f} {m['w1_nf']:>7.1f}  "
              f"{m['rmse_n03']:>7.4f} {m['w1_n03']:>7.1f}")

    # ========================================================
    # Step 6a: Architecture sweep
    # ========================================================
    if verbose:
        print("=" * 70)
        print("Step 6a: Architecture Sweep")
        print("=" * 70)

    data_clean = _prepare_d2s2d_data(
        n_aug=n_aug, preset=preset, mix_dist_types=mix_dist_types,
        verbose=verbose)

    arch_configs = [
        (32, 2), (32, 3),
        (64, 2), (64, 3),
        (128, 2), (128, 3), (128, 4),
        (256, 2), (256, 3), (256, 4),
        (512, 3),
    ]

    if verbose:
        print(f"\n  {'Config':<28} {'NF RMSE':>7} {'NF W1':>7}  "
              f"{'N3 RMSE':>7} {'N3 W1':>7}")
        print("  " + "-" * 64)

    for hidden_dim, n_hidden in arch_configs:
        tag = f"h{hidden_dim}_l{n_hidden}"
        path = os.path.join(output_dir, f"tune_{tag}.npz")
        history = _train_on_prepared_data(
            data_clean, output_path=path,
            hidden_dim=hidden_dim, n_hidden=n_hidden,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            patience=patience, verbose=False,
        )
        m = _evaluate(path, f"h={hidden_dim} l={n_hidden}")
        m['hidden_dim'] = hidden_dim
        m['n_hidden'] = n_hidden
        m['n_params'] = history['n_parameters']
        m['val_loss'] = history['best_val_loss']
        m['sweep'] = 'arch'
        results.append(m)
        if verbose:
            _print_row(m)

    # Find best architecture (by NF RMSE)
    arch_results = [r for r in results if r['sweep'] == 'arch']
    best_arch = min(arch_results, key=lambda r: r['rmse_nf'])
    best_hidden = best_arch['hidden_dim']
    best_layers = best_arch['n_hidden']
    if verbose:
        print(f"\n  Best arch: h={best_hidden} l={best_layers} "
              f"(RMSE={best_arch['rmse_nf']:.4f}, W1={best_arch['w1_nf']:.1f})")

    # ========================================================
    # Step 6b: Loss weighting sweep (using best architecture)
    # ========================================================
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Step 6b: Loss Weighting Sweep (h={best_hidden}, l={best_layers})")
        print("=" * 70)
        print(f"  {'Config':<28} {'NF RMSE':>7} {'NF W1':>7}  "
              f"{'N3 RMSE':>7} {'N3 W1':>7}")
        print("  " + "-" * 64)

    lambda_values = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    for lam in lambda_values:
        use_w1 = lam > 0
        tag = f"lam{lam:.1f}"
        path = os.path.join(output_dir, f"tune_{tag}.npz")
        history = _train_on_prepared_data(
            data_clean, output_path=path,
            hidden_dim=best_hidden, n_hidden=best_layers,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            patience=patience, use_w1_loss=use_w1,
            lambda_w1=lam, verbose=False,
        )
        m = _evaluate(path, f"λ_w1={lam:.1f}")
        m['lambda_w1'] = lam
        m['val_loss'] = history['best_val_loss']
        m['sweep'] = 'loss'
        results.append(m)
        if verbose:
            _print_row(m)

    # Find Pareto-optimal lambda (best W1 with RMSE < best_RMSE * 1.3)
    loss_results = [r for r in results if r['sweep'] == 'loss']
    best_mse_rmse = min(r['rmse_nf'] for r in loss_results)
    pareto_candidates = [r for r in loss_results
                         if r['rmse_nf'] < best_mse_rmse * 1.3]
    best_loss = min(pareto_candidates, key=lambda r: r['w1_nf'])
    best_lambda = best_loss.get('lambda_w1', 0.0)
    if verbose:
        print(f"\n  Best λ_w1: {best_lambda:.1f} "
              f"(RMSE={best_loss['rmse_nf']:.4f}, W1={best_loss['w1_nf']:.1f})")

    # ========================================================
    # Step 6c: Noise augmentation sweep
    # ========================================================
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Step 6c: Noise Augmentation Sweep "
              f"(h={best_hidden}, l={best_layers}, λ={best_lambda:.1f})")
        print("=" * 70)
        print(f"  {'Config':<28} {'NF RMSE':>7} {'NF W1':>7}  "
              f"{'N3 RMSE':>7} {'N3 W1':>7}")
        print("  " + "-" * 64)

    noise_fracs = [0.0, 0.3, 0.5, 0.7, 1.0]
    use_w1 = best_lambda > 0

    for frac in noise_fracs:
        tag = f"noise{frac:.1f}"
        path = os.path.join(output_dir, f"tune_{tag}.npz")
        data_noisy = _prepare_d2s2d_data(
            n_aug=n_aug, noise_augment=frac, preset=preset,
            mix_dist_types=mix_dist_types, verbose=False)
        history = _train_on_prepared_data(
            data_noisy, output_path=path,
            hidden_dim=best_hidden, n_hidden=best_layers,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            patience=patience, use_w1_loss=use_w1,
            lambda_w1=best_lambda, verbose=False,
        )
        m = _evaluate(path, f"noise_aug={frac:.0%}")
        m['noise_augment'] = frac
        m['val_loss'] = history['best_val_loss']
        m['sweep'] = 'noise'
        results.append(m)
        if verbose:
            _print_row(m)

    # ========================================================
    # Summary: train final best model
    # ========================================================
    noise_results = [r for r in results if r['sweep'] == 'noise']
    # Best noisy RMSE with NF RMSE < 30% degradation
    best_nf = min(r['rmse_nf'] for r in noise_results)
    noise_candidates = [r for r in noise_results
                        if r['rmse_nf'] < best_nf * 1.3]
    best_noise = min(noise_candidates, key=lambda r: r['rmse_n03'])
    best_noise_frac = best_noise.get('noise_augment', 0.0)

    if verbose:
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print("=" * 70)
        print(f"  Best architecture: h={best_hidden}, l={best_layers}")
        print(f"  Best λ_w1:         {best_lambda:.1f}")
        print(f"  Best noise_aug:    {best_noise_frac:.0%}")
        print(f"\n  Training final model...")

    # Train final model with best settings
    # Derive element count tag from preset (e.g., '7elem_checkered' -> '7elem')
    elem_tag = preset.split('_')[0] if '_' in preset else preset
    final_path = os.path.join(output_dir, f"depth_mlp_{elem_tag}_tuned.npz")
    data_final = _prepare_d2s2d_data(
        n_aug=n_aug, noise_augment=best_noise_frac, preset=preset,
        mix_dist_types=mix_dist_types, verbose=False)
    _train_on_prepared_data(
        data_final, output_path=final_path,
        hidden_dim=best_hidden, n_hidden=best_layers,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        patience=patience, use_w1_loss=use_w1,
        lambda_w1=best_lambda, verbose=False,
    )
    final_m = _evaluate(final_path, "FINAL")
    results.append({**final_m, 'sweep': 'final',
                    'hidden_dim': best_hidden, 'n_hidden': best_layers,
                    'lambda_w1': best_lambda, 'noise_augment': best_noise_frac})

    if verbose:
        print(f"\n  FINAL: RMSE(NF)={final_m['rmse_nf']:.4f} "
              f"W1(NF)={final_m['w1_nf']:.1f}  "
              f"RMSE(N3)={final_m['rmse_n03']:.4f} "
              f"W1(N3)={final_m['w1_n03']:.1f}")
        print(f"  Saved: {final_path}")

    return results


def _detect_arch(weight_dict: dict) -> str:
    """Auto-detect model architecture from weight key structure."""
    keys = set(weight_dict.keys())
    if any(k.startswith('decoders.') for k in keys):
        return 'elemnet'
    if any(k.startswith('conv1.') for k in keys):
        return 'conv1d'
    if any(k.startswith('enc_layers.') for k in keys):
        return 'transformer'
    if any(k.startswith('blocks.') for k in keys):
        return 'resmlp'
    from .model import _REGISTERED_LOADERS
    for name, prefix, _ in _REGISTERED_LOADERS:
        if any(k.startswith(prefix) for k in keys):
            return name
    return 'mlp'


def _detect_dims_mlp(weight_dict: dict):
    """Auto-detect MLP dimensions from weight shapes."""
    hidden_dim = weight_dict['layers.0.weight'].shape[0]
    n_linear = sum(1 for k, v in weight_dict.items()
                   if 'weight' in k and len(v.shape) == 2)
    n_hidden = n_linear - 1
    input_dim = weight_dict['layers.0.weight'].shape[1]
    last_key = max(
        (k for k in weight_dict if 'weight' in k and len(weight_dict[k].shape) == 2),
        key=lambda k: int(k.split('.')[1]))
    output_dim = weight_dict[last_key].shape[0]
    return input_dim, output_dim, hidden_dim, n_hidden


def _detect_dims_resmlp(weight_dict: dict):
    """Auto-detect ResMLP dimensions."""
    input_dim = weight_dict['proj_in.weight'].shape[1]
    hidden_dim = weight_dict['proj_in.weight'].shape[0]
    output_dim = weight_dict['proj_out.weight'].shape[0]
    n_hidden = sum(1 for k in weight_dict if k.startswith('blocks.') and k.endswith('.linear.weight'))
    return input_dim, output_dim, hidden_dim, n_hidden


def _detect_dims_conv1d(weight_dict: dict):
    """Auto-detect Conv1D dimensions."""
    input_dim = weight_dict['head_in.weight'].shape[1]  # flat_dim, not input_dim
    hidden_dim = weight_dict['head_in.weight'].shape[0]
    output_dim = weight_dict['head_out.weight'].shape[0]
    n_elem = weight_dict['conv1.weight'].shape[2]  # in_channels = n_elem
    n_filters = weight_dict['conv1.weight'].shape[0]
    flat_dim = weight_dict['head_in.weight'].shape[1]
    n_angles = flat_dim // n_filters
    n_blocks = sum(1 for k in weight_dict if k.startswith('head_blocks.') and k.endswith('.linear.weight'))
    n_hidden = n_blocks + 2  # conv layers + head blocks
    real_input_dim = n_angles * n_elem
    return real_input_dim, output_dim, hidden_dim, n_hidden, n_angles, n_elem, n_filters


def _detect_dims_elemnet(weight_dict: dict):
    """Auto-detect ElemNet dimensions."""
    input_dim = weight_dict['enc_in.weight'].shape[1]
    hidden_dim = weight_dict['enc_in.weight'].shape[0]
    n_elem = sum(1 for k in weight_dict if k.startswith('decoders.') and k.endswith('.weight'))
    n_depth = weight_dict['decoders.0.weight'].shape[0]
    output_dim = n_depth * n_elem
    n_blocks = sum(1 for k in weight_dict if k.startswith('enc_blocks.') and k.endswith('.linear.weight'))
    n_hidden = n_blocks + 1  # encoder blocks + 1
    return input_dim, output_dim, hidden_dim, n_hidden, n_depth, n_elem


def _detect_dims_transformer(weight_dict: dict):
    """Auto-detect Transformer dimensions."""
    n_elem = weight_dict['token_proj.weight'].shape[1]
    hidden_dim = weight_dict['token_proj.weight'].shape[0]
    output_dim = weight_dict['head.weight'].shape[0]
    n_angles = weight_dict['pos_embed'].shape[0]
    n_hidden = sum(1 for k in weight_dict
                   if k.startswith('enc_layers.') and k.endswith('.attention.out_proj.weight'))
    input_dim = n_angles * n_elem
    # Detect n_heads from attention key dim
    qkv_weight_key = 'enc_layers.0.attention.query_proj.weight'
    if qkv_weight_key in weight_dict:
        # query_proj.weight shape: (hidden_dim, hidden_dim)
        # n_heads = hidden_dim // head_dim, but we need head_dim
        # Default to 8 heads (32 dims per head for 256)
        n_heads = max(1, hidden_dim // 32)
    else:
        n_heads = 8
    return input_dim, output_dim, hidden_dim, n_hidden, n_angles, n_elem, n_heads


def load_model(
    model_path: str,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    n_hidden: Optional[int] = None,
) -> 'nn.Module':
    """Load a trained model from .npz weights.

    Auto-detects architecture (MLP, ResMLP, Conv1D, ElemNet, Transformer)
    and all dimensions from the weight file.
    """
    weights = mx.load(model_path)
    weight_dict = dict(weights) if not isinstance(weights, dict) else weights

    arch = _detect_arch(weight_dict)

    if arch == 'mlp':
        dims = _detect_dims_mlp(weight_dict)
        in_d, out_d, h_d, n_h = dims
        model = DepthMLP(
            input_dim or in_d, output_dim or out_d,
            hidden_dim or h_d, n_hidden if n_hidden is not None else n_h,
        )
    elif arch == 'resmlp':
        from .model import DepthResMLP
        dims = _detect_dims_resmlp(weight_dict)
        in_d, out_d, h_d, n_h = dims
        model = DepthResMLP(
            input_dim or in_d, output_dim or out_d,
            hidden_dim or h_d, n_hidden if n_hidden is not None else n_h,
        )
    elif arch == 'conv1d':
        from .model import DepthConv1D
        dims = _detect_dims_conv1d(weight_dict)
        in_d, out_d, h_d, n_h, n_ang, n_el, n_filt = dims
        model = DepthConv1D(
            input_dim or in_d, output_dim or out_d,
            hidden_dim or h_d, n_hidden if n_hidden is not None else n_h,
            n_angles=n_ang, n_elem=n_el, n_filters=n_filt,
        )
    elif arch == 'elemnet':
        from .model import DepthElemNet
        dims = _detect_dims_elemnet(weight_dict)
        in_d, out_d, h_d, n_h, n_dep, n_el = dims
        model = DepthElemNet(
            input_dim or in_d, output_dim or out_d,
            hidden_dim or h_d, n_hidden if n_hidden is not None else n_h,
            n_depth=n_dep, n_elem=n_el,
        )
    elif arch == 'transformer':
        from .model import DepthTransformer
        dims = _detect_dims_transformer(weight_dict)
        in_d, out_d, h_d, n_h, n_ang, n_el, n_hds = dims
        model = DepthTransformer(
            input_dim or in_d, output_dim or out_d,
            hidden_dim or h_d, n_hidden if n_hidden is not None else n_h,
            n_angles=n_ang, n_elem=n_el, n_heads=n_hds,
        )
    else:
        from .model import MODEL_REGISTRY, _REGISTERED_LOADERS
        detect = {n: d for n, _, d in _REGISTERED_LOADERS}.get(arch)
        if detect is None:
            raise ValueError(f"Unknown architecture detected: {arch}")
        in_d, out_d, h_d, n_h = detect(weight_dict)
        model = MODEL_REGISTRY[arch](
            input_dim or in_d, output_dim or out_d,
            hidden_dim or h_d, n_hidden if n_hidden is not None else n_h,
        )

    model.load_weights(list(weight_dict.items()))
    mx.eval(model.parameters())
    return model
