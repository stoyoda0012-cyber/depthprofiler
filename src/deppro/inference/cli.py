"""
CLI for AI Inference Engine
===========================

Subcommands:
    train     - Train DepthMLP on Parquet data
    train-h5  - Train DepthMLP from H5 with augmentation
    benchmark - Evaluate against benchmark H5
    infer     - Run inference on arbitrary I_obs
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional


def cmd_train(args: argparse.Namespace) -> None:
    """Train a DepthMLP model."""
    from .train import train_model

    history = train_model(
        train_path=args.train,
        val_path=args.val,
        output_path=args.output,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        use_physics_loss=not args.no_physics_loss,
        lambda_sum=args.lambda_sum,
        n_depth=args.n_depth,
        n_elem=args.n_elem,
        verbose=True,
    )


def cmd_train_h5(args: argparse.Namespace) -> None:
    """Train a DepthMLP from H5 benchmark data with augmentation."""
    from .train import train_from_h5

    history = train_from_h5(
        h5_path=args.h5,
        output_path=args.output,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        n_aug=args.n_aug,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        val_fraction=args.val_fraction,
        use_w1_loss=args.use_w1_loss,
        lambda_w1=args.lambda_w1,
        verbose=True,
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark against SLSQP on H5 data."""
    from .train import load_model
    from .evaluate import evaluate_on_dstk

    model = load_model(
        args.model,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
    )

    results = evaluate_on_dstk(
        model=model,
        h5_path=args.h5,
        h5_element_order=getattr(args, 'h5_order', False),
        verbose=True,
    )


def cmd_infer(args: argparse.Namespace) -> None:
    """Run inference on I_obs data."""
    import numpy as np
    import mlx.core as mx
    from .train import load_model
    from .postprocess import enforce_constraints

    model = load_model(
        args.model,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
    )

    # Load input
    I_obs = np.loadtxt(args.input, delimiter=',').astype(np.float32)
    if I_obs.ndim == 1:
        I_obs = I_obs.reshape(1, -1)

    print(f"Input: {I_obs.shape[0]} pixels, {I_obs.shape[1]} features")

    # Inference
    t0 = time.perf_counter()
    D_pred = model(mx.array(I_obs))
    mx.eval(D_pred)
    D_pred_np = np.array(D_pred)
    t1 = time.perf_counter()

    # Post-process
    D_pred_np = enforce_constraints(D_pred_np, args.n_depth, args.n_elem)

    print(f"Inference: {t1 - t0:.4f}s ({I_obs.shape[0] / (t1 - t0):,.0f} px/s)")

    # Save
    np.savetxt(args.output, D_pred_np, delimiter=',', fmt='%.6f')
    print(f"Output: {args.output}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='AI Inference Engine for ARXPS Depth Profiling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Common model args
    def add_model_args(p):
        p.add_argument('--input-dim', type=int, default=84,
                        help='I_obs dimension (default: 84 = 14 angles × 6 elem)')
        p.add_argument('--output-dim', type=int, default=120,
                        help='D(z) dimension (default: 120 = 20 depths × 6 elem)')
        p.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden layer width (default: 256)')
        p.add_argument('--n-hidden', type=int, default=3,
                        help='Number of hidden layers (default: 3)')
        p.add_argument('--n-depth', type=int, default=20,
                        help='Number of depth points (default: 20)')
        p.add_argument('--n-elem', type=int, default=6,
                        help='Number of elements (default: 6)')

    # --- train ---
    p_train = subparsers.add_parser('train', help='Train DepthMLP')
    p_train.add_argument('--train', required=True, help='Training Parquet path')
    p_train.add_argument('--val', required=True, help='Validation Parquet path')
    p_train.add_argument('--output', default='depth_mlp.npz',
                         help='Output model weights path')
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--batch-size', type=int, default=8192)
    p_train.add_argument('--lr', type=float, default=1e-3)
    p_train.add_argument('--patience', type=int, default=5)
    p_train.add_argument('--no-physics-loss', action='store_true',
                         help='Disable sum constraint penalty')
    p_train.add_argument('--lambda-sum', type=float, default=0.1,
                         help='Sum constraint penalty weight')
    add_model_args(p_train)

    # --- train-h5 ---
    p_train_h5 = subparsers.add_parser('train-h5',
                                        help='Train DepthMLP from H5 with augmentation')
    p_train_h5.add_argument('--h5', required=True, help='Benchmark H5 path')
    p_train_h5.add_argument('--output', default='depth_mlp.npz',
                            help='Output model weights path')
    p_train_h5.add_argument('--n-aug', type=int, default=500_000,
                            help='Number of augmented samples (default: 500000)')
    p_train_h5.add_argument('--epochs', type=int, default=50)
    p_train_h5.add_argument('--batch-size', type=int, default=8192)
    p_train_h5.add_argument('--lr', type=float, default=1e-3)
    p_train_h5.add_argument('--patience', type=int, default=10)
    p_train_h5.add_argument('--val-fraction', type=float, default=0.1,
                            help='Validation fraction (default: 0.1)')
    p_train_h5.add_argument('--use-w1-loss', action='store_true',
                            help='Use MSE + W1 CDF loss (anti-correlation)')
    p_train_h5.add_argument('--lambda-w1', type=float, default=1.0,
                            help='W1 CDF loss weight (default: 1.0)')
    add_model_args(p_train_h5)

    # --- benchmark ---
    p_bench = subparsers.add_parser('benchmark', help='Benchmark against SLSQP')
    p_bench.add_argument('--model', required=True, help='Model weights path')
    p_bench.add_argument('--h5', required=True, help='Benchmark H5 path')
    p_bench.add_argument('--h5-order', action='store_true',
                         help='Model was trained in H5 element order (train-h5 mode)')
    add_model_args(p_bench)

    # --- infer ---
    p_infer = subparsers.add_parser('infer', help='Run inference')
    p_infer.add_argument('--model', required=True, help='Model weights path')
    p_infer.add_argument('--input', required=True, help='Input CSV (I_obs)')
    p_infer.add_argument('--output', default='D_pred.csv', help='Output CSV')
    add_model_args(p_infer)

    args = parser.parse_args(argv)

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'train-h5':
        cmd_train_h5(args)
    elif args.command == 'benchmark':
        cmd_benchmark(args)
    elif args.command == 'infer':
        cmd_infer(args)
