"""
最適化手法の比較テスト
========================

SLSQP vs L-BFGS-B vs trust-constr (内点法) の比較

Usage:
    python -m deppro.test_optimization_methods
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.special import erf
from pathlib import Path
import time


def build_multilayer_profile(
    depth: np.ndarray,
    thicknesses: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    """Build C/Al₂O₃/TiO₂/SiO₂/Si profile."""
    n_depth = len(depth)
    z = depth
    sigmas = widths / (2 * np.sqrt(np.log(2)))

    def erf_layer(z_start, z_end, sigma_start, sigma_end):
        rising = 0.5 * (1 + erf((z - z_start) / sigma_start))
        falling = 0.5 * (1 + erf((z - z_end) / sigma_end))
        return np.maximum(rising - falling, 0)

    D = np.zeros((n_depth, 6))

    t_C = thicknesses[0]
    D_C = 0.3 * np.exp(-z / 3.0) * (1 - 0.5 * (1 + erf((z - t_C) / sigmas[0])))
    D[:, 0] = np.maximum(D_C, 0)
    D[:, 1] = 0.4 * erf_layer(thicknesses[0], thicknesses[1], sigmas[0], sigmas[1])
    D[:, 2] = 0.333 * erf_layer(thicknesses[1], thicknesses[2], sigmas[1], sigmas[2])
    D[:, 3] = 0.333 * erf_layer(thicknesses[2], thicknesses[3], sigmas[2], sigmas[3])
    D[:, 5] = 0.5 * (1 + erf((z - thicknesses[3]) / sigmas[3]))
    D[:, 4] = 1.5 * D[:, 1] + 2.0 * D[:, 2] + 2.0 * D[:, 3]

    D = np.clip(D, 1e-6, 1.0)
    total = D.sum(axis=1, keepdims=True)
    D = D / np.maximum(total, 1e-10)
    return D


def build_transfer_matrices(depth: np.ndarray, angles_deg: np.ndarray) -> List[np.ndarray]:
    """Build transfer matrices."""
    n_depth = len(depth)
    n_angles = len(angles_deg)
    dz = depth[1] - depth[0] if len(depth) > 1 else 1.0
    imfp_values = np.array([25.0, 20.0, 22.0, 20.0, 24.0, 28.0])

    T_matrices = []
    for elem_idx in range(6):
        T = np.zeros((n_angles, n_depth))
        for a, angle in enumerate(angles_deg):
            theta = np.radians(angle)
            lambda_eff = imfp_values[elem_idx] * np.cos(theta)
            for j in range(n_depth):
                T[a, j] = np.exp(-depth[j] / lambda_eff) * dz / lambda_eff
        T_matrices.append(T)
    return T_matrices


@dataclass
class OptimizationResult:
    """最適化結果"""
    method: str
    final_positions: np.ndarray
    final_chi2: float
    n_iterations: int
    elapsed_time: float
    success: bool
    errors: np.ndarray  # 真値との誤差


def solve_with_method(
    method: str,
    I_obs: np.ndarray,
    T_matrices: List[np.ndarray],
    depth: np.ndarray,
    fixed_widths: np.ndarray,
    initial_positions: np.ndarray,
    true_positions: np.ndarray,
    bounds: Tuple[float, float] = (1.0, 150.0),
    max_iter: int = 500,
) -> OptimizationResult:
    """指定した手法で最適化"""

    n_angles, n_elements = I_obs.shape
    n_interfaces = len(initial_positions)

    I_obs_norm = I_obs / np.maximum(I_obs.sum(axis=1, keepdims=True), 1e-10)

    def objective(thicknesses):
        D = build_multilayer_profile(depth, thicknesses, fixed_widths)
        I_pred = np.zeros_like(I_obs)
        for e in range(6):
            I_pred[:, e] = T_matrices[e] @ D[:, e]
        I_pred_norm = I_pred / np.maximum(I_pred.sum(axis=1, keepdims=True), 1e-10)
        return np.sum((I_obs_norm - I_pred_norm) ** 2)

    # 境界条件
    lb = np.full(n_interfaces, bounds[0])
    ub = np.full(n_interfaces, bounds[1])
    scipy_bounds = Bounds(lb=lb, ub=ub)

    start_time = time.time()

    if method == 'SLSQP':
        # SLSQP: 制約を辞書形式で指定
        constraints = []
        for i in range(1, n_interfaces):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[i] - x[i-1] - 1.0,
            })

        result = minimize(
            objective,
            initial_positions,
            method='SLSQP',
            bounds=scipy_bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10},
        )

    elif method == 'L-BFGS-B':
        # L-BFGS-B: 境界のみ（制約なし）
        result = minimize(
            objective,
            initial_positions,
            method='L-BFGS-B',
            bounds=scipy_bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10},
        )

    elif method == 'trust-constr':
        # trust-constr (内点法): NonlinearConstraintを使用
        def monotonicity_constraint(x):
            # x[i] - x[i-1] >= 1.0 を満たすべき
            return np.diff(x) - 1.0

        nlc = NonlinearConstraint(
            monotonicity_constraint,
            lb=0.0,  # >= 0
            ub=np.inf,
        )

        result = minimize(
            objective,
            initial_positions,
            method='trust-constr',
            bounds=scipy_bounds,
            constraints=nlc,
            options={'maxiter': max_iter, 'gtol': 1e-10},
        )

    elif method == 'COBYLA':
        # COBYLA: 微分不要の制約付き最適化
        constraints = []
        for i in range(1, n_interfaces):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[i] - x[i-1] - 1.0,
            })
        # 境界を制約として追加
        for i in range(n_interfaces):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - bounds[0]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: bounds[1] - x[i]})

        result = minimize(
            objective,
            initial_positions,
            method='COBYLA',
            constraints=constraints,
            options={'maxiter': max_iter, 'rhobeg': 1.0},
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - start_time
    final_positions = result.x
    errors = final_positions - true_positions

    return OptimizationResult(
        method=method,
        final_positions=final_positions,
        final_chi2=result.fun,
        n_iterations=result.nit if hasattr(result, 'nit') else -1,
        elapsed_time=elapsed,
        success=result.success,
        errors=errors,
    )


def run_comparison_test():
    """最適化手法の比較テスト"""

    print("=" * 70)
    print("最適化手法比較テスト")
    print("=" * 70)

    # 設定
    true_positions = np.array([5.0, 15.0, 35.0, 65.0])
    true_widths = np.array([2.0, 3.0, 4.0, 5.0])
    angles_deg = np.array([0, 30, 45, 55, 65, 75])
    noise_level = 0.02

    depth = np.linspace(0, 100, 101)

    # 真のプロファイル
    D_true = build_multilayer_profile(depth, true_positions, true_widths)
    T_matrices = build_transfer_matrices(depth, angles_deg)

    # 観測強度
    n_angles = len(angles_deg)
    n_elements = 6
    I_true = np.zeros((n_angles, n_elements))
    for e in range(n_elements):
        I_true[:, e] = T_matrices[e] @ D_true[:, e]

    np.random.seed(42)
    noise = np.random.normal(0, noise_level, I_true.shape)
    I_obs = I_true * (1 + noise)
    I_obs = np.maximum(I_obs, 1e-10)

    print(f"\n真の界面位置: {true_positions} Å")
    print(f"固定界面幅: {true_widths} Å")
    print(f"ノイズレベル: {noise_level * 100:.0f}%")

    # テストする手法
    methods = ['SLSQP', 'L-BFGS-B', 'trust-constr', 'COBYLA']

    # 異なる初期値でテスト
    perturbations = [
        ("小 (±3Å)", np.array([8.0, 18.0, 32.0, 62.0])),
        ("中 (±5Å)", np.array([10.0, 20.0, 30.0, 60.0])),
        ("大 (±10Å)", np.array([15.0, 25.0, 25.0, 55.0])),
    ]

    all_results: Dict[str, List[OptimizationResult]] = {m: [] for m in methods}

    for perturb_name, init_pos in perturbations:
        print(f"\n{'=' * 60}")
        print(f"初期値摂動: {perturb_name}")
        print(f"初期位置: {init_pos}")
        print(f"{'=' * 60}")

        for method in methods:
            try:
                result = solve_with_method(
                    method=method,
                    I_obs=I_obs,
                    T_matrices=T_matrices,
                    depth=depth,
                    fixed_widths=true_widths,
                    initial_positions=init_pos.copy(),
                    true_positions=true_positions,
                )
                all_results[method].append(result)

                mean_err = np.mean(np.abs(result.errors))
                print(f"\n  {method:15s}:")
                print(f"    最終位置: {result.final_positions.round(2)}")
                print(f"    誤差: {result.errors.round(2)} Å")
                print(f"    平均誤差: {mean_err:.3f} Å")
                print(f"    chi2: {result.final_chi2:.2e}")
                print(f"    反復回数: {result.n_iterations}")
                print(f"    計算時間: {result.elapsed_time*1000:.1f} ms")
                print(f"    収束: {'Yes' if result.success else 'No'}")

            except Exception as e:
                print(f"\n  {method:15s}: エラー - {e}")

    # サマリー
    print("\n" + "=" * 70)
    print("サマリー")
    print("=" * 70)

    print(f"\n{'手法':<15} | {'平均誤差':>10} | {'平均chi2':>12} | {'平均時間':>10} | {'成功率':>8}")
    print("-" * 70)

    for method in methods:
        results = all_results[method]
        if results:
            mean_errors = [np.mean(np.abs(r.errors)) for r in results]
            mean_chi2 = [r.final_chi2 for r in results]
            mean_time = [r.elapsed_time * 1000 for r in results]
            success_rate = sum(1 for r in results if r.success) / len(results) * 100

            print(f"{method:<15} | {np.mean(mean_errors):>8.3f} Å | {np.mean(mean_chi2):>10.2e} | {np.mean(mean_time):>8.1f} ms | {success_rate:>6.0f}%")

    # 手法の性質（一般論。上の表はこの実行の結果）
    print("\n" + "=" * 70)
    print("手法の性質")
    print("=" * 70)
    print("""
    - SLSQP: 等式・不等式制約を扱える
    - L-BFGS-B: 境界制約だけを扱う（単調性などの制約は入れられない）
    - trust-constr: 制約を扱う信頼領域法
    - COBYLA: 微分を使わない
    """)


if __name__ == "__main__":
    run_comparison_test()
