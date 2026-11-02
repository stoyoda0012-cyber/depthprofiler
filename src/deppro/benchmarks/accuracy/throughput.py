#!/usr/bin/env python
"""4層モデル スループット & 精度テスト"""

import numpy as np
import time
import multiprocessing as mp

from deppro import (
    TransferMatrix, InverseSolver, SolverConfig, ElementConfig, BatchInverseSolver
)

def create_4layer_profile(depth, variation=0.0):
    """4層モデルのdepth profileを生成"""
    n_depth = len(depth)
    d1 = 20.0 + variation * np.random.uniform(-5, 5)
    d2 = 50.0 + variation * np.random.uniform(-10, 10)
    d3 = 80.0 + variation * np.random.uniform(-10, 10)

    profile = np.zeros((n_depth, 3))
    for i, d in enumerate(depth):
        if d < d1:
            profile[i] = [0.8, 0.1, 0.1]
        elif d < d2:
            profile[i] = [0.05, 0.67, 0.28]
        elif d < d3:
            t = (d - d2) / (d3 - d2)
            profile[i] = [0.02, 0.67*(1-t), 0.28 + 0.70*t]
        else:
            profile[i] = [0.01, 0.01, 0.98]
    return profile


def main():
    print("=" * 65)
    print("4層モデル スループット & 精度テスト")
    print(f"CPU cores: {mp.cpu_count()}")
    print("=" * 65)

    # ===== セットアップ =====
    angles = np.array([15, 30, 45, 60, 75, 85])
    depth = np.linspace(0, 200, 101)
    n_depth = len(depth)
    step_size = depth[1] - depth[0]

    elements = [
        ElementConfig(symbol='C', orbital='1s', compound='C', binding_energy=285.0),
        ElementConfig(symbol='O', orbital='1s', compound='SiO2', binding_energy=532.0),
        ElementConfig(symbol='Si', orbital='2p', compound='Si', binding_energy=99.0),
    ]
    n_elements = len(elements)

    tm = TransferMatrix(depth=depth, step_size=step_size, angles=angles)
    solver_config = SolverConfig(method='L2', max_iter=100, tol=1e-6)
    D_prior = np.ones((n_depth, n_elements)) / n_elements

    results_summary = {}

    # ===== N=10 逐次処理テスト =====
    N = 10
    print(f"\n{'='*65}")
    print(f"テスト1: N={N} (逐次処理)")
    print(f"{'='*65}")

    np.random.seed(42)
    profiles = [create_4layer_profile(depth, variation=0.3) for _ in range(N)]

    # 順問題
    start_time = time.time()
    I_obs_list = []
    for D in profiles:
        tm.compute(D, elements)
        I_obs = tm.forward(D, normalize=True)
        I_obs_list.append(I_obs)
    gen_time = time.time() - start_time

    # 逆問題（逐次）
    start_time = time.time()
    reconstructed = []
    for I_obs in I_obs_list:
        solver = InverseSolver(tm, solver_config, D_prior, elements)
        result = solver.solve(I_obs)
        reconstructed.append(result.D_opt)
    solve_time = time.time() - start_time

    total_time = gen_time + solve_time

    mse_list = [np.mean((p - r)**2) for p, r in zip(profiles, reconstructed)]
    mean_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)

    print(f"  順問題時間:   {gen_time:.4f} 秒 ({gen_time/N*1000:.2f} ms/profile)")
    print(f"  逆問題時間:   {solve_time:.4f} 秒 ({solve_time/N*1000:.2f} ms/profile)")
    print(f"  合計時間:     {total_time:.4f} 秒")
    print(f"  スループット: {N/total_time:.1f} profiles/秒")
    print(f"  MSE:          {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"  RMSE:         {np.sqrt(mean_mse):.4f}")

    results_summary['N=10 Sequential'] = {
        'N': N, 'total_time': total_time, 'throughput': N/total_time,
        'mse': mean_mse, 'rmse': np.sqrt(mean_mse)
    }

    # ===== N=100 逐次バッチテスト =====
    N = 100
    print(f"\n{'='*65}")
    print(f"テスト2: N={N} (逐次バッチ)")
    print(f"{'='*65}")

    np.random.seed(42)
    profiles = [create_4layer_profile(depth, variation=0.3) for _ in range(N)]

    # 順問題
    start_time = time.time()
    I_obs_batch = np.zeros((N, len(angles), n_elements))
    for i, D in enumerate(profiles):
        tm.compute(D, elements)
        I_obs_batch[i] = tm.forward(D, normalize=True)
    gen_time = time.time() - start_time

    # 逆問題（逐次バッチ）
    batch_solver = BatchInverseSolver(tm, solver_config, D_prior)

    start_time = time.time()
    batch_results = batch_solver.solve_batch_sequential(I_obs_batch)
    solve_time = time.time() - start_time

    total_time = gen_time + solve_time

    mse_list = [np.mean((profiles[i] - r.D_opt)**2) for i, r in enumerate(batch_results)]
    mean_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)

    print(f"  順問題時間:   {gen_time:.4f} 秒 ({gen_time/N*1000:.2f} ms/profile)")
    print(f"  逆問題時間:   {solve_time:.4f} 秒 ({solve_time/N*1000:.2f} ms/profile)")
    print(f"  合計時間:     {total_time:.4f} 秒")
    print(f"  スループット: {N/total_time:.1f} profiles/秒")
    print(f"  MSE:          {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"  RMSE:         {np.sqrt(mean_mse):.4f}")

    results_summary['N=100 SeqBatch'] = {
        'N': N, 'total_time': total_time, 'throughput': N/total_time,
        'mse': mean_mse, 'rmse': np.sqrt(mean_mse)
    }

    # ===== N=100 並列処理テスト =====
    print(f"\n{'='*65}")
    print(f"テスト2b: N={N} (並列処理)")
    print(f"{'='*65}")

    start_time = time.time()
    batch_results = batch_solver.solve_batch(I_obs_batch)
    solve_time_parallel = time.time() - start_time

    total_time_parallel = gen_time + solve_time_parallel

    print(f"  順問題時間:   {gen_time:.4f} 秒 ({gen_time/N*1000:.2f} ms/profile)")
    print(f"  逆問題時間:   {solve_time_parallel:.4f} 秒 ({solve_time_parallel/N*1000:.2f} ms/profile)")
    print(f"  合計時間:     {total_time_parallel:.4f} 秒")
    print(f"  スループット: {N/total_time_parallel:.1f} profiles/秒")
    print(f"  並列効率:     {solve_time/solve_time_parallel:.2f}x")

    results_summary['N=100 Parallel'] = {
        'N': N, 'total_time': total_time_parallel, 'throughput': N/total_time_parallel,
        'mse': mean_mse, 'rmse': np.sqrt(mean_mse)
    }

    # ===== N=1000 逐次バッチテスト =====
    N = 1000
    print(f"\n{'='*65}")
    print(f"テスト3: N={N} (逐次バッチ)")
    print(f"{'='*65}")

    np.random.seed(42)
    profiles = [create_4layer_profile(depth, variation=0.3) for _ in range(N)]

    # 順問題
    start_time = time.time()
    I_obs_batch = np.zeros((N, len(angles), n_elements))
    for i, D in enumerate(profiles):
        tm.compute(D, elements)
        I_obs_batch[i] = tm.forward(D, normalize=True)
    gen_time = time.time() - start_time

    # 逆問題（逐次バッチ）
    start_time = time.time()
    batch_results = batch_solver.solve_batch_sequential(I_obs_batch)
    solve_time = time.time() - start_time

    total_time = gen_time + solve_time

    mse_list = [np.mean((profiles[i] - r.D_opt)**2) for i, r in enumerate(batch_results)]
    mean_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)

    print(f"  順問題時間:   {gen_time:.4f} 秒 ({gen_time/N*1000:.2f} ms/profile)")
    print(f"  逆問題時間:   {solve_time:.4f} 秒 ({solve_time/N*1000:.2f} ms/profile)")
    print(f"  合計時間:     {total_time:.4f} 秒")
    print(f"  スループット: {N/total_time:.1f} profiles/秒")
    print(f"  MSE:          {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"  RMSE:         {np.sqrt(mean_mse):.4f}")

    results_summary['N=1000 SeqBatch'] = {
        'N': N, 'total_time': total_time, 'throughput': N/total_time,
        'mse': mean_mse, 'rmse': np.sqrt(mean_mse)
    }

    # ===== N=1000 並列処理テスト =====
    print(f"\n{'='*65}")
    print(f"テスト3b: N={N} (並列処理)")
    print(f"{'='*65}")

    start_time = time.time()
    batch_results = batch_solver.solve_batch(I_obs_batch)
    solve_time_parallel = time.time() - start_time

    total_time_parallel = gen_time + solve_time_parallel

    print(f"  順問題時間:   {gen_time:.4f} 秒 ({gen_time/N*1000:.2f} ms/profile)")
    print(f"  逆問題時間:   {solve_time_parallel:.4f} 秒 ({solve_time_parallel/N*1000:.2f} ms/profile)")
    print(f"  合計時間:     {total_time_parallel:.4f} 秒")
    print(f"  スループット: {N/total_time_parallel:.1f} profiles/秒")
    print(f"  並列効率:     {solve_time/solve_time_parallel:.2f}x")

    results_summary['N=1000 Parallel'] = {
        'N': N, 'total_time': total_time_parallel, 'throughput': N/total_time_parallel,
        'mse': mean_mse, 'rmse': np.sqrt(mean_mse)
    }

    # ===== まとめ =====
    print(f"\n{'='*65}")
    print("結果まとめ")
    print(f"{'='*65}")
    print(f"{'テスト':<20} {'N':>6} {'時間(s)':>10} {'スループット':>15} {'RMSE':>10}")
    print("-" * 65)
    for name, data in results_summary.items():
        print(f"{name:<20} {data['N']:>6} {data['total_time']:>10.3f} {data['throughput']:>12.1f}/s {data['rmse']:>10.4f}")


if __name__ == '__main__':
    main()
