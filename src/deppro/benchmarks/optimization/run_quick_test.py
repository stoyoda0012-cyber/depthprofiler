#!/usr/bin/env python
"""Quick test of initial value sensitivity benchmark"""

import sys
import os

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
deppro_parent = os.path.join(script_dir, '..', '..', '..')  # deppro/ folder
python_root = os.path.join(script_dir, '..', '..', '..')     # python/ folder

sys.path.insert(0, python_root)
sys.path.insert(0, os.path.dirname(python_root))

from initial_value import run_benchmark

if __name__ == '__main__':
    n_workers = min(8, os.cpu_count() or 4)
    
    print('=' * 70)
    print('INITIAL VALUE SENSITIVITY BENCHMARK - QUICK TEST')
    print('=' * 70)
    print(f'Workers: {n_workers}')
    print()
    
    n_dist = 10  # Quick test with 10 distributions
    print(f'--- {n_dist} distributions (quick test) ---')
    print(f"{'Mode':<12} {'Rate':>10} {'Eff':>8} {'Success':>10} {'W1 Mean':>10} {'W1 Std':>8} {'Iters':>8}")
    print('-' * 70)
    
    for mode in ['zero', 'uniform', 'random']:
        try:
            r = run_benchmark(n_dist, mode, n_workers)
            print(f"{mode:<12} {r['rate']:>10.1f} {r['efficiency']:>7.1f}% {r['success_rate']:>9.1f}% {r['w1_mean']:>10.2f} {r['w1_std']:>8.2f} {r['iter_mean']:>8.1f}")
        except Exception as e:
            print(f'{mode:<12} ERROR: {e}')
            import traceback
            traceback.print_exc()
    
    print()
    print('=' * 70)
    print('QUICK TEST COMPLETE')
    print('=' * 70)
