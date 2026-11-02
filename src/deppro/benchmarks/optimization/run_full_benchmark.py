#!/usr/bin/env python
"""Full benchmark of initial value sensitivity"""

import sys
import os
import time
from datetime import datetime

from deppro.paths import ensure_outputs_dir

# Path setup (relative to this file: <repo>/src and the repo's parent dir)
_here = os.path.dirname(os.path.abspath(__file__))
deppro_dir = os.path.abspath(os.path.join(_here, '..', '..', '..'))
source_root = os.path.abspath(os.path.join(deppro_dir, '..', '..'))

sys.path.insert(0, deppro_dir)
sys.path.insert(0, source_root)

from initial_value import run_benchmark

if __name__ == '__main__':
    n_workers = min(8, os.cpu_count() or 4)
    
    # Output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = str(ensure_outputs_dir())
    
    output_file = os.path.join(output_dir, f'benchmark_initial_value_{timestamp}.txt')
    csv_file = os.path.join(output_dir, f'benchmark_initial_value_{timestamp}.csv')
    
    # Header
    header = f"""
{'='*80}
INITIAL VALUE SENSITIVITY BENCHMARK - FULL EVALUATION
{'='*80}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Workers: {n_workers}
Python: {sys.version}
{'='*80}

"""
    
    print(header)
    with open(output_file, 'w') as f:
        f.write(header)
    
    # CSV header
    csv_header = "Distribution_Count,Mode,Rate,Efficiency,SuccessRate,W1_Mean,W1_Std,Iters_Mean\n"
    with open(csv_file, 'w') as f:
        f.write(csv_header)
    
    # Run benchmarks
    results = []
    
    for n_dist in [100, 500, 1000, 2000, 5000, 10000]:
        print(f'\n--- {n_dist} distributions ---')
        print(f"{'Mode':<12} {'Rate':>10} {'Eff':>8} {'Success':>10} {'W1 Mean':>10} {'W1 Std':>8} {'Iters':>8}")
        print('-' * 80)
        
        section = f'\n--- {n_dist} distributions ---\n'
        section += f"{'Mode':<12} {'Rate':>10} {'Eff':>8} {'Success':>10} {'W1 Mean':>10} {'W1 Std':>8} {'Iters':>8}\n"
        section += '-' * 80 + '\n'
        
        for mode in ['zero', 'uniform', 'random']:
            t0 = time.perf_counter()
            try:
                r = run_benchmark(n_dist, mode, n_workers)
                elapsed = time.perf_counter() - t0
                
                line = f"{mode:<12} {r['rate']:>10.1f} {r['efficiency']:>7.1f}% {r['success_rate']:>9.1f}% {r['w1_mean']:>10.2f} {r['w1_std']:>8.2f} {r['iter_mean']:>8.1f}"
                print(line)
                section += line + '\n'
                
                # CSV line
                csv_line = f"{n_dist},{mode},{r['rate']:.1f},{r['efficiency']:.1f},{r['success_rate']:.1f},{r['w1_mean']:.2f},{r['w1_std']:.2f},{r['iter_mean']:.1f}\n"
                results.append(csv_line)
                
            except Exception as e:
                line = f'{mode:<12} ERROR: {str(e)[:60]}'
                print(line)
                section += line + '\n'
        
        with open(output_file, 'a') as f:
            f.write(section)
    
    # Write CSV
    with open(csv_file, 'w') as f:
        f.write(csv_header)
        f.writelines(results)
    
    footer = f"\n{'='*80}\nBENCHMARK COMPLETE\n{'='*80}\n"
    footer += f"Results saved to:\n  {output_file}\n  {csv_file}\n"
    
    print(footer)
    with open(output_file, 'a') as f:
        f.write(footer)
    
    print(f"✅ Output: {output_file}")
    print(f"✅ CSV: {csv_file}")
