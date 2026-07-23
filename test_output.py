"""
Quick check script to run long-horizon (200k steps) eps-greedy runs
and inspect asymptotic convergence behavior across different eps values.
"""
import numpy as np
import sys
sys.path.append('.')
from algorithms import simulate_vectorized_testbed

def main():
    print("Testing testbed...")
    for eps in [0.0, 0.01, 0.1]:
        r, o = simulate_vectorized_testbed(
            num_runs=2000, num_arms=10, num_steps=200000,
            epsilon=eps, seed=42)
        r100k = np.mean(r[-1000:]); o100k = np.mean(o[-1000:])
        print(f"eps={eps}: Reward 200k={r100k:.3f}, %Opt 200k={o100k:.1f}%")

if __name__ == '__main__':
    main()
