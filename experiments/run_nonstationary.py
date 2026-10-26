"""
Non-Stationary Benchmark Runner (Sutton & Barto Exercise 2.5 + Modern Tracking).

Demonstrates tracking ability in drifting reward environments:
- q_*(a) undergoes independent Gaussian random walk: increment ~ N(0, 0.01^2)
- Compares:
  1. Epsilon-Greedy with sample-average updates (1/k) [Fails: 1/k -> 0]
  2. Epsilon-Greedy with constant step-size (alpha = 0.1) [Tracks moving target]
  3. Standard UCB1 (c = 2) [Fails due to diminishing exploration & sample-avg Q]
  4. Discounted UCB (D-UCB, gamma = 0.99, c = 2) [Recovers tracking via discount]
  5. Sliding-Window UCB (SW-UCB, tau = 200, c = 2) [Tracks via bounded history]

Outputs publication-quality figure to figures/nonstationary_benchmark.png.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bandits.evaluate import (
    simulate_vectorized_nonstationary,
    simulate_vectorized_discounted_ucb,
    simulate_vectorized_sliding_window_ucb,
    simulate_vectorized_ucb,
)

NUM_RUNS = 2000
NUM_STEPS = 10000
NUM_ARMS = 10
WALK_STD = 0.01
SEED = 42

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def set_plot_style():
    """Apply clean publication-style matplotlib parameters."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.edgecolor": "#2D3748",
        "axes.linewidth": 1.0,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "legend.fontsize": 9,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CBD5E0",
        "figure.dpi": 300,
    })


def simulate_ucb_nonstationary_standard(num_runs, num_arms, num_steps, c, walk_std, seed):
    """Standard UCB1 running on drifting arms to show stale memory failure."""
    np.random.seed(seed)
    q_star = np.zeros((num_runs, num_arms), dtype=float)
    Q = np.zeros((num_runs, num_arms), dtype=float)
    N = np.zeros((num_runs, num_arms), dtype=float)

    avg_rewards = np.zeros(num_steps, dtype=float)
    pct_optimal = np.zeros(num_steps, dtype=float)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        step_num = t + 1
        q_star += np.random.normal(0.0, walk_std, size=(num_runs, num_arms))
        best_actions = np.argmax(q_star, axis=1)

        actions = np.zeros(num_runs, dtype=int)
        unpulled_mask = (N == 0)
        has_unpulled = np.any(unpulled_mask, axis=1)
        if np.any(has_unpulled):
            actions[has_unpulled] = np.argmax(unpulled_mask[has_unpulled], axis=1)

        all_pulled = ~has_unpulled
        if np.any(all_pulled):
            q_sub = Q[all_pulled]
            n_sub = N[all_pulled]
            ucb = q_sub + c * np.sqrt(np.log(step_num) / n_sub)
            ucb += np.random.uniform(0, 1e-9, ucb.shape)
            actions[all_pulled] = np.argmax(ucb, axis=1)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], 1.0)
        avg_rewards[t] = np.mean(rewards)

        # Standard 1/N sample-average update (stale memory failure)
        N[idx, actions] += 1.0
        Q[idx, actions] += (1.0 / N[idx, actions]) * (rewards - Q[idx, actions])

    return avg_rewards, pct_optimal


def run_nonstationary_benchmarks():
    print("=" * 70)
    print("  NON-STATIONARY BENCHMARK SUITE (2,000 Runs, 10,000 Steps)")
    print(f"  Random walk drift: σ = {WALK_STD}")
    print("=" * 70)

    results = {}

    print("\n[1/5] Simulating Sample Average Epsilon-Greedy (1/k)...")
    results["ε = 0.1 (Sample Avg 1/k)"] = simulate_vectorized_nonstationary(
        num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=NUM_STEPS,
        epsilon=0.1, use_sample_avg=True, walk_std=WALK_STD, seed=SEED,
    )

    print("[2/5] Simulating Constant Step-Size Epsilon-Greedy (α = 0.1)...")
    results["ε = 0.1 (Constant α = 0.1)"] = simulate_vectorized_nonstationary(
        num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=NUM_STEPS,
        epsilon=0.1, alpha=0.1, use_sample_avg=False, walk_std=WALK_STD, seed=SEED,
    )

    print("[3/5] Simulating Standard UCB1 (c = 2)...")
    results["Standard UCB1 (c = 2)"] = simulate_ucb_nonstationary_standard(
        num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=NUM_STEPS,
        c=2.0, walk_std=WALK_STD, seed=SEED,
    )

    print("[4/5] Simulating Discounted UCB (D-UCB, γ = 0.99, c = 2)...")
    results["D-UCB (γ = 0.99, c = 2)"] = simulate_vectorized_discounted_ucb(
        num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=NUM_STEPS,
        c=2.0, gamma=0.99, walk_std=WALK_STD, seed=SEED,
    )

    print("[5/5] Simulating Sliding-Window UCB (SW-UCB, τ = 200, c = 2)...")
    results["SW-UCB (τ = 200, c = 2)"] = simulate_vectorized_sliding_window_ucb(
        num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=NUM_STEPS,
        c=2.0, tau=200, walk_std=WALK_STD, seed=SEED,
    )

    colors = {
        "ε = 0.1 (Sample Avg 1/k)": "#A0AEC0",
        "ε = 0.1 (Constant α = 0.1)": "#319795",
        "Standard UCB1 (c = 2)": "#E53E3E",
        "D-UCB (γ = 0.99, c = 2)": "#805AD5",
        "SW-UCB (τ = 200, c = 2)": "#38A169",
    }

    # 1. Pure Exercise 2.5 Plot: Sample Average vs Constant Alpha
    fig_ex, (ax_r, ax_o) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    steps = np.arange(1, NUM_STEPS + 1)
    
    ex_configs = [
        ("ε = 0.1 (Sample Avg 1/k)", results["ε = 0.1 (Sample Avg 1/k)"], "#E53E3E"),
        ("ε = 0.1 (Constant α = 0.1)", results["ε = 0.1 (Constant α = 0.1)"], "#3182CE"),
    ]
    for label, (r_c, o_c), col in ex_configs:
        ax_r.plot(steps, r_c, label=label, color=col, linewidth=1.6)
        ax_o.plot(steps, o_c, label=label, color=col, linewidth=1.6)

    ax_r.set_ylabel("Average Reward")
    ax_r.set_title("Exercise 2.5: Sample-Average vs Constant Step-Size (Random Walk sigma = 0.01)")
    ax_r.grid(True)
    ax_r.legend(loc="upper left")

    ax_o.set_xlabel("Steps")
    ax_o.set_ylabel("% Optimal Action")
    ax_o.set_title("Exercise 2.5: % Optimal Action")
    ax_o.set_ylim(0, 100)
    ax_o.grid(True)
    ax_o.legend(loc="upper left")

    plt.tight_layout()
    ex_save_path = FIGURES_DIR / "nonstationary_exercise_2_5.png"
    plt.savefig(ex_save_path, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved Exercise 2.5 figure: {ex_save_path.name}")

    # 2. Broader Multi-Method Comparison Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for label, (r_curve, opt_curve) in results.items():
        color = colors.get(label, "#4A5568")
        lw = 1.8 if any(k in label for k in ("Constant", "D-UCB", "SW-UCB")) else 1.3
        ax1.plot(steps, r_curve, label=label, color=color, linewidth=lw)
        ax2.plot(steps, opt_curve, label=label, color=color, linewidth=lw)

    ax1.set_ylabel("Average Reward")
    ax1.set_title("Non-Stationary Tracking Benchmark: Average Reward (Random Walk sigma = 0.01)")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal Action")
    ax2.set_title("Non-Stationary Tracking Benchmark: % Optimal Action")
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    save_path = FIGURES_DIR / "nonstationary_benchmark.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved figure: {save_path.name}")

    # Metrics summary
    print("\n" + "=" * 70)
    print(f"  {'Algorithm':<30} | {'Mid (Step 1k) %Opt':<20} | {'Late (Step 10k) %Opt':<20}")
    print("  " + "─" * 30 + "─┼─" + "─" * 20 + "─┼─" + "─" * 20)
    for label, (r, o) in results.items():
        o1k = np.mean(o[900:1000])
        o10k = np.mean(o[-1000:])
        print(f"  {label:<30} | {o1k:<19.2f}% | {o10k:<19.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_nonstationary_benchmarks()
