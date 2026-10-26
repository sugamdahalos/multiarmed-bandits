"""
Stationary Benchmark Runner (Sutton & Barto Ch. 2).

Reproduces:
- Figure 2.2: Epsilon-Greedy comparison (eps = 0, 0.01, 0.1)
- Figure 2.4: UCB1 (c = 2) vs Epsilon-Greedy
- Section 2.6: Optimistic Initial Values (Q1 = 5, eps = 0)
- Section 2.8: Gradient Bandit with baseline (alpha = 0.1)
- Modern comparison: Gaussian Thompson Sampling

Generates publication-quality figures saved to figures/.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bandits.evaluate import (
    simulate_vectorized_testbed,
    simulate_vectorized_ucb,
    simulate_vectorized_gradient,
    simulate_vectorized_thompson,
)

NUM_RUNS = 2000
NUM_STEPS = 1000
NUM_ARMS = 10
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


def plot_dual_metric(
    results: dict,
    colors: dict,
    save_path: Path,
    title: str,
    num_steps: int = NUM_STEPS,
):
    """Plot publication-grade dual-panel figure (Average Reward & % Optimal Action)."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    steps = np.arange(1, num_steps + 1)

    for label, (r_curve, opt_curve) in results.items():
        color = colors.get(label, "#4A5568")
        lw = 1.8 if any(k in label for k in ("UCB", "Thompson", "Optimistic")) else 1.4
        ax1.plot(steps, r_curve, label=label, color=color, linewidth=lw)
        ax2.plot(steps, opt_curve, label=label, color=color, linewidth=lw)

    ax1.set_ylabel("Average Reward")
    ax1.set_title(f"{title}: Average Reward")
    ax1.grid(True)
    ax1.legend(loc="lower right")

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal Action")
    ax2.set_title(f"{title}: % Optimal Action")
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {save_path.name}")


def run_stationary_benchmarks():
    print("=" * 70)
    print("  STATIONARY BENCHMARK SUITE (2,000 Runs, 1,000 Steps)")
    print("=" * 70)

    # 1. Sutton & Barto Figure 2.2 (Epsilon-Greedy)
    print("\n[1/3] Simulating Epsilon-Greedy Family (Figure 2.2)...")
    eps_results = {}
    eps_colors = {
        "ε = 0 (greedy)": "#718096",
        "ε = 0.01": "#E53E3E",
        "ε = 0.1": "#3182CE",
    }
    for eps in [0.0, 0.01, 0.1]:
        label = "ε = 0 (greedy)" if eps == 0.0 else f"ε = {eps}"
        r, o = simulate_vectorized_testbed(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=NUM_STEPS,
            epsilon=eps, seed=SEED,
        )
        eps_results[label] = (r, o)

    plot_dual_metric(
        eps_results, eps_colors,
        FIGURES_DIR / "stationary_figure_2_2.png",
        "Sutton & Barto Figure 2.2 (ε-Greedy Suite)"
    )

    # 2. Comparison Across Algorithm Families
    print("\n[2/3] Simulating Algorithm Suite (UCB, Thompson, Optimistic, Gradient)...")
    full_results = {
        "ε = 0 (greedy)": eps_results["ε = 0 (greedy)"],
        "ε = 0.1": eps_results["ε = 0.1"],
    }

    print("  Running UCB1 (c = 2)...")
    full_results["UCB1 (c = 2)"] = simulate_vectorized_ucb(
        num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=NUM_STEPS,
        c=2.0, seed=SEED,
    )

    print("  Running Gaussian Thompson Sampling...")
    full_results["Thompson Sampling"] = simulate_vectorized_thompson(
        num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=NUM_STEPS,
        seed=SEED,
    )

    print("  Running Optimistic Greedy (Q₁ = 5, ε = 0)...")
    full_results["Optimistic Greedy (Q₁ = 5)"] = simulate_vectorized_testbed(
        num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=NUM_STEPS,
        epsilon=0.0, q_init=5.0, seed=SEED,
    )

    print("  Running Gradient Bandit (α = 0.1)...")
    full_results["Gradient Bandit (α = 0.1)"] = simulate_vectorized_gradient(
        num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=NUM_STEPS,
        alpha=0.1, use_baseline=True, seed=SEED,
    )

    full_colors = {
        "ε = 0 (greedy)": "#A0AEC0",
        "ε = 0.1": "#3182CE",
        "UCB1 (c = 2)": "#805AD5",
        "Thompson Sampling": "#38A169",
        "Optimistic Greedy (Q₁ = 5)": "#D53F8C",
        "Gradient Bandit (α = 0.1)": "#DD6B20",
    }

    plot_dual_metric(
        full_results, full_colors,
        FIGURES_DIR / "stationary_benchmark.png",
        "Stationary 10-Armed Bandit Benchmark Comparison"
    )

    # 3. Print Empirical Metrics Table
    print("\n" + "=" * 70)
    print(f"  {'Algorithm':<28} | {'Step 100 Reward':<16} | {'Step 1k Reward':<16} | {'Step 1k %Opt':<12}")
    print("  " + "─" * 28 + "─┼─" + "─" * 16 + "─┼─" + "─" * 16 + "─┼─" + "─" * 12)
    for label, (r, o) in full_results.items():
        r100 = np.mean(r[90:100])
        r1k = np.mean(r[-100:])
        o1k = np.mean(o[-100:])
        print(f"  {label:<28} | {r100:<16.4f} | {r1k:<16.4f} | {o1k:<11.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_stationary_benchmarks()
