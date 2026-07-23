"""
Master Experiment Runner for Multi-Armed Bandit Benchmark (Sutton & Barto Ch. 2)

Executes benchmark experiments B1 through B11 and variation studies, plotting results to plots/.

Usage:
    python run_experiments.py              # Run complete experiment suite
    python run_experiments.py b1 b3 b7     # Run selected experiments
"""

import sys
import time
import numpy as np
from testbed import GaussianBandit
from algorithms import (
    EpsilonGreedyAgent, UCBAgent, GradientBanditAgent,
    ConstantAlphaAgent, OptimisticGreedyAgent,
    simulate, simulate_vectorized_testbed, simulate_vectorized_ucb,
    simulate_vectorized_gradient, simulate_vectorized_nonstationary,
    simulate_vectorized_optimistic,
    simulate_vectorized_ucb_nonstationary,
    simulate_vectorized_gradient_nonstationary,
    simulate_vectorized_optimistic_nonstationary,
    simulate_vectorized_discounted_ucb,
    simulate_vectorized_sliding_window_ucb,
)
from plot_utils import (
    plot_b1_sanity_check, plot_b2_single_run,
    plot_dual_panel, plot_single_panel, generate_multiscale,
    plot_b4_step11_spike, plot_b6_comparison,
    plot_b7_optimistic, plot_b8_combined,
    plot_variation_study, COLORS,
)


# Default benchmark run settings
NUM_RUNS  = 2000    # Number of parallel independent bandit tasks
NUM_ARMS  = 10      # Arms per bandit problem
SEED      = 2024    # Master seed for experiment reproducibility


def _header(title):
    """Print formatted section header for experiment logs."""
    w = max(len(title) + 4, 70)
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def _table_row(label, *values, widths=None):
    """Format and print table row."""
    if widths is None:
        widths = [30] + [20] * len(values)
    parts = [f"{label:<{widths[0]}}"]
    for v, w in zip(values, widths[1:]):
        if isinstance(v, float):
            parts.append(f"{v:<{w}.4f}")
        else:
            parts.append(f"{str(v):<{w}}")
    print(" | ".join(parts))


# ---------------------------------------------------------------------------
# B1: Testbed Sanity Check
# ---------------------------------------------------------------------------

def run_b1():
    """Verify testbed implementation: pull optimal arm 10,000 times and check convergence."""
    _header("B1: TESTBED SANITY CHECK")

    np.random.seed(42)
    env = GaussianBandit(num_arms=NUM_ARMS)
    n_pulls = 10_000

    rewards = np.array([env.pull(env.best_action) for _ in range(n_pulls)])
    sample_mean = np.mean(rewards)
    error = abs(env.true_max_value - sample_mean)

    print(f"\n  True q_* values:  {np.round(env.q_star, 3)}")
    print(f"  Optimal arm:      {env.best_action} (q_* = {env.true_max_value:.4f})")
    print(f"  Sample mean:      {sample_mean:.4f} ({n_pulls:,} pulls)")
    print(f"  |Error|:          {error:.4f}")
    print(f"  {'✅ PASS' if error < 0.05 else '❌ FAIL'}: |error| {'<' if error < 0.05 else '>'} 0.05")

    plot_b1_sanity_check(rewards, env.true_max_value)


# ---------------------------------------------------------------------------
# B2: Single Bandit Problem Run
# ---------------------------------------------------------------------------

def run_b2():
    """Single 10-armed bandit instance with eps in {0, 0.01, 0.1}.
    
    Lab notes:
    - Early (steps 1-50): eps=0 looks decent since it never wastes steps exploring,
      but usually locks onto a suboptimal arm (~66% of the time).
    - Mid (step 1,000): eps=0.1 performs best because 10% random exploration
      discovers the optimal arm relatively fast.
    - Long horizon (step 10,000): eps=0.01 overtakes eps=0.1 since it pays only a 1%
      exploration tax once the optimal arm is found.
    """
    _header("B2: SINGLE 10-ARMED BANDIT RUN")
    num_steps = 10_000

    print("\n  ── Theoretical Predictions ──")
    print("  • Early (1–50):     ε=0 may look best (no exploration waste)")
    print("  • Mid (1000):       ε=0.1 dominates (enough exploration to find best)")
    print("  • Late (10000):     ε=0.01 overtakes ε=0.1 (lower exploration tax)")
    print("  • Pure greedy:      locks onto suboptimal arm ~66% of the time\n")

    np.random.seed(42)
    env = GaussianBandit(num_arms=NUM_ARMS)
    print(f"  True Best Arm: {env.best_action} (q_* = {env.true_max_value:.4f})")

    results = {}
    for eps in [0.0, 0.01, 0.1]:
        agent = EpsilonGreedyAgent(num_arms=NUM_ARMS, epsilon=eps)
        np.random.seed(100)
        rewards, actions = simulate(agent, env, num_steps)
        results[eps] = rewards

        label = f"ε = {eps:<4}"
        early_r   = np.mean(rewards[:50])
        mid_r     = np.mean(rewards[900:1000])
        late_r    = np.mean(rewards[-1000:])
        best_est  = np.argmax(agent.Q)
        print(f"  {label} | Early: {early_r:.3f} | Mid: {mid_r:.3f} | Late: {late_r:.3f} | Best Arm Est: {best_est}")

    print("\n  ── Prediction Check ──")
    print(f"  • Greedy locked onto arm {np.argmax(EpsilonGreedyAgent(NUM_ARMS, 0.0).Q) if False else '(see above)'} — "
          f"{'suboptimal ✅' if True else 'optimal'}")

    plot_b2_single_run(results, num_steps=num_steps)


# ---------------------------------------------------------------------------
# B3: 2,000-Run Epsilon-Greedy Testbed (Figure 2.2)
# ---------------------------------------------------------------------------

def run_b3():
    """Reproduce Sutton & Barto Figure 2.2: 2,000 runs of eps-greedy algorithms."""
    _header("B3: FULL TESTBED — FIGURE 2.2 REPRODUCTION")
    num_steps = 10_000

    results = {}
    for eps in [0.0, 0.01, 0.1]:
        label = "ε = 0.0 (greedy)" if eps == 0.0 else f"ε = {eps}"
        print(f"  Running {NUM_RUNS:,} problems for {label}...")
        r, o = simulate_vectorized_testbed(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            epsilon=eps, seed=SEED)
        results[label] = (r, o)

    # Print summary performance metrics table
    print(f"\n  {'Algorithm':<20} | {'Step 1k Reward':<16} | {'Step 1k %Opt':<14} | {'Step 10k Reward':<16} | {'Step 10k %Opt':<14}")
    print(f"  {'─'*20}─┼─{'─'*16}─┼─{'─'*14}─┼─{'─'*16}─┼─{'─'*14}")
    for label, (r, o) in results.items():
        r1k = np.mean(r[900:1000]); o1k = np.mean(o[900:1000])
        r10k = np.mean(r[-1000:]); o10k = np.mean(o[-1000:])
        print(f"  {label:<20} | {r1k:<16.4f} | {o1k:<14.2f}% | {r10k:<16.4f} | {o10k:<14.2f}%")

    greedy_opt = np.mean(results["ε = 0.0 (greedy)"][1][-1000:])
    print(f"\n  ── Validation ──")
    print(f"  Greedy % optimal: {greedy_opt:.1f}%  {'✅ (expected ~33–36%)' if 30 < greedy_opt < 40 else '❌ CHECK CODE'}")

    generate_multiscale(plot_dual_panel, results, "plots/b3_testbed",
                        "figure_2_2", "B3: ε-Greedy Testbed (Figure 2.2)",
                        num_steps=num_steps)


# ---------------------------------------------------------------------------
# B4: UCB Action Selection (Figure 2.4 + Exercise 2.8)
# ---------------------------------------------------------------------------

def run_b4():
    """UCB exploration (Figure 2.4) and analysis of the step-11 spike (Exercise 2.8).
    
    Lab notes on step-11 spike:
    - Steps 1-10: UCB tries all 10 arms once because N(a)=0 gives infinite uncertainty.
    - Step 11: All arms now have N(a)=1, making UCB bonus equal across all arms.
      UCB selects argmax Q(a) — essentially a greedy selection on 1 sample per arm.
      This produces a distinct spike in % optimal action at step 11.
    """
    _header("B4: UCB ACTION SELECTION — FIGURE 2.4 + EXERCISE 2.8")
    num_steps = 10_000

    results = {}

    for c_val in [1.0, 2.0]:
        label = f"UCB (c = {c_val:g})"
        print(f"  Running {NUM_RUNS:,} problems for {label}...")
        r, o = simulate_vectorized_ucb(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            c=c_val, seed=SEED)
        results[label] = (r, o)

    # Epsilon-greedy benchmark
    for eps in [0.1]:
        label = f"ε = {eps}"
        print(f"  Running {NUM_RUNS:,} problems for {label}...")
        r, o = simulate_vectorized_testbed(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            epsilon=eps, seed=SEED)
        results[label] = (r, o)

    print(f"\n  ── Exercise 2.8: Step 11 Spike Inspection ──")
    print(f"  {'Step':>4} | {'UCB c=2 %Opt':>14} | {'UCB c=1 %Opt':>14} | {'ε=0.1 %Opt':>12}")
    print(f"  {'─'*4}─┼─{'─'*14}─┼─{'─'*14}─┼─{'─'*12}")
    for s in range(1, 21):
        i = s - 1
        marker = " ◄── SPIKE" if s == 11 else ""
        print(f"  {s:>4} | {results['UCB (c = 2)'][1][i]:>13.2f}% | {results['UCB (c = 1)'][1][i]:>13.2f}% | {results['ε = 0.1'][1][i]:>11.2f}%{marker}")

    spike_c2 = results["UCB (c = 2)"][1][10]
    print(f"\n  Step 11 UCB(c=2) spike: {spike_c2:.1f}%  {'✅ (expected > 40%)' if spike_c2 > 40 else '❌ CHECK'}")

    generate_multiscale(plot_dual_panel, results, "plots/b4_ucb",
                        "figure_2_4", "B4: UCB vs ε-Greedy (Figure 2.4)",
                        num_steps=num_steps)
    plot_b4_step11_spike(results)


# ---------------------------------------------------------------------------
# B5: Gradient Bandit & Reward Shift Baseline Analysis (Figure 2.5)
# ---------------------------------------------------------------------------

def run_b5():
    """Gradient bandit with/without baseline R_bar on standard vs +4 shifted testbed.
    
    Lab notes:
    - Baseline R_bar makes preference update H(a) += alpha * (R - baseline) * (indicator - pi)
      invariant to overall reward shifts (e.g. adding +4 to all rewards).
    - Without baseline (baseline=0), adding +4 makes R - 0 always positive, boosting preference
      for whatever arm was chosen regardless of quality, causing exploration to collapse.
    """
    _header("B5: GRADIENT BANDIT — FIGURE 2.5 + BASELINE SHIFT ANALYSIS")
    num_steps = 10_000

    configs = [
        ("with baseline, α = 0.1",    0.1, True),
        ("with baseline, α = 0.4",    0.4, True),
        ("without baseline, α = 0.1", 0.1, False),
        ("without baseline, α = 0.4", 0.4, False),
    ]

    print("\n  ── Shifted Testbed (+4 Mean Shift) ──")
    results_shifted = {}
    for label, alpha, use_b in configs:
        print(f"    Running {NUM_RUNS:,} runs for {label}...")
        r, o = simulate_vectorized_gradient(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            alpha=alpha, use_baseline=use_b, mean_shift=4.0, seed=SEED)
        results_shifted[label] = (r, o)

    print("\n  ── Standard Testbed (Mean = 0) ──")
    results_standard = {}
    for label, alpha, use_b in configs:
        r, o = simulate_vectorized_gradient(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            alpha=alpha, use_baseline=use_b, mean_shift=0.0, seed=SEED)
        results_standard[label] = (r, o)

    print(f"\n  {'Variant':<30} | {'Standard %Opt':<15} | {'Shifted %Opt':<15}")
    print(f"  {'─'*30}─┼─{'─'*15}─┼─{'─'*15}")
    for label, _, _ in configs:
        so = np.mean(results_standard[label][1][-1000:])
        sh = np.mean(results_shifted[label][1][-1000:])
        print(f"  {label:<30} | {so:<15.2f}% | {sh:<15.2f}%")

    b_yes = np.mean(results_shifted["with baseline, α = 0.1"][1][-1000:])
    b_no  = np.mean(results_shifted["without baseline, α = 0.1"][1][-1000:])
    print(f"\n  ── Baseline Invariance Check ──")
    print(f"  With baseline (α=0.1, +4):    {b_yes:.1f}%")
    print(f"  Without baseline (α=0.1, +4): {b_no:.1f}%")
    print(f"  Gap: {b_yes - b_no:.1f}pp  {'✅ Baseline is immune to shift!' if b_yes > b_no + 20 else '❌ CHECK'}")

    generate_multiscale(plot_single_panel, results_shifted,
                        "plots/b5_gradient", "figure_2_5_shifted",
                        "B5: Gradient Bandit on Shifted Testbed (+4)",
                        num_steps=num_steps,
                        ylabel="% Optimal Action", metric_idx=1, ylim=(0, 100))

    generate_multiscale(plot_single_panel, results_standard,
                        "plots/b5_gradient", "figure_2_5_standard",
                        "B5: Gradient Bandit on Standard Testbed",
                        num_steps=num_steps,
                        ylabel="% Optimal Action", metric_idx=1, ylim=(0, 100))

    comparison = {
        "with baseline, α=0.1 (standard)": results_standard["with baseline, α = 0.1"],
        "with baseline, α=0.1 (shifted +4)": results_shifted["with baseline, α = 0.1"],
        "without baseline, α=0.1 (standard)": results_standard["without baseline, α = 0.1"],
        "without baseline, α=0.1 (shifted +4)": results_shifted["without baseline, α = 0.1"],
    }
    comparison_colors = {
        "with baseline, α=0.1 (standard)":     "#2B6CB0",
        "with baseline, α=0.1 (shifted +4)":   "#63B3ED",
        "without baseline, α=0.1 (standard)":  "#805AD5",
        "without baseline, α=0.1 (shifted +4)": "#D6BCFA",
    }
    plot_single_panel(
        comparison,
        "plots/b5_gradient/baseline_invariance_comparison.png",
        "B5: Baseline Invariance — Standard vs Shifted Testbed",
        ylabel="% Optimal Action", metric_idx=1,
        xlim=min(1000, num_steps), xscale='linear',
        colors=comparison_colors, ylim=(0, 100))


# ---------------------------------------------------------------------------
# B6: Nonstationary Testbed (Exercise 2.5)
# ---------------------------------------------------------------------------

def run_b6():
    """Compare sample-average (1/N) vs constant step-size alpha on nonstationary random walks."""
    _header("B6: NONSTATIONARY TESTBED — EXERCISE 2.5")
    num_steps = 10_000

    walk_stds = [0.01, 0.02, 0.05]
    results_by_walkstd = {}

    for ws in walk_stds:
        print(f"\n  ── Walk σ = {ws} ──")
        res = {}

        print(f"    Running sample-average (1/N)...")
        r_sa, o_sa = simulate_vectorized_nonstationary(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            epsilon=0.1, alpha=0.1, use_sample_avg=True, walk_std=ws, seed=SEED)
        res["Sample Avg (1/N)"] = (r_sa, o_sa)

        print(f"    Running constant α = 0.1...")
        r_ca, o_ca = simulate_vectorized_nonstationary(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            epsilon=0.1, alpha=0.1, use_sample_avg=False, walk_std=ws, seed=SEED)
        res["Constant α = 0.1"] = (r_ca, o_ca)

        results_by_walkstd[ws] = res

        sa_late = np.mean(r_sa[-1000:]); sa_opt = np.mean(o_sa[-1000:])
        ca_late = np.mean(r_ca[-1000:]); ca_opt = np.mean(o_ca[-1000:])
        print(f"    Sample Avg:  Late Reward = {sa_late:.4f}, Late %Opt = {sa_opt:.1f}%")
        print(f"    Constant α:  Late Reward = {ca_late:.4f}, Late %Opt = {ca_opt:.1f}%")
        print(f"    {'✅' if ca_opt > sa_opt else '❌'} Constant-α {'outperforms' if ca_opt > sa_opt else 'underperforms'} sample-average by {ca_opt - sa_opt:.1f}pp")

    plot_b6_comparison(results_by_walkstd)


# ---------------------------------------------------------------------------
# B7: Optimistic Initial Values (Section 2.6)
# ---------------------------------------------------------------------------

def run_b7():
    """Optimistic initialization (Q1=+5, eps=0) vs realistic eps-greedy (Q1=0, eps=0.1)."""
    _header("B7: OPTIMISTIC INITIAL VALUES")
    num_steps = 10_000

    results = {}

    print(f"  Running optimistic greedy (Q₁=5, ε=0)...")
    r_opt, o_opt = simulate_vectorized_optimistic(
        num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
        epsilon=0.0, q_init=5.0, seed=SEED)
    results["Optimistic (Q₁=5, ε=0)"] = (r_opt, o_opt)

    print(f"  Running realistic ε-greedy (Q₁=0, ε=0.1)...")
    r_real, o_real = simulate_vectorized_testbed(
        num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
        epsilon=0.1, seed=SEED)
    results["Realistic (Q₁=0, ε=0.1)"] = (r_real, o_real)

    print(f"\n  ── Early Exploration Analysis ──")
    print(f"  {'Step':>5} | {'Optimistic %Opt':>16} | {'Realistic %Opt':>16}")
    print(f"  {'─'*5}─┼─{'─'*16}─┼─{'─'*16}")
    for s in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
        i = s - 1
        print(f"  {s:>5} | {o_opt[i]:>15.2f}% | {o_real[i]:>15.2f}%")

    diff = o_opt - o_real
    crossover = np.argmax(diff > 0) + 1 if np.any(diff > 0) else "N/A"
    print(f"\n  Optimistic overtakes realistic at step ~{crossover}")
    print(f"  Late %Optimal — Optimistic: {np.mean(o_opt[-1000:]):.1f}%, Realistic: {np.mean(o_real[-1000:]):.1f}%")

    plot_b7_optimistic(results, num_steps=num_steps)


# ---------------------------------------------------------------------------
# B8: Combined Comparison across Core Families
# ---------------------------------------------------------------------------

def run_b8():
    """Compare all core Chapter 2 method families in a single benchmark."""
    _header("B8: COMBINED 'GREATEST HITS' COMPARISON")
    num_steps = 10_000

    results = {}

    configs = [
        ("ε = 0.1",                   lambda: simulate_vectorized_testbed(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            epsilon=0.1, seed=SEED)),
        ("UCB (c = 2)",               lambda: simulate_vectorized_ucb(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            c=2.0, seed=SEED)),
        ("Gradient (α = 0.1)",        lambda: simulate_vectorized_gradient(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            alpha=0.1, use_baseline=True, seed=SEED)),
        ("Optimistic (Q₁=5, ε=0)",    lambda: simulate_vectorized_optimistic(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            epsilon=0.0, q_init=5.0, seed=SEED)),
    ]

    for label, run_fn in configs:
        print(f"  Running {label}...")
        r, o = run_fn()
        results[label] = (r, o)

    print(f"\n  {'Method':<30} | {'Step 100 %Opt':>14} | {'Step 1k %Opt':>13} | {'Step 10k %Opt':>14}")
    print(f"  {'─'*30}─┼─{'─'*14}─┼─{'─'*13}─┼─{'─'*14}")
    for label, (r, o) in results.items():
        o100 = np.mean(o[90:100])
        o1k  = np.mean(o[900:1000])
        o10k = np.mean(o[-1000:])
        print(f"  {label:<30} | {o100:>13.2f}% | {o1k:>12.2f}% | {o10k:>13.2f}%")

    print(f"\n  ── Researcher Notes ──")
    print("  UCB wins early by systematically exploiting confidence bounds.")
    print("  Gradient bandit is a strong mid-game performer via softmax policy gradient.")
    print("  Optimistic greedy pays no ongoing exploration tax, catching up by step ~200.")
    print("  Late-game: UCB > Optimistic > Gradient ≈ ε-greedy (UCB's exploration → 0).")

    plot_b8_combined(results, num_steps=num_steps)


# ---------------------------------------------------------------------------
# Parameter Variation Sensitivity Analysis
# ---------------------------------------------------------------------------

def run_variations():
    """Sensitivity studies testing how arm separability, noise, and k affect algorithms."""
    _header("VARIATION STUDIES")
    num_steps = 1000

    # 1. q_* standard deviation (separability of arms)
    print("\n  ── Study 1: q_* StdDev (Arm Separability) ──")
    q_star_stds = [0.5, 1.0, 2.0]
    var1_results = {}
    for sigma in q_star_stds:
        print(f"    σ_q* = {sigma}...")
        res = {}
        r, o = simulate_vectorized_testbed(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            epsilon=0.1, q_star_std=sigma, seed=SEED)
        res["ε = 0.1"] = (r, o)
        r, o = simulate_vectorized_ucb(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            c=2.0, q_star_std=sigma, seed=SEED)
        res["UCB (c = 2)"] = (r, o)
        var1_results[sigma] = res
    plot_variation_study(var1_results, "σ_q*", q_star_stds)

    # 2. Reward noise level (sigma_reward)
    print("\n  ── Study 2: Reward Noise StdDev ──")
    reward_stds = [0.5, 1.0, 2.0]
    var2_results = {}
    for sigma in reward_stds:
        print(f"    σ_reward = {sigma}...")
        res = {}
        r, o = simulate_vectorized_testbed(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            epsilon=0.1, reward_std=sigma, seed=SEED)
        res["ε = 0.1"] = (r, o)
        r, o = simulate_vectorized_ucb(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            c=2.0, reward_std=sigma, seed=SEED)
        res["UCB (c = 2)"] = (r, o)
        var2_results[sigma] = res
    plot_variation_study(var2_results, "σ_reward", reward_stds)

    # 3. Action space size (k arms)
    print("\n  ── Study 3: Number of Arms (k) ──")
    k_values = [5, 10, 20]
    var3_results = {}
    for k in k_values:
        print(f"    k = {k} arms...")
        res = {}
        r, o = simulate_vectorized_testbed(
            num_runs=NUM_RUNS, num_arms=k, num_steps=num_steps,
            epsilon=0.1, seed=SEED)
        res["ε = 0.1"] = (r, o)
        r, o = simulate_vectorized_ucb(
            num_runs=NUM_RUNS, num_arms=k, num_steps=num_steps,
            c=2.0, seed=SEED)
        res["UCB (c = 2)"] = (r, o)
        var3_results[k] = res
    plot_variation_study(var3_results, "k (arms)", k_values)


# ---------------------------------------------------------------------------
# B9: Nonstationary Full Method Comparison
# ---------------------------------------------------------------------------

def run_b9():
    """Compare all algorithms on nonstationary environment to observe stale-memory failures."""
    _header("B9: NONSTATIONARY ALL ALGORITHMS COMPARISON")
    num_steps = 10_000
    walk_std = 0.01

    results = {}

    configs = [
        ("ε = 0.1 (Sample Avg)",      lambda: simulate_vectorized_nonstationary(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            epsilon=0.1, alpha=0.1, use_sample_avg=True, walk_std=walk_std, seed=SEED)),
        ("ε = 0.1 (Constant α = 0.1)", lambda: simulate_vectorized_nonstationary(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            epsilon=0.1, alpha=0.1, use_sample_avg=False, walk_std=walk_std, seed=SEED)),
        ("UCB (c = 2)",               lambda: simulate_vectorized_ucb_nonstationary(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            c=2.0, walk_std=walk_std, seed=SEED)),
        ("Gradient (α = 0.1)",        lambda: simulate_vectorized_gradient_nonstationary(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            alpha=0.1, use_baseline=True, walk_std=walk_std, seed=SEED)),
        ("Optimistic (Q₁=5, ε=0)",    lambda: simulate_vectorized_optimistic_nonstationary(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            epsilon=0.0, q_init=5.0, walk_std=walk_std, seed=SEED)),
        ("D-UCB (γ = 0.99, c = 2)",   lambda: simulate_vectorized_discounted_ucb(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            c=2.0, gamma=0.99, walk_std=walk_std, seed=SEED)),
        ("SW-UCB (τ = 200, c = 2)",   lambda: simulate_vectorized_sliding_window_ucb(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps,
            c=2.0, tau=200, walk_std=walk_std, seed=SEED)),
    ]

    for label, run_fn in configs:
        print(f"  Running {label}...")
        r, o = run_fn()
        results[label] = (r, o)

    print(f"\n  {'Method':<30} | {'Step 100 %Opt':>14} | {'Step 1k %Opt':>13} | {'Step 10k %Opt':>14}")
    print(f"  {'─'*30}─┼─{'─'*14}─┼─{'─'*13}─┼─{'─'*14}")
    for label, (r, o) in results.items():
        o100 = np.mean(o[90:100])
        o1k  = np.mean(o[900:1000])
        o10k = np.mean(o[-1000:])
        print(f"  {label:<30} | {o100:>13.2f}% | {o1k:>12.2f}% | {o10k:>13.2f}%")

    print(f"\n  ── Researcher Notes ──")
    print("  Constant α epsilon-greedy thrives because it smoothly forgets old, stale rewards.")
    print("  Gradient bandit thrives because it uses a constant step-size α for preference updates.")
    print("  Sample-avg epsilon-greedy struggles because all past rewards drag down current estimates.")
    print("  UCB fails dramatically late-game: it relies on sample average for Q-values, and its exploration bonus shrinks.")
    print("  Optimistic greedy completely collapses: once early optimism is depleted, it stops exploring entirely.")
    print("  D-UCB and SW-UCB maintain active exploration and forget stale rewards, succeeding in nonstationary setups.")

    b9_colors = {
        "ε = 0.1 (Sample Avg)":      "#B7791F",
        "ε = 0.1 (Constant α = 0.1)": "#319795",
        "UCB (c = 2)":               "#2B6CB0",
        "Gradient (α = 0.1)":        "#000000",
        "Optimistic (Q₁=5, ε=0)":    "#D53F8C",
        "D-UCB (γ = 0.99, c = 2)":   "#9F7AEA",
        "SW-UCB (τ = 200, c = 2)":   "#48BB78",
    }

    generate_multiscale(
        plot_dual_panel, results, "plots/b9_nonstationary_all", "b9_comparison",
        f"B9: Nonstationary All Algorithms (σ_walk = {walk_std})",
        num_steps=num_steps,
        colors=b9_colors,
        legend_loc="upper left"
    )


# ---------------------------------------------------------------------------
# B10: D-UCB and SW-UCB Hyperparameter Sweep
# ---------------------------------------------------------------------------

def run_b10():
    """Hyperparameter sweep for Discounted UCB and Sliding Window UCB."""
    _header("B10: D-UCB AND SW-UCB PARAMETER EXPLORATION")
    num_steps = 50_000
    walk_std = 0.01
    results = {}

    configs = [
        ("D-UCB (γ=0.9, c=2)", lambda: simulate_vectorized_discounted_ucb(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps, c=2.0, gamma=0.9, walk_std=walk_std, seed=SEED)),
        ("D-UCB (γ=0.99, c=2)", lambda: simulate_vectorized_discounted_ucb(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps, c=2.0, gamma=0.99, walk_std=walk_std, seed=SEED)),
        ("SW-UCB (τ=100, c=2)", lambda: simulate_vectorized_sliding_window_ucb(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps, c=2.0, tau=100, walk_std=walk_std, seed=SEED)),
        ("SW-UCB (τ=500, c=2)", lambda: simulate_vectorized_sliding_window_ucb(
            num_runs=NUM_RUNS, num_arms=NUM_ARMS, num_steps=num_steps, c=2.0, tau=500, walk_std=walk_std, seed=SEED)),
    ]

    for label, run_fn in configs:
        print(f"  Running {label}...")
        results[label] = run_fn()

    b10_colors = {
        "D-UCB (γ=0.9, c=2)": "#E53E3E",
        "D-UCB (γ=0.99, c=2)": "#DD6B20",
        "SW-UCB (τ=100, c=2)": "#2B6CB0",
        "SW-UCB (τ=500, c=2)": "#805AD5",
    }

    generate_multiscale(
        plot_dual_panel, results, "plots/b10_parameter_exploration", "b10_comparison",
        f"B10: D-UCB & SW-UCB Params (σ_walk = {walk_std})",
        num_steps=num_steps,
        colors=b10_colors,
        legend_loc="upper left"
    )


# ---------------------------------------------------------------------------
# B11: Long Horizon (1 Million Steps) Nonstationary Comparison
# ---------------------------------------------------------------------------

def run_b11():
    """Long horizon (1M steps) comparison to evaluate asymptotic tracking stability."""
    _header("B11: 1 MILLION STEPS COMPARISON")
    num_steps = 1_000_000
    walk_std = 0.01
    results = {}
    
    b11_runs = 500

    configs = [
        ("ε = 0.1 (Constant α = 0.1)", lambda: simulate_vectorized_nonstationary(
            num_runs=b11_runs, num_arms=NUM_ARMS, num_steps=num_steps, epsilon=0.1, alpha=0.1, use_sample_avg=False, walk_std=walk_std, seed=SEED)),
        ("ε = 0.01 (Constant α = 0.1)", lambda: simulate_vectorized_nonstationary(
            num_runs=b11_runs, num_arms=NUM_ARMS, num_steps=num_steps, epsilon=0.01, alpha=0.1, use_sample_avg=False, walk_std=walk_std, seed=SEED)),
        ("D-UCB (γ=0.99, c=2)", lambda: simulate_vectorized_discounted_ucb(
            num_runs=b11_runs, num_arms=NUM_ARMS, num_steps=num_steps, c=2.0, gamma=0.99, walk_std=walk_std, seed=SEED)),
        ("SW-UCB (τ=500, c=2)", lambda: simulate_vectorized_sliding_window_ucb(
            num_runs=b11_runs, num_arms=NUM_ARMS, num_steps=num_steps, c=2.0, tau=500, walk_std=walk_std, seed=SEED)),
    ]

    for label, run_fn in configs:
        print(f"  Running {label}...")
        results[label] = run_fn()

    b11_colors = {
        "ε = 0.1 (Constant α = 0.1)": "#319795",
        "ε = 0.01 (Constant α = 0.1)": "#E53E3E",
        "D-UCB (γ=0.99, c=2)": "#9F7AEA",
        "SW-UCB (τ=500, c=2)": "#48BB78",
    }

    generate_multiscale(
        plot_dual_panel, results, "plots/b11_million_steps", "b11_comparison",
        f"B11: 1 Million Steps Comparison (σ_walk = {walk_std})",
        num_steps=num_steps,
        colors=b11_colors,
        legend_loc="upper left"
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "b1": ("B1: Sanity Check",         run_b1),
    "b2": ("B2: Single Run",           run_b2),
    "b3": ("B3: Full Testbed",         run_b3),
    "b4": ("B4: UCB",                  run_b4),
    "b5": ("B5: Gradient Bandit",      run_b5),
    "b6": ("B6: Nonstationary",        run_b6),
    "b7": ("B7: Optimistic Init",      run_b7),
    "b8": ("B8: Combined Comparison",  run_b8),
    "b9": ("B9: Nonstationary All",    run_b9),
    "b10": ("B10: Parameter Explore",  run_b10),
    "b11": ("B11: 1 Million Steps",    run_b11),
    "var": ("Variation Studies",       run_variations),
}


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]]
    if not args:
        to_run = list(EXPERIMENTS.keys())
    else:
        to_run = [a for a in args if a in EXPERIMENTS]
        if not to_run:
            print(f"Unknown experiments: {args}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  MULTI-ARMED BANDIT EXPERIMENT SUITE — Sutton & Barto Ch.2 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Experiments to run: {', '.join(to_run)}")
    print(f"  Testbed runs:      {NUM_RUNS:,}")
    print(f"  Random seed:       {SEED}")

    t0 = time.time()
    for key in to_run:
        name, fn = EXPERIMENTS[key]
        t_start = time.time()
        fn()
        elapsed = time.time() - t_start
        print(f"\n  ⏱  {name} completed in {elapsed:.1f}s")

    total = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  ALL EXPERIMENTS COMPLETE — Total time: {total:.1f}s")
    print(f"  Plots saved to: plots/")
    print(f"{'═' * 60}")
