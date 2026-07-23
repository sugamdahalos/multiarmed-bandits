"""
Plotting utilities for Multi-Armed Bandit Experiments (Sutton & Barto Ch. 2).

Generates figures for benchmark runs B1 through B11 and parameter variation studies,
supporting early (1-100), medium (1-1000), and full horizon log-scale views.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Plotting Style & Palette Config
# ---------------------------------------------------------------------------

# Color palette mapped by experiment curve label
COLORS = {
    # Epsilon-greedy family
    "ε = 0.0 (greedy)":      "#718096",
    "ε = 0.01":              "#E53E3E",
    "ε = 0.1":               "#DD6B20",
    # UCB family
    "UCB (c = 1)":           "#805AD5",
    "UCB (c = 2)":           "#2B6CB0",
    # Gradient family
    "Gradient (α = 0.1)":    "#2F855A",
    "Gradient (α = 0.4)":    "#D69E2E",
    # Baseline comparison (B5)
    "with baseline, α = 0.1":    "#2B6CB0",
    "with baseline, α = 0.4":    "#DD6B20",
    "without baseline, α = 0.1": "#805AD5",
    "without baseline, α = 0.4": "#E53E3E",
    # Nonstationary algorithms
    "Sample Avg (1/N)":      "#B7791F",
    "Constant α = 0.1":      "#319795",
    # Optimistic init
    "Optimistic (Q₁=5, ε=0)": "#D53F8C",
    "Realistic (Q₁=0, ε=0.1)": "#DD6B20",
    # Nonstationary UCB variants
    "D-UCB (γ = 0.99, c = 2)": "#9F7AEA",
    "SW-UCB (τ = 200, c = 2)": "#48BB78",
}


def _apply_style():
    """Apply global matplotlib style defaults for clear figure readability."""
    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.sans-serif':  ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size':        10,
        'axes.edgecolor':   '#333333',
        'axes.linewidth':   0.8,
        'axes.titlesize':   13,
        'axes.titleweight': 'bold',
        'axes.labelsize':   11,
        'axes.labelweight': 'bold',
        'grid.color':       '#cccccc',
        'grid.linestyle':   '--',
        'grid.alpha':       0.6,
        'legend.fontsize':  9,
        'legend.framealpha': 0.9,
        'figure.dpi':       300,
    })


def _legend(ax, loc="lower right", title=None):
    """Add standard styled legend to axis."""
    ax.legend(loc=loc, title=title, frameon=True, facecolor="white",
              edgecolor="#cccccc", framealpha=0.95, title_fontsize=10)


def _smooth(curve, window=50):
    """Apply simple moving average smoothing for noisy curves."""
    if len(curve) <= window:
        return curve
    kernel = np.ones(window) / window
    return np.convolve(curve, kernel, mode='valid')


# ---------------------------------------------------------------------------
# Core Plot Panel Helpers
# ---------------------------------------------------------------------------

def plot_dual_panel(results: dict, save_path: str,
                    title: str, xlim: int = None,
                    xscale: str = 'linear', smooth_window: int = 0,
                    colors: dict = None, ylim_reward=None, ylim_optimal=None,
                    legend_loc: str = "lower right"):
    """Plot 2-panel figure comparing Average Reward and % Optimal Action."""
    _apply_style()
    cmap = colors or COLORS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for label, (r_curve, opt_curve) in results.items():
        color = cmap.get(label, None)
        lw = 1.6 if any(k in label for k in ("UCB", "Gradient", "Constant", "Optimistic", "D-UCB", "SW-UCB")) else 1.3

        rc = r_curve[:xlim] if xlim else r_curve
        oc = opt_curve[:xlim] if xlim else opt_curve

        if smooth_window > 0 and len(rc) > smooth_window:
            rc = _smooth(rc, smooth_window)
            oc = _smooth(oc, smooth_window)
            x = np.arange(smooth_window - 1, (xlim or len(r_curve)))
        else:
            x = np.arange(len(rc))

        ax1.plot(x, rc, label=label, color=color, linewidth=lw)
        ax2.plot(x, oc, label=label, color=color, linewidth=lw)

    ax1.set_ylabel("Average Reward")
    ax1.set_title(title, pad=10)
    ax1.set_xscale(xscale)
    ax1.grid(True, which="both")
    if ylim_reward:
        ax1.set_ylim(ylim_reward)
    _legend(ax1, loc=legend_loc, title="Algorithm")

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal Action")
    ax2.set_xscale(xscale)
    ax2.grid(True, which="both")
    if ylim_optimal:
        ax2.set_ylim(ylim_optimal)
    _legend(ax2, loc=legend_loc, title="Algorithm")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_single_panel(results: dict, save_path: str,
                      title: str, ylabel: str, metric_idx: int = 1,
                      xlim: int = None, xscale: str = 'linear',
                      smooth_window: int = 0, colors: dict = None,
                      ylim=None):
    """Plot single panel figure for a single chosen metric (0=reward, 1=%optimal)."""
    _apply_style()
    cmap = colors or COLORS
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for label, data in results.items():
        curve = data[metric_idx]
        color = cmap.get(label, None)
        lw = 1.6

        c = curve[:xlim] if xlim else curve
        if smooth_window > 0 and len(c) > smooth_window:
            c = _smooth(c, smooth_window)
            x = np.arange(smooth_window - 1, (xlim or len(curve)))
        else:
            x = np.arange(len(c))

        ax.plot(x, c, label=label, color=color, linewidth=lw)

    ax.set_xlabel("Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.set_xscale(xscale)
    ax.grid(True, which="both")
    if ylim:
        ax.set_ylim(ylim)
    _legend(ax, title="Algorithm")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_path}")


def generate_multiscale(plot_fn, results: dict, base_dir: str,
                        prefix: str, title_base: str,
                        num_steps: int, **kwargs):
    """Save figures across 3 time horizons: early (1-100), medium (1-1000), and full log-scale."""
    os.makedirs(base_dir, exist_ok=True)

    # Early phase: Steps 1-100 (raw un-smoothed linear plot)
    plot_fn(results, os.path.join(base_dir, f"{prefix}_early.png"),
            title=f"{title_base} — Early Phase (Steps 1–100)",
            xlim=100, xscale='linear', smooth_window=0, **kwargs)

    # Medium phase: Steps 1-1000 (light smoothing)
    medium_lim = min(1000, num_steps)
    plot_fn(results, os.path.join(base_dir, f"{prefix}_medium.png"),
            title=f"{title_base} — First 1,000 Steps",
            xlim=medium_lim, xscale='linear', smooth_window=10, **kwargs)

    # Full horizon: Log-scale x-axis with heavier smoothing
    sw = 50 if num_steps >= 5000 else (20 if num_steps >= 1000 else 0)
    plot_fn(results, os.path.join(base_dir, f"{prefix}_full.png"),
            title=f"{title_base} — Full Horizon ({num_steps:,} Steps, Log Scale)",
            xlim=num_steps, xscale='log', smooth_window=sw, **kwargs)


# ---------------------------------------------------------------------------
# Specific Experiment Plot Routines
# ---------------------------------------------------------------------------

def plot_b1_sanity_check(rewards: np.ndarray, true_value: float,
                         save_dir: str = "plots/b1_sanity_check"):
    """Plot histogram of rewards and running mean convergence for B1 sanity check."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    # Reward distribution histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(rewards, bins=80, density=True, color="#3182CE", alpha=0.7,
            edgecolor="white", linewidth=0.5, label="Reward samples")
    ax.axvline(true_value, color="#E53E3E", linewidth=2, linestyle="--",
               label=f"True q_*(best) = {true_value:.3f}")
    ax.axvline(np.mean(rewards), color="#2F855A", linewidth=2, linestyle=":",
               label=f"Sample mean = {np.mean(rewards):.3f}")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Density")
    ax.set_title("B1: Reward Distribution of Optimal Arm (10,000 Pulls)")
    _legend(ax, loc="upper left")
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_dir}/reward_histogram.png")

    # Running mean convergence curve
    fig, ax = plt.subplots(figsize=(9, 5))
    running_mean = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    ax.plot(running_mean, color="#3182CE", linewidth=1.2, label="Running mean")
    ax.axhline(true_value, color="#E53E3E", linewidth=1.5, linestyle="--",
               label=f"True q_*(best) = {true_value:.3f}")
    ax.fill_between(range(len(running_mean)),
                    true_value - 0.05, true_value + 0.05,
                    color="#E53E3E", alpha=0.1, label="±0.05 band")
    ax.set_xlabel("Number of Pulls")
    ax.set_ylabel("Running Mean Reward")
    ax.set_title("B1: Running Mean Convergence to True Value")
    ax.set_xscale('log')
    _legend(ax, loc="upper right")
    ax.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "running_mean_convergence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_dir}/running_mean_convergence.png")


def plot_b2_single_run(results: dict, save_dir: str = "plots/b2_single_run",
                       num_steps: int = 10000):
    """Plot step-by-step and cumulative rewards for a single 10-armed bandit run."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    b2_colors = {
        0.0:  "#718096",
        0.01: "#E53E3E",
        0.1:  "#DD6B20",
    }

    def _plot_b2(path, title, xlim, xscale, window):
        fig, ax = plt.subplots(figsize=(10, 5))
        for eps, rewards in results.items():
            label = f"ε = {eps} (greedy)" if eps == 0.0 else f"ε = {eps}"
            c = rewards[:xlim]
            if window > 0 and len(c) > window:
                smoothed = _smooth(c, window)
                x = np.arange(window - 1, xlim)
                ax.plot(x, smoothed, label=label, color=b2_colors.get(eps),
                        linewidth=1.4, alpha=0.9)
            else:
                ax.plot(c, label=label, color=b2_colors.get(eps),
                        linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Reward")
        ax.set_title(title)
        ax.set_xscale(xscale)
        ax.grid(True, which="both")
        _legend(ax, loc="lower right", title="Policy")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] {path}")

    _plot_b2(os.path.join(save_dir, "reward_early.png"),
             "B2: Single Run — Early Phase (Steps 1–100)", 100, 'linear', 0)
    _plot_b2(os.path.join(save_dir, "reward_medium.png"),
             "B2: Single Run — First 1,000 Steps", min(1000, num_steps), 'linear', 10)
    sw = 50 if num_steps >= 5000 else 10
    _plot_b2(os.path.join(save_dir, "reward_full.png"),
             f"B2: Single Run — Full {num_steps:,} Steps (Log Scale)",
             num_steps, 'log', sw)

    # Cumulative reward curves
    fig, ax = plt.subplots(figsize=(10, 5))
    for eps, rewards in results.items():
        label = f"ε = {eps} (greedy)" if eps == 0.0 else f"ε = {eps}"
        cum = np.cumsum(rewards)
        ax.plot(cum, label=label, color=b2_colors.get(eps), linewidth=1.4)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("B2: Cumulative Reward — Single Run")
    ax.set_xscale('log')
    ax.grid(True, which="both")
    _legend(ax, loc="upper left", title="Policy")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cumulative_reward.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_dir}/cumulative_reward.png")


def plot_b4_step11_spike(results: dict,
                         save_dir: str = "plots/b4_ucb"):
    """Zoomed plot for steps 1-20 showing the UCB step-11 spike (Exercise 2.8)."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    steps = np.arange(1, 21)

    for label in ["UCB (c = 2)", "UCB (c = 1)", "ε = 0.1"]:
        if label not in results:
            continue
        r_curve, opt_curve = results[label]
        color = COLORS.get(label)
        lw = 1.8 if "UCB" in label else 1.3
        ax1.plot(steps, r_curve[:20], label=label, color=color,
                 linewidth=lw, marker='o', markersize=4)
        ax2.plot(steps, opt_curve[:20], label=label, color=color,
                 linewidth=lw, marker='o', markersize=4)

    for ax in (ax1, ax2):
        ax.axvline(x=11, color='#E53E3E', linestyle='--', alpha=0.6, linewidth=1.2)
        ax.grid(True)
        ax.set_xticks(steps)

    # Highlight step 11 spike annotation
    if "UCB (c = 2)" in results:
        spike_val = results["UCB (c = 2)"][1][10]
        ax2.annotate('Step 11 Spike\n(All 10 arms tried once)',
                     xy=(11, spike_val),
                     xytext=(14, spike_val + 6),
                     arrowprops=dict(facecolor='#E53E3E', shrink=0.05,
                                    width=1.5, headwidth=6),
                     fontsize=9, fontweight='bold', color='#C53030',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5F5',
                                edgecolor='#FEB2B2', alpha=0.9))

    ax1.set_ylabel("Average Reward")
    ax1.set_title("Exercise 2.8: Steps 1–20 UCB Step-11 Spike Analysis")
    _legend(ax1, loc="upper left")

    ax2.set_xlabel("Step Number")
    ax2.set_ylabel("% Optimal Action")
    _legend(ax2, loc="upper left")

    plt.tight_layout()
    path = os.path.join(save_dir, "step11_spike_zoom.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


def plot_b6_comparison(results_by_walkstd: dict,
                       save_dir: str = "plots/b6_nonstationary"):
    """Plot sample average vs constant step-size alpha across walk_std drift speeds."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    for walk_std, results in results_by_walkstd.items():
        num_steps = len(list(results.values())[0][0])
        tag = f"walkstd_{str(walk_std).replace('.', 'p')}"

        generate_multiscale(
            plot_dual_panel, results, save_dir, tag,
            f"B6: Nonstationary Testbed (σ_walk = {walk_std})",
            num_steps=num_steps
        )


def plot_b7_optimistic(results: dict, num_steps: int,
                       save_dir: str = "plots/b7_optimistic"):
    """Plot optimistic vs realistic initialization comparison."""
    os.makedirs(save_dir, exist_ok=True)
    generate_multiscale(
        plot_dual_panel, results, save_dir, "optimistic",
        "B7: Optimistic Initial Values (Q₁=5, ε=0) vs ε-Greedy (Q₁=0, ε=0.1)",
        num_steps=num_steps
    )


def plot_b8_combined(results: dict, num_steps: int,
                      save_dir: str = "plots/b8_combined"):
    """Plot overall comparison of all four core action selection families."""
    os.makedirs(save_dir, exist_ok=True)
    generate_multiscale(
        plot_dual_panel, results, save_dir, "greatest_hits",
        "B8: Combined Comparison — All Chapter 2 Methods",
        num_steps=num_steps
    )


def plot_variation_study(all_variation_results: dict,
                          param_name: str, param_values: list,
                          save_dir: str = "plots/variations"):
    """Plot grid of subplots showing how environment parameters affect performance."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    n_vals = len(param_values)
    fig, axes = plt.subplots(2, n_vals, figsize=(5 * n_vals, 8), sharey='row')
    if n_vals == 1:
        axes = axes.reshape(2, 1)

    for col, pval in enumerate(param_values):
        results = all_variation_results[pval]
        for label, (r_curve, opt_curve) in results.items():
            color = COLORS.get(label)
            lw = 1.4
            rc = _smooth(r_curve[:1000], 10) if len(r_curve) >= 1000 else r_curve[:1000]
            oc = _smooth(opt_curve[:1000], 10) if len(opt_curve) >= 1000 else opt_curve[:1000]
            x = np.arange(len(rc))
            axes[0, col].plot(x, rc, label=label, color=color, linewidth=lw)
            axes[1, col].plot(x, oc, label=label, color=color, linewidth=lw)

        axes[0, col].set_title(f"{param_name} = {pval}", fontsize=11)
        axes[0, col].grid(True)
        axes[1, col].grid(True)
        axes[1, col].set_xlabel("Steps")

    axes[0, 0].set_ylabel("Average Reward")
    axes[1, 0].set_ylabel("% Optimal Action")
    _legend(axes[0, -1], loc="lower right")
    _legend(axes[1, -1], loc="lower right")

    fig.suptitle(f"Variation Study: Effect of {param_name} on Convergence",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, f"variation_{param_name.lower().replace(' ', '_').replace('*', 'star')}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")
