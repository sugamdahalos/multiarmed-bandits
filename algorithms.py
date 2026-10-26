"""
Backward compatibility shim for algorithms and simulators.
Imports from bandits.agents and bandits.evaluate.
"""

from bandits.agents import (
    EpsilonGreedyAgent,
    ConstantAlphaAgent,
    OptimisticGreedyAgent,
    UCBAgent,
    DiscountedUCBAgent,
    SlidingWindowUCBAgent,
    GradientBanditAgent,
    GaussianThompsonSamplingAgent,
)

from bandits.evaluate import (
    simulate,
    simulate_vectorized_testbed,
    simulate_vectorized_ucb,
    simulate_vectorized_gradient,
    simulate_vectorized_thompson,
    simulate_vectorized_nonstationary,
    simulate_vectorized_discounted_ucb,
    simulate_vectorized_sliding_window_ucb,
)


def simulate_vectorized_optimistic(
    num_runs: int = 2000,
    num_arms: int = 10,
    num_steps: int = 1000,
    epsilon: float = 0.0,
    q_init: float = 5.0,
    seed: int = None,
):
    """Vectorized optimistic-initial-values agent in a stationary environment."""
    return simulate_vectorized_testbed(
        num_runs=num_runs, num_arms=num_arms, num_steps=num_steps,
        epsilon=epsilon, q_init=q_init, seed=seed,
    )


def simulate_vectorized_optimistic_nonstationary(
    num_runs: int = 2000,
    num_arms: int = 10,
    num_steps: int = 10000,
    epsilon: float = 0.0,
    q_init: float = 5.0,
    walk_std: float = 0.01,
    seed: int = None,
):
    """Vectorized optimistic-initial-values agent in a nonstationary environment."""
    return simulate_vectorized_nonstationary(
        num_runs=num_runs, num_arms=num_arms, num_steps=num_steps,
        epsilon=epsilon, alpha=0.1, use_sample_avg=True,
        walk_std=walk_std, q_init=q_init, seed=seed,
    )


def simulate_vectorized_ucb_nonstationary(
    num_runs: int = 2000,
    num_arms: int = 10,
    num_steps: int = 10000,
    c: float = 2.0,
    walk_std: float = 0.01,
    seed: int = None,
):
    """Vectorized standard UCB on a nonstationary drifting bandit testbed."""
    import numpy as np
    if seed is not None:
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

        N[idx, actions] += 1.0
        Q[idx, actions] += (1.0 / N[idx, actions]) * (rewards - Q[idx, actions])

    return avg_rewards, pct_optimal


def simulate_vectorized_gradient_nonstationary(
    num_runs: int = 2000,
    num_arms: int = 10,
    num_steps: int = 10000,
    alpha: float = 0.1,
    use_baseline: bool = True,
    walk_std: float = 0.01,
    seed: int = None,
):
    """Vectorized Gradient Bandit on a nonstationary drifting bandit testbed."""
    import numpy as np
    if seed is not None:
        np.random.seed(seed)

    q_star = np.zeros((num_runs, num_arms), dtype=float)
    H = np.zeros((num_runs, num_arms), dtype=float)
    R_bar = np.zeros(num_runs, dtype=float)

    avg_rewards = np.zeros(num_steps, dtype=float)
    pct_optimal = np.zeros(num_steps, dtype=float)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        step_num = t + 1
        q_star += np.random.normal(0.0, walk_std, size=(num_runs, num_arms))
        best_actions = np.argmax(q_star, axis=1)

        H_shifted = H - np.max(H, axis=1, keepdims=True)
        exp_H = np.exp(H_shifted)
        pi = exp_H / np.sum(exp_H, axis=1, keepdims=True)

        cum_pi = np.cumsum(pi, axis=1)
        u = np.random.rand(num_runs, 1)
        actions = np.argmax(cum_pi > u, axis=1)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], 1.0)
        avg_rewards[t] = np.mean(rewards)

        baseline = R_bar.copy() if use_baseline else 0.0
        if use_baseline:
            R_bar += (1.0 / step_num) * (rewards - R_bar)

        one_hot = np.zeros_like(pi)
        one_hot[idx, actions] = 1.0
        H += alpha * (rewards - baseline)[:, np.newaxis] * (one_hot - pi)

    return avg_rewards, pct_optimal
