"""
Monte Carlo Simulation and Batch Vectorized Evaluators.

Batch simulation functions running parallel bandit problems with NumPy:
- simulate: Scalar single-agent single-environment runner.
- simulate_vectorized_testbed: Epsilon-greedy & optimistic greedy stationary benchmark.
- simulate_vectorized_ucb: UCB1 stationary benchmark.
- simulate_vectorized_thompson: Gaussian Thompson Sampling stationary benchmark.
- simulate_vectorized_gradient: Policy gradient stationary benchmark.
- simulate_vectorized_nonstationary: Sample-average vs constant-alpha tracking benchmark.
- simulate_vectorized_discounted_ucb: Discounted UCB non-stationary tracking.
- simulate_vectorized_sliding_window_ucb: Sliding-Window UCB non-stationary tracking.
"""

from typing import Optional, Tuple
import numpy as np


def simulate(agent, env, num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Execute a single agent on a single environment instance for num_steps."""
    rewards = np.zeros(num_steps, dtype=float)
    actions = np.zeros(num_steps, dtype=int)
    agent.reset()

    for t in range(num_steps):
        a = agent.select_action()
        r = env.pull(a)
        agent.update(a, r)
        if hasattr(env, "step"):
            env.step()
        rewards[t] = r
        actions[t] = a

    return rewards, actions


def simulate_vectorized_testbed(
    num_runs: int = 2000,
    num_arms: int = 10,
    num_steps: int = 1000,
    epsilon: float = 0.1,
    q_init: float = 0.0,
    q_star_std: float = 1.0,
    reward_std: float = 1.0,
    mean: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized epsilon-greedy simulation across num_runs parallel bandit problems."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.random.normal(mean, q_star_std, size=(num_runs, num_arms))
    best_actions = np.argmax(q_star, axis=1)

    Q = np.full((num_runs, num_arms), q_init, dtype=float)
    N = np.zeros((num_runs, num_arms), dtype=float)

    avg_rewards = np.zeros(num_steps, dtype=float)
    pct_optimal = np.zeros(num_steps, dtype=float)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        explore = np.random.rand(num_runs) < epsilon
        rand_a = np.random.randint(0, num_arms, size=num_runs)
        greedy_a = np.argmax(Q + np.random.uniform(0, 1e-9, Q.shape), axis=1)
        actions = np.where(explore, rand_a, greedy_a)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], reward_std)
        avg_rewards[t] = np.mean(rewards)

        N[idx, actions] += 1.0
        Q[idx, actions] += (1.0 / N[idx, actions]) * (rewards - Q[idx, actions])

    return avg_rewards, pct_optimal


def simulate_vectorized_ucb(
    num_runs: int = 2000,
    num_arms: int = 10,
    num_steps: int = 1000,
    c: float = 2.0,
    q_star_std: float = 1.0,
    reward_std: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized UCB1 simulation across num_runs parallel bandit problems."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.random.normal(0.0, q_star_std, size=(num_runs, num_arms))
    best_actions = np.argmax(q_star, axis=1)

    Q = np.zeros((num_runs, num_arms), dtype=float)
    N = np.zeros((num_runs, num_arms), dtype=float)

    avg_rewards = np.zeros(num_steps, dtype=float)
    pct_optimal = np.zeros(num_steps, dtype=float)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        step_num = t + 1
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

        rewards = np.random.normal(q_star[idx, actions], reward_std)
        avg_rewards[t] = np.mean(rewards)

        N[idx, actions] += 1.0
        Q[idx, actions] += (1.0 / N[idx, actions]) * (rewards - Q[idx, actions])

    return avg_rewards, pct_optimal


def simulate_vectorized_thompson(
    num_runs: int = 2000,
    num_arms: int = 10,
    num_steps: int = 1000,
    prior_mean: float = 0.0,
    prior_variance: float = 1.0,
    reward_std: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Gaussian Thompson Sampling across num_runs parallel bandit problems."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.random.normal(0.0, 1.0, size=(num_runs, num_arms))
    best_actions = np.argmax(q_star, axis=1)

    N = np.zeros((num_runs, num_arms), dtype=float)
    sum_rewards = np.zeros((num_runs, num_arms), dtype=float)

    avg_rewards = np.zeros(num_steps, dtype=float)
    pct_optimal = np.zeros(num_steps, dtype=float)
    idx = np.arange(num_runs)

    prior_prec = 1.0 / prior_variance
    reward_var = reward_std ** 2
    obs_prec_factor = 1.0 / reward_var

    for t in range(num_steps):
        # Posterior Gaussian distribution per arm
        post_prec = prior_prec + N * obs_prec_factor
        post_var = 1.0 / post_prec
        post_std = np.sqrt(post_var)
        post_mean = post_var * (prior_mean * prior_prec + sum_rewards * obs_prec_factor)

        # Thompson sample
        theta = np.random.normal(post_mean, post_std)
        theta += np.random.uniform(0, 1e-9, theta.shape)
        actions = np.argmax(theta, axis=1)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], reward_std)
        avg_rewards[t] = np.mean(rewards)

        N[idx, actions] += 1.0
        sum_rewards[idx, actions] += rewards

    return avg_rewards, pct_optimal


def simulate_vectorized_gradient(
    num_runs: int = 2000,
    num_arms: int = 10,
    num_steps: int = 1000,
    alpha: float = 0.1,
    use_baseline: bool = True,
    mean_shift: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Gradient Bandit simulation across num_runs parallel bandit problems."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.random.normal(mean_shift, 1.0, size=(num_runs, num_arms))
    best_actions = np.argmax(q_star, axis=1)

    H = np.zeros((num_runs, num_arms), dtype=float)
    R_bar = np.zeros(num_runs, dtype=float)

    avg_rewards = np.zeros(num_steps, dtype=float)
    pct_optimal = np.zeros(num_steps, dtype=float)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        step_num = t + 1

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


def simulate_vectorized_nonstationary(
    num_runs: int = 2000,
    num_arms: int = 10,
    num_steps: int = 10000,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    use_sample_avg: bool = True,
    walk_std: float = 0.01,
    q_init: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized non-stationary bandit where true action values follow random walks."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.zeros((num_runs, num_arms), dtype=float)
    Q = np.full((num_runs, num_arms), q_init, dtype=float)
    N = np.zeros((num_runs, num_arms), dtype=float)

    avg_rewards = np.zeros(num_steps, dtype=float)
    pct_optimal = np.zeros(num_steps, dtype=float)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        q_star += np.random.normal(0.0, walk_std, size=(num_runs, num_arms))
        best_actions = np.argmax(q_star, axis=1)

        explore = np.random.rand(num_runs) < epsilon
        rand_a = np.random.randint(0, num_arms, size=num_runs)
        greedy_a = np.argmax(Q + np.random.uniform(0, 1e-9, Q.shape), axis=1)
        actions = np.where(explore, rand_a, greedy_a)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], 1.0)
        avg_rewards[t] = np.mean(rewards)

        N[idx, actions] += 1.0
        step_size = (1.0 / N[idx, actions]) if use_sample_avg else alpha
        Q[idx, actions] += step_size * (rewards - Q[idx, actions])

    return avg_rewards, pct_optimal


def simulate_vectorized_discounted_ucb(
    num_runs: int = 2000,
    num_arms: int = 10,
    num_steps: int = 10000,
    c: float = 2.0,
    gamma: float = 0.99,
    walk_std: float = 0.01,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Discounted UCB (D-UCB) for non-stationary environments."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.zeros((num_runs, num_arms), dtype=float)
    W = np.zeros((num_runs, num_arms), dtype=float)
    N = np.zeros((num_runs, num_arms), dtype=float)

    avg_rewards = np.zeros(num_steps, dtype=float)
    pct_optimal = np.zeros(num_steps, dtype=float)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        q_star += np.random.normal(0.0, walk_std, size=(num_runs, num_arms))
        best_actions = np.argmax(q_star, axis=1)

        actions = np.zeros(num_runs, dtype=int)
        unpulled_mask = (N < 1e-5)
        has_unpulled = np.any(unpulled_mask, axis=1)
        if np.any(has_unpulled):
            actions[has_unpulled] = np.argmax(unpulled_mask[has_unpulled], axis=1)

        all_pulled = ~has_unpulled
        if np.any(all_pulled):
            n_sub = N[all_pulled]
            w_sub = W[all_pulled]
            q_sub = w_sub / n_sub
            n_total = np.sum(n_sub, axis=1, keepdims=True)
            ucb = q_sub + c * np.sqrt(np.log(n_total) / n_sub)
            ucb += np.random.uniform(0, 1e-9, ucb.shape)
            actions[all_pulled] = np.argmax(ucb, axis=1)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], 1.0)
        avg_rewards[t] = np.mean(rewards)

        N *= gamma
        W *= gamma
        N[idx, actions] += 1.0
        W[idx, actions] += rewards

    return avg_rewards, pct_optimal


def simulate_vectorized_sliding_window_ucb(
    num_runs: int = 2000,
    num_arms: int = 10,
    num_steps: int = 10000,
    c: float = 2.0,
    tau: int = 200,
    walk_std: float = 0.01,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Sliding Window UCB (SW-UCB) for non-stationary environments."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.zeros((num_runs, num_arms), dtype=float)
    actions_history = np.zeros((num_runs, tau), dtype=int)
    rewards_history = np.zeros((num_runs, tau), dtype=float)
    pos = 0

    W = np.zeros((num_runs, num_arms), dtype=float)
    N = np.zeros((num_runs, num_arms), dtype=float)

    avg_rewards = np.zeros(num_steps, dtype=float)
    pct_optimal = np.zeros(num_steps, dtype=float)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        q_star += np.random.normal(0.0, walk_std, size=(num_runs, num_arms))
        best_actions = np.argmax(q_star, axis=1)

        actions = np.zeros(num_runs, dtype=int)
        unpulled_mask = (N == 0)
        has_unpulled = np.any(unpulled_mask, axis=1)
        if np.any(has_unpulled):
            actions[has_unpulled] = np.argmax(unpulled_mask[has_unpulled], axis=1)

        all_pulled = ~has_unpulled
        if np.any(all_pulled):
            n_sub = N[all_pulled]
            w_sub = W[all_pulled]
            q_sub = w_sub / n_sub
            n_total = np.sum(n_sub, axis=1, keepdims=True)
            ucb = q_sub + c * np.sqrt(np.log(n_total) / n_sub)
            ucb += np.random.uniform(0, 1e-9, ucb.shape)
            actions[all_pulled] = np.argmax(ucb, axis=1)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], 1.0)
        avg_rewards[t] = np.mean(rewards)

        if t >= tau:
            old_actions = actions_history[:, pos]
            old_rewards = rewards_history[:, pos]
            N[idx, old_actions] -= 1.0
            W[idx, old_actions] -= old_rewards

        N[idx, actions] += 1.0
        W[idx, actions] += rewards

        actions_history[:, pos] = actions
        rewards_history[:, pos] = rewards
        pos = (pos + 1) % tau

    return avg_rewards, pct_optimal
