"""
Multi-Armed Bandit Environments (Sutton & Barto Ch. 2)

Implements:
1. GaussianBandit: Stationary testbed where q_*(a) ~ N(mean, 1) is fixed.
2. NonstationaryBandit: True values q_*(a) take independent random walks at each step.
"""

import numpy as np


class GaussianBandit:
    """Stationary k-armed bandit.
    
    Each arm's true action-value q_*(a) is sampled once from N(mean, variance).
    When an arm is pulled, reward R is returned as N(q_*(a), reward_std^2).
    """

    def __init__(self, num_arms: int = 10, mean: float = 0.0,
                 variance: float = 1.0, reward_std: float = 1.0):
        self.num_arms = num_arms
        self.reward_std = reward_std
        # Sample ground-truth action values q_*(a)
        self.q_star = np.random.normal(mean, np.sqrt(variance), size=num_arms)
        self.best_action = int(np.argmax(self.q_star))
        self.true_max_value = self.q_star[self.best_action]

    def pull(self, action: int) -> float:
        """Sample reward for the selected arm with additive Gaussian noise."""
        if not (0 <= action < self.num_arms):
            raise ValueError(f"Action {action} out of range [0, {self.num_arms})")
        return np.random.normal(self.q_star[action], self.reward_std)


class NonstationaryBandit:
    """Nonstationary bandit (Exercise 2.5).
    
    All q_*(a) start at initial_value (default 0.0). At every step, each arm's
    true value drifts by adding an independent N(0, walk_std) step.
    
    Useful for testing if step-size alpha can track a moving target better than
    a sample-average estimator.
    """

    def __init__(self, num_arms: int = 10, walk_std: float = 0.01,
                 initial_value: float = 0.0, reward_std: float = 1.0):
        self.num_arms = num_arms
        self.walk_std = walk_std
        self.reward_std = reward_std
        self.q_star = np.full(num_arms, initial_value, dtype=float)
        self.best_action = int(np.argmax(self.q_star))
        self.true_max_value = self.q_star[self.best_action]

    def step(self):
        """Drift all true values q_*(a) by one random walk step."""
        self.q_star += np.random.normal(0.0, self.walk_std, size=self.num_arms)
        self.best_action = int(np.argmax(self.q_star))
        self.true_max_value = self.q_star[self.best_action]

    def pull(self, action: int) -> float:
        """Sample noisy reward from current (drifting) value of q_*(action)."""
        return np.random.normal(self.q_star[action], self.reward_std)


def generate_stationary_testbed(num_runs: int, num_arms: int,
                                q_star_std: float = 1.0,
                                mean: float = 0.0) -> tuple:
    """Helper to pre-generate a batch of stationary bandit environments for vectorized runs."""
    q_star = np.random.normal(mean, q_star_std, size=(num_runs, num_arms))
    best_actions = np.argmax(q_star, axis=1)
    return q_star, best_actions


if __name__ == "__main__":
    # Quick sanity check: sample reward from best arm 10,000 times and check convergence to true q_*
    print("=" * 60)
    print(" TASK B1: Testbed Sanity Check")
    print("=" * 60)

    np.random.seed(42)
    env = GaussianBandit(num_arms=10)
    n_pulls = 10_000

    rewards = np.array([env.pull(env.best_action) for _ in range(n_pulls)])
    sample_mean = np.mean(rewards)
    sample_std = np.std(rewards, ddof=1)
    error = abs(env.true_max_value - sample_mean)

    print(f"\nTrue q_* values: {np.round(env.q_star, 4)}")
    print(f"Optimal Arm:   {env.best_action}  (q_* = {env.true_max_value:.4f})")
    print(f"Sample Mean:   {sample_mean:.4f}  (over {n_pulls:,} pulls)")
    print(f"Sample StdDev: {sample_std:.4f}  (expected ≈ 1.0)")
    print(f"|Error|:       {error:.4f}  (expected < 0.05)")

    assert error < 0.05, f"Sanity check failed: |error| = {error:.4f}"
    print("\n✅ Sanity check PASSED — sample mean within 0.05 of true value.")