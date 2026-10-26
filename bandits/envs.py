"""
Multi-Armed Bandit Environments (Sutton & Barto Ch. 2).

Provides:
- GaussianBandit: Standard stationary 10-armed Gaussian testbed.
- NonstationaryBandit: Drifting testbed where true action values follow random walks.
- generate_stationary_testbed: Vectorized batch pre-allocation helper.
"""

from typing import Tuple
import numpy as np


class GaussianBandit:
    """Stationary k-armed bandit environment (Sutton & Barto Section 2.2).

    Each arm's true action-value q_*(a) is sampled once from N(mean, variance).
    When an arm is pulled, reward R is returned as N(q_*(a), reward_std^2).
    """

    def __init__(
        self,
        num_arms: int = 10,
        mean: float = 0.0,
        variance: float = 1.0,
        reward_std: float = 1.0,
    ):
        self.num_arms = num_arms
        self.reward_std = reward_std
        # Sample ground-truth action values q_*(a) ~ N(mean, variance)
        self.q_star = np.random.normal(mean, np.sqrt(variance), size=num_arms)
        self.best_action = int(np.argmax(self.q_star))
        self.true_max_value = float(self.q_star[self.best_action])

    def pull(self, action: int) -> float:
        """Sample reward for the selected arm with additive Gaussian noise."""
        if not (0 <= action < self.num_arms):
            raise ValueError(f"Action {action} out of range [0, {self.num_arms})")
        return float(np.random.normal(self.q_star[action], self.reward_std))


class NonstationaryBandit:
    """Non-stationary k-armed bandit environment (Sutton & Barto Exercise 2.5).

    All arms start at initial_value (default 0.0). At every time step, each arm's
    true action-value q_*(a) takes an independent random walk:
        q_*(a) <- q_*(a) + N(0, walk_std^2)
    """

    def __init__(
        self,
        num_arms: int = 10,
        walk_std: float = 0.01,
        initial_value: float = 0.0,
        reward_std: float = 1.0,
    ):
        self.num_arms = num_arms
        self.walk_std = walk_std
        self.reward_std = reward_std
        self.q_star = np.full(num_arms, initial_value, dtype=float)
        self.best_action = int(np.argmax(self.q_star))
        self.true_max_value = float(self.q_star[self.best_action])

    def step(self):
        """Drift all true values q_*(a) by one independent Gaussian random walk step."""
        self.q_star += np.random.normal(0.0, self.walk_std, size=self.num_arms)
        self.best_action = int(np.argmax(self.q_star))
        self.true_max_value = float(self.q_star[self.best_action])

    def pull(self, action: int) -> float:
        """Sample noisy reward from the current (drifting) value of q_*(action)."""
        if not (0 <= action < self.num_arms):
            raise ValueError(f"Action {action} out of range [0, {self.num_arms})")
        return float(np.random.normal(self.q_star[action], self.reward_std))


def generate_stationary_testbed(
    num_runs: int,
    num_arms: int,
    q_star_std: float = 1.0,
    mean: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to pre-generate a batch of stationary bandit environments for vectorized runs."""
    q_star = np.random.normal(mean, q_star_std, size=(num_runs, num_arms))
    best_actions = np.argmax(q_star, axis=1)
    return q_star, best_actions
