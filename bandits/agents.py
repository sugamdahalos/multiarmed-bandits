"""
Multi-Armed Bandit Agents (Sutton & Barto Ch. 2 + Modern Extensions).

Implements single-agent OOP interfaces:
1. EpsilonGreedyAgent: Incremental sample-average (1/N) estimator.
2. ConstantAlphaAgent: Constant step-size alpha for exponential recency weighting.
3. OptimisticGreedyAgent: Pure greedy with optimistic initial values (Q1 = +5).
4. UCBAgent: Upper Confidence Bound (UCB1) action selection.
5. DiscountedUCBAgent: D-UCB with exponential discounting for non-stationary tracking.
6. SlidingWindowUCBAgent: SW-UCB with finite horizon window tau.
7. GradientBanditAgent: Softmax policy gradient with baseline invariance.
8. GaussianThompsonSamplingAgent: Bayesian posterior sampling for Gaussian bandits.
"""

from collections import deque
from typing import Optional
import numpy as np


class EpsilonGreedyAgent:
    """Epsilon-greedy agent using incremental sample-average action-value estimates.

    Selects a uniform random arm with probability epsilon (exploration),
    otherwise selects argmax Q(a) breaking ties uniformly at random (exploitation).
    """

    def __init__(self, num_arms: int, epsilon: float, q_init: float = 0.0):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.q_init = q_init
        self.Q = np.full(num_arms, q_init, dtype=float)
        self.N = np.zeros(num_arms, dtype=float)

    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.num_arms))
        max_val = np.max(self.Q)
        best_arms = np.flatnonzero(self.Q == max_val)
        return int(np.random.choice(best_arms))

    def update(self, action: int, reward: float):
        self.N[action] += 1.0
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])

    def reset(self):
        self.Q.fill(self.q_init)
        self.N.fill(0.0)


class ConstantAlphaAgent:
    """Epsilon-greedy agent with constant step-size alpha.

    Weights past rewards exponentially: w_i = alpha * (1 - alpha)^{t - i}.
    Enables tracking of non-stationary reward distributions.
    """

    def __init__(self, num_arms: int, epsilon: float, alpha: float, q_init: float = 0.0):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_init = q_init
        self.Q = np.full(num_arms, q_init, dtype=float)
        self.N = np.zeros(num_arms, dtype=float)

    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.num_arms))
        max_val = np.max(self.Q)
        best_arms = np.flatnonzero(self.Q == max_val)
        return int(np.random.choice(best_arms))

    def update(self, action: int, reward: float):
        self.N[action] += 1.0
        self.Q[action] += self.alpha * (reward - self.Q[action])

    def reset(self):
        self.Q.fill(self.q_init)
        self.N.fill(0.0)


class OptimisticGreedyAgent(EpsilonGreedyAgent):
    """Greedy agent (epsilon=0) with optimistic Q-value initialization.

    High initial estimates force systematic exploration in early steps
    without paying an asymptotic exploration tax.
    """

    def __init__(self, num_arms: int, q_init: float = 5.0):
        super().__init__(num_arms=num_arms, epsilon=0.0, q_init=q_init)


class UCBAgent:
    """Upper Confidence Bound (UCB1) action selection agent (Eq 2.10).

    Selects A_t = argmax_a [ Q(a) + c * sqrt(ln(t) / N(a)) ].
    Guarantees every arm is pulled once (N(a) = 0) before evaluating bounds.
    """

    def __init__(self, num_arms: int, c: float = 2.0):
        self.num_arms = num_arms
        self.c = c
        self.Q = np.zeros(num_arms, dtype=float)
        self.N = np.zeros(num_arms, dtype=float)
        self.t = 0

    def select_action(self) -> int:
        self.t += 1
        unpulled = np.flatnonzero(self.N == 0)
        if len(unpulled) > 0:
            return int(unpulled[0])
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        max_val = np.max(ucb)
        best_arms = np.flatnonzero(ucb == max_val)
        return int(np.random.choice(best_arms))

    def update(self, action: int, reward: float):
        self.N[action] += 1.0
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])

    def reset(self):
        self.Q.fill(0.0)
        self.N.fill(0.0)
        self.t = 0


class DiscountedUCBAgent:
    """Discounted UCB (D-UCB) for non-stationary environments (Garivier & Moulines, 2011).

    Applies discount factor gamma in (0, 1) to past counts and rewards:
        N_t(gamma, a) = sum_{s=1}^t gamma^{t-s} * I(A_s = a)
        W_t(gamma, a) = sum_{s=1}^t gamma^{t-s} * R_s * I(A_s = a)
    """

    def __init__(self, num_arms: int, c: float = 2.0, gamma: float = 0.99):
        self.num_arms = num_arms
        self.c = c
        self.gamma = gamma
        self.W = np.zeros(num_arms, dtype=float)
        self.N = np.zeros(num_arms, dtype=float)

    def select_action(self) -> int:
        unpulled = np.flatnonzero(self.N < 1e-5)
        if len(unpulled) > 0:
            return int(unpulled[0])
        q_est = self.W / self.N
        n_total = np.sum(self.N)
        ucb = q_est + self.c * np.sqrt(np.log(n_total) / self.N)
        max_val = np.max(ucb)
        best_arms = np.flatnonzero(ucb == max_val)
        return int(np.random.choice(best_arms))

    def update(self, action: int, reward: float):
        self.N *= self.gamma
        self.W *= self.gamma
        self.N[action] += 1.0
        self.W[action] += reward

    def reset(self):
        self.W.fill(0.0)
        self.N.fill(0.0)


class SlidingWindowUCBAgent:
    """Sliding-Window UCB (SW-UCB) for non-stationary environments (Garivier & Moulines, 2011).

    Maintains a sliding window of the most recent tau observations.
    """

    def __init__(self, num_arms: int, c: float = 2.0, tau: int = 200):
        self.num_arms = num_arms
        self.c = c
        self.tau = tau
        self.history = deque(maxlen=tau)
        self.W = np.zeros(num_arms, dtype=float)
        self.N = np.zeros(num_arms, dtype=float)

    def select_action(self) -> int:
        unpulled = np.flatnonzero(self.N == 0)
        if len(unpulled) > 0:
            return int(unpulled[0])
        q_est = self.W / self.N
        n_total = np.sum(self.N)
        ucb = q_est + self.c * np.sqrt(np.log(n_total) / self.N)
        max_val = np.max(ucb)
        best_arms = np.flatnonzero(ucb == max_val)
        return int(np.random.choice(best_arms))

    def update(self, action: int, reward: float):
        if len(self.history) == self.tau:
            old_action, old_reward = self.history.popleft()
            self.N[old_action] -= 1.0
            self.W[old_action] -= old_reward
        self.history.append((action, reward))
        self.N[action] += 1.0
        self.W[action] += reward

    def reset(self):
        self.history.clear()
        self.W.fill(0.0)
        self.N.fill(0.0)


class GradientBanditAgent:
    """Gradient Bandit using softmax action preferences H(a) (Section 2.8).

    Action probabilities: pi(a) = exp(H(a)) / sum_b exp(H(b)).
    Update: H(a) <- H(a) + alpha * (R - baseline) * (I(a=A) - pi(a)).
    """

    def __init__(self, num_arms: int, alpha: float = 0.1, use_baseline: bool = True):
        self.num_arms = num_arms
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.H = np.zeros(num_arms, dtype=float)
        self.r_bar = 0.0
        self.t = 0

    def _softmax(self) -> np.ndarray:
        shifted = self.H - np.max(self.H)
        exp_h = np.exp(shifted)
        return exp_h / np.sum(exp_h)

    def select_action(self) -> int:
        return int(np.random.choice(self.num_arms, p=self._softmax()))

    def update(self, action: int, reward: float):
        self.t += 1
        baseline = self.r_bar if self.use_baseline else 0.0
        if self.use_baseline:
            self.r_bar += (1.0 / self.t) * (reward - self.r_bar)
        pi = self._softmax()
        indicator = np.zeros(self.num_arms, dtype=float)
        indicator[action] = 1.0
        self.H += self.alpha * (reward - baseline) * (indicator - pi)

    def reset(self):
        self.H.fill(0.0)
        self.r_bar = 0.0
        self.t = 0


class GaussianThompsonSamplingAgent:
    """Thompson Sampling with Gaussian Conjugate Prior.

    Prior on true value: q_*(a) ~ N(mu_0, sigma_0^2).
    Observed rewards: R ~ N(q_*(a), sigma_reward^2).

    Posterior:
        mu_{post}(a) = (mu_0/sigma_0^2 + sum(R_a)/sigma_reward^2) / (1/sigma_0^2 + N(a)/sigma_reward^2)
        var_{post}(a) = 1 / (1/sigma_0^2 + N(a)/sigma_reward^2)

    Action selection samples theta_a ~ N(mu_{post}(a), var_{post}(a)) and picks argmax.
    """

    def __init__(
        self,
        num_arms: int,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        reward_std: float = 1.0,
    ):
        self.num_arms = num_arms
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.reward_variance = reward_std ** 2
        self.reward_std = reward_std

        self.N = np.zeros(num_arms, dtype=float)
        self.sum_rewards = np.zeros(num_arms, dtype=float)

    def select_action(self) -> int:
        # Precision formulation
        prior_prec = 1.0 / self.prior_variance
        obs_prec = self.N / self.reward_variance
        post_var = 1.0 / (prior_prec + obs_prec)
        post_mean = post_var * (self.prior_mean * prior_prec + self.sum_rewards / self.reward_variance)

        # Posterior Thompson sample
        theta = np.random.normal(post_mean, np.sqrt(post_var))
        max_val = np.max(theta)
        best_arms = np.flatnonzero(theta == max_val)
        return int(np.random.choice(best_arms))

    def update(self, action: int, reward: float):
        self.N[action] += 1.0
        self.sum_rewards[action] += reward

    def reset(self):
        self.N.fill(0.0)
        self.sum_rewards.fill(0.0)
