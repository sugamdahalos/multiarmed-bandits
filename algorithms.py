"""
Multi-Armed Bandit Agents & Batch Vectorized Simulators (Sutton & Barto Ch. 2)

Implements standard action selection strategies:
1. Epsilon-greedy (sample-average updates)
2. Upper Confidence Bound (UCB)
3. Gradient Bandit (softmax preferences + baseline)
4. Constant step-size alpha (tracking nonstationary rewards)
5. Optimistic initial values (driving early greedy exploration)

Includes vectorized NumPy simulators for running 2,000+ bandit problems in parallel.
"""

import numpy as np
from testbed import GaussianBandit


# ---------------------------------------------------------------------------
# Single-Agent Implementations
# ---------------------------------------------------------------------------

class EpsilonGreedyAgent:
    """Epsilon-greedy agent using incremental sample-average action-value estimates.
    
    With probability eps, picks a uniform random action to explore.
    Otherwise picks argmax Q(a) to exploit (breaking ties randomly).
    """

    def __init__(self, num_arms: int, epsilon: float, q_init: float = 0.0):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.q_init = q_init
        self.Q = np.full(num_arms, q_init)
        self.N = np.zeros(num_arms)

    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_arms)
        else:
            # Pick argmax with random tie-breaking if multiple arms share the max Q
            max_val = np.max(self.Q)
            return np.random.choice(np.flatnonzero(self.Q == max_val))

    def update(self, action: int, reward: float):
        # Incremental sample average update: Q(a) += (1/N) * (R - Q)
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])

    def reset(self):
        self.Q.fill(self.q_init)
        self.N.fill(0.0)


class UCBAgent:
    """Upper Confidence Bound (UCB) agent (Eq 2.10).
    
    Selects action A_t = argmax [ Q(a) + c * sqrt(ln(t) / N(a)) ].
    Unpulled arms (N(a) = 0) are forced first so every arm gets tried once
    before confidence bounds are calculated.
    """

    def __init__(self, num_arms: int, c: float):
        self.num_arms = num_arms
        self.c = c
        self.Q = np.zeros(num_arms)
        self.N = np.zeros(num_arms)
        self.t = 0

    def select_action(self) -> int:
        self.t += 1
        # Try any arm that hasn't been pulled yet
        unpulled = np.flatnonzero(self.N == 0)
        if len(unpulled) > 0:
            return unpulled[0]
        # Calculate UCB bonus for all arms
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return np.random.choice(np.flatnonzero(ucb == np.max(ucb)))

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])

    def reset(self):
        self.Q.fill(0.0)
        self.N.fill(0.0)
        self.t = 0


class GradientBanditAgent:
    """Gradient bandit using softmax action preferences H(a) (Section 2.8).
    
    Action probabilities pi(a) = exp(H(a)) / sum(exp(H(b))).
    Updates preferences based on reward relative to a running-average baseline R_bar.
    """

    def __init__(self, num_arms: int, alpha: float = 0.1,
                 use_baseline: bool = True):
        self.num_arms = num_arms
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.H = np.zeros(num_arms)
        self.r_bar = 0.0
        self.t = 0

    def _softmax(self) -> np.ndarray:
        # Subtract max for numerical stability before exponentiating
        shifted = self.H - np.max(self.H)
        exp_h = np.exp(shifted)
        return exp_h / np.sum(exp_h)

    def select_action(self) -> int:
        return np.random.choice(self.num_arms, p=self._softmax())

    def update(self, action: int, reward: float):
        self.t += 1
        baseline = self.r_bar if self.use_baseline else 0.0
        if self.use_baseline:
            # Update running average reward baseline
            self.r_bar += (1.0 / self.t) * (reward - self.r_bar)
        pi = self._softmax()
        indicator = np.zeros(self.num_arms)
        indicator[action] = 1.0
        # Policy gradient update (Eq 2.12)
        self.H += self.alpha * (reward - baseline) * (indicator - pi)

    def reset(self):
        self.H.fill(0.0)
        self.r_bar = 0.0
        self.t = 0


class ConstantAlphaAgent:
    """Epsilon-greedy agent with constant step-size alpha instead of 1/N(a).
    
    Useful for nonstationary environments: constant alpha gives exponentially
    decaying weights to older rewards, allowing the agent to track moving action values.
    """

    def __init__(self, num_arms: int, epsilon: float, alpha: float,
                 q_init: float = 0.0):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_init = q_init
        self.Q = np.full(num_arms, q_init)
        self.N = np.zeros(num_arms)

    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_arms)
        max_val = np.max(self.Q)
        return np.random.choice(np.flatnonzero(self.Q == max_val))

    def update(self, action: int, reward: float):
        self.N[action] += 1
        # Constant alpha update rule
        self.Q[action] += self.alpha * (reward - self.Q[action])

    def reset(self):
        self.Q.fill(self.q_init)
        self.N.fill(0.0)


class OptimisticGreedyAgent(EpsilonGreedyAgent):
    """Greedy agent (eps = 0) with optimistic Q initialization (e.g. Q_1 = +5).
    
    Why this explores:
    Because real rewards are around N(0, 1), initial rewards (~0) are much lower than +5.
    Every update sharply reduces Q(a) for the arm pulled, prompting the argmax rule to
    try a different untried arm. This forces systematic exploration of all arms in early steps.
    """

    def __init__(self, num_arms: int, q_init: float = 5.0):
        super().__init__(num_arms, epsilon=0.0, q_init=q_init)


# ---------------------------------------------------------------------------
# Scalar Simulator (single run helper)
# ---------------------------------------------------------------------------

def simulate(agent, env, num_steps: int):
    """Run a single agent on a single bandit environment for num_steps."""
    rewards = np.zeros(num_steps)
    actions = np.zeros(num_steps, dtype=int)
    agent.reset()
    for t in range(num_steps):
        a = agent.select_action()
        r = env.pull(a)
        agent.update(a, r)
        rewards[t] = r
        actions[t] = a
    return rewards, actions


# ---------------------------------------------------------------------------
# Vectorized Simulators (runs num_runs problems in parallel via NumPy)
# ---------------------------------------------------------------------------

def simulate_vectorized_testbed(num_runs: int = 2000, num_arms: int = 10,
                                num_steps: int = 1000, epsilon: float = 0.1,
                                q_star_std: float = 1.0, reward_std: float = 1.0,
                                mean: float = 0.0, q_init: float = 0.0,
                                seed: int = None):
    """Vectorized epsilon-greedy simulation across num_runs parallel bandit problems."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.random.normal(mean, q_star_std, size=(num_runs, num_arms))
    best_actions = np.argmax(q_star, axis=1)

    Q = np.full((num_runs, num_arms), q_init)
    N = np.zeros((num_runs, num_arms))

    avg_rewards = np.zeros(num_steps)
    pct_optimal = np.zeros(num_steps)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        # Epsilon-greedy action selection for all runs in parallel
        explore = np.random.rand(num_runs) < epsilon
        rand_a = np.random.randint(0, num_arms, size=num_runs)
        # Add tiny uniform jitter to break ties randomly among equal max Q values
        greedy_a = np.argmax(Q + np.random.uniform(0, 1e-9, Q.shape), axis=1)
        actions = np.where(explore, rand_a, greedy_a)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], reward_std)
        avg_rewards[t] = np.mean(rewards)

        # Incremental sample-average update across all runs
        N[idx, actions] += 1.0
        Q[idx, actions] += (1.0 / N[idx, actions]) * (rewards - Q[idx, actions])

    return avg_rewards, pct_optimal


def simulate_vectorized_ucb(num_runs: int = 2000, num_arms: int = 10,
                            num_steps: int = 1000, c: float = 2.0,
                            q_star_std: float = 1.0, reward_std: float = 1.0,
                            seed: int = None):
    """Vectorized UCB simulation across num_runs parallel bandit problems."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.random.normal(0.0, q_star_std, size=(num_runs, num_arms))
    best_actions = np.argmax(q_star, axis=1)

    Q = np.zeros((num_runs, num_arms))
    N = np.zeros((num_runs, num_arms))

    avg_rewards = np.zeros(num_steps)
    pct_optimal = np.zeros(num_steps)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        step_num = t + 1  # 1-indexed for ln(t) calculation

        actions = np.zeros(num_runs, dtype=int)

        # For runs with unpulled arms, pick the first unpulled arm
        unpulled_mask = (N == 0)
        has_unpulled = np.any(unpulled_mask, axis=1)

        if np.any(has_unpulled):
            actions[has_unpulled] = np.argmax(unpulled_mask[has_unpulled], axis=1)

        # For runs where all arms were tried at least once, use the UCB formula
        all_pulled = ~has_unpulled
        if np.any(all_pulled):
            q_sub = Q[all_pulled]
            n_sub = N[all_pulled]
            ucb = q_sub + c * np.sqrt(np.log(step_num) / n_sub)
            ucb += np.random.uniform(0, 1e-9, ucb.shape)  # Tie-breaker jitter
            actions[all_pulled] = np.argmax(ucb, axis=1)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], 1.0)
        avg_rewards[t] = np.mean(rewards)

        N[idx, actions] += 1.0
        Q[idx, actions] += (1.0 / N[idx, actions]) * (rewards - Q[idx, actions])

    return avg_rewards, pct_optimal


def simulate_vectorized_gradient(num_runs: int = 2000, num_arms: int = 10,
                                 num_steps: int = 1000, alpha: float = 0.1,
                                 use_baseline: bool = True,
                                 mean_shift: float = 0.0,
                                 seed: int = None):
    """Vectorized Gradient Bandit simulation across num_runs parallel bandit problems."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.random.normal(mean_shift, 1.0, size=(num_runs, num_arms))
    best_actions = np.argmax(q_star, axis=1)

    H = np.zeros((num_runs, num_arms))
    R_bar = np.zeros(num_runs)

    avg_rewards = np.zeros(num_steps)
    pct_optimal = np.zeros(num_steps)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        step_num = t + 1

        # Softmax probabilities with numerical stability trick
        H_shifted = H - np.max(H, axis=1, keepdims=True)
        exp_H = np.exp(H_shifted)
        pi = exp_H / np.sum(exp_H, axis=1, keepdims=True)

        # Sample actions according to policy probabilities
        cum_pi = np.cumsum(pi, axis=1)
        u = np.random.rand(num_runs, 1)
        actions = np.argmax(cum_pi > u, axis=1)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], 1.0)
        avg_rewards[t] = np.mean(rewards)

        # Baseline update
        baseline = R_bar.copy() if use_baseline else 0.0
        if use_baseline:
            R_bar += (1.0 / step_num) * (rewards - R_bar)

        # Preference update
        one_hot = np.zeros_like(pi)
        one_hot[idx, actions] = 1.0
        H += alpha * (rewards - baseline)[:, np.newaxis] * (one_hot - pi)

    return avg_rewards, pct_optimal


def simulate_vectorized_nonstationary(num_runs: int = 2000, num_arms: int = 10,
                                      num_steps: int = 10000,
                                      epsilon: float = 0.1,
                                      alpha: float = 0.1,
                                      use_sample_avg: bool = True,
                                      walk_std: float = 0.01,
                                      q_init: float = 0.0,
                                      seed: int = None):
    """Vectorized nonstationary bandit simulator where q_*(a) undergoes a random walk."""
    if seed is not None:
        np.random.seed(seed)

    # All arms start at 0 so no arm has an initial advantage
    q_star = np.zeros((num_runs, num_arms))

    Q = np.full((num_runs, num_arms), q_init)
    N = np.zeros((num_runs, num_arms))

    avg_rewards = np.zeros(num_steps)
    pct_optimal = np.zeros(num_steps)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        # Random walk drift on true action values
        q_star += np.random.normal(0.0, walk_std, size=(num_runs, num_arms))
        best_actions = np.argmax(q_star, axis=1)

        # Epsilon-greedy choice
        explore = np.random.rand(num_runs) < epsilon
        rand_a = np.random.randint(0, num_arms, size=num_runs)
        greedy_a = np.argmax(Q + np.random.uniform(0, 1e-9, Q.shape), axis=1)
        actions = np.where(explore, rand_a, greedy_a)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], 1.0)
        avg_rewards[t] = np.mean(rewards)

        # Update Q values using sample average (1/N) or constant step-size alpha
        N[idx, actions] += 1.0
        if use_sample_avg:
            step_size = 1.0 / N[idx, actions]
        else:
            step_size = alpha
        Q[idx, actions] += step_size * (rewards - Q[idx, actions])

    return avg_rewards, pct_optimal


def simulate_vectorized_optimistic(num_runs: int = 2000, num_arms: int = 10,
                                   num_steps: int = 1000,
                                   epsilon: float = 0.0,
                                   q_init: float = 5.0,
                                   seed: int = None):
    """Vectorized optimistic-initial-values agent in a stationary environment."""
    return simulate_vectorized_testbed(
        num_runs=num_runs, num_arms=num_arms, num_steps=num_steps,
        epsilon=epsilon, q_init=q_init, seed=seed
    )


def simulate_vectorized_optimistic_nonstationary(num_runs: int = 2000, num_arms: int = 10,
                                                 num_steps: int = 10000,
                                                 epsilon: float = 0.0,
                                                 q_init: float = 5.0,
                                                 walk_std: float = 0.01,
                                                 seed: int = None):
    """Vectorized optimistic-initial-values agent in a nonstationary environment."""
    return simulate_vectorized_nonstationary(
        num_runs=num_runs, num_arms=num_arms, num_steps=num_steps,
        epsilon=epsilon, alpha=0.1, use_sample_avg=True, 
        walk_std=walk_std, q_init=q_init, seed=seed
    )


def simulate_vectorized_ucb_nonstationary(num_runs: int = 2000, num_arms: int = 10,
                                          num_steps: int = 10000, c: float = 2.0,
                                          walk_std: float = 0.01,
                                          seed: int = None):
    """Vectorized UCB on a nonstationary drifting bandit testbed."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.zeros((num_runs, num_arms))
    best_actions = np.zeros(num_runs, dtype=int)

    Q = np.zeros((num_runs, num_arms))
    N = np.zeros((num_runs, num_arms))

    avg_rewards = np.zeros(num_steps)
    pct_optimal = np.zeros(num_steps)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        step_num = t + 1

        # Drift action values
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

        # Standard sample average update (fails in nonstationary environments)
        N[idx, actions] += 1.0
        Q[idx, actions] += (1.0 / N[idx, actions]) * (rewards - Q[idx, actions])

    return avg_rewards, pct_optimal


def simulate_vectorized_gradient_nonstationary(num_runs: int = 2000, num_arms: int = 10,
                                               num_steps: int = 10000, alpha: float = 0.1,
                                               use_baseline: bool = True,
                                               walk_std: float = 0.01,
                                               seed: int = None):
    """Vectorized Gradient Bandit on a nonstationary drifting bandit testbed."""
    if seed is not None:
        np.random.seed(seed)

    q_star = np.zeros((num_runs, num_arms))
    best_actions = np.zeros(num_runs, dtype=int)

    H = np.zeros((num_runs, num_arms))
    R_bar = np.zeros(num_runs)

    avg_rewards = np.zeros(num_steps)
    pct_optimal = np.zeros(num_steps)
    idx = np.arange(num_runs)

    for t in range(num_steps):
        step_num = t + 1

        # Drift action values
        q_star += np.random.normal(0.0, walk_std, size=(num_runs, num_arms))
        best_actions = np.argmax(q_star, axis=1)

        # Softmax probabilities
        H_shifted = H - np.max(H, axis=1, keepdims=True)
        exp_H = np.exp(H_shifted)
        pi = exp_H / np.sum(exp_H, axis=1, keepdims=True)

        cum_pi = np.cumsum(pi, axis=1)
        u = np.random.rand(num_runs, 1)
        actions = np.argmax(cum_pi > u, axis=1)

        pct_optimal[t] = np.mean(actions == best_actions) * 100.0

        rewards = np.random.normal(q_star[idx, actions], 1.0)
        avg_rewards[t] = np.mean(rewards)

        # Baseline update using running average
        baseline = R_bar.copy() if use_baseline else 0.0
        if use_baseline:
            R_bar += (1.0 / step_num) * (rewards - R_bar)

        # Preference update uses constant step size alpha
        one_hot = np.zeros_like(pi)
        one_hot[idx, actions] = 1.0
        H += alpha * (rewards - baseline)[:, np.newaxis] * (one_hot - pi)

    return avg_rewards, pct_optimal


def print_run_summary(label: str, agent, env: GaussianBandit,
                      num_steps: int, rewards: np.ndarray):
    """Print summary stats for a single agent run."""
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Steps:           {num_steps:,}")
    print(f"  True Best Arm:   {env.best_action} (q_* = {env.true_max_value:.4f})")
    print(f"  Agent Best Arm:  {np.argmax(agent.Q)} (Q̂ = {agent.Q[np.argmax(agent.Q)]:.4f})")
    print(f"  Mean Reward:     {np.mean(rewards):.4f}")
    print(f"\n  {'Arm':>3}  {'Pulls':>6}  {'q_*(a)':>8}  {'Q̂(a)':>8}")
    print(f"  {'─'*3}  {'─'*6}  {'─'*8}  {'─'*8}")
    for a in range(agent.num_arms):
        print(f"  {a:>3}  {int(agent.N[a]):>6}  {env.q_star[a]:>8.4f}  {agent.Q[a]:>8.4f}")


def simulate_vectorized_discounted_ucb(num_runs: int = 2000, num_arms: int = 10,
                                       num_steps: int = 10000, c: float = 2.0,
                                       gamma: float = 0.99,
                                       walk_std: float = 0.01,
                                       seed: int = None):
    """Vectorized Discounted UCB (D-UCB) for nonstationary environments.
    
    Exponentially decays counts N and rewards W by factor gamma at each step.
    """
    if seed is not None:
        np.random.seed(seed)

    q_star = np.zeros((num_runs, num_arms))
    
    W = np.zeros((num_runs, num_arms))
    N = np.zeros((num_runs, num_arms))

    avg_rewards = np.zeros(num_steps)
    pct_optimal = np.zeros(num_steps)
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

        # Discounted update: multiply previous totals by gamma before adding new observation
        N *= gamma
        W *= gamma
        N[idx, actions] += 1.0
        W[idx, actions] += rewards

    return avg_rewards, pct_optimal


def simulate_vectorized_sliding_window_ucb(num_runs: int = 2000, num_arms: int = 10,
                                           num_steps: int = 10000, c: float = 2.0,
                                           tau: int = 200,
                                           walk_std: float = 0.01,
                                           seed: int = None):
    """Vectorized Sliding Window UCB (SW-UCB) for nonstationary environments.
    
    Only considers the most recent tau observations to compute UCB bounds.
    """
    if seed is not None:
        np.random.seed(seed)

    q_star = np.zeros((num_runs, num_arms))
    
    actions_history = np.zeros((num_runs, tau), dtype=int)
    rewards_history = np.zeros((num_runs, tau))
    pos = 0
    
    W = np.zeros((num_runs, num_arms))
    N = np.zeros((num_runs, num_arms))

    avg_rewards = np.zeros(num_steps)
    pct_optimal = np.zeros(num_steps)
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

        # Sliding window update: drop observation older than tau steps
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
