"""
Unit Tests for Bandits Library (Sutton & Barto Ch. 2).
"""

import unittest
import numpy as np
from bandits.envs import GaussianBandit, NonstationaryBandit
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
    simulate_vectorized_thompson,
)


class TestBanditEnvironments(unittest.TestCase):
    def test_gaussian_bandit(self):
        env = GaussianBandit(num_arms=10, mean=0.0, variance=1.0)
        self.assertEqual(len(env.q_star), 10)
        reward = env.pull(0)
        self.assertIsInstance(reward, float)
        self.assertEqual(env.best_action, int(np.argmax(env.q_star)))

    def test_nonstationary_bandit(self):
        env = NonstationaryBandit(num_arms=10, walk_std=0.05, initial_value=0.0)
        initial_q = env.q_star.copy()
        env.step()
        self.assertFalse(np.allclose(initial_q, env.q_star))


class TestBanditAgents(unittest.TestCase):
    def test_epsilon_greedy(self):
        agent = EpsilonGreedyAgent(num_arms=10, epsilon=0.1)
        action = agent.select_action()
        self.assertTrue(0 <= action < 10)
        agent.update(action, 2.0)
        self.assertEqual(agent.N[action], 1.0)
        self.assertEqual(agent.Q[action], 2.0)

    def test_ucb_unpulled_first(self):
        agent = UCBAgent(num_arms=5, c=2.0)
        for expected_arm in range(5):
            action = agent.select_action()
            self.assertEqual(action, expected_arm)
            agent.update(action, 1.0)

    def test_thompson_sampling(self):
        agent = GaussianThompsonSamplingAgent(num_arms=5)
        action = agent.select_action()
        self.assertTrue(0 <= action < 5)
        agent.update(action, 1.0)
        self.assertEqual(agent.N[action], 1.0)
        self.assertEqual(agent.sum_rewards[action], 1.0)

    def test_d_ucb(self):
        agent = DiscountedUCBAgent(num_arms=5, gamma=0.9)
        agent.update(0, 10.0)
        agent.update(1, 5.0)
        self.assertAlmostEqual(agent.N[0], 0.9, places=5)
        self.assertAlmostEqual(agent.W[0], 9.0, places=5)

    def test_sw_ucb(self):
        agent = SlidingWindowUCBAgent(num_arms=3, tau=2)
        agent.update(0, 1.0)
        agent.update(1, 2.0)
        agent.update(2, 3.0)  # Should evict the first pull on arm 0
        self.assertEqual(agent.N[0], 0.0)
        self.assertEqual(agent.N[1], 1.0)
        self.assertEqual(agent.N[2], 1.0)


class TestVectorizedEvaluators(unittest.TestCase):
    def test_vectorized_shapes(self):
        num_runs = 50
        num_steps = 100
        r, o = simulate_vectorized_testbed(num_runs=num_runs, num_steps=num_steps, epsilon=0.1)
        self.assertEqual(r.shape, (num_steps,))
        self.assertEqual(o.shape, (num_steps,))
        self.assertTrue(0.0 <= o[-1] <= 100.0)


if __name__ == "__main__":
    unittest.main()
