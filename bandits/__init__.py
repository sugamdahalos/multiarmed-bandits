"""
Multi-Armed Bandit Benchmark Library (Sutton & Barto Ch. 2).
"""

from bandits.envs import (
    GaussianBandit,
    NonstationaryBandit,
    generate_stationary_testbed,
)
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
    simulate_vectorized_gradient,
    simulate_vectorized_nonstationary,
    simulate_vectorized_discounted_ucb,
    simulate_vectorized_sliding_window_ucb,
)

__all__ = [
    "GaussianBandit",
    "NonstationaryBandit",
    "generate_stationary_testbed",
    "EpsilonGreedyAgent",
    "ConstantAlphaAgent",
    "OptimisticGreedyAgent",
    "UCBAgent",
    "DiscountedUCBAgent",
    "SlidingWindowUCBAgent",
    "GradientBanditAgent",
    "GaussianThompsonSamplingAgent",
    "simulate",
    "simulate_vectorized_testbed",
    "simulate_vectorized_ucb",
    "simulate_vectorized_thompson",
    "simulate_vectorized_gradient",
    "simulate_vectorized_nonstationary",
    "simulate_vectorized_discounted_ucb",
    "simulate_vectorized_sliding_window_ucb",
]
