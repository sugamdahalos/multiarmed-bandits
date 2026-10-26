"""
Backward compatibility shim for testbed environments.
Imports from bandits.envs.
"""

from bandits.envs import (
    GaussianBandit,
    NonstationaryBandit,
    generate_stationary_testbed,
)

__all__ = [
    "GaussianBandit",
    "NonstationaryBandit",
    "generate_stationary_testbed",
]
