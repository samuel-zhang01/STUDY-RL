"""Week 22 mini-project: PPO-Lagrangian on a constrained gridworld with lava cost.

Public API:
    LavaGridWorld     — constrained gridworld where stepping into lava incurs cost
    PPOLagrangian     — PPO with primal-dual Lagrangian constraint on expected cost
    train             — entry point (CPU, < 60s)
"""
from .env import LavaGridWorld
from .agent import PPOLagrangian, PPOLagrangianConfig
from .train import train

__all__ = ["LavaGridWorld", "PPOLagrangian", "PPOLagrangianConfig", "train"]
