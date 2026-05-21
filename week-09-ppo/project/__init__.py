"""Mini-project for week 09 -- PPO with the 37 implementation details, on
BipedalWalker-v3.

Target metric (project/train.py): mean episode return on BipedalWalker-v3
trends strictly upward (>= +30) within 60s on CPU; on GPU we converge to
mean return >= 100 within 5min.
"""
from .solutions import (
    ppo_clip_loss,
    ppo_value_clip_loss,
    approx_kl,
    orthogonal_init_,
    adaptive_kl_penalty_update,
)
from .agent import PPOAgent
from .train import train

__all__ = [
    "ppo_clip_loss",
    "ppo_value_clip_loss",
    "approx_kl",
    "orthogonal_init_",
    "adaptive_kl_penalty_update",
    "PPOAgent",
    "train",
]
