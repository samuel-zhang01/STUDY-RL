"""Mini-project for week 10 -- SAC on Pendulum-v1, with automatic temperature.

Target metric (project/train.py): smoothed episode return >= -300 within 8k env
steps, under 5min on a single consumer GPU (under 60s on CPU at reduced budget).
"""
from .solutions import (
    tanh_gaussian_log_prob,
    soft_bellman_target,
    twin_critic_min,
    polyak_update,
    auto_temperature_loss,
)
from .agent import SACAgent
from .train import train

__all__ = [
    "tanh_gaussian_log_prob",
    "soft_bellman_target",
    "twin_critic_min",
    "polyak_update",
    "auto_temperature_loss",
    "SACAgent",
    "train",
]
