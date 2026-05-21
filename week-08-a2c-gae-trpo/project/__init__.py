"""Mini-project for week 08 -- A2C+GAE on LunarLander-Continuous, plus a TRPO toy.

Target metric (project/train.py): mean episode return on LunarLanderContinuous-v3
trending strictly upward (>= initial+30) within 60s on CPU; on a GPU we converge
to mean return >= 100 in 5min.
"""
from .solutions import (
    compute_gae,
    nstep_advantage,
    diagonal_gaussian_kl,
    conjugate_gradient,
    flat_params,
    set_flat_params,
)
from .agent import A2CAgent
from .train import train

__all__ = [
    "compute_gae",
    "nstep_advantage",
    "diagonal_gaussian_kl",
    "conjugate_gradient",
    "flat_params",
    "set_flat_params",
    "A2CAgent",
    "train",
]
