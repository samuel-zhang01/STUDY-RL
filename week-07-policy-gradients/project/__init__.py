"""Mini-project for week 07 -- REINFORCE with learned value baseline.

Target metric (project/train.py): mean episode return on CartPole-v1 >= 195 within
500 episodes for the REINFORCE+baseline variant, under 60s on CPU.
"""
from .solutions import (
    discount_cumsum,
    log_prob_normal,
    reinforce_loss,
    reinforce_with_baseline_loss,
    optimal_baseline,
    topk_offpolicy_multiplier,
    snips_estimator,
)
from .agent import REINFORCEAgent
from .train import train

__all__ = [
    "discount_cumsum",
    "log_prob_normal",
    "reinforce_loss",
    "reinforce_with_baseline_loss",
    "optimal_baseline",
    "topk_offpolicy_multiplier",
    "snips_estimator",
    "REINFORCEAgent",
    "train",
]
