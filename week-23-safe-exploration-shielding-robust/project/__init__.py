"""Week 23 mini-project: LTL-shielded gridworld + domain-randomised pendulum.

Public API:
    DangerGridWorld   — small gridworld with a designated unsafe region
    DFAShield         — minimal LTL-derived shield (synthesised by hand)
    QLearner          — tabular Q-learning agent
    train_shielded    — train Q-learning under the shield; expects ZERO unsafe transitions
    PendulumDR        — pendulum with randomised pole length
    train_pendulum_dr — train a tiny policy-gradient agent under domain randomisation
"""
from .env_grid import DangerGridWorld
from .shield import DFAShield, build_no_lava_visit_shield
from .q_learn import QLearner, train_shielded, train_unshielded
from .env_pendulum import PendulumDR
from .pendulum_train import train_pendulum_dr

__all__ = [
    "DangerGridWorld",
    "DFAShield",
    "build_no_lava_visit_shield",
    "QLearner",
    "train_shielded",
    "train_unshielded",
    "PendulumDR",
    "train_pendulum_dr",
]
