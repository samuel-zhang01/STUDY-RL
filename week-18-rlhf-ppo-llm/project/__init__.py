"""Mini-project for week 18 — RLHF & PPO for Language Models.

Self-contained, from-scratch PyTorch implementation of the three RLHF
stages (SFT, reward model, PPO) on a tiny character-level transformer.
Designed to converge to a non-trivial signal in well under a minute on
CPU and in seconds on a single GPU.
"""

from .model import CharTransformer, ValueHead, RewardModel
from .data import ToyPreferenceDataset, ALPHABET, encode, decode
from .sft import sft_train
from .rm import rm_train, reward_loss
from .ppo import ppo_train, ppo_step, compute_gae
from . import solutions  # noqa: F401

__all__ = [
    "CharTransformer",
    "ValueHead",
    "RewardModel",
    "ToyPreferenceDataset",
    "ALPHABET",
    "encode",
    "decode",
    "sft_train",
    "rm_train",
    "reward_loss",
    "ppo_train",
    "ppo_step",
    "compute_gae",
]
