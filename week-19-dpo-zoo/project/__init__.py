"""Mini-project for week 19 -- DPO and the Preference Zoo.

A self-contained PyTorch implementation of the DPO family on a tiny
character-level transformer: DPO, IPO, KTO, ORPO, SimPO and CPO.
The educational aim is to make the connection between the
KL-regularised RL objective (week 18) and each variant's loss as
transparent as possible.
"""

from .losses import (
    dpo_loss,
    ipo_loss,
    kto_loss,
    orpo_loss,
    simpo_loss,
    cpo_loss,
)
from .data import ToyPreferenceDataset, encode, decode
from .model import CharTransformer, TransformerConfig
from .train import train_variant, train, bakeoff
from . import solutions  # noqa: F401

__all__ = [
    "dpo_loss",
    "ipo_loss",
    "kto_loss",
    "orpo_loss",
    "simpo_loss",
    "cpo_loss",
    "ToyPreferenceDataset",
    "encode",
    "decode",
    "CharTransformer",
    "TransformerConfig",
    "train_variant",
    "train",
    "bakeoff",
]
