"""Counterfactual Learning-to-Rank utilities.

Implements the position-bias model (Joachims, Swaminathan, Schnabel WSDM
2017) and IPS-LTR + SNIPS-LTR pointwise objectives. Click model: with
probability p_k (examination at position k) the user inspects the item at
position k, then clicks if the item is relevant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
import numpy as np


def position_bias(K: int, eta: float = 1.0) -> np.ndarray:
    """Standard position-bias prior: p_k = (1/(k+1))^eta."""
    return (1.0 / (np.arange(1, K + 1))) ** eta


def simulate_clicks(
    relevance: np.ndarray,
    ranking: np.ndarray,
    pos_bias: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Generate Bernoulli clicks from the examination model.

    Returns a binary array of length len(ranking).
    """
    rng = np.random.default_rng(seed)
    rels = relevance[ranking]
    K = len(ranking)
    p = pos_bias[:K] * rels
    return rng.binomial(1, p)


def ips_ltr_value(
    clicks: np.ndarray,
    ranking: np.ndarray,
    pos_bias: np.ndarray,
    target_ranking: np.ndarray,
) -> float:
    """IPS-LTR estimate of the target ranking's expected click sum.

    For each clicked item, weight by target examination probability divided
    by its logged examination probability, if the item is in the target.
    """
    val = 0.0
    K = len(ranking)
    target_pos = {int(item): pos for pos, item in enumerate(target_ranking)}
    for k in range(K):
        item = int(ranking[k])
        if clicks[k] == 0:
            continue
        if item in target_pos:
            kp = target_pos[item]
            if kp < len(pos_bias):
                val += pos_bias[kp] / max(pos_bias[k], 1e-9)
    return val


def expected_relevance(ranking: np.ndarray, relevance: np.ndarray,
                       pos_bias: np.ndarray) -> float:
    K = min(len(ranking), len(pos_bias))
    return float((relevance[ranking[:K]] * pos_bias[:K]).sum())
