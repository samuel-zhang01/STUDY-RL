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
    The position-based model requires stable item relevance and logging
    coverage of every item the target can expose. Missing coverage is an
    identification failure, including when no click happens to be observed.
    """
    clicks = np.asarray(clicks, dtype=float)
    ranking = np.asarray(ranking)
    target_ranking = np.asarray(target_ranking)
    pos_bias = np.asarray(pos_bias, dtype=float)
    if (ranking.ndim != 1 or not len(ranking) or ranking.dtype.kind not in "iu"
            or target_ranking.ndim != 1
            or (len(target_ranking) and target_ranking.dtype.kind not in "iu")
            or clicks.shape != ranking.shape or not np.isfinite(clicks).all()
            or not np.isin(clicks, [0, 1]).all()
            or len(np.unique(ranking)) != len(ranking)
            or len(np.unique(target_ranking)) != len(target_ranking)
            or (ranking < 0).any() or (target_ranking < 0).any()):
        raise ValueError("Need unique nonnegative item IDs and aligned binary clicks")
    if (pos_bias.ndim != 1 or len(pos_bias) < max(len(ranking), len(target_ranking))
            or not np.isfinite(pos_bias).all()
            or (pos_bias < 0).any() or (pos_bias > 1).any()):
        raise ValueError("Examination probabilities must cover both rankings and lie in [0,1]")
    logged_positions = {int(item): k for k, item in enumerate(ranking)}
    if any(clicks[k] and pos_bias[k] == 0 for k in range(len(ranking))):
        raise ValueError("A click at zero examination probability is an impossible logged row")
    for k, item in enumerate(target_ranking):
        if pos_bias[k] > 0 and (int(item) not in logged_positions
                               or pos_bias[logged_positions[int(item)]] == 0):
            raise ValueError("Target exposure requires positive logged examination support for every item")
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
                val += pos_bias[kp] / pos_bias[k]
    return val


def expected_relevance(ranking: np.ndarray, relevance: np.ndarray,
                       pos_bias: np.ndarray) -> float:
    K = min(len(ranking), len(pos_bias))
    return float((relevance[ranking[:K]] * pos_bias[:K]).sum())
