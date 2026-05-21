"""Fairness audit utilities for a ranking system.

We implement:
- demographic parity gap: |P(top-k | A=F) - P(top-k | A=M)|
- equalised opportunity gap: |TPR_F - TPR_M| (TPR = P(top-k | qualified))
- exposure (Singh & Joachims 2018): expected position-bias-weighted exposure
  by group; the gap is the producer/consumer-fairness primitive
- individual fairness Lipschitz check: for each pair of candidates with
  identical skills but different gender-coded names, are their model scores
  the same?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np


def demographic_parity_gap(scores: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
    """`scores` is (n_jobs, n_candidates); we compute the fraction of top-k
    slots filled by group F vs group M, average over jobs."""
    n_jobs = scores.shape[0]
    rates_f, rates_m = [], []
    for j in range(n_jobs):
        order = np.argsort(-scores[j])[:k]
        g = groups[order]
        rates_f.append(np.mean(g == "F"))
        rates_m.append(np.mean(g == "M"))
    return float(abs(np.mean(rates_f) - np.mean(rates_m)))


def equalised_opportunity_gap(
    scores: np.ndarray, groups: np.ndarray, qualified: np.ndarray, k: int = 10
) -> float:
    """`qualified[j, c]` is a 0/1 ground-truth indicator for whether candidate
    c is *qualified* for job j (true match >= threshold). Compute the gap in
    P(in-top-k | qualified) across groups."""
    n_jobs = scores.shape[0]
    tpr_f, tpr_m = [], []
    for j in range(n_jobs):
        order = np.argsort(-scores[j])[:k]
        top_set = set(order.tolist())
        q = qualified[j]
        f_mask = (groups == "F") & (q == 1)
        m_mask = (groups == "M") & (q == 1)
        if f_mask.sum() > 0:
            tpr_f.append(np.mean([c in top_set for c in np.where(f_mask)[0]]))
        if m_mask.sum() > 0:
            tpr_m.append(np.mean([c in top_set for c in np.where(m_mask)[0]]))
    return float(abs(np.mean(tpr_f) - np.mean(tpr_m)))


def exposure_by_group(
    scores: np.ndarray, groups: np.ndarray, eta: float = 1.0
) -> Tuple[float, float]:
    """Position-bias-weighted exposure aggregated by group (averaged over jobs).
    Returns (exposure_F, exposure_M).
    """
    n_jobs, n_cands = scores.shape
    pos_bias = (1.0 / np.arange(1, n_cands + 1)) ** eta
    exp_f = exp_m = 0.0
    for j in range(n_jobs):
        order = np.argsort(-scores[j])
        for k, c in enumerate(order):
            if groups[c] == "F":
                exp_f += pos_bias[k]
            else:
                exp_m += pos_bias[k]
    n_f = int((groups == "F").sum())
    n_m = int((groups == "M").sum())
    return float(exp_f / max(n_f, 1) / n_jobs), float(exp_m / max(n_m, 1) / n_jobs)


def individual_fairness_check(
    score_fn: Callable[[np.ndarray], float],
    cand_vec: np.ndarray,
    noise_scale: float = 0.0,
) -> float:
    """For an identical skill vector, are the model's scores invariant to
    gender-coded perturbations? Returns max |delta|. (For a model that takes
    only the skill vector this is trivially 0; the check is useful when the
    model has been augmented with name features.)"""
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(8):
        v2 = cand_vec + rng.normal(0, noise_scale, size=cand_vec.shape)
        deltas.append(abs(score_fn(v2) - score_fn(cand_vec)))
    return float(max(deltas))


def gender_swap_audit(scores_with_gender: Callable[[np.ndarray, str], float],
                      cand_vec: np.ndarray) -> float:
    """The flagship audit. Given a scoring function that takes a candidate
    vector and a gender label, return the absolute score gap when we swap
    F <-> M. For a fair model, this is 0."""
    s_f = scores_with_gender(cand_vec, "F")
    s_m = scores_with_gender(cand_vec, "M")
    return float(abs(s_f - s_m))
