"""Baseline heuristic scorer for locally generated candidate and job data.

This teaching fixture combines skill similarity and experience fit using
illustrative weights; it does not reproduce an external system. It is
intentionally a hand-chosen rule. The
contextual bandit on top (`bandit.py`) gets to *re-rank* the top-k slate
produced by this scorer.
"""
from __future__ import annotations

import numpy as np

from .env import ATSData, _safe_norm


def heuristic_score(data: ATSData) -> np.ndarray:
    """Return (N, J) heuristic match score in [0, 100]."""
    cand_n = _safe_norm(data.candidates)
    jobs_n = _safe_norm(data.jobs)
    sim = cand_n @ jobs_n.T                                    # in [-1, 1]
    sim01 = 0.5 * (sim + 1.0)                                  # [0,1]
    sen_gap = np.abs(data.cand_years[:, None] / 10.0 - data.job_seniority[None, :])
    fit = np.exp(-sen_gap)                                     # [0,1]
    s = 100.0 * (0.7 * sim01 + 0.3 * fit)
    return s


def top_k_shortlist(scores: np.ndarray, job_idx: int, k: int = 10) -> np.ndarray:
    """Return top-k candidates sorted by descending heuristic score."""
    col = scores[:, job_idx]
    if not 1 <= k <= len(col):
        raise ValueError("k must be between 1 and the candidate count")
    return np.argsort(-col, kind="stable")[:k]


def heuristic_feature_score(features: np.ndarray) -> np.ndarray:
    """Exact heuristic score from env.context_features (normalised products)."""
    similarity = features[:, :-4].sum(axis=-1)
    fit = np.exp(-features[:, -2])
    return 100.0 * (0.7 * (similarity + 1.0) / 2.0 + 0.3 * fit)
