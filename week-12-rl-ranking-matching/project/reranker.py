"""LinTS contextual-bandit re-ranker that sits on top of the two-tower
retriever. Per-job (per-requisition) Bayesian linear head over the candidate
skill vector. The bandit explores in the space of top-N retrieved
candidates --- this is the standard 2025 production architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class LinTSReranker:
    """One linear posterior per job. Features are (skill_vec || two_tower_score)."""

    n_jobs: int
    d: int
    v: float = 0.5    # Posterior scale
    lam: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.A = np.stack([self.lam * np.eye(self.d) for _ in range(self.n_jobs)])
        self.b = np.zeros((self.n_jobs, self.d), dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)

    def _sample(self, j: int) -> np.ndarray:
        Ainv = np.linalg.inv(self.A[j])
        mu = Ainv @ self.b[j]
        cov = (self.v ** 2) * Ainv
        cov = 0.5 * (cov + cov.T) + 1e-6 * np.eye(self.d)
        return self._rng.multivariate_normal(mu, cov)

    def score_topN(self, j: int, feats: np.ndarray) -> np.ndarray:
        """Return TS score for each row in `feats` (top-N candidate features)."""
        theta = self._sample(j)
        return feats @ theta

    def update(self, j: int, x: np.ndarray, r: float) -> None:
        self.A[j] += np.outer(x, x)
        self.b[j] += r * x

    def mean_score(self, j: int, feats: np.ndarray) -> np.ndarray:
        Ainv = np.linalg.inv(self.A[j])
        mu = Ainv @ self.b[j]
        return feats @ mu


def make_features(ds, jid: int, cid: int, tt_score: float) -> np.ndarray:
    """Feature for the reranker: candidate-skill bag concatenated with the
    two-tower similarity and a constant bias."""
    skills = ds.cand_vecs[cid]
    return np.concatenate([skills, [tt_score, 1.0]])
