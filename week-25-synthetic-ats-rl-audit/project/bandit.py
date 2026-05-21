"""Contextual bandit reranker — Linear Thompson Sampling.

State : candidate-job feature vector x (see env.context_features)
Action: pick one candidate from the heuristic top-k shortlist
Reward: binary hire signal sampled from the ground-truth match probability.

We maintain a single shared posterior over a global theta (a la
disjoint-arms LinTS with K=1 effective arm but per-candidate features);
this matches how a Phase-1 shadow reranker would actually be deployed on
top of an existing scorer: one model, scored on each candidate feature.
"""
from __future__ import annotations

import numpy as np


class LinTSReranker:
    def __init__(self, d: int, v2: float = 1.0, lam: float = 1.0, seed: int = 0):
        self.d = d
        self.v2 = v2
        self.lam = lam
        self.A = lam * np.eye(d)
        self.b = np.zeros(d)
        self.rng = np.random.default_rng(seed)

    @property
    def theta_hat(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def sample_theta(self) -> np.ndarray:
        cov = self.v2 * np.linalg.inv(self.A)
        return self.rng.multivariate_normal(self.theta_hat, cov)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Posterior-sample once and score each row of X (a shortlist)."""
        theta = self.sample_theta()
        return X @ theta

    def rerank(self, X: np.ndarray) -> np.ndarray:
        return np.argsort(-self.score(X))

    def update(self, x: np.ndarray, r: float) -> None:
        self.A += np.outer(x, x)
        self.b += r * x


def softmax_propensity(scores: np.ndarray, beta: float = 4.0) -> np.ndarray:
    """Softmax over a shortlist to give a logged behaviour policy with overlap.

    Without overlap (positivity) you cannot do off-policy evaluation — the
    bandit literature is unanimous (week 11). Real ATS deployments must
    therefore inject a small softmax/epsilon perturbation around the
    argmax. We mimic that here so propensities are well-defined.
    """
    z = beta * (scores - scores.max())
    p = np.exp(z)
    return p / p.sum()
