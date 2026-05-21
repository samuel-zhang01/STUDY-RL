"""Off-policy evaluation: SNIPS + Doubly Robust + Direct method.

We follow the week-11 notation exactly. A log entry is a tuple
    (x, a, r, p_b, slate)
where:
  x      : context features at decision time
  a      : index *within the slate* of the action picked by the behaviour
           policy (so a in [0, len(slate)-1])
  r      : observed reward (binary hire)
  p_b    : behaviour propensity of picking a, P_b(a | x, slate)
  slate  : the shortlist features (k, d)

A target policy is a callable: pi(slate) -> probability vector over slate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

LogEntry = Tuple[np.ndarray, int, float, float, np.ndarray]


def ips(log: List[LogEntry], pi: Callable[[np.ndarray], np.ndarray]) -> float:
    s = 0.0
    for x, a, r, pb, slate in log:
        p = pi(slate)
        s += (p[a] / max(pb, 1e-8)) * r
    return s / len(log)


def snips(log: List[LogEntry], pi: Callable[[np.ndarray], np.ndarray]) -> float:
    num, den = 0.0, 0.0
    for x, a, r, pb, slate in log:
        w = pi(slate)[a] / max(pb, 1e-8)
        num += w * r
        den += w
    return num / max(den, 1e-8)


def snips_ci(log: List[LogEntry], pi: Callable[[np.ndarray], np.ndarray],
             alpha: float = 0.05) -> Tuple[float, float, float]:
    """Bootstrap CI for SNIPS. Returns (point, lo, hi)."""
    rng = np.random.default_rng(0)
    point = snips(log, pi)
    n = len(log)
    boots = []
    for _ in range(200):
        idx = rng.integers(0, n, size=n)
        boots.append(snips([log[i] for i in idx], pi))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


class RidgeRewardModel:
    """Tiny ridge model q_hat(slate, a) ~ r."""

    def __init__(self, d: int, lam: float = 1.0):
        self.d = d
        self.lam = lam
        self.theta = np.zeros(d)

    def fit(self, log: List[LogEntry]) -> "RidgeRewardModel":
        A = self.lam * np.eye(self.d)
        b = np.zeros(self.d)
        for x, a, r, pb, slate in log:
            phi = slate[a]
            A += np.outer(phi, phi)
            b += r * phi
        self.theta = np.linalg.solve(A, b)
        return self

    def q(self, slate: np.ndarray) -> np.ndarray:
        return slate @ self.theta


def doubly_robust(log: List[LogEntry], pi: Callable, qhat: RidgeRewardModel) -> float:
    s = 0.0
    for x, a, r, pb, slate in log:
        q_vec = qhat.q(slate)
        p_vec = pi(slate)
        baseline = float((p_vec * q_vec).sum())
        correction = (p_vec[a] / max(pb, 1e-8)) * (r - q_vec[a])
        s += baseline + correction
    return s / len(log)


def direct_method(log: List[LogEntry], pi: Callable, qhat: RidgeRewardModel) -> float:
    s = 0.0
    for x, a, r, pb, slate in log:
        q_vec = qhat.q(slate)
        p_vec = pi(slate)
        s += float((p_vec * q_vec).sum())
    return s / len(log)


def effective_sample_size(log: List[LogEntry], pi: Callable) -> float:
    ws = []
    for x, a, r, pb, slate in log:
        ws.append(pi(slate)[a] / max(pb, 1e-8))
    w = np.asarray(ws)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))
