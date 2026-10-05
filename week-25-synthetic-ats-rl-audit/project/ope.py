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


def _checked_entries(log, pi):
    if not log:
        raise ValueError("OPE requires a nonempty log")
    for x, a, r, pb, slate in log:
        p = np.asarray(pi(slate), dtype=float)
        if (p.shape != (len(slate),) or not np.isfinite(p).all()
                or (p < 0).any() or not np.isclose(p.sum(), 1.0)):
            raise ValueError("Target policy must return a probability vector over the slate")
        if not 0 <= a < len(slate) or not np.isfinite(pb) or not 0 < pb <= 1 or not np.isfinite(r):
            raise ValueError("Logged action, reward, or propensity is invalid; positivity is required")
        yield x, a, r, pb, slate, p


def ips(log: List[LogEntry], pi: Callable[[np.ndarray], np.ndarray]) -> float:
    s = 0.0
    for x, a, r, pb, slate, p in _checked_entries(log, pi):
        s += (p[a] / pb) * r
    return s / len(log)


def snips(log: List[LogEntry], pi: Callable[[np.ndarray], np.ndarray]) -> float:
    num, den = 0.0, 0.0
    for x, a, r, pb, slate, p in _checked_entries(log, pi):
        w = p[a] / pb
        num += w * r
        den += w
    if den == 0:
        raise ValueError("SNIPS is undefined with zero total target weight")
    return num / den


def snips_ci(log: List[LogEntry], pi: Callable[[np.ndarray], np.ndarray],
             alpha: float = 0.05) -> Tuple[float, float, float]:
    """IID-row percentile bootstrap for a FIXED target; returns (point, lo, hi).

    No coverage guarantee is supplied for adaptive/clustered logs or weak
    overlap. Freeze the target before calling; do not re-fit on bootstrap rows.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
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
    for x, a, r, pb, slate, p_vec in _checked_entries(log, pi):
        q_vec = qhat.q(slate)
        baseline = float((p_vec * q_vec).sum())
        correction = (p_vec[a] / pb) * (r - q_vec[a])
        s += baseline + correction
    return s / len(log)


def direct_method(log: List[LogEntry], pi: Callable, qhat: RidgeRewardModel) -> float:
    s = 0.0
    for x, a, r, pb, slate, p_vec in _checked_entries(log, pi):
        q_vec = qhat.q(slate)
        s += float((p_vec * q_vec).sum())
    return s / len(log)


def effective_sample_size(log: List[LogEntry], pi: Callable) -> float:
    ws = []
    for x, a, r, pb, slate, p in _checked_entries(log, pi):
        ws.append(p[a] / pb)
    w = np.asarray(ws)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))
