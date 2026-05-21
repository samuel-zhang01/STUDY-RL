"""Off-Policy Evaluation: IPS, SNIPS, Direct Method, Doubly Robust, SWITCH-DR.

All estimators consume a logged dataset

    log = list of (x_i, a_i, r_i, p_b(a_i | x_i))   for i = 1 .. n

and a target policy `pi` exposed as `pi(x) -> probability vector over K arms`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np


Logged = Tuple[np.ndarray, int, float, float]   # (x, a, r, p_b)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def ips(log: List[Logged], pi: Callable[[np.ndarray], np.ndarray]) -> float:
    n = len(log)
    s = 0.0
    for x, a, r, pb in log:
        s += r * pi(x)[a] / max(pb, 1e-12)
    return s / n


def snips(log: List[Logged], pi: Callable[[np.ndarray], np.ndarray]) -> float:
    num, den = 0.0, 0.0
    for x, a, r, pb in log:
        w = pi(x)[a] / max(pb, 1e-12)
        num += w * r
        den += w
    return num / max(den, 1e-12)


def direct_method(
    log: List[Logged],
    pi: Callable[[np.ndarray], np.ndarray],
    qhat: Callable[[np.ndarray], np.ndarray],
) -> float:
    s = 0.0
    for x, _a, _r, _pb in log:
        s += float(pi(x) @ qhat(x))
    return s / len(log)


def doubly_robust(
    log: List[Logged],
    pi: Callable[[np.ndarray], np.ndarray],
    qhat: Callable[[np.ndarray], np.ndarray],
) -> float:
    n = len(log)
    s = 0.0
    for x, a, r, pb in log:
        pix = pi(x)
        baseline = float(pix @ qhat(x))
        w = pix[a] / max(pb, 1e-12)
        s += baseline + w * (r - float(qhat(x)[a]))
    return s / n


def switch_dr(
    log: List[Logged],
    pi: Callable[[np.ndarray], np.ndarray],
    qhat: Callable[[np.ndarray], np.ndarray],
    tau: float = 10.0,
) -> float:
    """SWITCH-DR (Wang, Agarwal, Dudik 2017): use DR when w<=tau, else direct."""
    n = len(log)
    s = 0.0
    for x, a, r, pb in log:
        pix = pi(x)
        w = pix[a] / max(pb, 1e-12)
        baseline = float(pix @ qhat(x))
        if w <= tau:
            s += baseline + w * (r - float(qhat(x)[a]))
        else:
            s += baseline
    return s / n


# ---------------------------------------------------------------------------
# Reward model used by DR / Direct
# ---------------------------------------------------------------------------

@dataclass
class RidgeRewardModel:
    """One ridge regression per arm, fit on the logged data."""

    K: int
    d: int
    lam: float = 1.0

    def fit(self, log: List[Logged]) -> "RidgeRewardModel":
        A = np.stack([self.lam * np.eye(self.d) for _ in range(self.K)])
        b = np.zeros((self.K, self.d))
        for x, a, r, _pb in log:
            A[a] += np.outer(x, x)
            b[a] += r * x
        self.theta = np.stack([np.linalg.solve(A[a], b[a]) for a in range(self.K)])
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.theta @ x


# ---------------------------------------------------------------------------
# Variance estimates for sanity-checking
# ---------------------------------------------------------------------------

def ips_variance(log: List[Logged], pi: Callable[[np.ndarray], np.ndarray]) -> float:
    vals = np.array([r * pi(x)[a] / max(pb, 1e-12) for (x, a, r, pb) in log])
    return float(vals.var(ddof=1) / max(len(vals), 1))


def dr_variance(
    log: List[Logged],
    pi: Callable[[np.ndarray], np.ndarray],
    qhat: Callable[[np.ndarray], np.ndarray],
) -> float:
    vals = []
    for x, a, r, pb in log:
        pix = pi(x)
        baseline = float(pix @ qhat(x))
        w = pix[a] / max(pb, 1e-12)
        vals.append(baseline + w * (r - float(qhat(x)[a])))
    vals = np.array(vals)
    return float(vals.var(ddof=1) / max(len(vals), 1))


# ---------------------------------------------------------------------------
# Ground-truth on-policy value (only available because env is synthetic)
# ---------------------------------------------------------------------------

def true_policy_value(
    env, pi: Callable[[np.ndarray], np.ndarray], n: int = 5000, seed: int = 42
) -> float:
    rng = np.random.default_rng(seed)
    # Sample fresh contexts; do not consume env state.
    s = 0.0
    for _ in range(n):
        x = rng.normal(0, 1.0, size=env.d)
        x[-1] = 1.0
        x = x / max(1.0, np.linalg.norm(x))
        means = env.all_means(x)
        s += float(pi(x) @ means)
    return s / n
