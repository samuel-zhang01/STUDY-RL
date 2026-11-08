"""Off-Policy Evaluation: IPS, SNIPS, Direct Method, Doubly Robust, SWITCH-DR.

All estimators consume a logged dataset

    log = list of (x_i, a_i, r_i, p_b(a_i | x_i))   for i = 1 .. n

and a target policy `pi` exposed as `pi(x) -> probability vector over K arms`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np
from common.statistics import importance_weights


Logged = Tuple[np.ndarray, int, float, float]   # (x, a, r, p_b)


class UndefinedOverlapError(ValueError):
    """A valid finite log has zero total observed target-policy weight."""


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _checked_entries(log, pi):
    """Validate observed rows; this cannot establish unobserved population support."""
    if not log:
        raise ValueError("OPE requires a nonempty evaluation log")
    for x, a, r, pb in log:
        p = np.asarray(pi(x), dtype=float)
        if (p.ndim != 1 or not len(p) or not np.isfinite(p).all()
                or (p < 0).any() or not np.isclose(p.sum(), 1.0)):
            raise ValueError("target policy must return a finite probability vector")
        if (not isinstance(a, (int, np.integer)) or not 0 <= a < len(p)
                or not np.isfinite(r)):
            raise ValueError("logged action and reward must be valid")
        weight = float(importance_weights([p[a]], [pb])[0])
        yield x, a, float(r), p, weight


def _predictions(qhat, x, probabilities):
    q = np.asarray(qhat(x), dtype=float)
    if q.shape != probabilities.shape or not np.isfinite(q).all():
        raise ValueError("reward predictions must be finite and match the action space")
    return q

def ips(log: List[Logged], pi: Callable[[np.ndarray], np.ndarray]) -> float:
    n = len(log)
    s = 0.0
    for x, a, r, p, w in _checked_entries(log, pi):
        s += r*w
    return s / n


def snips(log: List[Logged], pi: Callable[[np.ndarray], np.ndarray]) -> float:
    num, den = 0.0, 0.0
    for x, a, r, p, w in _checked_entries(log, pi):
        num += w * r
        den += w
    if den == 0:
        raise UndefinedOverlapError("SNIPS is undefined with zero observed target overlap")
    return num / den


def direct_method(
    log: List[Logged],
    pi: Callable[[np.ndarray], np.ndarray],
    qhat: Callable[[np.ndarray], np.ndarray],
) -> float:
    s = 0.0
    for x, _a, _r, p, _w in _checked_entries(log, pi):
        s += float(p @ _predictions(qhat, x, p))
    return s / len(log)


def doubly_robust(
    log: List[Logged],
    pi: Callable[[np.ndarray], np.ndarray],
    qhat: Callable[[np.ndarray], np.ndarray],
) -> float:
    n = len(log)
    s = 0.0
    for x, a, r, pix, w in _checked_entries(log, pi):
        q = _predictions(qhat, x, pix)
        baseline = float(pix @ q)
        s += baseline + w * (r - float(q[a]))
    return s / n


def switch_dr(
    log: List[Logged],
    pi: Callable[[np.ndarray], np.ndarray],
    qhat: Callable[[np.ndarray], np.ndarray],
    tau: float = 10.0,
) -> float:
    """SWITCH-DR (Wang, Agarwal, Dudik 2017): use DR when w<=tau, else direct."""
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and nonnegative")
    n = len(log)
    s = 0.0
    for x, a, r, pix, w in _checked_entries(log, pi):
        q = _predictions(qhat, x, pix)
        baseline = float(pix @ q)
        if w <= tau:
            s += baseline + w * (r - float(q[a]))
        else:
            s += baseline
    return s / n


# ---------------------------------------------------------------------------
# Reward model used by DR / Direct
# ---------------------------------------------------------------------------

def independent_logged_sample(env, n=400, seed=0):
    """Independent uniform-logger rows from the same SYNTHETIC mechanisms.

    This deliberately separates nuisance fitting from target evaluation. It
    uses the simulator's reward means to sample new outcomes; ordinary offline
    data must instead provide a justified independent fitting split/design.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive number of fitting rows")
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        x = rng.normal(size=env.d)
        x[-1] = 1.
        x /= max(1., np.linalg.norm(x))
        a = int(rng.integers(env.K))
        r = float(rng.binomial(1, env.mean_reward(x, a)))
        rows.append((x, a, r, 1./env.K))
    return rows

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
    """IID-row plug-in variance of an IPS mean, not of adaptive logging streams."""
    vals = np.array([r*w for x, a, r, p, w in _checked_entries(log, pi)])
    if len(vals) < 2:
        raise ValueError("variance needs at least two independent evaluation rows")
    return float(vals.var(ddof=1) / len(vals))


def dr_variance(
    log: List[Logged],
    pi: Callable[[np.ndarray], np.ndarray],
    qhat: Callable[[np.ndarray], np.ndarray],
) -> float:
    vals = []
    for x, a, r, pix, w in _checked_entries(log, pi):
        q = _predictions(qhat, x, pix)
        baseline = float(pix @ q)
        vals.append(baseline + w * (r - float(q[a])))
    vals = np.array(vals)
    if len(vals) < 2:
        raise ValueError("variance needs at least two independent evaluation rows")
    return float(vals.var(ddof=1) / len(vals))


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
