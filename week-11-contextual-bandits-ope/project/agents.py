"""Contextual bandit agents: UCB1, KL-UCB, Thompson Sampling, LinUCB, LinTS.

All linear agents use the disjoint-arms parameterisation (one design matrix
A_a = lambda*I + sum_t x_t x_t^T per arm), which keeps the math identical to
the original LinUCB paper and avoids hybrid-arm bookkeeping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Stochastic (non-contextual) bandits
# ---------------------------------------------------------------------------

@dataclass
class UCB1:
    K: int

    def __post_init__(self) -> None:
        self.n = np.zeros(self.K, dtype=np.int64)
        self.mu = np.zeros(self.K, dtype=np.float64)
        self.t = 0

    def act(self) -> int:
        self.t += 1
        # Force one pull of each arm initially.
        for a in range(self.K):
            if self.n[a] == 0:
                return a
        bonus = np.sqrt(2.0 * np.log(self.t) / self.n)
        return int(np.argmax(self.mu + bonus))

    def update(self, a: int, r: float) -> None:
        self.n[a] += 1
        self.mu[a] += (r - self.mu[a]) / self.n[a]


def kl_bernoulli(p: float, q: float) -> float:
    eps = 1e-9
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


@dataclass
class KLUCB:
    K: int
    c: float = 3.0

    def __post_init__(self) -> None:
        self.n = np.zeros(self.K, dtype=np.int64)
        self.mu = np.zeros(self.K, dtype=np.float64)
        self.t = 0

    def _upper(self, mu_a: float, n_a: int) -> float:
        # Bisection for q in [mu_a, 1] with n_a*KL(mu_a, q) <= log(t)+c*log(log(t))
        if n_a == 0:
            return 1.0
        rhs = (np.log(self.t) + self.c * np.log(max(np.log(self.t), 1.0))) / n_a
        lo, hi = mu_a, 1.0 - 1e-9
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            if kl_bernoulli(mu_a, mid) > rhs:
                hi = mid
            else:
                lo = mid
        return lo

    def act(self) -> int:
        self.t += 1
        for a in range(self.K):
            if self.n[a] == 0:
                return a
        ub = np.array([self._upper(self.mu[a], int(self.n[a])) for a in range(self.K)])
        return int(np.argmax(ub))

    def update(self, a: int, r: float) -> None:
        self.n[a] += 1
        self.mu[a] += (r - self.mu[a]) / self.n[a]


@dataclass
class BetaThompson:
    K: int
    alpha0: float = 1.0
    beta0: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.alpha = np.full(self.K, self.alpha0, dtype=np.float64)
        self.beta = np.full(self.K, self.beta0, dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)

    def act(self) -> int:
        samples = self._rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, a: int, r: float) -> None:
        self.alpha[a] += r
        self.beta[a] += (1.0 - r)


@dataclass
class MOSS:
    """Minimax Optimal Strategy in the Stochastic case (Audibert & Bubeck 2009).

    Bonus uses log(max(1, T/(K n_a))) which removes the spurious log(T) factor
    and gives O(sqrt(KT)) worst-case regret.
    """

    K: int
    horizon: int

    def __post_init__(self) -> None:
        self.n = np.zeros(self.K, dtype=np.int64)
        self.mu = np.zeros(self.K, dtype=np.float64)

    def act(self) -> int:
        for a in range(self.K):
            if self.n[a] == 0:
                return a
        bonus = np.sqrt(np.maximum(np.log(self.horizon / (self.K * self.n)), 0.0) / self.n)
        return int(np.argmax(self.mu + bonus))

    def update(self, a: int, r: float) -> None:
        self.n[a] += 1
        self.mu[a] += (r - self.mu[a]) / self.n[a]


# ---------------------------------------------------------------------------
# Contextual linear bandits
# ---------------------------------------------------------------------------

@dataclass
class LinUCB:
    K: int
    d: int
    alpha: float = 1.0  # exploration radius
    lam: float = 1.0    # ridge regulariser

    def __post_init__(self) -> None:
        self.A = np.stack([self.lam * np.eye(self.d) for _ in range(self.K)])
        self.b = np.zeros((self.K, self.d), dtype=np.float64)

    def theta_hat(self, a: int) -> np.ndarray:
        return np.linalg.solve(self.A[a], self.b[a])

    def score(self, x: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.K)
        for a in range(self.K):
            Ainv_x = np.linalg.solve(self.A[a], x)
            mu = float(self.b[a] @ Ainv_x)  # = x.T @ A^{-1} @ b  (closed-form ridge)
            ucb = self.alpha * np.sqrt(max(float(x @ Ainv_x), 0.0))
            scores[a] = mu + ucb
        return scores

    def act(self, x: np.ndarray) -> int:
        return int(np.argmax(self.score(x)))

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x

    def policy_prob(self, x: np.ndarray, a: int) -> float:
        """Greedy w.r.t. UCB score; returns 1.0 for argmax arm, 0 otherwise."""
        return 1.0 if self.act(x) == a else 0.0


@dataclass
class LinTS:
    K: int
    d: int
    v: float = 0.25  # posterior-scale ("v^2" in Agrawal & Goyal 2013)
    lam: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.A = np.stack([self.lam * np.eye(self.d) for _ in range(self.K)])
        self.b = np.zeros((self.K, self.d), dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)

    def _posterior(self, a: int) -> Tuple[np.ndarray, np.ndarray]:
        Ainv = np.linalg.inv(self.A[a])
        mu = Ainv @ self.b[a]
        cov = (self.v ** 2) * Ainv
        # Symmetrise for numerical safety.
        cov = 0.5 * (cov + cov.T)
        return mu, cov

    def sample_theta(self, a: int) -> np.ndarray:
        mu, cov = self._posterior(a)
        return self._rng.multivariate_normal(mu, cov)

    def act(self, x: np.ndarray) -> int:
        scores = np.zeros(self.K)
        for a in range(self.K):
            theta = self.sample_theta(a)
            scores[a] = x @ theta
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x

    def policy_prob(self, x: np.ndarray, a: int, n_mc: int = 64) -> float:
        """Monte-Carlo estimate of P(LinTS picks a | x)."""
        wins = 0
        for _ in range(n_mc):
            scores = np.array([x @ self.sample_theta(k) for k in range(self.K)])
            if int(np.argmax(scores)) == a:
                wins += 1
        return wins / n_mc


@dataclass
class EpsilonGreedyLogger:
    """Behaviour policy used to generate the offline log: eps-greedy on the
    current ridge-regression mean. Records propensities pi_b(a|x) so OPE
    estimators have unbiased weights.
    """

    K: int
    d: int
    eps: float = 0.25
    lam: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.A = np.stack([self.lam * np.eye(self.d) for _ in range(self.K)])
        self.b = np.zeros((self.K, self.d), dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)

    def mean_scores(self, x: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.K)
        for a in range(self.K):
            theta = np.linalg.solve(self.A[a], self.b[a])
            scores[a] = x @ theta
        return scores

    def propensities(self, x: np.ndarray) -> np.ndarray:
        scores = self.mean_scores(x)
        greedy = int(np.argmax(scores))
        p = np.full(self.K, self.eps / self.K)
        p[greedy] += 1.0 - self.eps
        return p

    def act(self, x: np.ndarray) -> Tuple[int, float]:
        p = self.propensities(x)
        a = int(self._rng.choice(self.K, p=p))
        return a, float(p[a])

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x
