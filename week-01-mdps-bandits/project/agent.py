"""Bandit agents: epsilon-greedy, UCB1, Thompson Sampling (Beta-Bernoulli),
Boltzmann/softmax, and LinUCB for the contextual case."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ----------------------------- K-armed agents -----------------------------

@dataclass
class EpsilonGreedyAgent:
    K: int
    eps: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.counts = np.zeros(self.K, dtype=int)
        self.means = np.zeros(self.K, dtype=float)

    def select(self, t: int) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.K))
        # Break ties uniformly at random for unbiased behaviour at start.
        best = np.flatnonzero(self.means == self.means.max())
        return int(self.rng.choice(best))

    def update(self, a: int, r: float) -> None:
        self.counts[a] += 1
        self.means[a] += (r - self.means[a]) / self.counts[a]


@dataclass
class UCB1Agent:
    """Auer, Cesa-Bianchi & Fischer 2002. Bonus = sqrt(2 log t / n_a).

    Rewards must lie in [0, 1] for the regret bound to apply literally. For
    Gaussian rewards with known sub-Gaussian parameter sigma, replace 2 by
    2 sigma^2.
    """
    K: int
    c: float = 2.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.counts = np.zeros(self.K, dtype=int)
        self.means = np.zeros(self.K, dtype=float)

    def select(self, t: int) -> int:
        # Play each arm once up front (standard UCB1 initialisation).
        unplayed = np.flatnonzero(self.counts == 0)
        if unplayed.size > 0:
            return int(unplayed[0])
        bonuses = np.sqrt(self.c * np.log(max(t, 2)) / self.counts)
        ucb = self.means + bonuses
        best = np.flatnonzero(ucb == ucb.max())
        return int(self.rng.choice(best))

    def update(self, a: int, r: float) -> None:
        self.counts[a] += 1
        self.means[a] += (r - self.means[a]) / self.counts[a]


@dataclass
class ThompsonBernoulliAgent:
    """Beta(1,1)-Bernoulli Thompson Sampling."""
    K: int
    alpha0: float = 1.0
    beta0: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.alpha = np.full(self.K, self.alpha0, dtype=float)
        self.beta = np.full(self.K, self.beta0, dtype=float)

    def select(self, t: int) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(samples.argmax())

    def update(self, a: int, r: float) -> None:
        # r is 0 or 1.
        self.alpha[a] += r
        self.beta[a] += 1.0 - r


@dataclass
class BoltzmannAgent:
    """Softmax-of-Q with temperature tau. Pure-exploration limit: tau -> infty."""
    K: int
    tau: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.counts = np.zeros(self.K, dtype=int)
        self.means = np.zeros(self.K, dtype=float)

    def select(self, t: int) -> int:
        z = self.means / max(self.tau, 1e-6)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        return int(self.rng.choice(self.K, p=p))

    def update(self, a: int, r: float) -> None:
        self.counts[a] += 1
        self.means[a] += (r - self.means[a]) / self.counts[a]


# ----------------------------- Contextual: LinUCB -----------------------------

@dataclass
class LinUCBAgent:
    """Disjoint LinUCB (Li, Chu, Langford, Schapire 2010).

    Per-arm linear model r = x^T theta_a + noise; confidence
    ellipsoid via the Gram matrix ``A_a`` and response ``b_a``:

        theta_a_hat = A_a^{-1} b_a,
        UCB_a(x) = x^T theta_a_hat + alpha * sqrt(x^T A_a^{-1} x).
    """
    K: int
    d: int
    alpha: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.A = np.stack([np.eye(self.d) for _ in range(self.K)])  # (K, d, d)
        self.b = np.zeros((self.K, self.d))                          # (K, d)

    def select(self, x: np.ndarray, t: int = 0) -> int:
        ucbs = np.empty(self.K)
        for a in range(self.K):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            mean = float(x @ theta_hat)
            bonus = self.alpha * float(np.sqrt(x @ A_inv @ x))
            ucbs[a] = mean + bonus
        best = np.flatnonzero(ucbs == ucbs.max())
        return int(self.rng.choice(best))

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x


# --------------------------- Convenience factory ---------------------------

def make_agent(name: str, K: int, seed: int = 0, **kw) -> object:
    name = name.lower()
    if name in ("eps", "epsilon", "epsilon_greedy"):
        return EpsilonGreedyAgent(K=K, seed=seed, **kw)
    if name == "ucb1":
        return UCB1Agent(K=K, seed=seed, **kw)
    if name in ("ts", "thompson"):
        return ThompsonBernoulliAgent(K=K, seed=seed, **kw)
    if name in ("boltz", "boltzmann", "softmax"):
        return BoltzmannAgent(K=K, seed=seed, **kw)
    raise ValueError(f"unknown agent: {name}")
