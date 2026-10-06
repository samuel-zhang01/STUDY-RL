"""Synthetic synthetic insurance lead-broker contextual bandit environment.

Each round draws a lead context x in R^d (d=8 features that you can read as
postcode-band, vehicle-power, prior-claims, quote-channel, etc.). There are
K=6 broker arms; each broker has a hidden linear preference vector theta_a in
R^d. The conversion (bind) probability for broker a on lead x is

    p(x, a) = sigmoid( x @ theta_a )

and the realised binary reward r ~ Bernoulli(p(x, a)). This is the simplest
non-trivial *logistic* contextual bandit. LinUCB / LinTS fit linear reward
models here, so their linear-payoff regret theorems do not directly apply to
this misspecified testbed. Empirical regret remains well defined.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SyntheticLeadBandit:
    d: int = 8
    K: int = 6
    seed: int = 0
    noise_std: float = 0.1  # observation noise on the Bernoulli-rate logit

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # Hidden broker preference vectors. Spread broker quality so there is a
        # clear ordering on average but contexts can flip it on individual leads.
        base = rng.normal(0, 0.6, size=(self.K, self.d))
        # Inject differential expertise: broker k specialises in feature k % d.
        for k in range(self.K):
            base[k, k % self.d] += 1.0
        # Quality offsets (broker 0 best on average, broker K-1 worst).
        offsets = np.linspace(0.6, -0.6, self.K).reshape(-1, 1)
        self.theta = base + offsets * np.ones((self.K, self.d)) * 0.2
        self._rng = np.random.default_rng(self.seed + 1)

    # --- context drawing ----------------------------------------------------
    def sample_context(self) -> np.ndarray:
        # Sub-Gaussian feature distribution. Last coordinate is a bias.
        x = self._rng.normal(0, 1.0, size=self.d).astype(np.float64)
        x[-1] = 1.0  # intercept
        x = x / max(1.0, np.linalg.norm(x))  # bounded contexts: ||x|| <= 1
        return x

    # --- reward / expected reward ------------------------------------------
    def mean_reward(self, x: np.ndarray, a: int) -> float:
        logit = float(x @ self.theta[a])
        return 1.0 / (1.0 + np.exp(-logit))

    def all_means(self, x: np.ndarray) -> np.ndarray:
        logits = self.theta @ x
        return 1.0 / (1.0 + np.exp(-logits))

    def step(self, x: np.ndarray, a: int) -> float:
        p = self.mean_reward(x, a)
        return float(self._rng.binomial(1, p))

    def optimal_action(self, x: np.ndarray) -> int:
        return int(np.argmax(self.all_means(x)))

    def optimal_mean(self, x: np.ndarray) -> float:
        return float(self.all_means(x).max())
