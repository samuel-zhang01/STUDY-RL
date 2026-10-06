"""Bandit environments for week 1.

Three testbeds:

1. ``BernoulliBandit`` — K arms with means in [0, 1].
2. ``GaussianBandit`` — K arms with means drawn N(0, 1), unit-variance rewards.
3. ``SyntheticLeadBrokerBandit`` — synthetic contextual bandit imitating insurance
   lead routing: at each round a *lead* arrives with a feature vector ``x in R^d``;
   the agent picks one of K *brokers*; the reward is a Bernoulli "bind" event
   with logits ``x^T theta_a + b_a``. The latent ``theta_a`` matrix is the
   ground truth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class BernoulliBandit:
    means: np.ndarray  # shape (K,)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=float)
        assert ((self.means >= 0) & (self.means <= 1)).all()

    @property
    def K(self) -> int:
        return int(self.means.shape[0])

    @property
    def best_mean(self) -> float:
        return float(self.means.max())

    def pull(self, a: int) -> float:
        return float(self.rng.random() < self.means[a])

    def regret(self, a: int) -> float:
        return float(self.best_mean - self.means[a])


@dataclass
class GaussianBandit:
    means: np.ndarray
    sigma: float = 1.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=float)

    @property
    def K(self) -> int:
        return int(self.means.shape[0])

    @property
    def best_mean(self) -> float:
        return float(self.means.max())

    def pull(self, a: int) -> float:
        return float(self.rng.normal(self.means[a], self.sigma))

    def regret(self, a: int) -> float:
        return float(self.best_mean - self.means[a])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SyntheticLeadBrokerBandit:
    """Synthetic contextual bandit modelling synthetic insurance-style lead routing.

    State: a feature vector ``x`` per lead (postcode-hash, vehicle, prior-claim,
    derived embedding...). Action: pick one of K brokers. Reward: Bernoulli
    "bind" with probability ``sigmoid(x . theta_a + b_a)``.

    The latent broker parameters ``theta`` are drawn N(0, 1); each broker has
    a bias ``b_a`` reflecting baseline conversion. The optimal broker depends
    on the lead context, so non-contextual policies leave significant regret.
    """
    K: int = 8
    d: int = 6
    seed: int = 0
    bias_scale: float = 0.5
    theta_scale: float = 0.8

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.theta = self.theta_scale * self.rng.standard_normal((self.K, self.d))
        self.b = self.bias_scale * self.rng.standard_normal(self.K)

    def context(self) -> np.ndarray:
        x = self.rng.standard_normal(self.d) * 0.7
        # First feature is a constant 1.0 (intercept absorbed into bias).
        x[0] = 1.0
        return x

    def expected_reward(self, x: np.ndarray) -> np.ndarray:
        logits = self.theta @ x + self.b
        return _sigmoid(logits)

    def pull(self, x: np.ndarray, a: int) -> float:
        p = float(self.expected_reward(x)[a])
        return float(self.rng.random() < p)

    def best_action(self, x: np.ndarray) -> int:
        return int(self.expected_reward(x).argmax())

    def instantaneous_regret(self, x: np.ndarray, a: int) -> float:
        er = self.expected_reward(x)
        return float(er.max() - er[a])
