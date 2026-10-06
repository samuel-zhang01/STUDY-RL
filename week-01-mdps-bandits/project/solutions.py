"""Reference solutions for the Week 1 workbook exercises.

Each function name matches an ``exercise_N`` cell in ``workbook.ipynb``.
The workbook tests import these and check correctness on fixed seeds.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from .env import BernoulliBandit, GaussianBandit, SyntheticLeadBrokerBandit
from .agent import (
    EpsilonGreedyAgent,
    UCB1Agent,
    ThompsonBernoulliAgent,
    BoltzmannAgent,
    LinUCBAgent,
)


# ============================================================================
# Exercise 1 (star) — sample-average reward update
# ============================================================================
def exercise_1_incremental_mean(rewards: np.ndarray) -> np.ndarray:
    """Return the running sample-average after each observation.

    Implements ``Q_{n+1} = Q_n + (1/n) (R_n - Q_n)``.
    """
    rewards = np.asarray(rewards, dtype=float)
    out = np.zeros_like(rewards)
    Q = 0.0
    for n, r in enumerate(rewards, start=1):
        Q = Q + (r - Q) / n
        out[n - 1] = Q
    return out


# ============================================================================
# Exercise 2 (star) — Bernoulli regret of epsilon-greedy
# ============================================================================
def exercise_2_eps_greedy_regret(
    means: np.ndarray, T: int = 500, eps: float = 0.1, seed: int = 0
) -> np.ndarray:
    """Run epsilon-greedy and return cumulative regret per step (length T)."""
    env = BernoulliBandit(means=np.asarray(means, dtype=float),
                          rng=np.random.default_rng(seed))
    ag = EpsilonGreedyAgent(K=env.K, eps=eps, seed=seed)
    out = np.zeros(T)
    cum = 0.0
    for t in range(T):
        a = ag.select(t + 1)
        r = env.pull(a)
        ag.update(a, r)
        cum += env.regret(a)
        out[t] = cum
    return out


# ============================================================================
# Exercise 3 (star star) — UCB1 confidence bonus
# ============================================================================
def exercise_3_ucb1_bonus(counts: np.ndarray, t: int) -> np.ndarray:
    """Hoeffding bonus sqrt(2 log t / n_a). Unplayed arms get +inf."""
    counts = np.asarray(counts, dtype=float)
    bonus = np.where(counts > 0, np.sqrt(2.0 * np.log(max(t, 2)) / np.maximum(counts, 1)),
                     np.inf)
    return bonus


# ============================================================================
# Exercise 4 (star star) — Thompson Sampling posterior update
# ============================================================================
def exercise_4_thompson_posterior(
    prior_alpha: float, prior_beta: float, rewards: np.ndarray
) -> Tuple[float, float]:
    """Beta(a,b) posterior after a sequence of Bernoulli rewards."""
    s = float(np.sum(rewards))
    n = float(len(rewards))
    return prior_alpha + s, prior_beta + (n - s)


# ============================================================================
# Exercise 5 (star star) — algorithm bake-off, final regret
# ============================================================================
def exercise_5_bakeoff(
    means: np.ndarray, T: int = 1000, n_seeds: int = 10
) -> Dict[str, float]:
    """Return final cumulative regret per agent averaged across seeds."""
    means = np.asarray(means, dtype=float)
    K = len(means)
    res: Dict[str, List[float]] = {"eps-greedy(0.1)": [], "UCB1": [], "Thompson": []}
    for seed in range(n_seeds):
        env = BernoulliBandit(means=means, rng=np.random.default_rng(50 + seed))
        agents = {
            "eps-greedy(0.1)": EpsilonGreedyAgent(K=K, eps=0.1, seed=seed),
            "UCB1": UCB1Agent(K=K, c=2.0, seed=seed),
            "Thompson": ThompsonBernoulliAgent(K=K, seed=seed),
        }
        for agent_idx, (name, ag) in enumerate(agents.items()):
            env_copy = BernoulliBandit(
                means=means, rng=np.random.default_rng(50 + seed * 13 + agent_idx)
            )
            cum = 0.0
            for t in range(T):
                a = ag.select(t + 1)
                r = env_copy.pull(a)
                ag.update(a, r)
                cum += env_copy.regret(a)
            res[name].append(cum)
    return {k: float(np.mean(v)) for k, v in res.items()}


# ============================================================================
# Exercise 6 (star star) — MDP return computation
# ============================================================================
def exercise_6_discounted_return(rewards: np.ndarray, gamma: float) -> float:
    """G = sum_{k=0}^{T-1} gamma^k * r_k."""
    r = np.asarray(rewards, dtype=float)
    if r.size == 0:
        return 0.0
    discounts = gamma ** np.arange(r.size)
    return float((discounts * r).sum())


# ============================================================================
# Exercise 7 (star star) — LinUCB on the synthetic insurance contextual bandit
# ============================================================================
def exercise_7_linucb_regret(
    K: int = 6, d: int = 5, T: int = 1500, alpha: float = 1.0, seed: int = 0
) -> Tuple[float, float]:
    """Return (LinUCB final regret, eps-greedy final regret)."""
    env = SyntheticLeadBrokerBandit(K=K, d=d, seed=seed)
    lin = LinUCBAgent(K=K, d=d, alpha=alpha, seed=seed)
    eps = EpsilonGreedyAgent(K=K, eps=0.1, seed=seed)
    cum_lin = 0.0
    cum_eps = 0.0
    for t in range(T):
        x = env.context()
        a_lin = lin.select(x, t + 1)
        a_eps = eps.select(t + 1)
        r_lin = env.pull(x, a_lin)
        r_eps = env.pull(x, a_eps)
        lin.update(x, a_lin, r_lin)
        eps.update(a_eps, r_eps)
        cum_lin += env.instantaneous_regret(x, a_lin)
        cum_eps += env.instantaneous_regret(x, a_eps)
    return cum_lin, cum_eps


# ============================================================================
# Exercise 8 (star star star, open-ended) — SNIPS off-policy value estimator
# ============================================================================
def exercise_8_snips(
    rewards: np.ndarray, target_prob: np.ndarray, logged_prob: np.ndarray
) -> float:
    """Self-Normalised IPS estimate of E_{pi}[r] from logs of pi_b.

    SNIPS(pi) = sum_i w_i r_i / sum_i w_i  where w_i = pi(a_i|x_i) / pi_b(a_i|x_i).
    Returns 0.0 if the denominator is zero (vacuous estimate).
    """
    w = np.asarray(target_prob) / np.maximum(np.asarray(logged_prob), 1e-12)
    num = float((w * np.asarray(rewards)).sum())
    den = float(w.sum())
    return num / den if den > 0 else 0.0
