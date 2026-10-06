"""Reference solutions for the workbook exercises (week 11).

Each function takes whatever inputs the workbook hands it and returns the
reference answer. The workbook test cells call these and assert close-ness.
"""
from __future__ import annotations

from typing import Callable, List, Tuple
import numpy as np

from .agents import UCB1, KLUCB, BetaThompson, LinUCB, LinTS, EpsilonGreedyLogger, kl_bernoulli
from .env import SyntheticLeadBandit
from .ope import (
    ips,
    snips,
    direct_method,
    doubly_robust,
    switch_dr,
    RidgeRewardModel,
)


# E1 — Hoeffding -> UCB1 bonus derivation (numerical).
def exercise_1_ucb_bonus(t: int, n_a: int) -> float:
    """Return the UCB1 exploration bonus sqrt(2 log t / n_a)."""
    return float(np.sqrt(2.0 * np.log(t) / n_a))


# E2 — KL-UCB upper confidence: solve for q.
def exercise_2_klucb_upper(mu_hat: float, n_a: int, t: int, c: float = 3.0) -> float:
    rhs = (np.log(t) + c * np.log(max(np.log(t), 1.0))) / n_a
    lo, hi = mu_hat, 1.0 - 1e-9
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(mu_hat, mid) > rhs:
            hi = mid
        else:
            lo = mid
    return float(lo)


# E3 — Bernoulli Thompson on a 3-armed bandit.
def exercise_3_thompson_regret(T: int = 1000, seed: int = 0) -> float:
    means = np.array([0.2, 0.5, 0.8])
    rng = np.random.default_rng(seed)
    agent = BetaThompson(K=3, seed=seed)
    reg = 0.0
    for _ in range(T):
        a = agent.act()
        r = float(rng.binomial(1, means[a]))
        agent.update(a, r)
        reg += means.max() - means[a]
    return reg


# E4 — Train LinUCB on the synthetic insurance bandit and return final cumulative regret.
def exercise_4_linucb_regret(T: int = 1000, seed: int = 0) -> float:
    env = SyntheticLeadBandit(seed=seed)
    agent = LinUCB(K=env.K, d=env.d, alpha=1.0)
    reg = 0.0
    for _ in range(T):
        x = env.sample_context()
        a = agent.act(x)
        r = env.step(x, a)
        agent.update(x, a, r)
        reg += env.optimal_mean(x) - env.mean_reward(x, a)
    return reg


# E5 — Collect a small offline log and compute IPS / SNIPS for a uniform pi.
def exercise_5_ips_snips_uniform(n: int = 500, seed: int = 0) -> Tuple[float, float]:
    env = SyntheticLeadBandit(seed=seed)
    logger = EpsilonGreedyLogger(K=env.K, d=env.d, eps=0.3, seed=seed + 1)
    log = []
    for _ in range(n):
        x = env.sample_context()
        a, pb = logger.act(x)
        r = env.step(x, a)
        logger.update(x, a, r)
        log.append((x, a, r, pb))
    pi_uniform = lambda x, K=env.K: np.full(K, 1.0 / K)
    return float(ips(log, pi_uniform)), float(snips(log, pi_uniform))


# E6 — Doubly robust against a known-good softmax target.
def exercise_6_dr_softmax(n: int = 800, beta: float = 4.0, seed: int = 0) -> float:
    env = SyntheticLeadBandit(seed=seed)
    logger = EpsilonGreedyLogger(K=env.K, d=env.d, eps=0.3, seed=seed + 1)
    log = []
    for _ in range(n):
        x = env.sample_context()
        a, pb = logger.act(x)
        r = env.step(x, a)
        logger.update(x, a, r)
        log.append((x, a, r, pb))
    qhat = RidgeRewardModel(K=env.K, d=env.d).fit(log)

    def pi(x: np.ndarray) -> np.ndarray:
        logits = beta * env.theta @ x
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    return float(doubly_robust(log, pi, qhat))


# E7 — Effective sample size of importance weights.
def exercise_7_ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))


# E8 — SWITCH-DR estimate, observe how tau trades bias vs. variance.
def exercise_8_switch_dr(n: int = 800, tau: float = 10.0, seed: int = 0) -> float:
    env = SyntheticLeadBandit(seed=seed)
    logger = EpsilonGreedyLogger(K=env.K, d=env.d, eps=0.3, seed=seed + 1)
    log = []
    for _ in range(n):
        x = env.sample_context()
        a, pb = logger.act(x)
        r = env.step(x, a)
        logger.update(x, a, r)
        log.append((x, a, r, pb))
    qhat = RidgeRewardModel(K=env.K, d=env.d).fit(log)

    def pi(x: np.ndarray) -> np.ndarray:
        a = int(np.argmax(env.theta @ x))
        p = np.zeros(env.K)
        p[a] = 1.0
        return p

    return float(switch_dr(log, pi, qhat, tau=tau))


# E9 — *synthetic insurance-flavoured open-ended objective*: compare three counterfactual
# policies using SNIPS + DR and return the absolute estimation error against
# the on-policy ground-truth value for the softmax target. This is the
# meaningful objective exercise.
def exercise_9_synthetic_ope(n: int = 1500, seed: int = 0) -> Tuple[float, float, float]:
    env = SyntheticLeadBandit(seed=seed)
    logger = EpsilonGreedyLogger(K=env.K, d=env.d, eps=0.3, seed=seed + 1)
    log = []
    for _ in range(n):
        x = env.sample_context()
        a, pb = logger.act(x)
        r = env.step(x, a)
        logger.update(x, a, r)
        log.append((x, a, r, pb))
    qhat = RidgeRewardModel(K=env.K, d=env.d).fit(log)
    from .train import softmax_policy, true_policy_value

    pi = softmax_policy(env, beta=4.0)
    v_true = true_policy_value(env, pi, n=4000, seed=seed + 33)
    v_snips = snips(log, pi)
    v_dr = doubly_robust(log, pi, qhat)
    return float(v_true), float(v_snips), float(v_dr)
