"""Training loop for week 11: simulate logger, train LinUCB and LinTS online,
then run OPE estimators against three counterfactual policies.

Target: LinTS cumulative regret < 0.6 * T at T=2000, K=6, d=8.
Runtime: ~5-15 s on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np

from .env import SyntheticLeadBandit
from .agents import LinUCB, LinTS, EpsilonGreedyLogger
from .ope import (
    Logged,
    RidgeRewardModel,
    ips,
    snips,
    direct_method,
    doubly_robust,
    switch_dr,
    true_policy_value,
)


# ---------------------------------------------------------------------------
# Online training
# ---------------------------------------------------------------------------

def run_online(env: SyntheticLeadBandit, agent, T: int) -> Tuple[np.ndarray, np.ndarray]:
    rewards = np.zeros(T)
    regrets = np.zeros(T)
    for t in range(T):
        x = env.sample_context()
        a = agent.act(x)
        r = env.step(x, a)
        agent.update(x, a, r)
        rewards[t] = r
        regrets[t] = env.optimal_mean(x) - env.mean_reward(x, a)
    return rewards, regrets


# ---------------------------------------------------------------------------
# Offline logging via epsilon-greedy logger
# ---------------------------------------------------------------------------

def collect_logs(env: SyntheticLeadBandit, n: int, eps: float, seed: int) -> List[Logged]:
    logger = EpsilonGreedyLogger(K=env.K, d=env.d, eps=eps, seed=seed)
    log: List[Logged] = []
    for _ in range(n):
        x = env.sample_context()
        a, pb = logger.act(x)
        r = env.step(x, a)
        logger.update(x, a, r)
        log.append((x, a, r, pb))
    return log


# ---------------------------------------------------------------------------
# Counterfactual target policies
# ---------------------------------------------------------------------------

def uniform_policy(K: int) -> Callable[[np.ndarray], np.ndarray]:
    p = np.full(K, 1.0 / K)
    return lambda x, p=p: p


def greedy_oracle_policy(env: SyntheticLeadBandit) -> Callable[[np.ndarray], np.ndarray]:
    def pi(x: np.ndarray) -> np.ndarray:
        a = env.optimal_action(x)
        p = np.zeros(env.K)
        p[a] = 1.0
        return p

    return pi


def softmax_policy(env: SyntheticLeadBandit, beta: float = 4.0) -> Callable[[np.ndarray], np.ndarray]:
    """Boltzmann over the true logits — a 'good but imperfect' policy."""

    def pi(x: np.ndarray) -> np.ndarray:
        logits = beta * env.theta @ x
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    return pi


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    regrets: Dict[str, np.ndarray]
    ope: Dict[str, Dict[str, float]]
    ope_truth: Dict[str, float]


def train(seed: int = 0, T: int = 2000, n_log: int = 4000, eps_log: float = 0.3) -> TrainResult:
    env = SyntheticLeadBandit(seed=seed)

    # 1. Online training of LinUCB, LinTS.
    regrets: Dict[str, np.ndarray] = {}
    for name, agent in (
        ("LinUCB", LinUCB(K=env.K, d=env.d, alpha=1.0)),
        ("LinTS", LinTS(K=env.K, d=env.d, v=0.25, seed=seed + 7)),
    ):
        _, reg = run_online(SyntheticLeadBandit(seed=seed), agent, T)
        regrets[name] = reg.cumsum()

    # 2. Collect offline logs from epsilon-greedy logger.
    # Logs and oracle truth must use the same latent reward mechanisms.
    log = collect_logs(SyntheticLeadBandit(seed=seed), n_log, eps_log, seed + 17)

    # 3. Fit reward model for DR / Direct.
    qhat = RidgeRewardModel(K=env.K, d=env.d, lam=1.0).fit(log)

    # 4. Evaluate three counterfactual policies.
    policies = {
        "uniform": uniform_policy(env.K),
        "softmax-true": softmax_policy(env, beta=4.0),
        "oracle-greedy": greedy_oracle_policy(env),
    }

    ope: Dict[str, Dict[str, float]] = {}
    ope_truth: Dict[str, float] = {}
    for name, pi in policies.items():
        ope[name] = {
            "IPS": ips(log, pi),
            "SNIPS": snips(log, pi),
            "Direct": direct_method(log, pi, qhat),
            "DR": doubly_robust(log, pi, qhat),
            "SWITCH-DR(tau=10)": switch_dr(log, pi, qhat, tau=10.0),
        }
        ope_truth[name] = true_policy_value(env, pi)

    return TrainResult(regrets=regrets, ope=ope, ope_truth=ope_truth)


if __name__ == "__main__":  # pragma: no cover
    r = train()
    print("Final cumulative regret:", {k: float(v[-1]) for k, v in r.regrets.items()})
    for name, est in r.ope.items():
        print(f"\nPolicy: {name}  (truth = {r.ope_truth[name]:.3f})")
        for k, v in est.items():
            print(f"  {k:20s} = {v:.3f}")
