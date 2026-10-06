"""End-to-end bandit experiment runner.

Target: on a 10-armed Bernoulli testbed (T = 2000, 30 seeds), Thompson
Sampling and UCB1 should achieve cumulative regret < 50, and clearly beat
epsilon-greedy with eps=0.1 (which scales linearly).

This module finishes in well under 60 seconds on a single CPU.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .env import BernoulliBandit, SyntheticLeadBrokerBandit
from .agent import (
    EpsilonGreedyAgent,
    UCB1Agent,
    ThompsonBernoulliAgent,
    BoltzmannAgent,
    LinUCBAgent,
)


def run_bernoulli_experiment(
    means: np.ndarray,
    T: int = 2000,
    n_seeds: int = 30,
) -> Dict[str, np.ndarray]:
    """Run epsilon-greedy / UCB1 / Thompson / Boltzmann on a Bernoulli testbed.

    Returns a dict mapping agent name to an array of shape (n_seeds, T) of
    cumulative regret.
    """
    K = len(means)
    out: Dict[str, np.ndarray] = {
        name: np.zeros((n_seeds, T))
        for name in ("eps-greedy(0.1)", "UCB1", "Thompson", "Boltzmann(0.1)")
    }
    for seed in range(n_seeds):
        env = BernoulliBandit(means=means, rng=np.random.default_rng(1000 + seed))
        agents = {
            "eps-greedy(0.1)": EpsilonGreedyAgent(K=K, eps=0.1, seed=seed),
            "UCB1": UCB1Agent(K=K, c=2.0, seed=seed),
            "Thompson": ThompsonBernoulliAgent(K=K, seed=seed),
            "Boltzmann(0.1)": BoltzmannAgent(K=K, tau=0.1, seed=seed),
        }
        # Each agent runs against its own copy of the env to share the
        # ground-truth means but get independent reward noise.
        for agent_idx, (name, ag) in enumerate(agents.items()):
            env_copy = BernoulliBandit(
                means=means, rng=np.random.default_rng(1000 + seed * 7 + agent_idx)
            )
            cum_reg = 0.0
            for t in range(T):
                a = ag.select(t + 1)
                r = env_copy.pull(a)
                ag.update(a, r)
                cum_reg += env_copy.regret(a)
                out[name][seed, t] = cum_reg
    return out


def run_contextual_experiment(
    env: SyntheticLeadBrokerBandit,
    T: int = 2000,
    n_seeds: int = 20,
    alpha: float = 1.0,
) -> Dict[str, np.ndarray]:
    """LinUCB vs eps-greedy on the synthetic insurance lead-broker bandit."""
    K, d = env.K, env.d
    out: Dict[str, np.ndarray] = {
        "LinUCB": np.zeros((n_seeds, T)),
        "eps-greedy(0.1)": np.zeros((n_seeds, T)),
    }
    base_theta = env.theta.copy()
    base_b = env.b.copy()
    for seed in range(n_seeds):
        env_seed = SyntheticLeadBrokerBandit(K=K, d=d, seed=seed + 91)
        env_seed.theta = base_theta
        env_seed.b = base_b
        lin = LinUCBAgent(K=K, d=d, alpha=alpha, seed=seed)
        eps = EpsilonGreedyAgent(K=K, eps=0.1, seed=seed)
        cum_lin = 0.0
        cum_eps = 0.0
        for t in range(T):
            x = env_seed.context()
            a_lin = lin.select(x, t + 1)
            a_eps = eps.select(t + 1)
            r_lin = env_seed.pull(x, a_lin)
            r_eps = env_seed.pull(x, a_eps)
            lin.update(x, a_lin, r_lin)
            eps.update(a_eps, r_eps)
            cum_lin += env_seed.instantaneous_regret(x, a_lin)
            cum_eps += env_seed.instantaneous_regret(x, a_eps)
            out["LinUCB"][seed, t] = cum_lin
            out["eps-greedy(0.1)"][seed, t] = cum_eps
    return out


def train(seed: int = 0, T: int = 2000, n_seeds: int = 20) -> Dict[str, float]:
    """Run a quick experiment and return final-regret summary stats.

    Target metric: ``thompson_regret < eps_regret`` and ``ucb1_regret < eps_regret``.
    """
    means = np.array([0.10, 0.20, 0.25, 0.30, 0.35, 0.45, 0.50, 0.55, 0.60, 0.70])
    rng = np.random.default_rng(seed)
    means = np.sort(rng.permutation(means))  # shuffle but keep distribution
    res = run_bernoulli_experiment(means=means, T=T, n_seeds=n_seeds)
    finals = {k: float(v[:, -1].mean()) for k, v in res.items()}
    return finals


if __name__ == "__main__":
    import time

    t0 = time.time()
    finals = train()
    dt = time.time() - t0
    print(f"finished in {dt:.1f}s")
    for name, reg in finals.items():
        print(f"  {name:>20s}: final cumulative regret = {reg:7.2f}")
