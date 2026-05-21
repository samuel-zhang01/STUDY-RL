"""Train IQL, CQL, and Naive-Q baselines on the synthetic offline GridWorld.

Target metric (in module docstring as required):
    Final greedy-policy evaluation return >= 4.0 after 2000 grad steps for IQL.
The naive Q baseline is expected to *fail* (return well below 4.0) due to
extrapolation error — illustrating the core message of offline RL.

Runs in well under 60 seconds on CPU.
"""
from __future__ import annotations
import os
import sys
from typing import Dict, Any
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.seed import seed_everything

from .env import make_env, collect_logged_dataset
from .agent import IQLAgent, CQLAgent, NaiveQAgent


def evaluate_policy(env, policy_actions: np.ndarray, n_episodes: int = 50,
                    max_steps: int = 30, seed: int = 1234) -> float:
    """Run greedy policy and report mean return."""
    rng = np.random.default_rng(seed)
    returns = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        s = env.state
        G = 0.0
        for _ in range(max_steps):
            s_idx = env.s2idx(s)
            a = int(policy_actions[s_idx])
            s, r, done, _ = env.step(a)
            G += r
            if done:
                break
        returns.append(G)
    return float(np.mean(returns))


def train(seed: int = 0, n_steps: int = 2000, batch_size: int = 64,
          algo: str = "iql", record_every: int = 50,
          cql_alpha: float = 1.0, iql_tau: float = 0.7, iql_beta: float = 3.0,
          n_episodes_dataset: int = 200) -> Dict[str, Any]:
    """Train one agent on a freshly-collected offline dataset.

    Returns dict with curves and final eval.
    """
    seed_everything(seed)
    data = collect_logged_dataset(n_episodes=n_episodes_dataset, seed=seed)
    nS, nA = data["nS"], data["nA"]
    n = len(data["a"])
    rng = np.random.default_rng(seed)

    if algo == "iql":
        agent = IQLAgent(nS=nS, nA=nA, tau=iql_tau, beta=iql_beta)
    elif algo == "cql":
        agent = CQLAgent(nS=nS, nA=nA, alpha=cql_alpha)
    elif algo == "naive":
        agent = NaiveQAgent(nS=nS, nA=nA)
    else:
        raise ValueError(algo)

    env = make_env(seed=seed)
    eval_returns = []
    td_losses = []

    for step in range(n_steps):
        idx = rng.integers(0, n, size=batch_size)
        s_idx = data["s_idx"][idx]; a = data["a"][idx]
        r = data["r"][idx]; s2_idx = data["s2_idx"][idx]; done = data["done"][idx]

        if algo == "iql":
            agent.update_value(s_idx, a)
            loss = agent.update_q(s_idx, a, r, s2_idx, done)
        else:
            loss = agent.update(s_idx, a, r, s2_idx, done)
        td_losses.append(loss)

        if step % record_every == 0:
            ret = evaluate_policy(env, agent.policy())
            eval_returns.append((step, ret))

    final_ret = evaluate_policy(env, agent.policy(), n_episodes=100)
    return {
        "algo": algo,
        "td_losses": np.asarray(td_losses, dtype=np.float32),
        "eval_returns": np.asarray(eval_returns, dtype=np.float32),
        "final_return": final_ret,
        "Q": agent.Q.copy(),
        "policy": agent.policy(),
    }


if __name__ == "__main__":
    for algo in ("naive", "cql", "iql"):
        out = train(seed=0, algo=algo)
        print(f"{algo:>6s}  final_eval_return = {out['final_return']:.2f}")
