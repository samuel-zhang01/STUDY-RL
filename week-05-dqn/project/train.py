"""DQN training driver on LunarLander-v3.

Smoke target (CPU, ~30 s with 5000 steps):
    mean return of last 10 evaluation episodes > -250.
Full target (single consumer GPU, ~3-5 min for 200k steps):
    mean return of last 10 evaluation episodes >= 150.
"""
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

import gymnasium as gym

from .agent import DQNAgent, DQNConfig


def _make_env(env_id: str = "LunarLander-v3", seed: int = 0):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


@dataclass
class TrainLog:
    steps: List[int] = field(default_factory=list)
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)


def train(seed: int = 0, env_id: str = "LunarLander-v3", n_steps: int = 5000,
          eval_episodes: int = 5, eps_decay_steps: int = 30_000,
          warmup_steps: int = 1_000, buffer_size: int = 50_000,
          batch_size: int = 64, lr: float = 5e-4, gamma: float = 0.99,
          hidden=(128, 128), target_tau: float = 0.005,
          target_hard_every: int = 0, log_every: int = 200,
          eval_every: Optional[int] = None, **kwargs) -> TrainLog:
    """Single-env DQN. Returns a TrainLog. `n_steps` is the env-step budget."""
    env = _make_env(env_id, seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    cfg = DQNConfig(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden,
                    lr=lr, gamma=gamma, buffer_size=buffer_size,
                    batch_size=batch_size, warmup_steps=warmup_steps,
                    eps_decay_steps=eps_decay_steps,
                    target_tau=target_tau, target_hard_every=target_hard_every)
    agent = DQNAgent(cfg, seed=seed)
    log = TrainLog()

    s, _ = env.reset(seed=seed)
    ep_ret, ep_len = 0.0, 0
    start = time.time()
    for step in range(1, n_steps + 1):
        a = agent.act(np.asarray(s, dtype=np.float32))
        s2, r, term, trunc, _ = env.step(a)
        done = bool(term)  # only bootstrap-block on true terminal
        agent.push(s, a, r, s2, done)
        ep_ret += r; ep_len += 1
        loss = agent.learn()
        if loss is not None and step % log_every == 0:
            log.losses.append(loss); log.epsilons.append(agent.epsilon())
            log.steps.append(step)
        if term or trunc:
            log.episode_returns.append(ep_ret)
            log.episode_lengths.append(ep_len)
            s, _ = env.reset(seed=seed + step)
            ep_ret, ep_len = 0.0, 0
        else:
            s = s2

    log_meta = {
        "wall_clock_s": time.time() - start,
        "n_episodes": len(log.episode_returns),
        "mean_last_10": (float(np.mean(log.episode_returns[-10:]))
                        if log.episode_returns else float("nan")),
    }
    log.meta = log_meta  # type: ignore[attr-defined]
    env.close()
    return log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    res = train(seed=args.seed, n_steps=args.steps)
    print(f"Episodes: {len(res.episode_returns)}, "
          f"mean(last 10) return = {res.meta['mean_last_10']:.1f}, "
          f"wall = {res.meta['wall_clock_s']:.1f} s")
