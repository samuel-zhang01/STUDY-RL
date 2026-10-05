"""Train Rainbow-lite and run the ablation."""
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import gymnasium as gym

from .agent import RainbowAgent, RainbowConfig


@dataclass
class TrainLog:
    episode_returns: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def train(seed: int = 0, env_id: str = "LunarLander-v3", n_steps: int = 5000,
          double: bool = True, dueling: bool = True, per: bool = True,
          noisy: bool = True, reset_every: int = 0,
          eps_decay_steps: int = 3500, warmup_steps: int = 500,
          buffer_size: int = 10_000, batch_size: int = 64, lr: float = 5e-4,
          gamma: float = 0.99, hidden=(128, 128),
          target_tau: float = 0.005, log_every: int = 200, **kwargs) -> TrainLog:
    env = gym.make(env_id)
    env.reset(seed=seed); env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    cfg = RainbowConfig(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden,
                        lr=lr, gamma=gamma, buffer_size=buffer_size,
                        batch_size=batch_size, warmup_steps=warmup_steps,
                        eps_decay_steps=eps_decay_steps,
                        target_tau=target_tau,
                        double=double, dueling=dueling, per=per, noisy=noisy,
                        reset_every=reset_every)
    agent = RainbowAgent(cfg, seed=seed)
    log = TrainLog()
    s, _ = env.reset(seed=seed)
    ep_ret = 0.0
    start = time.time()
    for step in range(1, n_steps + 1):
        a = agent.act(np.asarray(s, dtype=np.float32))
        s2, r, term, trunc, _ = env.step(a)
        # A time limit ends collection but retains the continuation value.
        agent.push(s, a, r, s2, term)
        ep_ret += r
        loss = agent.learn()
        if loss is not None and step % log_every == 0:
            log.losses.append(loss)
        if term or trunc:
            log.episode_returns.append(ep_ret)
            s, _ = env.reset(seed=seed + step)
            ep_ret = 0.0
        else:
            s = s2
    log.meta = {
        "wall_clock_s": time.time() - start,
        "mean_last_10": (float(np.mean(log.episode_returns[-10:]))
                        if log.episode_returns else float("nan")),
    }
    env.close()
    return log


def ablate(seeds: int = 2, n_steps: int = 5000) -> Dict[str, np.ndarray]:
    """Ablate Double, Dueling, PER, NoisyNets one-at-a-time off."""
    out: Dict[str, list] = {"full": [], "no_double": [], "no_dueling": [],
                            "no_per": [], "no_noisy": []}
    for s in range(seeds):
        out["full"].append(
            train(seed=s, n_steps=n_steps).episode_returns)
        out["no_double"].append(
            train(seed=s, n_steps=n_steps, double=False).episode_returns)
        out["no_dueling"].append(
            train(seed=s, n_steps=n_steps, dueling=False).episode_returns)
        out["no_per"].append(
            train(seed=s, n_steps=n_steps, per=False).episode_returns)
        out["no_noisy"].append(
            train(seed=s, n_steps=n_steps, noisy=False).episode_returns)
    # pad to a uniform length (episode counts differ across runs)
    aligned = {}
    for k, v in out.items():
        L = min(len(r) for r in v)
        aligned[k] = np.array([r[:L] for r in v])
    return aligned


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    log = train(seed=args.seed, n_steps=args.steps)
    print(f"mean(last 10) = {log.meta['mean_last_10']:.1f}, "
          f"episodes = {len(log.episode_returns)}, "
          f"wall = {log.meta['wall_clock_s']:.1f}s")
