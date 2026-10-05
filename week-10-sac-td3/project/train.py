"""Train SAC on Pendulum-v1 with automatic temperature.

Target metric: smoothed episode return >= -300 within 8k env steps. Pendulum is
chosen over MuJoCo Hopper so the install footprint is trivial; the implementation
generalises directly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from common.seed import seed_everything

from .agent import ReplayBuffer, SACAgent


@dataclass
class TrainOutput:
    episode_returns: list[float] = field(default_factory=list)
    update_metrics: list[dict] = field(default_factory=list)
    entropy_trace: list[float] = field(default_factory=list)
    alpha_trace: list[float] = field(default_factory=list)


def train(
    *,
    seed: int = 0,
    env_id: str = "Pendulum-v1",
    n_steps: int = 8000,
    start_random: int = 1000,
    batch_size: int = 256,
    update_every: int = 1,
    updates_per_step: int = 1,
    buffer_size: int = 100_000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 5e-3,
    verbose: bool = False,
) -> TrainOutput:
    import gymnasium as gym

    seed_everything(seed)
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_scale = float(env.action_space.high[0])

    agent = SACAgent.build(obs_dim, act_dim, action_scale=action_scale,
                           lr=lr, gamma=gamma, tau=tau)
    buf = ReplayBuffer(buffer_size, obs_dim, act_dim)

    out = TrainOutput()
    ep_return = 0.0
    win = deque(maxlen=20)

    for step in range(n_steps):
        if step < start_random:
            a = env.action_space.sample()
        else:
            a = agent.act(obs)

        nxt, r, term, trunc, _ = env.step(a)
        done = float(term)  # Pendulum has no terminal; trunc is timelimit
        # Store normalised action in [-1, 1] for the buffer (SAC's actor outputs that scale)
        a_normalised = np.asarray(a, dtype=np.float32) / action_scale
        buf.add(obs, a_normalised, float(r), nxt, done)
        ep_return += float(r)

        if term or trunc:
            out.episode_returns.append(ep_return)
            win.append(ep_return)
            ep_return = 0.0
            obs, _ = env.reset()
        else:
            obs = nxt

        if step >= start_random and buf.size >= batch_size and step % update_every == 0:
            for _ in range(updates_per_step):
                batch = buf.sample(batch_size)
                m = agent.update(batch)
                out.update_metrics.append(m)
                out.entropy_trace.append(m["entropy"])
                out.alpha_trace.append(m["alpha"])

        if verbose and step % 500 == 0 and win:
            print(f"step {step:5d}  recent {np.mean(win):7.1f}  "
                  f"alpha {agent.log_alpha.exp().item():.3f}")

    env.close()
    return out


if __name__ == "__main__":  # pragma: no cover
    out = train(verbose=True)
    if out.episode_returns:
        print("final mean return:", np.mean(out.episode_returns[-20:]))
