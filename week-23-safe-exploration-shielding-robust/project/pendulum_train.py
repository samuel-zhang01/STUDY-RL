"""Tiny REINFORCE-with-baseline on the domain-randomised pendulum.

Compares two training regimes:
  * `randomise=True`  — lengths sampled in [0.5, 1.5] (broad training).
  * `randomise=False` — fixed length 1.0 (narrow training).

Both are then evaluated on a held-out length (e.g. 1.4 or 0.6) to measure
out-of-distribution generalisation. The expectation: the randomised policy
generalises better.

The agent is a tiny Gaussian-policy MLP (3-32-32-1) with fixed log-std.
This is *not* a competitive pendulum solver — it is a teaching artefact for
the domain-randomisation comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .env_pendulum import PendulumDR


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int = 3, act_dim: int = 1, hidden: int = 32,
                 log_std_init: float = -0.5):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor):
        return self.mu(obs), self.log_std.exp(), self.v(obs).squeeze(-1)

    def act(self, obs: torch.Tensor):
        mu, sd, v = self.forward(obs)
        dist = torch.distributions.Normal(mu, sd)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), v


@dataclass
class PendulumTrainConfig:
    n_epochs: int = 30
    steps_per_epoch: int = 600
    gamma: float = 0.97
    lr: float = 3e-3
    vf_coef: float = 0.5
    entropy_coef: float = 0.0


def _evaluate(net: GaussianPolicy, env: PendulumDR, n_episodes: int) -> float:
    rets = []
    for _ in range(n_episodes):
        o = env.reset()
        R = 0.0
        done = False
        while not done:
            with torch.no_grad():
                mu, _, _ = net(torch.as_tensor(o).unsqueeze(0))
                a = float(mu.item())
            o, r, done, _ = env.step(a)
            R += r
        rets.append(R)
    return float(np.mean(rets))


def train_pendulum_dr(seed: int = 0, randomise: bool = True,
                      cfg: Optional[PendulumTrainConfig] = None,
                      eval_lengths=(0.3, 1.0, 1.8),
                      eval_episodes: int = 6) -> Dict[str, list]:
    """Train a tiny policy-gradient agent and report eval returns at fixed lengths."""
    cfg = cfg or PendulumTrainConfig()
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = PendulumDR(randomise=randomise)
    env.reset(seed=seed)
    net = GaussianPolicy()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    history: Dict[str, List[float]] = {"train_return": []}

    for epoch in range(cfg.n_epochs):
        # one rollout
        obs_buf: List[np.ndarray] = []
        act_buf: List[torch.Tensor] = []
        logp_buf: List[torch.Tensor] = []
        v_buf: List[torch.Tensor] = []
        rew_buf: List[float] = []
        done_buf: List[bool] = []

        obs = env.reset()
        steps_in_buf = 0
        while steps_in_buf < cfg.steps_per_epoch:
            o_t = torch.as_tensor(obs).unsqueeze(0)
            a, lp, v = net.act(o_t)
            a_val = float(a.squeeze(0).item())
            n_obs, r, done, _ = env.step(a_val)
            obs_buf.append(obs); act_buf.append(a); logp_buf.append(lp); v_buf.append(v)
            rew_buf.append(r); done_buf.append(done)
            obs = n_obs if not done else env.reset()
            steps_in_buf += 1

        # Discounted returns + advantages (Monte Carlo with value baseline).
        T = len(rew_buf)
        returns = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in reversed(range(T)):
            running = rew_buf[t] + (0.0 if done_buf[t] else cfg.gamma * running)
            returns[t] = running
        returns_t = torch.as_tensor(returns)
        vals = torch.stack(v_buf).squeeze(-1)
        adv = returns_t - vals.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        logps = torch.stack(logp_buf).squeeze(-1)
        pi_loss = -(logps * adv).mean()
        v_loss = ((vals - returns_t) ** 2).mean()
        loss = pi_loss + cfg.vf_coef * v_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        history["train_return"].append(float(returns_t.mean().item()))

    # final eval at the requested lengths
    eval_results = {}
    for L in eval_lengths:
        eval_env = PendulumDR(randomise=False, fixed_length=L)
        eval_env.reset(seed=seed + 12345)
        eval_results[L] = _evaluate(net, eval_env, eval_episodes)
    history["eval"] = eval_results
    return history
