"""Minimal MAPPO (Multi-Agent PPO) implementation for the SimpleSpread env.

Centralised-critic, decentralised-execution paradigm: each agent has its own
actor (local observation only), but all share a critic that consumes the
global state. This is the dominant cooperative-MARL recipe of 2022-2026
(Yu et al. NeurIPS 2022 demonstrated MAPPO matches or beats QMIX, MADDPG,
COMA on most benchmarks).

CPU-friendly; converges in ~30s on SimpleSpread (3 agents).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .simple_spread import SimpleSpread


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp


class CentralisedCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float = 0.95,
                 lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
    returns = advantages + values
    return advantages, returns


def train_mappo(
    seed: int = 0,
    n_episodes: int = 60,
    n_agents: int = 3,
    clip_eps: float = 0.2,
    lr: float = 5e-4,
    update_epochs: int = 4,
    minibatch: int = 64,
    gamma: float = 0.95,
    lam: float = 0.95,
    device: str = "cpu",
    env=None,
) -> Dict[str, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if env is None:
        env = SimpleSpread(n_agents=n_agents, seed=seed)
    n_actions = getattr(env, "n_actions", 5)
    actor = Actor(env.obs_dim, n_actions).to(device)
    critic = CentralisedCritic(env.state_dim).to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr)

    returns_history: List[float] = []

    for ep in range(n_episodes):
        # ---- rollout ----
        # Per-agent buffers for one episode; we GAE them separately.
        ep_obs = [[] for _ in range(env.n_agents)]
        ep_act = [[] for _ in range(env.n_agents)]
        ep_logp = [[] for _ in range(env.n_agents)]
        ep_rew = [[] for _ in range(env.n_agents)]
        ep_val = [[] for _ in range(env.n_agents)]
        ep_state = []
        obs = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            obs_tensors = [torch.tensor(o, device=device) for o in obs]
            actions, logps = [], []
            with torch.no_grad():
                for i in range(env.n_agents):
                    a, lp = actor.act(obs_tensors[i])
                    actions.append(int(a.item()))
                    logps.append(float(lp.item()))
            state = env.global_state()
            with torch.no_grad():
                v = float(critic(torch.tensor(state, device=device).unsqueeze(0)).item())
            next_obs, rewards, done, _info = env.step(actions)
            for i in range(env.n_agents):
                ep_obs[i].append(obs[i])
                ep_act[i].append(actions[i])
                ep_logp[i].append(logps[i])
                ep_rew[i].append(rewards[i])
                ep_val[i].append(v)
            ep_state.append(state)
            ep_return += sum(rewards)
            obs = next_obs
        returns_history.append(ep_return)

        # ---- compute advantages per-agent then concatenate ----
        obs_buf, act_buf, logp_buf, adv_buf, ret_buf, state_buf = [], [], [], [], [], []
        for i in range(env.n_agents):
            rew = np.array(ep_rew[i], dtype=np.float32)
            val = np.array(ep_val[i], dtype=np.float32)
            adv, ret = compute_gae(rew, val, gamma=gamma, lam=lam)
            obs_buf.extend(ep_obs[i])
            act_buf.extend(ep_act[i])
            logp_buf.extend(ep_logp[i])
            adv_buf.extend(adv)
            ret_buf.extend(ret)
            state_buf.extend(ep_state)
        adv = np.array(adv_buf, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(np.stack(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.long, device=device)
        logp_old_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(np.array(ret_buf), dtype=torch.float32, device=device)
        state_t = torch.tensor(np.stack(state_buf), dtype=torch.float32, device=device)

        n = obs_t.shape[0]
        for _ in range(update_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, minibatch):
                mb = idx[start:start + minibatch]
                logits = actor(obs_t[mb])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_t[mb])
                ratio = (logp - logp_old_t[mb]).exp()
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()
                opt_a.zero_grad()
                actor_loss.backward()
                opt_a.step()

                v_pred = critic(state_t[mb])
                critic_loss = F.mse_loss(v_pred, ret_t[mb])
                opt_c.zero_grad()
                critic_loss.backward()
                opt_c.step()

    return {"returns": np.array(returns_history)}
