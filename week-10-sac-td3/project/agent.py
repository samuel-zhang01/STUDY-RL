"""SAC agent for continuous-action environments (e.g. Pendulum-v1)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.networks import mlp

from .solutions import (
    auto_temperature_loss,
    polyak_update,
    soft_bellman_target,
    tanh_gaussian_log_prob,
    twin_critic_min,
)


class TanhGaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.trunk = mlp([obs_dim, *hidden], output_activation=nn.ReLU)
        h = hidden[-1]
        self.mu_head = nn.Linear(h, act_dim)
        self.log_std_head = nn.Linear(h, act_dim)

    def _trunk_outputs(self, obs: torch.Tensor):
        z = self.trunk(obs)
        mu = self.mu_head(z)
        log_std = self.log_std_head(z).clamp(-5.0, 2.0)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self._trunk_outputs(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        u = mu + std * eps
        a = torch.tanh(u)
        logp = tanh_gaussian_log_prob(u, mu, log_std)
        return a, logp

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self._trunk_outputs(obs)
        return torch.tanh(mu)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, a], dim=-1)).squeeze(-1)


@dataclass
class SACAgent:
    actor: TanhGaussianPolicy
    q1: QNet
    q2: QNet
    q1_tgt: QNet
    q2_tgt: QNet
    log_alpha: torch.Tensor
    opt_actor: torch.optim.Optimizer
    opt_critic: torch.optim.Optimizer
    opt_alpha: torch.optim.Optimizer
    action_scale: float
    target_entropy: float
    gamma: float = 0.99
    tau: float = 5e-3

    @classmethod
    def build(cls, obs_dim: int, act_dim: int, *, action_scale: float = 1.0,
              lr: float = 3e-4, gamma: float = 0.99, tau: float = 5e-3,
              hidden=(256, 256)) -> "SACAgent":
        actor = TanhGaussianPolicy(obs_dim, act_dim, hidden=hidden)
        q1 = QNet(obs_dim, act_dim, hidden=hidden)
        q2 = QNet(obs_dim, act_dim, hidden=hidden)
        q1_tgt = QNet(obs_dim, act_dim, hidden=hidden); q1_tgt.load_state_dict(q1.state_dict())
        q2_tgt = QNet(obs_dim, act_dim, hidden=hidden); q2_tgt.load_state_dict(q2.state_dict())
        for p in q1_tgt.parameters(): p.requires_grad = False
        for p in q2_tgt.parameters(): p.requires_grad = False
        log_alpha = torch.zeros(1, requires_grad=True)

        opt_actor = torch.optim.Adam(actor.parameters(), lr=lr)
        opt_critic = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
        opt_alpha = torch.optim.Adam([log_alpha], lr=lr)
        return cls(
            actor=actor, q1=q1, q2=q2, q1_tgt=q1_tgt, q2_tgt=q2_tgt,
            log_alpha=log_alpha,
            opt_actor=opt_actor, opt_critic=opt_critic, opt_alpha=opt_alpha,
            action_scale=action_scale, target_entropy=-float(act_dim),
            gamma=gamma, tau=tau,
        )

    # ---- act ---------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            a = self.actor.act_deterministic(obs_t)
        else:
            a, _ = self.actor.sample(obs_t)
        return (a.squeeze(0).cpu().numpy() * self.action_scale)

    # ---- update ------------------------------------------------------------

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        s, a, r, s2, d = batch["s"], batch["a"], batch["r"], batch["s2"], batch["d"]
        # Critic update
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q_tgt = twin_critic_min(self.q1_tgt(s2, a2), self.q2_tgt(s2, a2))
            y = soft_bellman_target(r, d, q_tgt, logp2, self.log_alpha.exp(), gamma=self.gamma)

        q1v = self.q1(s, a)
        q2v = self.q2(s, a)
        critic_loss = 0.5 * (F.mse_loss(q1v, y) + F.mse_loss(q2v, y))
        self.opt_critic.zero_grad(); critic_loss.backward(); self.opt_critic.step()

        # Actor update (reparameterised)
        a_new, logp_new = self.actor.sample(s)
        q_min = twin_critic_min(self.q1(s, a_new), self.q2(s, a_new))
        actor_loss = (self.log_alpha.exp().detach() * logp_new - q_min).mean()
        self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()

        # Temperature update
        alpha_loss = auto_temperature_loss(self.log_alpha, logp_new, self.target_entropy)
        self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()

        # Polyak averaging of target nets
        polyak_update(self.q1_tgt, self.q1, self.tau)
        polyak_update(self.q2_tgt, self.q2, self.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "entropy": float(-logp_new.mean().item()),
        }


class ReplayBuffer:
    """Tiny ring buffer for off-policy training."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0; self.ptr = 0

    def add(self, s, a, r, s2, d) -> None:
        i = self.ptr
        self.obs[i] = s; self.acts[i] = a; self.rews[i] = r
        self.next_obs[i] = s2; self.dones[i] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "s":  torch.as_tensor(self.obs[idx], dtype=torch.float32),
            "a":  torch.as_tensor(self.acts[idx], dtype=torch.float32),
            "r":  torch.as_tensor(self.rews[idx], dtype=torch.float32),
            "s2": torch.as_tensor(self.next_obs[idx], dtype=torch.float32),
            "d":  torch.as_tensor(self.dones[idx], dtype=torch.float32),
        }
