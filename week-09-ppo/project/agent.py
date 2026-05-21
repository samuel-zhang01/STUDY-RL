"""PPO agent for continuous-action environments."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.networks import mlp

from .solutions import (
    approx_kl,
    orthogonal_init_,
    ppo_clip_loss,
    ppo_value_clip_loss,
)


class PPOActor(nn.Module):
    """Tanh-activated MLP with state-independent log-std (detail #15, #17)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64)):
        super().__init__()
        # Standard PPO continuous-actor: tanh activations.
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, act_dim)
        # Detail #17: initial log-std = 0.
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        # Detail #2: orthogonal init, policy head gain 0.01.
        orthogonal_init_(self.trunk, gain=math.sqrt(2.0))
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)

    def forward(self, obs: torch.Tensor):
        mu = self.mu_head(self.trunk(obs))
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mu)
        return mu, log_std

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu, log_std = self(obs)
        return torch.distributions.Normal(mu, log_std.exp())


class PPOCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 1])
        # Detail #29: value head gain 1.0.
        orthogonal_init_(self.net, gain=math.sqrt(2.0), final_layer_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class PPOAgent:
    actor: PPOActor
    critic: PPOCritic
    opt: torch.optim.Optimizer
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.02

    @classmethod
    def build(cls, obs_dim: int, act_dim: int, *,
              lr: float = 3e-4, hidden=(64, 64),
              clip_eps: float = 0.2, value_coef: float = 0.5,
              entropy_coef: float = 0.01,
              target_kl: float = 0.02) -> "PPOAgent":
        actor = PPOActor(obs_dim, act_dim, hidden=hidden)
        critic = PPOCritic(obs_dim, hidden=hidden)
        # Detail #3: Adam eps = 1e-5.
        opt = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=lr, eps=1e-5,
        )
        return cls(actor=actor, critic=critic, opt=opt,
                   clip_eps=clip_eps, value_coef=value_coef,
                   entropy_coef=entropy_coef, target_kl=target_kl)

    # --- act ----------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        dist = self.actor.dist(obs_t)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        v = self.critic(obs_t)
        # Detail #18: action clipping.
        a_np = a.clamp(-1.0, 1.0).cpu().numpy()
        return a_np, logp.cpu().numpy(), v.cpu().numpy()

    # --- update -------------------------------------------------------------

    def update_minibatch(
        self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        logp_old: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values_old: torch.Tensor,
    ) -> dict[str, float]:
        dist = self.actor.dist(obs)
        logp = dist.log_prob(acts).sum(-1)
        entropy = dist.entropy().sum(-1).mean()

        # Standardise advantages within the mini-batch (detail #6).
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pi_loss = ppo_clip_loss(logp, logp_old, adv, clip_eps=self.clip_eps)
        v = self.critic(obs)
        v_loss = ppo_value_clip_loss(v, values_old, returns, clip_eps=self.clip_eps)
        loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * entropy

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.opt.step()

        kl = approx_kl(logp, logp_old)
        return {
            "pi_loss": float(pi_loss.item()),
            "v_loss": float(v_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": kl,
        }
