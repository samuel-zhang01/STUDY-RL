"""Synchronous A2C agent with GAE for continuous-action environments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.networks import GaussianPolicy, mlp


class ValueHead(nn.Module):
    def __init__(self, obs_dim: int, hidden=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class A2CAgent:
    policy: GaussianPolicy
    value: ValueHead
    opt: torch.optim.Optimizer
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    @classmethod
    def build(cls, obs_dim: int, act_dim: int, *, lr: float = 3e-4,
              gamma: float = 0.99, lam: float = 0.95,
              hidden=(64, 64)) -> "A2CAgent":
        policy = GaussianPolicy(obs_dim, act_dim, hidden=hidden,
                                state_dependent_std=False)
        value = ValueHead(obs_dim, hidden=hidden)
        opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()),
                               lr=lr)
        return cls(policy=policy, value=value, opt=opt, gamma=gamma, lam=lam)

    # --- act ----------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mu, log_std = self.policy(obs_t)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        a = mu + std * eps
        logp = (-0.5 * (((a - mu) / std) ** 2 + 2 * log_std
                         + torch.log(torch.tensor(2.0 * np.pi)))).sum(-1)
        v = self.value(obs_t)
        return a.squeeze(0).cpu().numpy(), float(logp.item()), float(v.item())

    @torch.no_grad()
    def value_of(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return float(self.value(obs_t).item())

    # --- update -------------------------------------------------------------

    def _log_prob(self, obs: torch.Tensor, acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.policy(obs)
        std = log_std.exp()
        logp = (-0.5 * (((acts - mu) / std) ** 2 + 2 * log_std
                         + torch.log(torch.tensor(2.0 * np.pi)))).sum(-1)
        entropy = (0.5 + 0.5 * torch.log(torch.tensor(2.0 * np.pi)) + log_std).sum(-1)
        return logp, entropy

    def update(self, obs: np.ndarray, acts: np.ndarray, advantages: np.ndarray,
               returns: np.ndarray) -> dict[str, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        acts_t = torch.as_tensor(acts, dtype=torch.float32)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(correction=0) + 1e-8)

        logp, entropy = self._log_prob(obs_t, acts_t)
        v = self.value(obs_t)

        pi_loss = -(logp * adv_t).mean()
        v_loss = 0.5 * F.mse_loss(v, ret_t)
        ent = entropy.mean()
        loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * ent

        self.opt.zero_grad()
        loss.backward()
        params = list(self.policy.parameters()) + list(self.value.parameters())
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.opt.step()
        return {
            "pi_loss": float(pi_loss.item()),
            "v_loss": float(v_loss.item()),
            "entropy": float(ent.item()),
        }
