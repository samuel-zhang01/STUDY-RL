"""REINFORCE agent with an optional learned value baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.networks import mlp


class CategoricalPolicy(nn.Module):
    """Categorical (softmax) policy for discrete-action envs."""

    def __init__(self, obs_dim: int, n_actions: int, hidden=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions])

    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.net(obs))

    def forward(self, obs: torch.Tensor):  # pragma: no cover - convenience
        return self.dist(obs)


class ValueHead(nn.Module):
    def __init__(self, obs_dim: int, hidden=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class REINFORCEAgent:
    """A bundle of policy + (optional) value head + their optimisers.

    Use as:
        agent = REINFORCEAgent.build(obs_dim, n_actions, use_baseline=True)
        agent.update(obs, acts, rews)
    """

    policy: CategoricalPolicy
    value: ValueHead | None
    opt_pi: torch.optim.Optimizer
    opt_v: torch.optim.Optimizer | None
    gamma: float = 0.99
    use_baseline: bool = True

    @classmethod
    def build(
        cls,
        obs_dim: int,
        n_actions: int,
        *,
        use_baseline: bool = True,
        pi_lr: float = 5e-3,
        v_lr: float = 1e-2,
        gamma: float = 0.99,
        hidden=(64, 64),
    ) -> "REINFORCEAgent":
        policy = CategoricalPolicy(obs_dim, n_actions, hidden=hidden)
        opt_pi = torch.optim.Adam(policy.parameters(), lr=pi_lr)
        value: ValueHead | None = None
        opt_v: torch.optim.Optimizer | None = None
        if use_baseline:
            value = ValueHead(obs_dim, hidden=hidden)
            opt_v = torch.optim.Adam(value.parameters(), lr=v_lr)
        return cls(policy=policy, value=value, opt_pi=opt_pi, opt_v=opt_v,
                   gamma=gamma, use_baseline=use_baseline)

    # ----- rollout helpers ---------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        a = self.policy.dist(obs_t).sample().item()
        return int(a)

    # ----- update ------------------------------------------------------------

    def update(self, obs: np.ndarray, acts: np.ndarray, rews: np.ndarray) -> dict[str, float]:
        from .solutions import discount_cumsum

        returns_np = discount_cumsum(rews, self.gamma)
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        acts_t = torch.as_tensor(np.asarray(acts), dtype=torch.int64)
        ret_t = torch.as_tensor(returns_np, dtype=torch.float32)

        dist = self.policy.dist(obs_t)
        logp = dist.log_prob(acts_t)

        if self.use_baseline and self.value is not None and self.opt_v is not None:
            v = self.value(obs_t)
            adv = ret_t - v.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pi_loss = -(logp * adv).mean()
            v_loss = 0.5 * F.mse_loss(v, ret_t)

            self.opt_pi.zero_grad(); pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.opt_pi.step()

            self.opt_v.zero_grad(); v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.opt_v.step()

            return {"pi_loss": float(pi_loss.item()), "v_loss": float(v_loss.item())}

        # vanilla REINFORCE: standardise returns to keep optimiser scale sane
        adv = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
        pi_loss = -(logp * adv).mean()
        self.opt_pi.zero_grad(); pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.opt_pi.step()
        return {"pi_loss": float(pi_loss.item()), "v_loss": 0.0}

    # ----- variance probe ----------------------------------------------------

    def estimator_variance(self, obs, acts, rews) -> float:
        """Per-update empirical variance of (log_prob * advantage) terms."""
        from .solutions import discount_cumsum

        returns_np = discount_cumsum(rews, self.gamma)
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        acts_t = torch.as_tensor(np.asarray(acts), dtype=torch.int64)
        ret_t = torch.as_tensor(returns_np, dtype=torch.float32)
        with torch.no_grad():
            logp = self.policy.dist(obs_t).log_prob(acts_t)
            if self.use_baseline and self.value is not None:
                adv = ret_t - self.value(obs_t)
            else:
                adv = ret_t
        x = (logp * adv).cpu().numpy()
        return float(np.var(x))
