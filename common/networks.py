"""Small reusable network building blocks for the Deep RL lectures."""
from __future__ import annotations
from typing import Sequence, Callable
import torch
import torch.nn as nn


def mlp(sizes: Sequence[int], activation: Callable[..., nn.Module] = nn.ReLU,
        output_activation: Callable[..., nn.Module] | None = None,
        layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Standard discrete-action Q-network."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (128, 128),
                 layer_norm: bool = False) -> None:
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions], layer_norm=layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """Dueling architecture: separate value and advantage streams."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.trunk = mlp([obs_dim, *hidden], output_activation=nn.ReLU)
        h = hidden[-1]
        self.v_head = nn.Linear(h, 1)
        self.a_head = nn.Linear(h, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        v = self.v_head(z)
        a = self.a_head(z)
        return v + (a - a.mean(dim=-1, keepdim=True))


class GaussianPolicy(nn.Module):
    """Gaussian policy with state-dependent mean and (optionally) std. Used by PG/SAC."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (128, 128),
                 log_std_min: float = -5.0, log_std_max: float = 2.0,
                 state_dependent_std: bool = True) -> None:
        super().__init__()
        self.trunk = mlp([obs_dim, *hidden], output_activation=nn.Tanh)
        h = hidden[-1]
        self.mu = nn.Linear(h, act_dim)
        self.state_dependent_std = state_dependent_std
        if state_dependent_std:
            self.log_std = nn.Linear(h, act_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def forward(self, x: torch.Tensor):
        z = self.trunk(x)
        mu = self.mu(z)
        if self.state_dependent_std:
            log_std = self.log_std(z).clamp(self.log_std_min, self.log_std_max)
        else:
            log_std = self.log_std.clamp(self.log_std_min, self.log_std_max).expand_as(mu)
        return mu, log_std

    def sample(self, x: torch.Tensor):
        mu, log_std = self(x)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        a = mu + eps * std
        log_prob = -0.5 * (((a - mu) / std) ** 2 + 2 * log_std + torch.log(torch.tensor(2 * torch.pi)))
        return a, log_prob.sum(-1)
