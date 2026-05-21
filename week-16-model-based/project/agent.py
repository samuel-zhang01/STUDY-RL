"""Dreamer-lite actor and critic, trained in imagination on top of the RSSM."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, feat: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.net(feat)
        return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


class DreamerLite:
    """Container for {RSSM, actor, critic} parameters plus update rules."""
    def __init__(self, rssm, n_actions: int, hidden: int = 128, device: str = "cpu"):
        self.rssm = rssm
        self.device = torch.device(device)
        feat_dim = rssm.det_dim + rssm.stoch_dim
        self.actor = Actor(feat_dim, n_actions, hidden).to(self.device)
        self.critic = Critic(feat_dim, hidden).to(self.device)
        self.n_actions = n_actions
