"""Reference DQN agent (vanilla Mnih 2015 + Polyak target update).

Designed to be readable. Production tricks (PER, n-step, dueling) are
deferred to Week 6.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from common.networks import QNetwork
from common.replay import ReplayBuffer


@dataclass
class DQNConfig:
    obs_dim: int
    n_actions: int
    hidden: tuple = (128, 128)
    lr: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 100_000
    batch_size: int = 64
    warmup_steps: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 30_000
    target_tau: float = 0.005       # Polyak soft target update (per step)
    target_hard_every: int = 0      # if > 0, hard copy every N steps and ignore tau
    grad_clip: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DQNAgent:
    def __init__(self, cfg: DQNConfig, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.cfg = cfg
        self.q = QNetwork(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(cfg.device)
        self.q_target = QNetwork(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(cfg.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.buffer_size, cfg.obs_dim, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

    # ------------- behaviour --------------
    def epsilon(self) -> float:
        c = self.cfg
        if self.step_count >= c.eps_decay_steps:
            return c.eps_end
        frac = self.step_count / c.eps_decay_steps
        return c.eps_start + (c.eps_end - c.eps_start) * frac

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if (not deterministic) and self.rng.random() < self.epsilon():
            return int(self.rng.integers(0, self.cfg.n_actions))
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            q = self.q(x)
            return int(q.argmax(dim=-1).item())

    # ------------- learn --------------
    def push(self, s, a, r, s2, done) -> None:
        self.buf.push(np.asarray(s, dtype=np.float32), a, r,
                      np.asarray(s2, dtype=np.float32), done)

    def _soft_update(self) -> None:
        tau = self.cfg.target_tau
        for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)

    def _hard_update(self) -> None:
        self.q_target.load_state_dict(self.q.state_dict())

    def learn(self) -> Optional[float]:
        c = self.cfg
        if len(self.buf) < max(c.batch_size, c.warmup_steps):
            self.step_count += 1
            return None
        s, a, r, s2, d = self.buf.sample(c.batch_size)
        device = c.device
        s = torch.as_tensor(s, device=device)
        a = torch.as_tensor(a, dtype=torch.long, device=device)
        r = torch.as_tensor(r, device=device)
        s2 = torch.as_tensor(s2, device=device)
        d = torch.as_tensor(d, device=device)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s2).max(dim=1).values
            target = r + (1.0 - d) * c.gamma * q_next
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
        self.opt.step()
        if c.target_hard_every > 0:
            if self.step_count % c.target_hard_every == 0:
                self._hard_update()
        else:
            self._soft_update()
        self.step_count += 1
        return float(loss.item())
