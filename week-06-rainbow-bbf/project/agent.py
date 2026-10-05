"""Rainbow-lite agent: Double + Dueling + PER + Noisy. Modular flags.

Distributional (C51/QR-DQN) and n-step return are *not* implemented here to keep the
runtime small for the smoke test; the workbook explains them mathematically.
BBF-style resets are implemented as an option (`reset_every` > 0).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from common.replay import ReplayBuffer, PrioritizedReplayBuffer


# ---------------------- Noisy linear (Fortunato 2018) -------------------

class NoisyLinear(nn.Module):
    """Factorised-Gaussian noisy linear layer."""
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("eps_in", torch.zeros(in_features))
        self.register_buffer("eps_out", torch.zeros(out_features))
        bound = 1.0 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(in_features))
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(out_features))
        self.reset_noise()

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        self.eps_in = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
        self.eps_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_eps = self.eps_out.unsqueeze(1) * self.eps_in.unsqueeze(0)
            w = self.weight_mu + self.weight_sigma * w_eps
            b = self.bias_mu + self.bias_sigma * self.eps_out
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


# ---------------------- Rainbow network ----------------------------------

class RainbowQNet(nn.Module):
    """Dueling + optionally Noisy Q-network."""
    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 128),
                 dueling: bool = True, noisy: bool = True) -> None:
        super().__init__()
        self.dueling = dueling; self.noisy = noisy
        layers: list[nn.Module] = []
        sizes = [obs_dim, *hidden]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        h = hidden[-1]
        Lin = NoisyLinear if noisy else nn.Linear
        if dueling:
            self.v_head = Lin(h, 1)
            self.a_head = Lin(h, n_actions)
        else:
            self.q_head = Lin(h, n_actions)

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        if self.dueling:
            v = self.v_head(z)
            a = self.a_head(z)
            return v + (a - a.mean(dim=-1, keepdim=True))
        return self.q_head(z)


# ---------------------- Rainbow agent ------------------------------------

@dataclass
class RainbowConfig:
    obs_dim: int
    n_actions: int
    hidden: tuple = (128, 128)
    lr: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 30_000
    target_tau: float = 0.005
    grad_clip: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Rainbow flags
    double: bool = True
    dueling: bool = True
    noisy: bool = True
    per: bool = True
    per_alpha: float = 0.6
    per_beta: float = 0.4
    # BBF-style
    reset_every: int = 0           # 0 = never; >0 = shrink-and-perturb every N updates
    reset_alpha: float = 0.5       # interpolation toward fresh init


class RainbowAgent:
    def __init__(self, cfg: RainbowConfig, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.cfg = cfg
        self.q = RainbowQNet(cfg.obs_dim, cfg.n_actions, cfg.hidden,
                              dueling=cfg.dueling, noisy=cfg.noisy).to(cfg.device)
        self.q_target = RainbowQNet(cfg.obs_dim, cfg.n_actions, cfg.hidden,
                                     dueling=cfg.dueling, noisy=cfg.noisy).to(cfg.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        if cfg.per:
            self.buf = PrioritizedReplayBuffer(cfg.buffer_size, cfg.obs_dim,
                                                alpha=cfg.per_alpha, beta=cfg.per_beta,
                                                seed=seed)
        else:
            self.buf = ReplayBuffer(cfg.buffer_size, cfg.obs_dim, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.update_count = 0
        self._init_state = {k: v.detach().clone() for k, v in self.q.state_dict().items()}

    def epsilon(self) -> float:
        # With NoisyNets, exploration is parameter noise; eps still useful for warmup.
        if self.cfg.noisy:
            return 0.0
        c = self.cfg
        if self.step_count >= c.eps_decay_steps:
            return c.eps_end
        frac = self.step_count / c.eps_decay_steps
        return c.eps_start + (c.eps_end - c.eps_start) * frac

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        c = self.cfg
        if (not deterministic) and (not c.noisy) and self.rng.random() < self.epsilon():
            return int(self.rng.integers(0, c.n_actions))
        with torch.no_grad():
            was_training = self.q.training
            if c.noisy and not deterministic:
                self.q.reset_noise()
            if deterministic:
                self.q.eval()
            try:
                x = torch.as_tensor(obs, dtype=torch.float32, device=c.device).unsqueeze(0)
                q = self.q(x)
            finally:
                self.q.train(was_training)
            return int(q.argmax(dim=-1).item())

    def push(self, s, a, r, s2, done) -> None:
        self.buf.push(np.asarray(s, dtype=np.float32), a, r,
                      np.asarray(s2, dtype=np.float32), done)

    def _soft_update(self) -> None:
        tau = self.cfg.target_tau
        for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)

    def _shrink_and_perturb(self) -> None:
        alpha = self.cfg.reset_alpha
        with torch.no_grad():
            for k, p in self.q.state_dict().items():
                p_init = self._init_state[k]
                p.data.mul_(alpha).add_((1 - alpha) * p_init)
        # also reset optimiser moments to avoid carrying stale Adam state
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)

    def learn(self) -> Optional[float]:
        c = self.cfg
        if len(self.buf) < max(c.batch_size, c.warmup_steps):
            self.step_count += 1
            return None
        if c.per:
            s, a, r, s2, d, idx, w = self.buf.sample(c.batch_size)
            w_t = torch.as_tensor(w, device=c.device)
        else:
            s, a, r, s2, d = self.buf.sample(c.batch_size)
            idx = None; w_t = torch.ones(c.batch_size, device=c.device)
        device = c.device
        s = torch.as_tensor(s, device=device)
        a = torch.as_tensor(a, dtype=torch.long, device=device)
        r = torch.as_tensor(r, device=device)
        s2 = torch.as_tensor(s2, device=device)
        d = torch.as_tensor(d, device=device)
        if c.noisy:
            self.q.reset_noise(); self.q_target.reset_noise()
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if c.double:
                a_star = self.q(s2).argmax(dim=1, keepdim=True)
                q_next = self.q_target(s2).gather(1, a_star).squeeze(1)
            else:
                q_next = self.q_target(s2).max(dim=1).values
            target = r + (1.0 - d) * c.gamma * q_next
        td_err = (target - q_sa).detach()
        # Per-element loss with IS weights for PER
        # Huber per-sample
        diff = q_sa - target
        l = torch.where(diff.abs() < 1.0, 0.5 * diff ** 2, diff.abs() - 0.5)
        loss = (w_t * l).mean()
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
        self.opt.step()
        if c.per and idx is not None:
            self.buf.update_priorities(idx, td_err.cpu().numpy())
        self._soft_update()
        self.update_count += 1
        if c.reset_every > 0 and self.update_count % c.reset_every == 0:
            self._shrink_and_perturb()
        self.step_count += 1
        return float(loss.item())
