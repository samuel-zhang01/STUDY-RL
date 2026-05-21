"""PPO-Lagrangian agent for the constrained lava-gridworld.

Implements:
    * Categorical-policy MLP actor + two critics (reward V and cost V_c).
    * Generalised Advantage Estimation (GAE) on both reward and cost streams.
    * The PPO clipped surrogate, but with the *augmented* advantage
            A_total = A_r - lambda * A_c
      where lambda is the dual variable.
    * Lagrange multiplier updated by projected gradient ascent:
            lambda <- max(0, lambda + eta * (J_c_hat - d_target)).

Single-file, < 250 LoC, runs in well under a minute on a single CPU core for the
7x7 lava gridworld.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """MLP actor + two value heads (reward, cost). Separate trunks so each
    optimiser can update independently without graph-reuse problems."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.pi_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.v_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)
        self.vc_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h_pi = self.pi_trunk(obs)
        h_v = self.v_trunk(obs)
        return self.pi_head(h_pi), self.v_head(h_v).squeeze(-1), self.vc_head(h_v).squeeze(-1)

    def act(self, obs: torch.Tensor) -> Tuple[int, float, float, float]:
        """Sample an action; return (a, log_prob, v, v_c)."""
        logits, v, vc = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(v.item()), float(vc.item())


# ---------------------------------------------------------------------------
# Config + agent
# ---------------------------------------------------------------------------
@dataclass
class PPOLagrangianConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-3
    vf_lr: float = 5e-3
    lam_lr: float = 0.05
    train_iters: int = 4
    minibatch_size: int = 128
    ent_coef: float = 0.01
    cost_limit: float = 5.0   # discounted-cost budget d
    lam_init: float = 0.0
    lam_max: float = 50.0
    hidden: int = 64


@dataclass
class RolloutBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    acts: List[int] = field(default_factory=list)
    logps: List[float] = field(default_factory=list)
    rews: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    vals: List[float] = field(default_factory=list)
    vals_c: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    ep_starts: List[int] = field(default_factory=list)

    def push(self, o, a, lp, r, c, v, vc, d):
        self.obs.append(o); self.acts.append(a); self.logps.append(lp)
        self.rews.append(r); self.costs.append(c)
        self.vals.append(v); self.vals_c.append(vc); self.dones.append(d)

    def __len__(self) -> int:
        return len(self.acts)


def compute_gae(rews, vals, dones, gamma: float, lam: float, last_val: float = 0.0):
    """Generalised advantage estimation. Returns (advantages, returns) as np arrays."""
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_val = last_val
    for t in reversed(range(T)):
        nonterm = 1.0 - float(dones[t])
        delta = rews[t] + gamma * next_val * nonterm - vals[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
        next_val = vals[t]
    ret = adv + np.asarray(vals, dtype=np.float32)
    return adv, ret


class PPOLagrangian:
    """PPO with primal-dual Lagrangian constraint on discounted cost."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: PPOLagrangianConfig | None = None):
        self.cfg = cfg or PPOLagrangianConfig()
        self.net = ActorCritic(obs_dim, n_actions, hidden=self.cfg.hidden)
        self.opt_pi = torch.optim.Adam(
            list(self.net.pi_trunk.parameters()) + list(self.net.pi_head.parameters()),
            lr=self.cfg.pi_lr,
        )
        self.opt_v = torch.optim.Adam(
            list(self.net.v_trunk.parameters())
            + list(self.net.v_head.parameters())
            + list(self.net.vc_head.parameters()),
            lr=self.cfg.vf_lr,
        )
        self.lam = float(self.cfg.lam_init)
        self.history: Dict[str, list] = {
            "return": [], "cost": [], "lambda": [], "pi_loss": [], "v_loss": [],
        }

    # ------------------------------------------------------------------
    def update(self, buf: RolloutBuffer, last_val: float, last_val_c: float,
               ep_returns: List[float], ep_costs: List[float]) -> None:
        cfg = self.cfg
        adv_r, ret_r = compute_gae(buf.rews, buf.vals, buf.dones, cfg.gamma, cfg.gae_lambda, last_val)
        adv_c, ret_c = compute_gae(buf.costs, buf.vals_c, buf.dones, cfg.gamma, cfg.gae_lambda, last_val_c)
        # Standardise reward advantages (do NOT standardise cost advantages — keep them on the cost scale).
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)

        obs = torch.as_tensor(np.stack(buf.obs), dtype=torch.float32)
        acts = torch.as_tensor(buf.acts, dtype=torch.long)
        old_logps = torch.as_tensor(buf.logps, dtype=torch.float32)
        adv_r_t = torch.as_tensor(adv_r, dtype=torch.float32)
        adv_c_t = torch.as_tensor(adv_c, dtype=torch.float32)
        ret_r_t = torch.as_tensor(ret_r, dtype=torch.float32)
        ret_c_t = torch.as_tensor(ret_c, dtype=torch.float32)

        N = obs.shape[0]
        idx = np.arange(N)
        for _ in range(cfg.train_iters):
            np.random.shuffle(idx)
            for start in range(0, N, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                logits, v, vc = self.net(obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(acts[mb])
                ratio = torch.exp(logp - old_logps[mb])
                # Augmented advantage: A_r - lambda * A_c (mind the sign of the constraint).
                # We normalise the combined advantage by (1 + lam) to keep gradient magnitudes
                # comparable as lam grows; this is a standard PPO-Lagrangian trick.
                adv_total = (adv_r_t[mb] - self.lam * adv_c_t[mb]) / (1.0 + self.lam)
                clip_adv = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_total
                pi_loss = -(torch.min(ratio * adv_total, clip_adv)).mean()
                ent = dist.entropy().mean()
                pi_loss = pi_loss - cfg.ent_coef * ent
                self.opt_pi.zero_grad()
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.net.pi_trunk.parameters()) + list(self.net.pi_head.parameters()), 0.5)
                self.opt_pi.step()

                v_loss = F.mse_loss(v, ret_r_t[mb]) + F.mse_loss(vc, ret_c_t[mb])
                self.opt_v.zero_grad()
                v_loss.backward()
                self.opt_v.step()

        # ---- Dual variable update (projected gradient ascent) ----------
        mean_cost = float(np.mean(ep_costs)) if ep_costs else 0.0
        violation = mean_cost - cfg.cost_limit
        self.lam = float(np.clip(self.lam + cfg.lam_lr * violation, 0.0, cfg.lam_max))

        self.history["return"].append(float(np.mean(ep_returns)) if ep_returns else 0.0)
        self.history["cost"].append(mean_cost)
        self.history["lambda"].append(self.lam)
        self.history["pi_loss"].append(float(pi_loss.item()))
        self.history["v_loss"].append(float(v_loss.item()))
