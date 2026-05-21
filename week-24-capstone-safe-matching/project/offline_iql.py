"""Phase 1: IQL pretrain of a re-ranking policy from logged data.

IQL (Implicit Q-Learning, Kostrikov et al. 2021) keeps everything in-support
of the dataset using expectile regression. We adapt the standard IQL to our
*one-step* contextual-bandit setup (the episodes are trivially length 1 since
the recommendation decision is independent for each (job, candidate) pair).

In this contextual-bandit version IQL reduces to:
    V(s) = expectile_tau of Q(s, a) over a sampled from dataset
    Q(s, a) -> r (no bootstrap, because the next state is terminal)
    pi(a|s) propto exp(beta * (Q(s, a) - V(s))) advantage-weighted regression

We train Q-networks for actions 0 and 1, a V-network, and the policy network
(a small MLP outputting a Bernoulli logit). The whole pipeline runs in ~10
seconds on CPU for our dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    Candidate, JobPosting, LoggedInteraction, context_dim, context_vector,
)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class IQLConfig:
    tau: float = 0.7                  # expectile parameter
    beta: float = 3.0                 # AWR temperature
    n_epochs: int = 30
    batch_size: int = 256
    lr: float = 3e-3
    hidden: int = 64
    include_group: bool = True


def _expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(diff >= 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


def _build_dataset(data: Dict, cfg: IQLConfig):
    jobs_by_id = {j.job_id: j for j in data["jobs"]}
    cand_by_id = {c.candidate_id: c for c in data["candidates"]}
    obs, acts, rews, groups = [], [], [], []
    for it in data["interactions"]:
        j = jobs_by_id[it.job_id]
        c = cand_by_id[it.candidate_id]
        obs.append(context_vector(j, c, include_group=cfg.include_group))
        acts.append(it.action)
        rews.append(it.reward)
        groups.append(c.group)
    return (
        torch.as_tensor(np.stack(obs), dtype=torch.float32),
        torch.as_tensor(acts, dtype=torch.long),
        torch.as_tensor(rews, dtype=torch.float32),
        torch.as_tensor(groups, dtype=torch.long),
    )


class IQLPolicy(nn.Module):
    """Bernoulli (binary action) policy with two Q-heads + V-head + logits-head."""

    def __init__(self, ctx_dim: int, hidden: int = 64):
        super().__init__()
        self.q = MLP(ctx_dim, 2, hidden=hidden)
        self.v = MLP(ctx_dim, 1, hidden=hidden)
        self.pi = MLP(ctx_dim, 2, hidden=hidden)

    def action_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.pi(x)

    def act_prob(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.pi(x), dim=-1)


def iql_pretrain(data: Dict, cfg: Optional[IQLConfig] = None,
                 verbose: bool = False, seed: int = 0) -> Dict:
    """Train an IQL policy on the logged dataset. Returns dict with model,
    history, and final-evaluation diagnostics."""
    cfg = cfg or IQLConfig()
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs, acts, rews, groups = _build_dataset(data, cfg)
    N = obs.shape[0]
    in_dim = obs.shape[1]
    model = IQLPolicy(ctx_dim=in_dim, hidden=cfg.hidden)
    opt_q = torch.optim.Adam(model.q.parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(model.v.parameters(), lr=cfg.lr)
    opt_pi = torch.optim.Adam(model.pi.parameters(), lr=cfg.lr)
    history: Dict[str, List[float]] = {"q_loss": [], "v_loss": [], "pi_loss": []}

    for ep in range(cfg.n_epochs):
        idx = np.random.permutation(N)
        q_losses, v_losses, pi_losses = [], [], []
        for start in range(0, N, cfg.batch_size):
            mb = idx[start:start + cfg.batch_size]
            o = obs[mb]; a = acts[mb]; r = rews[mb]
            # ---- Q step: regress Q(s,a) -> r (terminal, no bootstrap).
            q_pred = model.q(o).gather(1, a.unsqueeze(-1)).squeeze(-1)
            q_loss = F.mse_loss(q_pred, r)
            opt_q.zero_grad(); q_loss.backward(); opt_q.step()
            # ---- V step: expectile of Q over the *sampled* action.
            with torch.no_grad():
                q_target = model.q(o).gather(1, a.unsqueeze(-1)).squeeze(-1)
            v_pred = model.v(o).squeeze(-1)
            v_loss = _expectile_loss(q_target - v_pred, cfg.tau)
            opt_v.zero_grad(); v_loss.backward(); opt_v.step()
            # ---- Pi step: advantage-weighted regression.
            with torch.no_grad():
                adv = q_target - model.v(o).squeeze(-1)
                w = torch.clamp(torch.exp(cfg.beta * adv), max=100.0)
            logits = model.pi(o)
            logp = F.log_softmax(logits, dim=-1).gather(1, a.unsqueeze(-1)).squeeze(-1)
            pi_loss = -(w * logp).mean()
            opt_pi.zero_grad(); pi_loss.backward(); opt_pi.step()
            q_losses.append(float(q_loss.item()))
            v_losses.append(float(v_loss.item()))
            pi_losses.append(float(pi_loss.item()))
        history["q_loss"].append(float(np.mean(q_losses)))
        history["v_loss"].append(float(np.mean(v_losses)))
        history["pi_loss"].append(float(np.mean(pi_losses)))
        if verbose and (ep % 5 == 0 or ep == cfg.n_epochs - 1):
            print(f"epoch {ep:2d}  q={history['q_loss'][-1]:.4f} "
                  f"v={history['v_loss'][-1]:.4f} pi={history['pi_loss'][-1]:.4f}")
    return {"model": model, "history": history, "cfg": cfg}
