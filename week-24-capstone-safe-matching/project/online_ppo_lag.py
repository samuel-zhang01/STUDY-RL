"""Phase 2: online PPO-Lagrangian fine-tuning on a simulated environment.

We treat each (job, candidate) pair as a one-step contextual bandit. The PPO
agent shares the policy network with the IQL pretrain (warm-started weights).

Episode = a single decision per pair. We collect rollouts by sampling pairs
from the dataset and using a Bernoulli reward model that scores positive
interview based on true skill match (see `data._true_relevance`).

Cost = demographic-parity gap measured over a sliding window of recent
recommendations:
    c_t = | mean_{recent}[a | g=0] - mean_{recent}[a | g=1] |

The moving-window cost is a diagnostic, not an immediate bandit cost with a
valid score-function gradient. We optimize a differentiable empirical gap in
policy recommendation probabilities alongside the reward PPO surrogate, and
update lambda using that same gap. This finite-sample surrogate supplies no
population fairness guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    Candidate, JobPosting, context_vector, _true_relevance,
)
from .offline_iql import IQLPolicy


@dataclass
class PPOLagConfig:
    n_updates: int = 25
    steps_per_update: int = 512
    clip_ratio: float = 0.2
    pi_lr: float = 1e-3
    vf_lr: float = 3e-3
    lam_lr: float = 0.1
    train_iters: int = 3
    minibatch_size: int = 128
    ent_coef: float = 0.01
    cost_limit: float = 0.10      # max tolerated parity gap
    lam_init: float = 0.5
    lam_max: float = 20.0
    window: int = 256             # sliding window for parity gap


class PPOActorCritic(nn.Module):
    """Wraps an IQL policy with two value heads (reward, cost)."""

    def __init__(self, base: IQLPolicy, ctx_dim: int, hidden: int = 64):
        super().__init__()
        self.base = base
        self.v_trunk = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.v_head = nn.Linear(hidden, 1)
        self.vc_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        logits = self.base.action_logits(x)
        h = self.v_trunk(x)
        return logits, self.v_head(h).squeeze(-1), self.vc_head(h).squeeze(-1)


def _sample_rollout(data: Dict, net: PPOActorCritic, n_steps: int,
                    cost_window: deque, cfg: PPOLagConfig, rng: np.random.Generator):
    """Sample (job, candidate) pairs uniformly, query the policy, build rollout."""
    jobs = data["jobs"]; cands = data["candidates"]
    obs_buf, act_buf, lp_buf, r_buf, c_buf, v_buf, vc_buf, group_buf = [], [], [], [], [], [], [], []
    n_recs_by_group = [0, 0]
    n_seen_by_group = [0, 0]
    for _ in range(n_steps):
        j = jobs[int(rng.integers(0, len(jobs)))]
        c = cands[int(rng.integers(0, len(cands)))]
        ctx = context_vector(j, c, include_group=net.base.include_group)
        x = torch.as_tensor(ctx, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, v, vc = net(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = int(dist.sample().item())
            lp = float(dist.log_prob(torch.tensor(a)).item())
        # reward model on the fly.
        rel = _true_relevance(j, c)
        if a == 1:
            reward = 1.0 if (rel > 0.55 and rng.random() < 0.9) else 0.0
        else:
            reward = 0.0
        # update sliding window for parity gap.
        cost_window.append((c.group, a))
        if len(cost_window) >= 32:
            g0 = [aa for gg, aa in cost_window if gg == 0]
            g1 = [aa for gg, aa in cost_window if gg == 1]
            p0 = float(np.mean(g0)) if g0 else 0.0
            p1 = float(np.mean(g1)) if g1 else 0.0
            cost = abs(p0 - p1)
        else:
            cost = 0.0
        obs_buf.append(ctx); act_buf.append(a); lp_buf.append(lp)
        r_buf.append(reward); c_buf.append(cost)
        v_buf.append(float(v.item())); vc_buf.append(float(vc.item()))
        group_buf.append(c.group)
        n_recs_by_group[c.group] += int(a == 1)
        n_seen_by_group[c.group] += 1
    parity = abs(
        (n_recs_by_group[0] / max(1, n_seen_by_group[0]))
        - (n_recs_by_group[1] / max(1, n_seen_by_group[1]))
    )
    return (np.stack(obs_buf), np.array(act_buf), np.array(lp_buf, dtype=np.float32),
            np.array(r_buf, dtype=np.float32), np.array(c_buf, dtype=np.float32),
            np.array(v_buf, dtype=np.float32), np.array(vc_buf, dtype=np.float32),
            float(np.mean(r_buf)), parity, np.array(group_buf))


def expected_parity_gap(logits, groups):
    """Differentiable |mean pi(show|g=0) - mean pi(show|g=1)| on one batch."""
    p_show = F.softmax(logits, dim=-1)[:, 1]
    if not (groups == 0).any() or not (groups == 1).any():
        raise ValueError("Parity requires observations from both groups")
    return (p_show[groups == 0].mean() - p_show[groups == 1].mean()).abs()


def ppo_lag_finetune(data: Dict, base: IQLPolicy,
                     cfg: Optional[PPOLagConfig] = None, seed: int = 0,
                     verbose: bool = False) -> Dict:
    """PPO-Lagrangian fine-tune of an IQL-pretrained policy.

    Returns: dict with `net`, `history` (return/cost/lambda/parity).
    """
    cfg = cfg or PPOLagConfig()
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ctx_dim = len(context_vector(data["jobs"][0], data["candidates"][0], include_group=base.include_group))
    # Preserve the offline checkpoint for a meaningful before/after comparison.
    net = PPOActorCritic(copy.deepcopy(base), ctx_dim)
    opt_pi = torch.optim.Adam(net.base.pi.parameters(), lr=cfg.pi_lr)
    opt_v = torch.optim.Adam(
        list(net.v_trunk.parameters()) + list(net.v_head.parameters())
        + list(net.vc_head.parameters()),
        lr=cfg.vf_lr,
    )
    lam = float(cfg.lam_init)
    history = {"return": [], "cost": [], "lambda": [], "parity": [], "expected_parity": []}
    cost_window = deque(maxlen=cfg.window)

    for it in range(cfg.n_updates):
        obs, acts, old_lp, rs, cs, vs, vcs, mean_r, parity, groups = _sample_rollout(
            data, net, cfg.steps_per_update, cost_window, cfg, rng,
        )
        # Targets: contextual bandit, returns = immediate reward / cost.
        adv_r = rs - vs
        # standardise reward advantage only
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        adv_total = adv_r / (1.0 + lam)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        groups_t = torch.as_tensor(groups)
        acts_t = torch.as_tensor(acts, dtype=torch.long)
        old_lp_t = torch.as_tensor(old_lp, dtype=torch.float32)
        adv_t = torch.as_tensor(adv_total, dtype=torch.float32)
        ret_r_t = torch.as_tensor(rs, dtype=torch.float32)
        ret_c_t = torch.as_tensor(cs, dtype=torch.float32)
        N = obs.shape[0]
        idx = np.arange(N)
        for _ in range(cfg.train_iters):
            np.random.shuffle(idx)
            for s in range(0, N, cfg.minibatch_size):
                mb = idx[s:s + cfg.minibatch_size]
                logits, v, vc = net(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(acts_t[mb])
                ratio = torch.exp(lp - old_lp_t[mb])
                clip = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_t[mb]
                pi_loss = -(torch.min(ratio * adv_t[mb], clip)).mean() \
                          - cfg.ent_coef * dist.entropy().mean()
                full_logits = net.base.action_logits(obs_t)
                gap = expected_parity_gap(full_logits, groups_t)
                pi_loss = pi_loss + lam * gap / (1.0 + lam)
                opt_pi.zero_grad()
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.base.pi.parameters(), 0.5)
                opt_pi.step()
                v_loss = F.mse_loss(v, ret_r_t[mb]) + F.mse_loss(vc, ret_c_t[mb])
                opt_v.zero_grad(); v_loss.backward(); opt_v.step()
        with torch.no_grad():
            expected_gap = float(expected_parity_gap(net.base.action_logits(obs_t), groups_t))
        lam = float(np.clip(lam + cfg.lam_lr * (expected_gap - cfg.cost_limit),
                            0.0, cfg.lam_max))
        history["return"].append(float(mean_r))
        history["cost"].append(float(cs.mean()))
        history["lambda"].append(lam)
        history["parity"].append(float(parity))
        history["expected_parity"].append(expected_gap)
        if verbose:
            print(f"it={it:3d} R={mean_r:+.3f} parity={parity:.3f} lam={lam:.3f}")
    return {"net": net, "history": history, "cfg": cfg}
