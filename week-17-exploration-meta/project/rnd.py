"""Random Network Distillation (RND) on a hard-exploration GridWorld toy.

We implement the canonical RND intrinsic-reward formulation:
- Fixed random target network f_target: state -> R^d.
- Trainable predictor f_pred trained to regress f_target on visited states.
- Intrinsic reward at step t: ||f_pred(s_t) - f_target(s_t)||_2^2,
  i.e. the "novelty" of the state w.r.t. what the predictor has seen.

Burda et al.\\ 2018, arXiv:1810.12894.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.gridworld import GridWorld
from common.seed import seed_everything


class RND(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.target = nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU(),
                                    nn.Linear(hidden, out_dim))
        self.predictor = nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU(),
                                       nn.Linear(hidden, hidden), nn.ELU(),
                                       nn.Linear(hidden, out_dim))
        for p in self.target.parameters():
            p.requires_grad_(False)

    def intrinsic(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t = self.target(obs)
        p = self.predictor(obs)
        return ((p - t) ** 2).sum(dim=-1)

    def predictor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        return self.intrinsic(obs).mean()


def make_hard_grid(seed: int = 0) -> GridWorld:
    """7x7 open GridWorld with sparse reward and no pits; goal at (6,6)."""
    # Exploration is hard because the distant goal is the only positive reward.
    return GridWorld(rows=7, cols=7, start=(0, 0), goals=((6, 6),), pits=(),
                     slip_prob=0.0, step_reward=0.0, goal_reward=1.0, pit_reward=-1.0,
                     rng=np.random.default_rng(seed))


def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32); v[idx] = 1.0; return v


def train_rnd_on_gridworld(seed: int = 0, n_episodes: int = 200, use_rnd: bool = True,
                           int_coef: float = 1.0, gamma: float = 0.99, lr: float = 1e-3,
                           eps: float = 0.4, max_steps: int = 100) -> dict:
    """Tabular Q-learning with optional RND intrinsic reward bonus."""
    seed_everything(seed)
    env = make_hard_grid(seed=seed)
    nS, nA = env.nS, env.nA
    rng = np.random.default_rng(seed)

    Q = np.zeros((nS, nA), dtype=np.float32)
    rnd = RND(in_dim=nS, hidden=32, out_dim=16)
    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=lr)

    returns = []
    visits = np.zeros(nS, dtype=np.int64)
    goal_hits = 0
    int_running = 1.0
    for ep in range(n_episodes):
        env.reset(seed=seed * 7919 + ep)
        s = env.state
        G_ext = 0.0
        for t in range(max_steps):
            s_idx = env.s2idx(s)
            visits[s_idx] += 1
            if rng.random() < eps:
                a = int(rng.integers(0, nA))
            else:
                a = int(np.argmax(Q[s_idx]))
            ns, r_ext, done, _ = env.step(a)
            ns_idx = env.s2idx(ns)

            # Intrinsic reward.
            if use_rnd:
                obs_t = torch.from_numpy(_onehot(ns_idx, nS)).unsqueeze(0)
                with torch.no_grad():
                    i_r = float(rnd.intrinsic(obs_t).item())
                int_running = 0.99 * int_running + 0.01 * i_r
                i_r_norm = i_r / (np.sqrt(int_running) + 1e-6)
                r_intrinsic = int_coef * i_r_norm
                # Update predictor.
                loss = rnd.predictor_loss(obs_t)
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                r_intrinsic = 0.0

            r_total = r_ext + r_intrinsic
            best_next = np.max(Q[ns_idx])
            target = r_total + (0.0 if done else gamma * best_next)
            Q[s_idx, a] += 0.2 * (target - Q[s_idx, a])

            s = ns; G_ext += r_ext
            if done:
                if r_ext > 0:
                    goal_hits += 1
                break
        returns.append(G_ext)

    return {
        "returns": np.asarray(returns, dtype=np.float32),
        "visits": visits.reshape(env.rows, env.cols),
        "goal_hits": goal_hits,
        "Q": Q,
    }
