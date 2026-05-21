"""Reference solutions for week-6 workbook exercises."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Ex 1: Double DQN target

def exercise_1_double_dqn_target(reward, done, q_online_next, q_target_next, gamma):
    """target = r + (1-d) * gamma * Q_target(s', argmax_a' Q_online(s', a'))."""
    a_star = q_online_next.argmax(dim=1, keepdim=True)
    q_next = q_target_next.gather(1, a_star).squeeze(1)
    return reward + (1.0 - done.float()) * gamma * q_next


# Ex 2: Dueling head

def exercise_2_dueling_head(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """v: (B,1), a: (B,A). Return Q = v + (a - mean(a))."""
    return v + (a - a.mean(dim=-1, keepdim=True))


# Ex 3: Importance-sampling weight for PER

def exercise_3_per_weight(prob: np.ndarray, N: int, beta: float) -> np.ndarray:
    """w_i = (N * P(i))^(-beta), normalised by max so w in [0, 1]."""
    w = (N * np.maximum(prob, 1e-12)) ** (-beta)
    return w / w.max()


# Ex 4: priority update post-step

def exercise_4_priority_update(old_priorities: np.ndarray, idx: np.ndarray,
                               td_errors: np.ndarray, eps: float) -> np.ndarray:
    p = old_priorities.copy()
    p[idx] = np.abs(td_errors) + eps
    return p


# Ex 5: NoisyLinear "factorised f" function

def exercise_5_noisy_f(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sign(x) * sqrt(|x|)."""
    return x.sign() * x.abs().sqrt()


# Ex 6: categorical projection (C51)

def exercise_6_categorical_project(p: torch.Tensor, support: torch.Tensor,
                                   reward: torch.Tensor, done: torch.Tensor,
                                   gamma: float) -> torch.Tensor:
    """Project the distributional Bellman target back onto a fixed support.

    p:        (B, N) probabilities over N atoms
    support:  (N,)   atom locations z_i
    reward:   (B,)
    done:     (B,)   in {0,1}
    Returns:  (B, N) projected target distribution.
    """
    B, N = p.shape
    Vmin, Vmax = float(support.min()), float(support.max())
    dz = (Vmax - Vmin) / (N - 1)
    # shifted/scaled atoms: Tz_j = r + gamma*(1-d)*z_j , clipped
    Tz = reward.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * gamma * support.unsqueeze(0)
    Tz = Tz.clamp(Vmin, Vmax)
    b = (Tz - Vmin) / dz
    l = b.floor().long(); u = b.ceil().long()
    # handle b exactly on an integer atom
    l[(u > 0) & (l == u)] -= 1
    u[(l < N - 1) & (l == u)] += 1
    m = torch.zeros_like(p)
    offset = torch.arange(B).unsqueeze(1).to(p.device) * N
    m.view(-1).index_add_(0, (l + offset).view(-1), (p * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1), (p * (b - l.float())).view(-1))
    return m


# Ex 7: n-step return helper

def exercise_7_nstep_return(rewards: list, q_bootstrap: float, gamma: float, n: int,
                            done_last: bool) -> float:
    """G_t^{(n)} = sum_{k=0..n-1} gamma^k r_k + (1 - done_last) * gamma^n * q_bootstrap."""
    G = 0.0
    for k, r in enumerate(rewards[:n]):
        G += (gamma ** k) * r
    if not done_last and len(rewards) >= n:
        G += (gamma ** n) * q_bootstrap
    return G


# Ex 8: BBF-style shrink-and-perturb

def exercise_8_shrink_perturb(theta: torch.Tensor, theta_init: torch.Tensor,
                              alpha: float) -> torch.Tensor:
    """theta <- alpha * theta + (1 - alpha) * theta_init  (BBF style soft reset)."""
    return alpha * theta + (1.0 - alpha) * theta_init


# Ex 9: ablation runner (smoke)

def exercise_9_smoke_ablate(n_steps: int = 1500):
    from .train import train
    res = {}
    for name, kwargs in {
        "full": dict(),
        "no_per": dict(per=False),
        "no_noisy": dict(noisy=False),
    }.items():
        log = train(seed=0, n_steps=n_steps, eps_decay_steps=int(0.7 * n_steps),
                    warmup_steps=200, buffer_size=5000, **kwargs)
        res[name] = log
    return res
