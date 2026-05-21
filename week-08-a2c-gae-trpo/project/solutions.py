"""Reference solutions for the week-08 workbook."""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------- E1/E2

def nstep_advantage(rewards: np.ndarray, values: np.ndarray, gamma: float, n: int,
                    bootstrap: float = 0.0) -> np.ndarray:
    """n-step advantage Abar_t = -V(s_t) + sum_{k<n} gamma^k r_{t+k} + gamma^n V(s_{t+n}).

    `values` is V(s_t) for t in [0, T-1]; `bootstrap` is V(s_T).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    T = len(rewards)
    out = np.zeros(T, dtype=np.float64)
    extended_values = np.concatenate([values, [bootstrap]])
    for t in range(T):
        upper = min(t + n, T)
        k = upper - t
        disc = gamma ** np.arange(k)
        out[t] = -values[t] + (disc * rewards[t:upper]).sum() + (gamma ** k) * extended_values[upper]
    return out


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(gamma, lam). Returns (advantages, returns)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    last_adv = 0.0
    for t in range(T - 1, -1, -1):
        v_next = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * v_next * nonterm - values[t]
        last_adv = delta + gamma * lam * nonterm * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


# ---------------------------------------------------------------------- E6

def diagonal_gaussian_kl(
    mu1: torch.Tensor, log_std1: torch.Tensor,
    mu2: torch.Tensor, log_std2: torch.Tensor,
) -> torch.Tensor:
    """Mean KL( N(mu1, exp(log_std1)^2) || N(mu2, exp(log_std2)^2) ) over the batch.
    Standard closed form: 0.5 * sum_i [ (s1/s2)^2 + ((mu2-mu1)/s2)^2 - 1 + 2*(log_std2-log_std1) ].
    """
    var1 = (2.0 * log_std1).exp()
    var2 = (2.0 * log_std2).exp()
    kl = 0.5 * (var1 / var2 + (mu2 - mu1) ** 2 / var2 - 1.0 + 2.0 * (log_std2 - log_std1))
    return kl.sum(-1).mean()


# ---------------------------------------------------------------------- E7

def conjugate_gradient(
    Av: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    n_iters: int = 10,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve Ax=b for symmetric positive-definite A via conjugate gradient.

    `Av(v)` returns A @ v. Returns x.
    """
    x = np.zeros_like(b, dtype=np.float64)
    r = b.astype(np.float64).copy()
    p = r.copy()
    rs_old = float(r @ r)
    for _ in range(n_iters):
        Ap = Av(p)
        denom = float(p @ Ap) + 1e-12
        alpha = rs_old / denom
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        if rs_new < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


# ---------------------------------------------------------------------- helpers

def flat_params(params: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def set_flat_params(params: Iterable[torch.Tensor], flat: torch.Tensor) -> None:
    i = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n
