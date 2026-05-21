"""Reference solutions for the week-09 workbook."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def ppo_clip_loss(
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO-clip policy loss. Returns scalar loss (negative-of-objective)."""
    ratio = torch.exp(logp - logp_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def ppo_value_clip_loss(
    value: torch.Tensor,
    value_old: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Value-clipped MSE (detail #8)."""
    v_clipped = value_old + torch.clamp(value - value_old, -clip_eps, clip_eps)
    v_loss1 = (value - returns) ** 2
    v_loss2 = (v_clipped - returns) ** 2
    return 0.5 * torch.max(v_loss1, v_loss2).mean()


def approx_kl(logp: torch.Tensor, logp_old: torch.Tensor) -> float:
    """Schulman's approximate-KL estimator (KL3 form).
    Closed form: (exp(log r) - 1 - log r).mean() for r = pi_new / pi_old.
    """
    log_ratio = logp - logp_old
    ratio = log_ratio.exp()
    return float(((ratio - 1.0) - log_ratio).mean().item())


def orthogonal_init_(
    module: nn.Module,
    gain: float = math.sqrt(2.0),
    final_layer_gain: float | None = None,
) -> None:
    """Orthogonal init (detail #2). Walks `module` and applies orthogonal init to Linear
    layers; if `final_layer_gain` is set, the *last* Linear in `module` uses that gain
    (typically 0.01 for policy heads and 1.0 for value heads).
    """
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    for i, m in enumerate(linears):
        g = gain
        if final_layer_gain is not None and i == len(linears) - 1:
            g = final_layer_gain
        nn.init.orthogonal_(m.weight, gain=g)
        nn.init.zeros_(m.bias)


def adaptive_kl_penalty_update(
    beta: float, measured_kl: float, target_kl: float,
    high_ratio: float = 1.5, low_ratio: float = 1.0 / 1.5,
    beta_min: float = 1e-4, beta_max: float = 1e2,
) -> float:
    """PPO-KL adaptive penalty rule (Schulman 2017, Sec. 4)."""
    if measured_kl > target_kl * high_ratio:
        beta *= 2.0
    elif measured_kl < target_kl * low_ratio:
        beta *= 0.5
    return max(beta_min, min(beta_max, beta))
