"""Reference solutions for the week-10 workbook."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------- E1

def tanh_gaussian_log_prob(
    u: torch.Tensor,
    mu: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    """log pi(a|s) for a = tanh(u), u ~ N(mu, exp(log_std)^2).

    Returns per-sample log-prob summed over action dimensions.
    Uses the numerically stable softplus form for the Jacobian.
    """
    std = log_std.exp()
    log_p_u = -0.5 * (((u - mu) / std) ** 2 + 2.0 * log_std
                       + math.log(2.0 * math.pi))
    # log|da/du| = log(1 - tanh^2(u)) = 2*(log 2 - u - softplus(-2u))   (stable)
    log_det_jac = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
    return (log_p_u - log_det_jac).sum(-1)


# ---------------------------------------------------------------------- E3

def twin_critic_min(q1_target: torch.Tensor, q2_target: torch.Tensor) -> torch.Tensor:
    """Conservative twin-critic target: min over two Q networks."""
    return torch.min(q1_target, q2_target)


@torch.no_grad()
def td3_target(rewards, dones, next_obs, target_actor, target_q1, target_q2,
               *, target_noise=0.2, noise_clip=0.5, gamma=0.99):
    """TD3 target smoothing for normalized [-1, 1] actions.

    The noise has the actor's action shape, is scaled before clipping,
    and affects both target critics evaluated at the same smoothed action.
    This helper is a workbook building block, not a full TD3 trainer.
    """
    next_action = target_actor(next_obs)
    noise = (torch.randn_like(next_action) * target_noise).clamp(-noise_clip, noise_clip)
    next_action = (next_action + noise).clamp(-1.0, 1.0)
    next_q = twin_critic_min(target_q1(next_obs, next_action), target_q2(next_obs, next_action))
    return rewards + gamma * (1.0 - dones.float()) * next_q


def soft_bellman_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_q: torch.Tensor,
    next_logp: torch.Tensor,
    alpha: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Soft Bellman target y = r + gamma * (1-d) * (Q_min - alpha * log pi)."""
    return rewards + gamma * (1.0 - dones) * (next_q - alpha * next_logp)


# ---------------------------------------------------------------------- E5

def auto_temperature_loss(
    log_alpha: torch.Tensor,
    logp: torch.Tensor,
    target_entropy: float,
) -> torch.Tensor:
    """Dual loss for the automatic temperature.

    L(alpha) = E[ -alpha * (log pi(a|s) + H_target) ]
             = -alpha * E[ log pi(a|s) + H_target ]

    We optimise log_alpha; alpha = exp(log_alpha). Detach logp because we are
    optimising in log_alpha only.
    """
    return -(log_alpha.exp() * (logp.detach() + target_entropy)).mean()


# ---------------------------------------------------------------------- helper

def polyak_update(target_net: nn.Module, source_net: nn.Module, tau: float) -> None:
    """tgt <- (1 - tau) * tgt + tau * src (in-place)."""
    with torch.no_grad():
        for t, s in zip(target_net.parameters(), source_net.parameters()):
            t.data.mul_(1.0 - tau).add_(tau * s.data)
