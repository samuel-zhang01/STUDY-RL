"""Reference solutions for the week-07 workbook exercises.

Every workbook exercise calls one of the functions in this file. They are written
to be readable first, fast second; nothing here is exotic.
"""
from __future__ import annotations

import numpy as np
import torch


# -------------------------------------------------------------------- E1 / E2 --

def log_prob_normal(x: float, mu: float, sigma: float = 1.0) -> float:
    """log N(x; mu, sigma^2). Used to verify the log-derivative identity."""
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)


# ---------------------------------------------------------------------- E3 ----

def discount_cumsum(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute returns-to-go G_t = sum_{t'>=t} gamma^(t'-t) r_{t'} in O(T).

    >>> discount_cumsum(np.array([1.0, 1.0, 1.0]), 0.5).tolist()
    [1.75, 1.5, 1.0]
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    out = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


# ---------------------------------------------------------------------- E4 ----

def reinforce_loss(log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Vanilla REINFORCE policy loss: -E[log pi(a|s) * G]."""
    return -(log_probs * returns).mean()


# ---------------------------------------------------------------------- E5 ----

def reinforce_with_baseline_loss(
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    standardize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (policy_loss, value_loss) for REINFORCE + V-baseline.

    `values` should be the *detached* value-net output for the policy loss; the
    caller is expected to compute the value-net forward separately for the
    value loss so gradients flow only to phi.
    """
    adv = returns - values.detach()
    if standardize:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    policy_loss = -(log_probs * adv).mean()
    value_loss = 0.5 * ((values - returns) ** 2).mean()
    return policy_loss, value_loss


def optimal_baseline(grad_norms_sq: np.ndarray, returns: np.ndarray) -> float:
    """Greensmith 2004 optimal scalar baseline b* = E[||g||^2 * G] / E[||g||^2]."""
    grad_norms_sq = np.asarray(grad_norms_sq, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    num = float((grad_norms_sq * returns).mean())
    den = float(grad_norms_sq.mean()) + 1e-12
    return num / den


# ---------------------------------------------------------------------- E7 ----

def topk_offpolicy_multiplier(probs: np.ndarray | torch.Tensor, K: int):
    """lambda_K(s, a) = K * (1 - pi(a|s))^(K-1).

    Chen et al. WSDM 2019, Eq. 5. Accepts numpy or torch; returns same type.
    """
    if isinstance(probs, torch.Tensor):
        return K * (1.0 - probs).clamp(min=0.0, max=1.0) ** (K - 1)
    probs = np.asarray(probs, dtype=np.float64)
    return K * np.clip(1.0 - probs, 0.0, 1.0) ** (K - 1)


# ---------------------------------------------------------------------- E8 ----

def snips_estimator(
    behaviour_log_probs: np.ndarray,
    target_log_probs: np.ndarray,
    rewards: np.ndarray,
) -> float:
    """Self-Normalised Importance Sampling estimator.

    V_SNIPS(pi_e) = sum_i w_i r_i / sum_i w_i, where w_i = pi_e(a_i|s_i)/pi_b(a_i|s_i).
    Lower variance than IPS, biased but consistent.
    """
    behaviour_log_probs = np.asarray(behaviour_log_probs, dtype=np.float64)
    target_log_probs = np.asarray(target_log_probs, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    # Clip log-ratio for numerical safety on extreme propensities.
    log_w = np.clip(target_log_probs - behaviour_log_probs, -20.0, 20.0)
    w = np.exp(log_w)
    return float((w * rewards).sum() / (w.sum() + 1e-12))
