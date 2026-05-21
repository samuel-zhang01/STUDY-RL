"""Reference solutions for the week-18 workbook exercises."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from .ppo import compute_gae
from .rm import reward_loss
from .data import heuristic_reward


# Exercise 1: Bradley--Terry log-likelihood.
def exercise_1_bt_loss(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """Return -log sigmoid(r_w - r_l) averaged over the batch."""
    return -F.logsigmoid(r_w - r_l).mean()


# Exercise 2: token-level KL between two categorical distributions.
def exercise_2_token_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) in nats, per sample (last dim is the vocabulary)."""
    logp = F.log_softmax(logits_p, dim=-1)
    logq = F.log_softmax(logits_q, dim=-1)
    p = logp.exp()
    return (p * (logp - logq)).sum(-1)


# Exercise 3: PPO clipped surrogate.
def exercise_3_ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    ratio = (new_logprobs - old_logprobs).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


# Exercise 4: GAE.
def exercise_4_gae(rewards, values, gamma=1.0, lam=0.95):
    return compute_gae(rewards, values, gamma=gamma, lam=lam)


# Exercise 5: adaptive KL controller update.
def exercise_5_kl_update(beta: float, observed: float, target: float, horizon: int) -> float:
    err = max(-0.2, min(0.2, observed / target - 1.0))
    return beta * (1.0 + err / horizon)


# Exercise 6: length-normalised reward (penalise verbosity by token count).
def exercise_6_length_normalised(reward: torch.Tensor, length: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    return reward - alpha * length.float()


# Exercise 7: KL-regularised optimal policy closed form (used in DPO derivation).
def exercise_7_optimal_policy_logprob(
    log_pi_ref: torch.Tensor,
    reward: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Return unnormalised log pi_star(y) = log pi_ref(y) + r(y) / beta.

    The caller is responsible for the partition function Z.
    """
    return log_pi_ref + reward / beta


# Exercise 8 (open-ended): a reward-model overoptimisation diagnostic
def exercise_8_overoptimisation_score(
    learned_reward: torch.Tensor, gold_reward: torch.Tensor
) -> float:
    """Return the gap between learned and gold reward (large gap = hacked)."""
    return float((learned_reward - gold_reward).mean().item())


# Exercise 9 (meaningful objective): heuristic-vs-learned reward correlation.
def exercise_9_reward_correlation(texts, scores) -> float:
    import numpy as np
    h = np.array([heuristic_reward(t) for t in texts])
    s = np.asarray(scores)
    if h.std() < 1e-8 or s.std() < 1e-8:
        return 0.0
    return float(((h - h.mean()) * (s - s.mean())).mean() / (h.std() * s.std()))
