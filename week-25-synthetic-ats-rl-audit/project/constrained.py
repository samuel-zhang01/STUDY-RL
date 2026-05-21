"""PPO-Lagrangian-style fairness constrained reranker.

We deliberately keep this *small and fast*: a contextual-bandit Lagrangian
update on top of LinTS. The constraint is demographic-parity-ish: the
expected selection rate of group B should not fall below `parity_floor`
times the selection rate of group A (a one-sided 80% rule analogue).

This is a single-step Lagrangian relaxation — full CMDP treatment is in
week 22. For this week we want a runnable demonstration on the synthetic
ATS data, not yet-another-PPO clone.

  L(theta, lambda) = E[reward] - lambda * (parity_floor * P(B selected) - P(A selected))
                                                       wait sign:
  cost c = max(0, parity_floor * sel_A - sel_B)  (B under-selected)
  Lagrangian update:
      theta_{t+1} = theta_t + eta * (grad reward - lambda_t * grad cost)
      lambda_{t+1} = max(0, lambda_t + eta_lam * c)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .env import ATSData, context_features, true_match_prob


@dataclass
class TrainStats:
    rewards: List[float]
    parity_gap: List[float]
    lam_hist: List[float]


def parity_cost(sel_A: float, sel_B: float, parity_floor: float = 0.8) -> float:
    """Cost is positive when B is under-selected relative to A."""
    return max(0.0, parity_floor * sel_A - sel_B)


def train_lagrangian(
    data: ATSData,
    n_rounds: int = 400,
    k_slate: int = 10,
    eta: float = 0.05,
    eta_lam: float = 0.5,
    parity_floor: float = 0.8,
    seed: int = 0,
) -> TrainStats:
    rng = np.random.default_rng(seed)
    n_cand, n_job = data.candidates.shape[0], data.jobs.shape[0]
    # Linear policy parameters
    d = data.candidates.shape[1] + 4
    theta = np.zeros(d)
    lam = 0.0
    p_match = true_match_prob(data)

    rewards, parity_gap, lams = [], [], []
    # Running ema of group selection rates
    sel_A = sel_B = 0.5
    decay = 0.05

    for t in range(n_rounds):
        j = int(rng.integers(0, n_job))
        # heuristic shortlist
        from .scorer import heuristic_score, top_k_shortlist
        # cache once outside in real code; cheap here
        s_mat = heuristic_score(data)
        shortlist = top_k_shortlist(s_mat, j, k=k_slate)
        feats = np.stack([context_features(data, c, j) for c in shortlist])
        logits = feats @ theta
        # Penalised score: deprioritise A picks when B under-selected
        groups = data.cand_group[shortlist]
        bias_pen = lam * (groups == 0).astype(np.float64)
        adj = logits - bias_pen
        a = int(np.argmax(adj))
        chosen = int(shortlist[a])
        g = int(data.cand_group[chosen])
        r = float(rng.binomial(1, p_match[chosen, j]))

        # Update theta with a simple online reward-weighted update
        theta = theta + eta * (r - 0.5) * feats[a]
        # Update group selection EMAs
        if g == 0:
            sel_A = (1 - decay) * sel_A + decay * 1.0
            sel_B = (1 - decay) * sel_B + decay * 0.0
        else:
            sel_A = (1 - decay) * sel_A + decay * 0.0
            sel_B = (1 - decay) * sel_B + decay * 1.0
        c = parity_cost(sel_A, sel_B, parity_floor)
        lam = max(0.0, lam + eta_lam * c)

        rewards.append(r)
        parity_gap.append(sel_B / max(sel_A, 1e-6))
        lams.append(lam)

    return TrainStats(rewards=rewards, parity_gap=parity_gap, lam_hist=lams)
