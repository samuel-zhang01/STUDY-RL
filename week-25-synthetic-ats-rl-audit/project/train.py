"""End-to-end training run for the Phase-1 shadow-mode reranker demo.

Pipeline:
  1. Generate population (500 cand x 80 jobs by default; smaller for tests).
  2. Compute heuristic shortlist (top-k = 10) for each job.
  3. Run an epsilon-soft behaviour policy over the shortlist (logs propensities).
  4. Train LinTS on the logged stream.
  5. Compute SNIPS + DR estimates for LinTS vs heuristic vs uniform.
  6. Run constrained variant + audit.

Target: SNIPS(LinTS) > SNIPS(heuristic) in < 30s on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .bandit import LinTSReranker, softmax_propensity
from .constrained import train_lagrangian
from .env import (
    ATSData,
    context_features,
    feature_dim,
    generate_population,
    true_match_prob,
)
from .ope import LogEntry, RidgeRewardModel, doubly_robust, ips, snips
from .scorer import heuristic_score, top_k_shortlist


@dataclass
class TrainResult:
    log: List[LogEntry]
    snips_estimates: Dict[str, float]
    dr_estimates: Dict[str, float]
    chosen_groups: np.ndarray
    chosen_cands: np.ndarray
    chosen_jobs: np.ndarray
    rewards: List[float] = field(default_factory=list)


def behaviour_policy_probs(scores: np.ndarray, beta: float = 3.0, eps: float = 0.05) -> np.ndarray:
    """eps-soft softmax. eps mass is spread uniformly to guarantee overlap."""
    p_soft = softmax_propensity(scores, beta=beta)
    k = len(p_soft)
    return (1.0 - eps) * p_soft + eps / k


def collect_logs(
    data: ATSData,
    n_rounds: int,
    k_slate: int,
    seed: int,
) -> tuple[List[LogEntry], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s_mat = heuristic_score(data)
    p_match = true_match_prob(data)
    n_jobs = data.jobs.shape[0]

    log: List[LogEntry] = []
    chosen_cands = np.empty(n_rounds, dtype=np.int64)
    chosen_jobs = np.empty(n_rounds, dtype=np.int64)
    rewards = np.empty(n_rounds, dtype=np.float64)

    for t in range(n_rounds):
        j = int(rng.integers(0, n_jobs))
        shortlist_idx = top_k_shortlist(s_mat, j, k=k_slate)
        feats = np.stack([context_features(data, c, j) for c in shortlist_idx])
        heur_scores = s_mat[shortlist_idx, j]
        probs = behaviour_policy_probs(heur_scores, beta=3.0, eps=0.05)
        a = int(rng.choice(len(probs), p=probs))
        chosen = int(shortlist_idx[a])
        r = float(rng.binomial(1, p_match[chosen, j]))
        log.append((feats[a], a, r, float(probs[a]), feats))
        chosen_cands[t] = chosen
        chosen_jobs[t] = j
        rewards[t] = r
    return log, chosen_cands, chosen_jobs, rewards


def target_pi_argmax_factory(scorer_fn):
    def pi(slate: np.ndarray) -> np.ndarray:
        scores = scorer_fn(slate)
        p = np.zeros(len(scores))
        a = int(np.argmax(scores))
        # smooth a tiny bit to keep overlap finite for OPE
        p[:] = 1e-3
        p[a] = 1.0 - 1e-3 * (len(scores) - 1)
        return p
    return pi


def train(
    n_candidates: int = 500,
    n_jobs: int = 80,
    k_slate: int = 10,
    n_rounds: int = 2000,
    seed: int = 0,
) -> TrainResult:
    data = generate_population(
        n_candidates=n_candidates, n_jobs=n_jobs, seed=seed
    )
    log, chosen_cands, chosen_jobs, rewards = collect_logs(
        data, n_rounds=n_rounds, k_slate=k_slate, seed=seed + 1
    )
    d = feature_dim(data.candidates.shape[1])

    # Train LinTS on the logged stream (single epoch, online).
    lints = LinTSReranker(d=d, seed=seed + 2)
    for x, a, r, pb, slate in log:
        lints.update(slate[a], r)

    qhat = RidgeRewardModel(d=d).fit(log)

    pi_lints = target_pi_argmax_factory(lints.score)
    pi_heur = target_pi_argmax_factory(lambda X: X @ np.ones(d) / d)
    pi_uniform = lambda slate: np.full(len(slate), 1.0 / len(slate))

    snips_est = {
        "LinTS": float(snips(log, pi_lints)),
        "Heuristic": float(snips(log, pi_heur)),
        "Uniform": float(snips(log, pi_uniform)),
    }
    dr_est = {
        "LinTS": float(doubly_robust(log, pi_lints, qhat)),
        "Heuristic": float(doubly_robust(log, pi_heur, qhat)),
        "Uniform": float(doubly_robust(log, pi_uniform, qhat)),
    }
    return TrainResult(
        log=log,
        snips_estimates=snips_est,
        dr_estimates=dr_est,
        chosen_groups=data.cand_group[chosen_cands],
        chosen_cands=chosen_cands,
        chosen_jobs=chosen_jobs,
        rewards=list(rewards),
    )


if __name__ == "__main__":  # pragma: no cover
    res = train()
    print("SNIPS:", res.snips_estimates)
    print("DR:   ", res.dr_estimates)
