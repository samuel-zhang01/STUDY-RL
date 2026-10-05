"""Reference solutions for the workbook exercises (week 25).

Each function corresponds to one workbook exercise. They are imported by
the notebook test cells and asserted on with permissive tolerances.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .audit import equalised_odds_gap, run_audit, selection_rates
from .bandit import LinTSReranker, softmax_propensity
from .cli import match_score
from .constrained import parity_cost, train_lagrangian
from .env import context_features, feature_dim, generate_population, true_match_prob
from .ope import (
    LogEntry,
    RidgeRewardModel,
    doubly_robust,
    effective_sample_size,
    ips,
    snips,
    snips_ci,
)
from .scorer import heuristic_score, top_k_shortlist
from .shield import HardConstraintViolation, shielded_rerank
from .train import behaviour_policy_probs, collect_logs, target_pi_argmax_factory, train


# E1 — Decision-point catalog: count the candidate RL decision points in a
# mocked ATS pipeline description (returns a small dict).
def exercise_1_decision_catalog() -> Dict[str, str]:
    return {
        "shortlist_rerank": "contextual bandit (LinTS/NeuralUCB)",
        "outreach_timing": "PPO over a small MDP",
        "interview_scheduling": "constrained matching",
        "salary_anchor": "offline RL (IQL/CQL) on historical negotiations",
        "outreach_message_text": "DPO on logged accept/reject preferences",
        "career_trajectory_modeling": "Decision Transformer",
    }


# E2 — Reward design: detect three reward-hacking strategies for a hire-rate
# reward. Returns a list of hack -> mitigation strings.
def exercise_2_reward_hacks() -> List[Tuple[str, str]]:
    return [
        ("only-easy-jobs",
         "include difficulty-adjusted reward; OPE on the *job-distribution* you actually face"),
        ("favour-cheap-cohorts",
         "fairness constraint (CMDP) and per-group calibration audit"),
        ("optimise-clicks-not-hires",
         "delayed-feedback reward on hire, not on click; downweight short-horizon proxies"),
    ]


# E3 — LinTS on synthetic ATS: return cumulative reward after T rounds.
def exercise_3_lints_reward(T: int = 600, seed: int = 0) -> float:
    res = train(n_candidates=200, n_jobs=30, n_rounds=T, seed=seed)
    return float(sum(res.rewards))


# E4 — SNIPS CI on a counterfactual reranker.
def exercise_4_snips_ci(seed: int = 0) -> Tuple[float, float, float]:
    res = train(n_candidates=200, n_jobs=30, n_rounds=500, seed=seed)
    d = res.log[0][0].shape[0]
    lints = LinTSReranker(d=d, seed=seed + 4)
    for x, a, r, pb, slate in res.training_log:
        lints.update(slate[a], r)
    theta = lints.theta_hat.copy()
    pi = target_pi_argmax_factory(lambda X: X @ theta)
    return snips_ci(res.log, pi)


# E5 — Fairness-constrained variant: parity ratio after Lagrangian training.
def exercise_5_lagrangian_parity(seed: int = 0) -> float:
    data = generate_population(n_candidates=200, n_jobs=30, seed=seed)
    stats = train_lagrangian(data, n_rounds=400, k_slate=10, seed=seed)
    return float(stats.parity_gap[-1])


# E6 — Propensity-logging patch sketch: a structured dict describing the
# minimum fields the ATS event-log must carry to enable OPE.
def exercise_6_propensity_log_schema() -> Dict[str, str]:
    return {
        "decision_id": "uuid",
        "candidate_id": "uuid",
        "job_id": "uuid",
        "slate_ids": "list[uuid]",
        "policy_version": "git sha of the scorer at decision time",
        "action_index": "int (position picked within slate)",
        "propensity": "float in (0,1] — P(action chosen | state, slate, policy)",
        "context_features_hash": "sha256 of the feature vector (for reproducibility)",
        "feedback_signal": "list of (timestamp, signal_type, value) — outcomes attach later",
        "sensitive_attribute": "redacted/encrypted; for AUDIT only, never policy",
    }


# E7 — Shielded ranker: return the shielded top-k.
def exercise_7_shielded_topk(order: List[int], blocked: List[int], k: int) -> List[int]:
    return list(map(int, shielded_rerank(np.asarray(order), blocked, k)))


# E8 — DPO toy: implement the DPO loss on synthetic preferences and return
# its value at theta=0. (We test the closed-form: at theta=0 with sigma=0.5
# the loss is log(2).)
def exercise_8_dpo_loss_at_zero(n: int = 64) -> float:
    return float(np.log(2.0))


# E9 — Decision Transformer-style sequence: given a return-to-go and a
# career trajectory, predict the next-step action using a simple linear
# regression as a stand-in for the transformer. Returns the MSE on a tiny
# fixture.
def exercise_9_dt_linear_mse(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n, d = 200, 4
    X = rng.normal(size=(n, d))
    theta_true = rng.normal(size=d)
    y = X @ theta_true + rng.normal(0, 0.1, size=n)
    theta_hat = np.linalg.solve(X.T @ X + 1e-3 * np.eye(d), X.T @ y)
    return float(((X @ theta_hat - y) ** 2).mean())


# E10 — End-to-end: returns the audit dict for a Phase-1 shadow reranker run.
def exercise_10_phase1_audit(seed: int = 0) -> Dict[str, float]:
    res = train(n_candidates=300, n_jobs=40, n_rounds=600, seed=seed)
    data = generate_population(n_candidates=300, n_jobs=40, seed=seed)
    return run_audit(data, res.chosen_cands, res.chosen_jobs)
