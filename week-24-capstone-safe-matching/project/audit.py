"""Phase 3: audit module — fairness, OPE, calibration, red-team.

All functions take either raw arrays (predictions + ground truth) or a trained
policy + dataset and return clean Python dicts the CLI prints.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import torch

from .data import (
    Candidate, JobPosting, LoggedInteraction, context_vector, _true_relevance,
)


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------
def demographic_parity_gap(actions: np.ndarray, groups: np.ndarray) -> float:
    """|P(a=1 | g=0) - P(a=1 | g=1)|. Lower is closer to demographic parity."""
    actions = np.asarray(actions); groups = np.asarray(groups)
    if (groups == 0).any():
        p0 = float(actions[groups == 0].mean())
    else:
        p0 = 0.0
    if (groups == 1).any():
        p1 = float(actions[groups == 1].mean())
    else:
        p1 = 0.0
    return float(abs(p0 - p1))


def equal_opportunity_gap(actions: np.ndarray, groups: np.ndarray,
                          y_true: np.ndarray) -> float:
    """|TPR(g=0) - TPR(g=1)| where positive class is y_true=1."""
    actions = np.asarray(actions); groups = np.asarray(groups); y = np.asarray(y_true)
    pos0 = (groups == 0) & (y == 1)
    pos1 = (groups == 1) & (y == 1)
    tpr0 = float(actions[pos0].mean()) if pos0.any() else 0.0
    tpr1 = float(actions[pos1].mean()) if pos1.any() else 0.0
    return float(abs(tpr0 - tpr1))


# ---------------------------------------------------------------------------
# Off-policy evaluation
# ---------------------------------------------------------------------------
def snips_ope(interactions: List[LoggedInteraction],
              target_action_probs: np.ndarray) -> float:
    """Self-Normalised Inverse Propensity Score estimator.

    target_action_probs[i] = pi_target(a_i | x_i), where a_i is the action taken
    in interaction i. Returns the SNIPS estimate of E_{pi_target}[reward].
    """
    rewards = np.array([it.reward for it in interactions], dtype=float)
    props = np.array([max(1e-3, it.propensity) for it in interactions], dtype=float)
    w = target_action_probs / props
    num = float((w * rewards).sum())
    den = float(w.sum())
    return num / max(1e-9, den)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_curve(probs: np.ndarray, y_true: np.ndarray,
                      n_bins: int = 10) -> Dict[str, np.ndarray]:
    """Reliability diagram bin centres + empirical positives per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2
    pos = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for p, y in zip(probs, y_true):
        idx = min(n_bins - 1, max(0, int(p * n_bins)))
        pos[idx] += y; counts[idx] += 1
    emp = np.where(counts > 0, pos / np.maximum(counts, 1), 0.0)
    return {"centres": centres, "empirical": emp, "counts": counts}


# ---------------------------------------------------------------------------
# Red-team / robustness
# ---------------------------------------------------------------------------
def red_team_eval(model_act_prob_fn, data: Dict, n_episodes: int = 200,
                  noise_levels: Iterable[float] = (0.0, 0.05, 0.10, 0.20),
                  swap_groups: bool = True) -> Dict:
    """Red-team battery:
        1. Skill-feature noise injection at several levels.
        2. Group-attribute swap (does removing/flipping g change the action
           distribution? - a counterfactual fairness probe).

    `model_act_prob_fn(ctx)` -> np.ndarray of shape (2,) action probabilities.
    """
    rng = np.random.default_rng(0)
    jobs = data["jobs"]; cands = data["candidates"]
    results = {"noise": {}, "group_flip_disagreement": None}
    for noise in noise_levels:
        rec_rates: List[float] = []
        for _ in range(n_episodes):
            j = jobs[int(rng.integers(0, len(jobs)))]
            c = cands[int(rng.integers(0, len(cands)))]
            ctx = context_vector(j, c, include_group=True)
            ctx = ctx + rng.normal(0, noise, size=ctx.shape).astype(np.float32)
            p = model_act_prob_fn(ctx)
            rec_rates.append(float(p[1]))
        results["noise"][float(noise)] = float(np.mean(rec_rates))
    # Group flip: count disagreement between (g=0) and (g=1) versions.
    if swap_groups:
        disagreement = 0
        total = 0
        for _ in range(n_episodes):
            j = jobs[int(rng.integers(0, len(jobs)))]
            c = cands[int(rng.integers(0, len(cands)))]
            ctx0 = context_vector(j, c, include_group=True)
            ctx0_flipped = ctx0.copy(); ctx0_flipped[-1] = 1.0 - ctx0[-1]
            p1 = model_act_prob_fn(ctx0)
            p2 = model_act_prob_fn(ctx0_flipped)
            if int(p1[1] > 0.5) != int(p2[1] > 0.5):
                disagreement += 1
            total += 1
        results["group_flip_disagreement"] = disagreement / max(1, total)
    return results


# ---------------------------------------------------------------------------
# End-to-end audit
# ---------------------------------------------------------------------------
def full_audit(net: torch.nn.Module, data: Dict,
               n_episodes: int = 500) -> Dict:
    """Run the audit on `net` (must implement `act_prob(x)` returning
    a (B,2) tensor) and return a clean dict.
    """
    rng = np.random.default_rng(0)
    jobs = data["jobs"]; cands = data["candidates"]
    actions: List[int] = []
    groups: List[int] = []
    y_true: List[int] = []
    probs: List[float] = []
    target_action_probs: List[float] = []
    for it in data["interactions"][:n_episodes]:
        j = next(jj for jj in jobs if jj.job_id == it.job_id)
        c = next(cc for cc in cands if cc.candidate_id == it.candidate_id)
        ctx = context_vector(j, c, include_group=True)
        with torch.no_grad():
            x = torch.as_tensor(ctx, dtype=torch.float32).unsqueeze(0)
            if hasattr(net, "act_prob"):
                p = net.act_prob(x).squeeze(0).numpy()
            else:
                # PPOActorCritic uses base.act_prob underneath
                p = net.base.act_prob(x).squeeze(0).numpy()
        actions.append(int(rng.random() < p[1]))
        groups.append(c.group)
        rel = _true_relevance(j, c)
        y_true.append(int(rel > 0.55))
        probs.append(float(p[1]))
        target_action_probs.append(float(p[it.action]))
    actions = np.array(actions); groups = np.array(groups); y_true = np.array(y_true)
    probs = np.array(probs); target_action_probs = np.array(target_action_probs)
    dp = demographic_parity_gap(actions, groups)
    eo = equal_opportunity_gap(actions, groups, y_true)
    snips = snips_ope(data["interactions"][:n_episodes], target_action_probs)
    cal = calibration_curve(probs, y_true)
    def act_prob_fn(ctx):
        with torch.no_grad():
            x = torch.as_tensor(ctx, dtype=torch.float32).unsqueeze(0)
            if hasattr(net, "act_prob"):
                return net.act_prob(x).squeeze(0).numpy()
            return net.base.act_prob(x).squeeze(0).numpy()
    redteam = red_team_eval(act_prob_fn, data, n_episodes=200)
    return {
        "demographic_parity_gap": dp,
        "equal_opportunity_gap": eo,
        "snips_value": snips,
        "calibration": cal,
        "red_team": redteam,
        "mean_rec_rate": float(actions.mean()),
        "n_episodes": n_episodes,
    }
