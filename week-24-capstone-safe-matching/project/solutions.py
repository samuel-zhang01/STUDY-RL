"""Reference solutions for the week-24 capstone workbook."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .data import (
    generate_dataset, context_vector, _true_relevance,
)
from .offline_iql import iql_pretrain, IQLConfig
from .online_ppo_lag import ppo_lag_finetune, PPOLagConfig
from .audit import (
    demographic_parity_gap, equal_opportunity_gap, snips_ope,
    calibration_curve, full_audit,
)
from .train import train as _train


def exercise_1(seed: int = 0):
    """Synthesise the dataset and verify shapes / class balance."""
    data = generate_dataset(n_candidates=200, n_jobs=20, n_interactions=600,
                            seed=seed)
    return {
        "n_jobs": len(data["jobs"]),
        "n_candidates": len(data["candidates"]),
        "n_interactions": len(data["interactions"]),
        "group_0_rate": float(np.mean([c.group == 0 for c in data["candidates"]])),
        "positive_outcome_rate": float(np.mean([i.reward for i in data["interactions"]])),
    }


def exercise_2(seed: int = 0):
    """Show the BIAS in the logged dataset (group-conditional recommend rate)."""
    data = generate_dataset(n_candidates=400, n_jobs=20, n_interactions=2000, seed=seed)
    groups = []; acts = []
    cands_by_id = {c.candidate_id: c for c in data["candidates"]}
    for it in data["interactions"]:
        c = cands_by_id[it.candidate_id]
        groups.append(c.group); acts.append(it.action)
    return demographic_parity_gap(np.array(acts), np.array(groups))


def exercise_3(seed: int = 0):
    """IQL pretrain: returns the trained model + final losses."""
    data = generate_dataset(n_candidates=400, n_jobs=20, n_interactions=2000, seed=seed)
    iql_cfg = IQLConfig(n_epochs=10)
    out = iql_pretrain(data, cfg=iql_cfg, seed=seed)
    return {"history": out["history"], "data": data, "model": out["model"]}


def exercise_4(seed: int = 0):
    """SNIPS off-policy evaluation of a uniformly random target policy."""
    data = generate_dataset(n_candidates=400, n_jobs=20, n_interactions=2000, seed=seed)
    # uniform target picks each action with prob 0.5.
    target_probs = np.full(len(data["interactions"]), 0.5)
    return float(snips_ope(data["interactions"], target_probs))


def exercise_5(seed: int = 0):
    """PPO-Lagrangian fine-tune. Returns final parity gap and the history."""
    data = generate_dataset(n_candidates=400, n_jobs=20, n_interactions=2000, seed=seed)
    iql_out = iql_pretrain(data, cfg=IQLConfig(n_epochs=10), seed=seed)
    ppo_out = ppo_lag_finetune(
        data, iql_out["model"],
        cfg=PPOLagConfig(n_updates=12, steps_per_update=256, cost_limit=0.08),
        seed=seed,
    )
    return {"history": ppo_out["history"], "final_parity": ppo_out["history"]["parity"][-1]}


def exercise_6(seed: int = 0):
    """End-to-end: data -> IQL -> PPO-Lag -> audit. Return full audit dict
    BEFORE and AFTER finetune so the learner can compare."""
    out = _train(seed=seed, iql_cfg=IQLConfig(n_epochs=8),
                 ppo_cfg=PPOLagConfig(n_updates=10, steps_per_update=256),
                 n_candidates=400, n_jobs=20, n_interactions=2000)
    return {"audit_before": out["audit_before"], "audit_after": out["audit_after"]}


def exercise_7(seed: int = 0):
    """Red-team: counterfactual-fairness probe.

    Returns the fraction of (job, candidate) pairs whose top-1 decision FLIPS
    when we flip the candidate's protected-attribute bit.
    """
    out = exercise_6(seed=seed)
    return out["audit_after"]["red_team"]["group_flip_disagreement"]


def exercise_8(seed: int = 0):
    """Open-ended (★★★): vary cost_limit in {0.0, 0.05, 0.10, 0.20} and report
    the (mean reward, parity gap) trade-off. This is the *Pareto frontier* of
    fair re-ranking."""
    pts = []
    for cl in (0.0, 0.05, 0.10, 0.20):
        out = _train(seed=seed, iql_cfg=IQLConfig(n_epochs=6),
                     ppo_cfg=PPOLagConfig(n_updates=8, steps_per_update=256, cost_limit=cl),
                     n_candidates=400, n_jobs=20, n_interactions=2000)
        a = out["audit_after"]
        pts.append({
            "cost_limit": cl,
            "parity_gap": a["demographic_parity_gap"],
            "rec_rate": a["mean_rec_rate"],
            "snips_value": a["snips_value"],
        })
    return pts
