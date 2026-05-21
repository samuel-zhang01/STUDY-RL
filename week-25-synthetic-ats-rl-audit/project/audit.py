"""Fairness + calibration audit. Prints a short report; returns metrics dict."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .env import ATSData, true_match_prob


def selection_rates(chosen_group: np.ndarray) -> Dict[str, float]:
    n = len(chosen_group)
    if n == 0:
        return {"A_rate": 0.0, "B_rate": 0.0, "parity_ratio": 1.0}
    rate_A = float((chosen_group == 0).mean())
    rate_B = float((chosen_group == 1).mean())
    ratio = rate_B / max(rate_A, 1e-6)
    return {"A_rate": rate_A, "B_rate": rate_B, "parity_ratio": ratio}


def calibration_buckets(probs: np.ndarray, outcomes: np.ndarray, n_buckets: int = 10) -> List[tuple]:
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            out.append((float(lo), float(hi), float("nan"), 0))
        else:
            out.append((float(lo), float(hi), float(outcomes[mask].mean()), int(mask.sum())))
    return out


def equalised_odds_gap(scores: np.ndarray, groups: np.ndarray, outcomes: np.ndarray,
                       threshold: float = 0.5) -> float:
    """Gap in TPR between groups at a given threshold; smaller is fairer."""
    preds = scores >= threshold
    tpr = []
    for g in (0, 1):
        mask = (groups == g) & (outcomes == 1)
        if mask.sum() == 0:
            tpr.append(np.nan)
        else:
            tpr.append(float(preds[mask].mean()))
    return float(abs(tpr[0] - tpr[1]))


def run_audit(data: ATSData, chosen_cands: np.ndarray, jobs_seen: np.ndarray) -> Dict:
    """chosen_cands[i] = candidate idx chosen for job jobs_seen[i]."""
    groups = data.cand_group[chosen_cands]
    sr = selection_rates(groups)
    p = true_match_prob(data)
    # Simulate outcomes at chosen pairs
    rng = np.random.default_rng(0)
    outcomes = rng.binomial(1, p[chosen_cands, jobs_seen])
    metrics = {
        **sr,
        "mean_match_prob": float(p[chosen_cands, jobs_seen].mean()),
        "hire_rate_sim": float(outcomes.mean()),
    }
    return metrics


def format_report(metrics: Dict) -> str:
    lines = ["=== ATS audit report ==="]
    for k, v in metrics.items():
        lines.append(f"  {k:>22s}: {v:.4f}")
    if metrics.get("parity_ratio", 1.0) < 0.8:
        lines.append("  WARNING: parity_ratio < 0.8 (four-fifths rule violation)")
    return "\n".join(lines)
