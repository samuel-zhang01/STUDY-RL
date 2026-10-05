"""Fairness + calibration audit. Prints a short report; returns metrics dict."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .env import ATSData, true_match_prob


def selection_rates(chosen_group: np.ndarray) -> Dict[str, float]:
    """Selected-group shares, not selection rates conditional on exposure.

    These cannot implement a four-fifths hiring-rate test without applicant
    denominators for each group. Keep the original keys for workbook callers.
    """
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
        mask = (probs >= lo) & ((probs <= hi) if hi == 1.0 else (probs < hi))
        if mask.sum() == 0:
            out.append((float(lo), float(hi), float("nan"), 0))
        else:
            out.append((float(lo), float(hi), float(outcomes[mask].mean()), int(mask.sum())))
    return out


def equalised_odds_gap(scores: np.ndarray, groups: np.ndarray, outcomes: np.ndarray,
                       threshold: float = 0.5) -> float:
    """Equalized-odds gap: maximum of the group TPR and FPR differences.

    Returns NaN if either conditional rate is unobservable in either group.
    """
    preds = scores >= threshold
    gaps = []
    for label in (0, 1):
        rates = []
        for g in (0, 1):
            mask = (groups == g) & (outcomes == label)
            if not mask.any():
                return float("nan")
            rates.append(float(preds[mask].mean()))
        gaps.append(abs(rates[0] - rates[1]))
    return float(max(gaps))


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
        lines.append("  Toy selected-group share ratio < 0.8; group exposure denominators are not recorded.")
    return "\n".join(lines)
