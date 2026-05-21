"""Reference solutions for the 10 graded workbook exercises (week 12)."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import numpy as np

from .env import ATSDataset, SKILLS
from .two_tower import train_two_tower, all_scores
from .reranker import LinTSReranker, make_features
from .cltr import position_bias, simulate_clicks, ips_ltr_value, expected_relevance
from .fairness import (
    demographic_parity_gap,
    equalised_opportunity_gap,
    exposure_by_group,
)
from .calibration import PlattCalibrator, IsotonicCalibrator, BetaCalibrator, ece, brier
from .train import ndcg_at_k, simulate_hire_logs


def exercise_1_jaccard(skills_a, skills_b) -> float:
    """E1 (★) - Jaccard similarity between two skill sets."""
    a = set(skills_a)
    b = set(skills_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def exercise_2_position_bias(K: int = 10, eta: float = 1.0) -> np.ndarray:
    """E2 (★) - Return p_k for positions 1..K."""
    return position_bias(K=K, eta=eta)


def exercise_3_two_tower_ndcg(seed: int = 0, epochs: int = 40) -> Tuple[float, float]:
    """E3 (★★) - Train two-tower, return (NDCG@10, NDCG@5)."""
    ds = ATSDataset(seed=seed, gender_skill_bias=0.25)
    model, _ = train_two_tower(ds, epochs=epochs, seed=seed)
    scores = all_scores(model, ds)
    rel = ds.true_match_matrix()
    return float(ndcg_at_k(scores, rel, k=10)), float(ndcg_at_k(scores, rel, k=5))


def exercise_4_ips_ltr(seed: int = 0) -> Tuple[float, float]:
    """E4 (★★) - simulate biased clicks for a logged ranking, estimate the
    expected-click value of a target ranking using IPS-LTR. Compare to the
    naive (un-corrected) estimator."""
    rng = np.random.default_rng(seed)
    n_items = 12
    relevance = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05])
    pb = position_bias(n_items, eta=1.0)
    logged = rng.permutation(n_items)
    clicks = simulate_clicks(relevance, logged, pb, seed=seed)
    target = np.argsort(-relevance)
    naive = float(clicks.sum())
    ips = ips_ltr_value(clicks, logged, pb, target)
    return naive, ips


def exercise_5_reranker_improvement(seed: int = 0, epochs: int = 40) -> Tuple[float, float]:
    """E5 (★★) - Train two-tower then LinTS reranker; return (tt_ndcg, rerank_ndcg)."""
    from .train import train
    out = train(seed=seed, epochs=epochs, n_logs_per_job=30)
    return float(out.metrics["tt_ndcg@10"]), float(out.metrics["rerank_ndcg@10"])


def exercise_6_dp_gap(seed: int = 0, epochs: int = 30) -> float:
    """E6 (★★) - Compute demographic-parity gap on the two-tower top-10."""
    ds = ATSDataset(seed=seed, gender_skill_bias=0.25)
    model, _ = train_two_tower(ds, epochs=epochs, seed=seed)
    scores = all_scores(model, ds)
    groups = np.array([c.gender for c in ds.candidates])
    return float(demographic_parity_gap(scores, groups, k=10))


def exercise_7_eo_gap(seed: int = 0, epochs: int = 30) -> float:
    """E7 (★★) - Equalised opportunity gap (TPR at qualification threshold 0.4)."""
    ds = ATSDataset(seed=seed, gender_skill_bias=0.25)
    model, _ = train_two_tower(ds, epochs=epochs, seed=seed)
    scores = all_scores(model, ds)
    groups = np.array([c.gender for c in ds.candidates])
    rel = ds.true_match_matrix()
    qualified = (rel >= 0.4).astype(int)
    return float(equalised_opportunity_gap(scores, groups, qualified, k=10))


def exercise_8_calibration(seed: int = 0) -> Dict[str, float]:
    """E8 (★★) - Fit Platt, Isotonic, Beta calibrators; return ECE for each."""
    rng = np.random.default_rng(seed)
    n = 1000
    # Synthetic over-confident scores: target probs are ~ score^2.
    raw = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < raw ** 2).astype(int)
    half = n // 2
    train_idx, eval_idx = np.arange(half), np.arange(half, n)
    out: Dict[str, float] = {"raw": ece(raw[eval_idx], labels[eval_idx])}
    for name, klass in (("platt", PlattCalibrator), ("isotonic", IsotonicCalibrator),
                        ("beta", BetaCalibrator)):
        cal = klass().fit(raw[train_idx], labels[train_idx])
        out[name] = ece(cal(raw[eval_idx]), labels[eval_idx])
    return out


def exercise_9_exposure(seed: int = 0, epochs: int = 30) -> Tuple[float, float]:
    """E9 (★★) - Exposure-by-group on the two-tower scores."""
    ds = ATSDataset(seed=seed, gender_skill_bias=0.25)
    model, _ = train_two_tower(ds, epochs=epochs, seed=seed)
    scores = all_scores(model, ds)
    groups = np.array([c.gender for c in ds.candidates])
    return exposure_by_group(scores, groups)


def exercise_10_ats_recommendation_memo(seed: int = 0, epochs: int = 40) -> Dict[str, float]:
    """E10 (★★★, open-ended) - The flagship objective exercise.

    Build the full ATS pipeline, audit fairness pre- and post-reranker,
    calibrate the final P(hire|c, j), and return a dictionary summarising
    the decision-relevant metrics for the recommendation memo:
        - NDCG@10 (two-tower) and (reranked)
        - DP gap (pre, post)
        - EO gap (pre, post)
        - ECE (raw, beta-calibrated)
    """
    from .train import train
    out = train(seed=seed, epochs=epochs)
    return {
        "tt_ndcg@10": float(out.metrics["tt_ndcg@10"]),
        "rerank_ndcg@10": float(out.metrics["rerank_ndcg@10"]),
        "dp_gap_pre": float(out.fairness_pre["dp_gap"]),
        "dp_gap_post": float(out.fairness_post["dp_gap"]),
        "eo_gap_pre": float(out.fairness_pre["eo_gap"]),
        "eo_gap_post": float(out.fairness_post["eo_gap"]),
        "ece_raw": float(out.calibration["raw"]["ece"]),
        "ece_beta": float(out.calibration["beta"]["ece"]),
    }
