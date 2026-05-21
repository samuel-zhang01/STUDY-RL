"""End-to-end training and evaluation script.

Pipeline:
1. Build ATSDataset.
2. Train a two-tower retriever (~15s on CPU).
3. Train a LinTS reranker on top of two-tower scores against simulated hire
   feedback (Bernoulli outcomes on the true match score).
4. Run a fairness audit on the pre- and post-reranker scores.
5. Calibrate the final P(hire|cand, job) with Platt, Isotonic, Beta and
   report ECE / Brier.

Target metrics (default seed):
- Two-tower NDCG@10 > 0.5
- LinTS reranker NDCG@10 > two-tower NDCG@10 (some improvement)
- Calibration: Beta-calibration ECE < raw-score ECE

Runtime: ~30s on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import torch

from .env import ATSDataset
from .two_tower import train_two_tower, all_scores, topk_for_job
from .reranker import LinTSReranker, make_features
from .fairness import demographic_parity_gap, equalised_opportunity_gap, exposure_by_group
from .calibration import PlattCalibrator, IsotonicCalibrator, BetaCalibrator, ece, brier


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def dcg(relevances: np.ndarray) -> float:
    K = len(relevances)
    return float(np.sum(relevances / np.log2(np.arange(2, K + 2))))


def ndcg_at_k(scores: np.ndarray, relevance_matrix: np.ndarray, k: int = 10) -> float:
    """Mean NDCG@k across jobs.
    `scores` and `relevance_matrix` are (n_jobs, n_candidates)."""
    n_jobs = scores.shape[0]
    out = 0.0
    for j in range(n_jobs):
        order = np.argsort(-scores[j])[:k]
        rels = relevance_matrix[j, order]
        ideal = np.sort(relevance_matrix[j])[::-1][:k]
        dcg_val = dcg(rels)
        idcg = dcg(ideal)
        out += dcg_val / max(idcg, 1e-9)
    return float(out / n_jobs)


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

@dataclass
class TrainOutput:
    losses: np.ndarray
    metrics: Dict[str, float]
    tt_scores: np.ndarray
    rerank_scores: np.ndarray
    fairness_pre: Dict[str, float]
    fairness_post: Dict[str, float]
    calibration: Dict[str, Dict[str, float]]


def simulate_hire_logs(ds, scores: np.ndarray, rng: np.random.Generator,
                        n_per_job: int = 30, eps: float = 0.3, top_n: int = 20):
    """Simulate (cand, hire) outcomes drawn from a Bernoulli on the true match.
    Logger is epsilon-greedy on `scores` (top-N ordered then exploration)."""
    logs = []
    n_cands = scores.shape[1]
    for j in range(scores.shape[0]):
        order = np.argsort(-scores[j])[:top_n]
        for _ in range(n_per_job):
            if rng.uniform() < eps:
                c = int(rng.choice(order))
                p_b = (1 - eps) * (1.0 / top_n if c in order else 0.0) + eps / top_n
            else:
                c = int(order[0])
                p_b = (1 - eps) + eps / top_n
            mu = ds.true_match(c, j)
            r = int(rng.binomial(1, mu))
            logs.append((j, c, r, p_b, mu))
    return logs


def train(
    seed: int = 0,
    epochs: int = 60,
    n_logs_per_job: int = 30,
    top_n: int = 20,
    k_eval: int = 10,
) -> TrainOutput:
    rng = np.random.default_rng(seed)
    ds = ATSDataset(seed=seed, gender_skill_bias=0.25)

    # 1. two-tower
    model, losses = train_two_tower(ds, epochs=epochs, seed=seed)
    tt_scores = all_scores(model, ds)

    # 2. ground-truth relevance for evaluation
    rel = ds.true_match_matrix()
    tt_ndcg = ndcg_at_k(tt_scores, rel, k=k_eval)

    # 3. LinTS reranker training
    feat_dim = ds.cand_vecs.shape[1] + 2  # skills + tt_score + bias
    reranker = LinTSReranker(n_jobs=ds.n_jobs, d=feat_dim, v=0.5, seed=seed + 1)
    # Sweep through (job, candidate) logs
    logs = simulate_hire_logs(ds, tt_scores, rng, n_per_job=n_logs_per_job, top_n=top_n)
    for j, c, r, _pb, _mu in logs:
        x = make_features(ds, j, c, float(tt_scores[j, c]))
        reranker.update(j, x, float(r))

    # Rerank: replace tt_scores with the reranker mean-score over top-N.
    rerank_scores = tt_scores.copy()
    for j in range(ds.n_jobs):
        order = np.argsort(-tt_scores[j])[:top_n]
        feats = np.stack([make_features(ds, j, c, float(tt_scores[j, c])) for c in order])
        mean = reranker.mean_score(j, feats)
        # Replace just those positions; leave others as tt scores.
        rerank_scores[j, order] = mean + 100.0  # ensure top-N stays on top

    rerank_ndcg = ndcg_at_k(rerank_scores, rel, k=k_eval)

    # 4. Fairness audit (pre = two-tower; post = reranker).
    groups = np.array([c.gender for c in ds.candidates])
    qualified = (rel >= 0.4).astype(int)
    fairness_pre = {
        "dp_gap": demographic_parity_gap(tt_scores, groups, k=k_eval),
        "eo_gap": equalised_opportunity_gap(tt_scores, groups, qualified, k=k_eval),
    }
    exp_f, exp_m = exposure_by_group(tt_scores, groups)
    fairness_pre["exposure_F"] = exp_f
    fairness_pre["exposure_M"] = exp_m

    fairness_post = {
        "dp_gap": demographic_parity_gap(rerank_scores, groups, k=k_eval),
        "eo_gap": equalised_opportunity_gap(rerank_scores, groups, qualified, k=k_eval),
    }
    exp_f2, exp_m2 = exposure_by_group(rerank_scores, groups)
    fairness_post["exposure_F"] = exp_f2
    fairness_post["exposure_M"] = exp_m2

    # 5. Calibration: turn cosine similarity (in [-1, 1]) into a probability.
    # Build (score, label) pairs from a held-out sample.
    test_logs = simulate_hire_logs(ds, tt_scores, np.random.default_rng(seed + 999),
                                    n_per_job=20, eps=0.5, top_n=top_n)
    raw = np.array([0.5 * (tt_scores[j, c] + 1.0) for (j, c, _r, _pb, _mu) in test_logs])
    raw = np.clip(raw, 0.0, 1.0)
    labels = np.array([r for (_j, _c, r, _pb, _mu) in test_logs])
    # Split 50/50 train/eval.
    idx = np.arange(len(raw))
    np.random.default_rng(seed).shuffle(idx)
    half = len(raw) // 2
    train_idx, eval_idx = idx[:half], idx[half:]

    calibration = {}
    calibration["raw"] = {
        "ece": ece(raw[eval_idx], labels[eval_idx]),
        "brier": brier(raw[eval_idx], labels[eval_idx]),
    }
    for name, klass in (("platt", PlattCalibrator), ("isotonic", IsotonicCalibrator),
                        ("beta", BetaCalibrator)):
        cal = klass().fit(raw[train_idx], labels[train_idx])
        p = cal(raw[eval_idx])
        calibration[name] = {
            "ece": ece(p, labels[eval_idx]),
            "brier": brier(p, labels[eval_idx]),
        }

    metrics = {
        "tt_ndcg@10": tt_ndcg,
        "rerank_ndcg@10": rerank_ndcg,
        "n_logs": len(logs),
    }
    return TrainOutput(losses=losses, metrics=metrics, tt_scores=tt_scores,
                       rerank_scores=rerank_scores,
                       fairness_pre=fairness_pre, fairness_post=fairness_post,
                       calibration=calibration)


if __name__ == "__main__":  # pragma: no cover
    out = train()
    print("metrics:", out.metrics)
    print("fairness pre:", out.fairness_pre)
    print("fairness post:", out.fairness_post)
    print("calibration:", out.calibration)
