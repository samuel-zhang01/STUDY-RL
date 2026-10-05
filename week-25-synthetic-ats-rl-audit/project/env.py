"""Synthetic ATS environment.

Generates a fake population of candidates and jobs with a *skills overlap*
ground-truth reward (probability of "good match", i.e. hire) plus a binary
*sensitive attribute* (group A / group B) that affects recorded experience.
Skills are sampled independently of the group.

Schema mimics what a real ATS pipeline produces:
  candidate: id, skills (dense vector), years_experience, group
  job:       id, required_skills (dense vector), seniority

The reward is computed from a logistic of cosine similarity in skill-space
minus a seniority mismatch penalty. The sensitive attribute leaks into the
data because group-B candidates have systematically lower years_experience
even when their skills are equally strong — exactly the kind of confounded
signal a naive scorer will latch onto.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ATSData:
    candidates: np.ndarray         # (N, d_skill)
    cand_years: np.ndarray         # (N,)
    cand_group: np.ndarray         # (N,) in {0, 1}
    jobs: np.ndarray               # (J, d_skill)
    job_seniority: np.ndarray      # (J,)  in [0, 1]
    reward_logits: np.ndarray      # (N, J)  ground-truth match logit


def _safe_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def generate_population(
    n_candidates: int = 500,
    n_jobs: int = 80,
    d_skill: int = 12,
    bias_strength: float = 0.6,
    seed: int = 0,
) -> ATSData:
    rng = np.random.default_rng(seed)
    # Candidate skills: sparse-ish in [0,1]
    cand = rng.beta(0.8, 1.2, size=(n_candidates, d_skill)).astype(np.float64)
    # Sensitive group: roughly balanced
    group = rng.integers(0, 2, size=n_candidates)
    # Years experience — group B systematically penalised in *recorded* years
    # (e.g. career gaps), but underlying skill is unbiased.
    base_years = rng.gamma(2.0, 3.0, size=n_candidates)
    years = base_years - bias_strength * group * rng.uniform(0.5, 2.5, size=n_candidates)
    years = np.clip(years, 0.0, 30.0)

    jobs = rng.beta(1.0, 1.0, size=(n_jobs, d_skill)).astype(np.float64)
    job_sen = rng.uniform(0.2, 0.9, size=n_jobs)

    cand_n = _safe_norm(cand)
    jobs_n = _safe_norm(jobs)
    sim = cand_n @ jobs_n.T                                # (N, J) in [-1, 1]
    sen_gap = np.abs(years[:, None] / 10.0 - job_sen[None, :])  # (N, J)
    logits = 2.5 * sim - 1.2 * sen_gap - 1.5
    # No explicit group term appears in the reward, but group affects recorded
    # years and therefore indirectly affects reward. This is not a causally
    # fair reward construction or a measure of unbiased latent competence.
    return ATSData(
        candidates=cand,
        cand_years=years,
        cand_group=group,
        jobs=jobs,
        job_seniority=job_sen,
        reward_logits=logits,
    )


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def true_match_prob(data: ATSData) -> np.ndarray:
    """Ground-truth hire probability, (N, J)."""
    return sigmoid(data.reward_logits)


def sample_reward(data: ATSData, cand_idx: int, job_idx: int, rng: np.random.Generator) -> int:
    p = float(sigmoid(data.reward_logits[cand_idx, job_idx]))
    return int(rng.binomial(1, p))


def context_features(data: ATSData, cand_idx: int, job_idx: int) -> np.ndarray:
    """Feature vector used by the bandit policy.

    Includes: element-wise skill product, years (scaled), seniority, gap.
    Deliberately EXCLUDES the sensitive group to model fairness-through-
    unawareness — and as we show in chapter 6, that is not enough.
    """
    c = data.candidates[cand_idx]
    j = data.jobs[job_idx]
    # Normalised elementwise products sum to cosine similarity, so the
    # heuristic baseline can be reconstructed exactly from a logged slate.
    prod = _safe_norm(c) * _safe_norm(j)
    extras = np.array([
        data.cand_years[cand_idx] / 30.0,
        data.job_seniority[job_idx],
        abs(data.cand_years[cand_idx] / 10.0 - data.job_seniority[job_idx]),
        1.0,  # bias term
    ])
    return np.concatenate([prod, extras]).astype(np.float64)


def feature_dim(d_skill: int) -> int:
    return d_skill + 4
