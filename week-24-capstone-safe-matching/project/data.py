"""Synthetic candidate-matching dataset for the capstone.

We synthesise:
  * ~50 jobs, each with a 16-dimensional skill-requirement vector and a small
    bias toward group 'A' or 'B' for the protected attribute (the bias is the
    historical hiring distortion we want to neutralise).
  * ~1000 candidates, each with the same 16-dim skill vector and a binary
    protected attribute g in {0, 1} drawn ~Bernoulli(0.5).
  * A *logged* dataset: for each (job, candidate) pair in a held-in subsample,
    a logging policy (epsilon-greedy on a noisy skill-match scorer) picked
    'show' / 'skip' and recorded the outcome (positive interview if skill_match
    > threshold AND a noisy biased prior in the logging policy was satisfied).

Two important things about the synthesis:
  * The true relevance is a clean function of skill overlap, with NO
    dependence on the protected attribute.
  * The logging policy is biased toward group 0 (mimicking historical
    discrimination). So the offline pretrain inherits the bias; the *online*
    Lagrangian phase has to correct it.

A "context" for the policy is `(job_skill_vec, candidate_skill_vec, candidate_group)`.
The action is binary: 1 = recommend, 0 = skip. The reward is +1 if a positive
interview happens, 0 otherwise. The cost (for the CMDP) is the *per-recommendation
demographic-parity gap*: the absolute difference between the empirical
recommendation rate on group 0 and group 1 over the most recent window.

All data lives in pure Python / numpy structures; no I/O is required to use the
package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np


SKILL_DIM = 16


@dataclass
class JobPosting:
    job_id: int
    skills: np.ndarray   # (SKILL_DIM,)
    title: str = "Engineer"


@dataclass
class Candidate:
    candidate_id: int
    skills: np.ndarray
    group: int   # 0 or 1, protected attribute
    name: str = ""


@dataclass
class LoggedInteraction:
    job_id: int
    candidate_id: int
    action: int          # 1 = recommended, 0 = skipped
    propensity: float    # P(action | context) under the logging policy
    reward: int          # 1 = positive interview, 0 = otherwise
    true_relevance: float


# ---------------------------------------------------------------------------
def _skills_for_job(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0, 1, size=SKILL_DIM)
    return v / np.linalg.norm(v)


def _skills_for_candidate(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0, 1, size=SKILL_DIM)
    return v / np.linalg.norm(v)


def _true_relevance(job: JobPosting, cand: Candidate) -> float:
    """Cosine similarity scaled into [0, 1]. Group-agnostic."""
    return float((job.skills @ cand.skills + 1.0) / 2.0)


def _logging_policy_score(job: JobPosting, cand: Candidate,
                          group_bias: float = 0.6) -> float:
    """Noisy score the *logging* policy uses to pick actions, with a bias toward
    group 0 (mimicking historical discrimination)."""
    rel = _true_relevance(job, cand)
    bias = group_bias if cand.group == 0 else (1.0 - group_bias)
    # Some noise + bias.
    return rel * bias


def generate_dataset(n_candidates: int = 1000, n_jobs: int = 50,
                     n_interactions: int = 4000, eps: float = 0.1,
                     seed: int = 0) -> Dict:
    """Generate jobs, candidates, and a logged interaction dataset.

    Returns a dict:
        jobs (List[JobPosting])
        candidates (List[Candidate])
        interactions (List[LoggedInteraction])
        logging_eps (float)
    """
    rng = np.random.default_rng(seed)
    jobs = [JobPosting(job_id=i, skills=_skills_for_job(rng),
                       title=f"Role-{i}") for i in range(n_jobs)]
    candidates = [Candidate(candidate_id=i, skills=_skills_for_candidate(rng),
                            group=int(rng.integers(0, 2)),
                            name=f"C{i:04d}") for i in range(n_candidates)]

    interactions: List[LoggedInteraction] = []
    for _ in range(n_interactions):
        j = jobs[int(rng.integers(0, n_jobs))]
        c = candidates[int(rng.integers(0, n_candidates))]
        score = _logging_policy_score(j, c)
        # eps-greedy: with prob eps pick a uniformly random action, else act
        # according to the logging score (threshold 0.5 chosen so ~50% of pairs
        # are recommended on average).
        p_show = (1 - eps) * (1.0 if score > 0.5 else 0.0) + eps * 0.5
        a = int(rng.random() < p_show)
        rel = _true_relevance(j, c)
        # Reward only depends on true relevance and a noisy threshold.
        outcome = int((rel > 0.55) and (rng.random() < 0.9))
        reward = outcome if a == 1 else 0
        # Recover the propensity of *the chosen action*.
        prop = p_show if a == 1 else (1.0 - p_show)
        interactions.append(LoggedInteraction(
            job_id=j.job_id, candidate_id=c.candidate_id,
            action=a, propensity=max(1e-3, prop), reward=reward,
            true_relevance=rel,
        ))
    return {
        "jobs": jobs,
        "candidates": candidates,
        "interactions": interactions,
        "logging_eps": eps,
    }


def context_vector(job: JobPosting, cand: Candidate,
                   include_group: bool = True) -> np.ndarray:
    """Featurise a (job, candidate) into a flat vector for the policy."""
    if include_group:
        return np.concatenate([job.skills, cand.skills, [float(cand.group)]]).astype(np.float32)
    return np.concatenate([job.skills, cand.skills]).astype(np.float32)


def context_dim(include_group: bool = True) -> int:
    return SKILL_DIM * 2 + (1 if include_group else 0)
