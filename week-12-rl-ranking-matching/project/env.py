"""Synthetic ATS candidate <-> job matching environment.

We bundle a small dataset of 200 fake candidates and 50 fake jobs. Each
candidate has a *skill bag* (a subset of a 24-skill vocabulary) and a
gender-coded first name (used only for the fairness audit). Each job has a
*required-skill bag* and a *nice-to-have* bag. Ground-truth match score is
the Jaccard similarity between candidate skills and required + 0.5 *
nice-to-have skills. Click / hire outcomes are drawn from a logistic on
this score plus position-bias (for the CLTR exercises).

The bundle ships as a pure-Python dataclass, no network calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple
import numpy as np


SKILLS: Tuple[str, ...] = (
    "python", "sql", "spark", "aws", "ml", "deep-learning", "nlp", "vision",
    "java", "scala", "kubernetes", "terraform", "react", "typescript",
    "fastapi", "postgres", "tableau", "excel", "stats", "causal-inference",
    "leadership", "communication", "spanish", "french",
)

# Gender-coded first names used only for the fairness audit. The model never
# sees these as input features; we audit whether a model that uses skills
# *alone* still discriminates because of skill <-> gender correlations in the
# generated data (the canonical "proxy variable" failure mode).
F_NAMES = ("Alice", "Beth", "Clara", "Diana", "Elena", "Fiona", "Grace",
           "Hannah", "Iris", "Julia", "Kira", "Lena", "Maya", "Nora", "Olivia",
           "Petra", "Quinn", "Rachel", "Sara", "Tessa", "Uma", "Vera",
           "Wendy", "Xenia", "Yara", "Zara")
M_NAMES = ("Adam", "Ben", "Carl", "David", "Eric", "Fred", "George",
           "Harry", "Ian", "Jack", "Kyle", "Leo", "Mike", "Nate", "Owen",
           "Paul", "Quentin", "Ryan", "Sam", "Tom", "Umar", "Victor",
           "Will", "Xavier", "Yusuf", "Zane")


@dataclass
class Candidate:
    cid: int
    name: str
    gender: str   # "F" or "M" -- used only for the audit
    skills: Tuple[str, ...]


@dataclass
class Job:
    jid: int
    title: str
    required: Tuple[str, ...]
    nice_to_have: Tuple[str, ...]


def _skills_to_vec(skills: Sequence[str]) -> np.ndarray:
    v = np.zeros(len(SKILLS), dtype=np.float64)
    idx = {s: i for i, s in enumerate(SKILLS)}
    for s in skills:
        v[idx[s]] = 1.0
    return v


@dataclass
class ATSDataset:
    n_candidates: int = 200
    n_jobs: int = 50
    seed: int = 0
    # If >0, inject a gender-skill correlation: female candidates are LESS
    # likely to list "leadership" and MORE likely to list "communication",
    # mirroring real CV-corpus biases.
    gender_skill_bias: float = 0.25

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.candidates = self._make_candidates(rng)
        self.jobs = self._make_jobs(rng)
        # Pre-compute skill vectors for speed.
        self.cand_vecs = np.stack([_skills_to_vec(c.skills) for c in self.candidates])
        self.job_req_vecs = np.stack([_skills_to_vec(j.required) for j in self.jobs])
        self.job_nice_vecs = np.stack([_skills_to_vec(j.nice_to_have) for j in self.jobs])

    def _make_candidates(self, rng: np.random.Generator) -> List[Candidate]:
        out: List[Candidate] = []
        for cid in range(self.n_candidates):
            gender = "F" if rng.uniform() < 0.5 else "M"
            name = rng.choice(F_NAMES if gender == "F" else M_NAMES)
            # Each candidate has 4-8 skills, sampled with a gendered bias.
            n_skills = int(rng.integers(4, 9))
            probs = np.ones(len(SKILLS)) / len(SKILLS)
            if gender == "F":
                # less leadership, more communication.
                probs[SKILLS.index("leadership")] *= max(1e-3, 1.0 - self.gender_skill_bias)
                probs[SKILLS.index("communication")] *= 1.0 + self.gender_skill_bias
            else:
                probs[SKILLS.index("leadership")] *= 1.0 + self.gender_skill_bias
                probs[SKILLS.index("communication")] *= max(1e-3, 1.0 - self.gender_skill_bias)
            probs = probs / probs.sum()
            skills = tuple(rng.choice(SKILLS, size=n_skills, replace=False, p=probs))
            out.append(Candidate(cid=cid, name=str(name), gender=gender, skills=skills))
        return out

    def _make_jobs(self, rng: np.random.Generator) -> List[Job]:
        titles = ("ML Engineer", "Data Scientist", "Backend Engineer",
                  "Frontend Engineer", "Data Analyst", "DevOps Engineer")
        out: List[Job] = []
        for jid in range(self.n_jobs):
            title = str(rng.choice(titles))
            n_req = int(rng.integers(3, 6))
            n_nice = int(rng.integers(1, 4))
            req = tuple(rng.choice(SKILLS, size=n_req, replace=False))
            remaining = [s for s in SKILLS if s not in req]
            nice = tuple(rng.choice(remaining, size=n_nice, replace=False))
            out.append(Job(jid=jid, title=f"{title} #{jid}", required=req, nice_to_have=nice))
        return out

    # ----- match scoring -----
    def true_match(self, cid: int, jid: int) -> float:
        c = self.cand_vecs[cid]
        r = self.job_req_vecs[jid]
        n = self.job_nice_vecs[jid]
        # Jaccard on required + 0.5 weight on nice-to-have intersection.
        req_inter = float((c * r).sum())
        req_union = float(((c + r) > 0).sum())
        jacc = req_inter / max(req_union, 1.0)
        nice_inter = float((c * n).sum()) / max(float(n.sum()), 1.0)
        return float(np.clip(jacc + 0.5 * nice_inter, 0.0, 1.0))

    def true_match_matrix(self) -> np.ndarray:
        # (n_jobs, n_candidates)
        M = np.zeros((self.n_jobs, self.n_candidates))
        for j in range(self.n_jobs):
            for c in range(self.n_candidates):
                M[j, c] = self.true_match(c, j)
        return M

    def candidate_feature(self, cid: int) -> np.ndarray:
        return self.cand_vecs[cid]

    def job_feature(self, jid: int) -> np.ndarray:
        return np.concatenate([self.job_req_vecs[jid], self.job_nice_vecs[jid]])
