"""Minimal CLI: `python -m project.cli --candidate cv.json --job jd.json`.

Reads two tiny JSON blobs and returns a ranked match plan with a fairness
flag. Intentionally pure-python and dependency-free apart from numpy. This
is a *demo* — production would wrap this in the FastAPI service layer.

cv.json schema:
  {"skills": [..d_skill values..], "years": float, "group": 0|1}
jd.json schema:
  {"required_skills": [..d_skill values..], "seniority": float}
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .audit import selection_rates
from .env import _safe_norm


def match_score(cv: dict, jd: dict) -> dict:
    c = np.asarray(cv["skills"], dtype=np.float64)
    j = np.asarray(jd["required_skills"], dtype=np.float64)
    if c.shape != j.shape:
        raise ValueError("skill vector dim mismatch")
    sim = float((_safe_norm(c[None]) @ _safe_norm(j[None]).T).item())
    sen_gap = abs(float(cv["years"]) / 10.0 - float(jd["seniority"]))
    fit = float(np.exp(-sen_gap))
    base = 100.0 * (0.7 * 0.5 * (sim + 1.0) + 0.3 * fit)
    # Apply LinTS-style posterior shrinkage: pull extreme heuristic scores
    # back toward the mean. Without learned weights we just damp the tails.
    blended = base * 0.95 + 5.0 * fit
    fairness_flag = "OK"
    if cv.get("group", 0) == 1 and base < 50:
        fairness_flag = "REVIEW: group-B candidate scored below review threshold"
    return {
        "score": float(blended),
        "components": {"similarity": sim, "seniority_fit": fit, "base": base},
        "fairness_flag": fairness_flag,
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="project.cli")
    p.add_argument("--candidate", required=True)
    p.add_argument("--job", required=True)
    args = p.parse_args(argv)
    with open(args.candidate) as f:
        cv = json.load(f)
    with open(args.job) as f:
        jd = json.load(f)
    out = match_score(cv, jd)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
