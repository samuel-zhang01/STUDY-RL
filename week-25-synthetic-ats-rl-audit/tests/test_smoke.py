"""Smoke tests for week 25."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)


def test_imports():
    import project  # noqa: F401
    from project import bandit, constrained, env, ope, scorer, shield  # noqa: F401


def test_train_runs_and_lints_competitive():
    from project.train import train

    res = train(n_candidates=150, n_jobs=25, n_rounds=400, seed=0)
    # LinTS should not be the *worst* policy by SNIPS.
    snips_vals = res.snips_estimates
    assert snips_vals["LinTS"] >= snips_vals["Uniform"] - 0.05, snips_vals


def test_constrained_runs():
    from project.constrained import train_lagrangian
    from project.env import generate_population

    data = generate_population(n_candidates=100, n_jobs=20, seed=0)
    stats = train_lagrangian(data, n_rounds=200)
    assert len(stats.rewards) == 200
    # Final B/A parity ratio should be > 0.5 after fairness control kicks in.
    assert stats.parity_gap[-1] > 0.5


def test_shield_blocks():
    import numpy as np
    from project.shield import HardConstraintViolation, shielded_rerank

    out = shielded_rerank(np.array([3, 1, 4, 1, 5, 9, 2]), blocked=[1, 9], top_k=3)
    assert list(out) == [3, 4, 5]
    try:
        shielded_rerank(np.array([1, 2, 3]), blocked=[1, 2, 3], top_k=1)
    except HardConstraintViolation:
        return
    raise AssertionError("expected HardConstraintViolation")


def test_audit_report_format():
    import numpy as np
    from project.audit import format_report, run_audit
    from project.env import generate_population

    data = generate_population(n_candidates=80, n_jobs=15, seed=0)
    chosen = np.arange(20)
    jobs = np.zeros(20, dtype=np.int64)
    metrics = run_audit(data, chosen, jobs)
    rep = format_report(metrics)
    assert "audit report" in rep
    assert "parity_ratio" in rep


def test_cli_match_score():
    from project.cli import match_score

    cv = {"skills": [0.5] * 12, "years": 5.0, "group": 0}
    jd = {"required_skills": [0.5] * 12, "seniority": 0.5}
    out = match_score(cv, jd)
    assert 0 <= out["score"] <= 100
    assert "fairness_flag" in out
