"""Tests for the reference workbook solutions."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)


def test_e1_catalog():
    from project.solutions import exercise_1_decision_catalog
    d = exercise_1_decision_catalog()
    assert len(d) >= 5


def test_e2_hacks():
    from project.solutions import exercise_2_reward_hacks
    hacks = exercise_2_reward_hacks()
    assert len(hacks) >= 3
    for hack, mitig in hacks:
        assert hack and mitig


def test_e3_lints_reward_positive():
    from project.solutions import exercise_3_lints_reward
    r = exercise_3_lints_reward(T=200)
    assert r > 0


def test_e4_snips_ci():
    from project.solutions import exercise_4_snips_ci
    p, lo, hi = exercise_4_snips_ci()
    assert lo <= p <= hi


def test_e5_parity():
    from project.solutions import exercise_5_lagrangian_parity
    r = exercise_5_lagrangian_parity()
    assert r > 0.3


def test_e6_schema_has_propensity():
    from project.solutions import exercise_6_propensity_log_schema
    s = exercise_6_propensity_log_schema()
    assert "propensity" in s


def test_e7_shielded():
    from project.solutions import exercise_7_shielded_topk
    out = exercise_7_shielded_topk([5, 4, 3, 2, 1], blocked=[4], k=3)
    assert out == [5, 3, 2]


def test_e8_dpo_at_zero():
    import math
    from project.solutions import exercise_8_dpo_loss_at_zero
    assert abs(exercise_8_dpo_loss_at_zero() - math.log(2.0)) < 1e-6


def test_e9_dt_mse():
    from project.solutions import exercise_9_dt_linear_mse
    assert exercise_9_dt_linear_mse() < 0.5


def test_e10_phase1_audit():
    from project.solutions import exercise_10_phase1_audit
    m = exercise_10_phase1_audit()
    assert "parity_ratio" in m and "hire_rate_sim" in m
