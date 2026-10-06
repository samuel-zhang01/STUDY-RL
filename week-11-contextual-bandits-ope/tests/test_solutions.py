"""Tests that the workbook reference solutions satisfy their contracts."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)


def test_ucb_bonus():
    from project.solutions import exercise_1_ucb_bonus
    val = exercise_1_ucb_bonus(t=100, n_a=10)
    assert abs(val - np.sqrt(2 * np.log(100) / 10)) < 1e-9


def test_klucb_upper_monotone():
    from project.solutions import exercise_2_klucb_upper
    a = exercise_2_klucb_upper(0.5, 10, 100)
    b = exercise_2_klucb_upper(0.5, 100, 100)
    assert a > 0.5 and b > 0.5
    # More samples -> tighter upper bound.
    assert b < a


def test_thompson_regret_sublinear():
    from project.solutions import exercise_3_thompson_regret
    reg = exercise_3_thompson_regret(T=600, seed=0)
    assert reg < 0.6 * 600, f"TS regret {reg} too high"


def test_linucb_regret():
    from project.solutions import exercise_4_linucb_regret
    reg = exercise_4_linucb_regret(T=400, seed=0)
    assert reg < 0.6 * 400


def test_ips_snips_uniform():
    from project.solutions import exercise_5_ips_snips_uniform
    v_ips, v_snips = exercise_5_ips_snips_uniform(n=300, seed=0)
    assert 0.0 <= v_snips <= 1.0
    assert -0.2 <= v_ips <= 1.5


def test_dr_softmax_close_to_truth():
    from project.solutions import exercise_6_dr_softmax
    from project.env import SyntheticLeadBandit
    from project.ope import true_policy_value
    from project.train import softmax_policy
    env = SyntheticLeadBandit(seed=0)
    truth = true_policy_value(env, softmax_policy(env, beta=4.0))
    est = exercise_6_dr_softmax(n=1200, seed=0)
    assert abs(est - truth) < 0.25, (est, truth)


def test_ess():
    from project.solutions import exercise_7_ess
    w = np.ones(100)
    assert abs(exercise_7_ess(w) - 100.0) < 1e-9
    w2 = np.array([10.0] + [0.0] * 99)
    assert exercise_7_ess(w2) < 2.0


def test_switch_dr():
    from project.solutions import exercise_8_switch_dr
    v = exercise_8_switch_dr(n=600, tau=5.0, seed=0)
    assert -0.1 <= v <= 1.2


def test_synthetic_ope_estimators_consistent():
    from project.solutions import exercise_9_synthetic_ope
    truth, snips_est, dr_est = exercise_9_synthetic_ope(n=1500, seed=0)
    # DR should be at least as accurate as SNIPS on this benign target.
    assert abs(dr_est - truth) <= abs(snips_est - truth) + 0.25


def test_training_oracle_matches_logged_reward_environment(monkeypatch):
    import project.train as trainer
    from project.env import SyntheticLeadBandit
    # The oracle is defined for seed 3's mechanisms; logger must use those
    # mechanisms too, even if it has an independent policy RNG seed.
    original = trainer.collect_logs
    oracle_theta = SyntheticLeadBandit(seed=3).theta
    def collect(env, *args, **kwargs):
        np.testing.assert_array_equal(env.theta, oracle_theta)
        return original(env, *args, **kwargs)
    monkeypatch.setattr(trainer, 'collect_logs', collect)
    result = trainer.train(seed=3, T=2, n_log=20)
    assert all(np.isfinite(v) for v in result.ope_truth.values())
