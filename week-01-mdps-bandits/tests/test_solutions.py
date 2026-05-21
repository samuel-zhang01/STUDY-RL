"""Contract tests for the reference solutions used by ``workbook.ipynb``."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK = os.path.dirname(HERE)
REPO = os.path.dirname(WEEK)
sys.path.insert(0, REPO)
for _m in list(sys.modules):
    if _m == "project" or _m.startswith("project."):
        del sys.modules[_m]
sys.path.insert(0, WEEK)

from project import solutions as S  # noqa: E402


def test_exercise_1_incremental_mean_matches_numpy():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(50)
    out = S.exercise_1_incremental_mean(r)
    ref = np.cumsum(r) / np.arange(1, 51)
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_exercise_2_eps_greedy_regret_monotone():
    means = np.array([0.2, 0.5, 0.8])
    reg = S.exercise_2_eps_greedy_regret(means, T=200, eps=0.1, seed=0)
    assert reg.shape == (200,)
    # Cumulative regret must be non-decreasing.
    assert np.all(np.diff(reg) >= -1e-9)
    assert reg[-1] >= 0.0


def test_exercise_3_ucb1_bonus_decreases_in_counts():
    bonus = S.exercise_3_ucb1_bonus(np.array([1, 10, 100]), t=100)
    assert bonus[0] > bonus[1] > bonus[2]
    # Unplayed arm => +inf
    bonus0 = S.exercise_3_ucb1_bonus(np.array([0, 10]), t=10)
    assert np.isinf(bonus0[0])


def test_exercise_4_thompson_posterior():
    a, b = S.exercise_4_thompson_posterior(1.0, 1.0, np.array([1, 0, 1, 1, 0]))
    assert a == 1.0 + 3.0
    assert b == 1.0 + 2.0


def test_exercise_5_bakeoff_ts_beats_epsilon():
    means = np.array([0.1, 0.25, 0.5, 0.75])
    finals = S.exercise_5_bakeoff(means, T=500, n_seeds=5)
    assert finals["Thompson"] < finals["eps-greedy(0.1)"]


def test_exercise_6_discounted_return():
    g = S.exercise_6_discounted_return(np.array([1.0, 1.0, 1.0]), gamma=0.5)
    assert abs(g - (1.0 + 0.5 + 0.25)) < 1e-12
    assert S.exercise_6_discounted_return(np.array([]), gamma=0.9) == 0.0


def test_exercise_7_linucb_beats_eps_on_contextual():
    lin_r, eps_r = S.exercise_7_linucb_regret(K=4, d=4, T=800, alpha=1.0, seed=0)
    assert lin_r < eps_r


def test_exercise_8_snips_basic():
    r = np.array([1.0, 0.0, 1.0, 1.0])
    pi = np.array([0.9, 0.1, 0.5, 0.5])
    pi_b = np.array([0.5, 0.5, 0.5, 0.5])
    v = S.exercise_8_snips(r, pi, pi_b)
    # By hand: w = [1.8, 0.2, 1.0, 1.0]; num = 1.8 + 0 + 1 + 1 = 3.8; den = 4.0
    assert abs(v - 3.8 / 4.0) < 1e-10
    # Edge: all zero weights -> 0.0
    assert S.exercise_8_snips(r, np.zeros(4), np.ones(4)) == 0.0
