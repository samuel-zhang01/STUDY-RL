"""Contract tests for the week-03 reference solutions."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK = os.path.dirname(HERE)
REPO = os.path.dirname(WEEK)
sys.path.insert(0, WEEK)
sys.path.insert(0, REPO)

from project import solutions as S  # noqa: E402
from project.blackjack import BlackjackEnv  # noqa: E402


def test_first_visit_returns_basic():
    # Trajectory: (s0, a0, 0), (s1, a1, 0), (s2, a2, 1). Gamma=1 -> all returns are 1.
    traj = [(0, 0, 0.0), (1, 1, 0.0), (2, 0, 1.0)]
    out = S.exercise_1_first_visit_returns(traj, gamma=1.0)
    assert out[(0, 0)] == [1.0]
    assert out[(1, 1)] == [1.0]
    assert out[(2, 0)] == [1.0]


def test_first_visit_only_counts_first():
    # (s,a) appears twice; only first visit is counted.
    traj = [(0, 0, 1.0), (0, 0, 1.0), (1, 0, 0.0)]
    out = S.exercise_1_first_visit_returns(traj, gamma=1.0)
    assert len(out[(0, 0)]) == 1


def test_every_visit_mc_converges():
    env = BlackjackEnv(seed=0)
    policy = lambda obs: 0  # always stick
    V = S.exercise_2_every_visit_mc(env, policy, n_episodes=3000, gamma=1.0, seed=0)
    # All-stick on Blackjack loses on average; values should be < 0 in most states.
    vs = np.array(list(V.values()))
    assert vs.mean() < 0.0


def test_blackjack_mc_control_runs_and_beats_random():
    Q, pi, returns = S.exercise_3_blackjack_mc_control(n_episodes=5000, eps=0.1, seed=0)
    assert len(Q) > 0
    # Random policy averages around -0.39 on Blackjack;
    # eps-soft MC control quickly beats that.
    assert np.mean(returns[-2000:]) > -0.20


def test_ordinary_is_basic():
    v = S.exercise_4_ordinary_is([1.0, 0.0, 1.0], [0.9, 0.1, 0.5], [0.5, 0.5, 0.5])
    # w = [1.8, 0.2, 1.0]; mean(w*r) = (1.8 + 0 + 1.0)/3 = 2.8/3
    assert abs(v - 2.8 / 3) < 1e-10


def test_wis_basic():
    v = S.exercise_5_weighted_is([1.0, 0.0, 1.0], [0.9, 0.1, 0.5], [0.5, 0.5, 0.5])
    # num = 2.8; den = 3.0
    assert abs(v - 2.8 / 3.0) < 1e-10
    # If all weights are equal the WIS estimate is the sample mean.
    v2 = S.exercise_5_weighted_is([1.0, 0.0], [0.5, 0.5], [0.5, 0.5])
    assert abs(v2 - 0.5) < 1e-10


def test_wis_lower_variance_than_ois():
    var_ois, var_wis = S.exercise_6_compare_is_wis_variance(n_trials=120, n_logs=120, seed=0)
    assert var_wis < var_ois


def test_offpolicy_recruitment_estimators_finite():
    v_ois, v_wis, v_oracle = S.exercise_7_offpolicy_recruitment(n_episodes=2000, seed=0)
    for v in (v_ois, v_wis, v_oracle):
        assert np.isfinite(v)


def test_is_variance_bound_holds():
    rng = np.random.default_rng(0)
    p_t = rng.uniform(0.05, 0.95, size=200)
    p_b = rng.uniform(0.05, 0.95, size=200)
    r = rng.standard_normal(200)
    v, ub = S.exercise_8_is_variance_bound(p_t, p_b, r)
    assert v <= ub + 1e-9
