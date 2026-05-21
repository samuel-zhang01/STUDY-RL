"""Contract tests for the week-02 reference solutions."""
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

from common.gridworld import GridWorld  # noqa: E402
from project import solutions as S  # noqa: E402


def test_exercise_1_bellman_operator_increases_value_toward_goal():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goals=((2, 2),))
    V = np.zeros(env.nS)
    V1 = S.exercise_1_bellman_operator(env, V, gamma=0.9)
    # Adjacent-to-goal states (1,2) and (2,1) get the goal reward 10 in one step.
    assert V1[env.s2idx((1, 2))] == 10.0
    assert V1[env.s2idx((2, 1))] == 10.0


def test_exercise_2_value_iteration_finds_finite_V_and_policy_at_start():
    V, pi, deltas = S.exercise_2_value_iteration(rows=5, cols=5, gamma=0.95)
    assert V.shape == (25,) and pi.shape == (25,)
    assert deltas[-1] < 1e-6
    # Optimal V at start should be > 0 (positive expected discounted reward to goal).
    assert V[0] > 0


def test_exercise_3_policy_iteration_matches_vi():
    V_vi, _, _ = S.exercise_2_value_iteration(gamma=0.95)
    V_pi, pi, n = S.exercise_3_policy_iteration(gamma=0.95)
    assert n >= 1
    np.testing.assert_allclose(V_vi, V_pi, atol=1e-5)


def test_exercise_4_slippery_world_value_drops_with_slip():
    v0 = S.exercise_4_slip_world_optimal_value(slip=0.0, gamma=0.95)
    v1 = S.exercise_4_slip_world_optimal_value(slip=0.4, gamma=0.95)
    assert v1 < v0


def test_exercise_5_contraction():
    rng = np.random.default_rng(0)
    env = GridWorld(rows=4, cols=4, start=(0, 0), goals=((3, 3),))
    V1 = rng.standard_normal(env.nS)
    V2 = rng.standard_normal(env.nS)
    tdiff, diff = S.exercise_5_contraction_factor(env, V1, V2, gamma=0.9)
    # Banach-style contraction: ||TV1 - TV2||_inf <= gamma ||V1 - V2||_inf.
    assert tdiff <= 0.9 * diff + 1e-9


def test_exercise_6_policy_evaluation_matches_iterative():
    env = GridWorld(rows=4, cols=4, start=(0, 0), goals=((3, 3),))
    pi = np.array([1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 0])
    V_lin = S.exercise_6_policy_evaluation_closed_form(env, pi, gamma=0.9)
    from common.tabular import policy_evaluation
    V_iter = policy_evaluation(env, pi, gamma=0.9, tol=1e-12)
    np.testing.assert_allclose(V_lin, V_iter, atol=1e-6)


def test_exercise_7_gpi_eventually_stable():
    env = GridWorld(rows=4, cols=4, start=(0, 0), goals=((3, 3),))
    V, pi, n = S.exercise_7_gpi(env, gamma=0.95, eval_sweeps=3)
    assert n >= 1
    # The resulting policy must reach the goal from the start with finite value.
    assert V[env.s2idx((0, 0))] > 0


def test_exercise_8_car_rental_runs():
    V, pi, n = S.exercise_8_car_rental(cap=4, gamma=0.9)
    assert V.shape == (25,)
    assert pi.shape == (25,)
    assert n >= 1
    assert np.isfinite(V).all()
