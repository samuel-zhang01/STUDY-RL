"""Contract tests for the week-23 solutions."""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)
sys.path.insert(0, os.path.dirname(WEEK))

from project import solutions as ref  # noqa: E402
from project.env_grid import DangerGridWorld  # noqa: E402


def test_dfa_shape():
    dfa = ref.exercise_1()
    assert dfa["initial"] == "q0"
    assert "q_bad" not in dfa["accepting"]
    # Sink absorbing.
    assert dfa["transitions"][("q_bad", "danger")] == "q_bad"
    assert dfa["transitions"][("q0", "danger")] == "q_bad"


def test_unsafe_predicate():
    env = DangerGridWorld()
    env.reset()
    # State (0, 0) moving 'down' goes to (1, 0): NOT in danger.
    assert not ref.exercise_2(env, env.s2idx((0, 0)), 2)
    # State (1, 1) moving 'right' goes to (1, 2): IS in danger.
    assert ref.exercise_2(env, env.s2idx((1, 1)), 1)


def test_shield_filter_picks_safe():
    env = DangerGridWorld()
    # In state (1,1), the preferred order is [right (1, into danger), down (2, into danger (2,1))], so shield must pick something else.
    # Action 0=up takes to (0,1) safe; action 3=left to (1,0) safe.
    out = ref.exercise_3(env, [1, 2, 0, 3], env.s2idx((1, 1)))
    assert out in (0, 3)


def test_train_shielded_no_unsafe():
    n = ref.exercise_5(seed=0, n_episodes=100)
    assert n == 0


def test_cbf_projection_feasible_passthrough():
    a = ref.exercise_7(a_proposed=0.5, h_val=2.0, dh_da=1.0, alpha=1.0)
    # constraint: 1 * a >= -1*2 = -2; 0.5 >= -2 trivially. Return as-is.
    assert abs(a - 0.5) < 1e-9


def test_cbf_projection_active():
    # h = 1, dh_da = -1, alpha = 1: constraint -a >= -1, i.e. a <= 1.
    # a_proposed = 2 violates: project to 1.
    a = ref.exercise_7(a_proposed=2.0, h_val=1.0, dh_da=-1.0, alpha=1.0)
    assert abs(a - 1.0) < 1e-9
