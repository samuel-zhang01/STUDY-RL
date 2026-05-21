import numpy as np
from common.gridworld import GridWorld


def test_basic_step():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goals=((2, 2),))
    s = env.reset(seed=42)
    assert s == (0, 0)
    s, r, done, _ = env.step(1)  # right
    assert s == (0, 1)
    assert r == -1.0
    assert not done


def test_wall_bump():
    env = GridWorld(rows=2, cols=2)
    env.reset()
    s, r, done, _ = env.step(0)  # up from (0,0) — wall
    assert s == (0, 0)
    assert not done


def test_goal_terminates_with_reward():
    env = GridWorld(rows=2, cols=2, start=(0, 1), goals=((1, 1),))
    env.reset()
    s, r, done, _ = env.step(2)
    assert s == (1, 1) and done and r == 10.0


def test_transition_table_probs_sum_to_one():
    env = GridWorld(rows=4, cols=4, slip_prob=0.2)
    P = env.transition_table()
    for (s, a), outs in P.items():
        p = sum(o[0] for o in outs)
        assert abs(p - 1.0) < 1e-9, (s, a, p)


def test_idx_roundtrip():
    env = GridWorld(rows=5, cols=7)
    for i in range(env.nS):
        assert env.s2idx(env.idx2s(i)) == i
