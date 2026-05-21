"""Smoke tests for week-04 mini-project."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import pytest

import _pkg_loader  # noqa: F401  -- registers w04_project
from w04_project import agents, env_utils, train, solutions  # noqa: E402


def test_eps_schedule_monotone():
    s = agents.make_eps_schedule(1.0, 0.1, 100)
    assert s(0) == 1.0
    assert s(50) == pytest.approx(0.55, rel=1e-6)
    assert s(100) == 0.1
    assert s(500) == 0.1


def test_eps_greedy_picks_argmax_when_eps_zero():
    Q = np.array([[1.0, 5.0, 2.0], [0.0, 0.0, 9.0]])
    rng = np.random.default_rng(0)
    assert agents.eps_greedy(Q, 0, 0.0, rng) == 1
    assert agents.eps_greedy(Q, 1, 0.0, rng) == 2


def test_sarsa_runs_and_returns_shape():
    env = env_utils.make_cliffwalking()
    Q, rs = agents.sarsa(env, n_episodes=40, alpha=0.5, eps_decay=20,
                         max_steps=200, seed=0)
    assert Q.shape[1] == env.action_space.n
    assert len(rs) == 40


def test_q_learning_beats_random_on_cliff():
    env = env_utils.make_cliffwalking()
    Q, rs = agents.q_learning(env, n_episodes=200, alpha=0.5, eps_decay=80,
                              max_steps=200, seed=0)
    assert np.mean(rs[-50:]) > -200, f"got {np.mean(rs[-50:])}"


def test_expected_sarsa_runs():
    env = env_utils.make_cliffwalking()
    _, rs = agents.expected_sarsa(env, n_episodes=40, eps_decay=20, seed=0)
    assert len(rs) == 40


def test_double_q_learning_runs():
    env = env_utils.make_cliffwalking()
    _, rs = agents.double_q_learning(env, n_episodes=40, eps_decay=20, seed=0)
    assert len(rs) == 40


def test_train_driver_smoke():
    res = train.train(n_seeds=2, n_episodes=80, taxi_episodes=200,
                      eps_decay=30, do_taxi=False)
    assert set(res.returns.keys()) >= {"SARSA", "Q-learning",
                                       "Expected SARSA", "Double Q"}
    for k in res.returns:
        assert res.returns[k].shape == (2, 80)
