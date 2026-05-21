"""Tests for week 13 reference solutions."""
from __future__ import annotations

import os
import sys
import numpy as np

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)


def test_puct():
    from project.solutions import exercise_1_puct
    s = exercise_1_puct(prior=0.5, parent_visits=100, child_visits=10, q=0.2)
    assert s > 0


def test_terminal_value():
    from project.solutions import exercise_2_terminal_value
    import numpy as np
    state = np.zeros((3, 3), dtype=np.int8)
    assert exercise_2_terminal_value(state) is None
    state[0, :] = -1  # player who just moved (encoded as -1) won
    assert exercise_2_terminal_value(state) == -1.0


def test_symmetries():
    from project.solutions import exercise_3_n_symmetries
    state = np.zeros((3, 3), dtype=np.int8)
    pi = np.ones(9) / 9.0
    assert exercise_3_n_symmetries(state, pi) == 8


def test_mcts_visits_sum_to_one():
    from project.solutions import exercise_4_mcts_visits
    pi = exercise_4_mcts_visits(n_simulations=20, seed=0)
    assert pi.shape == (9,)
    assert abs(pi.sum() - 1.0) < 1e-6


def test_alphazero_winrate_short():
    from project.solutions import exercise_5_alphazero_winrate
    wr = exercise_5_alphazero_winrate(n_iters=5, seed=0)
    assert wr >= 0.5


def test_gae():
    from project.solutions import exercise_6_gae
    adv, ret = exercise_6_gae([1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5])
    assert adv.shape == (4,) and ret.shape == (4,)
    # Returns should be > 0 (positive rewards).
    assert (ret > 0).all()


def test_coord_game_converges():
    from project.solutions import exercise_7_coord_game_return
    r = exercise_7_coord_game_return(n_episodes=150, seed=0)
    assert r >= 5.0  # at least half the max


def test_simple_spread_returns():
    from project.solutions import exercise_8_simple_spread
    first, last = exercise_8_simple_spread(n_episodes=20, seed=0)
    assert np.isfinite(first) and np.isfinite(last)


def test_self_play_dict():
    from project.solutions import exercise_9_self_play_vs_fixed
    d = exercise_9_self_play_vs_fixed(seed=0)
    assert "self_play_winrate" in d
    assert d["self_play_winrate"] >= 0.0


def test_az_rating_climb():
    from project.solutions import exercise_10_az_rating_climb
    wrs = exercise_10_az_rating_climb(n_iters=4, seed=0)
    assert wrs.shape == (4,)
    assert (wrs >= 0).all() and (wrs <= 1).all()
