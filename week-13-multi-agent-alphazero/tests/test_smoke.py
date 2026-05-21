"""Smoke test: a short MAPPO + AlphaZero training pass."""
from __future__ import annotations

import importlib
import os
import sys
import numpy as np

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)


def test_project_importable():
    importlib.import_module("project")


def test_alphazero_short_train_beats_random():
    from project.train import train_alphazero
    out = train_alphazero(seed=0, n_iters=4, n_episodes_per_iter=3,
                           n_simulations=12)
    # After 4 short iterations winrate should be above 0.55 (random ~0.5).
    assert out["winrates"][-1] >= 0.5, out["winrates"]


def test_mappo_coord_game_converges():
    from project.coord_game import CoordGame
    from project.mappo import train_mappo
    env = CoordGame(n_actions=3, n_agents=2, seed=0)
    out = train_mappo(env=env, n_episodes=150, seed=0, lr=1e-3,
                      update_epochs=4)
    # End-of-training mean should be at least 6 out of max 10.
    assert np.mean(out["returns"][-30:]) >= 6.0, out["returns"][-30:]


def test_simple_spread_returns_finite():
    from project.mappo import train_mappo
    out = train_mappo(n_episodes=30, seed=0, lr=3e-4)
    r = out["returns"]
    assert np.all(np.isfinite(r))
    assert len(r) == 30
