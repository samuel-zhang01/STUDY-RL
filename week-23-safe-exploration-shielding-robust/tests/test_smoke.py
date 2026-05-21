"""Smoke tests for week 23: shielding soundness and DR generalisation gap."""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)
sys.path.insert(0, os.path.dirname(WEEK))

from project.q_learn import train_shielded, train_unshielded  # noqa: E402
from project.pendulum_train import train_pendulum_dr, PendulumTrainConfig  # noqa: E402


def test_shield_zero_violations():
    h = train_shielded(n_episodes=200, seed=0)
    assert h["total_unsafe"] == 0, f"shield was violated: {h['total_unsafe']} unsafe transitions"


def test_unshielded_has_violations():
    h = train_unshielded(n_episodes=200, seed=0)
    # The unshielded agent should encounter at least some unsafe transitions
    # during exploration (otherwise the shield is doing no work).
    assert h["total_unsafe"] > 0


def test_shielded_solves_task():
    h = train_shielded(n_episodes=300, seed=0)
    # Last 50 episodes should hit the goal a clear majority of the time.
    assert sum(h["goal"][-50:]) >= 25


def test_pendulum_dr_runs():
    cfg = PendulumTrainConfig(n_epochs=8, steps_per_epoch=200)
    h = train_pendulum_dr(seed=0, randomise=True, cfg=cfg)
    assert "eval" in h
    # Eval returns finite.
    for k, v in h["eval"].items():
        assert np.isfinite(v)
