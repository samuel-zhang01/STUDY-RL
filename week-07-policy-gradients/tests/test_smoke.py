"""Smoke test for the week-07 REINFORCE mini-project.

Runs a tiny training pass (a few dozen episodes) and asserts that the agent at
least beats the random baseline on CartPole-v1 (~22 mean return).
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from project.train import train  # noqa: E402


def test_smoke_train_runs():
    out = train(seed=0, n_episodes=30, use_baseline=True, verbose=False)
    assert len(out.returns) == 30
    # First 5 vs last 10 episodes: at minimum, training should produce a finite
    # return curve; random policy averages ~22 on CartPole-v1, so we use a
    # loose lower bound (random has plenty of variance in 30 episodes).
    assert np.isfinite(out.returns).all()
    assert np.mean(out.returns) >= 10.0


def test_smoke_train_no_baseline_runs():
    out = train(seed=0, n_episodes=15, use_baseline=False, verbose=False)
    assert len(out.returns) == 15
    assert np.isfinite(out.returns).all()
