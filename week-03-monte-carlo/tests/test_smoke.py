"""Smoke tests for the week-03 MC project."""
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

from project.train import train  # noqa: E402


def test_blackjack_mc_control_beats_random_baseline():
    out = train(seed=0, n_episodes=8000)
    # All-hit baseline on Blackjack starting from sum >= 12 is around -0.40;
    # eps-soft MC control with eps=0.1 ceilings near -0.07. After 8k episodes
    # we expect the running average to be well above the all-hit floor.
    assert out["blackjack_avg_return_last_5k"] > -0.20
    assert out["n_state_action_visited"] > 50


def test_recruitment_off_policy_estimators_are_finite():
    out = train(seed=0, n_episodes=8000)
    for k in ("recruitment_V_ois", "recruitment_V_wis", "recruitment_V_oracle"):
        assert np.isfinite(out[k]), out
