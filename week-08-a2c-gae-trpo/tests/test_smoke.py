"""Smoke tests for the week-08 A2C+GAE mini-project."""
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
    out = train(seed=0, n_updates=4, num_envs=2, rollout_steps=64)
    assert len(out.update_losses) == 4
    for d in out.update_losses:
        assert np.isfinite(d["pi_loss"]) and np.isfinite(d["v_loss"])
