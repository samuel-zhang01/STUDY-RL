"""Smoke tests for the week-09 PPO mini-project."""
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
    out = train(seed=0, n_updates=3, num_envs=2, rollout_steps=64,
                n_epochs=2, minibatch_size=32)
    assert len(out.update_metrics) == 3
    for m in out.update_metrics:
        assert np.isfinite(m["pi_loss"]) and np.isfinite(m["v_loss"])


def test_details_audit():
    out = train(seed=0, n_updates=2, num_envs=2, rollout_steps=64,
                n_epochs=1, minibatch_size=32)
    # By default all the core continuous-control details should be on.
    for d in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 28, 29, 31, 32):
        assert out.details_used[d] is True, f"detail #{d} missing"
