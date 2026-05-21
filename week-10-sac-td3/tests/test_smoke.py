"""Smoke tests for the week-10 SAC mini-project."""
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
    out = train(seed=0, n_steps=600, start_random=100, batch_size=64,
                buffer_size=2000)
    # We expect a handful of gradient updates and at least a couple of episodes.
    assert len(out.update_metrics) > 0
    assert all(np.isfinite(m["critic_loss"]) for m in out.update_metrics)
    assert all(np.isfinite(m["actor_loss"]) for m in out.update_metrics)
