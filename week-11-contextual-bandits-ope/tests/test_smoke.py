"""Smoke test: 1-5s training pass that asserts sane behaviour."""
from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)


def test_project_importable():
    importlib.import_module("project")


def test_short_train_beats_random():
    from project.train import train

    res = train(T=400, n_log=600)
    # LinTS final cumulative regret should be far below the random upper bound
    # (~0.4 * T on this env). Anything below 0.6 * T is sane.
    assert res.regrets["LinTS"][-1] < 0.6 * 400, (
        f"LinTS regret too high: {res.regrets['LinTS'][-1]}"
    )
    # OPE estimates should be in [0, 1.5] (bounded Bernoulli rewards + slack
    # for high-variance IPS).
    for est in res.ope.values():
        for k, v in est.items():
            assert -0.1 <= v <= 1.5, f"{k}={v} out of plausible range"
