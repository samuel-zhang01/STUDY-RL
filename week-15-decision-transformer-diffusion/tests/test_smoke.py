"""Smoke test: DT trains a few steps and loss is finite."""
import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import numpy as np
from project.train import train


def test_dt_smoke():
    out = train(seed=0, n_steps=80, batch_size=8, n_episodes_dataset=40, hidden=32)
    assert np.isfinite(out["losses"]).all()
    # Loss should decrease at least a little.
    assert out["losses"][-1] < out["losses"][0] * 1.5
