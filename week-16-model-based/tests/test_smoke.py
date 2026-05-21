"""Smoke test: 3-iter Dreamer-lite training completes."""
import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import numpy as np
from project.train import train


def test_dreamer_smoke():
    out = train(seed=0, n_iters=3, episodes_per_iter=1, chunk_len=10,
                batch_size=8, imagine_horizon=6, max_steps=80)
    curve = out["eval_curve"]
    assert curve.shape == (3, 2)
    assert np.isfinite(curve).all()
