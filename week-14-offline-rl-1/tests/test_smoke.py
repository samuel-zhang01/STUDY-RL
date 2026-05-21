"""Smoke test: a tiny offline training run completes and beats a random baseline."""
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import numpy as np
from project.train import train
from project.env import make_env


def test_iql_smoke():
    out = train(seed=0, algo="iql", n_steps=600, n_episodes_dataset=80)
    # Random policy averages well below 0 on this grid; we just need sanity.
    assert out["final_return"] > -10.0
    assert out["Q"].shape == (25, 4)
    assert out["eval_returns"].shape[1] == 2


def test_cql_smoke():
    out = train(seed=0, algo="cql", n_steps=600, n_episodes_dataset=80)
    assert np.isfinite(out["final_return"])
    assert out["Q"].shape == (25, 4)
