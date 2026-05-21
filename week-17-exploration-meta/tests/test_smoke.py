import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import numpy as np
from project.rnd import train_rnd_on_gridworld
from project.maml import train_maml


def test_rnd_smoke():
    out = train_rnd_on_gridworld(seed=0, n_episodes=20, use_rnd=True)
    assert out["returns"].shape == (20,)


def test_maml_smoke():
    out = train_maml(seed=0, n_outer=10)
    assert out["rewards_curve"].shape == (10,)
    assert np.isfinite(out["rewards_curve"]).all()
