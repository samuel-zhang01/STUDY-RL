import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import numpy as np
from project import solutions as S


def test_rtg():
    r = np.array([1.0, 2.0, 3.0])
    rtg = S.exercise_1_returns_to_go(r)
    assert np.allclose(rtg, [6.0, 5.0, 3.0])


def test_causal_mask():
    M = S.exercise_2_causal_mask(4)
    assert M[0, 1] and not M[1, 0] and not M[0, 0]


def test_sample_minibatch_shapes():
    S_, A, R, T, M_ = S.exercise_3_sample_minibatch(seed=0, n_eps=20, K=6, B=3)
    assert S_.shape == (3, 6, 25)
    assert A.shape == (3, 6)


def test_diffusion_noising_variance():
    rng = np.random.default_rng(0)
    x0 = np.ones(100, dtype=np.float32)
    xt = S.exercise_7_diffusion_noising(x0, alpha_bar=0.0, rng=rng)
    # alpha_bar=0 means pure noise; variance should be near 1.
    assert abs(np.std(xt) - 1.0) < 0.3
