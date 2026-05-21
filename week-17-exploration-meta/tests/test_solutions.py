import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import numpy as np
from project import solutions as S


def test_random_target_outputs():
    out = S.exercise_1_random_target_outputs()
    assert out.shape == (3, 4)


def test_intrinsic_decreases():
    initial, final = S.exercise_2_intrinsic_decreases(n_steps=100)
    assert final < initial * 0.5


def test_task_balance():
    frac = S.exercise_5_meta_bandit_sampling(n_samples=500, seed=0)
    assert 0.3 < frac < 0.7


def test_inner_adaptation():
    pre, post = S.exercise_6_inner_adaptation_improves(seed=0)
    assert post > pre - 1e-3  # adaptation should not hurt; in expectation should help


def test_maml_curve():
    out = S.exercise_7_train_maml(seed=0, n_outer=30)
    assert out["rewards_curve"].shape == (30,)
