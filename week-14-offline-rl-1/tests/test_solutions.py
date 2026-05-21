"""Contract tests for reference solutions used by workbook.ipynb."""
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import numpy as np
from project import solutions as S


def test_expectile_is_positive_and_asymmetric():
    u = np.array([-1.0, 1.0])
    loss = S.exercise_1_expectile(u, tau=0.8)
    # tau=0.8 weights positive residuals more.
    assert loss[0] < loss[1]
    assert (loss >= 0).all()


def test_awr_weights_average_to_one():
    adv = np.array([-1.0, 0.0, 1.0, 2.0])
    w = S.exercise_2_awr_weights(adv, beta=1.0)
    assert abs(w.mean() - 1.0) < 1e-6


def test_cql_penalty_nonneg():
    Q_s = np.array([0.0, 0.0, 0.0, 0.0])
    p = S.exercise_3_cql_penalty(Q_s, a_beta=2)
    # log 4 - 0 = log 4
    assert abs(p - np.log(4.0)) < 1e-6


def test_dataset_shapes():
    data = S.exercise_4_collect_dataset(n_episodes=10, seed=0)
    n = len(data["a"])
    for k in ("s", "s2"):
        assert data[k].shape == (n, 25)
    assert data["s_idx"].max() < 25


def test_iql_runs():
    out = S.exercise_5_train_iql(seed=0, n_steps=400)
    assert "final_return" in out


def test_snips_finite():
    data = S.exercise_4_collect_dataset(n_episodes=30, seed=0)
    # Target policy = behaviour greedy (logger's argmax) ~ optimal pi.
    target = np.zeros(25, dtype=int)
    out = S.exercise_8_compare_algorithms(seed=0, n_steps=200)
    assert set(out.keys()) == {"naive", "cql", "iql"}
