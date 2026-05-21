"""Contract tests for the workbook solutions."""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)
sys.path.insert(0, os.path.dirname(WEEK))

from project import solutions  # noqa: E402


def test_dual_step():
    # Violation positive => lambda should rise.
    assert solutions.exercise_1(lam=0.5, eta=0.1, J_c=2.0, d=1.0) > 0.5
    # Violation negative => lambda decays, clipped at 0.
    assert solutions.exercise_1(lam=0.05, eta=1.0, J_c=0.0, d=1.0) == 0.0


def test_p3o_subgrad():
    assert solutions.exercise_2(J_c=2.0, d=1.0, kappa=10.0) == 10.0
    assert solutions.exercise_2(J_c=0.5, d=1.0, kappa=10.0) == 0.0


def test_gae_shape():
    rews = [0.1] * 10
    vals = [0.0] * 10
    dones = [False] * 9 + [True]
    adv, ret = solutions.exercise_3(rews, vals, dones, gamma=0.9, lam=0.95)
    assert adv.shape == (10,) and ret.shape == (10,)
    # last step is terminal => advantage equals the reward.
    np.testing.assert_allclose(adv[-1], rews[-1], atol=1e-6)


def test_focops_rownormalised():
    S, A = 4, 3
    pi = np.full((S, A), 1.0 / A)
    A_r = np.random.RandomState(0).randn(S, A)
    A_c = np.random.RandomState(1).randn(S, A)
    new = solutions.exercise_4(pi, A_r, A_c, lam=1.0, nu=0.5)
    np.testing.assert_allclose(new.sum(axis=1), 1.0, atol=1e-6)
    assert (new >= 0).all()


def test_cvar_known_case():
    samples = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # alpha=0.4 => take worst 2 out of 5 => mean of {0, 1} = 0.5
    assert abs(solutions.exercise_5(samples, alpha=0.4) - 0.5) < 1e-9


def test_pid_lagrangian_moves():
    lam, integ = solutions.exercise_8(
        lam_prev=0.0, integral_prev=0.0, J_c_hist=[2.0, 2.5], d=1.0,
    )
    assert lam > 0.0
    assert integ > 0.0
