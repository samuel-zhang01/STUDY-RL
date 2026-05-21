"""Tests for ``project.solutions`` — the reference implementations the
workbook is graded against."""
from __future__ import annotations
import numpy as np
import pytest

from project import solutions as S
from project.train import grpo_advantages, rloo_advantages


def test_exercise_1_extraction():
    assert S.exercise_1("The answer is 17") == 17
    assert S.exercise_1("step1: 3, step2: 4, so 7") == 7
    assert S.exercise_1("no number here") is None
    assert S.exercise_1("-2 + 3 = 1") == 1


def test_exercise_2_rloo_zero_sum():
    r = np.array([1.0, 0.0, 1.0, 0.0])
    a = S.exercise_2(r)
    # RLOO advantages sum to zero whenever the leave-one-out baseline is
    # symmetric across the group.
    assert abs(a.sum()) < 1e-9


def test_exercise_3_grpo_normalizes():
    r = np.array([[0.0, 1.0, 0.0, 1.0]])
    a = S.exercise_3(r)
    # mean zero, std ~ 1 (within numerical tolerance from eps).
    assert abs(a.mean()) < 1e-6
    assert abs(a.std() - 1.0) < 1e-2


def test_exercise_3_debias_is_centered():
    r = np.array([[0.2, 0.4, 0.6, 0.8]])
    a = S.exercise_3(r, debias=True)
    assert abs(a.mean()) < 1e-12


def test_exercise_4_rloo_lower_variance():
    rng = np.random.default_rng(0)
    r = (rng.uniform(size=(500, 8)) < 0.3).astype(float)
    v_reinforce, v_rloo = S.exercise_4(r)
    # RLOO advantage variance should not exceed raw-reward variance under
    # Bernoulli rewards; in fact for our generator it should be strictly less.
    assert v_rloo <= v_reinforce + 1e-9


def test_exercise_5_kl_nonneg():
    rng = np.random.default_rng(1)
    logp = -rng.uniform(0.1, 1.0, size=200)
    # equal distributions -> KL near zero
    assert abs(S.exercise_5(logp, logp)) < 1e-9
    # Make q assign lower log-prob systematically.
    logq = logp - 0.5
    assert S.exercise_5(logp, logq) > 0


def test_exercise_6_clipping_disables_large_ratios():
    ratio = np.array([2.0, 0.1, 1.0])
    adv = np.array([1.0, 1.0, 1.0])
    loss = S.exercise_6(ratio, adv, clip_eps=0.2)
    # ratio=2 with positive adv triggers clipping at 1.2; loss = -1.2.
    assert abs(loss[0] - (-1.2)) < 1e-9


def test_exercise_7_aggregations():
    s = np.array([[0.9, 0.9, 0.2], [0.8, 0.8, 0.8]])
    assert np.allclose(S.exercise_7(s, "min"),  [0.2, 0.8])
    assert np.allclose(S.exercise_7(s, "mean"), [0.6666667, 0.8])
    assert np.allclose(S.exercise_7(s, "prod"), [0.9 * 0.9 * 0.2, 0.8 ** 3])


def test_exercise_8_more_than_baseline():
    # Format bonus makes shaped reward >= base reward.
    assert S.exercise_8("3", target=3) >= 1.0
    assert S.exercise_8("hello", target=3) == 0.0
    assert 0.0 < S.exercise_8("answer 4", target=3) < 1.0  # wrong but formatted


def test_grpo_advantages_torch_match_numpy():
    import torch
    r = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    a_t = grpo_advantages(r)
    a_n = S.exercise_3(r.numpy())
    assert np.allclose(a_t.numpy(), a_n, atol=1e-3)


def test_rloo_advantages_torch_match_numpy():
    import torch
    r = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    a_t = rloo_advantages(r)
    a_n = np.stack([S.exercise_2(row) for row in r.numpy()])
    assert np.allclose(a_t.numpy(), a_n, atol=1e-9)
