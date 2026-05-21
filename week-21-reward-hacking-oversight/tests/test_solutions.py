"""Tests for ``project.solutions`` reference implementations."""
from __future__ import annotations
import numpy as np
import pytest

from project import solutions as S
from project.cai import compliance_score, critique, revise, dpo_loss
from project.hack_lab import proxy_reward, KEYWORDS


def test_exercise_1_goodhart_classification():
    assert S.exercise_1("noise_overfitting") == "regressional"
    assert S.exercise_1("out_of_distribution") == "extremal"
    assert S.exercise_1("spurious_correlation") == "causal"
    assert S.exercise_1("active_gaming") == "adversarial"
    with pytest.raises(ValueError):
        S.exercise_1("nonsense")


def test_exercise_2_proxy_truth_disagree():
    texts = [
        "the cat sat on the mat",                    # high truth
        "good good good good good good good good",   # gameable proxy hit
        "asdfqwer",                                  # low both
        "the dog ran across the field",              # high truth
    ]
    px, tx, corr = S.exercise_2(texts)
    # The keyword-stuffed line should score highly on proxy but poorly on
    # truth, breaking simple monotonicity.
    assert px[1] > px[0]   # gameable text beats real English on proxy
    assert tx[0] > tx[1]   # real English beats gameable on truth
    assert -1.0 <= corr <= 1.0


def test_exercise_3_kl_schedule_nonneg():
    assert S.exercise_3(0) == pytest.approx(0.04)
    # 100 steps * decay 0.001 -> 0.04 - 0.1 negative -> clamped at 0.
    assert S.exercise_3(20) == pytest.approx(0.04 - 0.02)
    # large step -> clamped at zero.
    assert S.exercise_3(1000) == 0.0


def test_exercise_4_length_penalty():
    assert S.exercise_4(1.0, 10, lam=0.01) == pytest.approx(0.9)
    assert S.exercise_4(1.0, 100, lam=0.02) == pytest.approx(-1.0)


def test_exercise_5_constitution_critique_and_revise():
    crit, rev, c_pre, c_post = S.exercise_5("HELLO Xworld")
    assert not crit["lowercase"]
    assert not crit["no_x"]
    assert not crit["ends_with_period"]
    assert rev == "hello world."
    assert c_post >= c_pre


def test_exercise_6_dpo_numpy_matches_torch():
    import torch
    rng = np.random.default_rng(0)
    a, b, c, d = (rng.normal(size=8) for _ in range(4))
    loss_np = S.exercise_6(a, b, c, d, beta=0.1)
    loss_t = dpo_loss(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        torch.tensor(c, dtype=torch.float32),
        torch.tensor(d, dtype=torch.float32),
        beta=0.1,
    )
    assert abs(loss_np - float(loss_t)) < 1e-4


def test_exercise_7_debate_monotone():
    # Adding rounds should not *decrease* the probability of truth winning.
    p1 = S.exercise_7(0.6, 1)
    p3 = S.exercise_7(0.6, 3)
    p5 = S.exercise_7(0.6, 5)
    assert p1 <= p3 <= p5
    assert S.exercise_7(1.0, 1) == 1.0  # oracle judge


def test_exercise_8_divergence_index():
    # Hand-crafted history where the proxy climbs and truth drops.
    hist = {
        "proxy":       [0.0] * 20 + [1.0] * 20,
        "true_quality": [-2.0] * 20 + [-3.0] * 20,
    }
    di = S.exercise_8(hist, window=20)
    # initial gap = 0 - (-2) = 2; final = 1 - (-3) = 4; di = 4/2 = 2.
    assert abs(di - 2.0) < 1e-6
