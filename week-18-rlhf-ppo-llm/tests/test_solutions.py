"""Tests for the reference solutions used by the workbook."""
import os
import sys
import math

THIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, THIS)

import torch  # noqa: E402

from project import solutions as S  # noqa: E402


def test_bt_loss_sign_and_value():
    r_w = torch.tensor([1.0, 2.0])
    r_l = torch.tensor([0.0, 0.0])
    loss = S.exercise_1_bt_loss(r_w, r_l)
    # -mean(log sigmoid(1), log sigmoid(2))  > 0
    expected = -0.5 * (math.log(1 / (1 + math.exp(-1.0))) + math.log(1 / (1 + math.exp(-2.0))))
    assert abs(float(loss) - expected) < 1e-5


def test_bt_loss_zero_at_winning():
    """Large margins drive the loss toward zero."""
    r_w = torch.tensor([20.0])
    r_l = torch.tensor([-20.0])
    loss = S.exercise_1_bt_loss(r_w, r_l)
    assert float(loss) < 1e-5


def test_token_kl_non_negative():
    torch.manual_seed(0)
    a, b = torch.randn(3, 5), torch.randn(3, 5)
    kl = S.exercise_2_token_kl(a, b)
    assert kl.shape == (3,)
    assert (kl >= -1e-6).all()


def test_token_kl_zero_for_identical():
    a = torch.randn(2, 5)
    kl = S.exercise_2_token_kl(a, a)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_ppo_loss_clip_engaged():
    new = torch.tensor([5.0])  # large positive log-ratio shift
    old = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    loss = S.exercise_3_ppo_loss(new, old, adv, clip_eps=0.2)
    # ratio = e^5 ~ 148; clipped = 1.2; adv > 0 picks min = 1.2.
    assert abs(float(loss) - (-1.2)) < 1e-5


def test_gae_matches_compute_gae():
    from project.ppo import compute_gae
    rewards = torch.randn(2, 6)
    values = torch.randn(2, 7)
    a1, r1 = compute_gae(rewards, values, gamma=0.99, lam=0.95)
    a2, r2 = S.exercise_4_gae(rewards, values, gamma=0.99, lam=0.95)
    assert torch.allclose(a1, a2) and torch.allclose(r1, r2)


def test_kl_controller_monotone():
    up = S.exercise_5_kl_update(0.1, 0.2, 0.1, 10)
    down = S.exercise_5_kl_update(0.1, 0.05, 0.1, 10)
    assert up > 0.1 > down


def test_length_norm():
    r = torch.tensor([1.0, 1.0])
    L = torch.tensor([0, 10])
    out = S.exercise_6_length_normalised(r, L, alpha=0.1)
    assert abs(float(out[0]) - 1.0) < 1e-6
    assert abs(float(out[1]) - 0.0) < 1e-6


def test_optimal_policy_logprob():
    out = S.exercise_7_optimal_policy_logprob(
        torch.tensor([0.0]), torch.tensor([2.0]), beta=0.5
    )
    assert abs(float(out) - 4.0) < 1e-6
