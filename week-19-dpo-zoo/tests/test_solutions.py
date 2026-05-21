"""Tests for week-19 reference solutions."""
import os
import sys

THIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, THIS)

import torch  # noqa: E402
torch.set_num_threads(1)

from project import solutions as S  # noqa: E402


def test_implicit_reward():
    lp = torch.tensor([0.0, -2.0])
    lref = torch.tensor([-1.0, -1.0])
    r = S.exercise_1_implicit_reward(lp, lref, beta=0.5)
    assert torch.allclose(r, torch.tensor([0.5, -0.5]))


def test_dpo_loss_value():
    z = torch.zeros(4)
    loss = S.exercise_2_dpo(z, z, z, z, beta=0.1)
    assert abs(float(loss) - 0.693) < 1e-2


def test_ipo_loss_value():
    z = torch.zeros(4)
    loss = S.exercise_3_ipo(z, z, z, z, beta=0.5)
    assert abs(float(loss) - 1.0) < 1e-5


def test_simpo_value():
    lp_w = torch.tensor([-1.0]); lp_l = torch.tensor([-5.0])
    L_w = torch.tensor([1]); L_l = torch.tensor([5])
    # Both have avg -1; margin 0; with gamma 0 loss = log 2.
    loss = S.exercise_4_simpo(lp_w, lp_l, L_w, L_l, beta=2.0, gamma=0.0)
    assert abs(float(loss) - 0.693) < 1e-2


def test_log1mexp_finite():
    x = torch.tensor([-0.1, -0.5, -2.0, -10.0])
    out = S.exercise_5_log1mexp(x)
    assert torch.isfinite(out).all()


def test_orpo_value():
    lp_w = torch.tensor([-1.0]); lp_l = torch.tensor([-3.0])
    nll_w = torch.tensor([1.0])
    out = S.exercise_6_orpo(lp_w, lp_l, nll_w, beta=0.1)
    assert torch.isfinite(out)


def test_win_rate():
    a = torch.tensor([1.0, 2.0, 0.0])
    b = torch.tensor([0.0, 1.0, 1.0])
    assert abs(S.exercise_7_win_rate(a, b) - 2.0 / 3.0) < 1e-6


def test_margin_signs_correctly():
    # Winner higher under theta than under ref; loser lower.
    lp_w = torch.tensor([-1.0]); lp_l = torch.tensor([-3.0])
    lref_w = torch.tensor([-2.0]); lref_l = torch.tensor([-2.0])
    margin = S.exercise_8_margin(lp_w, lp_l, lref_w, lref_l, beta=0.1)
    assert float(margin) > 0


def test_length_bias_diagnostic():
    chosen = torch.tensor([10, 20, 30])
    rejected = torch.tensor([5, 5, 5])
    gap = S.exercise_9_length_bias(chosen, rejected)
    assert gap == 15.0
