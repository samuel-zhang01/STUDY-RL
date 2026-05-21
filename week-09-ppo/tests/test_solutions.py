"""Unit tests for week-09 reference solutions."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from project.solutions import (  # noqa: E402
    ppo_clip_loss,
    ppo_value_clip_loss,
    approx_kl,
    orthogonal_init_,
    adaptive_kl_penalty_update,
)


def test_ppo_clip_loss_no_clip_when_ratio_equals_one():
    logp = torch.zeros(4)
    logp_old = torch.zeros(4)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    L = ppo_clip_loss(logp, logp_old, adv, clip_eps=0.2)
    # ratio = 1, surr1 = surr2 = adv; loss = -mean(adv) = 0
    assert abs(L.item()) < 1e-7


def test_ppo_clip_clips_positive_advantage_above_one_plus_eps():
    logp = torch.log(torch.tensor([1.5]))
    logp_old = torch.zeros(1)  # ratio = 1.5
    adv = torch.tensor([2.0])
    L = ppo_clip_loss(logp, logp_old, adv, clip_eps=0.2)
    # min(1.5*2, 1.2*2) = 2.4; loss = -2.4
    assert abs(L.item() + 2.4) < 1e-5


def test_ppo_value_clip_loss_matches_mse_when_unclipped():
    v = torch.tensor([1.0, 2.0])
    v_old = torch.tensor([1.0, 2.0])
    ret = torch.tensor([1.0, 2.5])
    L = ppo_value_clip_loss(v, v_old, ret, clip_eps=0.2)
    expected = 0.5 * ((v - ret) ** 2).mean().item()
    assert abs(L.item() - expected) < 1e-7


def test_approx_kl_is_nonneg_and_zero_at_match():
    logp = torch.randn(100)
    assert approx_kl(logp, logp.clone()) == 0.0
    assert approx_kl(logp + 0.1, logp) > 0.0


def test_orthogonal_init_sets_final_layer_gain():
    net = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4))
    orthogonal_init_(net, gain=2 ** 0.5, final_layer_gain=0.01)
    # Final layer weight should have small norm.
    w_final = net[-1].weight
    w_first = net[0].weight
    assert w_final.norm().item() < w_first.norm().item()


def test_adaptive_kl_penalty_doubles_when_overshooting():
    new_b = adaptive_kl_penalty_update(beta=1.0, measured_kl=0.05, target_kl=0.01)
    assert new_b == 2.0
    new_b = adaptive_kl_penalty_update(beta=1.0, measured_kl=0.002, target_kl=0.01)
    assert new_b == 0.5
