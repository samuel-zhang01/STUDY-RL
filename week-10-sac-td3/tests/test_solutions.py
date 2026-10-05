"""Unit tests for week-10 reference solutions."""
import math
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
    tanh_gaussian_log_prob,
    twin_critic_min,
    soft_bellman_target,
    auto_temperature_loss,
    polyak_update,
)


def test_tanh_gaussian_log_prob_change_of_variables():
    """log pi(a|s) = log N(u; mu, sigma^2) - sum log(1 - tanh^2 u). Check by hand."""
    torch.manual_seed(0)
    u = torch.tensor([[0.3, -0.5]])
    mu = torch.tensor([0.1, -0.2])
    log_std = torch.tensor([0.0, 0.0])
    logp = tanh_gaussian_log_prob(u, mu, log_std).item()
    # log N(u; mu, 1)
    log_p_u = (-0.5 * ((u - mu) ** 2) - 0.5 * math.log(2 * math.pi)).sum().item()
    a = torch.tanh(u)
    correction = torch.log(1.0 - a ** 2).sum().item()
    expected = log_p_u - correction
    assert abs(logp - expected) < 1e-5, (logp, expected)


def test_tanh_gaussian_log_prob_normalises_on_action_grid():
    """Integrate exp(log pi(a)) over a discretisation of (-1, 1) using
    change-of-variables a = tanh(u) ==> da = (1 - a^2) du. The integral
    of pi(a) da on a is the integral of pi(a)*(1-a^2) du on u."""
    mu = torch.zeros(1); log_std = torch.zeros(1)
    u = torch.linspace(-6.0, 6.0, 4001).unsqueeze(-1)
    a = torch.tanh(u)
    logp = tanh_gaussian_log_prob(u, mu, log_std)
    # integral over a from -1 to 1 = integral over u of pi(a(u)) * (da/du) du
    integrand = logp.exp() * (1.0 - a.squeeze(-1) ** 2)
    du = float(u[1] - u[0])
    total = float((integrand * du).sum())
    assert 0.97 < total < 1.03, total


def test_twin_critic_min():
    q1 = torch.tensor([1.0, 2.0, 3.0])
    q2 = torch.tensor([2.0, 1.0, 3.0])
    assert torch.equal(twin_critic_min(q1, q2), torch.tensor([1.0, 1.0, 3.0]))


def test_soft_bellman_target_no_terminal():
    r = torch.tensor([1.0]); d = torch.tensor([0.0])
    next_q = torch.tensor([2.0]); next_logp = torch.tensor([0.5])
    alpha = torch.tensor([0.1])
    y = soft_bellman_target(r, d, next_q, next_logp, alpha, gamma=0.9)
    expected = 1.0 + 0.9 * (2.0 - 0.1 * 0.5)
    assert abs(y.item() - expected) < 1e-6


def test_soft_bellman_target_with_done_drops_future():
    r = torch.tensor([1.0]); d = torch.tensor([1.0])
    y = soft_bellman_target(r, d, torch.tensor([100.0]), torch.tensor([0.0]),
                            torch.tensor([0.1]), gamma=0.9)
    assert abs(y.item() - 1.0) < 1e-6


def test_auto_temperature_loss_sign():
    log_alpha = torch.tensor([0.0], requires_grad=True)
    logp = torch.tensor([-2.0, -2.0, -2.0])  # entropy = 2
    L = auto_temperature_loss(log_alpha, logp, target_entropy=-1.0)
    # E[ logp + H_target ] = -2 + (-1) = -3; loss = -alpha * (-3) = 3 with alpha=1.
    assert abs(L.item() - 3.0) < 1e-6
    L.backward()
    # gradient wrt log_alpha of L = -alpha * E[logp + H_target]:
    # d/d log_alpha = -alpha * (logp + H_target) = -1 * (-3) = 3 ... wait sign:
    # actually L = -alpha*(-3) = 3*alpha; dL/d log_alpha = 3*alpha; at log_alpha=0 -> 3
    assert abs(log_alpha.grad.item() - 3.0) < 1e-6


def test_polyak_update():
    src = nn.Linear(2, 2); tgt = nn.Linear(2, 2)
    with torch.no_grad():
        src.weight.fill_(1.0); src.bias.zero_()
        tgt.weight.fill_(0.0); tgt.bias.zero_()
    polyak_update(tgt, src, tau=0.5)
    assert torch.allclose(tgt.weight, torch.full((2, 2), 0.5))


def test_td3_smoothing_uses_actions_and_both_critics_with_terminal_mask():
    from project.solutions import td3_target
    obs = torch.zeros(2, 3)
    actor = lambda s: torch.full((len(s), 2), 4.)
    q1 = lambda s, a: a.sum(-1)
    q2 = lambda s, a: a.sum(-1) - 1.
    out = td3_target(torch.ones(2), torch.tensor([0.,1.]), obs, actor, q1, q2,
        target_noise=0., gamma=.9)
    assert torch.allclose(out, torch.tensor([1.9, 1.]))
