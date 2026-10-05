"""Unit tests for the week-07 reference solutions."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from project.solutions import (  # noqa: E402
    discount_cumsum,
    log_prob_normal,
    reinforce_loss,
    reinforce_with_baseline_loss,
    optimal_baseline,
    topk_offpolicy_multiplier,
    snips_estimator,
)


def test_log_prob_normal_matches_torch():
    x, mu, sigma = 0.3, -0.2, 1.4
    ours = log_prob_normal(x, mu, sigma)
    ref = torch.distributions.Normal(mu, sigma).log_prob(torch.tensor(x)).item()
    assert abs(ours - ref) < 1e-6


def test_discount_cumsum_matches_reference():
    rew = np.array([1.0, 1.0, 1.0])
    out = discount_cumsum(rew, 0.5)
    assert np.allclose(out, [1.0 + 0.5 + 0.25, 1.0 + 0.5, 1.0])


def test_reinforce_loss_sign_and_value():
    logp = torch.tensor([np.log(0.5), np.log(0.25)])
    ret = torch.tensor([1.0, 2.0])
    L = reinforce_loss(logp, ret)
    expected = -(logp * ret).mean().item()
    assert abs(L.item() - expected) < 1e-6


def test_reinforce_with_baseline_loss_zero_mean_advantage():
    torch.manual_seed(0)
    logp = torch.randn(8)
    ret = torch.randn(8)
    val = torch.zeros(8, requires_grad=True)
    pi_loss, v_loss = reinforce_with_baseline_loss(logp, ret, val, standardize=True)
    assert torch.isfinite(pi_loss) and torch.isfinite(v_loss)
    # Value loss should produce a non-zero gradient on val.
    v_loss.backward()
    assert val.grad is not None and torch.any(val.grad != 0)


def test_optimal_baseline_recovers_mean_when_grad_constant():
    rng = np.random.default_rng(0)
    G = rng.normal(size=200)
    gnorms = np.ones_like(G)
    b = optimal_baseline(gnorms, G)
    assert abs(b - G.mean()) < 1e-9


def test_topk_multiplier_extremes():
    # K=1 should give 1 regardless of p.
    p = np.linspace(0, 1, 5)
    assert np.allclose(topk_offpolicy_multiplier(p, K=1), 1.0)
    # K>1 with p=0 gives K; with p=1 gives 0.
    assert topk_offpolicy_multiplier(np.array([0.0]), K=5)[0] == 5.0
    assert topk_offpolicy_multiplier(np.array([1.0]), K=5)[0] == 0.0


def test_snips_recovers_mean_for_target_equals_behaviour():
    rng = np.random.default_rng(1)
    logp = rng.normal(size=100)
    r = rng.normal(loc=2.0, size=100)
    v = snips_estimator(logp, logp, r)
    assert abs(v - r.mean()) < 1e-9


def test_snips_preserves_extreme_ratios_without_clipping():
    v = snips_estimator(np.array([-100., -99.]), np.array([0., 0.]), np.array([0., 1.]))
    assert abs(v - 1. / (np.e + 1.)) < 1e-12


def test_singleton_advantage_has_finite_losses_and_critic_gradient():
    logp = torch.tensor([-1.], requires_grad=True)
    values = torch.tensor([0.], requires_grad=True)
    actor_loss, value_loss = reinforce_with_baseline_loss(logp, torch.tensor([1.]), values)
    assert torch.isfinite(actor_loss) and torch.isfinite(value_loss)
    value_loss.backward()
    assert values.grad.item() == -1.
