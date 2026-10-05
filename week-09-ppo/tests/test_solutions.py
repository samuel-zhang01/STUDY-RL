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


def test_ppo_stores_the_sample_matching_old_log_prob():
    from project.agent import PPOAgent
    torch.manual_seed(0)
    agent = PPOAgent.build(2, 1, lr=0.)
    with torch.no_grad():
        agent.actor.mu_head.bias.fill_(4.)
    obs = np.zeros((2,2), dtype=np.float32)
    actions, old_logp, _ = agent.act(obs)
    assert np.any(actions > 1.)  # sampled latent, clipped only by the env
    with torch.no_grad():
        new_logp = agent.actor.dist(torch.tensor(obs)).log_prob(torch.tensor(actions)).sum(-1)
    np.testing.assert_allclose(new_logp.numpy(), old_logp, atol=1e-6)


def test_ppo_optional_losses_and_singleton_minibatch_are_honored():
    from project.agent import PPOAgent
    agent = PPOAgent.build(2,1,lr=0.)
    obs = torch.zeros(1,2); actions = torch.zeros(1,1)
    with torch.no_grad():
        old_logp = agent.actor.dist(obs).log_prob(actions).sum(-1)
        value = agent.critic(obs)
    metrics = agent.update_minibatch(obs, actions, old_logp, torch.tensor([2.]), value + 1., value + 3.,
        use_value_clip=False, use_advantage_norm=False)
    assert abs(metrics['pi_loss'] + 2.) < 1e-6
    assert abs(metrics['v_loss'] - .5) < 1e-6
    assert np.isfinite(list(metrics.values())).all()
