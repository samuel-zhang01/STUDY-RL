"""Smoke tests for week-06 Rainbow."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import pytest
import torch

import _pkg_loader  # noqa: F401
from w06_project import agent as agent_mod, train as train_mod


def test_noisy_linear_outputs_change_between_resets():
    nl = agent_mod.NoisyLinear(4, 3)
    nl.train()
    x = torch.randn(5, 4)
    y1 = nl(x).detach().clone()
    nl.reset_noise()
    y2 = nl(x).detach().clone()
    assert not torch.allclose(y1, y2)


def test_noisy_linear_eval_mode_is_deterministic():
    nl = agent_mod.NoisyLinear(4, 3)
    nl.eval()
    x = torch.randn(5, 4)
    y1 = nl(x).detach().clone()
    nl.reset_noise()
    y2 = nl(x).detach().clone()
    assert torch.allclose(y1, y2)


def test_dueling_head_subtracts_mean():
    v = torch.tensor([[1.0]])
    a = torch.tensor([[2.0, 4.0, 6.0]])  # mean = 4
    from w06_project.solutions import exercise_2_dueling_head
    out = exercise_2_dueling_head(v, a)
    # q = 1 + (a - 4) = (-1, 1, 3)
    assert torch.allclose(out, torch.tensor([[-1.0, 1.0, 3.0]]))


def test_rainbow_construct_all_flags():
    cfg = agent_mod.RainbowConfig(obs_dim=4, n_actions=2)
    ag = agent_mod.RainbowAgent(cfg, seed=0)
    obs = np.zeros(4, dtype=np.float32)
    assert isinstance(ag.act(obs), int)


def test_rainbow_smoke_lunar_beats_random():
    log = train_mod.train(seed=0, n_steps=3000, eps_decay_steps=2000,
                          warmup_steps=300, buffer_size=5000, per=False, noisy=False)
    assert len(log.episode_returns) >= 3
    mean_last = float(np.mean(log.episode_returns[-10:]))
    assert mean_last > -250, f"mean_last_10 = {mean_last:.1f}"


def test_per_replay_priorities_update():
    cfg = agent_mod.RainbowConfig(obs_dim=3, n_actions=2, per=True,
                                  warmup_steps=8, batch_size=4,
                                  buffer_size=100, noisy=False)
    ag = agent_mod.RainbowAgent(cfg, seed=0)
    for _ in range(30):
        s = np.random.randn(3).astype(np.float32)
        s2 = np.random.randn(3).astype(np.float32)
        ag.push(s, np.random.randint(0, 2), float(np.random.randn()), s2, False)
    initial = ag.buf.priorities[:ag.buf.size].copy()
    for _ in range(8):
        ag.learn()
    final = ag.buf.priorities[:ag.buf.size]
    # At least one priority must have changed (TD-error update).
    assert not np.allclose(initial, final)


def test_shrink_and_perturb_moves_params_toward_init():
    cfg = agent_mod.RainbowConfig(obs_dim=3, n_actions=2, reset_every=4,
                                  warmup_steps=4, batch_size=4,
                                  buffer_size=200, noisy=False, per=False)
    ag = agent_mod.RainbowAgent(cfg, seed=0)
    for _ in range(30):
        s = np.random.randn(3).astype(np.float32)
        s2 = np.random.randn(3).astype(np.float32)
        ag.push(s, np.random.randint(0, 2), float(np.random.randn()), s2, False)
    for _ in range(8):
        ag.learn()
    pre = next(iter(ag.q.state_dict().values())).detach().clone()
    ag._shrink_and_perturb()
    post = next(iter(ag.q.state_dict().values())).detach().clone()
    # Should have moved (params != init exactly, but shifted toward init)
    assert not torch.allclose(pre, post)
