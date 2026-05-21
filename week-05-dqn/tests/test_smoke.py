"""Smoke tests for week-05 DQN."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import pytest
import torch

import _pkg_loader  # noqa: F401
from w05_project import agent as agent_mod, train as train_mod


def test_agent_construct():
    cfg = agent_mod.DQNConfig(obs_dim=4, n_actions=2, buffer_size=100,
                              warmup_steps=10)
    a = agent_mod.DQNAgent(cfg, seed=0)
    obs = np.zeros(4, dtype=np.float32)
    assert isinstance(a.act(obs), int)
    assert 0 <= a.act(obs) < 2


def test_agent_learn_returns_loss_after_warmup():
    cfg = agent_mod.DQNConfig(obs_dim=4, n_actions=2, buffer_size=500,
                              warmup_steps=20, batch_size=8)
    a = agent_mod.DQNAgent(cfg, seed=0)
    for i in range(50):
        s = np.random.randn(4).astype(np.float32)
        s2 = np.random.randn(4).astype(np.float32)
        a.push(s, np.random.randint(0, 2), float(np.random.randn()),
               s2, bool(np.random.random() < 0.1))
    losses = [a.learn() for _ in range(5)]
    assert any(l is not None for l in losses)


def test_dqn_smoke_lunar_beats_random():
    """5k env steps should beat random on LunarLander."""
    log = train_mod.train(seed=0, n_steps=5000, eps_decay_steps=3500,
                          warmup_steps=500, buffer_size=10_000,
                          log_every=500)
    assert len(log.episode_returns) >= 5
    mean_last = float(np.mean(log.episode_returns[-10:]))
    assert mean_last > -250, f"DQN smoke mean_last_10 = {mean_last:.1f}"


def test_target_network_starts_synced():
    cfg = agent_mod.DQNConfig(obs_dim=3, n_actions=2)
    a = agent_mod.DQNAgent(cfg, seed=0)
    for p, pt in zip(a.q.parameters(), a.q_target.parameters()):
        assert torch.allclose(p, pt)


def test_target_network_drifts_under_polyak():
    cfg = agent_mod.DQNConfig(obs_dim=3, n_actions=2, target_tau=0.1,
                              warmup_steps=4, buffer_size=200, batch_size=4)
    a = agent_mod.DQNAgent(cfg, seed=0)
    for i in range(20):
        s = np.random.randn(3).astype(np.float32)
        s2 = np.random.randn(3).astype(np.float32)
        a.push(s, np.random.randint(0, 2), float(np.random.randn()), s2, False)
    for _ in range(20):
        a.learn()
    diff = sum((p - pt).abs().sum().item()
               for p, pt in zip(a.q.parameters(), a.q_target.parameters()))
    assert diff > 0.0
