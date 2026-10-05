"""Tests for the reference solutions in week-05."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch

import _pkg_loader  # noqa: F401
from w05_project import solutions as S


def test_ex1_gather():
    q_all = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a = torch.tensor([2, 0])
    out = S.exercise_1_gather_q(q_all, a)
    assert torch.allclose(out, torch.tensor([3.0, 4.0]))


def test_ex2_dqn_target():
    r = torch.tensor([1.0, 2.0])
    d = torch.tensor([0.0, 1.0])
    qn = torch.tensor([[0.0, 10.0], [-1.0, 5.0]])
    t = S.exercise_2_dqn_target(r, d, qn, gamma=0.9)
    assert torch.allclose(t, torch.tensor([1.0 + 0.9 * 10.0, 2.0]))


def test_ex3_polyak_endpoints():
    a = torch.nn.Linear(3, 1, bias=False)
    b = torch.nn.Linear(3, 1, bias=False)
    a.weight.data.fill_(1.0); b.weight.data.fill_(5.0)
    S.exercise_3_polyak(b, a, tau=0.5)
    assert torch.allclose(b.weight, torch.full_like(b.weight, 3.0))
    # tau=1 -> b becomes a
    S.exercise_3_polyak(b, a, tau=1.0)
    assert torch.allclose(b.weight, a.weight)


def test_ex4_eps_greedy_deterministic_when_eps_zero():
    rng = np.random.default_rng(0)
    q = np.array([1.0, 5.0, 2.0])
    for _ in range(20):
        assert S.exercise_4_eps_greedy(q, 0.0, rng) == 1


def test_ex5_td_error():
    q = torch.tensor([1.0, 2.0])
    t = torch.tensor([3.0, 0.0])
    out = S.exercise_5_td_error(q, t)
    assert torch.allclose(out, torch.tensor([2.0, 2.0]))


def test_ex6_huber_zero_at_match():
    p = torch.tensor([1.0, 2.0]); t = torch.tensor([1.0, 2.0])
    assert S.exercise_6_huber(p, t).item() == 0.0


def test_ex7_baird_diverges():
    norms = S.exercise_7_baird_divergence(n_iters=200, alpha=0.1)
    # Final norm should be much larger than initial.
    assert norms[-1] > 5 * norms[0]


def test_ex8_smoke_returns_log():
    log = S.exercise_8_smoke_dqn(n_steps=1000, seed=0)
    assert len(log.episode_returns) >= 1


def test_dqn_replay_time_limit_keeps_bootstrap(monkeypatch):
    import importlib
    import gymnasium as gym
    trainer = importlib.import_module('w05_project.train')
    class Env:
        observation_space = gym.spaces.Box(-1., 1., (2,))
        action_space = gym.spaces.Discrete(2)
        def reset(self, seed=None): return np.zeros(2, dtype=np.float32), {}
        def step(self, a): return np.ones(2, dtype=np.float32), 1., False, True, {}
        def close(self): pass
    monkeypatch.setattr(trainer, '_make_env', lambda *args: Env())
    instances = []
    original = trainer.DQNAgent
    def agent(*args, **kwargs):
        obj = original(*args, **kwargs); instances.append(obj); return obj
    monkeypatch.setattr(trainer, 'DQNAgent', agent)
    trainer.train(n_steps=1, hidden=(8,), warmup_steps=10)
    assert instances[0].buf.done[0] == 0.
