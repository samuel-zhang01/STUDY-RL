"""Tests for the reference solutions in week-06."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import torch

import _pkg_loader  # noqa: F401
from w06_project import solutions as S


def test_ex1_double_dqn_target():
    r = torch.tensor([1.0])
    d = torch.tensor([0.0])
    q_on  = torch.tensor([[5.0, 1.0]])  # argmax = 0
    q_tg  = torch.tensor([[3.0, 9.0]])  # value at action 0 = 3
    out = S.exercise_1_double_dqn_target(r, d, q_on, q_tg, 0.5)
    assert torch.allclose(out, torch.tensor([1.0 + 0.5 * 3.0]))


def test_ex2_dueling_head_mean_zero():
    v = torch.zeros(2, 1)
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    out = S.exercise_2_dueling_head(v, a)
    # sum of advantage portion is zero
    assert torch.allclose(out.sum(dim=-1), torch.tensor([0.0, 0.0]))


def test_ex3_per_weight_in_unit_interval():
    p = np.array([0.5, 0.3, 0.2])
    w = S.exercise_3_per_weight(p, N=10, beta=0.4)
    assert w.max() == 1.0 and (w >= 0).all()


def test_ex4_priority_update():
    p = np.ones(5)
    out = S.exercise_4_priority_update(p, np.array([1, 3]),
                                       np.array([-2.0, 0.5]), eps=1e-6)
    assert out[1] > 1.999 and out[3] > 0.499


def test_ex5_noisy_f_sign_preserved():
    x = torch.tensor([-4.0, 0.0, 9.0])
    out = S.exercise_5_noisy_f(x)
    assert torch.allclose(out, torch.tensor([-2.0, 0.0, 3.0]))


def test_ex6_categorical_project_preserves_total_mass():
    p = torch.tensor([[0.2, 0.3, 0.5]])
    support = torch.tensor([-1.0, 0.0, 1.0])
    out = S.exercise_6_categorical_project(p, support,
                                           reward=torch.tensor([0.1]),
                                           done=torch.tensor([0.0]),
                                           gamma=0.9)
    assert abs(out.sum().item() - 1.0) < 1e-5


def test_ex6_categorical_project_terminal_concentrates_on_reward_bin():
    # Done -> Tz = r. r = 0 should land entirely on the middle atom (z=0).
    p = torch.tensor([[1/3, 1/3, 1/3]])
    support = torch.tensor([-1.0, 0.0, 1.0])
    out = S.exercise_6_categorical_project(p, support,
                                           reward=torch.tensor([0.0]),
                                           done=torch.tensor([1.0]),
                                           gamma=0.9)
    assert out[0, 1] > 0.99


def test_ex7_nstep_return_with_and_without_bootstrap():
    # rewards = [1,1,1], n=2, gamma=1, done_last=False, q=10 -> 1+1+10=12
    assert S.exercise_7_nstep_return([1, 1, 1], 10.0, 1.0, 2, False) == 12.0
    # n>len(rewards): no bootstrap
    assert S.exercise_7_nstep_return([1, 1], 10.0, 1.0, 5, False) == 2.0
    # done_last=True: no bootstrap
    assert S.exercise_7_nstep_return([1, 1, 1], 10.0, 1.0, 2, True) == 2.0


def test_ex8_shrink_perturb_endpoints():
    th = torch.ones(3)
    th0 = torch.zeros(3)
    out = S.exercise_8_shrink_perturb(th, th0, alpha=1.0)
    assert torch.allclose(out, th)  # alpha=1 -> no change
    out = S.exercise_8_shrink_perturb(th, th0, alpha=0.0)
    assert torch.allclose(out, th0)  # alpha=0 -> full reset


def test_ex9_smoke_returns_three_runs():
    res = S.exercise_9_smoke_ablate(n_steps=800)
    assert set(res.keys()) == {"full", "no_per", "no_noisy"}


def test_noisy_agent_deterministic_action_uses_mean_parameters(monkeypatch):
    from w06_project.agent import RainbowAgent, RainbowConfig
    agent = RainbowAgent(RainbowConfig(obs_dim=2, n_actions=2, hidden=(8,), device='cpu'))
    def unexpected_noise():
        raise AssertionError('A deterministic action must not resample noise')
    monkeypatch.setattr(agent.q, 'reset_noise', unexpected_noise)
    obs = np.zeros(2, dtype=np.float32)
    actions = [agent.act(obs, deterministic=True) for _ in range(3)]
    assert len(set(actions)) == 1
    assert agent.q.training


def test_rainbow_replay_time_limit_keeps_bootstrap(monkeypatch):
    import importlib
    import gymnasium as gym
    trainer = importlib.import_module('w06_project.train')
    class Env:
        observation_space = gym.spaces.Box(-1., 1., (2,))
        action_space = gym.spaces.Discrete(2)
        def reset(self, seed=None): return np.zeros(2, dtype=np.float32), {}
        def step(self, a): return np.ones(2, dtype=np.float32), 1., False, True, {}
        def close(self): pass
    monkeypatch.setattr(trainer.gym, 'make', lambda *args: Env())
    instances = []
    original = trainer.RainbowAgent
    def agent(*args, **kwargs):
        obj = original(*args, **kwargs); instances.append(obj); return obj
    monkeypatch.setattr(trainer, 'RainbowAgent', agent)
    trainer.train(n_steps=1, hidden=(8,), warmup_steps=10)
    assert instances[0].buf.done[0] == 0.
