"""Smoke tests for week 18.

Run from the repo root: ``PYTHONPATH=. pytest week-18-rlhf-ppo-llm/tests -q``.

We do NOT depend on transformers/trl/peft -- those are documented in
``project.trl_recipe`` but the workbook's full RLHF run lives in
``project.ppo``, a pure-PyTorch implementation.
"""
import os
import sys
import time

THIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, THIS)

import torch  # noqa: E402

from project.data import ToyPreferenceDataset, heuristic_reward  # noqa: E402
from project.model import CharTransformer, RewardModel, TransformerConfig  # noqa: E402
from project.sft import sft_train  # noqa: E402
from project.rm import rm_train  # noqa: E402
from project.ppo import ppo_train, PPOConfig, compute_gae  # noqa: E402


def test_smoke_sft_rm_ppo():
    """50-step end-to-end smoke pass on a 5-pair dataset, < 10 s."""
    torch.manual_seed(0)
    t0 = time.time()
    ds = ToyPreferenceDataset(n=5, seed=0)
    cfg = TransformerConfig()
    policy = CharTransformer(cfg)
    rm = RewardModel(CharTransformer(cfg))
    sft_losses = sft_train(policy, ds.sft_strings(), steps=30, batch_size=2, seed=0)
    assert sft_losses[-1] < sft_losses[0]
    rm_losses, rm_accs = rm_train(rm, ds.pairs, steps=30, batch_size=2, seed=0)
    assert rm_losses[-1] < rm_losses[0]
    history, _, _ = ppo_train(
        policy, rm, ds.prompts(),
        cfg=PPOConfig(steps=3, batch_size=2, max_new_tokens=6, seed=0),
    )
    assert len(history["reward"]) == 3
    elapsed = time.time() - t0
    # Generous: this is a CPU integration test.
    assert elapsed < 90.0, f"smoke too slow: {elapsed:.1f}s"


def test_compute_gae_shape():
    rewards = torch.zeros(2, 4)
    values = torch.zeros(2, 5)
    adv, ret = compute_gae(rewards, values, gamma=1.0, lam=0.95)
    assert adv.shape == (2, 4)
    assert ret.shape == (2, 4)
    # Zero rewards + zero values must give zero advantages and returns.
    assert torch.allclose(adv, torch.zeros_like(adv))


def test_compute_gae_telescopes():
    """If lam == 1 and gamma == 1 with zero values, advantage = sum of future rewards."""
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.zeros(1, 4)
    adv, _ = compute_gae(rewards, values, gamma=1.0, lam=1.0)
    # adv[t] = sum_{k=t}^{T-1} rewards[k]
    assert torch.allclose(adv, torch.tensor([[6.0, 5.0, 3.0]]))


def test_heuristic_reward_sanity():
    # Polite, short.
    good = "Q: do you like tea? A: yes, tea is nice."
    # Shouty, long.
    bad = "Q: do you like tea? A: TEA IS TERRIBLE!! GO AWAY!! I HATE TEA!!"
    assert heuristic_reward(good) > heuristic_reward(bad)
