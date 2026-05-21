"""Smoke test for the GRPO mini-project.

Runs ~30 GRPO steps and asserts:
* the training loop does not crash,
* the reward mean is in [0, 1] at every step,
* the KL is finite,
* held-out greedy accuracy is at least 0 (sanity floor — random init is ~0).

Runs in well under 30 seconds on CPU.
"""
from __future__ import annotations
import math

import numpy as np
import pytest
import torch

from project.train import train


@pytest.mark.smoke
def test_grpo_smoke_runs():
    # Keep SFT brief to stay under the smoke-time budget.
    res = train(seed=0, num_steps=30, eval_every=15, batch_prompts=2,
                group_size=4, sft_warmup_steps=30)
    hist = res["history"]
    assert len(hist["reward_mean"]) == 30
    for r in hist["reward_mean"]:
        assert 0.0 <= r <= 1.0 + 1e-6, f"reward out of range: {r}"
    for k in hist["kl"]:
        assert math.isfinite(k)
    assert res["final_acc"] >= 0.0


@pytest.mark.smoke
def test_grpo_improves_above_random():
    """Train for ~200 steps and check accuracy beats a baseline of 0.05."""
    res = train(seed=0, num_steps=100, eval_every=100, batch_prompts=4,
                group_size=8, sft_warmup_steps=400)
    # After SFT warm-up + a short GRPO pass we should comfortably clear
    # 20% on the toy task.
    assert res["final_acc"] >= 0.2, f"final_acc was {res['final_acc']}"
