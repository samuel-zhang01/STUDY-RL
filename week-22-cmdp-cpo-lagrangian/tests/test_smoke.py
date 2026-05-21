"""Smoke test: PPO-Lagrangian on LavaGridWorld converges in < 30 s and beats
the random baseline on return while keeping cost finite."""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)
sys.path.insert(0, os.path.dirname(WEEK))  # repo root for `common`

from project.train import train  # noqa: E402
from project.env import LavaGridWorld  # noqa: E402


def random_baseline(n_episodes: int = 40, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    env = LavaGridWorld()
    env.reset(seed=seed)
    rets = []
    for _ in range(n_episodes):
        env.reset()
        R, done = 0.0, False
        while not done:
            a = int(rng.integers(0, env.n_actions))
            _, r, _, done, _ = env.step(a)
            R += r
        rets.append(R)
    return float(np.mean(rets))


def test_train_returns_history():
    h = train(seed=0, n_updates=12, steps_per_update=256, cost_limit=1.5)
    assert {"return", "cost", "lambda"}.issubset(h.keys())
    assert len(h["return"]) == 12
    assert all(np.isfinite(h["return"])), "non-finite returns produced"
    assert all(np.isfinite(h["cost"])), "non-finite costs produced"


def test_beats_random_baseline():
    h = train(seed=0, n_updates=40, steps_per_update=512, cost_limit=1.5)
    final = float(np.mean(h["return"][-5:]))
    base = random_baseline(seed=0)
    # PPO-Lag should clearly beat random in 40 updates.
    assert final > base + 0.1, f"final return {final:.3f} <= baseline {base:.3f} + 0.1"


def test_lambda_responds_to_violation():
    # With a near-zero budget, the multiplier must grow above its initial value
    # at least once during training.
    h = train(seed=0, n_updates=20, steps_per_update=256, cost_limit=0.0)
    assert max(h["lambda"]) > 0.0, "lambda never moved above 0 despite violations"
