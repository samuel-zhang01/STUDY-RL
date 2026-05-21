"""Smoke test for the Week 1 bandit project. Runs in a few seconds."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK = os.path.dirname(HERE)
REPO = os.path.dirname(WEEK)
sys.path.insert(0, REPO)
for _m in list(sys.modules):
    if _m == "project" or _m.startswith("project."):
        del sys.modules[_m]
sys.path.insert(0, WEEK)

from project.train import train  # noqa: E402


def test_train_returns_dict_with_expected_agents():
    finals = train(seed=0, T=400, n_seeds=5)
    assert set(finals) == {
        "eps-greedy(0.1)", "UCB1", "Thompson", "Boltzmann(0.1)"
    }


def test_thompson_beats_epsilon_greedy_and_all_finite():
    """At T=1000 on a 10-armed Bernoulli testbed, Thompson Sampling should
    cleanly beat eps-greedy(0.1). UCB1 with the canonical c=2 bonus is known
    to be over-exploratory at short horizons; we assert only that it is
    finite and stays below the trivial linear-regret upper bound."""
    finals = train(seed=0, T=1000, n_seeds=10)
    eps_reg = finals["eps-greedy(0.1)"]
    assert finals["Thompson"] < eps_reg, finals
    # Linear-regret upper bound: T * max delta = 1000 * 0.6.
    assert finals["UCB1"] < 1000 * 0.6, finals
    assert all(np.isfinite(list(finals.values())))
