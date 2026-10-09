"""Your five small implementations: no reference answers are imported here.

Run ``python -m learning_projects.check_exercises`` after editing this file.
Read docs/INTERACTIVE_LEARNING.md for predictions, hints, and extensions.
"""
from __future__ import annotations

from typing import Iterable


def mean_update(old_mean: float, reward: float, new_count: int) -> float:
    """Week 1: update this arm's mean; new_count includes its new reward.

    Example: old mean 1, new reward 0, new count 2 → 0.5.
    """
    raise NotImplementedError("Exercise 1: update the sample mean")


def discounted_return(rewards: Iterable[float], gamma: float) -> float:
    """Weeks 2–3: first reward has weight 1, then gamma, then gamma².

    Example: rewards [0, 4] with gamma 0.9 → 3.6.
    """
    raise NotImplementedError("Exercise 2: calculate the discounted return")


def q_target(reward: float, gamma: float, next_values: Iterable[float],
             terminated: bool) -> float:
    """Week 4: terminal reward has no future; a collector cutoff still does.

    Example: reward -1, gamma 0.9, next_values [4, -2], not terminal → 2.6.
    """
    raise NotImplementedError("Exercise 3: build the Q-learning target")


def ips(rewards: Iterable[float], logging_probabilities: Iterable[float],
        target_probabilities: Iterable[float]) -> float:
    """Week 11: average reward × target propensity / logging propensity.

    The three arrays contain OBSERVED rows, not all possible actions.
    This exercise assumes full support has already been checked separately.
    """
    raise NotImplementedError("Exercise 4: reweight observed rewards")


def shield(actions: Iterable[str], costs: dict[str, float], budget: float) -> list[str]:
    """Weeks 22–23: keep actions whose known one-step cost <= budget.

    This local rule assumes exact immediate costs; it is not a proof of safety
    over a trajectory with unknown dynamics.
    """
    raise NotImplementedError("Exercise 5: filter actions before choosing one")
