"""Reference answers. Attempt exercises.py before opening this file.

These small operations also power the reference project demonstrations.
"""
from __future__ import annotations

import math
from typing import Iterable


def mean_update(old_mean: float, reward: float, new_count: int) -> float:
    if new_count < 1:
        raise ValueError("new_count counts this arm after its new observation")
    return old_mean + (reward - old_mean) / new_count


def discounted_return(rewards: Iterable[float], gamma: float) -> float:
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    return sum(gamma ** step * reward for step, reward in enumerate(rewards))


def q_target(reward: float, gamma: float, next_values: Iterable[float],
             terminated: bool) -> float:
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if terminated:
        return reward
    values = list(next_values)
    if not values:
        raise ValueError("a nonterminal state needs next-action values")
    return reward + gamma * max(values)


def ips(rewards: Iterable[float], logging_probabilities: Iterable[float],
        target_probabilities: Iterable[float]) -> float:
    rewards, logger, target = map(list, (rewards, logging_probabilities, target_probabilities))
    if not rewards or len(rewards) != len(logger) or len(rewards) != len(target):
        raise ValueError("provide nonempty, equally sized observed-row arrays")
    if not all(math.isfinite(r) for r in rewards):
        raise ValueError("rewards must be finite")
    if any(not 0 < p <= 1 for p in logger) or any(not 0 <= p <= 1 for p in target):
        raise ValueError("observed propensities must be positive; probabilities lie in [0, 1]")
    # This observed-row formula cannot itself detect an unobserved action's support.
    # The study_bandit evaluator checks full context/action distributions separately.
    return sum(r * p / b for r, b, p in zip(rewards, logger, target)) / len(rewards)


def shield(actions: Iterable[str], costs: dict[str, float], budget: float) -> list[str]:
    if budget < 0 or not math.isfinite(budget):
        raise ValueError("budget must be finite and nonnegative")
    actions = list(actions)
    if any(a not in costs for a in actions):
        raise ValueError("every candidate action needs a predicted cost")
    if any(not math.isfinite(costs[a]) or costs[a] < 0 for a in actions):
        raise ValueError("predicted costs must be finite and nonnegative")
    return [a for a in actions if costs[a] <= budget]
