"""Synthetic study-session bandit: quiz, read, or explain, given fresh/tired context.

Rewards are invented Bernoulli outcomes, not measurements of a real person's
learning. The context is sampled independently of past choices: this is a
contextual bandit, not a model of long-term knowledge or fatigue.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import random
from typing import Callable

from . import reference

CONTEXTS = ("fresh", "tired")
ACTIONS = ("quiz", "read", "explain")
MEANS = {"fresh": (0.85, 0.55, 0.65), "tired": (0.35, 0.75, 0.45)}
TARGET = {"fresh": (1.0, 0.0, 0.0), "tired": (0.0, 1.0, 0.0)}


@dataclass(frozen=True)
class LogRow:
    context: str
    action: int
    reward: int
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class Estimates:
    ips: float
    snips: float
    effective_sample_size: float
    oracle: float


def probabilities(means: list[float], epsilon: float) -> tuple[float, ...]:
    """Full epsilon-greedy propensity, including random selection of best actions."""
    if not means or any(not math.isfinite(v) for v in means) or not 0 <= epsilon <= 1:
        raise ValueError("provide finite means and epsilon in [0, 1]")
    best = max(means)
    greedy = [i for i, v in enumerate(means) if math.isclose(v, best, abs_tol=1e-12)]
    return tuple(epsilon / len(means) + ((1 - epsilon) / len(greedy) if i in greedy else 0)
                 for i in range(len(means)))


def draw_action(probs: tuple[float, ...], rng: random.Random) -> int:
    if not probs or any(not 0 <= p <= 1 for p in probs) or not math.isclose(sum(probs), 1.0):
        raise ValueError("action probabilities must be nonnegative and sum to one")
    u, cumulative = rng.random(), 0.0
    for action, p in enumerate(probs):
        cumulative += p
        if u < cumulative:
            return action
    return len(probs) - 1  # Rounding at the last cumulative probability.


def collect(rounds: int = 1500, seed: int = 7, epsilon: float = 0.2,
            update: Callable = reference.mean_update) -> tuple[list[LogRow], dict[str, list[float]]]:
    if rounds < 1 or not 0 <= epsilon <= 1:
        raise ValueError("rounds must be positive; epsilon must be in [0, 1]")
    rng = random.Random(seed)
    means = {c: [0.0] * len(ACTIONS) for c in CONTEXTS}
    counts = {c: [0] * len(ACTIONS) for c in CONTEXTS}
    rows = []
    for _ in range(rounds):
        context = rng.choice(CONTEXTS)
        probs = probabilities(means[context], epsilon)
        action = draw_action(probs, rng)
        reward = int(rng.random() < MEANS[context][action])
        rows.append(LogRow(context, action, reward, probs))
        counts[context][action] += 1
        new_mean = update(means[context][action], reward, counts[context][action])
        if not math.isfinite(new_mean):
            raise ValueError("mean_update returned a nonfinite estimate")
        means[context][action] = new_mean
    return rows, means


def evaluate_logs(rows: list[LogRow], target: dict[str, tuple[float, ...]] = TARGET) -> Estimates:
    """Evaluate a target fixed before seeing these logs; reject support failures.

    Every row retains its logger's full action distribution, not just the chosen
    propensity. A positive target probability with zero logging probability is
    a missing-counterfactual problem even if that action never appears in rows.
    Adaptive logging is permitted here because its actual propensities are known;
    this function does not report independent-row confidence intervals.
    """
    if not rows:
        raise ValueError("cannot evaluate an empty log")
    for context in CONTEXTS:
        if context not in target or len(target[context]) != len(ACTIONS):
            raise ValueError("target needs one full action distribution per context")
        p = target[context]
        if any(not 0 <= v <= 1 for v in p) or not math.isclose(sum(p), 1.0):
            raise ValueError("target action probabilities must sum to one")
    weights, rewards = [], []
    for row in rows:
        if row.context not in CONTEXTS or not 0 <= row.action < len(ACTIONS):
            raise ValueError("invalid logged context or action")
        if row.reward not in (0, 1) or len(row.probabilities) != len(ACTIONS):
            raise ValueError("invalid reward or propensity vector")
        b = row.probabilities
        if any(not 0 <= p <= 1 for p in b) or not math.isclose(sum(b), 1.0):
            raise ValueError("logged distributions must sum to one")
        p = target[row.context]
        if any(pi > 0 and bi == 0 for pi, bi in zip(p, b)):
            raise ValueError("missing support: target selects an action the logger never tried")
        if b[row.action] <= 0:
            raise ValueError("an observed action cannot have zero logging propensity")
        weights.append(p[row.action] / b[row.action])
        rewards.append(row.reward)
    total = sum(weights)
    if total == 0:
        raise ValueError("zero target overlap in this finite sample: collect more evaluation data")
    weighted = sum(w * r for w, r in zip(weights, rewards))
    oracle = sum(sum(pi * mu for pi, mu in zip(target[c], MEANS[c])) for c in CONTEXTS) / 2
    return Estimates(weighted / len(rows), weighted / total,
                     total ** 2 / sum(w * w for w in weights), oracle)


def evaluate_greedy(means: dict[str, list[float]], rounds: int = 500,
                    seed: int = 1007) -> float:
    """Fresh simulator outcomes for the learned greedy policy; no updates."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    rng, reward = random.Random(seed), 0
    for _ in range(rounds):
        context = rng.choice(CONTEXTS)
        action = draw_action(probabilities(means[context], 0), rng)
        reward += int(rng.random() < MEANS[context][action])
    return reward / rounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--learner", action="store_true", help="use your exercises.mean_update")
    args = parser.parse_args()
    from . import exercises
    update = exercises.mean_update if args.learner else reference.mean_update
    try:
        rows, means = collect(args.rounds, args.seed, args.epsilon, update)
    except (ValueError, NotImplementedError) as exc:
        parser.error(f"{exc}. Attempt exercises.py, then run python -m learning_projects.check_exercises.")
    print("Your implementation" if args.learner else "Reference demonstration; use --learner after exercise 1.")
    print("Synthetic simulator, equally likely contexts; rewards are invented, not personal learning scores.")
    for context in CONTEXTS:
        print(f"{context}: estimates=" + ", ".join(f"{a}:{m:.3f}" for a, m in zip(ACTIONS, means[context])))
    print(f"Logged reward mean={sum(r.reward for r in rows) / len(rows):.3f}")
    print(f"Fresh greedy evaluation reward={evaluate_greedy(means, seed=args.seed + 1000):.3f}")
    print("Fixed evaluation target: quiz when fresh, read when tired (chosen before collecting logs).")
    try:
        estimates = evaluate_logs(rows)
    except ValueError as exc:
        print(f"OPE unavailable: {exc}")
    else:
        print(f"IPS={estimates.ips:.3f}; SNIPS={estimates.snips:.3f}; "
              f"ESS={estimates.effective_sample_size:.1f}/{len(rows)}")
        print(f"Simulator-only target oracle={estimates.oracle:.3f}; not an observable real-world label.")
    print("Repeat --seed 1, --seed 7, --seed 19; do not conclude a method wins from one lucky run.")


if __name__ == "__main__":
    main()
