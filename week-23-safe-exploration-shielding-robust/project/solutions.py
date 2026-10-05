"""Reference solutions for the week-23 workbook exercises."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .env_grid import DangerGridWorld
from .shield import DFAShield, build_no_lava_visit_shield
from .q_learn import (
    QLearner, train_shielded as _train_shielded, train_unshielded as _train_unshielded,
)
from .pendulum_train import train_pendulum_dr as _train_pendulum, PendulumTrainConfig


# -----------------------------------------------------------------------------
# Exercise 1 ★ — build the DFA for "G !danger" (two-state safety automaton).
# -----------------------------------------------------------------------------
def exercise_1() -> Dict:
    """Return a dictionary describing the safety automaton for G(¬danger).

    Keys: 'states' (list[str]), 'initial' (str), 'accepting' (set[str]),
    'transitions' (dict[(state, label) -> state]). Labels are 'danger' or '_'.
    """
    return {
        "states": ["q0", "q_bad"],
        "initial": "q0",
        "accepting": {"q0"},
        "transitions": {
            ("q0", "_"): "q0",
            ("q0", "danger"): "q_bad",
            ("q_bad", "_"): "q_bad",
            ("q_bad", "danger"): "q_bad",
        },
    }


# -----------------------------------------------------------------------------
# Exercise 2 ★ — one-step look-ahead unsafe predicate.
# -----------------------------------------------------------------------------
def exercise_2(env: DangerGridWorld, s_idx: int, a: int) -> bool:
    """Return True iff action `a` in state `s_idx` would enter a danger cell."""
    return env.adjacent_dangerous(s_idx, a)


# -----------------------------------------------------------------------------
# Exercise 3 ★★ — implement pre-shield filtering.
# -----------------------------------------------------------------------------
def exercise_3(env: DangerGridWorld, preferences: List[int], s_idx: int) -> int:
    """Given a list of action preferences (most preferred first) and the
    current state, return the most-preferred SAFE action."""
    shield = build_no_lava_visit_shield(env)
    return shield.filter(s_idx, list(preferences))


# -----------------------------------------------------------------------------
# Exercise 4 ★★ — train shielded vs unshielded and compare safety records.
# -----------------------------------------------------------------------------
def exercise_4(seed: int = 0, n_episodes: int = 300) -> Dict[str, Dict]:
    h_s = _train_shielded(n_episodes=n_episodes, seed=seed)
    h_u = _train_unshielded(n_episodes=n_episodes, seed=seed)
    return {"shielded": h_s, "unshielded": h_u}


# -----------------------------------------------------------------------------
# Exercise 5 ★★ — empirical check of the conditional soundness claim.
# -----------------------------------------------------------------------------
def exercise_5(seed: int = 0, n_episodes: int = 300) -> int:
    """Return observed unsafe transitions on the supplied deterministic grid.

    Formal soundness additionally requires safe initialization, a complete
    model, safe-set feasibility, and exact execution of the selected action.
    """
    h = _train_shielded(n_episodes=n_episodes, seed=seed)
    return int(h["total_unsafe"])


# -----------------------------------------------------------------------------
# Exercise 6 ★★ — robust pendulum: DR vs fixed length.
# -----------------------------------------------------------------------------
def exercise_6(seed: int = 0, n_epochs: int = 20, steps_per_epoch: int = 400):
    cfg = PendulumTrainConfig(n_epochs=n_epochs, steps_per_epoch=steps_per_epoch)
    dr = _train_pendulum(seed=seed, randomise=True, cfg=cfg)
    fixed = _train_pendulum(seed=seed, randomise=False, cfg=cfg)
    return {"dr": dr, "fixed": fixed}


# -----------------------------------------------------------------------------
# Exercise 7 ★★★ — CBF QP-projection in closed form for a 1-D linear constraint.
# -----------------------------------------------------------------------------
def exercise_7(a_proposed: float, h_val: float, dh_da: float, alpha: float = 1.0) -> float:
    """One-dimensional CBF safety filter: project `a_proposed` onto the
    half-space { a : dh_da * a >= -alpha * h_val } that enforces forward
    invariance of {h >= 0}.

    Returns the feasible action closest to a_proposed in Euclidean distance.
    """
    rhs = -alpha * h_val
    # constraint: dh_da * a >= rhs
    if dh_da == 0:
        if rhs > 0:
            raise ValueError("CBF constraint is infeasible: no action can satisfy it")
        return float(a_proposed)
    if dh_da * a_proposed >= rhs - 1e-12:
        return float(a_proposed)  # already feasible
    # Otherwise project: closest a satisfying equality.
    return float(rhs / dh_da)


# -----------------------------------------------------------------------------
# Exercise 8 ★★★ — adversarial domain shift on pendulum.
# -----------------------------------------------------------------------------
def exercise_8(seed: int = 0, n_epochs: int = 20, steps_per_epoch: int = 400):
    """Train a DR policy, then evaluate at adversarial-extreme lengths."""
    cfg = PendulumTrainConfig(n_epochs=n_epochs, steps_per_epoch=steps_per_epoch)
    h = _train_pendulum(seed=seed, randomise=True, cfg=cfg,
                        eval_lengths=(0.25, 0.5, 1.0, 1.5, 2.0))
    return h["eval"]
