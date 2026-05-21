"""A constrained gridworld with a lava-cost channel.

The environment is a 7x7 grid:
    S . . . . . .
    . . L L . . .
    . . L L . . .
    . . . . . . .
    . . . L L . .
    . . . L L . .
    . . . . . . G

  S = start (0,0), G = goal (6,6).
  L = lava cell; stepping onto lava incurs cost 1 but does NOT terminate.

Reward: -0.01 per step, +1 on reaching the goal, 0 elsewhere. Discount gamma in [0,1).
Cost: +1 on entering a lava cell, 0 otherwise.

The CMDP is
    maximise  E[sum_t gamma^t r_t]
    subject to E[sum_t gamma^t c_t] <= d.

This is *not* an unsafe-terminal MDP — the policy can in principle step on lava and
recover, which is exactly the situation Lagrangian methods address: the agent must
*trade off* a tiny lava-cost budget against the reward of a faster route.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np

# Action encoding: 0=up, 1=right, 2=down, 3=left.
_DELTAS: List[Tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]


@dataclass
class LavaGridWorld:
    rows: int = 7
    cols: int = 7
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (6, 6)
    lava: Tuple[Tuple[int, int], ...] = (
        (1, 2), (1, 3), (2, 2), (2, 3),
        (4, 3), (4, 4), (5, 3), (5, 4),
    )
    max_steps: int = 80
    slip_prob: float = 0.05
    step_reward: float = -0.01
    goal_reward: float = 1.0
    lava_cost: float = 1.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self) -> None:
        self.lava_set = set(self.lava)
        self.state = self.start
        self.t = 0

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return 4

    def s2idx(self, s: Tuple[int, int]) -> int:
        return s[0] * self.cols + s[1]

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start
        self.t = 0
        return self.s2idx(self.state)

    def step(self, action: int) -> Tuple[int, float, float, bool, dict]:
        """Returns (next_state_idx, reward, cost, done, info)."""
        if self.slip_prob > 0 and self.rng.random() < self.slip_prob:
            action = int(self.rng.integers(0, 4))
        dr, dc = _DELTAS[action]
        r, c = self.state
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            nr, nc = r, c
        self.state = (nr, nc)
        self.t += 1
        cost = self.lava_cost if self.state in self.lava_set else 0.0
        if self.state == self.goal:
            return self.s2idx(self.state), self.goal_reward, cost, True, {"goal": True}
        reward = self.step_reward
        done = self.t >= self.max_steps
        return self.s2idx(self.state), reward, cost, done, {}

    # ---- helpers for tabular pretty-printing in the notebook ----
    def one_hot_obs(self, idx: int) -> np.ndarray:
        v = np.zeros(self.n_states, dtype=np.float32)
        v[idx] = 1.0
        return v

    def render_ascii(self) -> str:
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.start:
                    row.append("S")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) in self.lava_set:
                    row.append("L")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)
