"""A 5x5 gridworld with a designated DANGER region.

Layout:
    S . . . .
    . . D . .
    . D D D .
    . . D . .
    . . . . G

S = start (0,0). G = goal (4,4). D = danger cells.

The safety property we will enforce in this week is:
    G (no danger visited)     [in LTL: G !danger]

This is a *safety* (path-bounded) property: any prefix of any trajectory must
never contain a danger cell.

The environment exposes:
    * step(a) -> (next_s, reward, done, info) where info['unsafe'] is True iff
      the agent just entered a danger cell.
    * label(s) -> {"danger"} or set() — the atomic propositions used by the LTL
      shield to detect violations.
    * adjacent_dangerous(s, a) -> True iff taking action a in state s would
      enter danger; used by the shield as a one-step look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Set, Dict

import numpy as np

_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left


@dataclass
class DangerGridWorld:
    rows: int = 5
    cols: int = 5
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (4, 4)
    danger: Tuple[Tuple[int, int], ...] = (
        (1, 2),
        (2, 1), (2, 2), (2, 3),
        (3, 2),
    )
    max_steps: int = 50
    step_reward: float = -0.1
    danger_reward: float = -1.0
    goal_reward: float = 5.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self):
        self.danger_set = set(self.danger)
        self.state = self.start
        self.t = 0

    @property
    def n_states(self):
        return self.rows * self.cols

    @property
    def n_actions(self):
        return 4

    def s2idx(self, s: Tuple[int, int]) -> int:
        return s[0] * self.cols + s[1]

    def idx2s(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.cols)

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start
        self.t = 0
        return self.s2idx(self.state)

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        dr, dc = _DELTAS[action]
        r, c = self.state
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            nr, nc = r, c
        self.state = (nr, nc)
        self.t += 1
        info: Dict = {}
        if self.state in self.danger_set:
            info["unsafe"] = True
            reward = self.danger_reward
            done = True
        elif self.state == self.goal:
            info["unsafe"] = False
            reward = self.goal_reward
            done = True
        else:
            info["unsafe"] = False
            reward = self.step_reward
            done = self.t >= self.max_steps
        return self.s2idx(self.state), reward, done, info

    # ----------------------------------------------------------
    # LTL labelling
    # ----------------------------------------------------------
    def label(self, s_idx: int) -> Set[str]:
        s = self.idx2s(s_idx)
        return {"danger"} if s in self.danger_set else set()

    def adjacent_dangerous(self, s_idx: int, a: int) -> bool:
        """One-step look-ahead: would taking action `a` in state `s` land in danger?

        This is the model the shield uses; it is *complete* for our gridworld
        because dynamics are deterministic.
        """
        s = self.idx2s(s_idx)
        dr, dc = _DELTAS[a]
        nr, nc = s[0] + dr, s[1] + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            nr, nc = s
        return (nr, nc) in self.danger_set

    def render_ascii(self) -> str:
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.start:
                    row.append("S")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) in self.danger_set:
                    row.append("D")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)
