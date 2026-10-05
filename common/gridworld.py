"""A tiny, dependency-free GridWorld used across early lectures (MDP, DP, MC, TD).

Coordinates: (row, col). Origin top-left. Actions: 0=up,1=right,2=down,3=left.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import numpy as np

ACTIONS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}
ACTION_NAMES = ["↑", "→", "↓", "←"]


@dataclass
class GridWorld:
    """Deterministic by default; pass `slip_prob` for a stochastic variant.

    Reward: -1 per step, +10 at any goal, -10 at any pit. Episode ends on goal/pit.
    """
    rows: int = 5
    cols: int = 5
    start: Tuple[int, int] = (0, 0)
    goals: Tuple[Tuple[int, int], ...] = ((4, 4),)
    pits: Tuple[Tuple[int, int], ...] = ()
    walls: Tuple[Tuple[int, int], ...] = ()
    slip_prob: float = 0.0
    step_reward: float = -1.0
    goal_reward: float = 10.0
    pit_reward: float = -10.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self) -> None:
        self.state = self.start
        self.terminal_set = set(self.goals) | set(self.pits)
        self.wall_set = set(self.walls)

    # ---------- Gym-ish API ----------
    def reset(self, seed: int | None = None) -> Tuple[int, int]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        # The simulator and transition_table share the same absorbing-state
        # convention: terminal-entry reward is paid once, then zero thereafter.
        if self.state in self.terminal_set:
            return self.state, 0.0, True, {}
        if self.slip_prob > 0 and self.rng.random() < self.slip_prob:
            action = int(self.rng.integers(0, 4))
        dr, dc = ACTIONS[action]
        r, c = self.state
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or (nr, nc) in self.wall_set:
            nr, nc = r, c  # bump into wall
        self.state = (nr, nc)
        if self.state in self.goals:
            return self.state, self.goal_reward, True, {}
        if self.state in self.pits:
            return self.state, self.pit_reward, True, {}
        return self.state, self.step_reward, False, {}

    # ---------- Tabular helpers (used in DP / MC / TD lectures) ----------
    @property
    def nS(self) -> int:
        return self.rows * self.cols

    @property
    def nA(self) -> int:
        return 4

    def s2idx(self, s: Tuple[int, int]) -> int:
        return s[0] * self.cols + s[1]

    def idx2s(self, idx: int) -> Tuple[int, int]:
        return (idx // self.cols, idx % self.cols)

    def transition_table(self) -> Dict[Tuple[int, int], List[Tuple[float, int, float, bool]]]:
        """Return P[s,a] = list of (prob, next_s_idx, reward, done) used by DP solvers."""
        P: Dict[Tuple[int, int], List[Tuple[float, int, float, bool]]] = {}
        for s_idx in range(self.nS):
            s = self.idx2s(s_idx)
            for a in range(self.nA):
                outcomes: Dict[Tuple[int, float, bool], float] = {}
                actions_dist = [(1.0 - self.slip_prob, a)]
                if self.slip_prob > 0:
                    for ap in range(self.nA):
                        actions_dist.append((self.slip_prob / 4.0, ap))
                if s in self.terminal_set:
                    P[(s_idx, a)] = [(1.0, s_idx, 0.0, True)]
                    continue
                for p, ap in actions_dist:
                    dr, dc = ACTIONS[ap]
                    nr, nc = s[0] + dr, s[1] + dc
                    if not (0 <= nr < self.rows and 0 <= nc < self.cols) or (nr, nc) in self.wall_set:
                        nr, nc = s
                    ns = (nr, nc)
                    if ns in self.goals:
                        r, done = self.goal_reward, True
                    elif ns in self.pits:
                        r, done = self.pit_reward, True
                    else:
                        r, done = self.step_reward, False
                    key = (self.s2idx(ns), r, done)
                    outcomes[key] = outcomes.get(key, 0.0) + p
                P[(s_idx, a)] = [(p, ns_idx, r, d) for (ns_idx, r, d), p in outcomes.items()]
        return P

    # ---------- ASCII render ----------
    def render(self, policy: np.ndarray | None = None, values: np.ndarray | None = None) -> str:
        out = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                s = (r, c)
                if s in self.goals:
                    cell = "  G  "
                elif s in self.pits:
                    cell = "  P  "
                elif s in self.wall_set:
                    cell = " ### "
                elif values is not None:
                    cell = f"{values[self.s2idx(s)]:+5.1f}"
                elif policy is not None:
                    cell = f"  {ACTION_NAMES[int(policy[self.s2idx(s)]) ]}  "
                else:
                    cell = "  .  "
                row.append(cell)
            out.append("|".join(row))
        return "\n".join(out)
