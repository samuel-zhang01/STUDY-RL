"""A minimal cooperative multi-agent environment in the style of MPE's
'simple_spread': N agents in a 2D world must each cover one of N landmarks.

This is a pure-NumPy reimplementation (no pettingzoo / gymnasium dependency)
designed for CPU MAPPO training in <2 minutes. Observations are local; the
critic in MAPPO gets the centralised state.

Reward: each agent gets the same negative sum, over landmarks, of the
nearest-agent distance, minus 0.1 per colliding pair. Episodes last 25 steps.
This coverage objective is not a one-to-one assignment cost: an agent can
be nearest to several landmarks, so Hungarian matching solves a different
constrained objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class SimpleSpread:
    n_agents: int = 3
    world_size: float = 1.0
    episode_len: int = 25
    collision_radius: float = 0.1
    seed: int = 0
    dt: float = 0.1
    agent_speed: float = 0.15

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self.agents = np.zeros((self.n_agents, 2), dtype=np.float64)
        self.landmarks = np.zeros((self.n_agents, 2), dtype=np.float64)
        self.reset()

    def reset(self) -> List[np.ndarray]:
        self.t = 0
        self.agents = self.rng.uniform(-self.world_size, self.world_size, size=(self.n_agents, 2))
        self.landmarks = self.rng.uniform(-self.world_size, self.world_size, size=(self.n_agents, 2))
        return self._obs()

    def _obs(self) -> List[np.ndarray]:
        """Local observation per agent: [self_pos, all_landmark_relative_pos,
        all_other_agent_relative_pos]. Dim = 2 + 2N + 2(N-1)."""
        obs = []
        for i in range(self.n_agents):
            pos = self.agents[i]
            lm_rel = (self.landmarks - pos).flatten()
            ot_rel = np.delete(self.agents - pos, i, axis=0).flatten()
            obs.append(np.concatenate([pos, lm_rel, ot_rel]).astype(np.float32))
        return obs

    def global_state(self) -> np.ndarray:
        """Centralised state for the MAPPO critic: full positions of agents
        and landmarks."""
        return np.concatenate([self.agents.flatten(), self.landmarks.flatten()]).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return 2 + 2 * self.n_agents + 2 * (self.n_agents - 1)

    @property
    def state_dim(self) -> int:
        return 4 * self.n_agents  # agents + landmarks

    @property
    def n_actions(self) -> int:
        # 5 discrete actions: no-op, up, down, left, right.
        return 5

    _act_map = np.array([
        [0.0, 0.0],
        [0.0, 1.0],   # up
        [0.0, -1.0],  # down
        [-1.0, 0.0],  # left
        [1.0, 0.0],   # right
    ])

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        self.t += 1
        for i, a in enumerate(actions):
            dv = self._act_map[a] * self.agent_speed
            self.agents[i] = np.clip(self.agents[i] + dv, -self.world_size, self.world_size)
        # Team reward (canonical simple_spread): minus sum over landmarks of
        # min_dist to any agent. Each agent receives the same shared reward
        # (the cooperative challenge is to figure out which landmark to go to).
        team_reward = 0.0
        for lm in self.landmarks:
            d = np.min(np.linalg.norm(self.agents - lm, axis=1))
            team_reward -= float(d)
        # Collisions
        col = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                d = float(np.linalg.norm(self.agents[i] - self.agents[j]))
                if d < self.collision_radius:
                    col += 1
        team_reward -= 0.1 * col
        rewards = [team_reward for _ in range(self.n_agents)]
        done = self.t >= self.episode_len
        return self._obs(), rewards, done, {"collisions": col}
