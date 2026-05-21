"""A 2-agent matrix coordination game for MAPPO smoke-testing.

Each step, both agents pick one of N actions; reward is +1 if they pick
the same action and 0 otherwise. The game has a *focal point* (Schelling
1960): action 0 is the safe default. Optimal policy is for both agents to
always pick the same action; the symmetric equilibrium is all-N pure
strategies; MAPPO should converge to one of them via shared training.

This is the canonical 'matrix MARL test' from Foerster et al. NeurIPS 2017.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class CoordGame:
    n_actions: int = 3
    n_agents: int = 2
    episode_len: int = 5
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self.last_actions = np.zeros(self.n_agents, dtype=np.int64)

    def reset(self) -> List[np.ndarray]:
        self.t = 0
        self.last_actions = np.zeros(self.n_agents, dtype=np.int64)
        return self._obs()

    def _obs(self) -> List[np.ndarray]:
        # Each agent sees a 1-hot of the other agent's last action.
        obs = []
        for i in range(self.n_agents):
            v = np.zeros(self.n_actions + 1, dtype=np.float32)
            other = self.last_actions[(i + 1) % self.n_agents]
            v[other] = 1.0
            v[-1] = self.t / self.episode_len
            obs.append(v)
        return obs

    def global_state(self) -> np.ndarray:
        v = np.zeros(self.n_actions * self.n_agents + 1, dtype=np.float32)
        for i, a in enumerate(self.last_actions):
            v[i * self.n_actions + a] = 1.0
        v[-1] = self.t / self.episode_len
        return v

    @property
    def obs_dim(self) -> int:
        return self.n_actions + 1

    @property
    def state_dim(self) -> int:
        return self.n_actions * self.n_agents + 1

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        self.t += 1
        match = int(actions[0] == actions[1])
        rewards = [float(match)] * self.n_agents
        self.last_actions = np.array(actions, dtype=np.int64)
        done = self.t >= self.episode_len
        return self._obs(), rewards, done, {"match": match}
