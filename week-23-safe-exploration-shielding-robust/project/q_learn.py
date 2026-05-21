"""Tabular Q-learning, optionally wrapped by the DFA shield."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from .env_grid import DangerGridWorld
from .shield import DFAShield, build_no_lava_visit_shield


@dataclass
class QLearner:
    n_states: int
    n_actions: int = 4
    alpha: float = 0.3
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 300
    seed: int = 0

    def __post_init__(self):
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)

    def eps(self, ep: int) -> float:
        return float(max(self.eps_end,
                         self.eps_start
                         - (self.eps_start - self.eps_end) * ep / max(1, self.eps_decay_episodes)))

    def preferences(self, s: int) -> List[int]:
        """Action ordering from best to worst (Q value), used by the shield."""
        return list(np.argsort(-self.Q[s]))

    def act(self, s: int, ep: int, shield: Optional[DFAShield] = None) -> int:
        if self.rng.random() < self.eps(ep):
            # epsilon-greedy explore — but still ask the shield to filter
            order = list(self.rng.permutation(self.n_actions))
        else:
            order = self.preferences(s)
        if shield is not None:
            return shield.filter(s, order)
        return order[0]

    def update(self, s: int, a: int, r: float, ns: int, done: bool) -> None:
        target = r + (0.0 if done else self.gamma * float(self.Q[ns].max()))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


def _run_one(env: DangerGridWorld, agent: QLearner, ep: int,
             shield: Optional[DFAShield]) -> Tuple[float, int, bool]:
    s = env.reset()
    R = 0.0
    unsafe_visits = 0
    done = False
    reached_goal = False
    while not done:
        a = agent.act(s, ep, shield=shield)
        ns, r, done, info = env.step(a)
        agent.update(s, a, r, ns, done)
        R += r
        if info.get("unsafe", False):
            unsafe_visits += 1
        if done and r == env.goal_reward:
            reached_goal = True
        s = ns
    return R, unsafe_visits, reached_goal


def train_shielded(n_episodes: int = 500, seed: int = 0,
                   env: Optional[DangerGridWorld] = None) -> Dict[str, list]:
    env = env or DangerGridWorld()
    env.reset(seed=seed)
    agent = QLearner(n_states=env.n_states, n_actions=env.n_actions, seed=seed)
    shield = build_no_lava_visit_shield(env)
    h = {"return": [], "unsafe": [], "goal": [], "n_blocked": []}
    for ep in range(n_episodes):
        R, u, g = _run_one(env, agent, ep, shield)
        h["return"].append(R); h["unsafe"].append(u); h["goal"].append(int(g))
        h["n_blocked"].append(shield.n_blocked)
    h["Q"] = agent.Q.copy()
    h["total_unsafe"] = int(sum(h["unsafe"]))
    h["shield_blocked"] = int(shield.n_blocked)
    return h


def train_unshielded(n_episodes: int = 500, seed: int = 0,
                     env: Optional[DangerGridWorld] = None) -> Dict[str, list]:
    env = env or DangerGridWorld()
    env.reset(seed=seed)
    agent = QLearner(n_states=env.n_states, n_actions=env.n_actions, seed=seed)
    h = {"return": [], "unsafe": [], "goal": []}
    for ep in range(n_episodes):
        R, u, g = _run_one(env, agent, ep, shield=None)
        h["return"].append(R); h["unsafe"].append(u); h["goal"].append(int(g))
    h["Q"] = agent.Q.copy()
    h["total_unsafe"] = int(sum(h["unsafe"]))
    return h
