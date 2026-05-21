"""Replay buffer + prioritized replay (used from week 5 onward)."""
from __future__ import annotations
from typing import NamedTuple
import numpy as np


class Transition(NamedTuple):
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    """A fixed-size circular replay buffer for vector-state transitions."""

    def __init__(self, capacity: int, state_dim: int, action_dtype=np.int64,
                 seed: int = 0) -> None:
        self.capacity = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=action_dtype)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done) -> None:
        i = self.idx
        self.s[i] = s; self.a[i] = a; self.r[i] = r; self.s2[i] = s2; self.done[i] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized experience replay (Schaul 2016) — sum-tree free
    naive version (fine for small/medium buffers used in the course)."""

    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6,
                 beta: float = 0.4, eps: float = 1e-6, seed: int = 0) -> None:
        super().__init__(capacity, state_dim, seed=seed)
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.max_priority = 1.0

    def push(self, s, a, r, s2, done) -> None:
        self.priorities[self.idx] = self.max_priority
        super().push(s, a, r, s2, done)

    def sample(self, batch_size: int):
        probs = self.priorities[: self.size] ** self.alpha
        probs /= probs.sum()
        idx = self.rng.choice(self.size, size=batch_size, p=probs)
        weights = (self.size * probs[idx]) ** (-self.beta)
        weights /= weights.max()
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx],
                idx, weights.astype(np.float32))

    def update_priorities(self, idx, td_errors) -> None:
        td = np.abs(td_errors) + self.eps
        self.priorities[idx] = td
        self.max_priority = max(self.max_priority, float(td.max()))
