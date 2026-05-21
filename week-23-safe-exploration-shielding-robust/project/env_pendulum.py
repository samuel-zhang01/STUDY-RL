"""Tiny pendulum with domain-randomised pole length, dependency-free.

State: (theta, theta_dot). Theta = 0 is upright.
Action: scalar torque in [-1, +1].
Reward: -(theta^2 + 0.1 * theta_dot^2 + 0.001 * u^2). Maximised when balanced upright.

We use a simple semi-implicit Euler integrator with discretisation dt and clamp
the angular velocity. The episode length is fixed (continuing task with a
horizon).

Domain randomisation: at each reset we sample the pole length from a uniform
distribution `[length_low, length_high]`; the dynamics depend on length
through the gravitational term and the inertia, so different lengths give
different transition probabilities (a textbook robust-RL setup).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np


@dataclass
class PendulumDR:
    dt: float = 0.05
    g: float = 9.81
    mass: float = 1.0
    max_torque: float = 1.0
    max_speed: float = 8.0
    horizon: int = 100
    length_low: float = 0.5
    length_high: float = 1.5
    randomise: bool = True
    fixed_length: float = 1.0  # used when randomise=False
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.theta = float(self.rng.uniform(-np.pi, np.pi))
        self.theta_dot = 0.0
        if self.randomise:
            self.length = float(self.rng.uniform(self.length_low, self.length_high))
        else:
            self.length = self.fixed_length
        self.t = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        # Use [cos theta, sin theta, theta_dot] so the agent does not need to
        # know length explicitly. Tests of generalisation become meaningful.
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot], dtype=np.float32)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        u = float(np.clip(action, -self.max_torque, self.max_torque))
        # Semi-implicit Euler for an inverted pendulum.
        I = self.mass * self.length ** 2
        torque_grav = -self.mass * self.g * self.length * np.sin(self.theta)
        self.theta_dot = float(np.clip(
            self.theta_dot + (self.dt / I) * (torque_grav + u),
            -self.max_speed, self.max_speed,
        ))
        self.theta = (self.theta + self.dt * self.theta_dot + np.pi) % (2 * np.pi) - np.pi
        cost = self.theta ** 2 + 0.1 * self.theta_dot ** 2 + 0.001 * u ** 2
        self.t += 1
        done = self.t >= self.horizon
        return self._obs(), float(-cost), done, {"length": self.length}
