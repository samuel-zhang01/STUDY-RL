"""Synthetic *recruitment delayed-feedback* MDP.

A candidate proceeds through a 4-stage funnel: screening -> interview ->
offer -> outcome. At each stage the agent picks one of two actions: ADVANCE
or REJECT. Only the terminal outcome (hire / no-hire) carries reward (+1 or
0); intermediate rewards are zero. The state encodes a coarse signal of fit
that becomes more informative as the funnel progresses, modelling the
recruiter's growing knowledge.

This exercises off-policy MC because we want to evaluate a *new* shortlisting
policy from logs collected under a different (e.g., uniform) behaviour.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class RecruitmentMDP:
    n_stages: int = 4
    seed: int = 0
    p_hire_advance: Tuple[float, ...] = (0.1, 0.2, 0.45, 0.7)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        # fit-score per candidate, drawn at reset; only revealed gradually.
        self.true_fit = 0.0
        self.stage = 0
        self.done = False
        # We model state as (stage, noisy fit bucket {0,1,2}).
        self.nS = self.n_stages * 3
        self.nA = 2  # 0 = reject, 1 = advance

    def reset(self, seed: int | None = None) -> Tuple[int, int]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.true_fit = float(self.rng.beta(2, 2))  # bias toward middle
        self.stage = 0
        self.done = False
        return self._obs()

    def _signal(self) -> int:
        # The signal becomes sharper as stage increases.
        # At stage s, the bucket is a noisy quantisation of true_fit.
        noise = self.rng.normal(0, 0.4 / (1 + self.stage))
        v = max(0.0, min(1.0, self.true_fit + noise))
        if v < 0.33:
            return 0
        if v < 0.66:
            return 1
        return 2

    def _obs(self) -> Tuple[int, int]:
        return (self.stage, self._signal())

    def state_index(self, obs) -> int:
        s, b = obs
        return s * 3 + b

    def step(self, a: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        assert not self.done
        if a == 0:  # reject
            self.done = True
            return self._obs(), 0.0, True, {}
        # advance: terminate at last stage with probability tied to fit
        if self.stage == self.n_stages - 1:
            self.done = True
            p = self.p_hire_advance[-1] * (0.5 + self.true_fit)
            r = 1.0 if self.rng.random() < min(0.95, p) else 0.0
            return self._obs(), r, True, {}
        self.stage += 1
        return self._obs(), 0.0, False, {}
