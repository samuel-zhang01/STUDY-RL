"""Tabular Blackjack as in Sutton & Barto Example 5.1.

State: (player_sum (12..21), dealer_showing (1..10), usable_ace (0/1)).
Action: 0 = stick, 1 = hit.
Reward: +1 win, 0 draw, -1 loss; otherwise 0 per step.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


def _draw_card(rng: np.random.Generator) -> int:
    # Cards uniformly in {1,...,10}; face cards (J,Q,K) collapse to 10.
    c = int(rng.integers(1, 14))
    return min(c, 10)


def _hand_value(cards):
    s = sum(cards)
    usable = 1 in cards and s + 10 <= 21
    if usable:
        s += 10
    return s, int(usable)


@dataclass
class BlackjackEnv:
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Deal until player sum >= 12 (Sutton convention).
        self.player = [_draw_card(self.rng), _draw_card(self.rng)]
        self.dealer = [_draw_card(self.rng), _draw_card(self.rng)]
        while True:
            s, _ = _hand_value(self.player)
            if s >= 12:
                break
            self.player.append(_draw_card(self.rng))
        self.done = False
        return self._obs()

    def _obs(self) -> Tuple[int, int, int]:
        s, u = _hand_value(self.player)
        return (s, self.dealer[0], u)

    def step(self, a: int):
        assert not self.done
        if a == 1:  # hit
            self.player.append(_draw_card(self.rng))
            s, _ = _hand_value(self.player)
            if s > 21:
                self.done = True
                return self._obs(), -1.0, True, {}
            return self._obs(), 0.0, False, {}
        # stick -> dealer plays
        self.done = True
        while True:
            ds, _ = _hand_value(self.dealer)
            if ds >= 17:
                break
            self.dealer.append(_draw_card(self.rng))
        ds, _ = _hand_value(self.dealer)
        ps, _ = _hand_value(self.player)
        if ds > 21 or ps > ds:
            r = 1.0
        elif ps < ds:
            r = -1.0
        else:
            r = 0.0
        return self._obs(), r, True, {}

    @staticmethod
    def state_index(obs):
        s, d, u = obs
        # s in 12..21 -> 10 vals; d in 1..10 -> 10 vals; u in {0,1}.
        return (s - 12) * 20 + (d - 1) * 2 + u

    @staticmethod
    def nS() -> int:
        return 10 * 10 * 2

    @staticmethod
    def nA() -> int:
        return 2
