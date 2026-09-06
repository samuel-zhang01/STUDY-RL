"""Minimal Tic-Tac-Toe (3x3) game environment used for the AlphaZero smoke test.

Implements a clean state representation, legal-move generator, terminal test,
and a canonical-form board for symmetry. Pure NumPy, no deps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


BOARD_SIZE = 3
WIN_LENGTH = 3


@dataclass
class TicTacToe:
    @staticmethod
    def initial_state() -> np.ndarray:
        # board: shape (3, 3); +1 = current player; -1 = opponent; 0 = empty.
        return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

    @staticmethod
    def legal_actions(state: np.ndarray) -> np.ndarray:
        # Returns binary mask of shape (9,) indicating legal positions.
        return (state.reshape(-1) == 0).astype(np.int8)

    @staticmethod
    def apply_action(state: np.ndarray, action: int) -> np.ndarray:
        # The current player's move (+1), then flip board so the next player
        # sees themselves as +1.
        s = state.copy()
        r, c = divmod(action, BOARD_SIZE)
        assert s[r, c] == 0, f"illegal action {action} on\n{state}"
        s[r, c] = 1
        return -s  # canonical form: next player is always +1

    @staticmethod
    def has_winning_line(state: np.ndarray, player: int) -> bool:
        # Check rows, cols, diagonals for `player` having WIN_LENGTH in a row.
        b = (state == player).astype(np.int8)
        # rows / cols
        for i in range(BOARD_SIZE):
            if b[i].sum() == WIN_LENGTH:
                return True
            if b[:, i].sum() == WIN_LENGTH:
                return True
        # diagonals
        if np.trace(b) == WIN_LENGTH:
            return True
        if np.trace(np.fliplr(b)) == WIN_LENGTH:
            return True
        return False

    @staticmethod
    def terminal_value(state: np.ndarray) -> Optional[float]:
        """Return -1 if the player who just moved (encoded as -1) won,
        +1 if the current player has a winning line (unreachable in normal
        play), 0 for a draw, or None while the game is ongoing.

        Convention: after apply_action() flips the board, the player who
        just moved sits at -1 in the new state. So:
        - if -1 has 3-in-a-row -> previous player won -> value from current
          player's perspective is -1.
        - draw if no empties and no winner -> 0.
        """
        if TicTacToe.has_winning_line(state, -1):
            return -1.0
        if TicTacToe.has_winning_line(state, 1):
            return 1.0  # cannot happen after a flip but safety net
        if (state == 0).sum() == 0:
            return 0.0
        return None

    @staticmethod
    def encode(state: np.ndarray) -> np.ndarray:
        """Two-plane encoding for the network: (player_pieces, opponent_pieces)."""
        return np.stack([(state == 1).astype(np.float32),
                         (state == -1).astype(np.float32)], axis=0)

    @staticmethod
    def symmetries(state: np.ndarray, pi: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Augment with 8 board symmetries (rotations + reflections)."""
        out = []
        pi2 = pi.reshape(BOARD_SIZE, BOARD_SIZE)
        for k in range(4):
            for flip in (False, True):
                s = np.rot90(state, k)
                p = np.rot90(pi2, k)
                if flip:
                    s = np.fliplr(s)
                    p = np.fliplr(p)
                out.append((s.copy(), p.reshape(-1).copy()))
        return out

    @staticmethod
    def display(state: np.ndarray) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = ["|".join(symbols[int(c)] for c in row) for row in state]
        return "\n".join(rows)

    n_actions = BOARD_SIZE * BOARD_SIZE
    obs_shape = (2, BOARD_SIZE, BOARD_SIZE)
