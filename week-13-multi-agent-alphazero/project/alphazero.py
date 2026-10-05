"""A small AlphaZero implementation for Tic-Tac-Toe (3x3).

Architecture:
- A small CNN (2 conv layers + 2-headed output: policy + value).
- MCTS with PUCT and Dirichlet noise at the root for exploration.
- Self-play training loop generating (state, pi_mcts, z) triples.

Designed to converge to near-optimal play on Tic-Tac-Toe in <2 minutes on a
single CPU. Knocks out the AlphaZero pipeline end-to-end on a problem small
enough to actually verify correctness (perfect play on TTT yields all draws).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tictactoe import TicTacToe


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class AZNetwork(nn.Module):
    def __init__(self, board_size: int = 3, n_actions: int = 9, hidden: int = 32):
        super().__init__()
        self.board_size = board_size
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(2, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        # Value head
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * board_size * board_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def predict(self, encoded_state: np.ndarray) -> Tuple[np.ndarray, float]:
        x = torch.tensor(encoded_state, dtype=torch.float32,
                         device=next(self.parameters()).device).unsqueeze(0)
        logits, value = self.forward(x)
        prior = F.softmax(logits, dim=-1).cpu().numpy()[0]
        return prior, float(value.item())


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    prior: float = 0.0
    value_sum: float = 0.0
    visits: int = 0
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    state: Optional[np.ndarray] = None  # the board *after* the parent's move
    terminal_value: Optional[float] = None  # if terminal

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0 or self.terminal_value is not None


class MCTS:
    def __init__(self, net: AZNetwork, n_simulations: int = 50, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
                 seed: int = 0) -> None:
        self.net = net
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self._rng = np.random.default_rng(seed)

    def run(self, root_state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        root = MCTSNode(state=root_state)
        self._expand(root, add_noise=add_noise)
        for _ in range(self.n_simulations):
            self._simulate(root)
        visits = np.zeros(TicTacToe.n_actions)
        for a, child in root.children.items():
            visits[a] = child.visits
        if visits.sum() == 0:
            # no legal moves explored -> uniform over legal
            legal = TicTacToe.legal_actions(root_state).astype(float)
            return legal / max(legal.sum(), 1.0)
        return visits / visits.sum()

    def _expand(self, node: MCTSNode, add_noise: bool = False) -> float:
        # Check terminal first.
        tv = TicTacToe.terminal_value(node.state)
        if tv is not None:
            node.terminal_value = tv
            return tv  # leaf value
        encoded = TicTacToe.encode(node.state)
        prior, value = self.net.predict(encoded)
        legal = TicTacToe.legal_actions(node.state)
        prior = prior * legal
        s = prior.sum()
        prior = prior / s if s > 0 else legal / legal.sum()
        if add_noise:
            mask = legal.astype(bool)
            # Draw only over legal actions so the mixture has the stated weight.
            noise = self._rng.dirichlet([self.dirichlet_alpha] * int(mask.sum()))
            prior[mask] = (1 - self.dirichlet_eps) * prior[mask] + self.dirichlet_eps * noise
            prior = prior * legal
            prior = prior / max(prior.sum(), 1e-9)
        for a, p in enumerate(prior):
            if legal[a] > 0:
                child_state = TicTacToe.apply_action(node.state, a)
                node.children[a] = MCTSNode(prior=float(p), state=child_state)
        return value

    def _simulate(self, root: MCTSNode) -> None:
        path = [root]
        node = root
        # Selection.
        while node.is_expanded() and node.terminal_value is None:
            a, node = self._select_child(node)
            path.append(node)
        # Expansion / evaluation.
        if node.terminal_value is not None:
            leaf_value = node.terminal_value
        else:
            leaf_value = self._expand(node)
        # Backup: alternate sign because we flip perspectives between plies.
        sign = 1.0  # leaf_value is already from the leaf player's perspective
        for n in reversed(path):
            n.visits += 1
            n.value_sum += sign * leaf_value
            sign *= -1.0

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        total_visits = sum(c.visits for c in node.children.values()) or 1
        best_a, best_score, best_c = -1, -1e18, None
        for a, c in node.children.items():
            u = self.c_puct * c.prior * math.sqrt(total_visits) / (1 + c.visits)
            score = -c.q + u   # negate q because child's q is from opponent's POV
            if score > best_score:
                best_score = score
                best_a = a
                best_c = c
        return best_a, best_c


# ---------------------------------------------------------------------------
# Self-play + training
# ---------------------------------------------------------------------------

def self_play_episode(
    net: AZNetwork,
    n_sim: int = 30,
    temperature_moves: int = 4,
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Play one game; return a list of (encoded_state, pi_mcts, z) tuples
    where z = +/-1/0 is the game result from the perspective of the player
    to move at that state."""
    rng = np.random.default_rng(seed)
    state = TicTacToe.initial_state()
    history: List[Tuple[np.ndarray, np.ndarray, int]] = []
    move_num = 0
    mcts = MCTS(net, n_simulations=n_sim, seed=seed)
    while True:
        pi = mcts.run(state, add_noise=True)
        # Symmetry augmentation: include 8 symmetries.
        for s_aug, pi_aug in TicTacToe.symmetries(state, pi):
            history.append((TicTacToe.encode(s_aug), pi_aug, 0))
        # Pick the actual move.
        if move_num < temperature_moves:
            a = int(rng.choice(len(pi), p=pi))
        else:
            a = int(np.argmax(pi))
        state = TicTacToe.apply_action(state, a)
        move_num += 1
        tv = TicTacToe.terminal_value(state)
        if tv is not None:
            break
    # Assign z to each (state, pi) in history. The terminal value tv is from
    # the perspective of the player who is now to move (which is the loser
    # if tv == -1, the draw player if tv == 0). Each prior state's player
    # alternates; we backfill z accordingly.
    out: List[Tuple[np.ndarray, np.ndarray, float]] = []
    n = len(history)
    # 8 augmentations per move -> each move has 8 entries in a row.
    n_moves = n // 8
    # The player to move at move m has flipped m times relative to the start.
    final_value = tv  # from final-state perspective
    # The player to move at the *next* state after move m is the same as
    # the player whose perspective tv reports; before move m it was the
    # opponent. Easier: the final move was made by some player; assign
    # z = ±1 alternating backward.
    z_seq = []
    z = -final_value  # the player who just moved sees -tv (since tv was from new-state POV)
    for m in range(n_moves - 1, -1, -1):
        z_seq.append(z)
        z = -z
    z_seq = z_seq[::-1]
    for m in range(n_moves):
        zm = z_seq[m]
        for k in range(8):
            i = m * 8 + k
            s_enc, pi_aug, _ = history[i]
            out.append((s_enc, pi_aug, float(zm)))
    return out


def train_step(
    net: AZNetwork,
    optimizer: torch.optim.Optimizer,
    batch: List[Tuple[np.ndarray, np.ndarray, float]],
    device: str = "cpu",
) -> Tuple[float, float]:
    if not batch:
        return 0.0, 0.0
    s = torch.tensor(np.stack([x[0] for x in batch]), dtype=torch.float32, device=device)
    pi = torch.tensor(np.stack([x[1] for x in batch]), dtype=torch.float32, device=device)
    z = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=device)
    logits, value = net(s)
    log_pi = F.log_softmax(logits, dim=-1)
    policy_loss = -(pi * log_pi).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, z)
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(policy_loss.item()), float(value_loss.item())


def evaluate_vs_random(
    net: AZNetwork,
    n_games: int = 20,
    n_sim: int = 20,
    seed: int = 0,
) -> Dict[str, float]:
    """Play against random; `winrate` is the score rate (a draw counts as 0.5)."""
    rng = np.random.default_rng(seed)
    wins = draws = losses = 0
    mcts = MCTS(net, n_simulations=n_sim, seed=seed, dirichlet_eps=0.0)
    for g in range(n_games):
        az_first = (g % 2 == 0)
        state = TicTacToe.initial_state()
        az_turn = az_first
        while True:
            tv = TicTacToe.terminal_value(state)
            if tv is not None:
                # The player to move would see tv from their perspective.
                # Was AZ's turn? Then AZ loses with prob -1 result, etc.
                result_for_az = tv if az_turn else -tv
                if result_for_az > 0:
                    wins += 1
                elif result_for_az < 0:
                    losses += 1
                else:
                    draws += 1
                break
            if az_turn:
                pi = mcts.run(state, add_noise=False)
                a = int(np.argmax(pi))
            else:
                legal = TicTacToe.legal_actions(state)
                a = int(rng.choice(np.where(legal > 0)[0]))
            state = TicTacToe.apply_action(state, a)
            az_turn = not az_turn
    return {"wins": wins, "draws": draws, "losses": losses,
            "winrate": (wins + 0.5 * draws) / n_games}
