"""Reference solutions for week-13 workbook exercises."""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch

from .tictactoe import TicTacToe
from .alphazero import AZNetwork, MCTS, evaluate_vs_random, self_play_episode, train_step
from .simple_spread import SimpleSpread
from .coord_game import CoordGame
from .mappo import train_mappo, Actor, CentralisedCritic, compute_gae


# E1 (star) - PUCT formula.
def exercise_1_puct(prior: float, parent_visits: int, child_visits: int,
                     q: float, c_puct: float = 1.5) -> float:
    """PUCT: -q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)."""
    return -q + c_puct * prior * np.sqrt(parent_visits) / (1 + child_visits)


# E2 (star) - TicTacToe terminal value.
def exercise_2_terminal_value(state: np.ndarray):
    return TicTacToe.terminal_value(state)


# E3 (star) - Symmetries.
def exercise_3_n_symmetries(state: np.ndarray, pi: np.ndarray) -> int:
    return len(TicTacToe.symmetries(state, pi))


# E4 (2-star) - One MCTS run and inspect visit distribution.
def exercise_4_mcts_visits(n_simulations: int = 30, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    net = AZNetwork()
    mcts = MCTS(net, n_simulations=n_simulations, seed=seed)
    state = TicTacToe.initial_state()
    return mcts.run(state, add_noise=False)


# E5 (2-star) - AlphaZero short training and winrate vs random.
def exercise_5_alphazero_winrate(n_iters: int = 8, seed: int = 0) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = AZNetwork()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    replay = []
    for it in range(n_iters):
        for ep in range(4):
            replay.extend(self_play_episode(net, n_sim=20, seed=seed + it * 100 + ep))
        replay = replay[-2000:]
        for _ in range(4):
            idx = np.random.choice(len(replay), size=min(64, len(replay)), replace=False)
            batch = [replay[i] for i in idx]
            train_step(net, opt, batch)
    return evaluate_vs_random(net, n_games=10, n_sim=15, seed=seed + 999)["winrate"]


# E6 (2-star) - GAE.
def exercise_6_gae(rewards, values, gamma: float = 0.95, lam: float = 0.95):
    return compute_gae(np.asarray(rewards, dtype=np.float32),
                       np.asarray(values, dtype=np.float32),
                       gamma=gamma, lam=lam)


# E7 (2-star) - Coordination game.
def exercise_7_coord_game_return(n_episodes: int = 200, seed: int = 0) -> float:
    env = CoordGame(n_actions=3, n_agents=2, seed=seed)
    out = train_mappo(env=env, n_episodes=n_episodes, seed=seed, lr=1e-3,
                      update_epochs=4)
    # Mean of last 30 episodes.
    return float(np.mean(out["returns"][-30:]))


# E8 (2-star) - Simple-spread learning curve.
def exercise_8_simple_spread(n_episodes: int = 60, seed: int = 0) -> Tuple[float, float]:
    out = train_mappo(n_episodes=n_episodes, seed=seed, lr=3e-4)
    r = out["returns"]
    # Mean(first 10), mean(last 10) - cooperative-coverage learning is slow.
    return float(np.mean(r[:10])), float(np.mean(r[-10:]))


# E9 (3-star, open-ended) - Self-play vs.\ fixed AlphaZero opponent.
def exercise_9_self_play_vs_fixed(seed: int = 0) -> Dict[str, float]:
    """Compare a self-play-trained net to an untrained net, both with MCTS.

    This does not train a second agent against a fixed opponent or isolate
    the effect of self-play. Small stochastic evaluations need not improve.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    net_a = AZNetwork()
    opt_a = torch.optim.Adam(net_a.parameters(), lr=2e-3)
    # Self-play
    replay = []
    for it in range(5):
        for ep in range(3):
            replay.extend(self_play_episode(net_a, n_sim=20, seed=seed + it * 100 + ep))
        replay = replay[-2000:]
        for _ in range(4):
            idx = np.random.choice(len(replay), size=min(64, len(replay)), replace=False)
            train_step(net_a, opt_a, [replay[i] for i in idx])
    wr_a = evaluate_vs_random(net_a, n_games=20, n_sim=15, seed=seed + 999)
    # Also evaluate a fresh untrained net for control.
    wr_random = evaluate_vs_random(AZNetwork(), n_games=20, n_sim=15, seed=seed + 999)
    return {"self_play_winrate": wr_a["winrate"], "untrained_winrate": wr_random["winrate"]}


# E10 (3-star, open-ended) - AlphaZero rating climb.
def exercise_10_az_rating_climb(n_iters: int = 8, seed: int = 0) -> np.ndarray:
    """Track winrate vs.\\ random across training iterations to plot the ELO-style climb."""
    from .train import train_alphazero
    out = train_alphazero(seed=seed, n_iters=n_iters, n_episodes_per_iter=4,
                           n_simulations=15)
    return out["winrates"]
