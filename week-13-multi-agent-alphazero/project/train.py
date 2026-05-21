"""Week-13 mini-project orchestration: AlphaZero on Tic-Tac-Toe + MAPPO on
SimpleSpread.

Target metrics (default seed):
- AlphaZero: after ~20 iterations of self-play, winrate vs random opponent
  >= 0.85 (mostly wins + draws, near-zero losses).
- MAPPO: by end of 60 episodes, mean episode return improves by at least
  20% over the first 5 episodes.

Runtime: 1-2 minutes total on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch

from .alphazero import AZNetwork, MCTS, self_play_episode, train_step, evaluate_vs_random
from .mappo import train_mappo


def train_alphazero(
    seed: int = 0,
    n_iters: int = 12,
    n_episodes_per_iter: int = 6,
    n_simulations: int = 30,
    batch_size: int = 64,
    train_epochs: int = 6,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = AZNetwork().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    replay: List[Tuple[np.ndarray, np.ndarray, float]] = []
    winrates: List[float] = []
    p_losses: List[float] = []
    v_losses: List[float] = []
    for it in range(n_iters):
        # Self-play.
        for ep in range(n_episodes_per_iter):
            replay.extend(self_play_episode(net, n_sim=n_simulations,
                                              seed=seed + it * 100 + ep))
        # Cap replay size.
        if len(replay) > 4000:
            replay = replay[-4000:]
        # Training.
        for _ in range(train_epochs):
            idx = np.random.choice(len(replay), size=min(batch_size, len(replay)), replace=False)
            batch = [replay[i] for i in idx]
            pl, vl = train_step(net, opt, batch, device=device)
            p_losses.append(pl)
            v_losses.append(vl)
        wr = evaluate_vs_random(net, n_games=10, n_sim=20, seed=seed + 999 + it)
        winrates.append(wr["winrate"])
    return {
        "winrates": np.array(winrates),
        "policy_losses": np.array(p_losses),
        "value_losses": np.array(v_losses),
    }


def train(seed: int = 0, az_iters: int = 12,
          mappo_episodes: int = 60,
          coord_episodes: int = 200) -> Dict[str, Dict]:
    az = train_alphazero(seed=seed, n_iters=az_iters)
    mappo = train_mappo(seed=seed, n_episodes=mappo_episodes)
    from .coord_game import CoordGame
    coord_env = CoordGame(n_actions=3, n_agents=2, seed=seed)
    coord = train_mappo(env=coord_env, seed=seed, n_episodes=coord_episodes,
                         lr=1e-3, update_epochs=4)
    return {"alphazero": az, "mappo": mappo, "coord": coord}


if __name__ == "__main__":  # pragma: no cover
    out = train()
    print("AlphaZero winrates over iterations:", out["alphazero"]["winrates"])
    print("MAPPO final episode return:", out["mappo"]["returns"][-1])
