"""End-to-end training driver for the week-4 mini-project.

Runs SARSA vs Q-learning vs Expected SARSA on CliffWalking, and
Q-learning on Taxi. Aggregates over a few seeds for the figure.

    python -m week-04-td-sarsa-q.project.train  # not the call pattern;
    use `python week-04-td-sarsa-q/project/train.py` from repo root.

Target wall-clock: under 60 s on CPU.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from .agents import sarsa, q_learning, expected_sarsa, double_q_learning
from .env_utils import make_cliffwalking, make_taxi


@dataclass
class TrainResult:
    returns: Dict[str, np.ndarray]  # name -> (n_seeds, n_episodes)
    final_mean: Dict[str, float]


def train(seed: int = 0, n_seeds: int = 5, n_episodes: int = 500,
          taxi_episodes: int = 1500, alpha: float = 0.5,
          eps_decay: int = 200, do_taxi: bool = True, **kwargs) -> TrainResult:
    """Run the full bake-off."""
    cliff_returns: Dict[str, List[List[float]]] = {
        "SARSA": [], "Q-learning": [], "Expected SARSA": [], "Double Q": []}

    for s in range(n_seeds):
        env = make_cliffwalking()
        _, rs_sarsa = sarsa(env, n_episodes=n_episodes, alpha=alpha,
                            eps_decay=eps_decay, seed=seed + s)
        env = make_cliffwalking()
        _, rs_q = q_learning(env, n_episodes=n_episodes, alpha=alpha,
                             eps_decay=eps_decay, seed=seed + s)
        env = make_cliffwalking()
        _, rs_e = expected_sarsa(env, n_episodes=n_episodes, alpha=alpha,
                                 eps_decay=eps_decay, seed=seed + s)
        env = make_cliffwalking()
        _, rs_d = double_q_learning(env, n_episodes=n_episodes, alpha=alpha,
                                    eps_decay=eps_decay, seed=seed + s)
        cliff_returns["SARSA"].append(rs_sarsa)
        cliff_returns["Q-learning"].append(rs_q)
        cliff_returns["Expected SARSA"].append(rs_e)
        cliff_returns["Double Q"].append(rs_d)

    returns = {k: np.asarray(v) for k, v in cliff_returns.items()}
    final_mean = {k: float(v[:, -100:].mean()) for k, v in returns.items()}

    if do_taxi:
        taxi_curves = []
        for s in range(min(n_seeds, 3)):
            env = make_taxi()
            _, rs = q_learning(env, n_episodes=taxi_episodes, alpha=0.5,
                               gamma=0.99, eps_decay=600,
                               max_steps=200, seed=seed + s)
            taxi_curves.append(rs)
        returns["Taxi Q-learning"] = np.asarray(taxi_curves)
        final_mean["Taxi Q-learning"] = float(returns["Taxi Q-learning"][:, -100:].mean())

    return TrainResult(returns=returns, final_mean=final_mean)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=500)
    args = p.parse_args()
    res = train(n_seeds=args.seeds, n_episodes=args.episodes)
    print("Final mean returns (last 100 episodes):")
    for k, v in res.final_mean.items():
        print(f"  {k:<18s} {v:+8.2f}")
