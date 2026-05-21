"""Train REINFORCE (+ optional baseline) on CartPole-v1.

Target metric: mean episode return on CartPole-v1 >= 195 within 500 episodes,
running in under 60s on CPU when use_baseline=True.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from common.seed import seed_everything

from .agent import REINFORCEAgent


@dataclass
class TrainOutput:
    returns: list[float]
    estimator_variances: list[float]
    final_mean_return: float
    solved_at: int | None


def _rollout_one_episode(env, agent: REINFORCEAgent, max_steps: int = 500):
    obs, _ = env.reset()
    O, A, R = [], [], []
    for _ in range(max_steps):
        a = agent.act(obs)
        nxt, r, term, trunc, _ = env.step(a)
        O.append(np.asarray(obs, dtype=np.float32))
        A.append(int(a))
        R.append(float(r))
        obs = nxt
        if term or trunc:
            break
    return (
        np.stack(O, axis=0),
        np.asarray(A, dtype=np.int64),
        np.asarray(R, dtype=np.float32),
    )


def train(
    *,
    seed: int = 0,
    n_episodes: int = 500,
    env_id: str = "CartPole-v1",
    use_baseline: bool = True,
    pi_lr: float = 5e-3,
    v_lr: float = 1e-2,
    gamma: float = 0.99,
    target_return: float = 195.0,
    log_every: int = 25,
    verbose: bool = False,
) -> TrainOutput:
    import gymnasium as gym

    seed_everything(seed)
    env = gym.make(env_id)
    env.reset(seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    agent = REINFORCEAgent.build(
        obs_dim, n_actions,
        use_baseline=use_baseline, pi_lr=pi_lr, v_lr=v_lr, gamma=gamma,
    )

    returns: list[float] = []
    variances: list[float] = []
    window: deque[float] = deque(maxlen=20)
    solved_at: int | None = None

    for ep in range(n_episodes):
        obs, acts, rews = _rollout_one_episode(env, agent)
        ep_return = float(rews.sum())
        returns.append(ep_return)
        window.append(ep_return)

        var = agent.estimator_variance(obs, acts, rews)
        variances.append(var)
        agent.update(obs, acts, rews)

        if len(window) == window.maxlen and solved_at is None:
            if np.mean(window) >= target_return:
                solved_at = ep

        if verbose and ep % log_every == 0:
            print(f"ep {ep:4d}  return {ep_return:6.1f}  "
                  f"window {np.mean(window):6.1f}  var {var:8.2f}")

    env.close()
    return TrainOutput(
        returns=returns,
        estimator_variances=variances,
        final_mean_return=float(np.mean(list(window))),
        solved_at=solved_at,
    )


if __name__ == "__main__":  # pragma: no cover
    out = train(verbose=True)
    print(f"final window mean = {out.final_mean_return:.1f}  solved_at={out.solved_at}")
