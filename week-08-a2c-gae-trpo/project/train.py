"""Train A2C+GAE on LunarLanderContinuous-v3.

Target metric: mean episode return trends strictly upward (final-window mean
exceeds initial-window mean by >= 30) within ~60s on CPU; with 5min on a GPU
you should see mean return >= 100.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from common.seed import seed_everything

from .agent import A2CAgent
from .solutions import compute_gae


@dataclass
class TrainOutput:
    episode_returns: list[float]
    update_losses: list[dict]
    final_mean_return: float


def _make_vector_env(env_id: str, num_envs: int, seed: int):
    import gymnasium as gym

    def thunk(i: int):
        def _f():
            e = gym.make(env_id)
            e.reset(seed=seed + i)
            return e
        return _f

    return gym.vector.SyncVectorEnv([thunk(i) for i in range(num_envs)])


def train(
    *,
    seed: int = 0,
    env_id: str = "LunarLanderContinuous-v3",
    num_envs: int = 8,
    rollout_steps: int = 128,
    n_updates: int = 200,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    entropy_coef: float = 0.01,
    target_window_lift: float = 30.0,
    verbose: bool = False,
) -> TrainOutput:
    seed_everything(seed)
    envs = _make_vector_env(env_id, num_envs, seed)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = A2CAgent.build(obs_dim, act_dim, lr=lr, gamma=gamma, lam=lam)
    agent.entropy_coef = entropy_coef

    obs, _ = envs.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)

    episode_returns: list[float] = []
    update_losses: list[dict] = []
    current_returns = np.zeros(num_envs, dtype=np.float64)

    for update in range(n_updates):
        all_obs = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        all_acts = np.zeros((rollout_steps, num_envs, act_dim), dtype=np.float32)
        all_rews = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        all_vals = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        all_dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        for t in range(rollout_steps):
            actions = np.zeros((num_envs, act_dim), dtype=np.float32)
            values = np.zeros(num_envs, dtype=np.float32)
            # We act per env with a single batched policy call.
            import torch
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                mu, log_std = agent.policy(obs_t)
                std = log_std.exp()
                eps = torch.randn_like(mu)
                a = (mu + std * eps).cpu().numpy()
                v = agent.value(obs_t).cpu().numpy()
            actions[:] = np.clip(a, -1.0, 1.0)
            values[:] = v

            nxt, r, term, trunc, _ = envs.step(actions)
            done = np.logical_or(term, trunc)

            all_obs[t] = obs
            all_acts[t] = actions
            all_rews[t] = r
            all_vals[t] = values
            all_dones[t] = done.astype(np.float32)

            current_returns += r
            for i in range(num_envs):
                if done[i]:
                    episode_returns.append(float(current_returns[i]))
                    current_returns[i] = 0.0

            obs = np.asarray(nxt, dtype=np.float32)

        # bootstrap value for incomplete final step
        import torch
        with torch.no_grad():
            last_vals = agent.value(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy()

        # Compute GAE per env, then stack.
        advs_list, rets_list = [], []
        for i in range(num_envs):
            adv, ret = compute_gae(
                rewards=all_rews[:, i], values=all_vals[:, i],
                dones=all_dones[:, i], last_value=float(last_vals[i]),
                gamma=gamma, lam=lam,
            )
            advs_list.append(adv); rets_list.append(ret)

        obs_flat = all_obs.reshape(-1, obs_dim)
        act_flat = all_acts.reshape(-1, act_dim)
        adv_flat = np.stack(advs_list, axis=1).reshape(-1)
        ret_flat = np.stack(rets_list, axis=1).reshape(-1)

        losses = agent.update(obs_flat, act_flat, adv_flat, ret_flat)
        update_losses.append(losses)

        if verbose and update % 20 == 0:
            window = episode_returns[-20:] if episode_returns else [0.0]
            print(f"upd {update:4d}  recent {np.mean(window):7.1f}  "
                  f"pi {losses['pi_loss']:+.3f}  v {losses['v_loss']:+.3f}  "
                  f"H {losses['entropy']:+.3f}")

    envs.close()

    final_mean = float(np.mean(episode_returns[-30:])) if len(episode_returns) >= 30 else 0.0
    return TrainOutput(
        episode_returns=episode_returns,
        update_losses=update_losses,
        final_mean_return=final_mean,
    )


if __name__ == "__main__":  # pragma: no cover
    out = train(verbose=True, n_updates=400)
    print(f"final mean return = {out.final_mean_return:.1f}")
