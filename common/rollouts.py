"""Episode boundaries and GAE for vector rollouts.

Termination removes the value bootstrap. Truncation keeps that bootstrap
but stops the advantage trace from crossing into a freshly reset episode.
"""
from __future__ import annotations

import inspect
import numpy as np


def sync_vector_env(env_fns):
    """Use same-step autoreset, including Gymnasium 1.0's legacy default."""
    import gymnasium as gym

    kwargs = {}
    if "autoreset_mode" in inspect.signature(gym.vector.SyncVectorEnv).parameters:
        kwargs["autoreset_mode"] = gym.vector.AutoresetMode.SAME_STEP
    return gym.vector.SyncVectorEnv(env_fns, **kwargs)


def transition_next_observations(next_obs, infos, dones):
    """Replace reset observations with final observations for bootstrapping.

    Same-step autoreset returns the next episode's initial observation as
    next_obs and supplies the just-finished episode's observation in info.
    Gymnasium 1.0 uses final_observation; later versions use final_obs.
    """
    result = np.asarray(next_obs).copy()
    key = next((key for key in ("final_obs", "final_observation") if key in infos), None)
    if key is not None:
        mask = np.asarray(infos.get("_" + key, dones), dtype=bool)
        for i in np.flatnonzero(mask):
            result[i] = infos[key][i]
    return result


def generalized_advantage_estimate(rewards, values, dones, last_value,
                                   gamma=0.99, lam=0.95, *,
                                   next_values=None, terminals=None):
    """Return (advantages, lambda-return targets) for one environment.

    dones marks any episode boundary. With terminals and next_values,
    truncations bootstrap from their final observation while the trace ends.
    Omitting these optional arrays retains the simple episodic convention
    where every done is a true terminal and next values follow the sequence.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    terminals = dones if terminals is None else np.asarray(terminals, dtype=np.float64)
    if next_values is None:
        next_values = np.concatenate([values[1:], [last_value]]) if len(values) else values.copy()
    else:
        next_values = np.asarray(next_values, dtype=np.float64)
    if any(arr.shape != rewards.shape for arr in (values, dones, terminals, next_values)):
        raise ValueError("All GAE arrays must have the same one-dimensional shape")
    if rewards.ndim != 1:
        raise ValueError("GAE expects one environment's one-dimensional rollout")
    advantages = np.zeros_like(rewards)
    last_adv = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * (1.0 - terminals[t]) * next_values[t] - values[t]
        last_adv = delta + gamma * lam * (1.0 - dones[t]) * last_adv
        advantages[t] = last_adv
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)
