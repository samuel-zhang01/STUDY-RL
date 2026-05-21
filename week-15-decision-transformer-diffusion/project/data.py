"""Build a Decision-Transformer-style dataset from the synthetic GridWorld logger.

We turn the per-step transitions from week 14 into (return-to-go, state, action)
triples grouped by episode, ready to be windowed into context-length K segments.
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Reuse week 14's dataset collector. The week-14 folder has a hyphen so we load
# the module by file path rather than `import`.
import importlib.util
_W14_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                              "week-14-offline-rl-1", "project", "env.py"))
_spec = importlib.util.spec_from_file_location("_w14_env", _W14_ENV_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)  # type: ignore
collect_logged_dataset = _mod.collect_logged_dataset
make_env = _mod.make_env


def build_dt_dataset(n_episodes: int = 200, context_len: int = 8, seed: int = 0):
    """Returns (episodes, context_len) where episodes is a list of dicts.

    Each episode dict has keys:
      states:     (T, nS) one-hot
      actions:    (T,) int64
      returns_to_go: (T,) float32  -- sum of future rewards (undiscounted)
      timesteps: (T,) int64
    """
    raw = collect_logged_dataset(n_episodes=n_episodes, seed=seed)
    s = raw["s"]; a = raw["a"]; r = raw["r"]; done = raw["done"].astype(bool)
    episodes = []
    start = 0
    for t in range(len(a)):
        if done[t]:
            ep_s = s[start:t + 1]
            ep_a = a[start:t + 1]
            ep_r = r[start:t + 1]
            # Undiscounted return-to-go: rtg[k] = sum_{j>=k} r[j]
            rtg = np.zeros_like(ep_r)
            acc = 0.0
            for k in range(len(ep_r) - 1, -1, -1):
                acc += float(ep_r[k]); rtg[k] = acc
            episodes.append({
                "states": ep_s.astype(np.float32),
                "actions": ep_a.astype(np.int64),
                "returns_to_go": rtg.astype(np.float32),
                "timesteps": np.arange(len(ep_a), dtype=np.int64),
            })
            start = t + 1
    return episodes, context_len, raw["nS"], raw["nA"]


def sample_minibatch(episodes, context_len: int, batch_size: int,
                     rng: np.random.Generator):
    """Sample fixed-length context windows (left-padded with zeros)."""
    B = batch_size; K = context_len
    nS = episodes[0]["states"].shape[1]
    S = np.zeros((B, K, nS), dtype=np.float32)
    A = np.zeros((B, K), dtype=np.int64)
    R = np.zeros((B, K), dtype=np.float32)
    T = np.zeros((B, K), dtype=np.int64)
    M = np.zeros((B, K), dtype=np.float32)  # mask
    for i in range(B):
        ep = episodes[int(rng.integers(0, len(episodes)))]
        L = len(ep["actions"])
        if L >= K:
            j = int(rng.integers(0, L - K + 1))
            S[i] = ep["states"][j:j + K]
            A[i] = ep["actions"][j:j + K]
            R[i] = ep["returns_to_go"][j:j + K]
            T[i] = ep["timesteps"][j:j + K]
            M[i] = 1.0
        else:
            S[i, -L:] = ep["states"]
            A[i, -L:] = ep["actions"]
            R[i, -L:] = ep["returns_to_go"]
            T[i, -L:] = ep["timesteps"]
            M[i, -L:] = 1.0
    return S, A, R, T, M
