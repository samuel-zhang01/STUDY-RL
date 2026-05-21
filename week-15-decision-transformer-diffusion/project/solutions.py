"""Reference solutions for the Week 15 workbook."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from .data import build_dt_dataset, sample_minibatch
from .model import DecisionTransformer
from .train import train, _rollout


def exercise_1_returns_to_go(rewards: np.ndarray) -> np.ndarray:
    """rtg[k] = sum_{j>=k} rewards[j]; vectorised reverse cumsum."""
    return np.cumsum(rewards[::-1])[::-1].astype(np.float32)


def exercise_2_causal_mask(L: int) -> np.ndarray:
    """Upper-triangular boolean mask: True at positions to be *masked* (not attended)."""
    return np.triu(np.ones((L, L), dtype=bool), k=1)


def exercise_3_sample_minibatch(seed: int = 0, n_eps: int = 30, K: int = 8, B: int = 4):
    eps, _, nS, nA = build_dt_dataset(n_eps, K, seed)
    rng = np.random.default_rng(seed)
    return sample_minibatch(eps, K, B, rng)


def exercise_4_train_dt(seed: int = 0, n_steps: int = 600):
    return train(seed=seed, n_steps=n_steps)


def exercise_5_conditional_generation(out: dict) -> tuple[float, float]:
    """Return (high_target_return, low_target_return) from training output dict."""
    return float(out["mean_return_high_target"]), float(out["mean_return_low_target"])


def exercise_6_action_distribution(out: dict, seed: int = 0) -> np.ndarray:
    """Compute the empirical action distribution at state s_idx=0 for the trained DT,
    conditioned on a high target return."""
    model: DecisionTransformer = out["model"]
    model.eval()
    K = model.context_len; nS = model.state_dim; nA = model.n_actions
    S_hist = np.zeros((1, K, nS), dtype=np.float32)
    S_hist[0, -1, 0] = 1.0
    A_hist = np.zeros((1, K), dtype=np.int64)
    R_hist = np.zeros((1, K), dtype=np.float32); R_hist[0, -1] = 3.0
    T_hist = np.zeros((1, K), dtype=np.int64)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(R_hist[..., None]).to(device),
                       torch.from_numpy(S_hist).to(device),
                       torch.from_numpy(A_hist).to(device),
                       torch.from_numpy(T_hist).to(device))
    p = F.softmax(logits[0, -1], dim=-1).cpu().numpy()
    model.train()
    return p


def exercise_7_diffusion_noising(x0: np.ndarray, alpha_bar: float,
                                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Forward diffusion: x_t = sqrt(alpha_bar) x0 + sqrt(1-alpha_bar) eps."""
    rng = rng or np.random.default_rng(0)
    eps = rng.standard_normal(x0.shape).astype(np.float32)
    return (np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * eps).astype(np.float32)


def exercise_8_dt_career_trajectory_demo(seed: int = 0) -> dict:
    """Meaningful objective: DT-style sequence modelling for "career trajectories".

    Each episode is a candidate trajectory; we treat returns-to-go as
    "expected future career outcome" and ask: given a high target outcome,
    what next stage does the model recommend?
    """
    out = train(seed=seed, n_steps=400)
    p_high = exercise_6_action_distribution(out, seed=seed)
    return {"action_dist_high_target": p_high,
            "mean_return_high": out["mean_return_high_target"],
            "mean_return_low": out["mean_return_low_target"]}
