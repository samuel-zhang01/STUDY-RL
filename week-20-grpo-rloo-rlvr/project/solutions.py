"""Reference solutions for the workbook exercises.

Every workbook exercise calls ``exercise_N(...)`` from this module — the
notebook cell scaffolds raise ``NotImplementedError`` until the learner
overrides them, and the hidden test cell asserts that the *reference*
implementation here matches an expected output.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np
import torch

from .env import exact_match_reward, extract_final_number, sympy_equiv_reward
from .train import grpo_advantages, rloo_advantages


# ---------------------------------------------------------------------------
# Exercise 1 (warm-up). Verifier extraction.
# ---------------------------------------------------------------------------

def exercise_1(text: str) -> int | None:
    """Return the final integer in ``text`` (or None). Mirror exactly the
    GSM8K answer-extraction convention.
    """
    return extract_final_number(text)


# ---------------------------------------------------------------------------
# Exercise 2 (core). RLOO baseline from a group of rewards.
# ---------------------------------------------------------------------------

def exercise_2(rewards: Sequence[float]) -> np.ndarray:
    """Return RLOO advantages for a 1-D vector of group rewards."""
    r = np.asarray(rewards, dtype=np.float64)
    G = len(r)
    if G < 2:
        return r - r.mean()
    return (G / (G - 1)) * (r - r.mean())


# ---------------------------------------------------------------------------
# Exercise 3 (core). GRPO group-relative advantages.
# ---------------------------------------------------------------------------

def exercise_3(rewards: np.ndarray, debias: bool = False, eps: float = 1e-6) -> np.ndarray:
    """Return GRPO advantages over a (B, G) batch. If ``debias`` is True,
    use the Dr.GRPO variant (no std normalisation).
    """
    r = np.asarray(rewards, dtype=np.float64)
    m = r.mean(axis=-1, keepdims=True)
    if debias:
        return r - m
    s = r.std(axis=-1, keepdims=True)
    return (r - m) / (s + eps)


# ---------------------------------------------------------------------------
# Exercise 4 (core). Variance reduction: REINFORCE vs RLOO.
# ---------------------------------------------------------------------------

def exercise_4(rewards: np.ndarray) -> tuple[float, float]:
    """Compare raw-reward and within-group-centred coefficient variance.

    This diagnostic measures coefficients alone, without policy score
    vectors. The self-including empirical group mean depends on the sampled
    action and is not an independent, unbiased REINFORCE baseline. RLOO uses
    the other samples' mean, giving G/(G-1) times the centred coefficient.
    A variance-optimal policy-gradient baseline also depends on score-vector
    magnitudes. Lower coefficient variance does not prove lower gradient
    variance or unbiasedness.

    Returns ``(var_reinforce, var_centered)``.
    """
    r = np.atleast_2d(np.asarray(rewards, dtype=np.float64))
    var_reinforce = r.var()
    centered = r - r.mean(axis=-1, keepdims=True)
    var_centered = centered.var()
    return float(var_reinforce), float(var_centered)


# ---------------------------------------------------------------------------
# Exercise 5 (core). KL-divergence k3 estimator.
# ---------------------------------------------------------------------------

def exercise_5(logp: np.ndarray, logq: np.ndarray) -> float:
    """Unbiased low-variance KL estimate KL(p || q) given samples ~ p with
    log-probs ``logp`` under p and ``logq`` under q. Uses Schulman's k3:

        k3 = exp(log q - log p) - (log q - log p) - 1 .

    For paired samples (same x), the empirical mean of k3 is unbiased for
    KL(p || q) and always non-negative.
    """
    log_ratio = logq - logp
    return float(np.mean(np.exp(log_ratio) - log_ratio - 1.0))


# ---------------------------------------------------------------------------
# Exercise 6 (core). PPO-clipped surrogate on a single token.
# ---------------------------------------------------------------------------

def exercise_6(ratio: np.ndarray, adv: np.ndarray, clip_eps: float = 0.2) -> np.ndarray:
    """Token-level clipped surrogate loss (negative of the objective).

    Returns ``-min(r*A, clip(r, 1-eps, 1+eps)*A)`` elementwise.
    """
    r = np.asarray(ratio); a = np.asarray(adv)
    s1 = r * a
    s2 = np.clip(r, 1 - clip_eps, 1 + clip_eps) * a
    return -np.minimum(s1, s2)


# ---------------------------------------------------------------------------
# Exercise 7 (core, matching-themed). Process-reward aggregation.
# ---------------------------------------------------------------------------

def exercise_7(step_rewards: np.ndarray, mode: str = "min") -> np.ndarray:
    """Aggregate per-step process-reward scores into a single trajectory score.

    Mirrors the ATS analogue: each interview *stage* yields a partial score;
    the overall candidate score is some aggregator over stage scores.

    Modes:
      - ``"min"``    : the weakest stage governs (conservative gating).
      - ``"prod"``   : multiplicative — interprets stage scores as success
                       probabilities and assumes stage-conditional independence.
      - ``"mean"``   : average — standard PRM aggregation in PRM800K.
    """
    s = np.asarray(step_rewards, dtype=np.float64)
    if mode == "min":
        return s.min(axis=-1)
    if mode == "prod":
        return np.prod(s, axis=-1)
    if mode == "mean":
        return s.mean(axis=-1)
    raise ValueError(f"unknown mode {mode}")


# ---------------------------------------------------------------------------
# Exercise 8 (★★★ open-ended). Reward-shape ablation: returns a callable.
# ---------------------------------------------------------------------------

def exercise_8(completion: str, target: int, fmt_bonus: float = 0.1) -> float:
    """A *shaped* reward that gives partial credit for emitting *any* number,
    plus a bonus for ending the line cleanly with ``<eos>``.

    The point of the open-ended exercise is to compare this against the
    plain exact-match reward and observe the format-hacking failure mode
    (more numbers does not mean more right answers).
    """
    base = exact_match_reward(completion, target)
    pred = extract_final_number(completion)
    fmt = fmt_bonus if pred is not None else 0.0
    return base + fmt


__all__ = [
    "exercise_1", "exercise_2", "exercise_3", "exercise_4",
    "exercise_5", "exercise_6", "exercise_7", "exercise_8",
]
