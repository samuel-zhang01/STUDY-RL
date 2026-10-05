"""Reference solutions for the week-22 workbook exercises.

Each exercise function is named `exercise_N` and matches the contract of the
corresponding workbook scaffold cell.
"""
from __future__ import annotations

from typing import Tuple, List

import numpy as np

from .env import LavaGridWorld
from .agent import (
    ActorCritic,
    PPOLagrangian,
    PPOLagrangianConfig,
    RolloutBuffer,
    compute_gae,
)
from .train import train as _train


# -----------------------------------------------------------------------------
# Exercise 1 ★ — derive the Lagrangian dual update direction
# -----------------------------------------------------------------------------
def exercise_1(lam: float, eta: float, J_c: float, d: float) -> float:
    """One projected-ascent step on lambda for J_c - d.

    L(pi, lam) = J_r - lam * (J_c - d), with lam >= 0.
    The dL/d(lam) = -(J_c - d), but we MAXIMISE over pi and MINIMISE over lam, hence
    we ascend lam in the direction *(J_c - d)* (the gradient of the *outer* problem
    after the dual swap). Project onto [0, +inf).
    """
    return float(max(0.0, lam + eta * (J_c - d)))


# -----------------------------------------------------------------------------
# Exercise 2 ★ — exact-penalty (P3O) gradient sign
# -----------------------------------------------------------------------------
def exercise_2(J_c: float, d: float, kappa: float) -> float:
    """Subgradient of [J_c - d]_+ wrt the policy-implied J_c."""
    if J_c > d:
        return float(kappa)
    if J_c < d:
        return 0.0
    return float(kappa / 2.0)  # any in [0, kappa] is a subgradient at the kink


# -----------------------------------------------------------------------------
# Exercise 3 ★★ — GAE for the cost stream
# -----------------------------------------------------------------------------
def exercise_3(rews, vals, dones, gamma, lam, last_val=0.0):
    return compute_gae(rews, vals, dones, gamma, lam, last_val)


# -----------------------------------------------------------------------------
# Exercise 4 ★★ — FOCOPS closed-form non-parametric update (one-shot)
# -----------------------------------------------------------------------------
def exercise_4(pi_old: np.ndarray, A_r: np.ndarray, A_c: np.ndarray,
               lam: float, nu: float) -> np.ndarray:
    """FOCOPS non-parametric optimal update:
            pi*(a|s) propto pi_old(a|s) * exp( (1/lam) * (A_r - nu * A_c) ).
    Then row-normalise per state. Here `pi_old` is shape (S, A); A_r, A_c shape (S,A).
    """
    logits = np.log(np.maximum(pi_old, 1e-12)) + (A_r - nu * A_c) / max(lam, 1e-8)
    logits = logits - logits.max(axis=1, keepdims=True)  # for numerical stability
    new = np.exp(logits)
    return new / new.sum(axis=1, keepdims=True)


# -----------------------------------------------------------------------------
# Exercise 5 ★★ — CVaR of a return distribution
# -----------------------------------------------------------------------------
def exercise_5(samples: np.ndarray, alpha: float) -> float:
    """Empirical CVaR at level alpha: mean of the worst alpha fraction.

    Uses exactly alpha probability mass, with fractional weight at the cutoff.
    The conditional-tail shorthand needs care at atoms/tied quantiles.
    """
    samples = np.sort(np.asarray(samples, dtype=float))
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all() or not 0 < alpha <= 1:
        raise ValueError("Need finite nonempty 1D samples and alpha in (0,1]")
    mass = alpha * len(samples)
    whole = int(np.floor(mass))
    remainder = mass - whole
    tail_sum = samples[:whole].sum()
    if remainder > 0:
        tail_sum += remainder * samples[whole]
    return float(tail_sum / mass)


# -----------------------------------------------------------------------------
# Exercise 6 ★★ — train PPO-Lagrangian and check constraint satisfaction
# -----------------------------------------------------------------------------
def exercise_6(seed: int = 0, n_updates: int = 30, cost_limit: float = 1.5):
    h = _train(seed=seed, n_updates=n_updates, steps_per_update=512,
               cost_limit=cost_limit)
    final_cost = float(np.mean(h["cost"][-5:]))
    final_return = float(np.mean(h["return"][-5:]))
    return {"history": h, "final_cost": final_cost, "final_return": final_return}


# -----------------------------------------------------------------------------
# Exercise 7 ★★★ — comparison: unconstrained vs constrained policy
# -----------------------------------------------------------------------------
def exercise_7(seed: int = 0, n_updates: int = 30):
    """Train PPO with the cost limit effectively disabled (very large d) and with
    a strict d=0.5. Return the two histories so the learner can compare."""
    h_unsafe = _train(seed=seed, n_updates=n_updates, steps_per_update=512,
                      cost_limit=1e6)
    h_safe = _train(seed=seed, n_updates=n_updates, steps_per_update=512,
                    cost_limit=0.5)
    return {"unsafe": h_unsafe, "safe": h_safe}


# -----------------------------------------------------------------------------
# Exercise 8 ★★★ — PID-Lagrangian one-step update
# -----------------------------------------------------------------------------
def exercise_8(lam_prev: float, integral_prev: float, J_c_hist: List[float],
               d: float, Kp: float = 0.1, Ki: float = 0.05, Kd: float = 0.05) -> Tuple[float, float]:
    """One step of PID-Lagrangian on lambda (Stooke et al., 2020).

    error_t = J_c_t - d
    integral_t = integral_{t-1} + error_t
    derivative_t = J_c_t - J_c_{t-1}   (approx, for t >= 1)
    lam_t = max(0, Kp*error + Ki*integral + Kd*derivative).
    Returns (lam_new, integral_new).
    """
    J_c = J_c_hist[-1]
    err = J_c - d
    integral = integral_prev + err
    deriv = (J_c_hist[-1] - J_c_hist[-2]) if len(J_c_hist) >= 2 else 0.0
    lam = max(0.0, Kp * err + Ki * integral + Kd * deriv)
    return float(lam), float(integral)
