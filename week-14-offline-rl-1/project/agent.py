"""Tabular IQL and CQL agents for the synthetic offline GridWorld.

Both agents are exact (tabular) implementations of the canonical updates,
keeping the maths transparent. The IQL agent uses expectile regression on
the value V plus advantage-weighted policy extraction; the CQL agent uses
the standard CQL(H) penalty `log sum exp Q(s,.) - E_{pi_beta}[Q(s,a)]`
added to a Bellman-error objective on Q.

Algebraic notes
---------------
IQL value update (expectile tau):
    L_V(V) = E[ L_2^tau (Q_hat(s,a) - V(s)) ]
    where L_2^tau(u) = |tau - 1{u<0}| * u^2.

IQL policy extraction (AWR):
    pi(a|s) prop exp( beta * (Q_hat(s,a) - V(s)) )

CQL penalty (discrete-action form):
    L_CQL = alpha * E_s [ log sum_a exp Q(s,a) - E_{a~pi_beta}[Q(s,a)] ].
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def expectile_loss_grad(u: np.ndarray, tau: float) -> np.ndarray:
    """Gradient w.r.t. the predictor of L_2^tau(u) = |tau-1{u<0}| u^2 (per-sample).

    d/dV  [ w * (Q - V)^2 ] = -2 w (Q - V).
    We return the residual r = w * (Q - V) so the caller can apply the step.
    """
    w = np.where(u < 0, 1.0 - tau, tau)
    return w * u


@dataclass
class IQLAgent:
    """Tabular IQL with expectile value regression + AWR policy extraction."""
    nS: int = 25
    nA: int = 4
    gamma: float = 0.95
    tau: float = 0.8      # expectile (>0.5 = upper expectile)
    beta: float = 3.0     # AWR temperature
    lr_q: float = 0.1
    lr_v: float = 0.1
    Q: np.ndarray = field(init=False)
    V: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float32)
        self.V = np.zeros(self.nS, dtype=np.float32)

    # ---- updates ----
    def update_value(self, s_idx: np.ndarray, a: np.ndarray) -> float:
        q_sa = self.Q[s_idx, a]
        u = q_sa - self.V[s_idx]
        grad = expectile_loss_grad(u, self.tau)  # = w * (Q - V)
        # Bucket-average across same states (tabular update).
        for s, g in zip(s_idx, grad):
            self.V[s] += self.lr_v * g
        return float((np.where(u < 0, 1.0 - self.tau, self.tau) * u ** 2).mean())

    def update_q(self, s_idx: np.ndarray, a: np.ndarray, r: np.ndarray,
                 s2_idx: np.ndarray, done: np.ndarray) -> float:
        target = r + self.gamma * (1.0 - done) * self.V[s2_idx]
        td = target - self.Q[s_idx, a]
        for s, ai, t in zip(s_idx, a, td):
            self.Q[s, ai] += self.lr_q * t
        return float((td ** 2).mean())

    # ---- policy extraction ----
    def policy(self) -> np.ndarray:
        """Return deterministic greedy w.r.t. AWR weights = argmax_a Q(s,a)."""
        return np.argmax(self.Q, axis=1)

    def policy_probs(self) -> np.ndarray:
        """Soft AWR policy: pi(a|s) ~ exp(beta * (Q(s,a) - V(s)))."""
        adv = self.Q - self.V[:, None]
        z = self.beta * adv
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return p


@dataclass
class CQLAgent:
    """Tabular CQL with discrete-action conservative penalty."""
    nS: int = 25
    nA: int = 4
    gamma: float = 0.95
    alpha: float = 1.0
    lr_q: float = 0.1
    Q: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float32)

    def update(self, s_idx: np.ndarray, a: np.ndarray, r: np.ndarray,
               s2_idx: np.ndarray, done: np.ndarray) -> float:
        # Standard Bellman target (max over next actions: off-policy).
        q_next_max = self.Q[s2_idx].max(axis=1)
        target = r + self.gamma * (1.0 - done) * q_next_max
        td = target - self.Q[s_idx, a]
        # Bellman update.
        for s, ai, t in zip(s_idx, a, td):
            self.Q[s, ai] += self.lr_q * t
        # CQL penalty: lower Q(s, .) toward Q(s, a_beta). Gradient w.r.t. Q[s, a']
        # of [ log sum_a' exp Q(s, a') - Q(s, a_beta) ] is softmax(Q(s, .))_a' - 1{a' == a_beta}.
        for s, ai in zip(s_idx, a):
            qs = self.Q[s]
            p = np.exp(qs - qs.max()); p /= p.sum()
            grad_pen = p.copy()
            grad_pen[ai] -= 1.0
            self.Q[s] -= self.lr_q * self.alpha * grad_pen
        return float((td ** 2).mean())

    def policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)


@dataclass
class NaiveQAgent:
    """Plain off-policy Q-learning on the offline dataset (extrapolation-error baseline)."""
    nS: int = 25
    nA: int = 4
    gamma: float = 0.95
    lr_q: float = 0.1
    Q: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float32)

    def update(self, s_idx, a, r, s2_idx, done):
        q_next_max = self.Q[s2_idx].max(axis=1)
        target = r + self.gamma * (1.0 - done) * q_next_max
        td = target - self.Q[s_idx, a]
        for s, ai, t in zip(s_idx, a, td):
            self.Q[s, ai] += self.lr_q * t
        return float((td ** 2).mean())

    def policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)
