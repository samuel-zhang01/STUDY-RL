"""Tabular IQL and CQL agents for the synthetic offline GridWorld.

Both agents are tabular teaching analogues without target networks,
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
    pi(a|s) prop mu_D(a|s) * exp( beta * (Q_hat(s,a) - V(s)) )

CQL penalty (discrete-action form):
    L_CQL = alpha * E_s [ log sum_a exp Q(s,a) - E_{a~pi_beta}[Q(s,a)] ].
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def expectile_loss_grad(u: np.ndarray, tau: float) -> np.ndarray:
    """Negative half-gradient of the expectile loss w.r.t. the predictor.

    d/dV  [ w * (Q - V)^2 ] = -2 w (Q - V).
    We return the residual r = w * (Q - V) so the caller can apply the step.
    """
    w = np.where(u < 0, 1.0 - tau, tau)
    return w * u


def _mean_update(table, indices, residuals, lr):
    """Average each parameter's residuals before one tabular update.

    Repeated samples in a minibatch must not multiply the learning rate.
    """
    sums = np.zeros_like(table)
    counts = np.zeros_like(table)
    np.add.at(sums, indices, residuals)
    np.add.at(counts, indices, 1)
    present = counts > 0
    table[present] += lr * sums[present] / counts[present]


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
    behavior_counts: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float32)
        self.V = np.zeros(self.nS, dtype=np.float32)

    # ---- updates ----
    def update_value(self, s_idx: np.ndarray, a: np.ndarray) -> float:
        q_sa = self.Q[s_idx, a]
        u = q_sa - self.V[s_idx]
        grad = expectile_loss_grad(u, self.tau)  # = w * (Q - V)
        # Bucket-average across same states (tabular update).
        _mean_update(self.V, s_idx, grad, self.lr_v)
        return float((np.where(u < 0, 1.0 - self.tau, self.tau) * u ** 2).mean())

    def update_q(self, s_idx: np.ndarray, a: np.ndarray, r: np.ndarray,
                 s2_idx: np.ndarray, done: np.ndarray) -> float:
        target = r + self.gamma * (1.0 - done) * self.V[s2_idx]
        td = target - self.Q[s_idx, a]
        _mean_update(self.Q, (s_idx, a), td, self.lr_q)
        return float((td ** 2).mean())

    # ---- policy extraction ----
    def policy(self) -> np.ndarray:
        """Greedy action under the fitted advantage-weighted BC policy."""
        return np.argmax(self.policy_probs(), axis=1)

    def fit_behavior(self, s_idx: np.ndarray, actions: np.ndarray) -> None:
        """Fit empirical action frequencies from the complete offline dataset."""
        self.behavior_counts = np.zeros_like(self.Q)
        np.add.at(self.behavior_counts, (s_idx, actions), 1)

    def policy_probs(self) -> np.ndarray:
        """Tabular weighted BC: pi(a|s) ∝ counts(s,a) exp(beta A(s,a)).

        Uniform behavior is assumed only before fit_behavior, or at states
        with no logged actions. Those unseen-state predictions have no support
        guarantee. Observed states assign zero mass to unobserved actions.
        """
        adv = self.Q - self.V[:, None]
        z = self.beta * adv
        if self.behavior_counts is not None:
            counts = self.behavior_counts.copy()
            counts[counts.sum(axis=1) == 0] = 1.0
            with np.errstate(divide="ignore"):
                z = z + np.log(counts)
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
        _mean_update(self.Q, (s_idx, a), td, self.lr_q)
        # CQL penalty: lower Q(s, .) toward Q(s, a_beta). Gradient w.r.t. Q[s, a']
        # of [ log sum_a' exp Q(s, a') - Q(s, a_beta) ] is softmax(Q(s, .))_a' - 1{a' == a_beta}.
        gradients = []
        for s, ai in zip(s_idx, a):
            qs = self.Q[s]
            p = np.exp(qs - qs.max()); p /= p.sum()
            grad_pen = p.copy()
            grad_pen[ai] -= 1.0
            gradients.append(grad_pen)
        _mean_update(self.Q, s_idx, -np.asarray(gradients), self.lr_q * self.alpha)
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
        _mean_update(self.Q, (s_idx, a), td, self.lr_q)
        return float((td ** 2).mean())

    def policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)
