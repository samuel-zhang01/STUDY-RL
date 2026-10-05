"""Training-only preprocessing and deterministic ridge gradient descent."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .solutions import exercise_5_impute_fit


@dataclass(frozen=True)
class Preprocessor:
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, features):
        filled, means = exercise_5_impute_fit(features)
        scales = filled.std(axis=0)
        scales = np.where(scales < 1e-12, 1.0, scales)
        return cls(means, scales)

    def transform(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.means) or np.isinf(x).any():
            raise ValueError("features must match fitted columns and have no infinity")
        return (np.where(np.isnan(x), self.means, x)-self.means)/self.scales


@dataclass(frozen=True)
class RidgeModel:
    weights: np.ndarray
    history: np.ndarray

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] + 1 != len(self.weights) or not np.isfinite(x).all():
            raise ValueError("finite features must match fitted model")
        return np.column_stack([np.ones(len(x)), x]) @ self.weights


def fit_ridge(features, outcome, steps=250, ridge=0.02):
    """Full-batch gradient descent with step 1/L on MSE + ridge*||w||^2.

    L=2*lambda_max(D.T D/n + ridge*P). The intercept is unpenalized.
    Returns loss history including the initial loss, with no test evaluation.
    """
    x, y = np.asarray(features, dtype=float), np.asarray(outcome, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not len(y):
        raise ValueError("features/outcome have incompatible or empty shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs must be finite")
    if not isinstance(steps, (int, np.integer)) or steps < 1 or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("steps must be positive integer and ridge nonnegative")
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0.0
    gram = design.T @ design / len(y) + ridge*penalty
    rhs = design.T @ y / len(y)
    rate = 1.0/(2.0*np.linalg.eigvalsh(gram).max())
    w = np.zeros(design.shape[1])
    history = []
    for iteration in range(steps+1):
        history.append(float(np.mean((design@w-y)**2)+ridge*(w@penalty@w)))
        if iteration < steps:
            w -= rate*2.0*(gram@w-rhs)
    return RidgeModel(w, np.asarray(history))


def patient_mse(outcome, prediction, patient_ids):
    """Average within-patient error, then average patients with equal weight."""
    y, p, ids = np.asarray(outcome), np.asarray(prediction), np.asarray(patient_ids)
    if y.ndim != 1 or p.shape != y.shape or ids.shape != y.shape or not len(y):
        raise ValueError("nonempty outcome, prediction and ids must match")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("outcome/prediction must be finite")
    squared = (y-p)**2
    return float(np.mean([squared[ids == pid].mean() for pid in np.unique(ids)]))


def patient_bootstrap_mse_interval(outcome, prediction, patient_ids, seed=0, draws=400):
    """95% percentile interval by resampling whole patients after model selection.

    Conditional on this fitted model and a representative IID patient sample;
    excludes model-fit/selection and biological-discovery uncertainty.
    """
    patient_mse(outcome, prediction, patient_ids)  # Validate shapes and finiteness.
    if not isinstance(draws, (int, np.integer)) or draws < 2:
        raise ValueError("draws must be an integer >=2")
    y, p, ids = np.asarray(outcome), np.asarray(prediction), np.asarray(patient_ids)
    risks = np.array([np.mean((y[ids == pid]-p[ids == pid])**2) for pid in np.unique(ids)])
    rng = np.random.default_rng(seed)
    means = rng.choice(risks, size=(draws, len(risks)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()
