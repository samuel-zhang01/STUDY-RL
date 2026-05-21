"""Probability calibration utilities: Platt scaling, isotonic regression,
Beta calibration (Kull, Filho, Flach 2017). Plus reliability-diagram, ECE,
Brier score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        # Logistic regression on a single feature (the score).
        lr = LogisticRegression()
        lr.fit(scores.reshape(-1, 1), labels.astype(int))
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        return self

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * scores + self.b)))


# ---------------------------------------------------------------------------
# Isotonic
# ---------------------------------------------------------------------------

@dataclass
class IsotonicCalibrator:
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(scores, labels.astype(float))
        return self

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(scores), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Beta calibration (Kull, Filho, Flach 2017)
# ---------------------------------------------------------------------------

@dataclass
class BetaCalibrator:
    a: float = 1.0
    b: float = 1.0
    c: float = 0.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "BetaCalibrator":
        """Beta calibration is logistic regression on
        (log(s), -log(1-s)) features; coefficients are constrained positive."""
        s = np.clip(scores, 1e-6, 1 - 1e-6)
        X = np.column_stack([np.log(s), -np.log(1 - s)])
        lr = LogisticRegression()
        lr.fit(X, labels.astype(int))
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.coef_[0, 1])
        self.c = float(lr.intercept_[0])
        return self

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        s = np.clip(scores, 1e-6, 1 - 1e-6)
        logit = self.a * np.log(s) - self.b * np.log(1 - s) + self.c
        return 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (Naeini et al. 2015)."""
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        e += (mask.sum() / n) * abs(acc - conf)
    return float(e)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centres, empirical_accuracy, bin_count)."""
    bins = np.linspace(0, 1, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins, dtype=int)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        cnt[i] = int(mask.sum())
        acc[i] = labels[mask].mean() if cnt[i] > 0 else np.nan
    return centres, acc, cnt
