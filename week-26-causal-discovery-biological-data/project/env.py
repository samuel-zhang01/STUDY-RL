"""Synthetic patient measurements and a separate confounded-treatment fixture.

Continuous values are generic biomarkers, not realistic RNA counts. Each patient
has repeated technical measurements sharing a target; patients are the split
and bootstrap units. Environment 2 changes a nuisance proxy's measurement rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PatientData:
    features: np.ndarray
    outcome: np.ndarray
    patient_ids: np.ndarray
    environments: np.ndarray
    latent_features: np.ndarray


def make_patient_data(seed=0, n_patients=240, repeats=3, missing_rate=0.3,
                      noise=0.8, target_shift=0.0):
    """Make three environments with globally unique IDs and repeated measurements.

    Y=1.5*Z+0.4*C+U_y except the optional target_shift in test environment.
    Proxy S=a_e*Y+U_s has a_e=(2,0,-2); neither S nor missingness causes Y.
    Missingness is MCAR here. target_shift deliberately violates target invariance.
    """
    if not isinstance(n_patients, (int, np.integer)) or n_patients < 4:
        raise ValueError("n_patients must be an integer >=4")
    if not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not np.isfinite(missing_rate) or not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be in [0,1)")
    if not np.isfinite(noise) or noise < 0 or not np.isfinite(target_shift):
        raise ValueError("noise must be nonnegative and target_shift finite")
    rng = np.random.default_rng(seed)
    xs, ys, ids, envs, latents = [], [], [], [], []
    for e, proxy_coefficient in enumerate((2.0, 0.0, -2.0)):
        z = rng.normal(0.4*e, 1.0, n_patients)
        c = rng.normal(0, 1.0, n_patients)
        y = 1.5*z + 0.4*c + rng.normal(0, 0.35, n_patients)
        if e == 2:
            y = y + target_shift*z
        latent = np.column_stack([z, c])
        rep_z, rep_c, rep_y = (np.repeat(a, repeats) for a in (z, c, y))
        x = np.column_stack([
            rep_z + rng.normal(0, noise, len(rep_z)),
            rep_c + rng.normal(0, 0.15, len(rep_c)),
            proxy_coefficient*rep_y + rng.normal(0, 0.3, len(rep_y)),
        ])
        x[rng.random(x.shape) < missing_rate] = np.nan
        xs.append(x); ys.append(rep_y)
        ids.append(np.repeat(np.arange(n_patients)+e*n_patients, repeats))
        envs.append(np.full(len(rep_y), e, dtype=int))
        latents.append(np.repeat(latent, repeats, axis=0))
    return PatientData(np.vstack(xs), np.concatenate(ys), np.concatenate(ids),
                       np.concatenate(envs), np.vstack(latents))


def make_confounded_data(seed=0, n=12000):
    """Measured binary confounder; true ATE=1.0 for every synthetic unit."""
    if not isinstance(n, (int, np.integer)) or n < 4:
        raise ValueError("n must be an integer >=4")
    rng = np.random.default_rng(seed)
    c = rng.binomial(1, 0.5, n)
    t = rng.binomial(1, np.where(c == 1, 0.8, 0.2))
    y0 = 2.0*c + rng.normal(0, 0.2, n)
    y1 = y0 + 1.0
    return {"confounder": c, "treatment": t, "outcome": np.where(t, y1, y0),
            "y0": y0, "y1": y1, "true_ate": 1.0}
