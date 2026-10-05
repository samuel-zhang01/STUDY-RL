"""Reference solutions for the eight workbook exercises.

Every data transformation takes explicit arrays. No function loads evaluation
data or silently adds support for unobserved treatment/confounder combinations.
"""
from __future__ import annotations

import numpy as np


def _vector(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return values


def _treatment_outcome(treatment, outcome):
    t = _vector(treatment, "treatment")
    y = _vector(outcome, "outcome")
    if t.shape != y.shape or not np.isin(t, [0, 1]).all():
        raise ValueError("treatment must be binary and match outcome")
    if len(np.unique(t)) != 2:
        raise ValueError("both treatment groups are required")
    return t, y


def exercise_1_naive_ate(treatment, outcome):
    """Observed mean difference; causal only under additional assumptions."""
    t, y = _treatment_outcome(treatment, outcome)
    return float(y[t == 1].mean() - y[t == 0].mean())


def exercise_2_adjusted_ate(treatment, outcome, confounder):
    """Empirical standardization over categorical pre-treatment strata.

    Identification additionally requires consistency, conditional exchangeability,
    and population positivity. Nonempty empirical cells do not prove these.
    """
    t, y = _treatment_outcome(treatment, outcome)
    c = _vector(confounder, "confounder")
    if c.shape != t.shape:
        raise ValueError("confounder must match treatment")
    total = 0.0
    for level in np.unique(c):
        rows = c == level
        treated = rows & (t == 1)
        untreated = rows & (t == 0)
        if not treated.any() or not untreated.any():
            raise ValueError(f"empirical overlap missing in stratum {level}")
        total += rows.mean() * (y[treated].mean() - y[untreated].mean())
    return float(total)


def exercise_3_equivalent_covariances(rho=0.8):
    """Two opposite linear Gaussian SCMs induce the same covariance.

    X=U, Y=rho*X+sqrt(1-rho**2)*V and its reversed construction
    have independent standard-normal disturbances U,V and unit marginals.
    """
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("rho must be finite with abs(rho)<1")
    forward = np.array([[1.0, rho], [rho, rho**2 + (1-rho**2)]])
    reverse = np.array([[rho**2 + (1-rho**2), rho], [rho, 1.0]])
    return forward, reverse


def exercise_4_patient_split(patient_ids, environments, test_environment=2,
                             validation_fraction=0.3, seed=0):
    """Hold out one environment, then split other environments by patient.

    All rows for a patient stay together. IDs must identify patients globally,
    rather than being reset to 0 for each hospital. Cross-environment patients
    are rejected so their allocation requires an explicit study design.
    """
    ids, env = np.asarray(patient_ids), np.asarray(environments)
    if ids.ndim != 1 or env.shape != ids.shape or not len(ids):
        raise ValueError("patient_ids and environments must be equal nonempty vectors")
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(env.dtype, np.integer):
        raise ValueError("this fixture requires integer patient and environment IDs")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    for pid in np.unique(ids):
        if len(np.unique(env[ids == pid])) != 1:
            raise ValueError("a patient occurs in multiple environments")
    if test_environment not in env:
        raise ValueError("held-out environment is absent")
    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    for level in np.unique(env):
        if level == test_environment:
            continue
        patients = rng.permutation(np.unique(ids[env == level]))
        if len(patients) < 2:
            raise ValueError("each training environment needs at least two patients")
        n_val = min(len(patients)-1, max(1, round(len(patients)*validation_fraction)))
        val_ids.extend(patients[:n_val])
        train_ids.extend(patients[n_val:])
    if not train_ids:
        raise ValueError("at least one training environment is required")
    return {
        "train": np.flatnonzero(np.isin(ids, train_ids)),
        "validation": np.flatnonzero(np.isin(ids, val_ids)),
        "test": np.flatnonzero(env == test_environment),
    }


def exercise_5_impute_fit(features):
    """Fit observed-only column means; an all-missing column gets zero.

    Returns imputed training data and fitted means. NaN denotes missingness;
    biological zeros remain observations. Never call this on held-out data.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or np.isinf(x).any():
        raise ValueError("features must be a nonempty matrix without infinity")
    observed = ~np.isnan(x)
    count = observed.sum(axis=0)
    means = np.divide(np.nansum(x, axis=0), count,
                      out=np.zeros(x.shape[1]), where=count > 0)
    return np.where(observed, x, means), means


def exercise_6_residual_spread(residuals, environments):
    """Descriptive mean/variance spread, NOT an ICP test or p-value."""
    r, e = _vector(residuals, "residuals"), np.asarray(environments)
    if e.shape != r.shape:
        raise ValueError("environments must match residuals")
    means = np.array([r[e == level].mean() for level in np.unique(e)])
    variances = np.array([r[e == level].var() for level in np.unique(e)])
    return {"mean_spread": float(np.ptp(means)),
            "variance_spread": float(np.ptp(variances))}


def exercise_7_task_randomize(features, column=2, probability=1.0, seed=0):
    """Training-only nuisance randomization retaining the same target labels.

    The nuisance column is specified by knowledge of the synthetic generator.
    This is an augmentation heuristic, not generative biology or causal discovery.
    Resampled values preserve a training marginal only approximately when p<1.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or not x.shape[0] or not np.isfinite(x).all():
        raise ValueError("features must be a nonempty finite matrix")
    if not isinstance(column, (int, np.integer)) or not 0 <= column < x.shape[1]:
        raise ValueError("column is out of bounds")
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    rng = np.random.default_rng(seed)
    out = x.copy()
    chosen = rng.random(len(x)) < probability
    permuted = rng.permutation(x[:, column])
    out[chosen, column] = permuted[chosen]
    return out


def exercise_8_matching_priority_value(scores, potential_improvements, budget=2):
    """Average simulator improvement for a fixed top-k candidate/lead shortlist.

    potential_improvements are fixture ground truth, not observed-only outcomes.
    Using observed outcomes for unchosen options would not identify policy value.
    Stable sorting makes ties deterministic. The budget is a positive integer.
    """
    scores = _vector(scores, "scores")
    gains = _vector(potential_improvements, "potential_improvements")
    if gains.shape != scores.shape:
        raise ValueError("potential_improvements must match scores")
    if not isinstance(budget, (int, np.integer)) or not 1 <= budget <= len(scores):
        raise ValueError("budget must be an integer in [1, number of candidates]")
    selected = np.argsort(-scores, kind="stable")[:budget]
    return {"selected": selected, "value": float(gains[selected].mean())}
