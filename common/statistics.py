"""Small statistical calculations with explicit units and assumptions.

These helpers never infer independence from an array shape. Callers must define
the independent unit before constructing the array. A seed-level interval, for
example, does not measure uncertainty about a different task or reward model.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import t


def _vector(values, name="values", minimum=1):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < minimum or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite 1D array with at least {minimum} entries")
    return x


def _confidence(confidence):
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return (1 - confidence) / 2


def independent_mean_interval(values, confidence=.95, unit="independent unit", method="t"):
    """Student-t interval for an IID-unit mean; exact under normal units.

    This is an approximation for other finite-variance populations. It is not
    appropriate for overlapping rolling windows, adaptively collected returns,
    heavy tails without adequate moments, or repeated rows from one patient.
    No interval is fabricated from a single observation.
    """
    x = _vector(values, minimum=2)
    alpha = _confidence(confidence)
    if method != "t" or not isinstance(unit, str) or not unit.strip():
        raise ValueError("method must be 't' and unit must describe the independent unit")
    estimate = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(len(x))
    half = float(t.ppf(1-alpha, len(x)-1) * se)
    return dict(estimate=estimate, low=estimate-half, high=estimate+half,
                n=len(x), unit=unit, sd=sd, standard_error=float(se),
                confidence=confidence, interval="Student-t interval for the unit mean")


def paired_bootstrap_difference(left, right, confidence=.95, draws=2000, seed=0,
                                unit="paired independent unit"):
    """Percentile interval for mean(left-right), resampling intact pairs.

    A pair must represent two measurements on the same independent unit. The
    procedure is conditional on this evaluation design and is approximate; a
    tiny number of pairs cannot establish reliable nominal coverage.
    """
    a, b = _vector(left, "left", 2), _vector(right, "right", 2)
    if a.shape != b.shape:
        raise ValueError("left and right must have identical paired shapes")
    alpha = _confidence(confidence)
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 100:
        raise ValueError("draws must be an integer >=100")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("unit must describe the independent pairs")
    delta = a-b
    rng = np.random.default_rng(seed)
    # Chunk draws so teaching examples cannot allocate draws*n unboundedly.
    means = np.concatenate([rng.choice(delta, size=(min(256, draws-start), len(delta)),
                                      replace=True).mean(axis=1)
                            for start in range(0, draws, 256)])
    lo, hi = np.quantile(means, [alpha, 1-alpha])
    return dict(estimate=float(delta.mean()), low=float(lo), high=float(hi), n=len(delta),
                unit=unit, confidence=confidence, draws=int(draws), seed=int(seed),
                interval="paired percentile bootstrap for mean(left-right)")


def importance_diagnostics(target_prob, logged_prob):
    """Validate logged-action probabilities and summarize observed weights.

    These are probabilities of the *observed action*, not full distributions.
    Valid logged rows need strictly positive behavior probabilities. Observed
    overlap does not prove population positivity or remove confounding. ESS is
    a weight-concentration diagnostic, not an interval or independent count.
    """
    target = _vector(target_prob, "target_prob")
    logged = _vector(logged_prob, "logged_prob")
    if target.shape != logged.shape or (target < 0).any() or (target > 1).any():
        raise ValueError("target and logged probabilities must match; target must be in [0,1]")
    if (logged <= 0).any() or (logged > 1).any():
        raise ValueError("every observed action needs a logged probability in (0,1]")
    with np.errstate(over="ignore", invalid="ignore"):
        weights = target/logged
    if not np.isfinite(weights).all():
        raise ValueError("importance weights overflow; inspect overlap before estimating")
    maximum = float(weights.max())
    if maximum == 0:
        raise ValueError("no observed target overlap: SNIPS and weight ESS are undefined")
    # ESS is invariant to scale; scaling avoids overflow in squared weights.
    scaled = weights/maximum
    ess = float(scaled.sum()**2 / np.dot(scaled, scaled))
    return dict(weights=weights, n=len(weights), effective_sample_size=ess,
                ess_fraction=ess/len(weights), max_weight=maximum,
                nonzero_weights=int(np.count_nonzero(weights)),
                population_support="not established from observed rows")


def importance_weights(target_prob, logged_prob):
    """Validated observed weights, permitting all-zero weights for ordinary IPS.

    Unlike SNIPS, ordinary IPS has a defined empirical value when every observed
    weight is zero. That value alone does not establish target-policy coverage.
    """
    target = _vector(target_prob, "target_prob")
    logged = _vector(logged_prob, "logged_prob")
    if target.shape != logged.shape or (target < 0).any() or (target > 1).any():
        raise ValueError("target and logged probabilities must match; target must be in [0,1]")
    if (logged <= 0).any() or (logged > 1).any():
        raise ValueError("every observed action needs a logged probability in (0,1]")
    with np.errstate(over="ignore", invalid="ignore"):
        weights = target/logged
    if not np.isfinite(weights).all():
        raise ValueError("importance weights overflow")
    return weights


def empirical_cvar(samples, alpha, tail="lower"):
    """Mean of exactly alpha empirical probability mass, including cutoff ties.

    Lower-tail return CVaR measures poor returns; upper-tail cost CVaR measures
    large costs. This descriptive empirical tail mean is not a risk guarantee.
    """
    x = np.sort(_vector(samples))
    if not np.isfinite(alpha) or not 0 < alpha <= 1 or tail not in {"lower", "upper"}:
        raise ValueError("alpha must be in (0,1] and tail lower or upper")
    if tail == "upper":
        x = x[::-1]
    mass = alpha*len(x)
    whole = int(np.floor(mass))
    fraction = mass-whole
    total = x[:whole].sum()
    if fraction > 0:
        total += fraction*x[whole]
    return float(total/mass)


def cluster_means(values, cluster_ids):
    """One equally weighted mean per cluster, before independent-unit inference."""
    x = _vector(values)
    ids = np.asarray(cluster_ids)
    if ids.ndim != 1 or ids.shape != x.shape:
        raise ValueError("cluster_ids must match values")
    if ids.dtype.kind in "fc" and not np.isfinite(ids).all():
        raise ValueError("cluster_ids must be observed")
    return np.array([x[ids == key].mean() for key in np.unique(ids)])


def bernoulli_zero_event_upper(n, confidence=.95):
    """Exact one-sided binomial upper bound after 0 events in n IID trials.

    A zero-violation rollout with dependent transitions is not n IID trials.
    For that case define independent episodes and an episode-level event first.
    """
    _confidence(confidence)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive number of independent Bernoulli trials")
    return float(-np.expm1(np.log1p(-confidence)/n))
