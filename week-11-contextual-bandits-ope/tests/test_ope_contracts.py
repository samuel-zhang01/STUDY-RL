"""Invalid-evidence and exact-small-fixture checks, independent of training wins."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from project.ope import (ips, snips, doubly_robust, direct_method, switch_dr,
    ips_variance, independent_logged_sample, RidgeRewardModel)
from project.env import SyntheticLeadBandit


def test_invalid_logging_probability_and_missing_overlap_are_explicit():
    pi = lambda x: np.array([0., 1.])
    for probability in [0, -.1, 1.1, np.nan]:
        with pytest.raises(ValueError):
            ips([(np.array([1.]), 0, 1., probability)], pi)
    rows = [(np.array([1.]), 0, 1., .5)]
    assert ips(rows, pi) == 0
    with pytest.raises(ValueError, match="undefined"):
        snips(rows, pi)
    with pytest.raises(ValueError):
        ips([], pi)
    with pytest.raises(ValueError):
        ips_variance(rows, pi)


def test_estimator_arithmetic_and_full_dr_limit():
    rows = [(np.array([1.]), 0, 1., .5), (np.array([1.]), 1, 0., .5)]
    pi = lambda x: np.array([.75, .25])
    q = lambda x: np.array([.6, .4])
    assert ips(rows, pi) == .75
    assert snips(rows, pi) == .75
    assert direct_method(rows, pi, q) == pytest.approx(.55)
    assert doubly_robust(rows, pi, q) == pytest.approx(.75)
    assert switch_dr(rows, pi, q, 2) == doubly_robust(rows, pi, q)
    assert switch_dr(rows, pi, q, 0) == direct_method(rows, pi, q)


def test_nuisance_sample_preserves_mechanisms_without_consuming_evaluation_rng():
    env, twin = SyntheticLeadBandit(seed=2), SyntheticLeadBandit(seed=2)
    rows = independent_logged_sample(env, 30, seed=900)
    np.testing.assert_array_equal(env.sample_context(), twin.sample_context())
    assert len(rows) == 30 and all(row[3] == 1/env.K for row in rows)
    model = RidgeRewardModel(env.K, env.d).fit(rows)
    assert model(env.sample_context()).shape == (env.K,)
