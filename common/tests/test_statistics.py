"""Numerical contracts and failure cases for educational inference helpers."""
import numpy as np
import pytest
from common.statistics import (independent_mean_interval, paired_bootstrap_difference,
    importance_diagnostics, importance_weights, empirical_cvar, cluster_means,
    bernoulli_zero_event_upper)


def test_mean_interval_uses_independent_units_and_sample_sd():
    result = independent_mean_interval([1, 2, 3, 4], unit="training seed")
    assert result["estimate"] == 2.5 and result["n"] == 4
    assert result["unit"] == "training seed"
    np.testing.assert_allclose(result["sd"], np.sqrt(5/3))
    np.testing.assert_allclose(result["high"], 4.554260256760879)
    with pytest.raises(ValueError):
        independent_mean_interval([1])
    with pytest.raises(ValueError):
        independent_mean_interval([1, np.nan])


def test_pairing_removes_shared_environment_variation():
    base = np.array([1., 10., -2., 30.])
    result = paired_bootstrap_difference(base+2, base, seed=4)
    assert result["estimate"] == result["low"] == result["high"] == 2
    assert result == paired_bootstrap_difference(base+2, base, seed=4)
    with pytest.raises(ValueError):
        paired_bootstrap_difference([1, 2], [1, 2, 3])


def test_importance_support_and_ess_contracts():
    d = importance_diagnostics([.5, .5, .5], [.25, .25, .25])
    np.testing.assert_allclose(d["weights"], [2, 2, 2])
    assert d["effective_sample_size"] == 3
    assert importance_diagnostics([1, 0, 0], [.01, .5, .5])["effective_sample_size"] == 1
    with pytest.raises(ValueError, match="undefined"):
        importance_diagnostics([0, 0], [.5, .5])
    np.testing.assert_array_equal(importance_weights([0, 0], [.5, .5]), [0, 0])
    for target, logged in [([1], [0]), ([1], [-1]), ([1.1], [.5]), ([1], [np.nan])]:
        with pytest.raises(ValueError):
            importance_weights(target, logged)


def test_cvar_fractional_mass_and_tail_direction():
    assert empirical_cvar([0, 10, 20], .5) == pytest.approx(10/3)
    assert empirical_cvar([0, 10, 20], .5, "upper") == pytest.approx(50/3)
    assert empirical_cvar([2, 2, 2], .1) == 2
    assert empirical_cvar([0, 10, 20], 1) == 10
    with pytest.raises(ValueError):
        empirical_cvar([], .5)


def test_cluster_weighting_and_zero_event_interval():
    np.testing.assert_array_equal(cluster_means([1, 3, 8], [10, 10, 20]), [2, 8])
    upper = bernoulli_zero_event_upper(100)
    assert (1-upper)**100 == pytest.approx(.05)
    assert 0 < bernoulli_zero_event_upper(1000) < upper < 1
