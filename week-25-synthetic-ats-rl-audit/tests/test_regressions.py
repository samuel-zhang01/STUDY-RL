import numpy as np
import pytest
from project.env import generate_population, context_features
from project.scorer import heuristic_score, heuristic_feature_score, top_k_shortlist
from project.audit import calibration_buckets, equalised_odds_gap
from project.ope import snips
from project.train import train


def test_logged_features_reconstruct_the_actual_heuristic_and_sort_shortlists():
    data = generate_population(n_candidates=20, n_jobs=2, seed=4)
    features = np.stack([context_features(data, c, 0) for c in range(20)])
    scores = heuristic_score(data)
    np.testing.assert_allclose(heuristic_feature_score(features), scores[:, 0])
    shortlist = top_k_shortlist(scores, 0, k=5)
    assert np.all(np.diff(scores[shortlist, 0]) <= 0)


def test_equalized_odds_includes_false_positives_and_calibration_includes_one():
    assert equalised_odds_gap(np.array([1., 1., 1., 0.]),
                             np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0])) == 1.
    assert calibration_buckets(np.array([1.]), np.array([1.]))[-1][-1] == 1


def test_ope_requires_positive_recorded_propensities():
    slate = np.eye(2)
    with pytest.raises(ValueError, match="positivity"):
        snips([(slate[0], 0, 1., 0., slate)], lambda _: np.array([.5, .5]))


def test_estimator_uses_fresh_logs_and_returns_synthetic_value_diagnostics():
    result = train(n_candidates=30, n_jobs=3, k_slate=5, n_rounds=30, seed=1)
    assert result.training_log is not result.log
    assert [row[2] for row in result.training_log] != [row[2] for row in result.log]
    assert set(result.ground_truth_values) == {"LinTS", "Heuristic", "Uniform"}
    assert all(0 <= value <= 1 for value in result.ground_truth_values.values())
