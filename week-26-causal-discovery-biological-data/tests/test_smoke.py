"""Meaningful fixture metrics, model math, and evaluation leakage safeguards."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

PACKAGE = "week26_project"
ROOT = Path(__file__).resolve().parents[1]
if PACKAGE not in sys.modules:
    spec = importlib.util.spec_from_file_location(PACKAGE, ROOT/"project"/"__init__.py",
                                                submodule_search_locations=[str(ROOT/"project")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
from week26_project import train as training
from week26_project.agent import (Preprocessor, fit_ridge, patient_mse,
                                 patient_bootstrap_mse_interval)


@pytest.mark.smoke
def test_default_fixture_converges_and_adjusts_confounding():
    report = training.train(seed=0)
    assert abs(report["adjusted_ate"]-1.0) < 0.05
    assert abs(report["naive_ate"]-1.0) > 0.8
    assert report["enhanced_test_mse"] < 0.8*report["baseline_test_mse"]
    assert report["stable_features_ablation_test_mse"] < report["enhanced_test_mse"]
    assert report["enhanced_test_mse"] < report["training_mean_baseline_test_mse"]
    for key in ("baseline_training_loss", "enhanced_training_loss"):
        curve = np.asarray(report[key])
        assert curve[-1] < curve[0]
        assert np.max(np.diff(curve)) < 1e-10


def test_ridge_matches_closed_form_on_nontrivial_fixture():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3)); y = 0.5+x@np.array([1.2, -0.3, 0.8])
    model = fit_ridge(x, y, steps=400, ridge=0.03)
    d = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0, 1, 1, 1])
    expected = np.linalg.solve(d.T@d/len(x)+0.03*penalty, d.T@y/len(x))
    np.testing.assert_allclose(model.weights, expected, atol=1e-8)


def test_preprocessing_transform_is_frozen():
    pre = Preprocessor.fit([[0, np.nan], [2, 4]])
    fitted_means = pre.means.copy()
    heldout = pre.transform([[100, np.nan]])
    np.testing.assert_array_equal(pre.means, fitted_means)
    np.testing.assert_allclose(heldout, [[99, 0]])


def test_test_labels_cannot_select_enhancement_or_fit_parameters(monkeypatch):
    original = training.make_patient_data
    before = training.train(seed=0, n_patients=40, steps=80)
    def altered(*args, **kwargs):
        data = original(*args, **kwargs)
        outcome = data.outcome.copy()
        outcome[data.environments == 2] += 1000
        return replace(data, outcome=outcome)
    monkeypatch.setattr(training, "make_patient_data", altered)
    after = training.train(seed=0, n_patients=40, steps=80)
    assert before["selected_randomization_probability"] == after["selected_randomization_probability"]
    assert before["candidate_selection"] == after["candidate_selection"]
    assert before["fitted_imputation_means"] == after["fitted_imputation_means"]
    assert before["enhanced_weights"] == after["enhanced_weights"]
    assert after["enhanced_test_mse"] > before["enhanced_test_mse"] + 10000


def test_patient_metric_and_bootstrap_do_not_weight_technical_repeats_as_patients():
    y = [0, 0, 0, 0]; p = [0, 0, 0, 2]; ids = [0, 0, 0, 1]
    assert patient_mse(y, p, ids) == 2.0  # Row MSE would be 1.0.
    interval = patient_bootstrap_mse_interval(y, p, ids, seed=0, draws=1000)
    np.testing.assert_allclose(interval, [0, 4])
