"""Small contract and failure fixtures for every workbook reference solution."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest

# A distinct alias avoids collisions with other weeks' packages named project.
PACKAGE = "week26_project"
ROOT = Path(__file__).resolve().parents[1]
if PACKAGE not in sys.modules:
    spec = importlib.util.spec_from_file_location(PACKAGE, ROOT/"project"/"__init__.py",
                                                submodule_search_locations=[str(ROOT/"project")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
from week26_project import solutions as s


def test_mean_difference_and_adjustment():
    c = np.array([0]*10 + [1]*10)
    t = np.array([0]*8 + [1]*2 + [0]*2 + [1]*8)
    y = 2*c+t
    assert s.exercise_1_naive_ate(t, y) == pytest.approx(2.2)
    assert s.exercise_2_adjusted_ate(t, y, c) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="overlap"):
        s.exercise_2_adjusted_ate(c, y, c)
    with pytest.raises(ValueError):
        s.exercise_1_naive_ate([0, 2], [1, 2])


def test_opposite_scms_share_covariance():
    forward, reverse = s.exercise_3_equivalent_covariances(0.6)
    np.testing.assert_allclose(forward, [[1, 0.6], [0.6, 1]])
    np.testing.assert_array_equal(forward, reverse)
    with pytest.raises(ValueError):
        s.exercise_3_equivalent_covariances(1.0)


def test_patient_split_keeps_every_repeat_and_environment():
    ids = np.repeat(np.arange(18), 3)
    env = np.repeat(np.repeat(np.arange(3), 6), 3)
    split = s.exercise_4_patient_split(ids, env, seed=9)
    sets = {k: set(ids[v]) for k, v in split.items()}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                or sets["validation"] & sets["test"])
    assert len(np.concatenate(list(split.values()))) == len(ids)
    assert set(env[split["test"]]) == {2}
    for idx in split.values():
        assert all(np.sum(ids[idx] == pid) == 3 for pid in np.unique(ids[idx]))
    bad = env.copy(); bad[0] = 1
    with pytest.raises(ValueError, match="multiple"):
        s.exercise_4_patient_split(ids, bad)


def test_impute_missing_but_keep_biological_zeros():
    x = np.array([[0, np.nan, np.nan], [2, 4, np.nan]])
    filled, means = s.exercise_5_impute_fit(x)
    np.testing.assert_allclose(means, [1, 4, 0])
    np.testing.assert_allclose(filled, [[0, 4, 0], [2, 4, 0]])
    assert np.isnan(x[0, 1])  # No mutation.


def test_descriptive_spread_is_not_a_test():
    score = s.exercise_6_residual_spread([-1, 1, -2, 2], [0, 0, 1, 1])
    assert score == {"mean_spread": 0.0, "variance_spread": 3.0}


def test_task_randomization_only_changes_nuisance_column():
    x = np.arange(60, dtype=float).reshape(20, 3)
    np.testing.assert_array_equal(s.exercise_7_task_randomize(x, probability=0), x)
    shuffled = s.exercise_7_task_randomize(x, probability=1, seed=5)
    np.testing.assert_array_equal(shuffled[:, :2], x[:, :2])
    np.testing.assert_array_equal(np.sort(shuffled[:, 2]), np.sort(x[:, 2]))
    assert not np.array_equal(shuffled[:, 2], x[:, 2])
    np.testing.assert_array_equal(s.exercise_7_task_randomize(x, seed=5), shuffled)
    with pytest.raises(ValueError):
        s.exercise_7_task_randomize(x, probability=1.1)


def test_meaningful_matching_objective_and_ties():
    result = s.exercise_8_matching_priority_value([0.9, 0.8, 0.1], [-0.2, 0.5, 1.0], 2)
    np.testing.assert_array_equal(result["selected"], [0, 1])
    assert result["value"] == pytest.approx(0.15)
    assert s.exercise_8_matching_priority_value([1, 1], [2, -2], 1)["value"] == 2
    with pytest.raises(ValueError):
        s.exercise_8_matching_priority_value([1], [2], 0)
