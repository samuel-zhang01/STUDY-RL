import numpy as np
from project.cai import revise, critique
from project.solutions import exercise_2, exercise_6


def test_constitution_revision_handles_case_and_reserves_punctuation_space():
    revised = revise("HELL DAMN " + "a" * 45)
    assert len(revised) <= 40
    assert all(critique(revised).values())
    assert revise("hello world.") == "hello world."
    assert critique("hello world.")["no_swearing"]
    assert not critique("HELL.")["no_swearing"]


def test_rank_correlation_has_no_artificial_order_for_tied_proxy_scores():
    _, _, correlation = exercise_2(["aaaa", "bbbb", "cccc"])
    assert correlation == 0


def test_numpy_dpo_stays_finite_at_extreme_negative_margin():
    assert np.isclose(exercise_6(np.array([-10000.]), np.array([0.]),
                                np.array([0.]), np.array([0.]), beta=1), 10000.)
