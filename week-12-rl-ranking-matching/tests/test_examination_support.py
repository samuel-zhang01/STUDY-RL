"""The position model needs support even when the finite log contains no click."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from project.cltr import ips_ltr_value


def test_position_ratio_has_a_hand_calculated_target_value():
    # Item1 moves from examination .5 to1: its one click contributes2.
    assert ips_ltr_value(np.array([0, 1]), np.array([0, 1]),
                         np.array([1., .5]), np.array([1, 0])) == 2


def test_missing_item_support_is_visible_without_an_observed_click():
    with pytest.raises(ValueError, match="support"):
        ips_ltr_value(np.array([0, 0]), np.array([0, 1]),
                      np.array([1., .5]), np.array([2, 0]))
    with pytest.raises(ValueError, match="support"):
        ips_ltr_value(np.array([0, 0]), np.array([0, 1]),
                      np.array([1., 0.]), np.array([1, 0]))


def test_impossible_clicks_and_unobserved_zero_exposure_differ():
    with pytest.raises(ValueError, match="impossible"):
        ips_ltr_value(np.array([0, 1]), np.array([0, 1]),
                      np.array([1., 0.]), np.array([0, 1]))
    assert ips_ltr_value(np.array([1, 0]), np.array([0, 1]),
                         np.array([1., 0.]), np.array([0, 1])) == 1
