"""Smoke tests for the week-02 DP project."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK = os.path.dirname(HERE)
REPO = os.path.dirname(WEEK)
sys.path.insert(0, WEEK)
sys.path.insert(0, REPO)

from project.train import train  # noqa: E402


def test_train_outputs_expected_keys():
    out = train()
    assert set(out) >= {
        "det_vi_sweeps", "det_pi_sweeps", "stoch_vi_V_start",
        "car_rental_V_at_5_5", "car_rental_iters",
    }


def test_vi_converges_and_pi_is_faster_per_sweep():
    out = train()
    # Both DP methods produce sensible values.
    assert out["det_vi_sweeps"] > 0
    assert out["det_pi_sweeps"] > 0
    assert out["det_vi_V_start"] > 0  # positive return reachable
    # Policy iteration converges in *far* fewer outer sweeps than VI inner iterations.
    assert out["det_pi_sweeps"] <= out["det_vi_sweeps"]


def test_car_rental_value_function_is_finite():
    out = train()
    assert np.isfinite(out["car_rental_V_at_5_5"])
    assert out["car_rental_iters"] >= 1
