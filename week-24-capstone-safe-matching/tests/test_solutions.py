"""Contract tests for the capstone workbook solutions."""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)
sys.path.insert(0, os.path.dirname(WEEK))

from project import solutions as ref  # noqa


def test_ex1_shapes():
    out = ref.exercise_1(seed=0)
    assert out["n_jobs"] == 20
    assert out["n_candidates"] == 200
    assert out["n_interactions"] == 600
    assert 0.0 < out["group_0_rate"] < 1.0


def test_ex2_bias_is_nonzero():
    bias = ref.exercise_2(seed=0)
    # The logging policy is biased; demographic-parity gap should be > 0.
    assert bias > 0.0


def test_ex3_iql_trains():
    out = ref.exercise_3(seed=0)
    h = out["history"]
    assert h["q_loss"][-1] <= h["q_loss"][0] * 1.5  # at most 50% worse, usually better


def test_ex4_snips_finite():
    v = ref.exercise_4(seed=0)
    assert np.isfinite(v)


def test_ex5_finetune_runs():
    out = ref.exercise_5(seed=0)
    h = out["history"]
    assert len(h["return"]) > 0
    assert np.isfinite(out["final_parity"])


def test_ex6_audit_keys():
    out = ref.exercise_6(seed=0)
    for k in ("demographic_parity_gap", "snips_value", "mean_rec_rate"):
        assert k in out["audit_before"]
        assert k in out["audit_after"]


def test_ex7_red_team_in_range():
    g = ref.exercise_7(seed=0)
    assert 0.0 <= g <= 1.0
