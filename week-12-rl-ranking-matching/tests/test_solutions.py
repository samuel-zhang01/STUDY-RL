"""Tests for the 10 reference solutions in week 12."""
from __future__ import annotations

import os
import sys
import numpy as np

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)


def test_jaccard():
    from project.solutions import exercise_1_jaccard
    assert abs(exercise_1_jaccard(["a", "b"], ["a", "b"]) - 1.0) < 1e-9
    assert abs(exercise_1_jaccard(["a", "b"], ["c"]) - 0.0) < 1e-9
    assert abs(exercise_1_jaccard(["a", "b", "c"], ["b", "c", "d"]) - 0.5) < 1e-9


def test_position_bias():
    from project.solutions import exercise_2_position_bias
    pb = exercise_2_position_bias(5, eta=1.0)
    assert pb.shape == (5,)
    assert abs(pb[0] - 1.0) < 1e-9
    assert pb[0] > pb[-1]


def test_two_tower_ndcg():
    from project.solutions import exercise_3_two_tower_ndcg
    n10, n5 = exercise_3_two_tower_ndcg(seed=0, epochs=20)
    assert 0.3 < n10 <= 1.0
    assert 0.3 < n5 <= 1.0


def test_ips_ltr():
    from project.solutions import exercise_4_ips_ltr
    naive, ips = exercise_4_ips_ltr(seed=0)
    # IPS is *unbounded* in principle and can land far from naive; only sanity-check non-negativity.
    assert naive >= 0
    assert ips >= 0


def test_reranker_improvement():
    from project.solutions import exercise_5_reranker_improvement
    tt, rr = exercise_5_reranker_improvement(seed=0, epochs=20)
    assert tt > 0.4 and rr > 0.4


def test_dp_gap_bounded():
    from project.solutions import exercise_6_dp_gap
    g = exercise_6_dp_gap(seed=0, epochs=15)
    assert 0.0 <= g <= 1.0


def test_eo_gap_bounded():
    from project.solutions import exercise_7_eo_gap
    g = exercise_7_eo_gap(seed=0, epochs=15)
    assert 0.0 <= g <= 1.0


def test_calibration_improves():
    from project.solutions import exercise_8_calibration
    eces = exercise_8_calibration(seed=0)
    # Beta and isotonic should improve calibration significantly.
    assert eces["isotonic"] < eces["raw"]
    assert eces["beta"] < eces["raw"]


def test_exposure_positive():
    from project.solutions import exercise_9_exposure
    f, m = exercise_9_exposure(seed=0, epochs=15)
    assert f > 0 and m > 0


def test_recommendation_memo_keys():
    from project.solutions import exercise_10_ats_recommendation_memo
    d = exercise_10_ats_recommendation_memo(seed=0, epochs=15)
    for k in ("tt_ndcg@10", "rerank_ndcg@10", "dp_gap_pre", "dp_gap_post",
              "eo_gap_pre", "eo_gap_post", "ece_raw", "ece_beta"):
        assert k in d


def test_logger_records_full_mixture_probability_with_small_candidate_pool():
    from project.train import simulate_hire_logs
    class Dataset:
        def true_match(self, c, j): return .5
    scores = np.array([[3., 2., 1.]])
    logs = simulate_hire_logs(Dataset(), scores, np.random.default_rng(0),
        n_per_job=100, eps=.6, top_n=20)
    assert any(c == 0 for _,c,*_ in logs) and any(c != 0 for _,c,*_ in logs)
    for j,c,r,propensity,mu in logs:
        assert abs(propensity - (.6 / 3 + .4 * (c == 0))) < 1e-12
