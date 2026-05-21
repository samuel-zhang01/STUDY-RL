"""Smoke test for week 12: short pipeline pass."""
from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)


def test_project_importable():
    importlib.import_module("project")


def test_short_pipeline_runs():
    from project.train import train
    out = train(seed=0, epochs=15, n_logs_per_job=10)
    # Two-tower should be at least better than random baseline (~0.4).
    assert out.metrics["tt_ndcg@10"] > 0.45, out.metrics
    # Reranker NDCG should be a sane number.
    assert 0.3 <= out.metrics["rerank_ndcg@10"] <= 1.0
    # DP gap and EO gap bounded in [0, 1].
    for d in (out.fairness_pre, out.fairness_post):
        assert 0.0 <= d["dp_gap"] <= 1.0
        assert 0.0 <= d["eo_gap"] <= 1.0
    # Calibration: at least one of platt/iso/beta should be no worse than raw.
    raw_ece = out.calibration["raw"]["ece"]
    best = min(out.calibration[k]["ece"] for k in ("platt", "isotonic", "beta"))
    assert best <= raw_ece + 0.05  # tolerate noise on tiny eval set
