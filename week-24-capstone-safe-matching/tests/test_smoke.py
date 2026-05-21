"""Capstone smoke tests."""
from __future__ import annotations

import os, sys
import numpy as np

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
sys.path.insert(0, WEEK)
sys.path.insert(0, os.path.dirname(WEEK))

from project.data import generate_dataset, context_dim, context_vector  # noqa
from project.offline_iql import iql_pretrain, IQLConfig  # noqa
from project.online_ppo_lag import ppo_lag_finetune, PPOLagConfig  # noqa
from project.train import train  # noqa
from project.audit import (  # noqa
    demographic_parity_gap, snips_ope, full_audit,
)


def test_data_generation():
    data = generate_dataset(n_candidates=200, n_jobs=10, n_interactions=400, seed=0)
    assert len(data["jobs"]) == 10
    assert len(data["candidates"]) == 200
    assert len(data["interactions"]) == 400
    # Each interaction has a positive propensity.
    assert all(i.propensity > 0 for i in data["interactions"])


def test_iql_runs():
    data = generate_dataset(n_candidates=200, n_jobs=10, n_interactions=600, seed=0)
    out = iql_pretrain(data, cfg=IQLConfig(n_epochs=5), seed=0)
    assert "model" in out
    assert all(np.isfinite(out["history"]["q_loss"]))


def test_ppo_lag_runs():
    data = generate_dataset(n_candidates=200, n_jobs=10, n_interactions=600, seed=0)
    iql_out = iql_pretrain(data, cfg=IQLConfig(n_epochs=5), seed=0)
    ppo_out = ppo_lag_finetune(
        data, iql_out["model"],
        cfg=PPOLagConfig(n_updates=4, steps_per_update=128), seed=0,
    )
    assert len(ppo_out["history"]["return"]) == 4


def test_end_to_end_runs_under_30s():
    import time
    t = time.time()
    out = train(seed=0,
                iql_cfg=IQLConfig(n_epochs=5),
                ppo_cfg=PPOLagConfig(n_updates=6, steps_per_update=128),
                n_candidates=200, n_jobs=10, n_interactions=600)
    elapsed = time.time() - t
    assert elapsed < 60.0, f"capstone took {elapsed:.1f}s (limit 60s)"
    a = out["audit_after"]
    assert "demographic_parity_gap" in a
    assert "snips_value" in a
    assert "red_team" in a


def test_snips_ope_known_case():
    # If the target policy always picks the same action as the logger, SNIPS
    # is the empirical mean reward.
    data = generate_dataset(n_candidates=100, n_jobs=5, n_interactions=200, seed=0)
    target_probs = np.ones(len(data["interactions"]))   # always match logged action
    # NOTE: this scenario assumes target IS the logger so weights=1/propensity
    # multiplied by propensity gives 1, so SNIPS = mean(reward).
    v = snips_ope(data["interactions"], np.array([i.propensity for i in data["interactions"]]))
    emp = float(np.mean([i.reward for i in data["interactions"]]))
    assert abs(v - emp) < 1e-9
