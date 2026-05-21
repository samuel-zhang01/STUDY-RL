"""Smoke tests for the reward-hack lab and the CAI mini-loop."""
from __future__ import annotations
import pytest

from project.hack_lab import hack_train
from project.cai import cai_train


@pytest.mark.smoke
def test_hack_lab_demonstrates_hacking():
    res = hack_train(seed=0, num_steps=20, sft_warmup_steps=40,
                     batch=4, max_new_tokens=12)
    h = res["history"]
    # Proxy reward should rise above its initial value.
    assert h["proxy"][-3:][0] >= h["proxy"][:3][0], (
        f"proxy did not rise: start {h['proxy'][:3]} end {h['proxy'][-3:]}")
    for L in h["length"]:
        assert 0 <= L <= 20


@pytest.mark.smoke
def test_kl_penalty_constrains_drift():
    res_free = hack_train(seed=0, num_steps=15, sft_warmup_steps=40,
                          batch=4, max_new_tokens=12)
    res_kl   = hack_train(seed=0, num_steps=15, sft_warmup_steps=40,
                          batch=4, max_new_tokens=12, kl_coef=1.0)
    assert res_kl["history"]["kl"][-1] <= res_free["history"]["kl"][-1] + 1e-3


@pytest.mark.smoke
def test_cai_loop_runs_and_revision_helps():
    res = cai_train(seed=0, num_steps=10, sft_warmup_steps=40,
                    batch=4, max_new_tokens=20)
    h = res["history"]
    import numpy as np
    pre = np.array(h["compliance_pre"])
    post = np.array(h["compliance_post"])
    assert post.mean() >= pre.mean() - 1e-6
