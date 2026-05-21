"""Smoke tests for week 19.

We do NOT depend on transformers/trl/peft -- those are documented in
``project.trl_recipe`` but the bake-off uses the pure-PyTorch losses
in ``project.losses``.
"""
import os
import sys
import time

THIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, THIS)

import torch  # noqa: E402
# Pin threads -- tiny ops thrash on 20-core hosts.
torch.set_num_threads(1)

from project.data import ToyPreferenceDataset  # noqa: E402
from project.model import CharTransformer  # noqa: E402
from project.train import train_variant, BakeoffConfig  # noqa: E402
from project.losses import (  # noqa: E402
    dpo_loss, ipo_loss, simpo_loss, orpo_loss, _log1mexp,
)


def test_dpo_loss_basic():
    z = torch.zeros(4)
    loss = dpo_loss(z, z, z, z, beta=0.1)
    assert abs(float(loss) - 0.693) < 1e-2


def test_ipo_loss_targets_finite_margin():
    z = torch.zeros(4)
    loss = ipo_loss(z, z, z, z, beta=0.5)
    assert abs(float(loss) - 1.0) < 1e-5


def test_simpo_reference_free():
    lp_w = torch.tensor([-2.0])
    lp_l = torch.tensor([-10.0])
    L_w = torch.tensor([2])
    L_l = torch.tensor([10])
    loss = simpo_loss(lp_w, lp_l, L_w, L_l, beta=2.0, gamma=0.0)
    assert abs(float(loss) - 0.693) < 1e-2


def test_log1mexp_against_naive():
    x = torch.tensor([-0.5, -1.0, -2.0, -5.0])
    naive = torch.log1p(-torch.exp(x))
    stable = _log1mexp(x)
    assert torch.allclose(naive, stable, atol=1e-5)


def test_orpo_runs_and_finite():
    lp_w = torch.tensor([-2.0, -3.0])
    lp_l = torch.tensor([-4.0, -5.0])
    nll_w = torch.tensor([2.0, 3.0])
    loss = orpo_loss(lp_w, lp_l, nll_w, beta=0.1)
    assert torch.isfinite(loss)


def test_smoke_train_variant_dpo():
    torch.manual_seed(0)
    t0 = time.time()
    ds = ToyPreferenceDataset(n=5, seed=0)
    m = CharTransformer()
    ref = CharTransformer()
    ref.load_state_dict(m.state_dict())
    h = train_variant(m, ref, ds.pairs, BakeoffConfig(variant="dpo", steps=10, batch_size=2))
    assert len(h["loss"]) == 10
    elapsed = time.time() - t0
    assert elapsed < 60.0, f"too slow: {elapsed:.1f}s"


def test_smoke_train_variant_simpo():
    torch.manual_seed(0)
    ds = ToyPreferenceDataset(n=5, seed=0)
    m = CharTransformer()
    h = train_variant(m, None, ds.pairs, BakeoffConfig(variant="simpo", steps=10, batch_size=2))
    assert len(h["loss"]) == 10
    assert max(h["acc"]) > 0.0
