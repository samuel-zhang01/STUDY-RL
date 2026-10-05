import pytest
import torch
from project.losses import _log1mexp
from project.model import CharTransformer
from project.train import train_variant, BakeoffConfig


def test_log1mexp_backward_is_finite_near_zero():
    value = torch.tensor([-1e-10, -100.], requires_grad=True)
    result = _log1mexp(value)
    result.sum().backward()
    assert torch.isfinite(result).all() and torch.isfinite(value.grad).all()
    assert value.grad[0] < 0


def test_dpo_rejects_missing_reference_before_updates():
    with pytest.raises(ValueError, match="frozen reference"):
        train_variant(CharTransformer(), None, [("Q:", "yes", "no")], BakeoffConfig(steps=1))
