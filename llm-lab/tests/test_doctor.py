import pytest
import torch

from llm_lab.doctor import check_kernel, inspect_hardware


def test_cpu_executes_gradients():
    assert check_kernel("cpu")["forward_backward"] == "passed"


def test_nf4_rejects_wrong_device_before_import():
    with pytest.raises(ValueError, match="requires --device cuda"):
        check_kernel("cpu", nf4=True)


def test_missing_cuda_reports_action(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA PyTorch wheel"):
        check_kernel("cuda")
    assert inspect_hardware()["gpus"] == []
