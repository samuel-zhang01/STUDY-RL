"""Inspect training hardware and exercise a real accelerator kernel."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform

import torch


def inspect_hardware() -> dict:
    info = {
        "python": platform.python_version(), "system": platform.system(),
        "machine": platform.machine(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(), "gpus": [],
    }
    for package in ("transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            info[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info[package] = None
    if info["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            with torch.cuda.device(index):
                bf16 = torch.cuda.is_bf16_supported()
            info["gpus"].append({
                "index": index, "name": prop.name,
                "vram_gib": round(prop.total_memory / 2**30, 2),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "bf16_supported": bf16,
            })
        info["torch_compiled_architectures"] = torch.cuda.get_arch_list()
    return info


def check_kernel(device: str, nf4: bool = False) -> dict:
    """A successful import is insufficient: execute forward AND backward."""
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Install the CUDA PyTorch wheel for your NVIDIA GPU/driver; see NVIDIA.md.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; use --device cpu or inspect your Apple PyTorch installation.")
    if nf4 and device != "cuda":
        raise ValueError("This lab's NF4 training path requires --device cuda; use ordinary LoRA on CPU/MPS.")
    x = torch.randn(2, 64, device=device, requires_grad=True)
    if nf4:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ValueError("Install requirements-nvidia.txt in the NVIDIA environment first.") from exc
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        layer = bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=dtype,
                                 compress_statistics=True, quant_type="nf4").to(device)
        # QLoRA keeps the quantized base frozen but needs gradients through it.
        layer.requires_grad_(False)
    else:
        layer = torch.nn.Linear(64, 32, bias=False, device=device)
    loss = layer(x).float().square().mean()
    loss.backward()
    if x.grad is None or not torch.isfinite(x.grad).all().item():
        raise RuntimeError("The accelerator kernel did not produce finite input gradients.")
    return {"device": device, "nf4": nf4, "forward_backward": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--check-nf4", action="store_true")
    args = parser.parse_args()
    info = inspect_hardware()
    print(json.dumps(info, indent=2), flush=True)
    device = args.device
    if device == "auto":
        device = "cuda" if info["cuda_available"] else "mps" if info["mps_available"] else "cpu"
    try:
        print(json.dumps(check_kernel(device, args.check_nf4), indent=2))
    except (ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Hardware check failed: {exc}\n")


if __name__ == "__main__":
    main()
