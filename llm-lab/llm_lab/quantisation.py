"""Inspectable CPU quantisation: real packed integers, not NF4 or a GEMM kernel.

Run from ``llm-lab`` with ``python -m llm_lab.quantisation``. No model download,
CUDA, bitsandbytes, or Transformers is needed. Quantisation is non-differentiable;
input tensors are detached and copied to CPU float64 for deterministic arithmetic.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import torch


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _bits(bits: int) -> int:
    bits = _integer(bits, "bits", 2)
    if bits > 8:
        raise ValueError("bits must be between 2 and 8 inclusive")
    return bits


def _values(values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    source = torch.as_tensor(values)
    if source.is_complex():
        raise ValueError("values must be real")
    x = source.detach().to(device="cpu", dtype=torch.float64)
    if x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError("values must be nonempty and finite")
    return x


def _pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned codes, least-significant bits first, including odd lengths."""
    packed: list[int] = []
    accumulator = occupied = 0
    for code in codes.reshape(-1).tolist():
        accumulator |= int(code) << occupied
        occupied += bits
        while occupied >= 8:
            packed.append(accumulator & 255)
            accumulator >>= 8
            occupied -= 8
    if occupied:
        packed.append(accumulator & 255)
    return torch.tensor(packed, dtype=torch.uint8)


@dataclass(frozen=True)
class QuantizedTensor:
    """Packed affine representation: reconstructed value = scale * (code - zero_point).

    ``payload`` is the actual integer storage. ``unpack`` allocates an integer
    tensor and ``dequantize`` allocates floating-point storage for inspection.
    Python object/shape/scale/zero-point metadata is deliberately excluded from
    ``payload_nbytes``; this object is not a compact production file format.
    """

    payload: torch.Tensor
    shape: tuple[int, ...]
    bits: int
    scale: float
    zero_point: int
    scheme: str

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def payload_nbytes(self) -> int:
        return self.payload.numel() * self.payload.element_size()

    @property
    def representable_range(self) -> tuple[float, float]:
        # Symmetric mode reserves the highest code, giving balanced signed levels.
        largest_code = (1 << self.bits) - (2 if self.scheme == "symmetric" else 1)
        return (-self.zero_point * self.scale,
                (largest_code - self.zero_point) * self.scale)

    def unpack(self) -> torch.Tensor:
        """Return unsigned integer codes in a newly allocated CPU uint8 tensor."""
        codes: list[int] = []
        accumulator = occupied = 0
        mask = (1 << self.bits) - 1
        for byte in self.payload.tolist():
            accumulator |= int(byte) << occupied
            occupied += 8
            while occupied >= self.bits and len(codes) < self.numel:
                codes.append(accumulator & mask)
                accumulator >>= self.bits
                occupied -= self.bits
        return torch.tensor(codes, dtype=torch.uint8).reshape(self.shape)

    def dequantize(self, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Reconstruct floating values; this is not integer matrix multiplication."""
        if not dtype.is_floating_point:
            raise ValueError("dequantization dtype must be floating point")
        return ((self.unpack().to(torch.float64) - self.zero_point) * self.scale).to(dtype)


def symmetric_quantize(
    values: torch.Tensor | Sequence[float], bits: int = 4,
    clip_value: float | None = None,
) -> QuantizedTensor:
    """Quantize onto balanced signed levels [-qmax, qmax], qmax = 2**(bits-1)-1.

    Uses nearest rounding with ties to even. A user-supplied ``clip_value`` fixes
    the maximum absolute representable value and may saturate outliers. The
    all-zero auto-range uses scale 1 instead of dividing by zero. Signed values
    are shifted by qmax into unsigned codes before bit packing.
    """
    bits = _bits(bits)
    x = _values(values)
    qmax = (1 << (bits - 1)) - 1
    maximum = float(x.abs().max()) if clip_value is None else float(clip_value)
    if not math.isfinite(maximum) or maximum < 0 or (clip_value is not None and maximum == 0):
        raise ValueError("clip_value must be finite and positive when supplied")
    scale = maximum / qmax if maximum else 1.0
    if scale == 0:
        raise ValueError("nonzero range is too small for a float64 scale")
    signed = torch.round(x.clamp(-maximum, maximum) / scale).clamp(-qmax, qmax)
    codes = (signed.to(torch.int64) + qmax).to(torch.uint8)
    return QuantizedTensor(_pack(codes, bits), tuple(x.shape), bits, scale, qmax, "symmetric")


def affine_quantize(
    values: torch.Tensor | Sequence[float], bits: int = 4,
    value_range: tuple[float, float] | None = None,
) -> QuantizedTensor:
    """Uniform unsigned affine quantisation, with an integer zero point.

    The automatic calibration range includes zero, even for constant or one-sign
    tensors. An explicit range must include zero too. Rounding the zero point can
    shift the reconstructed endpoints away from the calibration endpoints; see
    ``representable_range``. Quantisation rounds the signed scaled value first,
    then adds the integer zero point, using ties to even for each rounding.
    """
    bits = _bits(bits)
    x = _values(values)
    if value_range is None:
        lower, upper = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
    else:
        if len(value_range) != 2:
            raise ValueError("value_range must contain (lower, upper)")
        lower, upper = map(float, value_range)
        if (not math.isfinite(lower) or not math.isfinite(upper)
                or not lower <= 0 <= upper or lower >= upper):
            raise ValueError("value_range must be finite, increasing, and include zero")
    qmax = (1 << bits) - 1
    scale = (upper - lower) / qmax if upper > lower else 1.0
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("range cannot be represented by a finite positive float64 scale")
    zero_point = max(0, min(qmax, round(-lower / scale)))
    # Saturate to reconstructed endpoints before scaling to avoid overflow on outliers.
    minimum_value, maximum_value = -zero_point * scale, (qmax - zero_point) * scale
    codes = (torch.round(x.clamp(minimum_value, maximum_value) / scale) + zero_point)
    codes = codes.clamp(0, qmax).to(torch.uint8)
    return QuantizedTensor(_pack(codes, bits), tuple(x.shape), bits, scale, zero_point, "affine")


def low_rank_residual(
    original: torch.Tensor, reconstructed: torch.Tensor, rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return B, A for the best rank-r residual in Frobenius norm using an SVD.

    This is an oracle arithmetic example, not LoRA training or LoftQ. It requires
    the full original weights and optimises weight reconstruction, not task loss.
    For LoRA scaling alpha/r, absorb the scale into one factor as appropriate.
    """
    original, reconstructed = _values(original), _values(reconstructed)
    rank = _integer(rank, "rank", 1)
    if original.ndim != 2 or original.shape != reconstructed.shape:
        raise ValueError("original and reconstructed must be matrices of the same shape")
    if rank > min(original.shape):
        raise ValueError("rank cannot exceed the smaller matrix dimension")
    u, singular, vh = torch.linalg.svd(original - reconstructed, full_matrices=False)
    b = u[:, :rank] * singular[:rank]
    a = vh[:rank, :]
    return b, a


def estimate_lora_memory(
    base_parameters: int, adapter_parameters: int, base_bits: int = 4,
    scale_block_size: int = 64, scale_bytes: int = 4,
) -> dict[str, int]:
    """Illustrative storage accounting, not a peak-device-memory prediction.

    Assume frozen base weights, one fp32 scale per block for sub-16-bit base
    storage (no double quantisation), fp32 adapters and their gradients, and two
    fp32 Adam moments. Dense 16/32-bit base storage has no quantisation scales.
    Excludes activations, KV cache, allocator/kernel workspace, unquantised
    modules, reference models, transient dequantisation, and Python metadata.
    """
    base_parameters = _integer(base_parameters, "base_parameters")
    adapter_parameters = _integer(adapter_parameters, "adapter_parameters")
    base_bits = _integer(base_bits, "base_bits", 1)
    scale_block_size = _integer(scale_block_size, "scale_block_size", 1)
    scale_bytes = _integer(scale_bytes, "scale_bytes")
    if base_bits not in (2, 3, 4, 8, 16, 32):
        raise ValueError("base_bits must be one of 2, 3, 4, 8, 16, 32")
    result = {
        "frozen_base_payload": (base_parameters * base_bits + 7) // 8,
        "base_scales": ((base_parameters + scale_block_size - 1) // scale_block_size
                        * scale_bytes if base_bits < 16 else 0),
        "adapter_parameters": adapter_parameters * 4,
        "adapter_gradients": adapter_parameters * 4,
        "adapter_adam_moments": adapter_parameters * 8,
    }
    result["accounted_bytes"] = sum(result.values())
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, default=4, help="uniform quantisation bits, 2 to 8")
    parser.add_argument("--base-parameters", type=int, default=1_000_000)
    parser.add_argument("--adapter-parameters", type=int, default=10_000)
    args = parser.parse_args(argv)
    _bits(args.bits)
    x = torch.tensor([-1.0, -0.6, -0.1, 0.0, 0.3, 0.8, 1.0], dtype=torch.float32)
    print("CPU uniform quantisation demo: packed integer storage; NOT NF4 / QLoRA / integer GEMM")
    for quantize in (symmetric_quantize, affine_quantize):
        q = quantize(x, bits=args.bits)
        restored = q.dequantize()
        print(f"\n{q.scheme}: bits={q.bits}, scale={q.scale:.8g}, zero_point={q.zero_point}")
        print(f"  original:      {x.tolist()}")
        print(f"  integer codes: {q.unpack().tolist()}")
        print(f"  reconstructed: {[round(v, 6) for v in restored.tolist()]}")
        print(f"  max abs error: {(restored - x).abs().max().item():.8g}")
        print(f"  payload: {q.payload.tolist()} ({q.payload_nbytes} bytes, dtype={q.payload.dtype})")
        print(f"  original fp32 tensor: {x.numel() * x.element_size()} bytes")
    w = torch.tensor([[1.0, 0.0], [0.2, 0.2]], dtype=torch.float64)
    restored = symmetric_quantize(w, bits=2).dequantize()
    b, a = low_rank_residual(w, restored, rank=1)
    print("\nRank-1 oracle residual example (not a trained adapter):")
    print(f"  ||W-W_quant||_F: {torch.linalg.vector_norm(w-restored):.8f}")
    print(f"  ||W-(W_quant+BA)||_F: {torch.linalg.vector_norm(w-restored-b@a):.8f}")
    print("\nIllustrative bytes: frozen base + one fp32 scale per 64 quantised weights")
    for bits in (32, 16, 8, 4):
        estimate = estimate_lora_memory(args.base_parameters, args.adapter_parameters, bits)
        print(f"  base {bits:2d}-bit: {estimate}")
    print("Excluded: activations, KV cache, reference model, workspace, unquantised layers,")
    print("temporary dequantisation and object metadata. These are NOT measured peak memory.")


if __name__ == "__main__":
    main()
