"""Numerical invariants for the educational quantiser; no model downloads."""
from __future__ import annotations

import math

import pytest
import torch

from llm_lab.quantisation import (
    affine_quantize,
    estimate_lora_memory,
    low_rank_residual,
    main,
    symmetric_quantize,
)


@pytest.mark.parametrize("bits", range(2, 9))
def test_bit_packing_reconstructs_representable_codes_across_byte_boundaries(bits):
    # An odd number of values exercises partial last bytes, not only nibble pairs.
    levels = torch.arange(2**bits, dtype=torch.float64)
    values = torch.cat([levels, levels[:3]])
    q = affine_quantize(values, bits=bits, value_range=(0, 2**bits - 1))
    assert q.payload.dtype == torch.uint8
    assert q.payload_nbytes == math.ceil(values.numel() * bits / 8)
    torch.testing.assert_close(q.unpack().to(torch.float64), values)
    torch.testing.assert_close(q.dequantize(), values)


def test_four_bit_payload_is_actually_packed_not_one_byte_per_weight():
    q = affine_quantize(torch.arange(16, dtype=torch.float64), bits=4, value_range=(0, 15))
    assert q.payload.tolist() == [16, 50, 84, 118, 152, 186, 220, 254]
    assert q.payload_nbytes == 8


@pytest.mark.parametrize("quantize", [symmetric_quantize, affine_quantize])
def test_no_saturation_reconstruction_error_at_most_half_step(quantize):
    # Fixed calibration followed by points strictly inside the representable interval.
    probe = quantize(torch.tensor([-1.0, 2.0]), bits=4)
    low, high = probe.representable_range
    x = torch.linspace(low, high, 301, dtype=torch.float64).reshape(7, 43)
    if quantize is symmetric_quantize:
        q = quantize(x, bits=4, clip_value=high)
    else:
        q = quantize(x, bits=4, value_range=(-1.0, 2.0))
    assert q.shape == (7, 43)
    assert torch.max(torch.abs(x - q.dequantize())) <= q.scale / 2 + 1e-12


@pytest.mark.parametrize("quantize", [symmetric_quantize, affine_quantize])
def test_rounding_ties_are_even_before_integer_zero_point_is_added(quantize):
    x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=torch.float64)
    kwargs = {"clip_value": 3} if quantize is symmetric_quantize else {"value_range": (-3, 4)}
    q = quantize(x, bits=3, **kwargs)
    torch.testing.assert_close(q.dequantize(), torch.tensor([-2., -2., 0., 0., 2., 2.], dtype=torch.float64))


def test_symmetric_clipping_saturates_and_invalidates_unclipped_error_bound():
    x = torch.tensor([-4.0, -0.2, 0.0, 0.2, 4.0], dtype=torch.float64)
    q = symmetric_quantize(x, bits=4, clip_value=1.0)
    torch.testing.assert_close(q.dequantize()[[0, -1]], torch.tensor([-1., 1.], dtype=torch.float64))
    assert torch.max(torch.abs(x - q.dequantize())) == 3
    assert 3 > q.scale / 2


def test_affine_calibration_zero_point_and_saturation():
    q = affine_quantize(torch.tensor([-3., -1., 0., 1., 2., 4.]), bits=2,
                        value_range=(-1., 2.))
    assert q.scale == 1 and q.zero_point == 1
    assert q.unpack().tolist() == [0, 0, 1, 2, 3, 3]
    assert q.dequantize().tolist() == [-1, -1, 0, 1, 2, 2]


@pytest.mark.parametrize("quantize", [symmetric_quantize, affine_quantize])
def test_all_zero_tensor_has_finite_scale_and_exact_zero_reconstruction(quantize):
    q = quantize(torch.zeros(3, 5), bits=4)
    assert q.scale == 1.0
    assert q.payload_nbytes == 8
    assert torch.count_nonzero(q.dequantize()) == 0


@pytest.mark.parametrize("constant", [-7.0, 7.0])
def test_affine_constant_nonzero_tensor_includes_zero_in_calibration(constant):
    x = torch.full((5,), constant, dtype=torch.float64)
    q = affine_quantize(x)
    torch.testing.assert_close(q.dequantize(), x)
    assert q.scale > 0


def test_input_is_not_mutated_and_gradient_is_not_misrepresented():
    x = torch.tensor([-.5, .2, 1.], requires_grad=True)
    before = x.detach().clone()
    result = symmetric_quantize(x).dequantize()
    torch.testing.assert_close(x.detach(), before)
    assert not result.requires_grad


@pytest.mark.parametrize("bad_bits", [0, 1, 9, -4, 4.0, True])
@pytest.mark.parametrize("quantize", [symmetric_quantize, affine_quantize])
def test_reject_invalid_bit_widths(quantize, bad_bits):
    with pytest.raises(ValueError, match="bits"):
        quantize([1.0], bits=bad_bits)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [1 + 2j]])
@pytest.mark.parametrize("quantize", [symmetric_quantize, affine_quantize])
def test_reject_empty_nonfinite_and_complex_inputs(quantize, values):
    with pytest.raises(ValueError):
        quantize(values)


@pytest.mark.parametrize("value_range", [(1, 2), (-2, -1), (0, 0), (2, -1), (0, float("inf"))])
def test_explicit_affine_range_must_be_valid_and_contain_zero(value_range):
    with pytest.raises(ValueError, match="range"):
        affine_quantize([0., 1.], value_range=value_range)


@pytest.mark.parametrize("clip_value", [0, -1, float("inf"), float("nan")])
def test_explicit_symmetric_clipping_scale_must_be_positive(clip_value):
    with pytest.raises(ValueError, match="clip_value"):
        symmetric_quantize([0., 1.], clip_value=clip_value)


def test_rank_one_residual_fixes_constructed_weight_and_output_error():
    w = torch.tensor([[1., 0.], [.2, .2]], dtype=torch.float64)
    reconstructed = symmetric_quantize(w, bits=2).dequantize()
    b, a = low_rank_residual(w, reconstructed, rank=1)
    x = torch.tensor([1., 2.], dtype=torch.float64)
    assert b.shape == (2, 1) and a.shape == (1, 2)
    assert torch.linalg.vector_norm(w - reconstructed) > 0.28
    torch.testing.assert_close(reconstructed + b @ a, w)
    torch.testing.assert_close(reconstructed @ x + b @ (a @ x), torch.tensor([1., .6], dtype=torch.float64))


def test_rank_one_cannot_reconstruct_every_residual():
    # The identity residual has two nonzero singular values; rank one leaves error 1.
    w = torch.eye(2, dtype=torch.float64)
    b, a = low_rank_residual(w, torch.zeros_like(w), rank=1)
    torch.testing.assert_close(torch.linalg.vector_norm(w - b @ a), torch.tensor(1., dtype=torch.float64))


@pytest.mark.parametrize("rank", [0, 3, 1.5, True])
def test_invalid_correction_rank_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        low_rank_residual(torch.eye(2), torch.zeros(2, 2), rank)


def test_memory_accounting_counts_trainable_states_and_scale_overhead():
    estimate = estimate_lora_memory(1_000_000, 10_000, base_bits=4)
    assert estimate == {
        "frozen_base_payload": 500_000,
        "base_scales": 62_500,
        "adapter_parameters": 40_000,
        "adapter_gradients": 40_000,
        "adapter_adam_moments": 80_000,
        "accounted_bytes": 722_500,
    }
    assert estimate_lora_memory(1_000_000, 10_000, base_bits=32)["accounted_bytes"] == 4_160_000
    assert estimate_lora_memory(1, 0, base_bits=4)["base_scales"] == 4
    with pytest.raises(ValueError):
        estimate_lora_memory(-1, 10)
    with pytest.raises(ValueError):
        estimate_lora_memory(1, 10, scale_block_size=0)


def test_cli_states_the_scope_and_reports_real_integer_storage(capsys):
    main([])
    text = capsys.readouterr().out
    assert "dtype=torch.uint8" in text
    assert "NOT NF4" in text
    assert "NOT measured peak memory" in text
