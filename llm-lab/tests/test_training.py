"""No downloads: test real PEFT updates on a tiny randomly initialized Llama."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import sys

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_lab.inference import compute_dtype, resolve_device
from llm_lab.training import (
    cache_reference_logprobs, collate_completions, completion_logprobs,
    disable_dropout, dpo_losses, encode_completion, preference_scores, train_adapter,
)


class LocalChatTokenizer:
    """Deterministic template with EOS == padding and a trailing template suffix."""
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [4 + ord(c) % 26 for c in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert tokenize and return_dict is False
        prefix = [1] + [4 + ord(c) % 26 for c in messages[0]["content"]] + [3]
        if add_generation_prompt:
            return prefix
        return prefix + [4 + ord(c) % 26 for c in messages[1]["content"]] + [2, 42]


def tiny_adapter(attention_dropout=0.0):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=128, attention_dropout=attention_dropout,
                         bos_token_id=1, eos_token_id=2, pad_token_id=2)
    base = LlamaForCausalLM(config)
    return get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear",
                                         r=4, lora_alpha=8, lora_dropout=0.0, bias="none"))


def encoded(prompt="x", completion="7"):
    return encode_completion(LocalChatTokenizer(), prompt, completion, max_length=64)


def test_padding_masks_preserve_real_eos_and_completion_targets():
    short, long = encoded("x", "7"), encoded("long", "111")
    batch = collate_completions([short, long], pad_token_id=2)
    n = len(short["input_ids"])
    assert short["input_ids"][-1] == 2  # terminal EOS retained; template suffix removed
    assert batch["labels"][0, n - 1] == 2
    assert (batch["labels"][0, n:] == -100).all()  # padding has the very same input ID
    assert (batch["input_ids"][0, n:] == 2).all()
    assert batch["attention_mask"][0, n - 1] == 1
    assert (batch["attention_mask"][0, n:] == 0).all()
    assert (batch["labels"][0, :short["prompt_length"]] == -100).all()
    scores, counts = completion_logprobs(torch.zeros(*batch["labels"].shape, 64), batch["labels"])
    assert counts.tolist() == [2, 4]  # includes EOS but not prompt or padding
    assert torch.allclose(scores, -counts * math.log(64))


def test_prefix_mismatch_and_target_truncation_fail_loudly():
    class BrokenTemplate(LocalChatTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            if not kwargs["add_generation_prompt"]:
                result[0] = 59
            return result
    with pytest.raises(ValueError, match="prefix mismatch"):
        encode_completion(BrokenTemplate(), "x", "7", 64)
    with pytest.raises(ValueError, match="remove every completion target"):
        encode_completion(LocalChatTokenizer(), "long prompt", "7", 3)
    with pytest.raises(ValueError, match="not silently truncated"):
        encode_completion(LocalChatTokenizer(), "x", "1234567", 6)


def test_sft_updates_adapter_while_base_stays_exactly_frozen():
    model = tiny_adapter()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    train_adapter(model, [encoded()], "sft", 2, steps=2, batch_size=1,
                  gradient_accumulation=1, lr=0.01, log_every=0)
    changed = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert "lora_" in name
            changed.append(not torch.equal(parameter, before[name]))
        else:
            assert torch.equal(parameter, before[name]), name
    assert any(changed), "An actual LoRA weight must update, not merely the reported loss."


def test_accumulation_matches_one_batch_for_unequal_completion_lengths():
    # If microbatch losses were averaged equally, the two-token completion would
    # get disproportionate weight relative to the five-token completion.
    first = tiny_adapter()
    second = deepcopy(first)
    examples = [encoded("x", "7"), encoded("long", "1234")]
    common = dict(steps=1, lr=0.001, max_grad_norm=100.0, log_every=0)
    train_adapter(first, examples, "sft", 2, batch_size=2, gradient_accumulation=1, **common)
    train_adapter(second, examples, "sft", 2, batch_size=1, gradient_accumulation=2, **common)
    for (name, p1), (_, p2) in zip(first.named_parameters(), second.named_parameters()):
        if p1.requires_grad:
            assert torch.allclose(p1, p2, atol=2e-6, rtol=2e-5), name


def test_dpo_gradient_sign_and_frozen_reference():
    chosen = torch.tensor([-4.0], requires_grad=True)
    rejected = torch.tensor([-5.0], requires_grad=True)
    ref_chosen = torch.tensor([-4.0], requires_grad=True)
    ref_rejected = torch.tensor([-5.0], requires_grad=True)
    losses, margin = dpo_losses(chosen, rejected, ref_chosen, ref_rejected, 0.2)
    assert margin.item() == 0
    assert losses.item() == pytest.approx(math.log(2))
    losses.sum().backward()
    assert chosen.grad.item() == pytest.approx(-0.1)
    assert rejected.grad.item() == pytest.approx(0.1)
    assert ref_chosen.grad is None and ref_rejected.grad is None
    with pytest.raises(ValueError, match="strictly positive"):
        dpo_losses(chosen, rejected, ref_chosen, ref_rejected, 0)


def test_dpo_caches_initial_sft_policy_and_keeps_checkpointing_active():
    model = tiny_adapter(attention_dropout=0.1)
    train_adapter(model, [encoded()], "sft", 2, steps=3, batch_size=1,
                  gradient_accumulation=1, lr=0.01, log_every=0)
    pairs = [{"chosen": encoded("x", "7"), "rejected": encoded("x", "8")}]
    disable_dropout(model)
    cache = cache_reference_logprobs(model, pairs, 2)
    frozen_copy = cache.clone()
    model.eval()
    with torch.no_grad():
        current = torch.stack(preference_scores(model, pairs, 2, "cpu"), dim=-1).cpu()
        with model.disable_adapter():
            base = torch.stack(preference_scores(model, pairs, 2, "cpu"), dim=-1).cpu()
    assert torch.equal(cache, current)
    assert not torch.allclose(cache, base), "Loaded SFT reference must not silently revert to the base."
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    train_adapter(model, pairs, "dpo", 2, steps=1, batch_size=1,
                  gradient_accumulation=1, lr=0.001, reference_scores=cache, log_every=0)
    assert model.training and model.is_gradient_checkpointing
    assert torch.equal(cache, frozen_copy)
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if p.requires_grad)
    assert all(m.p == 0 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    assert all(getattr(m, "attention_dropout", 0) == 0 for m in model.modules())


def test_non_cuda_nf4_is_rejected_without_a_model_download():
    with pytest.raises(ValueError, match="this lab requires an NVIDIA CUDA GPU"):
        resolve_device("cpu", "nf4")


def test_saved_adapter_reloads_on_original_base_with_same_logits(tmp_path):
    model = tiny_adapter()
    # Snapshot the original backbone before training. unload() removes LoRA
    # layers without merging them into the base weights.
    original_base = deepcopy(model).unload()
    train_adapter(model, [encoded("x", "7")], "sft", 2, steps=2, batch_size=1,
                  gradient_accumulation=1, lr=0.01, log_every=0)
    batch = collate_completions([encoded("x", "7"), encoded("long", "11")], 2)
    model.eval()
    with torch.inference_mode():
        expected = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert not (tmp_path / "model.safetensors").exists()  # actual adapter-only artifact
    restored = PeftModel.from_pretrained(original_base, tmp_path, is_trainable=False).eval()
    with torch.inference_mode():
        actual = restored(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)
    assert not any(p.requires_grad for p in restored.parameters())


def test_completion_content_cannot_hide_an_embedded_eos():
    class EmbeddedEosTokenizer(LocalChatTokenizer):
        def encode(self, text, *, add_special_tokens):
            return [7, 2, 9]
    with pytest.raises(ValueError, match="contains an EOS token"):
        encode_completion(EmbeddedEosTokenizer(), "x", "7<eos>9", 64)


def test_cuda_precision_uses_bf16_or_conservative_fp32(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert compute_dtype("cuda") == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert compute_dtype("cuda") == torch.float32
    assert compute_dtype("cpu") == compute_dtype("mps") == torch.float32
