"""The mathematical core: completion masks, sequence log-probabilities, SFT, DPO.

No Trainer/TRL abstractions: each optimizer step shows exactly what is averaged.
SFT averages over target tokens, while DPO averages over preference pairs.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from .inference import compute_dtype, prompt_token_ids


def encode_completion(tokenizer, prompt: str, completion: str, max_length: int) -> dict:
    """Mask the prompt by its length, preserving an actual EOS even if pad == EOS.

    We reject overlong rows rather than silently train an empty or cut response.
    Some templates append whitespace after EOS; that is not part of generation.
    """
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("Completion must be a nonempty string.")
    prompt_ids = prompt_token_ids(tokenizer, prompt)
    full = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        tokenize=True, add_generation_prompt=False, return_dict=False,
    ))
    if full[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Chat-template token prefix mismatch: the generation prompt is not an exact "
            "prefix of the prompt+assistant sequence. Use a compatible chat template; "
            "this lab refuses to guess a masking boundary."
        )
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("An EOS token is required to train completion termination.")
    if eos in tokenizer.encode(completion, add_special_tokens=False):
        raise ValueError("Completion content contains an EOS token; remove embedded termination markers.")
    # EOS in the user/system prefix must not terminate the assistant completion.
    response = full[len(prompt_ids):]
    if eos in response:
        full = full[:len(prompt_ids) + response.index(eos) + 1]
    else:
        full.append(eos)
    if len(prompt_ids) >= max_length:
        raise ValueError(
            f"Prompt alone uses {len(prompt_ids)} tokens at --max-length {max_length}; "
            "truncation would remove every completion target. Increase --max-length."
        )
    if len(full) > max_length:
        raise ValueError(
            f"Example uses {len(full)} tokens > --max-length {max_length}. "
            "Increase --max-length or shorten the example; completions are not silently truncated."
        )
    labels = [-100] * len(prompt_ids) + full[len(prompt_ids):]
    if not any(x != -100 for x in labels[1:]):
        raise ValueError("Example has no causally predictable completion targets.")
    return {"input_ids": full, "labels": labels, "prompt_length": len(prompt_ids)}


def read_examples(path: str | Path, tokenizer, method: str, max_length: int) -> list[dict]:
    examples, seen = [], set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row_id = str(row["id"])
                if row_id in seen:
                    raise ValueError(f"duplicate id {row_id!r}")
                seen.add(row_id)
                if method == "sft":
                    item = encode_completion(tokenizer, row["prompt"], row["completion"], max_length)
                    item["id"] = row_id
                elif method == "dpo":
                    chosen = encode_completion(tokenizer, row["prompt"], row["chosen"], max_length)
                    rejected = encode_completion(tokenizer, row["prompt"], row["rejected"], max_length)
                    if chosen["input_ids"] == rejected["input_ids"]:
                        raise ValueError("chosen and rejected completions tokenize identically")
                    item = {"id": row_id, "chosen": chosen, "rejected": rejected}
                else:
                    raise ValueError(f"Unknown method {method!r}")
                item["prompt"] = row["prompt"]
                examples.append(item)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"Dataset is empty: {path}")
    return examples


def collate_completions(examples: list[dict], pad_token_id: int, device="cpu") -> dict:
    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    length = max(len(row["input_ids"]) for row in examples)
    ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(examples):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"])
        labels[i, :n] = torch.tensor(row["labels"])
        attention[i, :n] = 1
    return {"input_ids": ids.to(device), "attention_mask": attention.to(device),
            "labels": labels.to(device)}


def completion_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed completion log-probabilities and counts, one value per row.

    logits at position t predict labels at t+1. Float32 log-softmax protects the
    likelihood calculation when the model forward uses CUDA BF16.
    """
    targets = labels[:, 1:]
    mask = targets != -100
    counts = mask.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("A batch row has zero unmasked completion targets.")
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    selected = logp.gather(-1, targets.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(dim=-1), counts


def dpo_losses(chosen, rejected, reference_chosen, reference_rejected, beta: float):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO --beta must be finite and strictly positive.")
    margin = (chosen - rejected) - (reference_chosen - reference_rejected).detach()
    return -F.logsigmoid(beta * margin), margin


def autocast_context(device: str):
    if device == "cuda" and compute_dtype(device) == torch.bfloat16:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def disable_dropout(model):
    """Keep training/checkpointing enabled while making DPO likelihoods deterministic.

    Llama uses a float attention_dropout in functional attention, whereas PEFT
    adapters use nn.Dropout modules. Handle both rather than only calling eval().
    """
    dropout_attributes = ("attention_dropout", "dropout", "activation_dropout",
                          "attn_dropout", "resid_dropout", "embd_pdrop", "attn_pdrop",
                          "resid_pdrop", "hidden_dropout_prob", "attention_probs_dropout_prob")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        for attr in dropout_attributes:
            value = getattr(module, attr, None)
            if isinstance(value, (int, float)):
                setattr(module, attr, 0.0)
            config = getattr(module, "config", None)
            if isinstance(getattr(config, attr, None), (int, float)):
                setattr(config, attr, 0.0)


def _scores(model, examples, pad_id, device):
    batch = collate_completions(examples, pad_id, device)
    with autocast_context(device):
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    scores, counts = completion_logprobs(output.logits, batch["labels"])
    return scores, counts


def preference_scores(model, pairs, pad_id, device):
    # One shared padding width avoids chosen/rejected shape-dependent differences.
    combined = [p["chosen"] for p in pairs] + [p["rejected"] for p in pairs]
    scores, _ = _scores(model, combined, pad_id, device)
    return scores[:len(pairs)], scores[len(pairs):]


def cache_reference_logprobs(model, pairs, pad_id, device="cpu", batch_size=2) -> torch.Tensor:
    """Freeze the initial policy's outputs, including any loaded SFT adapter.

    Cache BEFORE optimizer updates. No reference forward can accidentally follow
    the changing adapter, and a second full base model is unnecessary.
    """
    was_training = model.training
    model.eval()
    cached = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chosen, rejected = preference_scores(model, pairs[start:start + batch_size], pad_id, device)
            cached.extend(torch.stack([chosen, rejected], dim=-1).cpu().tolist())
    model.train(was_training)
    return torch.tensor(cached, dtype=torch.float32)


def evaluate_objective(model, examples, method, pad_id, device="cpu", batch_size=2,
                       beta=0.1, reference_scores=None) -> dict:
    was_training = model.training
    model.eval()
    total_loss, denominator, margins, preferences = 0.0, 0, 0.0, 0.0
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            rows = examples[start:start + batch_size]
            if method == "sft":
                scores, counts = _scores(model, rows, pad_id, device)
                total_loss -= scores.sum().item()
                denominator += counts.sum().item()
            else:
                if reference_scores is None:
                    raise ValueError("DPO evaluation requires cached initial-policy scores.")
                chosen, rejected = preference_scores(model, rows, pad_id, device)
                ref = reference_scores[start:start + len(rows)].to(device)
                losses, margin = dpo_losses(chosen, rejected, ref[:, 0], ref[:, 1], beta)
                total_loss += losses.sum().item()
                denominator += len(rows)
                margins += margin.sum().item()
                preferences += ((margin > 0).float() + 0.5 * (margin == 0).float()).sum().item()
    model.train(was_training)
    result = {"loss": total_loss / denominator, "denominator": int(denominator),
              "averaging": "completion_tokens" if method == "sft" else "preference_pairs"}
    if method == "sft":
        result["completion_perplexity"] = math.exp(result["loss"]) if result["loss"] < 700 else None
    else:
        result.update(preference_margin=margins / denominator,
                      relative_preference_accuracy=preferences / denominator)
    return result


def _batch_indices(size, batch_size, rng):
    while True:
        order = list(range(size))
        rng.shuffle(order)
        for start in range(0, size, batch_size):
            yield order[start:start + batch_size]


def train_adapter(model, examples, method, pad_id, *, steps=30, batch_size=2,
                  gradient_accumulation=2, lr=2e-4, device="cpu", beta=0.1,
                  seed=7, max_grad_norm=1.0, reference_scores=None, log_every=1) -> list[dict]:
    """Run exactly `steps` optimizer updates with correctly weighted microbatches."""
    if min(steps, batch_size, gradient_accumulation) < 1 or lr <= 0:
        raise ValueError("Steps, batch size, accumulation, and learning rate must be positive.")
    if method not in {"sft", "dpo"} or not examples:
        raise ValueError("Training requires method sft/dpo and a nonempty dataset.")
    if method == "dpo" and (reference_scores is None or len(reference_scores) != len(examples)):
        raise ValueError("Cache one initial-policy reference score per preference row before DPO.")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters: load a trainable LoRA adapter.")
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    batches = _batch_indices(len(examples), batch_size, random.Random(seed))
    history, started = [], time.perf_counter()
    # HF gates gradient checkpointing on training mode. Disable stochastic
    # dropout explicitly rather than switching the DPO learner to eval mode.
    if method == "dpo":
        disable_dropout(model)
    model.train()
    for step in range(1, steps + 1):
        window = [next(batches) for _ in range(gradient_accumulation)]
        if method == "sft":
            denominator = sum(sum(x != -100 for x in examples[i]["labels"][1:])
                              for indices in window for i in indices)
        else:
            denominator = sum(len(indices) for indices in window)
        optimizer.zero_grad(set_to_none=True)
        reported_loss = 0.0
        for indices in window:
            rows = [examples[i] for i in indices]
            if method == "sft":
                scores, _ = _scores(model, rows, pad_id, device)
                loss_sum = -scores.sum()
            else:
                chosen, rejected = preference_scores(model, rows, pad_id, device)
                ref = reference_scores[indices].to(device)
                losses, _ = dpo_losses(chosen, rejected, ref[:, 0], ref[:, 1], beta)
                loss_sum = losses.sum()
            loss = loss_sum / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at optimizer step {step}; no checkpoint saved.")
            reported_loss += loss.detach().item()
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        record = {"step": step, "loss": reported_loss, "denominator": denominator,
                  "grad_norm": float(grad_norm), "elapsed_seconds": time.perf_counter() - started}
        history.append(record)
        if log_every and (step % log_every == 0 or step == steps):
            print(json.dumps(record), flush=True)
    return history
