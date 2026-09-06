"""Load an actual base model plus a saved LoRA adapter and generate a completion.

The checkpoint's run_config.json records the resolved base-model revision. An
adapter is not a standalone language model: reload that same base underneath it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from . import DEFAULT_MODEL


def resolve_device(device: str = "auto", quantization: str = "none") -> str:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("--device must be auto, cpu, mps, or cuda")
    if quantization not in {"none", "nf4"}:
        raise ValueError("--quantization must be none or nf4")
    # Reject this before downloading a model or importing bitsandbytes.
    if quantization == "nf4" and device != "cuda":
        raise ValueError(
            "NF4 QLoRA in this lab requires an NVIDIA CUDA GPU. Use "
            "--quantization none on CPU/MPS, or --device cuda on your NVIDIA machine."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable to this PyTorch installation. On your NVIDIA machine, "
            "install a CUDA-enabled PyTorch build and check torch.cuda.is_available()."
        )
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu or a supported Apple GPU.")
    if quantization == "nf4" and importlib.util.find_spec("bitsandbytes") is None:
        raise ValueError("NF4 requires bitsandbytes on the CUDA machine: pip install bitsandbytes")
    return device


def compute_dtype(device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    # Keep the teaching loop free of FP16 loss-scaling/overflow recovery.
    # RTX 5090 uses BF16; older devices use the more conservative FP32 path.
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


def resolve_model_source(model_name=None, adapter_path=None, revision=None) -> tuple[str, str | None]:
    """Use the recorded base revision when continuing or evaluating an adapter."""
    source = model_name or DEFAULT_MODEL
    if adapter_path:
        folder = Path(adapter_path)
        if not folder.is_dir():
            raise ValueError(f"Adapter directory does not exist: {folder}")
        config_path = folder / "run_config.json"
        if config_path.exists():
            saved = json.loads(config_path.read_text())
            recorded_model = saved["base_model_name"]
            recorded_revision = saved.get("base_revision")
        else:
            config = json.loads((folder / "adapter_config.json").read_text())
            recorded_model = config["base_model_name_or_path"]
            recorded_revision = config.get("revision")
        if model_name not in (None, DEFAULT_MODEL, recorded_model):
            raise ValueError(f"Adapter requires base {recorded_model!r}, not {model_name!r}.")
        if revision and recorded_revision and revision != recorded_revision:
            raise ValueError("Requested revision differs from the adapter's recorded base revision.")
        source = recorded_model
        revision = recorded_revision or revision
    return source, revision


def load_base_components(model_name=None, adapter_path=None, device="auto",
                         quantization="none", revision=None, cpu_threads=4):
    """Load onto one explicit device; never use inference-style automatic offload."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = resolve_device(device, quantization)
    if cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(cpu_threads)
    source, revision = resolve_model_source(model_name, adapter_path, revision)
    dtype = compute_dtype(device)
    kwargs = {"revision": revision, "dtype": dtype, "trust_remote_code": False}
    if quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if quantization == "none":
        model.to(device)
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    tokenizer_source = str(adapter_path) if adapter_path else source
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=False,
        **({} if adapter_path else {"revision": resolved_revision}),
    )
    if not tokenizer.chat_template:
        raise ValueError("This lab requires a tokenizer with a chat template; select an instruct/chat model.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad token or EOS token; none is configured.")
        tokenizer.pad_token = tokenizer.eos_token
    info = {
        "base_model_name": source, "base_revision": resolved_revision,
        "revision_kind": "hub_commit" if resolved_revision else "local_or_unresolved",
        "device": device, "compute_dtype": str(dtype), "quantization": quantization,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "cpu_threads": cpu_threads,
    }
    return model, tokenizer, info


def load_inference_model(model_name=DEFAULT_MODEL, adapter_path=None, device="auto",
                         quantization="none", revision=None, cpu_threads=4):
    """Return (model, tokenizer, metadata), optionally restoring an adapter."""
    model, tokenizer, info = load_base_components(
        model_name, adapter_path, device, quantization, revision, cpu_threads
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer, info


def prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A prompt must be a nonempty string.")
    return list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_dict=False,
    ))


def generate_completion_details(model, tokenizer, prompt: str,
                                max_new_tokens=32, temperature=0.0) -> dict:
    if max_new_tokens < 1 or temperature < 0:
        raise ValueError("max_new_tokens must be positive and temperature nonnegative.")
    ids = prompt_token_ids(tokenizer, prompt)
    device = next(model.parameters()).device
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if temperature > 0:
        kwargs["temperature"] = temperature
    model.eval()
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **kwargs)
    generated = output[0, len(ids):].tolist()
    eos_ids = tokenizer.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "generated_tokens": len(generated),
        "terminated": any(token in eos_ids for token in generated),
    }


def generate_completion(model, tokenizer, prompt: str, max_new_tokens=32, temperature=0.0) -> str:
    return generate_completion_details(model, tokenizer, prompt, max_new_tokens, temperature)["text"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter")
    parser.add_argument("--revision")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--quantization", default="none", choices=["none", "nf4"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        model, tokenizer, _ = load_inference_model(
            args.model, args.adapter, args.device, args.quantization, args.revision, args.cpu_threads
        )
        print(generate_completion(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
