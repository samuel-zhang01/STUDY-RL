"""Train an actual pretrained causal LM with LoRA SFT or reference-cached DPO.

Run from llm-lab, for example:
python -m llm_lab.train --method sft --data data/sft_train.jsonl \
    --eval-data data/sft_validation.jsonl --output runs/sft --steps 30
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shlex
import time

import torch

from . import DEFAULT_MODEL
from .inference import load_base_components, resolve_device
from .training import cache_reference_logprobs, disable_dropout, evaluate_objective, read_examples, train_adapter


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_metadata(device) -> dict:
    result = {"torch": torch.__version__, "python": platform.python_version(),
              "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(),
              "mps_available": torch.backends.mps.is_available()}
    for name in ("transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    if device == "cuda":
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        result.update(gpu_name=properties.name, gpu_memory_bytes=properties.total_memory,
                      gpu_compute_capability=[properties.major, properties.minor],
                      cuda_bf16_supported=torch.cuda.is_bf16_supported())
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=["sft", "dpo"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="Hub branch/tag/commit; resolved commit is recorded for reload.")
    parser.add_argument("--adapter", help="Continue a saved adapter; DPO caches this initial SFT policy as reference.")
    parser.add_argument("--data", required=True, help="Local JSONL: id,prompt,completion or id,prompt,chosen,rejected.")
    parser.add_argument("--eval-data", help="Held-out JSONL with the same schema.")
    parser.add_argument("--output", required=True, help="New or empty directory for adapter, tokenizer, and metrics.")
    parser.add_argument("--steps", type=int, default=30, help="Number of optimizer updates, not microbatches.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=192, help="Includes template/prompt; overlong rows are rejected.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--quantization", choices=["none", "nf4"], default="none")
    parser.add_argument("--lr", type=float, help="Default: 2e-4 SFT; 5e-5 DPO.")
    parser.add_argument("--beta", type=float, default=0.1, help="Positive DPO inverse reward scale.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="all-linear", help="all-linear or comma-separated layer names.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Trade recomputation for activation memory.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def run(args) -> dict:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    device = resolve_device(args.device, args.quantization)
    if min(args.steps, args.batch_size, args.gradient_accumulation, args.max_length,
           args.lora_r, args.lora_alpha, args.cpu_threads) < 1:
        raise ValueError("Steps, sizes, rank, alpha, and CPU threads must all be positive.")
    if not (0 <= args.lora_dropout < 1):
        raise ValueError("--lora-dropout must be in [0,1).")
    if not (0 < args.beta < float("inf")) or not (0 < args.max_grad_norm < float("inf")):
        raise ValueError("--beta and --max-grad-norm must be finite and positive.")
    args.lr = args.lr if args.lr is not None else (2e-4 if args.method == "sft" else 5e-5)
    if not (0 < args.lr < float("inf")):
        raise ValueError("--lr must be finite and positive.")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new run directory.")
    for path in (args.data, args.eval_data):
        if path and not Path(path).is_file():
            raise ValueError(f"Dataset file does not exist: {path}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, model_info = load_base_components(
        args.model, args.adapter, device, args.quantization, args.revision, args.cpu_threads
    )
    model.config.use_cache = False
    examples = read_examples(args.data, tokenizer, args.method, args.max_length)
    validation = read_examples(args.eval_data, tokenizer, args.method, args.max_length) if args.eval_data else examples
    if args.eval_data:
        shared_ids = {row["id"] for row in examples} & {row["id"] for row in validation}
        shared_prompts = {row["prompt"] for row in examples} & {row["prompt"] for row in validation}
        if shared_ids or shared_prompts:
            raise ValueError(
                f"Training/evaluation overlap: {len(shared_ids)} shared IDs and "
                f"{len(shared_prompts)} shared exact prompts. Provide a separate validation split."
            )
    if args.quantization == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        targets = args.target_modules if args.target_modules == "all-linear" else args.target_modules.split(",")
        config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout, bias="none", target_modules=targets,
                            task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Expected only LoRA parameters to be trainable; inspect your adapter configuration.")
    trainable_count, parameter_count = model.get_nb_trainable_parameters()
    model.print_trainable_parameters()
    config = {
        **vars(args), **model_info, "method": args.method,
        "trainable_parameters": trainable_count, "total_parameters": parameter_count,
        "trainable_fraction": trainable_count / parameter_count,
        "lora_configuration": json.loads(json.dumps(model.peft_config["default"].to_dict(), default=str)),
        "environment": hardware_metadata(device),
        "train_data_sha256": file_sha256(args.data),
        "eval_data_sha256": file_sha256(args.eval_data) if args.eval_data else None,
        "train_examples": len(examples), "eval_examples": len(validation),
        "evaluation_split": "separate_ids_and_exact_prompts" if args.eval_data else "training_data_not_held_out",
        "masking": "exact_chat_prompt_prefix; completion_plus_EOS_only; padding_by_length",
        "truncation": "reject_overlong_rows", "effective_batch_size_max": args.batch_size * args.gradient_accumulation,
        "loss_averaging": "completion_tokens" if args.method == "sft" else "preference_pairs",
    }
    config["base_revision"] = model_info["base_revision"]
    if args.adapter:
        config["initial_adapter_hashes"] = {
            p.name: file_sha256(p) for p in Path(args.adapter).iterdir()
            if p.name in {"adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"}
        }
    command = ["python", "-m", "llm_lab.inference", "--model", config["base_model_name"]]
    if config["base_revision"]:
        command += ["--revision", config["base_revision"]]
    command += ["--adapter", str(output), "--device", device, "--quantization", args.quantization,
                "--prompt", "What is 17 + 8? Reply with only the integer answer."]
    config["inference_command"] = shlex.join(command)
    output.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        # Include reference caching, training, and evaluation, starting with the
        # already-resident base+adapter allocation. This is measured, not estimated.
        torch.cuda.reset_peak_memory_stats()

    reference, eval_reference = None, None
    if args.method == "dpo":
        disable_dropout(model)
        print("Caching frozen initial-policy completion log-probabilities before any update.", flush=True)
        reference = cache_reference_logprobs(model, examples, tokenizer.pad_token_id, device, args.batch_size)
        eval_reference = (cache_reference_logprobs(model, validation, tokenizer.pad_token_id, device, args.batch_size)
                          if args.eval_data else reference)
        reference_data = {
            "reference": "initial_loaded_adapter" if args.adapter else "original_base_with_zero_LoRA_update",
            "train": [{"id": row["id"], "chosen": float(score[0]), "rejected": float(score[1])}
                      for row, score in zip(examples, reference)],
            "validation": [{"id": row["id"], "chosen": float(score[0]), "rejected": float(score[1])}
                           for row, score in zip(validation, eval_reference)],
        }
        (output / "reference_logprobs.json").write_text(json.dumps(reference_data, indent=2) + "\n")
        config["reference"] = reference_data["reference"]
        config["reference_cache_sha256"] = file_sha256(output / "reference_logprobs.json")
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    initial = evaluate_objective(model, validation, args.method, tokenizer.pad_token_id, device,
                                 args.batch_size, args.beta, eval_reference)
    print("Initial evaluation:", json.dumps(initial), flush=True)
    started = time.perf_counter()
    history = train_adapter(
        model, examples, args.method, tokenizer.pad_token_id, steps=args.steps, batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation, lr=args.lr, device=device, beta=args.beta,
        seed=args.seed, max_grad_norm=args.max_grad_norm, reference_scores=reference, log_every=args.log_every,
    )
    training_seconds = time.perf_counter() - started
    final = evaluate_objective(model, validation, args.method, tokenizer.pad_token_id, device,
                               args.batch_size, args.beta, eval_reference)
    model.peft_config["default"].revision = config["base_revision"]
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    metrics = {"method": args.method, "initial_eval": initial, "final_eval": final,
               "train_history": history, "training_seconds": training_seconds,
               "training_evaluation_save_seconds": time.perf_counter() - started,
               "optimizer_steps": len(history), "inference_command": config["inference_command"],
               "config": config}
    if device == "cuda":
        metrics["cuda_memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "measurement": "resident_model_then_reference_cache_training_and_evaluation",
        }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print("Final evaluation:", json.dumps(final), flush=True)
    print(f"Saved adapter and tokenizer to {output}\nReload with:\n{config['inference_command']}", flush=True)
    return metrics


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
