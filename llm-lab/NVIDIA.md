# LoRA and QLoRA on your RTX 5090

The desktop RTX 5090 uses Blackwell and has 32 GB GDDR7. Start by running the
135M model to check the complete pipeline, then increase model size and context
length separately. These are starting configurations, not measured VRAM or
throughput promises. [NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/).

## 1. Install a Blackwell-compatible environment

Use Python 3.12 on Linux, Windows, or WSL2 with GPU access. The commands below
are shell-independent except for virtual-environment activation. From the repo
root:

```bash
python -m venv .venv-llm
```

Activate with `source .venv-llm/bin/activate` on Linux/WSL2, or
`.venv-llm\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
nvidia-smi
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r llm-lab/requirements-nvidia.txt
python -m pip check
cd llm-lab
python -m llm_lab.doctor --device cuda
python -m llm_lab.doctor --device cuda --check-nf4
python -m pytest tests -q
```

This CUDA 13.0 wheel targets current Blackwell hardware. Use an up-to-date
NVIDIA driver: PyTorch documents minimum drivers 580.65.06 on Linux and 580.88
on Windows for its CUDA 13.0 builds. PyTorch introduced Blackwell support with
CUDA 12.8, but stopped publishing that wheel in its 2.12 release, so old advice
to install any CUDA wheel or always use `cu128` is insufficient. Check the
[PyTorch release guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/)
and [installation selector](https://pytorch.org/get-started/locally/) if your
environment differs.

The lab pins the Transformers/PEFT APIs and `bitsandbytes==0.50.2`. The official
[bitsandbytes build matrix](https://huggingface.co/docs/bitsandbytes/main/en/installation)
includes Blackwell `sm120` in CUDA 13 builds. A matching wheel and driver still
need a real kernel check: `doctor` executes forward **and backward**, and the
NF4 check propagates gradients through a frozen 4-bit linear layer.

The doctor report should identify your GPU, VRAM, compute capability `(12, 0)`,
CUDA runtime, and BF16 support. The training code selects BF16 on supported
CUDA hardware and records the actual selection. Do not use an environment
copied from macOS; recreate it and transfer the data/checkpoint files instead.

## 2. Run ordinary LoRA first

Inside `llm-lab`, with the NVIDIA environment active:

```bash
python -m llm_lab.train --device cuda --method sft --data data/sft_train.jsonl --eval-data data/sft_validation.jsonl --output runs/5090-lora-smoke --steps 2 --batch-size 1 --gradient-accumulation 2
python -m llm_lab.inference --device cuda --adapter runs/5090-lora-smoke --prompt "Compute 3 + 4. Reply with just the integer."
```

After this succeeds, run the [README experiment](README.md#run-the-full-small-experiment)
with `--device cuda`. Ordinary LoRA freezes the original weights and learns
floating-point adapters; it does not quantise the base.

## 3. Run actual 4-bit QLoRA

```bash
python -m llm_lab.train --device cuda --quantization nf4 --method sft --data data/sft_train.jsonl --eval-data data/sft_validation.jsonl --output runs/5090-qlora-smoke --steps 2 --batch-size 1 --gradient-accumulation 2 --gradient-checkpointing
python -m llm_lab.evaluate --device cuda --quantization nf4 --adapter runs/5090-qlora-smoke --data data/sft_test.jsonl --output runs/5090-qlora-smoke/evaluation.json
```

This path loads the frozen base through `BitsAndBytesConfig` with NF4 and
double quantisation, prepares it for low-bit training, and attaches trainable
LoRA weights. The matrix operations and adapters use a floating compute dtype;
“4-bit training” does not mean that gradients, activations, or optimizer states
are all 4-bit. [PEFT quantisation training](https://huggingface.co/docs/peft/developer_guides/quantization).

Continue a QLoRA SFT adapter with DPO using the **same** quantisation choice:

```bash
python -m llm_lab.train --device cuda --quantization nf4 --method dpo --adapter runs/5090-qlora-smoke --data data/preferences_train.jsonl --eval-data data/preferences_validation.jsonl --output runs/5090-qlora-dpo --steps 2 --batch-size 1 --gradient-accumulation 2 --lr 0.00005 --beta 0.1 --gradient-checkpointing
```

The reference is cached from the complete starting policy, including the SFT
adapter. DPO's chosen and rejected sequences use more work per pair than one
SFT example; treat batch sizes for these methods separately.

## 4. Scale the model after the small run works

| Experiment | Starting model | Batch / accumulation / max length | Purpose |
|---|---|---|---|
| LoRA setup | SmolLM2-135M-Instruct | 1 / 2 / 192 | Validate setup, masks, save, reload |
| LoRA next step | SmolLM2-1.7B-Instruct | 1 / 8 / 512 | Adapt a larger model without quantisation |
| QLoRA next step | Qwen2.5-7B-Instruct | 1 / 8 / 512 | Inspect NF4 memory/quality tradeoffs |

Example commands, still using the tiny fixtures **only for pipeline validation**:

```bash
python -m llm_lab.train --device cuda --model HuggingFaceTB/SmolLM2-1.7B-Instruct --method sft --data data/sft_train.jsonl --eval-data data/sft_validation.jsonl --output runs/5090-1p7b-lora --steps 2 --batch-size 1 --gradient-accumulation 8 --max-length 512 --gradient-checkpointing
python -m llm_lab.train --device cuda --quantization nf4 --model Qwen/Qwen2.5-7B-Instruct --method sft --data data/sft_train.jsonl --eval-data data/sft_validation.jsonl --output runs/5090-7b-qlora --steps 2 --batch-size 1 --gradient-accumulation 8 --max-length 512 --gradient-checkpointing
```

These download substantially larger checkpoints than the default model. Use
reviewed task-specific data before spending time on a longer run. The choices
have published chat templates and model cards:
[SmolLM2 1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct),
[Qwen2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
Actual memory depends on sequence lengths, attention kernels, quantisation
metadata, and adapter targets. Inspect recorded peak CUDA memory before
increasing batch size; 32 GB is not all available for weights.

## 5. Move adapters between machines

Copy the entire run directory, including adapter weights/config, tokenizer
files, and `run_config.json`, to the same relative location on the other machine.
The loader uses the recorded base model and Hub commit; it downloads or reuses
that base locally. For example, after transferring `runs/sft` to your Mac:

```bash
python -m llm_lab.inference --device mps --adapter runs/sft --prompt "Compute 3 + 4. Reply with just the integer."
```

Ordinary LoRA adapters are portable between these devices. A QLoRA adapter can
also be attached to the matching full-precision base on the Mac using
`--quantization none`, but this changes the effective base weights and memory
use. Re-evaluate the result; do not claim bit-identical QLoRA outputs. Load with
`--quantization nf4` on NVIDIA when evaluating the original quantised setup.
The saved adapter supports a new training stage; optimizer/RNG state for exact
mid-run resume is not saved.

## Troubleshoot by the failing layer

| Symptom | Check/action |
|---|---|
| `nvidia-smi` fails | Resolve the host driver or WSL GPU visibility before Python packages |
| CUDA unavailable in `doctor` | Confirm the active Python environment and reinstall the CUDA PyTorch wheel |
| `no kernel image` / `sm_120` error | Use the Blackwell-compatible wheel above; changing only a local CUDA toolkit does not replace compiled PyTorch kernels |
| CUDA check passes, NF4 check fails | Inspect bitsandbytes version and its CUDA backend; try ordinary LoRA to isolate the quantisation path |
| Out of memory | Batch size 1, shorter max length, checkpointing, then smaller model or NF4; accumulation preserves examples/update without keeping all microbatches in memory |
| Loss improves, answers deteriorate | Compare exact-match and format metrics; inspect target masks, rejected labels, length cutoffs, and overfitting |

The local validation machine has MPS and no NVIDIA device. The CUDA/NF4 paths
are implemented and their device selection is tested, but **RTX 5090 training
and larger-model VRAM have not been measured here**. Run the smoke commands
above on your 5090 before treating that hardware path as validated.
