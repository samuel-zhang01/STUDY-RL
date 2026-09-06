# Practical LLM post-training lab

Train a real pretrained language model with LoRA, save its adapter, reload it,
and measure its answers. Then continue with DPO and examine what quantisation
changes. This optional track accompanies weeks 18–21; start after the policy
gradient and PPO lessons in weeks 7–9.

The default is [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
It is already instruction-tuned: “base” below means the **starting checkpoint
before this lab's adaptation**, not a raw pretrained model. We use small,
locally generated addition examples to make targets, masks, rewards, and data
splits inspectable. The trainer performs real transformer forward/backward
passes and optimises real PEFT adapters. Results on this task do not establish
general reasoning ability.

## Lessons

| Lesson | What you should be able to explain and do |
|---|---|
| [1. LLMs as policies](lessons/01-llms-as-policies.md) | Map prompts, prefixes, tokens, EOS, and rewards onto an MDP; distinguish SFT, DPO, and online RL |
| [2. Real LoRA training](lessons/02-real-lora-training.md) | Derive the low-rank update, count trainable parameters, inspect completion masks, and save/reload an adapter |
| [3. Quantisation and QLoRA](lessons/03-quantisation-and-qlora.md) | Calculate codes, scales, errors, and memory; distinguish uniform int4, NF4, and compute precision |
| [4. Preferences and rewards](lessons/04-preferences-and-rewards.md) | Derive DPO, calculate its gradient direction, and connect preference learning to GRPO/RLOO and verifiers |
| [5. Running a training project](lessons/05-running-a-training-project.md) | Design splits, compare checkpoints, diagnose failures, and report results with evidence |

Work through the [numerical workbook](workbook.ipynb) before a longer training run.
For your **RTX 5090**, use the [NVIDIA setup and training guide](NVIDIA.md).

## Start learning without training a model

Read lessons 1 and 2, predict the worked examples, then work through the eight
numerical workbook exercises. Continue with lessons 3–5 and read the corresponding
functions in the trainer. The workbook and unit tests use small local tensors;
they do not download or train a pretrained language model. The training commands
below are optional exercises for when you are ready to use your hardware.

With the lab environment installed, from the repository root:

```bash
make llm-tests PY=python
make llm-notebook PY=python
```

The notebook target creates an ignored `workbook.executed.ipynb`; you can also
read the committed notebook directly. Virtual environments, model checkpoints,
and `runs/` outputs are ignored by Git. The small synthetic fixtures and written
lessons are intended to be committed.

## Install a small environment

Use Python 3.12. From the repository root on macOS/Linux:

```bash
python3.12 -m venv .venv-llm
source .venv-llm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r llm-lab/requirements.txt
cd llm-lab
python -m llm_lab.doctor
```

On NVIDIA, install the CUDA PyTorch wheel **before** the requirements, following
[NVIDIA.md](NVIDIA.md). Keep that environment separate from the course's optional
MuJoCo/Atari dependencies. All commands below run inside `llm-lab` with the
environment activated. The initial model download needs internet access and
roughly a few hundred MB of cache; training is local and does not call a paid API.

| Device | `--device` | Precision in this lab | Quantisation option |
|---|---|---|---|
| Apple Silicon | `mps` | Float32 LoRA | `none` |
| CPU | `cpu` | Float32 LoRA; slower | `none` |
| NVIDIA | `cuda` | BF16 when supported, otherwise FP32 | `none` or `nf4` |

`auto` chooses CUDA, then MPS, then CPU. Our NF4 implementation targets NVIDIA
CUDA. Other quantisation backends require their own integration; installing a
quantisation library alone does not change this trainer's supported devices.

## Run the full small experiment

1. Verify the data and run the tests. The committed fixtures can be reproduced
   exactly with the default seed. Train/validation/test contain 192/32/32 examples;
   an operand pair and its reversal always stay in the same split.

   ```bash
   python -m llm_lab.data --output data
   python -m pytest tests -q
   python -m llm_lab.quantisation
   ```

2. Evaluate the starting model with fixed greedy decoding:

   ```bash
   python -m llm_lab.evaluate --data data/sft_test.jsonl --output runs/base/evaluation.json
   ```

3. Train an SFT adapter. `--steps` counts **optimizer updates**, so batch size 2
   and accumulation 2 consume four examples per update. Try 2 steps first for a
   setup check; use 30 for the worked experiment. Output directories must be new.

   ```bash
   python -m llm_lab.train --method sft --data data/sft_train.jsonl --eval-data data/sft_validation.jsonl --output runs/sft --steps 30 --batch-size 2 --gradient-accumulation 2 --lr 0.0005
   python -m llm_lab.inference --adapter runs/sft --prompt "Compute 3 + 4. Reply with just the integer."
   python -m llm_lab.evaluate --adapter runs/sft --data data/sft_test.jsonl --output runs/sft/evaluation.json
   ```

4. Continue that adapter with preference optimisation. The reference scores are
   computed **once from the initial SFT policy** and then remain frozen. This is
   offline DPO; the script does not collect online rollouts.

   ```bash
   python -m llm_lab.train --method dpo --adapter runs/sft --data data/preferences_train.jsonl --eval-data data/preferences_validation.jsonl --output runs/dpo --steps 20 --batch-size 2 --gradient-accumulation 2 --lr 0.00005 --beta 0.1
   python -m llm_lab.evaluate --adapter runs/dpo --data data/sft_test.jsonl --output runs/dpo/evaluation.json
   ```

These are short teaching runs, not tuned hyperparameters. Use validation for
changes to learning rate, rank, steps, or data; inspect the test comparison after
those choices. The committed [validation report](VALIDATION.md) states exactly
which paths were executed and includes the observed results.

## What the code teaches

| File | Inspect this mechanism |
|---|---|
| [training.py](llm_lab/training.py) | Completion labels, causal shift, token sums, DPO margins, and gradient accumulation |
| [train.py](llm_lab/train.py) | Device selection, adapter creation, training loop, evaluation, checkpoint metadata |
| [inference.py](llm_lab/inference.py) | Load the recorded base revision plus adapter; apply the same chat template |
| [data.py](llm_lab/data.py) | Reproducible group splits and a strict integer verifier |
| [evaluate.py](llm_lab/evaluate.py) | Generate held-out answers; measure correctness and format separately |
| [quantisation.py](llm_lab/quantisation.py) | Actual integer encoding, packing, reconstruction, and error calculations |
| [doctor.py](llm_lab/doctor.py) | Inspect hardware and execute forward/backward kernels, including optional NF4 |

The training loop is deliberately visible PyTorch rather than hidden behind a
large trainer API. It uses the actual [PEFT LoRA implementation](https://huggingface.co/docs/peft/developer_guides/lora)
and [Transformers model loading](https://huggingface.co/docs/transformers/models).
The small CPU tests create random models locally; they never download weights.

Each completed run saves an adapter and tokenizer, `run_config.json`, and
`metrics.json`. The config records the base revision, dataset hashes, seed,
software, device, and hyperparameters. The adapter is **not a standalone model**;
the inference command restores its recorded base. Adapter checkpoints support
inference and a fresh follow-on training stage; they are not exact interrupted-run
resumption with optimizer/RNG state.

Evaluation reports include every generated answer, the data hash, decoding
settings, and counts. A correct integer receives credit only if the model also
terminated; a response cut off at the generation budget receives zero. A format
score alone is easy to exploit by always returning a number.

## Extend the task yourself

Replace the synthetic rows with your own reviewed `prompt`/`completion` records
and, for DPO, `prompt`/`chosen`/`rejected` records. Keep IDs stable, split by the
underlying source or problem, and keep evaluation independent. The included
integer verifier only applies to this addition task; design a suitable evaluator
for another domain.

For online verifiable-reward RL, continue to
[week 20](../week-20-grpo-rloo-rlvr/teaching_guide.md). Its small runnable project
and lesson 4 explain group baselines and reward failures. This lab adds real
pretrained-model SFT/DPO; full pretrained-model PPO/GRPO training is a further
extension, not an implemented command here.
