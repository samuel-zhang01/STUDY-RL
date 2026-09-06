# What was checked

Checked on 6 September 2026 with Python 3.12.14, PyTorch 2.14.0,
Transformers 5.16.1, PEFT 0.20.0, and Accelerate 1.14.0 on Apple Silicon/MPS.
`pip check` found no dependency conflicts in the validation environment.

## Lightweight checks

- **83 tests passed**, without downloading pretrained weights. They cover
  packed quantisation and error bounds, disjoint data splits, strict reward
  parsing, completion/EOS/padding masks, unequal-length gradient accumulation,
  frozen base weights, adapter save/reload, fixed SFT reference scores for DPO,
  embedded termination markers, precision selection, and device-selection failures. Tiny randomly initialised models exercise the
  training mechanics in seconds.
- **All 35 cells of the eight-exercise numerical workbook executed** in a fresh
  kernel. Its examples use local tensors; the workbook does not train a
  pretrained language model.
- Worked LoRA gradients, DPO margins, group advantages, quantisation codes,
  memory figures, and local lesson links were checked.
- The MPS hardware check executed forward and backward operations successfully.
- One benign PEFT warning in the offline checkpoint test concerns the random
  fixture's missing model-card/config location; the reload comparison passes.

Run the lightweight checks yourself from the repo root:

```bash
make llm-tests PY=python
make llm-notebook PY=python
```

## Earlier short adapter checks

Before the request to focus on the guide, two short LoRA checks had already
completed on SmolLM2-135M-Instruct: 30 SFT optimizer updates and 20 DPO updates.
They updated **2,442,240 adapter parameters**, with the starting transformer
weights frozen. No further model-training runs were started after that request.

The base revision was `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
These are illustrative single-seed checks on synthetic addition, not a model
quality benchmark. The table uses 32 pairs held out from this lab's adaptation;
the model's pretraining exposure is unknown.

| Checkpoint | Complete, strictly formatted integer answers | Correct answers under that strict rule |
|---|---:|---:|
| Starting instruct checkpoint | 0 / 32 | 0 / 32 |
| Reloaded SFT adapter | 32 / 32 | 19 / 32 |

The starting model often answered in prose. This result mixes instruction-format
compliance and answer correctness; it does **not** mean it had zero arithmetic
knowledge. Ten starting-model answers reached the generation cap. The scorer
does not reward an answer cut off before EOS.

SFT validation completion-token NLL fell from **2.8150 to 0.4715**.
DPO validation pair loss fell from **0.6931 to 0.5565**, relative to a frozen
initial-SFT reference. DPO generated-answer evaluation was not run, so no DPO
answer-quality improvement is claimed.

Inspect the recorded evidence:

- [Starting-model answers](results/base_evaluation.json)
- [Reloaded SFT answers](results/sft_evaluation.json)
- [SFT configuration and losses](results/sft_training.json)
- [DPO configuration and losses](results/dpo_training.json)

Final metadata, conservative FP32 fallback, and input-validation refinements were verified with the offline
tests; the earlier pretrained-model runs were not repeated after those changes.
Training checkpoints remain under ignored `runs/`; these small JSON reports
are the evidence intended for Git.

## NVIDIA scope

CUDA LoRA, NF4 QLoRA, BF16 selection, and gradient checkpointing are implemented.
The [RTX 5090 guide](NVIDIA.md) includes installation, actual kernel checks,
two-step smoke commands, and cautious larger-model starting settings.
This machine has no NVIDIA GPU, so RTX 5090 execution, CUDA peak memory, and
the 1.7B/7B examples remain unmeasured here. Run those optional commands on your
NVIDIA machine when you choose to start training.
