# Lesson 5 — Run a post-training experiment you can explain and reproduce

**Outcome:** take the lab from a successful command to an interpretable experiment:
clean splits, a recorded configuration, a verified checkpoint, and an honest comparison.
Use [the README](../README.md) for installation, current CLI commands, and actual measured results.
Complete [LoRA](02-real-lora-training.md), [quantisation](03-quantisation-and-qlora.md), and
[preferences/rewards](04-preferences-and-rewards.md) first.
For the RL connections revisit [week 9](../../week-09-ppo/teaching_guide.md),
[week 18](../../week-18-rlhf-ppo-llm/teaching_guide.md),
[week 19](../../week-19-dpo-zoo/teaching_guide.md), and
[week 21](../../week-21-reward-hacking-oversight/teaching_guide.md).

## Analogy: keep a laboratory notebook, not just the best photograph

A photograph shows that one experiment produced something interesting.
A laboratory notebook records what was changed, how it was measured, and what else happened.

| Experiment record | Training-project counterpart |
|---|---|
| Material batch | Dataset version, hashes, and split membership |
| Equipment and settings | Model revision, device, dtype, quantisation, optimiser configuration |
| Treatment | SFT/DPO objective, adapter targets, rank, alpha, learning rate, update count |
| Measurement protocol | Prompt template, decoding rule, verifier, held-out examples |
| Stored specimen | Adapter checkpoint plus its base-model identity |
| Replication | Repeated run under the same recorded configuration |

The analogy's limit is determinism: equal seeds do not guarantee identical results across kernels,
devices, and library versions. Reproduction may mean agreement within a stated numerical or statistical tolerance.
PyTorch documents this limitation in its [reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Define one question before opening a training curve

For this lab, a useful question is:

> On held-out synthetic addition prompts, does task-specific SFT improve correctly formatted
> exact answers relative to the initial instruction-tuned checkpoint, and what changes after DPO?

That gives a task, comparison, and outcome. It does not presume improvement.
A 30-step run is a practical mechanics exercise, not evidence of general arithmetic mastery,
reasoning ability, robust alignment, or production readiness.
“Base” in a comparison table means the supplied initial checkpoint; our default already has instruction post-training.

## Split and deduplicate before fitting anything

The lab uses JSONL records: an SFT record contains `id`, `prompt`, and `completion`;
a preference record contains `id`, `prompt`, `chosen`, and `rejected`. IDs must be unique within each file.
The demonstration preference is the correct integer versus an incorrect integer.
Read the generated files and check the schema before training on a custom dataset.

1. Decide the unit that must stay in one split: exact prompt, underlying problem, document,
   user, or conversation. For addition, consider keeping operand-pair variants together.
2. Deduplicate on the chosen unit before creating train/validation/test sets.
3. Use training data for SFT and DPO parameter updates.
4. Use validation data for learning-rate, checkpoint, and method choices.
5. Reserve `data/sft_test.jsonl` for a final comparison after those choices. The supplied
   train/validation/test sets contain 192/32/32 problems, grouped by unordered operand pair.
   Custom experiments with only train and validation files should label their results as validation performance.

A held-out string is not necessarily a held-out problem. `7+5` and `5+7` differ as text
but can test almost the same skill. A distribution-shift test might hold out operand ranges
or carry patterns; label that split separately from an ordinary random split.
Reusing training prompts between SFT and DPO is a valid staged design when stated explicitly.
Leaking validation answers into either stage is a different issue.

### Worked example: leakage can dominate a reported score

Suppose a validation file has 100 records, but 20 are duplicates of training problems.
A model solves all 20 leaked records and 40 of the remaining 80 unseen problems.

**Predict:** the aggregate score will overstate performance on unseen problems.

1. Reported score on the file: `(20+40)/100=60%`.
2. Score on the genuinely unseen subset: `40/80=50%`.
3. The ten-point difference comes entirely from changing the evaluation mixture.
4. If another run solves only 5 leaked examples but the same 40 unseen problems,
   its aggregate is `45%`, even though both runs score `50%` on unseen problems.

**Interpret:** a large “improvement” can be about overlap rather than transfer.
These are constructed numbers, not measured results from the lab.
Deduplicate before evaluating; do not remove difficult validation records after seeing their predictions.

## Make the first run deliberately inspectable

Use the README's short SFT command first. Its main interface is

```bash
python -m llm_lab.train --method sft \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --data data/sft_train.jsonl --eval-data data/sft_validation.jsonl \
  --output runs/sft --steps 30 --device auto
```

Run it from `llm-lab` in the documented environment.
Then initialise DPO from that saved adapter:

```bash
python -m llm_lab.train --method dpo \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --adapter runs/sft \
  --data data/preferences_train.jsonl --eval-data data/preferences_validation.jsonl \
  --output runs/dpo --steps 30 --device auto
```

The DPO reference is the initial SFT policy and is cached before optimisation.
Keep SFT and DPO output directories separate so the first stage remains a reviewable artifact.
The [README](../README.md) is authoritative if the command interface changes.

The local CPU/MPS route uses unquantised FP32 LoRA. NVIDIA CUDA LoRA and optional
CUDA NF4 QLoRA are also supported code paths; inspect the actual GPU name, available VRAM,
CUDA/PyTorch compatibility, and BF16 capability using the README's hardware checks.
Do not infer hardware capabilities from an ambiguous model number.
A CUDA code path is not a claim that it has been executed on your particular GPU;
consult the recorded validation results. [The RTX 5090/NVIDIA guide](../NVIDIA.md) contains the
actual setup checks; lesson 3 explains the NF4 route and its limitations.

## Record configuration separately from results

The trainer writes `run_config.json`, `metrics.json`, adapter files, and tokenizer files;
DPO additionally writes `reference_logprobs.json`.
Treat these as the beginning of an experiment record and retain the fields needed for your claim:

| Record | Why it matters |
|---|---|
| Base model ID and resolved revision | An adapter is relative to particular base weights |
| Dataset hashes and split definition | Same filenames can contain different data |
| Tokenizer and chat template | Formatting changes the sequence the model actually sees |
| Seed, package versions, device, dtype | Different numerical execution can change results |
| LoRA target modules, rank, alpha, dropout | Determines which corrections the model can learn |
| Objective, beta for DPO, learning rate, batch/accumulation, steps | Defines the optimisation experiment |
| Maximum sequence/completion lengths and masks | Defines what is supervised and what is truncated |
| Decoding and evaluation rule | Separates distribution changes from decoding changes |

Inspect the saved fields; if a custom workflow omits provenance, record it alongside the run.
For a tiny experiment, save a small table of prompt, expected answer, generated completion,
parsed result, and metric decision. That table often explains more than one average loss curve.

## Compare all three policies under one evaluation protocol

Evaluate the initial checkpoint, SFT adapter, and DPO adapter on the same held-out prompts.
Use the same chat template, decoding algorithm, maximum new tokens, EOS handling, and verifier.
Prefer deterministic decoding for the first comparison; if you sample, record temperature,
number of samples, and seeds and report the corresponding sampled metric.

Distinguish these quantities:

- **Completion NLL:** likelihood of the provided correct answer under teacher forcing.
- **Preference accuracy/margin:** how the model scores chosen versus rejected examples.
- **Generated exact-answer accuracy:** whether the generated answer satisfies the specified correctness rule.
- **Format compliance:** whether the answer follows the requested integer-only format.
- **Completion length/truncation:** whether apparent gains accompany unwanted verbosity or incomplete answers.

A DPO loss is not numerically comparable with an SFT loss. A preference margin is not a probability
that an answer is correct. Compare behavioural metrics to answer the project question.

### Worked example: separate format from correctness

On 20 held-out addition prompts, suppose a model produces 16 integer-only responses.
Of those 16, 12 are correct. The remaining four responses include unwanted explanations.
Suppose one of those four contains the correct final integer.

**Predict:** different verifiers will produce different success rates.

1. Integer-only format compliance: `16/20=80%`.
2. Correctness conditional on integer-only formatting: `12/16=75%`.
3. Strict correct-and-integer-only score: `12/20=60%`.
4. A permissive final-integer parser would score `(12+1)/20=65%` in this example.
5. Reporting only `75%` hides format failures by changing the denominator.

**Interpret:** say which metric is primary before choosing a checkpoint.
If the task requires only an integer, the permissive parser changes the task contract.
The example is an evaluation design exercise. The actual `score_completion` in
[data.py](../llm_lab/data.py) requires one ASCII integer (surrounding whitespace is allowed),
and exact match compares its canonical text with the expected answer.
[evaluate.py](../llm_lab/evaluate.py) also requires a real termination token: a response
cut off at the token budget scores zero even if its visible prefix looks correct.

## Know what your checkpoint can do

For inference, load the recorded base checkpoint and then the adapter, with its tokenizer/template.
Use evaluation mode and the README's reload command. Check a fixed batch's logits or deterministic
outputs before and after save/reload to confirm you are actually using the trained adapter.
[PEFT's checkpoint documentation](https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint)
explains why adapter checkpoints require their associated base model.

An adapter export is different from a merged full-model export.
A merged model incorporates the correction into base weights and may have different storage or
quantisation implications; do not relabel an adapter folder as a complete model.
Do not assume loading an adapter resumes an optimiser, learning-rate schedule, or RNG stream.
Exact continuation requires those states as well; this lab's adapter artifacts primarily support
inference and starting another post-training stage.

## Choose small ablations that answer a specific question

Change one factor while keeping data and evaluation fixed:

| Change | Question | Evidence to keep |
|---|---|---|
| Rank 4 versus 8 | Does more adapter capacity help here? | Trainable count, held-out score, memory |
| Short versus longer training | Is the model learning or overfitting? | Train/validation losses and generated answers |
| SFT versus SFT then DPO | What does preference training add? | Same held-out generations and preference diagnostics |
| CUDA LoRA versus NF4 QLoRA | What does base quantisation cost/save? | Actual peak memory, duration, reload behaviour, task score |

Do not compare different methods at a single noisy seed and declare a general winner.
Use a small repeated-seed comparison before making a performance claim.
If a run only establishes that gradients flow and an adapter reloads correctly, report precisely that.
Once this small project is interpretable, the next extension is a new task or a carefully specified
online verifier loop—not an automatic assumption that a larger model solves the remaining measurement problems.

## Self-checks

<details>
<summary>1. Your validation score improved after five hyperparameter sweeps. Is that an untouched test result?</summary>

No. The validation data helped select the configuration. Evaluate the selected configuration once on a
separate untouched test set for that claim, or label the result as validation performance.

</details>

<details>
<summary>2. What does a successful adapter reload establish?</summary>

That the saved parameter correction, base model, tokenizer, and execution path reconstruct the intended
model behaviour within the chosen tolerance. It does not establish improved task quality or exact optimiser resumption.

</details>

<details>
<summary>3. Can lower DPO loss alone justify a claim of better arithmetic?</summary>

No. It establishes better fit to the scored preference pairs under that objective.
Arithmetic performance requires held-out generated-answer evaluation, with correctness and formatting defined separately.

</details>
