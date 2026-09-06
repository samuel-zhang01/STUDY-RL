# Lesson 2 — Real LoRA: a small trainable update inside a pretrained transformer

**Outcome:** derive the adapter forward pass and gradients, calculate its parameter/memory cost,
and train, save, and reload an actual adapter using the lab.
Start with [lesson 1](01-llms-as-policies.md). Basic matrix multiplication and backpropagation are enough.
[Week 7](../../week-07-policy-gradients/teaching_guide.md) provides gradient intuition;
[week 18](../../week-18-rlhf-ppo-llm/teaching_guide.md) connects it to language-model training.

## Analogy: adjust a mixing desk through a small control panel

A mixing desk already maps many input signals to many outputs.
Instead of replacing every connection, add a small panel that combines inputs into a few
adjustment signals and sends those signals back to the outputs.

| Mixing-desk part | LoRA quantity |
|---|---|
| Existing connections | Frozen base weight `W_0` |
| Few adjustment channels | Rank `r` |
| Input-to-adjustment connections | Matrix `A` |
| Adjustment-to-output connections | Matrix `B` |
| Gain applied to the extra panel | Scale `s=alpha/r` |

The panel changes the function even though the original connections stay fixed.
The analogy's limit is capacity: a rank-`r` update cannot represent every arbitrary full-rank
matrix update at a single layer. Across layers, the network remains nonlinear and expressive.
LoRA does not mean only the final output head learns.
[The original LoRA paper](https://arxiv.org/abs/2106.09685) introduces this frozen-weight plus low-rank-update construction.

## Write down the shapes before the code

Use column-vector convention for one activation `x`:

$$W_0\in\mathbb R^{d_{out}\times d_{in}},\quad
A\in\mathbb R^{r\times d_{in}},\quad
B\in\mathbb R^{d_{out}\times r},\quad
h=W_0x+sB(Ax),\quad s=\alpha/r.$$

The effective matrix is `W_eff=W_0+sBA`; the adapter update has rank at most `r`.
A batched PyTorch linear layer stores the same weight shape but evaluates row-batched
activations as `X @ W_0.T + s (X @ A.T) @ B.T`.
Transposes change with the data convention, not with the underlying operation.

This lab defaults to rank `8`, alpha `16`, dropout `0`, and PEFT's `all-linear` targets.
Thus default scale is `16/8=2`. The output language-model head is excluded by this targeting convention.
Inspect the actual reported trainable parameter count rather than inferring it from the model name.
[PEFT's LoRA documentation](https://huggingface.co/docs/peft/main/package_reference/lora)
describes the configuration and default random-`A`, zero-`B` initialisation.

## Derive the gradients, including the first-step asymmetry

For one example, let `g=partial L/partial h`, a column vector of length `d_out`.
Hold the input fixed while differentiating this layer, and omit adapter dropout for clarity.

1. Set `u=Ax`, a vector of length `r`.
2. The output gradient reaches `B` through the outer product:
   `partial L/partial B = s g u^T`.
3. It reaches `A` through `B`:
   `partial L/partial A = s B^T g x^T`.
4. The input gradient remains `W_0^T g + s A^T B^T g`.
   Frozen weights can still participate in propagating gradients to earlier trainable adapters.

With initial `B=0`, the adapter adds exactly zero to the base output.
Initially `grad_A=0` but `grad_B` can be nonzero because `A` is random.
If both `A` and `B` were zero, both gradients would be zero in this construction.
“Every adapter tensor must have a nonzero first gradient” is therefore the wrong smoke check.

### Worked example: watch the first adapter update

Use a two-dimensional toy layer, rank `1`, scale `s=1`, and loss `L=||h−t||²/2`:

```text
W0 = [[1, 0], [0, 1]]       A = [[1, -1]]
B  = [[0], [0]]             x = [2, 1]       target t = [3, 0]
```

**Predict:** the first update changes `B`; `A` starts learning only after that path opens.

1. `Ax=2−1=1`; initial output `h=W0 x=(2,1)`.
2. Error/output gradient `g=h−t=(−1,1)`; loss `(1+1)/2=1`.
3. `grad_B=g×1=(−1,1)`; `grad_A=(0,0)`.
4. SGD with learning rate `0.1` gives `B_new=(0.1,−0.1)`; `A` is unchanged.
5. New output `(2,1)+(0.1,−0.1)×1=(2.1,0.9)`.
6. New loss `((−0.9)²+0.9²)/2=0.81`.
7. On the next pass, `B_new^T g_new=0.1(−0.9)−0.1(0.9)=−0.18`,
   so `grad_A=−0.18(2,1)=(−0.36,−0.18)`.

**Interpret:** the frozen base still contributes the main computation; the adapter learns a correction.
This tiny matrix is an explanation of the real adapter operation, not a replacement for training a language model.

## Parameter savings and memory savings are different calculations

A dense weight has `d_out d_in` parameters. A LoRA pair adds `r(d_in+d_out)`.
The ratio is

$$\frac{N_{adapter}}{N_{dense}}=\frac{r(d_{in}+d_{out})}{d_{in}d_{out}}.$$

### Worked example: a 4096 by 4096 weight with rank 8

**Predict:** adapters save much more optimiser-state memory than base-weight memory.

1. Dense parameters: `4096²=16,777,216`.
2. Adapter parameters: `8(4096+4096)=65,536`.
3. Adapter fraction: `65,536/16,777,216=1/256≈0.390625%`.
4. Under an explicitly **FP32 Adam** accounting model, each trainable parameter uses
   `4` bytes for its weight, `4` for its gradient, and `8` for two moment estimates: `16` bytes.
5. Full fine-tuning state for this weight: `16,777,216×16=256 MiB`.
6. Frozen FP32 base weight: `16,777,216×4=64 MiB`.
7. Adapter weight, gradient, and moments: `65,536×16=1 MiB`.
8. Frozen base plus trainable adapter state: `65 MiB`, compared with `256 MiB` in this accounting.

One MiB is `2^20` bytes. These figures exclude activations, temporary buffers, other layers,
allocator overhead, and any extra master-weight copies. They are not measured peak VRAM.
LoRA still needs a forward pass through the base model and activations for backpropagation.
Quantising the frozen base attacks a different memory term; continue with
[lesson 3: quantisation and QLoRA](03-quantisation-and-qlora.md).

## Completion masks: learn the answer and when it ends

For a conceptual token sequence, labels aligned with input positions might be

```text
input:   [prompt_0, prompt_1, answer_0, answer_1, EOS, PAD]
labels:  [  -100,     -100,  answer_0, answer_1, EOS, -100]
```

`−100` is an ignored target, not a vocabulary token.
After the causal shift, logits at input position `t−1` predict the label at position `t`.
The prompt still enters the model and influences its activations; masking its labels does not remove it.
An attention mask and a loss mask have different jobs.

If the padding ID equals EOS, mask by actual sequence lengths/attention metadata.
Masking every token whose ID equals EOS would also erase the real termination target.
Check that tokenisation preserves the prompt as an exact prefix of the prompt-plus-answer example;
chat headers, whitespace, and subword boundaries can otherwise move the loss boundary.
The trainer validates this prefix and retains real end-of-turn tokens while excluding padding.
Follow `encode_completion`, `collate_completions`, and `completion_logprobs` in
[training.py](../llm_lab/training.py) to connect each mask and causal shift to actual tensors.
Reject examples with no supervised completion targets rather than training on a silent empty loss.

## Train, save, reload, and verify

Follow [the README's SFT run](../README.md) with `--method sft`.
This uses an actual pretrained causal transformer and real PEFT LoRA parameters.
The default local route is unquantised FP32 LoRA on CPU or Apple MPS.
On NVIDIA CUDA, the trainer uses BF16 when supported, otherwise FP32 for a conservative fallback.
The optional `--quantization nf4` route requires CUDA in this lab; it is covered in lesson 3.
Use [the RTX 5090/NVIDIA guide](../NVIDIA.md) to inspect the GPU, available VRAM, and executable
CUDA kernels before selecting a precision mode.

The saved run contains adapter weights/configuration, tokenizer files, `run_config.json`, and `metrics.json`.
An adapter checkpoint is a learned change to a specified base checkpoint, not a standalone full language model.
A reload therefore needs the same base model/revision plus the adapter and compatible tokenizer/template.
The README gives the inference command; the run metadata records the corresponding invocation.
For DPO continuation, `--adapter runs/sft` loads the SFT adapter as the starting policy.

Verify three different things:

- **Training:** adapter parameters change and losses remain finite; frozen base parameters remain frozen.
- **Reload:** for fixed prompt tokens in evaluation mode, saved/reloaded adapter logits match the in-memory model within numerical tolerance.
- **Behaviour:** generated validation answers improve on the metric you actually chose, if the evidence shows improvement.

Passing the first two does not imply the third. A saved adapter also does not imply the
optimiser and RNG states required for exact training resumption were saved.

## Check your understanding

<details>
<summary>1. Why can the first gradient of A be zero without training being broken?</summary>

`grad_A=s B^T g x^T`, and the default initial `B` is zero.
Check that `B` receives an update and that `A` can receive gradients on subsequent passes.

</details>

<details>
<summary>2. Does halving LoRA rank halve total peak device memory?</summary>

It halves the adapter parameter count at fixed targets. Frozen base weights, many activations,
and other buffers remain. Total peak memory does not generally halve.

</details>

<details>
<summary>3. Can an adapter trained on one base revision be assumed interchangeable with another?</summary>

No. Matching shapes is insufficient: its learned correction is relative to particular base weights.
Record the base identifier/revision, adapter configuration, tokenizer, and prompt formatting.

</details>
