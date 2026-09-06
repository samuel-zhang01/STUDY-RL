# Lesson 03 — quantisation, LoRA, and what QLoRA actually saves

Start with [the lab README](../README.md) for setup and
[the preceding LoRA lesson](02-real-lora-training.md). Activate the lab
environment before running the commands below.
This lesson needs only multiplication, rounding, matrix shapes, and the idea of
freezing a parameter. Run its numerical demo without downloading a model:

```bash
cd llm-lab
python -m llm_lab.quantisation
```

The demo really stores packed integers and reconstructs them. It deliberately
uses uniform quantisation on tiny CPU tensors. It does not implement NF4,
quantised matrix-multiplication kernels, or a quantised transformer trainer.
That distinction lets you inspect the arithmetic without confusing it with a
claim about model quality, training throughput, or GPU memory.

## A ruler with fewer marks

Imagine writing a measurement using only the nearest mark on a short ruler.

| Ruler idea | Quantisation object |
|---|---|
| Distance between marks | Scale $s$ |
| Number written beside a mark | Integer code $q$ |
| Mark corresponding to physical zero | Zero point $z$ |
| Nearest mark | Rounding |
| Measurement beyond the ruler's end | Clipping/saturation |
| Recovering an approximate distance | Dequantisation $\hat x=s(q-z)$ |

**Limit:** uniformly spaced marks describe this tutorial's quantiser. NF4 uses
a nonuniform codebook, so its marks are not evenly spaced. Quantisation is
lossy; more careful decoding cannot recover a measurement that was discarded.

## 1. Symmetric quantisation, derived and computed

For $b$ bits, this implementation uses balanced signed levels

$$q_{\max}=2^{b-1}-1,\qquad q_{\rm signed}\in\{-q_{\max},\ldots,q_{\max}\}.$$

This convention has $2^b-1$ levels: one code is unused. Other quantisers use
an asymmetric signed range such as $[-8,7]$; always state the convention.
For maximum represented magnitude $M>0$, set $s=M/q_{\max}$ and compute

$$q_{\rm signed}=\operatorname{clip}(\operatorname{round}(x/s),-q_{\max},q_{\max}),
\qquad \hat x=sq_{\rm signed}.$$

1. Divide by scale to express the weight in units of one quantisation step.
2. Round to a representable integer.
3. Saturate to the available integer range.
4. Multiply by scale when reconstructing the approximate weight.

**Predict:** at four bits and $M=1$, will 0.8 reconstruct above or below 0.8?

Here $q_{\max}=7$ and $s=1/7$. The full calculation is:

| $x$ | $x/s$ | Signed integer | Reconstructed $\hat x$ | Absolute error |
|---:|---:|---:|---:|---:|
| -1 | -7 | -7 | -1 | 0 |
| -0.6 | -4.2 | -4 | -4/7 = -0.571429 | 0.028571 |
| -0.1 | -0.7 | -1 | -1/7 = -0.142857 | 0.042857 |
| 0 | 0 | 0 | 0 | 0 |
| 0.3 | 2.1 | 2 | 2/7 = 0.285714 | 0.014286 |
| 0.8 | 5.6 | 6 | 6/7 = 0.857143 | 0.057143 |
| 1 | 7 | 7 | 1 | 0 |

For inputs inside the represented range, nearest rounding gives
$|x-\hat x|\le s/2=1/14\approx0.071429$.
This is a scalar reconstruction bound, not a bound on generated-answer quality.
The module stores unsigned codes $q=q_{\rm signed}+7$ and zero point $z=7$;
the shifted storage format still reconstructs the signed values above.

### Rounding ties is part of the specification

We use **round to nearest, ties to even**, matching `torch.round`.
At scale 1, $(-2.5,-1.5,-0.5,0.5,1.5,2.5)$ rounds to
$(-2,-2,0,0,2,2)$. Thus 0.5 does not always round up.
For affine quantisation below, this module rounds $x/s$ **before** adding
integer zero point $z$. At exact half-step ties, rounding $x/s+z$ instead can
choose the other neighbouring code when $z$ is odd. Both are nearest rounding;
the tie rule and operation order must be declared for exact reproducibility.

## 2. Asymmetric/affine quantisation

An unsigned affine quantiser uses all $2^b$ integer codes $0,\ldots,2^b-1$.
For a calibration range $[L,U]$ containing zero,

$$s=\frac{U-L}{2^b-1},\qquad
z=\operatorname{clip}\left(\operatorname{round}(-L/s),0,2^b-1\right),$$

$$q=\operatorname{clip}\left(\operatorname{round}(x/s)+z,0,2^b-1\right),
\qquad\hat x=s(q-z).$$

The zero point allocates different numbers of levels to negative and positive
values. It is a coordinate offset, not a learned bias in the original layer.

**Predict:** can two-bit affine quantisation represent both -1 and 2 exactly?

Set $b=2$, $[L,U]=[-1,2]$, so $s=3/3=1$ and $z=1$.
The represented values are $(-1,0,1,2)$:

| Input | Rounded $x/s$ | Code after adding $z$ and clipping | Reconstruction |
|---:|---:|---:|---:|
| -1 | -1 | 0 | -1 |
| -0.4 | 0 | 1 | 0 |
| 0 | 0 | 1 | 0 |
| 0.6 | 1 | 2 | 1 |
| 2 | 2 | 3 | 2 |
| 4 | 4 | 3 | 2 |

The first five reconstruction errors are $(0,0.4,0,0.4,0)$, within $s/2=0.5$.
The last error is 2: clipping breaks the no-saturation error bound.

Integer zero-point rounding can shift reconstructed endpoints. At three bits
on the same calibration range, $s=3/7$, $z=\operatorname{round}(7/3)=2$,
and the reconstructed range is $[-6/7,15/7]$, not exactly $[-1,2]$.
The module exposes `representable_range` so that this is visible.

### Zero scales, constant tensors, and outliers

For an all-zero tensor, automatic scale formulas would divide by zero.
This module defines $s=1$ and encodes zero exactly. The chosen positive scale
is arbitrary because every reconstructed value is zero.

A constant positive tensor, such as $(7,7,7)$, is different. Automatic affine
calibration includes zero, giving $[0,7]$, $s=7/15$, $z=0$ at four bits;
code 15 reconstructs every 7 exactly. Constant negative tensors work analogously.
An explicitly supplied degenerate range or zero clipping magnitude is rejected.

For the symmetric example, forcing $M=1$ clips a weight 4 to 1: error 3.
If automatic calibration instead sees an outlier 100, then $s=100/7$;
a small weight 0.8 rounds to zero. Smaller blocks or per-channel scales can
reduce an outlier's influence on unrelated weights, at the cost of metadata.
Neither clipping nor finer calibration is always best for task performance.

## 3. Real integer storage is separate from low-precision arithmetic

Open [quantisation.py](../llm_lab/quantisation.py) and inspect these boundaries:

- `symmetric_quantize` / `affine_quantize` choose scale, zero point, and codes.
- `_pack` puts consecutive codes into a `torch.uint8` byte tensor.
- `QuantizedTensor.unpack()` allocates one byte per code for inspection.
- `dequantize()` allocates floating-point values for ordinary PyTorch operations.

Sixteen four-bit codes $(0,1,\ldots,15)$ occupy **eight bytes**. The first
byte is $0+(1\ll4)=16$; the second is $2+(3\ll4)=50$. The complete byte
payload is $(16,50,84,118,152,186,220,254)$.
Seven four-bit codes occupy $\lceil7\cdot4/8\rceil=4$ bytes; the final nibble is padding.
The corresponding seven fp32 values occupy $7\cdot4=28$ bytes.

These are payload sizes. Python objects, shape information, scales, and zero
points add overhead, especially for tiny tensors. Keeping the original tensor
and a reconstructed tensor alive also consumes their full floating-point memory.
Running `dequantize() @ x` demonstrates reconstruction; it does not demonstrate
a faster quantised matrix-multiplication kernel.

## 4. LoRA can learn alongside a frozen quantised base

For a linear layer using column-vector inputs,

$$W_0\in\mathbb R^{d_{out}\times d_{in}},\quad
A\in\mathbb R^{r\times d_{in}},\quad B\in\mathbb R^{d_{out}\times r},$$

$$y=W_0x+\frac{\alpha}{r}B(Ax).$$

LoRA freezes $W_0$ and trains A and B. QLoRA uses a frozen quantised base:

$$y=\operatorname{dequant}(Q(W_0))x+\frac{\alpha}{r}B(Ax).$$

The matrix multiplications and adapter gradients still use floating-point
computation. The base weight codes do not receive ordinary gradient updates.
The adapter learns the downstream objective; it need not reconstruct the
original unquantised weights exactly.

**Predict:** can rank one exactly repair every two-by-two quantisation error?

For this constructed case, let

$$W_0=\begin{bmatrix}1&0\\0.2&0.2\end{bmatrix},\quad
\hat W=\begin{bmatrix}1&0\\0&0\end{bmatrix},\quad
B=\begin{bmatrix}0\\0.2\end{bmatrix},\quad A=\begin{bmatrix}1&1\end{bmatrix}.$$

The balanced two-bit quantiser has $M=1$, $s=1$ and produces $\hat W$.
With $\alpha/r=1$, $BA=W_0-\hat W$, so this rank-one correction is exact.
For $x=(1,2)^T$, $W_0x=(1,0.6)^T$, $\hat Wx=(1,0)^T$, and
$B(Ax)=(0,0.6)^T$ restores the original output.
The weight error falls from $\sqrt{0.2^2+0.2^2}=0.282843$ to zero.

But residual $I_2$ has rank two; its best rank-one approximation leaves
Frobenius error 1. The module's `low_rank_residual` computes an SVD oracle
using access to original weights. It illustrates the rank limit; it is not
LoRA training, QLoRA, or an implementation of LoftQ.

## 5. Uniform int4 is not NF4

Both can store four-bit indices, but those indices mean different things.
Our uniform codebook has equally spaced reconstructed values. NF4 uses a
nonuniform codebook designed around a normal weight distribution, with more
resolution in densely populated regions. Block scales map those levels to
weight magnitudes. See the [bitsandbytes `LinearNF4` specification](https://huggingface.co/docs/bitsandbytes/main/en/reference/nn/linear4bit).

The original QLoRA recipe combines frozen four-bit base weights, LoRA,
NF4, quantisation of quantisation constants (double quantisation), and paged
optimisers to manage memory spikes. Double quantisation compresses scale
metadata; it does not turn each four-bit weight into a two-bit weight.
See [Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314).
An arbitrary four-bit tensor experiment therefore cannot reproduce the paper's
quality or hardware results.

## 6. Build a memory budget before choosing a model

For $P$ base parameters at $b$ bits, the ideal packed weight payload is
$\lceil Pb/8\rceil$ bytes. This is the first line of the budget, not its total.

The CLI's `estimate_lora_memory` assumes one fp32 scale per 64 quantised
weights and 10,000 fp32 adapter parameters for a 1,000,000-parameter base.
Adapters require 40,000 parameter bytes, 40,000 gradient bytes, and 80,000
bytes for two fp32 Adam moments: 160,000 bytes altogether.

| Frozen base storage | Weight payload | Assumed scales | Adapter training states | Accounted total |
|---|---:|---:|---:|---:|
| fp32 | 4,000,000 | 0 | 160,000 | 4,160,000 bytes |
| fp16/bf16 | 2,000,000 | 0 | 160,000 | 2,160,000 bytes |
| uniform int8 | 1,000,000 | 62,500 | 160,000 | 1,222,500 bytes |
| uniform int4 | 500,000 | 62,500 | 160,000 | 722,500 bytes |

These are illustrative accounting assumptions. They are not a measurement of
bitsandbytes NF4: actual block sizes, double quantisation, unquantised modules,
adapter dtype, and optimiser implementation change the result.
MB uses $10^6$ bytes; MiB uses $2^{20}$ bytes. Label units when comparing reports.

You still need to account for:

- **Compute dtype and temporary buffers:** a four-bit stored weight does not
  imply four-bit accumulation, activations, or intermediate dequantisation.
- **Training activations:** backward needs information from the forward pass;
  sequence length, batch size, attention implementation, and checkpointing matter.
- **KV cache during generation:** a simple estimate is
  $2\times L\times B\times T\times H_{KV}\times d_{head}\times\text{bytes per value}$.
  Here 2 counts keys and values, L is layers, B is batch, T cached tokens,
  and $H_{KV}$ counts KV heads, which can differ from query heads.
- **Optimiser and gradient storage:** LoRA restricts these to trainable parameters;
  freezing the base does not remove activations needed to backpropagate through it.
- **Other models and runtime:** a reward/reference model, allocator reservations,
  kernel workspaces, loading copies, and merging can add memory.

For example, 12 layers, batch 1, 1,024 tokens, four KV heads, head dimension 64,
and fp16 cache give $2(12)(1)(1024)(4)(64)(2)=12,582,912$ bytes = **12 MiB**.
Doubling cached sequence length doubles that simple KV estimate. It does not
necessarily double all training memory; some attention intermediates grow faster.
During training, decoder caching is commonly disabled, while activations remain.

## 7. Choose the local route honestly

| Lab route | Frozen base | Trainable part | What it teaches |
|---|---|---|---|
| CPU/MPS full-precision LoRA | Floating-point weights | Floating-point adapters | Actual adapter training on the available local device |
| This numerical CPU demo | Packed uniform integer tensor | No trained model | Quantisation, reconstruction, and memory arithmetic |
| CUDA NF4 QLoRA | bitsandbytes quantised base | Floating-point adapters | Training through a quantised base with the required backend |

CPU/MPS LoRA is useful, real fine-tuning; it is not QLoRA merely because the
adapter rank is small. This lab's NF4 training route explicitly requires a
working CUDA stack. That is a lab support choice, not a claim that bitsandbytes
has no other backends: its current compatibility documentation also lists
CPU, XPU and HPU configurations. MPS availability alone is not evidence of
CUDA NF4 support. Check [the backend installation table](https://huggingface.co/docs/bitsandbytes/main/en/installation)
against the actual device, operating system, and installed versions.

The following shows the current API sequence for a **single CUDA device**;
set `model_id` to the model selected in the README. This fragment is a recipe,
not a record of a GPU run performed for this lesson:

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

assert torch.cuda.is_available(), "This lab's NF4 route requires CUDA"
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
base = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map={"": 0},
    dtype=compute_dtype,
)
base.config.use_cache = False
base = prepare_model_for_kbit_training(base)
model = get_peft_model(base, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0,
    target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
))
model.print_trainable_parameters()
```

`prepare_model_for_kbit_training` belongs before adapter attachment; inspect
which parameters require gradients afterwards. `target_modules="all-linear"`
is the PEFT QLoRA-style setting, but architecture-specific layer names and
excluded output heads still deserve inspection. The actual runnable training
commands and chosen lab model are in the README. On the RTX 5090, follow
[the NVIDIA setup guide](../NVIDIA.md) to verify the installed CUDA kernels.
The lab's [loader](../llm_lab/inference.py) creates the quantised base;
[train.py](../llm_lab/train.py) prepares it and attaches adapters, while
[training.py](../llm_lab/training.py) implements the losses and optimiser steps.
[PEFT's quantisation guide](https://huggingface.co/docs/peft/main/developer_guides/quantization)
documents this preparation sequence. The
[Transformers bitsandbytes guide](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)
distinguishes storage/compute settings and cautions that automatic device
dispatch is intended for inference; it is not a substitute for a training placement plan.

### Saving and merging an adapter

An adapter checkpoint stores the learned low-rank additions and configuration;
it still needs the compatible base model and tokenizer. Record the base revision,
formatting template, quantisation settings, and package versions with the adapter.

For ordinary LoRA in floating point, merging computes
$W_{merged}=W_0+(\alpha/r)BA$ and can eliminate the separate adapter path.
For a quantised base, adding BA to packed integer codes is not this operation.
A merge may require reconstruction into floating point, extra memory, and
possibly requantisation. Requantisation can introduce a new error.
Loading an adapter trained against $\hat W_0$ onto the original $W_0$ also
changes its effective layer by $W_0-\hat W_0$.

For a one-weight example with representable levels $(0,1)$, frozen weight 1
and adapter delta -0.4 give effective value 0.6 before merging. Merging into
floating point preserves 0.6; requantising on that original grid rounds back
to 1 and loses the entire correction. Backend-specific merge support varies;
check the [PEFT merge API](https://huggingface.co/docs/peft/main/package_reference/lora)
and evaluate the resulting checkpoint again.

## Check that you can transfer the ideas

<details><summary>1. A four-bit tensor has uint8 storage. Is it secretly eight-bit?</summary>

Not if two four-bit indices are packed into each byte. Inspect payload size:
16 codes should occupy eight bytes. A uint8 array with one code per element
instead uses eight physical bits per stored code until packed.
</details>

<details><summary>2. Why can a rank-eight adapter fail to undo quantisation error?</summary>

The residual can have rank greater than eight, and adapter training optimises
task loss rather than weight reconstruction. Even an oracle rank-eight residual
cannot represent an arbitrary full-rank error exactly.
</details>

<details><summary>3. Four-bit base weights fit, but training runs out of memory. Which budget lines change first?</summary>

Inspect activation/sequence and batch costs, trainable parameter count and
optimiser state, additional reference models, and temporary buffers. Weight
payload fitting alone does not establish that a training step fits.
</details>

<details><summary>4. Is using NF4 alone sufficient to claim a reproduced QLoRA result?</summary>

No. State the exact base model, adapter coverage, objective, data, optimisation,
precision, backend, and evaluation. The numerical demo verifies arithmetic;
a completed training run and measured evaluation support training claims.
</details>

Validation: [test_quantisation.py](../tests/test_quantisation.py) checks packed
storage across bit widths, exact representable values, half-step error bounds,
saturation, tie rounding, zero/constant tensors, rank limits, and memory sums.
Run `python -m pytest -q tests/test_quantisation.py` from `llm-lab`.
The numerical module and these checks run locally without external model downloads.
