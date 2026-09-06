# Lesson 1 — A language model is a policy over the next token

**Outcome:** explain exactly what is being optimised in pre-training, supervised fine-tuning,
DPO, and online RL, and follow one answer from token probabilities to a training signal.
Start here, then continue to [real LoRA training](02-real-lora-training.md).
[The lab README](../README.md) contains the installation instructions and runnable commands.

Useful background: [week 7: policy gradients](../../week-07-policy-gradients/teaching_guide.md),
[week 9: PPO](../../week-09-ppo/teaching_guide.md), and
[week 18: RLHF](../../week-18-rlhf-ppo-llm/teaching_guide.md).
You need conditional probability, logarithms, and the idea of a gradient.

## First distinguish the stages

| Stage | Training examples | Main signal | Generates new answers during the update loop? |
|---|---|---|---|
| Pre-training | Large text corpus | Predict observed next tokens | Usually no: next-token teacher forcing |
| Supervised fine-tuning, SFT | Prompt plus desired completion | Raise likelihood of demonstrated completion | Usually no |
| Offline DPO | Prompt plus chosen/rejected completions | Fit their preference relative to a reference | No new rollout required |
| Online policy-gradient RL | Prompts, newly sampled completions, evaluated rewards | Increase expected reward, usually with regularisation | Yes |

LoRA specifies **which parameters may change**. It does not choose the objective.
Quantisation specifies how some numbers are represented. It is another independent choice.
You can train LoRA adapters with SFT or DPO; neither becomes online RL simply because adapters are used.
The DPO derivation is related to a regularised RL objective, but its offline training loop uses fixed pairs.

The lab starts from the already instruction-tuned
[SmolLM2-135M-Instruct checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
Here, “initial model” means that checkpoint before **our** task adaptation.
We do not pre-train a foundation model from scratch. We perform additional post-training on synthetic addition examples.

## Analogy: walking along a tree of sentences

Imagine every partial sentence as a junction with a branch for each possible next token.

| Sentence-tree object | RL quantity |
|---|---|
| Prompt and tokens already written | State `s_t=(x,y_<t)` |
| Choosing the next branch | Action `a_t=y_t` |
| Probabilities assigned to branches | Policy `pi_theta(a|s_t)` |
| Appending the chosen token | Deterministic state transition |
| Complete path ending in EOS | Episode/completion |
| Score assigned to the completed answer | Terminal reward |

EOS is an action indicating that the answer ends; it is not free padding.
Prefixes are distinct states even if they look semantically similar.
The analogy describes generation, not intelligence: branch probability says how likely the model
is to produce text, not whether the text is true. A chat template also contains special tokens
that are part of the model's input protocol rather than ordinary visible words.

## From token probabilities to answer probability

Fix a prompt `x`. Let `y=(y_1,...,y_T)` include the final EOS/end-of-turn token.
All logarithms below are natural, so log-probability and KL use nats.
The autoregressive chain rule gives

$$\pi_\theta(y\mid x)=\prod_{t=1}^T\pi_\theta(y_t\mid x,y_{<t}),$$

and therefore

$$\ell_\theta(x,y)=\log\pi_\theta(y\mid x)
=\sum_{t=1}^T\log\pi_\theta(y_t\mid x,y_{<t}).$$

A log-softmax converts model logits into token log probabilities.
Selecting the observed token from each vocabulary distribution and adding the selected values
produces the sequence log probability. Averaging instead produces a different quantity.
[Transformers' causal-language-model guide](https://huggingface.co/docs/transformers/main/en/tasks/language_modeling)
describes next-token prediction and teacher-forced training.

### Worked example: likelihood and correctness can disagree

Suppose the correct answer is `12`. For illustration only, assume the tokenizer represents
it as `1`, `2`, EOS, with conditional probabilities `0.6`, `0.5`, `0.8`.
The real tokenizer need not split these numbers this way.

**Predict:** multiplying several probabilities can give a small sequence probability even for a correct answer.

1. Correct response probability: `0.6 × 0.5 × 0.8 = 0.24`.
2. Sequence log probability: `log(0.24) = −1.427116`.
3. Summed negative log likelihood: `1.427116`.
4. Mean negative log likelihood across its three target tokens: `1.427116/3 = 0.475705`.
5. Suppose an incorrect answer is a single token `13`, then EOS, with probabilities `0.4`, `0.9`.
   Its sequence probability is `0.36`, log probability `−1.021651`.
6. An exact-answer verifier awards rewards `1` to `12` and `0` to `13`, despite the likelihood ordering.

**Interpret:** likelihood is a property of the model's current distribution. Correctness comes from a separate criterion.
The two responses share a prompt but have different continuation states and lengths.
A mean token log probability is not a normalised probability over complete answers.

## SFT: teach the model which continuation to imitate

For a desired response `y*`, completion-only SFT minimises

$$L_{SFT}=-\sum_t m_t\log\pi_\theta(y_t^*\mid x,y_{<t}^*),$$

where `m_t=1` on real completion targets and `0` on prompt/padding targets.
The lab normalises by the number of supervised completion tokens.
During **teacher forcing**, the preceding response tokens come from the example, even if the model
would have generated a different prefix. At inference the model conditions on its own generated prefix.
That difference is one reason lower training loss need not imply better free-running answers.

### Worked example: one cross-entropy update

At one prediction position, suppose a two-token vocabulary has probabilities `(0.6,0.4)`
and the demonstration requires token 1. Let logits initially be `(log 0.6,log 0.4)`.

**Predict:** gradient descent should raise token 1's logit relative to token 2's.

1. Loss: `−log 0.6 = 0.510826`.
2. Softmax-cross-entropy derivative in logits is `p−one_hot(target) = (−0.4,+0.4)`.
3. At learning rate `0.1`, new logits are `(log 0.6+0.04,log 0.4−0.04)`.
4. Their difference increases by `0.08`; token 1's probability becomes
   `0.6 exp(0.04)/(0.6 exp(0.04)+0.4 exp(−0.04)) ≈ 0.619038`.
5. There is no reward model in this step. The desired token itself specifies the target.

**Boundary:** real LoRA parameters influence many logits simultaneously. The calculation illustrates
one output-level gradient, not an assertion that every training step improves every example.

## Online RL: weight sampled answers by their consequences

For a fixed, parameter-independent terminal reward `R(x,y)`, the basic objective is

$$J(\theta)=\mathbb E_{x,y\sim\pi_\theta}[R(x,y)].$$

Differentiate the sequence probability, then use its log derivative:

$$\nabla J=\mathbb E\left[R(x,y)\nabla\log\pi_\theta(y|x)\right]
=\mathbb E\left[R(x,y)\sum_t\nabla\log\pi_\theta(y_t|s_t)\right].$$

A suitable baseline replaces `R` with an advantage without changing this expectation.
With advantage `+1`, a sampled answer is encouraged; with advantage `−1`, it is discouraged.
Token-level credit assignment, PPO clipping, and KL penalties add further machinery.
[Lesson 4](04-preferences-and-rewards.md) distinguishes these pieces and derives the group baselines.

The new lab actually runs **pretrained-model LoRA SFT and DPO**.
For an executable online group-RL loop, use
[week 20's small character-model trainer](../../week-20-grpo-rloo-rlvr/project/train.py).
That trainer teaches rollout mechanics; it is a separate toy from this pretrained-model lab.

## Check your understanding before continuing

<details>
<summary>1. If both token probabilities are 0.5, is a two-token answer's probability 0.5?</summary>

No. It is `0.5×0.5=0.25` under those conditionals. The average token probability is a different statistic.
Include any termination token needed to define a complete response.

</details>

<details>
<summary>2. Can SFT loss fall while exact-answer accuracy stays unchanged?</summary>

Yes. The correct answer's probability can rise without becoming the most likely decoded answer.
Teacher-forced improvement can also fail to transfer to prefixes generated at inference.
Evaluate generated answers separately from likelihood loss.

</details>

<details>
<summary>3. Which changes at each stage: objective, trainable parameters, or number format?</summary>

SFT versus DPO versus online RL changes the learning objective/data loop.
Full fine-tuning versus LoRA changes trainable parameters.
FP32/BF16 versus NF4 changes number representation and execution constraints.
These are separate axes, so name all three when describing an experiment.

</details>
