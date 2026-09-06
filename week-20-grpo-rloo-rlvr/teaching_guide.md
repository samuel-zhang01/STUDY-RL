# Week 20 teaching guide: what a group baseline does and does not cancel

Use with [the notes](notes.tex) and [the workbook](workbook.ipynb).
All numerical cases below are finite examples whose expectations can be enumerated.

**Practical companion:** [LLM lab lesson 4](../llm-lab/lessons/04-preferences-and-rewards.md)
connects DPO to online rewards; its [strict answer verifier](../llm-lab/llm_lab/data.py)
and [generation evaluation](../llm-lab/llm_lab/evaluate.py) expose format and cutoff failures.

## Before starting

Recall the score-function gradient from week 7 and token/sequence probability from week 18.
A verifier supplies the reward; RLOO and GRPO specify how that reward enters an update.
A correct advantage calculation cannot repair an incorrect answer parser.
An outcome reward scores a whole completion, even when its coefficient is broadcast to tokens.

## Analogy: several attempts at the same puzzle

A teacher asks a student to try one puzzle several times, then compares the attempts.

| Classroom comparison | Mathematical object |
|---|---|
| One puzzle | Fixed prompt `x` |
| Independent attempts | Completions `y_1,...,y_G` sampled from one policy |
| Marks | Rewards `r_1,...,r_G` |
| Mean of all *other* marks | Leave-one-out baseline `b_i` |
| Mark minus peer mean | RLOO advantage |
| Deviation divided by the group's standard deviation | GRPO advantage |

The comparison is within one puzzle; pooling easy and difficult puzzles changes the baseline.
The analogy's independence matters: copying another attempt or sampling without replacement
changes the argument below. Similar outputs are allowed; probabilistic independence is the assumption.
The teacher can still mark the wrong answer as correct if the marking rubric is flawed.

## Derivation: why excluding the current sample matters

Fix `x`, let `y_i` be i.i.d. from `p_theta`, and let reward `r(y)` have no direct parameter dependence.
Define the sequence score `S_i=grad_theta log p_theta(y_i|x)`.
Assume the expectations below exist and `G>=2`.
The desired gradient is `g=E[r_i S_i]`.

1. The score has mean zero:

   $$\mathbb E[S_i]=\sum_y p_\theta(y)\nabla\log p_\theta(y)
   =\nabla\sum_y p_\theta(y)=0.$$

2. Form `b_i=sum_{j≠i} r_j/(G−1)`. It is independent of `y_i` conditional on the prompt.
   Thus `E[b_i S_i]=E[b_i] E[S_i]=0`.

3. Subtracting it preserves the gradient:

   $$\mathbb E[(r_i-b_i)S_i]=g.$$

4. Rewrite it for efficient implementation:

   $$A_i^{RLOO}=\frac{G}{G-1}(r_i-\bar r).$$

5. A mean including `r_i` instead gives

   $$\mathbb E[(r_i-\bar r)S_i]=(1-1/G)g.$$

A fixed group size permits correcting this scale with `G/(G−1)`.
Dividing by a **random reward-dependent** standard deviation does more than a fixed rescaling.
GRPO's zero-sum coefficients do not prove an unbiased reward gradient or a bound on group probability.
[RLOO's paper](https://arxiv.org/abs/2402.14740) revisits sequence-level REINFORCE;
[DeepSeekMath](https://arxiv.org/abs/2402.03300) introduces group-relative optimisation.

## Worked example 1: four completions, three kinds of coefficient

**Predict.** In group rewards `(1,0,0,0)`, the successful completion should have a
positive coefficient and the others negative coefficients.

1. Mean reward: `r_bar=1/4=0.25`.
2. Centred rewards: `(0.75,−0.25,−0.25,−0.25)`.
3. RLOO baseline for the success is `0`; its advantage is `1`.
   Each failure's peer baseline is `1/3`; its advantage is `−1/3`.
4. Population group variance is
   `(0.75²+3×0.25²)/4=0.1875`; standard deviation is `sqrt(3)/4≈0.433013`.
5. Ignoring a tiny numerical epsilon, GRPO advantages are
   `(sqrt(3),−1/sqrt(3),−1/sqrt(3),−1/sqrt(3))`
   `≈(1.732051,−0.577350,−0.577350,−0.577350)`.

**Interpret.** These arrays share signs and sum to zero, but their scales differ.
With all rewards `(1,1,1,1)` or `(0,0,0,0)`, the centred numerator is zero:
all reward advantages are zero. Dividing by epsilon does not create a learning signal.
A KL term can still update the policy when the verifier advantage is zero.
At `G=1`, leave-one-out is undefined; the project's zero-centred fallback is not an unbiased RLOO estimator.

## Worked example 2: enumerate the actual gradient, not just reward variance

Let action `a∈{0,1}`, success probability `p=0.5`, reward `r=a`, and parameter
`theta=log(p/(1−p))`. Its score is `S=a−p`, so `d E[r]/d theta=p(1−p)=0.25`.
Use groups of two independent actions.

**Predict.** RLOO should preserve `0.25` in expectation; centring by the inclusive
mean should shrink it by one half.

| Group | Probability | Mean raw `r S` | Mean inclusive-centred `(r−r_bar)S` | Mean RLOO `(r−r_other)S` |
|---|---:|---:|---:|---:|
| `(0,0)` | 0.25 | 0 | 0 | 0 |
| `(0,1)` | 0.25 | 0.25 | 0.25 | 0.5 |
| `(1,0)` | 0.25 | 0.25 | 0.25 | 0.5 |
| `(1,1)` | 0.25 | 0.5 | 0 | 0 |

Expected gradients are respectively `0.25`, `0.125`, and `0.25`.
Raw group-gradient variance is `0.03125`; RLOO group-gradient variance is `0.0625`.
Here RLOO is unbiased but has **higher** variance than the raw estimator.
An ideal fixed baseline `b=0.5` gives `(a−0.5)²=0.25` for every action, with zero variance.
The finite peer estimate introduces noise that a known constant baseline does not have.

## A small KL calculation worth doing separately

For `p=(0.8,0.2)`, `q=(0.5,0.5)`, let `u=log(q/p)` and `k3=exp(u)−u−1`.
The two values are `0.095004` and `0.583709`.
Their `p`-weighted mean is `0.192745=KL(p||q)`.
The identity uses `E_p[q/p]=1`, requiring the appropriate support in both directions.
Changing the sampling policy to an old policy invalidates that identity without correction.
An unbiased scalar estimate also does not automatically give an unbiased gradient when
samples are treated as fixed during differentiation.

## Bridge to the actual project

- [project/train.py](project/train.py): `rloo_advantages` and `grpo_advantages`
  operate along the group axis. Read `debias` as removing group standardisation,
  not as a blanket proof of an unbiased full training algorithm.
- [project/agent.py](project/agent.py): `token_logprobs` aligns shifted targets;
  `completion_mask` includes the first EOS and excludes later positions.
- [project/solutions.py](project/solutions.py): exercise 4 compares variances of
  reward **coefficients**, not complete score-weighted gradients; it cannot prove variance reduction.
- [project/env.py](project/env.py): inspect `extract_final_number` and `exact_match_reward`
  together; parsing a number is weaker than verifying a valid reasoning trace.

The full Dr.GRPO recipe also changes length normalisation; dropping only the standard deviation
is one component. Fixed normalisation and averaging over all valid batch tokens are distinct choices.
See [the Dr.GRPO analysis](https://arxiv.org/abs/2503.20783) for the length and difficulty distinction.

## Retrieve and transfer

<details>
<summary>1. Is a detached inclusive group mean independent of the sampled response?</summary>

No. Detaching changes autograd, not statistical dependence. The mean still contains the reward
of that response. Leave-one-out removes the dependence under independent sampling.

</details>

<details>
<summary>2. Can all-zero verifier rewards be repaired by increasing group normalisation?</summary>

No. Their centred advantages remain zero. Change data difficulty, sampling coverage,
or verifier design based on evidence; scaling zero cannot provide reward information.

</details>

<details>
<summary>3. What baseline minimises squared gradient magnitude?</summary>

For a scalar baseline it is `b*=E[r||S||²]/E[||S||²]`, assuming a nonzero denominator.
The reward mean is optimal only under an additional zero-covariance condition.

</details>
