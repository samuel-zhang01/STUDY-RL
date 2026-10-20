# Week 07 teaching guide — learning probabilities directly

**Course navigation:** [Week 06](../week-06-rainbow-bbf/teaching_guide.md) ← **Week 07 / 26** → [Week 08](../week-08-a2c-gae-trpo/teaching_guide.md)

[Routes](../docs/LEARNING_MAP.md) · [All weeks](../docs/STUDY_GUIDE.md#week-by-week-teaching-guides) · [Notation](../docs/GLOSSARY.md) · [Progress](../docs/PROGRESS.md)

[Lecture PDF](notes.pdf) · [Workbook](workbook.ipynb) · [Extended reading PDF](extended_reading.pdf)

**You are here:** Why does reward multiply a log-probability gradient?

**Recall first:** [Week 04](../week-04-td-sarsa-q/teaching_guide.md). If unfamiliar, use the [repair tasks](../docs/LEARNING_MAP.md#if-a-prerequisite-is-missing).

**Visual anchor:** [See the connected diagram](../docs/CONCEPT_ATLAS.md#the-actor-and-critic-answer-different-questions). Use it to explain the mechanism before the derivation.

Use with [the workbook](workbook.ipynb) and [the notes](notes.tex).
The key distinction is between a scalar loss and its parameter gradient.

## Learning bridge

Follow [the staged learning path](../docs/LEARNING_PATH.md) when a topic is new.

- **Carry forward:** Retrieve the delivery’s two action values: stop 2 and continue 3.6 at gamma 0.9.
- **Add one difficulty:** Instead of choosing greedily from a value table, directly learn how often to choose each action. An advantage compares one action with the current policy’s average.
- **Try after the worked example:** In [the policy experiment](../interactive/index.html#policy), set the continue probability to 0.25. Predict the policy’s average return and whether a positive reward of 2 must have positive advantage.

<details><summary>Check your prediction</summary>

The average is 0.75×2+0.25×3.6=2.4. Stop has advantage 2−2.4=−0.4: positive reward can still be below the policy average.

</details>

**Carry it onward:** [Actor–critic and GAE](../week-08-a2c-gae-trpo/teaching_guide.md) replace noisy completed returns with critic-based advantage estimates.


## Mechanism at a glance

```mermaid
flowchart LR
    P["Policy probabilities"] --> A["Sample action"]
    A --> G["Observe return"]
    G --> U["Return minus baseline times score gradient"]
    U --> P
```

**Read the diagram:** The sampled action has a log-probability derivative. Multiplying it by a return relative to a suitable baseline changes probabilities in a reward-sensitive direction.

## Before you start

Weeks 04–06 improved a value estimate and acted greedily from it.
This week parameterises the action probabilities themselves: $\pi_\theta(a\mid s)$.
You need the chain rule, log probabilities, expectation, and returns-to-go.
Week 08 will replace complete returns with bootstrapped advantage estimates.

## Analogy: changing a lottery's ticket proportions

| Lottery idea | Policy-gradient object |
|---|---|
| Fraction of tickets assigned to an action | $\pi_\theta(a\mid s)$ |
| Knobs that change those fractions | Parameters $\theta$ |
| Prize revealed after a ticket is drawn | Return $G$ |
| Sensitivity of its log chance | Score $\nabla_\theta\log\pi_\theta(a\mid s)$ |
| Prize above a reference level | Advantage $G-b(s)$ |

The update changes how likely similar choices will be in future.
**Limit:** RL actions also change future states; the one-draw lottery omits that.
A baseline is a reference for gradient estimation, not an extra reward from the environment.

## Derivation: the likelihood-ratio gradient, one step at a time

Assume a differentiable policy with fixed support, finite horizon $T$,
policy-independent dynamics/initial states, and regularity to exchange
expectation and differentiation. Let $r_t$ denote the reward after action $a_t$.

$$J(\theta)=\mathbb E_{\tau\sim p_\theta}
\left[\sum_{t=0}^{T-1}\gamma^tr_t\right].$$

1. Differentiate the trajectory probability, because the environment reward
   has no direct parameter derivative in this likelihood-ratio calculation.
2. Use $\nabla p_\theta=p_\theta\nabla\log p_\theta$.
3. Dynamics factors cancel from the log derivative, giving
   $\nabla\log p_\theta(\tau)=\sum_t\nabla\log\pi_\theta(a_t\mid s_t)$.
4. Past rewards can be removed from each action's term: condition on the full
   history before that action; the conditional mean score is zero.
5. Set $G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k$. The result is

$$\nabla J=\mathbb E\left[\sum_{t=0}^{T-1}
\gamma^t\nabla\log\pi_\theta(a_t\mid s_t)G_t\right].$$

The outer $\gamma^t$ matters for this start-state discounted objective.
It can be absorbed into a discounted state-sampling distribution, but cannot
just disappear from an ordinary trajectory sum. At $\gamma=1$ it is one.
Finite-horizon values formally also depend on time; include time in state if needed.

## Example 1: two actions let us enumerate the exact gradient

**Predict:** can a sampled zero reward contribute no gradient despite a positive true gradient?

One-step policy has $p=\pi_\theta(1)=\sigma(\theta)=0.2$;
rewards are $r(1)=1$, $r(0)=0$, so $J=p$ and $dJ/d\theta=p(1-p)=0.16$.
The logistic score is $a-p$.

| Action | Probability | Reward | Score | Sample gradient $r(a-p)$ |
|---|---:|---:|---:|---:|
| 1 | 0.2 | 1 | 0.8 | 0.8 |
| 0 | 0.8 | 0 | -0.2 | 0 |

Mean gradient $0.2(0.8)+0.8(0)=0.16$.
Variance $0.2(0.8^2)-0.16^2=0.1024$.
For minimisation, the loss is $-r\log\pi$; its derivative is the negative
of the ascent gradient. Flipping that sign would learn the worse action.

**Interpretation:** correctness is in expectation, not in every sampled update.
At very small $p$, absolute variance tends to zero, but useful rewards become
rare and the variance relative to the squared mean grows.

## Derivation: why a baseline preserves the mean

For a state-only baseline held fixed with respect to the sampled action,

$$\mathbb E_{a\sim\pi}[b(s)\nabla\log\pi(a\mid s)]
=b(s)\sum_a\nabla\pi(a\mid s)=b(s)\nabla1=0.$$

Thus $G-b(s)$ preserves the expected score-function gradient.
For a scalar baseline minimizing total gradient variance, let $h=\nabla\log\pi$.
The gradient mean does not depend on $b$, so minimise
$\mathbb E[\|h\|^2(G-b)^2]$ instead.
Differentiate with respect to $b$ and set the result to zero:

$$b^*=\frac{\mathbb E[\|h\|^2G]}{\mathbb E[\|h\|^2]}.$$

A value baseline $\mathbb E[G\mid s]$ is useful but need not equal this weighted optimum.
See [Greensmith, Bartlett and Baxter, 2004](https://www.jmlr.org/papers/v5/greensmith04a.html).

## Example 2: the value baseline can increase variance

**Predict:** at $p=0.2$, is the mean reward 0.2 the variance-minimising baseline?

Using $b=0.2$, action 1 gives gradient $(1-0.2)(0.8)=0.64$;
action 0 gives $(0-0.2)(-0.2)=0.04$.
Mean $0.2(0.64)+0.8(0.04)=0.16$;
variance $0.2(0.64^2)+0.8(0.04^2)-0.16^2=0.0576$.

The optimal baseline is

$$b^*=\frac{0.2(0.8^2)(1)+0.8(0.2^2)(0)}
{0.2(0.8^2)+0.8(0.2^2)}=\frac{0.128}{0.16}=0.8.$$

With $b=0.8$, both sample gradients equal 0.16, hence variance zero.
For the edge case $p=0.8$, the unbaselined variance is 0.0064,
while the value baseline $b=0.8$ produces variance 0.0576: nine times larger.
The optimal baseline is then 0.2 and again has zero variance.

**Interpretation:** action-independent baselines preserve the mean; variance
reduction is a separate property that requires choosing the baseline well.

## Example 3: a discount factor that cannot be dropped

Two independent Bernoulli decisions share parameter $p=\sigma(\theta)$.
Only the second action pays its indicator reward; $\gamma=0.5$, $p=0.5$.
Then $J=0.5p$, so its derivative is $0.5p(1-p)=0.125$.
The first action's expected contribution is zero; the second contributes 0.125.
Dropping the outer discount gives 0.25, twice the correct start-state gradient.

## Bridge to this week's code

- [solutions.py](project/solutions.py), `discount_cumsum`, computes $G_t$;
  it does not supply the separate outer $\gamma^t$ weighting.
- `reinforce_loss` is sufficient for the one-step bandit examples above.
- `optimal_baseline` takes **squared gradient norms**, not squared log probabilities.
- [agent.py](project/agent.py), `REINFORCEAgent.update`, detaches the value estimate
  from the actor loss and standardises advantages within an episode.
  This is a practical estimator variant, not exactly the unnormalised unbiased
  formula above. It uses population standard deviation (`correction=0`), avoiding
  an undefined sample standard deviation for one observation. Centering a
  one-step episode still makes its only advantage zero, so this episode-normalised
  update supplies no actor signal for that case. For a bandit, use the separate
  `reinforce_loss` exercise or aggregate independent samples before standardising.
- `estimator_variance` measures variance of `log_prob * advantage` loss terms.
  It does **not** compute variance of parameter gradients; interpret its plot accordingly.
- `topk_offpolicy_multiplier` differentiates inclusion probability
  $1-(1-p)^K$ for $K$ independent draws with replacement. It is not a complete
  correction for arbitrary deterministic top-K ranking or logging-policy mismatch.

For the original discounted policy-gradient setting, see
[Sutton et al., Policy Gradient Methods for RL with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html).

## Choose your explanation depth: Why the best baseline is a weighted mean

Open the first layer for the idea, the second to calculate, and the third to check the assumptions. The worked examples above and investigation below provide the longer route.

<details class="learning-layer">
<summary>1 · Intuition</summary>

A baseline changes the noise of a policy-gradient sample without changing its expectation when it does not depend on the sampled action. Samples with larger score gradients have more influence on gradient variance, so the best baseline does not generally equal the ordinary average return.

</details>

<details class="learning-layer">
<summary>2 · Work a numerical case</summary>

For returns (2,5) with squared score magnitudes (1,4), the weighted baseline is (1×2+4×5)/(1+4) = 4.4. The ordinary mean is 3.5. The corresponding weighted squared residual sums are 7.2 and 11.25 respectively; the first is smaller for these fixed samples.

</details>

<details class="learning-layer">
<summary>3 · Derive it and state the limits</summary>

Write $z=\nabla_\theta\log\pi_\theta(A\mid s)$. Since $E[z\mid s]=\nabla_\theta\sum_a\pi_\theta(a\mid s)=0$, subtracting an action-independent $b(s)$ preserves the expected gradient. Minimise $E[\|z\|^2(G-b)^2\mid s]$ over $b$: differentiating gives $b^*=E[\|z\|^2G\mid s]/E[\|z\|^2\mid s]$ when the denominator is positive. This is a population result under finite moments. Fitting a baseline on the very same samples can introduce dependencies requiring additional care.

</details>

**Try it in Jupyter:** jump to [`optimal_baseline` in the workbook](workbook.ipynb#cell-26), then run the setup cells above it. Expect 4.4. Measure variance of gradient vectors across samples or runs; variance of scalar losses is a different quantity.

## Notebook investigation: plot the gradient estimator, not just its loss

Open [the actual Jupyter workbook](workbook.ipynb) and find **Exercises 1, 2, 4 and 5**.
The [GitHub week folder](https://github.com/samuel-zhang01/STUDY-RL/tree/main/week-07-policy-gradients)
contains this notebook, its `project/` imports and tests. Follow the
[download and Jupyter workflow](../docs/LEARNING_MAP.md#download-and-open-the-jupyter-materials)
so the shared `common/` package is available.

Use the two-action policy with p=0.2 and rewards `(0,1)`. Enumerate actions instead of training:
the scores are `a-p`, and the two unbaselined sample gradients are `(0,0.8)`.
Weight them by probabilities `(0.8,0.2)` to recover mean 0.16.
Repeat with baselines 0.2 and 0.8; the mean stays 0.16 while variances are 0.0576 and zero.

Plot one mark per action at its **score-weighted gradient**, with its probability shown explicitly.
Do not use a histogram of `log_prob * advantage` to label parameter-gradient variance: the workbook's
exercise 5 loss-term diagnostic does not calculate those derivatives. Add your enumerated check as
a separate notebook cell. Then change p to 0.8: the value baseline now increases variance relative
to using no baseline, which disproves a universal variance-reduction claim.

<details>
<summary>Check your interpretation: Why can two zero-mean advantages still give a positive expected gradient?</summary>

The advantages multiply different score gradients. Their probability-weighted sum can be zero while the probability-weighted sum of those products is positive.

</details>

## Retrieve and transfer

<details><summary>1. Why detach the critic's prediction in the actor loss?</summary>

The actor estimator multiplies the policy score by a fixed advantage target.
Differentiating the baseline through that loss adds a term not justified by the baseline lemma.
</details>

<details><summary>2. Is zero-mean advantage sufficient for minimal gradient variance?</summary>

No. The relevant squared error is weighted by squared score norms, which can
vary with the sampled action. Example 2 gives the exact counterexample.
</details>

<details><summary>3. Can you obtain true gradient variance from scalar log probabilities alone?</summary>

No. You need parameter derivatives of those log probabilities and per-sample
weighted gradient vectors, under a clearly specified sampling distribution.
</details>

Arithmetic check: local Python enumerated both action branches for all baseline
choices and all four two-step paths, including the discounted gradient 0.125.
