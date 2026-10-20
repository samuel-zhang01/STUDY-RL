# Week 03 teaching guide — learning from finished stories

**Course navigation:** [Week 02](../week-02-bellman-dp/teaching_guide.md) ← **Week 03 / 26** → [Week 04](../week-04-td-sarsa-q/teaching_guide.md)

[Routes](../docs/LEARNING_MAP.md) · [All weeks](../docs/STUDY_GUIDE.md#week-by-week-teaching-guides) · [Notation](../docs/GLOSSARY.md) · [Progress](../docs/PROGRESS.md)

[Lecture PDF](notes.pdf) · [Workbook](workbook.ipynb) · [Extended reading PDF](extended_reading.pdf)

**You are here:** Which visit and whose trajectory does this return estimate?

**Recall first:** [Week 01](../week-01-mdps-bandits/teaching_guide.md), [Week 02](../week-02-bellman-dp/teaching_guide.md). If unfamiliar, use the [repair tasks](../docs/LEARNING_MAP.md#if-a-prerequisite-is-missing).

**Visual anchor:** [See the connected diagram](../docs/CONCEPT_ATLAS.md#one-transition-different-targets). Use it to explain the mechanism before the derivation.

Use with [the workbook](workbook.ipynb) and [the notes](notes.tex).
Track reward timing and visit timing separately throughout this week.

## Learning bridge

Follow [the staged learning path](../docs/LEARNING_PATH.md) when a topic is new.

- **Carry forward:** Retrieve the reward timing and discounted return from [week 2](../week-02-bellman-dp/teaching_guide.md).
- **Add one difficulty:** Hide the map’s reward average. Estimate a route’s value using only completed trips; one lucky trip is still one sample.
- **Try after the worked example:** In [MC versus TD](../interactive/index.html#targets), compare the evidence each target needs. On paper, average two continuing trips whose last rewards are 4 and 0, with gamma 0.9. Reproduce this using [two-step delivery](../learning_projects/two_step.py).

<details><summary>Check your prediction</summary>

The returns are 3.6 and 0, so their average is 1.8. This estimate uses both finished trips; it does not reveal the true reward law from only two observations.

</details>

**Carry it onward:** [TD](../week-04-td-sarsa-q/teaching_guide.md) will update before the trip ends by inserting a next-state estimate.


## Mechanism at a glance

```mermaid
flowchart LR
    E["Complete sampled episode"] --> G["Return from each visit"]
    G --> F["Choose first visits in chronological order"]
    F --> V["Average returns for each state"]
```

**Read the diagram:** First-visit MC uses the first chronological visit of a state in each episode. Scanning backward and accepting the first encounter would select its last chronological visit.

## Before you start

Week 02 averaged outcomes using a known model. Monte Carlo (MC) averages actual
returns from completed episodes instead. You need means, conditional expectation,
and multiplication of probabilities along a trajectory.
Assume a fixed policy for prediction and independent episodes with finite
expected returns. Unvisited states have no MC estimate yet.

## Analogy: reviewing complete delivery trips

| Delivery record | MC concept |
|---|---|
| A complete trip | Episode |
| Total remaining profit after a stop | Return-to-go |
| First time a depot is visited on a trip | First visit |
| Every return to the same depot | Every visit |
| Old dispatcher's rules | Behaviour policy $b$ |
| New dispatcher's rules | Target policy $\pi$ |

You can evaluate a depot by averaging the remainder of completed trips.
**Limit:** two visits on the same trip are not two independent trips.
Reweighting old trips also cannot reveal routes that the old dispatcher never took.

## Derivation: returns and the first-visit estimator

Let $R_{t+1}$ follow the action at time $t$, and let the episode end at $T$.
Set $G_T=0$; repeatedly expand the return definition to obtain

$$G_t=R_{t+1}+\gamma G_{t+1}.$$

This backward recurrence computes all returns in $O(T)$ time.
For state-action pair $(s,a)$, define its first occurrence in episode $i$ as
$t_i=\min\{t:(S_t,A_t)=(s,a)\}$, if that set is nonempty.
Then average $G_{t_i}$ across the episodes that contain the pair.

1. Compute returns backwards, storing one return at each time.
2. Scan occurrences forwards and retain the first occurrence of each pair.
3. Increment the pair's count once for each contributing episode.
4. Apply the incremental mean from Week 01.

For a stationary Markov policy, each first-visit return has mean $Q^\pi(s,a)$.
Repeated visits within one episode can be dependent. Every-visit MC remains
consistent under suitable episodic assumptions, but is not generally unbiased
for a fixed finite number of episodes.

## Example 1: backwards computation is not backwards selection

**Predict:** which of the two A visits should first-visit MC keep?

A single episode has tuples $(A,x,1),(B,y,2),(A,x,3)$, then termination.
With $\gamma=0.5$, the returns are

$$G_2=3,\qquad G_1=2+0.5(3)=3.5,
\qquad G_0=1+0.5(3.5)=2.75.$$

First-visit samples: $(A,x)\mapsto2.75$ and $(B,y)\mapsto3.5$.
Every-visit samples for $(A,x)$ are $(2.75,3)$, giving mean $2.875$.
A backward loop that keeps the first unseen pair would instead keep 3 for A.
That is **last-visit** selection, despite correctly computing the returns.

**Interpretation:** computation order and the statistical definition of a visit
are independent choices. The project solution correctly uses two passes.

## Derivation: importance sampling changes the sampling distribution

Assume the same initial-state law and environment dynamics for both policies,
and coverage: $b(a\mid s)>0$ whenever $\pi(a\mid s)>0$.
The trajectory likelihood is an initial probability times action probabilities
times transition probabilities. Dividing target by behaviour cancels the
common initial-state and transition factors:

$$\rho(\tau)=\frac{p_\pi(\tau)}{p_b(\tau)}
=\prod_{t=0}^{T-1}\frac{\pi(A_t\mid S_t)}{b(A_t\mid S_t)}.$$

Hence, for integrable returns,

$$\mathbb E_b[\rho G]=\sum_\tau p_b(\tau)
\frac{p_\pi(\tau)}{p_b(\tau)}G(\tau)=\mathbb E_\pi[G].$$

Ordinary IS averages $\rho_iG_i$ over $N$ episodes.
Weighted IS (WIS, or SNIPS in the one-step setting) divides their sum by
$\sum_i\rho_i$ instead of $N$. It is a random ratio, generally biased in finite
samples. Its denominator must be nonzero; consistency needs coverage and moments.

## Example 2: rare matching paths carry large weights

**Predict:** is the weight of a fully matching deterministic-target path tiny or large?

There are three decisions, advance/reject, with behaviour probabilities $1/2$.
The target always advances. A path with three advances has

$$p_b=(1/2)^3=1/8,\quad p_\pi=1,\quad \rho=8.$$

Every path containing a reject has target probability and weight zero.
Take eight logged paths, two fully matching, with returns 4 and 6.
The other six have zero weight regardless of their observed returns.

$$\widehat V_{\rm OIS}=\frac{8(4)+8(6)}8=10,
\qquad \widehat V_{\rm WIS}=\frac{80}{16}=5.$$

Their effective sample size is $(\sum w)^2/\sum w^2=16^2/128=2$.
If returns were known to lie in $[0,6]$, OIS can still exceed 6; WIS is a
weighted average and stays inside the observed return range when defined.

**Interpretation:** matching paths are rare under behaviour but heavily weighted.
If none appears, the WIS denominator is zero; reporting a numerical zero after
adding epsilon is not evidence that the target's value is zero.

## Example 3: why there is no universal variance winner

Let a one-step behaviour choose actions with probabilities $(1/2,1/2)$;
target probabilities are $(3/4,1/4)$, so weights are $(1.5,0.5)$.
Choose deterministic rewards $(2/3,2)$.
Then $wR=1$ for either action, so one-sample OIS is always 1: variance 0.
One-sample WIS equals the sampled reward: mean $4/3$ and variance $4/9$.
The true target value is $(3/4)(2/3)+(1/4)2=1$.

**Interpretation:** WIS often helps, but lower variance is not a universal theorem.
The counterexample is exact enumeration, not a simulation artefact.

## Bridge to this week's code

- [solutions.py](project/solutions.py), `exercise_1_first_visit_returns`, computes
  `Gs` backwards, then walks the original trajectory forwards with `seen`.
- `exercise_3_blackjack_mc_control` explicitly implements **every-visit** control.
- `exercise_4_ordinary_is` and `exercise_5_weighted_is` accept per-row
  probabilities. For trajectories, provide the full trajectory action products.
- `exercise_7_offpolicy_recruitment` accumulates those products; it implements
  trajectory IS, despite an older section comment mentioning per-decision IS.
- [recruitment.py](project/recruitment.py) is a simulator: the oracle samples the
  target directly. A logged production dataset has no such free oracle.

For the original per-decision correction and its temporal structure, see
[Precup, Sutton and Singh, Eligibility Traces for Off-Policy Policy Evaluation](https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf).
This changes weights on each reward to include only the actions preceding it;
it does not repair absent coverage.

## Choose your explanation depth: Why self-normalisation changes an estimator

Open the first layer for the idea, the second to calculate, and the third to check the assumptions. The worked examples above and investigation below provide the longer route.

<details class="learning-layer">
<summary>1 · Intuition</summary>

Importance weights make rare logged choices count more when those choices are common under the target policy. Ordinary IS divides by the number of observations. Weighted IS divides by their total importance mass, so a random denominator changes both stability and finite-sample bias.

</details>

<details class="learning-layer">
<summary>2 · Work a numerical case</summary>

With rewards (1,0) and weights (2,1), ordinary IS is (2×1+1×0)/2 = 1. Weighted IS is 2/3. If every weight is multiplied by ten, the weighted answer stays 2/3 while the ordinary answer becomes 10. Real probability ratios cannot be rescaled arbitrarily when claiming unbiased ordinary IS.

</details>

<details class="learning-layer">
<summary>3 · Derive it and state the limits</summary>

Under support and the correct behaviour probabilities, $E_\mu[wR]=E_\pi[R]$ and $E_\mu[w]=1$. With suitable laws of large numbers, $(\sum_iw_iR_i)/(\sum_iw_i)$ converges to the target mean. At finite sample size, $E[X/Y]$ generally differs from $E[X]/E[Y]$: this is why ratio consistency does not prove unbiasedness. When the total weight is zero, the logged sample supplies no empirical target-support mass; a denominator floor does not create that information.

</details>

**Try it in Jupyter:** jump to [`exercise_5_weighted_is` in the workbook](workbook.ipynb#cell-22), then run the setup cells above it. Expected answers are 1 and 2/3 for the two estimators. A result of zero from an all-zero-weight sample must be labelled a missing-information convention, not a measured policy value.

## Notebook investigation: separate visit order from return computation

Open [the actual Jupyter workbook](workbook.ipynb) and find **Exercises 1, 4 and 5**.
The [GitHub week folder](https://github.com/samuel-zhang01/STUDY-RL/tree/main/week-03-monte-carlo)
contains this notebook, its `project/` imports and tests. Follow the
[download and Jupyter workflow](../docs/LEARNING_MAP.md#download-and-open-the-jupyter-materials)
so the shared `common/` package is available.

In exercise 1, use the repeated-pair trajectory `[(0,0,1), (1,1,2), (0,0,3)]` with gamma 0.5.
Write the return at each position before selecting visits: `(2.75,3.5,3)`.
The first-visit result for `(0,0)` must contain 2.75, not 3. A test that checks only the number of
entries misses this bug; check the retained value too.

For exercises 4–5, set rewards `(1,0)` and target/logged probabilities `(0.75,0.25)/(0.5,0.5)`.
Predict IPS 0.75 and WIS 0.75, then change only the logged sample to contain the rewarded action twice.
IPS becomes 1.5 while WIS is 1. The change is a different finite sample, not a different true policy.
If you plot repeated samples, use one dot per independent dataset and label bias and variability
separately; one estimator need not win on both.

<details>
<summary>Check your interpretation: What would an all-zero importance-weight array tell you about WIS?</summary>

Its denominator is zero, so it is undefined. This sample contains no target-weighted evidence; replacing the denominator by epsilon does not make its policy value zero.

</details>

## Retrieve and transfer

<details><summary>1. Must every-visit MC be better because it uses more samples?</summary>

No. Those returns are correlated within an episode; raw visit count is not
independent sample count. Compare estimator error across independent episodes.
</details>

<details><summary>2. How does the matching-path weight change at ten binary decisions?</summary>

It is $2^{10}=1024$, while its behaviour probability is $1/1024$.
The expectation identity survives; reliable finite-sample estimation becomes harder.
</details>

<details><summary>3. Does a positive logged propensity in every observed row prove coverage?</summary>

No. Unsupported target actions may be absent from all rows. Coverage must be
checked against the logging policy's action support, including unobserved actions.
</details>

Arithmetic check: local Python reproduced the three returns, both IS estimates,
ESS = 2, and the exact zero-variance OIS counterexample.
