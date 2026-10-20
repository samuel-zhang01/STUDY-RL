# Week 11 teaching guide — what a logged experiment can tell you

**Course navigation:** [Week 10](../week-10-sac-td3/teaching_guide.md) ← **Week 11 / 26** → [Week 12](../week-12-rl-ranking-matching/teaching_guide.md)

[Routes](../docs/LEARNING_MAP.md) · [All weeks](../docs/STUDY_GUIDE.md#week-by-week-teaching-guides) · [Notation](../docs/GLOSSARY.md) · [Progress](../docs/PROGRESS.md)

[Lecture PDF](notes.pdf) · [Workbook](workbook.ipynb) · [Extended reading PDF](extended_reading.pdf)

**You are here:** When can logged actions evaluate a different policy?

**Recall first:** [Week 01](../week-01-mdps-bandits/teaching_guide.md), [Week 03](../week-03-monte-carlo/teaching_guide.md). If unfamiliar, use the [repair tasks](../docs/LEARNING_MAP.md#if-a-prerequisite-is-missing).

**Visual anchor:** [See the connected diagram](../docs/CONCEPT_ATLAS.md#logged-data-and-support). Use it to explain the mechanism before the derivation.

Use this guide before exercises 5–9. The estimator is only as meaningful as its sampling assumptions.

## Learning bridge

Follow [the staged learning path](../docs/LEARNING_PATH.md) when a topic is new.

- **Carry forward:** Retrieve week 1’s chosen-action-only feedback and week 3’s probability-ratio correction.
- **Add one difficulty:** Add context to the delivery route or study activity, then record the probability with which the logger chose each observed action. Future-state learning is absent in this one-step problem.
- **Try after the worked example:** Run [the synthetic study bandit](../learning_projects/study_bandit.py), then [the OPE experiment](../interactive/index.html#ope). Predict whether B-only logs can evaluate a target that always chooses C; implement `ips` in [the coding exercises](../learning_projects/exercises.py).

<details><summary>Check your prediction</summary>

They cannot: two worlds with different C rewards produce the same B-only logs. More B rows, clipping or a denominator floor do not supply C outcomes.

</details>

**Carry it onward:** [Ranking and matching](../week-12-rl-ranking-matching/teaching_guide.md) add exposure and examination mechanisms to this selection problem.


## Mechanism at a glance

```mermaid
flowchart LR
    B["Logging policy chooses action"] --> R["Observed reward"]
    R --> W["Weight by target / logger probability"]
    W --> E["Evaluate a fixed target policy"]
```

**Read the diagram:** Importance weights change which action distribution an average represents. They require support; unobserved actions cannot acquire rewards through a large weight.

## Connect to what you know

A contextual bandit observes a context, chooses one action, and observes its reward.
There is no action-dependent future-state term in its one-step objective.
From importance sampling, recall that a probability ratio changes which population an average represents.
Here the data come from logger $b$ while the policy of interest is $\pi$.

## Analogy: correcting an uneven survey

A survey invited people from one neighborhood four times as often as another.
A respondent from the rarely invited neighborhood represents more people in a population estimate.

| Survey concept | Logged bandit concept |
|---|---|
| Recorded invitation probability | Logged action propensity $b(a|x)$ |
| Desired population composition | Target policy $\pi(a|x)$ |
| Respondent's answer | Observed reward |
| Reweighting a response | Importance ratio $w=\pi/b$ |
| Group never invited | Missing action support |

**Limit:** no weight reconstructs the unknown responses of a never-invited group.
Similarly, zero support is an identification problem, not a divide-by-zero inconvenience.
Observational action choice can also depend on unrecorded variables related to outcomes.
A correct formula does not remove that confounding.

## Derivation: IPS and the DR correction

For this derivation assume independent evaluation rows, a fixed target policy,
correct logged probabilities, finite moments, and the same context/reward mechanism at evaluation.
The logger must have positive probability wherever the target does.
Condition on all variables used to choose the action that affect the reward, or randomize the action.
Write $q(x,a)=\mathbb E[R|x,a]$.

1. The target value is $V(\pi)=\mathbb E_X\sum_a\pi(a|X)q(X,a)$.
2. A logger-generated row has conditional expected weighted reward

   $$\mathbb E_b[wR|x]=\sum_a b(a|x)\frac{\pi(a|x)}{b(a|x)}q(x,a).$$

3. Cancel $b$: this equals $\sum_a\pi(a|x)q(x,a)$.
   Averaging rows gives $\widehat V_{IPS}=n^{-1}\sum_iw_iR_i$.
4. Add a model $\hat q$ and correct only its residual:

   $$Z_{DR}=\sum_a\pi(a|x)\hat q(x,a)+\frac{\pi(a|x)}{\hat b(a|x)}[R-\hat q(x,a)].$$

5. Treat the fitted nuisance functions as fixed on each evaluation row.
   Subtract the target conditional value and take expectations:

   $$\mathbb E[Z_{DR}|x]-V(\pi|x)
   =\sum_a\pi(a|x)\left(1-\frac{b(a|x)}{\hat b(a|x)}\right)[\hat q(x,a)-q(x,a)].$$

6. Either a correct propensity model or a correct conditional reward model makes this product zero.
   Both models can be wrong; their errors need not cancel.

Fit nuisance models on another split or cross-fit them before invoking this calculation.
A policy selected to maximize its own evaluation estimate also needs fresh evaluation data.
See [Dudík, Langford, and Li](https://arxiv.org/abs/1103.4601) for the estimator and assumptions.

## Worked example 1 — calculate three estimates from the same rows

**Predict:** must IPS and SNIPS agree on a small sample?
Use one context, actions A/B, logger $b=(0.75,0.25)$, target $\pi=(0.5,0.5)$.
Suppose the four logged actions are A, A, A, B with rewards 1, 0, 1, 1.
Fix an independently trained reward model $\hat q=(0.6,0.4)$.

**Calculate:**

| Row | Action | Reward | Weight | Weighted reward | DR contribution |
|---:|---|---:|---:|---:|---:|
| 1 | A | 1 | $2/3$ | $2/3$ | $0.5+(2/3)(0.4)=23/30$ |
| 2 | A | 0 | $2/3$ | $0$ | $0.5+(2/3)(-0.6)=1/10$ |
| 3 | A | 1 | $2/3$ | $2/3$ | $23/30$ |
| 4 | B | 1 | $2$ | $2$ | $0.5+2(0.6)=17/10$ |

Weights sum to $4$, and weighted rewards sum to $10/3$.
Thus IPS $=(10/3)/4=5/6=0.833333$ and SNIPS $=5/6$.
DR $=(23/30+1/10+23/30+17/10)/4=5/6$ as well.
ESS $=4^2/[3(2/3)^2+2^2]=16/(16/3)=3$.

**Interpret:** agreement here follows from the action counts and the fixed model;
it does not certify that $5/6$ is the true policy value.
If true means are $(0.6,0.4)$, that value is $0.5$, and this particular sample is lucky.
Even a perfect reward-mean model leaves reward noise in DR.

## Worked example 2 — normalization changes the answer

**Predict:** what happens if a two-row sample contains one A and one B, both rewarded 1?
Use the same policies as above.

**Calculate:** weights are $2/3$ and $2$.
IPS $=[2/3+2]/2=4/3$, exceeding the reward range $[0,1]$.
SNIPS $=[2/3+2]/[2/3+2]=1$.
ESS $=(8/3)^2/[(2/3)^2+2^2]=1.6$.

**Interpret:** IPS is an unbiased estimator under the stated assumptions,
not a number forced into the reward range on every dataset.
SNIPS with nonnegative weights stays within observed reward extrema when its denominator is positive.
That boundedness does not make it unbiased, and its finite-sample bias has no universal sign.

## Worked example 3 — an impossible evaluation

**Predict:** can more rows solve a logger that always chooses A?
Take $b=(1,0)$ and target $\pi=(0,1)$.

**Calculate:** every observed row has target weight zero.
The SNIPS numerator is 0 and denominator is 0: the estimate is undefined.
Consider two worlds: B always rewards 0 in world one and always rewards 1 in world two.
Both worlds generate identical A-only logs, yet target values are 0 and 1.

**Interpret:** the logs cannot identify the target value without extra assumptions or data.
Replacing a denominator by $10^{-12}$ supplies a number, not evidence about B.
A reward model can extrapolate to B, but that is a modeling assumption.

## Code bridge

- [ope.py](project/ope.py): `ips`, `snips`, and `doubly_robust` correspond to the formulas.
  `snips` floors its denominator: detect zero target overlap before interpreting its returned zero.
- `RidgeRewardModel.fit` is a convenient nuisance model; the workbook's same-log fit is a demo.
  Split the log to study evaluation validity, and report the split along with estimates.
- `ips_variance` estimates variance of the mean under independent rows.
  Adaptive logging, clustered users, or delayed outcomes require a matching uncertainty analysis.
- [agents.py](project/agents.py): `EpsilonGreedyLogger.propensities` is the full action distribution.
  The propensity is the chance of the action, including exploration *and* exploitation routes.
- [train.py](project/train.py): the oracle value is available because the environment is synthetic.
  Logs and oracle now use the same latent reward parameters; changing those
  between collection and evaluation would measure a different environment.
  A real logged dataset does not contain an oracle for unchosen actions.
- The environment has logistic reward means. LinUCB and LinTS are linear
  approximations here; their correctly specified linear-payoff regret bounds
  do not directly certify this experiment.

MIPS marginalizes over action embeddings: both numerator and denominator must marginalize actions,
using target and logger policies respectively. Embedding dimension alone does not determine variance.
See [Saito and Joachims](https://proceedings.mlr.press/v162/saito22a.html).

## Choose your explanation depth: How a positive propensity can still leave almost no evidence

Open the first layer for the idea, the second to calculate, and the third to check the assumptions. The worked examples above and investigation below provide the longer route.

<details class="learning-layer">
<summary>1 · Intuition</summary>

Overlap answers whether evaluation is possible in principle. Effective sample size asks whether this particular weighted sample is concentrated on a few rows. A logger can give every action positive probability while almost never choosing the action that the target always selects.

</details>

<details class="learning-layer">
<summary>2 · Work a numerical case</summary>

Consider 100 actions and epsilon-greedy exploration 0.05. If the target always selects a non-greedy action, its logging probability is 0.05/100 = 0.0005. A matching row has weight 2000; all other rows have weight zero. Among 10,000 rows, the expected number of matches is only 5. The expected count is not a guarantee: a realised log can have fewer or none.

</details>

<details class="learning-layer">
<summary>3 · Derive it and state the limits</summary>

If exactly $m>0$ rows match, then $\mathrm{ESS}=(2000m)^2/(2000^2m)=m$. More generally, for nonnegative weights with nonzero total, Cauchy–Schwarz gives $1\le\mathrm{ESS}\le n$. This diagnostic depends on the observed weights; it is not a confidence interval and does not account for reward noise, temporal dependence or a misspecified logger. With $m=0$, the formula is $0/0$: report no empirical overlap rather than inventing an effective sample. Increasing sample count only helps if the logging distribution supplies relevant outcomes.

</details>

**Try it in Jupyter:** jump to [`exercise_7_ess` in the workbook](workbook.ipynb#cell-25), then run the setup cells above it. Construct five weights of 2000 and 9,995 zeros: expect ESS 5. Repeating the rare-event experiment changes m; plot that distribution alongside estimator error instead of presenting the expected count as observed evidence.

## Notebook investigation: diagnose the log before trusting its estimate

Open [the actual Jupyter workbook](workbook.ipynb) and find **Exercises 5, 6 and 7**.
The [GitHub week folder](https://github.com/samuel-zhang01/STUDY-RL/tree/main/week-11-contextual-bandits-ope)
contains this notebook, its `project/` imports and tests. Follow the
[download and Jupyter workflow](../docs/LEARNING_MAP.md#download-and-open-the-jupyter-materials)
so the shared `common/` package is available.

Print the logged-action target probabilities, logging probabilities and their ratios before
computing an estimate. Sort the ratios and plot cumulative normalised weight against the fraction
of rows. If one row contributes nearly all the weight, a large dataset can still have little useful
information for this target. Report maximum weight and ESS alongside IPS/SNIPS.

Add the guide's two-world fixture: logger always A, target always B. No amount of repeated A data
can distinguish reward 0 from reward 1 for B. This is stronger than a numerical warning about a
small denominator. The current helper's denominator floor must not be interpreted as an answer.
For exercise 6, create a training/evaluation split before fitting the reward model; keep the target
fixed while evaluating. Comparing that estimate with the same-log demonstration teaches which
assumption changed, without guaranteeing one particular sample moves up or down.

<details>
<summary>Check your interpretation: Can excellent ESS rule out confounding or a wrongly recorded propensity?</summary>

No. ESS describes weight concentration. It does not validate the logging mechanism, outcome definition, causal assumptions, or policy-selection split.

</details>

## Retrieval and transfer

<details>
<summary>1. Why does a huge dataset sometimes have tiny ESS?</summary>

A few rows may carry almost all target weight. ESS measures weight concentration, not the raw count,
and does not independently validate support, causal assumptions, or the reward model.
</details>

<details>
<summary>2. Does a perfect reward model make DR equal the direct method row by row?</summary>

Only if rewards equal their conditional means. With stochastic outcomes, the residual has mean zero but is not zero on each row.
</details>

<details>
<summary>3. Why keep evaluation data separate when choosing among 100 policies?</summary>

Choosing the largest noisy estimate introduces selection bias. Fresh evaluation data measures the selected policy without reusing the noise that helped select it.
</details>

Arithmetic checks: all IPS/SNIPS/DR contributions and ESS values were recomputed in Python.
The two indistinguishable-world argument is an identification counterexample, not a simulation result.
