# Week 24 teaching guide: separate learning, evaluation, and the decision claim

Use alongside [the notes](notes.tex) and [the workbook](workbook.ipynb).
This is a synthetic educational matching problem, not a validated production hiring system.

## The connections that matter

Week 14's offline learning uses actions that already have observed outcomes.
Week 11's off-policy evaluation asks how a different policy would perform under those logs.
Week 22's multiplier attempts to trade reward against a specified cost constraint.
These steps use related data, but their conclusions require different assumptions.
The project's action is binary recommend/skip, not a complete ranking or sequential hiring process.

## Analogy: learning from a restaurant's old menu choices

Suppose a restaurant logged which meal it offered and whether the customer enjoyed it.

| Restaurant concept | Matching quantity |
|---|---|
| Customer and available meal | Context: job and candidate features |
| Offer or skip | Binary action |
| Whether an offered meal was enjoyed | Observed action-dependent reward |
| Probability the old restaurant offered it | Logging propensity |
| New offering rule | Target policy |
| Budget on unequal offering rates | Chosen population constraint |

A skipped meal's missing enjoyment outcome is not evidence that the meal was bad.
In this simulator, skipping has reward zero by definition, while latent relevance is separately available.
Real unobserved outcomes cannot be filled in merely because the simulator supplies them.
The analogy also stops at causality: changing a group field is a sensitivity probe,
not automatically a realistic counterfactual intervention on a person.

## Derivation: IQL in a one-step bandit

Let a record be `(x,a,r)` from the logger `mu`.
A one-step episode has no future reward, so the Q regression target is simply `r`.
Let `tau∈(0,1)` be the expectile parameter and `beta>0` the AWR inverse temperature.

1. Fit action values on logged actions:

   $$L_Q=\mathbb E_D[(Q(x,a)-r)^2].$$

2. Fit a state value through asymmetric squared residuals:

   $$L_V=\mathbb E_D[|\tau-\mathbf1\{Q-V<0\}|(Q(x,a)-V(x))^2].$$

3. Give high-advantage logged actions larger cloning weights:

   $$w(x,a)=\exp(\beta[Q(x,a)-V(x)]),\qquad
   L_\pi=-\mathbb E_D[w(x,a)\log\pi(a|x)].$$

4. At one fixed context, with an unrestricted categorical policy and unclipped weights,
   the optimum normalises weighted behaviour frequencies:

   $$\pi^*(a|x)=\frac{\mu(a|x)w(x,a)}{\sum_b\mu(b|x)w(x,b)}.$$

The last result follows by a Lagrange multiplier for `sum_a pi(a|x)=1`.
It makes the support limit visible: a zero behaviour frequency remains zero.
Function approximation may generalise elsewhere, but that is not observed support.
[IQL](https://arxiv.org/abs/2110.06169) uses expectile value learning and advantage-weighted policy extraction.

## Worked example 1: compute an expectile and weighted policy

**Predict.** An upper expectile should lie above the ordinary mean of action values,
and weighted regression should move probability toward the higher-value logged action.

At one context, let `mu(skip)=mu(recommend)=0.5`, values `(0,1)`, `tau=0.75`, and `beta=2`.

1. For `0<V<1`, the weighted value loss is
   `0.5×0.25 V² + 0.5×0.75(1−V)²`.
2. Set its derivative to zero:
   `0.25V−0.75(1−V)=0`, giving `V=0.75`.
3. Advantages are `−0.75` and `+0.25`.
4. Weights are `exp(−1.5)=0.223130` and `exp(0.5)=1.648721`.
5. Recommend probability is
   `1.648721/(0.223130+1.648721)=0.880797`.
6. Expected reward under the supplied values rises from `0.5` to `0.880797`.

**Interpret.** This exact one-context result assumes values and frequencies are known.
With `mu(recommend)=0`, the policy extraction has no positive-action evidence to weight.
Also, because `V(x)` is common to both actions, its exponential factor cancels in the
ideal categorical normalisation; it still affects finite optimisation and clipped weights.

## Worked example 2: SNIPS with the propensity of the action actually taken

Let four logged decisions have rewards `(1,0,1,0)`.
Their recorded action propensities are `(0.5,0.25,0.5,0.75)`.
The fixed target assigns the same recorded actions probabilities `(0.75,0.5,0.25,0.25)`.

**Predict.** The target upweights some records and downweights others;
the unweighted reward mean `0.5` need not remain the estimated target value.

1. Importance weights are `(1.5,2,0.5,1/3)`.
2. Weighted reward sum is `1.5×1+2×0+0.5×1+(1/3)×0=2`.
3. Weight sum is `1.5+2+0.5+1/3=13/3`.
4. SNIPS is `2/(13/3)=6/13≈0.461538`.
5. IPS is `2/4=0.5`; the difference is the random normalising denominator.
6. Effective sample size is
   `(13/3)²/(1.5²+2²+0.5²+(1/3)²)=338/119≈2.840336`.

**Interpret.** Four records contain less than four equally weighted records' worth of
information by this weight-concentration diagnostic. ESS is not an exact confidence interval.
If a target action has positive probability where the logger had zero probability,
no denominator clipping can create the missing counterfactual evidence.
The support requirement comes from the importance-sampling change of measure;
[counterfactual risk minimisation](https://proceedings.mlr.press/v37/swaminathan15.html) builds on logged-policy correction.

## The logger's bias: check the threshold before telling the story

In [project/data.py](project/data.py), group 0 gets score `0.6×relevance` and group 1 gets `0.4×relevance`.
With relevance `0.7`, scores are `0.42` and `0.28`: both are below threshold `0.5`.
At epsilon `0.1`, each gets recommendation probability `epsilon/2=0.05`, not `0.1`.
With relevance `0.9`, scores are `0.54` and `0.36`:
recommendation probabilities now differ, `0.95` versus `0.05`.
This demonstrates uneven coverage without assuming every different score changes the action.

## Bridge to the actual project and audit claims

- [project/offline_iql.py](project/offline_iql.py): `_expectile_loss` weights residual signs;
  `iql_pretrain` clips the AWR weight at `100` and uses terminal Q targets.
- [project/online_ppo_lag.py](project/online_ppo_lag.py): `_sample_rollout` computes a rolling
  group gap, while `ppo_lag_finetune` updates the multiplier using full-rollout parity.
  Window history is not in the policy state, so this is a heuristic cost update.
- [project/audit.py](project/audit.py): `snips_ope` expects target probability for the
  **logged action**, not always the recommend action.
- `full_audit` uses a prefix of the supplied interaction list. The shipped pipeline
  reuses the dataset used for fitting; independent evaluation requires an explicit split.
- `calibration_curve` receives policy recommendation probabilities versus relevance labels.
  Decision probabilities are not predicted outcome probabilities; a diagonal is not required by RL.

For 40 recommendations among 100 group-0 contexts and 20 among 50 group-1 contexts,
DP gap is `|0.4−0.4|=0`. Raw recommendation counts differ but conditional rates agree.
If no qualified examples from a group are observed, its equal-opportunity rate is undefined;
a numerical fallback of zero should not be reported as measured zero performance.

## Retrieve and transfer

<details>
<summary>1. Why split data before fitting and choosing a target policy?</summary>

A target chosen using evaluation rewards can exploit evaluation noise.
Holding out data before fitting and selection gives evidence about that fixed policy's generalisation.

</details>

<details>
<summary>2. If everyone is recommended, what happens to demographic parity?</summary>

Both group rates are one, so DP gap is zero. That says nothing about selectivity,
capacity, qualification, or whether recommendations are useful. Include decision-relevant metrics.

</details>

<details>
<summary>3. Does flipping only the group feature prove counterfactual fairness?</summary>

No. It probes direct model sensitivity while holding all other features fixed.
A causal counterfactual requires assumptions about which other features would also change.

</details>
