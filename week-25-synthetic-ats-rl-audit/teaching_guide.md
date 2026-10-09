# Week 25 teaching guide: make an audit claim that the evidence can support

Use with [the notes](notes.tex) and [the workbook](workbook.ipynb).
This directory contains a synthetic ATS demonstration and a hypothetical architecture.
All data and calculations are constructed examples. They do not validate an external system.

## Learning bridge

Follow [the staged learning path](../docs/LEARNING_PATH.md) when a topic is new.

- **Carry forward:** Retrieve [week 24’s](../week-24-capstone-safe-matching/teaching_guide.md) separate learning/evaluation/constraint claims and week 12’s action-definition discipline.
- **Add one difficulty:** Audit the actual logged action: choosing one candidate from a fixed shortlist is different from ranking every candidate. Synthetic evidence supports the stated simulator first.
- **Try after the worked example:** Read [the audit implementation](project/audit.py), then write a one-paragraph claim with action, population, data-generating mechanism, support, evaluation split and limits. If your simulator’s new policy scores higher, does that establish a real hiring improvement?

<details><summary>Check your prediction</summary>

No. It supports a result under this synthetic reward function and shortlist process; real-world outcomes, consent, causal validity and transport were not measured by that run.

</details>

**Carry it onward:** [Causal biology](../week-26-causal-discovery-biological-data/teaching_guide.md) carries over overlap and audit discipline, then asks what observational evidence identifies about an intervention.

## What this capstone brings together

The key prerequisites are contextual bandits, off-policy evaluation, and constrained optimisation.
Start by stating the action: this project chooses **one candidate from a fixed shortlist**.
It does not evaluate every possible ranking over the entire candidate population.
Then state the outcome: simulated hire/reward is generated from a known synthetic function.
A higher estimate in this simulator is evidence about that setup, not a production deployment claim.

## Analogy: replaying an old tournament

You have records of which competitor an old selector entered and how they performed.
You want to assess a new selector without rerunning every event.

| Tournament object | OPE quantity |
|---|---|
| Competitors eligible for an event | Logged shortlist/slate |
| Old selector's actual choice | Logged action `a_i` |
| Probability of that choice | Behaviour propensity `mu_i` |
| New selector's probability for the same choice | Target probability `pi_i` |
| Performance of the entered competitor | Observed reward `r_i` |

The new selector can be evaluated through reweighting only where the old selection process
provided support. Competitors never eligible for any logged event cannot be restored by weights.
Eligibility and within-shortlist randomisation are separate stages of support.
The analogy also shows why a shortlist selection probability is not a full-ranking probability.

## Derivation: IPS and the doubly robust correction

Fix a context/shortlist `x` and let `q(x,a)=E[r|x,a]`.
Assume a stable reward mechanism, correctly recorded action probabilities,
and `mu(a|x)>0` wherever the target `pi(a|x)>0`.

1. Change measure over the action:

   $$\mathbb E_{a\sim\mu}\!\left[\frac{\pi(a|x)}{\mu(a|x)}r\right]
   =\sum_a\pi(a|x)q(x,a).$$

2. Averaging independent evaluation records gives IPS:
   `V_IPS=(1/N) sum_i w_i r_i`, with `w_i=pi_i/mu_i`.
3. SNIPS instead divides by `sum_i w_i`; its denominator is random,
   so it is generally biased in finite samples, with no fixed bias direction.
4. Given a separately fitted reward estimate `q_hat`, add a residual correction:

   $$\hat V_{DR}=\frac1N\sum_i\left[
   \sum_a\pi(a|x_i)\hat q(x_i,a)
   +w_i(r_i-\hat q(x_i,a_i))\right].$$

5. With correct propensities, the expected residual correction is
   `sum_a pi(a|x)[q(x,a)−q_hat(x,a)]`, cancelling reward-model error.

If the reward model is correct, the residual also has mean zero under suitable sampling
and fitting independence even when propensity estimates are imperfect.
Neither branch permits unsupported target actions or unrestricted adaptive reuse of evaluation data.
[Dudík et al.](https://arxiv.org/abs/1103.4601) derive doubly robust policy evaluation;
[Swaminathan and Joachims](https://proceedings.mlr.press/v37/swaminathan15.html) study learning from logged bandit feedback.

## Worked example 1: calculate three estimators and an ESS

There are two candidates A and B, repeated over three identical contexts.
Behaviour is `(0.5,0.5)`; target is `(0.75,0.25)`.
Logged actions are `(A,B,A)` and rewards `(1,0,0)`.
A separately supplied reward model predicts `q_hat(A)=0.6`, `q_hat(B)=0.2`.

**Predict.** The reward model's estimate can differ from the observed weighted reward;
DR should add corrections that reflect those residuals.

1. Weights: `(1.5,0.5,1.5)`; weighted reward sum `1.5`; weight sum `3.5`.
2. IPS: `1.5/3=0.5`.
3. SNIPS: `1.5/3.5=3/7≈0.428571`.
4. Direct model value: `0.75×0.6+0.25×0.2=0.5` for every record.
5. DR contributions are:
   first A: `0.5+1.5(1−0.6)=1.1`;
   B: `0.5+0.5(0−0.2)=0.4`;
   second A: `0.5+1.5(0−0.6)=−0.4`.
6. DR mean: `(1.1+0.4−0.4)/3=1.1/3≈0.366667`.
7. ESS: `3.5²/(1.5²+0.5²+1.5²)=49/19≈2.578947`.

**Interpret.** An individual DR contribution can lie outside the reward range.
Different point estimates do not establish which estimator is “right” for this finite sample.
If the target equals behaviour, all weights are one and IPS/SNIPS reduce to the reward mean.
If the shortlist excluded B entirely, a target choosing B would violate support regardless of sample size.

## Worked example 2: equal selection shares need not mean equal selection rates

Suppose group A has 80 eligible candidates and group B has 20.
The system selects 10 from each group.

**Predict.** Looking only at selected candidates will hide differences in selection probability.

1. Shares among the 20 selected candidates are `A_share=10/20=0.5` and `B_share=0.5`.
2. The selected-share ratio is `B_share/A_share=1`.
3. Group-conditional selection rates are `A_rate=10/80=0.125` and `B_rate=10/20=0.5`.
4. Their B/A ratio is `0.5/0.125=4`, and absolute gap is `0.375`.
5. Equal conditional rates of `0.25` would instead select 20 A candidates and 5 B candidates.
   Selected shares would be `0.8` and `0.2`, with share ratio `0.25`.

**Interpret.** The denominator defines the metric.
The project's `selection_rates` accepts only selected groups, so it computes shares.
It cannot compute eligible-group conditional rates without counts of eligible candidates.
A synthetic ratio threshold is a chosen teaching target, not a legal finding.
Repeated opportunities also require specifying whether the unit is a candidate,
a candidate-job pair, a shortlist exposure, or a decision.

## One posterior update: connect the audit to the bandit

Take ridge prior matrix `A=I_2`, vector `b=(0,0)`, feature `x=(1,2)`, and reward `r=1`.
The update is `A'=I+xx^T=[[2,2],[2,5]]`, `b'=(1,2)`.
Its determinant is `10−4=6`; inverse is `(1/6)[[5,−2],[−2,2]]`.
Thus posterior mean is `theta_hat=(1/6,1/3)`.
Candidate features `(1,0)` and `(0,1)` receive posterior-mean scores `1/6` and `1/3`.
Thompson sampling draws a parameter before scoring; it does not always choose the larger mean.
A sampled argmax is one random decision, not the marginal selection probability of that candidate.

## Bridge to the actual project: read these boundaries first

- [project/train.py](project/train.py): `collect_logs` randomises only within the
  heuristic shortlist. Its epsilon mass does not create support outside that shortlist.
- `train` fits `LinTSReranker` and a reward model on `training_log`, then evaluates
  a fixed posterior-mean exploitation policy on a fresh log stream. The stream
  shares the synthetic population and shortlist rule, but supplies no fitting labels.
- [project/bandit.py](project/bandit.py): `score` samples a fresh parameter on every call.
  The training driver freezes `theta_hat` for its OPE target; it does not evaluate
  the marginal Thompson-sampling distribution. The `LinTS` result key is retained
  for compatibility. Synthetic `ground_truth_values` give expected performance
  under the evaluation job distribution for comparison with estimator noise.
- [project/audit.py](project/audit.py): `run_audit` audits the supplied choices.
  The choices returned by `train` are **behaviour-policy** choices, not deployed LinTS choices.
- [project/constrained.py](project/constrained.py): `train_lagrangian` is a deterministic
  score-adjustment heuristic with a nondecreasing clipped-shortfall multiplier, not PPO.
- [project/ope.py](project/ope.py): `snips_ci` resamples records. Its percentile interval
  need not contain the original point estimate, and row bootstrap assumes appropriate independence.

The test asserting a particular LinTS-versus-uniform ordering is an empirical benchmark,
not an algorithmic guarantee. A failure deserves investigation rather than a changed theorem.
`equalised_odds_gap` checks both true-positive and false-positive gaps; missing
conditional outcomes yield NaN. `selection_rates` reports selected-group shares,
which cannot be interpreted as group-conditional applicant selection rates
without exposure denominators. Its toy ratio is not a four-fifths hiring audit.

## Retrieve and transfer

<details>
<summary>1. Can a misspecified reward model alone explain why DR is pessimistic?</summary>

No. With correct propensities the expected correction removes that model error under the
usual evaluation assumptions. Finite samples, fitting reuse, and target definition also matter.

</details>

<details>
<summary>2. What information must accompany a logged action probability?</summary>

The context at decision time, eligible action set/slate, actual selected action,
behaviour-policy version and randomisation rule, probability for that selected action,
and outcome definition plus observation time. Selection stages outside the slate need their own support analysis.

</details>

<details>
<summary>3. If a shadow ranker never changes displayed candidates, can it observe its own outcomes?</summary>

No. It observes outcomes generated by the active behaviour policy.
Shadow scores support debugging and supported counterfactual evaluation, not direct online causal validation.

</details>
