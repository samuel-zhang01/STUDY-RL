# 7 · Importance weighting: change the average, check the support

**Question:** can another policy be evaluated without taking its actions?

**Use with:** Monte Carlo off-policy evaluation, contextual bandits, ranking and applied audits. Sources: [S2.4, S4.1, S7.1 and R4.4–4.7](SOURCES.md#verified-section-map) supply probability and estimation foundations. The OPE estimators' algorithm-specific assumptions and references remain in Weeks 3 and 11.

## Derive the cancellation before using the formula

For a one-step problem, let $X$ be context, $A$ action, $R$ reward, $b(a\mid x)$ the logger, and $\pi(a\mid x)$ a fixed target policy. Let $q(x,a)=E[R\mid X=x,A=a]$. Under the shared reward/context mechanism and appropriate identification assumptions,

$$V(\pi)=E_X\sum_a\pi(a\mid X)q(X,a).$$

If $b(a\mid x)>0$ wherever the target has positive probability, define $w=\pi(A\mid X)/b(A\mid X)$. Conditioning on the context,

$$E_b[wR\mid x]=\sum_a b(a\mid x)\frac{\pi(a\mid x)}{b(a\mid x)}q(x,a)=\sum_a\pi(a\mid x)q(x,a).$$

The cancellation requires support. Correct algebra does not verify logged propensities or remove hidden confounding in observational actions.

## Calculate IPS, SNIPS and weight concentration

Use one context, two actions A/B, logger probabilities $(0.75,0.25)$ and target probabilities $(0.5,0.5)$. Two logged rows contain A and B, both with reward 1. Their weights are $2/3$ and 2.

$$\widehat V_{IPS}=\frac{2/3+2}{2}=4/3,\qquad \widehat V_{SNIPS}=\frac{2/3+2}{2/3+2}=1.$$

IPS can leave the reward range in a particular sample despite being unbiased under the stated assumptions. SNIPS normalizes the random total weight; it stays in the observed reward range for nonnegative weights and a positive denominator, but is generally biased in finite samples.

The weight-concentration effective sample size is

$$ESS=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}=\frac{(8/3)^2}{(2/3)^2+2^2}=1.6.$$

It describes concentration of observed weights. It does not incorporate reward noise, dependent rows, model misspecification or missing actions, and is not a confidence interval.

## Two worlds the logs cannot distinguish

Change the logger to $b=(1,0)$ and target to $\pi=(0,1)$. All observed rewards come from A. One world has B reward always 0; another has B reward always 1. They produce identical logs and target values 0 and 1.

No number of additional A-only rows identifies the B value. A denominator floor, weight clipping and a reward-model extrapolation cannot create the missing information. Extrapolation can answer a model-dependent question if that extra assumption is stated.

Distinguish **population support**, a property of the logging distribution, from **observed overlap**, a property of one finite sample. If B has probability 0.001 but never appears in a small sample, population support exists while the sample contains no direct B outcomes. The sample alone cannot prove population support either way.

## Uncertainty needs the whole design

For independent rows, one can examine variation of weighted contributions; for trajectories, prompts or patients, preserve their grouping. A bootstrap of a fixed target and fitted nuisance models gives a conditional analysis. If the target or nuisance models were learned using these rows, use justified sample splitting or cross-fitting and identify which parts are repeated.

With a rare target action, some bootstrap resamples have zero target weight. Report that failure frequency and investigate the original overlap. Silently omitting such resamples gives an interval conditional on the resampling estimate being defined.

## Check and transfer

1. Change the target to the logger. What happens to every weight?
2. If all observed rewards are 0, can ESS still be high?
3. Can a logger's nominal exploration rate substitute for event-specific recorded probabilities?

<details><summary>Answers</summary>

1. Weights are 1 wherever the logger can act; IPS and SNIPS become the ordinary sample mean.
2. Yes. ESS depends only on weights, so it cannot certify useful reward evidence or exclude unobserved rare rewards.
3. No. Eligibility filters, ties, mixtures and evolving policies can change the actual selection probability. Reconstruct and verify the probability used at the decision.

</details>

The website's **Importance weights** experiment makes missing overlap visible. Use it before interpreting a polished OPE plot.
