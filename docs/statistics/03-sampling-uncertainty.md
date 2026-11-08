# 3 · Sampling uncertainty: what another seed actually adds

**Question:** what is the difference between variable performance and an uncertain mean?

**Use with:** repeated-seed training and Monte Carlo evaluation. Sources: [S6.2–S6.5, S7.1 and R6.2–6.3](SOURCES.md#verified-section-map).

## Name the estimand before counting observations

Two valid questions need different repetitions:

- **Fixed-policy evaluation:** hold one trained policy fixed and sample new independent evaluation episodes. Estimate its mean return under the stated environment distribution.
- **Training-procedure evaluation:** retrain on independent training runs, evaluate each with a defined protocol, then compare run-level summaries. Estimate the performance of the whole training procedure, including its randomness.

One agent evaluated on a thousand episodes does not become a thousand independently trained agents. Episode-level error bars can be narrow while training outcomes vary greatly. Report what you repeated and what stayed fixed.

## Calculate SD and SE from four independent runs

Take run-level held-out means $8,10,12,14$. These are invented values for four independent repetitions of a fixed training/evaluation protocol.

The sample mean is $\bar x=44/4=11$. Deviations are $-3,-1,1,3$ and their squares sum to 20. The sample variance and SD are

$$s^2=20/(4-1)=20/3,\qquad s=\sqrt{20/3}\approx2.582.$$

The $n-1$ denominator accounts for estimating the mean from the same independent sample. The estimated standard error of the mean is

$$\widehat{SE}(\bar x)=s/\sqrt n=2.582/2\approx1.291.$$

**Interpret:** SD describes between-run spread. SE estimates how much the sample mean would fluctuate across repetitions of this four-run experiment. Both have reward units, but they describe different random quantities.

A mean ± SD plot shows spread, not a 95% confidence interval. A mean ± SE plot is also not automatically a 95% interval. Label the band, its independent unit and sample count.

## Derive the square-root rule

For independent identically distributed $X_i$ with variance $\sigma^2$,

$$\operatorname{Var}(\bar X)=\frac1{n^2}\sum_{i=1}^n\operatorname{Var}(X_i)=\frac{\sigma^2}{n}.$$

The standard deviation of the sample mean is $\sigma/\sqrt n$. Four times as many independent units halves this quantity if the population variability stays the same. It does not halve SD across individual runs.

The law of large numbers describes convergence under its assumptions. The central limit theorem describes a limiting standardized sampling distribution under its assumptions. Neither says that every mean based on 30 observations is well approximated by a normal distribution. Rare catastrophic outcomes, heavy tails and dependence can make simple approximations poor.

## A hidden clustering trap

Imagine four independently trained policies, each evaluated on 100 episodes. Copy each policy's mean into 100 identical rows and compute an SE using $n=400$. For the four means above, the copied rows give sample variance $2000/399$ and a misleading SE $\sqrt{(2000/399)/400}\approx0.11194$, rather than the run-level SE 1.291. No new training outcome has been observed.

This is pseudoreplication. Keep episodes nested inside their training runs when estimating variation across training procedures. With a hierarchical design, you may also study evaluation noise separately, but the estimand and resampling hierarchy must match that design.

## Check and transfer

1. If the same spread $s\approx2.582$ is observed across 16 independent runs, what SE results?
2. Is a million replay-buffer transitions enough to justify a million-unit SE?
3. What changes if the four run values above are checkpoints of one ongoing training run?

<details><summary>Answers</summary>

1. $2.582/4\approx0.6455$. This is a planning calculation holding spread fixed, not a prediction of the new observed SD.
2. No. Transitions can share trajectories, policies and environment history. The independent unit depends on the question.
3. They are dependent observations from one run. The independent-run SE formula is not justified. Use independent retraining for a training-procedure comparison, and treat checkpoint selection as part of the protocol.

</details>

Next use [Lesson 4](04-intervals-comparisons.md) to turn an SE into a carefully qualified interval and compare two agents on paired independent units.
