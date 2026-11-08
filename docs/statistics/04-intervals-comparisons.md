# 4 · Intervals and comparisons: uncertainty in the difference

**Question:** does agent A outperform B by an amount that matters for this experiment?

**Use with:** ablations, policy comparisons, LLM evaluation and safety rates. Sources: [S4.5, S8.1–S8.2, S9.1, R7.3 and R8.2–8.3](SOURCES.md#verified-section-map).

## An interval is a procedure with assumptions

For independent normally distributed unit-level values with unknown variance, a two-sided mean interval is

$$\bar x\pm t_{n-1,0.975}\frac{s}{\sqrt n}.$$

Its 95% coverage describes repetitions of the **procedure** under that model: about 95% of the resulting random intervals contain the fixed population mean. After observing one interval, the parameter is still fixed in this frequentist description. A Bayesian credible interval instead describes posterior probability under a prior and likelihood; the two statements are not interchangeable.

Apply the formula to Lesson 3's four run values: mean 11, SE $\sqrt{5/3}\approx1.291$, and $t_{3,0.975}\approx3.182$. The margin is $3.182(1.291)\approx4.109$, giving approximately $[6.89,15.11]$. Four units give a wide interval. The normal-model assumption matters strongly at this size; a successful formula call does not validate it.

## Calculate a paired agent comparison

Evaluate both agents on four independently generated test instances. Pair by **the same instance**, not merely by a numeric seed label.

| Instance | Agent A | Agent B | Difference A − B |
|---|---:|---:|---:|
| 1 | 10 | 9 | 1 |
| 2 | 20 | 18 | 2 |
| 3 | 30 | 29 | 1 |
| 4 | 40 | 38 | 2 |

The target is the mean difference over the instance distribution. Differences have mean $1.5$, squared deviations summing to 1, sample variance $1/3$, and SE $\sqrt{1/3}/2\approx0.288675$. Under an independent normal model for the differences, the paired t interval is

$$1.5\pm3.182(0.288675)\approx[0.581,2.419].$$

The instance difficulty makes both agents' raw scores vary widely. Pairing removes that shared component from each difference. Algebraically,

$$\operatorname{Var}(A-B)=\operatorname{Var}(A)+\operatorname{Var}(B)-2\operatorname{Cov}(A,B).$$

Positive covariance can help. Bad pairing or negative covariance can hurt. Reordering B's rows destroys the intended matched comparison.

## Statistical and practical thresholds differ

Suppose a practically useful gain must exceed 2 reward units. The illustrative interval excludes zero, but it includes values below 2. Evidence of a positive difference is not the same as evidence of a sufficiently large difference. Predefine the useful threshold and report the effect in understandable units.

Do not decide whether a difference is distinguishable from zero by checking overlap of separate agent confidence intervals. Analyze the difference directly with the design's covariance structure. Selecting the best of twenty agents on the same evaluation set also changes the interpretation: the selection procedure belongs inside the experiment or needs a fresh final holdout.

## A rare-event boundary case

With zero failures in $n$ independent Bernoulli trials and a stable failure probability $p$, an exact one-sided 95% upper bound solves $(1-p)^n=0.05$:

$$p_U=1-0.05^{1/n}.$$

For $n=100$, $p_U\approx0.02951$. Zero observed failures does not establish zero risk. If trials share a hidden failure cause, or a deployment distribution changes, this independent Bernoulli statement no longer applies unchanged.

## Check and transfer

1. What independent unit supports the paired interval above?
2. Would 100 evaluations of each one of two trained agents establish uncertainty across training seeds?
3. What happens to the zero-failure bound at $n=1000$?

<details><summary>Answers</summary>

1. An independently generated matched test instance; the four differences must follow the assumed sampling model.
2. No. That estimates evaluation noise conditional on those trained agents. Repeat training to assess the training procedure.
3. It becomes $1-0.05^{1/1000}\approx0.002991$. It remains an assumption-dependent upper bound, not a guarantee of deployment safety.

</details>

The website's **Paired comparisons** experiment shows why an agent-difference distribution can be much tighter than the two raw-score distributions. Use the weekly notebook to inspect interval sensitivity rather than treating a four-value fixture as a benchmark result.
