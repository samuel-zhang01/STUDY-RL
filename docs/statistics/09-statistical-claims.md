# 9 · Statistical claims: write the sentence the experiment supports

**Question:** what can you say after an encouraging result?

**Use with:** every project report, especially reward hacking, constraints, shielding and applied audits. Sources: [S7.1, S8.1, S9.1 and R7.3, R8.2–8.3](SOURCES.md#verified-section-map). The claim examples below are course applications.

## Start with the estimand, not the winner

An estimand is the quantity the question asks about: for example, mean held-out return difference between two frozen training procedures over a named environment distribution. Define its population, unit, metric, horizon and comparison. “Better” without those details is underspecified.

Prediction, policy value, fairness metrics and safety rates can concern different populations and denominators. An average reward improvement does not establish improvement for every subgroup or satisfaction of an individual safety constraint.

## Calculate a claim that changes with the denominator

An invented evaluation contains two groups. Group L has 90 decisions with 81 successes; group H has 10 decisions with 5 successes. Success rates are $81/90=0.9$ and $5/10=0.5$. The decision-weighted overall success rate is

$$\frac{81+5}{90+10}=0.86.$$

If the target population has equal group proportions, the standardized rate is

$$0.5(0.9)+0.5(0.5)=0.70.$$

Both calculations use the same group-specific observations. The first describes the sample mixture; the second estimates a different population mixture under a transport assumption about group-specific rates. A label such as “86% reliable” hides which population and outcome were measured.

The group gap is $0.9-0.5=0.4$. This observed gap needs uncertainty and a definition of the fairness question. It does not prove a causal effect of group membership, nor establish fairness under every criterion.

## Report a difference with its independent units

Suppose a prespecified comparison yields mean held-out return difference 1.5 and an illustrative 95% interval $[0.58,2.42]$ across four matched independent instances, using a normal model for differences.

A bounded sentence is: “On the stated synthetic instance distribution, A's mean return exceeded B's by 1.5 units; the normal-model paired interval was approximately 0.58 to 2.42 across four independent instances.”

Add the small-sample limitation, exact selection protocol and whether training was repeated. Avoid expanding the sentence into a claim about every task, customer outcome or deployment. Those require further evidence.

## Null tests answer a limited question

A p-value is a probability, under a specified null model, of a result at least as extreme as the observed one according to the chosen statistic. It is not the probability the null is true, the probability the result occurred by chance, or an effect size.

If twenty independent true null hypotheses are tested at 0.05 without adjustment, the chance of at least one false rejection is

$$1-0.95^{20}\approx0.6415.$$

This is an illustrative independent-tests calculation. Real ablations are often correlated. Predefine primary comparisons, report the whole comparison family and use a justified multiplicity or selection analysis. Showing only the smallest p-value conceals the procedure that generated it.

## Absence, rare events and safety

No observed failures is evidence about the sampled conditions, not a proof of impossibility. [Lesson 4](04-intervals-comparisons.md) gives the independent Bernoulli upper bound. For a shield with exact known dynamics, a formal invariant under a model is different from an estimated empirical failure rate. Report both with their assumptions; do not let either silently stand in for the other.

A constrained policy can satisfy an estimated average budget while violating a per-step limit or a subgroup budget. State the cost definition and denominator. Audit those quantities separately from return.

## A report you can reproduce

Include the experiment card from the [statistics entry page](README.md#the-experiment-card), the source revision, parameters, independent-unit count, aggregation rule, interval method and the code or notebook that produces the summary. Label invented fixtures, saved historical curves and newly executed results distinctly. An unexecuted calculation should not be written as an observed benchmark.

## Check and transfer

1. Why does the 0.86 sample rate differ from the 0.70 equal-group rate?
2. Does an interval excluding zero prove a practically useful improvement?
3. What extra evidence would support a new population or environment claim?

<details><summary>Answers</summary>

1. They use different group weights and therefore different estimands.
2. No. Compare the effect and interval with a predefined practically meaningful threshold and the full costs or harms.
3. Evaluation sampled from that target, or a justified transport model with explicit assumptions and sensitivity checks. Reusing the same synthetic task does not establish that extension.

</details>
