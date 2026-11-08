# 8 · Adaptive data: choosing what you observe changes the analysis

**Question:** why is a bandit log different from a fixed independent sample?

**Use with:** exploration, adaptive evaluation and contextual logging. Sources: [S2.5, S4.7, S16.3 and S17.1](SOURCES.md#verified-section-map). The stopping example and logging application below are original course calculations.

## State what is known before the next outcome

Let $\mathcal F_{t-1}$ represent the recorded history before round $t$. A bandit chooses $A_t$ using this history and possibly fresh randomization. For a stationary reward model with conditionally fresh reward noise,

$$E[R_t\mid\mathcal F_{t-1},A_t]=\mu_{A_t}.$$

The conditional-mean statement is compatible with adaptive actions. It does not say that the chronological observed rewards are identically distributed: the learner changes the proportions of actions with different means.

For an arm, one can imagine a pre-generated independent reward stream exposed whenever that arm is selected. But the number exposed by a fixed horizon depends on previous outcomes. Treating that adaptive sample size as though it were chosen independently can invalidate naive finite-sample inference.

## Calculate a stopping-rule bias exactly

An arm gives independent Bernoulli rewards with mean 0.5. Sample once. If the first reward is 1, stop and report the sample mean. If it is 0, sample once more and report the mean of the two rewards.

| Observed path | Probability | Reported mean |
|---|---:|---:|
| 1, then stop | 1/2 | 1 |
| 0,0 | 1/4 | 0 |
| 0,1 | 1/4 | 1/2 |

The expected reported mean is

$$\tfrac12(1)+\tfrac14(0)+\tfrac14(1/2)=5/8=0.625.$$

Each draw was generated from the correct stationary arm. The stopping rule favored an early lucky result, so the reported mean is biased upward. A fixed two-draw sample mean instead has expectation 0.5.

This counterexample does not show that all adaptive estimators fail. It shows why the sampling and stopping protocol belongs in the analysis. An inference procedure valid at a prespecified horizon need not retain its error control if repeatedly checked and stopped at a favorable time.

## Separate exploration from final evaluation

Use exploration data to learn and choose a policy. Then freeze the policy and evaluate it with a prespecified protocol on fresh independent units. That answers a conditional fixed-policy question. Repeat the entire train/select/evaluate pipeline over independent runs if the target is performance of the training procedure.

A confidence bonus for action selection serves another purpose. A UCB bonus encodes an exploration rule under its theorem's assumptions; it is not automatically an empirical 95% confidence interval for the final learned policy's value. A Bayesian posterior used by Thompson sampling likewise has a prior/likelihood interpretation, not a generic frequentist deployment guarantee.

## Changing loggers can still be analyzed

In one-step OPE, a logger may depend on prior history. If the actual conditional action probability $b_t(A_t\mid X_t,\mathcal F_{t-1})$ is recorded and positive on target actions, importance weighting can retain a useful conditional-mean identity under suitable reward-mechanism assumptions.

However, weighted rows may be dependent, and a fixed independent-row variance formula need not apply. Record the temporal protocol and use an inference method justified for it. The identity, the estimator's variance and the validity of an interval are three separate questions.

## Check and transfer

1. Reverse the stopping rule: stop after an initial 0; sample again after 1. What is the expected reported mean?
2. Is selecting the best seed and reporting only its evaluation a training-procedure comparison?
3. Does a stationary environment guarantee independent action choices?

<details><summary>Answers</summary>

1. Paths 0, (1,0), (1,1) give $0.5(0)+0.25(0.5)+0.25(1)=0.375$.
2. No. That reports a selected outcome. If seed selection is a defined algorithm, its selection budget and independent final evaluation must be part of the evaluated procedure.
3. No. A learning policy intentionally depends on history. Separate the reward model's conditional assumptions from dependence in the action sequence.

</details>
