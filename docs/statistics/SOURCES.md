# Textbook sources and application boundaries

These bridge lessons are original course explanations based on the two supplied books. They use new numerical fixtures rather than copied textbook exercises or instructor solutions. The tables below identify passages checked in the local PDFs on 8 November 2026. Page numbers mean **one-based PDF viewer pages**, not printed page numbers. They identify verified discussions, and do not necessarily identify the first page of a section.

## The local books

**S:** Kyle Siegrist, *Probability, Mathematical Statistics, and Stochastic Processes*, LibreTexts compilation dated 19 March 2025; local file `textbooks/Full.pdf`, 1,558 PDF pages. The PDF contains section-level source links and a detailed licensing appendix. Links below reproduce those source addresses. Consult the relevant section license before adapting its expressive content; the course prose and fixtures here are newly written.

**R:** Sheldon M. Ross, *Introduction to Probability and Statistics for Engineers and Scientists*, fifth edition, Academic Press/Elsevier, 2014; local file `textbooks/statbook3.pdf`, 730 PDF pages. The supplied file includes an instructor manual at PDF pages 8–66 before the textbook begins at PDF page 67. Its pagination changes again later, so do not calculate citations using one global page offset. Use chapter/section plus the checked PDF page. The course does not distribute this PDF, its manual, or their exercises and solutions.

## Verified section map

| ID | Section and verified PDF page | Supports |
|---|---|---|
| S2.4 | [Conditional Probability](https://stats.libretexts.org/@go/page/10132), §2.4, PDF p.181 | Conditioning, probability ratios and Bayes reasoning |
| S2.5 | [Independence](https://stats.libretexts.org/@go/page/10133), §2.5, PDF p.197 | Independence as a property of a joint distribution |
| S4.1 | [Definitions and Basic Properties](https://stats.libretexts.org/@go/page/10156), §4.1, PDF p.439 | Expectation and linearity |
| S4.3 | [Variance](https://stats.libretexts.org/@go/page/10158), §4.3, PDF p.460 | Variance and squared deviations |
| S4.5 | [Covariance and Correlation](https://stats.libretexts.org/@go/page/10160), §4.5, PDF p.478 | Variance of sums and paired differences |
| S4.7 | [Conditional Expected Value](https://stats.libretexts.org/@go/page/10163), §4.7, PDF p.507 | Iterated expectation and conditional-variance reasoning |
| S6.2 | [The Sample Mean](https://stats.libretexts.org/@go/page/10179), §6.2, PDF p.817 | Sampling variability of an average |
| S6.3 | [The Law of Large Numbers](https://stats.libretexts.org/@go/page/10180), §6.3, PDF p.828 | Convergence under stated sampling assumptions |
| S6.4 | [The Central Limit Theorem](https://stats.libretexts.org/@go/page/10181), §6.4, PDF p.834 | Approximate sampling distributions; not a universal small-sample guarantee |
| S6.5 | [The Sample Variance](https://stats.libretexts.org/@go/page/10182), §6.5, PDF p.844 | The sample-variance denominator and descriptive spread |
| S7.1 | [Estimators](https://stats.libretexts.org/@go/page/10189), §7.1, PDF p.896 | Bias, variance and mean squared error |
| S7.3 | [Maximum Likelihood](https://stats.libretexts.org/@go/page/10191), §7.3, PDF p.917 | Fitting a probability model |
| S7.4 | [Bayesian Estimation](https://stats.libretexts.org/@go/page/10192), §7.4, PDF p.929 | Prior, likelihood and posterior distinction |
| S8.1 | [Introduction to Set Estimation](https://stats.libretexts.org/@go/page/10200), §8.1, PDF p.960 | Interval procedures and coverage |
| S8.2 | [Estimation in the Normal Model](https://stats.libretexts.org/@go/page/10201), §8.2, PDF p.966 | Normal-model mean intervals |
| S9.1 | [Introduction to Hypothesis Testing](https://stats.libretexts.org/@go/page/10211), §9.1, PDF p.994 | Null models and error rates |
| S16.3 | [Introduction to Discrete-Time Chains](https://stats.libretexts.org/@go/page/10290), §16.3, PDF p.1322 | Transition probabilities and Markov structure |
| S17.1 | [Introduction to Martingales](https://stats.libretexts.org/@go/page/10299), §17.1, PDF p.1481 | Conditional-mean statements relative to observed history |
| R3.6 | Conditional Probability, §3.6, PDF p.130 | Finite-event conditioning |
| R4.4–4.7 | Expectation, Properties of Expected Value, Variance, Covariance and Variance of Sums, §§4.4–4.7; checked PDF pp.171, 175, 182, 185 | Means, variances and covariance terms |
| R6.2–6.3 | The Sample Mean and The Central Limit Theorem, §§6.2–6.3, PDF pp.271, 273 | Independent sampling and averaging |
| R7.2 | Maximum Likelihood Estimators, §7.2, PDF p.298 | Parametric fitting and model assumptions |
| R7.3 | Interval Estimates, §7.3, PDF p.309 | Confidence intervals and their sampling interpretation |
| R8.2–8.3 | Significance Levels and Tests Concerning the Mean of a Normal Population, §§8.2–8.3, PDF pp.360–361 | Hypothesis-test interpretation |
| R9.2–9.4 | Least Squares Estimators, Distribution of the Estimators, Statistical Inferences About Regression Parameters, §§9.2–9.4; PDF pp.421, 423, 429 | Regression fitting and conditional inference |
| R15.2 | Random Numbers, §15.2, PDF p.680 | Simulation as a numerical experiment |
| R15.3 | The Bootstrap Method, §15.3, PDF pp.683–689 | Resampling empirical observations |
| R15.4 | Permutation Tests, §15.4, PDF pp.690–697 | Randomization/exchangeability reasoning |

## What is an application rather than a textbook result?

The numerical bandits, seed comparisons, frozen-policy experiments, synthetic patients, prompt clusters and overlap counterexamples are course-created applications of the cited probability/statistics concepts. A source for a sample mean does not establish that RL seeds are independent, that a logger has support, or that a patient split identifies a causal effect. Those require the experiment's design and additional assumptions.

Modern RL, LLM training, safety algorithms and causal-discovery methods retain their primary research references in weekly lectures and reading packs. These two textbooks are not evidence for algorithm rankings, benchmark reproduction, clinical benefit, deployment safety, or newer algorithm theorems.

## Recheck a citation locally

Open the named local PDF at the viewer page and confirm the section heading. For example, Siegrist §4.7 is at PDF page 507; Ross §15.3 is discussed at PDF page 683. Read the assumptions surrounding the equation as well as the equation itself. Record the edition and PDF page when adding another source. Do not place the original PDFs or manual into public assets or weekly archives.
