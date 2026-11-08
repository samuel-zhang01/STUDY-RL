# 5 · Resampling and dependence: keep related evidence together

**Question:** which objects should a bootstrap sample draw?

**Use with:** trajectories, replay buffers, self-play matches, prompts, patients and OPE. Sources: [S4.5, S6.2, R15.3–15.4](SOURCES.md#verified-section-map). Cluster resampling below is a course application of those sampling ideas; the cluster assumptions must be justified for the actual design.

## What a bootstrap approximates

The ordinary nonparametric bootstrap samples observations with replacement from the empirical distribution, recomputes the statistic, and uses its repeated distribution as an approximation to sampling uncertainty. The point is to mimic the variation in the experiment. Resampling the wrong object mimics a different experiment.

For $x=(0,2)$, there are four equally likely ordered two-draw bootstrap samples:

| Sample | Mean |
|---|---:|
| (0,0) | 0 |
| (0,2) | 1 |
| (2,0) | 1 |
| (2,2) | 2 |

The bootstrap mean has expectation 1 and variance $[1+0+0+1]/4=0.5$. This exact enumeration shows what resampling computes. It does not establish good confidence-interval coverage with only two observations.

## Calculate why duplicated rows are misleading

Suppose three independent prompts have outcomes 0, 1 and 2, and each is duplicated into four token rows. The twelve recorded values are four zeros, four ones and four twos. The overall mean is 1.

If the independent unit is a prompt, the three prompt values have sample SD 1 and estimated SE $1/\sqrt3\approx0.57735$.

If the duplicates are incorrectly treated as twelve independent observations, their sample variance is $8/11$, and the reported SE becomes

$$\sqrt{(8/11)/12}=\sqrt{2/33}\approx0.24618.$$

The smaller number arose from pretending to have more independent evidence. A prompt bootstrap draws prompts and carries all their token rows together. It does not draw twelve unrelated tokens.

## Design the resampling unit

- For a fixed policy evaluated on independent episodes, resample episodes.
- For a training-procedure comparison, resample independent training runs and preserve the defined per-run evaluation summary.
- For both agents tested on the same independent instances, resample paired differences or matched pairs together.
- For patients with repeated measurements, keep all measurements of a sampled patient together.
- For completions sampled from one prompt, distinguish conditional completion variability from variability over prompts. A prompt-level generalization claim needs independent prompts.

Clusters should be independent or satisfy an appropriate weaker model. A handful of clusters provides little information about between-cluster variation. Unequal cluster sizes also force an estimand choice: an average patient and an average measurement weight the population differently.

## Frozen versus refitted uncertainty

A bootstrap of held-out predictions from one fitted reward model measures uncertainty conditional on that fitted model and the held-out sampling design. To approximate uncertainty of the full model-building pipeline, the repetition may need to repeat training, nuisance fitting, hyperparameter selection and evaluation at the appropriate independent level. Freezing those steps gives a narrower conditional question; label it accordingly.

For OPE ratio estimates, inspect resamples with zero total weight. Discarding them silently conditions on a selected set of bootstrap worlds. A large number of bootstrap draws cannot repair absent support or a poor independent-unit choice.

## Permutation is a different operation

A permutation test rearranges labels according to a justified null distribution or randomization design. It is not automatically valid because a computer can shuffle rows. Paired sign flips need an appropriate randomized-pair design or symmetry/exchangeability assumption; unrestricted shuffling can break prompt, patient or trajectory structure.

## Check and transfer

1. If every patient has ten scans, what should be held together for a patient-level generalization claim?
2. Does increasing bootstrap draws from 1,000 to 100,000 create more independent training runs?
3. If one prompt has 100 completions and another has two, should they receive equal weight?

<details><summary>Answers</summary>

1. All scans belonging to one patient. Split patients before fitting preprocessing or models.
2. No. More draws reduce simulation noise in the bootstrap approximation, not uncertainty due to limited original evidence.
3. It depends on the estimand. An average-prompt target weights prompts equally; an average-record target weights their numbers of records. State the choice before looking at results.

</details>

Try the website's **Clustered bootstrap** experiment, then explain which experiment each resampling scheme represents. Use [Lesson 9](09-statistical-claims.md) to write a conclusion with the correct unit and scope.
