# Statistics that makes an RL experiment interpretable

An agent can improve on a graph while the experiment still fails to answer your question. These lessons help you name the quantity being estimated, identify the independent observations, calculate uncertainty, and decide what a result supports.

Use a lesson when its weekly question arises. You do not need to finish a statistics textbook before starting the bandit playground. Every number below is an invented teaching fixture unless a section explicitly describes an executed experiment. A calculation checks arithmetic under a model; it does not verify that the model describes a real deployment.

## Choose the question you need

| Question | Bridge lesson | First useful weeks |
|---|---|---|
| What changes when I condition on an action or a context? | [1 · Probability and conditioning](01-probability-conditioning.md) | 1–3, 11, 15, 26 |
| How can an average be predictable while individual outcomes are noisy? | [2 · Expectation and variance](02-expectation-variance.md) | 2, 4, 6–8, 10, 16, 20 |
| What does another seed add, and why is SD different from SE? | [3 · Sampling and uncertainty](03-sampling-uncertainty.md) | 1, 3–7 |
| How uncertain is the difference between two agents? | [4 · Intervals and comparisons](04-intervals-comparisons.md) | 6, 9–10, 13, 17–19 |
| Which things should I resample together? | [5 · Resampling and dependence](05-resampling-dependence.md) | 3, 5, 8, 11, 13, 20, 24–26 |
| Why can a fitted model look good and still fail on new data? | [6 · Regression and validation](06-regression-validation.md) | 5, 9, 12, 14–16, 18–19, 21, 26 |
| Can logs collected by one policy evaluate another? | [7 · Importance weighting](07-importance-weighting.md) | 3, 11–12, 14, 24–25 |
| What changes when the learner chooses the data? | [8 · Adaptive data](08-adaptive-data.md) | 1, 17, 25 |
| What exactly can I claim after the experiment? | [9 · Statistical claims](09-statistical-claims.md) | 21–25, and every project report |

## A 45-minute first route

1. Spend ten minutes on Lesson 1's conditional-mean example. Explain which information is known before a draw.
2. Spend ten minutes on Lesson 2's two-outcome distribution. Calculate the expectation and variance before checking.
3. Spend fifteen minutes on Lesson 3's four independent runs. Calculate the SD and the SE and explain their different units of uncertainty.
4. Spend ten minutes on Lesson 4's paired differences. State the target quantity and one reason an interval could be invalid.

Then return to your weekly guide and the statistics investigation near the end of its workbook. The notebooks use the same distinction between a fixed trained policy's evaluation noise and uncertainty across the complete training procedure.

## The experiment card

Before running a comparison, fill in these seven lines:

- **Question and estimand:** for example, mean held-out return difference, A minus B, over a specified environment distribution.
- **Independent unit:** a newly trained agent, independent environment instance, prompt, patient, or another justified unit.
- **Fixed versus repeated:** what is held fixed and what is retrained, sampled, or refitted on each repetition?
- **Split and selection:** where are hyperparameters and checkpoints selected, and which data remain untouched for final evaluation?
- **Summary and uncertainty:** mean, SD, SE, interval method, independent-unit count, and pairing or clustering.
- **Failure test:** overlap, rare events, missing outcomes, distribution shift, or dependence that would defeat the interpretation.
- **Bounded conclusion:** the population, policy, conditions and limitations that the result actually covers.

The [source catalog](SOURCES.md) records the supplied textbook editions, verified PDF locations, and the boundary between textbook results and original course applications. The originals and instructor manual are local study inputs; they are not part of the public website or downloadable weekly bundles.
