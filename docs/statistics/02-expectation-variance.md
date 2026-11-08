# 2 · Expectation and variance: the average is not an outcome

**Question:** how can an estimator be correct on average while individual updates are noisy?

**Use with:** Bellman targets, MC/TD, distributional RL, policy-gradient baselines, SAC and RLOO. Sources: [S4.1, S4.3, S4.5, S4.7 and R4.4–4.7](SOURCES.md#verified-section-map).

## Calculate the whole small distribution

A terminal delivery pays 4 with probability 0.25 and 0 otherwise. Its expected reward is

$$E[R]=0.25(4)+0.75(0)=1.$$

No delivery pays 1. Expectation is a probability-weighted average over possible outcomes. Its units are reward units. Variance measures squared deviation from that mean:

$$\operatorname{Var}(R)=0.25(4-1)^2+0.75(0-1)^2=3.$$

Equivalently, $E[R^2]-E[R]^2=4-1=3$. Standard deviation is $\sqrt3\approx1.732$ reward units. Variance is in squared reward units.

**Predict:** if the discounted target is $Y=2+0.5R$, which quantity changes by 0.5 and which changes by its square?

$$E[Y]=2+0.5(1)=2.5,\qquad \operatorname{Var}(Y)=0.5^2(3)=0.75.$$

Adding a fixed 2 shifts the mean without changing spread. Multiplying by 0.5 scales deviations by 0.5 and squared deviations by 0.25.

## Conditioning separates two sources of variation

Suppose the next state is L or H, each with probability 0.5. The conditional reward means are 1 and 3; within either state, reward variance is 1.

The overall mean is $0.5(1)+0.5(3)=2$. The law of total variance gives

$$\operatorname{Var}(R)=E[\operatorname{Var}(R\mid S')]+\operatorname{Var}(E[R\mid S']).$$

The first term is 1. The second is $0.5(1-2)^2+0.5(3-2)^2=1$. Total variance is 2. Replacing a sampled reward by its exact conditional mean removes within-state noise and leaves variance 1.

This explains a limited variance reduction mechanism behind conditional averaging, such as an Expected SARSA next-action average with fixed Q-values. It does not prove that an entire learning algorithm has lower final control error: Q-values, action frequencies and future data all change during training.

## Why covariance matters

For random variables $X,Y$,

$$\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y).$$

Suppose two rewards are identical copies of the **same draw**: $Y=X$, each with variance 3. Their sum has variance 12, because covariance is 3. Two independent draws instead give variance 6. A trajectory can contain rewards tied together by the same hidden circumstance, so counting transitions does not automatically count independent evidence.

For an agent difference $A-B$, the covariance term has a minus sign. Shared test instances can reduce difference variance when they induce positive covariance. That is the rationale for a paired design, not a guarantee that arbitrary identical numeric seeds produce valid pairing.

## Correct mean does not mean best estimator

An estimator's mean squared error decomposes as

$$E[(\widehat\theta-\theta)^2]=\operatorname{Var}(\widehat\theta)+\operatorname{Bias}(\widehat\theta)^2.$$

An unbiased estimate with variance 4 has MSE 4. An estimate with bias 1 and variance 1 has MSE 2. This fixture explains why a small bias can accompany a useful variance reduction. It does not license hiding bias or assuming its sign. SNIPS, target bootstrapping and regularization require their own analyses.

## Check and transfer

1. If reward is multiplied by 10, what happens to mean, variance and SD?
2. Does a return distribution with SD 20 imply an SE of 20 for an average over independent runs?
3. Replace the exact conditional mean above with a biased fitted model. Does the variance calculation establish its MSE?

<details><summary>Answers</summary>

1. Mean is multiplied by 10, variance by 100, SD by 10.
2. No. SD describes outcome spread. The uncertainty of an independent sample mean depends on the number of units; see [Lesson 3](03-sampling-uncertainty.md).
3. No. The fitted model introduces estimation error and possibly bias. Lower output variability alone does not establish closeness to the true target.

</details>
