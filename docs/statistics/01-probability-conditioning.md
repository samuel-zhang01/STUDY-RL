# 1 · Probability and conditioning: what information is available?

**Question:** why can the same action have different average rewards in different logs?

**Use with:** Weeks 1–3, 11, 15 and 26. You need fractions and weighted averages. Sources: [S2.4, S2.5 and R3.6](SOURCES.md#verified-section-map).

## Start with the experiment

A probability model names possible outcomes and their probabilities. In a bandit, a row might contain a context $X$, selected action $A$, and observed reward $R$. They are random before the experiment. A recorded row, such as `(easy, B, 1)`, is one realization.

Conditioning means restricting attention to outcomes consistent with information we have observed. For events with $P(A)>0$,

$$P(B\mid A)=\frac{P(A\cap B)}{P(A)}.$$

The denominator changes the population being averaged. Conditioning is an operation on a distribution; it does not by itself describe an intervention or establish a causal relationship.

## Calculate: a route selected for easier deliveries

Invent two equally common delivery contexts: easy and hard. Route B succeeds with probabilities 0.9 and 0.3 respectively. A logger selects B with probability 0.8 in easy contexts and 0.2 in hard contexts.

**Predict:** will the success rate among B's logged deliveries equal B's success rate if we assign it to everyone?

First find the mass of each kind of B row:

$$P(E,A=B)=0.5(0.8)=0.4,\qquad P(H,A=B)=0.5(0.2)=0.1.$$

Therefore $P(A=B)=0.5$ and

$$P(E\mid A=B)=0.4/0.5=0.8,\quad P(H\mid A=B)=0.2.$$

The logged conditional mean is

$$E[R\mid A=B]=0.8(0.9)+0.2(0.3)=0.78.$$

Assigning B across the original equally weighted contexts instead gives

$$0.5(0.9)+0.5(0.3)=0.60.$$

The reward law within each context stayed the same. The context mixture changed. Neither 0.78 nor 0.60 is an arithmetic mistake: they answer different questions. The second quantity has an intervention interpretation only under a model that justifies using these context-specific response probabilities under assignment.

## Independence is a claim you can break

Variables $U,V$ are independent when their joint probabilities factor: $P(U=u,V=v)=P(U=u)P(V=v)$ for all relevant values. Independence is stronger than zero correlation. In this fixture, action and context are dependent: $P(A=B\mid E)=0.8$ differs from $P(A=B)=0.5$.

Randomizing the action with probability 0.5 in both contexts makes $A$ independent of $X$. It does not make rewards independent of context, and it does not make two observations from the same patient independent. Always name which variables and which sampling units an independence statement concerns.

## A Bellman connection

A value is a conditional expectation, not a guaranteed outcome. For a fixed policy,

$$V^\pi(s)=E_\pi[R_{t+1}+\gamma V^\pi(S_{t+1})\mid S_t=s].$$

The conditioning fixes the current state; the expectation averages the action, next state and reward randomness. The Markov assumption says the modeled state contains the history information needed for the next-outcome distribution given the action. A short sensor reading need not have that property.

## Check and transfer

1. Without looking, explain why the logged B mean is 0.78.
2. Change the hard-context B probability from 0.2 to 0.8. What happens?
3. If B is never chosen in hard contexts, can B's hard-context success rate be learned from these logs alone?

<details><summary>Answers and assumptions</summary>

1. Four-fifths of B rows come from easy contexts, so B's conditional average overweights easy deliveries.
2. B is chosen with probability 0.8 in both contexts. Its logged context mix becomes 50/50 and its logged mean becomes 0.60.
3. No. A hard-context response model could extrapolate, but the missing outcomes are not identified by these logs without further assumptions. This is the overlap issue in [Lesson 7](07-importance-weighting.md).

</details>

In your workbook, label the data-generating distribution and the distribution in the question before writing an average. That one step prevents many apparently mysterious changes in reported performance.
