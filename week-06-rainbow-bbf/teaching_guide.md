# Week 06 teaching guide — separate the reasons for each improvement

Use with [the workbook](workbook.ipynb) and [the notes](notes.tex).
For each Rainbow component, identify the mathematical object it changes.

## Learning bridge

Follow [the staged learning path](../docs/LEARNING_PATH.md) when a topic is new.

- **Carry forward:** Retrieve [week 5’s](../week-05-dqn/teaching_guide.md) online network, target network and replay rows.
- **Add one difficulty:** Ask which object an improvement changes: action selection, replay weights, value representation or parameter resets. Keep every other component fixed for an ablation.
- **Try after the worked example:** In [the Double-DQN exercise](project/solutions.py), let online next values be (2,4) and target next values be (5,3), with reward 0 and gamma 0.9. Predict the label. The browser courier is a target analogy; the weekly agent is scalar Rainbow-lite, not a complete BBF reproduction.

<details><summary>Check your prediction</summary>

The online network selects the second action, then the target network evaluates it: 0+0.9×3=2.7. Standard DQN’s target here would be 4.5.

</details>

**Carry it onward:** [Policy gradients](../week-07-policy-gradients/teaching_guide.md) will change the learned object from action values to action probabilities.

## Before you start

You need Week 05's batched DQN target, stopped gradient, and replay buffer.
Importance sampling from Week 03 returns here, but it reweights **buffer rows**.
That is distinct from correcting which actions a behaviour policy logged.

## Analogy: revising with a stack of practice cards

| Revision choice | Algorithm component |
|---|---|
| Practise cards with large recent mistakes more often | PER |
| Compensate for overrepresented cards in the score | IS weight |
| Separate general topic difficulty from answer-specific differences | Dueling head |
| Use another estimate to grade the answer you selected | Double DQN |
| Track several possible scores rather than one mean | Distributional value head |

**Limit:** a large mistake may reflect irreducible reward noise, not a useful
learning opportunity. Repeatedly selecting it can waste the entire practice budget.
Also, two estimates derived from shared training are not independent graders.

## Derivation: PER weights from a change of measure

Fix a replay buffer with $N$ rows and per-row gradient vectors $g_i$.
The uniform-replay gradient is $g_U=(1/N)\sum_i g_i$.
Let priorities be $p_i=|\delta_i|+\epsilon>0$ and sampling probabilities

$$P(i)=\frac{p_i^\alpha}{\sum_jp_j^\alpha}.$$

1. Sampling from $P$ without correction estimates $\sum_iP(i)g_i$.
2. Multiply row $i$ by $w_i=1/[NP(i)]$.
3. Expand the corrected expectation:

$$\mathbb E_{i\sim P}[w_ig_i]
=\sum_iP(i)\frac{1}{NP(i)}g_i=\frac1N\sum_i g_i.$$

The identity assumes fixed positive probabilities and treats weights as constants
for this update. The tempered choice $w_i=[NP(i)]^{-\beta}$ is exact at
$\beta=1$ **before normalisation**; $\beta<1$ only partially corrects sampling.
Dividing by a fixed full-buffer maximum rescales the expected gradient.
Dividing by a random minibatch maximum can additionally change its expectation.

## Example 1: biased sampling, exact correction, and a normalisation trap

**Predict:** if the hardest card is sampled three times as often, does its gradient dominate?

Let $N=2$, priorities $(1,3)$, $\alpha=1$, hence $P=(0.25,0.75)$.
Take scalar gradients $g=(2,6)$. Uniform objective gradient is $(2+6)/2=4$.
Unweighted prioritised expectation is $0.25(2)+0.75(6)=5$.
At $\beta=1$, weights are $(2,2/3)$, giving

$$0.25(2)(2)+0.75(2/3)(6)=1+3=4.$$

Normalising both weights by their full-buffer maximum 2 yields $(1,1/3)$;
the expectation becomes 2, a common scalar rescaling.
But with batch size one and normalisation by that batch's maximum, the one
weight is always 1. The expectation reverts to 5.

**Interpretation:** distinguish exact importance correction from stabilising
normalisation. PER's correction is relative to uniform replay, not the online policy.
See [Schaul et al., Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).

## Derivation: C51 projection preserves probability mass

Use equally spaced atoms $z_i=V_{\min}+i\Delta z$ and input masses $p_j$.
For reward $r$, terminal flag $d$, and discount $\gamma$, transform atom $j$:

$$\tilde z_j=\operatorname{clip}(r+\gamma(1-d)z_j,V_{\min},V_{\max}).$$

Let $b_j=(\tilde z_j-V_{\min})/\Delta z$, $l=\lfloor b_j\rfloor$,
and $u=\lceil b_j\rceil$. When $l<u$, send masses
$p_j(u-b_j)$ to $l$ and $p_j(b_j-l)$ to $u$.
Their sum is $p_j(u-l)=p_j$, because adjacent indices differ by one.
If $l=u$, put all of $p_j$ on that atom; naively applying both zero factors loses mass.
Without clipping, interpolation also preserves each transformed atom's mean.
Clipping can change the mean: the support range is a modelling choice.

## Example 2: project three atoms, then terminate

**Predict:** will the projection leave total probability equal to one?

Take support $z=(-1,0,1)$, masses $p=(0.2,0.3,0.5)$,
$r=0.5$, $\gamma=0.5$, and $d=0$.
Transformed atoms are $(0,0.5,1)$.

- Mass 0.2 lands exactly at 0.
- Mass 0.3 at 0.5 splits equally: 0.15 to 0 and 0.15 to 1.
- Mass 0.5 lands exactly at 1.

The projected masses are $(0,0.35,0.65)$, summing to 1.
The old mean is $-0.2+0+0.5=0.3$;
Bellman mean is $0.5+0.5(0.3)=0.65$, equal to the projected mean.

If $d=1$, every atom transforms to 0.5 regardless of the old distribution.
Projection becomes $(0,0.5,0.5)$. The true terminal return is deterministic
0.5; the two-bin representation is approximation caused by the fixed grid.
If terminal reward is 2, clipping gives $(0,0,1)$ with represented mean 1.

**Interpretation:** two occupied bins need not mean the environment is uncertain.
See [Bellemare, Dabney and Munos, A Distributional Perspective on RL](https://proceedings.mlr.press/v70/bellemare17a.html).

## Example 3: Double and Dueling are different operations

Online next values $(5,4)$ select action 0; target values $(3,6)$ evaluate it at 3.
With reward 1 and $\gamma=0.5$, Double DQN target is $1+0.5(3)=2.5$.
Vanilla DQN uses the target maximum and gives $1+0.5(6)=4$.
A lower target on this one example does not establish unbiasedness.

For a dueling head $V=5$ and raw advantages $(1,3)$, their mean is 2,
so $Q=(4,6)$ and the mean of Q is 5.
Adding 10 to both raw advantages gives $(11,13)$ and exactly the same Q.
Thus the **centred** residual is determined by Q, but the raw head offset
remains redundant. $V$ here is the action-uniform mean of Q, not generally $V^\pi$.
See [Wang et al., Dueling Network Architectures](https://proceedings.mlr.press/v48/wangf16.html).

## Bridge to this week's code

- [solutions.py](project/solutions.py), exercises 1, 2, 3, and 6 implement the
  Double target, dueling combination, normalised weights, and projection above.
- [agent.py](project/agent.py), `RainbowAgent.learn`, trains a **scalar**
  Rainbow-lite agent. C51 is a separate workbook component, not its trained head.
- `NoisyLinear.reset_noise` draws shared input/output noise factors; resulting
  weight entries are dependent, not independent Gaussian noises.
- `_shrink_and_perturb` is an educational reset mechanism. The full BBF method
  includes architecture, auxiliary prediction, replay, and schedule choices.
  It is not obtained by adding one reset line to this agent.
- In [train.py](project/train.py), `ablate` compares configurations; short runs
  diagnose wiring, while claims of performance improvement require repeated evaluation.

For the complete BBF experimental recipe, consult
[Schwarzer et al., Bigger, Better, Faster](https://proceedings.mlr.press/v202/schwarzer23a.html).

## Episode boundaries in the project

A true terminal ends the Bellman bootstrap. An external time limit ends
collection but retains the final state's continuation value. For example,
with reward 1, gamma 0.9, and final value 10, the truncated target is
$1+0.9(10)=10$; the terminal target is 1. The trainer uses the termination
flag for this mask and either flag to reset the environment.
See [Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## Retrieve and transfer

<details><summary>1. What does PER alpha = 0 do?</summary>

Every positive priority becomes $p_i^0=1$, so $P(i)=1/N$.
Unnormalised IS weights are all 1 for every beta.
</details>

<details><summary>2. Why must C51 treat an exact atom carefully?</summary>

Floor and ceiling coincide. Both naive interpolation coefficients become zero;
all mass must instead be assigned to that exact atom.
</details>

<details><summary>3. Does a copied target network provide independent noise?</summary>

No. Online and target weights share their training history and are identical
just after copying. Double DQN reduces coupled selection/evaluation error without
promising perfect independence or zero bias.
</details>

Arithmetic check: local Python reproduced all sampling expectations, both C51
projections and means, Double target 2.5, and dueling outputs (4, 6).
