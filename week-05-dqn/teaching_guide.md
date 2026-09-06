# Week 05 teaching guide — why a Q-network needs careful targets

Use beside [the workbook](workbook.ipynb) and [the notes](notes.tex).
The main calculation is supervised regression with a deliberately frozen target.

## Before you start

Recall Week 04's Q-learning target $r+\gamma\max_bQ(s',b)$.
You also need neural-network gradients, batched tensors, and gradient descent.
A network shares parameters across states; a tabular update changes one entry.
Week 06 will change target selection, replay sampling, and the network head.

## Analogy: a student copying a periodically revised answer sheet

| Story element | DQN object |
|---|---|
| Student's current predictions | Online network $Q_\theta$ |
| Current answer sheet | Target network $Q_{\theta^-}$ |
| Exercises from earlier lessons | Replay transitions |
| Revising the answer sheet | Hard copy or Polyak update |
| Error on a chosen exercise | TD error |

Holding the answer sheet fixed briefly gives the student a stable regression target.
**Limit:** the sheet is produced from the student's own estimates, not an oracle.
A frozen target can be wrong, and replay data can cover the state space poorly.
Neither mechanism proves neural Q-learning will converge.

## Derivation: what receives the gradient

A replay row is $(s_i,a_i,r_i,s'_i,d_i)$, where $d_i$ flags true termination.
The online network returns one value per action. Select the value for the
recorded action: $q_i=Q_\theta(s_i,a_i)$.
Treat the next-state target as constant during this optimisation step:

$$y_i=r_i+\gamma(1-d_i)\max_bQ_{\theta^-}(s'_i,b),
\qquad \delta_i=y_i-q_i.$$

For half squared error $L_i=\tfrac12(q_i-y_i)^2$,

$$\frac{\partial L_i}{\partial q_i}=q_i-y_i=-\delta_i,
\qquad \nabla_\theta L_i=-\delta_i\nabla_\theta Q_\theta(s_i,a_i).$$

1. Predict **all** action values for each recorded state.
2. Gather the column corresponding to its recorded action.
3. Build $y_i$ without autograd tracking.
4. Minimise the mean loss; a positive TD error increases the selected prediction.

For unit-threshold Huber loss, replace $q_i-y_i$ in the derivative with
$\operatorname{clip}(q_i-y_i,-1,1)$.
This bounds the derivative with respect to the prediction, not necessarily the
full parameter-gradient norm, since the network Jacobian can be large.
Gradient-norm clipping in the agent is a separate operation.

## Example 1: one batch, two very different targets

**Predict:** which row should have no dependence on the next-state network?

Take batch size $B=2$, three actions, and online predictions

$$q_{\rm all}=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},
\qquad a=(2,0),$$

where actions use zero-based indices. Gathered predictions are $(3,4)$.
Rewards are $(1,-2)$; terminal flags are $(0,1)$; $\gamma=0.5$.
Target-network next values are $(0,4,2)$ and $(100,100,100)$.

First target: $y_1=1+0.5(4)=3$.
Second target: $y_2=-2+0.5(0)(100)=-2$.
Errors $\delta=(0,-6)$; unit Huber losses are $(0,6-0.5)=(0,5.5)$.
Mean loss is $2.75$; its derivatives with respect to gathered predictions
are $(0,0.5)$, including division by batch size.

**Interpretation:** descent pushes the second prediction down. The 100s have
no role on a terminal row. If `(B,1)` predictions are compared with `(B,)`
targets, broadcasting can create a `(B,B)` loss and compare the wrong rows.

## Example 2: a tiny linear network exposes the sign

**Predict:** if the target exceeds the prediction, should the weight increase?

Let $Q_\theta(s,a)=\theta x$, scalar feature $x=2$, initial $\theta=1$,
fixed target $y=3$, and learning rate $\eta=0.1$.
Prediction $q=2$; error $\delta=1$; half squared loss $0.5$.
Gradient $(q-y)x=(-1)(2)=-2$.
Descent gives $\theta'=1-0.1(-2)=1.2$, so $q'=2.4$ and loss $0.18$.

Now suppose the target itself were $y(\theta)=1+0.5\theta$.
At $\theta=1$, the residual is $q-y=0.5$.
The semi-gradient is $0.5(2)=1$;
the full sampled-residual gradient is $0.5(2-0.5)=0.75$.

**Interpretation:** stopping a target gradient changes the update mathematically.
Neither derivative is a typographical variant of the other. Optimising a sampled
squared residual is also different from squaring the expected Bellman residual.

## Example 3: how quickly does a target catch up?

**Predict:** does a soft update immediately copy the online network?

For one parameter, target $\theta^-=2$, online $\theta=10$, and $\tau=0.25$:
first soft update gives $0.75(2)+0.25(10)=4$;
second gives $0.75(4)+0.25(10)=5.5$.
With fixed online weights, the gap follows

$$\theta^-_k-10=(1-\tau)^k(2-10).$$

At $\tau=0$ the target never moves; at $\tau=1$ it copies immediately.
Periodic hard copies can be represented by a time-varying $\tau$ equal to
zero between copy times and one on a copy time. Constant-$\tau$ updates differ.

## Bridge to this week's code

- [agent.py](project/agent.py), `DQNAgent.learn`, uses `gather`, `torch.no_grad`,
  and `smooth_l1_loss` in exactly the sequence derived above.
- `DQNConfig.target_hard_every > 0` selects hard copies; otherwise
  `_soft_update` uses `target_tau` after learning.
- [solutions.py](project/solutions.py), exercises 1–3 and 6, isolate gather,
  target construction, target averaging, and Huber loss for hand comparisons.
- `exercise_7_baird_divergence` is a specific linear counterexample; one divergent
  example shows a guarantee fails, not that every combination must diverge.
- `exercise_9_action_value_trace` exposes model predictions. A large predicted
  best-versus-second-best gap is not measured causal lift over a uniform policy.
  Even exact immediate action lift over uniform would use best minus **mean**.

The original stabilisation mechanisms and their Atari setting appear in
[Mnih et al., Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236).
The continuation mask should follow
[Gymnasium's termination/truncation semantics](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## Retrieve and transfer

<details><summary>1. Why does replay remain off-policy after shuffling?</summary>

Shuffling changes sample order. It does not change which historical behaviour
policy selected each action. Sampling distribution and temporal correlation differ.
</details>

<details><summary>2. Is a low TD loss proof of a good policy?</summary>

No. It may only show agreement with inaccurate targets on a narrow replay
set. Evaluate return separately and inspect coverage and target calibration.
</details>

<details><summary>3. Can a large Huber residual still produce a large parameter gradient?</summary>

Yes. The scalar residual derivative is bounded, but it multiplies the network
Jacobian. That is why parameter-gradient clipping is a separate control.
</details>

Arithmetic check: local Python reproduced gather/targets, Huber mean 2.75,
scalar loss decrease 0.5 → 0.18, and target updates 2 → 4 → 5.5.
