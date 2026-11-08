# 6 · Regression and validation: a good fit answers a conditional question

**Question:** why can lower training error accompany worse decisions?

**Use with:** Q-functions, reward models, contextual bandits, offline learning, world models and causal-biology prediction. Sources: [S7.3, R7.2 and R9.2–9.4](SOURCES.md#verified-section-map). Dataset splitting and leakage examples are course applications, not a new textbook theorem about modern neural agents.

## Separate the target from the fitted object

A regression model estimates a conditional mean or another specified functional of outcomes given inputs. Squared-error minimization targets $E[Y\mid X=x]$ at the population level when the expectation exists. A fitted function $\widehat q(x)$ is an estimate produced from finite data and an optimization procedure.

To see why the mean appears, write $Y=m(X)+\epsilon$ with $E[\epsilon\mid X]=0$. Conditional on $X=x$,

$$E[(Y-f(x))^2\mid x]=\operatorname{Var}(Y\mid x)+(m(x)-f(x))^2.$$

The cross term vanishes because the conditional residual mean is zero. The second term is minimized at $f(x)=m(x)$. This statement does not guarantee that a finite neural model finds that function, nor that the conditional mean supports a causal interpretation.

## Calculate an overfitting example

Two training points are $(x,y)=(0,0)$ and $(1,1)$. Model A predicts $f_A(x)=x$. Model B predicts $f_B(x)=x+10x(x-1)$. Both have zero training squared error.

On an independent test point $x=0.5$ with outcome $y=0.5$, model A predicts 0.5 and has squared error 0. Model B predicts $0.5+10(0.5)(-0.5)=-2$ and has squared error $(-2-0.5)^2=6.25$.

Two interpolators can fit every supplied training observation and disagree badly elsewhere. This is an original counterexample, not an empirical claim that a particular architecture behaves this way.

## Freeze the right split

Use training data to fit parameters, validation data to choose hyperparameters or checkpoints, and a final held-out set for the predefined comparison. If the final set influences another model choice, it has become part of selection. Report the complete selection protocol or use fresh final evaluation data.

Split the independent entity before fitting anything that learns from outcomes or distributions: imputation, scaling when appropriate, feature selection, reward modeling and augmentation can all leak information. If the target is new patients, split patients; if it is new tasks, split tasks. Randomly splitting correlated rows can test memorization of the same entity rather than generalization.

For a simple arithmetic illustration, suppose one patient's 20 records all encode a stable patient-specific outcome. A random row split can place that patient in both training and test. A model using an identifier may predict those test records perfectly without learning anything about a new patient.

## The RL distribution adds another distinction

A low TD loss means agreement with targets on the sampled training distribution. Those targets may be bootstrapped from a changing model, and a greedy policy can visit regions that replay data barely cover. Falling TD loss is therefore not equivalent to rising held-out return.

Likewise, reward-model preference accuracy measures a predictive task on a particular pair distribution. Optimizing the learned score changes the policy's output distribution. Re-evaluate actual task outcomes and specified failures on a held-out protocol rather than treating the learned score as the outcome of interest.

In model-based RL, reconstruction loss and one-step prediction error are useful diagnostics. Planning asks about multi-step consequences under actions selected by the planner; that distribution can differ from the model's training distribution.

## Check and transfer

1. Does a correct regression conditional mean prove that intervening on X changes Y?
2. If the same held-out set selects twenty successive checkpoints, is the selected checkpoint's score a fresh final test?
3. How should a new-environment claim change the split?

<details><summary>Answers</summary>

1. No. Identification needs a causal model and assumptions beyond conditional prediction; see Week 26.
2. No. The set participated in selection. Freeze the selection procedure and evaluate it on a fresh final set, or explicitly assess that whole procedure.
3. Hold out environments according to the target deployment question, and state the shift assumptions. A row split within known environments answers a different question.

</details>
