# Glossary — STUDY-RL

Symbols and terms used across the curriculum. Weekly derivations should still
define their local notation. For a connected example, see [STUDY_GUIDE.md](STUDY_GUIDE.md).

## Reward indexing: translate before calculating

The repository uses two common conventions. In early notes, $r_{t+1}$ is
received after action $a_t$; some later notes and arrays call that same reward
$r_t$. This table uses the **after-action** convention. If a local formula uses
$r_t$, shift the reward index throughout, without shifting states or actions.

| Transition | After-action notation | Array notation |
|---|---|---|
| $s_t \xrightarrow{a_t} s_{t+1}$ | reward $r_{t+1}$ | `rewards[t]` |
| Return from $s_t$ | $r_{t+1}+\gamma r_{t+2}+\cdots$ | `rewards[t] + gamma * rewards[t+1] + ...` |

An actual terminal state has zero future return; the reward for entering it is
already counted. A data-collection truncation can still require a bootstrap
from the last real observation. These differ from a finite-horizon task ending
at its specified horizon; see the [time-limit explanation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## Set / function notation

| Symbol | Meaning |
|---|---|
| $\mathcal{S}$ | State space |
| $\mathcal{A}$ | Action space |
| $\mathcal{R}$ | Reward set |
| $\pi(a \mid s)$ | Policy: probability mass for discrete actions, or density for continuous actions, conditional on state $s$ |
| $\pi^\star$ | Optimal policy |
| $\gamma \in [0,1)$ | Discount factor |
| $T$ or $H$ | Horizon |
| $\mathbb{P}, \mathbb{E}$ | Probability, expectation |
| $\Delta(\mathcal{X})$ | Simplex over set $\mathcal{X}$ |
| $\mathrm{KL}(p \,\Vert\, q)$ | Kullback–Leibler divergence |
| $\mathrm{TV}$ | Total-variation distance |

## Returns, values, advantages

| Symbol | Definition |
|---|---|
| $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k+1}$ | Sample return from time $t$; stop the sum at termination for an episode |
| $V^\pi(s) = \mathbb{E}^\pi[G_t \mid s_t=s]$ | State value |
| $Q^\pi(s,a) = \mathbb{E}^\pi[G_t \mid s_t=s, a_t=a]$ | Action value |
| $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ | Advantage |
| $\delta_t = r_{t+1} + \gamma\hat V(s_{t+1}) - \hat V(s_t)$ | Sample TD error using a value estimate; mask the bootstrap on true termination |
| $\hat{A}^\mathrm{GAE}_t = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}$ | GAE-$\lambda$ advantage |

## Operators

| Symbol | Definition |
|---|---|
| $T_\pi V(s) = \mathbb{E}_{a\sim\pi, s'\sim P}[r + \gamma V(s')]$ | Bellman expectation operator |
| $T_\star V(s) = \max_a \mathbb{E}_{s'}[r + \gamma V(s')]$ | Bellman optimality operator |

Both are $\gamma$-contractions in $\ell_\infty$ on bounded value functions for
a discounted MDP with $0\le\gamma<1$ and bounded rewards. This does not make an
arbitrary neural-network training update a contraction. Undiscounted episodic
tasks need their own assumptions and analysis.

## Distinctions that prevent common mistakes

| Terms | Distinction |
|---|---|
| Reward / return / value | One feedback number / a trajectory's accumulated rewards / expected return under a specified policy |
| State / observation | A Markov state suffices to predict the next outcome given the action; a sensor observation need not |
| Prediction / control | Evaluate a fixed policy / improve or optimise a policy |
| Behaviour policy $b$ / target policy $\pi$ | Generates the data / is evaluated or improved |
| On-policy / off-policy | Uses data from the relevant target policy / uses data from a different policy, with algorithm-specific corrections or assumptions |
| Online / offline | Can gather more environment data / learns from a fixed dataset; off-policy learning can still be online |
| Bootstrapping | Uses a learned estimate inside a learning target; unrelated to resampling a dataset for a statistical bootstrap |
| Bias / variance | Systematic estimation error in expectation / variability across samples; a single run does not reveal either by itself |
| Advantage / TD error | Expected relative action value / a sampled residual that may estimate advantage under suitable assumptions |
| Exact / estimated KL | A distribution-level nonnegative expectation / a finite-sample estimator, which can be negative for some estimators |
| Clipped surrogate / hard constraint | Alters an optimisation objective / explicitly restricts feasible solutions; PPO clipping does not enforce a hard KL limit |
| Expected cost / per-step safety | Averages cost over trajectories / restricts individual actions or outcomes under stated assumptions |

Symbols are reused between papers. $\alpha$ may be a learning rate, SAC
temperature, or PER exponent; $\lambda$ may be GAE's trace parameter or a cost
multiplier; $\rho$ may be a policy ratio or a state distribution. Always attach
the meaning, range, and conditioning to the local derivation.

## Algorithms (one-line)

| Acronym | Family | What it is |
|---|---|---|
| DP | foundations | Dynamic programming (value/policy iteration) |
| MC | foundations | Monte-Carlo (sample-average) prediction/control |
| TD | foundations | Temporal-difference learning |
| SARSA | foundations | On-policy TD control |
| Q-learning | foundations | Off-policy TD control |
| Expected SARSA | foundations | TD control with expected next-step action |
| DQN | deep value | Q-learning with NN + replay + target net |
| Double DQN | deep value | DQN with decoupled selection/eval to fight max-bias |
| Dueling DQN | deep value | Split value + advantage streams |
| PER | deep value | Prioritised experience replay |
| C51, QR-DQN, IQN | deep value | Distributional RL variants |
| Rainbow | deep value | DDQN + Dueling + PER + Noisy + C51 + n-step |
| BBF | deep value | Modern high-replay-ratio DQN with periodic resets |
| REINFORCE | policy gradient | Vanilla policy gradient (Williams 1992) |
| A2C / A3C | policy gradient | Synchronous / async advantage actor-critic |
| GAE | policy gradient | Generalised advantage estimator |
| TRPO | policy gradient | Trust-region PG (KL-constrained) |
| PPO | policy gradient | Clipped surrogate PG (Schulman 2017) |
| DDPG | policy gradient | Deterministic policy gradient (continuous) |
| TD3 | policy gradient | DDPG + twin Q + delayed updates + target smoothing |
| SAC | policy gradient | Max-entropy stochastic actor-critic |
| CrossQ | policy gradient | SAC simplification using BatchRenorm (2024) |
| LinUCB / LinTS | bandits | Linear contextual bandits with UCB / Thompson |
| IPS / SNIPS / DR | OPE | Inverse / self-normalised / doubly-robust estimators |
| CQL | offline RL | Conservative Q-learning (Kumar 2020) |
| IQL | offline RL | Implicit Q-learning with expectile regression (Kostrikov 2021) |
| AWR / AWAC | offline RL | Advantage-weighted regression / actor-critic |
| TD3+BC | offline RL | TD3 with behaviour-cloning regularisation |
| DT | offline RL | Decision Transformer (sequence model) |
| Dreamer V3 | model-based | Latent RSSM + actor-critic in imagination (Hafner 2023) |
| MuZero | model-based | MCTS with learned dynamics + value/policy (Schrittwieser 2020) |
| TD-MPC2 | model-based | Trajectory optimisation on a learned model (Hansen 2024) |
| MAPPO | multi-agent | PPO with centralised critic and parameter sharing |
| MADDPG | multi-agent | Centralised-critic deterministic PG |
| QMIX | multi-agent | Monotone value decomposition |
| PSRO | multi-agent | Policy-space response oracles (population best-response) |
| AlphaZero | multi-agent | Self-play + MCTS + value/policy net |
| RND / ICM | exploration | Random-network-distillation / curiosity intrinsic rewards |
| MAML | meta-RL | Model-Agnostic Meta-Learning (Finn 2017) |
| PEARL | meta-RL | Probabilistic embeddings for actor-critic RL |
| AD / DPT | meta-RL | Algorithm Distillation / Decision-Pretrained Transformer |
| RLHF | LLM RL | Reinforcement learning from human feedback; this course studies the reward-model + PPO pipeline |
| DPO | LLM RL | Direct preference optimisation (Rafailov 2023) |
| IPO, KTO, ORPO, SimPO | LLM RL | Distinct preference/post-training objectives; not interchangeable DPO flags |
| SFT | LLM training | Supervised fine-tuning on demonstrated responses; the lab masks prompt and padding targets |
| LoRA | LLM training | Freeze a base weight and learn a low-rank correction `W = W0 + (alpha/r) BA`; [worked derivation](../llm-lab/lessons/02-real-lora-training.md) |
| QLoRA | LLM training | Train floating-point LoRA adapters through a frozen quantised base; [quantisation lesson](../llm-lab/lessons/03-quantisation-and-qlora.md) |
| NF4 | Quantisation | A nonuniform 4-bit codebook motivated by normally distributed weights; differs from uniform signed int4 |
| BF16 | Numerical precision | A 16-bit floating-point format with fewer significand bits than FP32 but the same exponent width; the lab uses it on supported CUDA hardware |
| GRPO | LLM RL | Group-relative PO (critic-free, DeepSeekMath 2024) |
| RLOO | LLM RL | REINFORCE leave-one-out (Ahmadian 2024) |
| RLVR | LLM RL | RL with verifiable rewards (R1/Kimi-K1.5 lineage) |
| PRM / ORM | LLM RL | Process / outcome reward model |
| CAI | LLM RL | Constitutional AI |
| CMDP | safety | Constrained Markov decision process |
| CPO | safety | Constrained Policy Optimization (Achiam 2017) |
| PPO-Lagrangian | safety | PPO with an adaptive cost multiplier; for $L=J_r-\lambda(J_c-d)$, maximise in policy and minimise in $\lambda\ge0$ |
| FOCOPS | safety | First-order constrained policy optimisation |
| Shielding | safety | Action filtering relative to a safety model/specification; LTL synthesis is one construction |
| CBF | safety | Control-barrier function (safe-set forward invariance) |
| W2S | safety | Weak-to-strong generalisation (OpenAI 2023) |

## Matching / ranking jargon

| Term | Meaning |
|---|---|
| Two-tower | Dual-encoder retrieval (query/doc → embedding → dot product) |
| Cross-encoder | Joint encoder over (query, doc) for fine-grained relevance |
| Reranker | Policy that re-orders the top-K retrieved candidates |
| Propensity | $\pi_b(a \mid x)$ — behaviour-policy probability of the logged action |
| CLTR | Counterfactual learning to rank (Joachims line) |
| Demographic parity | $\mathbb{P}(\hat{Y}=1 \mid A=0) = \mathbb{P}(\hat{Y}=1 \mid A=1)$ |
| Equalised odds | TPR + FPR balanced across sensitive groups |
| Exposure | Expected position-weighted attention each candidate receives |
| Calibration | $\mathbb{P}(\text{outcome} \mid \hat{p}=p) = p$ |

## Safety terms

| Term | Meaning |
|---|---|
| Reward hacking | Policy maximises proxy reward while harming the true objective |
| Specification gaming | Reward hacking when the spec was the bug |
| Goodhart | "When a measure becomes a target, it ceases to be a good measure" |
| KL anchor | KL-penalty back to a reference policy (RLHF lever vs hacking) |
| Spec gaming | Policy exploits an unintended loophole in the reward specification |
| Constitutional AI | Iterative critique + revise loop using a written constitution |
| Scalable oversight | Methods to evaluate AI outputs that exceed the evaluator's level |
| Debate, IDA | Two-player + iterated distillation for scalable oversight |

## Causal biology (week 26)

| Term | Meaning |
|---|---|
| DAG | Directed acyclic graph; an arrow represents a proposed direct causal relationship under the model's assumptions |
| SCM | Structural causal model: assignments of each variable from its parents and external noise; interventions replace an assignment |
| $P(Y\mid A=a)$ / $P(Y\mid\mathrm{do}(A=a))$ | Outcomes among those observed with $A=a$ / outcomes under an intervention that sets $A=a$; confounding can make these differ |
| ATE | Average treatment effect, $\mathbb{E}[Y(1)-Y(0)]$, for a specified population and intervention |
| Confounder / mediator / collider | A common cause of treatment and outcome / a variable on a causal path / a common effect; their roles determine whether adjustment helps or biases an estimate |
| Identification / estimation / discovery | Whether assumptions and available distributions determine a causal quantity / estimating that quantity from finite data / learning graph constraints from data and assumptions |
| Consistency, exchangeability, positivity | Observed outcome matches the specified received intervention; no remaining confounding conditional on adjustment variables; both treatments occur in the target covariate strata |
| Markov equivalence | Different causal graphs can imply the same conditional independences; observational evidence may identify an equivalence class rather than one graph |
| Environment | A patient group, tissue, laboratory, or other domain with potentially different distributions or mechanisms; its label alone does not establish an intervention |
| Invariance | Stability of a specified conditional mechanism across environments; observed stability needs assumptions to support a causal interpretation |
| PC / FCI / S-FCI | Constraint-based graph-discovery methods; FCI allows latent confounding, and S-FCI uses constraints across environments |
| ICP | Invariant causal prediction; its causal conclusions require assumptions about which mechanisms remain stable and statistical tests, beyond a residual diagnostic |
| MCAR / MAR / MNAR | Missing completely at random / missingness explained by observed variables / missingness still depends on unobserved values |
| Task-aware enhancement | Preprocessing or generation judged by a specified downstream task; better reconstruction or prediction alone does not establish preservation of intervention effects |
| Patient-level split | Keep every record from a patient in the same partition; fit preprocessing on training patients and select settings on validation patients |

For identification assumptions, see [Hernán and Robins, *Causal Inference:
What If*](https://miguelhernan.org/whatifbook). For the multi-environment
discovery method, see [Li, Jaber, and Bareinboim (NeurIPS
2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/368cba57d00902c752eaa9e4770bbbbe-Abstract-Conference.html).
The [Week 26 guide](../week-26-causal-discovery-biological-data/teaching_guide.md)
explains which assumptions and methods its toy project actually implements.

## Acronyms

CMDP, CPO, FOCOPS, P3O, OPE, IPS, SNIPS, DR, GAE, TRPO, PPO, SAC, TD3, DDPG, DQN, PER,
C51, QR-DQN, IQN, MAML, PEARL, RND, ICM, NGU, RLHF, DPO, IPO, KTO, ORPO, SimPO, CPO,
GRPO, RLOO, RLVR, PRM, ORM, CAI, W2S, MCTS, PUCT, MADDPG, MAPPO, QMIX, PSRO, AlphaZero,
LinUCB, LinTS, MOSS, SlateQ, IQL, CQL, AWR, AWAC, BCQ, BEAR, DT, IDQL, Diffuser, MBPO,
PETS, Dreamer, MuZero, TD-MPC, DIAMOND, RSSM, ELBO, KL, TV.
