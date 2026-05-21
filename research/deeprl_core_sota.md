# Deep RL Core — SOTA Dossier

*Compiled: May 2026. Target reader: a curriculum learner with solid ML/optimization background, building up to research-level Deep RL across a 24-week self-study programme.*

This dossier is **opinionated**: it filters the canon of Deep RL down to the parts that still matter in 2025–2026, gives a working derivation for each idea, lists the hyper-parameters that actually move the needle, and flags the failure modes that cost real wall-clock time. Each section ends with a `Curriculum slot:` directive mapping the topic onto the 24-week plan (W1–W24).

---

## 0. Orientation — what is still SOTA in 2026?

The field has bifurcated.

1. **Sample-efficient model-free** (Atari-100k, DMC-100k, MuJoCo-50k). The reigning champions are **BBF** (Schwarzer et al., ICML 2023) for discrete control, and **BRO** (Nauman et al., NeurIPS 2024 spotlight) plus **SimBa / SimBa-V2** (Lee et al., ICLR 2025) for continuous control. The common recipe: *scale critic width, layer-norm everywhere, residual blocks, high replay ratio, periodic resets, distributional value heads*. **CrossQ** (Bhatt et al., ICLR 2024) showed that with careful BatchNorm placement you can drop target networks entirely and beat SAC at UTD=1.
2. **Model-based / world models**. **DreamerV3** (Hafner et al., 2023, published Nature 2025) and **TD-MPC2** (Hansen et al., ICLR 2024) are the workhorses. DreamerV3 holds the only published *one-set-of-hyperparameters* result across 150+ tasks including Minecraft diamond from scratch.
3. **Massively parallel on-policy**. With GPU sims (Isaac Gym, MJX, Brax) PPO is still the default; **FastTD3** (2025) and **REPPO** (2026) port SAC-style pathwise gradients to massive parallelism.
4. **Multi-agent**. **MAPPO** with its 5-trick recipe is still the strongest cooperative baseline; **PSRO / Fusion-PSRO** (2024) dominate zero-sum/competitive settings.
5. **The plasticity story**. Post-2023 we now treat *plasticity loss / dormant neurons / capacity collapse* as first-class problems — handled via resets (BBF), LayerNorm + residuals (SimBa), regularization (BRO), and weight normalization.

`Curriculum slot:` W1 — read this section as orientation; revisit at W24 retrospective.

---

## 1. The DQN family

### 1.1 Vanilla DQN (Mnih 2013/15)

The Bellman optimality operator $\mathcal{T}^* Q(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q(s',a')]$ is a $\gamma$-contraction in $\ell_\infty$. DQN approximates the fixed point with a network $Q_\theta$ and a *target network* $Q_{\bar\theta}$ updated every $C$ steps, minimising
$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\big[(r + \gamma \max_{a'} Q_{\bar\theta}(s',a') - Q_\theta(s,a))^2\big].$$
Three engineering tricks make it work: **experience replay** (decorrelates samples, restores i.i.d.-ish gradients), **target network** (Polyak τ or hard-copy every C, breaks the moving-target instability), **reward clipping** $r \in [-1,1]$.

**Pitfalls.** (i) Q-value drift on sparse-reward / long-horizon tasks; (ii) over-estimation from $\max$; (iii) replay buffer staleness if buffer too large vs. policy change; (iv) gradient clipping (1.0–10.0) is *non-optional*.

### 1.2 Double DQN (van Hasselt 2016)

Decouple selection and evaluation:
$$y = r + \gamma\, Q_{\bar\theta}\!\big(s', \arg\max_{a'} Q_\theta(s',a')\big).$$
Reduces optimistic bias from Jensen + $\max$. Cheap; always on in modern code.

### 1.3 Dueling (Wang 2016)

Factor $Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')$. The mean-subtraction (not max-subtraction) is what stabilises training. Helps in states where action choice barely matters.

### 1.4 Prioritized Experience Replay (Schaul 2016)

Sample with probability $p_i \propto |\delta_i|^\alpha$, importance-sample-correct with $w_i = (N p_i)^{-\beta}$, anneal $\beta:0.4 \to 1.0$. Sumtree implementation is the standard. **Pitfall**: PER + n-step + Double can over-amplify noisy targets early; clip the IS weights at 1.

### 1.5 NoisyNets (Fortunato 2018)

Replace ε-greedy with parametric noise in the last layer(s): $y = (\mu^W + \sigma^W \odot \epsilon^W) x + \mu^b + \sigma^b \odot \epsilon^b$, factorised Gaussian noise. Exploration becomes state-dependent and learned. Removes the ε schedule (one fewer hyper-param).

### 1.6 C51 — Categorical DQN (Bellemare 2017)

Model the **return distribution** $Z(s,a)$ as a discrete on a fixed support $\{z_i\}_{i=0}^{N-1}$ with $N=51$, $V_{\min}=-10, V_{\max}=10$. The distributional Bellman operator
$$\mathcal{T}^\pi Z(s,a) \stackrel{D}{=} R(s,a) + \gamma Z(S', A')$$
is then projected back to the support via **categorical projection** $\Phi$:
$$(\Phi \mathcal{T} Z)_i = \sum_j \big[1 - |[\mathcal{T} z_j]_{[V_{\min},V_{\max}]} - z_i|/\Delta z\big]_+ p_j(s',\pi(s')).$$
Train with KL between projected target and prediction. **Pitfall**: support must bracket true returns, otherwise mass accumulates at boundaries — failing silently.

### 1.7 QR-DQN (Dabney 2018)

Drop the fixed support: learn $N$ **quantile locations** $\theta_i$ at fractions $\tau_i = (2i-1)/(2N)$. Loss is asymmetric Huber quantile regression:
$$\mathcal{L} = \sum_i \mathbb{E}_{j}[\rho^\kappa_{\tau_i}(y_j - \theta_i(s,a))],\qquad \rho^\kappa_\tau(u) = |\tau - \mathbb{1}(u<0)| \cdot \mathcal{L}_\kappa(u)/\kappa.$$
Wasserstein-friendly, no projection. The standard $N=200$, $\kappa=1$.

### 1.8 IQN (Dabney 2018b)

Make the quantile network *implicit*: sample $\tau \sim U[0,1]$, condition the network on $\phi(\tau) = \cos(i\pi\tau)$, output $Z_\tau(s,a)$. Variable-sample efficiency, distortion-risk policies (CVaR etc.) come for free. Big jump over QR-DQN on Atari, half the gap to Rainbow.

### 1.9 Rainbow (Hessel 2018)

Combine **Double + Dueling + PER + multi-step (n=3) + NoisyNet + C51**. The ablation is the actual lesson: multi-step and distributional are the two highest-impact components; PER second; NoisyNet smallest but free.

### 1.10 Munchausen DQN (Vieillard 2020)

Add a **scaled log-policy** to the reward:
$$r_t \leftarrow r_t + \alpha \tau \log \pi(a_t | s_t),\quad \pi \propto \exp(Q/\tau).$$
Implicitly performs KL regularisation between successive policies (an entropy-regularised soft-Q iteration). Clip $\log\pi$ at $-1$ for stability. With IQN backbone, **M-IQN** matched/beat Rainbow without PER or n-step — single most "ROI-per-line-of-code" trick in DQN-land.

### 1.11 BBF — Bigger, Better, Faster (Schwarzer 2023)

The current Atari-100k SOTA. Recipe:
* **15-layer Impala-style ResNet** with width multiplier, ~5× SR-SPR capacity.
* **Replay ratio 8** with periodic **shrink-and-perturb resets** every 40k gradient steps (interpolate weights with re-init: $\theta \leftarrow \alpha \theta + (1-\alpha)\theta_\text{init}$).
* **Discount-factor annealing** $\gamma: 0.97 \to 0.997$.
* **n-step annealing** $n: 10 \to 3$.
* **EMA target update** $\tau=0.005$.
* Categorical distributional head + Double + Dueling + NoisyNet + self-prediction SPR auxiliary loss.

Reaches super-human Atari-100k median in ~6 GPU-hours.

**Pitfall**: without the resets, high replay ratio collapses to plasticity loss — *dormant neuron fraction* shoots past 50%.

### 1.12 Replay-ratio scaling and BBF descendants

Modern rule of thumb: **resets enable RR up to 8–16**, beyond that returns diminish unless you also widen the net (BRO finding). DroQ-style **dropout + LayerNorm** is the cheaper alternative to BBF resets for off-policy continuous control.

`Curriculum slot:` W2–W4. W2 = vanilla / Double / Dueling / PER. W3 = NoisyNet + C51 + QR-DQN + IQN. W4 = Rainbow, M-DQN, SPR, BBF — implement BBF-lite on MinAtar.

---

## 2. Policy-gradient family

### 2.1 REINFORCE and baselines

For trajectory $\tau$ with return $G(\tau)$:
$$\nabla_\theta J(\theta) = \mathbb{E}_\tau\big[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, (G_t - b(s_t))\big].$$
The baseline $b(s)$ does **not** bias the gradient (because $\mathbb{E}[\nabla \log \pi] = 0$) and minimises variance when $b^*(s) = V^\pi(s)$. Learn $b$ by MC regression on $G_t$.

### 2.2 A2C / A3C (Mnih 2016)

Replace return with $n$-step bootstrapped advantage:
$$A_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t).$$
A3C ran 16 async CPU actors; A2C is the synchronous, GPU-friendly version. Modernly: A2C is mostly a stepping stone, replaced by PPO.

### 2.3 GAE (Schulman 2015)

Bias-variance dial. With TD residual $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$:
$$\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}.$$
**Derivation**: $\hat A^{(k)}_t = -V(s_t) + \sum_{l=0}^{k-1}\gamma^l r_{t+l} + \gamma^k V(s_{t+k})$ is the $k$-step advantage. GAE is the geometric mixture $(1-\lambda)\sum_{k\ge 1} \lambda^{k-1} \hat A^{(k)}$; algebraic collapse using $\sum_k \lambda^{k-1}(\delta_t + \gamma\delta_{t+1} + \ldots)$ yields the closed form. $\lambda=1$ is MC (high var), $\lambda=0$ is one-step TD (high bias). **Standard**: $\lambda=0.95, \gamma=0.99$.

### 2.4 Natural policy gradient

The vanilla PG gradient is in Euclidean param space, but the meaningful geometry is the policy manifold. The **Fisher information matrix** $F = \mathbb{E}[\nabla\log\pi \,\nabla\log\pi^\top]$ defines the natural gradient $\tilde g = F^{-1} g$. Update $\theta \leftarrow \theta + \alpha F^{-1}g$ is invariant to policy reparameterisation.

### 2.5 TRPO (Schulman 2015) — full KL derivation

Maximise the surrogate $L_{\theta_\text{old}}(\theta) = \mathbb{E}_{s,a \sim \pi_{\theta_\text{old}}}[\frac{\pi_\theta}{\pi_{\theta_\text{old}}} A^{\pi_{\theta_\text{old}}}]$ subject to mean KL constraint $\bar D_\text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta) \le \delta$.

Second-order Taylor: $\bar D_\text{KL} \approx \tfrac12 (\theta-\theta_\text{old})^\top F (\theta-\theta_\text{old})$. Lagrangian gives the natural-gradient step
$$\theta \leftarrow \theta_\text{old} + \sqrt{\tfrac{2\delta}{g^\top F^{-1} g}} F^{-1} g,$$
solved by **conjugate gradient** on $Fx = g$ (10 CG iterations typical) plus a **backtracking line search** that re-checks the KL constraint and surrogate improvement. Theorem (monotonic improvement): $J(\pi') \ge L_{\pi}(\pi') - C \max_s D_\text{KL}$, $C = 4\epsilon\gamma/(1-\gamma)^2$.

**Pitfalls**: CG with bad preconditioning blows up; Fisher-vector products require sample efficiency; large batch (≥4096) is essential.

### 2.6 PPO (Schulman 2017) — clip and KL variants

PPO is the first-order, batched approximation to TRPO. **Clip** variant:
$$L^{\text{CLIP}}(\theta) = \mathbb{E}\big[\min(r_t(\theta) \hat A_t,\, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t)\big],$$
$r_t = \pi_\theta/\pi_{\theta_\text{old}}$, $\epsilon=0.2$. **KL-penalty** variant adds $-\beta\,\bar D_\text{KL}$ with adaptive $\beta$ targeting $D_\text{KL}^\star=0.01$ (double $\beta$ if exceeded by 50%, halve if below). Empirically the **clip variant wins** in all published comparisons; KL-penalty is back in vogue for RLHF (KL to a reference policy is part of the objective there).

### 2.7 The "37 Implementation Details" (Engstrom 2020 / Huang et al. ICLR 2022)

The gap between paper PPO and working PPO is *enormous*. The Huang blog is canon — every entry costs you days if missed:
1. Vectorised envs (typically 8–64 parallel).
2. Orthogonal init, gain $\sqrt2$ on hidden, $0.01$ on policy head, $1$ on value head.
3. **Adam $\epsilon=1\text{e}{-5}$** (not default $1\text{e}{-8}$).
4. Linear LR annealing.
5. **GAE with $\lambda=0.95$**.
6. **Minibatch advantage normalisation** (after computing GAE on whole rollout).
7. Clipped value loss: $\max((V-\hat R)^2, (V_\text{clip}-\hat R)^2)$.
8. Entropy bonus $\beta\approx 0.0$ for MuJoCo, $0.01$ for Atari.
9. Global grad-norm clip $0.5$.
10. **Approx KL early-stop**: break the SGD epochs if $\widehat{D}_\text{KL} > 1.5 D_\text{KL}^\star$.
11. Reward scaling/clipping; observation normalisation with running statistics.
12. Continuous-action: state-independent log-std, tanh **before** the Gaussian (squashing trick à la SAC) or *no* tanh; do not mix.

Engstrom 2020 ("Implementation Matters") showed that ~80% of PPO-vs-TRPO performance delta is implementation tricks, not the algorithm.

`Curriculum slot:` W5–W7. W5 = REINFORCE/A2C/GAE. W6 = TRPO derivation + CG/FIM. W7 = PPO from scratch with all 37 details, reproduce CleanRL on Atari + MuJoCo.

---

## 3. Continuous control — off-policy actor-critic

### 3.1 DDPG (Lillicrap 2016)

Deterministic policy $\mu_\theta(s)$ with deterministic-PG theorem (Silver 2014): $\nabla J = \mathbb{E}[\nabla_a Q(s,a)|_{a=\mu(s)}\,\nabla_\theta \mu(s)]$. Critic = DQN-style; explore with OU/Gaussian action noise. **Pitfalls**: brittle, wildly hyper-param sensitive, severe Q overestimation; today only used as a teaching baseline.

### 3.2 TD3 (Fujimoto 2018)

Three fixes that should be in *every* deterministic actor-critic since:
1. **Twin critics**, target uses $\min(Q_1, Q_2)$ — clipped-double-Q.
2. **Delayed policy update**: critic 2× per actor update.
3. **Target-policy smoothing**: $a' = \mu_{\bar\theta}(s') + \text{clip}(\epsilon, -c, c)$, $\epsilon \sim \mathcal{N}(0, \sigma)$.

Solid baseline; matches early SAC on MuJoCo.

### 3.3 SAC (Haarnoja 2018a/b) — automatic temperature

Maximum-entropy RL: $J = \mathbb{E}[\sum r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))]$. Soft Bellman:
$$Q^\pi(s,a) = r + \gamma \mathbb{E}_{s',a'\sim\pi}[Q^\pi(s',a') - \alpha \log\pi(a'|s')].$$
Policy update by reparameterised gradient through tanh-Gaussian: $a = \tanh(\mu + \sigma \odot \epsilon)$, $\epsilon \sim \mathcal{N}(0,I)$. **Automatic temperature** (Haarnoja 2018b): maintain $\alpha$ by gradient descent on
$$\mathcal{L}(\alpha) = \mathbb{E}_{a\sim\pi}[-\alpha (\log\pi(a|s) + \bar{\mathcal{H}})],$$
target entropy $\bar{\mathcal{H}} = -\dim(\mathcal{A})$.

**Pitfalls**: forget the tanh log-det Jacobian correction in $\log\pi$ and the policy will silently saturate; $\sigma$ unconstrained explodes — clamp $\log\sigma \in [-20, 2]$.

### 3.4 REDQ (Chen 2021)

Use a **large critic ensemble** (size $N=10$), random subset of $M=2$ for target min; high **UTD=20**. Roughly matches model-based sample efficiency on MuJoCo, purely model-free. Big compute cost.

### 3.5 DroQ (Hiraoka 2022)

Same idea as REDQ but **2 critics** with dropout (p=0.01) + LayerNorm; the dropout-induced uncertainty replaces the ensemble. ~5× cheaper, matches REDQ. The default "high-UTD SAC" today.

### 3.6 TQC (Kuznetsov 2020)

SAC with QR-DQN-style **distributional critic** per ensemble member (5 nets × 25 quantiles), and **truncate top $d$ quantiles** of the pooled distribution before computing the soft-Q target. Tunes the bias dial via $d$. Beats SAC by 25%+ on Humanoid.

### 3.7 CrossQ (Bhatt 2024)

**Drop the target network entirely.** BatchNorm in the critic, applied with the trick that current-state and next-state batches must be concatenated before BN so that running statistics match the policy distribution. Result: UTD=1 matches REDQ/DroQ at UTD=20. Wall-clock king for continuous control.

Recipe in a nutshell:
* BN-MLP critic, 2 hidden layers ×2048 (wide).
* BN-MLP actor, 256×256.
* No target networks.
* Concat `(s,a)` and `(s',a' ~ π)` along batch axis before each critic forward pass.

**Pitfall**: BN's running stats need careful handling under distribution shift; switch to *eval mode* at inference, never during training-target computation.

### 3.8 BRO (Nauman 2024)

Take SAC, **scale critic to ~5M params** with residual+LayerNorm blocks (the SimBa-style block), **distributional value** (HL-Gauss two-hot loss), **optimistic exploration** (multiple actor proposals, pick the one with highest UCB-style Q), **layer-norm everywhere**, periodic resets, UTD=10. First model-free agent to solve Dog and Humanoid in DMC. *BRO-Fast* (RR=2) matches TD-MPC2 sample efficiency. Probably the best off-policy continuous-control baseline as of mid-2026.

### 3.9 SimBa / SimBa-V2 (Lee 2024 / 2025)

Architectural primitive, not an algorithm:
* **Running-stat observation normalization**.
* **Residual MLP block**: `x → LN → Linear → ReLU → Linear → +x`.
* **Pre-output LayerNorm**.
* Width 512–1024, depth 2–4 blocks.

Drop into SAC/PPO/TD-MPC and you get +30–50% sample efficiency for free. **SimBa-V2** swaps to hyperspherical (RMSNorm + weight projection onto sphere) normalisation; cleaner scaling laws.

### 3.10 FastTD3 / REPPO

GPU-parallel-sim era. FastTD3 (2025): TD3 + LayerNorm + distributional critic + huge batch size (~65k transitions/step), scales to Isaac humanoid training in minutes. REPPO (2026): relative-entropy pathwise PG, SAC-like off-policy gradient but stable with massive parallelism.

`Curriculum slot:` W8–W10. W8 = DDPG + TD3 on Pendulum/Mujoco. W9 = SAC (auto-α) + REDQ/DroQ + TQC. W10 = CrossQ + BRO + SimBa block; reproduce DMC dog on a single GPU.

---

## 4. Distributional RL theory

Bellemare, Dabney, Rowland (2023, MIT Press, "Distributional Reinforcement Learning") is the canonical text. Core results:

1. **Distributional Bellman equation**: $Z^\pi(s,a) \stackrel{D}{=} R + \gamma Z^\pi(S',A')$. The operator $\mathcal{T}^\pi$ on distributions is a $\gamma$-contraction in the **Wasserstein-$p$** metric (Cramér / quantile / Wasserstein, depending on parameterisation).
2. **Approximation choice matters.** Categorical (C51) is contraction in Cramér $\bar\ell_2$ *after the projection*. Quantile (QR-DQN) is contraction in $W_1$. KL-projection is **not** a contraction — so categorical works only because we project in Cramér, not KL, sense.
3. **Categorical projection** $\Phi_\mathcal{C}$: given a target distribution with atoms $\{(\hat z_j, p_j)\}$, redistribute mass to fixed support $\{z_i\}$ by:
$$(\Phi_\mathcal{C} \mu)_i = \sum_j p_j \cdot \big[1 - |z_i - \text{clip}(\hat z_j, V_\min, V_\max)|/\Delta z\big]_+.$$
4. **Why does it help?** Three answers: (a) richer auxiliary signal regularises the representation; (b) better-calibrated targets reduce TD-target variance after $\min$ aggregation; (c) enables risk-sensitive policies (CVaR, distortion measures).
5. **Two-hot / HL-Gauss** (Farebrother 2024): map a regression target $y$ onto a soft histogram with a Gaussian kernel, train with cross-entropy. Strictly better than MSE for value-net training; DreamerV3, BRO, FastTD3 all use it.

**Pitfall**: support choice. Atari uses $[-10,10]$ with 51 atoms; for continuous control with wildly varying scales, normalize returns *before* setting support or use quantile-based / two-hot loss instead.

`Curriculum slot:` W11. Read Bellemare book Ch.1–7, implement HL-Gauss critic, plot CDFs.

---

## 5. Recurrent RL and partial observability

### 5.1 DRQN baseline

Replace the post-conv FC with an LSTM (Hausknecht 2015). Backprop-through-time of length 8–16; sample whole episodes or fixed-length sub-sequences.

### 5.2 R2D2 (Kapturowski 2019)

Distributed replay of **fixed-length 80-step sequences**, two-step **burn-in** of 40 steps (recompute h with stale weights but don't backprop) to mitigate the *recurrent state staleness* problem. **Invertible value-function rescaling** $h(x) = \text{sign}(x)(\sqrt{|x|+1}-1)+\epsilon x$ for wide reward scales. Prioritized replay using max + mean of TD errors over the sequence. The reference recurrent RL recipe.

### 5.3 Agent57 (Badia 2020)

R2D2 + **NGU intrinsic motivation** (episodic + lifelong novelty via random network distillation and embedding-based count) + **multi-arm bandit meta-controller** picking $(\beta,\gamma)$ pair across a family of policies. First agent to be super-human on all 57 Atari games. Engineering monument; today rarely re-implemented in full.

### 5.4 Recurrent SAC and the POMDP literature

Ni 2021 ("Recurrent Off-policy Baselines for Memory-based Continuous Control") showed that a well-tuned **GRU-SAC / GRU-TD3** is a strong baseline for POMDPs and beats most bespoke memory-RL algorithms. Implementation gotchas:
* Replay full episodes; sample a fixed-length window with burn-in.
* Store the hidden state with the transition; recompute on update.
* Layer-norm inside GRU/LSTM is *very* helpful for stability.
* 2024 trend: replace GRU with **Mamba / S4 / Transformer-XL with relative pos enc** for long-horizon memory; DreamerV3 uses a "Block GRU".

`Curriculum slot:` W12. Implement GRU-PPO on a delayed-cartpole POMDP, then R2D2-lite on MinAtar with sequence replay.

---

## 6. Modern actor-critic best practices (2024–2026)

A compact "if you're starting a new continuous-control codebase in 2026" checklist:

1. **Architecture**
   * SimBa residual MLP block (or BRO's wider variant). Width 512–1024.
   * LayerNorm before each Linear; or RMSNorm (SimBa-V2).
   * Spectral normalization on critic last layer if you see explosions.
2. **Output heads**
   * Two-hot / HL-Gauss for value/critic.
   * Squashed-Gaussian or normalising-flow actor.
3. **Optimizer**
   * AdamW, lr 3e-4 baseline, lr 1e-4 for very wide nets.
   * Adam $\epsilon = 1\text{e}{-5}$.
   * Linear warm-up first 5–10k steps.
4. **Stabilizers**
   * Twin critics with $\min$ aggregation (or 5-critic ensemble if you can afford it).
   * **Periodic resets** every $T$ updates (BBF). For continuous control, layer-wise re-init of last 2 layers every 250k env steps.
   * Dormant-neuron monitoring (>10% dormant ⇒ resets are needed).
5. **Plasticity hygiene**
   * LayerNorm + residuals (SimBa)
   * Weight decay (1e-4–1e-2)
   * Effective-learning-rate stabilisation (Lyle 2024)
6. **Replay ratio**: start UTD=1, ramp to 4–10 if you have resets/LN; don't exceed 10 without monitoring dormancy.
7. **Distributional value**: HL-Gauss is the default in 2026.
8. **Action-space**: tanh-squash with Jacobian; do *not* use raw Gaussian on bounded actions.
9. **Reward scale**: symlog (DreamerV3) or percentile-based return normalization is now standard.

`Curriculum slot:` W13. Re-architect your W10 SAC into a SimBa-block actor-critic; ablate LN, residuals, resets.

---

## 7. Multi-agent RL

### 7.1 MADDPG (Lowe 2017)

**Centralised training, decentralised execution (CTDE)**: each agent has its own actor $\mu_i(o_i)$; the critic $Q_i(s, a_1, \ldots, a_N)$ sees joint state and joint action. Works for mixed cooperative-competitive. Brittle, but historically important.

### 7.2 QMIX (Rashid 2018)

CTDE for cooperative value-based MARL. Each agent has $Q_i(o_i, a_i)$; the **mixing network** $f_\text{mix}$ produces $Q_\text{tot}(s, \mathbf a) = f_\text{mix}(Q_1, \ldots, Q_N; s)$ subject to $\partial Q_\text{tot}/\partial Q_i \ge 0$ (enforced via non-negative weights from a hypernetwork conditioned on $s$). The monotonicity ensures $\arg\max_{\mathbf a} Q_\text{tot} = (\arg\max_{a_1} Q_1, \ldots)$ — i.e. *decentralised execution is consistent*. **Pitfall**: monotonicity excludes non-monotonic interactions (e.g. coordination games). Variants: QPLEX, QTRAN.

### 7.3 MAPPO (Yu 2022)

Just PPO with parameter sharing across homogeneous agents and a **centralised value function** $V(s)$. With the following 5 tricks it beats QMIX on SMAC:
1. Value normalisation (running mean/std).
2. Agent-specific global state (concat agent id one-hot).
3. PopArt-style return rescaling.
4. **Death masking** in SMAC.
5. Careful hyper-param (clip 0.2, GAE 0.95, 15 PPO epochs).

2024 nuance: IPPO (independent PPO, no central critic) often matches MAPPO; the central critic helps mainly when credit assignment is non-trivial.

### 7.4 PSRO (Lanctot 2017) and Fusion-PSRO (2024)

Game-theoretic MARL for zero-sum / general-sum:
1. Maintain a population $\Pi = \{\pi_1, \ldots, \pi_k\}$.
2. Compute the **empirical game** payoff matrix $U_{ij}$.
3. Compute a meta-strategy $\sigma$ via Nash / $\alpha$-rank / uniform.
4. **Oracle**: train a best response $\pi_{k+1}$ vs. $\sum_i \sigma_i \pi_i$ using any single-agent RL.
5. Repeat.

Convergence to (approximate) Nash in 2-player zero-sum under exact best response. **Fusion-PSRO (2024)** warm-starts the oracle from a Nash-mixture-distilled policy, big efficiency win. Heavy compute; used in AlphaStar-style stacks.

### 7.5 Nash-Q

Tabular foundation (Hu & Wellman 2003): Q-update uses the Nash payoff of the stage game defined by the Q-tables. Conceptually important; in deep RL replaced by PSRO and its descendants.

`Curriculum slot:` W14–W15. W14 = QMIX + MAPPO on SMAC-Lite/MPE. W15 = PSRO on Kuhn / Leduc poker, then a small Overcooked task.

---

## 8. Sample-efficient RL benchmarks

The two reference suites, and what counts as SOTA in 2026:

* **Atari 100k** (Kaiser 2020): 100k env steps (~2 hours of game-play). Human-normalised score, IQM aggregated across 26 games (Agarwal 2021's *rliable* protocol). 2026 SOTA: **BBF** ~1.045 median, ~2.5 IQM; **EfficientZero V2** competitive on model-based side; **SR-SPR** and **SPR** are the strong prior baselines.
* **DMControl 100k / 500k** (Tassa 2018): 27-task suite, proprio + pixel. 2026 SOTA: **BRO** on proprio (~0.95 normalised), **TD-MPC2** on pixel.
* **DMC Hard subset** (Dog Run/Walk, Humanoid Run): BRO is the first model-free to crack these without curriculum.
* **MetaWorld MT50 / ML45**: TD-MPC2 dominates; SimBa-based off-policy catching up.
* **MyoSuite** (musculoskeletal): TD-MPC2 / BRO neck and neck.

**Evaluation methodology** (Agarwal "Deep RL at the Edge of the Statistical Precipice" 2021): report **IQM with 95% stratified bootstrap CI**, **performance profiles**, and **probability of improvement**. Mean and median are not enough.

`Curriculum slot:` W16. Reproduce one Atari-100k seed with BBF-lite; analyse with `rliable`.

---

## 9. Self-play and curricula

### 9.1 Self-play essentials

* **Naive self-play**: train vs. current policy; works in transitive games (Connect-Four), oscillates in non-transitive (rock-paper-scissors).
* **Fictitious self-play / fictitious play**: train vs. average policy distribution. Converges to Nash in 2-player zero-sum.
* **Population-based training (PBT) / Prioritized Fictitious Self-Play** (AlphaStar): maintain a league of frozen historical opponents; sample by win-rate; periodically copy strong policies and perturb hyper-params.
* **PSRO** (Section 7.4) is the principled generalisation.

### 9.2 Curricula

* **Automatic curriculum learning** (Portelas 2020 survey): goal generation by absolute/relative learning progress.
* **PLR / Prioritized Level Replay** (Jiang 2021): sample environment levels by TD-error / value loss.
* **ACCEL** (Parker-Holder 2022): evolve levels via mutation + PLR; SOTA on procgen-style domains.
* **Open-Endedness** (POET, Enhanced POET, Minecraft Voyager 2023): co-evolve environments and agents.

### 9.3 Skill discovery / unsupervised RL

* **DIAYN, APS, APT**: information-theoretic skills.
* **URLB** benchmark (Laskin 2021): pre-train task-agnostic, fine-tune on tasks.
* 2024 trend: language-conditioned skill discovery powered by VLMs.

`Curriculum slot:` W17. Self-play tic-tac-toe → Connect-Four → tiny poker. Then PLR on a 10-level Procgen subset.

---

## 10. Recent (2024–2026) trends and reading list

A non-exhaustive picture of where the frontier moved between the BBF/Rainbow era and today:

1. **Plasticity is first-class.** Papers: Nikishin 2022 (primacy bias), Sokar 2023 (dormant neuron), Lyle 2023/2024 (effective learning rate), Plasticine benchmark 2025.
2. **Width and depth scale RL** — but only with regularization (BRO/SimBa). Old folklore "RL nets must stay small" is dead.
3. **Two-hot / HL-Gauss** loss replaces MSE for value heads (Farebrother 2024 "Stop Regressing").
4. **World models matured.** DreamerV3 is general; TD-MPC2 is the strongest at continuous control with proprio; Genie/Genie-2 (2024) show internet-scale video → playable worlds. Diffusion world models (DIAMOND 2024) are rising.
5. **Massive parallelism rewrites algorithm choice.** With Isaac Lab + 4096 parallel envs, PPO/FastTD3-style methods minutes-train policies that took GPU-days a year before.
6. **RLHF pulled PPO back into the spotlight** — and revealed deep failures (reward hacking, KL collapse). DPO/GRPO/RLOO/ORPO/KTO are the offline / no-critic offshoots; GRPO (DeepSeek 2024) is the current reasoning-RL workhorse.
7. **MARL went game-theoretic.** PSRO derivatives + LLM-grounded agents are the active frontier.
8. **Offline RL → IQL / CQL / TD3+BC / Diffusion Policies.** Outside scope here but a future curriculum week.
9. **Generalist agents.** Gato → RT-X → OpenVLA: behaviour cloning + RL fine-tune on cross-embodiment data.

Must-read 2024–2026 papers (with the slot they belong to):
* Schwarzer 2023 *BBF* (W4)
* Nauman 2024 *BRO* (W10)
* Bhatt 2024 *CrossQ* (W10)
* Lee 2024 *SimBa*, 2025 *SimBa-V2* (W13)
* Farebrother 2024 *Stop Regressing* (W11)
* Lyle 2023/2024 *plasticity / effective LR* (W13)
* Hafner 2023 *DreamerV3* + Hansen 2024 *TD-MPC2* (model-based slot, W18 of follow-on plan)
* Yu 2022 *MAPPO* + 2024 *Fusion-PSRO* (W14–W15)
* Agarwal 2021 *Statistical Precipice* (W16)
* Huang 2022 *37 Implementation Details* (W7)

`Curriculum slot:` W18 = read 5 of the above end-to-end, reproduce one figure each.

---

## 11. 24-week curriculum mapping (TL;DR)

| Wk | Topic | Key paper(s) |
|----|-------|--------------|
| 1 | Orientation, MDPs, contraction maps | Sutton & Barto ch.3–4 |
| 2 | DQN, Double, Dueling, PER | Mnih 2015, vanH 2016, Wang 2016, Schaul 2016 |
| 3 | NoisyNet, C51, QR-DQN, IQN | Fortunato, Bellemare, Dabney |
| 4 | Rainbow, M-DQN, BBF | Hessel 2018, Vieillard 2020, Schwarzer 2023 |
| 5 | REINFORCE, A2C, baselines, GAE | Williams 1992, Mnih 2016, Schulman 2015 |
| 6 | Natural PG, Fisher, TRPO | Kakade 2001, Schulman 2015 |
| 7 | PPO + 37 details | Schulman 2017, Huang 2022 |
| 8 | DDPG, TD3 | Lillicrap, Fujimoto |
| 9 | SAC, REDQ, DroQ, TQC | Haarnoja, Chen, Hiraoka, Kuznetsov |
| 10 | CrossQ, BRO, SimBa | Bhatt, Nauman, Lee |
| 11 | Distributional RL theory + HL-Gauss | Bellemare book, Farebrother 2024 |
| 12 | Recurrent: DRQN, R2D2, Agent57, GRU-SAC | Kapturowski, Badia, Ni |
| 13 | Plasticity & modern arch best practice | Nikishin, Sokar, Lyle |
| 14 | MARL value-based: VDN, QMIX, MAPPO | Sunehag, Rashid, Yu |
| 15 | Game-theoretic MARL: PSRO, Nash-Q, Fusion-PSRO | Lanctot, Hu&Wellman, Yang 2024 |
| 16 | Benchmarks: rliable, Atari-100k, DMC | Agarwal 2021 |
| 17 | Self-play, league play, curricula, PLR/ACCEL | Vinyals, Jiang, Parker-Holder |
| 18 | Recent SOTA paper-reading deep-dive | (see list) |
| 19 | Model-based: World Models, MuZero, DreamerV3, TD-MPC2 | Ha, Schrittwieser, Hafner, Hansen |
| 20 | Offline RL: BCQ, CQL, IQL, TD3+BC, Diffusion Policy | Fujimoto, Kumar, Kostrikov, Chi |
| 21 | RLHF / DPO / GRPO + reasoning | Ouyang, Rafailov, Shao 2024 |
| 22 | Exploration: RND, NGU, ICM, BYOL-Explore | Burda, Badia, Pathak |
| 23 | Generalisation: domain randomisation, PCG, ProcGen, XLand | OpenAI, Cobbe, Team 2023 |
| 24 | Capstone project + retrospective | — |

Topics 19–24 are pointers for a follow-on dossier; this one is bounded to the **core** Deep RL stack (DQN family, PG, off-policy AC, distributional, recurrent, modern arch, MARL, benchmarks, self-play).

---

## Sources

- [BBF — Bigger, Better, Faster (Schwarzer 2023)](https://arxiv.org/abs/2305.19452)
- [BRO — Bigger, Regularized, Optimistic (Nauman 2024)](https://arxiv.org/abs/2405.16158)
- [SimBa — Simplicity Bias (Lee 2024)](https://arxiv.org/abs/2410.09754)
- [CrossQ (Bhatt 2024 ICLR)](https://arxiv.org/abs/1902.05605)
- [TD-MPC2 (Hansen 2024)](https://arxiv.org/abs/2310.16828)
- [DreamerV3 (Hafner 2023)](https://arxiv.org/abs/2301.04104)
- [Munchausen RL (Vieillard 2020)](https://arxiv.org/abs/2007.14430)
- [Implicit Quantile Networks (Dabney 2018)](https://arxiv.org/abs/1806.06923)
- [TQC (Kuznetsov 2020)](https://arxiv.org/abs/2005.04269)
- [DroQ (Hiraoka 2022)](https://arxiv.org/abs/2110.02034)
- [Agent57 (Badia 2020)](http://proceedings.mlr.press/v119/badia20a/badia20a.pdf)
- [R2D2 (Kapturowski 2019)](https://openreview.net/pdf?id=r1lyTjAqYX)
- [QMIX (Rashid 2018)](https://arxiv.org/abs/1803.11485)
- [MAPPO (Yu 2022)](https://sites.google.com/view/mappo)
- [Fusion-PSRO (2024)](https://arxiv.org/abs/2405.21027)
- [PSRO survey (2024)](https://arxiv.org/html/2403.02227v2)
- [37 Implementation Details of PPO (Huang 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Recurrent Off-policy Baselines for POMDPs (Ni 2021)](https://arxiv.org/abs/2110.12628)
- [Plasticine: Plasticity in Deep RL (2025)](https://arxiv.org/abs/2504.17490)
- [Normalization and Effective LR in RL (Lyle 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/c04d37be05ba74419d2d5705972a9d64-Paper-Conference.pdf)
- [Recent Advances in RL for Continuous Control SOTA 2026 (Raffin)](https://araffin.github.io/slides/advances-rl-sota-2026/)
