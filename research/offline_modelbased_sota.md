# Offline + Model-Based RL — SOTA Dossier

A two-track research dossier feeding a 24-week curriculum. Track A is **Offline RL** (learning from a fixed log, no environment interaction). Track B is **Model-Based RL** (learn a world model, plan/imagine in it). The two tracks interleave: week-by-week, the student alternates between an offline-RL method and a model-based method, with shared infrastructure (replay handling, OPE, planning) introduced just-in-time. Each entry gives derivation/intuition, when to use, hyperparameters, code references, and curriculum slot.

---

## Track A — Offline RL

The unifying problem: given a dataset `D = {(s, a, r, s')}` collected by an unknown behavior policy `μ`, learn a policy `π` whose value `J(π)` is high under the true MDP, **without** new rollouts. The central pathology is **extrapolation error / OOD-action bootstrapping**: a learned `Q(s, a)` queried at actions never seen in `D` is unconstrained and tends to be over-optimistic; bootstrapping then amplifies the error. Every method below is a different answer to *"how do we keep π close to the data distribution, or how do we make Q honest about unseen actions?"*.

### A.1 BCQ — Batch-Constrained Q-Learning (Fujimoto 2019)

**Intuition.** First clean formulation of the OOD problem. Train a CVAE behavior model `μ̂(a|s)`; at the Bellman target only consider actions sampled from `μ̂` (perturbed by a small δ-network). Effectively a *support-constrained* TD3.

**When.** Historical baseline; useful when the behavior policy is highly multi-modal and you need an explicit generative behavior model.

**Hyperparameters.** δ-net perturbation `Φ ∈ [0.05, 0.5]`, 10 action samples at target. Two Q-networks, twin clipped target (TD3 style). LR 1e-3.

**Code.** `sfujim/BCQ` (PyTorch).

**Curriculum slot.** **Week 3**, after a TD3/SAC online refresher (weeks 1–2).

### A.2 BEAR — Bootstrapping Error Accumulation Reduction (Kumar 2019)

**Intuition.** Constrain π to share *support* (not distribution) with μ via MMD between samples. Less conservative than KL-based methods.

**When.** Datasets where behavior policy is wide; mostly superseded by CQL/IQL but conceptually important.

**Hyperparameters.** MMD kernel σ=20, MMD threshold ε≈0.05, n=5 action samples. Lagrangian dual gradient on the support constraint.

**Code.** `aviralkumar2907/BEAR`.

**Curriculum slot.** **Week 3** (paired reading with BCQ).

### A.3 CQL — Conservative Q-Learning (Kumar 2020)

**Intuition.** Push down Q-values on OOD actions, push up on dataset actions. Adds a regularizer `α · (E_s log Σ_a exp Q(s,a) − E_{s,a~D} Q(s,a))` to standard Bellman. This yields a *lower bound* on the true Q-function. Single most-cited offline-RL method.

**When.** Default baseline for any new offline benchmark. Robust, simple, no behavior model needed.

**Hyperparameters.** α ∈ {1, 5, 10} (auto-tuned via Lagrangian targeting a gap ≈ 5–10), 10 importance samples for the log-sum-exp, SAC backbone, LR 3e-4, batch 256.

**Code.** `aviralkumar2907/CQL`, `tinkoff-ai/CORL` reproduction.

**Curriculum slot.** **Week 4** — the workhorse implementation week.

### A.4 IQL — Implicit Q-Learning (Kostrikov 2021)

**Intuition.** Never query Q on OOD actions at all. Use **expectile regression** of V(s) toward the upper tail of `Q(s, a)` over dataset actions: `V = argmin E[L_τ(Q − V)]` with τ ≈ 0.7–0.9. Target uses `r + γ V(s')` (no `max` over actions). Policy extracted by **advantage-weighted regression**.

**When.** Strong default. Trains fast, no Lagrangian, two-stage (V, then π) is stable. Best CQL-alternative.

**Hyperparameters.** expectile τ = 0.7 (locomotion) / 0.9 (AntMaze); AWR temperature β ∈ {3, 10}; clip AWR weights at 100; LR 3e-4 (Q,V), 3e-4 (π) with cosine decay.

**Code.** `ikostrikov/implicit_q_learning` (JAX); CORL PyTorch port.

**Curriculum slot.** **Week 5**.

### A.5 AWR / AWAC

**Intuition.** Policy improvement = weighted regression with weights `exp((Q − V)/β)`. AWR (Peng 2019) uses Monte-Carlo returns; AWAC (Nair 2020) uses a learned Q for offline-to-online fine-tuning.

**When.** AWAC is the canonical *offline-pretrain → online-finetune* method.

**Hyperparameters.** β = 0.3 (AWR) / 2.0 (AWAC), weight clip 20.

**Code.** `hari-sikchi/AWR`, `vitchyr/rlkit` for AWAC.

**Curriculum slot.** **Week 5** (taught alongside IQL — same AWR extractor).

### A.6 TD3+BC (Fujimoto 2021)

**Intuition.** "Minimalist offline RL." Take TD3, add a BC term to the actor loss: `−λ Q(s, π(s)) + (π(s) − a)²`, with `λ = α / mean|Q|` for scale invariance. One line of code; competitive with CQL/IQL on D4RL locomotion.

**When.** Always run as a sanity baseline; if it wins, complex methods aren't justified.

**Hyperparameters.** α = 2.5, normalize states, twin Q, target smoothing 0.005.

**Code.** `sfujim/TD3_BC`.

**Curriculum slot.** **Week 4** (paired with CQL — the regularizer-vs-constraint dichotomy).

### A.7 ReBRAC — Revisited Behavior-Regularized Actor-Critic (Tarasov 2023)

**Intuition.** BRAC done right: deeper nets (3-layer 256), layer-norm in critic, separate actor/critic BC penalties (`β_actor`, `β_critic`), large discount, large batch. Often matches or beats diffusion methods on D4RL — the "boring baseline that won."

**When.** Default modern minimalist baseline circa 2024–2026.

**Hyperparameters.** `β_actor ∈ [0.001, 1.0]` per task, `β_critic ∈ [0.0, 0.1]`, LayerNorm in Q, γ=0.99, batch 1024.

**Code.** `tinkoff-ai/ReBRAC` (CORL).

**Curriculum slot.** **Week 6** — "tuning matters" week; reproduce ReBRAC on D4RL halfcheetah-medium.

### A.8 Decision Transformer (Chen 2021)

**Intuition.** Reframe offline RL as **return-conditioned sequence modelling**. Tokens are `(R̂_t, s_t, a_t)`; train GPT-style on next-action prediction; at inference, condition on a target return-to-go. No Bellman, no Q-function.

**When.** Long-horizon, sparse-reward, or multi-task tabular/visual datasets. Scales gracefully with parameters.

**Hyperparameters.** context K = 20–50, 6 layers / 8 heads / dim 128, dropout 0.1, AdamW LR 1e-4 warmup, target-return = best in dataset × 1.1.

**Code.** `kzl/decision-transformer`. 2024 successors: Q-DT, Q-value Regularized Transformer (Hu 2024), Diffusion-Branch DT (Wen 2024).

**Curriculum slot.** **Week 7**.

### A.9 Trajectory Transformer (Janner 2021)

**Intuition.** Like DT but tokenize *everything* (states, actions, rewards, returns) and use **beam-search planning** with a value heuristic at inference. Transformer is a discrete world model + behavior model + value model fused.

**When.** Whenever planning at inference is acceptable; planning compensates for weaker policy training.

**Hyperparameters.** discretize each dim into 100 bins, K=10 plan horizon, beam width 64, 4 layers.

**Code.** `jannerm/trajectory-transformer`.

**Curriculum slot.** **Week 8** — first taste of *learned-model planning*, the bridge to Track B.

### A.10 Diffuser (Janner 2022)

**Intuition.** Model entire trajectories `τ = (s_0, a_0, ..., s_T)` with a diffusion model, then **guide sampling** with a reward classifier ∇log p(O|τ) (à la classifier guidance). Planning = denoising.

**When.** Long-horizon, goal-conditioned, multi-task. Slow inference (≥20 denoising steps) but expressive.

**Hyperparameters.** T=100 diffusion steps (DDIM 20 at sample time), horizon H=32, U-Net (Janner config), guide scale 0.1.

**Code.** `jannerm/diffuser`.

**Curriculum slot.** **Week 9**.

### A.11 Diffusion Policies (Chi 2023) / Diffusion-QL (Wang 2022)

**Intuition.** Replace the Gaussian/Tanh actor with a *conditional diffusion model* over actions: `π(a|s) = denoise(ε | s)`. Two flavors: (i) **Diffusion Policy** for visuomotor BC (Chi 2023 — IL on real-robot data, now industry standard); (ii) **Diffusion-QL** for offline RL — diffusion actor + standard Q with policy-improvement loss `−Q(s, π(s))` added to denoising loss.

**When.** Multi-modal action distributions (real-robot manipulation, human demos). Diffusion Policy is the IL workhorse of 2024–2026.

**Hyperparameters.** N=5–20 reverse steps, EMA on actor (decay 0.995), η ∈ [1, 5] for Q-weighting, receding-horizon T_a < T_p (predict T_p actions, execute T_a).

**Code.** `columbia-ai-robotics/diffusion_policy`, `Zhendong-Wang/Diffusion-Policies-for-Offline-RL`.

**Curriculum slot.** **Week 10–11** (two weeks: visuomotor BC, then Diffusion-QL).

### A.12 IDQL — Implicit Diffusion Q-Learning (Hansen-Estruch 2023)

**Intuition.** Marry IQL critic with a diffusion behavior model. Sample K candidate actions from the diffusion behavior policy, re-weight by `exp((Q − V)/β)`. Sidesteps the unstable end-to-end Diffusion-QL gradient.

**When.** D4RL-style benchmarks; SOTA on AntMaze when paired with the right τ.

**Hyperparameters.** K = 32–128 candidates, IQL τ = 0.7–0.9, β = 3–10, denoise steps 5.

**Code.** `philippe-eecs/IDQL`.

**Curriculum slot.** **Week 11**.

### A.13 Q-Transformer (Chebotar 2023)

**Intuition.** Discretize each action *dimension* into bins, autoregress over dimensions with a transformer Q-head, apply CQL-style conservatism per dim. Scales offline TD-learning to **large, multi-task, real-robot** datasets (RT-1/RT-2 style).

**When.** Robot fleets, mixed-quality data, language-conditioned manipulation.

**Hyperparameters.** 256 bins/dim, 16 layers, vision-language backbone (e.g., EfficientNet+USE), CQL α auto-tuned.

**Code.** `lucidrains/q-transformer` (community PyTorch); Google paper code not public.

**Curriculum slot.** **Week 12** — the offline-RL scaling capstone.

### A.14 SfBC — Selecting from Behavior Candidates (Chen 2023)

**Intuition.** Two-stage: (1) train a diffusion behavior model on the dataset; (2) at inference, sample N candidate actions, pick `argmax_a Q(s, a)` for a learned Q. Conceptual ancestor of IDQL.

**When.** When you want a clean decoupling of policy class (diffusion) and improvement (Q-argmax).

**Hyperparameters.** N=10–50 candidates, simple FQE-style Q.

**Code.** `Bin-Chen/SfBC`.

**Curriculum slot.** **Week 11** (taught with IDQL).

### A.15 OPE — Off-Policy Evaluation

Before deploying any offline-trained π, **estimate J(π) from D without rolling out**.

- **Importance Sampling (IS) / Per-Decision IS / Weighted IS.** Unbiased (for finite horizons) but variance explodes as `(π/μ)^H`. Useful only for short horizons or near-on-policy.
- **Doubly Robust (Jiang & Li 2016).** Combine IS with a learned value baseline; unbiased if either is correct. Variance bounded by min(IS, model) up to a constant.
- **FQE — Fitted Q Evaluation.** Repeatedly regress `Q_k(s,a) → r + γ Q_{k-1}(s', π(s'))`. Low variance, biased. The empirical workhorse — most reliable single estimator across D4RL/NeoRL studies.
- **DICE family — DualDICE / GenDICE / BestDICE (Nachum & Dai).** Estimate the *stationary state-action ratio* `d^π/d^μ` by solving a convex dual program. Marginalized IS — variance ~constant in horizon. Sensitive to function approximation.
- **MAGIC, MRDR.** Adaptive blends of DR with model-based estimates.

**When to use what.**
- Short horizon (H<30), near-on-policy → weighted IS.
- General-purpose, single number → FQE.
- Long horizon, off-policy distribution shift → DICE.
- Safety-critical decisions → report a *set*: FQE + DICE + DR, and take the min.

**Curriculum slot.** **Week 13** — OPE bootcamp before any offline-to-online transition.

### A.16 Benchmarks

- **D4RL (Fu 2020).** MuJoCo locomotion (medium/medium-replay/medium-expert/expert), AntMaze (umaze/medium/large + diverse/play), Adroit, Kitchen, Carla. Still the lingua franca but **maintenance has shifted to Minari** (Farama Foundation), which provides typed datasets and a standardized HDF5 schema — pip-installable, integrates with Gymnasium.
- **NeoRL & NeoRL-2 (2022 / 2025).** "Near real-world": narrow data distributions from deployed controllers, delayed actions, external disturbances, safety constraints. The 2025 NeoRL-2 paper shows that **CQL, TD3+BC, MOPO, MOBILE, RAMBO, COMBO all routinely fail to beat the behavior policy** — a sobering reality check vs D4RL.
- **RL Unplugged (DeepMind).** Atari + DM Control + locomotion at scale.
- **Visual-manipulation.** RT-X / Open-X-Embodiment (1M+ robot trajectories), used to train RT-2, OpenVLA, π0.

**Curriculum slot.** **Week 2** (D4RL + Minari setup), **Week 14** (NeoRL-2 reality check).

---

## Track B — Model-Based RL

The unifying problem: learn `T̂(s'|s,a)` and `r̂(s,a)`, then **plan or imagine** to derive a policy. Trade-offs are between (i) model class expressivity (Gaussian ensemble → discrete VAE → diffusion), (ii) planning method (CEM/MPPI vs MCTS vs imagination rollouts), and (iii) sample efficiency vs asymptotic performance.

### B.1 PETS — Probabilistic Ensembles with Trajectory Sampling (Chua 2018)

**Intuition.** Ensemble of probabilistic dynamics nets (Gaussian heads); plan with **CEM** over action sequences using *trajectory sampling* to propagate epistemic+aleatoric uncertainty.

**When.** Low-dim continuous control, ≤200k env steps. Still a strong baseline.

**Hyperparameters.** 5-net ensemble, 200 CEM samples, 5 elites, plan horizon 25, 5 CEM iterations.

**Code.** `kchua/handful-of-trials`, `facebookresearch/mbrl-lib`.

**Curriculum slot.** **Week 15** — first model-based week.

### B.2 MBPO — Model-Based Policy Optimization (Janner 2019)

**Intuition.** "Dyna with short rollouts." Train SAC on a 1:1 mix of real and **k-step model rollouts** (k=1–15, scheduled). Theoretical: short rollouts bound the compounding-error penalty.

**When.** MuJoCo locomotion at <300k steps. The textbook example of model-based acceleration.

**Hyperparameters.** ensemble of 7 dyn-nets, k = 1 → 15 schedule, 400 model rollouts per env step, SAC inside.

**Code.** `JannerM/mbpo`, `mbrl-lib`.

**Curriculum slot.** **Week 15**.

### B.3 Dreamer v1 / v2 / v3 (Hafner 2019 / 2020 / 2023)

**Intuition.** Learn a **RSSM** (recurrent state-space model) with stochastic discrete latents (v2+); train actor & critic *entirely in imagination* via backprop through the world model.

- **v1 (PlaNet/Dreamer):** continuous latents, CEM planning then actor-critic.
- **v2:** **categorical latents** (32×32 one-hot), KL balancing, beats Rainbow on Atari with 200M frames.
- **v3 (2023):** **same hyperparameters across 150+ tasks** — DMC, Atari, Crafter, Minecraft (first agent to mine diamond from scratch). Adds symlog two-hot reward heads, robust loss scaling, EMA target critic.

**When.** v3 is the **default model-based agent** for partially observable, pixel-based control circa 2024–2026.

**Hyperparameters.** (v3) image 64×64, 32×32 categorical, RSSM hidden 4096 (XL), imagination horizon H=15, KL free-bits 1.0, actor entropy 3e-4, batch 16×64.

**Code.** `danijar/dreamerv3` (JAX), `NM512/dreamerv3-torch`.

**Curriculum slot.** **Week 16–17** (two weeks: implement DreamerV3 on a DMC suite, then on Crafter).

### B.4 TD-MPC / TD-MPC2 (Hansen 2022 / 2024)

**Intuition.** Latent-space MPC: learn a latent dynamics + reward + Q + π, plan via **MPPI in latent space** with a terminal Q bootstrap. v2 (2024) introduces **task embeddings** and a single **317M-param** model that handles 80 tasks across DMC, Meta-World, ManiSkill.

**When.** Continuous control where you want planning + value bootstrap. Often beats DreamerV3 on continuous-control suites (per the TD-MPC2 paper).

**Hyperparameters.** plan horizon 5, 512 MPPI samples, 64 elites, 6 iters, temperature 0.5, latent dim 512, ensemble of 5 Qs, LayerNorm everywhere.

**Code.** `nicklashansen/tdmpc2`.

**Curriculum slot.** **Week 18**.

### B.5 IRIS (Micheli 2022) and successors

**Intuition.** Tokenize observations with a **VQ-VAE**, learn an **autoregressive transformer** over `(token, action)` sequences as a world model, train a PPO/Dreamer-style policy in its imagination. First transformer world model to be SOTA on Atari 100k.

**When.** Discrete-action visual environments; benchmarks at <100k env steps.

**Successors (2024–2025).** **Δ-IRIS** (encode stochastic deltas, SOTA on Crafter + better Atari 100k), **STORM** (stochastic transformer + KL prior, matches DreamerV3 with faster wall-clock), **Masked Latent Transformers (2025)** (efficient context-aware tokenization).

**Hyperparameters.** VQ-VAE codebook 1024, transformer 10 layers, context 20 tokens, imagination H=15.

**Code.** `eloialonso/iris`, `vmicheli/delta-iris`.

**Curriculum slot.** **Week 19**.

### B.6 DIAMOND — Diffusion as Model of Environment Dreams (Alonso 2024, NeurIPS Spotlight)

**Intuition.** **Diffusion world model** over pixels (not tokens). EDM-style denoising of the next frame conditioned on past frames+actions. Preserves fine visual detail that VQ-tokenization discards — *visual details matter*. Hits **1.46 mean HNS on Atari 100k**, a new best for world-model agents, and demonstrates real-time interactive Counter-Strike: GO play from a static-data-trained diffusion engine.

**When.** Visually rich environments where tokenization is lossy; "neural game engine" applications.

**Hyperparameters.** EDM denoiser U-Net, 10 sampling steps at imagination, 100 at high-quality eval; condition on 4 past frames + action; PPO actor-critic on imagined rollouts.

**Code.** `eloialonso/diamond`.

**Curriculum slot.** **Week 20** — capstone for the world-model thread.

### B.7 MuZero (Schrittwieser 2020)

**Intuition.** AlphaZero without a known model. Learn three heads — **representation** `h(s)`, **dynamics** `g(s_latent, a) → (s'_latent, r̂)`, **prediction** `f(s_latent) → (π̂, v̂)` — purely from value/policy/reward targets (no observation reconstruction). Plan with **MCTS in latent space**.

**When.** Discrete-action games (Go, chess, Atari) where MCTS is feasible.

**Hyperparameters.** num_simulations 50 (Atari) / 800 (Go), Dirichlet noise α=0.3, c_puct=1.25, n-step=5, replay reanalyze.

**Code.** `werner-duvaud/muzero-general`, `opendilab/LightZero`.

**Curriculum slot.** **Week 21**.

### B.8 EfficientZero (Ye 2021) → EfficientZero V2 (Wang 2024)

**Intuition.** MuZero + three sample-efficiency tricks: (1) self-supervised consistency loss on dynamics, (2) value-prefix prediction (sum of future rewards), (3) off-policy correction via reanalyze.

**V2 (2024)** generalizes to **continuous control + discrete action + vector + pixel** via Gaussian policy heads (inspired by Sampled MuZero) and **search-based value estimation**. Beats DreamerV3 on 50/66 tasks across Atari 100k, Proprio Control, Vision Control.

**When.** Best published sample-efficient agent on Atari 100k (along with DIAMOND); now also a strong continuous-control contender.

**Hyperparameters.** num_simulations 50, K-step unroll 5, SimSiam consistency weight 2.0, Gaussian policy with 20 action samples for continuous.

**Code.** `YeWR/EfficientZero`, `opendilab/LightZero` (V2).

**Curriculum slot.** **Week 22**.

### B.9 Stochastic MuZero (Antonoglou 2022)

**Intuition.** Vanilla MuZero is deterministic — fails in stochastic environments (backgammon, poker, real games with hidden info). Adds an **after-state** node and a **chance node** with learned outcome distribution, training the outcome head with VQ-VAE-style discretization.

**When.** Stochastic dynamics or partial observability.

**Code.** `opendilab/LightZero` (in progress per their tracker).

### B.10 Sampled MuZero (Hubert 2021)

**Intuition.** Generalize MCTS to large/continuous action spaces by **sampling K actions** from a learned proposal at each node; MCTS now runs over the subset, with importance correction in the policy improvement.

**When.** Continuous control via MCTS (precursor to EfficientZero V2's continuous extension).

### B.11 UniZero (2024)

**Intuition.** Latent **transformer** world model + MCTS — replaces MuZero's RNN dynamics with attention, enabling longer planning horizons and better generalization across tasks.

**Code.** `opendilab/LightZero`.

**Curriculum slot.** **Week 22** (taught alongside EfficientZero V2 as "the LightZero family").

### B.12 World-Models-for-Robotics: Gato → RT-1 → RT-2 → OpenVLA → π0 → Gemini Robotics (2022–2026)

**Intuition.** Treat robotic control as **token prediction** with VLMs. Not "model-based" in the planning sense — the *implicit* world model lives inside the LLM's pretraining; actions are emitted as language tokens or discretized motor tokens.

- **Gato (DeepMind 2022)** — 604-task multimodal transformer; proof-of-concept.
- **RT-1 / RT-2 (Google 2022/2023)** — VLM (PaLI-X / PaLM-E) fine-tuned to output discretized end-effector deltas.
- **OpenVLA (2024)** — open 7B VLA on Llama-2 + DINOv2 + SigLIP, trained on OpenX-Embodiment.
- **π0 (Physical Intelligence 2024)** — flow-matching action head atop a VLM; SOTA dexterous manipulation.
- **Gemini Robotics / Robotics-ER / Robotics 1.5 (Google, 2025)** — multimodal VLA with embodied-reasoning variant.

**When.** Real-world manipulation with large pretrained backbones; data scale > algorithm cleverness.

**Curriculum slot.** **Week 23** — the "scale + foundation models" capstone.

---

## Curriculum Slot Map (24 weeks, interleaved)

| Wk | Theme | Track A (Offline) | Track B (Model-Based) |
|----|-------|-------------------|-----------------------|
| 1  | Online RL refresher | SAC, TD3 | – |
| 2  | Datasets & infra | D4RL + Minari setup | – |
| 3  | OOD problem | BCQ, BEAR | – |
| 4  | Conservatism | CQL, TD3+BC | – |
| 5  | Implicit methods | IQL, AWR/AWAC | – |
| 6  | Tuning matters | ReBRAC | – |
| 7  | Sequence modeling | Decision Transformer | – |
| 8  | Planning from sequences | Trajectory Transformer | – |
| 9  | Trajectory diffusion | Diffuser | – |
| 10 | Visuomotor IL | Diffusion Policy (Chi) | – |
| 11 | Diffusion + Q | Diffusion-QL, SfBC, IDQL | – |
| 12 | Scale | Q-Transformer | – |
| 13 | Evaluation | OPE (IS/DR/FQE/DICE) | – |
| 14 | Reality check | NeoRL-2 | – |
| 15 | Dyna basics | – | PETS, MBPO |
| 16 | Latent world models I | – | DreamerV3 (DMC) |
| 17 | Latent world models II | – | DreamerV3 (Crafter) |
| 18 | Latent MPC | – | TD-MPC2 |
| 19 | Transformer WMs | – | IRIS, Δ-IRIS, STORM |
| 20 | Diffusion WMs | – | DIAMOND |
| 21 | MCTS in latent | – | MuZero |
| 22 | LightZero family | – | EfficientZero V2, Stochastic/Sampled MuZero, UniZero |
| 23 | Foundation models | – | Gato → RT-2 → OpenVLA → π0 → Gemini Robotics |
| 24 | Capstone | Offline-pretrain (IQL+Diffusion-Policy) → online finetune in DreamerV3 latent | Combine both tracks |

---

## Cross-Cutting Themes (introduced just-in-time)

- **OOD pathology** (wk 3) underlies both tracks: in offline RL it bites the Bellman target; in model-based RL it bites the model rollout. Same beast, two names — *compounding error*.
- **Diffusion** appears in *both* tracks (policy class wk 10–11, world model wk 20); the same DDPM/EDM toolkit suffices.
- **Transformers** also appear in both (DT/TT/Q-Transformer in A; IRIS/UniZero/VLAs in B).
- **Planning vs amortized policy.** Trajectory Transformer + Diffuser (offline) ↔ TD-MPC2 + MuZero (model-based) — the same trade-off across tracks.
- **Evaluation discipline.** OPE in wk 13 is mandatory before any deployment claim; for model-based RL, the analogous discipline is **model rollout horizon ablation** (introduced wk 15 with MBPO).

---

## Recommended References (single-source-of-truth links)

- D4RL: `Farama-Foundation/D4RL`, Minari: `Farama-Foundation/Minari`, NeoRL-2: `arxiv.org/abs/2503.19267`.
- Offline RL reproductions: `tinkoff-ai/CORL` (single-file PyTorch for CQL, IQL, TD3+BC, AWAC, ReBRAC, DT, SAC-N, EDAC).
- Model-based: `facebookresearch/mbrl-lib`, `danijar/dreamerv3`, `nicklashansen/tdmpc2`, `eloialonso/diamond`, `opendilab/LightZero` (MuZero / EfficientZero / Sampled / Stochastic / UniZero in one repo).
- Robotics-VLA: `openvla/openvla`, `Physical-Intelligence/openpi`.

End of dossier.
