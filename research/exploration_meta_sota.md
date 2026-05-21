# Exploration + Meta-RL — SOTA Dossier

*Compiled May 2026. Scope: 2024–2026 state of the art for deep RL exploration, intrinsic motivation, unsupervised pre-training, meta-RL, in-context RL, and open-ended learning. Each entry: when it matters, intuition, key equations, citation/arxiv/date, and a "curriculum slot" mapping into a 24-week self-study plan.*

---

## Part I — Exploration & Intrinsic Motivation

The unifying problem: in MDPs with sparse or deceptive reward, on-policy gradient signal vanishes. Exploration bonuses give the agent a dense, learned signal — count, prediction error, information gain, or skill diversity. The 2024–2026 frontier has consolidated around **representation-aware** bonuses (latent prediction error, not pixel error) and **world-model-coupled** exploration.

### 1. Count-based & Pseudocounts (CTS, PixelCNN, Hash counts)

- **When it matters**: tabular-like discrete state spaces and tasks where novelty is well-defined by visitation. Foundational baseline.
- **Intuition**: replace `1/sqrt(N(s))` UCB-style bonus with a *density model* `ρ(s)` so that pseudocount `N̂(s) = ρ(s)(1-ρ'(s))/(ρ'(s)-ρ(s))` generalises to high-dim observations.
- **Key eq**: `r⁺ = r + β / sqrt(N̂(s) + ε)`.
- **Refs**: Bellemare et al. "Unifying Count-Based Exploration and Intrinsic Motivation" (arxiv:1606.01868, 2016); Tang et al. "#Exploration: A Study of Count-Based Exploration" with SimHash (arxiv:1611.04717, 2017); Ostrovski et al. PixelCNN counts (arxiv:1703.01310, 2017).
- **Curriculum slot**: **Week 7** — taught alongside MBIE-EB / UCB intuition right after model-free DQN.

### 2. ICM — Intrinsic Curiosity Module

- **When it matters**: image-based environments with controllable dynamics (VizDoom, Mario). Decouples curiosity from pixel-level noise via an *inverse model*.
- **Intuition**: predict the next latent `φ(sₜ₊₁)` from `φ(sₜ), aₜ` via a forward model; reward the agent for prediction error in a feature space trained to be controllable.
- **Key eq**: `r_int = (η/2) ||f(φ(sₜ),aₜ) − φ(sₜ₊₁)||²`, with inverse loss `‖a − g(φ(sₜ),φ(sₜ₊₁))‖²` shaping φ.
- **Ref**: Pathak et al. "Curiosity-Driven Exploration by Self-Supervised Prediction" (arxiv:1705.05363, 2017).
- **Curriculum slot**: **Week 8**.

### 3. RND — Random Network Distillation

- **When it matters**: hard-exploration Atari (Montezuma's Revenge); simple to bolt onto PPO; no inverse model needed; robust to noisy-TV problem because target is deterministic.
- **Intuition**: a fixed random net f̂ maps observations to features; a trained predictor f tries to match it. Prediction error → novelty signal that decays with visitation.
- **Key eq**: `r_int = ‖f̂(s) − f(s; θ)‖²`, minimise the same over a sliding buffer.
- **Ref**: Burda et al. (arxiv:1810.12894, Oct 2018). Still the most-used baseline in 2025 papers per ICLR/NeurIPS surveys.
- **Curriculum slot**: **Week 8** (paired with ICM).

### 4. NGU & Agent57

- **When it matters**: long-horizon Atari, when both episodic novelty (short-term) and lifelong novelty (RND-style, long-term) are needed.
- **Intuition**: NGU mixes an *episodic* memory-based k-NN bonus with an RND lifelong modulator. Agent57 adds a meta-controller that picks among a family of policies indexed by `(β, γ)` — the first agent to beat the human baseline on all 57 Atari games.
- **Key eq**: `r_int = r^episodic · min(max(α_RND, 1), L)`, with α_RND = 1+(err−μ)/σ.
- **Refs**: Badia et al. "Never Give Up" (arxiv:2002.06038, Feb 2020); Badia et al. "Agent57" (arxiv:2003.13350, Mar 2020).
- **Curriculum slot**: **Week 9**.

### 5. Go-Explore

- **When it matters**: deceptive sparse reward, when you can save & restore state (or use a goal-conditioned policy).
- **Intuition**: maintain an archive of visited cells; *return* to a promising cell deterministically, then *explore* from it stochastically. Separates the two failure modes (forgetting promising states vs. failing to explore).
- **Ref**: Ecoffet et al. "First return, then explore" (Nature 2021, arxiv:2004.12919, Apr 2020).
- **Curriculum slot**: **Week 9**.

### 6. BYOL-Explore

- **When it matters**: visually rich 3D environments (DM-HARD-8); state of the art on the ten hardest Atari exploration games as of 2022, still competitive in 2025 benchmarks.
- **Intuition**: a *single* self-supervised latent-prediction loss (à la BYOL) trains both the world model and the curiosity bonus. The exponential-moving-average target avoids representation collapse on noisy observations.
- **Key eq**: `L = ‖sg(target(φ(sₜ₊ₖ))) − pred(h(sₜ), a_{t:t+k-1})‖²`; intrinsic reward = same per-step loss.
- **Ref**: Guo et al. (arxiv:2206.08332, Jun 2022).
- **Curriculum slot**: **Week 10**.

### 7. Surprise-driven & Bayesian surprise (VIME, EMI)

- **When it matters**: continuous control with structured dynamics; theoretically principled (information gain ≈ KL between posteriors).
- **Key eq**: `r_int = KL(p(θ|s,a,s')‖p(θ|s,a))` — Bayesian surprise over dynamics parameters.
- **Refs**: Houthooft et al. "VIME" (arxiv:1605.09674, 2016); Kim et al. "EMI" (arxiv:1810.01176, 2018).
- **Curriculum slot**: **Week 10**.

### 8. Empowerment

- **When it matters**: agents need an *intrinsic notion of control* without task reward; pre-training for downstream control tasks.
- **Intuition**: maximise the mutual information between the agent's k-step actions and resulting future states: `E(s) = max_{p(a|s)} I(A^k; S' | s)`. Equivalent to "how much can I causally influence my future?"
- **Refs**: Mohamed & Rezende "Variational Information Maximisation for Intrinsically Motivated RL" (arxiv:1509.08731, 2015); recent revival in skill-conditioned form via CIC and Choreographer.
- **Curriculum slot**: **Week 11**.

### 9. Diversity: DIAYN, SMERL, EDL, CIC

- **When it matters**: unsupervised skill pre-training; later compose or fine-tune skills for downstream tasks.
- **Intuition**: maximise `I(s; z)` — make skills `z` discriminable from the states they visit. SMERL conditions on task reward and only injects diversity when near-optimal. EDL flips the direction (state-conditional skill). CIC uses contrastive estimation for high-dim skill spaces.
- **Key eq (DIAYN)**: `r_int(s, z) = log q_φ(z|s) − log p(z)` with q_φ a learned skill discriminator.
- **Refs**: Eysenbach et al. "DIAYN" (arxiv:1802.06070, Feb 2018); Kumar et al. "SMERL" (arxiv:2010.14484, 2020); Campos et al. "EDL" (arxiv:2002.03647, 2020); Laskin et al. "CIC" (arxiv:2202.00161, 2022). 2024–25 follow-ups: ComSD (contrastive dynamic skills, 2024); analyses (arxiv:2501.11533) show DIAYN does *not* by itself promote spatial exploration in MiniGrid — diversity ≠ coverage.
- **Curriculum slot**: **Week 12**.

### 10. Unsupervised RL — URLB, Plan2Explore, APT/APS, Choreographer

- **When it matters**: pre-train a single agent without rewards, then fine-tune in <100k steps on many tasks. Now the standard setup for foundation-model-style RL.
- **Intuition**: Plan2Explore uses *ensemble disagreement* over latent dynamics (Dreamer world model) as intrinsic reward → drives the agent to maximally informative states. APT uses particle-based state entropy. URLB unifies 8 such methods on 12 DMC tasks.
- **Key eq (Plan2Explore)**: `r_int = Var_{ensemble}( f_i(h_t, a_t) )`.
- **Refs**: Sekar et al. "Plan2Explore" (arxiv:2005.05960, May 2020); Laskin et al. "URLB" (arxiv:2110.15191, Oct 2021); Liu & Abbeel "APT/APS" (2021); "Choreographer" (Mazzaglia et al., ICLR 2023); 2024 follow-ups push +17% over Plan2Explore on URLB via better latent-model regularisation.
- **Curriculum slot**: **Week 13** — full URLB sweep as a lab.

### 11. Posterior Sampling / Bayesian exploration (PSRL, BootDQN, RPF)

- **When it matters**: provably efficient exploration with regret bounds; tabular & linear MDPs; bootstrap heads for deep RL.
- **Intuition**: sample a hypothesis MDP from the posterior, act optimally w.r.t. it for an episode — *Thompson sampling* lifted to MDPs. Deep approximation: train K bootstrapped Q-heads, sample one per episode.
- **Refs**: Osband et al. "PSRL" (arxiv:1306.0940, 2013); "Bootstrapped DQN" (arxiv:1602.04621, 2016); "Randomised Prior Functions" (arxiv:1806.03335, 2018).
- **Curriculum slot**: **Week 14**.

### 12. 2025–2026 frontier on exploration

- *Latent-prediction families consolidating*: BYOL-Explore-style targets are now standard inside Dreamer-V3 / TD-MPC2 pipelines. Pixel-level prediction error is essentially dead as a baseline.
- *In-context exploration*: ICEE / In-Context Curiosity (arxiv:2510.00347, Oct 2025) distils RND-style bonuses into DPT-style transformers so that *exploration itself* is meta-learned — the agent does Thompson-sampling-like behaviour purely in-context.
- *World-model coupled count*: 2025 work blends hash-counts in a learned latent (post-BYOL projector), recovering count-bonus theoretical guarantees while scaling to pixels.
- **Curriculum slot**: **Week 15** — capstone exploration lab combining BYOL-Explore + ICEE.

---

## Part II — Meta-RL & Multi-Task RL

The unifying problem: amortise the cost of learning over a *distribution* of tasks `p(M)`, so that a new task from p is solved in O(few) rollouts instead of O(millions).

### 13. MAML & Reptile (gradient-based meta-RL)

- **When it matters**: small task distributions, dense reward, when policy gradient itself adapts quickly to a new task with 1–5 grad steps.
- **Intuition**: meta-learn an *initialisation* θ such that one gradient step on task τ yields a good policy. Reptile is the first-order, no-Hessian variant.
- **Key eq (MAML)**: `θ ← θ − β ∇_θ Σ_τ L_τ(θ − α ∇_θ L_τ(θ))`.
- **Refs**: Finn et al. "MAML" (arxiv:1703.03400, Mar 2017); Nichol et al. "Reptile" (arxiv:1803.02999, 2018); Stadie et al. "E-MAML/E-RL²" (arxiv:1803.01118, 2018) — adds explicit exploration credit.
- **Curriculum slot**: **Week 16**.

### 14. RL² (memory-based meta-RL)

- **When it matters**: when the task identity can be inferred from a short history; RNN/Transformer policy across episodes within a "trial".
- **Intuition**: treat the *learning algorithm* as the RNN's hidden-state update. Train across many MDPs from p(M); a single forward pass on a new MDP implements amortised inference + policy.
- **Refs**: Duan et al. "RL²" (arxiv:1611.02779, Nov 2016); Wang et al. "Learning to Reinforcement Learn" (arxiv:1611.05763, 2016).
- **2025 extension**: "Boosting Meta-RL via RL inside RL²" (RLJ 2025) — outer RL on the inner RL² agent gives sample-efficiency gains on Meta-World.
- **Curriculum slot**: **Week 16**.

### 15. PEARL (probabilistic context-conditioned meta-RL)

- **When it matters**: continuous control with smooth task variation (locomotion at varying velocities); off-policy meta-RL.
- **Intuition**: encode trajectory context `c` into a Gaussian latent `z ~ q(z|c)` (permutation-invariant); condition the SAC policy on z; posterior sampling over z gives Thompson-style exploration.
- **Key eq**: `q(z|c) ∝ Π_i q(z|c_i)`, with `ELBO = E_q[R] − β KL(q(z|c) ‖ p(z))`.
- **Ref**: Rakelly et al. (arxiv:1903.08254, Mar 2019).
- **Curriculum slot**: **Week 17**.

### 16. VariBAD (Bayes-adaptive deep RL)

- **When it matters**: when Bayes-optimal exploration matters (you want the agent to *probe* before exploiting); small-to-medium task distributions.
- **Intuition**: meta-learn a VAE that, given history, produces a belief b_t over MDPs; policy conditions on (s_t, b_t) — explicitly Bayes-adaptive.
- **Key eq**: `π(a|s,b)` with `b = q_φ(m|τ_{:t})`, ELBO over reconstructing reward and dynamics.
- **Ref**: Zintgraf et al. (arxiv:1910.08348, Oct 2019).
- **2025 extension**: "Bayesian Meta-RL with Laplace Variational Recurrent Networks" (arxiv:2505.18591, May 2025) — sharper posteriors, better uncertainty calibration. SDVT, TrMRL, ECET are the 2024-25 transformer-based variBAD variants.
- **Curriculum slot**: **Week 17**.

### 17. In-context RL — AD, DPT, DICP

- **When it matters**: this is the **dominant 2024–2026 paradigm**. Pre-train a transformer on RL learning histories; new task adaptation is *just* sequence prefilling — no gradient updates.
- **Algorithm Distillation (AD)**: Laskin et al. (arxiv:2210.14215, Oct 2022). Train a causal transformer on cross-episode (s,a,r) sequences from a *source RL algorithm's training history*. The model imitates the algorithm's improvement curve — at inference, just keep extending the context and the policy improves in-context.
- **DPT (Decision-Pretrained Transformer)**: Lee et al. (arxiv:2306.14892, Jun 2023). Supervised pretraining: predict the *optimal* action given a query state and a context of (s,a,r,s'); provably implements Bayesian posterior sampling. **Scales to in-context RL across hundreds of tasks**.
- **DICP** (Distilling RL Algorithms for In-Context Model-Based Planning, arxiv:2502.19009, Feb 2025): combines AD with learned world models; SOTA on Dark Room / Watermaze with far fewer env interactions.
- **In-Context Curiosity / ICEE** (arxiv:2510.00347, Oct 2025): distils exploration bonuses into DPT for bandits; bridges the "DPT replicates suboptimal source" gap.
- **Key intuition**: amortise *the act of learning* into a sequence model. Bayesian-posterior-sampling-equivalent when trained correctly.
- **Curriculum slot**: **Weeks 18–19** — two-week deep dive; build a from-scratch DPT on Dark Room.

### 18. Gato & RT-2 (generalist transformer policies)

- **When it matters**: multi-modal multi-task multi-embodiment — one transformer for hundreds of tasks (Atari, captioning, robot block-stacking).
- **Intuition**: tokenise *everything* (pixels, text, actions, proprio) into a unified sequence; decoder-only transformer predicts the next token; the same weights answer "what's in this image" and "what torque on this joint".
- **Refs**: Reed et al. "Gato" (arxiv:2205.06175, May 2022); Brohan et al. "RT-2" (arxiv:2307.15818, Jul 2023) — VLM grounded for robot control. 2025 successors: RT-X, OpenVLA, OCTO — all open-source generalist policies trained on the Open-X-Embodiment dataset.
- **Curriculum slot**: **Week 20**.

### 19. Open-ended: POET, Enhanced POET, AdA

- **When it matters**: you have neither a fixed task nor a fixed environment; you want emergent curricula.
- **POET** (Wang et al., arxiv:1901.01753, Jan 2019): co-evolves environments and agents; transfers stepping-stone solutions across niches; Enhanced POET (arxiv:2003.08536, Mar 2020) generalises the environment representation.
- **AdA — Adaptive Agent** (Team et al., DeepMind, arxiv:2301.07608, Jan 2023): meta-RL transformer trained in **XLand 2.0** — 10¹⁵ task space. Human-timescale adaptation on held-out tasks. Demonstrates that scale + diversity + memory ≈ in-context RL with foundation-model behaviour.
- **2024–25 line**: OMNI (open-endedness via LLM-graded interestingness), OMNI-EPIC (LLMs generate POET-style environment code, 2024); GENIE-2 (Google DeepMind, 2024) as a learned open-ended world generator. The **curriculum-from-LLM** paradigm has largely replaced hand-tuned POET niches.
- **Curriculum slot**: **Week 21**.

### 20. Voyager & LLM-as-RL-agent

- **When it matters**: long-horizon embodied agents in vast environments (Minecraft, web, OS); when an LLM can write *executable code as skills*.
- **Intuition**: replace policy gradient with (i) LLM-proposed curriculum, (ii) iterative code-as-skills library, (iii) self-verification via execution feedback. Voyager dominates prior Minecraft SOTA (3.3× unique items, 15.3× faster tech-tree).
- **Ref**: Wang et al. "Voyager" (arxiv:2305.16291, May 2023).
- **2024–25 successors**: GITM (Ghost-in-the-Minecraft), Co-Voyager (multi-agent collaboration), JARVIS-1, MP5, Auto-MC-Reward; broader trend: **LLM proposes goals + verifies, RL/code executes** — the dominant pattern for open-ended embodied agents in 2025.
- **Curriculum slot**: **Week 22**.

### 21. 2025–2026 frontier on meta/multi-task

- *In-context RL is winning*: across NeurIPS 2024 / ICLR 2025, DPT/AD-style transformers replace MAML/PEARL as the default meta-RL baseline. The shift is "from bilevel optimisation to sequence modelling".
- *Foundation policies*: OpenVLA, π0 (Physical Intelligence, 2024), RT-X family — generalist transformers trained on millions of cross-embodiment robot trajectories, fine-tuned with RLHF/preference learning.
- *LLM-driven curricula*: the open-ended loop is now LLM-mediated (OMNI-EPIC, Eureka-style reward design, Voyager-style skill libraries). Hand-coded POET niches are obsolete.
- *Bayesian rigour returning*: Flow-Matching DPT (2025) and Laplace-VariBAD restore principled posteriors on top of the scale-driven transformer trend.
- **Curriculum slots**: **Weeks 23–24** — capstone projects: (23) re-implement DPT on a Meta-World-Lite distribution; (24) Voyager-style LLM-curriculum on Crafter.

---

## 24-Week Curriculum Map (consolidated)

| Week | Topic |
|------|-------|
| 1-6  | Foundations: MDPs, DP, TD, DQN, PPO, SAC, Dreamer-V3 |
| 7    | Count-based & pseudocount exploration |
| 8    | ICM + RND |
| 9    | NGU / Agent57 / Go-Explore |
| 10   | BYOL-Explore + VIME/EMI |
| 11   | Empowerment |
| 12   | Diversity: DIAYN/SMERL/EDL/CIC |
| 13   | Unsupervised RL: URLB sweep + Plan2Explore |
| 14   | Bayesian exploration: PSRL, BootDQN, RPF |
| 15   | Capstone exploration: BYOL-Explore + ICEE 2025 |
| 16   | MAML/Reptile + RL² |
| 17   | PEARL + VariBAD (+ 2025 Laplace variant) |
| 18-19| In-context RL: AD, DPT, DICP, ICEE |
| 20   | Gato, RT-2, OpenVLA |
| 21   | Open-ended: POET → AdA → OMNI-EPIC |
| 22   | Voyager + LLM-as-agent |
| 23   | Capstone: DPT on Meta-World-Lite |
| 24   | Capstone: LLM-curriculum on Crafter |

---

## Key takeaways for May 2026

1. **Exploration has converged on latent-prediction**: BYOL-Explore-family bonuses, paired with Dreamer-V3 / TD-MPC2 world models, are the de-facto recipe. RND remains the simple baseline; pixel-prediction error is dead.
2. **Meta-RL has been absorbed by in-context RL**: AD/DPT/DICP supplant MAML/PEARL as defaults. The line between "meta-learning" and "foundation-model RL" has effectively dissolved.
3. **Open-endedness is LLM-mediated**: the curriculum generator is now an LLM (OMNI-EPIC, Eureka, Voyager), not hand-engineered niches.
4. **Bayesian principles are quietly returning** under the transformer hood — DPT ≈ posterior sampling, Flow-Matching DPT and Laplace-VariBAD make this explicit.
5. **Skill-discovery ≠ exploration**: 2025 analyses confirm DIAYN-style mutual-information skills do *not* automatically give spatial coverage — combine with state-entropy or count bonuses.

## Sources

- [In-context RL with Algorithm Distillation (arxiv:2210.14215)](https://arxiv.org/abs/2210.14215)
- [Distilling RL Algorithms for In-Context Model-Based Planning — DICP (arxiv:2502.19009)](https://arxiv.org/abs/2502.19009)
- [Supervised Pretraining Can Learn In-Context RL — DPT (arxiv:2306.14892)](https://arxiv.org/abs/2306.14892)
- [In-Context Curiosity (arxiv:2510.00347)](https://arxiv.org/html/2510.00347)
- [Exploration by Random Network Distillation (arxiv:1810.12894)](https://arxiv.org/abs/1810.12894)
- [BYOL-Explore (arxiv:2206.08332)](https://arxiv.org/pdf/2206.08332)
- [URLB (arxiv:2110.15191)](https://arxiv.org/abs/2110.15191)
- [Enhanced POET (arxiv:2003.08536)](https://arxiv.org/abs/2003.08536)
- [Voyager (arxiv:2305.16291)](https://arxiv.org/abs/2305.16291)
- [VariBAD (arxiv:1910.08348)](https://arxiv.org/abs/1910.08348)
- [Bayesian Meta-RL with Laplace VRNs (arxiv:2505.18591)](https://arxiv.org/pdf/2505.18591)
- [Boosting Meta-RL via RL inside RL² (RLJ 2025)](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_52.pdf)
- [The impact of intrinsic rewards on exploration (arxiv:2501.11533)](https://arxiv.org/html/2501.11533v1)
- [State of RL in 2025 — DataRoot Labs](https://datarootlabs.com/blog/state-of-reinforcement-learning-2025)
