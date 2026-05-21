# LLM RL — SOTA Dossier

*Compiled May 2026. Audience: an ML/DL practitioner new to RL, running on a single consumer GPU (e.g. 24 GB VRAM), using HuggingFace TRL + PEFT/LoRA. Each section ends with a Curriculum slot for a 24‑week deep‑RL self-study plan.*

---

## 0. Map of the territory

LLM post-training RL has, between 2022 and 2026, gone through four overlapping eras:

1. **Classical RLHF (2022–2023)** — PPO against a learned scalar reward model (InstructGPT lineage). Heavy infrastructure, four models in memory (policy, reference, reward, critic).
2. **Offline / direct preference optimization (2023–2024)** — DPO and its zoo (IPO, KTO, ORPO, SimPO, CPO, sDPO). No on-policy sampling, no reward model, no critic. Cheap, popular, fragile.
3. **Verifiable-reward reasoning RL (2024–2025)** — DeepSeek-R1, Kimi K1.5, Tülu-3 RLVR, Open-Reasoner-Zero. GRPO/RLOO/REINFORCE-style critic-free objectives driven by rule-based correctness signals (exact match, code execution, formal proof).
4. **Refinement and de-biasing (2025–2026)** — DAPO, Dr.GRPO, VAPO, λ-GRPO, shrinkage baselines, noise-corrected GRPO; iterative/online DPO; generative & rubric reward models; inference-time RL (best-of-N tuning, MCTS, AlphaProof).

The center of gravity in 2026 is **critic-free policy gradients with verifiable or generative rewards on small (1–32 B) reasoning models**, with DPO-family methods relegated to cheap "preference polish" steps in larger pipelines.

---

## 1. PPO for RLHF — the InstructGPT lineage

**Motivation.** Ouyang et al. 2022 ("Training language models to follow instructions with human feedback", [arXiv:2203.02155](https://arxiv.org/abs/2203.02155), Mar 2022) framed alignment as: (1) SFT on demos, (2) train a Bradley–Terry reward model on pairwise human preferences, (3) optimize the policy with PPO against that RM, with a per-token KL penalty against the SFT reference. A 1.3 B InstructGPT was preferred over 175 B GPT-3 — the experimental result that turned RLHF into industry default.

**Loss.** The clipped surrogate
$$\mathcal L^{PPO}(\theta) = \mathbb E_t\!\left[\min\!\big(r_t(\theta)\hat A_t,\;\text{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)\hat A_t\big)\right] - \beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$$
with $r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{\text{old}}(a_t|s_t)$ and $\hat A_t$ a GAE estimate from the critic. The full RLHF reward is $r_\phi(x,y) - \beta \log\pi_\theta(y|x)/\pi_{\text{ref}}(y|x)$.

**When it shines.** Open-ended generation with a well-calibrated RM (helpfulness, harmlessness, style); production pipelines where you can afford four models in VRAM.

**Failure modes.** Reward hacking, length inflation, KL collapse if $\beta$ is mistuned, critic instability on long horizons (sparse end-of-sequence reward), RM overoptimization (Gao et al. 2023 scaling laws, [PMLR v202](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf)).

**Reference implementations.** `trl.PPOTrainer` (legacy, now `PPOv2Trainer`); OpenRLHF; DeepSpeed-Chat; verl. On a single 24 GB GPU you can run PPO on Pythia-160M or Qwen2.5-0.5B with LoRA on the actor and a frozen RM offloaded; expect ~1 k prompts/hour.

**2025–2026 follow-ups.** PPO has largely been displaced by GRPO/RLOO for reasoning, but VAPO ([arXiv:2504.05118](https://arxiv.org/html/2504.05118v1)) revives PPO with value-pretraining, length-adaptive GAE and decoupled clip, hitting 60.4 AIME24 on Qwen-32B in 5 k steps — currently the strongest PPO-family reasoning recipe.

**Curriculum slot:** weeks 9–10 (after classical RL foundations).

---

## 2. DPO and the offline preference zoo

### 2.1 DPO — Direct Preference Optimization
Rafailov et al. 2023, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290). The Bradley–Terry-plus-KL RLHF optimum is closed form in $\pi_\theta$, so the reward model disappears and we minimize
$$\mathcal L_{DPO} = -\mathbb E_{(x,y_w,y_l)}\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right).$$
**Shines** when you have static preference pairs and want a one-shot, stable, supervised-style training run. **Fails** on distribution shift (offline data ≠ on-policy distribution), is biased toward decreasing $\log\pi(y_l)$ aggressively (the "DPO drifts down" phenomenon), and over-fits length.

### 2.2 IPO — Identity Preference Optimization
Azar et al. 2024 ("A General Theoretical Paradigm…", [arXiv:2310.12036](https://arxiv.org/abs/2310.12036), AISTATS 2024). Generalizes DPO to a $\Psi$PO family; choosing $\Psi=\text{Identity}$ gives a squared-loss objective on the implicit log-ratio gap that **does not saturate** when preferences are deterministic, fixing DPO's tendency to push $\beta\to\infty$ effectively.

### 2.3 KTO — Kahneman–Tversky Optimization
Ethayarajh et al. 2024, [arXiv:2402.01306](https://arxiv.org/abs/2402.01306). Treats the alignment objective through prospect theory: instead of pairs, it consumes **unpaired binary** "this output is desirable / undesirable" signals, applying a Kahneman–Tversky utility (concave gains, convex losses) around a reference utility. Matches DPO from 1B to 30B parameters. Practical win: you can mine signals from thumbs-up/down logs without pairwise comparisons.

### 2.4 ORPO — Odds-Ratio Preference Optimization
Hong et al. 2024, [arXiv:2403.07691](https://arxiv.org/abs/2403.07691), EMNLP 2024. Monolithic: combines SFT NLL with a log-odds-ratio penalty between chosen and rejected — no reference model, single stage. On UltraFeedback alone, Phi-2/Llama-2-7B/Mistral-7B + ORPO surpass much larger Llama-2-Chat/Zephyr. **Shines** when you want to fold preference learning into the same gradient step as SFT.

### 2.5 SimPO — Simple Preference Optimization
Meng, Xia, Chen 2024, [arXiv:2405.14734](https://arxiv.org/abs/2405.14734), NeurIPS 2024. Reference-free implicit reward = **average log-prob** $\frac1{|y|}\log\pi_\theta(y|x)$, plus a target margin $\gamma$:
$$\mathcal L_{SimPO} = -\log\sigma\!\Big(\tfrac\beta{|y_w|}\log\pi_\theta(y_w|x) - \tfrac\beta{|y_l|}\log\pi_\theta(y_l|x) - \gamma\Big).$$
Beats DPO by 6.4 pts on AlpacaEval-2; Gemma-2-9B-it + SimPO hits 72.4 % LC win rate. Length-normalization is the key trick — also its risk (can under-train short answers).

### 2.6 CPO — Contrastive Preference Optimization
Xu et al. 2024 (machine translation), [arXiv:2401.08417](https://arxiv.org/abs/2401.08417). Adds an SFT-style NLL regularizer to a contrastive log-ratio loss and **removes** the reference model — halves memory and forward cost vs DPO. Strong default for MT and any domain where you want SFT + alignment fused.

### 2.7 sDPO — stepwise DPO
Kim et al. 2024. Splits preference data into chunks; after each chunk, the **reference model is replaced by the freshly trained policy**. Mitigates DPO's reference drift and is a stepping stone to fully iterative DPO.

### 2.8 Sigmoid vs hinge variants
DPO uses a *sigmoid* (logistic) on the margin; the *hinge* variant (Slic-HF, Zhao 2023) clips it. IPO's squared loss is the third member. The 2024–2025 unification papers ([Rainbow PO arXiv:2410.04203](https://arxiv.org/html/2410.04203), [DPO survey arXiv:2410.15595](https://arxiv.org/html/2410.15595v3)) treat all of these as choices of the link function in $\Psi$PO.

**Reference impls.** `trl.DPOTrainer`, `KTOTrainer`, `ORPOTrainer`, `CPOTrainer`. SimPO available via `DPOTrainer(loss_type="simpo")` and the upstream [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO) repo.

**TRL/LoRA recipe (Qwen2.5-0.5B, 24 GB):** LoRA r=16, α=32, target=`["q_proj","k_proj","v_proj","o_proj"]`; batch 4, grad-accum 8; β=0.1 (DPO/IPO), β=2.0 γ=1.0 (SimPO); lr 5e-6 for full FT, 1e-4 for LoRA; one epoch over 60 k pairs ≈ 2 h.

**Failure modes (all DPO variants).** Distribution shift; "reward hacking via likelihood displacement" (Razin et al. 2024); preference dataset noise amplification; collapse of diversity. Iterative / online DPO (§6) addresses most of these.

**Curriculum slot:** weeks 6–8.

---

## 3. Critic-free policy gradients: GRPO, RLOO, REINFORCE, ReMax, VinePPO

### 3.1 GRPO — Group Relative Policy Optimization
Introduced in DeepSeekMath, Shao et al. 2024, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300). For each prompt $x$ sample $G$ outputs $\{y_i\}$, score each with reward $r_i$, and use the **group-relative normalized advantage**
$$\hat A_i = (r_i - \mathrm{mean}(r))/\mathrm{std}(r)$$
broadcast to every token, plugged into a PPO-style clipped surrogate **without a value network**, with a KL term to the reference. Eliminates the critic (≈ halves memory) and is the engine behind DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948), Jan 2025).

**Shines** on verifiable tasks (math, code) where many rollouts per prompt are cheap. **Fails** when all $G$ rollouts succeed or all fail (zero advantage, wasted compute) and exhibits a **length bias**: incorrect long answers are under-penalized because the per-token loss is divided by token count.

### 3.2 Dr.GRPO — bias-removed GRPO
Liu et al. 2025, "Understanding R1-Zero-Like Training: A Critical Perspective", [arXiv:2503.20783](https://arxiv.org/abs/2503.20783), COLM 2025. Drops both the **std-dev normalization** (introduces a difficulty bias toward easy prompts) and the **per-response length normalization** (causes length inflation on wrong answers). Equivalent to using $\hat A_i = r_i - \mathrm{mean}(r)$ and summing token losses. State-of-the-art math RL on Qwen2.5-Math-7B in 27 h on 8×A100.

### 3.3 DAPO — Decoupled-clip & Dynamic-sampling Policy Optimization
ByteDance Seed, Yu et al. 2025, [arXiv:2503.14476](https://arxiv.org/abs/2503.14476). Four upgrades on GRPO: (i) **Clip-Higher** — asymmetric clip $\varepsilon_{\text{low}}\neq\varepsilon_{\text{high}}$ to let low-prob exploration tokens get probability mass; (ii) **Dynamic Sampling** — over-sample and discard prompts whose group accuracy is 0 or 1; (iii) **Token-level policy gradient loss** — sum, don't average, over tokens (same fix as Dr.GRPO); (iv) **Overlong reward shaping** to handle truncation. Hits 50 on AIME24 with Qwen2.5-32B at half the steps of R1-Zero; the strongest open recipe as of mid-2025. Implemented in [verl](https://verl.readthedocs.io/en/latest/algo/dapo.html).

### 3.4 RLOO — REINFORCE Leave-One-Out
Ahmadian et al. 2024, "Back to Basics", [arXiv:2402.14740](https://arxiv.org/abs/2402.14740). Sample $k$ completions per prompt and use the **average of the other $k-1$** as a per-sample baseline; treat the whole completion as a single action with a sequence-level reward. Outperforms a heavily tuned PPO at 6.9 B params, 3× faster, 70 % less RAM. Available as `trl.RLOOTrainer`.

### 3.5 ReMax
Li et al. 2023/2024, [arXiv:2310.10505](https://arxiv.org/abs/2310.10505). Uses the **greedy decode** as a baseline (a kind of pseudo-deterministic reference reward) — removes the critic and reduces variance using the deterministic-transition / trajectory-reward structure of RLHF. Practically simpler than RLOO (one extra forward, no value net).

### 3.6 VinePPO — better credit assignment with MC vines
Kazemnejad et al. 2024, [arXiv:2410.01679](https://arxiv.org/abs/2410.01679). For each prefix in a rollout, branch out $K$ short MC rollouts to estimate $V(s)$ on the fly — eliminates the learned critic, gives **dense per-step credit** for reasoning chains. Achieves PPO's peak with 9× fewer iterations, 3× less wall-clock on MATH. The leading "let's actually credit-assign" alternative to GRPO.

**TRL recipe (GRPO on SmolLM2-360M-Instruct or Qwen2.5-0.5B, 24 GB).** `GRPOTrainer` with `num_generations=8`, `max_completion_length=512`, `temperature=0.9`, `learning_rate=1e-6`, `beta=0.04` (KL), LoRA r=32 α=64 on all linear layers. Reward = exact-match on GSM8K. Expect ~1500 prompts/h, AIME-style benchmarks need 24–48 h. See [HF cookbook: GRPO with TRL](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl) and [SmolLM-GRPO blog](https://huggingface.co/blog/prithivMLmods/smollm-grpo-ft).

**Curriculum slot:** weeks 11–13.

---

## 4. RLVR — Reinforcement Learning with Verifiable Rewards

**Idea.** Replace the learned reward model with a **programmatic verifier** that returns 1 iff the output is correct. Eliminates reward-model overoptimization at the cost of needing a verifiable domain.

### 4.1 The canonical models
- **DeepSeek-R1-Zero / R1** — Guo et al. 2025, [arXiv:2501.12948](https://arxiv.org/abs/2501.12948). Pure RL from a base model (no SFT) with rule rewards (correctness + formatting); emergent long CoT, "aha moments", self-reflection. R1 then adds a cold-start SFT and multi-stage RL.
- **Kimi K1.5** — Moonshot AI 2025, [arXiv:2501.12599](https://arxiv.org/pdf/2501.12599). RL with long-context (128 k) and multimodal verifiable rewards; introduces length penalties to prevent overthinking and a partial-rollouts technique.
- **Tülu-3** — Lambert et al. 2024, [arXiv:2411.15124](https://arxiv.org/abs/2411.15124). First fully open RLVR recipe at scale (SFT → DPO → RLVR with PPO on math/IF tasks). Open dataset + eval suite.
- **Open-Reasoner-Zero** — Hu et al. 2025, [arXiv:2503.24290](https://arxiv.org/abs/2503.24290). Minimalist: vanilla PPO, GAE λ=γ=1, **no KL penalty**, rule rewards, on the base Qwen2.5-32B. Beats DeepSeek-R1-Zero-Qwen-32B in 1/10 the steps. Demonstrates KL regularization is dispensable on verifiable tasks.
- **DAPO / Dr.GRPO** — see §3, the algorithmic side of RLVR.

### 4.2 Verifier design
- **Math.** SymPy / latex2sympy equivalence checks; numeric tolerance for floats; format guards (`\boxed{}`).
- **Code.** Hidden unit tests; sandbox execution; partial credit by test pass rate.
- **Tool-use / agents.** Final-state equality; goal predicates; trajectory cost (negative reward for tokens / API calls).
- **Open-ended.** Multiple-choice reformulation ([arXiv:2511.02463](https://arxiv.org/pdf/2511.02463)), or **generative verifier** prompting a separate judge LLM.

**Failure modes.** Spec gaming the verifier (e.g. `assert True`), brittle math regex, "format reward" hacking, distribution narrowing (the policy forgets non-verifiable skills). The standard mitigation is mixing RLVR with a small fraction of preference data and a KL anchor; Tülu-3 mixes RLVR with DPO.

**Curriculum slot:** weeks 14–16.

---

## 5. Process Reward Models vs Outcome Reward Models

| | ORM | PRM |
|---|---|---|
| Signal | final answer only | per reasoning step |
| Data | answer key | step labels (human or auto) |
| Use | RL reward / BoN reranker | dense RL reward / search guide |
| Cost | cheap | expensive |

- **PRM800K** — Lightman et al. 2023, [arXiv:2305.20050](https://arxiv.org/abs/2305.20050). 800 K human step labels on MATH; first big PRM dataset. Showed PRMs beat ORMs at BoN re-ranking.
- **Math-Shepherd** — Wang et al. 2023/2024, [arXiv:2312.08935](https://arxiv.org/pdf/2312.08935). **Automatic** step labels via MC rollouts from each step: a step is "good" iff its completions disproportionately reach the right answer.
- **OmegaPRM** — Luo et al. 2024, [arXiv:2406.06592](https://arxiv.org/pdf/2406.06592). MCTS-based automatic PRM data generation; far more efficient than vanilla MC. Drove the math gains in DeepMind's Gemini math reasoning.
- **Generative Reward Models (GenRM)** — Zhang et al. 2024, [arXiv:2410.12832](https://arxiv.org/html/2410.12832v1). Reward is the *probability of token "yes"* under a judge LLM prompted with a rubric. Composes naturally with chain-of-thought judgment.
- **Rubric-RM / OpenRubrics** — [arXiv:2510.07743](https://arxiv.org/html/2510.07743v1), 2025. Synthetic rubric generation; +6.8 % over scalar RMs.
- **Survey** — [arXiv:2510.08049](https://arxiv.org/pdf/2510.08049), 2025, "A Survey of Process Reward Models".

**When PRMs shine.** Long CoT search/MCTS, where you want to prune branches early. **When ORMs win.** RLVR with cheap exact-match — the marginal gain of step labels is small and they add reward-hacking surface.

**Curriculum slot:** weeks 17–18.

---

## 6. Iterative / online DPO, self-rewarding, Constitutional AI, RLAIF, weak-to-strong

### 6.1 Iterative & online DPO
- **Online DPO with fast-slow chasing** — [arXiv:2406.05534](https://arxiv.org/html/2406.05534v1), 2024.
- **Iterative DPO comprehensive study** — [arXiv:2503.12854](https://arxiv.org/abs/2503.12854), 2025.
- **Iterative length-regularized DPO** — Liu et al. 2024, [arXiv:2406.11817](https://arxiv.org/pdf/2406.11817). Pushes Llama-3-8B to GPT-4-level on AlpacaEval.
- **TR-DPO** (Trust-Region DPO) — dynamically updates the reference policy to avoid overoptimization.

Pattern: sample → score with RM (or self-judge) → form preference pairs → DPO → repeat. Closes the offline-distribution-shift gap of vanilla DPO at modest cost.

### 6.2 Self-Rewarding Language Models
Yuan et al. 2024, [arXiv:2401.10020](https://arxiv.org/abs/2401.10020). The same LLM plays generator, judge, and learner: LLM-as-a-judge prompting produces rewards, DPO updates the policy, repeat. Llama-2-70B after 3 iterations beats Claude 2 / Gemini Pro / GPT-4-0613 on AlpacaEval 2. Failure mode: reward-model drift / self-confirmation. Process-based variant [arXiv:2503.03746](https://arxiv.org/html/2503.03746v1).

### 6.3 Constitutional AI / RLAIF
Bai et al. 2022, "Constitutional AI: Harmlessness from AI Feedback", [arXiv:2212.08073](https://arxiv.org/pdf/2212.08073). Two-stage: (i) SL phase — model critiques and revises its own outputs against a "constitution"; (ii) RL phase — an AI preference model trained on AI-generated pairwise comparisons replaces the human RM (this is the original RLAIF). Backbone of Claude's harmlessness training. Modern extensions: principled rubric-RLAIF, deliberative alignment (OpenAI 2024).

### 6.4 Weak-to-strong & scalable oversight
- Burns et al. 2023 (OpenAI Superalignment), "Weak-to-Strong Generalization", [arXiv:2312.09390](https://arxiv.org/pdf/2312.09390). Strong models fine-tuned on weak supervisors' labels can *exceed* the supervisor — early evidence that scalable oversight may work. Auxiliary-confidence loss boosts the recovery rate.
- Quantifying-the-gain follow-up [arXiv:2405.15116](https://arxiv.org/pdf/2405.15116).
- Related agenda: debate, market-making, recursive reward modeling, iterated amplification — all part of the "how do we align superhuman models" research direction.

**Curriculum slot:** weeks 19–20.

---

## 7. Reward hacking, KL control, length bias, Goodhart

- **RM over-optimization scaling laws** — Gao, Schulman, Hilton 2023, [Gao et al.](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf). Proxy-vs-gold reward gap grows monotonically with KL distance; analytic laws.
- **Catastrophic Goodhart** — [arXiv:2407.14503](https://arxiv.org/html/2407.14503v2), 2024. Shows KL regularization is *insufficient* under heavy-tailed reward misspecification.
- **Length bias.** DPO, PPO and GRPO all systematically inflate response length. Mitigations: SimPO's length normalization; Dr.GRPO's removal of `/|y|` divisor; iterative length-regularized DPO (Liu 2024); explicit length-penalty terms in DAPO and Kimi K1.5.
- **Information-theoretic RM** — [arXiv:2402.09345](https://arxiv.org/html/2402.09345v3), NeurIPS 2024. Regularize RM with mutual information.
- **Robust RM (RRM)** — ICLR 2025; data-augmentation-based reward hacking mitigation.
- **Shrinkage baselines for RLVR** — [arXiv:2511.03710](https://arxiv.org/html/2511.03710), 2025. James–Stein-style shrinkage on the group baseline reduces variance dramatically.
- **Noise-corrected GRPO** — [arXiv:2510.18924](https://arxiv.org/pdf/2510.18924), 2025. Unbiased gradient estimator under noisy reward labels.

**Practical KL hygiene.** For preference RL: $\beta\in[0.05,0.2]$. For RLVR: many recent recipes (Open-Reasoner-Zero, R1-Zero) drop the KL term entirely on the base model; if you keep it, $\beta\sim 0.01$–$0.04$ and clip its contribution.

**Curriculum slot:** weeks 21–22 (after main methods are internalized).

---

## 8. Inference-time RL and search

- **Best-of-N (BoN)** — sample $N$, rerank with an RM/verifier, return the top one. Snell et al. 2024 ([arXiv:2408.03314](https://arxiv.org/abs/2408.03314)) show that for many tasks, scaling BoN can outperform scaling pretraining FLOPs.
- **BoN-aware fine-tuning** — Chow et al. 2024, [arXiv:2412.15287](https://arxiv.org/pdf/2412.15287). Trains the policy to be *optimal under BoN selection*, not just under greedy decode.
- **Is BoN the best of them?** — [arXiv:2503.21878](https://arxiv.org/abs/2503.21878), 2025. Coverage/scaling theory of BoN; near-optimal under broad conditions.
- **MCTS-augmented reasoning** — ReST-MCTS*, AlphaLLM, rStar-Math, MCTS-DPO. The PRM guides node value, the LLM generates moves. Trade-off: orders-of-magnitude more compute, big jumps on hard math.
- **AlphaProof / AlphaGeometry 2** — DeepMind, 2024–2025; [Nature, Nov 2025](https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/). AlphaProof: Lean-formal RL loop — generator proposes tactics, Lean verifies, RL on success; massive test-time MCTS search. IMO silver-medal level. AlphaGeometry 2: symbolic-deductive engine + LLM, also MCTS. The clearest demonstration that **RL + formal verifier + tree search > scale** in narrow but hard domains.

**Curriculum slot:** weeks 22–23.

---

## 9. Practical TRL recipes on a single consumer GPU

### 9.1 Hardware budget (24 GB VRAM)
| Model | Method | LoRA? | Effective batch | Notes |
|---|---|---|---|---|
| Pythia-160M | PPO / RLOO | full | 32 | toy reasoning, fast iteration |
| TinyLlama-1.1B | DPO/SimPO/ORPO | r=16 | 16 | classic preference RL playground |
| SmolLM2-360M-Instruct | GRPO/Dr.GRPO | r=32 | 8 | recommended GRPO entry point |
| Qwen2.5-0.5B-Instruct | GRPO + RLVR | r=32 | 8 | enough capacity for GSM8K signal |
| Qwen2.5-1.5B-Instruct | GRPO + RLVR | r=64 | 4, grad-accum 4 | tight; use vLLM for rollouts |
| Qwen2.5-3B / Llama-3.2-3B | DPO | r=16, 4-bit (QLoRA) | 2, grad-accum 8 | bitsandbytes nf4 |

### 9.2 Sane defaults
- **LoRA.** r=16–64, α=2r, dropout=0.05, target all linear projections (`q,k,v,o,gate,up,down`). The 2025 "LoRA Without Regret" study (Thinking Machines) shows LoRA matches full FT at 67 % compute when applied to all linear layers with appropriate rank.
- **Optimizer.** AdamW, lr 1e-6 (RL), 5e-6 (DPO full), 1e-4 (DPO LoRA); cosine schedule with 3 % warmup; gradient clip 1.0.
- **KL coefficient.** DPO/SimPO β=0.1; GRPO β=0.04 (or 0); PPO β starts at 0.02, adaptive.
- **Rollouts.** TRL+vLLM integration since `trl>=0.13` — keep the generator process separate from the trainer to hide rollout latency.
- **Numerics.** bf16 for the actor; fp32 for the policy log-probs in the loss (numerical stability of the importance ratio); flash-attention 2/3.
- **Generations per prompt (GRPO).** 4–8 on a single GPU; 16+ on multi-GPU.

### 9.3 Reference repos & docs
- [`huggingface/trl`](https://github.com/huggingface/trl) — DPO, KTO, ORPO, CPO, SimPO, PPOv2, RLOO, GRPO.
- [`huggingface/peft`](https://github.com/huggingface/peft) — LoRA / QLoRA / DoRA.
- [`OpenRLHF`](https://github.com/OpenRLHF/OpenRLHF), [`verl`](https://github.com/volcengine/verl), [`Open-Reasoner-Zero`](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) — production-scale RL.
- [HF GRPO cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl); [SmolLM-GRPO blog](https://huggingface.co/blog/prithivMLmods/smollm-grpo-ft); [TRL GRPO docs](https://huggingface.co/docs/trl/grpo_trainer).
- [`princeton-nlp/SimPO`](https://github.com/princeton-nlp/SimPO); [allenai/Tulu-3](https://github.com/allenai/open-instruct).
- [RLHF Book — Lambert](https://rlhfbook.com/) — the best free reference text as of 2026.
- [awesome-RLVR](https://github.com/opendilab/awesome-RLVR) — living list of RLVR papers.

**Curriculum slot:** woven through weeks 6–24 as the hands-on track.

---

## 10. Suggested 24-week curriculum skeleton

| Weeks | Theme | Methods/papers |
|---|---|---|
| 1–2 | RL foundations | MDPs, Bellman, tabular Q/SARSA |
| 3–4 | Deep RL | DQN, A2C, PPO, GAE |
| 5 | Bandits & exploration | UCB, Thompson, KL-control |
| 6–8 | Offline preference RL | DPO, IPO, KTO, ORPO, SimPO, CPO, sDPO |
| 9–10 | Classical RLHF | InstructGPT, PPO-RLHF, reward modeling |
| 11–13 | Critic-free PG | RLOO, ReMax, VinePPO, GRPO, Dr.GRPO, DAPO |
| 14–16 | RLVR | DeepSeek-R1, Kimi K1.5, Tülu-3, Open-Reasoner-Zero; verifier design |
| 17–18 | Reward modeling | PRM800K, Math-Shepherd, OmegaPRM, GenRM, Rubric-RM |
| 19–20 | Iterative & scalable oversight | online/iterative DPO, self-rewarding LMs, CAI/RLAIF, weak-to-strong |
| 21–22 | Reward hacking & robustness | Gao scaling laws, catastrophic Goodhart, length bias mitigation, shrinkage/noise-corrected GRPO |
| 22–23 | Inference-time RL | best-of-N, BoN-aware FT, MCTS, AlphaProof / AlphaGeometry |
| 24 | Capstone | end-to-end RLVR run on Qwen2.5-0.5B with GRPO+DAPO tricks on GSM8K/MATH |

---

## 11. Key reading list (chronological, citation-style)

- Ouyang et al., *InstructGPT*, [arXiv:2203.02155](https://arxiv.org/abs/2203.02155), Mar 2022.
- Bai et al., *Constitutional AI*, [arXiv:2212.08073](https://arxiv.org/abs/2212.08073), Dec 2022.
- Gao, Schulman, Hilton, *Scaling Laws for Reward Model Overoptimization*, ICML 2023.
- Lightman et al., *Let's Verify Step by Step* (PRM800K), [arXiv:2305.20050](https://arxiv.org/abs/2305.20050), May 2023.
- Rafailov et al., *DPO*, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290), May 2023.
- Li et al., *ReMax*, [arXiv:2310.10505](https://arxiv.org/abs/2310.10505), Oct 2023.
- Azar et al., *A General Theoretical Paradigm / IPO*, [arXiv:2310.12036](https://arxiv.org/abs/2310.12036), Oct 2023.
- Yuan et al., *Self-Rewarding Language Models*, [arXiv:2401.10020](https://arxiv.org/abs/2401.10020), Jan 2024.
- Xu et al., *CPO*, [arXiv:2401.08417](https://arxiv.org/abs/2401.08417), Jan 2024.
- Ethayarajh et al., *KTO*, [arXiv:2402.01306](https://arxiv.org/abs/2402.01306), Feb 2024.
- Shao et al., *DeepSeekMath / GRPO*, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300), Feb 2024.
- Ahmadian et al., *Back to Basics / RLOO*, [arXiv:2402.14740](https://arxiv.org/abs/2402.14740), Feb 2024.
- Hong et al., *ORPO*, [arXiv:2403.07691](https://arxiv.org/abs/2403.07691), Mar 2024.
- Meng, Xia, Chen, *SimPO*, [arXiv:2405.14734](https://arxiv.org/abs/2405.14734), May 2024.
- Luo et al., *OmegaPRM*, [arXiv:2406.06592](https://arxiv.org/abs/2406.06592), Jun 2024.
- Snell et al., *Scaling LLM Test-Time Compute…*, [arXiv:2408.03314](https://arxiv.org/abs/2408.03314), Aug 2024.
- Kazemnejad et al., *VinePPO*, [arXiv:2410.01679](https://arxiv.org/abs/2410.01679), Oct 2024.
- Zhang et al., *Generative Reward Models*, [arXiv:2410.12832](https://arxiv.org/abs/2410.12832), Oct 2024.
- Lambert et al., *Tülu 3*, [arXiv:2411.15124](https://arxiv.org/abs/2411.15124), Nov 2024.
- Chow et al., *Inference-Aware Fine-Tuning for BoN*, [arXiv:2412.15287](https://arxiv.org/abs/2412.15287), Dec 2024.
- DeepSeek-AI, *DeepSeek-R1*, [arXiv:2501.12948](https://arxiv.org/abs/2501.12948), Jan 2025.
- Moonshot AI, *Kimi K1.5*, [arXiv:2501.12599](https://arxiv.org/abs/2501.12599), Jan 2025.
- Yu et al., *DAPO*, [arXiv:2503.14476](https://arxiv.org/abs/2503.14476), Mar 2025.
- Liu et al., *Understanding R1-Zero-Like Training / Dr.GRPO*, [arXiv:2503.20783](https://arxiv.org/abs/2503.20783), Mar 2025.
- Hu et al., *Open-Reasoner-Zero*, [arXiv:2503.24290](https://arxiv.org/abs/2503.24290), Mar 2025.
- ByteDance, *VAPO*, [arXiv:2504.05118](https://arxiv.org/abs/2504.05118), Apr 2025.
- "Shrinkage Baselines for RLVR", [arXiv:2511.03710](https://arxiv.org/abs/2511.03710), Nov 2025.
- DeepMind, *AlphaProof* (Nature, Nov 2025); [blog](https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/).
- *A Survey of Process Reward Models*, [arXiv:2510.08049](https://arxiv.org/abs/2510.08049), Oct 2025.
- *A Comprehensive Survey of Direct Preference Optimization*, [arXiv:2410.15595](https://arxiv.org/abs/2410.15595), 2024–2025.

---

*End of dossier. Suggested next step: pair this with the 24-week schedule in the repo root and start week 1 on tabular MDPs while pre-downloading Qwen2.5-0.5B-Instruct and the GSM8K/MATH splits for the week-14 RLVR capstone.*
