# Bandits + Matching RL — Applied SOTA Dossier

*Compiled May 2026. Applied targets: (1) synthetic insurance insurance lead↔broker matching, (2) ATS candidate↔job matching. The unifying problem: a partially observed, delayed-feedback decision system that consumes mostly unstructured input (CVs, job descriptions, quote forms, broker books of business) and must predict, then act on, the **alignment** of two heterogeneous entities. This dossier walks the literature from classical stochastic bandits up to 2024–2026 off-policy CLTR and fairness-aware two-sided matching, attaching curriculum slots for a 24-week self-study plan.*

---

## 1. Stochastic Bandits — the irreducible core

**Intuition.** A K-armed bandit is the smallest non-trivial decision problem: pull one of K arms, observe a noisy reward, repeat. Regret — the gap to the oracle that always pulls the best arm — is the universal currency. Every matching algorithm we will build is, under the hood, a structured bandit that exploits side information to share statistics between arms.

**UCB1** (Auer, Cesa-Bianchi, Fischer 2002). Pick arm `a` maximising `μ̂_a + sqrt(2 log t / n_a)`. The bonus is a Hoeffding upper confidence bound; regret is `O(sum_a (log T)/Δ_a)` and matches the Lai–Robbins lower bound up to constants. The key intuition: optimism in the face of uncertainty (OFU) — be optimistic exactly proportional to your ignorance.

**KL-UCB** (Garivier & Cappé 2011, Maillard et al.). Replaces Hoeffding with the tight KL bound: arm `a` is feasible if `n_a · KL(μ̂_a, q) ≤ log t + c log log t`. Constant-optimal for Bernoulli rewards, which matters for binary conversion signals (lead → bound policy; candidate → hire) where Hoeffding is loose near 0 or 1.

**Thompson Sampling** (Thompson 1933; revived Chapelle & Li 2011; regret analysis Agrawal & Goyal 2012, Russo & Van Roy 2014). Maintain a posterior over reward parameters; at each step sample once, act greedily w.r.t. the sample. Bayesian, easy to compose with priors, robust under delay, and empirically dominant in industry A/B comparisons. Russo & Van Roy's *information-theoretic regret* analysis gives the cleanest unified bound `sqrt(Γ · H · T)` where Γ is information ratio and H is entropy of the optimal action.

**MOSS / MOSS-anytime** (Audibert & Bubeck 2009; Ménard & Garivier 2017). Minimax-optimal: `O(sqrt(KT))` worst-case regret without a `log T` factor. KL-UCB++ and OCUCB close further constants. Practically less important than TS but theoretically the right reference when your environment is adversarial-ish (e.g., broker behaviour during a hard market).

> **Curriculum slot — Week 1.** Implement UCB1, KL-UCB, TS, MOSS on a Bernoulli testbed; reproduce Lai–Robbins lower bound empirically. Read Lattimore & Szepesvári Ch. 7–10.

---

## 2. Contextual Bandits — where matching actually lives

A lead arrives with features `x` (postcode, vehicle, prior claims, free-text description). A candidate arrives with embedding `φ(CV)`. The arm is a broker / job opening. Reward is bind / hire / placement. This is the contextual bandit.

**LinUCB** (Li, Chu, Langford, Schapire 2010, WWW — the Yahoo! news article paper). Per-arm linear model `r = x^T θ_a + ε`. Confidence ellipsoid `||θ_a - θ̂_a||_{A_a} ≤ α`. Act by `argmax_a x^T θ̂_a + α sqrt(x^T A_a^{-1} x)`. Regret `Õ(d sqrt(T))`. Disjoint vs. hybrid variants — hybrid shares a global feature across arms and is what you want when most arms (brokers, jobs) have few observations.

**LinTS** (Agrawal & Goyal 2013). Gaussian posterior on `θ_a`, sample, act greedily. `Õ(d^{3/2} sqrt(T))` frequentist regret; empirically as good as LinUCB and easier to tune (one variance scalar vs. confidence radius).

**Neural bandits.** The linearity assumption breaks for CV ↔ JD text. Two routes:
- **NeuralUCB** (Zhou, Li, Gu 2020): UCB with the NTK-based confidence ellipsoid over network outputs. Theoretical `Õ(d̃ sqrt(T))` with effective dimension d̃.
- **NeuralTS** (Zhang et al. 2021): Bayesian-linear head on top of a deep encoder, sample from a Laplace / NTK posterior.
- **Bootstrapped TS / Deep TS** (Osband et al. 2016, 2018; Riquelme et al. *Deep Bayesian Bandits Showdown* 2018). Train K bootstrap heads on resampled replay; act greedily w.r.t. a randomly chosen head. The Showdown paper still stands in 2026: bootstrap + ε-greedy fine-tuning is the most reliable production baseline; full Bayesian neural nets are fragile.

For synthetic insurance and ATS the right stack is: a frozen text encoder (LLM-distilled cross-encoder), then a small Bayesian linear head per broker / requisition with LinTS or bootstrapped TS. This gives calibrated uncertainty where it matters (the last layer) without paying for full BNN inference.

> **Curriculum slot — Week 2–3.** LinUCB and LinTS on the Open Bandit Dataset (ZOZO). Then port the architecture to synthetic insurance's anonymised lead stream. Read Bietti, Agarwal, Langford *Contextual Bandit Bake-off* (2018, updated 2021).

---

## 3. Off-Policy Evaluation — the only way you'll ever measure these

You cannot A/B test every policy. You need to score a *new* policy `π` using logs from the *behaviour* policy `π_b`.

**IPS / IS** (Horvitz–Thompson 1952; Strehl et al. 2010). `V̂(π) = (1/n) Σ r_i · π(a_i|x_i)/π_b(a_i|x_i)`. Unbiased but variance explodes when π and π_b disagree. Requires logged propensities — you must log `π_b(a|x)`, not just `a`.

**SNIPS** (Swaminathan & Joachims 2015b). Self-normalised IPS: divide by `Σ w_i`. Biased but consistent; dramatically lower variance; the default in production OPE.

**Doubly Robust (DR)** (Dudík, Langford, Li 2011; Jiang & Li 2016 for RL). `V̂_DR = (1/n) Σ [Q̂(x,a) + w_i (r_i - Q̂(x_i, a_i))]`. Unbiased if *either* the propensity *or* the reward model `Q̂` is correct. Critical in matching where Q̂ is a strong text-similarity prior.

**Switch / SWITCH-DR** (Wang, Agarwal, Dudík 2017). Use DR when `w_i ≤ τ`, fall back to the direct method otherwise. Bounds variance at the cost of small bias. Empirically robust; what you want for tail brokers / rare requisitions.

**BanditNet / counterfactual risk minimisation** (Joachims, Swaminathan, de Rijke 2018). Training a deep policy directly against the SNIPS estimator with self-normalisation and propensity overfitting controls. Re-derives the CRM principle (Swaminathan & Joachims 2015a) at deep-net scale.

**OPE for ranking / slates**: 
- Slate-IPS (Swaminathan et al. 2017): assumes linear decomposition of rewards across slate positions.
- **PI / Pseudoinverse** estimator (Swaminathan et al. 2017 NeurIPS) — handles combinatorial action spaces.
- **OPE for ranking 2023–2025**: Saito & Joachims *Off-Policy Evaluation for Large Action Spaces via Embeddings* (ICML 2022, extended 2024) — the **MIPS** estimator. This is the most important practical advance for ATS / synthetic insurance: when you have thousands of brokers or jobs, embed the actions and do IPS over the embedding space. Variance drops from O(K) to O(d_embed).

> **Curriculum slot — Week 4–5.** Implement IPS, SNIPS, DR, SWITCH, MIPS on Open Bandit Pipeline (Saito et al.). Reproduce the MIPS variance plot.

---

## 4. Counterfactual Learning to Rank (CLTR)

Ranking is bandit feedback with position bias: users only click what they see, and they see only what we ranked highly.

**IPS-LTR / Unbiased LTR** (Joachims, Swaminathan, Schnabel WSDM 2017). Inverse propensity weighting by *position bias* `p_k` (probability a user examines position k). Reward each click by `1/p_k`. Estimate `p_k` via online interventions (RandTopN swaps) or EM (Wang et al. 2018).

**SNIPS-LTR** and **Dual Learning Algorithm** (Ai et al. SIGIR 2018) jointly learn ranker and propensities.

**Off-policy CLTR — Joachims line continued**: 
- Oosterhuis & de Rijke 2020–2022: counterfactual evaluation for top-k truncated rankings.
- Vardasbi, Oosterhuis, de Rijke 2020: cascade-based CLTR.
- **2024–2026 work**: Oosterhuis *Doubly-Robust Estimation for Unbiased LTR* (SIGIR 2023); Gupta et al. *Safe Deployment for Counterfactual LTR* (WSDM 2024); Jagerman et al. on logarithmic-regret CLTR (2024); recent two-stage CLTR with LLM-judged labels (Yan et al. KDD 2025) that blends offline LLM relevance with logged clicks via DR.

For ATS: position bias is *enormous* (recruiters scan ~10 candidates / req). Logging propensities for the shortlist policy is the single highest-ROI engineering task — without it, every offline metric is biased.

> **Curriculum slot — Week 6.** Reproduce Joachims-Swaminathan-Schnabel WSDM 2017 on the MSLR-WEB10K + simulated clicks pipeline. Then layer DR-CLTR.

---

## 5. RL for Recommendation

When actions compound (one match changes the next), you graduate from bandits to full RL.

**SlateQ** (Ie et al. IJCAI 2019, Google). Decomposes the Q-function over a slate `A = (a_1,...,a_k)` under a *single-choice* user model: `Q(s, A) = Σ_a P(a chosen | A) · Q(s, a)`. Makes slate RL tractable. The matching analogue: a recruiter session is a slate; a broker's daily inbox is a slate.

**REINFORCE for Recs / Top-K off-policy correction** (Chen et al. WSDM 2019, YouTube). Policy gradient with importance-weight clipping and top-K correction `λ_K(a|s) = K(1 - π(a|s))^{K-1}`. Live-deployed; one of the few peer-reviewed industrial RL-recsys papers. Variance reduction via λ_K-corrected REINFORCE.

**Action embeddings**: when |A| is huge (millions of jobs), parameterise the policy as `π(a|s) ∝ exp(f(s)^T g(a))` with shared embeddings. Wolpertinger (Dulac-Arnold et al. 2015) — k-NN in embedding space then Q-rank. In 2025, the dominant pattern is two-tower retrieval + cross-encoder re-ranking + bandit / RL exploration on the top-N.

**Conservative recsys RL** (CQL — Kumar et al. 2020; BCQ — Fujimoto et al. 2019; IQL — Kostrikov et al. 2022). Offline RL with explicit conservatism penalties; essential in matching where ε-greedy exploration on a real recruiter is unacceptable. 2024–2025 work: *Decision-Transformer for recsys* (Zhao et al. 2023) and *Diffusion policies for recommendation* (Wang et al. 2024) — promising but still less robust than CQL+two-tower in production.

> **Curriculum slot — Week 7–9.** SlateQ on RecSim-NG; REINFORCE-Top-K with off-policy correction. Then CQL on a logged synthetic insurance dataset.

---

## 6. Two-Sided Matching Markets

Both synthetic insurance and ATS are *two-sided*: brokers reject leads; candidates reject jobs. A unilateral recommender loses to mechanism-design-aware policies.

**Foundations**: Gale–Shapley deferred acceptance (1962); Roth's medical residency redesign (1984, 2003); Hatfield–Milgrom matching with contracts (2005). Stability dominates pure efficiency in repeated markets because unstable matches are unwound by the agents themselves.

**RL on matching markets**:
- *Learning to Match* (Hu et al. KDD 2018) — Didi-Chuxing rider/driver matching as an MDP with bipartite assignment.
- *Spatio-Temporal Multi-Agent RL for Dispatch* (Li et al. 2019, KDD).
- *Min-Cost Flow + RL for ride-hailing* (Qin et al. 2020).
- **Two-sided bandits**: Liu, Mania, Jordan *Competing Bandits in Matching Markets* (AISTATS 2020) — both sides explore, regret bounds w.r.t. stable matches; Sankararaman, Basu, Sankagiri (NeurIPS 2021); *Decentralised Two-Sided Bandit Learning* (Basu et al. 2021).
- **Fair-matching 2024–2026**: Patro et al. *FairRec* (WWW 2020) — two-sided exposure fairness; Biswas et al. on group-fair matching (2023); Wang & Joachims *Fairness in Ranking under Uncertainty* (NeurIPS 2021); recent *Stable + Fair Matching under Bandit Feedback* (Cen, Shah 2024); Jagadeesan, Garg, Steinhardt *Supply-side Equilibria in Recommender Systems* (NeurIPS 2024) — recommender pushes producers (brokers / employers) into equilibrium behaviours; the platform must model this.

**Mechanism-design considerations**: 
- Incentive compatibility: if brokers learn that down-weighting a lead increases future high-value allocations, they game it. Match synthetic insurance's allocation to be DSIC w.r.t. broker bid signals; Myerson-optimal where revenue matters.
- Repeated games: brokers/recruiters are long-lived; a one-shot CTR maximiser destroys supply.

> **Curriculum slot — Week 10–12.** Read Roth & Sotomayor; implement Gale–Shapley; reproduce Liu-Mania-Jordan on synthetic competing bandits.

---

## 7. Calibration — the bridge from score to decision

Matching scores get thresholded, ranked, fed into auctions, and reported to brokers. Miscalibration corrupts every downstream step.

- **Platt scaling** (Platt 1999): sigmoid fit on validation set. One-parameter, robust, low capacity.
- **Isotonic regression** (Zadrozny & Elkan 2002): non-parametric monotone; needs more data, exact on training set.
- **Beta calibration** (Kull, Filho, Flach 2017, AISTATS): three-parameter Beta-family; strictly more expressive than Platt; recommended default in 2024+ for binary classifiers.
- **Temperature scaling** (Guo et al. ICML 2017): single-parameter logit rescaling; the default for deep nets.
- **Deep calibration**: focal loss (Mukhoti et al. 2020), label smoothing (caution: hurts calibration on some tasks per Müller et al. 2019), *Mix-n-Match* (Zhang, Kailkhura, Han 2020). Recent: *Spline calibration* (Gupta et al. 2021), *Histogram binning with debiasing* (Kumar et al. 2019), *Calibration under distribution shift* (Ovadia et al. 2019, ongoing).
- **Calibration for ranking / matching specifically**: a calibrated `P(bind | lead, broker)` lets you optimise *expected revenue* across brokers; without it, ranking is correct in-distribution but absolute thresholds (auto-accept / auto-reject) are wrong.

> **Curriculum slot — Week 13.** Calibrate the synthetic insurance binding-probability model with Platt, isotonic, Beta; compute ECE, MCE, Brier; do reliability diagrams.

---

## 8. Fairness in Matching

Three families:
- **Demographic parity**: `P(match | A=0) = P(match | A=1)`. Easy to enforce, but distorts welfare.
- **Equalised odds / opportunity** (Hardt, Price, Srebro 2016): conditional on true qualification.
- **Individual fairness** (Dwork et al. 2012): similar individuals get similar treatment under a task-specific metric.

**Bandit-fairness**:
- Joseph et al. *Fairness in Learning: Classic and Contextual Bandits* (NeurIPS 2016) — never prefer a worse arm with positive probability w.r.t. a sensitive attribute.
- Patil et al. *Achieving Fairness in the Stochastic Multi-Armed Bandit Problem* (AAAI 2020).
- **Two-sided fairness 2023–2026**: producer + consumer fairness; *exposure fairness in ranking* (Singh & Joachims 2018); *FairCo* online controller (Morik et al. SIGIR 2020); recent *Fair Allocation in Two-Sided Markets under Uncertainty* (2024) and *Fair Off-Policy Learning* (Kallus et al. 2024).
- Regulatory pressure: EU AI Act (in force 2024, high-risk for ATS), UK FCA Consumer Duty (insurance). Calibration-by-group is now table-stakes.

For ATS: equalised opportunity by protected class on `P(hire | qualified)`; individual fairness across CVs that differ only in named entities (a known LLM failure mode — see Bertrand & Mullainathan 2004 echoed in LLM-resume audits 2024).

> **Curriculum slot — Week 14.** Audit the ATS shortlist model on a synthetic protected-attribute benchmark; deploy in-processing constraint (Agarwal et al. 2018 reductions) and post-processing (Hardt et al.).

---

## 9. Embedding-Based Retrieval + RL Re-Ranking

The 2025 canonical matching stack is a three-layer cascade:

1. **Retrieval (two-tower / dual encoder)**: `s(x, a) = f(x)^T g(a)`. ANN over millions of candidates. Trained with in-batch negatives + hard negative mining (Xiong et al. *ANCE* 2021), or contrastive (SimCSE, E5, BGE, ModernBERT, Snowflake-arctic-embed 2024). For CV↔JD: domain-adapted E5 / bge-m3 + LoRA on labelled hire pairs.
2. **Cross-encoder re-ranking**: full attention over `[x; a]`. Slow but accurate; ColBERT v2 / monoT5 / RankZephyr (2024) / RankGPT (Sun et al. 2023) — LLM-as-reranker is now SOTA but expensive.
3. **Policy layer**: contextual bandit / RL over the top-N from the re-ranker. Optimises long-term reward (bind, hire), exploration, and fairness constraints. This is where synthetic insurance / ATS add the most value over off-the-shelf retrieval.

For unstructured → aligned: the retrieval layer makes the unstructured CV / lead form addressable; the cross-encoder produces a strong supervised relevance signal; the bandit layer corrects for behavioural-policy bias and explores.

> **Curriculum slot — Week 15–17.** Train a two-tower CV↔JD encoder; cross-encoder re-ranker; integrate with the LinTS policy from Week 3.

---

## 10. Cold-Start

New broker, new requisition, new market segment — no history, no clicks.

- **Meta-learning bandits**: MAML-style priors (Finn et al. 2017); *Meta-Thompson Sampling* (Kveton et al. ICML 2021); *Meta-Learning Bandit Policies* (Cella et al. 2020).
- **Bayesian priors from hierarchy**: a new broker inherits the posterior of the *category* of brokers (motor / commercial / home), then updates. Empirical Bayes shrinkage à la James–Stein.
- **LLM priors**: zero-shot relevance from an LLM cross-encoder gives a non-trivial Q̂ for cold arms (and a DR baseline).
- **Curriculum / warm-up exploration**: forced exploration budget on new arms; Lattimore's *Bandits with Side Observations* and *graph-feedback bandits* let you propagate observations from similar arms.

> **Curriculum slot — Week 18.** Implement hierarchical TS where each new broker inherits a category prior; benchmark vs. flat TS on a cold-start simulator.

---

## 11. Delayed and Partial Feedback

The applied killer. synthetic insurance: bind decision in days, claims feedback in years. ATS: shortlist feedback in days, hire in weeks, retention in months.

- **Delayed feedback models** (Chapelle WWW 2014, Criteo): joint model of conversion *and* delay distribution; train against censored observations with EM / survival likelihood.
- *Counterfactual Delayed Reward* (Vernade et al. 2017, 2020): bandits with stochastic delays — UCB and TS analogues, regret depends on E[delay].
- *Bandits with partial monitoring* (Bartók et al. 2014): you observe a signal, not the reward; the matching analogue is broker "considered but didn't bind" — a partial signal that's not nothing.
- **Survival modelling** for time-to-bind / time-to-hire: Cox PH, DeepSurv (Katzman et al. 2018), DeepHit (Lee et al. 2018), and 2024 *Neural survival bandits* (Hu et al.). Treat conversion as a censored time-to-event problem; the reward at evaluation is the predicted survival-adjusted expectation.
- **Composite reward shaping**: short-term proxies (broker opens lead, recruiter requests interview) regressed against eventual hire / bind to construct a low-variance immediate reward. Calibration of the proxy → outcome map is essential.
- 2024–2026: *Off-Policy Evaluation under Delayed Rewards* (Saito et al. 2024); *Bandits with Stochastically Delayed and Aggregated Feedback* (Cesa-Bianchi et al. 2024).

> **Curriculum slot — Week 19–20.** Implement Chapelle-style delayed-feedback EM; integrate with the LinTS policy; simulate synthetic insurance's 90-day bind window with realistic censoring.

---

## 12. Production Case Studies (Public)

- **Netflix**: *Artwork personalisation* (Chandrashekar et al. 2017) — contextual bandit picking thumbnails. *Calibrated recommendations* (Steck RecSys 2018). Still the cleanest published contextual-bandit deployment.
- **Spotify**: *Reinforcement Learning for Slate-based Recommendation* (McInerney et al. RecSys 2018); *Where to Next?* podcast recsys with bandits (2020); ongoing work on contextual TS for home shelf personalisation.
- **YouTube**: Chen et al. WSDM 2019 (REINFORCE), Ie et al. 2019 (SlateQ), Wilhelm et al. 2018 (DPP-based diversity).
- **LinkedIn**: *XNLT* — economic-graph-aware ranking; *Recruiter Search* (Borisyuk et al. KDD 2017); *Two-tower-based candidate retrieval* (Geyik et al. 2019); *Fairness-aware ranking* (Geyik, Ambler, Kenthapadi KDD 2019) — the most relevant published ATS-side fairness work.
- **Indeed**: published two-tower job-search models (2020+), and 2023 work on calibrated CTR.
- **ZipRecruiter**: less public, but their *two-sided marketplace* and *bidirectional matching* posts describe a TS-style bandit for active-candidate ↔ job notification.
- **Insurance comparators (MoneySuperMarket, Compare the Market)**: limited public material, but the broker-allocation problem is structurally identical to ad allocation — Bottou et al. 2013 *Counterfactual Reasoning and Learning Systems* (Microsoft adCenter) is the canonical reference.
- **Stitch Fix**: *Multi-armed bandits and the Stitch Fix experimentation platform* (engineering blog, ongoing); strong on TS + human-in-the-loop.

> **Curriculum slot — Week 21.** Deep-read LinkedIn fairness-aware ranking + Bottou 2013 + Chen 2019; map their architectures onto synthetic insurance and ATS.

---

## 13. Synthesis: the synthetic insurance / ATS Reference Architecture

Putting it together:

```
unstructured input (CV / quote form / broker book)
   │
   ▼
LLM-based feature extraction + structured field parsing
   │
   ▼
two-tower retrieval (top-1000)
   │
   ▼
cross-encoder relevance (top-50)
   │
   ▼
calibrated reward model Q̂(x, a) → P(bind / hire)
   │
   ▼
contextual bandit / RL policy (LinTS + DR + MIPS + fairness constraints)
   │
   ▼
allocation with mechanism-design constraints (DSIC, exposure floors)
   │
   ▼
logged decisions WITH propensities → OPE pipeline (SNIPS / DR / SWITCH)
   │
   ▼
delayed-feedback survival head → reward attribution
   │
   ▼
counterfactual training loop (CRM / BanditNet / DR-CLTR)
```

The pipeline must log: context features, candidate set, propensities, action taken, immediate proxy signal, delayed outcome, sensitive attributes (for fairness audit), and version hashes of every component. Without propensities you cannot do OPE; without versioning you cannot do regression analysis on policy changes.

> **Curriculum slot — Week 22–23.** Build a minimum-viable end-to-end pipeline on synthetic synthetic insurance data; then port to anonymised ATS data; run an offline DR evaluation against the current production heuristic.

> **Curriculum slot — Week 24.** Capstone: write a deployment plan with monitoring (calibration drift, propensity overlap, fairness dashboards, delay-distribution shift) and a kill-switch protocol.

---

## 14. Reading list (compressed)

- Lattimore & Szepesvári, *Bandit Algorithms* (2020) — the textbook.
- Sutton & Barto, *Reinforcement Learning* 2nd ed. (2018) — RL foundation.
- Bietti, Agarwal, Langford, *A Contextual Bandit Bake-off* (JMLR 2021).
- Joachims, Swaminathan, Schnabel, *Unbiased Learning-to-Rank with Biased Feedback* (WSDM 2017).
- Saito & Joachims, *Off-Policy Evaluation for Large Action Spaces via Embeddings* (ICML 2022, extended 2024).
- Chen et al., *Top-K Off-Policy Correction for a REINFORCE Recommender* (WSDM 2019).
- Ie et al., *SlateQ* (IJCAI 2019).
- Liu, Mania, Jordan, *Competing Bandits in Matching Markets* (AISTATS 2020).
- Geyik, Ambler, Kenthapadi, *Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search* (KDD 2019).
- Bottou et al., *Counterfactual Reasoning and Learning Systems* (JMLR 2013).
- Chapelle, *Modeling Delayed Feedback in Display Advertising* (WWW 2014).
- Kull, Filho, Flach, *Beta calibration* (AISTATS 2017).
- Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017).
- Riquelme, Tucker, Snoek, *Deep Bayesian Bandits Showdown* (ICLR 2018).
- Russo & Van Roy, *Learning to Optimize via Information-Directed Sampling* (NeurIPS 2014).
- Jagadeesan, Garg, Steinhardt, *Supply-side Equilibria in Recommender Systems* (NeurIPS 2024).

---

*End of dossier. The thread is meant to be re-entered: each curriculum slot is one week of focused implementation + one short writeup back into this repo. By Week 24 the student has a defensible, calibrated, fairness-aware, delay-robust matching pipeline applicable to both synthetic insurance and the ATS, grounded in 60+ years of bandit and matching theory.*
