# Adversarial learner review — 20 October 2026

Acceptance target: **9.5/10**, with no unresolved mathematical, missing-material,
misleading-feedback, data-loss or essential-navigation defect in the inspected
student workflows. Scores are editorial judgements, not measured learning
outcomes or probabilities that every claim is correct.

## What this round covers

Two content reviewers read all **52 lecture and further-reading documents**,
checked central equations, assumptions, source identities and implementation
claims, and enriched all **26 teaching guides**. The course now provides
**78 explanation layers**: intuition, a numerical case, and derivation with
assumptions. Their worked cases were checked against actual helpers or direct
algebra. A separate reviewer audited the reader, revision and progress workflow;
the integrating reviewer also challenged selected mathematical corrections.

This is a document-wide reading with claim-directed primary-source checks. It
is not a formal verification of every proof, a reproduction of every research
paper, or a systematic audit of all supporting research dossiers. Source links
inside each corrected reading identify the relevant primary material.

## Substantive corrections

The audit found errors that a navigation-only inspection could not expose.
Corrections include UCB pull-count versus regret dependence, the direction of
the InfoNCE information bound, top-K sampling with versus without replacement,
Beta calibration arithmetic, importance-weight support and effective sample
size, and MCTS value perspective. Later lectures now distinguish TD3+BC scaling,
discounted target-return updates, KTO utility and loss, ORPO token normalisation,
FOCOPS projection direction, exact-penalty signs and assumptions, and control
barrier feasibility. Invalid source identifiers, unsupported quotations,
invented guarantees and unmeasured performance claims were removed or replaced
with supported, scoped explanations.

The reading packs teach what a result requires and give a concrete investigation.
They distinguish a paper's algorithm from the bundled small implementation.
For example, Week 23 explicitly notes that the action shield filters execution
while the supplied Q-learning backup remains optimistic over all next actions.
That teaching simplification is not a constrained-Bellman or deployment guarantee.

## Reader, retrieval and progress

Each week has three diagnostic questions: a misconception, a calculation or
assumption check, and a transfer problem. Every answer choice has explanatory
feedback. The **78 repair links** target specific guide headings; **76 notebook
links** target verified one-based cells; **47 GitHub links** target source lines.
Two open-ended investigations use the workbook entry because no exact existing
cell implements their extension. A Week 7 baseline calculation was appended to
the workbook without shifting its existing cells.

The revision queue combines missed, uncertain and due questions across weeks.
Answers and confidence remain separate from manual milestones. Practice intervals
are disclosed; immediate retries do not manufacture longer review intervals.
They are a convenience schedule, not a validated learning model or a course grade.

The reader supports text size, line width, section bookmarks, equation links,
keyboard-scrollable mathematics, enlarged diagrams and focused reading. Every
reading links to its GitHub source and a correction form. The footer links to
Samuel's GitHub profile and the repository. The start page explains these tools
and the transition into local Jupyter.

Progress uses browser-local storage, with a visible save status and downloadable
backups. Existing schema-1 notes, milestones and legacy drafts remain importable.
Independent tab edits merge by field; concurrent note versions are preserved for
review. Imports keep differing notes. Page hiding flushes pending changes;
blocked storage reports failure while export retains unsaved changes. There is
no automatic device sync. Quiz history stores the latest record per question,
not an append-only log of simultaneous answers from different tabs.

## Acceptance and executed evidence

The content reviewers score their inspected scopes **9.50/10** and **9.53/10**;
the independent usability reviewer scores the tested workflow **9.5/10** after repairs. No open veto remains in those reviewed scopes.
The checks below record the tested release candidate; publication also checks the committed build identity.

| Check | Observed result |
|---|---|
| Content coverage | 52/52 lecture/readings read; 26 guides with three explanation layers each |
| Worked cases | 26/26 new layered cases checked against helpers or direct algebra |
| Portal unit tests | 34 passed: rendered links/math, plots, state, concurrent note handling and diagnostics |
| Simulation/state regression | 41 passed; includes overlapping portal cases, so these are not 41 additional unique tests |
| New learning browser suite | 7 workflow groups and 9 WCAG A/AA audits passed, zero reported violations or browser errors |
| Independent reader/revision review | 154 exact link navigations, 7 workflow groups and 12 WCAG A/AA audits passed; 320/390/1440px, 130% reading text, screenshots and keyboard interactions inspected |
| Final complete-course browser suite | 12 workflow groups and 25 WCAG A/AA audits passed on rebuilt content; all 26 guides/lectures, 22 visual experiments, notebook downloads, search and progress; zero browser errors or failed requests |
| Final bundle validation | 26/26 ZIPs match source; all extracted notebook setup cells import successfully |
| Final PDF verification | 50 PDFs rebuilt with no final overflow or undefined-reference warnings; all 52 PDFs pass text extraction; representative pages visually inspected |
| Final release hygiene | 684 files checked, zero release-tree issues; whitespace check passed |

The independent review found mathematical labels duplicated by hidden KaTeX
markup. Section labels, bookmark labels and search text now use one readable
representation. A narrow-screen pass found a long Python symbol escaping its
line; inline code now wraps. Scrollable equations receive keyboard focus only
when they overflow. The final formatting review verified these fixes and a
single concise revision-page introduction. Complete-course integration also
caught a removed Week 20 proof link; the reading now links directly to the
RLOO unbiasedness theorem again. The accompanying memory and stability claims
were scoped to the actual architecture and estimator assumptions.

## Document-by-document coverage

Every row below records a complete source-text reading. Checks focus on central
formulas, assumptions, numerical examples, attribution and correspondence with
the actual course code. “Retained” means the reviewer found no new correction
in that document's checked claims; it does not mean every published theorem was
re-proved. Use the linked source for the full explanations and citations.

| Week | Document | Main correction or checked scope |
|---|---|---|
| 01 | [Lecture](../week-01-mdps-bandits/notes.tex) | Original Thompson theorem attribution and proof conflated later gap-dependent frequentist and Bayesian bounds. Removed unsupported application discount defaults. |
| 01 | [Further reading](../week-01-mdps-bandits/extended_reading.tex) | Replaced unverified quotations, invented empirical gains, and overbroad theorem/default claims with source-linked contribution/assumption/investigation notes. Added source-specific contribution, assumptions, numerical/notebook investigations; removed unverified quotations instead of presenting paraphrases as author quotations. |
| 02 | [Lecture](../week-02-bellman-dp/notes.tex) | Learning objective contradicted scoped GPI discussion. Removed unsupported PI complexity and specified stable tie-breaking. |
| 02 | [Further reading](../week-02-bellman-dp/extended_reading.tex) | Replaced unverified quotations, invented empirical gains, and overbroad theorem/default claims with source-linked contribution/assumption/investigation notes. Added source-specific contribution, assumptions, numerical/notebook investigations; removed unverified quotations instead of presenting paraphrases as author quotations. |
| 03 | [Lecture](../week-03-monte-carlo/notes.tex) | Learning objective contradicted corrected MC theorem. Distinguished continuing returns, integrability and changing objective. |
| 03 | [Further reading](../week-03-monte-carlo/extended_reading.tex) | Replaced unverified quotations, invented empirical gains, and overbroad theorem/default claims with source-linked contribution/assumption/investigation notes. Added source-specific contribution, assumptions, numerical/notebook investigations; removed unverified quotations instead of presenting paraphrases as author quotations. |
| 04 | [Lecture](../week-04-td-sarsa-q/notes.tex) | Removed false uniqueness claim. Double-Q is not generically unbiased. |
| 04 | [Further reading](../week-04-td-sarsa-q/extended_reading.tex) | Replaced unverified quotations, invented empirical gains, and overbroad theorem/default claims with source-linked contribution/assumption/investigation notes. Added source-specific contribution, assumptions, numerical/notebook investigations; removed unverified quotations instead of presenting paraphrases as author quotations. |
| 05 | [Lecture](../week-05-dqn/notes.tex) | Separated buffer size from target staleness. Corrected PyTorch gather failure description. |
| 05 | [Further reading](../week-05-dqn/extended_reading.tex) | Replaced unverified quotations, invented empirical gains, and overbroad theorem/default claims with source-linked contribution/assumption/investigation notes. Added source-specific contribution, assumptions, numerical/notebook investigations; removed unverified quotations instead of presenting paraphrases as author quotations. |
| 06 | [Lecture](../week-06-rainbow-bbf/notes.tex) | Replaced unsupported Gaussian constant with standard derived sub-Gaussian bound. Removed unsupported IQN effect size. |
| 06 | [Further reading](../week-06-rainbow-bbf/extended_reading.tex) | Replaced unverified quotations, invented empirical gains, and overbroad theorem/default claims with source-linked contribution/assumption/investigation notes. Added source-specific contribution, assumptions, numerical/notebook investigations; removed unverified quotations instead of presenting paraphrases as author quotations. |
| 07 | [Lecture](../week-07-policy-gradients/notes.tex) | Learning objective overstated pathwise variance advantage. Dropping discount need not optimise another objective and bias is not guaranteed small. |
| 07 | [Further reading](../week-07-policy-gradients/extended_reading.tex) | Replaced unverified quotations, invented empirical gains, and overbroad theorem/default claims with source-linked contribution/assumption/investigation notes. Added source-specific contribution, assumptions, numerical/notebook investigations; removed unverified quotations instead of presenting paraphrases as author quotations. |
| 08 | [Lecture](../week-08-a2c-gae-trpo/notes.tex) | Average KL does not imply per-state or individual guarantees. Removed exclusive KL safety claim. |
| 08 | [Further reading](../week-08-a2c-gae-trpo/extended_reading.tex) | Retained previously grounded reading companion after reviewing contribution, theoretical/empirical scope, and notebook connections; no substantive correction required this pass. |
| 09 | [Lecture](../week-09-ppo/notes.tex) | Removed memorisation/all37 requirement. Removed unsupported universal implementation effect split. |
| 09 | [Further reading](../week-09-ppo/extended_reading.tex) | Retained previously grounded reading companion after reviewing contribution, theoretical/empirical scope, and notebook connections; no substantive correction required this pass. |
| 10 | [Lecture](../week-10-sac-td3/notes.tex) | Removed volatile superiority claim. Removed unsupported replay-use and efficiency multipliers. |
| 10 | [Further reading](../week-10-sac-td3/extended_reading.tex) | Replaced unverified quotations, invented empirical gains, and overbroad theorem/default claims with source-linked contribution/assumption/investigation notes. Added source-specific contribution, assumptions, numerical/notebook investigations; removed unverified quotations instead of presenting paraphrases as author quotations. |
| 11 | [Lecture](../week-11-contextual-bandits-ope/notes.tex) | Removed duplicated qualifier. Distinguished count and regret coefficients. |
| 11 | [Further reading](../week-11-contextual-bandits-ope/extended_reading.tex) | Replaced unverified quotations, invented empirical gains, and overbroad theorem/default claims with source-linked contribution/assumption/investigation notes. Added source-specific contribution, assumptions, numerical/notebook investigations; removed unverified quotations instead of presenting paraphrases as author quotations. |
| 12 | [Lecture](../week-12-rl-ranking-matching/notes.tex) | Distillation not guaranteed to retain teacher accuracy. Removed unsupported current SOTA and 5–10% hiring claim. |
| 12 | [Further reading](../week-12-rl-ranking-matching/extended_reading.tex) | Replaced unverified quotations, invented empirical gains, and overbroad theorem/default claims with source-linked contribution/assumption/investigation notes. Added source-specific contribution, assumptions, numerical/notebook investigations; removed unverified quotations instead of presenting paraphrases as author quotations. |
| 13 | [Lecture](../week-13-multi-agent-alphazero/notes.tex) | Made maximise-policy/minimise-value sign consistent. Corrected zero-count tie handling, player perspective and invented visit distribution. |
| 13 | [Further reading](../week-13-multi-agent-alphazero/extended_reading.tex) | Replaced unverified quotations, invented empirical gains, and overbroad theorem/default claims with source-linked contribution/assumption/investigation notes. Added source-specific contribution, assumptions, numerical/notebook investigations; removed unverified quotations instead of presenting paraphrases as author quotations. |
| 14 | [Lecture](../week-14-offline-rl-1/notes.tex) | Separated exact non-parametric AWR improvement from its parametric projection and a chosen hard KL budget. Corrected reversed TD3+BC scaling and overstated ReBRAC leaderboard claim. |
| 14 | [Further reading](../week-14-offline-rl-1/extended_reading.tex) | Replaced unverified quotation-style passages, universal rankings and overstated transfer claims with primary-linked mechanism/scope annotations, derivation prompts and exact course implementation boundaries. |
| 15 | [Lecture](../week-15-decision-transformer-diffusion/notes.tex) | Corrected target-return recurrence for the defined discounted return. Replaced convex-hull criterion by conditional support and stochastic-luck limits. |
| 15 | [Further reading](../week-15-decision-transformer-diffusion/extended_reading.tex) | Replaced unverified quotation-style passages, universal rankings and overstated transfer claims with primary-linked mechanism/scope annotations, derivation prompts and exact course implementation boundaries. |
| 16 | [Lecture](../week-16-model-based/notes.tex) | Made reward/continuation arrival convention explicit across RSSM and return equations. Removed inaccurate blanket attribution of actor estimators across Dreamer versions. |
| 16 | [Further reading](../week-16-model-based/extended_reading.tex) | Replaced unverified quotation-style passages, universal rankings and overstated transfer claims with primary-linked mechanism/scope annotations, derivation prompts and exact course implementation boundaries. |
| 17 | [Lecture](../week-17-exploration-meta/notes.tex) | Added absorbing-terminal bootstrap mask to RND Q-learning pseudocode. Separated MAML symmetry/stationarity from a claimed unique optimal initialisation. |
| 17 | [Further reading](../week-17-exploration-meta/extended_reading.tex) | Replaced unverified quotation-style passages, universal rankings and overstated transfer claims with primary-linked mechanism/scope annotations, derivation prompts and exact course implementation boundaries. |
| 18 | [Lecture](../week-18-rlhf-ppo-llm/notes.tex) | Corrected deterministic-preference claim and beta-versus-fitted-margin confusion. Separated reference-KL controller provenance, horizon units and token/sequence KL targets. |
| 18 | [Further reading](../week-18-rlhf-ppo-llm/extended_reading.tex) | Replaced unverified quotation-style passages, universal rankings and overstated transfer claims with primary-linked mechanism/scope annotations, derivation prompts and exact course implementation boundaries. |
| 19 | [Lecture](../week-19-dpo-zoo/notes.tex) | Removed duplicate beta from KTO by defining the raw log-ratio separately. Corrected ORPO mean-token normalisation and false endpoint-gradient claim. |
| 19 | [Further reading](../week-19-dpo-zoo/extended_reading.tex) | Replaced unverified quotation-style passages, universal rankings and overstated transfer claims with primary-linked mechanism/scope annotations, derivation prompts and exact course implementation boundaries. |
| 20 | [Lecture](../week-20-grpo-rloo-rlvr/notes.tex) | Corrected false premises that PPO requires dense reward/stationary states and token credit assignment is fictitious. Corrected GenRM authors/description and distinction between datasets, methods and preference reward models. |
| 20 | [Further reading](../week-20-grpo-rloo-rlvr/extended_reading.tex) | Replaced unverified quotation-style passages, universal rankings and overstated transfer claims with primary-linked mechanism/scope annotations, derivation prompts and exact course implementation boundaries. Restored an explicit cross-document link to notes.tex#thm:rloo-unbiased with a proof reconstruction and self-inclusion counterexample investigation. |
| 21 | [Lecture](../week-21-reward-hacking-oversight/notes.tex) | Qualified remaining universal specification-gaming claim. |
| 21 | [Further reading](../week-21-reward-hacking-oversight/extended_reading.tex) | Detailed pack retained: independent primary-source recheck found its scoped interpretations and course-created examples consistent; no corrective source edit required. |
| 22 | [Lecture](../week-22-cmdp-cpo-lagrangian/notes.tex) | Added occupancy nonnegativity; distinguished finite feasible bounded LP duality from strict Slater feasibility. Restored discounted visitation weighting in the policy-gradient identity. |
| 22 | [Further reading](../week-22-cmdp-cpo-lagrangian/extended_reading.tex) | Replaced unverified quotation-style passages, universal rankings and overstated transfer claims with primary-linked mechanism/scope annotations, derivation prompts and exact course implementation boundaries. |
| 23 | [Lecture](../week-23-safe-exploration-shielding-robust/notes.tex) | Corrected learned-shield attribution and limited assurance to its model/uncertainty assumptions. Corrected action/state gradient dimension mismatch and added zero-authority/infeasible projection cases. |
| 23 | [Further reading](../week-23-safe-exploration-shielding-robust/extended_reading.tex) | Replaced unverified quotation-style passages, universal rankings and overstated transfer claims with primary-linked mechanism/scope annotations, derivation prompts and exact course implementation boundaries. |
| 24 | [Lecture](../week-24-capstone-safe-matching/notes.tex) | Reconciled introductory implementation description with actual probability-gap actor and multiplier objective. Corrected Safety Gym source identifier; arXiv:1910.01708 is an unrelated batch-RL paper. |
| 24 | [Further reading](../week-24-capstone-safe-matching/extended_reading.tex) | Replaced unverified quotation-style passages, universal rankings and overstated transfer claims with primary-linked mechanism/scope annotations, derivation prompts and exact course implementation boundaries. |
| 25 | [Lecture](../week-25-synthetic-ats-rl-audit/notes.tex) | Separated greedy bandit reduction from linear-bandit statistical regret assumptions. Corrected LinTS posterior arithmetic, uncertainty interpretation, and factorization complexity. |
| 25 | [Further reading](../week-25-synthetic-ats-rl-audit/extended_reading.tex) | Replaced unverified quotation-style passages, universal rankings and overstated transfer claims with primary-linked mechanism/scope annotations, derivation prompts and exact course implementation boundaries. |
| 26 | [Lecture](../week-26-causal-discovery-biological-data/notes.tex) | Full lecture reviewed; retained measured-confounder identification, observational-equivalence and patient-split boundaries. New guide arithmetic independently verified; no corrective lecture edit required. |
| 26 | [Further reading](../week-26-causal-discovery-biological-data/extended_reading.tex) | Retained the detailed reading pack, rechecked primary-source identities and short passages, and added an accessible PMC primary-article mirror after the publisher returned HTTP 403. |

## Limits that remain

Automated accessibility checks and keyboard inspection do not certify every
assistive technology. Browser storage can be cleared or evicted, and mobile
operating systems can terminate a browser without a final event; keep a separate
backup. The host remains a static teaching portal; Python runs in downloaded
Jupyter workbooks. Passing a supplied reference solution does not demonstrate an
independent implementation. Large training benchmarks and every historical
saved curve were not rerun in this editorial pass. A study with human learners
is still needed to measure pacing, comprehension and retention.

The earlier [9 October review](archive/LEARNER_REVIEW_2026-10-09.md) and
[interactive validation](archive/INTERACTIVE_VALIDATION_2026-10-09.md) retain
their original dates and scopes. Current source history preserves the previous
portal audit without creating another competing entry document.
