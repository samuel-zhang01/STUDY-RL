# Study progress tracker

A markdown checklist you can tick off as you work through the curriculum.
For each week, mark a box when you've (a) explained the main idea without the
notes, (b) reproduced a worked numerical example, (c) implemented and tested an
exercise on a changed input without calling the reference solution, and (d)
evaluated the mini-project. Use the weekly guides linked from
[STUDY_GUIDE.md](STUDY_GUIDE.md). Record execution failures separately from
conceptual difficulties; both help identify the next useful task.

## A record for each completed week

Copy this small template into your personal notes:

```text
Week:
Explain: The algorithm changes ____ because ____.
Calculate: My prediction was ____; the hand calculation gave ____.
Transfer: I changed ____; I expected ____; I observed ____.
Evaluate: Data/source, seed(s), baseline, metric, result, and limitation.
Revisit: One misconception to test again next session.
```

## Track 1 — Foundations
- [ ] **Week 01 — MDPs & bandits** — PDF · Workbook · Tests · Project (bandit testbed)
- [ ] **Week 02 — Bellman & DP** — PDF · Workbook · Tests · Project (VI / PI / Car Rental)
- [ ] **Week 03 — Monte Carlo** — PDF · Workbook · Tests · Project (Blackjack)
- [ ] **Week 04 — TD / SARSA / Q-learning** — PDF · Workbook · Tests · Project (Cliff + Taxi)

## Track 2 — Deep value
- [ ] **Week 05 — DQN** — PDF · Workbook · Tests · Project (LunarLander)
- [ ] **Week 06 — Rainbow + BBF** — PDF · Workbook · Tests · Project (ablation)

## Track 3 — Policy methods
- [ ] **Week 07 — Policy gradients & REINFORCE** — PDF · Workbook · Tests · Project (CartPole + Acrobot)
- [ ] **Week 08 — A2C, GAE, TRPO** — PDF · Workbook · Tests · Project (LunarLander-Continuous)
- [ ] **Week 09 — PPO + the 37 details** — PDF · Workbook · Tests · Project (BipedalWalker)
- [ ] **Week 10 — SAC, TD3, CrossQ** — PDF · Workbook · Tests · Project (Pendulum / Hopper)

## Track 4 — Bandits & matching
- [ ] **Week 11 — Contextual bandits + OPE** — PDF · Workbook · Tests · Project (synthetic insurance toy)
- [ ] **Week 12 — RL ranking & fair matching** — PDF · Workbook · Tests · Project (ATS toy)

## Track 5 — Multi-agent
- [ ] **Week 13 — MARL + AlphaZero** — PDF · Workbook · Tests · Project (MAPPO + Tic-Tac-Toe AZ)

## Track 6 — Offline RL
- [ ] **Week 14 — Offline RL fundamentals** — PDF · Workbook · Tests · Project (IQL / CQL)
- [ ] **Week 15 — Decision Transformer + diffusion** — PDF · Workbook · Tests · Project (DT)

## Track 7 — Model-based
- [ ] **Week 16 — Dreamer / MuZero / TD-MPC2** — PDF · Workbook · Tests · Project (Dreamer-lite)

## Track 8 — Exploration + meta
- [ ] **Week 17 — RND, ICM, MAML, in-context RL** — PDF · Workbook · Tests · Project (RND + MAML)

## Track 9 — LLM RL
- [ ] **Week 18 — RLHF / PPO for LMs** — PDF · Workbook · Tests · Project (toy RLHF + practical LLM lab)
- [ ] **Week 19 — DPO zoo (IPO, KTO, ORPO, SimPO)** — PDF · Workbook · Tests · Project (bake-off)
- [ ] **Week 20 — GRPO / RLOO / RLVR / PRMs** — PDF · Workbook · Tests · Project (verifier-rewarded math)
- [ ] **Week 21 — Reward hacking + scalable oversight** — PDF · Workbook · Tests · Project (hack lab + CAI)

## Track 10 — Safety-critical RL
- [ ] **Week 22 — CMDPs, CPO, PPO-Lagrangian** — PDF · Workbook · Tests · Project (PPO-Lag)
- [ ] **Week 23 — Safe exploration, shielding, robust** — PDF · Workbook · Tests · Project (LTL shield)
- [ ] **Week 24 — Capstone: safe candidate matching** — PDF · Workbook · Tests · Project (CLI)

## Track 11 — Applied audit
- [ ] **Week 25 — synthetic ATS RL enhancement audit** — PDF · Workbook · Tests · Project (synthetic-ATS)

## Track 12 — Causal biology
- [ ] **Week 26 — Causal discovery across biological environments** — PDF · Workbook · Tests · Project (synthetic patients, confounding, enhancement ablations)
- [ ] Distinguish association, causal effect estimation, and causal graph discovery.
- [ ] Derive an adjusted effect and name its identification assumptions.
- [ ] Explain an observationally equivalent graph and what additional evidence could distinguish it.
- [ ] Evaluate noisy-data enhancement using separate patients and an unseen environment; report the baseline and uncertainty.

---

Completing all 26 weeks gives you practice with the main algorithm families,
their mathematical foundations, synthetic application audits, and causal
reasoning about biological data. Several
projects are deliberately small approximations; completion does not imply
reproduction of every cited paper or validation of a production system.


## Optional practical LLM post-training track

Start with [the lab guide](../llm-lab/README.md); reading and the numerical workbook
require no pretrained-model training. Check learning outcomes yourself; passing
reference-code tests does not establish mastery.

- [ ] Explain the token MDP and distinguish SFT, DPO, and online RL.
- [ ] Derive LoRA shapes, gradients, and parameter savings by hand.
- [ ] Calculate quantisation codes/error and distinguish NF4 storage from compute precision.
- [ ] Derive a DPO margin and explain the fixed reference and group baselines.
- [ ] Complete the eight numerical workbook exercises and explain the results.
- [ ] Design data splits, a verifier, and a fair checkpoint comparison.
- [ ] Optional: run the RTX 5090 hardware checks and a two-step LoRA/QLoRA smoke test.
- [ ] Optional: train, save, reload, and evaluate an adapter on a task you chose.
