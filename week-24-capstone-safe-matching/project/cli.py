"""Command-line interface for the safe matching capstone.

Usage:
    python -m project.cli rank \
        --job '{"title": "Engineer", "skills": [...16 floats...]}' \
        --candidates candidates.json \
        [--checkpoint runs/model.pt]

`candidates.json` is a list of objects:
    {"name": "Alice", "group": 0, "skills": [16 floats]}

If `--checkpoint` is not provided, the CLI trains a fresh model from scratch
(takes ~30 s on CPU) and uses that. The output is a ranked list:
    1. Alice    score=0.83  group=0
    2. Bob      score=0.71  group=1
    ...
Followed by an audit summary on the held-out interactions.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np
import torch

from .data import (
    JobPosting, Candidate, generate_dataset, context_vector, SKILL_DIM,
)
from .train import train
from .audit import full_audit


def _load_job(arg: str) -> JobPosting:
    obj = json.loads(arg)
    skills = np.asarray(obj["skills"], dtype=np.float32)
    if skills.shape[0] != SKILL_DIM:
        raise ValueError(f"job skills must have length {SKILL_DIM}")
    skills = skills / max(1e-9, np.linalg.norm(skills))
    return JobPosting(job_id=-1, title=obj.get("title", "Engineer"), skills=skills)


def _load_candidates(path: str) -> List[Candidate]:
    with open(path) as f:
        objs = json.load(f)
    cands = []
    for i, o in enumerate(objs):
        skills = np.asarray(o["skills"], dtype=np.float32)
        if skills.shape[0] != SKILL_DIM:
            raise ValueError(f"candidate {i} skills must have length {SKILL_DIM}")
        skills = skills / max(1e-9, np.linalg.norm(skills))
        cands.append(Candidate(candidate_id=i, name=o.get("name", f"C{i}"),
                               skills=skills, group=int(o.get("group", 0))))
    return cands


def rank(net, job: JobPosting, candidates: List[Candidate]) -> List[Dict]:
    out = []
    for c in candidates:
        ctx = context_vector(job, c, include_group=True)
        with torch.no_grad():
            x = torch.as_tensor(ctx, dtype=torch.float32).unsqueeze(0)
            if hasattr(net, "act_prob"):
                p = net.act_prob(x).squeeze(0).numpy()
            else:
                p = net.base.act_prob(x).squeeze(0).numpy()
        out.append({"name": c.name, "score": float(p[1]), "group": int(c.group)})
    out.sort(key=lambda d: -d["score"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="safe-matching-cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rank", help="rank candidates for a job")
    p.add_argument("--job", required=True, help="JSON literal for the job posting")
    p.add_argument("--candidates", required=True, help="path to candidates JSON list")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cost-limit", type=float, default=0.10)
    p.add_argument("--ppo-updates", type=int, default=15)
    p.add_argument("--no-finetune", action="store_true",
                   help="use IQL-only (skip PPO-Lagrangian)")

    args = ap.parse_args(argv)
    if args.cmd != "rank":
        ap.print_help(); return 1

    job = _load_job(args.job)
    cands = _load_candidates(args.candidates)
    from .online_ppo_lag import PPOLagConfig
    ppo_cfg = PPOLagConfig(cost_limit=args.cost_limit, n_updates=args.ppo_updates)
    out = train(seed=args.seed, ppo_cfg=ppo_cfg)
    net = out["iql"]["model"] if args.no_finetune else out["ppo"]["net"]

    ranked = rank(net, job, cands)
    print("\nRanked candidates:")
    for i, r in enumerate(ranked, 1):
        print(f"  {i:2d}. {r['name']:20s}  score={r['score']:.3f}  group={r['group']}")
    a = out["audit_after"]
    print("\nFairness + OPE audit (after Lagrangian fine-tune):")
    print(f"  demographic_parity_gap   = {a['demographic_parity_gap']:.4f}")
    print(f"  equal_opportunity_gap    = {a['equal_opportunity_gap']:.4f}")
    print(f"  snips_value (off-policy) = {a['snips_value']:.4f}")
    print(f"  mean_recommend_rate      = {a['mean_rec_rate']:.4f}")
    print(f"  group_flip_disagreement  = {a['red_team']['group_flip_disagreement']:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
