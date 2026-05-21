"""End-to-end capstone pipeline: data -> IQL -> PPO-Lagrangian -> audit.

Run as
    python -m week_24_capstone_safe_matching.project.train
or import:
    from project.train import train
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional

import numpy as np
import torch

from .data import generate_dataset
from .offline_iql import iql_pretrain, IQLConfig
from .online_ppo_lag import ppo_lag_finetune, PPOLagConfig
from .audit import full_audit


def train(seed: int = 0,
          iql_cfg: Optional[IQLConfig] = None,
          ppo_cfg: Optional[PPOLagConfig] = None,
          n_candidates: int = 1000, n_jobs: int = 50, n_interactions: int = 4000,
          verbose: bool = False) -> Dict:
    """Synthesise data, IQL-pretrain, PPO-Lagrangian finetune, audit."""
    data = generate_dataset(n_candidates=n_candidates, n_jobs=n_jobs,
                            n_interactions=n_interactions, seed=seed)
    iql_cfg = iql_cfg or IQLConfig()
    iql_out = iql_pretrain(data, cfg=iql_cfg, verbose=verbose, seed=seed)
    base = iql_out["model"]
    # Audit BEFORE Lagrangian finetune (to show the bias inherited from logged data).
    audit_before = full_audit(base, data, n_episodes=400)
    ppo_cfg = ppo_cfg or PPOLagConfig()
    ppo_out = ppo_lag_finetune(data, base, cfg=ppo_cfg, seed=seed, verbose=verbose)
    audit_after = full_audit(ppo_out["net"], data, n_episodes=400)
    return {
        "data": data,
        "iql": iql_out,
        "ppo": ppo_out,
        "audit_before": audit_before,
        "audit_after": audit_after,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cost-limit", type=float, default=0.10)
    ap.add_argument("--ppo-updates", type=int, default=25)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    ppo_cfg = PPOLagConfig(cost_limit=args.cost_limit, n_updates=args.ppo_updates)
    out = train(seed=args.seed, ppo_cfg=ppo_cfg, verbose=args.verbose)
    a_b = out["audit_before"]; a_a = out["audit_after"]
    print("=" * 60)
    print(" CAPSTONE AUDIT SUMMARY")
    print("=" * 60)
    for k in ("demographic_parity_gap", "equal_opportunity_gap",
              "snips_value", "mean_rec_rate"):
        print(f"  {k:30s}  before={a_b[k]:+.4f}  after={a_a[k]:+.4f}")
    print(f"  group_flip_disagreement       "
          f"before={a_b['red_team']['group_flip_disagreement']:.3f}  "
          f"after={a_a['red_team']['group_flip_disagreement']:.3f}")
    print("=" * 60)


if __name__ == "__main__":  # pragma: no cover
    main()
