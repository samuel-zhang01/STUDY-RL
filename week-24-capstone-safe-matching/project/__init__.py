"""Week 24 capstone: end-to-end SAFE candidate-job matching policy.

Pipeline:
    1. data.py            — synthesise candidates + jobs + logged interactions
    2. offline_iql.py     — IQL pretrain a re-ranking policy on the logged data
    3. online_ppo_lag.py  — fine-tune online with a fairness-cost Lagrangian
    4. audit.py           — fairness metrics, OPE (SNIPS), calibration, red-team
    5. cli.py             — command-line interface that ranks a candidate JSON
                            for a job posting and prints the audit summary

This is the synthesis of:
  * offline RL (weeks 14-15)
  * CMDPs (week 22)
  * ranking & matching (week 12)
  * scalable oversight (week 21)
"""
from .data import generate_dataset, JobPosting, Candidate, LoggedInteraction
from .offline_iql import iql_pretrain, IQLConfig
from .online_ppo_lag import ppo_lag_finetune, PPOLagConfig
from .audit import (
    demographic_parity_gap,
    equal_opportunity_gap,
    snips_ope,
    calibration_curve,
    red_team_eval,
    full_audit,
)
from .train import train

__all__ = [
    "generate_dataset", "JobPosting", "Candidate", "LoggedInteraction",
    "iql_pretrain", "IQLConfig",
    "ppo_lag_finetune", "PPOLagConfig",
    "demographic_parity_gap", "equal_opportunity_gap",
    "snips_ope", "calibration_curve", "red_team_eval", "full_audit",
    "train",
]
