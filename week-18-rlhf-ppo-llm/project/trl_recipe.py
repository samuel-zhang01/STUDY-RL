"""Compatibility pointer to the executable pretrained-model training lab.

The former printed TRL snippets used outdated trainer APIs. Real LoRA SFT/DPO,
save/reload, evaluation, and NVIDIA QLoRA now live in llm-lab. The week's local
PyTorch PPO project remains the runnable toy RLHF implementation.
"""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sft", "rm", "ppo"], default="sft")
    args = parser.parse_args()
    lab = Path(__file__).resolve().parents[2] / "llm-lab"
    print(f"Executable pretrained-model LoRA/QLoRA training: {lab / 'README.md'}")
    print(f"NVIDIA / RTX 5090 setup: {lab / 'NVIDIA.md'}")
    if args.stage == "sft":
        print("From llm-lab with its environment active, run:")
        print("python -m llm_lab.train --method sft --data data/sft_train.jsonl "
              "--eval-data data/sft_validation.jsonl --output runs/sft --steps 30")
    else:
        print("The new lab implements pretrained-model SFT and DPO. "
              "For the educational reward-model/PPO pipeline, use this week's project/train.py; "
              "a full pretrained-model PPO/RM trainer is not implemented in the lab.")


if __name__ == "__main__":
    main()
