"""Compatibility pointer to the actual LoRA DPO training lab.

The former printed bake-off conflated algorithm-specific TRL APIs. The week's
local losses still compare its toy variants. llm-lab implements executable DPO
on a pretrained model with a fixed initial-policy reference.
"""
from pathlib import Path


def main():
    lab = Path(__file__).resolve().parents[2] / "llm-lab"
    print(f"Follow the SFT stage first: {lab / 'README.md'}")
    print("Then, from llm-lab with its environment active:")
    print("python -m llm_lab.train --method dpo --adapter runs/sft "
          "--data data/preferences_train.jsonl --eval-data data/preferences_validation.jsonl "
          "--output runs/dpo --steps 20 --lr 0.00005 --beta 0.1")
    print("For NVIDIA QLoRA, add --device cuda --quantization nf4; "
          "see NVIDIA.md for installation and a matching SFT run.")
    print("This command implements DPO. IPO/KTO/ORPO/SimPO remain separate "
          "algorithms illustrated by the week's toy project.")


if __name__ == "__main__":
    main()
