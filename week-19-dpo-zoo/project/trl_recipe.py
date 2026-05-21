"""TRL/PEFT recipes for the DPO zoo on a real LM (Qwen2.5-0.5B + LoRA).

This module imports nothing heavy at module load.  Each recipe lives
as a string; running ``python -m project.trl_recipe --variant dpo``
prints the script.
"""
from __future__ import annotations
import argparse


BAKEOFF_RECIPE = r'''
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

variant = "{variant}"  # one of: sigmoid (DPO), ipo, kto_pair, simpo, cpo, orpo
ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:5%]")

cfg = DPOConfig(
    loss_type=variant,
    beta=0.1 if variant != "simpo" else 2.0,
    simpo_gamma=1.0,
    max_length=512,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    bf16=True,
    output_dir=f"runs/{{variant}}",
    logging_steps=10,
    save_steps=200,
    max_steps=500,
)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    task_type="CAUSAL_LM",
)

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    ref_model=None if variant in {{"simpo", "cpo", "orpo"}} else "Qwen/Qwen2.5-0.5B-Instruct",
    args=cfg,
    train_dataset=ds,
    peft_config=peft_cfg,
)
trainer.train()
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=["sigmoid", "ipo", "kto_pair", "simpo", "cpo", "orpo"],
        required=True,
    )
    args = ap.parse_args()
    print(BAKEOFF_RECIPE.format(variant=args.variant))


if __name__ == "__main__":
    main()
