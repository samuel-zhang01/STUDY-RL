"""Optional: HuggingFace TRL recipe for the same pipeline on a real LM.

This module is **not** exercised by the test suite — it exists as
documentation, to be copy-pasted into a real GPU session.  We import
nothing at module load to avoid pulling transformers/peft/bitsandbytes
into the rest of the project.

Usage (uncomment when you have the stack installed and a GPU):

    python -m week_18_rlhf_ppo_llm.project.trl_recipe --stage sft
    python -m week_18_rlhf_ppo_llm.project.trl_recipe --stage rm
    python -m week_18_rlhf_ppo_llm.project.trl_recipe --stage ppo
"""
from __future__ import annotations
import argparse


SFT_RECIPE = r'''
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model_name = "HuggingFaceTB/SmolLM2-360M"  # or EleutherAI/pythia-160m
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16")

lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                  lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)

ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:2%]")
ds = ds.map(lambda x: {"text": x["chosen"][0]["content"] + tok.eos_token})

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds,
    args=TrainingArguments(output_dir="runs/sft", per_device_train_batch_size=4,
                            gradient_accumulation_steps=4, num_train_epochs=1,
                            bf16=True, logging_steps=10, save_steps=200,
                            learning_rate=5e-5),
    max_seq_length=512,
)
trainer.train()
'''

RM_RECIPE = r'''
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from peft import LoraConfig
from trl import RewardTrainer, RewardConfig

model_name = "HuggingFaceTB/SmolLM2-360M"
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:5%]")

cfg = RewardConfig(output_dir="runs/rm", per_device_train_batch_size=4,
                    gradient_accumulation_steps=4, num_train_epochs=1,
                    bf16=True, logging_steps=10, learning_rate=1e-4,
                    max_length=512)

peft_cfg = LoraConfig(r=16, lora_alpha=32,
                      target_modules=["q_proj","k_proj","v_proj","o_proj"],
                      task_type="SEQ_CLS")

trainer = RewardTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok,
                        peft_config=peft_cfg)
trainer.train()
'''

PPO_RECIPE = r'''
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOv2Trainer

policy_id = "runs/sft"
rm_id = "runs/rm"
tok = AutoTokenizer.from_pretrained(policy_id)
tok.pad_token = tok.eos_token

policy = AutoModelForCausalLM.from_pretrained(policy_id, torch_dtype="bfloat16")
ref    = AutoModelForCausalLM.from_pretrained(policy_id, torch_dtype="bfloat16")
rm     = AutoModelForSequenceClassification.from_pretrained(rm_id, num_labels=1)

config = PPOConfig(
    output_dir="runs/ppo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,
    num_ppo_epochs=4,
    kl_coef=0.05,
    cliprange=0.2,
    bf16=True,
    response_length=128,
    total_episodes=2000,
)

prompts = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:1%]")
trainer = PPOv2Trainer(
    config=config, tokenizer=tok,
    policy=policy, ref_policy=ref,
    reward_model=rm, train_dataset=prompts,
)
trainer.train()
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sft", "rm", "ppo"], required=True)
    args = ap.parse_args()
    print({"sft": SFT_RECIPE, "rm": RM_RECIPE, "ppo": PPO_RECIPE}[args.stage])


if __name__ == "__main__":
    main()
