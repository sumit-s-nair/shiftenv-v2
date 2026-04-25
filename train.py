"""
GRPO training loop for the C2Rust RL environment.

Uses HuggingFace TRL's GRPOTrainer (with Unsloth acceleration when available)
to fine-tune DeepSeek-Coder-V2-Lite-Instruct with compiler-shaped rewards.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Reward function (passed to GRPOTrainer) ──────────────────────────────────

def make_reward_fn(env_cfg: dict, reward_cfg: dict):
    """
    Factory returning a reward function compatible with TRL's GRPOTrainer.

    GRPOTrainer calls reward_fn(prompts, completions) → list[float].
    We run each completion through the Rust compiler and return shaped rewards.
    """
    from tester.compiler import compile_and_evaluate

    timeout = env_cfg.get("timeout_seconds", 30)
    run_clippy = env_cfg.get("run_clippy", True)
    unsafe_penalty = reward_cfg.get("unsafe_penalty_per_block", -0.1)
    clippy_bonus = reward_cfg.get("clippy_bonus", 0.1)

    def reward_fn(prompts: list[str], completions: list[str]) -> list[float]:
        rewards = []
        for rust_code in completions:
            result = compile_and_evaluate(
                rust_code=rust_code,
                reference_output=None,
                timeout=timeout,
                run_clippy=run_clippy,
                unsafe_penalty_per_block=unsafe_penalty,
                clippy_bonus=clippy_bonus,
            )
            rewards.append(result.reward)
        return rewards

    return reward_fn


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_grpo_dataset(data_dir: str) -> "datasets.Dataset":
    """Build a HuggingFace Dataset of C→Rust prompts for GRPO training."""
    import datasets as hf_datasets

    data_path = Path(data_dir)
    records = []

    _SYSTEM = (
        "You are an expert Rust programmer. "
        "Translate the following C code to safe, idiomatic Rust. "
        "Do NOT use unsafe blocks. Return only valid Rust source code."
    )

    for c_file in sorted(data_path.glob("*.c")):
        c_source = c_file.read_text(encoding="utf-8")
        prompt = f"{_SYSTEM}\n\nC source:\n```c\n{c_source}\n```\n\nRust translation:\n```rust\n"
        records.append({"prompt": prompt, "c_file": c_file.name})

    return hf_datasets.Dataset.from_list(records)


# ── Main training entry ──────────────────────────────────────────────────────

def train(config_path: str = "configs/config.yaml") -> None:
    cfg = _load_config(config_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    env_cfg = cfg["env"]
    reward_cfg = cfg["reward"]
    log_cfg = cfg.get("logging", {})

    # ── wandb ────────────────────────────────────────────────────────────────
    try:
        import wandb
        wandb.init(
            project=log_cfg.get("wandb_project", "c2rust-rl"),
            config=cfg,
        )
    except ImportError:
        print("wandb not installed — skipping experiment tracking")

    # ── Model loading (Unsloth preferred) ────────────────────────────────────
    load_4bit = model_cfg.get("quantization", "4bit") == "4bit"
    model, tokenizer = _load_model(model_cfg["name"], load_4bit, train_cfg)

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = build_grpo_dataset(env_cfg["data_dir"])
    print(f"Training on {len(dataset)} C programs")

    # ── GRPOTrainer ──────────────────────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir="checkpoints",
        num_train_epochs=max(1, train_cfg.get("max_episodes", 1000) // max(len(dataset), 1)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        num_generations=train_cfg.get("group_size", 4),
        max_completion_length=model_cfg.get("max_tokens", 2048),
        temperature=model_cfg.get("temperature", 0.2),
        logging_steps=log_cfg.get("log_every_n_episodes", 10),
        save_steps=log_cfg.get("save_checkpoint_every", 100),
        report_to="wandb" if "wandb" in dir() else "none",
    )

    reward_fn = make_reward_fn(env_cfg, reward_cfg)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model("final_model")
    print("Training complete. Model saved to ./final_model")


def _load_model(model_name: str, load_4bit: bool, train_cfg: dict):
    """Load model with Unsloth (preferred) or plain transformers."""
    lora_r = train_cfg.get("lora_r", 16)
    lora_alpha = train_cfg.get("lora_alpha", 32)
    lora_dropout = train_cfg.get("lora_dropout", 0.05)

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=load_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        return model, tokenizer
    except ImportError:
        pass

    # Fallback to PEFT + transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    import torch

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=load_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if load_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
