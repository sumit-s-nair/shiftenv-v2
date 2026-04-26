import argparse
import os
import re
import statistics
import time
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

try:
    import wandb
except Exception:
    wandb = None

from C2RustAI import clean_rust_output, remove_self_import, strip_self_includes
from reward import compute_reward

# ---------------------------------------------------------------------------
# GLOBAL LOGGING
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"  # INFO | DEBUG | NONE

def log(msg: str):
    if LOG_LEVEL == "NONE":
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------------------------------

# QLoRA config — 4-bit NF4 with bfloat16 compute, fits a 7B model in ~4 GB VRAM.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Qwen2.5-Coder-7B-Instruct: ~4 GB at 4-bit, fits T4 (16 GB) with room for
# activations, KV cache, and LoRA gradients during online GRPO training.
DEFAULT_MODEL       = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_ADAPTER_DIR = "lora_adapters"
MAX_NEW_TOKENS      = 1024
GRPO_GROUP_SIZE     = 4
LORA_R              = 16
LORA_ALPHA          = 32
_SAVE_EVERY         = 10
_ADV_EPS            = 1e-8


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

_state = {
    "model_name": DEFAULT_MODEL,
    "adapter_path": DEFAULT_ADAPTER_DIR,
    "group_size": GRPO_GROUP_SIZE,
    "online_training": False,
    "debug": False,
    "use_wandb": False,
    "debug_log": "debug_log.md",
    "lr": 2e-5,
    "step": 0,
    "files_processed": 0,
    "tokenizer": None,
    "model": None,
    "optimizer": None,
}


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

def configure(online_training=True, debug=False,
              model_name=DEFAULT_MODEL,
              adapter_path=DEFAULT_ADAPTER_DIR,
              debug_log="debug_log.md",
              group_size=GRPO_GROUP_SIZE,
              lr=5e-6,
              use_wandb=False):

    log(f"Configuring system | online={online_training}, debug={debug}")

    _state.update(
        online_training=online_training,
        debug=debug,
        use_wandb=bool(use_wandb and online_training),
        model_name=model_name,
        adapter_path=adapter_path,
        debug_log=debug_log,
        group_size=group_size,
        lr=lr,
        step=0,
        files_processed=0,
        tokenizer=None,
        model=None,
        optimizer=None,
    )

    if _state["use_wandb"]:
        if wandb is None:
            raise RuntimeError(
                "WandB logging requested, but the 'wandb' package is not installed. "
                "Install it with: pip install wandb"
            )

        wandb_mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
        if wandb.run is None:
            log(f"Initializing Weights & Biases (mode={wandb_mode})...")
            wandb.init(
                project="c2rust-rl",
                mode=wandb_mode,
                config={
                    "model": _state["model_name"],
                    "learning_rate": _state["lr"],
                    "group_size": _state["group_size"],
                    "online_training": _state["online_training"],
                },
            )


# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a compiler-like C-to-Rust translator. "
    "Output ONLY valid Rust source code. "
    "No explanations, no markdown fences, no commentary."
)


def _build_messages(c_code: str, include_modules: list[str]) -> list[dict]:
    dep_hint = "\n".join(
        f"- {m} → use crate::{m}::*;" for m in include_modules
    ) if include_modules else "(none)"

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Convert the following C code to Rust.\n\n"
            f"Dependencies:\n{dep_hint}\n\n"
            f"```c\n{c_code}\n```"
        )},
    ]


def _apply_chat_template(tokenizer, messages: list[dict]) -> str:
    """Format messages using the model's chat template and add the generation prompt."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    log(f"Prompt built | size={len(prompt)} chars")
    return prompt


def _extract_include_modules(c_code: str, module_name: str) -> list[str]:
    includes = re.findall(r'#include\s+"([^"]+)"', c_code)
    return [
        os.path.splitext(os.path.basename(inc))[0]
        for inc in includes
        if os.path.splitext(os.path.basename(inc))[0] != module_name
    ]


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def _ensure_model():
    if _state["model"] is not None:
        return _state["tokenizer"], _state["model"], _state["optimizer"]

    log(f"Loading tokenizer: {_state['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(_state["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Loading base model (QLoRA 4-bit)...")

    base = AutoModelForCausalLM.from_pretrained(
        _state["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    log("Model loaded")

    if _state["online_training"]:
        # Required before attaching LoRA to a quantized model: casts layer norms
        # to float32 and enables gradient checkpointing to save VRAM.
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        log("Attaching LoRA adapters...")
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()
    else:
        model = base

    model.eval()

    optimizer = None
    if _state["online_training"]:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=_state["lr"])
        log(f"Optimizer initialized | params={len(trainable)}")

    _state.update(tokenizer=tokenizer, model=model, optimizer=optimizer)
    return tokenizer, model, optimizer


# ---------------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------------

def _generate(tokenizer, model, prompt: str, n: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    log(f"Generating {n} samples in parallel...")

    texts, resp_ids = [], []

    with torch.no_grad():
        # Parallel generation instead of sequential loop
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=n,
        )

        for i in range(n):
            ids = out[i][prompt_len:]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
            resp_ids.append(ids)

    log("Generation complete")
    return texts, resp_ids, inputs


# ---------------------------------------------------------------------------
# GRPO STEP
# ---------------------------------------------------------------------------

def _grpo_step(model, optimizer, inputs, resp_ids, rewards):
    rewards_t = torch.tensor(rewards, dtype=torch.float32)

    mean_r = rewards_t.mean()
    std_r = rewards_t.std(unbiased=False) + _ADV_EPS
    advantages = (rewards_t - mean_r) / std_r

    log(f"GRPO step | mean={mean_r:.4f} std={std_r:.4f}")

    prompt_len = inputs["input_ids"].shape[1]
    loss_terms = []

    model.train()

    for i, (ids, adv) in enumerate(zip(resp_ids, advantages)):
        log(f"  sample {i} adv={adv:.4f}")

        full = torch.cat([inputs["input_ids"][0], ids]).unsqueeze(0)

        labels = full.clone()
        labels[:, :prompt_len] = -100

        out = model(full, labels=labels)

        log(f"  sample {i} loss={out.loss.item():.4f}")

        # FIXED: Removed the negative sign to correctly minimize the negated policy gradient
        loss_terms.append((adv.to(model.device)) * out.loss)

    if loss_terms:
        loss = torch.stack(loss_terms).mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        log(f"GRPO update complete | loss={loss.item():.4f}")

        return loss.item()

    return 0.0


# ---------------------------------------------------------------------------
# MAIN CONVERT
# ---------------------------------------------------------------------------

def convert_c_to_rust(file_path: str, output_path: str) -> str:

    module_name   = os.path.splitext(os.path.basename(file_path))[0]
    rust_filename = f"{module_name}.rs"

    log(f"Converting module: {module_name}")

    with open(file_path) as f:
        c_code = f.read()

    c_code = strip_self_includes(c_code, module_name)

    # Model must be loaded before building the prompt so we can use its
    # tokenizer's apply_chat_template.
    tokenizer, model, optimizer = _ensure_model()

    includes = _extract_include_modules(c_code, module_name)
    messages = _build_messages(c_code, includes)
    prompt   = _apply_chat_template(tokenizer, messages)

    texts, resp_ids, inputs = _generate(
        tokenizer, model, prompt, _state["group_size"]
    )

    reward_pairs = [
        compute_reward(clean_rust_output(t), module_name)
        for t in texts
    ]
    rewards = [score for score, _ in reward_pairs]
    reward_infos = [info for _, info in reward_pairs]

    log("Rewards:")
    for i, r in enumerate(rewards):
        log(f"  sample {i}: {r:.4f}")

    best = int(max(range(len(rewards)), key=lambda i: rewards[i]))
    log(f"Best sample = {best} (reward={rewards[best]:.4f})")

    rust = clean_rust_output(texts[best])
    rust = remove_self_import(rust, module_name)
    best_info = reward_infos[best]
    best_success = int(best_info.error_count == 0 and best_info.compilation_score >= 1.0)
    loss_val = 0.0

    if _state["online_training"] and not _state["debug"]:
        _state["step"] += 1
        loss_val = _grpo_step(model, optimizer, inputs, resp_ids, rewards)
        log(f"GRPO step {_state['step']} complete | loss={loss_val:.4f}")

        if _state["step"] % _SAVE_EVERY == 0 and hasattr(model, "save_pretrained"):
            model.save_pretrained(_state["adapter_path"])
            log(f"Adapters saved → {_state['adapter_path']}")
    elif _state["online_training"] and _state["debug"]:
        log("Debug mode is enabled; skipping GRPO parameter update.")

    _state["files_processed"] += 1
    if _state["use_wandb"] and wandb is not None and wandb.run is not None:
        wandb.log(
            {
                "reward": float(rewards[best]),
                "reward_mean": float(sum(rewards) / len(rewards)),
                "compile_success": best_success,
                "unsafe_count": int(best_info.unsafe_count),
                "warning_count": int(best_info.warning_count),
                "error_count": int(best_info.error_count),
                "compilation_score": float(best_info.compilation_score),
                "loss": float(loss_val),
                "module_name": module_name,
                "files_processed": int(_state["files_processed"]),
                "train_step": int(_state["step"]),
            }
        )

    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, rust_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rust)

    log(f"Saved → {out_path}")
    return rust_filename