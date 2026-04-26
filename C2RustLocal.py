import argparse
import html
import json
import os
import re
import statistics
import time
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
# Allow HF Spaces to override via env so adapters land on the /data volume
DEFAULT_ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "lora_adapters")
MAX_NEW_TOKENS      = 1024
GRPO_GROUP_SIZE     = 4
LORA_R              = 16
LORA_ALPHA          = 32
_SAVE_EVERY         = 5        # save adapter + checkpoint every N steps
_ADV_EPS            = 1e-8
_CHECKPOINT_FILE    = "training_checkpoint.json"  # sidecar saved next to adapter dir

# Fix 1 — generation diversity
GEN_TEMPERATURE     = 1.2      # higher → more diverse rollouts → non-zero advantage signal
GEN_TOP_P           = 0.95

# Fix 3 — cheap KL penalty via L2 on LoRA params (which start at 0 = base model)
# Equivalent to a Gaussian prior over LoRA deltas; prevents policy from drifting too far.
KL_BETA             = 5e-4    # scale relative to policy loss (~0–2)


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
    "history": [],
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
        history=[],
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

def _checkpoint_path() -> str:
    """Return path to the JSON sidecar that stores step/history for resumption."""
    return os.path.join(_state["adapter_path"], _CHECKPOINT_FILE)


def _save_checkpoint(model, tokenizer):
    """Persist LoRA adapters, tokenizer, and training state to disk."""
    adapter_dir = _state["adapter_path"]
    os.makedirs(adapter_dir, exist_ok=True)

    # Save LoRA weights
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(adapter_dir)

    # Save tokenizer so the adapter dir is self-contained
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(adapter_dir)

    # Save resumable training state
    ckpt = {
        "step": _state["step"],
        "files_processed": _state["files_processed"],
        "history": _state["history"],
    }
    with open(_checkpoint_path(), "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)

    log(f"Checkpoint saved → {adapter_dir} (step={_state['step']})")

    # Auto-push to Hugging Face Hub periodically if token is available
    hf_repo = "sumit-s-nair/c2rust-qwen-adapter"
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and hasattr(model, "push_to_hub"):
        try:
            # Push every 20 steps (since _SAVE_EVERY is 5, this runs on step 20, 40, etc.)
            if _state.get("step", 0) > 0 and _state.get("step", 0) % 20 == 0:
                log(f"Auto-saving to Hugging Face Hub → {hf_repo} ...")
                model.push_to_hub(hf_repo, token=hf_token, safe_serialization=True)
                if hasattr(tokenizer, "push_to_hub"):
                    tokenizer.push_to_hub(hf_repo, token=hf_token)
                log("Hub auto-push complete.")
        except Exception as e:
            log(f"WARNING: Failed to auto-push to hub: {e}")


def _load_checkpoint():
    """Restore step/history from a previous checkpoint if one exists."""
    ckpt_file = _checkpoint_path()
    if not os.path.exists(ckpt_file):
        return
    try:
        with open(ckpt_file, encoding="utf-8") as f:
            ckpt = json.load(f)
        _state["step"] = int(ckpt.get("step", 0))
        _state["files_processed"] = int(ckpt.get("files_processed", 0))
        _state["history"] = list(ckpt.get("history", []))
        log(f"Resumed from checkpoint: step={_state['step']}, "
            f"files_processed={_state['files_processed']}")
    except Exception as e:
        log(f"WARNING: Could not load checkpoint ({e}); starting fresh.")


def _ensure_model():
    if _state["model"] is not None:
        return _state["tokenizer"], _state["model"], _state["optimizer"]

    log(f"Loading tokenizer: {_state['model_name']}")

    adapter_dir = _state["adapter_path"]
    saved_adapter_exists = (
        os.path.isdir(adapter_dir)
        and os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
    )

    # Prefer loading the tokenizer from a saved adapter dir if available
    tok_source = adapter_dir if saved_adapter_exists else _state["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
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

        if saved_adapter_exists:
            log(f"Loading existing LoRA adapters from {adapter_dir} ...")
            model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
            # Restore step / history counters
            _load_checkpoint()
        else:
            log("Attaching fresh LoRA adapters...")
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
            temperature=GEN_TEMPERATURE,  # Fix 1: 1.2 for diverse rollouts
            top_p=GEN_TOP_P,
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

    # Skip update if all rewards identical — no gradient signal available.
    # (Previously this silently ran with loss=0 wasting 3–4 seconds.)
    if std_r <= _ADV_EPS * 2:
        log("GRPO step skipped — zero reward variance (all samples identical)")
        return 0.0

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

        loss_terms.append((adv.to(model.device)) * out.loss)

    if loss_terms:
        policy_loss = torch.stack(loss_terms).mean()

        # Fix 3 — KL proxy: L2 regularisation on LoRA params.
        # LoRA deltas are initialised to 0 (= base model), so penalising their
        # L2 norm approximates a KL penalty against the reference model without
        # loading a second 7B copy into VRAM.
        lora_l2 = sum(
            p.pow(2).sum()
            for p in model.parameters()
            if p.requires_grad
        )
        loss = policy_loss + KL_BETA * lora_l2

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        log(f"GRPO update complete | policy_loss={policy_loss.item():.4f} "
            f"kl_proxy={lora_l2.item():.4f} total_loss={loss.item():.4f}")

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
    best_unsafe = int(best_info.unsafe_count)
    loss_val = 0.0

    if _state["online_training"] and not _state["debug"]:
        _state["step"] += 1
        loss_val = _grpo_step(model, optimizer, inputs, resp_ids, rewards)
        log(f"GRPO step {_state['step']} complete | loss={loss_val:.4f}")

        if _state["step"] % _SAVE_EVERY == 0:
            _save_checkpoint(model, tokenizer)
    elif _state["online_training"] and _state["debug"]:
        log("Debug mode is enabled; skipping GRPO parameter update.")

    mean_reward = float(sum(rewards) / len(rewards))
    max_reward = float(rewards[best])
    _state["history"].append(
        {
            "step": int(_state["step"]),
            "module": module_name,
            "loss": float(loss_val),
            "mean_reward": mean_reward,
            "max_reward": max_reward,
            "compile_success": int(best_success),
            "unsafe_count": best_unsafe,
        }
    )

    _state["files_processed"] += 1
    if _state["use_wandb"] and wandb is not None and wandb.run is not None:
        wandb_payload = {
            "train/step": int(_state["step"]),
            "train/loss": float(loss_val),
            "reward/mean": mean_reward,
            "reward/max": max_reward,
            "compile_success": int(best_success),
            "unsafe_count": best_unsafe,
            "warning_count": int(best_info.warning_count),
            "error_count": int(best_info.error_count),
            "compilation_score": float(best_info.compilation_score),
            "module_name": module_name,
            "files_processed": int(_state["files_processed"]),
        }

        if hasattr(wandb, "Html"):
            escaped_rust = html.escape(rust)
            wandb_payload["samples/best_rust_code"] = wandb.Html(
                f"<h3>Module: {module_name}</h3><pre><code class='language-rust'>{escaped_rust}</code></pre>"
            )

        wandb.log(wandb_payload)

    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, rust_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rust)

    log(f"Saved → {out_path}")
    return rust_filename


# ---------------------------------------------------------------------------
# SUBMISSION REPORT GENERATOR
# ---------------------------------------------------------------------------

def generate_submission_report(output_dir: str):
    """Generate PNG graphs and a Markdown summary for hackathon submissions."""
    history = _state.get("history", [])
    if not history:
        log("No history to plot. Skipping report generation.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(history)

    if "step" not in df.columns or df["step"].nunique() <= 1:
        df["step"] = list(range(1, len(df) + 1))

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle("C2Rust RL Training Metrics (Qwen2.5-7B GRPO)", fontsize=16)

    axes[0].plot(df["step"], df["loss"], color="red", marker="o", label="GRPO Loss")
    axes[0].set_title("Policy Loss over Time")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(df["step"], df["max_reward"], color="blue", marker="o", label="Max Reward")
    axes[1].plot(df["step"], df["mean_reward"], color="lightblue", marker="x", label="Mean Reward")
    axes[1].axhline(1.0, color="green", linestyle="--", label="Perfect Score (1.0)")
    axes[1].set_title("Compiler Reward Evolution")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    axes[2].bar(df["step"], df["unsafe_count"], color="orange", label="Unsafe Blocks")
    axes[2].set_title("Reduction in Unsafe Code")
    axes[2].set_xlabel("Training Step (Module)")
    axes[2].set_ylabel("Unsafe Keyword Count")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    graph_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(graph_path)
    plt.close(fig)
    log(f"Submission graphs saved to {graph_path}")

    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    log(f"Raw training history saved to {history_path}")

    md_path = os.path.join(output_dir, "hackathon_report.md")
    final_mean_reward = float(df["mean_reward"].iloc[-1]) if not df.empty else 0.0
    success_count = int(df["compile_success"].sum()) if "compile_success" in df.columns else 0

    try:
        raw_table = df.to_markdown(index=False)
    except Exception:
        raw_table = df.to_string(index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# C2Rust RL Training Summary\n\n")
        f.write(f"**Total Modules Processed:** {len(df)}\n")

    # Final Hub Push
    hf_repo = "sumit-s-nair/c2rust-qwen-adapter"
    hf_token = os.environ.get("HF_TOKEN")
    model = _state.get("model")
    tokenizer = _state.get("tokenizer")
    if hf_token and model is not None and hasattr(model, "push_to_hub"):
        log(f"Finalizing: Pushing model to Hugging Face Hub → {hf_repo} ...")
        try:
            model.push_to_hub(hf_repo, token=hf_token, safe_serialization=True)
            if tokenizer and hasattr(tokenizer, "push_to_hub"):
                tokenizer.push_to_hub(hf_repo, token=hf_token)
            log("Final Hub push complete.")
        except Exception as e:
            log(f"WARNING: Final push to hub failed: {e}")
        f.write(f"**Final Mean Reward:** {final_mean_reward:.3f}\n")
        f.write(f"**Total Successful Compiles:** {success_count} / {len(df)}\n\n")
        f.write("![Training Curves](training_curves.png)\n\n")
        f.write("### Raw Data Log\n")
        f.write(raw_table)

    log(f"Markdown report saved to {md_path}")