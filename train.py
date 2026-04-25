"""
GRPO training for C2Rust — headless, streams rich logs to stdout, saves
training-curve graphs to logs/ after every checkpoint and at run end.

Usage:
    python -u train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Terminal helpers ──────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def banner(text: str, char: str = "=", width: int = 68) -> None:
    bar = char * width
    print(f"\n{bar}", flush=True)
    print(f"  {text}", flush=True)
    print(f"{bar}\n", flush=True)


# ── Reward tracker ────────────────────────────────────────────────────────────

class RewardTracker:
    """
    GRPOTrainer-compatible reward function that also accumulates per-step
    stats for logging and graphing.
    """

    def __init__(self, env_cfg: dict, reward_cfg: dict) -> None:
        from tester.compiler import compile_and_evaluate
        self._compile = compile_and_evaluate
        self._timeout = env_cfg.get("timeout_seconds", 30)
        self._run_clippy = env_cfg.get("run_clippy", True)
        self._unsafe_penalty = reward_cfg.get("unsafe_penalty_per_block", -0.1)
        self._clippy_bonus = reward_cfg.get("clippy_bonus", 0.1)

        # Per-interval buffers (flushed by the callback on each log step)
        self._buf_rewards: list[float] = []
        self._buf_compiled: list[int] = []
        self._buf_unsafe: list[int] = []
        self._buf_ecodes: list[str] = []

        # Full history kept for graphing
        self.history: list[dict] = []

    def __call__(self, prompts: list[str], completions: list[str]) -> list[float]:
        from agent.code_writer import _extract_rust
        rewards = []
        for raw in completions:
            rust_code = _extract_rust(raw)
            result = self._compile(
                rust_code=rust_code,
                reference_output=None,
                timeout=self._timeout,
                run_clippy=self._run_clippy,
                unsafe_penalty_per_block=self._unsafe_penalty,
                clippy_bonus=self._clippy_bonus,
            )
            rewards.append(result.reward)
            self._buf_rewards.append(result.reward)
            self._buf_compiled.append(1 if result.success else 0)
            self._buf_unsafe.append(result.unsafe_count)
            for e in result.errors[:3]:
                if e.code:
                    self._buf_ecodes.append(e.code)
        return rewards

    def flush(self, step: int) -> dict:
        """Drain buffers, record a history entry, return stats dict."""
        if not self._buf_rewards:
            return {}
        r = self._buf_rewards
        stats = {
            "step": step,
            "reward_mean": mean(r),
            "reward_max": max(r),
            "reward_min": min(r),
            "reward_std": stdev(r) if len(r) > 1 else 0.0,
            "compile_rate": mean(self._buf_compiled),
            "unsafe_mean": mean(self._buf_unsafe),
            "n_samples": len(r),
            "top_errors": _fmt_top_errors(self._buf_ecodes),
        }
        self.history.append(stats)
        self._buf_rewards.clear()
        self._buf_compiled.clear()
        self._buf_unsafe.clear()
        self._buf_ecodes.clear()
        return stats


def _fmt_top_errors(codes: list[str], n: int = 4) -> str:
    if not codes:
        return "none"
    return "  ".join(f"E{c}×{k}" for c, k in Counter(codes).most_common(n))


# ── Trainer callback ──────────────────────────────────────────────────────────

def _make_callback(tracker: RewardTracker, graph_dir: Path, graph_every: int):
    """Return a TrainerCallback that streams rich logs and saves graphs."""
    from transformers import TrainerCallback

    class C2RustCallback(TrainerCallback):
        def __init__(self):
            self._t0 = time.time()
            self._best = float("-inf")
            self._last_graph_step = -1

        def _elapsed(self) -> str:
            s = time.time() - self._t0
            h, m = int(s // 3600), int(s % 3600 // 60)
            return f"{h}h{m:02d}m" if h else f"{m}m{int(s%60):02d}s"

        def on_log(self, args, state, control, logs=None, **kw):
            step = state.global_step
            stats = tracker.flush(step)
            if not stats:
                return

            loss = (logs or {}).get("loss", float("nan"))
            lr   = (logs or {}).get("learning_rate", float("nan"))

            new_best = stats["reward_mean"] > self._best
            if new_best:
                self._best = stats["reward_mean"]
            mark = "  ★ NEW BEST" if new_best else ""

            print(f"\n{'─'*68}", flush=True)
            print(
                f"  step {step:>5}  |  {self._elapsed()}  |  "
                f"loss {loss:.4f}  |  lr {lr:.2e}",
                flush=True,
            )
            print(
                f"  reward   mean={stats['reward_mean']:+.3f}  "
                f"max={stats['reward_max']:+.3f}  "
                f"min={stats['reward_min']:+.3f}  "
                f"std={stats['reward_std']:.3f}{mark}",
                flush=True,
            )
            print(
                f"  compile  {stats['compile_rate']*100:5.1f}%  |  "
                f"unsafe {stats['unsafe_mean']:.2f} avg  |  "
                f"samples {stats['n_samples']}",
                flush=True,
            )
            if stats["top_errors"] != "none":
                print(f"  errors   {stats['top_errors']}", flush=True)
            print(f"{'─'*68}", flush=True)

            if (
                step // graph_every > self._last_graph_step // graph_every
                and len(tracker.history) >= 2
            ):
                self._last_graph_step = step
                _save_graphs(tracker.history, graph_dir)

        def on_epoch_end(self, args, state, control, **kw):
            banner(
                f"EPOCH {int(state.epoch)} END  |  "
                f"step={state.global_step}  |  "
                f"best={self._best:+.3f}  |  {self._elapsed()}",
                char="█",
            )
            if len(tracker.history) >= 2:
                _save_graphs(tracker.history, graph_dir)

        def on_save(self, args, state, control, **kw):
            log(f"checkpoint saved  step={state.global_step}")

        def on_train_end(self, args, state, control, **kw):
            if tracker.history:
                _save_graphs(tracker.history, graph_dir)
            banner(
                f"TRAINING COMPLETE  |  {self._elapsed()}  |  "
                f"best_reward={self._best:+.3f}",
                char="█",
            )

    return C2RustCallback()


# ── Graphs ────────────────────────────────────────────────────────────────────

def _save_graphs(history: list[dict], graph_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed — skipping graphs")
        return

    steps      = [h["step"] for h in history]
    r_mean     = [h["reward_mean"] for h in history]
    r_max      = [h["reward_max"] for h in history]
    r_min      = [h["reward_min"] for h in history]
    compile_pct = [h["compile_rate"] * 100 for h in history]
    unsafe_avg  = [h["unsafe_mean"] for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    fig.suptitle("C2Rust · GRPO Training Curves", fontsize=13, fontweight="bold", y=0.98)

    # ── Reward ────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(steps, r_mean, color="#2563eb", linewidth=2, label="mean reward")
    ax.fill_between(steps, r_min, r_max, alpha=0.18, color="#2563eb", label="min–max band")
    ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Reward", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)

    # ── Compile rate ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(steps, compile_pct, color="#16a34a", linewidth=2)
    ax.set_ylabel("Compile Rate (%)", fontsize=11)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.25)

    # ── Unsafe blocks ─────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(steps, unsafe_avg, color="#dc2626", linewidth=2)
    ax.set_ylabel("Avg unsafe{} blocks", fontsize=11)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = graph_dir / "training_curves.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    log(f"graph saved → {out}")


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset(data_dirs: list[str]) -> "datasets.Dataset":
    import datasets as hf_datasets

    _SYSTEM = (
        "You are an expert Rust programmer. "
        "Translate the following C code to safe, idiomatic Rust. "
        "Do NOT use unsafe blocks. Return only valid Rust source code."
    )

    records: list[dict] = []
    seen: set[str] = set()

    for base_str in data_dirs:
        base = Path(base_str)
        if not base.exists():
            log(f"data dir not found, skipping: {base}")
            continue
        for c_file in sorted(base.rglob("*.c")):
            rel = str(c_file.relative_to(base))
            if rel in seen:
                continue
            seen.add(rel)
            src = c_file.read_text(encoding="utf-8", errors="replace")
            prompt = (
                f"{_SYSTEM}\n\n"
                f"C source — {c_file.name}:\n```c\n{src}\n```\n\n"
                f"Rust translation:\n```rust\n"
            )
            records.append({"prompt": prompt, "c_file": str(c_file)})

    log(f"dataset: {len(records)} C files from {data_dirs}")
    if not records:
        raise ValueError(f"No .c files found in {data_dirs}")
    return hf_datasets.Dataset.from_list(records)


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(
    model_name: str,
    load_4bit: bool,
    train_cfg: dict,
    model_revision: str | None = None,
):
    lora_r       = train_cfg.get("lora_r", 16)
    lora_alpha   = train_cfg.get("lora_alpha", 32)
    lora_dropout = train_cfg.get("lora_dropout", 0.05)

    try:
        from unsloth import FastLanguageModel
        log("loading model via Unsloth …")
        unsloth_kwargs = dict(
            model_name=model_name,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=load_4bit,
        )
        if model_revision:
            unsloth_kwargs["revision"] = model_revision
        model, tokenizer = FastLanguageModel.from_pretrained(**unsloth_kwargs)
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
        log("Unsloth model ready")
        return model, tokenizer
    except ImportError:
        pass

    log("Unsloth not found — falling back to HuggingFace transformers …")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    import torch

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=load_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if load_4bit else None

    tokenizer_kwargs = {"trust_remote_code": True}
    if model_revision:
        tokenizer_kwargs["revision"] = model_revision

    model_kwargs = dict(
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if model_revision:
        model_kwargs["revision"] = model_revision

    log(f"loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    log(f"loading model weights …")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

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
    log("HuggingFace model ready")
    return model, tokenizer


# ── Hub upload ────────────────────────────────────────────────────────────────

def _push_to_hub(model, tokenizer, hub_cfg: dict) -> None:
    repo_id = hub_cfg.get("repo_id") or os.environ.get("HF_REPO_ID", "")
    token   = os.environ.get("HF_TOKEN", "")

    if not repo_id:
        log("Hub upload skipped — set hub.repo_id in config or HF_REPO_ID env var")
        return
    if not token:
        log("Hub upload skipped — HF_TOKEN not set")
        return

    log(f"pushing model to huggingface.co/{repo_id} …")
    try:
        private = hub_cfg.get("private", False)
        model.push_to_hub(repo_id, token=token, private=private)
        tokenizer.push_to_hub(repo_id, token=token, private=private)
        log(f"model pushed → https://huggingface.co/{repo_id}")
        _patch_code_writer(repo_id)
    except Exception as exc:
        log(f"Hub upload failed: {exc}")


def _patch_code_writer(repo_id: str) -> None:
    path = Path("agent/code_writer.py")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    updated = re.sub(
        r'model_name: str = "[^"]+"',
        f'model_name: str = "{repo_id}"',
        text,
        count=1,
    )
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        log(f"agent/code_writer.py default model → {repo_id}")


def _load_grpo_classes():
    try:
        from trl import GRPOConfig, GRPOTrainer
        return GRPOConfig, GRPOTrainer
    except Exception:
        pass

    try:
        from trl.trainer.grpo_config import GRPOConfig
        from trl.trainer.grpo_trainer import GRPOTrainer
        return GRPOConfig, GRPOTrainer
    except Exception as exc:
        trl_ver = "not installed"
        try:
            import trl
            trl_ver = getattr(trl, "__version__", "unknown")
        except Exception:
            pass

        raise ImportError(
            "TRL GRPO APIs are unavailable. Install compatible versions: "
            "trl>=0.14.0, transformers>=4.46.0, accelerate>=0.34.0, "
            f"datasets>=2.21.0. Detected trl={trl_ver}."
        ) from exc


def _resolve_grpo_batch_shape(train_cfg: dict[str, Any]) -> tuple[int, int, int]:
    """
    Ensure TRL's global-batch-size divisibility constraint for GRPO.

    Constraint: (per_device_train_batch_size * world_size) % num_generations == 0
    """
    requested_g = max(1, int(train_cfg.get("group_size", 4)))
    per_device_bs = max(
        1,
        int(train_cfg.get("per_device_train_batch_size", requested_g)),
    )
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    global_bs = per_device_bs * world_size

    if global_bs % requested_g == 0:
        return requested_g, per_device_bs, world_size

    valid_g = [g for g in range(1, global_bs + 1) if global_bs % g == 0]
    adjusted_g = max((g for g in valid_g if g <= requested_g), default=1)
    log(
        "WARNING: requested training.group_size="
        f"{requested_g} is incompatible with global batch size {global_bs} "
        f"(per_device_train_batch_size={per_device_bs}, world_size={world_size}). "
        f"Using num_generations={adjusted_g}."
    )
    return adjusted_g, per_device_bs, world_size


# ── Main ──────────────────────────────────────────────────────────────────────

def train(config_path: str = "configs/config.yaml") -> None:
    cfg       = _load_config(config_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    env_cfg   = cfg["env"]
    reward_cfg = cfg["reward"]
    log_cfg   = cfg.get("logging", {})

    requested_group_size = max(1, int(train_cfg.get("group_size", 4)))
    num_generations, per_device_bs, world_size = _resolve_grpo_batch_shape(train_cfg)
    global_bs = per_device_bs * world_size
    grad_accum_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 4)))

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    banner("C2Rust  ·  GRPO Training")
    log(f"config:     {config_path}")
    log(f"model:      {model_cfg['name']}")
    model_revision = model_cfg.get("revision")
    if model_revision:
        log(f"revision:   {model_revision}")
    else:
        log("WARNING: model.revision is unset — remote code may change between runs")
    log(f"quant:      {model_cfg.get('quantization', '4bit')}")
    log(f"lr:         {train_cfg.get('learning_rate', 1e-5):.2e}")
    log(f"group size: {num_generations} (requested {requested_group_size})")
    log(
        "batch:      "
        f"per_device={per_device_bs}, world_size={world_size}, global={global_bs}, "
        f"grad_accum={grad_accum_steps}"
    )
    log(f"max tokens: {model_cfg.get('max_tokens', 2048)}")

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            log(f"GPU:        {name}  ({vram:.0f} GB VRAM)")
        else:
            log("WARNING: no CUDA GPU detected — model load will fail")
    except Exception:
        pass

    # wandb
    wandb_ok = False
    try:
        import wandb
        wandb.init(project=log_cfg.get("wandb_project", "c2rust-rl"), config=cfg)
        wandb_ok = True
        log("wandb initialized")
    except Exception as e:
        log(f"wandb skipped: {e}")

    load_4bit = model_cfg.get("quantization", "4bit") == "4bit"
    model, tokenizer = _load_model(
        model_cfg["name"],
        load_4bit,
        train_cfg,
        model_revision=model_revision,
    )

    data_dirs = [env_cfg.get("data_dir", "data/c_programs"), "data/repos"]
    dataset = build_dataset(data_dirs)

    log_every  = log_cfg.get("log_every_n_episodes", 10)
    save_every = log_cfg.get("save_checkpoint_every", 100)
    graph_every = log_cfg.get("graph_every_n_steps", save_every)

    GRPOConfig, GRPOTrainer = _load_grpo_classes()

    tracker  = RewardTracker(env_cfg, reward_cfg)
    callback = _make_callback(tracker, log_dir, graph_every)

    n_epochs = max(1, train_cfg.get("max_episodes", 1000) // max(len(dataset), 1))
    log(f"epochs:     {n_epochs}  (={len(dataset)} samples × {n_epochs} passes)")

    grpo_config = GRPOConfig(
        output_dir="checkpoints",
        num_train_epochs=n_epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        num_generations=num_generations,
        max_completion_length=model_cfg.get("max_tokens", 2048),
        temperature=model_cfg.get("temperature", 0.2),
        logging_steps=log_every,
        save_steps=save_every,
        report_to="wandb" if wandb_ok else "none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[tracker],
        args=grpo_config,
        train_dataset=dataset,
        callbacks=[callback],
    )

    banner("Starting GRPO training …", char="-")
    trainer.train()

    trainer.save_model("final_model")
    tokenizer.save_pretrained("final_model")
    log("model saved → ./final_model")

    _push_to_hub(model, tokenizer, cfg.get("hub", {}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C2Rust GRPO training")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
