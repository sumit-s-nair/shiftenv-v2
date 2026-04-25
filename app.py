"""
HuggingFace Spaces Gradio demo for C2Rust repo migration.

Tabs:
  🚀 Training   — start/monitor GRPO training with live log streaming
  🗂️ Repo Demo  — convert a full C repo module-by-module with the loaded model
  ⚡ Compiler   — paste any Rust code and see the reward score instantly
  ℹ️ About      — how it works
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Generator

import gradio as gr

from agent.code_writer import CodeWriter
from env.c2rust_repo_env import C2RustRepoEnv
from tester.compiler import compile_and_evaluate

# ── Agent (loaded once at startup) ───────────────────────────────────────────

_AGENT: CodeWriter | None = None
_AGENT_ERROR: str = ""

print("Loading DeepSeek-Coder-V2-Lite-Instruct …", flush=True)
try:
    _AGENT = CodeWriter()
    print("Model ready.", flush=True)
except RuntimeError as e:
    _AGENT_ERROR = str(e)
    print(f"WARNING: model not loaded — {_AGENT_ERROR}", flush=True)
    print("Compiler Test tab will still work. Set Space hardware to L40S for generation.", flush=True)

# ── Constants ─────────────────────────────────────────────────────────────────

REPOS_DIR = "data/repos"
AVAILABLE_REPOS = [d.name for d in Path(REPOS_DIR).iterdir() if d.is_dir()]

# ── Training subprocess ───────────────────────────────────────────────────────

_training_proc: subprocess.Popen | None = None


def start_training(config_path: str) -> Generator[str, None, None]:
    global _training_proc

    if _training_proc is not None and _training_proc.poll() is None:
        yield "⚠️  Training is already running. Wait for it to finish.\n"
        return

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    hf_token  = os.environ.get("HF_TOKEN", "")
    hf_repo   = os.environ.get("HF_REPO_ID", "")

    warnings = []
    if not wandb_key:
        warnings.append("WANDB_API_KEY not set — training will run without wandb logging.")
    if not hf_token or not hf_repo:
        warnings.append("HF_TOKEN / HF_REPO_ID not set — model will NOT be pushed to Hub after training.")
    for w in warnings:
        yield f"⚠️  {w}\n"

    yield f"▶ Starting training: python train.py --config {config_path}\n\n"

    env = os.environ.copy()
    _training_proc = subprocess.Popen(
        [sys.executable, "train.py", "--config", config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    for line in _training_proc.stdout:
        yield line

    code = _training_proc.wait()
    yield f"\n{'✅ Training complete' if code == 0 else f'❌ Training exited with code {code}'}.\n"


def stop_training() -> str:
    global _training_proc
    if _training_proc is None or _training_proc.poll() is not None:
        return "No training process is running."
    _training_proc.terminate()
    return "Training process terminated."


def training_status() -> str:
    if _training_proc is None:
        return "Not started."
    code = _training_proc.poll()
    if code is None:
        return "🟢 Running"
    return f"{'✅ Finished' if code == 0 else '❌ Failed'} (exit code {code})"


# ── Reward badge ──────────────────────────────────────────────────────────────

def _reward_badge(reward: float) -> str:
    pct = int((reward + 1.0) / 2.0 * 100)
    colour = "#22c55e" if reward >= 0 else "#ef4444"
    return (
        f'<div style="display:inline-block;padding:6px 14px;border-radius:8px;'
        f'background:{colour};color:#fff;font-weight:bold;font-size:1.1em;">'
        f'Reward: {reward:+.2f} ({pct}%)</div>'
    )


def _status_html(info: dict) -> str:
    converted = info.get("converted_count", 0)
    total     = info.get("total_modules", 1)
    pct       = int(converted / total * 100)
    lines = [
        f"<b>Module:</b> {info.get('module', '?')}",
        f"<b>Progress:</b> {converted}/{total} modules ({pct}%)",
        f"<b>Build:</b> {'✅ OK' if info.get('module_success') else '❌ FAIL'}",
        f"<b>Tests:</b> {info.get('test_passed', 'N/A')}",
        f"<b>Unsafe blocks:</b> {info.get('unsafe_count', 0)}",
    ]
    if info.get("error_type") and info["error_type"] != "none":
        lines.append(f"<b>Error type:</b> {info['error_type']}")
    return "<br>".join(lines)


# ── Repo conversion ───────────────────────────────────────────────────────────

def run_repo_conversion(repo_name: str, max_retries: int) -> Generator:
    if _AGENT is None:
        yield (
            "", "", "",
            f'<div style="color:#ef4444;font-weight:bold;">Model not loaded: {_AGENT_ERROR}</div>',
            "❌ This Space needs GPU hardware to run the model.\n"
            "Go to Space Settings → Hardware → upgrade to L40S or A100.",
        )
        return

    env = C2RustRepoEnv(repos_dir=REPOS_DIR, max_retries_per_module=max_retries)
    obs, info = env.reset(repo_name=repo_name)

    log_lines: list[str] = [
        f"=== Repo: {repo_name} ===",
        f"Conversion order: {info['conversion_order']}",
    ]
    all_rust: dict[str, str] = {}
    reward = 0.0

    while not env._done:
        module = obs["current_module"]
        c_src  = obs["current_c_source"]

        log_lines.append(f"\n→ Converting {module}  (retry {obs['retry_count']})")
        yield (
            f"```c\n// {obs.get('current_rel_path', module + '.c')}\n{c_src}\n```",
            "(generating…)",
            "",
            f"Converting <b>{module}</b>…",
            "\n".join(log_lines),
        )

        rust_code = _AGENT.generate(obs)  # type: ignore[union-attr]
        obs, reward, terminated, truncated, info = env.step({"rust_code": rust_code})

        if info["module_success"]:
            all_rust[module] = rust_code
            log_lines.append(f"  ✅ {module} compiled  reward={reward:+.2f}")
        else:
            log_lines.append(f"  ❌ {module} failed    reward={reward:+.2f}")
            if info.get("error_type"):
                log_lines.append(f"     {info['error_type']}  line={info.get('error_line')}")

        yield (
            f"```c\n// {obs.get('current_rel_path', module + '.c')}\n{c_src}\n```",
            f"```rust\n// {module}.rs\n{rust_code}\n```",
            _reward_badge(reward),
            _status_html(info),
            "\n".join(log_lines),
        )

        if terminated:
            log_lines.append("\n🎉 Full repo migrated and tests passed!")
            break
        if truncated:
            log_lines.append(f"\n⚠️  Max retries reached for {module}.")
            break

    env.close()

    final_rust = "\n\n".join(
        f"// ── {stem}.rs ──\n{code}" for stem, code in all_rust.items()
    )
    yield (
        f"```c\n// {repo_name} — original C\n```",
        f"```rust\n// {repo_name} — final Rust\n{final_rust}\n```",
        _reward_badge(reward),
        _status_html(info if "info" in dir() else {}),
        "\n".join(log_lines),
    )


# ── Quick compiler test ───────────────────────────────────────────────────────

def quick_compile_test(rust_code: str) -> str:
    if not rust_code.strip():
        return "Paste some Rust code above."
    result = compile_and_evaluate(rust_code, run_clippy=False)
    lines = [
        f"Compiled: {result.success}",
        f"Reward:   {result.reward:+.2f}",
        f"Unsafe:   {result.unsafe_count}",
    ]
    if result.errors:
        lines.append("Errors:")
        for e in result.errors[:5]:
            lines.append(f"  [{e.code or '??'}] line {e.line}: {e.message}")
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

print("Building Gradio UI…", flush=True)

with gr.Blocks(title="C2Rust — Compiler-as-Oracle RL Demo") as demo:
    gr.Markdown(
        """
        # C2Rust — Teaching LLMs to Migrate C to Safe Rust
        **OpenEnv Hackathon India 2026 · Theme #2: Long-Horizon Planning**

        The Rust compiler (`rustc`) acts as the reward oracle — no human labels needed.
        """
    )

    # ── Training tab ─────────────────────────────────────────────────────────
    with gr.Tab("🚀 Training"):
        gr.Markdown(
            "Run GRPO fine-tuning directly in this Space. Logs stream live below. "
            "Make sure `WANDB_API_KEY`, `HF_TOKEN`, and `HF_REPO_ID` are set in Space secrets."
        )
        with gr.Row():
            config_tb  = gr.Textbox(value="configs/config.yaml", label="Config path", scale=3)
            status_tb  = gr.Textbox(value="Not started.", label="Status", interactive=False, scale=1)

        with gr.Row():
            train_btn = gr.Button("▶ Start Training", variant="primary")
            stop_btn  = gr.Button("⏹ Stop", variant="stop")
            refresh_btn = gr.Button("↻ Refresh status")

        wandb_md = gr.Markdown(
            f"**wandb project:** `c2rust-rl` — "
            f"[open dashboard](https://wandb.ai/home) after training starts."
        )
        log_stream = gr.Textbox(label="Training log", lines=25, interactive=False,
                                show_copy_button=True)

        train_btn.click(fn=start_training, inputs=[config_tb], outputs=[log_stream])
        stop_btn.click(fn=stop_training, outputs=[status_tb])
        refresh_btn.click(fn=training_status, outputs=[status_tb])

    # ── Repo demo tab ─────────────────────────────────────────────────────────
    with gr.Tab("🗂️ Repo Demo"):
        gr.Markdown("Convert a full C repository to safe Rust, module by module.")
        with gr.Row():
            repo_dd  = gr.Dropdown(AVAILABLE_REPOS,
                                   value=AVAILABLE_REPOS[0] if AVAILABLE_REPOS else None,
                                   label="Sample C Repository")
            retry_sl = gr.Slider(1, 5, value=2, step=1, label="Max retries per module")

        run_btn = gr.Button("🚀 Convert Repository", variant="primary", size="lg")

        with gr.Row():
            c_panel    = gr.Markdown(label="C Source",    value="*C source appears here*")
            rust_panel = gr.Markdown(label="Rust Output", value="*Rust output appears here*")

        with gr.Row():
            reward_html = gr.HTML(value="")
            status_html = gr.HTML(value="")

        log_box = gr.Textbox(label="Conversion Log", lines=15, interactive=False)

        run_btn.click(
            fn=run_repo_conversion,
            inputs=[repo_dd, retry_sl],
            outputs=[c_panel, rust_panel, reward_html, status_html, log_box],
        )

    # ── Compiler test tab ─────────────────────────────────────────────────────
    with gr.Tab("⚡ Compiler Test"):
        gr.Markdown("Paste any Rust code and see its reward score instantly. Works without GPU.")
        rust_input = gr.Textbox(label="Rust code", lines=20,
                                value='fn main() {\n    println!("Hello, Rust!");\n}')
        test_btn = gr.Button("Compile & Score", variant="secondary")
        test_out = gr.Textbox(label="Result", lines=8, interactive=False)
        test_btn.click(fn=quick_compile_test, inputs=[rust_input], outputs=[test_out])

    # ── About tab ─────────────────────────────────────────────────────────────
    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            ## How it works

            1. **Choose module** — C files ordered by `#include` dependencies (leaves first)
            2. **Rewrite** — DeepSeek-Coder-V2-Lite-Instruct (16B, 4-bit QLoRA) generates Rust
            3. **Compile** — `cargo build --message-format=json` gives structured errors
            4. **Reward** — shaped signal from −1.0 (total failure) to +1.0 (perfect, no unsafe)
            5. **Retry** — exact compiler errors fed back into the prompt

            ## Reward table

            | Outcome | Reward |
            |---|---|
            | Clean compile + tests pass + no `unsafe` | **+1.0** |
            | Compiles + tests pass + `unsafe` used | **+0.3** |
            | Ownership error (E0505, E0382) | **−0.3** |
            | Lifetime error (E0597, E0106) | **−0.5** |
            | Tests fail | **−0.8** |
            | Does not compile | **−1.0** |

            Each `unsafe` block: additional **−0.1**.

            ## Model & Training

            **DeepSeek-Coder-V2-Lite-Instruct** (16B MoE, ~2.4B active params per forward pass)
            fine-tuned with **GRPO** (Group Relative Policy Optimization) via HuggingFace TRL + Unsloth.
            Same RL algorithm DeepSeek used to train R1. Hardware: 1×L40S (48 GB VRAM).
            """
        )

print("Launching Gradio on port 7860…", flush=True)

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
    )
