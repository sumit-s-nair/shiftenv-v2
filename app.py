"""
HuggingFace Spaces Gradio demo for C2Rust repo migration.

Uses the fine-tuned (or base) DeepSeek-Coder-V2-Lite-Instruct model
to convert a full multi-file C repository to safe Rust module-by-module,
showing live compiler feedback and reward scores at each step.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import gradio as gr

from agent.code_writer import CodeWriter
from env.c2rust_repo_env import C2RustRepoEnv
from tester.compiler import compile_and_evaluate

# ── Agent (loaded once at startup) ───────────────────────────────────────────

_AGENT: CodeWriter | None = None
_AGENT_ERROR: str = ""

print("Loading DeepSeek-Coder-V2-Lite-Instruct …")
try:
    _AGENT = CodeWriter()
    print("Model ready.")
except RuntimeError as e:
    _AGENT_ERROR = str(e)
    print(f"WARNING: model not loaded — {_AGENT_ERROR}")
    print("The compiler test tab will still work. Switch this Space to GPU hardware to enable generation.")

# ── Constants ─────────────────────────────────────────────────────────────────

REPOS_DIR = "data/repos"
AVAILABLE_REPOS = [d.name for d in Path(REPOS_DIR).iterdir() if d.is_dir()]


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


# ── Main conversion logic ─────────────────────────────────────────────────────

def run_repo_conversion(repo_name: str, max_retries: int) -> Generator:
    """
    Generator yielding (c_panel, rust_panel, reward_html, status_html, log)
    after each module step — drives Gradio live updates.
    """
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

with gr.Blocks(title="C2Rust — Compiler-as-Oracle RL Demo") as demo:
    gr.Markdown(
        """
        # C2Rust — Teaching LLMs to Migrate C to Safe Rust
        **OpenEnv Hackathon India 2026 · Theme #2: Long-Horizon Planning**

        The Rust compiler (`rustc`) acts as the reward oracle — no human labels needed.
        Watch DeepSeek-Coder-V2-Lite convert an entire C repository module-by-module.
        """
    )

    with gr.Tab("🗂️ Repo Migration"):
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

    with gr.Tab("⚡ Quick Compiler Test"):
        gr.Markdown("Paste any Rust code and check if it compiles + see its reward score.")
        rust_input = gr.Textbox(label="Rust code", lines=20,
                               value='fn main() {\n    println!("Hello, Rust!");\n}')
        test_btn = gr.Button("Compile & Score", variant="secondary")
        test_out = gr.Textbox(label="Result", lines=8, interactive=False)
        test_btn.click(fn=quick_compile_test, inputs=[rust_input], outputs=[test_out])

    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            ## How it works

            1. **Choose module** — C files ordered by `#include` dependencies (leaves first)
            2. **Rewrite** — DeepSeek-Coder-V2-Lite-Instruct generates Rust
            3. **Compile** — `cargo build --message-format=json` gives structured errors
            4. **Reward** — shaped signal from −1.0 (total failure) to +1.0 (perfect, no unsafe)
            5. **Retry** — exact compiler errors are fed back into the prompt

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

            ## Model

            **DeepSeek-Coder-V2-Lite-Instruct** (16B MoE, ~2.4B active params)
            loaded in 4-bit QLoRA via Unsloth. Fine-tuned with **GRPO** on an L40S (48 GB).
            """
        )

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
    )
