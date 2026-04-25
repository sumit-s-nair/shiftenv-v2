"""
HuggingFace Spaces Gradio demo for C2Rust repo migration.

Converts a full multi-file C repository to safe Rust module-by-module,
showing live compiler feedback and reward scores at each step.

Set OPENAI_API_KEY in your Space secrets before running.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Generator

import gradio as gr

from env.c2rust_repo_env import C2RustRepoEnv
from tester.compiler import compile_and_evaluate
from analyzer.static import parse_c_ast

# ── Constants ─────────────────────────────────────────────────────────────────

REPOS_DIR = "data/repos"
AVAILABLE_REPOS = [d.name for d in Path(REPOS_DIR).iterdir() if d.is_dir()]

_SYSTEM_PROMPT = """\
You are an expert Rust programmer specialising in migrating C code to safe, idiomatic Rust.

Rules:
- Do NOT use unsafe blocks. Use Box, Vec, HashMap, iterators, and enums instead of raw pointers.
- Generate idiomatic Rust with proper ownership — no .clone() spam.
- The output must be a complete, compilable Rust module (no fn main unless this is main.rs).
- If converting a module that other modules depend on, declare public types and functions with pub.
- Return ONLY the Rust source inside a single ```rust ... ``` block.\
"""

# ── OpenAI helper ─────────────────────────────────────────────────────────────

def _openai_translate(
    c_source: str,
    module_name: str,
    converted_context: dict[str, str],
    errors: list[dict],
    prev_rust: str,
    model: str,
) -> str:
    from openai import OpenAI
    client = OpenAI()

    context_block = ""
    if converted_context:
        snippets = []
        for stem, code in converted_context.items():
            snippets.append(f"// Already converted: {stem}.rs\n" + code[:600] + ("\n// ...\n" if len(code) > 600 else ""))
        context_block = "\n\nAlready converted modules (for type/API reference):\n" + "\n---\n".join(snippets)

    error_block = ""
    if errors:
        lines = ["\nCompiler errors from previous attempt (fix these):"]
        for e in errors[:8]:
            lines.append(f"  [{e.get('code','??')}] line {e.get('line','?')}: {e.get('message','')}")
        if prev_rust:
            lines.append(f"\nPrevious attempt:\n```rust\n{prev_rust}\n```")
        error_block = "\n".join(lines)

    user_msg = (
        f"Translate this C module ({module_name}.c) to safe Rust:{context_block}{error_block}\n\n"
        f"```c\n{c_source}\n```\n\nRust translation:"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"```rust\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


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
    total = info.get("total_modules", 1)
    pct = int(converted / total * 100)
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

def run_repo_conversion(
    repo_name: str,
    model: str,
    max_retries: int,
) -> Generator:
    """
    Generator that yields (c_panel, rust_panel, reward_html, status_html, log)
    after each module step — drives the Gradio live update.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        yield "", "", "", "", "❌ OPENAI_API_KEY not set in environment."
        return

    env = C2RustRepoEnv(repos_dir=REPOS_DIR, max_retries_per_module=max_retries)
    obs, info = env.reset(repo_name=repo_name)

    log_lines: list[str] = [f"=== Repo: {repo_name} ===",
                             f"Conversion order: {info['conversion_order']}"]
    all_rust: dict[str, str] = {}

    while not env._done:
        module = obs["current_module"]
        c_src  = obs["current_c_source"]
        errors = obs["compiler_errors"]
        prev   = obs["previous_rust"]

        log_lines.append(f"\n→ Converting {module}.c  (retry {obs['retry_count']})")
        yield (
            f"```c\n// {module}.c\n{c_src}\n```",
            "(generating...)",
            "",
            f"Converting <b>{module}</b>...",
            "\n".join(log_lines),
        )

        rust_code = _openai_translate(
            c_source=c_src,
            module_name=module,
            converted_context=dict(all_rust),
            errors=errors,
            prev_rust=prev,
            model=model,
        )

        obs, reward, terminated, truncated, info = env.step({"rust_code": rust_code})

        if info["module_success"]:
            all_rust[module] = rust_code
            log_lines.append(f"  ✅ {module} compiled  reward={reward:+.2f}")
        else:
            log_lines.append(f"  ❌ {module} failed    reward={reward:+.2f}")
            if info.get("error_type"):
                log_lines.append(f"     error_type={info['error_type']}  line={info.get('error_line')}")

        yield (
            f"```c\n// {module}.c\n{c_src}\n```",
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
        f"```c\n// {repo_name} — original C\n(see left panel for each module)\n```",
        f"```rust\n// {repo_name} — final Rust\n{final_rust}\n```",
        _reward_badge(reward if 'reward' in dir() else 0.0),
        _status_html(info if 'info' in dir() else {}),
        "\n".join(log_lines),
    )


# ── Single-file quick test ────────────────────────────────────────────────────

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

with gr.Blocks(title="C2Rust — Compiler-as-Oracle RL Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # C2Rust — Teaching LLMs to Migrate C to Safe Rust
        **OpenEnv Hackathon India 2026 · Theme #2: Long-Horizon Planning**

        The Rust compiler (`rustc`) acts as the reward oracle — no human labels needed.
        Watch the agent convert an entire C repository module-by-module.
        """
    )

    with gr.Tab("🗂️ Repo Migration"):
        with gr.Row():
            repo_dd   = gr.Dropdown(AVAILABLE_REPOS, value=AVAILABLE_REPOS[0] if AVAILABLE_REPOS else None,
                                    label="Sample C Repository")
            model_dd  = gr.Dropdown(["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                                    value="gpt-4o", label="OpenAI Model")
            retry_sl  = gr.Slider(1, 5, value=2, step=1, label="Max retries per module")

        run_btn = gr.Button("🚀 Convert Repository", variant="primary", size="lg")

        with gr.Row():
            c_panel    = gr.Markdown(label="C Source", value="*C source appears here*")
            rust_panel = gr.Markdown(label="Rust Output", value="*Rust output appears here*")

        with gr.Row():
            reward_html = gr.HTML(value="")
            status_html = gr.HTML(value="")

        log_box = gr.Textbox(label="Conversion Log", lines=15, interactive=False)

        run_btn.click(
            fn=run_repo_conversion,
            inputs=[repo_dd, model_dd, retry_sl],
            outputs=[c_panel, rust_panel, reward_html, status_html, log_box],
        )

    with gr.Tab("⚡ Quick Compiler Test"):
        gr.Markdown("Paste any Rust code and check if it compiles + see its reward score.")
        rust_input = gr.Code(language="rust", label="Rust code", lines=20,
                             value='fn main() {\n    println!("Hello, Rust!");\n}')
        test_btn   = gr.Button("Compile & Score", variant="secondary")
        test_out   = gr.Textbox(label="Result", lines=8, interactive=False)
        test_btn.click(fn=quick_compile_test, inputs=[rust_input], outputs=[test_out])

    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            ## How it works

            1. **Choose module** — select a C file from the repo (ordered by dependencies)
            2. **Rewrite** — GPT-4o (or fine-tuned DeepSeek-Coder-V2-Lite) generates Rust
            3. **Compile** — `cargo build --message-format=json` gives structured errors
            4. **Reward** — shaped signal from −1.0 (total failure) to +1.0 (perfect, no unsafe)
            5. **Retry** — errors are fed back into the prompt for error-conditioned re-generation

            ## Reward table

            | Outcome | Reward |
            |---|---|
            | Clean compile + tests pass + no `unsafe` | **+1.0** |
            | Compiles + tests pass + `unsafe` used | **+0.3** |
            | Ownership error (E0505, E0382) | **−0.3** |
            | Lifetime error (E0597, E0106) | **−0.5** |
            | Tests fail | **−0.8** |
            | Does not compile | **−1.0** |

            Each `unsafe` block costs an additional **−0.1**.

            ## Training

            The RL fine-tuning uses **GRPO** (Group Relative Policy Optimization) via HuggingFace TRL
            on **DeepSeek-Coder-V2-Lite-Instruct** (16B, 4-bit QLoRA) running on a single **L40S** (48 GB).
            """
        )

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
