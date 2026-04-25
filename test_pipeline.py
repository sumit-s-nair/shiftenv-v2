"""
Pipeline debug script — validates the compiler/reward loop with OpenAI as a
drop-in agent before committing to the expensive DeepSeek fine-tuning run.

Usage:
    export OPENAI_API_KEY=sk-...
    python test_pipeline.py                          # all 6 C programs
    python test_pipeline.py --file simple_malloc.c   # one program
    python test_pipeline.py --model gpt-4o-mini      # cheaper model
    python test_pipeline.py --no-clippy              # skip clippy (faster)
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

from tester.compiler import compile_and_evaluate, CompilerResult
from analyzer.static import parse_c_ast

# ── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert Rust programmer specialising in migrating C to safe, idiomatic Rust.

Rules:
- Do NOT use unsafe blocks. Use Box, Vec, slices, and iterators instead of raw pointers.
- The output must be a complete, compilable Rust program (include fn main if C has one).
- Return ONLY the Rust source inside a ```rust ... ``` block, nothing else.\
"""

def _build_prompt(c_source: str, errors: list[dict] | None = None, prev_rust: str = "") -> str:
    lines = [f"Translate this C program to safe, idiomatic Rust:\n\n```c\n{c_source}\n```"]
    if errors:
        lines.append("\nYour previous attempt had these compiler errors — fix them:")
        for e in errors[:8]:
            code = e.get("code") or "??"
            line = e.get("line") or "?"
            msg  = e.get("message", "")
            lines.append(f"  [{code}] line {line}: {msg}")
        if prev_rust:
            lines.append(f"\nPrevious attempt:\n```rust\n{prev_rust}\n```")
    lines.append("\nRust translation:")
    return "\n".join(lines)


# ── OpenAI call ───────────────────────────────────────────────────────────────

def _call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI
    import re

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    text = resp.choices[0].message.content or ""

    # Extract ```rust ... ``` block
    m = re.search(r"```rust\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


# ── Per-file evaluation loop (with one retry) ─────────────────────────────────

def evaluate_file(
    c_path: Path,
    model: str,
    run_clippy: bool,
    max_retries: int = 2,
) -> dict:
    c_source = c_path.read_text(encoding="utf-8")
    ref_path = c_path.with_suffix(".ref.txt")
    reference_output = ref_path.read_text(encoding="utf-8") if ref_path.exists() else None

    ast = parse_c_ast(c_source)

    rust_code = ""
    result: CompilerResult | None = None
    history: list[dict] = []

    for attempt in range(1, max_retries + 1):
        errors_for_prompt = (
            [{"code": e.code, "line": e.line, "message": e.message} for e in result.errors]
            if result and result.errors else None
        )
        prompt = _build_prompt(c_source, errors_for_prompt, rust_code)
        rust_code = _call_openai(prompt, model)

        result = compile_and_evaluate(
            rust_code=rust_code,
            reference_output=reference_output,
            run_clippy=run_clippy,
        )

        history.append({
            "attempt": attempt,
            "reward": result.reward,
            "success": result.success,
            "test_passed": result.test_passed,
            "unsafe_count": result.unsafe_count,
            "error_count": len(result.errors),
            "error_type": result.error_type,
            "error_line": result.error_line,
        })

        if result.success and result.test_passed is not False:
            break  # done

    return {
        "file": c_path.name,
        "ast_difficulty": ast["difficulty"]["score"],
        "ast_parser": ast["parser"],
        "unsafe_patterns": ast["unsafe_patterns"],
        "final_reward": result.reward,
        "final_success": result.success,
        "final_test_passed": result.test_passed,
        "final_unsafe_count": result.unsafe_count,
        "clippy_clean": result.clippy_clean,
        "attempts": len(history),
        "history": history,
        "rust_code": rust_code,
        "compiler_errors": [
            {"code": e.code, "line": e.line, "message": e.message, "rendered": e.rendered}
            for e in result.errors
        ],
    }


# ── Pretty printer ────────────────────────────────────────────────────────────

_REWARD_COLOUR = {
    True:  "\033[32m",  # green  — success
    False: "\033[31m",  # red    — failure
}
_RESET = "\033[0m"


def _reward_bar(reward: float, width: int = 20) -> str:
    filled = int((reward + 1.0) / 2.0 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {reward:+.2f}"


def print_result(r: dict, verbose: bool = False) -> None:
    ok = r["final_success"] and r["final_test_passed"] is not False
    colour = _REWARD_COLOUR[ok]
    status = "PASS" if ok else "FAIL"

    print(f"\n{'─'*60}")
    print(f"{colour}{status}{_RESET}  {r['file']}")
    print(f"  Reward   : {_reward_bar(r['final_reward'])}")
    print(f"  Compiled : {r['final_success']}")
    print(f"  Tests    : {r['final_test_passed']}")
    print(f"  Unsafe   : {r['final_unsafe_count']}  |  Clippy: {r['clippy_clean']}")
    print(f"  Attempts : {r['attempts']}")
    print(f"  AST diff : {r['ast_difficulty']} ({r['ast_parser']})")
    if r["unsafe_patterns"]:
        print(f"  Patterns : {', '.join(r['unsafe_patterns'])}")

    if r["history"]:
        print("  History  :", " → ".join(
            f"{'✓' if h['success'] else '✗'}{h['reward']:+.2f}" for h in r["history"]
        ))

    if not ok and r["compiler_errors"]:
        print("  Errors:")
        for e in r["compiler_errors"][:5]:
            print(f"    [{e['code'] or '??'}] line {e['line']}: {e['message'][:80]}")

    if verbose and r["rust_code"]:
        print("\n  Generated Rust:")
        for line in r["rust_code"].splitlines()[:40]:
            print(f"    {line}")
        if r["rust_code"].count("\n") > 40:
            print("    ...")


def print_summary(results: list[dict]) -> None:
    total     = len(results)
    compiled  = sum(1 for r in results if r["final_success"])
    tests_ok  = sum(1 for r in results if r["final_test_passed"] is True)
    unsafe_0  = sum(1 for r in results if r["final_unsafe_count"] == 0)
    mean_rew  = sum(r["final_reward"] for r in results) / total if total else 0.0

    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    print(f"  Files tested  : {total}")
    print(f"  Compile rate  : {compiled}/{total}  ({100*compiled//total}%)")
    print(f"  Test-pass rate: {tests_ok}/{total}  ({100*tests_ok//total}%)")
    print(f"  Unsafe-free   : {unsafe_0}/{total}  ({100*unsafe_0//total}%)")
    print(f"  Mean reward   : {mean_rew:+.3f}")
    print(f"{'═'*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="C2Rust pipeline debug with OpenAI")
    parser.add_argument("--file",    help="Single .c filename in data/c_programs/")
    parser.add_argument("--model",   default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--retries", type=int, default=2, help="Max retry attempts per file")
    parser.add_argument("--no-clippy", action="store_true", help="Skip clippy check")
    parser.add_argument("--verbose", action="store_true", help="Print generated Rust code")
    parser.add_argument("--data-dir", default="data/c_programs")
    parser.add_argument("--output-dir", default="generated_rust", help="Directory to save .rs output files")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    try:
        import openai  # noqa: F401
    except ImportError:
        print("ERROR: openai package not installed.  Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if args.file:
        c_files = [data_dir / args.file]
    else:
        c_files = sorted(data_dir.glob("*.c"))

    if not c_files:
        print(f"No .c files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model      : {args.model}")
    print(f"Files      : {[f.name for f in c_files]}")
    print(f"Clippy     : {not args.no_clippy}")
    print(f"Retries    : {args.retries}")
    print(f"Output dir : {out_dir.resolve()}")

    results = []
    for c_path in c_files:
        print(f"\n→ Evaluating {c_path.name} ...", flush=True)
        try:
            r = evaluate_file(c_path, args.model, run_clippy=not args.no_clippy, max_retries=args.retries)
            results.append(r)
            print_result(r, verbose=args.verbose)

            # Save generated Rust to <output_dir>/<stem>.rs
            if r["rust_code"]:
                rs_path = out_dir / c_path.with_suffix(".rs").name
                rs_path.write_text(r["rust_code"], encoding="utf-8")
                status = "✓" if r["final_success"] else "✗"
                print(f"  Saved     : {rs_path}  {status}")

        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
