"""
Pipeline debug script — validates the compiler/reward loop using the
DeepSeek-Coder-V2-Lite-Instruct agent before the full GRPO training run.

Usage:
    python test_pipeline.py                          # all 6 C programs
    python test_pipeline.py --file simple_malloc.c   # one program
    python test_pipeline.py --no-clippy              # skip clippy (faster)
    python test_pipeline.py --quantization 8bit      # less VRAM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.code_writer import CodeWriter
from tester.compiler import compile_and_evaluate, CompilerResult
from analyzer.static import parse_c_ast


# ── Per-file evaluation loop ──────────────────────────────────────────────────

def evaluate_file(
    c_path: Path,
    agent: CodeWriter,
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

    obs: dict = {
        "c_source": c_source,
        "c_ast": ast,
        "previous_rust": "",
        "compiler_errors": [],
        "error_count": 0,
        "retry_count": 0,
        "migration_context": {},
    }

    for attempt in range(1, max_retries + 1):
        rust_code = agent.generate(obs)

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
            break

        # Update obs for error-conditioned retry
        obs["previous_rust"] = rust_code
        obs["compiler_errors"] = [
            {"code": e.code, "line": e.line, "message": e.message}
            for e in result.errors
        ]
        obs["error_count"] = len(result.errors)
        obs["retry_count"] = attempt

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

_GREEN = "\033[32m"
_RED   = "\033[31m"
_RESET = "\033[0m"


def _reward_bar(reward: float, width: int = 20) -> str:
    filled = int((reward + 1.0) / 2.0 * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {reward:+.2f}"


def print_result(r: dict, verbose: bool = False) -> None:
    ok = r["final_success"] and r["final_test_passed"] is not False
    print(f"\n{'─'*60}")
    print(f"{_GREEN if ok else _RED}{'PASS' if ok else 'FAIL'}{_RESET}  {r['file']}")
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
    total    = len(results)
    compiled = sum(1 for r in results if r["final_success"])
    tests_ok = sum(1 for r in results if r["final_test_passed"] is True)
    unsafe_0 = sum(1 for r in results if r["final_unsafe_count"] == 0)
    mean_rew = sum(r["final_reward"] for r in results) / total if total else 0.0

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
    parser = argparse.ArgumentParser(description="C2Rust pipeline debug with DeepSeek-Coder")
    parser.add_argument("--file",         help="Single .c filename in data/c_programs/")
    parser.add_argument("--retries",      type=int, default=2, help="Max retry attempts per file")
    parser.add_argument("--no-clippy",    action="store_true",  help="Skip clippy check")
    parser.add_argument("--verbose",      action="store_true",  help="Print generated Rust code")
    parser.add_argument("--data-dir",     default="data/c_programs")
    parser.add_argument("--output-dir",   default="generated_rust")
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit"])
    parser.add_argument("--no-unsloth",   action="store_true",  help="Force plain transformers")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    c_files  = [data_dir / args.file] if args.file else sorted(data_dir.glob("*.c"))
    if not c_files:
        print(f"No .c files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DeepSeek-Coder-V2-Lite-Instruct …")
    agent = CodeWriter(
        use_unsloth=not args.no_unsloth,
        quantization=args.quantization,
    )

    print(f"Files      : {[f.name for f in c_files]}")
    print(f"Clippy     : {not args.no_clippy}")
    print(f"Retries    : {args.retries}")
    print(f"Output dir : {out_dir.resolve()}")

    results = []
    for c_path in c_files:
        print(f"\n→ Evaluating {c_path.name} …", flush=True)
        try:
            r = evaluate_file(c_path, agent, run_clippy=not args.no_clippy, max_retries=args.retries)
            results.append(r)
            print_result(r, verbose=args.verbose)

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
