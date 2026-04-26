"""Reward function for C-to-Rust translation quality via compiler feedback.

Used as the reward signal during GRPO training. Scores generated Rust code on:
  - Compilation success (primary signal, 70% weight)
  - Warning count (secondary, small penalty per warning)
  - Unsafe block count (penalise for unsafe{})
  - Potential panic sites (.unwrap()/.expect())

Return value is a float in [0, 1]; higher is better.
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field


_CARGO_TOML = """\
[package]
name = "reward_check"
version = "0.1.0"
edition = "2021"

[dependencies]
"""

# Per-error penalty used for partial credit when compilation fails.
# 8+ errors → score floors at 0.
_ERROR_PENALTY = 0.12

# Per-unit penalties applied after the compilation score.
_WARN_PENALTY   = 0.05   # per warning  (capped at 5)
_UNSAFE_PENALTY = 0.10   # per unsafe{} (capped at 3)
_UNWRAP_PENALTY = 0.02   # per .unwrap/.expect (capped at 5)


@dataclass
class RewardInfo:
    compilation_score: float = 0.0
    error_count:       int   = 0
    warning_count:     int   = 0
    unsafe_count:      int   = 0
    unwrap_count:      int   = 0
    errors:            list  = field(default_factory=list)
    total:             float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_project(tmpdir: str, rust_code: str) -> None:
    """Write a minimal Cargo project.

    Any `use crate::X` references get stub modules so the checker flags real
    errors rather than cascading "unresolved module" noise.
    """
    src_dir = os.path.join(tmpdir, "src")
    os.makedirs(src_dir, exist_ok=True)

    with open(os.path.join(tmpdir, "Cargo.toml"), "w") as f:
        f.write(_CARGO_TOML)

    deps = set(re.findall(r"use\s+crate::(\w+)", rust_code))
    mod_header = "\n".join(f"pub mod {d};" for d in deps)

    entry = "main.rs" if re.search(r"\bfn\s+main\s*\(", rust_code) else "lib.rs"
    full_code = (mod_header + "\n\n" if mod_header else "") + rust_code

    with open(os.path.join(src_dir, entry), "w",encoding="utf-8") as f:
        f.write(full_code)

    for dep in deps:
        with open(os.path.join(src_dir, f"{dep}.rs"), "w") as f:
            f.write("// stub\n")


def _run_cargo_check(tmpdir: str, timeout: int) -> tuple[bool | None, str]:
    """Run `cargo check` and return (success, stderr).

    Returns (None, reason) when cargo is unavailable or times out.
    """
    try:
        result = subprocess.run(
            ["cargo", "check", "--quiet"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except FileNotFoundError:
        return None, "cargo_not_found"


def _parse_diagnostics(stderr: str) -> tuple[int, int, list[str]]:
    """Count distinct errors and warnings from cargo stderr text."""
    errors: list[str] = []
    warning_count = 0

    for line in stderr.splitlines():
        s = line.lstrip()
        if s.startswith("error[") or (
            s.startswith("error")
            and "aborting" not in s
            and "could not compile" not in s
        ):
            errors.append(s)
        elif s.startswith("warning[") or (
            s.startswith("warning") and "warning(s)" not in s
        ):
            warning_count += 1

    return len(errors), warning_count, errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward(
    rust_code: str,
    module_name: str = "module",
    timeout: int = 30,
) -> tuple[float, RewardInfo]:
    """Evaluate generated Rust code and return (reward ∈ [0,1], info).

    Args:
        rust_code:   Generated Rust source to evaluate.
        module_name: Module name (used for context in future extensions).
        timeout:     Seconds before cargo check is killed.
    """
    info = RewardInfo()

    if not rust_code.strip():
        return 0.0, info

    # --- Static analysis (compiler-free) ---
    info.unsafe_count = len(re.findall(r"\bunsafe\s*\{", rust_code))
    info.unwrap_count = len(re.findall(r"\.(unwrap|expect)\s*\(", rust_code))

    # --- Compilation check ---
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_project(tmpdir, rust_code)
        success, stderr = _run_cargo_check(tmpdir, timeout)

    if success is None:
        # Compiler unavailable — give neutral score so static penalties still apply.
        info.compilation_score = 0.5
    elif success:
        info.compilation_score = 1.0
        _, info.warning_count, _ = _parse_diagnostics(stderr)
    else:
        info.error_count, info.warning_count, info.errors = _parse_diagnostics(stderr)
        # Partial credit: each additional error subtracts _ERROR_PENALTY, floor 0.
        info.compilation_score = max(0.0, 1.0 - info.error_count * _ERROR_PENALTY)

    # --- Aggregate ---
    raw = (
        info.compilation_score
        - _WARN_PENALTY   * min(info.warning_count, 5)
        - _UNSAFE_PENALTY * min(info.unsafe_count,  3)
        - _UNWRAP_PENALTY * min(info.unwrap_count,   5)
    )
    info.total = max(0.0, min(1.0, raw))
    return info.total, info


# ---------------------------------------------------------------------------
# Quick CLI for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Score a Rust file with the reward function.")
    parser.add_argument("file", help="Path to .rs file")
    parser.add_argument("--timeout", type=int, default=30)
    a = parser.parse_args()

    with open(a.file) as f:
        code = f.read()

    score, info = compute_reward(code, timeout=a.timeout)
    print(f"Reward : {score:.4f}")
    print(f"Compile: {info.compilation_score:.2f}  errors={info.error_count}  warnings={info.warning_count}")
    print(f"Unsafe : {info.unsafe_count}  Unwraps: {info.unwrap_count}")
    if info.errors:
        print("Errors:")
        for e in info.errors[:10]:
            print(f"  {e}")
    sys.exit(0 if score >= 0.9 else 1)
