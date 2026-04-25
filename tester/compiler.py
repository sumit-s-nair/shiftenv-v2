"""
Rust compiler bridge: writes Rust code to a temp file, invokes rustc with
JSON error output, parses the result into a structured reward signal.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CompilerError:
    message: str
    code: Optional[str]  # e.g. "E0505" (use-after-move)
    level: str           # "error" | "warning" | "note"
    line: Optional[int]
    column: Optional[int]
    filename: Optional[str]
    rendered: str        # human-readable rustc text

    @classmethod
    def from_json(cls, obj: dict) -> "CompilerError":
        spans = obj.get("spans", [])
        primary = next((s for s in spans if s.get("is_primary")), spans[0] if spans else {})
        return cls(
            message=obj.get("message", ""),
            code=(obj.get("code") or {}).get("code"),
            level=obj.get("level", "error"),
            line=primary.get("line_start"),
            column=primary.get("column_start"),
            filename=primary.get("file_name"),
            rendered=obj.get("rendered", ""),
        )


@dataclass
class CompilerResult:
    success: bool
    errors: list[CompilerError]
    reward: float
    unsafe_count: int
    test_passed: Optional[bool]  # None if not tested
    clippy_clean: Optional[bool]
    binary_path: Optional[str]
    raw_stderr: str
    error_type: str              # dominant error category
    error_line: Optional[int]

    @property
    def dominant_error_code(self) -> Optional[str]:
        for e in self.errors:
            if e.code:
                return e.code
        return None


# ── Error categorisation ────────────────────────────────────────────────────

_OWNERSHIP_CODES = {"E0505", "E0506", "E0507", "E0508", "E0382", "E0384"}
_LIFETIME_CODES  = {"E0597", "E0598", "E0106", "E0261", "E0262"}

def _categorise_errors(errors: list[CompilerError]) -> str:
    codes = {e.code for e in errors if e.code}
    if codes & _LIFETIME_CODES:
        return "lifetime_error"
    if codes & _OWNERSHIP_CODES:
        return "ownership_error"
    if errors:
        return "compile_error"
    return "none"


# ── Unsafe block counting ───────────────────────────────────────────────────

def _count_unsafe(rust_code: str) -> int:
    return len(re.findall(r"\bunsafe\s*\{", rust_code))


# ── Reward computation ──────────────────────────────────────────────────────

def _compute_reward(
    success: bool,
    errors: list[CompilerError],
    unsafe_count: int,
    test_passed: Optional[bool],
    clippy_clean: Optional[bool],
    unsafe_penalty_per_block: float = -0.1,
    performance_bonus: float = 0.2,
    clippy_bonus: float = 0.1,
) -> float:

    if not success:
        error_type = _categorise_errors(errors)
        if error_type == "ownership_error":
            return -0.3
        if error_type == "lifetime_error":
            return -0.5
        return -1.0

    # Compiled successfully
    if test_passed is False:
        base = -0.8
    elif unsafe_count > 0:
        base = 0.3
    else:
        base = 1.0

    # Cumulative unsafe penalty
    base += unsafe_count * unsafe_penalty_per_block

    # Clippy bonus (only for otherwise-good translations)
    if clippy_clean and unsafe_count == 0 and test_passed is not False:
        base += clippy_bonus

    return max(-1.0, min(1.0, base))


# ── Clippy runner ───────────────────────────────────────────────────────────

def _run_clippy(src_path: str, timeout: int = 30) -> bool:
    """Return True if clippy reports no warnings on the given source file."""
    try:
        result = subprocess.run(
            ["clippy-driver", "--edition", "2021", src_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0 and "warning" not in result.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── C reference runner ──────────────────────────────────────────────────────

def _run_binary(binary_path: str, timeout: int = 10) -> Optional[str]:
    try:
        r = subprocess.run([binary_path], capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return None


# ── Main public API ─────────────────────────────────────────────────────────

def compile_and_evaluate(
    rust_code: str,
    reference_output: Optional[str] = None,
    timeout: int = 30,
    run_clippy: bool = True,
    unsafe_penalty_per_block: float = -0.1,
    clippy_bonus: float = 0.1,
) -> CompilerResult:
    """
    Write *rust_code* to a temp file, compile with rustc, parse JSON errors,
    optionally run the binary and compare output against *reference_output*.

    Returns a fully-populated CompilerResult.
    """
    unsafe_count = _count_unsafe(rust_code)

    with tempfile.TemporaryDirectory(prefix="c2rust_") as tmpdir:
        src_path = os.path.join(tmpdir, "main.rs")
        bin_path = os.path.join(tmpdir, "main")

        Path(src_path).write_text(rust_code, encoding="utf-8")

        # ── Compile ──────────────────────────────────────────────────────────
        compile_cmd = [
            "rustc",
            "--edition", "2021",
            "--error-format=json",
            "-o", bin_path,
            src_path,
        ]
        try:
            proc = subprocess.run(
                compile_cmd,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return CompilerResult(
                success=False, errors=[], reward=-1.0, unsafe_count=unsafe_count,
                test_passed=None, clippy_clean=None, binary_path=None,
                raw_stderr="<timeout>", error_type="compile_error", error_line=None,
            )
        except FileNotFoundError:
            return CompilerResult(
                success=False, errors=[], reward=-1.0, unsafe_count=unsafe_count,
                test_passed=None, clippy_clean=None, binary_path=None,
                raw_stderr="<rustc not found>", error_type="compile_error", error_line=None,
            )

        raw_stderr = proc.stderr

        # ── Parse JSON errors ────────────────────────────────────────────────
        errors: list[CompilerError] = []
        for line in raw_stderr.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("level") in ("error", "warning"):
                errors.append(CompilerError.from_json(obj))

        real_errors = [e for e in errors if e.level == "error"]
        success = proc.returncode == 0 and not real_errors

        # ── Test binary ──────────────────────────────────────────────────────
        test_passed: Optional[bool] = None
        binary_path_out: Optional[str] = None
        if success and os.path.exists(bin_path):
            # Copy binary out of tmpdir so caller can inspect if needed
            import shutil
            persistent_dir = tempfile.mkdtemp(prefix="c2rust_bin_")
            persistent_bin = os.path.join(persistent_dir, "main")
            shutil.copy2(bin_path, persistent_bin)
            os.chmod(persistent_bin, 0o755)
            binary_path_out = persistent_bin

            if reference_output is not None:
                actual = _run_binary(persistent_bin, timeout=10)
                if actual is not None:
                    test_passed = actual.strip() == reference_output.strip()

        # ── Clippy ───────────────────────────────────────────────────────────
        clippy_clean: Optional[bool] = None
        if success and run_clippy:
            clippy_clean = _run_clippy(src_path, timeout=timeout)

        # ── Reward ───────────────────────────────────────────────────────────
        reward = _compute_reward(
            success=success,
            errors=real_errors,
            unsafe_count=unsafe_count,
            test_passed=test_passed,
            clippy_clean=clippy_clean,
            unsafe_penalty_per_block=unsafe_penalty_per_block,
            clippy_bonus=clippy_bonus,
        )

        error_type = _categorise_errors(real_errors)
        error_line = real_errors[0].line if real_errors else None

        return CompilerResult(
            success=success,
            errors=real_errors,
            reward=reward,
            unsafe_count=unsafe_count,
            test_passed=test_passed,
            clippy_clean=clippy_clean,
            binary_path=binary_path_out,
            raw_stderr=raw_stderr,
            error_type=error_type,
            error_line=error_line,
        )
