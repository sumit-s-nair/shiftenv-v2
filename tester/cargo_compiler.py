"""
Cargo-based build bridge for whole-repo C→Rust migration.

Manages a temporary Cargo project, writes converted Rust modules into it,
compiles with `cargo build --message-format=json`, and parses structured output
into a shaped reward signal.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tester.compiler import CompilerError, _categorise_errors, _count_unsafe


# ── Cargo project manager ─────────────────────────────────────────────────────

class CargoProject:
    """
    A temporary Cargo binary project.  Call add_module() / add_stub() to
    populate src/, then build() to compile.
    """

    def __init__(self, project_name: str, tmpdir: str) -> None:
        self.project_name = project_name
        self.root = Path(tmpdir)
        self.src = self.root / "src"
        self.src.mkdir(parents=True, exist_ok=True)
        self._write_cargo_toml()

    def _write_cargo_toml(self) -> None:
        (self.root / "Cargo.toml").write_text(
            f'[package]\nname = "{self.project_name}"\nversion = "0.1.0"\nedition = "2021"\n',
            encoding="utf-8",
        )

    def add_module(self, stem: str, rust_code: str) -> None:
        """Write a converted Rust module (stem == 'main' → src/main.rs)."""
        path = self.src / ("main.rs" if stem == "main" else f"{stem}.rs")
        path.write_text(rust_code, encoding="utf-8")

    def add_stub(self, stem: str) -> None:
        """Write a minimal stub so cargo build doesn't fail on missing modules."""
        stub = (
            f"// auto-stub for unconverted module '{stem}'\n"
            "#![allow(dead_code, unused_variables, unused_imports)]\n\n"
            f"pub fn _{stem}_stub() {{}}\n"
        )
        (self.src / f"{stem}.rs").write_text(stub, encoding="utf-8")

    def build(self, timeout: int = 120) -> "CargoResult":
        try:
            proc = subprocess.run(
                ["cargo", "build", "--message-format=json"],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return CargoResult(
                success=False, errors=[], reward=-1.0, unsafe_count=0,
                test_passed=None, binary_path=None,
                raw_output="<timeout>", error_type="compile_error", error_line=None,
            )
        except FileNotFoundError:
            return CargoResult(
                success=False, errors=[], reward=-1.0, unsafe_count=0,
                test_passed=None, binary_path=None,
                raw_output="<cargo not found>", error_type="compile_error", error_line=None,
            )

        errors: list[CompilerError] = []
        for line in proc.stdout.splitlines() + proc.stderr.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # cargo --message-format=json wraps rustc diagnostics under "message"
            if obj.get("reason") == "compiler-message":
                msg = obj.get("message", {})
                if msg.get("level") in ("error", "warning"):
                    errors.append(CompilerError.from_json(msg))
            elif obj.get("level") in ("error", "warning"):
                errors.append(CompilerError.from_json(obj))

        real_errors = [e for e in errors if e.level == "error"]
        success = proc.returncode == 0

        binary = None
        if success:
            binary = str(self.root / "target" / "debug" / self.project_name)

        return CargoResult(
            success=success,
            errors=real_errors,
            reward=0.0,  # caller fills this in
            unsafe_count=0,
            test_passed=None,
            binary_path=binary,
            raw_output=proc.stdout + proc.stderr,
            error_type=_categorise_errors(real_errors),
            error_line=real_errors[0].line if real_errors else None,
        )

    def run(self, stdin_input: str = "", timeout: int = 10) -> Optional[str]:
        binary = self.root / "target" / "debug" / self.project_name
        if not binary.exists():
            return None
        try:
            r = subprocess.run(
                [str(binary)],
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return r.stdout
        except (subprocess.TimeoutExpired, PermissionError):
            return None


@dataclass
class CargoResult:
    success: bool
    errors: list[CompilerError]
    reward: float
    unsafe_count: int
    test_passed: Optional[bool]
    binary_path: Optional[str]
    raw_output: str
    error_type: str
    error_line: Optional[int]


# ── Public API ────────────────────────────────────────────────────────────────

def compile_repo_and_evaluate(
    converted: dict[str, str],       # stem → rust_code (already converted)
    stubs: list[str],                 # stems to fill with auto-stubs
    project_name: str = "c2rust_project",
    stdin_input: str = "",
    reference_output: Optional[str] = None,
    timeout: int = 120,
    unsafe_penalty_per_block: float = -0.1,
    clippy_bonus: float = 0.1,
) -> CargoResult:
    """
    Build a temporary Cargo project from *converted* modules + *stubs*,
    run the binary, compare to *reference_output*, compute reward.
    """
    tmpdir = tempfile.mkdtemp(prefix="c2rust_cargo_")
    try:
        proj = CargoProject(project_name, tmpdir)

        for stem, code in converted.items():
            proj.add_module(stem, code)
        for stem in stubs:
            proj.add_stub(stem)

        result = proj.build(timeout=timeout)

        # Count unsafe across all converted modules
        total_unsafe = sum(_count_unsafe(code) for code in converted.values())
        result.unsafe_count = total_unsafe

        # Run and test if build succeeded and no stubs remain
        if result.success and not stubs and stdin_input:
            actual = proj.run(stdin_input=stdin_input, timeout=10)
            if actual is not None and reference_output is not None:
                result.test_passed = actual.strip() == reference_output.strip()

        # Compute reward
        all_converted = not stubs
        result.reward = _repo_reward(result, all_converted, total_unsafe, unsafe_penalty_per_block)

        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _repo_reward(
    result: CargoResult,
    all_converted: bool,
    unsafe_count: int,
    unsafe_penalty: float,
) -> float:
    if not result.success:
        if result.error_type == "ownership_error":
            return -0.3
        if result.error_type == "lifetime_error":
            return -0.5
        return -1.0

    if not all_converted:
        # Partial build — reward for making progress without breaking things
        return 0.1 + unsafe_count * unsafe_penalty

    # Full project compiled
    if result.test_passed is False:
        base = -0.8
    elif unsafe_count > 0:
        base = 0.3
    else:
        base = 1.0

    base += unsafe_count * unsafe_penalty
    return max(-1.0, min(1.0, base))
