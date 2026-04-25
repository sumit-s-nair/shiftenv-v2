"""
Repo-level OpenRL environment for C→Rust migration.

Handles both flat repos (kvstore) and nested-directory repos (repo/1).
Nested C files like core/engine.c become flat Rust module IDs: "core_engine".
The Cargo project mirrors this flat layout in src/.

Episode flow:
  reset()  → loads a C repo, determines conversion order by dependency analysis
  step()   → agent provides Rust for current module; cargo build; reward
  done     → all modules converted + tests pass, or retry budget exhausted
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

from analyzer.static import parse_c_ast
from memory.store import MigrationStore
from tester.cargo_compiler import CargoResult, compile_repo_and_evaluate


Observation = dict[str, Any]
Action      = dict[str, str]
Info        = dict[str, Any]


# ── Module loading ────────────────────────────────────────────────────────────

def _load_c_modules(repo_path: Path) -> dict[str, dict]:
    """
    Recursively load all .c files from a repo.

    Returns:
        {module_id: {"source": str, "rel_path": str, "dir": str, "stem": str}}

    module_id is a flat, filesystem-safe name:
      - flat/foo.c            → "foo"
      - nested/core/engine.c  → "core_engine"
    """
    modules: dict[str, dict] = {}
    for c_path in sorted(repo_path.rglob("*.c")):
        rel = c_path.relative_to(repo_path)
        parts = list(rel.with_suffix("").parts)   # e.g. ["core", "engine"]
        module_id = "_".join(parts)                # "core_engine"
        stem = parts[-1]                            # "engine"
        modules[module_id] = {
            "source":   c_path.read_text(encoding="utf-8"),
            "rel_path": str(rel),
            "dir":      str(rel.parent),
            "stem":     stem,
        }
    return modules


# ── Dependency analysis ───────────────────────────────────────────────────────

def _header_to_module_id(include_path: str, file_dir: str, all_ids: set[str]) -> Optional[str]:
    """
    Resolve a #include "..." path to a module_id.

    Examples:
      "engine.h"           (from core/engine.c)  → "core_engine" (same dir)
      "core/engine.h"      (from main.c)          → "core_engine"
      "../module1/proc.h"  (from core/engine.c)   → "module1_proc"
    """
    p = Path(include_path).with_suffix("")          # remove .h
    parts = list(p.parts)

    # Handle relative paths: resolve against the file's directory
    if parts[0] == "..":
        base = Path(file_dir).parent
        parts = parts[1:]
    elif parts[0] == ".":
        base = Path(file_dir)
        parts = parts[1:]
    else:
        # May be relative to repo root OR same directory
        base = Path(file_dir)

    resolved = "_".join((list(base.parts) + parts)) if base != Path(".") else "_".join(parts)

    if resolved in all_ids:
        return resolved

    # Fallback: match by stem only (last component)
    stem = parts[-1]
    candidates = [mid for mid in all_ids if mid == stem or mid.endswith("_" + stem)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _build_dep_graph(modules: dict[str, dict]) -> dict[str, list[str]]:
    all_ids = set(modules.keys())
    graph: dict[str, list[str]] = {mid: [] for mid in all_ids}
    for mid, info in modules.items():
        for m in re.finditer(r'#include\s+"([^"]+)"', info["source"]):
            dep = _header_to_module_id(m.group(1), info["dir"], all_ids)
            if dep and dep != mid:
                graph[mid].append(dep)
    return graph


def _topological_order(graph: dict[str, list[str]]) -> list[str]:
    """Kahn's algorithm — dependencies before dependents."""
    in_deg = {n: 0 for n in graph}
    for deps in graph.values():
        for d in deps:
            in_deg[d] = in_deg.get(d, 0)   # ensure key present
    # recount properly
    in_deg = {n: len(deps) for n, deps in graph.items()}

    queue = [n for n, d in in_deg.items() if d == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for n, deps in graph.items():
            if node in deps:
                in_deg[n] -= 1
                if in_deg[n] == 0:
                    queue.append(n)
    for n in graph:
        if n not in order:
            order.append(n)

    # Main entry always last
    mains = [n for n in order if n == "main" or n.endswith("_main")]
    for m in mains:
        order.remove(m)
        order.append(m)
    return order


# ── Environment ───────────────────────────────────────────────────────────────

class C2RustRepoEnv:
    """
    Gym-style environment for whole-repo C→Rust migration.

    Converts one module per step in dependency order.
    Cargo builds the growing partial project after each conversion.
    Stubs are generated for unconverted modules so cargo doesn't fail on missing symbols.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        repos_dir: str = "data/repos",
        max_retries_per_module: int = 5,
        timeout: int = 120,
        store_path: str = "migration_state_repo.json",
        seed: Optional[int] = None,
    ) -> None:
        self.repos_dir = Path(repos_dir)
        self.max_retries = max_retries_per_module
        self.timeout = timeout
        self.store = MigrationStore(store_path)

        import random
        self._rng = random.Random(seed)

        # Episode state
        self._repo_name: str = ""
        self._repo_path: Path = Path(".")
        self._modules: dict[str, dict] = {}        # id → {source, rel_path, dir, stem}
        self._asts: dict[str, dict] = {}
        self._order: list[str] = []
        self._converted: dict[str, str] = {}       # id → rust_code
        self._current_idx: int = 0
        self._retry_count: int = 0
        self._previous_rust: str = ""
        self._compiler_errors: list[dict] = []
        self._done: bool = False
        self._stdin_input: str = ""
        self._reference_output: str = ""
        self._project_name: str = "c2rust_project"

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        repo_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> tuple[Observation, Info]:
        if seed is not None:
            self._rng.seed(seed)

        repos = [d for d in self.repos_dir.iterdir() if d.is_dir()]
        if not repos:
            raise FileNotFoundError(f"No repos in {self.repos_dir}")
        repo = self.repos_dir / repo_name if repo_name else self._rng.choice(repos)

        self._repo_path = repo
        self._repo_name = repo.name
        self._project_name = re.sub(r"[^a-z0-9_]", "_", repo.name.lower())

        self._modules = _load_c_modules(repo)
        self._asts    = {mid: parse_c_ast(info["source"]) for mid, info in self._modules.items()}

        graph       = _build_dep_graph(self._modules)
        self._order = _topological_order(graph)

        test_in  = repo / "tests" / "test_input.txt"
        test_out = repo / "tests" / "expected_output.txt"
        self._stdin_input      = test_in.read_text(encoding="utf-8")  if test_in.exists()  else ""
        self._reference_output = test_out.read_text(encoding="utf-8") if test_out.exists() else ""

        self._converted = {}
        self._current_idx = 0
        self._retry_count = 0
        self._previous_rust = ""
        self._compiler_errors = []
        self._done = False

        info: Info = {
            "repo": self._repo_name,
            "conversion_order": self._order,
            "total_modules": len(self._order),
            "action": "reset",
        }
        return self._build_obs(), info

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, Info]:
        if self._done:
            raise RuntimeError("Episode finished — call reset() first")

        rust_code: str = action.get("rust_code", "")
        current_id = self._order[self._current_idx]

        candidate  = {**self._converted, current_id: rust_code}
        pending    = self._order[self._current_idx + 1:]

        result: CargoResult = compile_repo_and_evaluate(
            converted=candidate,
            stubs=pending,
            project_name=self._project_name,
            stdin_input=self._stdin_input,
            reference_output=self._reference_output or None,
            timeout=self.timeout,
        )

        self._compiler_errors = [
            {"message": e.message, "code": e.code, "level": e.level,
             "line": e.line, "column": e.column, "rendered": e.rendered}
            for e in result.errors
        ]
        self._previous_rust = rust_code
        self._retry_count  += 1

        if result.success:
            self._converted[current_id] = rust_code
            self._current_idx += 1
            self._retry_count  = 0
            self._previous_rust = ""
            self._compiler_errors = []

        self.store.update(
            module=f"{self._repo_name}/{current_id}",
            success=result.success,
            retry_count=self._retry_count,
            error_type=result.error_type,
            reward=result.reward,
            rust_code=rust_code if result.success else None,
        )

        all_done   = self._current_idx >= len(self._order)
        terminated = all_done and result.test_passed is not False
        truncated  = self._retry_count >= self.max_retries

        if terminated or truncated:
            self._done = True

        info: Info = {
            "repo": self._repo_name,
            "module": current_id,
            "rel_path": self._modules[current_id]["rel_path"],
            "module_success": result.success,
            "error_type": result.error_type,
            "error_line": result.error_line,
            "compiler_output": result.raw_output[:2000],
            "unsafe_count": result.unsafe_count,
            "test_passed": result.test_passed,
            "converted_count": len(self._converted),
            "total_modules": len(self._order),
        }
        return self._build_obs(), result.reward, terminated, truncated, info

    def close(self) -> None:
        self.store.save()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_obs(self) -> Observation:
        current_id = self._order[self._current_idx] if self._current_idx < len(self._order) else ""
        return {
            "repo_name":         self._repo_name,
            "module_ids":        list(self._modules.keys()),
            "c_sources":         {mid: info["source"] for mid, info in self._modules.items()},
            "c_asts":            self._asts,
            "conversion_order":  self._order,
            "current_module":    current_id,
            "current_c_source":  self._modules[current_id]["source"] if current_id else "",
            "current_rel_path":  self._modules[current_id]["rel_path"] if current_id else "",
            "converted_modules": dict(self._converted),
            "pending_modules":   self._order[self._current_idx:],
            "previous_rust":     self._previous_rust,
            "compiler_errors":   self._compiler_errors,
            "error_count":       len(self._compiler_errors),
            "retry_count":       self._retry_count,
            "migration_context": self.store.get_context(),
        }

    @property
    def current_module(self) -> str:
        return self._order[self._current_idx] if self._current_idx < len(self._order) else ""

    @property
    def reward_range(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"C2RustRepoEnv(repo={self._repo_name!r}, "
            f"progress={len(self._converted)}/{len(self._order)})"
        )
