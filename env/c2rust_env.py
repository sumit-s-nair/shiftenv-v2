"""OpenEnv wrapper for the C-to-Rust migration task."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from reward import compute_reward


class C2RustEnv:
    """Gym/OpenEnv style wrapper around repo-native reward logic."""

    metadata = {"render_modes": []}

    def __init__(self, data_dir: Optional[str] = None, max_retries: int = 5):
        self.max_retries = max(1, int(max_retries))
        self._rng = random.Random()

        self.data_dir, self._c_files = self._discover_c_files(data_dir)

        # Episode state
        self._current_file: Optional[Path] = None
        self._c_source = ""
        self._c_ast: Dict[str, Any] = {}
        self._retry_count = 0
        self._done = False

    @staticmethod
    def _discover_c_files(data_dir: Optional[str]) -> Tuple[Path, list[Path]]:
        candidates: list[Path] = []
        if data_dir:
            candidates.append(Path(data_dir))

        # Prefer this repository's default source tree first.
        candidates.extend([
            Path("c_migration_test_3"),
            Path("data/c_programs"),
            Path("."),
        ])

        seen: set[Path] = set()
        for candidate in candidates:
            root = candidate.expanduser().resolve()
            if root in seen or not root.exists() or not root.is_dir():
                continue
            seen.add(root)

            c_files = [
                f
                for f in root.rglob("*.c")
                if ".git" not in f.parts and "__pycache__" not in f.parts
            ]
            if c_files:
                return root, sorted(c_files)

        fallback = Path(data_dir).expanduser().resolve() if data_dir else Path(".").resolve()
        return fallback, []

    @staticmethod
    def _extract_action_text(action: Any) -> str:
        if isinstance(action, str):
            return action
        if isinstance(action, dict):
            for key in ("rust_code", "text", "output", "answer"):
                value = action.get(key)
                if isinstance(value, str):
                    return value
        return ""

    @staticmethod
    def _analyze_c_source(c_source: str) -> Dict[str, Any]:
        # Lightweight parser fallback: avoids hard dependency on libclang at import time.
        include_matches = re.findall(r'^\s*#include\s+["<]([^">]+)[">]', c_source, flags=re.MULTILINE)
        function_matches = re.findall(
            r"^\s*(?:static\s+)?(?:inline\s+)?[\w\*\s]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{",
            c_source,
            flags=re.MULTILINE,
        )

        return {
            "line_count": len(c_source.splitlines()),
            "includes": sorted(set(include_matches)),
            "functions": sorted(set(function_matches)),
        }

    def _refresh_files(self) -> None:
        self.data_dir, self._c_files = self._discover_c_files(str(self.data_dir))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        del options

        if seed is not None:
            self._rng.seed(seed)

        if not self._c_files:
            self._refresh_files()
        if not self._c_files:
            raise FileNotFoundError(
                f"No .c files found under {self.data_dir}. "
                "Set data_dir to your C source root (for this repo, c_migration_test_3)."
            )

        self._current_file = self._rng.choice(self._c_files)
        self._c_source = self._current_file.read_text(encoding="utf-8", errors="ignore")
        self._c_ast = self._analyze_c_source(self._c_source)

        self._retry_count = 0
        self._done = False

        obs = {
            "c_source": self._c_source,
            "c_ast": self._c_ast,
            "previous_rust": "",
            "compiler_errors": [],
            "retry_count": self._retry_count,
        }
        info = {
            "file": str(self._current_file),
            "source_root": str(self.data_dir),
        }
        return obs, info

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode finished, call reset() before step().")

        if self._current_file is None:
            raise RuntimeError("Environment not initialized, call reset() before step().")

        rust_code = self._extract_action_text(action)
        module_name = self._current_file.stem

        reward, reward_info = compute_reward(rust_code, module_name=module_name)
        success = bool(reward_info.error_count == 0 and reward_info.compilation_score >= 1.0)
        errors = reward_info.errors

        self._retry_count += 1

        terminated = success
        truncated = self._retry_count >= self.max_retries
        if terminated or truncated:
            self._done = True

        obs = {
            "c_source": self._c_source,
            "c_ast": self._c_ast,
            "previous_rust": rust_code,
            "compiler_errors": errors,
            "retry_count": self._retry_count,
        }

        info = {
            "module": self._current_file.name,
            "success": success,
            "unsafe_count": reward_info.unsafe_count,
            "warning_count": reward_info.warning_count,
            "error_count": reward_info.error_count,
            "errors": errors,
            "compilation_score": reward_info.compilation_score,
        }

        return obs, reward, terminated, truncated, info

    @property
    def observation_space(self) -> dict:
        return {"type": "dict", "description": "C source, static summary, and compiler feedback"}

    @property
    def action_space(self) -> dict:
        return {"type": "text", "description": "Generated Rust code as string"}

    def close(self) -> None:
        return None
