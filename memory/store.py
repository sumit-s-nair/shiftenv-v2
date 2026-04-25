"""
JSON-backed persistent store for migration state.

Tracks per-module progress and aggregated statistics used to build the
`migration_context` slice of every observation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


_DEFAULT_MODULE: dict[str, Any] = {
    "status": "pending",       # pending | in_progress | migrated | failed
    "retry_count": 0,
    "last_error_type": None,
    "best_reward": None,
    "rust_code": None,
}


class MigrationStore:
    def __init__(self, path: str = "migration_state.json") -> None:
        self.path = Path(path)
        self._data: dict[str, Any] = self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"modules": {}, "global": {"total": 0, "migrated": 0}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Module accessors ─────────────────────────────────────────────────────

    def _module(self, name: str) -> dict[str, Any]:
        mods = self._data.setdefault("modules", {})
        if name not in mods:
            mods[name] = dict(_DEFAULT_MODULE)
            self._data.setdefault("global", {})["total"] = len(mods)
        return mods[name]

    def is_migrated(self, name: str) -> bool:
        return self._module(name)["status"] == "migrated"

    def update(
        self,
        module: str,
        success: bool,
        retry_count: int,
        error_type: Optional[str],
        reward: float,
        rust_code: Optional[str] = None,
    ) -> None:
        mod = self._module(module)
        mod["retry_count"] = retry_count
        mod["last_error_type"] = error_type

        if mod["best_reward"] is None or reward > mod["best_reward"]:
            mod["best_reward"] = reward

        if success:
            mod["status"] = "migrated"
            if rust_code:
                mod["rust_code"] = rust_code
        elif mod["status"] != "migrated":
            mod["status"] = "in_progress"

        # Update global counters
        g = self._data.setdefault("global", {})
        all_mods = self._data["modules"]
        g["total"] = len(all_mods)
        g["migrated"] = sum(1 for m in all_mods.values() if m["status"] == "migrated")

        self.save()

    # ── Context builder ──────────────────────────────────────────────────────

    def get_context(self) -> dict[str, Any]:
        """Return a lightweight summary for the observation's migration_context."""
        g = self._data.get("global", {})
        total = g.get("total", 0)
        migrated = g.get("migrated", 0)

        error_dist: dict[str, int] = {}
        migrated_modules: list[str] = []

        for name, mod in self._data.get("modules", {}).items():
            if mod["status"] == "migrated":
                migrated_modules.append(name)
            et = mod.get("last_error_type")
            if et:
                error_dist[et] = error_dist.get(et, 0) + 1

        return {
            "total_modules": total,
            "migrated_count": migrated,
            "migration_pct": round(migrated / total * 100, 1) if total else 0.0,
            "migrated_modules": migrated_modules,
            "error_distribution": error_dist,
        }

    def all_modules(self) -> dict[str, dict[str, Any]]:
        return dict(self._data.get("modules", {}))
