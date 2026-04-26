"""Typed Action, Observation, and State models for the C2Rust RL environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State


@dataclass
class C2RustAction(Action):
    """Action: submit generated Rust code for a given C module."""
    rust_code: str = ""        # The Rust translation to evaluate
    module_name: str = ""      # Which C module this translation is for


@dataclass
class C2RustObservation(Observation):
    """Observation returned after each step."""
    c_source: str = ""                         # C source code to translate
    module_name: str = ""                      # Name of the module
    reward: float = 0.0                        # Compiler-based reward ∈ [0, 1]
    compile_success: bool = False              # Did it compile cleanly?
    unsafe_count: int = 0                      # Number of unsafe{} blocks
    warning_count: int = 0                     # Compiler warnings
    error_count: int = 0                       # Compiler errors
    compiler_errors: List[str] = field(default_factory=list)
    done: bool = False                         # Episode complete?
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class C2RustState(State):
    """Episode state — tracked across steps."""
    current_module: str = ""
    files_served: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
