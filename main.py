"""
Entry point for manual evaluation / smoke-testing of the C2Rust environment.

Usage:
    python main.py                          # run one episode with a dummy agent
    python main.py --file data/c_programs/simple_malloc.c
    python main.py --train                  # kick off full GRPO training
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_episode(c_file: str | None = None) -> None:
    """Run a single episode in the environment with a trivial stub agent."""
    from env.c2rust_env import C2RustEnv

    env = C2RustEnv()
    obs, info = env.reset()

    if c_file:
        # Force the file for debugging (monkey-patch current file)
        p = Path(c_file)
        env._current_file = p
        env._c_source = p.read_text(encoding="utf-8")
        ref = p.with_suffix(".ref.txt")
        env._reference_output = ref.read_text(encoding="utf-8") if ref.exists() else ""
        from analyzer.static import parse_c_ast
        env._c_ast = parse_c_ast(env._c_source)
        obs = env._build_obs()

    print(f"\nC source ({len(obs['c_source'])} chars):")
    print(obs["c_source"][:500], "..." if len(obs["c_source"]) > 500 else "")

    # Stub agent: produce a minimal (wrong) Rust program to trigger compiler
    stub_rust = """fn main() {\n    println!("stub");\n}\n"""
    action = {"rust_code": stub_rust}

    obs2, reward, terminated, truncated, info = env.step(action)

    print(f"\nReward: {reward:.3f}")
    print(f"Error type: {info['error_type']}")
    print(f"Unsafe count: {info['unsafe_count']}")
    print(f"Test passed: {info['test_passed']}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="C2Rust RL environment entry point")
    parser.add_argument("--file", help="Specific .c file to test", default=None)
    parser.add_argument("--train", action="store_true", help="Run GRPO training")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    if args.train:
        from train import train
        train(args.config)
    else:
        run_episode(args.file)


if __name__ == "__main__":
    main()
