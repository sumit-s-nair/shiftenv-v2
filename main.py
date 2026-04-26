import argparse
import json
import os
import re
import subprocess
import sys
from time import sleep

from choose_module import pick_next_module
from gen_cargo import generate_cargo
from mark_migrated import mark_node_migrated
from update_memory import reconcile_migration_progress


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate a C repository to Rust.")
    parser.add_argument("--source",       default="c_migration_test_3",         help="C source directory")
    parser.add_argument("--output",       default="rust_output",                 help="Rust output directory")
    parser.add_argument("--state",        default="migration_state copy 2.json", help="Migration state JSON")
    parser.add_argument("--migrator-data", default="migrator_data",              dest="migrator_data")
    parser.add_argument(
        "--engine", choices=["openai", "local"], default="openai",
        help="'openai' uses C2RustAI (GPT-4o); 'local' uses C2RustLocal (LoRA+online RL)",
    )
    # Debug flags — only meaningful with --engine local
    parser.add_argument(
        "--debug", action="store_true",
        help="(local engine) sample G responses per module, log rewards to markdown instead of training",
    )
    parser.add_argument(
        "--debug-log", default="debug_log.md", dest="debug_log",
        help="Path for the debug markdown log (default: debug_log.md)",
    )
    return parser.parse_args()


def _make_package_name(source_path: str) -> str:
    raw  = os.path.basename(os.path.abspath(source_path))
    name = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    return f"_{name}" if name and name[0].isdigit() else name


def _ensure_state(state_path: str, source_path: str, migrator_data: str) -> None:
    if os.path.exists(state_path):
        return
    print(f"State '{state_path}' not found — running analysis on '{source_path}'…")
    result = subprocess.run(
        [sys.executable, "analyzer.py", source_path, "--out", migrator_data],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Analysis failed:\n{result.stderr}")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())
    from memory_generate import initialize_migration_state
    initialize_migration_state(
        os.path.join(migrator_data, "dependencies.json"),
        state_path,
    )


def _all_migrated(state_path: str) -> bool:
    with open(state_path) as f:
        data = json.load(f)
    stats = data["global_stats"]
    return stats["migrated_files"] >= stats["total_files"]


def main() -> None:
    args = _parse_args()

    if args.engine == "openai":
        from C2RustAI import convert_c_to_rust
    else:
        import C2RustLocal
        C2RustLocal.configure(
            online_training=True,
            debug=args.debug,
            debug_log=args.debug_log,
        )
        convert_c_to_rust = C2RustLocal.convert_c_to_rust

    _ensure_state(args.state, args.source, args.migrator_data)
    os.makedirs(args.output, exist_ok=True)

    package_name = _make_package_name(args.source)

    while not _all_migrated(args.state):
        node = pick_next_module(args.state)
        if node is None:
            print("No valid next module (possible cycle or deadlock).")
            break

        node_clean = node.removeprefix("root/")
        if not node_clean.endswith(".c"):
            node_clean += ".c"

        c_file_path = os.path.join(args.source, node_clean)
        if not os.path.exists(c_file_path):
            print(f"  Skipping (not found): {c_file_path}")
            mark_node_migrated(args.state, node)
            continue

        print(f"\nMigrating: {node}")
        rust_file = convert_c_to_rust(c_file_path, args.output)
        print(f"  → {rust_file}")

        mark_node_migrated(args.state, node)
        reconcile_migration_progress(args.state)
        sleep(0.5)

    generate_cargo(args.output, package_name)
    print("\nMigration finished.")


if __name__ == "__main__":
    main()
