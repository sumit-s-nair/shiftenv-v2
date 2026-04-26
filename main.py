import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from time import sleep

import health_check  
from choose_module import pick_next_module
from gen_cargo import generate_cargo
from mark_migrated import mark_node_migrated
from update_memory import reconcile_migration_progress
from reward import compute_reward


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate a C repository to Rust.")
    parser.add_argument("--source",       default="tests",         help="C source directory")
    parser.add_argument("--output",       default="rust_output",   help="Rust output directory")
    parser.add_argument("--state",        default="migration_state.json", help="Migration state JSON")
    parser.add_argument("--migrator-data", default="migrator_data",dest="migrator_data")
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
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging (only applies to local engine)",
    )
    return parser.parse_args()


def _make_package_name(source_path: str) -> str:
    raw  = os.path.basename(os.path.abspath(source_path))
    name = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    return f"_{name}" if name and name[0].isdigit() else name


def _ensure_state_for_folder(state_path: str, target_path: str, migrator_data: str) -> None:
    """Ensure state exists for a specific directory path."""
    if os.path.exists(state_path):
        os.remove(state_path)  # Force regeneration for new folder
    
    print(f"Regenerating state for folder: {target_path}")
    result = subprocess.run(
        [sys.executable, "analyzer.py", target_path, "--out", migrator_data],
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
        top_level_folder=None, # Analyzer already filtered to this path
    )


def _all_migrated(state_path: str) -> bool:
    if not os.path.exists(state_path):
        return True
    with open(state_path) as f:
        data = json.load(f)
    stats = data["global_stats"]
    return stats["migrated_files"] >= stats["total_files"]


def _cleanup_epoch(migrator_data: str, output_dir: str, state_file: str) -> None:
    """Delete all files generated during an epoch."""
    # Delete migrator data
    if os.path.exists(migrator_data):
        shutil.rmtree(migrator_data)
        print(f"  Cleaned up migration data: {migrator_data}")
    
    # Delete rust output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"  Cleaned up output directory: {output_dir}")
    
    # Delete migration state
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"  Cleaned up state file: {state_file}")


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
            use_wandb=args.wandb,
        )
        convert_c_to_rust = C2RustLocal.convert_c_to_rust

    os.makedirs(args.output, exist_ok=True)
    package_name = _make_package_name(args.source)

    # Hardcoded target epoch paths
    target_epochs = [
        "data_basics",
        os.path.join("easy", "folder_1"),
        os.path.join("easy", "folder_2"),
        os.path.join("easy", "folder_3"),
        os.path.join("medium", "folder_1"),
        os.path.join("medium", "folder_2"),
        os.path.join("medium", "folder_3"),
    ]

    epoch = 0

    # Iterate through the epochs exactly once
    for current_folder in target_epochs:
        target_path = os.path.join(args.source, current_folder)
        
        # Skip if the directory doesn't exist to prevent crashes
        if not os.path.exists(target_path):
            print(f"Directory not found, skipping: {target_path}")
            continue

        epoch += 1
        print(f"\n{'='*60}")
        print(f"STARTING EPOCH {epoch}: {current_folder}")
        print(f"{'='*60}")

        # Pass the precise path into the state generator
        _ensure_state_for_folder(args.state, target_path, args.migrator_data)

        while not _all_migrated(args.state):
            node = pick_next_module(args.state)
            if node is None:
                print("No valid next module (possible cycle or deadlock).")
                break

            node_clean = node.removeprefix("root/")
            if not node_clean.endswith(".c"):
                node_clean += ".c"

            # analyzer.py uses the target_path as its root now
            c_file_path = os.path.join(target_path, node_clean)
            
            # Fallback check in case analyzer stored relative to original args.source
            if not os.path.exists(c_file_path):
                fallback_path = os.path.join(args.source, node_clean)
                if os.path.exists(fallback_path):
                    c_file_path = fallback_path
                else:
                    print(f"  Skipping (not found): {c_file_path}")
                    mark_node_migrated(args.state, node)
                    continue

            print(f"\nEpoch {epoch} | {current_folder} | Migrating: {node}")
            rust_file = convert_c_to_rust(c_file_path, args.output)
            print(f"  → {rust_file}")

            # Compute and print rewards for OpenAI engine
            if args.engine == "openai":
                rust_file_path = os.path.join(args.output, rust_file)
                if os.path.exists(rust_file_path):
                    with open(rust_file_path) as f:
                        rust_code = f.read()
                    reward_score, reward_info = compute_reward(rust_code, timeout=30)
                    print(f"  Reward: {reward_score:.4f} (compile: {reward_info.compilation_score:.2f} | "
                          f"errors: {reward_info.error_count} | warnings: {reward_info.warning_count} | "
                          f"unsafe: {reward_info.unsafe_count} | unwrap: {reward_info.unwrap_count})")

            mark_node_migrated(args.state, node)
            reconcile_migration_progress(args.state)
            sleep(0.5)

        # Cleanup at the end of the specified epoch folder
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} COMPLETE: {current_folder}")
        print(f"{'='*60}")
        _cleanup_epoch(args.migrator_data, args.output, args.state)

    generate_cargo(args.output, package_name)

    if args.engine == "local":
        print("\nGenerating submission graphs and markdown report...")
        C2RustLocal.generate_submission_report(args.output)

    print(f"\nMigration finished.")
    print(f"Total epochs processed: {epoch}")


if __name__ == "__main__":
    main()