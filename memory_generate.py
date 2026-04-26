import json
import os


def _normalize_import(name):
    """Convert 'foo.h' / 'foo.c' -> 'foo'"""
    base = os.path.basename(name)
    if base.endswith((".h", ".c")):
        return os.path.splitext(base)[0]
    return base


def initialize_migration_state(deps_path, state_path="migration_state.json", top_level_folder=None):
    """
    Reads dependencies.json and creates migration_state.json.
    Skips self-imports (e.g., foo.h inside module foo).
    If top_level_folder is specified, only includes modules from that top-level folder.
    """
    if os.path.exists(state_path):
        print(f"State file already exists at {state_path}. Skipping.")
        return

    if not os.path.exists(deps_path):
        print(f"Error: Dependency file not found at {deps_path}")
        return

    with open(deps_path, 'r') as f:
        deps = json.load(f)

    state = {
        "modules": {},
        "global_stats": {
            "total_files": 0,
            "migrated_files": 0
        }
    }

    repo_modules = {
    module_name
    for folder_entries in deps.get("modules", {}).values()
    for module_name in folder_entries.keys()
    }

    for folder, module_entries in deps.get("modules", {}).items():
        # Filter by top-level folder if specified
        if top_level_folder is not None:
            folder_prefix = folder.split('\\')[0] if '\\' in folder else folder
            if folder_prefix != top_level_folder:
                continue
                
        state["modules"][folder] = {}

        for module_name, data in module_entries.items():
            state["global_stats"]["total_files"] += 1

            exports = data.get("exports", [])
            includes = data.get("includes", [])

            # --- FILTER IMPORTS ---
            filtered_imports = {}
            for inc in includes:
                normalized = _normalize_import(inc)

                # Skip self-import (foo.h inside foo)
                if normalized not in repo_modules:
                    continue

                if normalized == module_name:
                    continue

                filtered_imports[inc] = False

            total_imports = len(filtered_imports)
            total_exports = len(exports)
            
            state["modules"][folder][module_name] = {
                "migrated": False,
                "exports": {name: False for name in exports},
                "imports": filtered_imports,
            

            "stats": {
            "total_imports": total_imports,
            "total_exports": total_exports,
            "migrated_imports": 0,
            "migrated_exports": 0
            }
            }


    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"Success: Initialized migration state at {state_path}")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Generate migration_state.json from dependencies.json")
    _p.add_argument("--deps",  default="migrator_data/dependencies.json", help="Path to dependencies.json")
    _p.add_argument("--state", default="migration_state.json",            help="Output migration state path")
    _a = _p.parse_args()
    initialize_migration_state(_a.deps, _a.state)