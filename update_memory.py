import json
import os


def _normalize_import(name):
    """
    Convert include names like 'foo.h' or 'foo.c' -> 'foo'
    Keeps system headers (e.g. stdio.h) unchanged unless desired.
    """
    base = os.path.basename(name)

    if base.endswith((".h", ".c")):
        return os.path.splitext(base)[0]

    return base

def reconcile_migration_progress(state_path):
    if not os.path.exists(state_path):
        print(f"Error: State file {state_path} not found.")
        return

    with open(state_path, 'r') as f:
        state = json.load(f)

    # 1. Collect fully migrated modules
    fully_migrated_modules = set()

    for folder in state["modules"].values():
        for module_name, module_data in folder.items():
            all_exports_done = (
                all(module_data["exports"].values())
                if module_data["exports"] else True
            )

            if module_data["migrated"] or all_exports_done:
                fully_migrated_modules.add(module_name)

    # 2. Reconcile + compute stats
    changes_made = 0
    total_migrated = 0

    # Global aggregates (NEW)
    global_total_imports = 0
    global_total_exports = 0
    global_migrated_imports = 0
    global_migrated_exports = 0

    for folder in state["modules"].values():
        for module_name, module_data in folder.items():

            # --- UPDATE IMPORTS ---
            for imp_name in list(module_data["imports"].keys()):
                normalized = _normalize_import(imp_name)

                if normalized in fully_migrated_modules:
                    if not module_data["imports"][imp_name]:
                        module_data["imports"][imp_name] = True
                        changes_made += 1

            # --- CHECK MIGRATION ---
            all_imps_done = all(module_data["imports"].values()) if module_data["imports"] else True
            all_exps_done = all(module_data["exports"].values()) if module_data["exports"] else True

            if all_imps_done and all_exps_done and not module_data["migrated"]:
                module_data["migrated"] = True
                changes_made += 1

            if module_data["migrated"]:
                total_migrated += 1

            # --- TRACK EXPORT CHANGES (FIX) ---
            prev_migrated_exports = module_data.get("stats", {}).get("migrated_exports", 0)
            current_migrated_exports = sum(1 for v in module_data["exports"].values() if v)

            if current_migrated_exports != prev_migrated_exports:
                changes_made += 1


            # --- COMPUTE STATS ---
            total_imports = len(module_data["imports"])
            total_exports = len(module_data["exports"])

            migrated_imports = sum(1 for v in module_data["imports"].values() if v)
            migrated_exports = current_migrated_exports

            module_data["stats"] = {
                "total_imports": total_imports,
                "total_exports": total_exports,
                "migrated_imports": migrated_imports,
                "migrated_exports": migrated_exports
            }

            # Update global aggregates
            global_total_imports += total_imports
            global_total_exports += total_exports
            global_migrated_imports += migrated_imports
            global_migrated_exports += migrated_exports

    # 3. Update global stats
    state["global_stats"]["migrated_files"] = total_migrated

    # Optional global metrics (NEW)
    state["global_stats"]["total_imports"] = global_total_imports
    state["global_stats"]["total_exports"] = global_total_exports
    state["global_stats"]["migrated_imports"] = global_migrated_imports
    state["global_stats"]["migrated_exports"] = global_migrated_exports

    # 4. Persist
    if changes_made > 0:
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"Reconciliation successful. {changes_made} updates saved to {state_path}.")
    else:
        print("No new progress detected.")

