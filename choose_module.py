import heapq
import json
from collections import defaultdict


def _normalize(path: str) -> str:
    return path.replace(".h", "").replace(".c", "").replace("../", "").replace("./", "")


def _load(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def _build_graph(data: dict) -> tuple[dict, dict, dict]:
    """Return (nodes, deps, dependents) from migration-state data.

    nodes      : node_id → module info
    deps       : node_id → set of dependency node_ids
    dependents : node_id → set of node_ids that depend on it
    """
    modules    = data["modules"]
    nodes      = {}
    deps       = defaultdict(set)   # node → its dependencies
    dependents = defaultdict(set)   # node → nodes that import it

    for group, subs in modules.items():
        for name in subs:
            nodes[f"{group}/{name}"] = subs[name]

    for group, subs in modules.items():
        for name, info in subs.items():
            src = f"{group}/{name}"
            for imp in info.get("imports", {}):
                dep_key = _normalize(imp)
                # Match dep_key against the last segment of known node ids.
                matched = next(
                    (n for n in nodes if n.split("/")[-1] == dep_key), None
                )
                if matched:
                    deps[src].add(matched)
                    dependents[matched].add(src)

    return nodes, deps, dependents


def _migrated_set(data: dict) -> set[str]:
    return {
        f"{group}/{name}"
        for group, subs in data["modules"].items()
        for name, info in subs.items()
        if info.get("migrated", False)
    }


def _score(node: str, info: dict, deps: dict, dependents: dict) -> tuple:
    """Lower score = higher priority (topological + complexity heuristic)."""
    return (
        len(info.get("exports", {})),   # prefer smaller API surface
        len(deps[node]),                # prefer fewer dependencies
        len(dependents[node]),          # prefer lower downstream impact
    )


def pick_next_module(json_path: str, debug: bool = False) -> str | None:
    """Return the node_id of the next module to migrate, or None on deadlock."""
    data      = _load(json_path)
    nodes, deps, dependents = _build_graph(data)
    migrated  = _migrated_set(data)

    heap: list = []
    for node, info in nodes.items():
        if node in migrated:
            continue
        unresolved = [d for d in deps[node] if d not in migrated]
        if not unresolved:
            heapq.heappush(heap, (_score(node, info, deps, dependents), node))

    if not heap:
        if debug:
            print("No candidates. Blocked modules:")
            for node in nodes:
                if node not in migrated:
                    blocked_by = [d for d in deps[node] if d not in migrated]
                    print(f"  {node} blocked by {blocked_by}")
        return None

    return heapq.heappop(heap)[1]
