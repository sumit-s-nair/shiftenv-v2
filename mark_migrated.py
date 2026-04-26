import json


def mark_node_migrated(json_path: str, node_path: str) -> None:
    """Set migrated=True and mark all exports done for the given node."""
    with open(json_path) as f:
        data = json.load(f)

    parts = node_path.split("/")
    node  = data["modules"]
    try:
        for part in parts:
            node = node[part]
    except KeyError:
        raise ValueError(f"Invalid node path: {node_path!r}")

    node["migrated"] = True
    if "exports" in node:
        node["exports"] = {k: True for k in node["exports"]}
    if "stats" in node:
        node["stats"]["migrated_exports"] = node["stats"].get("total_exports", 0)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
