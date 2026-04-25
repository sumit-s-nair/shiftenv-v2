"""
C source analyser using tree-sitter.

Extracts function signatures, pointer/malloc/free patterns, struct definitions,
and a difficulty assessment for Rust migration.

Falls back to a regex-based heuristic when tree-sitter is unavailable so that
the rest of the system still runs in lightweight environments.
"""

from __future__ import annotations

import re
from typing import Any

# ── Tree-sitter bootstrap ────────────────────────────────────────────────────

try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser

    _C_LANGUAGE = Language(tsc.language())
    _PARSER = Parser(_C_LANGUAGE)
    _TS_AVAILABLE = True
except Exception:
    _TS_AVAILABLE = False


# ── Public API ───────────────────────────────────────────────────────────────

def parse_c_ast(source: str) -> dict[str, Any]:
    """
    Parse *source* (a C source string) and return a structured dict:

    {
        "functions":      [{"name": str, "return_type": str, "params": [...]}],
        "structs":        [{"name": str, "fields": [...]}],
        "pointer_ops":    {"malloc_count": int, "free_count": int, "raw_pointers": int},
        "difficulty":     {"score": float, "reasons": [str]},
        "unsafe_patterns": [str],   # patterns that will need unsafe in naive translation
    }
    """
    if _TS_AVAILABLE:
        return _parse_with_treesitter(source)
    return _parse_with_regex(source)


# ── Tree-sitter implementation ───────────────────────────────────────────────

def _parse_with_treesitter(source: str) -> dict[str, Any]:
    tree = _PARSER.parse(source.encode())
    root = tree.root_node

    functions = _extract_functions_ts(root, source)
    structs = _extract_structs_ts(root, source)
    pointer_ops = _count_pointer_ops_ts(root, source)
    unsafe_patterns = _identify_unsafe_patterns_ts(root, source)
    difficulty = _assess_difficulty(functions, structs, pointer_ops, unsafe_patterns)

    return {
        "functions": functions,
        "structs": structs,
        "pointer_ops": pointer_ops,
        "difficulty": difficulty,
        "unsafe_patterns": unsafe_patterns,
        "parser": "tree-sitter",
    }


def _text(node, source: str) -> str:
    return source[node.start_byte:node.end_byte]


def _extract_functions_ts(root, source: str) -> list[dict]:
    functions = []
    for node in _walk(root):
        if node.type != "function_definition":
            continue
        name = ""
        return_type = ""
        params = []

        for child in node.children:
            if child.type == "function_declarator":
                for sub in child.children:
                    if sub.type == "identifier":
                        name = _text(sub, source)
                    elif sub.type == "parameter_list":
                        params = _extract_params(sub, source)
            elif child.type in ("primitive_type", "type_identifier", "pointer_declarator"):
                return_type = _text(child, source)

        if name:
            functions.append({"name": name, "return_type": return_type, "params": params})

    return functions


def _extract_params(param_list_node, source: str) -> list[dict]:
    params = []
    for child in param_list_node.children:
        if child.type == "parameter_declaration":
            param_text = _text(child, source).strip()
            is_pointer = "*" in param_text
            params.append({"text": param_text, "is_pointer": is_pointer})
    return params


def _extract_structs_ts(root, source: str) -> list[dict]:
    structs = []
    for node in _walk(root):
        if node.type not in ("struct_specifier", "type_definition"):
            continue
        name = ""
        fields = []
        for child in node.children:
            if child.type == "type_identifier" and not name:
                name = _text(child, source)
            elif child.type == "field_declaration_list":
                for fc in child.children:
                    if fc.type == "field_declaration":
                        fields.append(_text(fc, source).strip().rstrip(";"))
        if name:
            structs.append({"name": name, "fields": fields})
    return structs


def _count_pointer_ops_ts(root, source: str) -> dict[str, int]:
    text = source
    return {
        "malloc_count": len(re.findall(r"\bmalloc\s*\(", text)),
        "free_count": len(re.findall(r"\bfree\s*\(", text)),
        "raw_pointers": text.count("*"),
        "void_pointers": len(re.findall(r"\bvoid\s*\*", text)),
        "function_pointers": len(re.findall(r"\(\s*\*\s*\w+\s*\)", text)),
    }


def _identify_unsafe_patterns_ts(root, source: str) -> list[str]:
    patterns = []
    if re.search(r"\bmalloc\b|\bcalloc\b|\brealloc\b", source):
        patterns.append("heap_allocation")
    if re.search(r"\bfree\b", source):
        patterns.append("manual_free")
    if re.search(r"\bvoid\s*\*", source):
        patterns.append("void_pointer")
    if re.search(r"\(\s*\*\s*\w+\s*\)\s*\(", source):
        patterns.append("function_pointer")
    if re.search(r"\bunion\b", source):
        patterns.append("union")
    if re.search(r"\bsetjmp\b|\blongjmp\b", source):
        patterns.append("setjmp_longjmp")
    return patterns


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


# ── Regex fallback ───────────────────────────────────────────────────────────

def _parse_with_regex(source: str) -> dict[str, Any]:
    functions = []
    for m in re.finditer(
        r"(\w[\w\s\*]+?)\s+(\w+)\s*\(([^)]*)\)\s*\{",
        source,
    ):
        ret_type = m.group(1).strip()
        name = m.group(2).strip()
        param_str = m.group(3).strip()
        params = [
            {"text": p.strip(), "is_pointer": "*" in p}
            for p in param_str.split(",") if p.strip() and p.strip() != "void"
        ]
        functions.append({"name": name, "return_type": ret_type, "params": params})

    structs = []
    for m in re.finditer(r"typedef\s+struct\s*\{([^}]+)\}\s*(\w+)\s*;", source, re.S):
        fields_raw = m.group(1).strip().splitlines()
        fields = [f.strip().rstrip(";") for f in fields_raw if f.strip()]
        structs.append({"name": m.group(2), "fields": fields})

    pointer_ops = {
        "malloc_count": len(re.findall(r"\bmalloc\s*\(", source)),
        "free_count": len(re.findall(r"\bfree\s*\(", source)),
        "raw_pointers": source.count("*"),
        "void_pointers": len(re.findall(r"\bvoid\s*\*", source)),
        "function_pointers": len(re.findall(r"\(\s*\*\s*\w+\s*\)", source)),
    }

    unsafe_patterns = _identify_unsafe_patterns_ts(None, source)
    difficulty = _assess_difficulty(functions, structs, pointer_ops, unsafe_patterns)

    return {
        "functions": functions,
        "structs": structs,
        "pointer_ops": pointer_ops,
        "difficulty": difficulty,
        "unsafe_patterns": unsafe_patterns,
        "parser": "regex-fallback",
    }


# ── Difficulty assessment ────────────────────────────────────────────────────

def _assess_difficulty(
    functions: list, structs: list, pointer_ops: dict, unsafe_patterns: list
) -> dict[str, Any]:
    score = 0.0
    reasons = []

    if pointer_ops.get("malloc_count", 0) > 0:
        score += 0.2
        reasons.append("heap allocation (malloc/free → Box/Vec)")

    if pointer_ops.get("void_pointers", 0) > 0:
        score += 0.3
        reasons.append("void pointers (require generics or enums)")

    if pointer_ops.get("function_pointers", 0) > 0:
        score += 0.2
        reasons.append("function pointers (→ fn traits or closures)")

    if "union" in unsafe_patterns:
        score += 0.3
        reasons.append("C unions (→ enum or unsafe union)")

    for fn in functions:
        for param in fn.get("params", []):
            if param.get("is_pointer"):
                score += 0.05
                break

    if any(s.get("fields") for s in structs):
        ptr_fields = sum(
            1 for s in structs for f in s.get("fields", []) if "*" in f
        )
        if ptr_fields:
            score += ptr_fields * 0.1
            reasons.append(f"{ptr_fields} pointer field(s) in structs")

    score = min(1.0, score)
    return {"score": round(score, 2), "reasons": reasons}
