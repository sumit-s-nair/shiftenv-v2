import os
import json
import argparse
import clang.cindex
from clang.cindex import CursorKind, TypeKind, TranslationUnit


class CToRustAnalyzer:
    def __init__(self, repo_path):
        self.repo_path = os.path.abspath(repo_path)
        self.index = clang.cindex.Index.create()

        # Internal Data Storage
        self.ast_data = {}
        self.pointer_data = []
        self.dep_graph = {
            "modules": {},
            "call_graph": {}
        }

    def _get_rel_path(self, path):
        return os.path.relpath(path, self.repo_path)

    def _get_module_name(self, filepath):
        """Normalize module name: foo.c / foo.h -> foo"""
        base = os.path.basename(filepath)
        return os.path.splitext(base)[0]

    def analyze_repository(self):
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(('.c', '.h')):
                    abs_path = os.path.join(root, file)
                    self._process_file(abs_path)

    def _process_file(self, filepath):
        rel_path = self._get_rel_path(filepath)
        folder = os.path.dirname(rel_path) or "root"
        module_name = self._get_module_name(filepath)

        print(f"Analyzing {rel_path}...")

        options = TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        args = ['-x', 'c', f'-I{self.repo_path}']

        try:
            tu = self.index.parse(filepath, args=args, options=options)

            # Ensure folder exists
            if folder not in self.dep_graph["modules"]:
                self.dep_graph["modules"][folder] = {}

            # Ensure module exists (MERGED .c/.h)
            if module_name not in self.dep_graph["modules"][folder]:
                self.dep_graph["modules"][folder][module_name] = {
                    "metrics": {"import_count": 0, "export_count": 0},
                    "includes": [],
                    "exports": []
                }

            file_entry = self.dep_graph["modules"][folder][module_name]

            # AST remains per physical file
            filename = os.path.basename(filepath)
            self.ast_data[filename] = self._serialize_ast(tu.cursor, filepath)

            # Walk semantics
            self._walk_semantics(tu.cursor, filepath, file_entry, current_func=None)

            # Deduplicate
            file_entry["includes"] = list(set(file_entry["includes"]))
            file_entry["exports"] = list(set(file_entry["exports"]))

            # Update metrics
            file_entry["metrics"]["import_count"] = len(file_entry["includes"])
            file_entry["metrics"]["export_count"] = len(file_entry["exports"])

        except Exception as e:
            print(f"Failed to parse {rel_path}: {e}")

    def _serialize_ast(self, cursor, origin_file):
        """Recursively convert AST to JSON, filtered to origin file."""
        if cursor.kind != CursorKind.TRANSLATION_UNIT:
            if not cursor.location.file or os.path.abspath(cursor.location.file.name) != os.path.abspath(origin_file):
                return None

        node = {
            "kind": str(cursor.kind).split('.')[1],
            "name": cursor.spelling,
            "type": cursor.type.spelling if cursor.type.spelling else None,
            "location": f"{cursor.location.line}:{cursor.location.column}" if cursor.location.file else "0:0",
            "children": []
        }

        for child in cursor.get_children():
            serialized_child = self._serialize_ast(child, origin_file)
            if serialized_child:
                node["children"].append(serialized_child)

        return node

    def _walk_semantics(self, cursor, filepath, file_entry, current_func):
        is_in_file = cursor.location.file and os.path.abspath(cursor.location.file.name) == os.path.abspath(filepath)
        filename = os.path.basename(filepath)

        # Includes
        if cursor.kind == CursorKind.INCLUSION_DIRECTIVE:
            file_entry["includes"].append(cursor.spelling)

        # Function definitions
        if cursor.kind == CursorKind.FUNCTION_DECL and is_in_file:
            if cursor.is_definition():
                current_func = cursor.spelling

                if current_func not in self.dep_graph["call_graph"]:
                    self.dep_graph["call_graph"][current_func] = set()

                file_entry["exports"].append(current_func)

        # Call expressions
        if cursor.kind == CursorKind.CALL_EXPR and current_func:
            if cursor.spelling:
                self.dep_graph["call_graph"][current_func].add(cursor.spelling)

        # Pointer analysis
        if is_in_file and (
            cursor.type.kind == TypeKind.POINTER or
            cursor.type.kind == TypeKind.INCOMPLETEARRAY
        ):
            if cursor.kind in (
                CursorKind.VAR_DECL,
                CursorKind.PARM_DECL,
                CursorKind.FIELD_DECL
            ):
                self.pointer_data.append({
                    "file": filename,
                    "name": cursor.spelling,
                    "parent_func": current_func,
                    "type": cursor.type.spelling,
                    "is_const": cursor.type.get_pointee().is_const_qualified()
                })

        for child in cursor.get_children():
            self._walk_semantics(child, filepath, file_entry, current_func)

    def _finalize_call_graph(self):
        """Convert sets to lists for JSON serialization"""
        for func, callees in self.dep_graph["call_graph"].items():
            self.dep_graph["call_graph"][func] = sorted(list(callees))

    def export_json(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Finalize call graph
        self._finalize_call_graph()

        files_to_save = {
            'ast.json': self.ast_data,
            'pointers.json': self.pointer_data,
            'dependencies.json': self.dep_graph,
            'call_graph.json': self.dep_graph["call_graph"]
        }

        for filename, data in files_to_save.items():
            with open(os.path.join(output_dir, filename), 'w') as f:
                json.dump(data, f, indent=2)

        print(f"Exported JSON files to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path", help="Path to the C repository")
    parser.add_argument("--out", default="./output", help="Output directory")
    args = parser.parse_args()

    analyzer = CToRustAnalyzer(args.repo_path)
    analyzer.analyze_repository()
    analyzer.export_json(args.out)