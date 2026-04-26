import os
import re


def _find_entry_file(src_dir: str, rust_files: list[str]) -> str:
    """Return the .rs file that contains `fn main()`, or the first file."""
    for fname in rust_files:
        with open(os.path.join(src_dir, fname)) as f:
            if re.search(r"\bfn\s+main\s*\(", f.read()):
                return fname
    return rust_files[0]


def generate_cargo(root: str, package_name: str) -> None:
    print("\nGenerating Cargo project…")

    src_dir = os.path.join(root, "src")
    os.makedirs(src_dir, exist_ok=True)

    # Move loose .rs files into src/
    rust_files = [f for f in os.listdir(root) if f.endswith(".rs")]
    for f in rust_files:
        os.rename(os.path.join(root, f), os.path.join(src_dir, f))

    # Also discover any .rs already in src/ (re-runs, partial state)
    all_rs = [f for f in os.listdir(src_dir) if f.endswith(".rs")]
    if not all_rs:
        print("WARNING: No .rs files found — empty Cargo project.")
        _write_cargo_toml(root, package_name)
        return

    entry_file = _find_entry_file(src_dir, all_rs)
    print(f"Entry file: {entry_file}")

    # Rename entry → main.rs
    main_rs = os.path.join(src_dir, "main.rs")
    if entry_file != "main.rs":
        os.rename(os.path.join(src_dir, entry_file), main_rs)

    # Prepend `mod X;` for every non-entry module
    modules = [
        os.path.splitext(f)[0]
        for f in os.listdir(src_dir)
        if f.endswith(".rs") and f != "main.rs"
    ]
    mod_header = "\n".join(f"mod {m};" for m in modules)

    with open(main_rs) as f:
        main_code = f.read()
    with open(main_rs, "w") as f:
        f.write(f"{mod_header}\n\n{main_code}" if mod_header else main_code)

    _write_cargo_toml(root, package_name)
    print(f"Cargo project ready: {root}")


def _write_cargo_toml(root: str, package_name: str) -> None:
    content = (
        f'[package]\nname = "{package_name}"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n'
    )
    with open(os.path.join(root, "Cargo.toml"), "w") as f:
        f.write(content)
