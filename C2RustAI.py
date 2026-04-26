import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def strip_self_includes(c_code: str, module_name: str) -> str:
    pattern = rf'#include\s+"{re.escape(module_name)}(\.h|\.c)"'
    return re.sub(pattern, "", c_code)


def clean_rust_output(text: str) -> str:
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(("pub ", "fn ", "use ", "mod ", "#[")):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def remove_self_import(rust_code: str, module_name: str) -> str:
    return "\n".join(
        line for line in rust_code.splitlines()
        if f"use crate::{module_name}" not in line
    )


def convert_c_to_rust(file_path: str, output_path: str) -> str:
    """Convert *file_path* (.c) to Rust, write to *output_path*, return filename."""
    base_name      = os.path.basename(file_path)
    module_name    = os.path.splitext(base_name)[0]
    rust_filename  = f"{module_name}.rs"

    with open(file_path) as f:
        c_code = f.read()

    c_code = strip_self_includes(c_code, module_name)

    includes = re.findall(r'#include\s+"([^"]+)"', c_code)
    include_modules = [
        os.path.splitext(os.path.basename(inc))[0]
        for inc in includes
        if os.path.splitext(os.path.basename(inc))[0] != module_name
    ]
    dep_hint = (
        "\n".join(f"- {m} → use crate::{m}::*;" for m in include_modules)
        if include_modules
        else "(none)"
    )

    prompt = f"""\
You are a compiler-like translator converting C source code into Rust.

STRICT OUTPUT RULES:
- Output ONLY valid Rust code
- NO explanations, NO markdown, NO extra text

PROJECT STRUCTURE:
- Each C file becomes a Rust module with the same name
- Modules are declared elsewhere — do NOT add `mod` here
- Use `use crate::<module>::*;` to import dependencies

DETECTED DEPENDENCIES:
{dep_hint}

CONVERSION RULES:
- Preserve functionality exactly
- Prefer safe Rust (no unsafe unless necessary)
- Convert int → i32 unless context requires otherwise
- Convert printf → println!
- Mark externally-used functions as `pub`

INPUT C CODE:
{c_code}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Return only raw Rust code. No explanations, no markdown."},
            {"role": "user",   "content": prompt},
        ],
    )
    rust_code = response.choices[0].message.content

    rust_code = clean_rust_output(rust_code)
    rust_code = remove_self_import(rust_code, module_name)

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, rust_filename), "w") as f:
        f.write(rust_code)

    return rust_filename
