"""
LLM agent wrapper for C→Rust code generation.

Loads DeepSeek-Coder-V2-Lite-Instruct in 4-bit QLoRA via Unsloth (when
available) or plain HuggingFace transformers as a fallback.  Constructs
error-conditioned prompts and returns raw Rust source strings.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# DeepSeek's bundled modeling_deepseek.py imports is_torch_fx_available which
# was removed in transformers 4.46+. Patch it back before any model loading so
# the import succeeds regardless of transformers version.
try:
    from transformers.utils.import_utils import is_torch_fx_available  # noqa: F401
except ImportError:
    import transformers.utils.import_utils as _tfu
    _tfu.is_torch_fx_available = lambda: False


_SYSTEM_PROMPT = """\
You are an expert Rust programmer specialising in migrating C code to safe, idiomatic Rust.

Rules:
- Do NOT use unsafe blocks.  If you think you need unsafe, redesign using Rust ownership, Box, Rc, or Arc instead.
- Use idiomatic Rust: iterators, pattern matching, enums, Option/Result instead of raw pointers and null checks.
- The output must be a complete, compilable Rust program (include fn main if present in C).
- Do not include any explanation outside the code block.  Return ONLY valid Rust source.
"""

_RETRY_HEADER = """\
Your previous attempt failed to compile.  Study the errors carefully and fix them.
"""


class CodeWriter:
    """
    Wraps a causal language model to generate Rust translations from C.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to DeepSeek-Coder-V2-Lite-Instruct.
    use_unsloth:
        If True (and unsloth is installed), load via FastLanguageModel for
        speed.  Falls back to plain transformers automatically.
    quantization:
        "4bit" (default) or "8bit".
    max_new_tokens:
        Maximum tokens to generate.
    temperature:
        Sampling temperature (0.0 = greedy).
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        use_unsloth: bool = True,
        quantization: str = "4bit",
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._loaded = False

        self._load(use_unsloth, quantization)

    # ── Model loading ────────────────────────────────────────────────────────

    def _load(self, use_unsloth: bool, quantization: str) -> None:
        load_4bit = quantization == "4bit"

        if use_unsloth:
            try:
                from unsloth import FastLanguageModel
                self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_name,
                    max_seq_length=4096,
                    dtype=None,
                    load_in_4bit=load_4bit,
                )
                FastLanguageModel.for_inference(self._model)
                self._loaded = True
                return
            except ImportError:
                pass  # unsloth not installed — fall through

        # Plain HuggingFace transformers fallback
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise RuntimeError(
                "transformers is not installed. Run: pip install transformers bitsandbytes"
            ) from e

        try:
            import torch
        except ImportError as e:
            raise RuntimeError("torch is not installed. Run: pip install torch") from e

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU detected. DeepSeek-Coder-V2-Lite (16B) requires a GPU "
                "with at least 24 GB VRAM. This Space must be running on GPU hardware."
            )

        try:
            bnb_cfg = None
            if load_4bit:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            self._model.eval()
            self._loaded = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {self.model_name}: {e}\n"
                "Check transformers version — DeepSeek-Coder-V2 requires transformers<4.46."
            ) from e

    # ── Prompt construction ──────────────────────────────────────────────────

    def _build_prompt(self, obs: dict[str, Any]) -> str:
        c_source: str = obs.get("c_source", "")
        previous_rust: str = obs.get("previous_rust", "")
        compiler_errors: list = obs.get("compiler_errors", [])
        retry_count: int = obs.get("retry_count", 0)

        parts: list[str] = [_SYSTEM_PROMPT, "\n---\n"]

        if retry_count > 0 and compiler_errors:
            parts.append(_RETRY_HEADER)
            parts.append("\nCompiler errors from previous attempt:\n")
            for err in compiler_errors[:10]:   # cap at 10 to stay in context
                code = err.get("code") or "??"
                line = err.get("line") or "?"
                msg = err.get("message", "")
                parts.append(f"  [{code}] line {line}: {msg}\n")
            if previous_rust:
                parts.append(f"\nYour previous Rust attempt:\n```rust\n{previous_rust}\n```\n")

        parts.append(f"\nC source to translate:\n```c\n{c_source}\n```\n")
        parts.append("\nGenerate the complete Rust translation:\n```rust\n")
        return "".join(parts)

    # ── Inference ────────────────────────────────────────────────────────────

    def generate(self, obs: dict[str, Any]) -> str:
        """
        Given an observation dict, return a Rust source string.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        prompt = self._build_prompt(obs)

        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return _extract_rust(generated)

    def __call__(self, obs: dict[str, Any]) -> dict[str, str]:
        return {"rust_code": self.generate(obs)}


# ── Code extraction helper ───────────────────────────────────────────────────

def _extract_rust(text: str) -> str:
    """Pull the first ```rust ... ``` block; fall back to the full text."""
    m = re.search(r"```rust\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()
