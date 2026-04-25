# C2Rust — Teaching LLMs to Migrate C to Safe Rust

## The Problem

C powers the operating systems, embedded firmware, and network infrastructure that the world depends on — but its manual memory management makes it a perpetual source of critical vulnerabilities (CVEs, heap overflows, use-after-free bugs). Rust offers memory safety guarantees enforced at compile time, with no runtime overhead. The problem: migrating real C codebases to Rust by hand is slow, expensive, and demands deep expertise in both languages simultaneously. Automated tools exist but produce `unsafe` Rust — code that preserves C's danger rather than eliminating it. We train a model to produce *safe*, idiomatic Rust from first principles.

## The Environment

The agent sees a single C module at a time and must produce a complete Rust translation. At each step:

- **Observation** — the raw C source, its tree-sitter AST, the previous Rust attempt (if any), structured compiler errors from the last attempt, retry count, and a summary of what other modules have already been migrated.
- **Action** — a full Rust source file (string).
- **Episode** — terminates when the translation compiles cleanly and passes semantic equivalence tests, or when the retry budget (5 attempts) is exhausted.

```
C source → [LLM agent] → Rust code → rustc + tests → reward → [RL update]
                ↑_______________________________|
```

## The Reward Signal

The Rust compiler (`rustc --error-format=json`) acts as the reward oracle — no human labels required. Structured JSON output is parsed into error categories that drive a shaped reward:

| Compiler outcome | Reward | Reasoning |
|---|---|---|
| Clean compile + all tests pass + no `unsafe` | **+1.0** | Perfect migration |
| Clean compile + tests pass + `unsafe` used | **+0.3** | Compiles but cheated |
| Ownership error (E0505, E0382, …) | **−0.3** | Recoverable, line-level feedback |
| Lifetime error (E0597, E0106, …) | **−0.5** | Structural rethink needed |
| Tests fail (semantic mismatch) | **−0.8** | Correct structure, wrong output |
| Does not compile | **−1.0** | Total failure |

**Shaped bonuses:** −0.1 per `unsafe` block (cumulative), +0.1 for zero clippy warnings.

## Results

> _Training in progress on HuggingFace Spaces (1×L40S, 36 hr run)._

Reward curve and before/after examples will be embedded here once the training run completes.

**Quantitative target (36 hr run):**

| Metric | Baseline (untrained) | Fine-tuned target |
|---|---|---|
| Compile rate | ~10% | >70% |
| Test-pass rate | ~5% | >40% |
| `unsafe`-free rate | ~30% | >80% |
| Mean reward | −0.85 | >+0.4 |

## How to Run

```bash
# 1. Clone and install (requires Rust toolchain)
git clone https://github.com/YOUR_USERNAME/c2rust-rl
cd c2rust-rl
pip install unsloth trl datasets wandb pyyaml tree-sitter tree-sitter-c

# 2. Install Rust (if not already)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 3. Smoke-test the compiler bridge
python main.py

# 4. Full GRPO training run
python train.py --config configs/config.yaml
```

## Links

- HuggingFace Space: *(link TBD after deployment)*
- Training Notebook: [notebooks/training_colab.ipynb](notebooks/training_colab.ipynb)
- Demo Video: *(link TBD)*
- Blog Post: *(link TBD)*

---

**OpenEnv Hackathon India 2026 — Theme #2: Long-Horizon Planning**

> *"We fine-tuned DeepSeek Coder V2 Lite 16B directly on an L40S using GRPO — the same RL algorithm DeepSeek used to train R1 — with the Rust compiler itself as the reward oracle. No human labels. The compiler teaches the agent."*
