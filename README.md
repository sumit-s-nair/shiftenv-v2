---
title: C2Rust RL
emoji: 🦀
colorFrom: orange
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# C2Rust — Teaching LLMs to Migrate C to Safe Rust

## The Problem

C powers the operating systems, embedded firmware, and network infrastructure that the world depends on — but its manual memory management makes it a perpetual source of critical vulnerabilities (CVEs, heap overflows, use-after-free bugs). Rust offers memory safety guarantees enforced at compile time, with no runtime overhead. The problem: migrating real C codebases to Rust by hand is slow, expensive, and demands deep expertise in both languages simultaneously. Automated tools exist but produce `unsafe` Rust — code that preserves C's danger rather than eliminating it. We train a model to produce *safe*, idiomatic Rust from first principles.

## The Environment

The agent sees one C module at a time (ordered by `#include` dependencies — leaves first) and must produce a complete Rust translation. At each step:

- **Observation** — raw C source, tree-sitter AST, previous Rust attempt, structured `rustc` errors, retry count, and a summary of already-migrated modules.
- **Action** — a full Rust source file.
- **Episode** — terminates when the translation compiles and passes semantic equivalence tests, or when the retry budget (5 attempts) is exhausted.

```
C source → [DeepSeek-Coder-V2-Lite] → Rust code → cargo build + tests → reward → [GRPO update]
                    ↑__________________________________________________|
```

## The Reward Signal

`rustc --error-format=json` acts as the reward oracle — no human labels required.

| Compiler outcome | Reward |
|---|---|
| Clean compile + tests pass + no `unsafe` | **+1.0** |
| Compiles + tests pass + `unsafe` used | **+0.3** |
| Ownership error (E0505, E0382, …) | **−0.3** |
| Lifetime error (E0597, E0106, …) | **−0.5** |
| Tests fail (semantic mismatch) | **−0.8** |
| Does not compile | **−1.0** |

**Shaped bonuses:** −0.1 per `unsafe` block (cumulative), +0.1 for zero clippy warnings.

## Results

> _Training in progress on HuggingFace Spaces (1×L40S, 36 hr run)._

Reward curve and before/after examples will be embedded here once the training run completes.

**Quantitative targets (36 hr run):**

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
pip install unsloth trl datasets wandb pyyaml tree-sitter tree-sitter-c gradio

# 2. Install Rust (if not already)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 3. Smoke-test the compiler bridge
python main.py

# 4. Full GRPO training + automatic Hub upload
HF_TOKEN=hf_xxx HF_REPO_ID=YOUR_USERNAME/c2rust-deepseek-coder-v2-lite \
    python train.py --config configs/config.yaml
```

## Deploying to HuggingFace Spaces

### 1. Set your Hub repo in config

Edit `configs/config.yaml`:
```yaml
hub:
  repo_id: "YOUR_USERNAME/c2rust-deepseek-coder-v2-lite"
  private: false
```

### 2. Add secrets to your Space

In **Space → Settings → Variables and secrets**:

| Secret | Value |
|---|---|
| `HF_TOKEN` | from hf.co/settings/tokens (write access) |
| `WANDB_API_KEY` | from wandb.ai/authorize |
| `HF_REPO_ID` | `YOUR_USERNAME/c2rust-deepseek-coder-v2-lite` |

### 3. Push to the Space

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/c2rust-rl
git push space main
```

The Space detects the `Dockerfile`, builds the image, and starts `app.py`.  
After training completes, the fine-tuned model is pushed automatically to the Hub repo and `agent/code_writer.py` is updated to load from it.

## Links

- HuggingFace Space: *(link TBD after deployment)*
- Fine-tuned Model: *(link TBD after training)*
- Training Notebook: [notebooks/training_colab.ipynb](notebooks/training_colab.ipynb)
- Demo Video: *(link TBD)*
- Blog Post: *(link TBD)*

---

**OpenEnv Hackathon India 2026 — Theme #2: Long-Horizon Planning**

> *"We fine-tuned DeepSeek Coder V2 Lite 16B directly on an L40S using GRPO — the same RL algorithm DeepSeek used to train R1 — with the Rust compiler itself as the reward oracle. No human labels. The compiler teaches the agent."*
