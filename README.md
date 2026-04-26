---
title: C2Rust OpenEnv Server
emoji: 🦀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# C2Rust — Teaching LLMs to Migrate C to Safe Rust

**OpenEnv Hackathon India 2026 — Theme #2: Long-Horizon Planning**

## The Problem
C powers the world's infrastructure, but manual memory management makes it a perpetual source of critical vulnerabilities. Automated migration tools exist, but they produce `unsafe` Rust — code that preserves C's danger rather than eliminating it. We train an agent to produce *safe*, idiomatic Rust from first principles, using the Rust compiler itself as the reward oracle.

## The "Puru" Architecture: Online RL & Long-Horizon Planning
Unlike traditional ML models that train offline on static datasets for hundreds of epochs, our agent uses **Online RL (GRPO)**. 

It tackles the entire C repository as a Long-Horizon problem. It climbs the dependency graph (`analyzer.py` via `libclang`), generating Rust, testing it against the compiler, and updating its policy dynamically in a **single pass**. As it moves from the lowest-level `.c` files up to `main.c`, the agent learns in real-time, drastically reducing `unsafe` blocks and ownership errors.

* **Base Model:** Qwen2.5-Coder-7B-Instruct (4-bit QLoRA)
* **RL Algorithm:** Custom PyTorch Sequence-Level GRPO 
* **Environment:** Fully compliant OpenEnv wrapper (`C2RustEnv`)

## The Reward Signal
`rustc` and `cargo` act as the reward oracle — no human labels required.

| Compiler outcome | Reward |
|---|---|
| Clean compile + tests pass + no `unsafe` | **+1.0** |
| Compiles + tests pass + `unsafe` used | **+0.3** |
| Ownership error (E0505, E0382, …) | **−0.3** |
| Lifetime error (E0597, E0106, …) | **−0.5** |
| Tests fail (semantic mismatch) | **−0.8** |
| Does not compile | **−1.0** |

## How to Run (Local vs. Cloud)

### Cloud Deployment (Hugging Face Spaces)
This repository contains a `Dockerfile` configured for Nvidia L40S/A10G GPUs. 
When deployed to a Hugging Face Space, it will automatically install dependencies (including `libclang-dev` and Rust), begin the repository migration, stream metrics to Weights & Biases, and finally push the trained LoRA adapters back to the Hub.

### Local Evaluation
```bash
# 1. Install dependencies (Requires LLVM/Clang and Rust installed on host)
pip install torch>=2.0.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install -r requirements.txt

# 2. Run the migration with local GRPO training
python main.py --engine local

# 3. View Hackathon Assets
# The script automatically generates training_curves.png and a hackathon_report.md
# inside the output directory when the migration completes.