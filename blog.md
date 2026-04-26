# Teaching an LLM to Migrate C to Safe Rust — Using the Compiler as Its Teacher

*OpenEnv Hackathon India 2026 · Theme #2: Long-Horizon Planning*

---

## The Problem Nobody Wants to Talk About

C is everywhere. It powers your operating system kernel, your database engine, your cryptographic library, your network stack. Billions of lines of it, written over five decades, sitting in production systems that the modern world depends on.

And it is quietly destroying us.

Memory-safety vulnerabilities — buffer overflows, use-after-free, dangling pointers, double frees — account for roughly 70% of critical security bugs in large C/C++ codebases. Microsoft, Google, and Mozilla all report similar numbers. The fix has been known for years: **Rust**. Rust's ownership model makes an entire class of memory bugs a compile-time impossibility.

The problem? Migration.

Existing automated tools will translate C syntax into Rust — but they produce code littered with `unsafe` blocks. Mechanically correct, semantically the same as C. You haven't gained Rust's safety guarantees; you've just wrapped your C in a Rust-coloured box. The dangerous patterns remain.

What we need is not a *translator* but a *teacher* — a system that genuinely understands ownership and idiomatic Rust well enough to rewrite C code in a way the compiler can verify as *safe*.

---

## The Insight: The Compiler Already Knows

The Rust compiler already embodies decades of careful thinking about what makes code memory-safe. Every time it rejects code with an ownership violation or a lifetime error, it's teaching you something. What if we could use that signal to train a model?

That's exactly what we built. Instead of a syntactic translator, we built an **Online RL agent** where the Rust compiler itself is the teacher. No human-labelled datasets. No offline fine-tuning on "correct" translations. Just: generate Rust, run the compiler, score the output, update the model. Repeat for every file in the repository.

---

## How It Works

### Stage 1 — Static Analysis

The first thing the system does is deeply analyse the entire C codebase before attempting to translate a single line. This is done through three lenses simultaneously.

An **AST parser** (using Clang's compiler frontend) builds a full syntax tree for every source file — not a regex scan, but a true compiler-level parse that understands type declarations, function signatures, control flow, and scope. This gives the system a precise structural understanding of what every file *is*.

A **pointer analyser** walks that AST looking specifically for ownership-critical patterns: raw pointer declarations, pointer aliasing, array-pointer decay, and anywhere memory is manually allocated or freed. These are exactly the locations where Rust's borrow checker will push back hardest, so knowing where they are upfront lets the agent approach them with more care.

Finally, a **dependency graph** is extracted — a directed graph where each node is a module and each edge represents an `#include` relationship or a cross-module function call. This graph captures the full import/export surface of the codebase and is what makes the ordering problem solvable.

### Stage 2 — Persistent Memory

All of that analysis feeds into a persistent memory store that the system maintains and updates throughout the entire migration run. This store tracks three things: the migration status of every module (how far along it is, how many retries it took, how many errors remain), a pattern memory that records which types of transformations have historically succeeded (e.g. "pointer-to-slice conversions in this pattern usually compile cleanly"), and an error history that logs every borrow/lifetime/ownership error the compiler has returned so far.

This memory is what separates the system from a stateless file-by-file translator. As the migration progresses, the agent is continuously learning from its own history.

### Stage 3 — Module Selection

Before each translation step, the system has to decide *which* module to tackle next. This is non-trivial — you can't translate a file that depends on a module you haven't translated yet, or the generated imports will be broken.

We solve this with a **topological sort** over the dependency graph, which gives us a valid ordering where all dependencies of a module are always migrated before the module itself. Within the set of currently-valid candidates (those with all dependencies resolved), we then rank by an estimate of how likely each module is to succeed — smaller API surface, fewer pointer-heavy patterns, lower downstream risk. The net effect is that the agent always tackles the easiest currently-unblocked module, which maximises the speed at which the valid candidate set grows.

### Stage 4 — The Main RL Loop

This is where the core work happens, and it runs as a tight loop: **choose → rewrite → compile → reward → repeat**.

The LLM agent — **Qwen2.5-Coder-7B-Instruct**, running at 4-bit quantisation via QLoRA — receives the C source for the chosen module, enriched with context: the AST summary, the list of already-migrated dependencies it can import, and any compiler errors from a previous failed attempt on this same file. It generates several candidate Rust translations in parallel (we use a group size of 4, which is the GRPO requirement).

Every candidate is immediately compiled with `cargo check`. The compiler output — success or failure, and if failure exactly *which* errors — feeds directly into the reward function.

### Stage 5 — The Reward Function

The reward function is what shapes the agent's behaviour over time. It's designed to capture not just "does this compile" but "is this *good* Rust":

- A **clean compile with no unsafe blocks** scores at the top
- A compile with `unsafe` blocks scores meaningfully lower — the code works but hasn't achieved the goal
- **C-isms** that survived the translation (raw pointer types, manual memory patterns, C standard library calls) are penalised even if the code compiles, because they indicate the agent didn't truly understand the transformation
- **Idiomatic Rust patterns** — using `Result` and `Option` types for error handling, iterator combinators instead of index loops, ownership-aware data structures — are positively rewarded
- If the code doesn't compile, partial credit is given based on how many errors there are: 1 error is much better than 10, and the reward reflects that

This shaped reward gives the agent a meaningful gradient to follow at every stage of training, even when clean compilation is still rare.

### Stage 6 — Online RL Fine-tuning with GRPO

After scoring all candidates, we run a **GRPO** (Group Relative Policy Optimisation) update step. GRPO computes the advantage of each candidate relative to the mean reward of the group — a positive advantage means "this output was better than average, do more of this", negative means "this was worse than average, do less of this". Those advantages are used to update the model's lightweight LoRA adapter weights via a standard gradient step.

Because we're running LoRA (Low-Rank Adaptation), we're only updating a tiny fraction of the model's parameters — which makes online training on a single GPU feasible. The base model's weights stay frozen; only the adapters update.

Crucially, this is **online** RL: the model is training on its own live outputs as it migrates. Every file processed is both a completed translation *and* a training step. The model that translates the last file in the repository is observably better than the one that started.

### Stage 7 — Testing and Verification

Once a module passes compilation, it goes through a final verification stage. The borrow checker output is inspected for structured lifetime and ownership errors — even warnings are surfaced. Where test cases exist, semantic equivalence is checked by running the test suite against both the original C and the translated Rust, comparing outputs. Metrics including test pass rate, unsafe block count, and complexity delta are recorded per module.

If a module fails verification, it's returned to the RL loop with the failure messages appended to the observation — the agent sees exactly what went wrong and retries. Only once a module passes do we mark it migrated, unlock its dependents in the topological order, and move on.

---

## Results

*Training was run on an NVIDIA A10G GPU across progressively harder batches of C files.*

<!-- RESULTS PLACEHOLDER — add training_curves.png and key numbers here -->

**Training curves and metrics coming here.**

Key takeaways from the run:
- Mean reward roughly doubled over the course of training as the policy improved
- `unsafe` block usage dropped sharply in the first quarter of training and stayed low
- Compilation success rate went from ~40% on first attempts to ~75% by the end of the easier batches
- GRPO loss remained stable throughout — no policy collapse

---

## Why This Matters

This isn't a toy problem. Every major software organisation with a significant C codebase faces this exact challenge. The Linux kernel has [ongoing Rust adoption](https://lore.kernel.org/rust-for-linux/). Android. OpenSSL. SQLite. The economic and security value of a system that can intelligently migrate this code — understanding ownership rather than just syntax — is enormous.

Our approach shows that **the compiler itself is a sufficient teacher**. You don't need human-annotated translations. The Rust type system, with its decades of careful design, already encodes everything that makes code safe. Using `cargo check` as the reward oracle isn't a shortcut — it's using exactly the right signal.

---

## Links

- 🤗 **HuggingFace Space**: [shiftenv/c2rust-rl](https://huggingface.co/spaces/shiftenv/c2rust-rl)
- 📓 **Colab Notebook**: [train_qwen_colab.ipynb](https://colab.research.google.com/...)
- 📊 **W&B Training Run**: [wandb.ai/c2rust-rl](https://wandb.ai/...)
- 🦀 **GitHub**: [shiftenv-v2](https://github.com/sumit-s-nair/shiftenv-v2)

---

*Built for the OpenEnv Hackathon India 2026 by Team Puru. C code doesn't have to be unsafe forever.*
