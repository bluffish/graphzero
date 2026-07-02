# AGENTS.md

Operating instructions for coding agents working in GraphZero. Read this file
before every task.

**Working code only. Finish the job. Plausibility is not correctness.**

GraphZero is a performance-first Rust codebase for async graph/search pipelines.
The first target is a modular `GraphEngine` abstraction tested with fake and
Whittle-backed engines before any compiler backend exists.

---

## 0. Non-Negotiables

These rules override everything else in this file when in conflict.

1. **No flattery, no filler.** Start with the answer or the action.
2. **Disagree when you disagree.** If the premise is wrong, say so before doing
   the work.
3. **Never fabricate.** Not file paths, not test results, not APIs, not commit
   hashes. If you do not know, inspect or run the code.
4. **Stop when confused.** If two interpretations materially change the output,
   ask before editing.
5. **Touch only what you must.** No drive-by refactors, formatting churn, or
   adjacent cleanup unrelated to the request.
6. **Performance claims require measurement.** Do not say something is faster
   without a benchmark or profile.
7. **Runtime reward requires measurement.** Rows enter replay only after
   `GraphEngine::measure` returns a terminal measure result.

---

## 1. Project Priorities

GraphZero optimizes for:

1. **Modularity.** Search depends on `GraphEngine`, not concrete game/compiler
   state. Feature extraction is separate. Replay is separate. Evaluation is
   separate.
2. **Code cleanliness.** Interfaces are small, typed, deterministic, and easy
   to test. Avoid pass-through modules and speculative hooks.
3. **Performance.** Hot loops use typed handles, bounded queues, batching,
   arenas/caches where justified, and no JSON/object churn.
4. **Correctness.** Deterministic candidate enumeration, deterministic apply,
   measured-before-replay, and contract tests for every engine adapter.

When these conflict, prefer correctness first, then performance, then minimal
surface area.

---

## 2. Architecture Rules

- Crate names use `gz-*`. The repo and binary are `graphzero`.
- `gz-engine` is the foundational crate. It owns dependency-light engine
  traits, hashes, options, result types, and errors.
- `gz-engine` must not depend on torch, Python, concrete engine
  implementations, search, replay, or an async runtime choice.
- `gz-search` must not depend on `gz-engine-whittle`, future compiler adapters,
  replay storage, or training code.
- Search stores `E::Graph` and `E::Candidate` handles only. It never owns full
  graph bodies.
- Candidate semantics are engine-owned. Search must not assume site-level,
  family-level, compiler-specific, or Whittle-specific candidate structure.
- `FeatureExtractor<E>` is separate from `GraphEngine`.
- `GraphEngine::measure` owns terminal measurement. Async scheduling belongs to
  the orchestrator or `EngineServer`.
- Replay rows store portable identities such as `GraphHash` and
  `PortableSearchActionRef`, not process-local handles.
- Whittle is the first concrete engine adapter; `gz-engine-fake` is deferred
  until search/orchestration tests need it.

---

## 3. Performance Rules

- Design hot paths batch-first. Single-call APIs can be wrappers.
- Use bounded queues. Do not add unbounded channels in actor/eval/measure/replay
  paths.
- Avoid JSON, dynamic maps, strings, and allocation-heavy objects in search hot
  loops.
- Cache only with a clear key and invalidation rule. Cache keys must include
  graph hash, action-set hash, engine version, and measure config where relevant.
- Prefer stable integer handles and compact structs over cloned graph/state
  bodies.
- If a change is for speed, define the benchmark before editing and report the
  before/after numbers.
- If a profile points to a bottleneck, fix the bottleneck, not the symptom.

---

## 4. Before Writing Code

Goal: understand the problem and the codebase before producing a diff.

- State the plan in one or two sentences before editing. For non-trivial work,
  include verification.
- Read the files you will touch and the files that call them.
- Match existing patterns in the repo unless the task explicitly changes the
  architecture.
- Surface assumptions out loud. Do not bury them in code.
- If two approaches exist and the tradeoff matters, present both before editing.
- For performance work, identify the success metric first.

---

## 5. Writing Code

Goal: the minimum code that solves the stated problem.

- No features beyond what was asked.
- No abstraction for a single use unless it is the requested interface boundary.
- No speculative configurability or "future extensibility."
- Handle failures that can actually happen. Do not add elaborate error handling
  for impossible states.
- Prefer deleting code over adding code when that solves the problem.
- Keep public interfaces small and strongly typed.
- Use explicit names for domain concepts. Avoid vague names like `manager`,
  `helper`, `thing`, or `data` when a domain name exists.
- Comments should explain non-obvious reasoning, invariants, or performance
  constraints. Do not narrate obvious code.

---

## 6. Surgical Changes

Goal: clean, reviewable diffs.

- Do not improve adjacent code, comments, formatting, or imports that are not
  part of the task.
- Do not refactor working code just because you are in the file.
- Do not delete pre-existing dead code unless asked.
- Clean up orphans created by your own changes.
- Match the local style exactly.
- Every changed line must trace to the request. If not, revert that line.

---

## 7. Verification

Goal: define success as something executable, then run it.

For every task:

1. State the success criteria before writing code.
2. Add or update tests/benchmarks where practical.
3. Run the relevant verification.
4. Read the output.
5. Fix causes, not tests.

Use the narrowest verification during iteration and broader verification before
calling work done.

Performance work must include:

```text
baseline measurement
change
new measurement
percent change or absolute delta
benchmark command
```

---

## 8. Tool Use

- Prefer `rg`/`rg --files` for search.
- Prefer running code to guessing about code.
- Use `cargo test`, `cargo bench`, `cargo clippy`, and `cargo fmt` once the Rust
  workspace exists.
- Use `apply_patch` for manual edits.
- Do not use destructive git commands unless explicitly requested.
- The worktree may be dirty. Never revert user changes unless explicitly asked.
- When reading logs, errors, or stack traces, read the whole thing.

---

## 9. Communication

- Direct, concise, factual.
- Findings and risks first for reviews/debugging.
- Say what changed and how it was verified.
- If verification was not run, say that plainly.
- Do not end with vague offers. Give concrete next steps when useful.

---

## 10. When To Ask

Ask before proceeding when:

- The request has two plausible meanings and the choice changes the design.
- The change touches a load-bearing interface, storage schema, or migration
  path.
- You need credentials or production resources.
- The stated goal conflicts with the literal request.

Proceed without asking when:

- The ambiguity can be resolved by reading code.
- The task is trivial and reversible.
- The user already answered the question in this session.

---

## 11. Project Context

### Stack

- Language: Rust planned.
- Package manager: Cargo.
- Runtime: async single-process first, process boundaries later.
- Storage: RocksDB + compact binary replay rows.

### Current Layout

```text
specs/
  CODEBASE_OUTLINE.md
  GZ_EVAL.md
  GZ_ENGINE.md
  GZ_ENGINE_WHITTLE.md
  GZ_ORCHESTRATOR.md
  GZ_ORCHESTRATOR_SERIAL_IMPL.md
  GZ_ORCHESTRATOR_FEATURIZED_IMPL.md
  GZ_ORCHESTRATOR_MULTI_WORKER_IMPL.md
  GZ_EVAL_SERVICE.md
  GZ_FEATURES.md
  GZ_ORCHESTRATOR_REPLAY_IMPL.md
  GZ_EVAL_PROTOCOL.md
  GZ_EVAL_SERVICE_IMPL.md
  GZ_PYTHON.md
  GZ_PYTHON_FRAMEWORK_IMPL.md
  GZ_REPLAY.md
  GZ_SEARCH.md
  GZ_SEARCH_GUMBEL_MCTS.md
  GZ_TRAINING_DATA_IMPL.md
```

Planned layout:

```text
crates/
  gz-engine/
  gz-engine-fake/
  gz-engine-whittle/
  gz-features/
  gz-search/
  gz-eval/
  gz-replay/
  gz-orchestrator/
  gz-cli/
```

### Commands

These are pending until the Rust workspace exists.

```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test --all
cargo bench
```

---

## 12. Project Learnings

Add corrections here when the user rejects an approach or a real bug teaches a
new rule. Write one concrete rule per line.

- Use `graphzero` for the repo and binary name, and `gz-*` for crate names.
- Keep `GraphEngine::measure` as the measurement interface; do not introduce a
  separate runtime oracle unless the design is explicitly reopened.
- Do not implement a learner before the async engine/search/measure/replay
  pipeline works.
- Do not start with a compiler backend; validate the architecture with the
  Whittle-backed engine first.
- Skip `gz-engine-fake` for now unless search/orchestration tests need a
  smaller deterministic adapter.
- Engineer GraphZero for maximum measured performance subject to correctness;
  avoid convenience designs that add avoidable overhead to hot paths.
- Prefer simple, clean, concise syntax; avoid ceremony unless it improves
  clarity or correctness.
- Avoid redundant tests and over-verification; verify the behavior and risk at
  the narrowest useful scope.
- Architect `gz-search` around many parallel async-driven Gumbel-MCTS selfplay
  workers; greedy search is only the first implementation slice.
- Model STOP as a search-level action appended by `gz-search`, never as a
  `GraphEngine` candidate.
- Use same-index opponent trajectory alignment only; root_step equals the
  learner episode step.
- Keep Rust tests out of `src/` files; put tests in each crate's `tests/`
  directory instead.
