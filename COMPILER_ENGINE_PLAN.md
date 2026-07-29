# Lightweight NVIDIA CompilerEngine for GraphZero

## Summary

Implement a concrete, search-agnostic `CompilerEngine` in `graphs/rust/compiler`. It will own compiler states and actions, delegate rewrite semantics to `TransformBank`, and use the existing compile/evaluator contracts. GraphZero remains responsible for its adapter, features, STOP, reward projection, replay, and orchestration.

V1 is NVIDIA-only, uses local `EvaluationExecutor`, masks STOP through GraphZero while candidates exist, and deterministically completes a cloned state when measurement happens at `max_steps`.

## Compiler API and State

- Add `compiler::engine` and `compiler::actions`, exporting:
  - `CompilerEngine`, `CompilerStateId`, and `CompilerCandidateId`.
  - Portable `CompilerStateHash` and `CompilerActionHash`.
  - Plain `CompilerStateView` containing the module, canonical node order, and source fingerprint.
  - Plain candidate descriptions exposing definition ID, match site, and typed parameters.
  - `CompilerMeasurement::Passed { accuracy, performance }` and `AccuracyFailed { accuracy }`.
- Construct the engine from one required NVIDIA domain record containing target, bufferization/lowering policies, payload compiler, evaluator, and measurement workload shape. Do not add a backend trait or generic pipeline abstraction.
- Insert roots as imported programs plus their exact evaluation workload. Normalize each root through target assignment and mandatory model-to-execution completion before search begins.
- Keep tensor backings and evaluation tensors once per root; child states contain full `Module` copies plus a root reference.
- Use compact generation-checked handles and one private refcounted arena implementation for states and candidates. Candidates retain their parent state; batched release reclaims states only after all graph and candidate references are gone.
- Deduplicate equivalent child states by portable state hash. Cache canonical bytes, node order, enumerated actions, analysis products, and the compiled program on the state record; all caches disappear with that state.

## Enumeration, Apply, and Artifacts

- Build one frozen transform bank at engine construction:
  - All target-independent optional optimization definitions.
  - NVIDIA provider-selection definitions.
  - Mandatory model completion definitions remain outside the search bank.
- Derive the action-set identity from an explicit engine schema version, ordered definition IDs, and resolved target fingerprint. Semantic changes require a version bump.
- Enumerate the complete canonical action list once per state, then apply caller limits only while copying handles to the output. A limited request must never alter later enumeration.
- Apply exactly one state-bound `TransformAction`:
  - Reject unknown, released, stale, or cross-parent handles.
  - Interpret through `TransformBank`, apply to an owned module clone, and run rewrite verification.
  - Leave the parent unchanged on every failure.
  - Compute canonical bytes/hash and deduplicate the successful child.
- Add standard byte access traits to existing opaque fingerprints needed by the engine; do not add string parsing or getter methods.
- Define an exact versioned binary state artifact containing schema identity, source fingerprint, engine-domain fingerprint, state hash, and canonical module bytes.
  - Restoration requires the matching source root to already be registered locally.
  - Reject unknown versions, wrong domains, missing roots, trailing bytes, and hash mismatches.
  - Do not embed external tensor payloads or implement remote transport in v1.

## Measurement and GraphZero Boundary

- `CompilerEngine::measure` always works on a clone:
  1. Deterministically complete unresolved NVIDIA provider choices.
  2. Compile through the selected NVIDIA compile path.
  3. Run accuracy first.
  4. Run performance only after accuracy passes.
- Cache the compiled program on the searched state; rely on the existing evaluator’s bounded request/result caches for repeated execution.
- Return typed compiler outcomes, not GraphZero rewards:
  - Accuracy pass returns both reports.
  - Accuracy failure returns its report and skips performance.
  - Compile or evaluator failures remain typed `CompilerEngineError` results.
- Expose only the compiler information required by an external feature extractor: canonical module/node order and transform action semantics. Do not add model tokens, priors, STOP, replay records, or GraphZero dependencies.
- The later GraphZero adapter will:
  - Map native handles and hashes directly into `GraphEngine`.
  - Use `mask_stop = true`; STOP becomes selectable only when no compiler candidates remain.
  - Map `MeasureOptions.samples` to measured iterations and require unsupported timeout behavior to be disabled initially.
  - Score successful measurements as negative execution-median milliseconds.
  - Map `AccuracyFailed` to a valid reward of `-10_000.0`.
  - Map compile/evaluator errors to measured-invalid failures.
  - Implement `CompilerFeatureExtractor` using the canonical state/action views.
- Update compiler documentation to describe the implemented API and local-only measurement boundary rather than the current design-only status.

## Test Plan

- Add external compiler tests covering:
  - Stable state/action/domain hashes across independent engine instances.
  - Canonical and limit-independent enumeration.
  - Exact apply, parent immutability, state deduplication, and verifier failure atomicity.
  - Unknown, released, stale, and cross-parent handles.
  - Multi-root isolation and reference-counted release.
  - State artifact round-trip and every rejection case.
  - Max-step-style measurement completing only a clone.
  - Accuracy pass running performance and reporting execution latency.
  - Accuracy failure skipping performance.
  - Compile/evaluator failures remaining distinguishable.
  - Cache hits using fake payload compiler and evaluator call counters.
- Use fake compiler/evaluator implementations for ordinary tests; add an ignored NVIDIA-device end-to-end test.
- Verify with:
  - `cargo fmt --all -- --check`
  - `cargo clippy -p compiler --all-targets -- -D warnings`
  - `cargo test -p compiler`
  - The ignored NVIDIA test on a configured CUDA machine.

## Assumptions

- Intel engine support, remote measurement bundles, model features, GraphZero adapter code, and GraphZero’s missing threaded single-agent orchestration are follow-on work outside this graphs-side implementation.
- Numerical checks remain necessary because valid typed IR can still be changed incorrectly by a rewrite, floating-point transformation, lowering, code generation, or runtime implementation.
- Public records expose fields directly. Constructors exist only where they validate invariants; no getters/setters, convenience builders, forwarding modules, speculative traits, or inline tests are added.
