# Codebase Improvements

Reviewed state: `a7ccf3a`

This file records the concrete code-quality and architecture findings from the
current GraphZero review. The high-level crate shape is sound: `gz-engine` stays
foundational, `gz-search` is mostly engine-generic, replay uses portable
identities, and evaluation/features/replay/orchestration are separate modules.
The main issues are lifecycle ownership, shallow orchestration duplication, and
large implementation files with too much locality loss.

## Disposition (2026-07-05)

Findings triaged into three work orders; two items adjusted on review:

- Findings 1+2 -> `specs/GZ_SEARCH_RELEASE_IMPL.md` (do first). Both
  verified real. Exposure note: production configs (self-average /
  policy references, stub/torch evaluators) do not hit either leak;
  greedy/beam/random references and the measure evaluator do.
- Findings 4+6 -> `specs/GZ_LANES_DEEPEN_IMPL.md`. The duplication has
  already forced multi-site fixes twice (capacity leak, opponent
  rollout); the work order also removes the per-episode summary-husk
  retention on replay paths.
- Finding 5 -> `specs/GZ_GUMBEL_SPLIT_IMPL.md` (mechanical motion,
  last).
- Finding 3: type-level split DECLINED -- churn across every equality
  oracle for an invariant the debug generation checks already enforce
  dynamically; released-handle documentation folded into the lanes
  work order instead. Also note replay paths now return trace-dropped
  husks (empty steps) since the capacity-leak fix, shrinking the
  exposed surface further.

## Priority Order

1. Fix engine handle release ownership outside Gumbel.
2. Fix temporary Whittle evaluator graph handle release.
3. Make returned orchestrator episodes explicit about released local handles.
4. Deepen the threaded lane orchestration module.
5. Split the Gumbel MCTS implementation into smaller internal modules.
6. Preserve replay error detail at the orchestrator seam.

## 1. Release Ownership Is Incomplete Outside Gumbel

Severity: High

Status: Fixed in `8c9612f`.

Files:

- `crates/gz-search/src/greedy.rs`
- `crates/gz-search/src/beam.rs`
- `crates/gz-search/src/random.rs`
- `crates/gz-search/src/episode.rs`
- `crates/gz-orchestrator/src/reference.rs`

Finding:

`GreedySearch`, `BeamSearch`, and `RandomSearch` create engine-owned graph and
candidate handles through `GraphEngine::candidates` and `GraphEngine::apply`,
but `SearchEpisode` does not record the full set of created handles. The Gumbel
path has explicit `created_graphs` and `created_candidates` ownership, and the
orchestrator releases those handles. The non-Gumbel search path lacks the same
contract.

Observed call sites:

- `crates/gz-search/src/greedy.rs:89` calls `engine.candidates`.
- `crates/gz-search/src/greedy.rs:102` calls `engine.apply`.
- `crates/gz-search/src/beam.rs:391` calls `engine.candidates`.
- `crates/gz-search/src/beam.rs:411` calls `engine.apply`.
- `crates/gz-search/src/random.rs:90` calls `engine.candidates`.
- `crates/gz-search/src/random.rs:133` calls `engine.apply`.
- `crates/gz-orchestrator/src/reference.rs:129`, `:165`, and `:201` use these
  searches as reference providers without releasing created handles.

Risk:

Long Whittle runs that enable `--reference greedy`, `--reference beam`, or
`--reference random` can bypass the release contract and accumulate arena
handles. This is a correctness and memory-growth risk.

Recommended fix:

Add created-handle ownership to `SearchEpisode`, mirroring the Gumbel episode
contract. The reference providers should project portable replay data first,
then release the created graph and candidate handles before returning. Add
focused tests that run each reference provider on `WhittleEngine` and verify the
arena counters return to the expected retained root/source state.

## 2. `WhittleMeasureEvaluator` Leaks Temporary Apply Results

Severity: High

Status: Fixed in `8c9612f`.

File:

- `crates/gz-eval-whittle/src/lib.rs`

Finding:

`WhittleMeasureEvaluator` computes policy logits by applying each candidate to
the input graph, measuring the resulting graph, and comparing the reward delta.
The temporary `applied.after` graph handles are not released.

Observed call site:

- `crates/gz-eval-whittle/src/lib.rs:43` calls `engine.apply`.

Risk:

Every evaluation request can allocate one temporary graph per candidate. Gumbel
MCTS evaluates many leaves and candidate sets, so this can dominate memory
growth in selfplay benchmarks.

Recommended fix:

Track temporary graph handles created while scoring candidates and call
`engine.release(&created_graphs, &[])` before returning. Ensure release still
happens when a later candidate measurement fails after earlier temporary graphs
were created. Add a `gz-eval-whittle` integration test that evaluates a request
with multiple candidates and checks the Whittle arena count after evaluation.

## 3. Orchestrator Results Expose Released Local Handles

Severity: Medium

File:

- `crates/gz-orchestrator/src/serial.rs`

Finding:

`SerialGumbelOrchestrator` releases `episode.created_graphs` and
`episode.created_candidates` before returning `OrchestratedEpisode`, but the
returned `GumbelEpisode` still contains process-local graph and candidate
handles. Those handles are already released by the time callers receive them.

Observed call site:

- `crates/gz-orchestrator/src/serial.rs:73` releases episode handles before
  returning the episode.

Risk:

Current replay/training callers mostly use portable contexts, so this is not
necessarily a live bug. The interface is still misleading: it exposes handles
that callers must not use, and that invariant is not encoded in the type.

Recommended fix:

Introduce a returned episode type that contains only portable replay/training
data after release, or document and enforce the invariant at the orchestrator
interface. Prefer a type-level split if callers do not need local handles after
orchestration.

## 4. Threaded Lane Orchestration Is Shallow And Duplicated

Severity: Medium

Status: Fixed in `8dfa9f1`.

File:

- `crates/gz-orchestrator/src/lanes.rs`

Finding:

`lanes.rs` has four public run paths with very similar lifecycle logic:

- `run`
- `run_with_replay`
- `run_featurized`
- `run_featurized_with_replay`

The same concerns repeat across these implementations: lane admission, blocked
task parking, eval batching, response routing, episode finalization, release
ordering, replay append, feature extraction, and opponent context handling.

Observed public seams:

- `crates/gz-orchestrator/src/lanes.rs:121`
- `crates/gz-orchestrator/src/lanes.rs:212`
- `crates/gz-orchestrator/src/lanes.rs:323`
- `crates/gz-orchestrator/src/lanes.rs:421`

Risk:

Lifecycle bugs are likely to be fixed in one path but missed in another. This
is especially risky around handle release, replay append ordering, feature
snapshot lifetimes, and evaluation response draining.

Recommended fix:

Deepen the lane runner into one internal pipeline with small policy adapters for
optional replay, optional feature extraction, and optional reference generation.
The external interface can stay close to the current four entry points, but the
implementation should concentrate lifecycle ownership in one place.

## 5. Gumbel MCTS Has Too Much Implementation In One File

Severity: Medium

Status: Fixed in `2d2de83`.

File:

- `crates/gz-search/src/gumbel/`

Finding:

`gumbel.rs` contains public config/types, episode task state, root task state,
tree storage, sequential-halving math, schedule math, and engine work handling
in one large implementation file.

Observed regions:

- `crates/gz-search/src/gumbel.rs:392` starts `GumbelEpisodeTask`.
- `crates/gz-search/src/gumbel.rs:757` starts `GumbelRootTask`.
- `crates/gz-search/src/gumbel.rs:1489` starts tree internals.
- `crates/gz-search/src/gumbel.rs:1744` starts math/schedule helpers.

Risk:

The external interface is reasonably small, but implementation locality is poor.
Changing schedule math, tree reuse, root sampling, or work polling requires
working inside the same large file. That makes subtle search bugs harder to
review and test.

Recommended fix:

Keep the public `gz-search` interface stable, but split internal implementation
modules by concept:

- `gumbel/task.rs` for episode/root task state machines.
- `gumbel/tree.rs` for tree storage and visit accounting.
- `gumbel/schedule.rs` for sequential-halving and considered-action math.
- `gumbel/types.rs` for public records if that improves readability.

Do this after the release leaks are fixed, because it is lower-risk cleanup.

## 6. Replay Errors Are Erased At The Orchestrator Seam

Severity: Low

Status: Fixed in `8dfa9f1`.

File:

- `crates/gz-orchestrator/src/lanes.rs`

Finding:

Replay append failures are converted to a generic internal engine error:

- `crates/gz-orchestrator/src/lanes.rs:1373`

The original `ReplayError` detail is dropped.

Risk:

Operational failures in RocksDB, binary encoding, or replay schema handling are
harder to diagnose from orchestrator logs and benchmark failures.

Recommended fix:

Preserve replay error detail in the mapped `EngineError` message, or add a
dedicated error variant at the orchestrator seam if the current engine error
type should not carry replay internals. Keep the message compact, but include
the replay error kind.

## Top Recommendation

Fix findings 1 and 2 first. They are correctness issues in the engine-handle
lifecycle and can affect long-running selfplay memory behavior. The lane runner
and Gumbel file structure are worth improving after ownership is correct,
because those refactors will be easier to verify once the release contract is
consistent across all search paths.
