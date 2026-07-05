# Lane Pipeline Consolidation Implementation Spec

Status: implementation work order (CODEBASE_IMPROVEMENTS.md findings
4+6, plus the finding-3 disposition and the episode-husk retention
wart)

Purpose: collapse the four near-identical lane run loops in
`lanes.rs` (`run`, `run_with_replay`, `run_featurized`,
`run_featurized_with_replay`) into one internal pipeline with small
policy adapters. The duplication is not cosmetic -- it has already
produced real divergence risk twice: the candidate-capacity leak fix
and the opponent-rollout integration each had to be hand-applied to
multiple copies of the same lifecycle, and a miss in one copy would
have shipped. Admission, gating, parking, reply draining, completion,
projection, release ordering, replay append, and rollout interception
must live in exactly one place.

Authority: `GZ_ORCHESTRATOR.md`, `GZ_ENGINE_RELEASE_IMPL.md` (release
ordering), `GZ_OPPONENT_IMPL.md` (rollout hooks are lane-driven).
Contract wins; report conflicts.

Read before starting:

```text
crates/gz-orchestrator/src/lanes.rs   all four run paths + the lane
                                      fns + OpponentRollout +
                                      release_episode_handles +
                                      clear_replayed_episode_trace
crates/gz-orchestrator/src/pool.rs    WorkerPool -- unchanged
crates/gz-orchestrator/tests/         threaded.rs, featurized.rs,
                                      replay_integration.rs,
                                      featurized_process.rs -- the
                                      equality oracles
```

## Hard Constraints

```text
Every stage ends with the full verification battery green; commit per
stage.
The four public entry points and their signatures are UNCHANGED for
batch paths. Replay-path return shape may change (stage 2) but only
as specced there.
Replay stores must be BYTE-IDENTICAL pre/post refactor for fixed
seeds on all four paths -- verify with a worktree build of the
pre-refactor commit writing to a second store and a byte comparison
of sampled rows plus counters (the house cross-commit discipline).
Batch-path episode results must compare equal (the existing threaded
== serial oracles are the guard).
Lifecycle order within the pipeline is contractual and documented in
one place: complete -> intercept rollout -> extract features ->
project -> append (ack) -> release -> observe -> trace-drop.
```

## Stage 1: The Pipeline

```text
One internal lane runner generic over two adapters:
  features: NoFeatures | Extract(extractor)   (what parks with a row)
  sink:     Collect | Replay { store-provider-rollout bundle }
The adapters own only their step of the lifecycle; the runner owns
admission (rollout admission BEFORE root admission -- the starvation
rule), gate polling, parking, reply draining with version
observation, completion, and release ordering. The four public fns
become thin constructors of adapter combinations.
No logic changes ride along. Any behavior difference found while
unifying (there will be at least one -- the four copies have already
drifted) is REPORTED in the work-order review, not silently picked;
the replay-path variant is authoritative where they disagree.
```

## Stage 2: Stop Retaining Episode Husks On Replay Paths

```text
Replay lanes currently push every completed episode (trace-dropped
husk, ~1 KB) into a per-lane Vec for the run summary -- O(episodes)
retention on unbounded runs, and the site of the 20 MB/episode
capacity leak. Replace with aggregated per-lane counts (episodes,
appended, dropped, and whatever the summary actually reads).
This CHANGES the replay-run return shape: ThreadedReplayRun keeps
counts and batch sizes, loses per-episode husks. Callers (summarize()
in gz-cli, tests) read counts already -- update the few that touch
lane.episodes on replay paths. Batch paths keep full episodes:
they are the equality-oracle surface and callers inspect them.
```

## Stage 3: Small Fixes In The Same File

```text
Finding 6: map_replay_error preserves the ReplayError detail in the
engine error message (compact, one line, includes the kind) instead
of the constant "replay sink failed".
Finding 3 disposition (documentation, NOT a type split): batch-path
returned episodes contain engine handles that the orchestrator has
already released. Document the invariant on the returned types
("handles are opaque identifiers for equality/inspection; deref is a
contract violation; debug builds panic via generation checks") and
in GZ_ORCHESTRATOR.md. The type-level split was considered and
declined: churn across every oracle test for an invariant the debug
generation checks already enforce dynamically.
```

Acceptance checklist:

```text
one lifecycle implementation; the four entry points are adapter
wiring only (reviewer check: no admission/parking/release logic
outside the runner)
byte-identical stores and counters vs the pre-refactor commit on all
four paths; batch episode equality oracles untouched
opponent rollout tests pass unchanged (priority admission, interception,
version tracking)
replay-path memory: per-lane retention is O(1) in episodes
a replay append failure surfaces its ReplayError kind in the error
message (test: fail an append via a closed store or schema mismatch
and assert the message)
```

## Out Of Scope

```text
changing the eval batcher, pool, or channel topology
serial.rs (its replay-less single-engine loop is not part of the
duplication cluster)
adding new run modes or adapter kinds beyond the existing four
```
