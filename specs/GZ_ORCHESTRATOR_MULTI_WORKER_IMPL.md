# Multi-Worker Driver Implementation Spec

Status: implementation work order

Purpose: implement the multi-worker selfplay drivers in `gz-orchestrator`:
first a single-threaded batched driver that proves cross-worker eval batching
over the task protocol, then a threaded driver with engine lanes and an eval
batcher thread. Both drive many width-1 `GumbelEpisodeTask` workers and must
reproduce the serial orchestrator's episodes bit-for-bit.

Authority: `GZ_ORCHESTRATOR.md` owns the design contract. This document is
the ordered work plan. If they disagree, `GZ_ORCHESTRATOR.md` wins; report
the conflict instead of improvising.

Read before starting:

```text
specs/GZ_ORCHESTRATOR.md                 (design contract)
specs/GZ_ORCHESTRATOR_SERIAL_IMPL.md     (previous work order; shapes reused)
crates/gz-orchestrator/src/serial.rs     (drive loop + expand servicing)
crates/gz-search/src/work.rs             (protocol types)
crates/gz-eval/src/types.rs              (Evaluator::evaluate_batch contract)
crates/gz-engine-whittle/src/generator.rs (WhittleGraphGenerator)
```

## Hard Constraints

```text
Work in the stage order below. Each stage must compile and pass
`cargo test --all` before the next stage starts.
All changes live in gz-orchestrator. gz-search, gz-eval, gz-engine, and the
Whittle crates must not change.
No new dependencies. Threading uses std only: std::thread::scope,
std::sync::mpsc (sync_channel for bounded channels). No tokio, no
crossbeam, no rayon, no async.
std::time::Instant is allowed only in the stage-B batcher flush logic.
Never in tasks, never in stage A.
The stage-0 goldens in gz-search and every existing test must pass
unchanged at every stage.
Only portable data crosses a lane boundary in stage B: the message sent to
the batcher carries EvalRequest plus routing ids, never E::Graph,
E::Candidate, or measure options. This is structural, not advisory: the
message type must not have generic engine parameters.
Every channel is bounded, with capacity chosen so the send that could
deadlock provably cannot block (rules below).
Protocol errors and driver bugs are EngineError::Internal with the stable
messages listed below. No new error enum.
Every stage ends with: cargo fmt, cargo test --all,
cargo clippy --all-targets --all-features -- -D warnings.
```

## Stage 0: Prerequisites

Outstanding review findings from the serial work order. Do these first.

```text
1. Commit the current working tree before any new work: one commit for the
   golden fixtures, one for the serial slice. The goldens must exist in git
   history before this work order touches the orchestrator.
2. Add the two missing protocol tests to gz-search/tests/gumbel_task.rs:
   - dropping a task with an outstanding token is safe (create task, poll
     to Work, drop the task; no panic)
   - a rejected ApplyResult masks the action and the next poll emits new
     work without consuming a resume (hand-drive: resume with a rejected
     ApplyResult, then poll and observe the next Work item)
```

These are gz-search test additions only; no gz-search source changes.

## Design Summary

Both drivers share one core: a worker pool of `GumbelEpisodeTask` slots that
are driven inline (Expand/Apply/Measure serviced immediately against an
engine) until each task either parks on an Eval or completes. The drivers
differ only in what happens to parked evals:

```text
stage A (batch.rs):  collect parked requests, call Evaluator::evaluate_batch
                     directly, resume in slot order. One thread, one engine,
                     fully deterministic.
stage B (lanes.rs):  each lane thread runs the same pool against its own
                     engine and sends parked requests to a batcher thread;
                     the batcher forms time/size-bounded batches, calls the
                     same Evaluator, and routes outputs back per lane.
```

Both require the portable `gz_eval::Evaluator`, not `EngineEvaluator`.
Engine-aware evaluators are lane-local and unbatchable by design; they stay
on the serial orchestrator. Use `RandomValueEvaluator` or scripted portable
evaluators in tests.

The per-episode equality oracle: because tasks have one outstanding token
and the required evaluators are order-independent (a gz-eval contract for
`RandomValueEvaluator`: batch grouping and request order must not change a
row's value), an episode produced by either driver must equal, field for
field, the episode `SerialGumbelOrchestrator` produces for the same config,
root, and evaluator. Whittle arena state differing between runs (one shared
engine, interleaved episodes) must not change results; graphs are
deduplicated by hash and engine behavior is deterministic per graph.

## Stage 1: Shared Pieces

New crate layout:

```text
crates/gz-orchestrator/src/
  lib.rs        add exports
  ids.rs        unchanged
  serial.rs     service_work moves out (below)
  service.rs    crate-private inline work servicing
  root.rs       RootSource
  pool.rs       crate-private worker pool
  batch.rs      stage A
  lanes.rs      stage B (stage 3)
```

### Episode record

Rename `SerialEpisode` to `OrchestratedEpisode` (same fields) and keep a
back-compat alias so nothing outside the crate breaks:

```rust
pub struct OrchestratedEpisode<G, C> {
    pub worker_id: WorkerId,
    pub episode_id: EpisodeId,
    pub episode: GumbelEpisode<G, C>,
}

pub type SerialEpisode<G, C> = OrchestratedEpisode<G, C>;
```

### Inline servicing

Move `serial.rs`'s private `service_work` and expand servicing into
`service.rs` as crate-private functions, and split out an engine-only
variant used by the pool:

```rust
pub(crate) fn service_engine_work<E: GraphEngine>(
    engine: &mut E,
    work: &SearchWork<E::Graph, E::Candidate>,
) -> EngineResult<Option<SearchWorkResult<E::Graph, E::Candidate>>>;
```

Returns `Ok(Some(result))` for Expand/Apply/Measure, `Ok(None)` for Eval
(the caller decides what to do with eval work), and an
`internal("unsupported search work")` error for unknown kinds. The serial
orchestrator keeps its behavior by composing this with an immediate
evaluator call.

### RootSource

```rust
pub trait RootSource<E: GraphEngine> {
    fn next_root(&mut self, engine: &mut E) -> EngineResult<Option<E::Graph>>;
}
```

Rules:

```text
None means exhausted: no more episodes will be admitted from this source.
next_root receives the engine because root generation is engine-owned work
(WhittleGraphGenerator::sample_into) and graph handles are instance-local.
Provide a blanket impl for FnMut(&mut E) -> EngineResult<Option<E::Graph>>
so tests can use closures.
Provide CountedRoots<F> (or equivalent) that yields N roots from a factory
closure, for budgeted selfplay runs.
```

### Worker pool

`pool.rs`, crate-private. One slot per worker:

```rust
enum Slot<G, C> {
    Idle,
    Running(GumbelEpisodeTask<G, C>),
    Parked {
        task: GumbelEpisodeTask<G, C>,
        token: WorkToken,
        request: EvalRequest,
    },
}
```

Pool operations, all deterministic by slot order:

```text
admit(engine, root_source, search, identity, context):
  for each Idle slot in slot index order, pull a root; on Some, create a
  GumbelEpisodeTask and mark Running; on None, stop admitting; episode ids
  are assigned at admission, in admission order, from the caller's counter

drive(engine):
  for each Running slot in slot index order, loop:
    poll the task
    Work(Eval w)   -> extract w.request, drop the rest of w, park the slot
                      with (token, request); next slot
    Work(other)    -> service_engine_work inline, resume, poll again
    Blocked        -> internal("worker blocked") (unreachable: the pool
                      resolves every non-eval token before polling again)
    Done(episode)  -> record OrchestratedEpisode, slot becomes Idle
  returns completed episodes in completion order

parked(): slots currently Parked, in slot index order
resume(slot, output): task.resume(token, Eval(output)); slot -> Running
active(): true when any slot is Running or Parked
```

Rules:

```text
The pool never touches an evaluator.
The pool retains nothing from EvalWork except the request and token; graph
and candidate copies are dropped at park time.
WorkerId = slot index (offset by lane in stage B).
The eval request stored at park time is the one sent out; the task's own
retained copy still performs resume validation. Do not validate twice in
the pool.
```

## Stage 2: Batched Driver (Single-Threaded)

`batch.rs`:

```rust
pub struct BatchedGumbelOrchestrator<E, V> {
    engine: E,
    evaluator: V,
    search: GumbelMcts,
    workers: NonZeroUsize,
}

pub struct BatchedRun<G, C> {
    pub episodes: Vec<OrchestratedEpisode<G, C>>,
    pub batch_sizes: Vec<usize>,
}

impl<E, V> BatchedGumbelOrchestrator<E, V>
where
    E: GraphEngine,
    V: Evaluator,
{
    pub fn new(engine: E, evaluator: V, search: GumbelMcts, workers: NonZeroUsize) -> Self;

    pub fn run<R: RootSource<E>>(
        &mut self,
        roots: &mut R,
        context: GumbelEpisodeContext,
    ) -> EngineResult<BatchedRun<E::Graph, E::Candidate>>;
}
```

The scheduling loop, in strict phases:

```text
loop:
  1. admit into idle slots (slot order) until slots or roots run out
  2. drive all Running slots (slot order); collect completed episodes
  3. if nothing is Parked and nothing is Running: return
  4. collect parked requests in slot order; record batch size
  5. evaluate_batch(requests, &mut outputs); one output per request in
     request order (gz-eval contract)
  6. resume parked slots with their outputs, in the same slot order
  7. repeat
```

Rules:

```text
context applies to every episode; per-episode contexts and opponent
trajectory leases are out of scope.
eval errors map through eval_error_to_engine_error and abort the run.
Fail-fast: the first EngineError from any slot aborts the whole run.
Per-worker fault isolation is deferred.
episode ids are global across the run, assigned at admission.
batch_sizes records every evaluate_batch call's request count, in order.
Blocked from a task is internal("batched driver blocked").
```

Determinism: for a fixed engine config, search config, root sequence, and
order-independent deterministic evaluator, `run` returns identical episodes
and identical batch_sizes on every invocation.

### Stage 2 tests (`tests/batch.rs`)

Use `WhittleEngine`, `WhittleGraphGenerator` (fixed seeds), and
`RandomValueEvaluator`, mirroring the existing serial tests.

```text
one worker, K roots: episodes equal SerialGumbelOrchestrator run on a fresh
  identical engine, root by root, field for field
W workers (W >= 4), K > W roots: each episode equals the serial episode for
  the same root; episode ids are 0..K in admission order
first batch size equals W when K >= W and max_steps > 0
K not a multiple of W drains fully: episodes.len() == K
empty root source returns zero episodes and zero batches
evaluator failure aborts: an Evaluator returning Err aborts the run with an
  error (scripted failing evaluator)
determinism: two identical runs produce identical episodes and batch_sizes
```

The multi-worker equality test is the point of this stage. It proves task
isolation, request routing, and batching neutrality in one assertion.

## Stage 3: Threaded Driver (Lanes + Batcher)

`lanes.rs`:

```rust
pub struct ThreadedOrchestratorConfig {
    pub workers_per_lane: NonZeroUsize,
    pub max_batch: NonZeroUsize,
    pub flush_after: Duration,
}

pub struct ThreadedGumbelOrchestrator<E, V> {
    engines: Vec<E>,
    evaluator: V,
    search: GumbelMcts,
    config: ThreadedOrchestratorConfig,
}

pub struct LaneEpisodes<G, C> {
    pub lane: usize,
    pub episodes: Vec<OrchestratedEpisode<G, C>>,
}

pub struct ThreadedRun<G, C> {
    pub lanes: Vec<LaneEpisodes<G, C>>,
    pub batch_sizes: Vec<usize>,
}

impl<E, V> ThreadedGumbelOrchestrator<E, V>
where
    E: GraphEngine + Send,
    V: Evaluator + Send,
{
    pub fn new(
        engines: Vec<E>,
        evaluator: V,
        search: GumbelMcts,
        config: ThreadedOrchestratorConfig,
    ) -> Self;

    pub fn run<R: RootSource<E> + Send>(
        self,
        root_sources: Vec<R>,
        context: GumbelEpisodeContext,
    ) -> EngineResult<ThreadedRun<E::Graph, E::Candidate>>;
}
```

Rules:

```text
lanes = engines.len(); root_sources.len() must equal lanes (validated,
internal("lane count mismatch") otherwise); each lane owns one engine and
one root source because graph handles are instance-local.
WorkerId = lane * workers_per_lane + slot.
EpisodeId = (lane as u64) << 32 | per-lane admission counter. Per-lane ids
are deterministic; no cross-lane coordination exists.
run consumes self and uses std::thread::scope: lane threads and one batcher
thread; scope guarantees all threads join before run returns.
```

### Message types and channels

```rust
struct EvalJob {
    lane: usize,
    slot: usize,
    token: WorkToken,
    request: EvalRequest,
}

struct EvalReply {
    slot: usize,
    token: WorkToken,
    output: EvalOutput,
}
```

`EvalJob` has no generic engine parameters. That is the structural
enforcement of the portable-only rule.

Channels, all `std::sync::mpsc::sync_channel` (bounded):

```text
intake: lanes -> batcher, capacity = lanes * workers_per_lane
reply[lane]: batcher -> lane, capacity = workers_per_lane
```

Capacity proofs (state these as comments in the code):

```text
intake can hold every possible outstanding eval at once (one per worker),
so a lane's send can only block transiently while the batcher is copying
out of the queue, never on a full steady state; the batcher never blocks
on anything a lane holds, so there is no cycle.
reply[lane] can hold one reply per worker in that lane, which is the
maximum outstanding; the batcher's reply sends therefore never block, so
the batcher -> lane edge cannot participate in a deadlock cycle.
```

### Lane thread loop

```text
loop:
  admit from this lane's root source into idle slots
  drive running slots inline against this lane's engine
  if nothing Running and nothing Parked: break     (lane done)
  send an EvalJob for each newly parked slot, slot order
  recv at least one EvalReply (blocking), then drain all immediately
    available replies without blocking
  resume the matching slots, then loop
on exit: drop this lane's intake sender clone
recv error (batcher gone) while evals are outstanding:
  internal("eval backend unavailable")
```

### Batcher thread loop

```text
loop:
  first = intake.recv(); on Disconnected: return Ok(batch_sizes)
  deadline = Instant::now() + flush_after
  collect into batch until batch.len() == max_batch, using
    recv_timeout(remaining); Timeout or Disconnected ends collection
  evaluate_batch over the batch's requests (request order = arrival order);
    record batch size
  on eval error: return it (dropping reply senders unblocks the lanes)
  route each output to reply[job.lane] as an EvalReply
```

### Error policy

```text
Fail-fast. Each thread returns EngineResult; run joins everything via the
scope, then returns the first error by this precedence: batcher error,
then lanes in lane order. A lane that errors drops its channels; other
lanes and the batcher finish or unwind through disconnects. No thread
outlives run.
```

### Determinism

```text
Per-episode: every episode still equals the serial episode for the same
root, config, and order-independent evaluator, regardless of batch
composition or timing. This is the invariant that matters.
Per-run: batch_sizes and cross-lane interleaving are timing-dependent and
carry no correctness weight. Episode order within a lane is deterministic.
```

### Stage 3 tests (`tests/threaded.rs`)

Shared test helper in `tests/common/mod.rs`: `SlowEvaluator<V>` wrapping an
inner evaluator, sleeping a fixed `Duration` per `evaluate_batch` call.
Test-only; sleeping in production drivers stays forbidden.

```text
one lane, one worker: equals SerialGumbelOrchestrator per root
two lanes, W workers each, fixed generator seeds per lane: every episode
  equals the serial episode for the same root (run each lane's root
  sequence serially on a fresh engine); per-lane episode order matches
episode conservation: total episodes == total roots across lanes
occupancy: 1 lane, 8 workers, SlowEvaluator(inner=RandomValueEvaluator,
  ~20ms), flush_after generous (>= 250ms), max_batch = 8: after the first
  batch, every batch while 8 workers remain active has size 8; assert
  batch_sizes sum equals total eval count and average size >= workers/2
  (conservative bound so CI timing noise cannot flake it)
eval failure: failing evaluator aborts the run with an error and all
  threads join (run returns; no hang)
drain: K roots per lane with K not a multiple of anything in particular;
  run returns with all episodes and no residual threads
```

## Stage 4: Docs And Bookkeeping

```text
Update specs/GZ_ORCHESTRATOR.md Deferred list: bounded eval batcher and
worker pool are now implemented (single-process); move them out of
Deferred, leaving queued/process engine lanes, replay sink, ratio
controller, cancellation, metrics deferred.
Add GZ_ORCHESTRATOR.md, GZ_ORCHESTRATOR_SERIAL_IMPL.md, and this file to
the specs list in AGENTS.md (known drift from the serial work order).
Commit stage by stage; do not squash the goldens commit from stage 0.
```

## Final Verification

```bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
```

Acceptance checklist:

```text
gz-search untouched except the two stage-0 test additions; goldens intact
serial orchestrator behavior unchanged (its tests pass unmodified, modulo
  the OrchestratedEpisode rename alias)
batched driver: multi-worker equality test against serial passes
threaded driver: per-lane equality test against serial passes
EvalJob carries no engine-generic data
all channels bounded, with capacity justifications in comments
no thread outlives run (std::thread::scope everywhere)
stable error messages: "worker blocked", "batched driver blocked",
  "unsupported search work", "lane count mismatch",
  "eval backend unavailable"
no new dependencies; std-only threading; Instant only in the batcher
```

## Out Of Scope

```text
wave search and multiple outstanding tokens per task
per-episode GumbelEpisodeContext and opponent trajectory leases
queued or process engine lanes (engine work stays inline on its lane)
measure concurrency limits
replay integration, ratio control, metrics, shutdown signals beyond
  natural drain
Python or process-backed evaluators
retry or per-worker fault isolation
```
