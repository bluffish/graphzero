# gz-orchestrator Spec

Status: draft

Purpose: define the execution crate that drives GraphZero search workers,
routes their engine and eval work, and batches that work across many
concurrent selfplay workers. The first implementation is serial and boring,
but the worker protocol is designed for the two regimes that dominate later:

```text
wave Gumbel-MCTS workers with many in-flight simulations per tree
compiler engines where apply and measure are expensive, batchable, and
hardware-limited
```

## Core Decision

A search task is a pure state machine. It never calls an engine or an
evaluator directly. Every engine and eval interaction crosses one poll/resume
work protocol, and the orchestrator decides where that work executes.

```text
gz-search:
  I need this graph expanded / this candidate applied / this graph measured /
  this eval request answered.
  Here is a token.
  Resume me with the result for that token.

gz-orchestrator:
  collect work requests from many tasks
  route each request to the right engine lane or eval backend
  batch compatible work
  resume the matching task/token with the result
```

The protocol covers engine work, not just eval:

```text
Whittle apply/measure are microsecond arena operations. A compiler engine's
apply is a real rewrite on a large graph, and its measure is lowering plus
hardware timing.
If engine calls are hardwired inside search, an expensive engine blocks its
worker and can never be queued, limited, or batched.
BatchGraphEngine already defines batch semantics; only a request boundary can
exploit it across workers.
Wave search has many in-flight simulations per tree, each suspended on apply,
expansion, or eval at different times. One suspension mechanism must serve
all of them.
```

The current direct `GumbelMcts::run(... evaluator ...)` shape stays as a
compatibility wrapper and as the equivalence oracle for the task refactor. It
is not the long-term execution path.

## Role

`gz-orchestrator` answers:

```text
Which search workers run, where does their engine and eval work execute, when
are they resumed, and when are measured episodes handed to replay?
```

It owns:

```text
worker ids
episode ids
search task driving
work routing for engine and eval requests
engine lane ownership and task-to-lane assignment
single-process eval batching
measure concurrency limits later
replay sink driving later
ratio/backpressure gating later
bounded eval queues for the threaded single-process driver
shutdown and cancellation later
```

It does not own:

```text
search algorithms
tree node layout
wave tree math
candidate enumeration semantics
STOP insertion
eval model implementation
Whittle-specific evaluator logic
replay schema
training
Python or torch in the default crate
```

## Dependency Contract

Default allowed:

```text
std
gz-engine
gz-eval
gz-search
```

Allowed as dev-dependencies for integration tests:

```text
gz-engine-whittle
gz-eval-whittle
```

Allowed later behind explicit features:

```text
tokio or another async runtime
gz-replay
gz-features
process-backed eval clients
metrics backend
```

Forbidden in the default crate:

```text
torch/Python bindings
rocksdb
gz-engine-whittle
future concrete compiler adapters
trainer code
```

Domain-specific orchestration binaries may compose concrete engine/eval crates.
The default `gz-orchestrator` library should stay engine-neutral.

## Work Protocol

The protocol types belong in `gz-search`, because the task owns search state.
`gz-search` gains no new dependencies from them: options, results, and hashes
come from `gz-engine`; `EvalRequest`/`EvalOutput` come from `gz-eval`.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WorkToken(u64);

pub enum SearchPoll<G, C, R> {
    Work(SearchWork<G, C>),
    Blocked,
    Done(R),
}

#[non_exhaustive]
pub enum SearchWork<G, C> {
    Expand(ExpandWork<G>),
    Apply(ApplyWork<G, C>),
    Measure(MeasureWork<G>),
    Eval(EvalWork<G, C>),
}

pub struct ExpandWork<G> {
    pub token: WorkToken,
    pub graph: G,
    pub options: CandidateOptions,
}

pub struct ApplyWork<G, C> {
    pub token: WorkToken,
    pub graph: G,
    pub candidate: C,
}

pub struct MeasureWork<G> {
    pub token: WorkToken,
    pub graph: G,
    pub options: MeasureOptions,
}

pub struct EvalWork<G, C> {
    pub token: WorkToken,
    pub graph: G,
    pub candidates: Vec<C>,
    pub request: EvalRequest,
    pub measure_options: MeasureOptions,
}
```

Results are delivered through one resume entry point:

```rust
#[non_exhaustive]
pub enum SearchWorkResult<G, C> {
    Expand(ExpandResult<C>),
    Apply(ApplyResult<G, C>),
    Measure(MeasureResult<G>),
    Eval(EvalOutput),
}

pub struct ExpandResult<C> {
    pub graph_hash: GraphHash,
    pub candidates: Vec<ExpandedCandidate<C>>,
}

pub struct ExpandedCandidate<C> {
    pub candidate: C,
    pub candidate_hash: CandidateHash,
    pub kind: CandidateKindId,
    pub tags: CandidateTags,
    pub static_prior: f32,
}
```

`ExpandedCandidate` is a compact projection of `CandidateInfo`. Work results
must not carry `display_name`, `subjects`, or adapter metadata; those are
diagnostics, not hot-path search inputs.

Protocol rules:

```text
Tokens are allocated by the task and unique for the task's lifetime.
The orchestrator keys in-flight work by (WorkerId, WorkToken).
poll may return Work any number of times before any resume. Multiple
outstanding tokens are the normal wave-search state, not an error.
Each issued token must be resumed exactly once.
Resume order across tokens is unspecified; tasks must accept any order.
resume with an unknown or already-resumed token is an error.
resume with a result variant that does not match the requested work kind is
an error.
Blocked means the task cannot progress until at least one resume. Polling
again without an intervening resume returns Blocked again.
poll after Done is an error.
Dropping a task with outstanding tokens is legal. The orchestrator discards
late results for dropped tasks; nothing is poisoned.
resume validates results before search uses them. EvalOutput is validated
against the stored EvalRequest using gz-eval validation helpers.
SearchWork and SearchWorkResult are non-exhaustive; drivers must fail loudly
on unknown work kinds, never skip them.
```

Failure rules:

```text
Tasks never observe backend or transport failures.
Expected domain failures travel inside normal results: ApplyResult.rejected,
MeasureResult.failure.
Retry, timeout, rerouting, and backend-crash policy belong to the
orchestrator. If the orchestrator cannot produce a result for a token, it
aborts and drops the task.
```

Eval alignment rules:

```text
EvalWork.candidates[i] corresponds to request.actions[i].
request.actions[candidates.len()] is STOP.
Search builds the ordered action list. Drivers, batchers, and evaluators must
not insert, remove, or reorder actions.
```

Portability rule:

```text
EvalWork.graph, EvalWork.candidates, and EvalWork.measure_options exist only
to drive lane-local EngineEvaluator backends. They are engine-instance-local
values and must never cross a lane or process boundary. Batched and remote
backends consume EvalWork.request alone.
```

## Engine Identity

Tasks build portable contexts without touching the engine:

```rust
pub struct EngineIdentity {
    pub engine_id: EngineId,
    pub engine_version: EngineVersion,
    pub action_set_hash: ActionSetHash,
}
```

Rules:

```text
A task captures EngineIdentity from its lane's engine at construction.
Result hashes plus captured identity are sufficient to build
ReplayGraphContext and PortableSearchActionRef values inside the task.
The lane engine's identity must not change during a task's lifetime.
Servicing a task's work against a different engine instance than its lane
engine is a driver bug; graph and candidate handles are instance-local.
```

## Gumbel Task Boundary

Serial Gumbel-MCTS becomes a task before the orchestrator is implemented.

```rust
pub struct GumbelRootTask<G, C> {
    // search-owned tree state, RNG state, in-flight simulation state
}

impl<G, C> GumbelRootTask<G, C> {
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: GumbelSearchContext,
    ) -> EngineResult<Self>;

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, GumbelRootResult<G, C>>>;

    pub fn resume(
        &mut self,
        token: WorkToken,
        result: SearchWorkResult<G, C>,
    ) -> EngineResult<()>;
}

pub struct GumbelEpisodeTask<G, C> {
    // current graph, recorded steps, active root task
}

impl<G, C> GumbelEpisodeTask<G, C> {
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: GumbelEpisodeContext,
    ) -> EngineResult<Self>;

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, GumbelEpisode<G, C>>>;

    pub fn resume(
        &mut self,
        token: WorkToken,
        result: SearchWorkResult<G, C>,
    ) -> EngineResult<()>;
}
```

`poll` takes no engine. That is the point of the boundary.

The Gumbel task suspends at these points:

```text
root expansion: Expand, then Eval for the root node
simulation frontier: Apply for the selected edge, then Expand and Eval for
the new leaf when it is not already in the tree
terminal STOP re-eval: Eval with an adjusted EvalPositionContext when
opponent-trajectory alignment moves the effective leaf depth
final episode measurement: Measure, emitted once by the episode task after
the last root search
```

The terminal STOP re-eval is a real suspension point in the existing serial
implementation (`stop_value`), not an expansion. The state machine must model
it explicitly.

The root task is the first refactor target. The episode task is built on top
of the root task.

## Wave Readiness

The serial v1 task keeps at most one token outstanding. The protocol already
permits more, which is exactly what wave Gumbel-MCTS needs:

```text
at each sequential-halving target, every eligible considered root action can
launch a simulation concurrently
each in-flight simulation suspends independently on apply, expansion, or eval
virtual visits mark in-flight descent paths so later launches diversify
backup on resume clears virtual marks and applies real visits
```

Wave tree math (virtual visits, in-flight bookkeeping, halving-level
barriers) is a future `gz-search` spec. This spec fixes what that math can
rely on:

```text
poll may emit many Work items before any resume
resumes arrive in any order
wave width is search config: it changes exploration, changes results, and
must enter SearchConfigHash
wave width 1 must reproduce serial run-to-completion results exactly
```

Do not implement wave math in the serial slice. Do not design any driver,
queue, or batcher that assumes one outstanding token per task.

## Engine Lanes

An engine lane is one engine instance plus the driver that services
`Expand`/`Apply`/`Measure` work against it.

Rules:

```text
Every task is bound to exactly one lane for its lifetime.
Engine work never crosses lanes. Only portable refs and portable EvalRequest
data cross lanes or processes.
Many tasks may share one lane to share dedup, candidate, and transition
caches. Lane assignment is orchestrator policy, not search state.
```

Lane implementations:

```text
inline lane:
  service work immediately on the calling thread
  correct for cheap engines like Whittle
  the serial orchestrator is an inline lane plus an inline eval backend

queued lane:
  bounded queue in front of one or more engine threads
  groups compatible work into BatchGraphEngine batch calls
  separate concurrency limits per operation kind
  measure gets the tightest limit: compiler measure is lowering plus hardware
  timing, and hardware slots are the scarcest resource in the pipeline
  apply/expand may run wider than measure

process lane later:
  the same work protocol over a process boundary for crash isolation or
  remote engines
```

This replaces the earlier EngineServer sketch. EngineServer is the queued
lane, and it is optional per engine, not a mandatory hop. Whittle never pays
for a queue; a compiler engine never blocks a worker thread.

Backpressure:

```text
every queue is bounded
a task with outstanding tokens parks at zero cost; Blocked is the natural
backpressure state
episode admission is gated by the ratio controller at episode start, not
inside the hot loop
```

## Eval Routing

Serial v1 routes immediately:

```text
Work(Eval w)
  -> evaluator.evaluate(engine, EngineEvalRequest { w.graph, w.candidates,
     w.request, w.measure_options })
  -> task.resume(w.token, SearchWorkResult::Eval(output))
```

Async v2 routes through a bounded batcher:

```text
many tasks poll
many Eval work items collected
batcher groups compatible requests
configured eval backend produces EvalOutput rows
orchestrator resumes matching (WorkerId, WorkToken) pairs
```

Compatibility key for batching must include at least:

```text
engine id
engine version
action set hash
measure config hash when the backend measures
feature schema hash when the backend uses features
model version or evaluator config identity
```

Do not batch requests together merely because their action counts match.
Engine-aware `EngineEvaluator` backends are lane-local by definition and are
never batched across lanes.

Opponent trajectory tables:

```text
trajectory tables are eval-backend state, not request payload
the orchestrator registers a table with the backend at episode start and
receives the trajectory_id used in EvalPositionContext
EvalRequest rows carry trajectory_id plus position indexes only
the batcher resolves trajectory_id + opponent_row while collating batches
the lease is dropped when the episode ends; the backend must not be left
holding tables for dead episodes
```

## Serial Orchestrator

The first orchestrator is synchronous and exists to prove the ownership
boundary and the protocol. It is not the final performance path.

Draft layout:

```text
crates/gz-orchestrator/
  Cargo.toml
  src/
    lib.rs
    ids.rs
    serial.rs
```

Draft ids:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WorkerId(u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EpisodeId(u64);
```

Draft serial type:

```rust
pub struct SerialGumbelOrchestrator<E, V> {
    worker_id: WorkerId,
    next_episode_id: u64,
    engine: E,
    evaluator: V,
    search: GumbelMcts,
}

pub struct OrchestratedEpisode<G, C> {
    pub worker_id: WorkerId,
    pub episode_id: EpisodeId,
    pub episode: GumbelEpisode<G, C>,
}

pub type SerialEpisode<G, C> = OrchestratedEpisode<G, C>;

impl<E, V> SerialGumbelOrchestrator<E, V>
where
    E: GraphEngine,
    V: EngineEvaluator<E>,
{
    pub fn new(worker_id: WorkerId, engine: E, evaluator: V, search: GumbelMcts) -> Self;

    pub fn run_from_root(
        &mut self,
        context: GumbelEpisodeContext,
    ) -> EngineResult<SerialEpisode<E::Graph, E::Candidate>>;

    pub fn run(
        &mut self,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<SerialEpisode<E::Graph, E::Candidate>>;
}
```

`run_from_root` uses `engine.root()`; `run` starts from the supplied graph,
matching the `gz-search` naming convention.

Serial run loop:

```text
create GumbelEpisodeTask with the engine's EngineIdentity
loop:
  match task.poll():
    Work(Expand w)  -> engine.candidates + candidate_info rows; resume
    Work(Apply w)   -> engine.apply; resume
    Work(Measure w) -> engine.measure; resume
    Work(Eval w)    -> evaluator.evaluate; resume
    Blocked         -> internal error: the serial driver resolves every token
                       before polling again, so Blocked is unreachable
    Done(episode)   -> assign EpisodeId; return SerialEpisode
```

## Future Async Orchestrator

The async orchestrator runs many search tasks concurrently:

```text
worker tasks own search state
orchestrator owns bounded queues
eval batcher owns compatible batch formation
queued engine lanes own expensive engine execution
replay sink owns measured-row persistence
ratio controller gates episode admission
```

The async path uses the same `SearchWork`/`resume` protocol. Search must not
know whether eval is random, measured, neural, local, remote, sync, or
batched, nor whether its engine lane is inline or queued.

Later async pieces:

```text
queued engine lane driver
measure concurrency limiter
ReplaySink
ratio controller
CancellationToken
Metrics
```

The single-process worker pool and bounded eval batcher now exist. Queued
engine lanes, replay, ratio control, cancellation, and metrics remain later
work.

## Measurement And Replay Boundary

Rules:

```text
EvalOutput.value is search value, not replay reward.
Replay rows require a GraphEngine::measure result.
Rows enter replay only after final measurement exists.
The episode task emits Measure work for the final graph; the orchestrator
routes it like any other engine work, subject to measure concurrency limits.
The serial orchestrator may return measured episodes but does not define a
replay row schema.
```

`gz-replay` will own durable row types and storage. `gz-orchestrator` will
eventually feed a replay sink; it must not duplicate replay schemas.

## Determinism

```text
A task's behavior is a pure function of its config, root, captured
EngineIdentity, and the observed sequence of protocol events. Tasks must not
read wall-clock time, OS RNG, or thread identity.
For fixed engine config/version, root, search config, and deterministic
evaluator, the serial driver must reproduce the current run-to-completion
results exactly: same selected action refs, same policy targets, same visit
statistics, same final graph hash, same stop reason, same SearchConfigHash.
Any driver that delivers the same protocol event sequence must produce the
same result.
Async drivers may reorder events across tokens. They must never reorder,
duplicate, or drop events within a token.
Wave width changes exploration and must change SearchConfigHash.
```

## Error Handling

Initial serial errors reuse `EngineResult`.

Search task protocol errors:

```text
unknown work token
double resume of a token
result variant does not match the requested work kind
invalid eval output for the stored request
poll after Done
```

Backend failures never reach tasks; the orchestrator retries, reroutes, or
drops the task. Do not create a large orchestrator error hierarchy until
async queues and replay introduce real non-engine failure modes.

## Test Strategy

Equivalence comes first. It is the regression harness for the entire task
refactor:

```text
before the refactor, capture golden episodes from the current
run-to-completion implementation per seed/config/root: selected action refs,
policy targets, visit statistics, final hashes, stop reasons
the task-driven path must match those goldens exactly, both through the
serial driver and through the compatibility wrappers
```

Search task tests belong in `gz-search`:

```text
root task emits Expand for the root, then Eval for the root
resume with an unknown token is rejected
resume with a mismatched result variant is rejected
resume with an invalid eval output length is rejected
double resume of a token is rejected
poll after Done is rejected
dropping a task with an outstanding token is safe
terminal STOP re-eval emits Eval with the adjusted position context when
opponent alignment requires it
episode task emits Measure exactly once, after the final root search
episode task completes through STOP and through max steps
```

Orchestrator tests belong in `gz-orchestrator`:

```text
serial orchestrator drives a Gumbel episode to completion
serial orchestrator matches the run-to-completion goldens
serial orchestrator increments episode ids
serial orchestrator routes eval work through the configured evaluator
Blocked in the serial driver is reported as an internal error
Whittle integration works with WhittleEngine + WhittleMeasureEvaluator
```

Use Whittle only as a dev-dependency integration test. The crate itself
remains engine-neutral.

## Implementation Plan

1. Capture run-to-completion golden fixtures in `gz-search` before touching
   the kernel. They are the equivalence oracle for every later step.
2. Add `WorkToken`, `SearchPoll`, `SearchWork`, `SearchWorkResult`,
   `ExpandResult`, `ExpandedCandidate`, and `EngineIdentity` to `gz-search`.
3. Refactor the serial Gumbel simulation loop into an explicit state machine
   with suspension points at apply, expansion, eval, and terminal STOP
   re-eval.
4. Implement `GumbelRootTask::poll`/`resume` with at most one outstanding
   token.
5. Reimplement `GumbelMcts::search_root` as a compatibility wrapper that
   drives `GumbelRootTask` inline. All existing tests and the goldens must
   pass unchanged.
6. Implement `GumbelEpisodeTask` on top of the root task, including the final
   Measure work item. Reimplement `GumbelMcts::run` as a wrapper.
7. Add the `gz-orchestrator` crate skeleton with ids and the serial driver.
8. Add Whittle integration tests with `WhittleMeasureEvaluator`.
9. Update `GZ_SEARCH.md` and `GZ_SEARCH_GUMBEL_MCTS.md` so their deferred
   lists point at this spec for the work protocol.
10. Decide whether the direct evaluator APIs in `gz-search` stay public,
    become test helpers, or are deleted once the task path is canonical.

The poll/resume refactor is the load-bearing step. Do not build async
batching, lanes, or wave math until the serial state machine matches the
goldens.

## Deferred

```text
wave tree math spec (virtual visits, in-flight bookkeeping, halving barriers)
async runtime choice
queued engine lanes
process lanes
measure concurrency limiter
replay sink
ratio/backpressure controller
worker supervision
shutdown/cancellation
metrics
root generators
multi-process orchestration
```
