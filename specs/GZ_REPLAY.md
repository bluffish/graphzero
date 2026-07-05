# gz-replay Spec

Status: draft

Purpose: define the durable replay store for GraphZero selfplay. Replay
persists per-step training rows and episode traces produced by orchestrator
workers, admits data only from measured episodes, and exposes a sampling
boundary so the Python trainer never reads storage directly.

The row/label model is decided, not open. The primary regime is the compiler
engine, where measurement is expensive: the schema must never require
per-step measurement.

## Decided Model

One episode produces one expensive measurement and many training rows:

```text
episode: s0 -> s1 -> ... -> sT          (no measurement during search)
measure(sT) once                         (the one expensive measurement)
outcome = sign(learner_reward - reference_reward)   in {-1.0, 0.0, +1.0}
rows: one per step, for s0 .. s(T-1)
every row carries the same episode outcome as its value target
```

Rules this fixes:

```text
Rows are per-step. Each row is the pre-action state s_t with the Gumbel
completed-Q policy target produced by the root search at s_t.
Labels are episode-level. value_target is the episode outcome, identical on
every row of the episode (AlphaZero z).
Admission is episode-level. An episode is replay-eligible iff its final
measurement is measured && valid with a finite scalar reward. Rows inherit
admission from their episode. Row graphs are never individually measured.
The outcome is a comparison. The raw scalar (runtime, cost) is not the
label; the sign of learner-vs-reference is. The raw scalar is stored as
reward_target so graded targets can be derived later without regeneration.
```

This amends the older CODEBASE_OUTLINE wording "rows enter replay only
after the row graph has a MeasureResult" to: rows enter replay only after
their episode's final graph has a valid MeasureResult.

Consciously closed door: no per-step cost data exists in replay. Same-index
graded/pairwise targets would require data regeneration. Accepted, because
per-step measurement is prohibitive in the compiler regime.

## Role

`gz-replay` answers:

```text
What selfplay data is durable, what exactly is a training row, and how do
downstream consumers sample it?
```

It owns:

```text
replay schema, invariants, and binary encoding
RocksDB storage, column families, and key layout
durable ReplayEpisodeId assignment
episode-level admission enforcement at the write boundary
sampling API, window semantics, and sampling determinism
produced/consumed row counters for ratio control
schema versioning
```

It does not own:

```text
episode generation
episode -> row projection (gz-orchestrator, rules defined here)
outcome comparison execution (gz-orchestrator, rules defined here)
opponent/reference trajectory generation or measurement scheduling
feature extraction
training
network or Python protocol serving (a later service layer wraps sampling)
```

## Dependency Contract

Allowed:

```text
std
gz-engine with the serde feature
gz-features (FeatureSchemaConfig plus GZFR header validation only)
rocksdb
serde
postcard
```

Forbidden:

```text
tokio or any async runtime
torch/Python bindings
gz-search
gz-eval
gz-orchestrator
concrete engine adapters
rand crates; sampling uses a small internal deterministic RNG
```

`gz-replay` depends only on `gz-engine` plus the light `gz-features` schema
and row-header API. It does not depend on concrete extractors or feature
extraction. `GumbelEpisode` lives in `gz-search`, so projection into replay
records belongs to
`gz-orchestrator`; `gz-replay` accepts already-portable records and never
sees an engine handle.

## Records

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ReplayEpisodeId(u64);   // store-assigned, monotonic, durable

pub struct ReplayEpisodeRecord {
    pub root: ReplayGraphContext,
    pub final_graph: ReplayGraphContext,
    pub steps: Vec<SearchStepRef>,
    pub final_measure: MeasureSummary,
    pub outcome: ReplayOutcome,
    pub search_config_hash: SearchConfigHash,
    pub row_count: u32,
}

pub struct ReplayOutcome {
    pub value_target: Option<f32>,       // -1.0 | 0.0 | +1.0 when Some
    pub learner_reward: f32,
    pub reference: Option<ReplayReference>,
}

pub struct ReplayReference {
    pub kind: ReplayReferenceKind,
    pub reward: f32,
    pub final_graph: Option<ReplayGraphContext>,
    pub trajectory_id: Option<u64>,
    pub search_config_hash: Option<SearchConfigHash>,
    pub model_version: Option<ModelVersion>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayReferenceKind {
    RootBaseline,
    Greedy,
    Beam,
    Random,
    Gumbel,
}

pub struct ReplayRow {
    pub step_index: u32,
    pub root: ReplayGraphContext,
    pub state: ReplayGraphContext,
    pub action_history: Vec<PortableSearchActionRef>,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub policy_target: Vec<f32>,
    pub selected_action: PortableSearchActionRef,
    pub value_target: Option<f32>,
    pub reward_target: Option<f32>,
    pub final_measure: MeasureSummary,
    pub model_version: Option<ModelVersion>,
    pub search_config_hash: SearchConfigHash,
    pub feature_row: Option<Vec<u8>>,     // GZFR bytes
}
```

Record rules:

```text
ReplayRow does not store ReplayEpisodeId; storage keys carry it, and
sampling returns (ReplayEpisodeId, ReplayRow) pairs.
state is the pre-action graph context s_t.
legal_actions is the exact ordered action list the root search evaluated at
s_t: engine candidates in enumeration order, then STOP last.
policy_target[i] scores legal_actions[i]; lengths must match.
action_history is the selected action refs from the root to s_t;
action_history.len() == step_index.
value_target and reward_target duplicate episode-level values onto every
row so sampling never needs a join.
model_version is the step's root eval model version; it may differ across
rows if a checkpoint swap lands mid-episode.
No engine-local handles, graph bodies, artifacts, display strings, or
adapter metadata anywhere in stored records. `feature_row` is the sole
exception to the old "no graph bodies" wording: it stores a portable encoded
FeatureRow (GZFR), not an engine body or artifact. Replay data with feature
rows is bound to one FeatureSchemaHash; changing the feature schema means
regenerating replay data.
```

## Outcome Rules

Defined here, computed by the orchestrator during projection:

```text
learner_reward = episode final_measure.scalar_reward (admission guarantees
it exists and is finite)
reference kinds:
  RootBaseline: the root graph's measured reward; one measurement per root,
  amortized across every episode from that root
  Greedy/Beam/Random: algorithmic reference trajectories for cheap-measure
  engines; their search_config_hash records the reference kernel config
  Gumbel: the policy opponent -- the terminal reward of a greedy
  (one-simulation, no-noise, temperature-0) rollout of the current
  published checkpoint from the fixed root, replayed once per observed
  model version; model_version records which checkpoint played it and
  search_config_hash the rollout kernel; rollout episodes never enter
  the store; episodes admitted before the first rollout completes are
  unlabeled (bounded burst runs that admit everything at once therefore
  stay unlabeled)
  SelfAverage: a reward EMA of the learner's own recent episode rewards
  on that lane; adaptive, so labels do not saturate on repeated or single
  roots; unlabeled until the EMA seeds (the first in-flight admissions per
  lane); carries no reference graph or trajectory
value_target = sign(learner_reward - reference.reward): +1.0 win, -1.0
loss, 0.0 exact tie
no reference configured, or reference measurement missing/invalid:
  value_target = None; rows are stored and remain policy-target training
  data
value orientation matches the eval scale: higher is better, targets live in
[-1, 1]
```

Graded targets (for example tanh of the reward delta) are deferred; they
must be derivable later from stored learner_reward and reference.reward
without regenerating data.

## Admission Rules

Enforced by `gz-replay` at the write boundary:

```text
append_episode rejects the episode unless final_measure.measured,
final_measure.valid, and scalar_reward is Some(finite) -> NotMeasured.
rows.len() must equal record.row_count and steps.len() -> InvalidRecord.
row step_index values must be contiguous from 0 -> InvalidRecord.
per-row invariants (lengths, finite non-negative policy mass, STOP last,
value_target in {-1, 0, +1} when Some, action_history length) are
validated -> InvalidRecord.
an episode and all its rows are written in one RocksDB WriteBatch with the
counter updates: replay never contains a partial episode.
```

Measurement failures are data at the orchestrator level (the episode is
simply not replay-eligible and is dropped or logged); they are never stored
rows.

## Storage Layout

Schema version: 4. Version 4 records the expander fields in the persisted
FeatureSchemaConfig metadata. Version 3 added optional per-row GZFR feature
payloads and the persisted FeatureSchemaConfig metadata. Version 1, 2, and 3
stores fail to open with SchemaMismatch.

```text
RocksDB, one database directory, column families:
  meta      schema version, next episode seq, produced/consumed counters,
            optional feature schema config
  episodes  key: episode_seq u64 BE            value: ReplayEpisodeRecord
  rows      key: episode_seq u64 BE || step_index u32 BE   value: ReplayRow
  row_index key: row_seq u64 BE                value: rows key
```

Rules:

```text
Keys are episode-major (decided; closes CODEBASE_OUTLINE design question 1).
Ordered keys give episode iteration and windowed scans for free.
row_index assigns every row a dense global sequence number so uniform
window sampling is O(1) lookups, not scans.
Values are postcard-encoded via serde. Hashes and ids serialize as raw
bytes through gz-engine's binary serde representation.
meta stores a schema version written at creation; opening a store with a
mismatched version fails with SchemaMismatch, never migrates silently.
next episode seq is recovered from the last episodes key on open; no
counter can drift from the data.
feature schema config is persisted once. `ensure_feature_schema(config)` is
idempotent for the same config and rejects mismatches with InvalidRecord.
Rows with `feature_row = Some(bytes)` require a configured schema and the
GZFR header hash must match it. Featureless rows remain legal. One episode
must be all featured or all featureless.
```

## Write And Sample API

```rust
pub struct ReplayStore { /* Arc<DB> internally; Send + Sync */ }

impl ReplayStore {
    pub fn open(path: &Path) -> ReplayResult<Self>;

    pub fn append_episode(
        &self,
        record: &ReplayEpisodeRecord,
        rows: &[ReplayRow],
    ) -> ReplayResult<ReplayEpisodeId>;

    pub fn episode(&self, id: ReplayEpisodeId) -> ReplayResult<Option<ReplayEpisodeRecord>>;

    pub fn ensure_feature_schema(&self, config: &FeatureSchemaConfig) -> ReplayResult<()>;
    pub fn feature_schema(&self) -> ReplayResult<Option<FeatureSchemaConfig>>;

    pub fn sample_rows(
        &self,
        config: SampleConfig,
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>>;

    pub fn counters(&self) -> ReplayCounters;
}

#[derive(Clone, Copy, Debug)]
pub struct SampleConfig {
    pub batch: NonZeroUsize,
    pub window_rows: NonZeroU64,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayCounters {
    pub produced_rows: u64,
    pub consumed_rows: u64,
}
```

Sampling rules:

```text
sample_rows draws batch row sequence numbers uniformly, with replacement,
from the last window_rows rows (clamped to what exists), using an internal
deterministic RNG seeded from SampleConfig.seed.
identical (store contents, config) -> identical sample; the trainer varies
the seed per batch.
sampling an empty store returns Empty.
sample_rows adds batch to consumed_rows; append_episode adds row_count to
produced_rows. sample_ratio = consumed / produced is computed by the
orchestrator's ratio controller, not here.
Python never reads RocksDB directly; a later service layer wraps this API.
```

Concurrency:

```text
ReplayStore is Send + Sync; append and sample take &self and may run from
different threads (orchestrator sink thread, trainer service thread).
Single-writer discipline is recommended but not required for correctness.
```

## Errors

```rust
pub type ReplayResult<T> = Result<T, ReplayError>;

pub enum ReplayError {
    NotMeasured,
    InvalidRecord,
    SchemaMismatch,
    Empty,
    Storage, // wraps the rocksdb error message, bounded
}
```

Keep it small; do not mirror rocksdb's error surface.

## Projection Contract (gz-orchestrator side)

Implemented in `gz-orchestrator`, not in this crate. The contract lives here
so the schema and producer cannot drift:

```text
projection input: a completed GumbelEpisode plus an optional Reference
one ReplayRow per GumbelStep; the final graph produces no row of its own
row.state = step_ref.before context; selected_action = step.selected_action
legal_actions/policy_target come from the step's root search, in the exact
eval order (candidates then STOP)
episodes whose final measure is unmeasured or invalid are not projected
reference = None stores policy-only rows with value_target = None
reference = Some(r) stores ReplayReference metadata and value_target =
sign(learner_reward - r.final_reward) on every row
reward_target = Some(learner_reward) on every row
action_history contains prior selected actions from the same episode
```

`GumbelRootResult` and `GumbelStep` expose
`legal_actions: Vec<PortableSearchActionRef>`, the root node's ordered action
refs. Projection copies those refs into rows so the policy target remains
action-aligned after engine-local handles are gone.

## Crate Shape

```text
crates/gz-replay/
  Cargo.toml
  src/
    lib.rs
    error.rs
    keys.rs
    records.rs
    sample.rs
    store.rs
  tests/
    store.rs
    sample.rs
```

Keep modules flat. No service, server, or trainer modules until a real
consumer exists.

## Test Strategy

Temp-dir stores; no engine adapters; records built by hand from portable
types.

```text
append then read back: episode and rows roundtrip byte-identically
admission rejects unmeasured, invalid, and non-finite-reward episodes
admission rejects row_count/steps/step_index/length mismatches
partial-episode atomicity: a rejected append leaves no episode, rows, or
counter changes behind
episode ids are monotonic and survive reopen; next id recovers from data
sampling is deterministic for fixed seed and contents
sampling respects the window (rows older than window_rows are never drawn)
sampling an empty store returns Empty
counters: produced advances on append, consumed on sample, both survive
reopen
schema version mismatch fails to open
value_target validation accepts only -1/0/+1
```

## Implementation Plan

1. Add `legal_actions` to `GumbelRootResult` and `GumbelStep` in
   `gz-search`; all existing tests and goldens must pass unchanged.
2. Add `crates/gz-replay` with rocksdb/serde/postcard deps and the
   `gz-engine` serde feature; verify the serde feature actually covers
   every stored gz-engine type, extending gz-engine's serde derives if any
   are missing.
3. Implement records, invariant validation, and errors.
4. Implement keys, schema version, and store open/recovery.
5. Implement append_episode with WriteBatch atomicity and counters.
6. Implement row_index and sample_rows with the internal seeded RNG.
7. Tests per the strategy above.
8. Wire the orchestrator side separately (projection + replay sink +
   RootBaseline/algorithmic reference plumbing); that work belongs with
   `gz-orchestrator`.
9. Update CODEBASE_OUTLINE and AGENTS.md; run fmt, test --all, clippy.

## Deferred

```text
ReplaySampleService process/network protocol for the Python trainer
graded/tanh targets derived from stored rewards
per-step measurement modes and dense cost data
graph-hash secondary indexes
retention, eviction, and compaction policy
compression
episode resume/remeasure via artifacts and resolvers
cross-run deduplication
auxiliary value targets (root_search_value bootstrapping)
```
