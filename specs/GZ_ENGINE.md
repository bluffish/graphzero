# gz-engine Contract

Status: draft

Purpose: define the detailed contract for `gz-engine`, the foundation crate for
GraphZero's engine abstraction. This crate owns the engine traits, the portable
engine-boundary value types, and the contract tests that every engine adapter
must pass.

`gz-engine` replaces the need for a separate `gz-core` crate at the first
implementation stage. It may own small cross-pipeline identifiers when doing so
prevents dependency cycles, but it must not grow into an orchestration, replay,
feature, or search crate.

## Role

`gz-engine` answers one question:

```text
What can a deterministic graph engine do, and what typed data crosses that
engine boundary?
```

It owns:

```text
GraphEngine and BatchGraphEngine traits
engine-neutral options and result structs
portable graph/candidate/config/version identifiers
portable search action references
candidate metadata needed by policy/replay/logging
measurement summaries produced by engines
engine-neutral error categories
adapter contract test helpers
```

It does not own:

```text
async scheduling
bounded queues
EngineServer implementation
search policies
feature extraction
evaluator/model code
replay storage schema
RocksDB integration
runtime actor/episode/request id allocation
concrete fake, Whittle, or compiler engines
```

## Dependency Contract

Allowed by default:

```text
std
smallvec if CandidateInfo subjects need inline storage
```

Allowed behind explicit features:

```text
serde support for hashes, options, results, and metadata
```

Forbidden:

```text
tokio or any async runtime
async-trait
rocksdb
torch/Python bindings
concrete engine dependencies
gz-search
gz-replay
gz-orchestrator
gz-engine-fake
gz-engine-whittle
logging/tracing framework dependencies
```

The default feature set must not pull in storage, async runtime, model, or
adapter dependencies.

## Core Traits

`GraphEngine` is sync by design. Async execution, timeouts, batching policy,
backpressure, process boundaries, and crash isolation belong to
`gz-orchestrator` and its `EngineServer` wrapper.

```rust
pub trait GraphEngine {
    type Graph: Copy + Eq + std::hash::Hash + Send + Sync + 'static;
    type Candidate: Copy + Eq + std::hash::Hash + Send + Sync + 'static;

    fn engine_id(&self) -> EngineId;
    fn engine_version(&self) -> EngineVersion;
    fn action_set_hash(&self) -> ActionSetHash;

    fn root(&self) -> Self::Graph;

    fn hash(&self, graph: Self::Graph) -> EngineResult<GraphHash>;

    fn candidates(
        &mut self,
        graph: Self::Graph,
        options: CandidateOptions,
        out: &mut Vec<Self::Candidate>,
    ) -> EngineResult<()>;

    fn candidate_info(
        &self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<CandidateInfo>;

    fn apply(
        &mut self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<ApplyResult<Self::Graph, Self::Candidate>>;

    fn measure(
        &mut self,
        graph: Self::Graph,
        options: MeasureOptions,
    ) -> EngineResult<MeasureResult<Self::Graph>>;

    fn release(
        &mut self,
        graphs: &[Self::Graph],
        candidates: &[Self::Candidate],
    ) -> EngineResult<()> {
        let _ = (graphs, candidates);
        Ok(())
    }

    fn export_graph(&self, graph: Self::Graph) -> EngineResult<GraphArtifact>;
}
```

Trait rules:

```text
Graph and Candidate are engine-local handles, not serialized graph bodies.
Search may store Graph and Candidate handles but must not inspect them.
Candidate semantics are engine-owned.
root() must be deterministic for a given engine config/version.
hash(graph) must return the canonical portable identity for that graph.
candidates(graph, options, out) must clear out before writing candidates so
callers can reuse allocation without stale entries.
candidate_info(graph, candidate) must describe the same candidate apply() sees.
measure(graph, options) measures the supplied graph and owns reward/score
production for that measurement.
release(graphs, candidates) frees engine-local resources for handles the caller
owns. Using a released handle afterwards is a contract violation; engines may
reuse released slots. The default implementation is a no-op for engines that
retain handles forever.
export_graph(graph) is for diagnostics/import-export, never hot-loop state.
```

Handle ownership:

```text
The creator of a handle owns it and is responsible for release when downstream
portable data has been copied. Root sources own the roots they yield; search
must not release those source-owned roots. Search owns graphs created by
apply() and candidates created by candidates(). Orchestrator lanes call
release() after episode projection and replay append/drop handling. release()
is a lane-thread engine call, not a SearchWork variant; episodes are complete
before release runs.
```

`BatchGraphEngine` defines batch semantics, not async semantics.

```rust
pub trait BatchGraphEngine: GraphEngine {
    fn candidates_batch(
        &mut self,
        graphs: &[Self::Graph],
        options: CandidateOptions,
    ) -> Vec<EngineResult<Vec<Self::Candidate>>>;

    fn apply_batch(
        &mut self,
        jobs: &[ApplyJob<Self::Graph, Self::Candidate>],
    ) -> Vec<EngineResult<ApplyResult<Self::Graph, Self::Candidate>>>;

    fn measure_batch(
        &mut self,
        graphs: &[Self::Graph],
        options: MeasureOptions,
    ) -> Vec<EngineResult<MeasureResult<Self::Graph>>>;
}
```

Batch rules:

```text
result length equals input length
result order equals input order
batch behavior equals ordered single-call behavior
one failed row does not poison unrelated rows
batch methods may use internal batching/caches, but observable results must be
the same as single-call methods
```

Default batch implementations may loop over single-call methods. Adapters can
override them when the underlying engine has real batch capability.

## Hash and Version Types

`gz-engine` owns opaque fixed-width identifiers that are shared across search,
features, eval, replay, and orchestration.

```rust
pub struct GraphHash([u8; 32]);
pub struct CandidateHash([u8; 32]);
pub struct ActionSetHash([u8; 32]);
pub struct MeasureConfigHash([u8; 32]);
pub struct SearchConfigHash([u8; 32]);

pub struct EngineId([u8; 16]);
pub struct EngineVersion([u8; 16]);
pub struct ModelVersion([u8; 16]);
```

Required behavior:

```text
Copy
Clone
Eq
PartialEq
Ord
PartialOrd
Hash
Debug
Display as lowercase hex
FromStr from lowercase or uppercase hex
as_bytes()
from_bytes()
try_from_hex()
```

Serialization rules:

```text
binary encoding stores raw bytes
human-readable encoding stores lowercase hex
invalid hex length is an error
invalid hex character is an error
```

Semantics:

```text
GraphHash identifies the engine's canonical graph state.
CandidateHash identifies an engine-stable candidate within a graph/action set.
ActionSetHash identifies legal candidate semantics for an engine/config.
MeasureConfigHash identifies settings that affect measure output.
SearchConfigHash identifies settings that affect search output.
EngineId identifies the engine adapter family.
EngineVersion changes when engine behavior can change hashes, candidates,
apply results, exported artifacts, or measurement semantics.
ModelVersion tags evaluator outputs, even though model code lives elsewhere.
```

`gz-engine` defines the storage shape. The owning crate computes the values.

`ActionSetHash` is owned by the engine instance/config. Callers do not pass it
through `CandidateOptions`. It changes when the legal candidate universe or
candidate identity rules change: enabled rewrite families, Whittle action rules,
compiler rewrite registry, candidate legality checks, or any adapter config that
changes what a candidate means. It should not change for per-call projection
options such as result limits, deterministic ordering, batching, logging, or
timeouts.

`EngineVersion` is intentionally conservative in the first design. If
measurement semantics change but candidate/apply semantics do not, this single
version may over-invalidate candidate or apply caches. That is acceptable until
measurement shows the invalidation cost matters. If it does, split the version
surface into narrower graph/action/apply/measure versions instead of reusing
stale cache entries across a broad version change.

`CandidateHash` derivation must include:

```text
graph_hash
action_set_hash
engine-defined stable candidate key
```

It must not be derived from vector position alone.

If an adapter detects that two different semantic graphs or candidates share the
same hash, it must return `EngineError::Internal` and refuse to silently alias
them. Hash collisions are engine bugs for GraphZero's purposes.

## Portable References

Process-local graph handles cannot enter replay or cross future process
boundaries. Use portable references there. The spec separates pure graph
identity from action-context identity so a graph hash does not accidentally mean
"this graph under these candidate rules."

```rust
pub struct PortableGraphId {
    pub graph_hash: GraphHash,
    pub engine_id: EngineId,
    pub engine_version: EngineVersion,
}

pub struct ReplayGraphContext {
    pub graph: PortableGraphId,
    pub action_set_hash: ActionSetHash,
}

pub struct PortableCandidateRef {
    pub context: ReplayGraphContext,
    pub candidate_hash: CandidateHash,
}

pub enum PortableSearchActionRef {
    Candidate(PortableCandidateRef),
    Stop { context: ReplayGraphContext },
}

pub struct SearchStepRef {
    pub before: ReplayGraphContext,
    pub action: PortableSearchActionRef,
    pub after: ReplayGraphContext,
}
```

Rules:

```text
PortableGraphId contains no engine-local Graph handle.
PortableGraphId is pure graph identity: graph hash plus engine family/version.
ReplayGraphContext adds the action-set semantics needed to interpret candidates.
PortableCandidateRef is only meaningful with its graph/action context.
PortableSearchActionRef::Candidate wraps an engine-owned candidate reference.
PortableSearchActionRef::Stop is search-owned control flow at a graph context.
Replay rows store portable refs and contexts, not E::Graph or E::Candidate.
Cache keys that depend on graph semantics must include engine/version/config
fields as needed.
CandidateHash by itself is not a durable action reference.
STOP has no CandidateHash and must not be passed to GraphEngine::apply().
SearchStepRef is the portable transition shape for episode traces.
SearchStepRef.action context must equal SearchStepRef.before.
For Candidate actions, SearchStepRef.after identifies the graph produced by
applying the candidate, or the rejected/no-op result recorded by the owning
episode trace.
For Stop actions, SearchStepRef.after must equal SearchStepRef.before.
```

Portable references are identifiers, not resolvers. They do not guarantee that
a future process can reconstruct `E::Graph` or `E::Candidate`. Replay rows are
training/sampling records by default. Workflows that need resume, remeasure,
debug export, or deterministic reconstruction must also store graph artifacts,
engine-owned replay state, or use an explicit resolver boundary that can map
portable refs back to engine-local handles.

Resolver shape, if a workflow needs it:

```rust
pub trait EngineReplayResolver<E: GraphEngine> {
    fn resolve_graph(&mut self, graph: PortableGraphId) -> EngineResult<E::Graph>;
    fn resolve_candidate(
        &mut self,
        candidate: PortableCandidateRef,
    ) -> EngineResult<E::Candidate>;
}
```

That resolver is not part of the first hot-path engine contract. It belongs in
the adapter/replay bridge for workflows that actually need reconstruction.

## Options

Options are engine-neutral knobs. They describe the requested operation, not a
concrete adapter's private config.

```rust
pub struct CandidateOptions {
    pub max_candidates: Option<usize>,
    pub deterministic_order: bool,
}

pub struct MeasureOptions {
    pub config_hash: MeasureConfigHash,
    pub samples: u32,
    pub timeout_ms: Option<u64>,
    pub deterministic: bool,
}
```

Rules:

```text
CandidateOptions describes the requested enumeration shape for one call.
CandidateOptions must not include ActionSetHash; action-set identity comes from
the engine instance.
CandidateOptions must not expose Whittle/compiler-specific concepts.
CandidateOptions fields may filter, limit, or order returned candidates, but
must not redefine candidate semantics.
If a setting changes the legal candidate universe or candidate identity rules,
it belongs in engine config and must affect ActionSetHash.
Rejected rewrites are apply outcomes, not candidate enumeration output.
MeasureOptions must include enough information to make measure cache keys safe.
Adapter-specific options belong in adapter construction config, not per-call
engine-neutral options, unless all engines can give the field the same meaning.
```

Default candidate enumeration should be deterministic.
Changing CandidateOptions must not change engine.action_set_hash().

## Candidate Metadata

Candidate handles are engine-local. `CandidateInfo` is the portable metadata
used by policy heads, replay, diagnostics, and logs.

```rust
pub struct CandidateInfo {
    pub candidate_hash: CandidateHash,
    pub graph_hash: GraphHash,
    pub action_set_hash: ActionSetHash,
    pub kind: CandidateKindId,
    pub display_name: String,
    pub static_prior: f32,
    pub tags: CandidateTags,
    pub subjects: Vec<SubjectId>,
    pub metadata: CandidateMetadata,
}

pub struct CandidateKindId(u32);
pub struct SubjectId(u64);
pub struct CandidateTags(u64);
pub struct CandidateMetadata {
    pub bytes: Vec<u8>,
}
```

Rules:

```text
candidate_hash must match the candidate passed to apply().
graph_hash must match hash(graph).
static_prior must be finite.
display_name is for humans and logs, not stable identity.
subjects are optional portable references to affected domain objects.
metadata is adapter-owned opaque data and must not be interpreted by search.
```

Search may use:

```text
candidate_hash
kind
static_prior
tags
```

Search must not infer domain semantics from `display_name`, `subjects`, or
opaque metadata.

## Apply Results

```rust
pub struct ApplyJob<G, C> {
    pub graph: G,
    pub candidate: C,
}

pub struct ApplyResult<G, C> {
    pub before: G,
    pub after: G,
    pub before_hash: GraphHash,
    pub after_hash: GraphHash,
    pub candidate: C,
    pub candidate_hash: CandidateHash,
    pub changed: bool,
    pub rejected: Option<RewriteRejection>,
    pub metrics: ApplyMetrics,
}

pub struct RewriteRejection {
    pub code: ErrorCode,
    pub message: ErrorMessage,
}

pub struct ApplyMetrics {
    pub elapsed_ms: Option<f32>,
    pub engine_steps: Option<u64>,
}
```

Rules:

```text
before_hash must match hash(before).
candidate_hash must match candidate_info(before, candidate).candidate_hash.
after_hash must match hash(after) when rejected is None.
A non-stop accepted transition must either change graph hash or explicitly set
changed=false for an engine-defined no-op.
Rejected transitions are expected data, not panics.
Stale candidates must return EngineError::StaleCandidate.
```

## Measure Results

`GraphEngine::measure` measures the supplied graph. It is not restricted to
"terminal" graphs by the engine contract. Search may choose to measure only the
last graph in an episode, but that is a search/replay lifecycle decision, not a
precondition of `measure()`.

```rust
pub struct MeasureResult<G> {
    pub graph: G,
    pub graph_hash: GraphHash,
    pub config_hash: MeasureConfigHash,
    pub measured: bool,
    pub valid: bool,
    pub latency: Option<LatencyStats>,
    pub scalar_reward: Option<f32>,
    pub failure: Option<MeasureFailure>,
    pub metadata: MeasureMetadata,
}

pub struct MeasureSummary {
    pub graph_hash: GraphHash,
    pub config_hash: MeasureConfigHash,
    pub measured: bool,
    pub valid: bool,
    pub latency: Option<LatencyStats>,
    pub scalar_reward: Option<f32>,
    pub failure_code: Option<ErrorCode>,
}

pub struct LatencyStats {
    pub mean_ms: f32,
    pub median_ms: f32,
    pub p95_ms: f32,
    pub samples_ms: Vec<f32>,
}

pub struct MeasureFailure {
    pub code: ErrorCode,
    pub message: ErrorMessage,
}

pub struct MeasureMetadata {
    pub bytes: Vec<u8>,
}
```

Rules:

```text
graph_hash must match hash(graph).
latency and scalar_reward are reward-eligible only when valid == true.
latency fields and samples must be finite and non-negative.
scalar_reward must be finite when present.
measured=false means no measurement was produced.
valid=false means measurement completed but should not be used as a valid
reward target unless a downstream config explicitly allows failed rows.
measurement failures are data, not process crashes.
```

Examples:

```text
WhittleTestEngine can measure any Whittle state with a simple score such as node
count or another deterministic domain statistic.
FutureCompilerEngine can measure any materializable compiler graph by running
correctness checks, lowering, and timing when requested by MeasureOptions.
```

`MeasureSummary` is the compact replay-facing projection. `MeasureResult`
remains the full engine result.

## Graph Artifacts

Graph artifacts are for diagnostics, import/export, and CLI inspection. They
are not hot-loop state and must not be required by search.

```rust
pub struct GraphArtifact {
    pub graph_hash: GraphHash,
    pub format: GraphArtifactFormat,
    pub bytes: Vec<u8>,
}

pub enum GraphArtifactFormat {
    Text,
    Json,
    Dot,
    Binary,
    AdapterSpecific(u32),
}
```

Rules:

```text
export_graph must not be called in search hot loops.
artifact bytes may be large and are not stored in MCTS nodes.
artifact format is descriptive only; concrete adapters own exact encoding.
```

## Errors

```rust
pub type EngineResult<T> = Result<T, EngineError>;

pub enum EngineError {
    UnknownGraph {
        graph_hash: Option<GraphHash>,
    },
    UnknownCandidate {
        candidate_hash: Option<CandidateHash>,
    },
    StaleCandidate {
        expected_graph_hash: GraphHash,
        actual_graph_hash: GraphHash,
        candidate_hash: CandidateHash,
    },
    Timeout {
        operation: OperationKind,
        limit_ms: u64,
    },
    Internal {
        code: ErrorCode,
        message: ErrorMessage,
    },
}

pub struct ErrorCode(u32);
pub struct ErrorMessage(String);

pub enum OperationKind {
    Root,
    Hash,
    Candidates,
    CandidateInfo,
    Apply,
    Measure,
    ExportGraph,
}
```

Rules:

```text
EngineError must not contain concrete engine state.
EngineError must not contain full graph or candidate bodies.
EngineError means the engine could not complete the requested operation.
EngineError::Internal is for adapter bugs or unexpected failures.
Expected rejected rewrites should be ApplyResult.rejected when an apply result
can still be produced.
Expected failed measurements should be MeasureResult.failure when a measurement
result can still be produced.
Timeout is reported by the layer that owns the timeout.
```

`ErrorMessage` should be bounded at construction time once implementation
starts. Logs can attach richer context outside this crate.

## Determinism Invariants

For a fixed engine configuration and version:

```text
root() returns the same root handle semantics
hash(g) returns the same GraphHash for the same semantic graph
candidates(g, options) returns the same CandidateHash set
candidates(g, options) returns deterministic order when requested
candidate_info(g, c) returns stable portable fields
apply(g, c) returns the same after_hash or same rejection
measure(g, options) returns deterministic outputs when options.deterministic
is true and the adapter supports deterministic measurement
export_graph(g) returns an artifact for the same semantic graph
```

Panics indicate engine bugs. Unknown graphs, stale candidates, rejected
rewrites, failed measurements, and timeouts are expected outcomes.

## Cache-Key Rules

`gz-engine` provides the pieces used in cache keys. Owning crates compose the
actual keys.

Required ingredients:

```text
candidate enumeration: ReplayGraphContext + CandidateOptions
candidate info: PortableCandidateRef
apply result: PortableCandidateRef
measure result: PortableGraphId + MeasureConfigHash
feature cache: PortableGraphId + feature extractor config outside gz-engine
replay row graph identity: ReplayGraphContext
search action identity: PortableSearchActionRef
```

If a cache key needs concrete engine state, the key does not belong in
`gz-engine`.

## Contract Tests

`gz-engine` owns reusable contract tests that adapters run against themselves.

Minimum adapter contract:

```text
create engine
read engine_id and engine_version
obtain root
hash root
enumerate root candidates
fetch CandidateInfo for each root candidate
apply every root candidate
re-enumerate candidates after apply
apply a known deterministic path
export root graph
measure root and a known graph reached by apply()
```

Determinism checks:

```text
root hash matches across engine instances
candidates(root) CandidateHash list matches across runs
engine.action_set_hash() is stable across CandidateOptions changes
candidate_info(root, candidate) stable fields match across runs
apply(root, candidate) after_hash matches across runs
known path final hash matches across runs
batch APIs match ordered single-call APIs
measure cache key fields are present in MeasureResult
```

Negative checks:

```text
unknown graph returns UnknownGraph
unknown candidate returns UnknownCandidate
stale candidate returns StaleCandidate
invalid measure options are rejected by MeasureOptions::new
failed measurement returns data, not a panic
```

Compiler-specific measurement checks are deferred until a compiler engine
exists, but the future compiler adapter must reject:

```text
NaN outputs
Inf outputs
missing output key
shape mismatch
allclose mismatch
lowering failure
measure timeout
```

## Public API Constraints

Value types should implement:

```text
Copy where representation allows it
Clone
Eq
PartialEq
Ord where deterministic ordering is useful
Hash
Debug
Send
Sync
'static
```

Do not implement `Default` for hashes, versions, or ids unless a real zero value
is documented. Accidental zero identifiers should be hard to create.

Public constructors must validate invariants:

```text
hex length and characters
finite float fields
non-negative latency fields
bounded error messages
valid option ranges
```

## Implementation Plan

Implement `gz-engine` in small layers. Each layer should compile and have
focused unit tests before the next layer starts.

### 1. Crate Skeleton

Create:

```text
crates/gz-engine/Cargo.toml
crates/gz-engine/src/lib.rs
```

Initial dependencies:

```text
std only
serde optional feature, disabled by default
```

Expose modules:

```rust
pub mod error;
pub mod hash;
pub mod measure;
pub mod metadata;
pub mod options;
pub mod refs;
pub mod traits;
```

`lib.rs` should re-export the public contract types so downstream crates can
import from `gz_engine::*` while implementation stays organized.

Verification:

```bash
cargo test -p gz-engine
cargo clippy -p gz-engine --all-targets --all-features
```

### 2. Hash And Version Newtypes

Implement:

```rust
GraphHash
CandidateHash
ActionSetHash
MeasureConfigHash
SearchConfigHash
EngineId
EngineVersion
ModelVersion
```

Required APIs:

```rust
as_bytes()
from_bytes()
try_from_hex()
Display
FromStr
Debug
```

Rules:

```text
no Default impls
lowercase hex Display
uppercase and lowercase hex accepted by FromStr
binary serde as raw bytes when serde is enabled
human-readable serde as hex when serde is enabled
```

Tests:

```text
byte roundtrip for every id type
hex roundtrip for every id type
uppercase parse
invalid length rejection
invalid character rejection
stable Ord/Hash behavior for map keys
serde roundtrip behind serde feature
```

### 3. Error Types

Implement:

```rust
EngineResult<T>
EngineError
ErrorCode
ErrorMessage
OperationKind
```

Rules:

```text
ErrorMessage constructor validates maximum byte length
EngineError contains no graph or candidate bodies
Display is concise and stable enough for logs
std::error::Error is implemented
serde support is feature-gated
```

Tests:

```text
ErrorMessage rejects overlong messages
EngineError preserves stale candidate context
Display includes operation/code where relevant
serde roundtrip behind serde feature
```

### 4. Portable References

Implement:

```rust
PortableGraphId
ReplayGraphContext
PortableCandidateRef
PortableSearchActionRef
SearchStepRef
```

Rules:

```text
SearchStepRef constructor validates action context == before
SearchStepRef constructor validates Stop after == before
SearchStepRef does not try to prove apply correctness
all refs are Copy when their fields allow it
all refs are stable map keys
```

Tests:

```text
portable graph id equality and ordering
candidate ref includes graph/action context
search step rejects candidate context mismatch
serde roundtrip behind serde feature
```

### 5. Options

Implement:

```rust
CandidateOptions
MeasureOptions
```

Defaults:

```rust
CandidateOptions {
    max_candidates: None,
    deterministic_order: true,
}
```

`MeasureOptions` should not have a silent default until there is a documented
default `MeasureConfigHash`. Prefer explicit construction.

Tests:

```text
CandidateOptions default is deterministic and unlimited
CandidateOptions has no ActionSetHash field
MeasureOptions rejects samples == 0 if that is invalid for the first engines
timeout validation if timeout constraints are added
```

### 6. Metadata And Result Types

Implement:

```rust
CandidateInfo
CandidateKindId
SubjectId
CandidateTags
CandidateMetadata
ApplyJob
ApplyResult
RewriteRejection
ApplyMetrics
MeasureResult
MeasureSummary
LatencyStats
MeasureFailure
MeasureMetadata
GraphArtifact
GraphArtifactFormat
```

Rules:

```text
constructors validate finite floats
LatencyStats validates finite non-negative values
CandidateInfo validates finite static_prior
MeasureResult validates finite scalar_reward when present
opaque metadata stays bytes for the first implementation
MeasureSummary can be built from MeasureResult without engine-local handles
```

Tests:

```text
LatencyStats rejects NaN, Inf, and negative values
CandidateInfo rejects NaN static_prior
MeasureResult rejects NaN scalar_reward
MeasureSummary drops engine-local graph handle
GraphArtifact preserves graph_hash and format
```

### 7. Traits And Default Batch Implementations

Implement:

```rust
GraphEngine
BatchGraphEngine
```

Provide default `BatchGraphEngine` methods that loop over single-call methods.

Rules:

```text
Graph and Candidate associated types require Copy + Eq + Hash + Send + Sync + 'static
candidates() contract says implementations clear out before writing
GraphEngine remains sync
no async runtime or async-trait
```

Tests can use a tiny local test engine inside `gz-engine` only for trait/default
batch behavior. The real fake engine lives in `gz-engine-fake`.

Tests:

```text
default candidates_batch matches ordered single calls
default apply_batch matches ordered single calls
default measure_batch matches ordered single calls
one row failure does not stop later rows
candidates() test helper detects stale output entries
```

### 8. Contract Test Harness

Implement reusable adapter contract helpers under a test-support feature or a
public `contract` module if downstream adapter crates need it in their tests.

Shape:

```rust
pub trait EngineContractFixture {
    type Engine: GraphEngine;

    fn make_engine(&self) -> Self::Engine;
    fn known_path(&self) -> Vec<<Self::Engine as GraphEngine>::Candidate>;
    fn unknown_graph(&self) -> Option<<Self::Engine as GraphEngine>::Graph>;
    fn unknown_candidate(&self) -> Option<<Self::Engine as GraphEngine>::Candidate>;
}
```

Start with helpers, not macros, unless macros clearly reduce boilerplate.

Checks:

```text
root/hash/candidates/candidate_info/apply/export/measure smoke
determinism across two engine instances
batch APIs equal single-call APIs
action_set_hash stable across CandidateOptions changes
unknown graph/candidate behavior when fixture supports it
failed measurement is data, not panic
```

### 9. Feature-Gated Serde Sweep

After all core types exist, add or finish serde support behind `serde`.

Rules:

```text
serde is optional and disabled by default
human-readable hash/id serialization is hex
binary hash/id serialization is raw bytes
opaque metadata serializes as bytes
```

Verification:

```bash
cargo test -p gz-engine
cargo test -p gz-engine --features serde
cargo clippy -p gz-engine --all-targets --all-features
```

### 10. Readiness Criteria

`gz-engine` is ready for `gz-engine-fake` when:

```text
all public contract types compile
all constructor invariants have unit tests
default batch implementations are tested
contract test helpers can be called by another crate
no async runtime dependency exists
no concrete engine dependency exists
serde feature is optional
cargo test -p gz-engine passes
cargo clippy -p gz-engine --all-targets --all-features passes
```

## Open Questions

1. Should `EngineVersion` and `ModelVersion` stay 128-bit values or use the same
   256-bit shape as graph/config hashes?
2. When should the single conservative `EngineVersion` split into narrower
   graph/action/apply/measure semantic versions?
3. Should `CandidateMetadata` and `MeasureMetadata` be opaque bytes, typed enums,
   or omitted until a concrete adapter needs them?
