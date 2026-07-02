# gz-features Spec

Status: draft

Purpose: define the feature extraction crate that turns engine states and
legal actions into the tensors the neural evaluator and trainer consume.
gz-features owns the feature schema, the portable per-state feature row, the
batched tensor encoding that crosses the process boundary to Python, and the
decoding of model outputs back into per-row results.

## Architecture Overview

Where features sit in the pipeline:

```text
lane thread (has engine)          batcher thread (no engine)         Python
─────────────────────────        ───────────────────────────       ─────────
worker parks on Eval
  -> FeatureExtractor.extract()   FeatureRow crosses (portable)
     graph body -> FeatureRow  ─>  FeatureCollator.collate()
                                   rows -> one padded FeatureBatch
                                   (bytes; fixed binary layout)   ─> frombuffer
                                                                     .view(shape)
                                                                     model forward
                                   decode_outputs(bytes, counts) <─  flat outputs
                                   -> per-row logits + value
                                   -> resume workers
```

The five decisions this spec fixes (review these before implementation):

```text
1. One concrete representation, not per-engine associated types. This
   AMENDS the FeatureExtractor sketch in CODEBASE_OUTLINE. Every engine
   maps into the same FeatureRow (node tokens + node attrs + typed edges +
   action features + position). Payoff: one wire format, one collator, one
   Python input parser, non-generic orchestrator plumbing. Engines that
   need more express it through the schema config (bigger vocab, attr_dim,
   more edge types), not through new types.

2. Extraction and collation are separate types with separate homes.
   FeatureExtractor<E> is engine-side code (needs graph bodies) and runs on
   lanes. FeatureCollator is schema-only code and runs in the batcher. The
   FeatureRow between them is fully portable.

3. Static shapes, v1. Every batch tensor has one shape, fixed by the schema
   (max_nodes, max_edges, max_actions, max_subjects) and the collator's
   batch capacity. Whittle's engine capacity makes max_nodes free. This is
   what makes torch.compile(fullgraph) and CUDA-graph capture trivial on
   the Python side. Shape buckets for the compiler regime are a later
   schema evolution, not a v1 concern.

4. The wire format IS the tensor format, both directions. gz-features owns
   the batch byte layout and the model-output byte layout, so Rust and
   Python cannot drift apart without FeatureSchemaHash catching it. Python
   ingest is frombuffer + view; zero per-row Python work.

5. The Whittle extractor lives in gz-engine-whittle (new module, new dep on
   gz-features). It needs direct arena access; exposing the arena publicly
   so a sibling crate can read it would be worse. This AMENDS
   GZ_ENGINE_WHITTLE.md's dependency list.

Defaults chosen for the open questions (cheap to change until the first
checkpoint is trained, expensive after):
   max_actions = 256 for Whittle (invariant: search max_candidates + 1 must
   fit; enforced at config validation)
   node_attr_dim = 0 for Whittle v1 (pure op tokens; structural scalars
   like depth/fanout are a schema evolution)
   max_subjects = 8 (Whittle's matched-node array size, exactly)

Deliberately deferred: action-history features (EvalRequest carries no
history; train/serve parity forbids training-only features), opponent
trajectory blocks (job 2; the batch layout uses named sections so a second
graph block is additive), positional encodings (cheap ones become node
attrs later; expensive ones live inside the compiled model), shape buckets.
```

## Role

`gz-features` answers:

```text
What tensors does the model see for one state and its legal actions, how are
many states packed into one batch, and how do model outputs map back to
per-action results?
```

It owns:

```text
FeatureSchema, FeatureSchemaConfig, and FeatureSchemaHash
FeatureRow, ActionFeature, FeatureEdge, PositionFeatures
the FeatureExtractor<E> trait
FeatureCollator, the batch byte encoding, and the output byte decoding
batch/section layout versioning
state feature caching rules
FeatureError
```

It does not own:

```text
concrete extractors (engine adapters own them)
eval transport, sockets, or process lifecycle (eval service, later)
model code, torch, Python
EvalRequest/EvalOutput (gz-eval)
where extraction runs (gz-orchestrator wires lanes/batcher)
training
```

## Dependency Contract

Allowed:

```text
std
gz-engine
blake3 (FeatureSchemaHash)
```

Forbidden:

```text
tokio or any async runtime
torch/Python bindings
serde (rows never serialize; the batch encoding is hand-rolled)
gz-eval, gz-search, gz-replay, gz-orchestrator
concrete engine adapters
```

`PositionFeatures` is defined here rather than importing
`EvalPositionContext` from gz-eval, so gz-features depends only on
gz-engine. The orchestrator maps one to the other.

## Schema

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureSchemaConfig {
    pub name: String,               // e.g. "whittle-v1"
    pub node_vocab_size: u16,       // includes PAD
    pub node_attr_dim: u16,
    pub edge_type_count: u8,
    pub action_kind_vocab_size: u32, // includes PAD and STOP
    pub max_nodes: u32,
    pub max_edges: u32,
    pub max_actions: u32,
    pub max_subjects: u32,
}

pub struct FeatureSchema { /* validated config + derived hash */ }

impl FeatureSchema {
    pub fn new(config: FeatureSchemaConfig) -> FeatureResult<Self>;
    pub fn config(&self) -> &FeatureSchemaConfig;
    pub fn hash(&self) -> FeatureSchemaHash;
}

pub struct FeatureSchemaHash([u8; 32]);   // Display/FromStr hex, as_bytes,
                                          // same conventions as gz-engine hashes
```

Token conventions (fixed, documented, part of the encoding version):

```text
node tokens:   0 = PAD, engine ops start at 1
action kinds:  0 = PAD, 1 = STOP, engine CandidateKindId k maps to k + 2
subject slots: u32::MAX = PAD
edge types:    engine-defined, 0-based; padding is expressed by edge_count,
               not a reserved type
```

Config validation:

```text
node_vocab_size >= 2, action_kind_vocab_size >= 3 (PAD + STOP + one kind)
max_nodes, max_edges, max_actions, max_subjects >= 1
name non-empty
```

`FeatureSchemaHash` derivation: blake3 over a domain prefix
("gz-features-schema-v1"), the encoding version constant, and every config
field, length-delimited. Any change to any field, the token conventions, or
the byte layout changes the hash. The hash rides with EngineVersion,
ActionSetHash, and ModelVersion in the evaluator/trainer fail-fast tag
check; a checkpoint trained under one schema must be rejected by an
evaluator collating another.

Batch capacity is deliberately NOT part of the schema hash: it is transport
configuration, carried in the batch header and checked by the evaluator
against its compiled capacity. Same schema, different capacities is legal
across deployments.

## Feature Rows

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureRow {
    pub node_count: u32,
    pub node_tokens: Vec<u16>,        // len == node_count
    pub node_attrs: Vec<f32>,         // len == node_count * node_attr_dim
    pub edges: Vec<FeatureEdge>,
    pub actions: Vec<ActionFeature>,  // candidates in enumeration order, STOP last
    pub position: PositionFeatures,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureEdge {
    pub src: u32,
    pub dst: u32,
    pub edge_type: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionFeature {
    pub kind_token: u32,
    pub static_prior: f32,
    pub subjects: Vec<u32>,           // node indices, len <= max_subjects
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionFeatures {
    pub root_step: u32,
    pub leaf_depth: u32,
    pub budget_fraction: f32,
    pub budget_step: f32,
}
```

Row rules (validated by the collator while packing, where bounds checks are
free; `FeatureRow::validate(&self, &FeatureSchema)` exists for tests):

```text
node_count <= max_nodes; edges.len() <= max_edges;
actions.len() <= max_actions — extraction must NEVER truncate actions;
overflow is FeatureError::ActionOverflow, and the deployment invariant is
candidate_options.max_candidates + 1 <= max_actions
tokens < their vocab sizes; edge endpoints and subjects < node_count
attrs length == node_count * attr_dim; finite floats throughout
actions is exactly the search action list: engine candidates in enumeration
order, then STOP (kind_token 1, empty subjects, static_prior 0.0) last —
index i here must correspond to EvalRequest.actions[i]
```

## Extractor Trait

```rust
pub trait FeatureExtractor<E: GraphEngine> {
    fn schema(&self) -> &FeatureSchema;

    fn extract(
        &mut self,
        engine: &E,
        graph: E::Graph,
        candidates: &[E::Candidate],
        position: PositionFeatures,
    ) -> FeatureResult<FeatureRow>;
}
```

Rules:

```text
One call produces the complete row. The state/candidate split from the old
CODEBASE_OUTLINE sketch is an implementation detail behind extract, not API.
&E, not &mut E: extraction reads graph bodies, hashes, and candidate_info;
it must not mutate engine state.
Extractors are engine-side code (they need graph bodies the GraphEngine
trait deliberately does not expose) and run on lanes; they are Send,
constructed one per lane.
Deterministic: fixed (engine config, graph, candidates, position, schema)
produces an identical FeatureRow.
Subjects come from CandidateInfo.subjects; extractors call candidate_info
themselves.
```

Caching:

```text
The state-derived portion (tokens, attrs, edges) is cached inside the
extractor keyed by PortableGraphId; the schema is fixed per instance so it
is not part of the key. Cache hits clone the cached vectors (Arc-backed
sharing is a deferred optimization). Unbounded, like the engine caches,
until measurement says otherwise. Action features and position are per-call
and never cached.
```

## Batch Encoding

```rust
pub struct FeatureCollator { /* schema + batch capacity + scratch */ }

impl FeatureCollator {
    pub fn new(schema: FeatureSchema, batch_capacity: NonZeroUsize) -> Self;

    /// Clears `out` and writes one encoded batch.
    pub fn collate_into(
        &mut self,
        rows: &[FeatureRow],
        out: &mut Vec<u8>,
    ) -> FeatureResult<()>;

    pub fn decode_outputs(
        &self,
        bytes: &[u8],
        action_counts: &[u32],
    ) -> FeatureResult<Vec<RowOutput>>;
}

pub struct RowOutput {
    pub policy_logits: Vec<f32>,   // truncated to the row's true action count
    pub value: f32,
}
```

Batch layout, exact and versioned. All integers little-endian; every
section starts at a 4-byte-aligned offset (zero-padded); B is
batch_capacity, N = max_nodes, E = max_edges, A = max_actions,
S = max_subjects, D = node_attr_dim:

```text
header:
  magic "GZFB", encoding version u32, FeatureSchemaHash 32 bytes,
  batch_capacity u32, row_count u32, N u32, E u32, A u32, S u32, D u32
sections, in this order:
  node_count      [B] u32
  node_tokens     [B, N] u16          (PAD = 0)
  node_attrs      [B, N, D] f32       (zeros in padding; absent when D = 0)
  edge_count      [B] u32
  edge_src        [B, E] u32
  edge_dst        [B, E] u32
  edge_type       [B, E] u8
  action_count    [B] u32
  action_kind     [B, A] u32          (PAD = 0)
  action_prior    [B, A] f32
  subject_count   [B, A] u8
  action_subjects [B, A, S] u32       (PAD = u32::MAX)
  position        [B, 4] f32          (root_step, leaf_depth as f32,
                                       budget_fraction, budget_step)
```

Rows beyond row_count are all-padding. `1 <= rows.len() <= batch_capacity`,
else `FeatureError::BatchOverflow` / `EmptyBatch`.

Output layout (model -> Rust):

```text
magic "GZFO", encoding version u32, row_count u32, A u32
  value  [B] f32
  policy [B, A] f32     (raw logits; padded action slots are ignored by
                         decode, never masked with non-finite values)
```

`decode_outputs` validates magic, version, row_count == action_counts.len(),
and A, then slices per row by true action count. Finiteness validation stays
where it already lives (gz-eval output validation at task resume).

A read-side parser (`FeatureBatchView::parse(&[u8])`) is public: it
validates the header and exposes typed section slices. It exists for tests,
diagnostics, and the future Python conformance test; the hot path never
parses its own batches.

Performance rules:

```text
collate_into reuses the caller's buffer; no per-batch Vec<u8> allocation
one pass over rows; section offsets are computed, not discovered
no compression, no varints, no self-describing container (considered
safetensors; rejected to avoid per-batch JSON header parsing — the schema
hash plus encoding version does the compatibility job)
```

## Whittle Extractor (contract here, code in gz-engine-whittle)

New module `features` in `gz-engine-whittle`, new dependency on
`gz-features` (this amends GZ_ENGINE_WHITTLE.md's allowed deps).

```rust
pub struct WhittleFeatureExtractor { /* schema + state cache */ }

impl WhittleFeatureExtractor {
    pub fn new(engine: &WhittleEngine) -> Self;  // schema derived from engine config
}

impl FeatureExtractor<WhittleEngine> for WhittleFeatureExtractor { ... }
```

Mapping:

```text
schema: name "whittle-v1", node vocab {PAD, Input, Const, And, Or, Not,
Output} (size 7), attr_dim 0, edge types {arg0 = 0, arg1 = 1} (size 2),
max_nodes = engine capacity, max_edges = 2 * capacity,
max_actions = 256, max_subjects = 8,
action kind vocab = rule count + 2
node tokens: op codes in arena node order, offset per the token convention
edges: one edge per present arg, direction argument -> consumer
actions: rule_id + 2 as kind token, CandidateInfo.static_prior,
CandidateInfo.subjects as node indices (Whittle subjects are node ids;
match_len <= 8 == max_subjects by construction)
STOP appended last by the extractor
```

Golden tests: fixed fixture graphs (the x0 AND x0 contract fixture plus one
generator graph at a fixed seed) produce fingerprint-stable FeatureRows and
batch bytes, captured with the same assert-print-paste procedure as the
gumbel goldens. Determinism and cache-consistency (extract twice, identical
row) tests alongside.

## Errors

```rust
pub type FeatureResult<T> = Result<T, FeatureError>;

pub enum FeatureError {
    InvalidSchema,
    NodeOverflow,
    EdgeOverflow,
    ActionOverflow,
    SubjectOverflow,
    InvalidRow,
    BatchOverflow,
    EmptyBatch,
    InvalidEncoding,
}
```

Small and structural. Engine failures during extraction surface as
`EngineError` from the extractor, not wrapped.

## Test Strategy

gz-features (no engine deps; rows built by hand):

```text
schema config validation; hash stable across runs; hash changes for every
config field
FeatureRow::validate catches every bound and length violation
collate golden: hand-built rows -> byte fingerprint literal
collate/parse roundtrip through FeatureBatchView for every section,
including padding values and section alignment
collate rejects overflow/empty; validates rows while packing
decode_outputs: truncation by action count; header mismatches rejected
two collate calls with identical rows produce identical bytes
```

gz-engine-whittle:

```text
Whittle goldens as above
action list alignment: extractor actions.len() == candidates.len() + 1 and
STOP is last
subjects index real nodes (bounds-checked against node_count)
schema hash is stable for a fixed engine config and changes when capacity
or the rule set changes
```

## Implementation Plan

1. Commit the current tree if dirty.
2. `crates/gz-features` skeleton: errors, schema config + validation +
   hash, `PositionFeatures`, `FeatureRow`/`ActionFeature`/`FeatureEdge` +
   `validate`. Unit tests.
3. `FeatureCollator`: collate_into, the exact layout above,
   `FeatureBatchView::parse`, `decode_outputs`. Golden + roundtrip tests.
4. `WhittleFeatureExtractor` in gz-engine-whittle: schema derivation from
   engine config, arena mapping, state cache, goldens. Add the gz-features
   dependency.
5. Docs: replace CODEBASE_OUTLINE's gz-features section sketch with a
   pointer to this spec plus the concrete-representation decision; amend
   GZ_ENGINE_WHITTLE.md's dependency list; AGENTS.md already lists this
   spec.
6. cargo fmt / test --all / clippy -D warnings; goldens for gz-search
   untouched.

## Deferred

```text
wiring into the orchestrator eval path (eval service work order)
opponent trajectory sections in the batch layout (job 2; additive)
action-history features
shape buckets for the compiler regime
Arc-backed cache sharing; bounded caches
structural node attrs (depth, fanout)
normalization constants in-schema
Python-side conformance test (lands with python/evaluator)
```
