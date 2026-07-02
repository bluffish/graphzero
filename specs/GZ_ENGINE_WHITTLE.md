# gz-engine-whittle Spec

Status: draft

Purpose: define the first non-fake `GraphEngine` adapter for GraphZero. This
crate implements Whittle boolean rewrite graphs behind the `gz-engine`
contract, using the existing Whittle native engine semantics as the reference
behavior.

This crate is an architecture validator, not GraphZero's final compiler
backend. It must still be engineered like production hot-path code.

## Role

`gz-engine-whittle` answers:

```text
Can the generic GraphZero engine/search/replay pipeline run against a real
rewrite domain without knowing anything about that domain?
```

It owns:

```text
Whittle graph arena and handles
Whittle rewrite candidate arena and handles
Whittle candidate enumeration
Whittle rewrite application
Whittle canonical graph hashing
Whittle cost measurement
Whittle training graph generation
Whittle graph import/export artifacts
Whittle EngineContractFixture
optional parity fixtures against legacy Whittle native behavior
```

It does not own:

```text
MCTS/search
neural evaluator/model code
generic feature schema/collation
replay storage
Python bindings
PyTorch
async runtime
compiler graph rewrites
```

## Reference Semantics

The source of truth for phase one is the existing Whittle native rewrite
semantics in `/home/ubuntu/whittlezero`:

```text
native/engine/core.h
native/engine/core.cpp
native/engine/rewrites.h
native/engine/rewrites.cpp
engine/graph.py
engine/candidates.py
engine/canonical.py
engine/generate.py
engine/rules.py
```

Relevant reference concepts:

```text
Graph:
  arity: int
  capacity: int
  op: Vec<i8>
  arg0: Vec<u32>
  arg1: Vec<u32>
  output_node: u32

Candidate:
  rule: int
  root: u32
  matched: [u32; 8]
  len: u8

Canonical artifact:
  compact graph serialized as WAV1 bytes
```

The adapter must match:

```text
compact_graph
serialize_raw / WAV1 bytes
enumerate_graph(include_reverse_constant_folding=false)
apply_graph
cost = compact graph node count
sample_training_circuit-style random-walked training graph generation
```

The first implementation should be a Rust port of these semantics, not a
Python bridge.

## Crate Shape

```text
crates/gz-engine-whittle/
  Cargo.toml
  src/
    lib.rs
    config.rs
    graph.rs
    candidate.rs
    arena.rs
    canonical.rs
    enumerate.rs
    apply.rs
    generator.rs
    measure.rs
    metadata.rs
    artifact.rs
    contract_fixture.rs
  tests/
    contract.rs
    golden.rs
    parity_optional.rs
```

Dependencies:

```text
gz-engine
gz-features
blake3 or another explicitly chosen fast hash for 32-byte ids
rand + rand_chacha or another explicitly chosen deterministic RNG
```

Forbidden:

```text
tokio
async-trait
rocksdb
pyo3
torch
gz-search
gz-replay
```

Whittle owns its concrete `WhittleFeatureExtractor` because it needs direct
arena access. Generic schema, row validation, collation, and wire formats stay
in `gz-features`.

If `blake3` is used, all hash derivations in this crate must be documented and
covered by golden tests. Do not add a generic serialization framework for the
hot path.

## Public API

```rust
pub struct WhittleEngine {
    config: WhittleEngineConfig,
    graphs: GraphArena,
    candidates: CandidateArena,
    caches: WhittleCaches,
}

#[derive(Clone, Debug)]
pub struct WhittleEngineConfig {
    pub root: WhittleRoot,
    pub include_reverse_constant_folding: bool,
    pub measure_mode: WhittleMeasureMode,
    pub cache_candidates: bool,
    pub cache_transitions: bool,
}

#[derive(Clone, Debug)]
pub enum WhittleRoot {
    Input { arity: u16, capacity: u16, input_index: u16 },
    Artifact(Vec<u8>),
}

pub type WhittleRng = rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct WhittleGraphGeneratorConfig {
    pub arity: u16,
    pub capacity: u16,
    pub exception_terms_min: u16,
    pub exception_terms_max: u16,
    pub prewalk_steps_min: u16,
    pub prewalk_steps_max: u16,
}

pub struct WhittleGraphGenerator {
    config: WhittleGraphGeneratorConfig,
    rng: WhittleRng,
}

pub struct GeneratedWhittleGraph {
    pub graph: WhittleGraphId,
    pub seed_graph: WhittleGraphId,
    pub prewalk_steps_requested: u16,
    pub prewalk_steps_applied: u16,
    pub start_cost: u32,
    pub final_cost: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhittleMeasureMode {
    NegativeCost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WhittleGraphId(u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WhittleCandidateId(u32);
```

`WhittleGraphId` and `WhittleCandidateId` are process-local handles. They are
cheap to copy and satisfy `GraphEngine::Graph` / `GraphEngine::Candidate`.
Replay and checkpoints must use `ReplayGraphContext`, `PortableCandidateRef`,
and graph artifacts.

## Graph Representation

Internal graph bodies should be compact and cache-friendly:

```rust
pub struct WhittleGraph {
    pub arity: u16,
    pub capacity: u16,
    pub output_node: u32,
    pub op: Box<[OpCode]>,
    pub arg0: Box<[u32]>,
    pub arg1: Box<[u32]>,
    pub canonical: Box<[u8]>,
    pub hash: GraphHash,
}

#[repr(u8)]
pub enum OpCode {
    Input = 0,
    Const = 1,
    And = 2,
    Or = 3,
    Not = 4,
    Output = 5,
}
```

Rules:

```text
1. Store only compact graphs in the arena.
2. `canonical` is the WAV1 compact serialization.
3. `hash` is derived from canonical bytes and engine/version tags.
4. Graph insertion must deduplicate by GraphHash when possible.
5. Graph handles are stable for the engine instance lifetime.
6. Invalid graph artifacts return EngineError::Internal or UnknownGraph, never panic.
```

Input graph construction must match existing Whittle:

```text
arity input nodes at ids 0..arity-1
one output node pointing at input_index
capacity >= node_count
```

## Graph Generator

`gz-engine-whittle` owns Whittle-specific training graph generation. This is not
part of the generic `GraphEngine` trait.

Purpose:

```text
generate random-walked arity-n Whittle graphs like whittlezero's
sample_training_circuit()
```

API shape:

```rust
impl WhittleGraphGeneratorConfig {
    pub fn validate(self) -> Result<Self, WhittleGeneratorConfigError>;
}

impl WhittleGraphGenerator {
    pub fn from_seed(config: WhittleGraphGeneratorConfig, seed: u64) -> Self;

    pub fn sample_into(
        &mut self,
        engine: &mut WhittleEngine,
    ) -> EngineResult<GeneratedWhittleGraph>;
}
```

Default config should match current Whittle training defaults:

```text
arity = 6
capacity = 256
exception_terms_min = 5
exception_terms_max = 7
prewalk_steps_min = 4
prewalk_steps_max = 64
```

The generator has two stages.

### Stage 1: Truth-Table Exception Seed

This matches `truth_table_seed()` from `whittlezero/engine/generate.py`.

Algorithm:

```text
bit_count = 1 << arity
max_terms = min(exception_terms_max, bit_count / 2)
min_terms = min(exception_terms_min, max_terms)
term_count = uniform_int_inclusive(min_terms, max_terms)
selected_assignments = sample_without_replacement(0..bit_count, term_count)
exceptions_are_true = random_bool()

start with arity input nodes:
  op[i] = Input
  arg0[i] = i
  arg1[i] = NO_NODE

for each selected assignment:
  build one conjunction term
  for each variable:
    literal = input var when assignment bit is 1
    literal = cached Not(input var) when assignment bit is 0
    term = And(term, literal)
  dnf = Or(dnf, term)

if exceptions_are_true:
  root = dnf
else:
  root = Not(dnf)

append Output(root)
compact graph
```

Capacity rule:

```text
if the seed graph exceeds capacity, return EngineError::Internal with a compact
message; do not panic
```

`arity` must be small enough that `1 << arity` fits in the chosen integer type.
The first implementation may cap `arity <= 16`; raise that only when needed and
tested.

Performance rule:

```text
do not allocate the full truth table just to sample selected assignments;
term_count is small, so use a small set, Floyd sampling, or partial shuffle
over a compact scratch buffer sized to the chosen strategy
```

### Stage 2: Random Rewrite Prewalk

This matches `sample_training_circuit()` from `whittlezero/engine/generate.py`.

Algorithm:

```text
steps = uniform_int_inclusive(prewalk_steps_min, prewalk_steps_max)
last_rule = None

for step in 0..steps:
  blocked_rule = inverse_rule_id(last_rule) if last_rule exists else None
  candidates = enumerate_rewrites(graph, include_reverse_constant_folding=true)
  remove candidates whose rule_id == blocked_rule
  sample one candidate by category weight
  if no candidate remains:
    break
  graph = apply_rewrite(graph, candidate)
  last_rule = candidate.rule_id
```

Prewalk enumeration always enables reverse constant folding, regardless of the
runtime `WhittleEngineConfig.include_reverse_constant_folding` used for search.
This mirrors existing Whittle generation. It does not change the engine
instance's `action_set_hash()` because it is generator-internal setup, not
runtime candidate enumeration.

Category weights:

```text
0.5:
  commutativity
  associativity
  consensus

1.0:
  all other rule categories
```

Immediate inverse blocking:

```text
after applying rule R, the next prewalk step must not sample inverse_rule_id(R)
```

RNG rules:

```text
1. `from_seed(config, seed)` must be deterministic across platforms for a fixed
   GraphZero version.
2. Exact byte-for-byte parity with Python's `random.Random` is not required.
3. Golden tests should lock representative generated graph artifacts for fixed
   seeds once the Rust RNG is chosen.
4. Do not use thread-local or global RNG state.
```

Output rules:

```text
1. sample_into inserts both the seed graph and final walked graph into the
   engine arena.
2. Returned GeneratedWhittleGraph.graph is the final walked graph.
3. Generated graphs are compact and deduplicated by GraphHash.
4. Metadata records requested/applied prewalk steps and start/final costs.
```

Validation:

```text
arity > 0
capacity >= arity + 1
exception_terms_min <= exception_terms_max
prewalk_steps_min <= prewalk_steps_max
exception_terms_max > 0
```

## Candidate Representation

```rust
pub struct WhittleCandidate {
    pub graph: WhittleGraphId,
    pub graph_hash: GraphHash,
    pub candidate_hash: CandidateHash,
    pub rule_id: u16,
    pub root: u32,
    pub match_len: u8,
    pub matched: [u32; 8],
}
```

Candidate key:

```text
rule_id
root
match_len
matched[0..match_len]
```

Candidate hash:

```text
hash("whittle-candidate-v1",
     graph_hash,
     action_set_hash,
     rule_id,
     root,
     match_len,
     matched[0..match_len])
```

Rules:

```text
1. CandidateHash must not depend on vector position.
2. CandidateInfo.kind = CandidateKindId(rule_id).
3. CandidateInfo.subjects = matched nodes as SubjectId.
4. CandidateInfo.display_name is debug text only.
5. CandidateInfo.metadata stores compact binary candidate fields, not JSON.
6. candidate_info must reject candidates whose graph handle does not match the
   supplied graph.
```

## STOP

`gz-engine-whittle` enumerates rewrite candidates only. It does not append a
STOP candidate.

Reason:

```text
STOP is search/episode control, not a Whittle rewrite transform.
The existing Whittle feature/eval path appends STOP outside rewrite
enumeration. Keep the same separation.
```

If a later search algorithm needs STOP as a first-class action, add it at the
search/eval layer or explicitly reopen the `GraphEngine` candidate contract.

## Action Set

`WhittleEngineConfig.include_reverse_constant_folding` is part of action-set
semantics and must affect `action_set_hash()`.

Default:

```text
include_reverse_constant_folding = false
```

This matches existing runtime rewrite enumeration. Training prewalk generation
may opt into reverse constant folding, but that should be a distinct engine
instance/config if exposed through `GraphEngine`.

`CandidateOptions` behavior:

```text
deterministic_order = true:
  emit candidates in existing Whittle native body-order semantics

max_candidates = Some(n):
  truncate after deterministic enumeration

deterministic_order = false:
  may use faster internal order later, but first implementation should still
  return deterministic order until a benchmark proves the alternative matters
```

Changing `CandidateOptions` must not change `action_set_hash()`.

## GraphEngine Implementation

```rust
impl GraphEngine for WhittleEngine {
    type Graph = WhittleGraphId;
    type Candidate = WhittleCandidateId;

    fn engine_id(&self) -> EngineId;
    fn engine_version(&self) -> EngineVersion;
    fn action_set_hash(&self) -> ActionSetHash;
    fn root(&self) -> WhittleGraphId;
    fn hash(&self, graph: WhittleGraphId) -> EngineResult<GraphHash>;
    fn candidates(
        &mut self,
        graph: WhittleGraphId,
        options: CandidateOptions,
        out: &mut Vec<WhittleCandidateId>,
    ) -> EngineResult<()>;
    fn candidate_info(
        &self,
        graph: WhittleGraphId,
        candidate: WhittleCandidateId,
    ) -> EngineResult<CandidateInfo>;
    fn apply(
        &mut self,
        graph: WhittleGraphId,
        candidate: WhittleCandidateId,
    ) -> EngineResult<ApplyResult<WhittleGraphId, WhittleCandidateId>>;
    fn measure(
        &mut self,
        graph: WhittleGraphId,
        options: MeasureOptions,
    ) -> EngineResult<MeasureResult<WhittleGraphId>>;
    fn export_graph(&self, graph: WhittleGraphId) -> EngineResult<GraphArtifact>;
}
```

Implementation rules:

```text
candidates() must call out.clear() before writing.
hash() returns the stored graph hash; it does not recompute on every call.
apply() must verify candidate.graph_hash == hash(graph), otherwise return
EngineError::StaleCandidate.
apply() returns compact graph handles.
changed = before_hash != after_hash.
rejected = None for successful Whittle rewrites.
invalid handles return EngineError::UnknownGraph/UnknownCandidate.
```

`export_graph()`:

```text
format = GraphArtifactFormat::Binary
bytes = canonical WAV1 compact graph bytes
```

## Measurement

Whittle measurement is deterministic structural cost, not runtime latency.

Phase-one mode:

```text
WhittleMeasureMode::NegativeCost
```

For graph `g`:

```text
cost = compact node count
scalar_reward = -cost as f32
latency = None
measured = true
valid = true
metadata = compact binary fields:
  version byte
  cost u32 little-endian
  arity u16 little-endian
  capacity u16 little-endian
```

Reason: GraphZero should maximize scalar reward, and lower Whittle cost is
better. Higher-level PTP/sign/graded targets can still be computed later from
measured costs or trajectory comparisons; the engine-level measurement remains
simple and deterministic.

`MeasureOptions`:

```text
samples must be accepted but ignored for deterministic cost measurement
timeout_ms should be accepted but no-op unless future measurement gets expensive
deterministic must be true or false with the same result
config_hash must match the engine's measure config hash if the adapter chooses
to enforce exact mode identity
```

Do not put raw graph bytes in `MeasureMetadata`.

## Hashing

The adapter computes ids, not `gz-engine`.

Recommended derivations:

```text
EngineId:
  first 16 bytes of hash("gz-engine-whittle")

EngineVersion:
  first 16 bytes of hash("whittle-rules-v1", rule table version,
                         canonical format version, measurement version)

ActionSetHash:
  hash("whittle-action-set-v1", engine_version,
       include_reverse_constant_folding)

GraphHash:
  hash("whittle-graph-v1", engine_id, engine_version, canonical_wav1_bytes)

MeasureConfigHash:
  computed by config builder from WhittleMeasureMode
```

Use domain prefixes for every hash. Never hash ambiguous concatenations without
length or fixed-size fields.

## Caches

Initial caches:

```text
graph hash -> WhittleGraphId
graph id -> Vec<WhittleCandidateId> when cache_candidates is true
(graph hash, candidate hash) -> WhittleGraphId when cache_transitions is true
```

Rules:

```text
1. Cache keys include action_set_hash where candidate semantics matter.
2. Cache hit paths must not allocate graph bodies.
3. Cache misses may allocate once into arena.
4. Caches are engine-instance-local.
5. No global mutable cache.
```

Eviction can be deferred for the first implementation. Add bounded caches only
after benchmarks show unbounded arenas/caches are a problem for long smoke runs.

## BatchGraphEngine

First implementation may use `BatchGraphEngine` default ordered loops.

Override only when measured:

```text
candidates_batch can parallelize enumeration across graph handles
apply_batch can reuse transition cache and compact graph insertion
measure_batch can vectorize cost lookup
```

Any override must preserve `gz-engine` batch contract:

```text
output length equals input length
output order equals input order
one failed row does not poison other rows
observable result equals ordered single-call result
```

## Error Mapping

```text
bad graph handle             -> EngineError::UnknownGraph
bad candidate handle         -> EngineError::UnknownCandidate
candidate from another graph -> EngineError::StaleCandidate
bad artifact bytes           -> EngineError::Internal
unsupported rule in apply    -> EngineError::Internal
capacity exceeded            -> EngineError::Internal or RewriteRejection
```

Existing Whittle enumeration filters candidates whose replacement would exceed
capacity, so capacity exceeded during `apply()` indicates stale/corrupt
candidate data or a bug.

## Contract Fixture

`gz-engine-whittle` must provide:

```rust
pub struct WhittleContractFixture;

impl EngineContractFixture for WhittleContractFixture {
    type Engine = WhittleEngine;
}
```

Fixture requirements:

```text
root graph:
  simple x0 AND x0 graph, compacted, capacity >= 16

known_path:
  first candidate should include AndIdempotent and reduce cost

unknown_graph:
  invalid WhittleGraphId outside arena

unknown_candidate:
  invalid WhittleCandidateId outside arena
```

The contract test must run through `gz_engine::run_engine_contract`.

## Golden Tests

Required golden tests:

```text
input graph WAV1 artifact matches legacy format
truth-table exception seed builds a compact arity-n graph under capacity
fixed generator seeds produce stable WAV1 artifacts
prewalk uses reverse constant folding candidates internally
prewalk blocks immediate inverse rules
simple x0 AND x0 enumeration includes AndIdempotent
candidate order matches captured legacy native order for at least 3 fixtures
apply AndIdempotent preserves hash-equivalent truth fixture and reduces cost
reverse constant folding disabled by default
reverse constant folding enabled by config changes ActionSetHash
export_graph bytes roundtrip through adapter import
measure returns scalar_reward = -cost
candidate_info graph_hash/action_set_hash match engine values
```

Optional parity tests may compare against `/home/ubuntu/whittlezero` native
extension when available. They must be opt-in or skipped when the old repo is
not present. Runtime code must not depend on that repo.

## CLI Expectations

Once `gz-cli` exists, these should work:

```bash
graphzero smoke-engine --engine whittle
graphzero probe-actions --engine whittle --root simple
graphzero apply-path --engine whittle --path "AndIdempotent"
graphzero measure --engine whittle --root simple
```

The CLI prints graph hashes, candidate hashes, display names, cost, and
export-artifact paths. It must not print or parse engine-local handles as
durable ids.

## Open Questions

1. Should phase one implement the rewrite rules manually in Rust, or generate
   them from a small table to reduce drift from the C++ reference?
2. Should `WhittleMeasureMode::NegativeCost` be the only measure mode, or should
   the adapter also expose bounded `1 / (1 + cost)` for direct scalar reward
   experiments?
3. Should import/resolve APIs live in this crate before `gz-replay` exists, or
   wait until replay needs deterministic reconstruction?
