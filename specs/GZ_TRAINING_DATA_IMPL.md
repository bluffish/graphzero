# Training Data Path Implementation Spec

Status: implementation work order

Purpose: close the Rust-side gaps between "selfplay writes replay rows" and
"a Python trainer can consume batches". Three pieces plus one trivial one:
feature payloads stored in replay rows (schema v3), a training-batch
targets encoding in gz-features, a standalone `graphzero replay-serve`
sample server, and an args pass-through on the evaluator spawn config.
After this work order the bare-minimum loop is unblocked: the next work
order is pure Python (torch model, checkpoints, trainer).

The bare-minimum training loop this enables is PHASE-ALTERNATED, not
concurrent: selfplay run -> store closes -> replay-serve + trainer ->
checkpoint published -> next selfplay run. No ratio-controlled concurrent
loop, no hot swap, no in-orchestrator service hosting. The architecture
already supports upgrading to concurrent later; do not build any of it now.

Authority: `GZ_REPLAY.md`, `GZ_FEATURES.md`, `GZ_EVAL_PROTOCOL.md` own the
contracts; each gains an amendment in this work order (listed per stage,
made explicitly, never silently). Contract wins on conflict; report
conflicts.

Read before starting:

```text
specs/GZ_REPLAY.md                        (schema being bumped to v3)
specs/GZ_FEATURES.md                      (row codec + targets additions)
crates/gz-replay/src/records.rs, store.rs
crates/gz-orchestrator/src/project.rs, lanes.rs, pool.rs
crates/gz-features/src/collator.rs
crates/gz-search/src/gumbel.rs            (episode-loop budget formulas)
crates/gz-cli/src/                        (new subcommand home)
```

## The Design Decision Being Implemented

Replay rows store portable refs, not graph bodies — so nothing in replay
can currently be fed to a network. Decision (amends GZ_REPLAY.md's "no
graph bodies" rule): rows gain an **encoded FeatureRow** — a portable
model input, not an engine body; engine handles and artifacts remain
forbidden. Consequence, stated in the spec amendment: replay data is bound
to one FeatureSchemaHash; changing the feature schema means regenerating
data. Accepted for this stage of the project. The artifact-based,
schema-flexible alternative stays deferred.

## Hard Constraints

```text
Stage order below; every stage ends with cargo fmt / cargo test --all /
cargo clippy --all-targets --all-features -- -D warnings and
python3 -m pytest python/tests. Commit per stage; stage 0 commits any
dirty tree.
gz-search may gain exactly one public method (stage 3); nothing else in
it changes and the gumbel goldens are untouched.
gz-eval, gz-engine, gz-engine-whittle, python/ untouched.
New dependency edges: gz-replay -> gz-features (light: std + gz-engine +
blake3). Amend GZ_REPLAY.md's dependency contract.
No serde in gz-features (its codecs stay hand-rolled); replay stores the
encoded row as opaque bytes.
Bounded everything; buffers reused in the serve loop; fail-fast error
policy throughout.
No wall-clock; sampling and serving stay deterministic given (store,
request).
```

## Stage 0: Commit

Commit any dirty tree.

## Stage 1: gz-features — Row Codec And Training Targets

Single-row codec (magic `GZFR`), hand-rolled like the batch codec:

```rust
pub fn encode_feature_row(row: &FeatureRow, schema: &FeatureSchema, out: &mut Vec<u8>);
pub fn decode_feature_row(bytes: &[u8]) -> FeatureResult<FeatureRow>;
/// Header-only check: magic, encoding version, schema hash equality.
pub fn validate_feature_row_header(bytes: &[u8], expected: &FeatureSchemaHash) -> FeatureResult<()>;
```

Layout: magic, `ENCODING_VERSION`, `FeatureSchemaHash` (32B), then
length-prefixed fields in `FeatureRow` declaration order, little-endian,
matching the existing per-field encodings. The row is validated against
the schema during encode (reuse `FeatureRow::validate`); decode validates
structurally (lengths, finiteness) without a schema.

Training targets block (magic `GZFT`):

```rust
pub struct RowTargets {
    pub policy: Vec<f32>,          // length == the row's action count
    pub value: Option<f32>,        // -1 | 0 | +1 when Some
    pub reward: f32,
}

pub fn encode_training_targets(
    targets: &[RowTargets],
    capacity: usize,
    max_actions: usize,
    out: &mut Vec<u8>,
) -> FeatureResult<()>;

pub struct TrainingTargetsView { /* parse() for tests + diagnostics */ }
```

Layout: magic, encoding version, capacity u32, row_count u32,
max_actions u32, then sections in order, 4-byte aligned like GZFB:
`policy [B, A] f32` (zero-padded), `value [B] f32` (0.0 when invalid),
`value_valid [B] u8`, `reward [B] f32`. Rows beyond row_count are all
padding.

Tests: roundtrip + golden byte literals for both codecs; header rejection
(magic/version/hash) for GZFR; policy-length-vs-A validation; the targets
view exposes exactly what was encoded. Amend GZ_FEATURES.md with both
layouts (it owns all wire formats).

## Stage 2: gz-replay — Schema v3

Records:

```rust
pub struct ReplayRow {
    // ... existing fields ...
    pub feature_row: Option<Vec<u8>>,     // GZFR bytes
}
```

Store:

```rust
impl ReplayStore {
    /// Idempotent; first call persists, later calls must match.
    pub fn ensure_feature_schema(&self, config: &FeatureSchemaConfig) -> ReplayResult<()>;
    pub fn feature_schema(&self) -> ReplayResult<Option<FeatureSchemaConfig>>;
}
```

Rules:

```text
SCHEMA_VERSION -> 3 (a v2 store fails to open; existing mechanism).
The FeatureSchemaConfig is persisted once in the meta CF (postcard of a
replay-owned mirror struct or the config fields directly — gz-features
has no serde; a small mirror in gz-replay with From impls is fine).
ensure_feature_schema with a different config than stored ->
InvalidRecord.
Admission: rows with feature_row require the store to have a schema
(NotConfigured-style InvalidRecord otherwise) and the bytes must pass
gz_features::validate_feature_row_header against the stored config's
hash. Featureless rows remain legal (non-featurized selfplay).
All rows within one episode must agree: all featured or all featureless.
```

Amend GZ_REPLAY.md: dependency contract (gz-features allowed), the records
section, the "no graph bodies" rule (portable feature-row bytes allowed;
handles/artifacts still forbidden; schema-bound data consequence stated),
schema version note.

Tests: v3 roundtrip with features; header-mismatch rejection; mixed
featured/featureless episode rejected; ensure_feature_schema idempotence
and mismatch; feature_schema survives reopen; v2 store fails to open.

## Stage 3: Projection Produces Feature Payloads

gz-search gains one public method (goldens unaffected — pure accessor):

```rust
impl GumbelMcts {
    /// The budget values the episode loop passes to root search at `step`;
    /// projection-time re-extraction must reproduce eval-time positions.
    pub fn root_budget(&self, step: usize) -> (f32, f32);  // (fraction, per-step)
}
```

Unit test in gz-search: the values match what the scripted evaluator
observes in `EvalRequest.position` during an episode run.

Orchestrator:

```text
project_episode gains feature_rows: Option<&[Vec<u8>]> (one GZFR blob per
step, same order); Some requires len == steps.len(); rows get
feature_row = Some(bytes.clone()) — or restructure to move; do not
double-clone.
The featurized replay lane builds the blobs after each episode completes,
before projection: for each step i, enumerate candidates on
step.before (same CandidateOptions as search — deterministic enumeration
reproduces the action list), build PositionFeatures { root_step: i,
leaf_depth: 0, root_budget(i) }, extract, assert the extracted action
count equals step.legal_actions.len() (internal error otherwise — this
is the alignment guarantee between policy_target and the stored feature
row), encode.
The extractor's state cache makes the graph side of re-extraction cheap;
re-enumeration cost is Whittle-fine and flagged with a comment as a
compiler-regime revisit (capture-at-eval-time is the future
optimization).
The non-featurized replay lane passes None (those stores cannot train a
model; that is fine and documented).
run_featurized_with_replay calls store.ensure_feature_schema(collator
schema config) before spawning threads.
```

Tests: featurized replay integration test now asserts sampled rows carry
feature bytes that decode to rows whose action counts match
legal_actions, and that the store's feature_schema equals the extractor's;
non-featurized replay still appends featureless rows; length-mismatch and
alignment failures are exercised via a wrapping test extractor.

## Stage 4: replay-serve

Protocol (amend GZ_EVAL_PROTOCOL.md with a "Sample Protocol" chapter — it
is the language-neutral wire doc and the Python trainer client implements
this next):

```text
same framing as the eval protocol (u32 LE length, u8 type + body)
SAMPLE_PROTOCOL_VERSION = 1

1 SHELLO       client -> server: protocol_version u32, encoding_version u32
2 SHELLO_ACK   server -> client: protocol_version u32,
               feature_schema_hash 32B, max_batch u32, produced_rows u64
3 SAMPLE       client -> server: batch u32 (<= max_batch), window u64,
               seed u64
4 SAMPLE_RESULT server -> client: gzfb_len u32, GZFB bytes, GZFT bytes
5 ERROR        server -> client: code u32, msg_len u16, utf8 <= 512B;
               close after sending
error codes: 1 protocol, 2 encoding, 3 empty store, 4 bad request,
5 missing features
one client at a time; sequential requests; responses in request order
```

Implementation lives in gz-cli (`src/serve.rs`; gz-replay's contract keeps
it out of the storage crate, and the binary already composes everything):

```text
graphzero replay-serve --replay-dir PATH --socket PATH --max-batch B
open store; feature_schema() must be Some (error out otherwise, message
says the store was not produced by featurized selfplay)
build FeatureSchema from the stored config; one FeatureCollator with
capacity B; reusable scratch: Vec<FeatureRow>, Vec<RowTargets>,
Vec<u8> x2
per SAMPLE: store.sample_rows -> decode each feature_row (a row without
features is error 5) -> RowTargets from the row's policy_target /
value_target / reward_target -> collate_into + encode_training_targets ->
one SAMPLE_RESULT
serve one client, then accept the next (loop; the trainer reconnects
between phases); SIGINT/termination handling is out of scope — the
process is killed by its operator
determinism: identical (store, SAMPLE fields) -> identical response bytes
```

Tests (gz-cli integration): build a store via a small featurized selfplay
run, start serve() on a thread against a temp socket, drive it with a
Rust test client using gz-eval-service framing: handshake fields correct;
a sampled batch round-trips (GZFB parses, row_count == batch, GZFT policy
rows match the sampled rows' policy targets; value_valid mirrors label
presence); batch > max_batch -> error 4; featureless store -> startup
error; empty store -> error 3; determinism across two identical SAMPLEs.

## Stage 5: Evaluator Spawn Pass-Through

```rust
pub struct EvaluatorProcessConfig {
    // ... existing ...
    pub extra_args: Vec<String>,   // appended after --socket PATH
}
```

Default empty; spawn appends them verbatim. One unit test (scripted args
visible in the failure message of a bad-binary spawn, or via a tiny echo
fixture — implementer's choice, keep it cheap). This is what the next
work order uses for `--backend torch --checkpoint-dir ...`.

## Stage 6: Docs And Final Verification

```text
GZ_REPLAY.md, GZ_FEATURES.md, GZ_EVAL_PROTOCOL.md amendments per stages
1-4 (verify all landed)
CODEBASE_OUTLINE.md: gz-cli section gains replay-serve
AGENTS.md: this spec listed
```

```bash
cargo fmt --all -- --check
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings
python3 -m pytest python/tests
target/debug/graphzero selfplay --replay-dir /tmp/gz-train-smoke \
  --episodes 8 --evaluator stub
target/debug/graphzero replay-serve --replay-dir /tmp/gz-train-smoke \
  --socket /tmp/gz-serve.sock --max-batch 32 &   # then sample once with
                                                 # the test client or a
                                                 # 5-line python script
```

Acceptance checklist:

```text
a featurized selfplay store round-trips through replay-serve into GZFB +
GZFT bytes a numpy client could consume (the Python trainer client is the
next work order; the Rust test client proves the bytes)
feature-row/policy-target alignment is asserted at projection, not
assumed
schema v3 gates old stores; feature schema config persists and is
enforced store-wide
non-featurized selfplay still works end to end (featureless rows legal)
gz-search change is the one accessor + its test; goldens untouched
serve loop reuses buffers; responses deterministic
all three contract amendments are explicit in the spec documents
```

## Out Of Scope

```text
the Python trainer, torch model, checkpoints (next work order)
concurrent selfplay + training; ratio-controlled serving; hot swap
capture-at-eval-time feature payloads (compiler-regime optimization)
artifact storage / schema-flexible refeaturization
retention, sharding, or multi-consumer sampling policy changes
opponent trajectory anything
```
