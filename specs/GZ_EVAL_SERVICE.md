# gz-eval-service Spec

Status: draft

Purpose: define the process-backed evaluator — the wire protocol between the
Rust orchestrator and a Python evaluator process, the Python evaluator
serving a deterministic stub model, the featurized eval path inside the
orchestrator, and the synthetic load generator. After this work order, the
entire neural-eval transport is built and verified end to end with zero ML:
`graphzero selfplay` can route every leaf eval through a Python process and
produce episodes identical to an in-process run.

## Architecture Overview

```text
lane thread                    batcher thread                 Python process
───────────                    ──────────────                 ──────────────
worker parks on Eval
  extractor.extract()          collect FeaturizedEvalJobs
  (engine in hand)      ──>    collator.collate_into()
FeatureRow crosses             GZFB batch bytes        ──UDS──> frombuffer/view
(portable, no handles)                                          stub model
                               decode_outputs()        <─UDS──  GZFO output bytes
                               EvalOutput per row               + ModelVersion
                               resume (worker, token)
```

The decisions this spec fixes (review before implementation):

```text
1. New crate gz-eval-service owns the protocol: framing, handshake, the
   client, process spawn/lifecycle, and the stub model's Rust reference
   implementation. It depends on gz-features and gz-engine only. The
   orchestrator gains default deps on gz-features and gz-eval-service
   (amending GZ_ORCHESTRATOR.md, same precedent as gz-replay).

2. The evaluator is a child process of the orchestrator. Spawned, health-
   checked via PING, killed on drop. Fail-fast on handshake mismatch or
   crash; restart policy is deferred.

3. The payload IS the gz-features encoding. EVAL frames carry GZFB batch
   bytes verbatim; results carry GZFO bytes verbatim. The service layer
   adds only framing, batch ids, and the handshake. Rust and Python cannot
   disagree about tensor layout without FeatureSchemaHash failing the
   handshake.

4. The stub model is a bit-exact cross-language function over the encoded
   batch (integer arithmetic + power-of-two divisions only), implemented
   once in Rust and once in numpy. It powers the equality oracle: selfplay
   through the socket must equal selfplay through the in-process stub,
   field for field. This extends the project's oracle discipline across
   the language boundary and catches encoding/alignment/routing bugs with
   zero ML noise.

5. The featurized path is a new backend boundary in the orchestrator, not
   a replacement: run()/run_with_replay() over portable Evaluators stay
   untouched. A FeatureEvalBackend trait abstracts "bytes in, outputs out"
   so the in-process stub and the socket client are interchangeable.

6. Single in-flight batch per connection in v1. The protocol carries
   batch_id from day one so pipelining (2-3 in flight) is a client change
   later, not a protocol change. ModelVersion rides every EVAL_RESULT so
   checkpoint hot-swap also lands later without protocol change.

Deliberately deferred: the trainer, checkpoint loading/hot-swap, torch,
batcher pipelining, evaluator restart policy, shared-memory transport,
opponent trajectory registration (job 2).
```

## Role

`gz-eval-service` answers:

```text
How do feature batches reach an out-of-process model and how do outputs
come back, versioned and verified?
```

It owns:

```text
the wire protocol: frame format, types, handshake, error frames
the blocking client (connect, handshake, eval round trip, ping)
evaluator child-process spawn, readiness probe, and kill-on-drop
the stub model reference implementation (Rust)
the FeatureEvalBackend trait and its two v1 backends
the synthetic load generator
```

It does not own:

```text
feature schema or encoding (gz-features)
where extraction/collation run (gz-orchestrator)
the Python implementation's internals (python/evaluator, specced below)
models, torch, checkpoints, training
retry/restart policy (fail-fast v1)
```

## Dependency Contract

`gz-eval-service` (Rust):

```text
allowed: std, gz-engine, gz-eval (EvalOutput), gz-features
forbidden: tokio, torch/Python bindings, serde, gz-search, gz-replay,
gz-orchestrator, engine adapters
transport is std::os::unix::net; this crate is unix-only, which is
acceptable and documented
```

`python/evaluator`:

```text
allowed: Python stdlib, numpy
forbidden in this work order: torch, any framework, any pip-only dependency
(numpy and pytest are installed as system packages)
```

`gz-orchestrator` amendments: gz-features and gz-eval-service move to
default dependencies (update GZ_ORCHESTRATOR.md's dependency contract).

## Wire Protocol

Unix domain socket; the orchestrator owns the socket path (a temp dir per
run). Every frame:

```text
u32 LE body_length, then body: u8 frame_type + fields
body_length covers the type byte; sanity cap 256 MiB, larger is a protocol
error and closes the connection
integers LE; hashes/ids as raw bytes
```

Frame types:

```text
1 HELLO       client -> server, once, immediately after connect:
              protocol_version u32, encoding_version u32,
              feature_schema_hash 32B, batch_capacity u32,
              engine_id 16B, engine_version 16B, action_set_hash 32B
2 HELLO_ACK   server -> client: protocol_version u32, model_version 16B
3 EVAL        client -> server: batch_id u64, GZFB bytes
4 EVAL_RESULT server -> client: batch_id u64, model_version 16B, GZFO bytes
5 PING        client -> server: nonce u64
6 PONG        server -> client: nonce u64
7 ERROR       server -> client: code u32, msg_len u16, utf8 message
              (bounded 512 bytes); the server closes after sending
```

Rules:

```text
PROTOCOL_VERSION = 1, a constant in both implementations.
The server validates every HELLO field it can check (protocol version,
encoding version, schema hash, batch capacity against its own config) and
replies ERROR + close on any mismatch. Engine tags are recorded and echoed
into future checkpoint compatibility checks; the stub server accepts any.
Responses arrive in request order; batch_id must match the oldest
outstanding EVAL, else the client fails the connection.
GZFB row_count and header dims are validated server-side against the
handshake before evaluation.
Any malformed frame in either direction is fatal to the connection;
fail-fast, no resync.
```

## Stub Model

Defined over the *encoded* batch so both implementations consume identical
input. For row `i < row_count`, with `a = node_count[i]`,
`c = action_count[i]`, all arithmetic in u64 with wraparound:

```text
value[i]     = (((a * 2654435761 + c * 40503) % 4096) - 2048) / 2048.0
logits[i][j] = (((a + 31*j + 7*c) % 64) - 32) / 32.0     for j < c
logits[i][j] = 0.0                                        for c <= j < A
rows i >= row_count: all zeros
```

Integer-mod results are < 2^24 and divisions are by powers of two, so both
languages produce bit-identical f32. The stub's model version is the
constant 16 bytes `67 7a 2d 73 74 75 62 2d 76 31 00 00 00 00 00 00`
(ascii "gz-stub-v1" zero-padded), hardcoded identically on both sides.

The Rust reference implementation operates on `FeatureBatchView` and is
used by the in-process backend and by conformance-test expectations.

## Rust API

```rust
pub trait FeatureEvalBackend {
    fn eval(
        &mut self,
        batch_bytes: &[u8],
        action_counts: &[u32],
    ) -> ServiceResult<BackendOutputs>;
}

pub struct BackendOutputs {
    pub model_version: ModelVersion,
    pub rows: Vec<RowOutput>,          // gz-features RowOutput
}

pub struct StubBackend { /* collator-compatible; pure Rust */ }

pub struct ProcessBackend { /* owns the connection; single in-flight */ }

pub struct EvaluatorProcess { /* child + socket path; kill on drop */ }

impl EvaluatorProcess {
    pub fn spawn(config: EvaluatorProcessConfig) -> ServiceResult<Self>;
    pub fn connect(&self, hello: Hello) -> ServiceResult<ProcessBackend>;
}

pub struct EvaluatorProcessConfig {
    pub python: PathBuf,           // default "python3"
    pub module: String,            // default "evaluator"
    pub working_dir: PathBuf,      // default "python/"
    pub socket_path: PathBuf,
    pub ready_timeout: Duration,   // default 10s
}
```

Rules:

```text
spawn starts the child with --socket <path>, inherits stderr, then probes
readiness: retry connect + PING until ready_timeout, else kill + error.
connect performs the handshake and returns a ready backend.
Drop for EvaluatorProcess kills and reaps the child; no orphans.
ProcessBackend::eval writes one EVAL frame and blocks for its EVAL_RESULT;
ERROR frames or connection loss map to ServiceError::Backend and are
fail-fast for the caller.
ServiceError is small: Handshake, Protocol, Backend, Io(bounded message).
```

## Python Evaluator

```text
python/evaluator/
  __init__.py
  __main__.py     argparse: --socket PATH; binds, serves one client
  server.py       accept loop, frame codec, dispatch
  codec.py        GZFB parse via numpy frombuffer with computed offsets;
                  GZFO encode; header/section validation
  stub.py         the stub model, vectorized numpy, exact formulas above
  tests/
    fixtures/batch_v1.bin    checked-in golden batch (generated by a Rust
                             test helper, committed)
    test_codec.py            parses the fixture; asserts header fields and
                             spot values per section
    test_stub.py             stub outputs on the fixture match literals
                             computed from the formulas
```

Rules:

```text
stdlib + numpy only; single-threaded; one client; blocking IO.
codec.py computes section offsets from header dims exactly as
GZ_FEATURES.md specifies (4-byte section alignment included) and never
copies section data (views into one read buffer).
The server validates HELLO and every EVAL header, replying ERROR + close
on mismatch (codes: 1 protocol, 2 encoding, 3 schema, 4 capacity,
5 malformed).
No logging on the hot path; startup and errors go to stderr.
Run tests with python3 -m pytest python/evaluator/tests.
```

## Orchestrator Featurized Path

The existing portable-Evaluator paths (`run`, `run_with_replay`) are
untouched. New alongside them:

```text
pool: the Parked payload gains an optional FeatureRow. drive() accepts an
optional extractor (&mut dyn FeatureExtractor<E>); when present, the park
step extracts while graph and candidates are still in hand and stores the
row; extraction errors are fail-fast. action_count for output decoding is
request.actions.len(), retained alongside.

lane -> batcher message: FeaturizedEvalJob { lane, slot, token, row,
action_count } — portable, no engine generics (same structural rule as
EvalJob).

featurized batcher: reuses the existing size/deadline collection logic;
collator.collate_into -> backend.eval -> RowOutput per row ->
EvalOutput { model_version, policy_logits, value } -> replies routed by
(lane, slot, token). Single in-flight v1.

entry point:
  pub fn run_featurized<R, X, B>(
      self,
      root_sources: Vec<R>,
      context: GumbelEpisodeContext,
      extractors: Vec<X>,               // one per lane
      backend: B,
      replay: Option<ReplayRuntime<'_, P>>-shaped optional replay support
  ) -> EngineResult<ThreadedRun<...>>
  where X: FeatureExtractor<E> + Send, B: FeatureEvalBackend + Send;
signature details are implementation-shaped, but replay must compose (the
CLI wants featurized eval + replay in one run) rather than adding a third
and fourth method later.

batch_capacity = the batcher's max_batch; the collator and handshake use
the same value.
```

CLI: `graphzero selfplay --evaluator random|stub|process-stub`. `random`
is the existing default; `stub` uses StubBackend in-process; `process-stub`
spawns the Python evaluator (flags `--python`, `--socket-dir` optional).

## Test Strategy

gz-eval-service unit (no Python):

```text
frame codec roundtrip for every frame type; oversized/undersized frames
rejected
handshake against a tiny in-Rust test server: accept, and each mismatch
field -> Handshake error
stub reference: golden literals for hand-built batches; padded rows all
zero
StubBackend output shapes/truncation via decode_outputs
EvaluatorProcess: spawn failure (bad binary path) errors within timeout;
drop kills the child (no zombie: waitpid reaps)
```

Python (pytest): codec fixture parse + stub literals, as above.

Cross-language conformance (Rust integration test, requires python3+numpy;
fail loudly with an install message if spawn fails, do not skip silently):

```text
spawn the real evaluator; handshake succeeds; PING works
for several seeded synthetic batches: ProcessBackend output ==
StubBackend output, bit-identical
handshake with a corrupted schema hash is rejected with the schema error
code
```

Orchestrator integration:

```text
featurized selfplay with StubBackend on Whittle: completes, deterministic
across two identical runs
the oracle: featurized selfplay through the Python process == featurized
selfplay through StubBackend, episodes field-equal (this is the
acceptance test of the entire work order)
featurized + replay: rows land in the store as in the existing replay
integration tests
CLI smoke: --evaluator stub and --evaluator process-stub both run
```

## Implementation Plan

1. Prerequisite: gz-features implemented, reviewed, committed. Commit any
   dirty tree.
2. `crates/gz-eval-service`: frame codec, Hello types, protocol constants,
   stub reference implementation, StubBackend, FeatureEvalBackend. Unit
   tests including the in-Rust test server.
3. `python/evaluator`: codec, stub, server, __main__, pytest suite, and
   the Rust helper that (re)generates the committed fixture.
4. EvaluatorProcess spawn/readiness/drop-kill + ProcessBackend + the
   cross-language conformance tests.
5. Orchestrator featurized path: pool park hook, FeaturizedEvalJob,
   featurized batcher over FeatureEvalBackend, run_featurized with
   optional replay. StubBackend integration tests.
6. CLI --evaluator flag; the Python end-to-end equality test; load
   generator example (`examples/eval_load.rs`: seeded synthetic rows,
   N batches through a chosen backend, prints p50/p95 latency and
   rows/s).
7. Docs: GZ_ORCHESTRATOR.md dependency amendment; CODEBASE_OUTLINE
   python/evaluator note; AGENTS.md lists this spec. Full verification:
   fmt, test --all, clippy -D warnings, pytest, and a manual
   `graphzero selfplay --evaluator process-stub --episodes 8`.
```

Every stage compiles and passes `cargo test --all` before the next; gz-search,
gz-engine, gz-eval, gz-replay, and the goldens are untouched throughout.

## Deferred

```text
trainer, checkpoint manifest/loading/hot-swap (protocol already carries
ModelVersion)
batcher pipelining (protocol already carries batch_id)
evaluator restart/backoff policy
torch, the real Exphormer model, GPU placement
shared-memory transport
opponent trajectory registration and resolution (job 2)
multiple concurrent client connections
```
