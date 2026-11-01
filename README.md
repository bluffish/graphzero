# GraphZero

GraphZero is a performance-first Rust/Python system for asynchronous graph
search, self-play, replay, measurement, and training. Search and orchestration
are generic over the `GraphEngine` boundary; Whittle is the current concrete
engine.

## Quick start

GraphZero requires a Rust toolchain with Rust 2024 edition support. The Python
components require Python 3.12 or newer and use
[uv](https://docs.astral.sh/uv/).

Build the CLI:

```bash
cargo build --release --bin graphzero
```

Run a short self-play job with the built-in stub evaluator:

```bash
cargo run --release --bin graphzero -- selfplay \
  --replay-dir ./replay \
  --episodes 2 \
  --lanes 1 \
  --workers-per-lane 1 \
  --evaluator stub
```

The `graphzero` binary also provides `replay-init`, `distill-generate`, and
`replay-serve` commands.

## Repository layout

- `crates/gz-engine`: generic engine traits and shared types
- `crates/gz-engine-whittle`: Whittle engine adapter
- `crates/gz-search`: greedy, beam, PUCT, and Gumbel search
- `crates/gz-orchestrator`: asynchronous search/evaluation/measurement lanes
- `crates/gz-replay`: RocksDB-backed replay storage
- `crates/gz-eval*`: evaluator interfaces and process service
- `crates/gz-measure*`: local and distributed measurement
- `crates/gz-cli`: the `graphzero` binary
- `python/gz`: model, evaluator, replay, checkpoint, and trainer code
- `configs`: self-play, training, and distillation configurations
- `specs`: protocol specifications

## Verification

```bash
cargo fmt --all --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
PYTHONPATH=.:python uv run --project python pytest -q python/tests
```

See
[`specs/distributed-measurement-protocol.md`](specs/distributed-measurement-protocol.md)
for the distributed measurement protocol.
