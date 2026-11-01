# GraphZero

AlphaZero-style search for graph rewriting. A model learns which rewrites make
a graph faster, and self-play generates the training data.

The search code works with any engine that implements `GraphEngine`. Two are
included: Whittle (boolean circuits) and a compute-graph compiler.

## Run

Needs Rust (2024 edition), Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cargo run --release --bin graphzero -- selfplay \
  --replay-dir ./replay --episodes 2 --lanes 1 --workers-per-lane 1 \
  --evaluator stub
```

## Test

```bash
cargo test --workspace
PYTHONPATH=.:python uv run --project python pytest -q python/tests
```
