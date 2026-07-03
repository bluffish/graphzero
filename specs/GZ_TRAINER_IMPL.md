# Trainer Implementation Spec (Trainer Work Order 3)

Status: implementation work order

Purpose: implement `python/gz/trainer` and the run supervisor per
GZ_TRAINER.md — the concurrent loop: unbounded selfplay with an in-process
sample service on GPU 0, continuous training on GPU 1, checkpoints
hot-swapping mid-run. After this work order, one command runs the whole
learning loop unattended and the first learning curve exists.

EXECUTION ORDER: after trainer work orders 1 (Rust prerequisites) and 2
(evaluator hot-swap).

Authority: `GZ_TRAINER.md` (losses, config, supervisor, metrics,
acceptance), `GZ_PYTHON.md` (layout, layering), `GZ_TRAINING_DATA_IMPL.md`
+ `GZ_EVAL_PROTOCOL.md` (sample protocol). Contract wins; report
conflicts.

Read before starting:

```text
specs/GZ_TRAINER.md                    (the contract; keep it open)
python/gz/codec/                       (GZFB parsing; GZFT parsing is
                                        stage 1 here)
python/gz/checkpoints/                 (publish path)
python/gz/model/exphormer.py           (build, ArchConfig, BatchStager)
crates/gz-features/src/ (GZFT layout)  (the bytes being decoded)
crates/gz-cli/src/serve.rs             (the protocol being spoken)
```

## Hard Constraints

```text
Every stage ends with python3 -m pytest python/tests and cargo test --all
green. Commit per stage; stage 0 commits any dirty tree.
GZ_PYTHON.md layering holds: trainer imports proto/codec/model/
checkpoints/common only; torch imports stay lazy; nothing imports
trainer but its __main__. The layering test is extended, not weakened.
Rust untouched in this work order.
The training data path reuses gz.codec views + pinned staging; no
per-step Python loops over rows; the only per-step allocations are what
torch forces.
Fail-fast: the supervisor never restarts children; any child exit or
trainer exception aborts the run with the culprit named.
No wall-clock in anything that affects results; time is allowed in
polling, metrics timestamps, and the supervisor.
```

## Stage 0: Commit

Commit any dirty tree.

## Stage 1: GZFT Parsing In gz/codec

`gz/codec/targets.py`: `TargetsView.parse(buf)` mirroring the Rust GZFT
layout (magic, encoding version, capacity, row_count, max_actions, then
policy [B, A] f32 / value [B] f32 / value_valid [B] u8 / reward [B] f32,
4-byte aligned) — zero-copy numpy views, header validation, same style as
`BatchView`. Tests: hand-built bytes with literals; zero-copy mutation
visibility; header rejection cases; and a fixture test against bytes
produced by the Rust encoder (extend `gen_python_fixtures` with a
committed GZFT fixture).

## Stage 2: Sampler

`gz/trainer/sampler.py`:

```text
connects to the sample socket; validates SHELLO_ACK protocol/encoding
versions; decodes the FeatureSchemaConfig from the ack (the codec from
GZ_MODEL_TORCH_IMPL stage 1) and exposes it plus the schema hash and
max_batch
sample(batch, window, seed) -> (BatchView, TargetsView, produced_rows)
startup wait: poll SHELLO until produced_rows >= min_startup_rows with
backoff (0.5s), bounded by startup_timeout
reconnect with backoff on socket loss, abort after reconnect_limit
consecutive failures
per-step seed = blake2b(run_seed, step) folded to u64 — deterministic,
documented
```

Tests against the real Rust serve (same pattern as existing cross-language
tests): handshake fields, a sampled batch parses through both views,
deterministic identical responses for identical requests, startup wait
against an initially-empty store that fills.

## Stage 3: Data And Loss Step

```text
gz/trainer/data.py   BatchView+TargetsView -> cuda tensors on the trainer
                     device via a BatchStager (manifest-free: built from
                     the sampler's schema config) plus target staging for
                     policy/value/valid/reward
gz/trainer/loop.py   exactly GZ_TRAINER.md's step:
                     Lp = soft-target CE with padded slots masked to -inf,
                     averaged over rows; Lv = BCE(sigmoid(2*v_raw),
                     (label+1)/2) over valid rows, 0 when none;
                     L = Lp + value_weight * Lv
                     AdamW, grad clip, cosine schedule over total_steps
                     with warmup, weight EMA update per step
                     returns a StepMetrics record (losses, grad_norm, lr,
                     value accuracy over valid, fraction valid, label
                     mean)
```

Loss unit tests with hand-built tiny batches: CE against a hand-computed
literal; BCE literal including the tie (0.5) case; zero-valid batch gives
Lv = 0 and finite gradients; padded action slots provably do not
contribute (perturb a padded logit target, loss unchanged).

## Stage 4: Publish And EMA

`gz/trainer/publish.py`: builds the manifest from the sampler's schema
config + ArchConfig + training_step, publishes the EMA weights through
gz/checkpoints. EMA: a parameter-for-parameter shadow copy updated per
step (`decay * ema + (1 - decay) * live`, buffers copied), initialized
from the live weights at step 0. Tests: EMA arithmetic literals; publish
round-trips through DirectorySource; model_version changes across
publishes with changed weights.

## Stage 5: Supervisor

`gz/trainer/driver.py` + `__main__.py` (`python -m gz.trainer --config
run.toml`), config exactly per GZ_TRAINER.md's `[trainer]/[selfplay]/
[paths]` tables (tomllib):

```text
bootstrap:
  1. run graphzero selfplay --evaluator stub --episodes B0 (subprocess,
     blocking) into the replay dir
  2. run graphzero replay-serve (subprocess); sampler reads the schema;
     build model on the trainer device; publish version 0; stop serve
steady state:
  3. spawn graphzero selfplay --evaluator torch --checkpoint-dir ...
     --eval-device <cfg> --eval-poll-interval <cfg> --episodes 0
     --serve-socket <path> --reference self-average
     --replay-backlog <cfg>
  4. trainer loop to total_steps, publishing every publish_interval
  5. publish final checkpoint; SIGKILL selfplay; exit 0
supervision: poll the selfplay child every few seconds from the trainer
loop; child exited -> abort naming it and its status. Trainer exception
-> kill child, re-raise.
test knob: [trainer] step_sleep (default 0) sleeps that long per step —
exists solely so the backpressure acceptance test can slow the trainer.
metrics: JSONL per GZ_TRAINER.md, one file per run dir, flushed per line.
```

## Stage 6: The End-To-End Acceptance Run

One integration test (torch + both GPUs required; fail loudly, never
skip) with tiny settings — total_steps ~40, publish_interval 10, small
selfplay config, poll_interval small — asserting GZ_TRAINER.md's
acceptance list mechanically:

```text
run completes; metrics JSONL nonempty and monotone in step
concurrency: selfplay stderr timestamps interleave with step timestamps
hot swap happened: the selfplay child's stderr contains
event=checkpoint_swapped at least once and event=checkpoint_rejected
never (GZFT targets do not carry per-row model_version, so the swap is
asserted from the evaluator's stderr events; per-row versions live only
in the store records)
backpressure: rerun with step_sleep large and a small backlog cap; the
store's produced_rows stays near the cap instead of growing unboundedly
kill safety: SIGKILL the selfplay child mid-run in a dedicated test; the
supervisor aborts naming it; the store reopens intact
```

Plus one manual longer smoke documented in the spec footer (30+ minutes,
default config) whose JSONL is eyeballed for moving losses — the actual
first learning curve is an operator activity, not CI.

## Stage 7: Docs And Final Verification

```text
GZ_TRAINER.md marked implemented-by; CODEBASE_OUTLINE python/ note;
AGENTS.md lists this spec.
```

```bash
python3 -m pytest python/tests
cargo test --all
python3 -m gz.trainer --config configs/train-whittle.toml   # committed
                                                            # example config
```

Acceptance checklist: GZ_TRAINER.md's list, verbatim, plus:

```text
layering test extended to trainer and green
GZFT fixture committed and cross-checked against Rust bytes
per-step sampling deterministic given (run_seed, step)
example config committed under configs/
```

## Out Of Scope

```text
everything in GZ_TRAINER.md's Non-Goals list
selfplay periodic stats (label-mean proxy suffices)
benchmark-campaign tuning of the default config
```
