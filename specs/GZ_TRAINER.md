# gz Trainer Spec (Concurrent)

Status: draft — minimal feature set, concurrent execution. Selfplay and
training run simultaneously on disjoint GPUs (this box: 2x RTX PRO 6000 —
evaluator on cuda:0, trainer on cuda:1). Phase alternation is not built;
the ratio gate and checkpoint hot-swap regulate the loop instead.

Purpose: define `python/gz/trainer` and the run supervisor that closes the
loop: one long-lived selfplay process (torch evaluator + in-process sample
service) and one long-lived trainer process, coupled only through the
sample socket and the checkpoint directory.

Implemented by `python/gz/trainer/` for the initial concurrent trainer
path: sample client, target parsing, tensor staging, loss step, EMA
checkpoint publishing, and supervisor CLI.

Assumes implemented: GZ_FEATURES_EXPANDER_IMPL, GZ_TRAINING_DATA_IMPL,
GZ_MODEL_TORCH_IMPL.

## Topology

```text
GPU 0                                   GPU 1
graphzero selfplay (long-lived)         python -m gz.trainer (long-lived)
  torch evaluator child (hot-swaps) <── checkpoints/  <── publish every K
  replay writer -> RocksDB store            steps (EMA weights)
  sample service thread ────socket────> sampler: seeded SAMPLE per step
  admission gated by produced-consumed
  backlog (the live ratio control)
```

Coupling is data only: the socket and the checkpoint dir. Either process
can be killed and the other throttles/starves gracefully; the supervisor
enforces fail-fast anyway.

## Decisions

```text
labels: sign labels against the SELF-AVERAGE reference (per-lane reward
EMA of the learner's own recent episodes; adaptive, non-saturating).
value loss: logistic on the sign labels. policy loss: soft-target CE
against stored completed-Q targets.
optimizer: AdamW + cosine/warmup + grad clip. weight EMA published
(decay 0.999) — distinct from the reward EMA reference.
concurrent from day one: selfplay runs unbounded; the trainer trains
continuously; checkpoints hot-swap into the evaluator mid-run; the replay
backpressure gate is the live producer/consumer regulator.
no gating/arena, no resume, fresh runs only.
```

## Rust Prerequisites (trainer work order stage 1)

```text
1. SelfAverageProvider: ReferenceProvider with one reward EMA per lane
   (decay default 0.99); first episode per lane seeds the EMA, reference
   None. ReplayReferenceKind::SelfAverage; GZ_REPLAY.md outcome rules
   updated. CLI --reference self-average [--reference-ema-decay D].
2. In-process sample service: graphzero selfplay --serve-socket PATH runs
   the existing replay-serve loop as a thread inside the selfplay process,
   sharing the &ReplayStore (append and sample are &self and internally
   serialized; the single-writer problem dissolves because there is one
   process). Accepts sequential clients; trainer reconnects freely.
   Standalone graphzero replay-serve remains for offline use.
3. Unbounded selfplay: --episodes 0 = run until killed. The store's
   WriteBatch atomicity makes kill-at-any-point safe by construction;
   note it where the flag is implemented.
4. --evaluator torch: spawns the evaluator child with --backend torch
   --checkpoint-dir DIR --device DEV via extra_args; flags --checkpoint-dir
   and --eval-device on the CLI.
```

## Python Prerequisites

```text
1. Evaluator hot-swap (per GZ_PYTHON.md's contract): a loader thread in
   the evaluator polls latest.json (default every 10s); on a new
   model_version it builds + loads + warms the new model off to the side,
   then hands it over; the serving loop swaps between batches. A
   checkpoint with mismatched tags is refused loudly and the old model
   keeps serving. model_version already rides every EVAL_RESULT, so
   replay rows record exactly which weights produced them.
2. The model exposes its pre-tanh value scalar for training (serving
   still returns tanh(v)).
```

## Trainer

Layout per GZ_PYTHON.md (`sampler.py`, `data.py`, `loop.py`, `publish.py`,
`driver.py`, thin `__main__`).

```text
sampler.py  connects to the sample socket, validates SHELLO_ACK versions,
            builds FeatureSchemaConfig from the ack; one seeded SAMPLE per
            step (seed = run_seed mixed with global step). At startup,
            waits with backoff until produced_rows >= min_startup_rows
            (default: one batch) before the first step. Reconnects with
            backoff on socket loss; aborts after reconnect_limit.
data.py     (GZFB, GZFT) bytes -> cuda:1 tensors via gz.codec views and
            preallocated pinned staging (same pattern as the evaluator)
loop.py     continuous, no phases:
              logits, v_raw = model(batch)           # live weights, GPU 1
              policy: mask padded slots to -inf;
                Lp = mean_rows( -sum_a pi_target[a] * log_softmax(logits)[a] )
              value: y = (label+1)/2 in {0, 0.5, 1}; p = sigmoid(2*v_raw)
                Lv = BCE(p, y) over value_valid rows (zero valid -> Lv = 0)
              L = Lp + value_weight * Lv
              AdamW + grad clip; cosine schedule over total_steps with
              warmup; update weight EMA
              every publish_interval steps: publish EMA checkpoint
publish.py  gz/checkpoints publish; manifest binds the ack's schema config
            + ArchConfig; training_step recorded; model_version derived
```

Config: one TOML. `[trainer]` lr 3e-4, warmup_steps 200, batch 256,
window_rows 200_000, total_steps, publish_interval 500, value_weight 1.0,
ema_decay 0.999, grad_clip 1.0, min_startup_rows, seed, device "cuda:1".
`[selfplay]` lanes/workers/simulations/max_steps/reference passthrough,
max_row_backlog (default = window_rows), eval device "cuda:0".
`[paths]` replay dir, checkpoint dir, run dir. Defaults are starting
points for the benchmark campaign, not claims.

## The Supervisor (driver.py)

```text
bootstrap (once per run, sequential):
  1. graphzero selfplay --evaluator stub --episodes B0 (small, e.g. 64)
     into the run's replay dir — creates the store and its schema
  2. graphzero replay-serve briefly; trainer reads the schema from the
     ack, builds the model on cuda:1, publishes version 0 (random init),
     stops serve
steady state (concurrent):
  3. spawn: graphzero selfplay --evaluator torch --checkpoint-dir ...
     --eval-device cuda:0 --episodes 0 --serve-socket PATH
     --reference self-average --replay-backlog max_row_backlog
  4. run the trainer loop against the socket until total_steps
  5. shutdown: publish the final checkpoint, SIGKILL the selfplay process
     (kill-safe by store atomicity), exit 0
supervision: poll both children; either exiting early kills the other and
aborts with the culprit and its exit status in the message. No restarts.
```

Flow control needs no new code: the trainer's sampling advances
consumed_rows; selfplay's existing backpressure gate throttles admission
when produced - consumed exceeds max_row_backlog. Slow trainer -> selfplay
idles; slow selfplay -> the trainer resamples the window (staleness rises;
the samples-per-row metric makes it visible).

## Metrics

JSON lines in the run dir, one stream (no phase boundaries):

```text
per step (every log_interval): step, Lp, Lv, grad_norm, lr, value
accuracy on valid rows, fraction valid, label mean among valid (the live
win-rate proxy), max reward_target seen (the record proxy),
produced_rows/consumed_rows from the latest ack or sample response,
samples_per_row = (steps * batch) / produced_rows
per publish: training_step, model_version
```

Selfplay-side periodic stats (episodes/s, per-checkpoint win rates) are a
later Rust addition; the label-mean proxy is enough to see learning.

## Non-Goals (deferred, with triggers)

```text
frozen-checkpoint Gumbel opponent + gating  -> self-average labels stop
                                               correlating with the record
graded/tanh targets                         -> sign labels too coarse
HL-Gauss value head                         -> value loss plateaus
Muon/mixed optimizer                        -> first curve exists; A/B
resume, optimizer-state checkpoints         -> runs long enough to hurt
per-root reward EMA / persistence           -> single-graph runs at scale
child restart policy in the supervisor      -> long runs die to transients
remote trainer host                         -> second box exists (the
                                               couplings already permit it)
wandb                                       -> implemented: optional
                                               [wandb] project mirror of
                                               the step/publish JSONL
                                               (train/*, perf/*, publish/*
                                               groups); JSONL stays the
                                               source of truth
DDP                                         -> one training GPU saturated
```

## Acceptance For The Work Order (when written)

```text
a supervisor run from empty dirs reaches total_steps unattended, with
selfplay and training demonstrably concurrent (overlapping timestamps in
the metrics and selfplay output)
the evaluator hot-swaps at least once mid-run: replay rows exist with two
or more distinct model_versions, and no eval errors during swaps
loss curves move; value accuracy beats the label base rate; label mean
drifts upward from ~0
backpressure engages under a deliberately slowed trainer (test knob) and
selfplay throughput drops instead of the backlog growing unboundedly
kill -9 of either child mid-run -> supervisor aborts cleanly, store
reopens intact afterwards
deterministic per-step sampling given (run_seed, step); training itself
is not bit-reproducible (CUDA) and is not claimed to be
```
