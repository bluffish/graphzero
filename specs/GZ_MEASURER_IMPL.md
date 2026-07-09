# Measurer Implementation Spec

Status: design for review

Purpose: create the measurer -- the single component that owns the
reference trajectory lifecycle AND authoritative final measurement.
Today those responsibilities are scattered: 44 per-lane
PolicyReferenceProviders each hold their own theta_B bar and play their
own challenger rollouts; label semantics (sign, length tie-break, tie
coin) live in projection called from two lane completion paths (where
the 2026-07-08 positions-vs-moves off-by-one hid); "the store only
ever contains labeled rows" is enforced by drop-branches in those same
two paths; and every episode deep-clones the full reference trajectory
into its context. After this work order: one global arena, one
reference snapshot registry that lanes read, label computation and
store admission in exactly one place, and the measure-cache scaffolding
the Jetson farm needs. The gamma mix is deleted, not migrated.

Provenance: whittlezero has ONE arena (whittle_arena.py) -- per-lane
bars were our porting artifact, and mirror2's 44 bars converged to
identical values anyway (1,144 consecutive challengers at exactly -82).
The farm direction requires measurement to be a cacheable, idempotent,
centrally-owned operation (distributed-training-direction memory).

Authority: GZ_REPLAY.md (labeling contract), GZ_GATED_POLICY_IMPL.md
(gate semantics -- unchanged, relocated), GZ_OPPONENT_IMPL.md (rollout
machinery -- execution stays in lanes). Contract wins; report
conflicts.

Read before starting:

```text
crates/gz-orchestrator/src/reference.rs   PolicyReferenceProvider,
                                          ReferenceProvider trait, the
                                          gamma machinery to delete
crates/gz-orchestrator/src/lanes.rs       OpponentRollout (try_admit /
                                          intercept), ReplayMode
                                          complete() x2 (drop branches,
                                          projection call sites),
                                          admission_open gating
crates/gz-orchestrator/src/project.rs     project_episode + sign_target
                                          (moves INTO the measurer)
crates/gz-cli/src/selfplay.rs             CliReferenceProvider forwarding,
                                          reference_gamma validation
python/gz/trainer/driver.py               reference_gamma field,
                                          OpponentTracker (gate-line
                                          consumer -- format is an API)
```

## Design decisions this spec fixes

```text
1. The gamma mix is DELETED (Stage 0), not moved: reference_gamma
   knob, --reference-gamma flag, driver field, gamma/mix_seed/draws/
   mix_unit in PolicyReferenceProvider, and the latest-vs-gated draw.
   The registry holds exactly one live reference: the gated bar.

2. One global arena. Registry state is shared (Arc) across lanes:
   {current: Option<Arc<ReferenceSnapshot>>, last_challenged, pending
   ticket}. Gate semantics byte-identical to GZ_GATED_POLICY_IMPL.md
   (strict >, dueness anchored on last MEASURED challenge, unmeasured
   retries) -- relocated, not redesigned.

3. Rollout EXECUTION stays in lanes; the registry owns dueness and
   admission of exactly one challenger per version via a CAS ticket:
   claim_challenge(version) -> bool. The 44-seeds-per-run and
   44-challenges-per-version patterns collapse to 1. Seed rollout:
   whichever lane claims the cold-start ticket plays it; every other
   lane's admission_open() reads registry.current().is_some().

4. ReferenceSnapshot is immutable and shared: {ref_id: u64, version,
   final_reward, steps (positions + opponent features), search_config
   hash}. Episodes pin the Arc at admission (retirement = refcount --
   replaces today's per-episode deep clone of 65 steps of features).
   Labels always compare against the PINNED snapshot: reference
   vintage becomes explicit (ref_id recorded in ReplayReference
   telemetry) instead of implicit in admission-time capture.

5. The measurer thread absorbs the replay writer. Lanes ship a
   CompletedEpisodeArtifact {episode trajectory + feature rows +
   final_measure + pinned ref_id + episode_id} over the existing
   replay channel; the measurer computes the label (sign -> length
   tie-break in MOVES -> salted coin; semantics and unit tests move
   from project.rs verbatim), builds record+rows, and writes the
   store. The only-labeled-rows invariant becomes structural: the
   measurer refuses unlabeled artifacts for gated runs by
   construction; lane drop-branches are deleted.

6. Per-node search measurement is OUT OF SCOPE and stays a lane-local
   SearchWork item forever (invariant). Phase 1 measurer trusts the
   artifact's final_measure (lanes still measure inline); it keys its
   ledger by final-graph hash and counts distinct-vs-repeat finals.
   The farm phase later swaps the lane measure servicing strategy to
   route AUTHORITATIVE finals through the measurer's cache; the
   counters from Stage 3 size that farm. No behavior change now.

7. Gate telemetry keeps the exact stderr line format
   (event=policy_gate accepted= challenger= best= steps= version=) --
   the trainer driver's OpponentTracker parses it; the line is an API.
   One global bar means opponent_best_cost in wandb stops being a
   min-over-lanes and becomes the bar itself.

8. Transport is phase-1 in-process (channel + Arc). The artifact
   payload must stay portable (no engine handles) so phase 2 can move
   the measurer behind a UDS socket and phase 3 can put the measure
   cache in front of farm dispatch without touching lane code.
   Explicitly out of scope here.
```

## What does NOT change

```text
- Gate comparison stays reward-only (no length term) -- the duration
  margin's continuous form, if adopted, lands as one line inside the
  measurer's sign_target later.
- Episodes still cannot start before the first reference exists (the
  pair value head consumes opponent features DURING play). Seed-first
  admission gating survives; it just reads the registry.
- Non-policy references (root/greedy/beam/random/self-average) keep
  their per-lane providers untouched; the registry path is
  gated-policy only. Self-average keeps drop-and-observe.
- Search, featurization, eval transport, trainer: untouched.
```

## Stage 0: delete the gamma mix

Remove reference_gamma end to end: CLI flag + usage + validation,
driver field + arg passing, provider fields (gamma, mix_seed, draws),
mix_unit(), gated_with_gamma constructor, and the latest slot IF it
has no other consumer (check: admission_ready() uses it -- replace
with current.is_some()). Configs in configs/ are historical records:
leave them. VERIFIED: _dataclass_from_dict rejects unknown keys, so
old gamma-bearing configs become unloadable after this stage -- the
correct strictness (relaunching a historical config should force a
conscious edit, not silently drop a knob that changed its outcome).

Gate: full battery green; a stub-evaluator smoke run with
reference=gated-policy produces labeled episodes identical in count
to pre-change with gamma=0.0.

## Stage 1: registry replaces per-lane providers (gated-policy path)

New crate crates/gz-measurer, created here and grown through Stage 3.
Dependency rule (compiler-enforced transport invariant): gz-measurer
depends on gz-engine + gz-features + gz-replay ONLY -- no gz-search,
no gz-orchestrator, so it cannot name an engine handle or a lane type
by construction. Reference and ReferenceStep move down into it; lanes
convert GumbelEpisode into the portable artifact before shipping. The
phase-2 socket server becomes a thin binary over this same crate (the
gz-eval-service pattern).

registry.rs in gz-measurer:
ReferenceRegistry {current, last_challenged, pending} behind a Mutex
(44 lanes x admission-rate reads; contention is negligible against
eval latency). PolicyReferenceProvider becomes a thin adapter over
Arc<ReferenceRegistry> during this stage (trait surface unchanged) so
lanes.rs churn is limited to: seed/challenge admission consults
claim_challenge() instead of local dueness, admission_open() reads the
registry, episode_context() pins Arc<ReferenceSnapshot>.

Gate: policy-reference integration tests updated to one-seed
semantics (episodes_appended == episodes, dropped == 0, exactly ONE
seed gate line per run, exactly one challenge per version across all
lanes). Battery green. A 15-minute live leg on the blind config shows
startup <= previous (expect faster: one seed rollout, not 44) and
step-1 label mix within noise of blind-1.

## Stage 2: measurer owns labeling + store admission

Move project_episode/sign_target into gz-measurer (the crate exists
from Stage 1; the portable-payload rule is its dependency tree). Lanes'
complete() paths shrink to: intercept rollouts (unchanged), release
handles, ship artifact. Drop-branches and both projection call sites
deleted; episodes_appended/dropped accounting moves to the measurer
and is reported through the existing summary channel.

Gate: store parity test -- a recorded set of artifacts run through
old projection and new measurer produce byte-identical records and
rows (including tie-coin determinism: episode_id salting unchanged).
Battery green; tie-break unit tests relocated verbatim plus one new
test pinning "unlabeled artifact + gated run = refused".

## Stage 3: measure ledger + farm-sizing counters

Measurer keys a ledger by final-graph hash: distinct finals, repeat
rate, per-version distinct counts. Emitted per publish interval into
the metrics JSONL (event=measure_ledger) and logged to wandb under
measure/*. No behavior change -- this is the instrument that decides
whether the Jetson farm is sized by episode throughput or by
distinct-graph discovery rate.

Gate: counters visible in a live run; repeat rate on a converged
policy checkpoint demonstrably >50% (the blind-1 store's 300/300
same-basin stops predict high repeat rates; if measured low, the
cache premise of the farm design needs revisiting BEFORE any farm
spec is written).
```
