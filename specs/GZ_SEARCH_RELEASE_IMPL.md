# Search Release Completeness Implementation Spec

Status: implementation work order (CODEBASE_IMPROVEMENTS.md findings 1+2)

Purpose: complete the engine-handle release contract outside Gumbel.
GreedySearch, BeamSearch, and RandomSearch create graph and candidate
handles via `candidates()` and `apply()` but SearchEpisode records no
created-handle ownership, and the reference providers built on them
never release. WhittleMeasureEvaluator applies every candidate to
score it and never releases the temporary `after` graphs. Both bypass
the release contract that GZ_ENGINE_RELEASE_IMPL.md established for
the Gumbel path -- the same class of leak that has now cost this
project three production incidents.

Exposure, for prioritization: production configs run self-average or
policy references (no engine work), so nothing currently deployed
leaks. `--reference greedy|beam|random` leak per-admission on every
lane (per-episode in fixed-root mode), and WhittleMeasureEvaluator
(orchestrator tests, serial_gumbel_bench) leaks one graph per
candidate per eval -- it poisons every bench memory number.

Authority: `GZ_ENGINE.md` (release contract: creator owns; sources own
roots), `GZ_ENGINE_RELEASE_IMPL.md` (incl. its amendments -- read the
probe-discipline note). Contract wins; report conflicts.

Read before starting:

```text
crates/gz-search/src/episode.rs      SearchEpisode -- gains ownership
crates/gz-search/src/greedy.rs:89,102    candidates/apply call sites
crates/gz-search/src/beam.rs:391,411
crates/gz-search/src/random.rs:90,133
crates/gz-orchestrator/src/reference.rs  providers; Reference.steps
crates/gz-eval-whittle/src/lib.rs:43     the per-candidate apply
crates/gz-engine-whittle/src/engine.rs   Drop stats (arena_stats) --
                                         promote into a diagnostic
```

## Hard Constraints

```text
Every stage ends with cargo fmt --all -- --check, cargo clippy
--all-targets --all-features -- -D warnings, cargo test --all,
python3 -m pytest python/tests green. Commit per stage.
No behavioral change to search results, labels, or stored bytes:
releases happen strictly after projection into portable data.
Roots are source-owned and never tracked or released by episodes.
Release-order rule from the Gumbel path applies: dedup means a
"temporary" handle may alias a live one; release only through the
engine's refcounted release, never assume uniqueness.
```

## Stage 1: Arena Occupancy Diagnostic

```text
Promote the GZ_ARENA_STATS Drop computation into a public
WhittleEngine diagnostic:
  pub fn arena_occupancy(&self) -> ArenaOccupancy
  { graphs_live, graph_refs, candidates_live, candidate_refs }
The Drop printer reuses it. This is the oracle every test below
asserts against: capture occupancy after engine setup (root +
enumerated root candidates), run the code under test, release, assert
occupancy returns exactly to the captured baseline.
```

## Stage 2: SearchEpisode Ownership + Provider Release

```text
SearchEpisode gains created_graphs / created_candidates (moved out,
not cloned), mirroring GumbelEpisode. Greedy, beam, and random record
every id returned by candidates() and every apply().after (excluding
results that dedup onto the root -- same root-exclusion rule as
track_created_handles; the release_protected root guard is the
backstop, not the mechanism).

Reference becomes fully portable: ReferenceStep currently carries an
engine handle (`graph: G`) that nothing outside reference.rs reads --
project.rs consumes only portable contexts. Drop the handle (steps
become Vec<ReplayGraphContext> or a portable ReferenceStep), drop the
type parameter from Reference if that falls out naturally, and have
each provider release the episode's created handles BEFORE returning
its Reference. Provider signatures keep &mut engine access, so the
release happens inside reference().

Tests (per provider: greedy, beam, random on WhittleEngine):
  occupancy returns to baseline after each reference() call
  repeated calls (8x) do not grow occupancy or slot peaks
  the returned Reference is unchanged bit-for-bit from before this
  work order (labels contract intact) -- assert against captured
  pre-change values, not re-derived ones
```

## Stage 3: WhittleMeasureEvaluator Release

```text
Track every applied.after during candidate scoring and release before
returning -- INCLUDING when a later candidate's apply or measure
fails after earlier temporaries were created (collect first, release
in a single pass on both success and error paths; a small guard
struct or explicit match, implementer's choice).

Tests (gz-eval-whittle):
  multi-candidate eval returns occupancy to baseline
  a mid-batch failure (inject via a candidate whose measure cannot
  produce a scalar, or a stale handle in debug) still releases the
  earlier temporaries
  logits unchanged bit-for-bit from before the change
```

## Stage 4: Docs

```text
GZ_ENGINE.md ownership rule gains one sentence: ALL search kernels
and evaluators that create handles own and release them; Gumbel is
not special. CODEBASE_IMPROVEMENTS.md findings 1-2 marked fixed with
commit hashes.
```

Acceptance checklist:

```text
occupancy baseline round-trip for every provider and the measure
evaluator, including failure paths
no stored-byte or label changes (existing goldens plus the captured
Reference assertions)
a leak probe at the 1023 shape with --reference greedy holds flat
RSS across 32 episodes (paste numbers into the commit message);
remember the probe discipline: gumbel_scale > 0
```

## Out Of Scope

```text
fixed-root memoization of deterministic providers (GZ_OPPONENT_IMPL
Stage 2 -- it reduces cost, not correctness, and lands cleaner on top
of this contract)
generated-root source-owned root release (~6 KB/episode, tracked in
GZ_ENGINE_RELEASE_IMPL amendments)
```
