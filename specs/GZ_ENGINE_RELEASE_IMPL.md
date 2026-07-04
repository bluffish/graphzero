# Engine Arena Release Implementation Spec

Status: implementation work order

Purpose: stop the selfplay memory leak that froze the box. The Whittle
engine retains every applied graph body, every enumerated candidate
body, and cache entries for both, forever. Measured at the 1024-action
config: ~5.3 MB retained per replay row (~100 KB per expansion is
candidate bodies alone), 300-900 MB/s at run throughput; box memory
went 6% to 100% in 26 minutes and the machine thrash-froze (no swap,
so the kernel evicts executable pages long before the OOM killer
fires). Every prior run leaked too, ~4x slower at 255-wide actions;
they ended or died of other causes first. This gates every long run.

Authority: `GZ_ENGINE.md` (contract change lands there), `GZ_ORCHESTRATOR.md`
(tasks stay pure state machines), `GZ_ENGINE_WHITTLE.md`.
Contract wins; report conflicts.

Read before starting:

```text
crates/gz-engine/src/            GraphEngine trait (release joins it)
crates/gz-engine-whittle/src/engine.rs
  GraphArena / CandidateArena    Vec arenas, no reuse -- become slabs
  Caches { candidates, transitions }  keyed by GraphHash; entries must
                                 die with their graphs
crates/gz-search/src/gumbel.rs   GumbelEpisodeTask sees every created
                                 handle (ApplyResult.after, ExpandResult
                                 candidates); episode result carries them
crates/gz-orchestrator/src/lanes.rs  lanes own engines; release after
                                 projection + append
crates/gz-cli/src/selfplay.rs    fixed-root mode: the source-owned root
                                 is never released
```

## Hard Constraints

```text
Every stage ends with cargo fmt --all -- --check, cargo clippy
--all-targets --all-features -- -D warnings, cargo test --all,
python3 -m pytest python/tests green. Commit per stage.
No behavioral change to search or labels: release happens strictly
after an episode's projection and append. All equality oracles and
goldens pass untouched.
Handle safety is the review focus: a released id must never be
dereferenced. Episodes own the handles they create exclusively (roots
come from the source and are excluded). Debug builds get generation
checks; release builds stay zero-overhead on the hot path.
The GraphEngine addition is default-no-op so non-Whittle engines and
every existing test compile unchanged.
```

## Stage 1: Contract

`GraphEngine` gains:

```rust
/// Frees engine resources for handles this caller owns. Using a
/// released handle afterwards is a contract violation; engines may
/// reuse the slots. Default: no-op (engines may retain forever).
fn release(
    &mut self,
    graphs: &[Self::Graph],
    candidates: &[Self::Candidate],
) -> EngineResult<()> {
    let _ = (graphs, candidates);
    Ok(())
}
```

GZ_ENGINE.md: contract text, ownership rule (creator owns; sources own
roots), and the explicit note that release is a lane-thread call, not a
SearchWork variant -- episodes are done when it runs.

## Stage 2: Whittle Slab Arenas

```text
GraphArena / CandidateArena become slab allocators: free list of slot
indexes; insert pops the free list before growing; release pushes.
Ids stay u32 slot indexes in release builds. Debug builds add a
generation counter per slot, checked on every dereference (panic on
stale handle) -- cfg(debug_assertions) only.
Cache invalidation: releasing a graph drops caches.candidates entry
for its hash and the transitions entries keyed by it; releasing a
candidate is covered by its parent graph's entry removal (candidate
ids only reach the cache through that entry). The fixed root's cache
entries survive because the root is never released.
The engine root graph (WhittleRoot) is never releasable: release of
the root id is an error.
Tests: slot reuse round-trip; arena len bounded across N
insert/release cycles; debug stale-handle panic; cache entries gone
after release; releasing the root errors.
```

## Stage 3: Episode Handle Tracking

```text
GumbelEpisodeTask accumulates created handles: every ApplyResult.after
graph (including rejected-then-masked applies' graphs if any were
created -- audit ApplyResult), every ExpandResult candidate. The root
passed in is NOT tracked. Tree reuse keeps handles within the episode;
at Done, the episode result gains created_graphs / created_candidates
(moved out, not cloned).
The final selected graphs per step and the episode's final graph are
episode-created and ARE released -- projection has already copied
portable contexts by then; nothing downstream holds engine handles.
Audit and document that claim in the work order review: replay records
hold ReplayGraphContext (portable), never Graph handles.
Tests: tracked counts equal expanded_nodes/eval counts from stats;
serial == threaded equality unchanged.
```

## Stage 4: Lane Release + CLI

```text
Replay lanes call engine.release(&episode.created_graphs,
&episode.created_candidates) after append succeeds (and also on
episode DROP paths -- unmeasured/invalid episodes must release too).
The serial driver releases after episode completion likewise.
CLI: no new flags; release is unconditional (the no-op default keeps
other engines unaffected).
The bounded-memory proof: rerun the leak probe from the post-mortem
(32 lanes x 8 workers, 1023 candidates, 48/8 sims, max-steps 128,
fixed root, stub evaluator): peak RSS at 192 episodes must be within
2x of peak RSS at 64 episodes (it was 5x before: 12.2 -> 60.5 GB).
Paste both numbers into the commit message.
```

## Stage 5: Docs

```text
GZ_ENGINE.md release contract (stage 1); GZ_ENGINE_WHITTLE.md slab +
generation notes; CODEBASE_OUTLINE design rule 6 gains the ownership
sentence; AGENTS.md lists this spec.
```

Acceptance checklist:

```text
leak probe flat: RSS(192 eps) < 2x RSS(64 eps) at the 1024 config
all equality oracles and goldens untouched
debug-build stale-handle dereference panics; release builds add no
hot-path cost (bench eval-rows/s within noise of baseline)
drop paths release; fixed root survives across episodes
```

## Out Of Scope

```text
cross-episode caches or transposition tables built on released slots
extractor cache redesign (already bounded)
compiler-engine specifics (the contract is the enabler; queued lanes
release identically)
```
