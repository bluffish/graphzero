# Gumbel Module Split Implementation Spec

Status: implementation work order (CODEBASE_IMPROVEMENTS.md finding 5;
run AFTER GZ_SEARCH_RELEASE_IMPL and GZ_LANES_DEEPEN_IMPL -- pure
code motion is cheapest when nothing else is moving)

Purpose: `gz-search/src/gumbel.rs` holds public types, the episode
and root task state machines, tree storage, sequential-halving and
schedule math, and work servicing in one ~2000-line file. Locality is
poor: reviewing a schedule change means scrolling past tree storage,
and the file has been the site of subtle bugs (budget crediting, noise
seeding, handle tracking) that are exactly the kind reviewers miss in
a file this size.

This is MECHANICAL CODE MOTION ONLY. No logic edits, no renames of
public items, no signature changes, no "while I'm here" cleanups. Any
improvement spotted during the move is reported in the review notes,
not applied.

Authority: `GZ_SEARCH_GUMBEL_MCTS.md`. Contract wins; report
conflicts.

## Hard Constraints

```text
Public API of gz-search is unchanged: same paths, same names --
`gz_search::gumbel::X` re-exports preserved via mod declarations, and
anything currently reachable as `gz_search::X` stays reachable.
Hash domains are bit-frozen: gumbel_search_config_hash inputs and the
gz-search-gumbel-mcts-v3 domain string do not move in ways that
change any computed hash. Golden fingerprint tests are the oracle and
must pass UNTOUCHED (no golden updates in this work order -- an
updated golden means the split changed behavior and is a defect).
cargo public-api is not in the toolchain; the reviewer instead
verifies no `pub` item was added or removed (grep-level check).
Full verification battery per stage; commit per stage.
```

## Stages

```text
1. gumbel/ directory: types.rs (public configs, episode/step records,
   contexts), schedule.rs (sequential halving, considered-action and
   budget math, budget_fraction), tree.rs (tree storage, visit
   accounting, subtree compaction/reuse), task.rs (GumbelEpisodeTask +
   GumbelRootTask state machines, work servicing, handle tracking).
   gumbel.rs becomes gumbel/mod.rs holding GumbelMcts, policy_rollout,
   and the re-export surface. Move tests alongside their code where
   they are unit tests; integration tests stay put.
2. Docs: GZ_SEARCH_GUMBEL_MCTS.md gains a one-paragraph module map;
   CODEBASE_IMPROVEMENTS.md finding 5 marked fixed.
```

Acceptance checklist:

```text
all suites green with zero golden-file changes
diff reviewable as pure motion (git diff --color-moved shows moved
blocks, not edits; reviewer spot-checks any non-moved hunk)
no public item added, removed, or renamed
file sizes: no module over ~600 lines
```

## Out Of Scope

```text
any behavior or performance change
splitting other large files (engine.rs, lanes.rs) -- separate
decisions after their own work orders land
```
