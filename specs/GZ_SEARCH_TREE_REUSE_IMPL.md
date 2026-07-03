# Gumbel Tree Reuse Implementation Spec

Status: implementation work order

Purpose: carry the selected child's subtree from one root search into the
next inside an episode, so step N+1 starts from an evaluated root with
carried visit statistics instead of rebuilding the tree from scratch.
This removes the per-step root eval and credits carried visits against
the sequential-halving schedule — the measured gap vs the predecessor
system (~59 NN evals/position here vs ~16 there at 64 simulations) is
mostly this. Selfplay is the training loop's bottleneck (8x row reuse,
evaluator GPU at ~22%); this is the highest-leverage change.

Authority: `GZ_SEARCH_GUMBEL_MCTS.md` (the search contract; amend it in
stage 3), `GZ_ORCHESTRATOR.md` (tasks stay pure state machines).
Contract wins; report conflicts.

Read before starting:

```text
crates/gz-search/src/gumbel.rs      (everything happens here)
  GumbelEpisodeTask::poll           Root Done arm — where reuse hooks in
  GumbelEpisodeTask::new_root_task  the fresh-tree path being bypassed
  GumbelRootTask::new / poll        EmitNodeExpand -> EmitNodeEval ->
                                    Running; seeded tasks skip to Running
  Tree / Node                       children are indices into tree.nodes
  start_run_state                   root gumbels seeded by (seed, root_step)
  start_descent / best_eligible     visits == target eligibility (see below)
  finish_root                       selection + GumbelRootStats
crates/gz-search/src/hash.rs        gumbel_search_config_hash
crates/gz-cli/src/selfplay.rs       search() constructs GumbelMctsConfig
```

## Hard Constraints

```text
Every stage ends with cargo fmt --all -- --check, cargo clippy
--all-targets --all-features -- -D warnings, cargo test --all, and
python3 -m pytest python/tests green. Commit per stage; stage 0 commits
any dirty tree. Check git status first: a sibling session may be
running sweeps from this tree — do not commit files you did not change.
reuse OFF is bit-identical to today: every existing test passes
unchanged, and the off-path code emits the same work sequence, same
tokens, same results. The regression oracle is the whole point of the
flag.
Tasks stay pure state machines: reuse adds no engine or evaluator calls
outside poll/resume work items, and no interior mutability.
No replay schema or encoding changes. GumbelStep and the GZFR row are
untouched.
No cross-episode reuse, no transposition/eval caches, no staleness
re-evaluation. Subtree carry within one episode only.
```

## Semantics

One new config field, threaded everywhere explicitly:

```text
GumbelMctsConfig.tree_reuse: bool
  included in gumbel_search_config_hash unconditionally (all hashes
  change; stores are scratch and search_config_hash carries no
  cross-version compatibility promise — note it in the commit message)
  no Default impl exists; every construction site chooses. CLI default
  is true.
```

Reuse mechanics, hooked into `GumbelEpisodeTask::poll`'s
`SearchPoll::Done(result)` arm, where the finished `GumbelRootTask` is
still in hand:

```text
if tree_reuse and the selected action is not Stop:
  extract the subtree rooted at the selected child: walk reachable
  node indices from tree.nodes[old_root].children[selected], compact
  into a fresh Vec with remapped child indices, preserving every node
  payload (graph handle, context, candidates, eval_actions, action
  refs, summaries, logits, priors, value, model_version, per-action
  visits/value_sum/q, masked actions). Graph handles stay valid for
  the episode by the GraphEngine contract.
  the next root task is constructed seeded: root_context = carried
  node.context (assert it equals the episode's current_context),
  tree.context = the NEW GumbelSearchContext (fresh root_step, budget,
  temperature — new evals use the new position context), state jumps
  straight to Running with a RunState built by start_run_state over
  the carried root node. No root Expand, no root Eval.
selected Stop, missing child, or tree_reuse off: exactly today's path.
```

Carried values, logits, and priors were computed under the previous
step's position context and possibly an older model version — accepted
staleness, same policy as the predecessor system; a hot swap propagates
within at most max_steps moves. Root Gumbel noise is NOT carried:
`start_run_state` already reseeds from (config.seed, root_step), which
differs per step, so fresh noise and a fresh considered set fall out for
free and determinism is preserved.

The schedule must credit carried visits — this is where the eval savings
come from, and the one behavioral subtlety:

```text
best_eligible currently requires node.visits[action] == target_visits,
which assumes visits progress exactly with the schedule. Carried visits
violate that (some actions start above zero), and would end the search
immediately. Under tree_reuse:
  eligibility becomes visits <= target_visits (an action behind the
  schedule catches up; an action already at/above its target is
  satisfied)
  a schedule slot with no eligible action is SKIPPED: advance
  schedule_index without consuming a simulation or a descent — this,
  not the root-eval skip, is the bulk of the savings (the selected
  child typically carries a large share of the previous step's budget)
with tree_reuse off, the == comparison and today's finish-early
behavior are preserved verbatim (the off-path oracle above).
```

`GumbelRootStats` gains `carried_nodes: usize` and
`carried_root_visits: u32` (zero on fresh trees) so tests and future
metrics can see reuse working. `run.simulations` keeps counting only
simulations actually consumed.

## Stage 0: Commit

Commit any dirty tree (subject to the sibling-session constraint above).

## Stage 1: Subtree Carry In gz-search

The config field, the hash change, subtree extraction, seeded
construction, schedule crediting, and stats — all behind the flag.

Tests (crates/gz-search/tests/):

```text
off-path regression: every existing gumbel test passes with
tree_reuse: false added to its config literal and nothing else changed
determinism: two serial runs of the same seeded episode with reuse on
produce identical steps, policy targets, and final measures
subtree carry: after step 0 selects action a, the next root's node
count equals the reachable-subtree size under a's child, the new root's
per-action visits equal the old child's, and carried_nodes/
carried_root_visits report them
visit conservation: sum of the seeded root's action visits equals the
old tree's visit count into that child's subtree
eval savings: on a generated Whittle root with simulations=16, every
step >= 1 reports eval_count strictly below step 0's, and stats show
nonzero carried_root_visits
stop and terminal edges: an episode whose selection hits Stop, and one
whose reused root has no candidates (stop-only action set), both
complete cleanly
schedule skip: a hand-seeded tree where every considered action already
exceeds its target finishes with zero consumed simulations and still
selects/finishes normally
```

## Stage 2: Driver Equality Oracles

Parameterize the existing serial == batched == threaded episode-equality
tests over tree_reuse on/off (same seeds). Reuse is per-task state, so
driver interleaving must not change results — this is the strongest
correctness check available and it must pass with reuse ON.

Golden fingerprints: existing golden tests stay pinned with reuse off;
add reuse-on fingerprints next to them (assert-print-paste discipline).

## Stage 3: CLI, Docs, Benchmark

```text
gz-cli: --tree-reuse true|false (default true) -> GumbelMctsConfig;
usage string; validation none (bool parse rejects garbage). The
selfplay stub/process/torch paths all inherit it via search().
CODEBASE_OUTLINE gz-cli usage line gains the flag.
GZ_SEARCH_GUMBEL_MCTS.md: a Reuse section documenting the semantics
block above, including the staleness acceptance and the <= eligibility
change under the flag.
AGENTS.md lists this spec.
```

Benchmark (paste numbers into the commit message): the matched
comparison from the review —

```bash
target/release/graphzero selfplay --replay-dir /tmp/gz-reuse-bench \
  --episodes 512 --lanes 2 --workers-per-lane 16 --simulations 64 \
  --max-steps 64 --max-batch 32 --evaluator torch \
  --checkpoint-dir <abs path to a version_0-only dir> \
  --eval-device cuda:0 --python-dir python --reference self-average
```

run with --tree-reuse false then true (same seed), idle GPUs (check
nvidia-smi first — a sweep may be running), subtract a small-run
startup baseline. Report rows/s and eval-rows/row both ways.

Acceptance: at simulations=64 on Whittle, reuse cuts mean NN evals per
position (steps >= 1) by at least 35% versus reuse off. If it does not,
report the measured breakdown (root-eval savings vs schedule credit)
instead of tuning past the target.

## Acceptance Checklist

```text
reuse off is bit-identical: full suite green with no golden changes
serial == threaded equality holds with reuse on
eval savings demonstrated in tests (sims=16) and benchmark (sims=64,
>= 35% fewer evals/position)
per-step determinism with reuse on across runs and drivers
no replay/GZFR changes; python tests untouched and green
--tree-reuse parses, defaults true, appears in usage and outline
```

## Out Of Scope

```text
cross-episode tree or eval caching, transposition tables
wave MCTS
staleness-triggered re-evaluation of carried nodes
changing == eligibility on the reuse-off path (today's finish-early
behavior after apply-rejection masking is preserved as-is)
compiler-engine tuning
```
