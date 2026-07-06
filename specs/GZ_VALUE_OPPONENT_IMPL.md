# Opponent-Conditioned Value Head Implementation Spec

Status: implementation work order

Purpose: give the value head the opponent it is being scored against.
Today `value_raw = value(g_readout)` predicts sign(learner final
reward - reference reward) with the reference INVISIBLE: the network
absorbs the current bar into its weights and must relearn it every
time the bar moves -- per publish under the policy reference, per
episode drift under self-average. whittlezero conditions the value
head on the opponent explicitly (`value_input: pair | scalar`,
model/value.py: "the value predicts E[sgn(r_self - r_opp)] from the
concatenated pair") and its strongest single-graph runs used it.

Stage 1 (this work order) is the `scalar` mode: the reference's final
reward as a value-head input. For a sign target the opponent's final
scalar is the sufficient statistic, it exists for EVERY reference
kind (all `Reference`s carry `final_reward`), and it needs no
opponent model at serving. Stage 2 (`pair`, timestep-aligned opponent
embeddings) is specced as a follow-up gated on Stage 1 results.

Authority: `GZ_FEATURES.md` (schema change lands there), `GZ_MODEL.md`,
`GZ_REPLAY.md` (labels unchanged -- inputs only). Contract wins;
report conflicts.

Read before starting:

```text
python/gz/model/exphormer.py         value head + BatchStager
crates/gz-features/src/{row,codec,collator,schema}.rs
                                     row/batch encodings + schema hash
crates/gz-search/src/gumbel/types.rs GumbelOpponentContext (extends)
crates/gz-orchestrator/src/pool.rs   admission -- the reference must
                                     exist BEFORE task creation now
crates/gz-orchestrator/src/lanes.rs  reference computation + rollout
python/gz/trainer/{data,loop}.py     training batch + loss (unchanged)
../whittlezero/model/value.py        the reference implementation
```

## Design

```text
One new per-row feature pair, ALWAYS carried (rows, batches, store):
  opponent_reward  f32 (bf16 on wire): the admission-time reference's
                   final_reward, scaled by 1/opponent_reward_scale
                   (schema field, default 256 -- whittle rewards are
                   -cost in (-256, 0), so inputs land in (-1, 0))
  opponent_present u8: 0 for unlabeled episodes (reference None) and
                   for opponent rollout episodes themselves

Model consumption is gated separately by [arch] value_input =
"single" (default, today's behavior) | "scalar":
  scalar: value head input = concat(g_readout, opponent_reward,
  opponent_present) -- in_dim + 2. Unlabeled rows carry (0, 0), the
  whittlezero missing-opponent zeros fallback; their value targets
  are masked anyway, so the flag mostly serves the policy trunk's
  gradient hygiene.
Carrying the feature unconditionally means flipping value_input is an
[arch] experiment, not a store-schema migration; the schema hash
still moves ONCE when this lands (new sections), a clean run
boundary like the v2 encodings.
```

Consistency invariant (the point of the design): the scalar fed at
eval time, the scalar stored in the row, and the reward used to
compute that row's value label all come from the SAME admission-time
`Reference`. Episodes labeled against a bar are evaluated seeing that
bar.

## Stages

```text
1. Plumbing the scalar to search evals.
   GumbelOpponentContext gains final_reward: f32 (alongside
   trajectory_id/row_count); EvalOpponentContext mirrors it.
   Admission reorder in pool.admit: the reference is currently
   computed AFTER admit returns, but the episode context needs it at
   task creation. Admission gains a per-episode context callback
   (roots -> reference -> opponent context) supplied by the lane
   mode; replay modes compute provider.reference() there and stash it
   for projection (same value, single computation); non-replay modes
   return None. Opponent rollout episodes (OpponentRollout) pass
   opponent = None.
   Tree::position already forwards context.opponent into
   EvalPositionContext -- no search kernel changes. Not part of the
   search config hash (same rationale as export_position).

2. Features and encodings.
   FeatureRow gains opponent_reward: f32, opponent_present: bool.
   Extractor signature: PositionFeatures is the natural carrier --
   add the two fields there (they are position-of-play context), so
   pool extraction and feature_rows_for_episode pass them through
   the existing plumb. Row codec v2 and batch encoding gain the two
   sections (bf16 + u8); python BatchView/BatchStager mirror; schema
   hash covers the new layout automatically. export_position does
   NOT zero these -- opponent context is exactly what that switch is
   meant to keep (graph + opponent).

3. Model.
   ArchConfig gains value_input: str = "single". BatchStager stages
   the two tensors; forward passes them to the value head only when
   value_input == "scalar" (concat before self.value). Checkpoint
   manifests already carry arch config, so serving picks the right
   head shape automatically and old checkpoints stay loadable under
   their own arch.

4. Trainer.
   The trainer consumes features from the batch encoding, so the
   scalar arrives with the features; no targets change (value_target
   stays sign, loss untouched). Verify TrainingStager surfaces the
   new fields to the model identically to the eval stager.

5. Tests.
   Unit: admission callback provides the same Reference used for the
   episode's projection (identity, not just equality). Codec: layout
   pinning + roundtrip for the new sections. Model: value_input
   single vs scalar shapes; scalar head output CHANGES when the
   opponent scalar changes (the conditioning is live) and matches
   single-mode when arch says single. Integration: fixed-seed run
   with value_input single is bit-identical to today except the
   schema hash; a policy-reference run stores rows whose
   opponent_reward equals the labeled reference reward (scaled).
```

Acceptance checklist:

```text
value_input = "single" runs are behaviorally identical to today
(new schema hash aside); "scalar" is a one-line [arch] flip
opponent_reward in stored rows == the admission reference's
final_reward / scale for labeled rows, (0,0) for unlabeled and
rollout rows
serving and training see identical opponent inputs for the same row
all suites green; measurement run: 5k-step A/B (single vs scalar,
policy reference, fixed root) comparing value_accuracy convergence
and cost trajectory -- paste both curves' summary into the review
```

## Stage 2 (separate work order, gated on Stage 1 results)

```text
value_input = "pair", whittlezero's full design: the opponent
trajectory's states embedded through the SAME trunk, paired with the
learner state at the comparable timestep (their index rule:
opp_state[min(t + offset, len-1)]).
Slot semantics and symmetrization (whittle_self_play.py ValueItem
emission): the slots are NOT randomized -- self is always the state
whose target z is being predicted, opp is the other trajectory's
state, and the head's contract is the asymmetric E[sgn(r_self -
r_opp)]. What prevents slot bias is `ptp_value_perspective: both`:
every pair also emits its MIRROR (self=opp_state, opp=self_state,
z=-z, player ids swapped) -- deterministic both-ways augmentation
rather than a per-sample coin flip, doubling pair data. Note this
requires value-only samples (the mirrored rows have no policy
target), which is why whittlezero runs SEPARATE policy and value
replays; adopting "both" here means a value-only sample stream, a
real architectural addition beyond the embedding plumbing. Their
per-side player_id bits ride along as inputs for turn parity in
alternating PTP; our fixed-root references have no turns, so they
reduce to a self/opp indicator. For the fixed-root policy/gated-policy
references the opponent trajectory is one rollout per model version,
so its per-timestep embeddings can be computed ONCE per swap and
memoized -- but they must be computed by the serving model
(evaluator-side) and delivered per eval row (~dim bf16 values/row on
the wire), and the trainer needs the same embeddings for stored rows,
which resurrects the stale-embedding question whittlezero solved by
shipping best_model in checkpoints. Substantial plumbing; only worth
it if Stage 1 shows conditioning matters but the scalar is too
coarse.
```

## Out Of Scope

```text
distributional value heads (whittlezero's hl_gauss/categorical) --
orthogonal axis, separate experiment
changing the value TARGET (stays sign vs admission reference)
per-timestep opponent scalars for self-average (no trajectory exists)
```
