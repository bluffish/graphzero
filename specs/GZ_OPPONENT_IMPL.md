# Opponent Configuration Implementation Spec

Status: Stage 3 implemented (2026-07-05, simplified); Stages 1-2 open

Purpose: make the opponent (reference) a first-class, parameterized
config swap and add the missing PolicyOpponent. The provider machinery
already exists -- ReferenceProvider<E> with self-average (EMA), greedy,
beam, root-baseline, random, none, selected by [selfplay] reference --
but beam width is hardwired, deterministic opponents recompute an
identical trajectory (and pay an identical measurement) every episode
in fixed-root mode, and there is no net-based opponent.

Vocabulary: the code keeps "reference" (ReplayReference, the provider
trait, the store schema all use it); "opponent" is the same concept.
Config accepts the short kind names below; "self-average" stays as an
accepted alias for "ema".

Authority: `GZ_REPLAY.md` (labeling contract), `GZ_TRAINER.md`.
Contract wins; report conflicts.

Read before starting:

```text
crates/gz-orchestrator/src/reference.rs  the trait + all providers
crates/gz-cli/src/selfplay.rs            provider() construction,
                                         ReferenceMode parse
python/gz/trainer/driver.py              [selfplay] reference plumb
crates/gz-orchestrator/src/lanes.rs      where reference() is called
                                         relative to projection
```

## Stage 1: Config Surface

```text
[selfplay] reference kinds: ema | greedy | beam | random | root | none
("self-average" parses as ema; existing stores/enums unchanged --
ReplayReferenceKind stays append-only).
Parameters join the config instead of code:
  reference_ema_decay      (exists)         ema
  reference_beam_width     (new, default 4) beam
  reference_rollout_seed   (exists via seed) random
CLI flags mirror them; driver plumbs through both spawns. Unknown
combinations (beam_width without beam) are rejected, same style as
the torch-only flags.
```

## Stage 2: Fixed-Root Memoization

```text
Greedy, beam, and root-baseline are deterministic per root. In fixed
root mode every episode pays an identical trajectory AND an identical
measurement -- the exact cost the one-measure-per-episode design
exists to avoid, and prohibitive in the compiler regime.
A memoizing wrapper caches the computed Reference by root context
(one entry in fixed mode; bounded map in generated mode is NOT built
-- generated mode keeps per-episode computation, documented).
With memoization, greedy/beam opponents cost ONE extra measurement
per run in fixed-root mode, making them affordable baselines for the
compiler case.
Tests: fixed-root run with beam opponent performs exactly one
reference trajectory (count via engine measure stats or a counting
provider wrapper); labels match the unmemoized provider bit-for-bit.
```

## Stage 3: PolicyOpponent (fixed-root first)

Implemented, with simplifications against the text below (per review):
plain greedy rollout only -- simulations hardwired to 1, no
reference_rollout_sims knob (add it when a run wants shallow-search
rollouts). Mechanics: GumbelMcts::policy_rollout() derives the
one-simulation/one-considered/no-noise search; RootSource::fixed_root
hands the shared root to rollouts without consuming episode budget;
ReferenceProvider gained rollout_due/begin_rollout/finish_rollout
hooks driven by the lane loop (OpponentRollout in lanes.rs), which
watches model versions on eval replies, admits one rollout episode
ahead of root admission (a busy pool cannot starve it), intercepts its
completion before projection, and never appends or counts it.
Labels bind at admission, so bounded runs that admit every episode
up front (episodes <= lanes x workers) finish unlabeled; continuous
runs label everything after the first rollout. Failed (unmeasured)
rollouts keep the previous scalar and retry while the version still
differs.

```text
kind = policy: the opponent is the network itself playing the graph.
Fixed-root design -- a scalar benchmark, refreshed per checkpoint:
  on each evaluator hot-swap (new model_version observed on eval
  replies), the lane runs ONE opponent rollout from the fixed root
  through the normal episode machinery: temperature 0, gumbel_scale 0,
  simulations = reference_rollout_sims (new config, default 1 =
  argmax policy rollout; >1 = shallow search rollout), measured once.
  Its rows are NOT appended to replay; its terminal cost becomes the
  reference scalar for subsequent episodes (kind = Gumbel in
  ReplayReferenceKind, model_version recorded).
  Until the first rollout completes, episodes are unlabeled -- same
  admission rule as an unseeded EMA.
This gives "am I beating what my current net does without search" --
a harder, non-drifting bar than the EMA, refreshed at publish cadence
(one rollout + one measurement per swap).
Generated-root PolicyOpponent (per-episode rollout through the eval
path) is explicitly out of scope: it multiplies eval and measurement
load per episode and needs reference work to flow through the
batcher; design it when a generated-root run wants it.
Tests: rollout excluded from replay counters; reference updates on
swap (two published checkpoints -> two distinct reference scalars);
labels compare against the current scalar; unlabeled before first
rollout.
```

## Stage 4: Docs

```text
GZ_REPLAY.md outcome rules gain the policy kind's semantics;
config docs list kinds + parameters; AGENTS.md lists this spec.
```

Acceptance checklist:

```text
switching opponents is a one-line config change, all kinds smoke-run
under the trainer loop
beam width configurable; ema alias accepted
fixed-root greedy/beam pay one reference computation per run
policy opponent refreshes per swap, never enters replay, labels
against the newest completed rollout
all suites green; no store schema changes beyond the append-only kind
```

## Out Of Scope

```text
per-episode policy rollouts in generated-root mode
frozen-checkpoint LEAGUES (multiple opponents, ratings) -- future
arena gating (opponent selection for selfplay, not labeling)
```
