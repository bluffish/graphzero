#[allow(dead_code)]
mod common;

use common::{TestEngine, measure_options};
use gz_engine::{CandidateOptions, GraphEngine, ModelVersion, PortableSearchActionRef};
use gz_eval::EvalOutput;
use gz_search::{
    EngineIdentity, ExpandResult, ExpandedCandidate, GumbelEpisodeContext, GumbelMcts,
    GumbelMctsConfig, GumbelPlayer, GumbelValueMode, SearchPoll, SearchWork, SearchWorkResult,
    SymmetricEpisode, SymmetricSelfplayEpisodeTask,
};
use std::collections::HashSet;
use std::num::NonZeroUsize;

fn search_with_options(
    max_steps: usize,
    simulations: usize,
    max_considered: usize,
    no_backtrack: bool,
    mask_stop: bool,
) -> GumbelMcts {
    GumbelMcts::new(GumbelMctsConfig {
        max_steps,
        simulations: NonZeroUsize::new(simulations).unwrap(),
        max_considered_actions: NonZeroUsize::new(max_considered).unwrap(),
        seed: 9,
        gumbel_scale: 0.0,
        gumbel_noise_overlap: -1.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        mask_stop,
        no_backtrack,
        value_mode: GumbelValueMode::SymmetricSelfplay,
        candidate_options: CandidateOptions::default(),
        measure_options: measure_options(),
    })
}

fn expand(engine: &mut TestEngine, graph: u8, options: CandidateOptions) -> ExpandResult<u8> {
    let mut candidates = Vec::new();
    engine.candidates(graph, options, &mut candidates).unwrap();
    ExpandResult {
        graph_hash: engine.hash(graph).unwrap(),
        candidates: candidates
            .into_iter()
            .map(|candidate| {
                let info = engine.candidate_info(graph, candidate).unwrap();
                ExpandedCandidate {
                    candidate,
                    candidate_hash: info.candidate_hash,
                    kind: info.kind,
                    tags: info.tags,
                    static_prior: info.static_prior,
                }
            })
            .collect(),
    }
}

fn drive(engine: TestEngine, max_steps: usize) -> (SymmetricEpisode<u8, u8>, usize, usize, usize) {
    drive_with_budget(engine, max_steps, 24, 2)
}

fn drive_with_budget(
    engine: TestEngine,
    max_steps: usize,
    simulations: usize,
    max_considered: usize,
) -> (SymmetricEpisode<u8, u8>, usize, usize, usize) {
    drive_with_options(
        engine,
        max_steps,
        simulations,
        max_considered,
        false,
        false,
        true,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn drive_with_options(
    engine: TestEngine,
    max_steps: usize,
    simulations: usize,
    max_considered: usize,
    no_backtrack: bool,
    reorder_applies: bool,
    mask_stop: bool,
    rewrite_after_opponent_stop: bool,
) -> (SymmetricEpisode<u8, u8>, usize, usize, usize) {
    drive_with_reuse_options(
        engine,
        max_steps,
        simulations,
        max_considered,
        no_backtrack,
        reorder_applies,
        mask_stop,
        rewrite_after_opponent_stop,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn drive_with_reuse_options(
    mut engine: TestEngine,
    max_steps: usize,
    simulations: usize,
    max_considered: usize,
    no_backtrack: bool,
    reorder_applies: bool,
    mask_stop: bool,
    rewrite_after_opponent_stop: bool,
    tree_reuse: bool,
) -> (SymmetricEpisode<u8, u8>, usize, usize, usize) {
    let mut config = search_with_options(
        max_steps,
        simulations,
        max_considered,
        no_backtrack,
        mask_stop,
    )
    .config();
    config.tree_reuse = tree_reuse;
    let search = GumbelMcts::new(config);
    let mut task = SymmetricSelfplayEpisodeTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelEpisodeContext { noise_seed: 17 },
    );
    let mut evals = 0;
    let mut measure_pairs = 0;
    let mut pending_applies = Vec::new();
    let mut pending_evals = Vec::new();
    let mut max_pending_evals = 0;
    let mut eval_rounds = 0;
    let mut created_graphs = Vec::new();
    let mut created_candidates = Vec::new();
    loop {
        match task.poll().unwrap() {
            SearchPoll::Work(work) => {
                let token = work.token();
                let result = match work {
                    SearchWork::Expand(work) => {
                        let expanded = expand(&mut engine, work.graph, work.options);
                        created_candidates.extend(
                            expanded
                                .candidates
                                .iter()
                                .map(|candidate| candidate.candidate),
                        );
                        Some(SearchWorkResult::Expand(expanded))
                    }
                    SearchWork::Apply(work) => {
                        if reorder_applies {
                            pending_applies.push((token, work));
                            None
                        } else {
                            let applied =
                                GraphEngine::apply(&mut engine, work.graph, work.candidate)
                                    .unwrap();
                            created_graphs.push(applied.after);
                            Some(SearchWorkResult::Apply(applied))
                        }
                    }
                    SearchWork::MeasurePair(work) => {
                        measure_pairs += 1;
                        Some(SearchWorkResult::MeasurePair([
                            engine.measure(work.graphs[0], work.options).unwrap(),
                            engine.measure(work.graphs[1], work.options).unwrap(),
                        ]))
                    }
                    SearchWork::Eval(work) => {
                        assert!(work.opponent.is_some());
                        evals += 1;
                        pending_evals.push((token, work));
                        max_pending_evals = max_pending_evals.max(pending_evals.len());
                        None
                    }
                    _ => panic!("unsupported symmetric work"),
                };
                if let Some(result) = result {
                    task.resume(token, result).unwrap();
                    let handles = task.take_releasable();
                    engine
                        .release(&handles.graphs, &handles.candidates)
                        .unwrap();
                }
            }
            SearchPoll::Blocked => {
                assert!(!pending_applies.is_empty() || !pending_evals.is_empty());
                for (token, work) in pending_applies.drain(..).rev() {
                    let result =
                        GraphEngine::apply(&mut engine, work.graph, work.candidate).unwrap();
                    created_graphs.push(result.after);
                    task.resume(token, SearchWorkResult::Apply(result)).unwrap();
                    let handles = task.take_releasable();
                    engine
                        .release(&handles.graphs, &handles.candidates)
                        .unwrap();
                }
                if !pending_evals.is_empty() {
                    eval_rounds += 1;
                }
                for (token, work) in pending_evals.drain(..).rev() {
                    let mut logits = vec![0.0; work.request.actions.len()];
                    let opponent_inactive = work
                        .opponent
                        .as_ref()
                        .is_some_and(|opponent| opponent.position.budget_step.is_sign_negative());
                    if rewrite_after_opponent_stop && opponent_inactive && logits.len() > 1 {
                        let opponent = work.opponent.as_ref().unwrap();
                        assert_eq!(opponent.position.root_step, 0);
                        assert_eq!(opponent.position.budget_fraction, 1.0);
                        logits[0] = 100.0;
                        *logits.last_mut().unwrap() = -100.0;
                    } else {
                        *logits.last_mut().unwrap() = 100.0;
                    }
                    task.resume(
                        token,
                        SearchWorkResult::Eval(EvalOutput {
                            model_version: ModelVersion::from_bytes([3; 16]),
                            policy_logits: logits,
                            value: f32::from(work.graph) / 16.0,
                        }),
                    )
                    .unwrap();
                    let handles = task.take_releasable();
                    engine
                        .release(&handles.graphs, &handles.candidates)
                        .unwrap();
                }
            }
            SearchPoll::Done(episode) => {
                assert!(pending_applies.is_empty());
                assert!(pending_evals.is_empty());
                assert_eq!(
                    measure_pairs, 1,
                    "only one final episode pair may be measured"
                );
                let handles = task.take_releasable();
                engine
                    .release(&handles.graphs, &handles.candidates)
                    .unwrap();
                assert_handles_accounted(
                    created_graphs,
                    &engine.released_graphs,
                    &episode.created_graphs,
                );
                assert_handles_accounted(
                    created_candidates,
                    &engine.released_candidates,
                    &episode.created_candidates,
                );
                return (episode, evals, max_pending_evals, eval_rounds);
            }
        }
    }
}

fn assert_handles_accounted<T>(mut created: Vec<T>, released: &[T], owned: &[T])
where
    T: Copy + Ord + std::fmt::Debug,
{
    let mut accounted = released.to_vec();
    accounted.extend_from_slice(owned);
    created.sort_unstable();
    accounted.sort_unstable();
    assert_eq!(accounted, created);
}

fn reuse_fixture() -> TestEngine {
    TestEngine::new()
        .candidates(0, [1, 2])
        .candidates(1, [3, 4])
        .candidates(2, [4, 3])
        .candidates(3, [5, 6])
        .candidates(4, [6, 5])
        .candidates(5, [7, 8])
        .candidates(6, [8, 7])
        .apply(0, 1, 1)
        .apply(0, 2, 2)
        .apply(1, 3, 3)
        .apply(1, 4, 4)
        .apply(2, 3, 3)
        .apply(2, 4, 4)
        .apply(3, 5, 5)
        .apply(3, 6, 6)
        .apply(4, 5, 5)
        .apply(4, 6, 6)
        .reward(5, 5.0)
        .reward(6, 6.0)
        .reward(7, 7.0)
        .reward(8, 8.0)
}

fn no_backtrack_reuse_fixture() -> TestEngine {
    TestEngine::new()
        .candidates(0, [1])
        .candidates(1, [2, 3, 4])
        .candidates(4, [5])
        .apply(0, 1, 1)
        .apply(1, 2, 1)
        .apply(1, 3, 1)
        .apply(1, 4, 4)
        .apply(4, 5, 5)
        .reward(1, 3.0)
        .reward(4, 2.0)
        .reward(5, 1.0)
}

#[test]
fn symmetric_search_uses_neural_values_at_terminal_leaves() {
    let engine = TestEngine::new()
        .candidates(0, [1, 2])
        .apply(0, 1, 1)
        .apply(0, 2, 2)
        .reward(1, 10.0)
        .reward(2, 5.0);

    let (episode, evals, _, _) = drive(engine, 1);

    assert!(evals > 2);
    assert_eq!(episode.p1.steps.len(), 1);
    assert_eq!(episode.p2.steps.len(), 1);
    assert_eq!(episode.p1.final_measure.scalar_reward, Some(5.0));
    assert_eq!(episode.p2.final_measure.scalar_reward, Some(10.0));
    assert!(matches!(
        episode.p1.steps[0].action,
        gz_search::SearchAction::Candidate(2)
    ));
    assert!(matches!(
        episode.p2.steps[0].action,
        gz_search::SearchAction::Candidate(1)
    ));
    for actor in [&episode.p1, &episode.p2] {
        let step = &actor.steps[0];
        assert_eq!(step.policy_target.len(), 2);
        assert_eq!(step.legal_actions.len(), 2);
        assert!(
            step.legal_actions
                .iter()
                .all(|action| !matches!(action, PortableSearchActionRef::Stop { .. }))
        );
    }
}

#[test]
fn both_players_can_stop_and_end_before_the_rewrite_budget() {
    let engine = TestEngine::new()
        .candidates(0, [1])
        .apply(0, 1, 1)
        .reward(0, 4.0)
        .reward(1, 4.0);

    let (episode, _, _, _) = drive_with_options(engine, 4, 24, 2, false, false, false, false);

    for actor in [&episode.p1, &episode.p2] {
        assert!(actor.stopped);
        assert!(!actor.blocked);
        assert_eq!(actor.steps.len(), 1);
        let step = &actor.steps[0];
        assert!(matches!(step.action, gz_search::SearchAction::Stop));
        assert_eq!(step.before, step.after);
        assert_eq!(step.engine_candidate_count, 1);
        assert_eq!(step.action_count, 2);
        assert_eq!(step.policy_target.len(), 2);
        assert!(matches!(
            step.legal_actions.last(),
            Some(PortableSearchActionRef::Stop { .. })
        ));
    }
    assert_eq!(episode.p2.steps[0].root_q_max, 0.0);
}

#[test]
fn stopping_retires_only_that_player_while_the_opponent_continues() {
    let engine = TestEngine::new()
        .candidates(0, [1])
        .apply(0, 1, 1)
        .reward(0, 4.0)
        .reward(1, 4.0);

    let (episode, _, _, _) = drive_with_options(engine, 4, 24, 2, false, false, false, true);

    assert!(episode.p1.stopped);
    assert!(!episode.p1.blocked);
    assert!(matches!(
        episode.p1.steps[0].action,
        gz_search::SearchAction::Stop
    ));
    assert!(!episode.p2.stopped);
    assert!(episode.p2.blocked);
    assert!(matches!(
        episode.p2.steps[0].action,
        gz_search::SearchAction::Candidate(1)
    ));
    assert_eq!(episode.p1.final_graph, 0);
    assert_eq!(episode.p2.final_graph, 1);
}

#[test]
fn stop_remains_selectable_when_no_backtrack_masks_every_rewrite() {
    let engine = TestEngine::new()
        .candidates(0, [1])
        .apply(0, 1, 0)
        .reward(0, 4.0);

    let (episode, _, _, _) = drive_with_options(engine, 4, 24, 2, true, false, false, true);

    assert!(episode.p1.stopped);
    assert!(episode.p2.stopped);
    assert!(
        episode
            .p2
            .steps
            .iter()
            .all(|step| matches!(step.action, gz_search::SearchAction::Stop))
    );
}

#[test]
fn empty_candidate_roots_force_pass_without_policy_rows() {
    let engine = TestEngine::new().reward(0, 4.0);

    let (episode, evals, _, _) = drive(engine, 3);

    assert_eq!(evals, 0);
    assert!(episode.p1.blocked);
    assert!(episode.p2.blocked);
    assert!(episode.p1.steps.is_empty());
    assert!(episode.p2.steps.is_empty());
    assert_eq!(episode.p1.final_measure.scalar_reward, Some(4.0));
    assert_eq!(episode.p2.final_measure.scalar_reward, Some(4.0));
}

#[test]
fn first_player_is_canonical_root_actor() {
    let engine = TestEngine::new()
        .candidates(0, [1])
        .apply(0, 1, 1)
        .reward(1, 1.0);
    let (episode, _, _, _) = drive(engine, 1);
    assert_eq!(episode.p1.steps[0].before, 0);
    assert_eq!(episode.p2.steps[0].before, 0);
    assert_eq!(GumbelPlayer::One.opponent(), GumbelPlayer::Two);
}

#[test]
fn resumed_position_uses_requested_pair_state_and_player() {
    let mut engine = TestEngine::new().candidates(1, [3]).candidates(2, [4]);
    let search = search_with_options(8, 1, 1, false, false);
    let identity = EngineIdentity::from_engine(&engine);
    let contexts = [
        Some(identity.context(engine.hash(1).unwrap())),
        Some(identity.context(engine.hash(2).unwrap())),
    ];
    let mut task = SymmetricSelfplayEpisodeTask::from_position(
        &search,
        identity,
        [1, 2],
        contexts,
        [3, 4],
        [false, false],
        [false, false],
        GumbelPlayer::Two,
        [HashSet::new(), HashSet::new()],
        GumbelEpisodeContext { noise_seed: 17 },
    );

    let SearchPoll::Work(SearchWork::Expand(work)) = task.poll().unwrap() else {
        panic!("resumed symmetric task did not expand the requested player");
    };
    assert_eq!(work.graph, 2);
    let token = work.token;
    task.resume(
        token,
        SearchWorkResult::Expand(expand(&mut engine, work.graph, work.options)),
    )
    .unwrap();

    let SearchPoll::Work(SearchWork::Eval(work)) = task.poll().unwrap() else {
        panic!("resumed symmetric task did not evaluate its explicit pair state");
    };
    assert_eq!(work.graph, 2);
    assert_eq!(work.request.position.root_step, 4);
    let opponent = work
        .opponent
        .expect("symmetric eval must include the opponent");
    assert_eq!(opponent.graph, 1);
    assert_eq!(opponent.position.root_step, 3);

    let handles = task.take_all_handles();
    engine
        .release(&handles.graphs, &handles.candidates)
        .unwrap();
}

#[test]
fn speculative_first_visits_preserve_no_backtrack_with_reordered_applies() {
    let engine = || {
        TestEngine::new()
            .candidates(0, [1, 2])
            .candidates(2, [3, 4])
            .candidates(3, [5, 6])
            .candidates(4, [6, 5])
            .apply(0, 1, 0)
            .apply(0, 2, 2)
            .apply(2, 3, 3)
            .apply(2, 4, 4)
            .reward(3, 3.0)
            .reward(4, 4.0)
            .reward(5, 5.0)
            .reward(6, 6.0)
    };

    for tree_reuse in [false, true] {
        let (sequential, sequential_evals, _, _) =
            drive_with_reuse_options(engine(), 2, 24, 2, true, false, true, false, tree_reuse);
        let (wave, wave_evals, _, _) =
            drive_with_reuse_options(engine(), 2, 24, 2, true, true, true, false, tree_reuse);

        assert_eq!(wave_evals, sequential_evals, "reuse={tree_reuse}");
        for (wave_actor, sequential_actor) in
            [(&wave.p1, &sequential.p1), (&wave.p2, &sequential.p2)]
        {
            assert_eq!(wave_actor.steps.len(), sequential_actor.steps.len());
            for (wave_step, sequential_step) in wave_actor.steps.iter().zip(&sequential_actor.steps)
            {
                assert_eq!(wave_step.action, sequential_step.action);
                assert_eq!(wave_step.policy_target, sequential_step.policy_target);
                assert_eq!(
                    wave_step.root_search_value,
                    sequential_step.root_search_value
                );
            }
            assert_eq!(
                wave_actor.final_measure.scalar_reward,
                sequential_actor.final_measure.scalar_reward
            );
        }
    }
}

#[test]
fn speculative_first_visits_preserve_duplicate_after_states() {
    let engine = || {
        TestEngine::new()
            .candidates(0, [1, 2])
            .candidates(1, [3, 4])
            .apply(0, 1, 1)
            .apply(0, 2, 1)
            .apply(1, 3, 3)
            .apply(1, 4, 4)
            .reward(3, 3.0)
            .reward(4, 4.0)
    };

    let (sequential, sequential_evals, _, _) =
        drive_with_options(engine(), 2, 24, 2, true, false, true, false);
    let (wave, wave_evals, _, _) = drive_with_options(engine(), 2, 24, 2, true, true, true, false);

    assert_eq!(wave_evals, sequential_evals);
    for (wave_actor, sequential_actor) in [(&wave.p1, &sequential.p1), (&wave.p2, &sequential.p2)] {
        assert_eq!(wave_actor.steps.len(), sequential_actor.steps.len());
        for (wave_step, sequential_step) in wave_actor.steps.iter().zip(&sequential_actor.steps) {
            assert_eq!(wave_step.action, sequential_step.action);
            assert_eq!(wave_step.policy_target, sequential_step.policy_target);
            assert_eq!(
                wave_step.root_search_value,
                sequential_step.root_search_value
            );
        }
    }
}

#[test]
fn speculative_apply_handles_are_returned_when_search_is_aborted() {
    let mut engine = TestEngine::new().candidates(0, [1, 2, 3, 4, 5, 6, 7, 8]);
    let search = search_with_options(2, 48, 8, false, true);
    let mut task = SymmetricSelfplayEpisodeTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelEpisodeContext { noise_seed: 17 },
    );
    let mut speculative_applies = 0;

    while speculative_applies < 8 {
        let SearchPoll::Work(work) = task.poll().unwrap() else {
            panic!("symmetric preflight blocked before all apply results arrived");
        };
        let token = work.token();
        let result = match work {
            SearchWork::Expand(work) => {
                SearchWorkResult::Expand(expand(&mut engine, work.graph, work.options))
            }
            SearchWork::Apply(work) => {
                speculative_applies += 1;
                SearchWorkResult::Apply(
                    GraphEngine::apply(&mut engine, work.graph, work.candidate).unwrap(),
                )
            }
            SearchWork::Eval(work) => SearchWorkResult::Eval(EvalOutput {
                model_version: ModelVersion::from_bytes([3; 16]),
                policy_logits: vec![0.0; work.request.actions.len()],
                value: 0.0,
            }),
            _ => panic!("unexpected work before symmetric root preflight"),
        };
        task.resume(token, result).unwrap();
    }

    let mut handles = task.take_all_handles();
    handles.graphs.sort_unstable();
    assert_eq!(handles.graphs, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    engine
        .release(&handles.graphs, &handles.candidates)
        .unwrap();
}

#[test]
fn tree_reuse_is_deterministic_and_carries_promoted_root_statistics() {
    let run =
        || drive_with_reuse_options(reuse_fixture(), 3, 24, 2, false, false, true, false, true).0;

    let first = run();
    let second = run();
    for (left, right) in [(&first.p1, &second.p1), (&first.p2, &second.p2)] {
        assert_eq!(left.steps, right.steps);
        assert_eq!(left.root_stats, right.root_stats);
        assert_eq!(left.final_context, right.final_context);
        assert!(left.root_stats.iter().all(|stats| stats.simulations == 24));
    }
    assert!(
        first
            .p1
            .root_stats
            .iter()
            .chain(&first.p2.root_stats)
            .skip(1)
            .any(|stats| stats.carried_nodes > 0 && stats.carried_root_visits > 0)
    );
}

#[test]
fn symmetric_tree_reuse_changes_the_search_identity() {
    let fresh = search_with_options(3, 24, 2, false, true);
    let mut config = fresh.config();
    config.tree_reuse = true;
    let reused = GumbelMcts::new(config);

    assert_eq!(
        fresh.search_config_hash().to_string(),
        "d4bc66f8e16347df82eecf5ddd85d36372174bbf6b40c7370509853127410fd8"
    );
    assert_eq!(
        gz_search::symmetric_selfplay_search_config_hash(fresh.search_config_hash()).to_string(),
        "04b125e03ae9f6843df8c4a0980edf0bd7799c86b7d0da009a470498918ff186"
    );
    assert_ne!(fresh.search_config_hash(), reused.search_config_hash());
}

#[test]
fn tree_reuse_reduces_eval_work_while_adding_the_full_root_budget() {
    let (_, fresh_evals, _, _) =
        drive_with_reuse_options(reuse_fixture(), 3, 24, 2, false, false, true, false, false);
    let (reused, reused_evals, _, _) =
        drive_with_reuse_options(reuse_fixture(), 3, 24, 2, false, false, true, false, true);

    assert!(
        reused_evals < fresh_evals,
        "reuse evals {reused_evals}, fresh evals {fresh_evals}"
    );
    assert!(
        reused
            .p1
            .root_stats
            .iter()
            .chain(&reused.p2.root_stats)
            .all(|stats| stats.simulations == 24)
    );
    println!("symmetric fresh evals={fresh_evals}, reuse evals={reused_evals}");
}

#[test]
fn tree_reuse_carries_a_stop_transition_to_the_remaining_player() {
    let engine = TestEngine::new()
        .candidates(0, [1])
        .apply(0, 1, 1)
        .reward(0, 4.0)
        .reward(1, 4.0);

    let (episode, _, _, _) =
        drive_with_reuse_options(engine, 4, 24, 2, false, false, false, true, true);

    assert!(episode.p1.stopped);
    assert!(matches!(
        episode.p1.steps[0].action,
        gz_search::SearchAction::Stop
    ));
    assert!(!episode.p2.stopped);
    assert!(episode.p2.root_stats[0].carried_nodes > 0);
    assert!(episode.p2.root_stats[0].carried_root_visits > 0);
    assert_eq!(episode.p2.root_stats[0].simulations, 24);
}

#[test]
fn tree_reuse_skips_a_known_inactive_player_without_losing_the_subtree() {
    let engine = TestEngine::new()
        .candidates(0, [1])
        .candidates(1, [2])
        .apply(0, 1, 1)
        .apply(1, 2, 2)
        .reward(0, 4.0)
        .reward(1, 3.0)
        .reward(2, 2.0);

    let (episode, _, _, _) =
        drive_with_reuse_options(engine, 2, 24, 2, false, false, false, true, true);

    assert!(episode.p1.stopped);
    assert_eq!(episode.p2.steps.len(), 2);
    assert!(
        episode
            .p2
            .root_stats
            .iter()
            .all(|stats| stats.carried_nodes > 0 && stats.simulations == 24)
    );
}

#[test]
fn tree_reuse_preserves_no_backtrack_across_cached_branches() {
    let (episode, _, _, _) = drive_with_reuse_options(
        no_backtrack_reuse_fixture(),
        2,
        24,
        2,
        true,
        true,
        true,
        false,
        true,
    );

    assert!(
        episode
            .p1
            .root_stats
            .iter()
            .chain(&episode.p2.root_stats)
            .any(|stats| stats.carried_nodes > 0)
    );
    for step in episode.p1.steps.iter().chain(&episode.p2.steps) {
        assert_ne!(
            step.before, step.after,
            "reuse selected a no-backtrack loop"
        );
    }
}

#[test]
fn tree_reuse_excludes_carried_masks_from_the_root_considered_set() {
    let (episode, _, _, _) = drive_with_reuse_options(
        no_backtrack_reuse_fixture(),
        2,
        24,
        2,
        true,
        true,
        true,
        false,
        true,
    );

    assert!(
        episode
            .p1
            .root_stats
            .iter()
            .chain(&episode.p2.root_stats)
            .all(|stats| stats.simulations == 24),
        "a reused root lost simulation budget"
    );
}

#[test]
fn symmetric_root_replenishes_the_considered_set_after_new_masks() {
    let (episode, _, _, _) = drive_with_reuse_options(
        no_backtrack_reuse_fixture(),
        2,
        24,
        2,
        true,
        true,
        true,
        false,
        false,
    );

    assert!(
        episode
            .p1
            .root_stats
            .iter()
            .chain(&episode.p2.root_stats)
            .all(|stats| stats.simulations == 24),
        "a root lost simulation budget after new masks"
    );
}

#[test]
fn aborting_a_promoted_tree_returns_every_retained_handle() {
    let (completed, _, _, _) =
        drive_with_reuse_options(reuse_fixture(), 3, 24, 2, false, false, true, false, true);
    let first_root_evals = completed.p1.root_stats[0].eval_count;

    let mut engine = reuse_fixture();
    let mut config = search_with_options(3, 24, 2, false, true).config();
    config.tree_reuse = true;
    let search = GumbelMcts::new(config);
    let mut task = SymmetricSelfplayEpisodeTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelEpisodeContext { noise_seed: 17 },
    );
    let mut created_graphs = Vec::new();
    let mut created_candidates = Vec::new();
    let mut evals = 0;

    loop {
        let SearchPoll::Work(work) = task.poll().unwrap() else {
            panic!("sequential symmetric task blocked before promoted-tree abort");
        };
        let token = work.token();
        let result = match work {
            SearchWork::Expand(work) => {
                let expanded = expand(&mut engine, work.graph, work.options);
                created_candidates.extend(
                    expanded
                        .candidates
                        .iter()
                        .map(|candidate| candidate.candidate),
                );
                SearchWorkResult::Expand(expanded)
            }
            SearchWork::Apply(work) => {
                let applied = GraphEngine::apply(&mut engine, work.graph, work.candidate).unwrap();
                created_graphs.push(applied.after);
                SearchWorkResult::Apply(applied)
            }
            SearchWork::MeasurePair(work) => SearchWorkResult::MeasurePair([
                engine.measure(work.graphs[0], work.options).unwrap(),
                engine.measure(work.graphs[1], work.options).unwrap(),
            ]),
            SearchWork::Eval(work) => {
                evals += 1;
                SearchWorkResult::Eval(EvalOutput {
                    model_version: ModelVersion::from_bytes([3; 16]),
                    policy_logits: vec![0.0; work.request.actions.len()],
                    value: f32::from(work.graph) / 16.0,
                })
            }
            _ => panic!("unsupported symmetric work"),
        };
        task.resume(token, result).unwrap();
        let handles = task.take_releasable();
        engine
            .release(&handles.graphs, &handles.candidates)
            .unwrap();

        if evals > first_root_evals {
            let handles = task.take_all_handles();
            engine
                .release(&handles.graphs, &handles.candidates)
                .unwrap();
            assert_handles_accounted(created_graphs, &engine.released_graphs, &[]);
            assert_handles_accounted(created_candidates, &engine.released_candidates, &[]);
            break;
        }
    }
}
