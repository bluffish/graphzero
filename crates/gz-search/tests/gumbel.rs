mod common;

use common::{TestEngine, measure_options};
use gz_engine::ModelVersion;
use gz_eval::{EvalOutput, EvalRequest, EvalResult, Evaluator};
use gz_search::{
    GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig, GumbelOpponentContext, GumbelSearchContext,
    GumbelStopReason, SearchAction, considered_visit_sequence,
};
use std::collections::BTreeMap;
use std::num::NonZeroUsize;

#[derive(Clone)]
struct EvalRow {
    logits: Vec<f32>,
    value: f32,
}

#[derive(Default)]
struct RecordedEvaluator {
    rows: BTreeMap<u8, EvalRow>,
    requests: Vec<EvalRequest>,
}

impl RecordedEvaluator {
    fn row(mut self, graph: u8, logits: impl Into<Vec<f32>>, value: f32) -> Self {
        self.rows.insert(
            graph,
            EvalRow {
                logits: logits.into(),
                value,
            },
        );
        self
    }
}

impl Evaluator for RecordedEvaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvalRequest],
        out: &mut Vec<EvalOutput>,
    ) -> EvalResult<()> {
        out.clear();

        for request in requests {
            request.validate_ref()?;
            self.requests.push(request.clone());
            let graph = request.context.graph.graph_hash.as_bytes()[0];
            let row = self.rows.get(&graph).cloned().unwrap_or(EvalRow {
                logits: vec![0.0; request.action_count()],
                value: 0.0,
            });

            out.push(EvalOutput {
                model_version: ModelVersion::from_bytes([7; 16]),
                policy_logits: row.logits,
                value: row.value,
            });
        }

        Ok(())
    }
}

fn config(max_steps: usize) -> GumbelMctsConfig {
    GumbelMctsConfig {
        max_steps,
        simulations: NonZeroUsize::new(1).unwrap(),
        max_considered_actions: NonZeroUsize::new(8).unwrap(),
        seed: 0,
        gumbel_scale: 0.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        candidate_options: gz_engine::CandidateOptions::default(),
        measure_options: measure_options(),
    }
}

fn reuse_config(max_steps: usize) -> GumbelMctsConfig {
    let mut config = config(max_steps);
    config.tree_reuse = true;
    config
}

#[test]
fn sequential_halving_schedule_matches_expected_shape() {
    assert_eq!(
        considered_visit_sequence(4, 8),
        vec![0, 0, 0, 0, 1, 1, 2, 2]
    );
    assert_eq!(considered_visit_sequence(1, 4), vec![0, 1, 2, 3]);
}

#[test]
fn root_request_appends_stop_and_selected_candidate_is_reused() {
    let mut engine = TestEngine::new()
        .candidates(0, [1, 2])
        .candidates(20, [])
        .apply(0, 2, 20)
        .reward(20, 20.0);
    let mut evaluator = RecordedEvaluator::default()
        .row(0, [0.0, 4.0, -10.0], 0.0)
        .row(20, [0.0], 1.0);
    let search = GumbelMcts::new(config(1));

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_eq!(episode.final_graph, 20);
    assert_eq!(episode.stop_reason, GumbelStopReason::MaxSteps);
    assert_eq!(episode.steps.len(), 1);
    assert_eq!(episode.steps[0].selected_rank, 1);
    assert_eq!(episode.steps[0].action_count, 3);
    assert!(matches!(
        episode.steps[0].action,
        SearchAction::Candidate(2)
    ));
    assert_eq!(engine.apply_calls, vec![(0, 2)]);
    assert_eq!(evaluator.requests[0].action_count(), 3);
}

#[test]
fn episode_records_created_engine_handles() {
    let mut engine = TestEngine::new()
        .candidates(0, [1, 2])
        .candidates(20, [])
        .apply(0, 2, 20)
        .reward(20, 20.0);
    let mut evaluator = RecordedEvaluator::default()
        .row(0, [0.0, 4.0, -10.0], 0.0)
        .row(20, [0.0], 1.0);
    let search = GumbelMcts::new(config(1));

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_eq!(episode.created_graphs, vec![20]);
    assert_eq!(episode.created_candidates, Vec::<u8>::new());
}

#[test]
fn stop_is_selected_through_eval_policy_and_never_applied() {
    let mut engine = TestEngine::new().candidates(0, [1]).reward(0, 0.0);
    let mut evaluator = RecordedEvaluator::default().row(0, [-10.0, 10.0], 0.0);
    let search = GumbelMcts::new(config(3));

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_eq!(episode.final_graph, 0);
    assert_eq!(episode.stop_reason, GumbelStopReason::SelectedStop);
    assert!(matches!(episode.steps[0].action, SearchAction::Stop));
    assert!(engine.apply_calls.is_empty());
}

#[test]
fn root_budget_matches_episode_eval_positions() {
    let mut engine = TestEngine::new()
        .candidates(0, [1])
        .candidates(1, [2])
        .reward(2, 2.0);
    let mut evaluator = RecordedEvaluator::default()
        .row(0, [10.0, -10.0], 0.0)
        .row(1, [10.0, -10.0], 0.0)
        .row(2, [0.0], 1.0);
    let search = GumbelMcts::new(config(2));

    search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    let root_positions = evaluator
        .requests
        .iter()
        .filter(|request| request.position.leaf_depth == 0)
        .map(|request| {
            (
                request.position.root_step as usize,
                request.position.budget_fraction,
                request.position.budget_step,
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(
        root_positions,
        vec![
            (0, search.root_budget(0).0, search.root_budget(0).1),
            (1, search.root_budget(1).0, search.root_budget(1).1),
        ]
    );
}

#[test]
fn rejected_root_candidate_is_masked_before_final_selection() {
    let mut engine = TestEngine::new()
        .candidates(0, [1])
        .rejected(0, 1)
        .reward(0, 0.0);
    let mut evaluator = RecordedEvaluator::default().row(0, [10.0, 0.0], 0.0);
    let mut cfg = config(1);
    cfg.max_considered_actions = NonZeroUsize::new(1).unwrap();
    let search = GumbelMcts::new(cfg);

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_eq!(episode.final_graph, 0);
    assert_eq!(episode.stop_reason, GumbelStopReason::SelectedStop);
    assert!(matches!(episode.steps[0].action, SearchAction::Stop));
    assert_eq!(engine.apply_calls, vec![(0, 1)]);
}

#[test]
fn repeated_graph_on_path_stops_simulation_without_depth_budget() {
    let mut engine = TestEngine::new()
        .candidates(0, [1])
        .apply(0, 1, 0)
        .reward(0, 3.0);
    let mut evaluator = RecordedEvaluator::default().row(0, [10.0, -10.0], 3.0);
    let mut cfg = config(1);
    cfg.simulations = NonZeroUsize::new(2).unwrap();
    cfg.max_considered_actions = NonZeroUsize::new(1).unwrap();
    let search = GumbelMcts::new(cfg);

    let result = search
        .search_root(
            &mut engine,
            &mut evaluator,
            0,
            GumbelSearchContext::default(),
        )
        .unwrap();

    assert_eq!(result.stats.simulations, 2);
    assert_eq!(result.selected_after, 0);
    assert!(matches!(result.selected_action, SearchAction::Candidate(1)));
}

#[test]
fn search_config_hash_changes_when_seed_changes() {
    let mut left = config(1);
    let mut right = config(1);
    right.seed = 1;

    assert_ne!(
        GumbelMcts::new(left).search_config_hash(),
        GumbelMcts::new(right).search_config_hash()
    );

    left.max_steps = 2;
    assert_ne!(
        GumbelMcts::new(left).search_config_hash(),
        GumbelMcts::new(config(1)).search_config_hash()
    );

    let reuse = reuse_config(1);
    assert_ne!(
        GumbelMcts::new(reuse).search_config_hash(),
        GumbelMcts::new(config(1)).search_config_hash()
    );
}

#[test]
fn opponent_context_uses_same_index_alignment_and_stop_terminal_row() {
    let mut engine = TestEngine::new().candidates(0, []).reward(0, 0.0);
    let mut evaluator = RecordedEvaluator::default().row(0, [0.0], 0.0);
    let search = GumbelMcts::new(config(1));

    let result = search
        .search_root(
            &mut engine,
            &mut evaluator,
            0,
            GumbelSearchContext {
                root_step: 1,
                opponent: Some(GumbelOpponentContext {
                    trajectory_id: 9,
                    row_count: 4,
                    final_reward: -2.0,
                }),
                ..GumbelSearchContext::default()
            },
        )
        .unwrap();

    assert!(matches!(result.selected_action, SearchAction::Stop));
    assert_eq!(evaluator.requests.len(), 2);
    assert_eq!(evaluator.requests[0].position.leaf_depth, 0);
    assert_eq!(evaluator.requests[0].position.opponent_row(), Some(1));
    assert_eq!(evaluator.requests[1].position.leaf_depth, 2);
    assert_eq!(evaluator.requests[1].position.opponent_row(), Some(3));
}

#[test]
fn tree_reuse_on_is_deterministic() {
    fn run() -> gz_search::GumbelEpisode<u8, u8> {
        let mut engine = TestEngine::new()
            .candidates(0, [1])
            .candidates(1, [2])
            .candidates(2, [3])
            .reward(3, 3.0);
        let mut evaluator = RecordedEvaluator::default()
            .row(0, [10.0, -10.0], 0.0)
            .row(1, [10.0, -10.0], 0.1)
            .row(2, [10.0, -10.0], 0.2)
            .row(3, [0.0], 0.3);
        let mut config = reuse_config(3);
        config.simulations = NonZeroUsize::new(8).unwrap();
        config.max_considered_actions = NonZeroUsize::new(2).unwrap();
        config.seed = 42;

        GumbelMcts::new(config)
            .run(
                &mut engine,
                &mut evaluator,
                0,
                GumbelEpisodeContext::default(),
            )
            .unwrap()
    }

    assert_eq!(run(), run());
}

#[test]
fn tree_reuse_skips_later_root_evals() {
    let mut engine = TestEngine::new()
        .candidates(0, [1])
        .candidates(1, [2])
        .candidates(2, [3])
        .reward(3, 3.0);
    let mut evaluator = RecordedEvaluator::default()
        .row(0, [10.0, -10.0], 0.0)
        .row(1, [10.0, -10.0], 0.1)
        .row(2, [10.0, -10.0], 0.2)
        .row(3, [0.0], 0.3);
    let mut config = reuse_config(3);
    config.simulations = NonZeroUsize::new(16).unwrap();
    config.max_considered_actions = NonZeroUsize::new(2).unwrap();
    let search = GumbelMcts::new(config);

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_eq!(episode.steps.len(), 3);
    assert_eq!(episode.root_stats.len(), 3);
    assert_eq!(episode.root_stats[0].carried_nodes, 0);
    assert_eq!(episode.root_stats[0].carried_root_visits, 0);
    assert!(episode.root_stats[1..].iter().all(|stats| {
        stats.eval_count < episode.root_stats[0].eval_count && stats.carried_root_visits > 0
    }));
    // Budget crediting: a reused root runs at most
    // max(simulations - carried, simulations / 4) fresh simulations, and
    // each simulation costs at most one eval (stop re-evals included).
    assert!(episode.root_stats[1..].iter().all(|stats| {
        let budget = 16usize
            .saturating_sub(stats.carried_root_visits as usize)
            .max(4);
        stats.simulations <= budget && stats.eval_count <= budget
    }));
    let root_eval_steps = evaluator
        .requests
        .iter()
        .filter(|request| request.position.leaf_depth == 0)
        .map(|request| request.position.root_step)
        .collect::<Vec<_>>();

    assert_eq!(root_eval_steps, vec![0]);
}

#[test]
fn tree_reuse_stop_selection_completes_cleanly() {
    let mut engine = TestEngine::new().candidates(0, [1]).reward(0, 0.0);
    let mut evaluator = RecordedEvaluator::default().row(0, [-10.0, 10.0], 0.0);
    let search = GumbelMcts::new(reuse_config(3));

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_eq!(episode.stop_reason, GumbelStopReason::SelectedStop);
    assert_eq!(episode.final_graph, 0);
}
