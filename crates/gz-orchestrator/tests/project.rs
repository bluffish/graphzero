use gz_engine::{CandidateOptions, MeasureOptions};
use gz_engine_whittle::{WhittleCandidateId, WhittleEngine, WhittleEngineConfig, WhittleGraphId};
use gz_eval_whittle::WhittleMeasureEvaluator;
use gz_orchestrator::project::project_episode;
use gz_orchestrator::reference::{Reference, ReferenceStep};
use gz_orchestrator::{SerialGumbelOrchestrator, WorkerId};
use gz_replay::{ReplayReferenceKind, ReplayStore};
use gz_search::{
    GumbelEpisode, GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig, GumbelStopReason,
};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "gz-orchestrator-project-test-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();

        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn projected_episode_appends_to_replay_store() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let episode = run_episode();
    let reference = reference(&episode, episode.final_measure.scalar_reward.unwrap() - 1.0);

    let (record, rows) = project_episode(&episode, Some(&reference), None, 7).unwrap();
    let id = store.append_episode(&record, &rows).unwrap();

    assert_eq!(store.episode(id).unwrap(), Some(record));
    assert_eq!(store.counters().produced_rows, rows.len() as u64);
}

#[test]
fn labels_follow_win_loss_tie_sign_rule() {
    let episode = run_episode();
    let learner = episode.final_measure.scalar_reward.unwrap();

    for (reference_reward, expected) in [(learner - 1.0, Some(1.0)), (learner + 1.0, Some(-1.0))] {
        let reference = reference(&episode, reference_reward);
        let (record, rows) = project_episode(&episode, Some(&reference), None, 7).unwrap();

        assert_eq!(record.outcome.value_target, expected);
        assert!(rows.iter().all(|row| row.value_target == expected));
    }
}

#[test]
fn exact_ties_coin_flip_to_hard_signs_deterministically() {
    let episode = run_episode();
    let learner = episode.final_measure.scalar_reward.unwrap();
    let reference = reference(&episode, learner);

    let first = project_episode(&episode, Some(&reference), None, 7)
        .unwrap()
        .0
        .outcome
        .value_target
        .unwrap();
    assert!(first == 1.0 || first == -1.0);

    // Same episode id -> same coin; the flip is a deterministic label,
    // not per-sample noise.
    let again = project_episode(&episode, Some(&reference), None, 7)
        .unwrap()
        .0
        .outcome
        .value_target
        .unwrap();
    assert_eq!(first, again);

    // Both signs are reachable across episode ids.
    let signs: std::collections::HashSet<i8> = (0..64)
        .map(|id| {
            let target = project_episode(&episode, Some(&reference), None, id)
                .unwrap()
                .0
                .outcome
                .value_target
                .unwrap();
            target as i8
        })
        .collect();
    assert_eq!(signs.len(), 2);
}

#[test]
fn reference_none_yields_policy_only_rows_that_append() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let episode = run_episode();

    let (record, rows) = project_episode(&episode, None, None, 7).unwrap();

    assert_eq!(record.outcome.value_target, None);
    assert!(rows.iter().all(|row| row.value_target.is_none()));
    store.append_episode(&record, &rows).unwrap();
}

#[test]
fn ineligible_episode_projects_to_none() {
    let mut episode = run_episode();
    episode.final_measure.valid = false;

    assert!(project_episode(&episode, None, None, 7).is_none());
}

#[test]
fn row_count_matches_steps_and_stop_row_is_decision_state() {
    let episode = run_episode();

    assert_eq!(episode.stop_reason, GumbelStopReason::SelectedStop);
    let (record, rows) = project_episode(&episode, None, None, 7).unwrap();
    let last_step = episode.steps.last().unwrap();
    let last_row = rows.last().unwrap();

    assert_eq!(record.row_count as usize, episode.steps.len());
    assert_eq!(rows.len(), episode.steps.len());
    assert_eq!(last_row.state, last_step.step_ref.before);
    assert_eq!(last_row.selected_action, last_step.selected_action);
    assert!(matches!(
        last_row.selected_action,
        gz_engine::PortableSearchActionRef::Stop { .. }
    ));
}

#[test]
fn feature_rows_are_attached_in_step_order() {
    let episode = run_episode();
    let feature_rows = (0..episode.steps.len())
        .map(|index| vec![index as u8, 99])
        .collect::<Vec<_>>();

    let (_, rows) = project_episode(&episode, None, Some(&feature_rows), 7).unwrap();

    assert_eq!(
        rows.iter()
            .map(|row| row.feature_row.clone().unwrap())
            .collect::<Vec<_>>(),
        feature_rows
    );
}

#[test]
fn feature_row_length_mismatch_rejects_projection() {
    let episode = run_episode();
    let feature_rows = Vec::new();

    assert!(project_episode(&episode, None, Some(&feature_rows), 7).is_none());
}

fn run_episode() -> GumbelEpisode<WhittleGraphId, WhittleCandidateId> {
    let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
    let search = search(&engine);
    let mut orchestrator = SerialGumbelOrchestrator::new(
        WorkerId::new(0),
        engine,
        WhittleMeasureEvaluator::new(),
        search,
    );

    orchestrator
        .run_from_root(GumbelEpisodeContext::default())
        .unwrap()
        .episode
}

fn search(engine: &WhittleEngine) -> GumbelMcts {
    GumbelMcts::new(GumbelMctsConfig {
        max_steps: 2,
        simulations: NonZeroUsize::new(2).unwrap(),
        max_considered_actions: NonZeroUsize::new(4).unwrap(),
        seed: 0,
        gumbel_scale: 0.0,
        gumbel_noise_overlap: -1.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        mask_stop: false,
        no_backtrack: false,
        candidate_options: CandidateOptions::default(),
        measure_options: measure_options(engine),
    })
}

fn measure_options(engine: &WhittleEngine) -> MeasureOptions {
    engine.measure_options()
}

fn reference(
    episode: &GumbelEpisode<WhittleGraphId, WhittleCandidateId>,
    final_reward: f32,
) -> Reference {
    Reference {
        kind: ReplayReferenceKind::RootBaseline,
        final_reward,
        final_graph: Some(episode.root_context),
        steps: vec![ReferenceStep {
            context: episode.root_context,
            features: None,
        }],
        search_config_hash: None,
        model_version: None,
    }
}
