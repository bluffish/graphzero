use gz_engine::{
    ActionSetHash, CandidateHash, EngineId, EngineIdentity, EngineVersion, GraphHash,
    MeasureConfigHash, MeasureSummary, ModelVersion, PortableCandidateRef, PortableGraphId,
    PortableSearchActionRef, ReplayGraphContext, SearchConfigHash,
};
use gz_measurer::{
    CompletedEpisodeArtifact, CompletedEpisodeStep, MeasuredSymmetricGame, MeasurerAdmissionStatus,
    MeasurerError, ReplayMeasurer, horizon_value_targets, project_episode,
};
use gz_replay::{
    ReplayContract, ReplayDataMode, ReplayEpisodeId, ReplayRowKind, ReplayStore,
    SymmetricRewardTargetScale, SymmetricRewardTieEpsilon,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "gz-measurer-service-test-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn symmetric_game_uses_reward_then_length_and_preserves_draws() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    bind_store(&store, ReplayDataMode::SymmetricSelfplay);
    let mut measurer = ReplayMeasurer::new(
        &store,
        SymmetricRewardTieEpsilon::ZERO,
        SymmetricRewardTargetScale::HARD,
    );

    let admission = measurer
        .admit_symmetric(MeasuredSymmetricGame {
            lane: 2,
            p1_artifact: artifact(1, -4.0, 2, false, version(1)),
            p2_artifact: artifact(2, -4.0, 1, false, version(2)),
        })
        .unwrap();
    assert_eq!(
        admission.status,
        MeasurerAdmissionStatus::Appended { row_count: 3 }
    );
    assert_eq!(target(&store, 0), -1.0);
    assert_eq!(target(&store, 1), 1.0);

    measurer
        .admit_symmetric(MeasuredSymmetricGame {
            lane: 2,
            p1_artifact: artifact(3, -5.0, 1, false, version(3)),
            p2_artifact: artifact(4, -5.0, 1, false, version(4)),
        })
        .unwrap();
    assert_eq!(target(&store, 2), 0.0);
    assert_eq!(target(&store, 3), 0.0);

    let summary = measurer.finish();
    assert_eq!(summary.episodes_appended, 2);
    assert_eq!(summary.replay_rows, 5);
    assert_eq!(summary.lanes[2].episodes_appended, 2);
    assert_eq!(summary.measure_ledger.finals, 4);
    assert_eq!(store.episode_counters(), (2, 0));
}

#[test]
fn symmetric_game_uses_length_within_reward_tie_epsilon() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let epsilon = SymmetricRewardTieEpsilon::new(0.01).unwrap();
    store
        .ensure_contract(
            &ReplayContract::unfeaturized(
                ReplayDataMode::SymmetricSelfplay,
                EngineIdentity::from_context(context(0)),
            )
            .with_symmetric_reward_tie_epsilon(epsilon),
        )
        .unwrap();
    let mut measurer = ReplayMeasurer::new(&store, epsilon, SymmetricRewardTargetScale::HARD);

    measurer
        .admit_symmetric(MeasuredSymmetricGame {
            lane: 0,
            p1_artifact: artifact(1, -4.0, 2, false, version(1)),
            p2_artifact: artifact(2, -4.005, 1, false, version(2)),
        })
        .unwrap();
    assert_eq!(target(&store, 0), -1.0);
    assert_eq!(target(&store, 1), 1.0);
}

#[test]
fn symmetric_game_uses_graded_reward_margin_targets() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let scale = SymmetricRewardTargetScale::new(5.0).unwrap();
    store
        .ensure_contract(
            &ReplayContract::unfeaturized(
                ReplayDataMode::SymmetricSelfplay,
                EngineIdentity::from_context(context(0)),
            )
            .with_symmetric_reward_target_scale(scale),
        )
        .unwrap();
    let mut measurer = ReplayMeasurer::new(&store, SymmetricRewardTieEpsilon::ZERO, scale);

    measurer
        .admit_symmetric(MeasuredSymmetricGame {
            lane: 0,
            p1_artifact: artifact(1, -100.0, 1, false, version(1)),
            p2_artifact: artifact(2, -105.0, 1, false, version(2)),
        })
        .unwrap();

    let expected = 1.0_f32.tanh();
    assert!((target(&store, 0) - expected).abs() < 1.0e-6);
    assert!((target(&store, 1) + expected).abs() < 1.0e-6);
}

#[test]
fn unmeasured_game_is_dropped_without_replay_rows() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    bind_store(&store, ReplayDataMode::SymmetricSelfplay);
    let mut p1 = artifact(1, -4.0, 1, false, version(1));
    p1.final_measure.measured = false;
    p1.final_measure.scalar_reward = None;
    let mut measurer = ReplayMeasurer::new(
        &store,
        SymmetricRewardTieEpsilon::ZERO,
        SymmetricRewardTargetScale::HARD,
    );

    let admission = measurer
        .admit_symmetric(MeasuredSymmetricGame {
            lane: 0,
            p1_artifact: p1,
            p2_artifact: artifact(2, -5.0, 1, false, version(2)),
        })
        .unwrap();

    assert_eq!(
        admission.status,
        MeasurerAdmissionStatus::Dropped {
            reason: MeasurerError::Unmeasured
        }
    );
    assert_eq!(store.counters().produced_rows, 0);
    assert_eq!(measurer.stats().episodes_dropped, 1);
}

#[test]
fn stop_is_not_counted_as_a_length_tiebreak_rewrite() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    bind_store(&store, ReplayDataMode::SymmetricSelfplayStop);
    let mut measurer = ReplayMeasurer::new(
        &store,
        SymmetricRewardTieEpsilon::ZERO,
        SymmetricRewardTargetScale::HARD,
    );
    let mut p2 = artifact(2, -4.0, 2, false, version(2));
    add_stop_options(&mut p2);

    measurer
        .admit_symmetric(MeasuredSymmetricGame {
            lane: 0,
            p1_artifact: artifact(1, -4.0, 1, true, version(1)),
            p2_artifact: p2,
        })
        .unwrap();

    assert_eq!(target(&store, 0), 1.0);
    assert_eq!(target(&store, 1), -1.0);
    assert_eq!(store.episode_counters(), (1, 1));
    let metrics = store.symmetric_selfplay_metrics().unwrap();
    assert_eq!(metrics.p1_episode_len_ema, 1.0);
    assert_eq!(metrics.p2_episode_len_ema, 2.0);
}

#[test]
fn horizon_targets_follow_v8_v32_recurrence() {
    let mut artifact = artifact(7, -4.0, 3, false, version(1));
    let search_values = [0.0, 0.5, -0.25];
    for (step, value) in artifact.steps.iter_mut().zip(search_values) {
        step.root_search_value = Some(value);
    }

    let targets = horizon_value_targets(&artifact, 1.0).unwrap();
    for (head, lambda) in [8.0_f32 / 9.0, 32.0 / 33.0].into_iter().enumerate() {
        let last = (1.0 - lambda) * search_values[2] + lambda;
        let middle = (1.0 - lambda) * search_values[1] + lambda * last;
        let first = (1.0 - lambda) * search_values[0] + lambda * middle;
        for (actual, expected) in targets
            .iter()
            .map(|target| target[head])
            .zip([first, middle, last])
        {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    artifact.steps[0].root_search_value = None;
    assert_eq!(
        horizon_value_targets(&artifact, 1.0),
        Err(MeasurerError::InvalidRootSearchValue)
    );
}

#[test]
fn projection_appends_a_value_only_terminal_row() {
    let mut artifact = artifact(7, -4.0, 2, false, version(1));
    artifact.terminal_feature_row = Some(vec![9, 8, 7]);

    let (record, rows) = project_episode(&artifact).unwrap();

    assert_eq!(record.row_count, 3);
    let terminal = rows.last().unwrap();
    assert_eq!(terminal.kind, ReplayRowKind::Terminal);
    assert_eq!(terminal.step_index, 2);
    assert_eq!(terminal.state, artifact.final_graph);
    assert_eq!(terminal.action_history.len(), 2);
    assert!(terminal.legal_actions.is_empty());
    assert!(terminal.policy_target.is_empty());
    assert_eq!(terminal.selected_action, None);
    assert_eq!(terminal.feature_row.as_deref(), Some([9, 8, 7].as_slice()));
}

#[test]
fn projection_rejects_misaligned_feature_rows() {
    let mut artifact = artifact(1, -4.0, 2, false, version(1));
    artifact.feature_rows = Some(vec![vec![1]]);
    assert_eq!(
        project_episode(&artifact),
        Err(MeasurerError::FeatureRowCountMismatch)
    );
}

fn target(store: &ReplayStore, id: u64) -> f32 {
    store
        .episode(ReplayEpisodeId::new(id))
        .unwrap()
        .unwrap()
        .outcome
        .value_target
        .unwrap()
}

fn add_stop_options(artifact: &mut CompletedEpisodeArtifact) {
    let stop = PortableSearchActionRef::stop(artifact.final_graph);
    for step in &mut artifact.steps {
        step.legal_actions.push(stop);
        step.policy_target.push(0.0);
    }
}

fn artifact(
    seed: u8,
    reward: f32,
    rewrites: usize,
    stopped: bool,
    model_version: ModelVersion,
) -> CompletedEpisodeArtifact {
    let state = context(seed);
    let candidate = PortableSearchActionRef::candidate(PortableCandidateRef::new(
        state,
        CandidateHash::from_bytes([seed; 32]),
    ));
    let stop = PortableSearchActionRef::stop(state);
    let mut steps = (0..rewrites)
        .map(|_| CompletedEpisodeStep {
            before: state,
            after: state,
            selected_action: candidate,
            legal_actions: if stopped {
                vec![candidate, stop]
            } else {
                vec![candidate]
            },
            policy_target: if stopped { vec![1.0, 0.0] } else { vec![1.0] },
            root_value: Some(0.25),
            root_search_value: Some(0.25),
            model_version: Some(model_version),
        })
        .collect::<Vec<_>>();
    if stopped {
        steps.push(CompletedEpisodeStep {
            before: state,
            after: state,
            selected_action: stop,
            legal_actions: vec![stop],
            policy_target: vec![1.0],
            root_value: Some(0.25),
            root_search_value: Some(0.25),
            model_version: Some(model_version),
        });
    }
    CompletedEpisodeArtifact {
        root: state,
        final_graph: state,
        final_measure: MeasureSummary {
            graph_hash: state.graph.graph_hash,
            config_hash: MeasureConfigHash::from_bytes([6; 32]),
            measured: true,
            valid: true,
            latency: None,
            scalar_reward: Some(reward),
            failure_code: None,
        },
        stop_selected: stopped,
        search_config_hash: SearchConfigHash::from_bytes([7; 32]),
        steps,
        feature_rows: None,
        terminal_feature_row: None,
    }
}

fn version(seed: u8) -> ModelVersion {
    ModelVersion::from_bytes([seed; 16])
}

fn bind_store(store: &ReplayStore, mode: ReplayDataMode) {
    store
        .ensure_contract(&ReplayContract::unfeaturized(
            mode,
            EngineIdentity::from_context(context(0)),
        ))
        .unwrap();
}

fn context(seed: u8) -> ReplayGraphContext {
    ReplayGraphContext::new(
        PortableGraphId::new(
            GraphHash::from_bytes([seed; 32]),
            EngineId::from_bytes([1; 16]),
            EngineVersion::from_bytes([2; 16]),
        ),
        ActionSetHash::from_bytes([3; 32]),
    )
}
