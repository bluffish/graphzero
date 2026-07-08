use gz_engine::{CandidateOptions, EngineResult, GraphEngine};
use gz_engine_whittle::{
    WhittleCandidateId, WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleGraphId,
};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_eval_service::StubBackend;
use gz_features::{
    FeatureExtractor, FeatureResult, FeatureRow, FeatureSchema, FeatureSchemaConfig,
    PositionFeatures, decode_feature_row,
};
use gz_orchestrator::reference::{ReferenceProvider, RootBaselineProvider};
use gz_orchestrator::{
    CountedRoots, FeaturizedRuntime, ReplayRuntime, ThreadedGumbelOrchestrator,
    ThreadedOrchestratorConfig,
};
use gz_replay::{ReplayStore, SampleConfig};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

type Roots = CountedRoots<fn(&mut WhittleEngine) -> EngineResult<WhittleGraphId>>;

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "gz-orchestrator-featurized-test-{}-{id}",
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

fn root_factory(engine: &mut WhittleEngine) -> EngineResult<WhittleGraphId> {
    Ok(engine.root())
}

fn roots(count: u64) -> Roots {
    CountedRoots::new(count, root_factory)
}

fn engines(count: usize) -> Vec<WhittleEngine> {
    (0..count)
        .map(|_| WhittleEngine::new(WhittleEngineConfig::default()).unwrap())
        .collect()
}

fn extractors(engines: &[WhittleEngine]) -> Vec<WhittleFeatureExtractor> {
    engines.iter().map(WhittleFeatureExtractor::new).collect()
}

fn search(engine: &WhittleEngine) -> GumbelMcts {
    GumbelMcts::new(GumbelMctsConfig {
        max_steps: 2,
        simulations: NonZeroUsize::new(2).unwrap(),
        max_considered_actions: NonZeroUsize::new(4).unwrap(),
        seed: 11,
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
        measure_options: engine.measure_options(),
    })
}

fn config(workers_per_lane: usize) -> ThreadedOrchestratorConfig {
    ThreadedOrchestratorConfig {
        workers_per_lane: NonZeroUsize::new(workers_per_lane).unwrap(),
        max_batch: NonZeroUsize::new(8).unwrap(),
        flush_after: Duration::from_millis(20),
    }
}

fn evaluator() -> RandomValueEvaluator {
    RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: 0,
        ..RandomValueEvaluatorConfig::default()
    })
    .unwrap()
}

#[test]
fn featurized_selfplay_is_deterministic() {
    let left = run_stub(2, 2, 3);
    let right = run_stub(2, 2, 3);

    assert_eq!(left, right);
    assert_eq!(left.lanes.len(), 2);
    assert!(left.lanes.iter().all(|lane| lane.episodes.len() == 3));
    assert!(!left.batch_sizes.is_empty());
}

#[test]
fn featurized_replay_appends_rows() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(2);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let feature_config = extractors[0].schema().config().clone();
    let providers = engines
        .iter()
        .map(|engine| RootBaselineProvider::new(engine.measure_options()))
        .collect::<Vec<_>>();
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(2));
    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(2), roots(2)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
            },
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
                length_tiebreak: false,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_dropped, 0);
    assert_eq!(run.episodes_appended, 4);
    assert!(store.counters().produced_rows > 0);
    assert_eq!(
        store.feature_schema().unwrap(),
        Some(feature_config.clone())
    );

    let sample = store
        .sample_rows(SampleConfig {
            batch: NonZeroUsize::new(store.counters().produced_rows as usize).unwrap(),
            window_rows: std::num::NonZeroU64::new(store.counters().produced_rows).unwrap(),
            seed: 0,
        })
        .unwrap();
    for (episode_id, row) in sample {
        let record = store.episode(episode_id).unwrap().unwrap();
        let reference = record.outcome.reference.as_ref().unwrap();
        let feature_row = decode_feature_row(row.feature_row.as_ref().unwrap()).unwrap();
        assert_eq!(feature_row.actions.len(), row.legal_actions.len());
        assert!(feature_row.position.opponent_present);
        assert_eq!(
            feature_row.position.opponent_reward,
            reference.reward / feature_config.opponent_reward_scale
        );
    }
}

/// Never supplies a reference and never expects one: rows are stored
/// unlabeled instead of dropped (the reference=none pipeline shape).
struct NoReferenceProvider;

impl<E: GraphEngine> ReferenceProvider<E> for NoReferenceProvider {
    fn reference(
        &mut self,
        _engine: &mut E,
        _root: E::Graph,
    ) -> EngineResult<Option<gz_orchestrator::reference::Reference>> {
        Ok(None)
    }

    fn expects_reference(&self) -> bool {
        false
    }
}

#[test]
fn featurized_replay_unlabeled_rows_have_no_opponent_scalar() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));
    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![NoReferenceProvider],
                backpressure: None,
                length_tiebreak: false,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 1);
    let sample = store
        .sample_rows(SampleConfig {
            batch: NonZeroUsize::new(store.counters().produced_rows as usize).unwrap(),
            window_rows: std::num::NonZeroU64::new(store.counters().produced_rows).unwrap(),
            seed: 0,
        })
        .unwrap();
    for (episode_id, row) in sample {
        let record = store.episode(episode_id).unwrap().unwrap();
        let feature_row = decode_feature_row(row.feature_row.as_ref().unwrap()).unwrap();
        assert!(record.outcome.reference.is_none());
        assert!(!feature_row.position.opponent_present);
        assert_eq!(feature_row.position.opponent_reward, 0.0);
    }
}

#[test]
fn featurized_replay_schema_error_includes_replay_detail() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let mut stored_config = extractors[0].schema().config().clone();
    stored_config.name = "stored-mismatch".to_owned();
    store.ensure_feature_schema(&stored_config).unwrap();
    let providers = engines
        .iter()
        .map(|engine| RootBaselineProvider::new(engine.measure_options()))
        .collect::<Vec<_>>();
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));

    let error = orchestrator
        .run_featurized_with_replay(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
            },
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
                length_tiebreak: false,
            },
        )
        .unwrap_err();

    assert!(error.to_string().contains("invalid replay record"));
}

#[test]
fn featurized_rejects_lane_and_schema_mismatches() {
    let engine_set = engines(2);
    let gumbel = search(&engine_set[0]);
    let mut extractor_set = extractors(&engine_set);
    extractor_set.pop();
    let orchestrator = ThreadedGumbelOrchestrator::new(engine_set, evaluator(), gumbel, config(2));
    let error = orchestrator
        .run_featurized(
            vec![roots(1), roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors: extractor_set,
                backends: vec![StubBackend],
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("lane count mismatch"));

    let engine_set = engines(2);
    let gumbel = search(&engine_set[0]);
    let mut extractor_set = extractors(&engine_set);
    let schema = FeatureSchema::new(FeatureSchemaConfig {
        name: "mismatch".to_owned(),
        ..extractor_set[0].schema().config().clone()
    })
    .unwrap();
    let wrapped = vec![
        WrappedExtractor::matching(extractor_set.remove(0)),
        WrappedExtractor::with_schema(extractor_set.remove(0), schema),
    ];
    let orchestrator = ThreadedGumbelOrchestrator::new(engine_set, evaluator(), gumbel, config(2));
    let error = orchestrator
        .run_featurized(
            vec![roots(1), roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors: wrapped,
                backends: vec![StubBackend],
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("feature schema mismatch"));
}

#[test]
fn featurized_extraction_failure_aborts_run() {
    let engines = engines(1);
    let search = search(&engines[0]);
    let extractors = vec![FailingExtractor {
        inner: WhittleFeatureExtractor::new(&engines[0]),
    }];
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));

    let error = orchestrator
        .run_featurized(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
            },
        )
        .unwrap_err();

    assert!(error.to_string().contains("feature extraction failed"));
}

fn run_stub(
    lanes: usize,
    workers_per_lane: usize,
    roots_per_lane: u64,
) -> gz_orchestrator::ThreadedRun<WhittleGraphId, WhittleCandidateId> {
    let engines = engines(lanes);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let orchestrator =
        ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(workers_per_lane));
    orchestrator
        .run_featurized(
            (0..lanes).map(|_| roots(roots_per_lane)).collect(),
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
            },
        )
        .unwrap()
}

struct WrappedExtractor {
    inner: WhittleFeatureExtractor,
    schema: FeatureSchema,
}

impl WrappedExtractor {
    fn matching(inner: WhittleFeatureExtractor) -> Self {
        let schema = inner.schema().clone();
        Self { inner, schema }
    }

    fn with_schema(inner: WhittleFeatureExtractor, schema: FeatureSchema) -> Self {
        Self { inner, schema }
    }
}

impl FeatureExtractor<WhittleEngine> for WrappedExtractor {
    fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    fn extract(
        &mut self,
        engine: &WhittleEngine,
        graph: WhittleGraphId,
        candidates: &[WhittleCandidateId],
        position: PositionFeatures,
    ) -> FeatureResult<FeatureRow> {
        self.inner.extract(engine, graph, candidates, position)
    }
}

struct FailingExtractor {
    inner: WhittleFeatureExtractor,
}

impl FeatureExtractor<WhittleEngine> for FailingExtractor {
    fn schema(&self) -> &FeatureSchema {
        self.inner.schema()
    }

    fn extract(
        &mut self,
        _engine: &WhittleEngine,
        _graph: WhittleGraphId,
        _candidates: &[WhittleCandidateId],
        _position: PositionFeatures,
    ) -> FeatureResult<FeatureRow> {
        Err(gz_features::FeatureError::InvalidRow("forced failure"))
    }
}
