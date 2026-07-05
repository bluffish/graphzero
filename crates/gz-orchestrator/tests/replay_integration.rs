use gz_engine::{CandidateOptions, EngineResult, GraphEngine, ModelVersion};
use gz_engine_whittle::{WhittleEngine, WhittleEngineConfig, WhittleGraphId};
use gz_eval::{EvalOutput, EvalRequest, EvalResult, Evaluator};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_orchestrator::reference::{
    GreedyReferenceProvider, PolicyReferenceProvider, ReferenceProvider, RootBaselineProvider,
};
use gz_orchestrator::{
    CountedRoots, ReplayBackpressure, ReplayRuntime, RootSource, ThreadedGumbelOrchestrator,
    ThreadedOrchestratorConfig,
};
use gz_replay::{ReplayEpisodeRecord, ReplayReferenceKind, ReplayStore, SampleConfig};
use gz_search::{
    GreedySearch, GreedySearchConfig, GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig,
};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
            "gz-orchestrator-replay-test-{}-{id}",
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

fn search(engine: &WhittleEngine) -> GumbelMcts {
    GumbelMcts::new(GumbelMctsConfig {
        max_steps: 2,
        simulations: NonZeroUsize::new(2).unwrap(),
        max_considered_actions: NonZeroUsize::new(4).unwrap(),
        seed: 7,
        gumbel_scale: 0.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse: false,
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
        seed: 13,
        ..RandomValueEvaluatorConfig::default()
    })
    .unwrap()
}

fn root_providers(engines: &[WhittleEngine]) -> Vec<RootBaselineProvider> {
    engines
        .iter()
        .map(|engine| RootBaselineProvider::new(engine.measure_options()))
        .collect()
}

#[test]
fn root_baseline_replay_appends_every_eligible_episode() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(2);
    let search = search(&engines[0]);
    let providers = root_providers(&engines);
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(2));
    let run = orchestrator
        .run_with_replay(
            vec![roots(3), roots(2)],
            GumbelEpisodeContext::default(),
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
            },
        )
        .unwrap();
    let total = run
        .run
        .lanes
        .iter()
        .map(|lane| lane.episodes.len())
        .sum::<usize>() as u64;
    let row_count = replay_records(&store, run.episodes_appended)
        .iter()
        .map(|record| record.row_count as u64)
        .sum::<u64>();

    assert_eq!(run.episodes_appended + run.episodes_dropped, total);
    assert_eq!(run.episodes_dropped, 0);
    assert_eq!(store.counters().produced_rows, row_count);

    let sample = store
        .sample_rows(SampleConfig {
            batch: NonZeroUsize::new(1).unwrap(),
            window_rows: NonZeroU64::new(store.counters().produced_rows).unwrap(),
            seed: 0,
        })
        .unwrap();
    assert!(sample[0].1.feature_row.is_none());
}

#[test]
fn replay_store_contents_are_deterministic_for_identical_runs() {
    let left = run_root_replay(1, 1, 4);
    let right = run_root_replay(1, 1, 4);

    assert_eq!(left, right);
}

#[test]
fn greedy_reference_labels_are_valid_and_present() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let providers = engines
        .iter()
        .map(|engine| {
            GreedyReferenceProvider::new(GreedySearch::new(GreedySearchConfig {
                max_steps: 2,
                candidate_options: CandidateOptions::default(),
                measure_options: engine.measure_options(),
            }))
        })
        .collect::<Vec<_>>();
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(2));
    let run = orchestrator
        .run_with_replay(
            vec![roots(4)],
            GumbelEpisodeContext::default(),
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
            },
        )
        .unwrap();
    let records = replay_records(&store, run.episodes_appended);
    let labels = records
        .iter()
        .filter_map(|record| record.outcome.value_target)
        .collect::<Vec<_>>();

    assert!(!labels.is_empty());
    assert!(
        labels
            .iter()
            .all(|label| *label == -1.0 || *label == 0.0 || *label == 1.0)
    );
}

#[test]
fn backpressure_gate_allows_consumer_to_drain_and_complete() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(2);
    let search = search(&engines[0]);
    let providers = root_providers(&engines);
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(2));
    let done = AtomicBool::new(false);
    let observations = Mutex::new(Vec::new());
    let max_backlog = NonZeroU64::new(1).unwrap();
    let overshoot_bound = 2 * 2 * 2;

    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !done.load(Ordering::Acquire) {
                let _ = store.sample_rows(SampleConfig {
                    batch: NonZeroUsize::new(1).unwrap(),
                    window_rows: NonZeroU64::new(16).unwrap(),
                    seed: 1,
                });
                let counters = store.counters();
                observations.lock().unwrap().push(
                    counters
                        .produced_rows
                        .saturating_sub(counters.consumed_rows),
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        });

        let run = orchestrator
            .run_with_replay(
                vec![roots(6), roots(6)],
                GumbelEpisodeContext::default(),
                ReplayRuntime {
                    store: &store,
                    providers,
                    backpressure: Some(ReplayBackpressure {
                        max_row_backlog: max_backlog,
                        gate_poll: Duration::from_millis(1),
                    }),
                },
            )
            .unwrap();
        assert_eq!(run.episodes_dropped, 0);
        done.store(true, Ordering::Release);
    });

    assert!(store.counters().produced_rows > 0);
    assert!(
        observations
            .lock()
            .unwrap()
            .iter()
            .all(|backlog| { *backlog <= max_backlog.get() + overshoot_bound })
    );
}

fn run_root_replay(
    lanes: usize,
    workers_per_lane: usize,
    roots_per_lane: u64,
) -> Vec<ReplayEpisodeRecord> {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(lanes);
    let search = search(&engines[0]);
    let providers = root_providers(&engines);
    let orchestrator =
        ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(workers_per_lane));
    let run = orchestrator
        .run_with_replay(
            (0..lanes).map(|_| roots(roots_per_lane)).collect(),
            GumbelEpisodeContext::default(),
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
            },
        )
        .unwrap();

    replay_records(&store, run.episodes_appended)
}

fn replay_records(store: &ReplayStore, count: u64) -> Vec<ReplayEpisodeRecord> {
    (0..count)
        .map(|id| {
            store
                .episode(gz_replay::ReplayEpisodeId::new(id))
                .unwrap()
                .unwrap()
        })
        .collect()
}

/// The engine root as every episode's root, plus the fixed_root hook the
/// policy opponent needs.
struct FixedRoots {
    remaining: u64,
}

impl RootSource<WhittleEngine> for FixedRoots {
    fn next_root(&mut self, engine: &mut WhittleEngine) -> EngineResult<Option<WhittleGraphId>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        Ok(Some(engine.root()))
    }

    fn fixed_root(&mut self, engine: &mut WhittleEngine) -> EngineResult<Option<WhittleGraphId>> {
        Ok(Some(engine.root()))
    }
}

/// Wraps the random evaluator and stamps a model version that switches
/// after a fixed number of evals -- a checkpoint hot-swap in miniature.
struct SwitchingEvaluator {
    inner: RandomValueEvaluator,
    evals: u64,
    switch_after: u64,
}

impl Evaluator for SwitchingEvaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvalRequest],
        out: &mut Vec<EvalOutput>,
    ) -> EvalResult<()> {
        self.inner.evaluate_batch(requests, out)?;
        for output in out.iter_mut() {
            self.evals += 1;
            output.model_version = if self.evals <= self.switch_after {
                ModelVersion::from_bytes([1; 16])
            } else {
                ModelVersion::from_bytes([2; 16])
            };
        }
        Ok(())
    }
}

#[test]
fn policy_reference_refreshes_per_model_version_and_skips_replay() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let episodes = 8;
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        SwitchingEvaluator {
            inner: evaluator(),
            evals: 0,
            switch_after: 12,
        },
        search,
        config(1),
    );
    let run = orchestrator
        .run_with_replay(
            vec![FixedRoots {
                remaining: episodes,
            }],
            GumbelEpisodeContext::default(),
            ReplayRuntime {
                store: &store,
                providers: vec![PolicyReferenceProvider::new()],
                backpressure: None,
            },
        )
        .unwrap();

    // Rollout episodes never reach the store or the counters.
    assert_eq!(run.episodes_appended, episodes);
    assert_eq!(run.episodes_dropped, 0);
    let records = replay_records(&store, run.episodes_appended);
    let row_count = records
        .iter()
        .map(|record| record.row_count as u64)
        .sum::<u64>();
    assert_eq!(store.counters().produced_rows, row_count);

    // The first admission precedes the first completed rollout.
    assert!(records[0].outcome.reference.is_none());

    // Labeled episodes carry the rollout scalar: kind Gumbel, versioned.
    let references = records
        .iter()
        .filter_map(|record| record.outcome.reference.as_ref())
        .collect::<Vec<_>>();
    assert!(!references.is_empty());
    for reference in &references {
        assert_eq!(reference.kind, ReplayReferenceKind::Gumbel);
        assert!(reference.model_version.is_some());
        assert!(reference.search_config_hash.is_some());
        assert!(reference.final_graph.is_some());
    }

    // The mid-run version switch produced a fresh rollout: both versions
    // appear across the run's labels.
    let mut versions = references
        .iter()
        .filter_map(|reference| reference.model_version)
        .collect::<Vec<_>>();
    versions.dedup();
    assert_eq!(
        versions,
        vec![
            ModelVersion::from_bytes([1; 16]),
            ModelVersion::from_bytes([2; 16]),
        ]
    );
}

struct CountingSelfAverage {
    inner: gz_orchestrator::reference::SelfAverageProvider,
    observed: std::sync::Arc<AtomicU64>,
}

impl ReferenceProvider<WhittleEngine> for CountingSelfAverage {
    fn reference(
        &mut self,
        engine: &mut WhittleEngine,
        root: WhittleGraphId,
    ) -> EngineResult<Option<gz_orchestrator::reference::Reference<WhittleGraphId>>> {
        self.inner.reference(engine, root)
    }

    fn observe(&mut self, learner_reward: f32) {
        self.observed.fetch_add(1, Ordering::Relaxed);
        ReferenceProvider::<WhittleEngine>::observe(&mut self.inner, learner_reward);
    }
}

#[test]
fn self_average_reference_labels_after_the_first_episode() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let observed = std::sync::Arc::new(AtomicU64::new(0));
    let providers = vec![CountingSelfAverage {
        inner: gz_orchestrator::reference::SelfAverageProvider::new(0.9),
        observed: observed.clone(),
    }];
    // workers_per_lane = 1 makes admission order deterministic: exactly the
    // first episode is admitted before any completion can seed the EMA.
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));
    let run = orchestrator
        .run_with_replay(
            vec![roots(5)],
            GumbelEpisodeContext::default(),
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 5);
    assert_eq!(observed.load(Ordering::Relaxed), 5);

    let mut labeled = 0;
    for id in 0..5 {
        let record = store
            .episode(gz_replay::ReplayEpisodeId::new(id))
            .unwrap()
            .unwrap();
        let value_target = record.outcome.value_target;
        if id == 0 {
            assert_eq!(value_target, None, "first admission has no EMA yet");
            assert!(record.outcome.reference.is_none());
        } else if let Some(target) = value_target {
            assert!(target == 1.0 || target == 0.0 || target == -1.0);
            let reference = record.outcome.reference.unwrap();
            assert_eq!(reference.kind, gz_replay::ReplayReferenceKind::SelfAverage);
            assert!(reference.final_graph.is_none());
            labeled += 1;
        }
    }
    assert!(labeled >= 1, "later episodes must be labeled");
}
