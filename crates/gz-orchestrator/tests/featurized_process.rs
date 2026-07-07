use gz_engine::{CandidateOptions, EngineResult, GraphEngine};
use gz_engine_whittle::{
    WhittleCandidateId, WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleGraphId,
};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_eval_service::{
    EvaluatorProcess, EvaluatorProcessConfig, FeatureEvalBackend, Hello, ProcessBackend,
    STUB_MODEL_VERSION, StubBackend,
};
use gz_orchestrator::{
    CountedRoots, FeaturizedRuntime, ThreadedGumbelOrchestrator, ThreadedOrchestratorConfig,
};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

type Roots = CountedRoots<fn(&mut WhittleEngine) -> EngineResult<WhittleGraphId>>;

static NEXT_SOCKET: AtomicU64 = AtomicU64::new(0);

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
        seed: 17,
        gumbel_scale: 0.0,
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

fn config() -> ThreadedOrchestratorConfig {
    ThreadedOrchestratorConfig {
        workers_per_lane: NonZeroUsize::new(2).unwrap(),
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
fn process_stub_selfplay_matches_in_process_stub() {
    require_numpy();
    let stub = run_with_backend(StubBackend);
    let (mut process, backend) = process_backend();
    let process_run = run_with_backend(backend);

    assert_eq!(process_run, stub);
    for lane in &process_run.lanes {
        for episode in &lane.episodes {
            for step in &episode.episode.steps {
                assert_eq!(step.model_version, STUB_MODEL_VERSION);
            }
        }
    }
    assert_child_exits(&mut process);
}

fn run_with_backend<B>(
    backend: B,
) -> gz_orchestrator::ThreadedRun<WhittleGraphId, WhittleCandidateId>
where
    B: FeatureEvalBackend + Send,
{
    let engines = engines(2);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config());

    orchestrator
        .run_featurized(
            vec![roots(3), roots(3)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![backend],
            },
        )
        .unwrap()
}

fn process_backend() -> (EvaluatorProcess, ProcessBackend) {
    let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
    let extractor = WhittleFeatureExtractor::new(&engine);
    let hello = Hello::new(
        extractor.schema().hash(),
        config().max_batch.get() as u32,
        engine.engine_id(),
        engine.engine_version(),
        engine.action_set_hash(),
    );
    let mut process = EvaluatorProcess::spawn(EvaluatorProcessConfig {
        working_dir: python_dir(),
        socket_path: temp_socket(),
        ready_timeout: Duration::from_secs(10),
        io_timeout: Duration::from_secs(10),
        ..EvaluatorProcessConfig::default()
    })
    .unwrap_or_else(|error| {
        panic!("failed to spawn Python evaluator: {error}; requires python3 + numpy")
    });
    let backend = process.connect(&hello).unwrap_or_else(|error| {
        panic!("failed to connect Python evaluator: {error}; requires python3 + numpy")
    });
    (process, backend)
}

fn temp_socket() -> PathBuf {
    let id = NEXT_SOCKET.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "gz-orchestrator-featurized-process-{}-{id}.sock",
        std::process::id()
    ))
}

fn python_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python")
}

fn assert_child_exits(process: &mut EvaluatorProcess) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match process.try_wait().unwrap() {
            Some(status) => {
                assert!(status.success(), "Python evaluator exited with {status}");
                return;
            }
            None if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            None => panic!("Python evaluator did not exit after backend dropped"),
        }
    }
}

fn require_numpy() {
    let status = std::process::Command::new("python3")
        .arg("-c")
        .arg("import numpy")
        .status()
        .expect("failed to run python3; featurized process tests require python3 + numpy");
    assert!(
        status.success(),
        "python3 -c 'import numpy' failed; featurized process tests require python3 + numpy"
    );
}
