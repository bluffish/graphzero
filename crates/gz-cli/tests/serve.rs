use gz_cli::selfplay::{EvaluatorMode, ReferenceMode, SelfplayConfig, run as run_selfplay};
use gz_cli::serve::{ReplayServeConfig, SAMPLE_PROTOCOL_VERSION, run_one};
use gz_features::{
    ENCODING_VERSION, FeatureBatchView, TrainingTargetsView, decode_feature_schema_config,
};
use gz_replay::ReplayStore;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("gz-cli-serve-test-{}-{id}", std::process::id()));
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
fn replay_serve_returns_feature_batch_and_targets() {
    let dir = TestDir::new();
    let summary = run_selfplay(SelfplayConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        episodes: 2,
        lanes: 1,
        workers_per_lane: 2,
        reference: ReferenceMode::Root,
        reference_ema_decay: 0.99,
        seed: 5,
        max_steps: 2,
        simulations: 2,
        max_batch: 2,
        evaluator: EvaluatorMode::Stub,
        python_dir: None,
        checkpoint_dir: None,
        eval_device: None,
        eval_poll_interval: None,
        serve_socket: None,
        serve_max_batch: 512,
        replay_backlog: None,
    })
    .unwrap();
    let expected_schema_config = ReplayStore::open(dir.path())
        .unwrap()
        .feature_schema()
        .unwrap();
    let socket = dir.path().join("sample.sock");
    let server_config = ReplayServeConfig {
        replay_dir: dir.path().to_path_buf(),
        socket: socket.clone(),
        max_batch: 2,
    };
    let server = std::thread::spawn(move || run_one(server_config));
    let mut stream = connect_retry(&socket);

    let mut hello = Vec::new();
    hello.extend_from_slice(&SAMPLE_PROTOCOL_VERSION.to_le_bytes());
    hello.extend_from_slice(&ENCODING_VERSION.to_le_bytes());
    write_frame(&mut stream, 1, &[&hello]);
    let (frame_type, ack) = read_frame(&mut stream);
    assert_eq!(frame_type, 2);
    assert_eq!(
        u32::from_le_bytes(ack[0..4].try_into().unwrap()),
        SAMPLE_PROTOCOL_VERSION
    );
    assert_eq!(u32::from_le_bytes(ack[36..40].try_into().unwrap()), 2);
    assert_eq!(
        u64::from_le_bytes(ack[40..48].try_into().unwrap()),
        summary.rows_produced
    );
    let schema_config = decode_feature_schema_config(&ack[48..]).unwrap();
    assert_eq!(Some(schema_config), expected_schema_config);

    let mut sample = Vec::new();
    sample.extend_from_slice(&1u32.to_le_bytes());
    sample.extend_from_slice(&summary.rows_produced.to_le_bytes());
    sample.extend_from_slice(&123u64.to_le_bytes());
    write_frame(&mut stream, 3, &[&sample]);
    let (frame_type, result) = read_frame(&mut stream);
    assert_eq!(frame_type, 4);
    let gzfb_len = u32::from_le_bytes(result[0..4].try_into().unwrap()) as usize;
    let gzfb = &result[4..4 + gzfb_len];
    let gzft = &result[4 + gzfb_len..];
    let batch = FeatureBatchView::parse(gzfb).unwrap();
    let targets = TrainingTargetsView::parse(gzft).unwrap();

    assert_eq!(batch.batch_capacity, 2);
    assert_eq!(batch.row_count, 1);
    assert_eq!(targets.capacity, 2);
    assert_eq!(targets.row_count, 1);
    assert_eq!(targets.max_actions, batch.max_actions);
    assert_eq!(
        targets.policy.len(),
        (targets.capacity * targets.max_actions) as usize
    );

    drop(stream);
    server.join().unwrap().unwrap();
}

#[test]
fn replay_serve_rejects_featureless_store() {
    let dir = TestDir::new();
    run_selfplay(SelfplayConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        episodes: 1,
        lanes: 1,
        workers_per_lane: 1,
        reference: ReferenceMode::Root,
        reference_ema_decay: 0.99,
        seed: 7,
        max_steps: 1,
        simulations: 1,
        max_batch: 1,
        evaluator: EvaluatorMode::Random,
        python_dir: None,
        checkpoint_dir: None,
        eval_device: None,
        eval_poll_interval: None,
        serve_socket: None,
        serve_max_batch: 512,
        replay_backlog: None,
    })
    .unwrap();

    let error = run_one(ReplayServeConfig {
        replay_dir: dir.path().to_path_buf(),
        socket: dir.path().join("sample.sock"),
        max_batch: 1,
    })
    .unwrap_err();

    assert!(error.contains("store was not produced by featurized selfplay"));
}

fn connect_retry(path: &Path) -> UnixStream {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match UnixStream::connect(path) {
            Ok(stream) => return stream,
            Err(error) if Instant::now() < deadline => {
                assert!(
                    matches!(
                        error.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                    ),
                    "{error}"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("{error}"),
        }
    }
}

fn write_frame(stream: &mut UnixStream, frame_type: u8, parts: &[&[u8]]) {
    let body_len = 1 + parts.iter().map(|part| part.len()).sum::<usize>();
    stream.write_all(&(body_len as u32).to_le_bytes()).unwrap();
    stream.write_all(&[frame_type]).unwrap();
    for part in parts {
        stream.write_all(part).unwrap();
    }
}

fn read_frame(stream: &mut UnixStream) -> (u8, Vec<u8>) {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).unwrap();
    let body_len = u32::from_le_bytes(len) as usize;
    let mut body = vec![0; body_len];
    stream.read_exact(&mut body).unwrap();
    (body[0], body[1..].to_vec())
}

// ---- in-process sample service (shared store, live producer) ----

use gz_cli::serve::run_shared;
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleGraphGenerator,
    WhittleGraphGeneratorConfig,
};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_eval_service::StubBackend;
use gz_orchestrator::reference::RootBaselineProvider;
use gz_orchestrator::{
    CountedRoots, FeaturizedRuntime, ReplayBackpressure, ReplayRuntime, ThreadedGumbelOrchestrator,
    ThreadedOrchestratorConfig,
};
use gz_search::{GumbelMcts, GumbelMctsConfig};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn live_setup(
    dir: &TestDir,
    socket: &Path,
) -> (
    Arc<ReplayStore>,
    Vec<WhittleEngine>,
    Vec<WhittleFeatureExtractor>,
    GumbelMcts,
) {
    let store = Arc::new(ReplayStore::open(dir.path()).unwrap());
    // Engine capacity must match the generator config, as the CLI does.
    let generator = WhittleGraphGeneratorConfig::default();
    let engines = vec![
        WhittleEngine::new(WhittleEngineConfig {
            root: gz_engine_whittle::WhittleRoot::Input {
                arity: generator.arity,
                capacity: generator.capacity,
                input_index: 0,
            },
            ..WhittleEngineConfig::default()
        })
        .unwrap(),
    ];
    let extractors: Vec<_> = engines.iter().map(WhittleFeatureExtractor::new).collect();
    store
        .ensure_feature_schema(extractors[0].schema().config())
        .unwrap();
    let search = GumbelMcts::new(GumbelMctsConfig {
        max_steps: 2,
        simulations: NonZeroUsize::new(2).unwrap(),
        max_considered_actions: NonZeroUsize::new(4).unwrap(),
        seed: 9,
        gumbel_scale: 0.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        candidate_options: gz_engine::CandidateOptions {
            max_candidates: Some(255),
            deterministic_order: true,
        },
        measure_options: engines[0].measure_options(),
    });

    let serve_store = store.clone();
    let serve_socket = socket.to_path_buf();
    std::thread::spawn(move || {
        let _ = run_shared(serve_store, serve_socket, 8);
    });

    (store, engines, extractors, search)
}

fn generated_roots(
    count: u64,
    seed: u64,
) -> CountedRoots<
    impl FnMut(&mut WhittleEngine) -> gz_engine::EngineResult<gz_engine_whittle::WhittleGraphId>,
> {
    let mut generator =
        WhittleGraphGenerator::from_seed(WhittleGraphGeneratorConfig::default(), seed);
    CountedRoots::new(count, move |engine: &mut WhittleEngine| {
        generator
            .sample_into(engine)
            .map(|generated| generated.graph)
    })
}

/// Connects, handshakes, and blocks until the store reports rows.
fn wait_for_rows(socket: &Path) -> UnixStream {
    loop {
        let mut stream = connect_retry(socket);
        let mut hello = Vec::new();
        hello.extend_from_slice(&SAMPLE_PROTOCOL_VERSION.to_le_bytes());
        hello.extend_from_slice(&ENCODING_VERSION.to_le_bytes());
        write_frame(&mut stream, 1, &[&hello]);
        let (frame_type, ack) = read_frame(&mut stream);
        assert_eq!(frame_type, 2);
        let produced = u64::from_le_bytes(ack[40..48].try_into().unwrap());
        if produced > 0 {
            return stream;
        }
        drop(stream);
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn sample_once(stream: &mut UnixStream, batch: u32, seed: u64) {
    let mut sample = Vec::new();
    sample.extend_from_slice(&batch.to_le_bytes());
    sample.extend_from_slice(&u64::MAX.to_le_bytes());
    sample.extend_from_slice(&seed.to_le_bytes());
    write_frame(stream, 3, &[&sample]);
    let (frame_type, _) = read_frame(stream);
    assert_eq!(frame_type, 4);
}

#[test]
fn in_process_sample_service_serves_during_production() {
    let dir = TestDir::new();
    let socket = dir.path().join("live.sock");
    let (store, engines, extractors, search) = live_setup(&dir, &socket);
    let done = Arc::new(AtomicBool::new(false));

    let sampler_done = done.clone();
    let sampler_socket = socket.clone();
    let sampler = std::thread::spawn(move || {
        let mut stream = wait_for_rows(&sampler_socket);
        let mut before_done = 0u64;
        let mut total = 0u64;
        while !sampler_done.load(Ordering::Acquire) {
            sample_once(&mut stream, 1, total);
            total += 1;
            if !sampler_done.load(Ordering::Acquire) {
                before_done += 1;
            }
        }
        (before_done, total)
    });

    let providers = vec![RootBaselineProvider::new(engines[0].measure_options())];
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        RandomValueEvaluator::new(RandomValueEvaluatorConfig::default()).unwrap(),
        search,
        ThreadedOrchestratorConfig {
            workers_per_lane: NonZeroUsize::new(2).unwrap(),
            max_batch: NonZeroUsize::new(2).unwrap(),
            flush_after: Duration::from_millis(1),
        },
    );
    let run = orchestrator
        .run_featurized_with_replay(
            vec![generated_roots(24, 3)],
            gz_search::GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backend: StubBackend,
            },
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
            },
        )
        .unwrap();
    done.store(true, Ordering::Release);
    let (before_done, total) = sampler.join().unwrap();

    assert_eq!(run.episodes_appended, 24);
    assert!(before_done >= 1, "no sample overlapped production");
    assert!(total >= before_done);
    let counters = store.counters();
    assert!(counters.consumed_rows >= total);
    assert!(counters.produced_rows > 0);
}

#[test]
fn live_backpressure_gates_production_until_the_consumer_drains() {
    let dir = TestDir::new();
    let socket = dir.path().join("gate.sock");
    let (store, engines, extractors, search) = live_setup(&dir, &socket);
    let done = Arc::new(AtomicBool::new(false));

    let consumer_done = done.clone();
    let consumer_socket = socket.clone();
    let consumer = std::thread::spawn(move || {
        let mut stream = wait_for_rows(&consumer_socket);
        let mut seed = 0u64;
        while !consumer_done.load(Ordering::Acquire) {
            sample_once(&mut stream, 2, seed);
            seed += 1;
        }
    });

    // 12 episodes at ~2 rows each against a 4-row backlog cap: production
    // cannot finish unless the consumer keeps draining the backlog.
    let providers = vec![RootBaselineProvider::new(engines[0].measure_options())];
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        RandomValueEvaluator::new(RandomValueEvaluatorConfig::default()).unwrap(),
        search,
        ThreadedOrchestratorConfig {
            workers_per_lane: NonZeroUsize::new(1).unwrap(),
            max_batch: NonZeroUsize::new(1).unwrap(),
            flush_after: Duration::from_millis(1),
        },
    );
    let (result_tx, result_rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let store = &store;
        scope.spawn(move || {
            let run = orchestrator.run_featurized_with_replay(
                vec![generated_roots(12, 5)],
                gz_search::GumbelEpisodeContext::default(),
                FeaturizedRuntime {
                    extractors,
                    backend: StubBackend,
                },
                ReplayRuntime {
                    store,
                    providers,
                    backpressure: Some(ReplayBackpressure {
                        max_row_backlog: NonZeroU64::new(4).unwrap(),
                        gate_poll: Duration::from_millis(1),
                    }),
                },
            );
            result_tx.send(run).unwrap();
        });
        let run = result_rx
            .recv_timeout(Duration::from_secs(120))
            .expect("gated production did not finish; backpressure never released")
            .unwrap();
        assert_eq!(run.episodes_appended, 12);
    });
    done.store(true, Ordering::Release);
    consumer.join().unwrap();

    let counters = store.counters();
    assert!(counters.consumed_rows > 0);
    assert!(counters.produced_rows >= 12);
}

#[test]
fn replay_serve_reacks_a_repeated_hello_on_a_live_connection() {
    let dir = TestDir::new();
    let socket = dir.path().join("rehello.sock");
    let (store, engines, extractors, search) = live_setup(&dir, &socket);

    let providers = vec![RootBaselineProvider::new(engines[0].measure_options())];
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        RandomValueEvaluator::new(RandomValueEvaluatorConfig::default()).unwrap(),
        search,
        ThreadedOrchestratorConfig {
            workers_per_lane: NonZeroUsize::new(1).unwrap(),
            max_batch: NonZeroUsize::new(1).unwrap(),
            flush_after: Duration::from_millis(1),
        },
    );
    orchestrator
        .run_featurized_with_replay(
            vec![generated_roots(2, 3)],
            gz_search::GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backend: StubBackend,
            },
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
            },
        )
        .unwrap();

    let mut stream = wait_for_rows(&socket);
    sample_once(&mut stream, 1, 7);

    let mut hello = Vec::new();
    hello.extend_from_slice(&SAMPLE_PROTOCOL_VERSION.to_le_bytes());
    hello.extend_from_slice(&ENCODING_VERSION.to_le_bytes());
    write_frame(&mut stream, 1, &[&hello]);
    let (frame_type, ack) = read_frame(&mut stream);
    assert_eq!(frame_type, 2);
    let produced = u64::from_le_bytes(ack[40..48].try_into().unwrap());
    assert!(produced > 0);

    // The connection keeps sampling after the re-ack.
    sample_once(&mut stream, 1, 8);
}
