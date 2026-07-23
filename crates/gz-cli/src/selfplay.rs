use gz_engine::{CandidateOptions, EngineIdentity, EngineResult, GraphEngine, ModelVersion};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleFeatureExtractorConfig,
    WhittleGraphGenerator, WhittleGraphGeneratorConfig, WhittleGraphId, WhittleRoot,
};
use gz_eval_service::{
    EvaluatorProcess, EvaluatorProcessConfig, Hello, STUB_MODEL_VERSION, StubBackend,
};
use gz_features::FeatureSchemaHash;
use gz_measure_agent::{named_test_profile_hash, submission as whittle_measure_submission};
use gz_measurer::MeasureLedgerSnapshot;
use gz_orchestrator::{
    AdmissionSmoothingConfig, FeaturizedRuntime, RemoteMeasurementRuntime, ReplayBackpressure,
    ReplayRuntime, RootSource, ThreadedGumbelOrchestrator, ThreadedOrchestratorConfig,
};
use gz_replay::{ReplayContract, ReplayCounters, ReplayDataMode, ReplayEpisodeId, ReplayStore};
use gz_search::{GumbelMcts, GumbelMctsConfig, GumbelValueMode};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use crate::remote_measure::{RemoteCoordinator, RemoteMeasureConfig};

const WHITTLE_FEATURE_MAX_ENGINE_CANDIDATES: usize = 255;

#[derive(Clone, Debug)]
pub struct SelfplayConfig {
    pub replay_dir: Option<PathBuf>,
    pub episodes: u64,
    pub lanes: usize,
    pub workers_per_lane: usize,
    pub seed: u64,
    pub max_steps: usize,
    pub simulations: usize,
    pub max_considered: usize,
    pub gumbel_scale: f32,
    pub c_visit: f32,
    pub c_scale: f32,
    pub gumbel_noise_overlap: f32,
    pub tree_reuse: bool,
    pub max_candidates: usize,
    pub max_batch: usize,
    pub evaluator: EvaluatorMode,
    pub python_dir: Option<PathBuf>,
    pub checkpoint_dir: Option<PathBuf>,
    pub checkpoint_pointer: Option<String>,
    pub eval_device: Option<String>,
    pub eval_poll_interval: Option<f32>,
    pub serve_socket: Option<PathBuf>,
    pub serve_max_batch: usize,
    pub replay_backlog: Option<u64>,
    pub replay_retain: Option<u64>,
    pub position_features: bool,
    pub no_backtrack: bool,
    pub mask_stop: bool,
    pub eval_processes: usize,
    pub admission_stagger_ms: u64,
    pub admission_smoothing: bool,
    pub remote_measure: RemoteMeasureConfig,
}

#[derive(Clone, Debug)]
pub struct ReplayInitConfig {
    pub replay_dir: Option<PathBuf>,
    pub max_candidates: usize,
    pub mask_stop: bool,
}

impl Default for ReplayInitConfig {
    fn default() -> Self {
        Self {
            replay_dir: None,
            max_candidates: WHITTLE_FEATURE_MAX_ENGINE_CANDIDATES,
            mask_stop: false,
        }
    }
}

impl ReplayInitConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_dir.is_none() {
            return Err("missing required --replay-dir".to_owned());
        }
        if self.max_candidates == 0 {
            return Err("--max-candidates must be greater than zero".to_owned());
        }
        feature_max_actions(self.max_candidates)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayInitSummary {
    pub feature_schema_hash: FeatureSchemaHash,
    pub max_actions: u32,
}

impl Default for SelfplayConfig {
    fn default() -> Self {
        Self {
            replay_dir: None,
            episodes: 16,
            lanes: 2,
            workers_per_lane: 8,
            seed: 0,
            max_steps: 8,
            simulations: 8,
            max_considered: 8,
            gumbel_scale: 1.0,
            c_visit: 50.0,
            c_scale: 1.0,
            gumbel_noise_overlap: 0.5,
            tree_reuse: true,
            max_candidates: WHITTLE_FEATURE_MAX_ENGINE_CANDIDATES,
            max_batch: 16,
            evaluator: EvaluatorMode::Stub,
            python_dir: None,
            checkpoint_dir: None,
            checkpoint_pointer: None,
            eval_device: None,
            eval_poll_interval: None,
            serve_socket: None,
            serve_max_batch: 512,
            replay_backlog: None,
            replay_retain: None,
            position_features: true,
            no_backtrack: true,
            mask_stop: false,
            eval_processes: 1,
            admission_stagger_ms: 0,
            admission_smoothing: false,
            remote_measure: RemoteMeasureConfig::default(),
        }
    }
}

impl SelfplayConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_dir.is_none() {
            return Err("missing required --replay-dir".to_owned());
        }
        for (value, name) in [
            (self.lanes, "--lanes"),
            (self.workers_per_lane, "--workers-per-lane"),
            (self.max_steps, "--max-steps"),
            (self.simulations, "--simulations"),
            (self.max_considered, "--max-considered"),
            (self.max_candidates, "--max-candidates"),
            (self.max_batch, "--max-batch"),
            (self.serve_max_batch, "--serve-max-batch"),
            (self.eval_processes, "--eval-processes"),
        ] {
            if value == 0 {
                return Err(format!("{name} must be greater than zero"));
            }
        }
        feature_max_actions(self.max_candidates)?;
        if !self.gumbel_scale.is_finite() || self.gumbel_scale < 0.0 {
            return Err("--gumbel-scale must be finite and non-negative".to_owned());
        }
        if !self.gumbel_noise_overlap.is_finite() || self.gumbel_noise_overlap >= 1.0 {
            return Err("--gumbel-noise-overlap must be < 1 (negative disables)".to_owned());
        }
        if !self.c_visit.is_finite() || self.c_visit < 0.0 {
            return Err("--c-visit must be finite and non-negative".to_owned());
        }
        if !self.c_scale.is_finite() || self.c_scale < 0.0 {
            return Err("--c-scale must be finite and non-negative".to_owned());
        }
        if !self.mask_stop && !self.position_features {
            return Err(
                "STOP-enabled symmetric selfplay requires --position-features true".to_owned(),
            );
        }
        if self.evaluator == EvaluatorMode::Torch && self.checkpoint_dir.is_none() {
            return Err("--evaluator torch requires --checkpoint-dir".to_owned());
        }
        if self.evaluator != EvaluatorMode::Torch {
            if self.checkpoint_dir.is_some() {
                return Err("--checkpoint-dir requires --evaluator torch".to_owned());
            }
            if self.checkpoint_pointer.is_some() {
                return Err("--checkpoint-pointer requires --evaluator torch".to_owned());
            }
            if self.eval_device.is_some() {
                return Err("--eval-device requires --evaluator torch".to_owned());
            }
            if self.eval_poll_interval.is_some() {
                return Err("--eval-poll-interval requires --evaluator torch".to_owned());
            }
        }
        if let Some(interval) = self.eval_poll_interval
            && (!interval.is_finite() || interval < 0.0)
        {
            return Err("--eval-poll-interval must be zero (disabled) or positive".to_owned());
        }
        if let Some(pointer) = &self.checkpoint_pointer
            && (pointer.is_empty()
                || PathBuf::from(pointer)
                    .file_name()
                    .and_then(|name| name.to_str())
                    != Some(pointer.as_str()))
        {
            return Err("--checkpoint-pointer must be a checkpoint file name".to_owned());
        }
        if self.serve_socket.is_some() && self.episodes != 0 {
            return Err("--serve-socket requires --episodes 0 (unbounded)".to_owned());
        }
        if self.episodes == 0 && self.serve_socket.is_none() {
            return Err("--episodes 0 (unbounded) requires --serve-socket".to_owned());
        }
        if self.eval_processes > self.lanes {
            return Err("--eval-processes cannot exceed --lanes".to_owned());
        }
        if self.eval_processes > 1
            && !matches!(
                self.evaluator,
                EvaluatorMode::ProcessStub | EvaluatorMode::Torch
            )
        {
            return Err("--eval-processes requires --evaluator process-stub|torch".to_owned());
        }
        if self.admission_stagger_ms > u64::MAX / 1_000_000 {
            return Err("--admission-stagger-ms is too large".to_owned());
        }
        if self.admission_smoothing && self.admission_stagger_ms != 0 {
            return Err(
                "--admission-smoothing and --admission-stagger-ms are mutually exclusive"
                    .to_owned(),
            );
        }
        if self.replay_backlog == Some(0) {
            return Err("--replay-backlog must be greater than zero".to_owned());
        }
        if self.replay_retain == Some(0) {
            return Err("--replay-retain must be greater than zero".to_owned());
        }
        self.remote_measure.validate()?;
        Ok(())
    }

    pub fn evaluator_extra_args(&self) -> Vec<String> {
        match self.evaluator {
            EvaluatorMode::Stub | EvaluatorMode::ProcessStub => Vec::new(),
            EvaluatorMode::Torch => {
                let checkpoint_dir = self
                    .checkpoint_dir
                    .as_ref()
                    .expect("validated checkpoint_dir exists");
                let device = self.eval_device.as_deref().unwrap_or("cuda:0");
                let mut args = vec![
                    "--backend".to_owned(),
                    "torch".to_owned(),
                    "--checkpoint-dir".to_owned(),
                    checkpoint_dir.display().to_string(),
                ];
                if let Some(pointer) = &self.checkpoint_pointer {
                    args.push("--checkpoint-pointer".to_owned());
                    args.push(pointer.clone());
                }
                args.push("--device".to_owned());
                args.push(device.to_owned());
                if let Some(interval) = self.eval_poll_interval {
                    args.push("--poll-interval".to_owned());
                    args.push(interval.to_string());
                }
                args
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluatorMode {
    Stub,
    ProcessStub,
    Torch,
}

impl EvaluatorMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stub => "stub",
            Self::ProcessStub => "process-stub",
            Self::Torch => "torch",
        }
    }
}

impl FromStr for EvaluatorMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "stub" => Ok(Self::Stub),
            "process-stub" => Ok(Self::ProcessStub),
            "torch" => Ok(Self::Torch),
            _ => Err(format!("unknown evaluator: {value}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelfplaySummary {
    pub evaluator: EvaluatorMode,
    pub model_version: Option<ModelVersion>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub rows_produced: u64,
    pub wins: u64,
    pub losses: u64,
    pub ties: u64,
    pub eval_batch_count: usize,
    pub mean_eval_batch_size: f64,
    pub replay_rows: u64,
    pub counters: ReplayCounters,
    pub measure_ledger: MeasureLedgerSnapshot,
}

pub fn init_replay(config: ReplayInitConfig) -> Result<ReplayInitSummary, String> {
    config.validate()?;

    let replay_dir = config
        .replay_dir
        .as_ref()
        .expect("validated replay_dir exists");
    let max_actions = feature_max_actions(config.max_candidates)?;
    let store = ReplayStore::open(replay_dir).map_err(|error| error.to_string())?;
    let engine = WhittleEngine::new(whittle_engine_config()).map_err(|error| error.to_string())?;
    let extractor = WhittleFeatureExtractor::with_config(
        &engine,
        WhittleFeatureExtractorConfig {
            max_actions,
            ..WhittleFeatureExtractorConfig::default()
        },
    );
    let schema = extractor.schema();
    store
        .ensure_contract(&ReplayContract::featurized(
            replay_data_mode(config.mask_stop),
            schema.config().clone(),
            EngineIdentity::from_engine(&engine),
        ))
        .map_err(|error| error.to_string())?;

    Ok(ReplayInitSummary {
        feature_schema_hash: schema.hash(),
        max_actions: schema.config().max_actions,
    })
}

pub fn run(config: SelfplayConfig) -> Result<SelfplaySummary, String> {
    config.validate()?;

    let replay_dir = config
        .replay_dir
        .as_ref()
        .expect("validated replay_dir exists");
    let store = Arc::new(
        ReplayStore::open_with_retention(replay_dir, config.replay_retain)
            .map_err(|error| error.to_string())?,
    );
    let engines = (0..config.lanes)
        .map(|_| WhittleEngine::new(whittle_engine_config()).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let search = search(&engines[0], &config)?;
    let roots = root_sources(&config);
    let schema = feature_extractor(&engines[0], &config)
        .schema()
        .config()
        .clone();
    store
        .ensure_contract(&ReplayContract::featurized(
            replay_data_mode(config.mask_stop),
            schema,
            EngineIdentity::from_engine(&engines[0]),
        ))
        .map_err(|error| error.to_string())?;

    let job_capacity = config
        .lanes
        .checked_mul(config.workers_per_lane)
        .ok_or_else(|| "remote measurement job capacity overflow".to_owned())?;
    let mut remote_coordinator = RemoteCoordinator::start(&config.remote_measure, job_capacity)?;
    if let Some(coordinator) = remote_coordinator.as_mut() {
        coordinator.wait_for_agent(config.remote_measure.startup_timeout)?;
    }
    let remote_measurement = remote_coordinator.as_ref().map(|coordinator| {
        let profile_hash = named_test_profile_hash(
            config
                .remote_measure
                .profile
                .as_deref()
                .expect("validated remote measure profile"),
        );
        RemoteMeasurementRuntime::new(
            coordinator.handle(),
            Arc::new(
                move |engine: &WhittleEngine, graph: WhittleGraphId, options| {
                    whittle_measure_submission(engine, graph, options, profile_hash)
                },
            ),
        )
    });

    if let Some(socket) = config.serve_socket.clone() {
        let serve_store = Arc::clone(&store);
        let serve_max_batch = config.serve_max_batch;
        std::thread::spawn(move || {
            if let Err(error) = crate::serve::run_shared(serve_store, socket, serve_max_batch) {
                eprintln!("sample service failed: {error}");
                std::process::exit(1);
            }
        });
    }

    match config.evaluator {
        EvaluatorMode::Stub => run_stub(config, store, engines, search, roots, remote_measurement),
        EvaluatorMode::ProcessStub | EvaluatorMode::Torch => {
            run_process(config, store, engines, search, roots, remote_measurement)
        }
    }
}

const fn replay_data_mode(mask_stop: bool) -> ReplayDataMode {
    if mask_stop {
        ReplayDataMode::SymmetricSelfplay
    } else {
        ReplayDataMode::SymmetricSelfplayStop
    }
}

fn run_stub(
    config: SelfplayConfig,
    store: Arc<ReplayStore>,
    engines: Vec<WhittleEngine>,
    search: GumbelMcts,
    roots: Vec<GeneratedRoots>,
    measurement: Option<RemoteMeasurementRuntime<WhittleEngine>>,
) -> Result<SelfplaySummary, String> {
    let extractors = engines
        .iter()
        .map(|engine| feature_extractor(engine, &config))
        .collect::<Vec<_>>();
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, search, threaded_config(&config)?);
    let featurized = FeaturizedRuntime {
        extractors,
        backends: vec![StubBackend],
    };
    let replay = ReplayRuntime {
        store: &store,
        backpressure: replay_backpressure(&config),
    };
    let run = match measurement {
        Some(measurement) => {
            orchestrator.run_featurized_with_replay_remote(roots, featurized, replay, measurement)
        }
        None => orchestrator.run_featurized_with_replay(roots, featurized, replay),
    }
    .map_err(|error| error.to_string())?;

    summarize(&store, run, EvaluatorMode::Stub, Some(STUB_MODEL_VERSION))
}

fn run_process(
    config: SelfplayConfig,
    store: Arc<ReplayStore>,
    engines: Vec<WhittleEngine>,
    search: GumbelMcts,
    roots: Vec<GeneratedRoots>,
    measurement: Option<RemoteMeasurementRuntime<WhittleEngine>>,
) -> Result<SelfplaySummary, String> {
    let extractors = engines
        .iter()
        .map(|engine| feature_extractor(engine, &config))
        .collect::<Vec<_>>();
    let mut processes = Vec::with_capacity(config.eval_processes);
    for index in 0..config.eval_processes {
        processes.push(
            EvaluatorProcess::spawn(EvaluatorProcessConfig {
                working_dir: config
                    .python_dir
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("python")),
                socket_path: process_socket_path(index),
                ready_timeout: Duration::from_secs(10),
                io_timeout: Duration::from_secs(300),
                extra_args: config.evaluator_extra_args(),
                ..EvaluatorProcessConfig::default()
            })
            .map_err(|error| error.to_string())?,
        );
    }
    let hello = Hello::new(
        extractors
            .first()
            .ok_or_else(|| "missing feature extractor".to_owned())?
            .schema()
            .hash(),
        config.max_batch as u32,
        engines[0].engine_id(),
        engines[0].engine_version(),
        engines[0].action_set_hash(),
    );
    let mut backends = Vec::with_capacity(processes.len());
    for process in &mut processes {
        backends.push(process.connect(&hello).map_err(|error| error.to_string())?);
    }
    let model_version = backends[0].model_version();
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, search, threaded_config(&config)?);
    let featurized = FeaturizedRuntime {
        extractors,
        backends,
    };
    let replay = ReplayRuntime {
        store: &store,
        backpressure: replay_backpressure(&config),
    };
    let run = match measurement {
        Some(measurement) => {
            orchestrator.run_featurized_with_replay_remote(roots, featurized, replay, measurement)
        }
        None => orchestrator.run_featurized_with_replay(roots, featurized, replay),
    }
    .map_err(|error| error.to_string())?;
    for process in &mut processes {
        wait_for_process_exit(process)?;
    }

    summarize(&store, run, config.evaluator, Some(model_version))
}

fn summarize(
    store: &ReplayStore,
    run: gz_orchestrator::ThreadedReplayRun,
    evaluator: EvaluatorMode,
    model_version: Option<ModelVersion>,
) -> Result<SelfplaySummary, String> {
    let counters = store.counters();
    let (wins, losses, ties) = label_counts(store)?;
    let evals = run.batch_sizes.iter().sum::<usize>();
    let mean_eval_batch_size = if run.batch_sizes.is_empty() {
        0.0
    } else {
        evals as f64 / run.batch_sizes.len() as f64
    };

    Ok(SelfplaySummary {
        evaluator,
        model_version,
        episodes_appended: run.episodes_appended,
        episodes_dropped: run.episodes_dropped,
        rows_produced: counters.produced_rows,
        wins,
        losses,
        ties,
        eval_batch_count: run.batch_sizes.len(),
        mean_eval_batch_size,
        replay_rows: run.replay_rows,
        counters,
        measure_ledger: run.measure_ledger,
    })
}

fn replay_backpressure(config: &SelfplayConfig) -> Option<ReplayBackpressure> {
    config.replay_backlog.map(|cap| ReplayBackpressure {
        max_row_backlog: NonZeroU64::new(cap).expect("validated nonzero"),
        gate_poll: Duration::from_millis(1),
    })
}

fn threaded_config(config: &SelfplayConfig) -> Result<ThreadedOrchestratorConfig, String> {
    Ok(ThreadedOrchestratorConfig {
        workers_per_lane: nonzero(config.workers_per_lane, "workers_per_lane")?,
        max_batch: nonzero(config.max_batch, "max_batch")?,
        admission_stagger: Duration::from_millis(config.admission_stagger_ms),
        admission_smoothing: admission_smoothing(config)?,
        flush_after: Duration::from_millis(3),
    })
}

fn admission_smoothing(
    config: &SelfplayConfig,
) -> Result<Option<AdmissionSmoothingConfig>, String> {
    if !config.admission_smoothing {
        return Ok(None);
    }
    let max_steps = u64::try_from(config.max_steps)
        .map_err(|_| "max_steps exceeds admission work range".to_owned())?;
    let simulations = u64::try_from(config.simulations)
        .map_err(|_| "simulations exceeds admission work range".to_owned())?;
    let initial_episode_eval_work = max_steps
        .checked_mul(2)
        .ok_or_else(|| "admission work estimate overflow".to_owned())?
        .checked_mul(
            simulations
                .checked_add(1)
                .ok_or_else(|| "admission work estimate overflow".to_owned())?,
        )
        .and_then(NonZeroU64::new)
        .ok_or_else(|| "admission work estimate overflow".to_owned())?;
    Ok(Some(AdmissionSmoothingConfig {
        initial_episode_eval_work,
    }))
}

fn process_socket_path(index: usize) -> PathBuf {
    std::env::temp_dir().join(format!("gz-evaluator-{}-{index}.sock", std::process::id()))
}

fn wait_for_process_exit(process: &mut EvaluatorProcess) -> Result<(), String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match process.try_wait().map_err(|error| error.to_string())? {
            Some(status) if status.success() => return Ok(()),
            Some(status) => return Err(format!("Python evaluator exited with {status}")),
            None if std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            None => return Err("Python evaluator did not exit".to_owned()),
        }
    }
}

fn search(engine: &WhittleEngine, config: &SelfplayConfig) -> Result<GumbelMcts, String> {
    Ok(GumbelMcts::new(GumbelMctsConfig {
        max_steps: config.max_steps,
        simulations: nonzero(config.simulations, "simulations")?,
        max_considered_actions: nonzero(config.max_considered, "max_considered")?,
        seed: config.seed,
        gumbel_scale: config.gumbel_scale,
        gumbel_noise_overlap: config.gumbel_noise_overlap,
        c_visit: config.c_visit,
        c_scale: config.c_scale,
        temperature_moves: 0,
        tree_reuse: config.tree_reuse,
        export_position: config.position_features,
        mask_stop: config.mask_stop,
        no_backtrack: config.no_backtrack,
        value_mode: GumbelValueMode::SymmetricSelfplay,
        candidate_options: feature_candidate_options(config),
        measure_options: engine.measure_options(),
    }))
}

fn feature_candidate_options(config: &SelfplayConfig) -> CandidateOptions {
    CandidateOptions {
        max_candidates: Some(config.max_candidates),
        deterministic_order: true,
    }
}

fn feature_extractor(engine: &WhittleEngine, config: &SelfplayConfig) -> WhittleFeatureExtractor {
    WhittleFeatureExtractor::with_config(
        engine,
        WhittleFeatureExtractorConfig {
            max_actions: feature_max_actions(config.max_candidates).expect("validated max_actions"),
            ..WhittleFeatureExtractorConfig::default()
        },
    )
}

fn feature_max_actions(max_candidates: usize) -> Result<u32, String> {
    let candidates = u32::try_from(max_candidates)
        .map_err(|_| "--max-candidates exceeds schema action limit".to_owned())?;
    candidates
        .checked_add(1)
        .ok_or_else(|| "--max-candidates exceeds schema action limit".to_owned())
}

fn root_sources(config: &SelfplayConfig) -> Vec<GeneratedRoots> {
    let base = config.episodes / config.lanes as u64;
    let extra = config.episodes % config.lanes as u64;

    (0..config.lanes)
        .map(|lane| {
            let count = base + u64::from((lane as u64) < extra);
            GeneratedRoots {
                remaining: (config.episodes != 0).then_some(count),
                generator: WhittleGraphGenerator::from_seed(
                    whittle_generator_config(),
                    config.seed ^ ((lane as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03)),
                ),
            }
        })
        .collect()
}

fn whittle_engine_config() -> WhittleEngineConfig {
    let generator = whittle_generator_config();
    WhittleEngineConfig {
        root: WhittleRoot::Input {
            arity: generator.arity,
            capacity: generator.capacity,
            input_index: 0,
        },
        ..WhittleEngineConfig::default()
    }
}

fn whittle_generator_config() -> WhittleGraphGeneratorConfig {
    WhittleGraphGeneratorConfig::default()
}

fn label_counts(store: &ReplayStore) -> Result<(u64, u64, u64), String> {
    let mut wins = 0;
    let mut losses = 0;
    let mut ties = 0;

    for id in 0..store.episode_sequence_end() {
        let Some(record) = store
            .episode(ReplayEpisodeId::new(id))
            .map_err(|error| error.to_string())?
        else {
            continue;
        };

        match record.outcome.value_target {
            Some(1.0) => wins += 1,
            Some(-1.0) => losses += 1,
            Some(0.0) => ties += 1,
            _ => {}
        }
    }

    Ok((wins, losses, ties))
}

fn nonzero(value: usize, name: &str) -> Result<NonZeroUsize, String> {
    NonZeroUsize::new(value).ok_or_else(|| format!("{name} must be greater than zero"))
}

pub struct GeneratedRoots {
    remaining: Option<u64>,
    generator: WhittleGraphGenerator,
}

impl RootSource<WhittleEngine> for GeneratedRoots {
    fn next_root(&mut self, engine: &mut WhittleEngine) -> EngineResult<Option<WhittleGraphId>> {
        match self.remaining.as_mut() {
            Some(0) => return Ok(None),
            Some(remaining) => *remaining -= 1,
            None => {}
        }
        self.generator.sample_root_into(engine).map(Some)
    }

    fn episode_roots_are_owned(&self) -> bool {
        true
    }
}
