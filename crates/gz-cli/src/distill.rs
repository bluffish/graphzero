use gz_engine::{
    CandidateOptions, EngineIdentity, GraphEngine, GraphHash, MeasureSummary, PortableCandidateRef,
    PortableGraphId, PortableSearchActionRef, ReplayGraphContext, SearchStepRef,
};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleFeatureExtractorConfig,
    WhittleGraphGenerator, WhittleGraphGeneratorConfig, WhittleRoot,
};
use gz_features::{FeatureExtractor, PositionFeatures, encode_feature_row};
use gz_replay::{
    ReplayContract, ReplayDataMode, ReplayEpisodeRecord, ReplayOutcome, ReplayRow, ReplayRowKind,
    ReplayStore,
};
use gz_search::reducing_uniform_distill_config_hash;
use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::time::{Duration, Instant};

const WORKER_SEED_STRIDE: u64 = 0xd1b5_4a32_d192_ed03;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyTeacher {
    ReducingUniform,
}

impl PolicyTeacher {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReducingUniform => "reducing-uniform",
        }
    }
}

impl FromStr for PolicyTeacher {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "reducing-uniform" => Ok(Self::ReducingUniform),
            _ => Err(format!("unknown policy teacher: {value}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DistillGenerateConfig {
    pub replay_dir: Option<PathBuf>,
    pub states: u64,
    pub workers: usize,
    pub max_attempts: u64,
    pub seed: u64,
    pub max_candidates: usize,
    pub max_steps: usize,
    pub position_features: bool,
    pub teacher: PolicyTeacher,
}

impl Default for DistillGenerateConfig {
    fn default() -> Self {
        Self {
            replay_dir: None,
            states: 100_000,
            workers: std::thread::available_parallelism().map_or(1, usize::from),
            max_attempts: 0,
            seed: 42,
            max_candidates: 1023,
            max_steps: 64,
            position_features: true,
            teacher: PolicyTeacher::ReducingUniform,
        }
    }
}

impl DistillGenerateConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_dir.is_none() {
            return Err("missing required --replay-dir".to_owned());
        }
        if self.states == 0 {
            return Err("--states must be greater than zero".to_owned());
        }
        if self.workers == 0 {
            return Err("--workers must be greater than zero".to_owned());
        }
        if self.max_candidates == 0 {
            return Err("--max-candidates must be greater than zero".to_owned());
        }
        let max_actions = self
            .max_candidates
            .checked_add(1)
            .ok_or_else(|| "--max-candidates exceeds schema action limit".to_owned())?;
        u32::try_from(max_actions)
            .map_err(|_| "--max-candidates exceeds schema action limit".to_owned())?;
        if self.attempt_limit() < self.states {
            return Err("--max-attempts must be zero or at least --states".to_owned());
        }
        Ok(())
    }

    fn attempt_limit(&self) -> u64 {
        if self.max_attempts == 0 {
            self.states.saturating_mul(10)
        } else {
            self.max_attempts
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TeacherLabel {
    pub policy_target: Vec<f32>,
    pub selected_candidate: Option<usize>,
    pub improving_actions: usize,
}

#[must_use]
pub fn reducing_uniform_label(current_reward: f32, successor_rewards: &[f32]) -> TeacherLabel {
    let improving = successor_rewards
        .iter()
        .enumerate()
        .filter_map(|(index, reward)| (*reward > current_reward).then_some(index))
        .collect::<Vec<_>>();
    let mut policy_target = vec![0.0; successor_rewards.len() + 1];
    if improving.is_empty() {
        *policy_target.last_mut().expect("STOP slot exists") = 1.0;
    } else {
        let probability = 1.0 / improving.len() as f32;
        for index in &improving {
            policy_target[*index] = probability;
        }
    }
    TeacherLabel {
        policy_target,
        selected_candidate: improving.first().copied(),
        improving_actions: improving.len(),
    }
}

#[derive(Clone, Debug)]
pub struct DistillGenerateSummary {
    pub states: u64,
    pub attempts: u64,
    pub duplicate_states: u64,
    pub candidate_overflows: u64,
    pub stop_targets: u64,
    pub improving_actions: u64,
    pub elapsed: Duration,
}

pub fn generate(config: DistillGenerateConfig) -> Result<DistillGenerateSummary, String> {
    config.validate()?;
    let replay_dir = config
        .replay_dir
        .as_ref()
        .expect("validated replay directory exists");
    let store = ReplayStore::open(replay_dir).map_err(|error| error.to_string())?;
    if store.counters().produced_rows != 0 || store.episode_counters().0 != 0 {
        return Err("distillation replay directory is not empty".to_owned());
    }
    let initial = worker_state(&config)?;
    let schema = initial.extractor.schema().config().clone();
    store
        .ensure_contract(&ReplayContract::featurized(
            ReplayDataMode::Standard,
            schema,
            EngineIdentity::from_engine(&initial.engine),
        ))
        .map_err(|error| error.to_string())?;

    let started = Instant::now();
    let stop = Arc::new(AtomicBool::new(false));
    let next_attempt = Arc::new(AtomicU64::new(0));
    let channel_capacity = config.workers.saturating_mul(2).max(1);
    let (tx, rx) = sync_channel::<Result<GeneratedAttempt, String>>(channel_capacity);
    let mut seen = HashSet::<GraphHash>::with_capacity(config.states as usize);
    let mut accepted = 0_u64;
    let mut attempts = 0_u64;
    let mut duplicate_states = 0_u64;
    let mut candidate_overflows = 0_u64;
    let mut stop_targets = 0_u64;
    let mut improving_actions = 0_u64;
    let mut first_error = None;

    std::thread::scope(|scope| {
        for worker in 0..config.workers {
            let worker_config = config.clone();
            let worker_tx = tx.clone();
            let worker_stop = Arc::clone(&stop);
            let worker_next_attempt = Arc::clone(&next_attempt);
            scope.spawn(move || {
                run_worker(
                    worker,
                    &worker_config,
                    worker_tx,
                    worker_stop,
                    worker_next_attempt,
                );
            });
        }
        drop(tx);

        while let Ok(result) = rx.recv() {
            attempts += 1;
            let attempt = match result {
                Ok(attempt) => attempt,
                Err(error) => {
                    stop.store(true, Ordering::Release);
                    first_error.get_or_insert(error);
                    continue;
                }
            };
            if accepted >= config.states || first_error.is_some() {
                continue;
            }
            let GeneratedAttempt::Example(example) = attempt else {
                candidate_overflows += 1;
                continue;
            };
            if !seen.insert(example.graph_hash) {
                duplicate_states += 1;
                continue;
            }
            if let Err(error) = store.append_episode(&example.record, &[example.row]) {
                stop.store(true, Ordering::Release);
                first_error.get_or_insert_with(|| error.to_string());
                continue;
            }
            accepted += 1;
            stop_targets += u64::from(example.improving_actions == 0);
            improving_actions += example.improving_actions as u64;
            if accepted.is_multiple_of(1000) || accepted == config.states {
                let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
                eprintln!(
                    "event=distill_generate states={accepted} attempts={attempts} rows_per_s={:.3}",
                    accepted as f64 / elapsed,
                );
            }
            if accepted == config.states {
                stop.store(true, Ordering::Release);
            }
        }
    });

    if let Some(error) = first_error {
        return Err(error);
    }
    if accepted != config.states {
        return Err(format!(
            "generated {accepted} of {} states before reaching the attempt limit; attempts={attempts} duplicates={duplicate_states} candidate_overflows={candidate_overflows}",
            config.states,
        ));
    }
    Ok(DistillGenerateSummary {
        states: accepted,
        attempts,
        duplicate_states,
        candidate_overflows,
        stop_targets,
        improving_actions,
        elapsed: started.elapsed(),
    })
}

fn run_worker(
    worker: usize,
    config: &DistillGenerateConfig,
    tx: SyncSender<Result<GeneratedAttempt, String>>,
    stop: Arc<AtomicBool>,
    next_attempt: Arc<AtomicU64>,
) {
    let mut state = match worker_state(config) {
        Ok(state) => state,
        Err(error) => {
            let _ = tx.send(Err(error));
            stop.store(true, Ordering::Release);
            return;
        }
    };
    state.generator = WhittleGraphGenerator::from_seed(
        WhittleGraphGeneratorConfig::default(),
        config.seed ^ (worker as u64 + 1).wrapping_mul(WORKER_SEED_STRIDE),
    );
    let attempt_limit = config.attempt_limit();

    while !stop.load(Ordering::Acquire) {
        let attempt = next_attempt.fetch_add(1, Ordering::Relaxed);
        if attempt >= attempt_limit {
            return;
        }
        let result = generate_attempt(&mut state, config);
        let failed = result.is_err();
        if tx.send(result).is_err() {
            return;
        }
        if failed {
            stop.store(true, Ordering::Release);
            return;
        }
    }
}

struct WorkerState {
    engine: WhittleEngine,
    generator: WhittleGraphGenerator,
    extractor: WhittleFeatureExtractor,
}

fn worker_state(config: &DistillGenerateConfig) -> Result<WorkerState, String> {
    let generator_config = WhittleGraphGeneratorConfig::default();
    let engine = WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Input {
            arity: generator_config.arity,
            capacity: generator_config.capacity,
            input_index: 0,
        },
        ..WhittleEngineConfig::default()
    })
    .map_err(|error| error.to_string())?;
    let max_actions = u32::try_from(config.max_candidates + 1)
        .map_err(|_| "--max-candidates exceeds schema action limit".to_owned())?;
    let extractor = WhittleFeatureExtractor::with_config(
        &engine,
        WhittleFeatureExtractorConfig {
            max_actions,
            ..WhittleFeatureExtractorConfig::default()
        },
    );
    Ok(WorkerState {
        engine,
        generator: WhittleGraphGenerator::from_seed(generator_config, config.seed),
        extractor,
    })
}

enum GeneratedAttempt {
    Example(Box<GeneratedExample>),
    CandidateOverflow,
}

struct GeneratedExample {
    graph_hash: GraphHash,
    record: ReplayEpisodeRecord,
    row: ReplayRow,
    improving_actions: usize,
}

fn generate_attempt(
    state: &mut WorkerState,
    config: &DistillGenerateConfig,
) -> Result<GeneratedAttempt, String> {
    let root = state
        .generator
        .sample_root_into(&mut state.engine)
        .map_err(|error| error.to_string())?;
    let candidate_options = CandidateOptions {
        max_candidates: Some(config.max_candidates + 1),
        deterministic_order: true,
    };
    let mut candidates = Vec::new();
    if let Err(error) = state
        .engine
        .candidates(root, candidate_options, &mut candidates)
    {
        let _ = state.engine.release(&[root], &[]);
        return Err(error.to_string());
    }

    let result = if candidates.len() > config.max_candidates {
        Ok(GeneratedAttempt::CandidateOverflow)
    } else {
        build_example(state, root, &candidates, config)
            .map(Box::new)
            .map(GeneratedAttempt::Example)
    };
    let released = state.engine.release(&[root], &candidates);
    match (result, released) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error.to_string()),
    }
}

fn build_example(
    state: &mut WorkerState,
    root: <WhittleEngine as GraphEngine>::Graph,
    candidates: &[<WhittleEngine as GraphEngine>::Candidate],
    config: &DistillGenerateConfig,
) -> Result<GeneratedExample, String> {
    let measure_options = state.engine.measure_options();
    let root_measure = state
        .engine
        .measure(root, measure_options)
        .map_err(|error| error.to_string())?;
    let current_reward = measured_reward(&root_measure)?;
    let root_context = context(&state.engine, root_measure.graph_hash);
    let mut legal_actions = Vec::with_capacity(candidates.len() + 1);
    let mut outcomes = Vec::with_capacity(candidates.len());

    for candidate in candidates.iter().copied() {
        let info = state
            .engine
            .candidate_info(root, candidate)
            .map_err(|error| error.to_string())?;
        legal_actions.push(PortableSearchActionRef::candidate(
            PortableCandidateRef::new(root_context, info.candidate_hash),
        ));
        let applied = state
            .engine
            .apply(root, candidate)
            .map_err(|error| error.to_string())?;
        let measured = state.engine.measure(applied.after, measure_options);
        let released = state.engine.release(&[applied.after], &[]);
        let measured = measured.map_err(|error| error.to_string())?;
        released.map_err(|error| error.to_string())?;
        outcomes.push(CandidateOutcome {
            context: context(&state.engine, measured.graph_hash),
            summary: MeasureSummary::from(&measured),
            reward: measured_reward(&measured)?,
        });
    }
    legal_actions.push(PortableSearchActionRef::stop(root_context));

    let successor_rewards = outcomes
        .iter()
        .map(|outcome| outcome.reward)
        .collect::<Vec<_>>();
    let label = match config.teacher {
        PolicyTeacher::ReducingUniform => {
            reducing_uniform_label(current_reward, &successor_rewards)
        }
    };
    let (selected_action, final_context, final_measure, stopped) = match label.selected_candidate {
        Some(index) => (
            legal_actions[index],
            outcomes[index].context,
            outcomes[index].summary.clone(),
            false,
        ),
        None => (
            *legal_actions.last().expect("STOP action exists"),
            root_context,
            MeasureSummary::from(&root_measure),
            true,
        ),
    };
    let reward = final_measure
        .scalar_reward
        .ok_or_else(|| "distillation final graph has no scalar reward".to_owned())?;
    let search_config_hash = reducing_uniform_distill_config_hash(
        config.max_steps,
        config.position_features,
        CandidateOptions {
            max_candidates: Some(config.max_candidates),
            deterministic_order: true,
        },
        measure_options,
    );
    let position = if config.position_features {
        PositionFeatures {
            root_step: 0,
            leaf_depth: 0,
            budget_fraction: 1.0,
            budget_step: if config.max_steps == 0 {
                0.0
            } else {
                1.0 / config.max_steps as f32
            },
            opponent_reward: 0.0,
            opponent_present: false,
        }
    } else {
        PositionFeatures {
            root_step: 0,
            leaf_depth: 0,
            budget_fraction: 0.0,
            budget_step: 0.0,
            opponent_reward: 0.0,
            opponent_present: false,
        }
    };
    let feature_row = state
        .extractor
        .extract(&state.engine, root, candidates, position)
        .map_err(|error| format!("feature extraction failed: {error:?}"))?;
    let mut feature_bytes = Vec::new();
    encode_feature_row(&feature_row, state.extractor.schema(), &mut feature_bytes)
        .map_err(|error| format!("feature encoding failed: {error:?}"))?;
    let step = SearchStepRef::new(root_context, selected_action, final_context)
        .map_err(|error| error.to_string())?;
    let record = ReplayEpisodeRecord {
        root: root_context,
        final_graph: final_context,
        steps: vec![step],
        final_measure: final_measure.clone(),
        outcome: ReplayOutcome::new(None, reward, stopped),
        search_config_hash,
        row_count: 1,
    };
    let row = ReplayRow {
        kind: ReplayRowKind::Decision,
        step_index: 0,
        root: root_context,
        state: root_context,
        action_history: Vec::new(),
        legal_actions,
        policy_target: label.policy_target,
        selected_action: Some(selected_action),
        value_target: None,
        horizon_value_targets: None,
        reward_target: Some(reward),
        final_measure,
        model_version: None,
        search_config_hash,
        feature_row: Some(feature_bytes),
    };
    Ok(GeneratedExample {
        graph_hash: root_measure.graph_hash,
        record,
        row,
        improving_actions: label.improving_actions,
    })
}

struct CandidateOutcome {
    context: ReplayGraphContext,
    summary: MeasureSummary,
    reward: f32,
}

fn measured_reward<G>(measure: &gz_engine::MeasureResult<G>) -> Result<f32, String> {
    match (measure.measured, measure.valid, measure.scalar_reward) {
        (true, true, Some(reward)) if reward.is_finite() => Ok(reward),
        _ => Err("Whittle measurement did not produce a valid scalar reward".to_owned()),
    }
}

fn context(engine: &WhittleEngine, graph_hash: GraphHash) -> ReplayGraphContext {
    ReplayGraphContext::new(
        PortableGraphId::new(graph_hash, engine.engine_id(), engine.engine_version()),
        engine.action_set_hash(),
    )
}
