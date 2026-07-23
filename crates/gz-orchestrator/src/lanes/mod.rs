use crate::admission::{AdmissionSmoothingConfig, EvalPressure, build_admission_shaper};
use crate::internal;
use crate::leases::ModelLeaseRegistry;
use crate::measurement::run_measure_gateway;
use crate::root::RootSource;
use gz_engine::{
    EngineError, EngineIdentity, EngineResult, ErrorCode, ErrorMessage, GraphEngine, MeasureResult,
};
use gz_eval::EvalOutput;
use gz_eval_service::{FeatureEvalBackend, ModelGeneration};
use gz_features::{
    FeatureCollator, FeatureExtractor, FeatureRow, FeatureSchema, FeatureSchemaHash,
};
use gz_measurer::{MeasureLedgerSnapshot, MeasuredSymmetricGame, MeasurerAdmission};
use gz_replay::{ReplayContract, ReplayError, ReplayStore};
use gz_search::{GumbelMcts, GumbelValueMode, WorkToken};
use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::time::Duration;

mod batcher;
mod lane;
mod projection;
mod replay_sink;

use batcher::run_featurized_batcher;
use lane::{FeaturizedReplayMode, LaneRuntime, merge_lane_measurer_summary, run_lane_pipeline};
use replay_sink::run_replay_sink;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadedOrchestratorConfig {
    pub workers_per_lane: NonZeroUsize,
    pub max_batch: NonZeroUsize,
    pub flush_after: Duration,
    pub admission_stagger: Duration,
    pub admission_smoothing: Option<AdmissionSmoothingConfig>,
}

pub struct ThreadedGumbelOrchestrator<E> {
    engines: Vec<E>,
    search: GumbelMcts,
    config: ThreadedOrchestratorConfig,
}

pub struct ReplayRuntime<'a> {
    pub store: &'a ReplayStore,
    pub backpressure: Option<ReplayBackpressure>,
}

pub struct FeaturizedRuntime<X, B> {
    pub extractors: Vec<X>,
    /// One batcher thread per backend; lanes are assigned round-robin
    /// (lane % backends.len()). Multiple evaluator processes parallelize
    /// the per-batch host work (decode/stage/encode runs on one thread
    /// per process) and keep the GPU's kernel queue dense.
    pub backends: Vec<B>,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplayBackpressure {
    pub max_row_backlog: NonZeroU64,
    pub gate_poll: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayLaneSummary {
    pub lane: usize,
    pub episodes_completed: u64,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub replay_rows: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreadedReplayRun {
    pub lanes: Vec<ReplayLaneSummary>,
    pub batch_sizes: Vec<usize>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub replay_rows: u64,
    pub measure_ledger: MeasureLedgerSnapshot,
}

struct FeaturizedEvalJob {
    lane: usize,
    slot: usize,
    token: WorkToken,
    row: FeatureRow,
    action_count: u32,
    model: ModelGeneration,
}

pub(crate) struct EvalReply {
    slot: usize,
    token: WorkToken,
    output: EvalOutput,
}

pub(crate) struct MeasureReply<G> {
    pub slot: usize,
    pub token: WorkToken,
    pub result: EngineResult<MeasureResult<G>>,
}

pub(crate) enum LaneReply<G> {
    Eval(EvalReply),
    Measure(MeasureReply<G>),
}

struct FeaturizedBatcherContext {
    model_registry: Arc<ModelLeaseRegistry>,
    eval_pressure: Option<Arc<EvalPressure>>,
}

enum ReplayJob {
    Symmetric {
        game: Box<MeasuredSymmetricGame>,
        ack: SyncSender<EngineResult<MeasurerAdmission>>,
    },
}

struct PipelineExit {
    closed: Arc<AtomicBool>,
    armed: bool,
}

impl PipelineExit {
    fn new(closed: Arc<AtomicBool>) -> Self {
        Self {
            closed,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PipelineExit {
    fn drop(&mut self) {
        if self.armed {
            self.closed.store(true, Ordering::Release);
        }
    }
}

impl<E> ThreadedGumbelOrchestrator<E>
where
    E: GraphEngine + Send,
    E::Graph: Send,
    E::Candidate: Send,
{
    pub const fn new(
        engines: Vec<E>,
        search: GumbelMcts,
        config: ThreadedOrchestratorConfig,
    ) -> Self {
        Self {
            engines,
            search,
            config,
        }
    }

    pub fn run_featurized_with_replay<R, X, B>(
        self,
        root_sources: Vec<R>,
        featurized: FeaturizedRuntime<X, B>,
        replay: ReplayRuntime<'_>,
    ) -> EngineResult<ThreadedReplayRun>
    where
        R: RootSource<E> + Send,
        X: FeatureExtractor<E> + Send,
        B: FeatureEvalBackend + Send,
    {
        self.run_featurized_with_replay_inner(root_sources, featurized, replay, None)
    }

    pub fn run_featurized_with_replay_remote<R, X, B>(
        self,
        root_sources: Vec<R>,
        featurized: FeaturizedRuntime<X, B>,
        replay: ReplayRuntime<'_>,
        measurement: crate::RemoteMeasurementRuntime<E>,
    ) -> EngineResult<ThreadedReplayRun>
    where
        R: RootSource<E> + Send,
        X: FeatureExtractor<E> + Send,
        B: FeatureEvalBackend + Send,
    {
        self.run_featurized_with_replay_inner(root_sources, featurized, replay, Some(measurement))
    }

    fn run_featurized_with_replay_inner<R, X, B>(
        self,
        root_sources: Vec<R>,
        featurized: FeaturizedRuntime<X, B>,
        replay: ReplayRuntime<'_>,
        measurement: Option<crate::RemoteMeasurementRuntime<E>>,
    ) -> EngineResult<ThreadedReplayRun>
    where
        R: RootSource<E> + Send,
        X: FeatureExtractor<E> + Send,
        B: FeatureEvalBackend + Send,
    {
        if self.search.config().value_mode != GumbelValueMode::SymmetricSelfplay {
            return Err(internal("replay selfplay requires symmetric search"));
        }

        let lanes = self.engines.len();
        if root_sources.len() != lanes || featurized.extractors.len() != lanes {
            return Err(internal("lane count mismatch"));
        }
        let engine_identity = validate_engine_identities(&self.engines)?;
        let schema_hash = validate_feature_schemas::<E, X>(&featurized.extractors)?;
        validate_backend_count(featurized.backends.len(), lanes)?;
        let data_mode = if self.search.config().mask_stop {
            gz_replay::ReplayDataMode::SymmetricSelfplay
        } else {
            gz_replay::ReplayDataMode::SymmetricSelfplayStop
        };
        let feature_schema = first_schema::<E, X>(&featurized.extractors, schema_hash)?;
        replay
            .store
            .ensure_contract(&ReplayContract::featurized(
                data_mode,
                feature_schema.config().clone(),
                engine_identity,
            ))
            .map_err(map_replay_error)?;

        let workers_per_lane = self.config.workers_per_lane.get();
        let worker_capacity = lanes
            .checked_mul(workers_per_lane)
            .ok_or_else(|| internal("worker count overflow"))?;
        let evals_per_worker = self
            .search
            .config()
            .max_considered_actions
            .get()
            .min(self.search.config().simulations.get());
        let intake_capacity = worker_capacity
            .checked_mul(evals_per_worker)
            .ok_or_else(|| internal("wave eval capacity overflow"))?;
        let backend_count = featurized.backends.len();
        let mut intake_txs = Vec::with_capacity(backend_count);
        let mut intake_rxs = Vec::with_capacity(backend_count);
        for _ in 0..backend_count {
            let (tx, rx) = sync_channel(intake_capacity);
            intake_txs.push(tx);
            intake_rxs.push(rx);
        }
        let (replay_tx, replay_rx) = sync_channel(worker_capacity);
        let mut reply_txs: Vec<SyncSender<LaneReply<E::Graph>>> = Vec::with_capacity(lanes);
        let mut reply_rxs = Vec::with_capacity(lanes);
        let reply_capacity = workers_per_lane
            .checked_mul(evals_per_worker)
            .ok_or_else(|| internal("wave reply capacity overflow"))?;
        for _ in 0..lanes {
            let (tx, rx) = sync_channel(reply_capacity);
            reply_txs.push(tx);
            reply_rxs.push(rx);
        }
        let (measure_tx, measure_rx, measure_coordinator, measure_encoder) = match measurement {
            Some(measurement) => {
                let (tx, rx) = tokio::sync::mpsc::channel(worker_capacity);
                (
                    Some(tx),
                    Some(rx),
                    Some(measurement.coordinator),
                    Some(measurement.encoder),
                )
            }
            None => (None, None, None, None),
        };

        let config = self.config;
        let eval_pressure = Arc::new(EvalPressure::default());
        let pipeline_closed = Arc::new(AtomicBool::new(false));
        let admission_shaper = build_admission_shaper(
            lanes,
            backend_count,
            config.workers_per_lane,
            config.max_batch,
            config.admission_stagger,
            config.admission_smoothing,
            Arc::clone(&eval_pressure),
        )?;
        let search = &self.search;
        let backends = featurized.backends;
        let model_registries = backends
            .iter()
            .map(|backend| ModelLeaseRegistry::new(backend.model_generation()).map(Arc::new))
            .collect::<EngineResult<Vec<_>>>()?;
        let extractors = featurized.extractors;
        let engines = self.engines;
        let store = replay.store;
        let backpressure = replay.backpressure;
        validate_collator_capacity(
            &FeatureCollator::new(feature_schema.clone(), config.max_batch),
            config,
        )?;
        let (batch_results, sink_result, lane_results, gateway_result) =
            std::thread::scope(|scope| {
                let gateway_handle = match (measure_coordinator, measure_rx) {
                    (Some(coordinator), Some(measure_rx)) => {
                        let reply_txs = reply_txs.clone();
                        let pipeline_closed = Arc::clone(&pipeline_closed);
                        Some(scope.spawn(move || {
                            let mut exit = PipelineExit::new(pipeline_closed);
                            let result = run_measure_gateway(coordinator, measure_rx, reply_txs);
                            if result.is_ok() {
                                exit.disarm();
                            }
                            result
                        }))
                    }
                    (None, None) => None,
                    _ => unreachable!("remote measurement runtime is complete"),
                };
                let mut batch_handles = Vec::with_capacity(backend_count);
                for ((backend, intake_rx), model_registry) in backends
                    .into_iter()
                    .zip(intake_rxs)
                    .zip(model_registries.iter().cloned())
                {
                    let batch_capacity = backend.batch_capacity().unwrap_or(config.max_batch);
                    let collator = FeatureCollator::new(feature_schema.clone(), batch_capacity);
                    let reply_txs = reply_txs.clone();
                    let eval_pressure = Arc::clone(&eval_pressure);
                    let pipeline_closed = Arc::clone(&pipeline_closed);
                    batch_handles.push(scope.spawn(move || {
                        let mut exit = PipelineExit::new(pipeline_closed);
                        let result = run_featurized_batcher(
                            backend,
                            collator,
                            intake_rx,
                            reply_txs,
                            config,
                            FeaturizedBatcherContext {
                                model_registry,
                                eval_pressure: Some(eval_pressure),
                            },
                        );
                        if result.is_ok() {
                            exit.disarm();
                        }
                        result
                    }));
                }
                drop(reply_txs);
                let sink_handle = scope.spawn(move || run_replay_sink(store, replay_rx));
                let mut lane_handles = Vec::with_capacity(lanes);

                for (lane, ((((engine, roots), extractor), reply_rx), model_registry)) in
                    engines
                        .into_iter()
                        .zip(root_sources)
                        .zip(extractors)
                        .zip(reply_rxs)
                        .zip((0..lanes).map(|lane| {
                            Arc::clone(&model_registries[lane % model_registries.len()])
                        }))
                        .enumerate()
                {
                    let intake_tx = intake_txs[lane % backend_count].clone();
                    let replay_tx = replay_tx.clone();
                    let eval_pressure = Arc::clone(&eval_pressure);
                    let pipeline_closed = Arc::clone(&pipeline_closed);
                    let admission_shaper = admission_shaper.clone();
                    let measure_tx = measure_tx.clone();
                    let measure_encoder = measure_encoder.clone();
                    lane_handles.push(scope.spawn(move || {
                        let mut exit = PipelineExit::new(Arc::clone(&pipeline_closed));
                        let result = run_lane_pipeline(
                            engine,
                            roots,
                            LaneRuntime {
                                lane,
                                lanes,
                                search,
                                workers_per_lane: config.workers_per_lane,
                                pool_capacity: config.workers_per_lane,
                                admission_stagger: config.admission_stagger,
                                admission_shaper,
                                eval_pressure,
                                pipeline_closed,
                                intake_tx,
                                reply_rx,
                                measure_tx,
                                measure_encoder,
                            },
                            FeaturizedReplayMode::new(
                                lane,
                                extractor,
                                replay_tx,
                                store,
                                backpressure,
                                model_registry,
                            ),
                        );
                        if result.is_ok() {
                            exit.disarm();
                        }
                        result
                    }));
                }

                drop(intake_txs);
                drop(replay_tx);
                drop(measure_tx);

                let lane_results = lane_handles
                    .into_iter()
                    .map(|handle| {
                        handle
                            .join()
                            .unwrap_or_else(|_| Err(internal("worker blocked")))
                    })
                    .collect::<Vec<_>>();
                let batch_results = batch_handles
                    .into_iter()
                    .map(|handle| {
                        handle
                            .join()
                            .unwrap_or_else(|_| Err(internal("eval backend unavailable")))
                    })
                    .collect::<Vec<_>>();
                let sink_result = sink_handle
                    .join()
                    .unwrap_or_else(|_| Err(internal("replay sink failed")));

                let gateway_result = gateway_handle.map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err(internal("measure gateway failed")))
                });

                (batch_results, sink_result, lane_results, gateway_result)
            });

        let mut batch_sizes = Vec::new();
        for result in batch_results {
            batch_sizes.extend(result?);
        }
        let measurer_summary = sink_result?;
        if let Some(result) = gateway_result {
            result?;
        }
        let mut lane_summaries = Vec::with_capacity(lane_results.len());
        for result in lane_results {
            let mut result = result?;
            merge_lane_measurer_summary(&mut result, &measurer_summary);
            lane_summaries.push(result);
        }

        Ok(ThreadedReplayRun {
            lanes: lane_summaries,
            batch_sizes,
            episodes_appended: measurer_summary.episodes_appended,
            episodes_dropped: measurer_summary.episodes_dropped,
            replay_rows: measurer_summary.replay_rows,
            measure_ledger: measurer_summary.measure_ledger,
        })
    }
}

fn map_replay_error(error: ReplayError) -> EngineError {
    EngineError::Internal {
        code: ErrorCode::new(1),
        message: ErrorMessage::new(format!("replay sink failed: {error}"))
            .expect("replay error message is bounded"),
    }
}

fn validate_engine_identities<E>(engines: &[E]) -> EngineResult<EngineIdentity>
where
    E: GraphEngine,
{
    let Some(first) = engines.first().map(EngineIdentity::from_engine) else {
        return Err(internal("missing engine lane"));
    };
    for engine in &engines[1..] {
        if EngineIdentity::from_engine(engine) != first {
            return Err(internal("engine identity mismatch"));
        }
    }
    Ok(first)
}

fn validate_feature_schemas<E, X>(extractors: &[X]) -> EngineResult<FeatureSchemaHash>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let Some(first) = extractors.first() else {
        return Err(internal("missing feature schema"));
    };
    let hash = first.schema().hash();
    for extractor in &extractors[1..] {
        if extractor.schema().hash() != hash {
            return Err(internal("feature schema mismatch"));
        }
    }
    Ok(hash)
}

fn first_schema<E, X>(extractors: &[X], hash: FeatureSchemaHash) -> EngineResult<FeatureSchema>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let schema = extractors
        .first()
        .ok_or_else(|| internal("missing feature schema"))?
        .schema();
    if schema.hash() != hash {
        return Err(internal("feature schema mismatch"));
    }
    Ok(schema.clone())
}

fn validate_backend_count(backends: usize, lanes: usize) -> EngineResult<()> {
    if backends == 0 {
        return Err(internal("no eval backends"));
    }
    if backends > lanes {
        return Err(internal("more eval backends than lanes"));
    }
    Ok(())
}

fn validate_collator_capacity(
    collator: &FeatureCollator,
    config: ThreadedOrchestratorConfig,
) -> EngineResult<()> {
    if collator.batch_capacity() != config.max_batch {
        return Err(internal("feature batch capacity mismatch"));
    }
    Ok(())
}
