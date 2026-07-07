use crate::EpisodeId;
use crate::pool::{Admission, WorkerPool};
use crate::project::project_episode;
use crate::reference::{Reference, ReferenceProvider, RolloutOutcome};
use crate::root::RootSource;
use crate::serial::OrchestratedEpisode;
use crate::service::internal;
use gz_engine::{
    CandidateOptions, EngineError, EngineResult, ErrorCode, ErrorMessage, GraphEngine, ModelVersion,
};
use gz_eval::{EvalOutput, EvalRequest, Evaluator, eval_error_to_engine_error, validate_outputs};
use gz_eval_service::{BackendOutputs, FeatureEvalBackend};
use gz_features::{
    FeatureCollator, FeatureExtractor, FeatureRow, FeatureSchema, FeatureSchemaHash,
    OpponentStateFeatures, PositionFeatures, encode_feature_row,
};
use gz_replay::{ReplayEpisodeRecord, ReplayError, ReplayRow, ReplayStore};
use gz_search::{
    EngineIdentity, GumbelEpisode, GumbelEpisodeContext, GumbelMcts, GumbelOpponentContext,
    WorkToken,
};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, sync_channel};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadedOrchestratorConfig {
    pub workers_per_lane: NonZeroUsize,
    pub max_batch: NonZeroUsize,
    pub flush_after: Duration,
}

pub struct ThreadedGumbelOrchestrator<E, V> {
    engines: Vec<E>,
    evaluator: V,
    search: GumbelMcts,
    config: ThreadedOrchestratorConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LaneEpisodes<G, C> {
    pub lane: usize,
    /// Completed batch-path episodes. Engine handles inside each episode are
    /// opaque identifiers only; the lane has already released them.
    pub episodes: Vec<OrchestratedEpisode<G, C>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreadedRun<G, C> {
    /// Batch-path episode equality surface. Engine handles inside returned
    /// episodes have already been released and must not be dereferenced.
    pub lanes: Vec<LaneEpisodes<G, C>>,
    pub batch_sizes: Vec<usize>,
}

pub struct ReplayRuntime<'a, P> {
    pub store: &'a ReplayStore,
    pub providers: Vec<P>,
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
    pub search_contexts: u64,
    pub replay_rows: u64,
    pub reference_steps: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreadedReplayRun {
    pub lanes: Vec<ReplayLaneSummary>,
    pub batch_sizes: Vec<usize>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub search_contexts: u64,
    pub replay_rows: u64,
    pub reference_steps: u64,
}

struct EvalJob {
    lane: usize,
    slot: usize,
    token: WorkToken,
    request: EvalRequest,
}

struct FeaturizedEvalJob {
    lane: usize,
    slot: usize,
    token: WorkToken,
    row: FeatureRow,
    action_count: u32,
}

struct EvalReply {
    slot: usize,
    token: WorkToken,
    output: EvalOutput,
}

struct ReplayJob {
    record: ReplayEpisodeRecord,
    rows: Vec<ReplayRow>,
    ack: SyncSender<EngineResult<()>>,
}

impl<E, V> ThreadedGumbelOrchestrator<E, V>
where
    E: GraphEngine + Send,
    E::Graph: Send,
    E::Candidate: Send,
    V: Evaluator + Send,
{
    pub const fn new(
        engines: Vec<E>,
        evaluator: V,
        search: GumbelMcts,
        config: ThreadedOrchestratorConfig,
    ) -> Self {
        Self {
            engines,
            evaluator,
            search,
            config,
        }
    }

    pub fn run<R>(
        self,
        root_sources: Vec<R>,
        context: GumbelEpisodeContext,
    ) -> EngineResult<ThreadedRun<E::Graph, E::Candidate>>
    where
        R: RootSource<E> + Send,
    {
        let lanes = self.engines.len();
        if root_sources.len() != lanes {
            return Err(internal("lane count mismatch"));
        }
        validate_engine_identities(&self.engines)?;

        let workers_per_lane = self.config.workers_per_lane.get();
        let intake_capacity = lanes * workers_per_lane;
        // Intake can hold every possible outstanding eval at once. The batcher
        // never waits on a lane while holding jobs, so this bounded channel
        // cannot form a steady-state send cycle.
        let (intake_tx, intake_rx) = sync_channel(intake_capacity);
        let mut reply_txs = Vec::with_capacity(lanes);
        let mut reply_rxs = Vec::with_capacity(lanes);

        for _ in 0..lanes {
            // A lane can have at most one outstanding eval per worker. This
            // capacity lets the batcher route all lane replies without blocking.
            let (tx, rx) = sync_channel(workers_per_lane);
            reply_txs.push(tx);
            reply_rxs.push(rx);
        }

        let config = self.config;
        let search = &self.search;
        let evaluator = self.evaluator;
        let engines = self.engines;

        let (batch_result, lane_results) = std::thread::scope(|scope| {
            let batch_handle =
                scope.spawn(move || run_batcher(evaluator, intake_rx, reply_txs, config));
            let mut lane_handles = Vec::with_capacity(lanes);

            for (lane, ((engine, roots), reply_rx)) in engines
                .into_iter()
                .zip(root_sources)
                .zip(reply_rxs)
                .enumerate()
            {
                let intake_tx = intake_tx.clone();
                lane_handles.push(scope.spawn(move || {
                    run_lane_pipeline(
                        engine,
                        roots,
                        LaneRuntime {
                            lane,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            context,
                            intake_tx,
                            reply_rx,
                        },
                        CollectMode::new(),
                    )
                }));
            }

            drop(intake_tx);

            let lane_results = lane_handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err(internal("worker blocked")))
                })
                .collect::<Vec<_>>();
            let batch_result = batch_handle
                .join()
                .unwrap_or_else(|_| Err(internal("eval backend unavailable")));

            (batch_result, lane_results)
        });

        let batch_sizes = batch_result?;
        let mut lanes = Vec::with_capacity(lane_results.len());

        for result in lane_results {
            lanes.push(result?);
        }

        Ok(ThreadedRun { lanes, batch_sizes })
    }

    pub fn run_with_replay<R, P>(
        self,
        root_sources: Vec<R>,
        context: GumbelEpisodeContext,
        replay: ReplayRuntime<'_, P>,
    ) -> EngineResult<ThreadedReplayRun>
    where
        R: RootSource<E> + Send,
        P: ReferenceProvider<E> + Send,
    {
        let lanes = self.engines.len();
        if root_sources.len() != lanes || replay.providers.len() != lanes {
            return Err(internal("lane count mismatch"));
        }
        validate_engine_identities(&self.engines)?;

        let workers_per_lane = self.config.workers_per_lane.get();
        let intake_capacity = lanes * workers_per_lane;
        let (intake_tx, intake_rx) = sync_channel(intake_capacity);
        let (replay_tx, replay_rx) = sync_channel(intake_capacity);
        let mut reply_txs = Vec::with_capacity(lanes);
        let mut reply_rxs = Vec::with_capacity(lanes);

        for _ in 0..lanes {
            let (tx, rx) = sync_channel(workers_per_lane);
            reply_txs.push(tx);
            reply_rxs.push(rx);
        }

        let config = self.config;
        let search = &self.search;
        let evaluator = self.evaluator;
        let engines = self.engines;
        let providers = replay.providers;
        let store = replay.store;
        let backpressure = replay.backpressure;

        let (batch_result, sink_result, lane_results) = std::thread::scope(|scope| {
            let batch_handle =
                scope.spawn(move || run_batcher(evaluator, intake_rx, reply_txs, config));
            let sink_handle = scope.spawn(move || run_replay_sink(store, replay_rx));
            let mut lane_handles = Vec::with_capacity(lanes);

            for (lane, (((engine, roots), provider), reply_rx)) in engines
                .into_iter()
                .zip(root_sources)
                .zip(providers)
                .zip(reply_rxs)
                .enumerate()
            {
                let intake_tx = intake_tx.clone();
                let replay_tx = replay_tx.clone();
                lane_handles.push(scope.spawn(move || {
                    run_lane_pipeline(
                        engine,
                        roots,
                        LaneRuntime {
                            lane,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            context,
                            intake_tx,
                            reply_rx,
                        },
                        ReplayMode::new(provider, replay_tx, store, backpressure),
                    )
                }));
            }

            drop(intake_tx);
            drop(replay_tx);

            let lane_results = lane_handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err(internal("worker blocked")))
                })
                .collect::<Vec<_>>();
            let batch_result = batch_handle
                .join()
                .unwrap_or_else(|_| Err(internal("eval backend unavailable")));
            let sink_result = sink_handle
                .join()
                .unwrap_or_else(|_| Err(internal("replay sink failed")));

            (batch_result, sink_result, lane_results)
        });

        let batch_sizes = batch_result?;
        let episodes_appended = sink_result?;
        let mut lanes = Vec::with_capacity(lane_results.len());
        let mut episodes_dropped = 0;
        let mut search_contexts = 0;
        let mut replay_rows = 0;
        let mut reference_steps = 0;

        for result in lane_results {
            let result = result?;
            episodes_dropped += result.episodes_dropped;
            search_contexts += result.search_contexts;
            replay_rows += result.replay_rows;
            reference_steps += result.reference_steps;
            lanes.push(result);
        }

        Ok(ThreadedReplayRun {
            lanes,
            batch_sizes,
            episodes_appended,
            episodes_dropped,
            search_contexts,
            replay_rows,
            reference_steps,
        })
    }

    pub fn run_featurized<R, X, B>(
        self,
        root_sources: Vec<R>,
        context: GumbelEpisodeContext,
        featurized: FeaturizedRuntime<X, B>,
    ) -> EngineResult<ThreadedRun<E::Graph, E::Candidate>>
    where
        R: RootSource<E> + Send,
        X: FeatureExtractor<E> + Send,
        B: FeatureEvalBackend + Send,
    {
        let lanes = self.engines.len();
        if root_sources.len() != lanes || featurized.extractors.len() != lanes {
            return Err(internal("lane count mismatch"));
        }
        validate_engine_identities(&self.engines)?;
        let schema_hash = validate_feature_schemas::<E, X>(&featurized.extractors)?;
        validate_backend_count(featurized.backends.len(), lanes)?;

        let workers_per_lane = self.config.workers_per_lane.get();
        let intake_capacity = lanes * workers_per_lane;
        let backend_count = featurized.backends.len();
        let mut intake_txs = Vec::with_capacity(backend_count);
        let mut intake_rxs = Vec::with_capacity(backend_count);
        for _ in 0..backend_count {
            let (tx, rx) = sync_channel(intake_capacity);
            intake_txs.push(tx);
            intake_rxs.push(rx);
        }
        let mut reply_txs = Vec::with_capacity(lanes);
        let mut reply_rxs = Vec::with_capacity(lanes);

        for _ in 0..lanes {
            let (tx, rx) = sync_channel(workers_per_lane);
            reply_txs.push(tx);
            reply_rxs.push(rx);
        }

        let config = self.config;
        let search = &self.search;
        let backends = featurized.backends;
        let extractors = featurized.extractors;
        let engines = self.engines;
        let feature_schema = first_schema::<E, X>(&extractors, schema_hash)?;
        validate_collator_capacity(
            &FeatureCollator::new(feature_schema.clone(), config.max_batch),
            config,
        )?;
        let _ = self.evaluator;

        let (batch_results, lane_results) = std::thread::scope(|scope| {
            let mut batch_handles = Vec::with_capacity(backend_count);
            for (backend, intake_rx) in backends.into_iter().zip(intake_rxs) {
                let collator = FeatureCollator::new(feature_schema.clone(), config.max_batch);
                let reply_txs = reply_txs.clone();
                batch_handles.push(scope.spawn(move || {
                    run_featurized_batcher(backend, collator, intake_rx, reply_txs, config)
                }));
            }
            drop(reply_txs);
            let mut lane_handles = Vec::with_capacity(lanes);

            for (lane, (((engine, roots), extractor), reply_rx)) in engines
                .into_iter()
                .zip(root_sources)
                .zip(extractors)
                .zip(reply_rxs)
                .enumerate()
            {
                let intake_tx = intake_txs[lane % backend_count].clone();
                lane_handles.push(scope.spawn(move || {
                    run_lane_pipeline(
                        engine,
                        roots,
                        LaneRuntime {
                            lane,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            context,
                            intake_tx,
                            reply_rx,
                        },
                        FeaturizedCollectMode::new(extractor),
                    )
                }));
            }

            drop(intake_txs);

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

            (batch_results, lane_results)
        });

        let mut batch_sizes = Vec::new();
        for result in batch_results {
            batch_sizes.extend(result?);
        }
        let mut lanes = Vec::with_capacity(lane_results.len());

        for result in lane_results {
            lanes.push(result?);
        }

        Ok(ThreadedRun { lanes, batch_sizes })
    }

    pub fn run_featurized_with_replay<R, X, B, P>(
        self,
        root_sources: Vec<R>,
        context: GumbelEpisodeContext,
        featurized: FeaturizedRuntime<X, B>,
        replay: ReplayRuntime<'_, P>,
    ) -> EngineResult<ThreadedReplayRun>
    where
        R: RootSource<E> + Send,
        X: FeatureExtractor<E> + Send,
        B: FeatureEvalBackend + Send,
        P: ReferenceProvider<E> + Send,
    {
        let lanes = self.engines.len();
        if root_sources.len() != lanes
            || featurized.extractors.len() != lanes
            || replay.providers.len() != lanes
        {
            return Err(internal("lane count mismatch"));
        }
        validate_engine_identities(&self.engines)?;
        let schema_hash = validate_feature_schemas::<E, X>(&featurized.extractors)?;
        validate_backend_count(featurized.backends.len(), lanes)?;

        let workers_per_lane = self.config.workers_per_lane.get();
        let intake_capacity = lanes * workers_per_lane;
        let backend_count = featurized.backends.len();
        let mut intake_txs = Vec::with_capacity(backend_count);
        let mut intake_rxs = Vec::with_capacity(backend_count);
        for _ in 0..backend_count {
            let (tx, rx) = sync_channel(intake_capacity);
            intake_txs.push(tx);
            intake_rxs.push(rx);
        }
        let (replay_tx, replay_rx) = sync_channel(intake_capacity);
        let mut reply_txs = Vec::with_capacity(lanes);
        let mut reply_rxs = Vec::with_capacity(lanes);

        for _ in 0..lanes {
            let (tx, rx) = sync_channel(workers_per_lane);
            reply_txs.push(tx);
            reply_rxs.push(rx);
        }

        let config = self.config;
        let search = &self.search;
        let backends = featurized.backends;
        let extractors = featurized.extractors;
        let engines = self.engines;
        let providers = replay.providers;
        let store = replay.store;
        let backpressure = replay.backpressure;
        let feature_schema = first_schema::<E, X>(&extractors, schema_hash)?;
        store
            .ensure_feature_schema(feature_schema.config())
            .map_err(map_replay_error)?;
        validate_collator_capacity(
            &FeatureCollator::new(feature_schema.clone(), config.max_batch),
            config,
        )?;
        let _ = self.evaluator;

        let (batch_results, sink_result, lane_results) = std::thread::scope(|scope| {
            let mut batch_handles = Vec::with_capacity(backend_count);
            for (backend, intake_rx) in backends.into_iter().zip(intake_rxs) {
                let collator = FeatureCollator::new(feature_schema.clone(), config.max_batch);
                let reply_txs = reply_txs.clone();
                batch_handles.push(scope.spawn(move || {
                    run_featurized_batcher(backend, collator, intake_rx, reply_txs, config)
                }));
            }
            drop(reply_txs);
            let sink_handle = scope.spawn(move || run_replay_sink(store, replay_rx));
            let mut lane_handles = Vec::with_capacity(lanes);

            for (lane, ((((engine, roots), extractor), provider), reply_rx)) in engines
                .into_iter()
                .zip(root_sources)
                .zip(extractors)
                .zip(providers)
                .zip(reply_rxs)
                .enumerate()
            {
                let intake_tx = intake_txs[lane % backend_count].clone();
                let replay_tx = replay_tx.clone();
                lane_handles.push(scope.spawn(move || {
                    run_lane_pipeline(
                        engine,
                        roots,
                        LaneRuntime {
                            lane,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            context,
                            intake_tx,
                            reply_rx,
                        },
                        FeaturizedReplayMode::new(
                            extractor,
                            provider,
                            replay_tx,
                            store,
                            backpressure,
                        ),
                    )
                }));
            }

            drop(intake_txs);
            drop(replay_tx);

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

            (batch_results, sink_result, lane_results)
        });

        let mut batch_sizes = Vec::new();
        for result in batch_results {
            batch_sizes.extend(result?);
        }
        let episodes_appended = sink_result?;
        let mut lanes = Vec::with_capacity(lane_results.len());
        let mut episodes_dropped = 0;
        let mut search_contexts = 0;
        let mut replay_rows = 0;
        let mut reference_steps = 0;

        for result in lane_results {
            let result = result?;
            episodes_dropped += result.episodes_dropped;
            search_contexts += result.search_contexts;
            replay_rows += result.replay_rows;
            reference_steps += result.reference_steps;
            lanes.push(result);
        }

        Ok(ThreadedReplayRun {
            lanes,
            batch_sizes,
            episodes_appended,
            episodes_dropped,
            search_contexts,
            replay_rows,
            reference_steps,
        })
    }
}

struct LaneRuntime<'a, J> {
    lane: usize,
    search: &'a GumbelMcts,
    workers_per_lane: NonZeroUsize,
    context: GumbelEpisodeContext,
    intake_tx: SyncSender<J>,
    reply_rx: Receiver<EvalReply>,
}

struct EpisodeFeatureRows<C> {
    rows: Vec<Vec<u8>>,
    candidates: Vec<C>,
}

trait LaneMode<E>
where
    E: GraphEngine,
{
    type Job;
    type Output;

    fn begin(
        &mut self,
        search: &GumbelMcts,
        identity: EngineIdentity,
        context: GumbelEpisodeContext,
    ) {
        let _ = (search, identity, context);
    }

    fn before_root_admission<R>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        next_episode_id: &mut u64,
    ) -> EngineResult<()>
    where
        R: RootSource<E>,
    {
        let _ = (pool, engine, roots, next_episode_id);
        Ok(())
    }

    fn gate_open(&self) -> bool {
        true
    }

    fn gate_poll(&self) -> Option<Duration> {
        None
    }

    fn episode_context(
        &mut self,
        engine: &mut E,
        episode_id: EpisodeId,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<GumbelEpisodeContext> {
        let _ = (engine, episode_id, root);
        Ok(context)
    }

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<OrchestratedEpisode<E::Graph, E::Candidate>>>;

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
    ) -> EngineResult<()>;

    fn observe_version(&mut self, version: ModelVersion) {
        let _ = version;
    }

    fn complete(
        &mut self,
        engine: &mut E,
        search: &GumbelMcts,
        completed: OrchestratedEpisode<E::Graph, E::Candidate>,
    ) -> EngineResult<()>;

    fn finish(self, lane: usize) -> Self::Output;
}

fn run_lane_pipeline<E, R, M>(
    mut engine: E,
    mut roots: R,
    runtime: LaneRuntime<'_, M::Job>,
    mut mode: M,
) -> EngineResult<M::Output>
where
    E: GraphEngine,
    R: RootSource<E>,
    M: LaneMode<E>,
{
    let identity = EngineIdentity::from_engine(&engine);
    let worker_id_base = (runtime.lane * runtime.workers_per_lane.get()) as u64;
    let mut pool = WorkerPool::new(runtime.workers_per_lane, worker_id_base);
    let mut roots_exhausted = false;
    let mut next_episode_id = (runtime.lane as u64) << 32;
    mode.begin(runtime.search, identity, runtime.context);

    loop {
        if !roots_exhausted {
            mode.before_root_admission(&mut pool, &mut engine, &mut roots, &mut next_episode_id)?;
            if mode.gate_open() {
                let mut admission = Admission {
                    search: runtime.search,
                    identity,
                    context: runtime.context,
                    next_episode_id: &mut next_episode_id,
                };
                roots_exhausted = pool.admit(
                    &mut engine,
                    &mut roots,
                    &mut admission,
                    |engine, id, root, context| mode.episode_context(engine, id, root, context),
                )?;
            } else if !pool.active()
                && let Some(gate_poll) = mode.gate_poll()
            {
                // The gate limits admission only. In-flight episodes always
                // finish, so backlog can overshoot by at most total workers
                // times rows per episode. This sleep is the throttled-idle
                // path that prevents a fully gated lane from busy-spinning.
                std::thread::sleep(gate_poll);
            }
        }

        for completed in mode.drive(&mut engine, &mut pool)? {
            mode.complete(&mut engine, runtime.search, completed)?;
        }

        mode.send_parked(runtime.lane, &mut pool, &runtime.intake_tx)?;

        if roots_exhausted && !pool.active() {
            return Ok(mode.finish(runtime.lane));
        }

        if pool.has_parked()
            && let Some(version) = receive_replies(&mut engine, &mut pool, &runtime.reply_rx)?
        {
            mode.observe_version(version);
        }
    }
}

struct CollectMode<G, C> {
    episodes: Vec<OrchestratedEpisode<G, C>>,
}

impl<G, C> CollectMode<G, C> {
    fn new() -> Self {
        Self {
            episodes: Vec::new(),
        }
    }
}

impl<E> LaneMode<E> for CollectMode<E::Graph, E::Candidate>
where
    E: GraphEngine,
{
    type Job = EvalJob;
    type Output = LaneEpisodes<E::Graph, E::Candidate>;

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<OrchestratedEpisode<E::Graph, E::Candidate>>> {
        pool.drive(engine, "worker blocked", None, |_, _, _| {})
    }

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
    ) -> EngineResult<()> {
        send_plain_parked(lane, pool, intake_tx)
    }

    fn complete(
        &mut self,
        engine: &mut E,
        _search: &GumbelMcts,
        completed: OrchestratedEpisode<E::Graph, E::Candidate>,
    ) -> EngineResult<()> {
        release_episode_handles(engine, &completed.episode, &[])?;
        self.episodes.push(completed);
        Ok(())
    }

    fn finish(self, lane: usize) -> Self::Output {
        LaneEpisodes {
            lane,
            episodes: self.episodes,
        }
    }
}

struct FeaturizedCollectMode<X, G, C> {
    extractor: X,
    episodes: Vec<OrchestratedEpisode<G, C>>,
}

impl<X, G, C> FeaturizedCollectMode<X, G, C> {
    fn new(extractor: X) -> Self {
        Self {
            extractor,
            episodes: Vec::new(),
        }
    }
}

impl<E, X> LaneMode<E> for FeaturizedCollectMode<X, E::Graph, E::Candidate>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    type Job = FeaturizedEvalJob;
    type Output = LaneEpisodes<E::Graph, E::Candidate>;

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<OrchestratedEpisode<E::Graph, E::Candidate>>> {
        pool.drive(
            engine,
            "worker blocked",
            Some(&mut self.extractor),
            |_, _, _| {},
        )
    }

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
    ) -> EngineResult<()> {
        send_featurized_parked(lane, pool, intake_tx)
    }

    fn complete(
        &mut self,
        engine: &mut E,
        _search: &GumbelMcts,
        completed: OrchestratedEpisode<E::Graph, E::Candidate>,
    ) -> EngineResult<()> {
        release_episode_handles(engine, &completed.episode, &[])?;
        self.episodes.push(completed);
        Ok(())
    }

    fn finish(self, lane: usize) -> Self::Output {
        LaneEpisodes {
            lane,
            episodes: self.episodes,
        }
    }
}

struct ReplayMode<'a, P> {
    provider: P,
    replay_tx: SyncSender<ReplayJob>,
    store: &'a ReplayStore,
    backpressure: Option<ReplayBackpressure>,
    references: HashMap<EpisodeId, Option<Reference>>,
    summary: ReplayLaneSummary,
    rollout: Option<OpponentRollout>,
}

impl<'a, P> ReplayMode<'a, P> {
    fn new(
        provider: P,
        replay_tx: SyncSender<ReplayJob>,
        store: &'a ReplayStore,
        backpressure: Option<ReplayBackpressure>,
    ) -> Self {
        Self {
            provider,
            replay_tx,
            store,
            backpressure,
            references: HashMap::new(),
            summary: ReplayLaneSummary {
                lane: 0,
                episodes_completed: 0,
                episodes_appended: 0,
                episodes_dropped: 0,
                search_contexts: 0,
                replay_rows: 0,
                reference_steps: 0,
            },
            rollout: None,
        }
    }
}

impl<E, P> LaneMode<E> for ReplayMode<'_, P>
where
    E: GraphEngine,
    P: ReferenceProvider<E>,
{
    type Job = EvalJob;
    type Output = ReplayLaneSummary;

    fn begin(
        &mut self,
        search: &GumbelMcts,
        identity: EngineIdentity,
        context: GumbelEpisodeContext,
    ) {
        self.rollout = Some(OpponentRollout::new(search, identity, context));
    }

    fn before_root_admission<R>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        next_episode_id: &mut u64,
    ) -> EngineResult<()>
    where
        R: RootSource<E>,
    {
        let mut rollout = self
            .rollout
            .take()
            .ok_or_else(|| internal("missing opponent rollout"))?;
        let result = rollout.try_admit(pool, engine, roots, &mut self.provider, next_episode_id);
        self.rollout = Some(rollout);
        result
    }

    fn gate_open(&self) -> bool {
        replay_gate_open(self.store, self.backpressure)
    }

    fn gate_poll(&self) -> Option<Duration> {
        self.backpressure.map(|backpressure| backpressure.gate_poll)
    }

    fn episode_context(
        &mut self,
        engine: &mut E,
        episode_id: EpisodeId,
        root: E::Graph,
        mut context: GumbelEpisodeContext,
    ) -> EngineResult<GumbelEpisodeContext> {
        let reference = self.provider.reference(engine, root)?;
        context.opponent = reference.as_ref().map(opponent_context);
        self.references.insert(episode_id, reference);
        Ok(context)
    }

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<OrchestratedEpisode<E::Graph, E::Candidate>>> {
        pool.drive(engine, "worker blocked", None, |_, _, _| {})
    }

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
    ) -> EngineResult<()> {
        send_plain_parked(lane, pool, intake_tx)
    }

    fn observe_version(&mut self, version: ModelVersion) {
        if let Some(rollout) = &mut self.rollout {
            rollout.observe_version(version);
        }
    }

    fn complete(
        &mut self,
        engine: &mut E,
        _search: &GumbelMcts,
        mut completed: OrchestratedEpisode<E::Graph, E::Candidate>,
    ) -> EngineResult<()> {
        let mut rollout = self
            .rollout
            .take()
            .ok_or_else(|| internal("missing opponent rollout"))?;
        if rollout.intercept(engine, &mut self.provider, &completed)? {
            self.rollout = Some(rollout);
            return Ok(());
        }
        self.rollout = Some(rollout);

        let reference = self
            .references
            .remove(&completed.episode_id)
            .ok_or_else(|| internal("missing replay reference"))?;
        self.summary.episodes_completed += 1;
        self.summary.search_contexts += episode_search_contexts(&completed.episode);
        self.summary.reference_steps += reference
            .as_ref()
            .map_or(0, |reference| reference.steps.len() as u64);

        if let Some((record, rows)) = project_episode(
            &completed.episode,
            reference.as_ref(),
            None,
            completed.episode_id.value(),
        ) {
            let reward = record.outcome.learner_reward;
            self.summary.replay_rows += rows.len() as u64;
            let append = append_replay_job(&self.replay_tx, record, rows);
            release_episode_handles(engine, &completed.episode, &[])?;
            append?;
            self.provider.observe(reward);
            self.summary.episodes_appended += 1;
        } else {
            release_episode_handles(engine, &completed.episode, &[])?;
            self.summary.episodes_dropped += 1;
        }

        clear_replayed_episode_trace(&mut completed.episode);
        Ok(())
    }

    fn finish(mut self, lane: usize) -> Self::Output {
        self.summary.lane = lane;
        self.summary
    }
}

struct FeaturizedReplayMode<'a, X, P> {
    extractor: X,
    replay: ReplayMode<'a, P>,
    candidate_options: CandidateOptions,
    export_position: bool,
}

impl<'a, X, P> FeaturizedReplayMode<'a, X, P> {
    fn new(
        extractor: X,
        provider: P,
        replay_tx: SyncSender<ReplayJob>,
        store: &'a ReplayStore,
        backpressure: Option<ReplayBackpressure>,
    ) -> Self {
        Self {
            extractor,
            replay: ReplayMode::new(provider, replay_tx, store, backpressure),
            candidate_options: CandidateOptions::default(),
            export_position: true,
        }
    }
}

impl<E, X, P> LaneMode<E> for FeaturizedReplayMode<'_, X, P>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
    P: ReferenceProvider<E>,
{
    type Job = FeaturizedEvalJob;
    type Output = ReplayLaneSummary;

    fn begin(
        &mut self,
        search: &GumbelMcts,
        identity: EngineIdentity,
        context: GumbelEpisodeContext,
    ) {
        self.replay.begin(search, identity, context);
        self.candidate_options = search.config().candidate_options;
        self.export_position = search.config().export_position;
    }

    fn before_root_admission<R>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        next_episode_id: &mut u64,
    ) -> EngineResult<()>
    where
        R: RootSource<E>,
    {
        self.replay
            .before_root_admission(pool, engine, roots, next_episode_id)
    }

    fn gate_open(&self) -> bool {
        self.replay.gate_open()
    }

    fn gate_poll(&self) -> Option<Duration> {
        self.replay.gate_poll()
    }

    fn episode_context(
        &mut self,
        engine: &mut E,
        episode_id: EpisodeId,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<GumbelEpisodeContext> {
        let reference = self.replay.provider.reference_with_features(
            engine,
            root,
            &mut self.extractor,
            self.candidate_options,
            self.export_position,
        )?;
        let mut context = context;
        context.opponent = reference.as_ref().map(opponent_context);
        self.replay.references.insert(episode_id, reference);
        Ok(context)
    }

    fn drive(
        &mut self,
        engine: &mut E,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
    ) -> EngineResult<Vec<OrchestratedEpisode<E::Graph, E::Candidate>>> {
        let references = &self.replay.references;
        pool.drive(
            engine,
            "worker blocked",
            Some(&mut self.extractor),
            |episode_id, root_step, row| {
                attach_reference_opponent(references, episode_id, root_step, row);
            },
        )
    }

    fn send_parked(
        &mut self,
        lane: usize,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        intake_tx: &SyncSender<Self::Job>,
    ) -> EngineResult<()> {
        send_featurized_parked(lane, pool, intake_tx)
    }

    fn observe_version(&mut self, version: ModelVersion) {
        self.replay.observe_version(version);
    }

    fn complete(
        &mut self,
        engine: &mut E,
        search: &GumbelMcts,
        mut completed: OrchestratedEpisode<E::Graph, E::Candidate>,
    ) -> EngineResult<()> {
        let mut rollout = self
            .replay
            .rollout
            .take()
            .ok_or_else(|| internal("missing opponent rollout"))?;
        if rollout.intercept_with_features(
            engine,
            &mut self.replay.provider,
            &completed,
            &mut self.extractor,
        )? {
            self.replay.rollout = Some(rollout);
            return Ok(());
        }
        self.replay.rollout = Some(rollout);

        let reference = self
            .replay
            .references
            .remove(&completed.episode_id)
            .ok_or_else(|| internal("missing replay reference"))?;
        let feature_rows = feature_rows_for_episode(
            engine,
            &mut self.extractor,
            search,
            &completed.episode,
            reference.as_ref(),
        )?;
        self.replay.summary.episodes_completed += 1;
        self.replay.summary.search_contexts += episode_search_contexts(&completed.episode);
        self.replay.summary.reference_steps += reference
            .as_ref()
            .map_or(0, |reference| reference.steps.len() as u64);

        if let Some((record, rows)) = project_episode(
            &completed.episode,
            reference.as_ref(),
            Some(&feature_rows.rows),
            completed.episode_id.value(),
        ) {
            let reward = record.outcome.learner_reward;
            self.replay.summary.replay_rows += rows.len() as u64;
            let append = append_replay_job(&self.replay.replay_tx, record, rows);
            release_episode_handles(engine, &completed.episode, &feature_rows.candidates)?;
            append?;
            self.replay.provider.observe(reward);
            self.replay.summary.episodes_appended += 1;
        } else {
            release_episode_handles(engine, &completed.episode, &feature_rows.candidates)?;
            self.replay.summary.episodes_dropped += 1;
        }

        clear_replayed_episode_trace(&mut completed.episode);
        Ok(())
    }

    fn finish(mut self, lane: usize) -> Self::Output {
        self.replay.summary.lane = lane;
        self.replay.summary
    }
}

fn send_featurized_parked<G, C>(
    lane: usize,
    pool: &mut WorkerPool<G, C>,
    intake_tx: &SyncSender<FeaturizedEvalJob>,
) -> EngineResult<()>
where
    G: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
{
    for parked in pool.take_unsent_parked() {
        let row = parked.row.ok_or_else(|| internal("missing feature row"))?;
        intake_tx
            .send(FeaturizedEvalJob {
                lane,
                slot: parked.slot,
                token: parked.token,
                row,
                action_count: parked.action_count,
            })
            .map_err(|_| internal("eval backend unavailable"))?;
    }
    Ok(())
}

fn episode_search_contexts<G, C>(episode: &GumbelEpisode<G, C>) -> u64 {
    episode
        .root_stats
        .iter()
        .map(|stats| stats.portable_contexts as u64)
        .sum()
}

fn send_plain_parked<G, C>(
    lane: usize,
    pool: &mut WorkerPool<G, C>,
    intake_tx: &SyncSender<EvalJob>,
) -> EngineResult<()>
where
    G: Copy + Eq + std::hash::Hash,
    C: Copy + Eq + std::hash::Hash,
{
    for parked in pool.take_unsent_parked() {
        intake_tx
            .send(EvalJob {
                lane,
                slot: parked.slot,
                token: parked.token,
                request: parked.request,
            })
            .map_err(|_| internal("eval backend unavailable"))?;
    }
    Ok(())
}

fn attach_reference_opponent(
    references: &HashMap<EpisodeId, Option<Reference>>,
    episode_id: EpisodeId,
    root_step: u32,
    row: &mut FeatureRow,
) {
    let Some(Some(reference)) = references.get(&episode_id) else {
        return;
    };
    attach_opponent_step(reference, root_step as usize, row);
}

fn attach_opponent_step(reference: &Reference, step_index: usize, row: &mut FeatureRow) {
    let Some(step) = aligned_reference_step(reference, step_index) else {
        return;
    };
    row.opponent = step.features.clone();
}

fn aligned_reference_step(
    reference: &Reference,
    step_index: usize,
) -> Option<&crate::reference::ReferenceStep> {
    if reference.steps.is_empty() {
        return None;
    }
    reference
        .steps
        .get(step_index)
        .or_else(|| reference.steps.last())
}

fn feature_rows_for_episode<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    episode: &GumbelEpisode<E::Graph, E::Candidate>,
    reference: Option<&Reference>,
) -> EngineResult<EpisodeFeatureRows<E::Candidate>>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let schema = extractor.schema().clone();
    let mut out = Vec::with_capacity(episode.steps.len());
    let mut candidates = Vec::new();
    let mut created_candidates = Vec::new();

    for (index, step) in episode.steps.iter().enumerate() {
        candidates.clear();
        engine.candidates(
            step.before,
            search.config().candidate_options,
            &mut candidates,
        )?;
        created_candidates.extend(candidates.iter().copied());
        // Mirror the eval-side export gate: rows must train the model on
        // the same position inputs it served with.
        let position = replay_position_features(search, extractor.schema(), index, reference)?;
        let mut row = extractor
            .extract(engine, step.before, &candidates, position)
            .map_err(|_| internal("feature extraction failed"))?;
        if let Some(reference) = reference {
            attach_opponent_step(reference, index, &mut row);
        }
        if row.actions.len() != step.legal_actions.len() {
            return Err(internal("feature row action count mismatch"));
        }

        let mut bytes = Vec::new();
        encode_feature_row(&row, &schema, &mut bytes)
            .map_err(|_| internal("feature row encoding failed"))?;
        out.push(bytes);
    }

    Ok(EpisodeFeatureRows {
        rows: out,
        candidates: created_candidates,
    })
}

fn opponent_context(reference: &Reference) -> GumbelOpponentContext {
    GumbelOpponentContext {
        trajectory_id: 0,
        row_count: reference.steps.len() as u32,
        final_reward: reference.final_reward,
    }
}

fn replay_position_features(
    search: &GumbelMcts,
    schema: &FeatureSchema,
    index: usize,
    reference: Option<&Reference>,
) -> EngineResult<PositionFeatures> {
    let (root_step, budget_fraction, budget_step) = if search.config().export_position {
        let (budget_fraction, budget_step) = search.root_budget(index);
        (
            u32::try_from(index).map_err(|_| internal("root step overflow"))?,
            budget_fraction,
            budget_step,
        )
    } else {
        (0, 0.0, 0.0)
    };
    let scale = schema.config().opponent_reward_scale;
    let opponent_reward = reference.map_or(0.0, |reference| reference.final_reward / scale);

    Ok(PositionFeatures {
        root_step,
        leaf_depth: 0,
        budget_fraction,
        budget_step,
        opponent_reward,
        opponent_present: reference.is_some(),
    })
}

fn release_episode_handles<E>(
    engine: &mut E,
    episode: &GumbelEpisode<E::Graph, E::Candidate>,
    extra_candidates: &[E::Candidate],
) -> EngineResult<()>
where
    E: GraphEngine,
{
    if extra_candidates.is_empty() {
        return engine.release(&episode.created_graphs, &episode.created_candidates);
    }

    let mut candidates =
        Vec::with_capacity(episode.created_candidates.len() + extra_candidates.len());
    candidates.extend_from_slice(&episode.created_candidates);
    candidates.extend_from_slice(extra_candidates);
    engine.release(&episode.created_graphs, &candidates)
}

fn reference_steps_for_gumbel_episode<G, C>(
    episode: &GumbelEpisode<G, C>,
) -> Vec<crate::reference::ReferenceStep> {
    let mut steps = Vec::with_capacity(episode.steps.len() + 1);
    match episode.steps.first() {
        Some(step) => steps.push(crate::reference::ReferenceStep {
            context: step.step_ref.before,
            features: None,
        }),
        None => steps.push(crate::reference::ReferenceStep {
            context: episode.final_context,
            features: None,
        }),
    }
    steps.extend(
        episode
            .steps
            .iter()
            .map(|step| crate::reference::ReferenceStep {
                context: step.step_ref.after,
                features: None,
            }),
    );
    steps
}

fn reference_steps_for_gumbel_episode_with_features<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    episode: &GumbelEpisode<E::Graph, E::Candidate>,
    final_reward: f32,
) -> EngineResult<(Vec<crate::reference::ReferenceStep>, Vec<E::Candidate>)>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let mut candidates = Vec::new();
    let steps = (|| {
        let mut steps = Vec::with_capacity(episode.steps.len() + 1);

        match episode.steps.first() {
            Some(step) => steps.push(reference_step_with_features(
                engine,
                extractor,
                search,
                GumbelReferenceStepInput {
                    graph: step.before,
                    context: step.step_ref.before,
                    index: 0,
                    final_reward,
                },
                &mut candidates,
            )?),
            None => steps.push(reference_step_with_features(
                engine,
                extractor,
                search,
                GumbelReferenceStepInput {
                    graph: episode.final_graph,
                    context: episode.final_context,
                    index: 0,
                    final_reward,
                },
                &mut candidates,
            )?),
        }

        for (index, step) in episode.steps.iter().enumerate() {
            steps.push(reference_step_with_features(
                engine,
                extractor,
                search,
                GumbelReferenceStepInput {
                    graph: step.after,
                    context: step.step_ref.after,
                    index: index + 1,
                    final_reward,
                },
                &mut candidates,
            )?);
        }

        Ok(steps)
    })();

    match steps {
        Ok(steps) => Ok((steps, candidates)),
        Err(error) => {
            engine.release(&[], &candidates)?;
            Err(error)
        }
    }
}

struct GumbelReferenceStepInput<G> {
    graph: G,
    context: gz_engine::ReplayGraphContext,
    index: usize,
    final_reward: f32,
}

fn reference_step_with_features<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    input: GumbelReferenceStepInput<E::Graph>,
    created_candidates: &mut Vec<E::Candidate>,
) -> EngineResult<crate::reference::ReferenceStep>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let mut candidates = Vec::new();
    engine.candidates(
        input.graph,
        search.config().candidate_options,
        &mut candidates,
    )?;
    created_candidates.extend(candidates.iter().copied());
    let position = replay_position_features(search, extractor.schema(), input.index, None)?;
    let scale = extractor.schema().config().opponent_reward_scale;
    let row = extractor
        .extract(
            engine,
            input.graph,
            &candidates,
            PositionFeatures {
                opponent_reward: input.final_reward / scale,
                opponent_present: true,
                ..position
            },
        )
        .map_err(|_| internal("reference feature extraction failed"))?;

    Ok(crate::reference::ReferenceStep {
        context: input.context,
        features: Some(OpponentStateFeatures {
            node_count: row.node_count,
            node_tokens: row.node_tokens,
            node_attrs: row.node_attrs,
            edges: row.edges,
            position: row.position,
        }),
    })
}

/// Drives opponent rollout episodes for rollout-based reference providers
/// (the policy opponent). Tracks the newest model version seen on eval
/// replies; when the provider reports a rollout due, admits one greedy
/// (single-simulation, no-noise) episode from the fixed root and feeds
/// its measured terminal reward back to the provider. Rollout episodes
/// never reach the replay store or the run summary.
struct OpponentRollout {
    search: GumbelMcts,
    identity: EngineIdentity,
    latest_version: Option<ModelVersion>,
    in_flight: Option<EpisodeId>,
}

impl OpponentRollout {
    fn new(search: &GumbelMcts, identity: EngineIdentity, _context: GumbelEpisodeContext) -> Self {
        Self {
            search: search.policy_rollout(),
            identity,
            latest_version: None,
            in_flight: None,
        }
    }

    fn observe_version(&mut self, version: ModelVersion) {
        self.latest_version = Some(version);
    }

    /// Runs before root admission so a busy pool cannot starve the
    /// rollout: the freed slot goes to the rollout first.
    fn try_admit<E, R, P>(
        &mut self,
        pool: &mut WorkerPool<E::Graph, E::Candidate>,
        engine: &mut E,
        roots: &mut R,
        provider: &mut P,
        next_episode_id: &mut u64,
    ) -> EngineResult<()>
    where
        E: GraphEngine,
        R: RootSource<E>,
        P: ReferenceProvider<E>,
    {
        if self.in_flight.is_some() {
            return Ok(());
        }
        let Some(version) = self.latest_version else {
            return Ok(());
        };
        if !provider.rollout_due(Some(version)) {
            return Ok(());
        }
        let Some(root) = roots.fixed_root(engine)? else {
            return Ok(());
        };

        let episode_id = EpisodeId::new(*next_episode_id);
        let admitted = pool.admit_direct(
            &self.search,
            self.identity,
            root,
            GumbelEpisodeContext {
                noise_seed: 0,
                opponent: None,
            },
            episode_id,
        );
        if admitted {
            *next_episode_id += 1;
            provider.begin_rollout(version);
            self.in_flight = Some(episode_id);
        }
        Ok(())
    }

    /// Claims a completed rollout episode: releases its handles and
    /// reports the outcome to the provider. Returns true when the episode
    /// was a rollout and must not be projected, appended, or counted.
    fn intercept<E, P>(
        &mut self,
        engine: &mut E,
        provider: &mut P,
        completed: &OrchestratedEpisode<E::Graph, E::Candidate>,
    ) -> EngineResult<bool>
    where
        E: GraphEngine,
        P: ReferenceProvider<E>,
    {
        if self.in_flight != Some(completed.episode_id) {
            return Ok(false);
        }
        self.in_flight = None;
        release_episode_handles(engine, &completed.episode, &[])?;

        let measure = &completed.episode.final_measure;
        let reward = if measure.measured && measure.valid {
            measure.scalar_reward.filter(|reward| reward.is_finite())
        } else {
            None
        };
        provider.finish_rollout(reward.map(|final_reward| RolloutOutcome {
            final_reward,
            final_graph: completed.episode.final_context,
            steps: reference_steps_for_gumbel_episode(&completed.episode),
            search_config_hash: completed.episode.search_config_hash,
        }));
        Ok(true)
    }

    fn intercept_with_features<E, P, X>(
        &mut self,
        engine: &mut E,
        provider: &mut P,
        completed: &OrchestratedEpisode<E::Graph, E::Candidate>,
        extractor: &mut X,
    ) -> EngineResult<bool>
    where
        E: GraphEngine,
        P: ReferenceProvider<E>,
        X: FeatureExtractor<E>,
    {
        if self.in_flight != Some(completed.episode_id) {
            return Ok(false);
        }
        self.in_flight = None;

        let measure = &completed.episode.final_measure;
        let reward = if measure.measured && measure.valid {
            measure.scalar_reward.filter(|reward| reward.is_finite())
        } else {
            None
        };

        let (steps, feature_candidates) = match reward {
            Some(final_reward) => reference_steps_for_gumbel_episode_with_features(
                engine,
                extractor,
                &self.search,
                &completed.episode,
                final_reward,
            )?,
            None => (Vec::new(), Vec::new()),
        };
        release_episode_handles(engine, &completed.episode, &feature_candidates)?;

        provider.finish_rollout(reward.map(|final_reward| RolloutOutcome {
            final_reward,
            final_graph: completed.episode.final_context,
            steps,
            search_config_hash: completed.episode.search_config_hash,
        }));
        Ok(true)
    }
}

fn clear_replayed_episode_trace<G, C>(episode: &mut GumbelEpisode<G, C>) {
    // Drop the backing buffers, not just the elements: clear() keeps
    // capacity, and created_candidates alone reaches millions of ids per
    // episode (~20 MB). Completed episodes are retained for the run
    // summary, so kept capacity is a per-episode leak on unbounded runs.
    episode.steps = Vec::new();
    episode.root_stats = Vec::new();
    episode.created_graphs = Vec::new();
    episode.created_candidates = Vec::new();
}

fn append_replay_job(
    replay_tx: &SyncSender<ReplayJob>,
    record: ReplayEpisodeRecord,
    rows: Vec<ReplayRow>,
) -> EngineResult<()> {
    let (ack, done) = sync_channel(1);
    replay_tx
        .send(ReplayJob { record, rows, ack })
        .map_err(|_| internal("replay sink failed"))?;
    done.recv().map_err(|_| internal("replay sink failed"))?
}

/// Resumes every pending reply; returns the newest model version seen so
/// callers can drive version-triggered opponent rollouts.
fn receive_replies<E>(
    engine: &mut E,
    pool: &mut WorkerPool<E::Graph, E::Candidate>,
    reply_rx: &Receiver<EvalReply>,
) -> EngineResult<Option<ModelVersion>>
where
    E: GraphEngine,
{
    let reply = reply_rx
        .recv()
        .map_err(|_| internal("eval backend unavailable"))?;
    let mut version = reply.output.model_version;
    pool.resume(engine, reply.slot, reply.token, reply.output)?;

    loop {
        match reply_rx.try_recv() {
            Ok(reply) => {
                version = reply.output.model_version;
                pool.resume(engine, reply.slot, reply.token, reply.output)?;
            }
            Err(TryRecvError::Empty) => return Ok(Some(version)),
            Err(TryRecvError::Disconnected) => return Err(internal("eval backend unavailable")),
        }
    }
}

fn replay_gate_open(store: &ReplayStore, backpressure: Option<ReplayBackpressure>) -> bool {
    let Some(backpressure) = backpressure else {
        return true;
    };
    let counters = store.counters();
    let backlog = counters
        .produced_rows
        .saturating_sub(counters.consumed_rows);

    backlog <= backpressure.max_row_backlog.get()
}

fn run_batcher<V>(
    mut evaluator: V,
    intake_rx: Receiver<EvalJob>,
    reply_txs: Vec<SyncSender<EvalReply>>,
    config: ThreadedOrchestratorConfig,
) -> EngineResult<Vec<usize>>
where
    V: Evaluator,
{
    let mut batch_sizes = Vec::new();

    loop {
        let first = match intake_rx.recv() {
            Ok(job) => job,
            Err(_) => return Ok(batch_sizes),
        };
        let mut batch = vec![first];
        let deadline = Instant::now() + config.flush_after;

        while batch.len() < config.max_batch.get() {
            let now = Instant::now();
            let remaining = deadline.saturating_duration_since(now);
            match intake_rx.recv_timeout(remaining) {
                Ok(job) => batch.push(job),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
            }
        }

        let requests = batch
            .iter()
            .map(|job| job.request.clone())
            .collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(requests.len());
        evaluator
            .evaluate_batch(&requests, &mut outputs)
            .map_err(eval_error_to_engine_error)?;
        validate_outputs(&requests, &outputs).map_err(eval_error_to_engine_error)?;
        batch_sizes.push(batch.len());

        for (job, output) in batch.into_iter().zip(outputs) {
            let _ = reply_txs[job.lane].send(EvalReply {
                slot: job.slot,
                token: job.token,
                output,
            });
        }
    }
}

/// Batches eval jobs and keeps one submitted batch in flight: while batch
/// N runs on the backend, batch N+1 is collected and submitted before N's
/// outputs are received, so a pipelining backend (the evaluator process)
/// overlaps its request read and staging with GPU compute. Non-pipelining
/// backends compute at submit and the loop degenerates to the historical
/// serial behavior.
///
/// Liveness: while a batch is in flight, collection is bounded by the
/// flush window and may come up empty (every parked eval can be inside
/// the in-flight batch, and new jobs only arrive after its replies), so
/// the loop always progresses to receive-and-route.
fn run_featurized_batcher<B>(
    mut backend: B,
    mut collator: FeatureCollator,
    intake_rx: Receiver<FeaturizedEvalJob>,
    reply_txs: Vec<SyncSender<EvalReply>>,
    config: ThreadedOrchestratorConfig,
) -> EngineResult<Vec<usize>>
where
    B: FeatureEvalBackend,
{
    type Routing = Vec<(usize, usize, WorkToken, u32)>;

    // Up to PIPELINE_DEPTH submitted batches ride the backend at once
    // (the evaluator moves outputs off its static buffers at launch, so
    // its GPU queue holds a batch while the previous one drains); replies
    // are FIFO. Depth 2 hides the server's per-batch host work under the
    // preceding batch's compute.
    const PIPELINE_DEPTH: usize = 2;

    let mut batch_sizes = Vec::new();
    let mut batch = Vec::with_capacity(config.max_batch.get());
    let mut rows = Vec::with_capacity(config.max_batch.get());
    let mut action_counts = Vec::with_capacity(config.max_batch.get());
    let mut bytes = Vec::new();
    let mut in_flight: std::collections::VecDeque<(Routing, gz_eval_service::PendingBatch)> =
        std::collections::VecDeque::with_capacity(PIPELINE_DEPTH);
    let mut intake_open = true;

    while intake_open || !in_flight.is_empty() {
        batch.clear();
        if intake_open && in_flight.len() < PIPELINE_DEPTH {
            if in_flight.is_empty() {
                // Nothing on the backend: block for work.
                match intake_rx.recv() {
                    Ok(job) => batch.push(job),
                    Err(_) => intake_open = false,
                }
            } else {
                // A batch is in flight: collect only within the flush
                // window so its replies are never held hostage to intake.
                match intake_rx.recv_timeout(config.flush_after) {
                    Ok(job) => batch.push(job),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => intake_open = false,
                }
            }
            if !batch.is_empty() {
                let deadline = Instant::now() + config.flush_after;
                while batch.len() < config.max_batch.get() {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    match intake_rx.recv_timeout(remaining) {
                        Ok(job) => batch.push(job),
                        Err(RecvTimeoutError::Timeout) => break,
                        Err(RecvTimeoutError::Disconnected) => {
                            intake_open = false;
                            break;
                        }
                    }
                }
            }
        }

        let submitted = if batch.is_empty() {
            false
        } else {
            let mut routing: Routing = Vec::with_capacity(batch.len());
            rows.clear();
            action_counts.clear();
            for job in batch.drain(..) {
                routing.push((job.lane, job.slot, job.token, job.action_count));
                action_counts.push(job.action_count);
                rows.push(job.row);
            }
            collator
                .collate_into(&rows, &mut bytes)
                .map_err(|_| internal("feature collation failed"))?;
            let pending = backend
                .submit(&bytes, &action_counts)
                .map_err(|_| internal("feature eval backend failed"))?;
            in_flight.push_back((routing, pending));
            true
        };

        // Drain the oldest reply when the pipeline is full, when this
        // round collected nothing (idle lanes are waiting on replies),
        // or when intake closed and only the tail remains.
        let must_drain = in_flight.len() >= PIPELINE_DEPTH || (!submitted && !in_flight.is_empty());
        if !must_drain {
            continue;
        }
        if let Some((routing, pending)) = in_flight.pop_front() {
            let outputs = backend
                .receive(pending)
                .map_err(|_| internal("feature eval backend failed"))?;
            let counts = routing
                .iter()
                .map(|&(_, _, _, action_count)| action_count)
                .collect::<Vec<_>>();
            validate_backend_outputs(&outputs, &counts)?;
            batch_sizes.push(routing.len());

            for ((lane, slot, token, _), row) in routing.into_iter().zip(outputs.rows) {
                let _ = reply_txs[lane].send(EvalReply {
                    slot,
                    token,
                    output: EvalOutput {
                        model_version: outputs.model_version,
                        policy_logits: row.policy_logits,
                        value: row.value,
                    },
                });
            }
        }
    }

    Ok(batch_sizes)
}

fn validate_backend_outputs(outputs: &BackendOutputs, action_counts: &[u32]) -> EngineResult<()> {
    if outputs.rows.len() != action_counts.len() {
        return Err(internal("eval output count mismatch"));
    }
    for (row, &action_count) in outputs.rows.iter().zip(action_counts) {
        if row.policy_logits.len() != action_count as usize {
            return Err(internal("eval output length mismatch"));
        }
        if !row.value.is_finite() || row.policy_logits.iter().any(|value| !value.is_finite()) {
            return Err(internal("invalid eval output"));
        }
    }
    Ok(())
}

fn run_replay_sink(store: &ReplayStore, replay_rx: Receiver<ReplayJob>) -> EngineResult<u64> {
    let mut episodes_appended = 0;

    while let Ok(job) = replay_rx.recv() {
        let result = store
            .append_episode(&job.record, &job.rows)
            .map(|_| ())
            .map_err(map_replay_error);
        let failed = result.clone().err();
        let _ = job.ack.send(result);
        if let Some(error) = failed {
            return Err(error);
        }
        episodes_appended += 1;
    }

    Ok(episodes_appended)
}

fn map_replay_error(error: ReplayError) -> EngineError {
    EngineError::Internal {
        code: ErrorCode::new(1),
        message: ErrorMessage::new(format!("replay sink failed: {error}"))
            .expect("replay error message is bounded"),
    }
}

fn validate_engine_identities<E>(engines: &[E]) -> EngineResult<()>
where
    E: GraphEngine,
{
    let Some(first) = engines.first().map(EngineIdentity::from_engine) else {
        return Ok(());
    };
    for engine in &engines[1..] {
        if EngineIdentity::from_engine(engine) != first {
            return Err(internal("engine identity mismatch"));
        }
    }
    Ok(())
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
