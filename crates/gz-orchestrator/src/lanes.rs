use crate::EpisodeId;
use crate::pool::{Admission, WorkerPool};
use crate::project::project_episode;
use crate::reference::{Reference, ReferenceProvider, RolloutOutcome};
use crate::root::RootSource;
use crate::serial::OrchestratedEpisode;
use crate::service::internal;
use gz_engine::{EngineResult, GraphEngine, ModelVersion};
use gz_eval::{EvalOutput, EvalRequest, Evaluator, eval_error_to_engine_error, validate_outputs};
use gz_eval_service::{BackendOutputs, FeatureEvalBackend};
use gz_features::{
    FeatureCollator, FeatureExtractor, FeatureRow, FeatureSchema, FeatureSchemaHash,
    PositionFeatures, encode_feature_row,
};
use gz_replay::{ReplayEpisodeRecord, ReplayError, ReplayRow, ReplayStore};
use gz_search::{EngineIdentity, GumbelEpisode, GumbelEpisodeContext, GumbelMcts, WorkToken};
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
    pub episodes: Vec<OrchestratedEpisode<G, C>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreadedRun<G, C> {
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
    pub backend: B,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplayBackpressure {
    pub max_row_backlog: NonZeroU64,
    pub gate_poll: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThreadedReplayRun<G, C> {
    pub run: ThreadedRun<G, C>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
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
                    run_lane(
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
    ) -> EngineResult<ThreadedReplayRun<E::Graph, E::Candidate>>
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
                    run_lane_with_replay(
                        engine,
                        roots,
                        provider,
                        ReplayLaneRuntime {
                            lane,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            context,
                            intake_tx,
                            reply_rx,
                            replay_tx,
                            store,
                            backpressure,
                        },
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

        for result in lane_results {
            let result = result?;
            episodes_dropped += result.episodes_dropped;
            lanes.push(result.lane);
        }

        Ok(ThreadedReplayRun {
            run: ThreadedRun { lanes, batch_sizes },
            episodes_appended,
            episodes_dropped,
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

        let workers_per_lane = self.config.workers_per_lane.get();
        let intake_capacity = lanes * workers_per_lane;
        let (intake_tx, intake_rx) = sync_channel(intake_capacity);
        let mut reply_txs = Vec::with_capacity(lanes);
        let mut reply_rxs = Vec::with_capacity(lanes);

        for _ in 0..lanes {
            let (tx, rx) = sync_channel(workers_per_lane);
            reply_txs.push(tx);
            reply_rxs.push(rx);
        }

        let config = self.config;
        let search = &self.search;
        let backend = featurized.backend;
        let extractors = featurized.extractors;
        let engines = self.engines;
        let feature_schema = first_schema::<E, X>(&extractors, schema_hash)?;
        let collator = FeatureCollator::new(feature_schema, config.max_batch);
        validate_collator_capacity(&collator, config)?;
        let _ = self.evaluator;

        let (batch_result, lane_results) = std::thread::scope(|scope| {
            let batch_handle = scope.spawn(move || {
                run_featurized_batcher(backend, collator, intake_rx, reply_txs, config)
            });
            let mut lane_handles = Vec::with_capacity(lanes);

            for (lane, (((engine, roots), extractor), reply_rx)) in engines
                .into_iter()
                .zip(root_sources)
                .zip(extractors)
                .zip(reply_rxs)
                .enumerate()
            {
                let intake_tx = intake_tx.clone();
                lane_handles.push(scope.spawn(move || {
                    run_featurized_lane(
                        engine,
                        roots,
                        extractor,
                        FeaturizedLaneRuntime {
                            lane,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            context,
                            intake_tx,
                            reply_rx,
                        },
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

    pub fn run_featurized_with_replay<R, X, B, P>(
        self,
        root_sources: Vec<R>,
        context: GumbelEpisodeContext,
        featurized: FeaturizedRuntime<X, B>,
        replay: ReplayRuntime<'_, P>,
    ) -> EngineResult<ThreadedReplayRun<E::Graph, E::Candidate>>
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
        let backend = featurized.backend;
        let extractors = featurized.extractors;
        let engines = self.engines;
        let providers = replay.providers;
        let store = replay.store;
        let backpressure = replay.backpressure;
        let feature_schema = first_schema::<E, X>(&extractors, schema_hash)?;
        store
            .ensure_feature_schema(feature_schema.config())
            .map_err(map_replay_error)?;
        let collator = FeatureCollator::new(feature_schema, config.max_batch);
        validate_collator_capacity(&collator, config)?;
        let _ = self.evaluator;

        let (batch_result, sink_result, lane_results) = std::thread::scope(|scope| {
            let batch_handle = scope.spawn(move || {
                run_featurized_batcher(backend, collator, intake_rx, reply_txs, config)
            });
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
                let intake_tx = intake_tx.clone();
                let replay_tx = replay_tx.clone();
                lane_handles.push(scope.spawn(move || {
                    run_featurized_lane_with_replay(
                        engine,
                        roots,
                        extractor,
                        provider,
                        FeaturizedReplayLaneRuntime {
                            lane,
                            search,
                            workers_per_lane: config.workers_per_lane,
                            context,
                            intake_tx,
                            reply_rx,
                            replay_tx,
                            store,
                            backpressure,
                        },
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

        for result in lane_results {
            let result = result?;
            episodes_dropped += result.episodes_dropped;
            lanes.push(result.lane);
        }

        Ok(ThreadedReplayRun {
            run: ThreadedRun { lanes, batch_sizes },
            episodes_appended,
            episodes_dropped,
        })
    }
}

struct LaneRuntime<'a> {
    lane: usize,
    search: &'a GumbelMcts,
    workers_per_lane: NonZeroUsize,
    context: GumbelEpisodeContext,
    intake_tx: SyncSender<EvalJob>,
    reply_rx: Receiver<EvalReply>,
}

struct ReplayLaneRuntime<'a> {
    lane: usize,
    search: &'a GumbelMcts,
    workers_per_lane: NonZeroUsize,
    context: GumbelEpisodeContext,
    intake_tx: SyncSender<EvalJob>,
    reply_rx: Receiver<EvalReply>,
    replay_tx: SyncSender<ReplayJob>,
    store: &'a ReplayStore,
    backpressure: Option<ReplayBackpressure>,
}

struct FeaturizedLaneRuntime<'a> {
    lane: usize,
    search: &'a GumbelMcts,
    workers_per_lane: NonZeroUsize,
    context: GumbelEpisodeContext,
    intake_tx: SyncSender<FeaturizedEvalJob>,
    reply_rx: Receiver<EvalReply>,
}

struct FeaturizedReplayLaneRuntime<'a> {
    lane: usize,
    search: &'a GumbelMcts,
    workers_per_lane: NonZeroUsize,
    context: GumbelEpisodeContext,
    intake_tx: SyncSender<FeaturizedEvalJob>,
    reply_rx: Receiver<EvalReply>,
    replay_tx: SyncSender<ReplayJob>,
    store: &'a ReplayStore,
    backpressure: Option<ReplayBackpressure>,
}

struct ReplayLaneResult<G, C> {
    lane: LaneEpisodes<G, C>,
    episodes_dropped: u64,
}

struct EpisodeFeatureRows<C> {
    rows: Vec<Vec<u8>>,
    candidates: Vec<C>,
}

fn run_lane<E, R>(
    mut engine: E,
    mut roots: R,
    runtime: LaneRuntime<'_>,
) -> EngineResult<LaneEpisodes<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    R: RootSource<E>,
{
    let identity = EngineIdentity::from_engine(&engine);
    let worker_id_base = (runtime.lane * runtime.workers_per_lane.get()) as u64;
    let mut pool = WorkerPool::new(runtime.workers_per_lane, worker_id_base);
    let mut episodes = Vec::new();
    let mut roots_exhausted = false;
    let mut next_episode_id = (runtime.lane as u64) << 32;

    loop {
        if !roots_exhausted {
            let mut admission = Admission {
                search: runtime.search,
                identity,
                context: runtime.context,
                next_episode_id: &mut next_episode_id,
            };
            roots_exhausted = pool.admit(&mut engine, &mut roots, &mut admission)?.1;
        }

        for completed in pool.drive(&mut engine, "worker blocked", None)? {
            release_episode_handles(&mut engine, &completed.episode, &[])?;
            episodes.push(completed);
        }

        for parked in pool.take_unsent_parked() {
            runtime
                .intake_tx
                .send(EvalJob {
                    lane: runtime.lane,
                    slot: parked.slot,
                    token: parked.token,
                    request: parked.request,
                })
                .map_err(|_| internal("eval backend unavailable"))?;
        }

        if roots_exhausted && !pool.active() {
            return Ok(LaneEpisodes {
                lane: runtime.lane,
                episodes,
            });
        }

        if pool.has_parked() {
            let reply = runtime
                .reply_rx
                .recv()
                .map_err(|_| internal("eval backend unavailable"))?;
            pool.resume(reply.slot, reply.token, reply.output)?;

            loop {
                match runtime.reply_rx.try_recv() {
                    Ok(reply) => pool.resume(reply.slot, reply.token, reply.output)?,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        return Err(internal("eval backend unavailable"));
                    }
                }
            }
        }
    }
}

fn run_lane_with_replay<E, R, P>(
    mut engine: E,
    mut roots: R,
    mut provider: P,
    runtime: ReplayLaneRuntime<'_>,
) -> EngineResult<ReplayLaneResult<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    R: RootSource<E>,
    P: ReferenceProvider<E>,
{
    let identity = EngineIdentity::from_engine(&engine);
    let worker_id_base = (runtime.lane * runtime.workers_per_lane.get()) as u64;
    let mut pool = WorkerPool::new(runtime.workers_per_lane, worker_id_base);
    let mut episodes = Vec::new();
    let mut references = HashMap::<EpisodeId, Option<Reference<E::Graph>>>::new();
    let mut roots_exhausted = false;
    let mut next_episode_id = (runtime.lane as u64) << 32;
    let mut episodes_dropped = 0;
    let mut rollout = OpponentRollout::new(runtime.search, identity, runtime.context);

    loop {
        if !roots_exhausted {
            rollout.try_admit(
                &mut pool,
                &mut engine,
                &mut roots,
                &mut provider,
                &mut next_episode_id,
            )?;
            if replay_gate_open(runtime.store, runtime.backpressure) {
                let mut admission = Admission {
                    search: runtime.search,
                    identity,
                    context: runtime.context,
                    next_episode_id: &mut next_episode_id,
                };
                let (admitted, exhausted) = pool.admit(&mut engine, &mut roots, &mut admission)?;
                roots_exhausted = exhausted;

                for (episode_id, root) in admitted {
                    references.insert(episode_id, provider.reference(&mut engine, root)?);
                }
            } else if !pool.active()
                && let Some(backpressure) = runtime.backpressure
            {
                // The gate limits admission only. In-flight episodes always
                // finish, so backlog can overshoot by at most total workers
                // times rows per episode. This sleep is the throttled-idle
                // path that prevents a fully gated lane from busy-spinning.
                std::thread::sleep(backpressure.gate_poll);
            }
        }

        for mut completed in pool.drive(&mut engine, "worker blocked", None)? {
            if rollout.intercept(&mut engine, &mut provider, &completed)? {
                continue;
            }
            let reference = references
                .remove(&completed.episode_id)
                .ok_or_else(|| internal("missing replay reference"))?;

            if let Some((record, rows)) =
                project_episode(&completed.episode, reference.as_ref(), None)
            {
                let reward = record.outcome.learner_reward;
                let append = append_replay_job(&runtime.replay_tx, record, rows);
                release_episode_handles(&mut engine, &completed.episode, &[])?;
                append?;
                provider.observe(reward);
            } else {
                episodes_dropped += 1;
                release_episode_handles(&mut engine, &completed.episode, &[])?;
            }

            clear_replayed_episode_trace(&mut completed.episode);
            episodes.push(completed);
        }

        for parked in pool.take_unsent_parked() {
            runtime
                .intake_tx
                .send(EvalJob {
                    lane: runtime.lane,
                    slot: parked.slot,
                    token: parked.token,
                    request: parked.request,
                })
                .map_err(|_| internal("eval backend unavailable"))?;
        }

        if roots_exhausted && !pool.active() {
            return Ok(ReplayLaneResult {
                lane: LaneEpisodes {
                    lane: runtime.lane,
                    episodes,
                },
                episodes_dropped,
            });
        }

        if pool.has_parked() {
            let reply = runtime
                .reply_rx
                .recv()
                .map_err(|_| internal("eval backend unavailable"))?;
            rollout.observe_version(reply.output.model_version);
            pool.resume(reply.slot, reply.token, reply.output)?;

            loop {
                match runtime.reply_rx.try_recv() {
                    Ok(reply) => {
                        rollout.observe_version(reply.output.model_version);
                        pool.resume(reply.slot, reply.token, reply.output)?;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        return Err(internal("eval backend unavailable"));
                    }
                }
            }
        }
    }
}

fn run_featurized_lane<E, R, X>(
    mut engine: E,
    mut roots: R,
    mut extractor: X,
    runtime: FeaturizedLaneRuntime<'_>,
) -> EngineResult<LaneEpisodes<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    R: RootSource<E>,
    X: FeatureExtractor<E>,
{
    let identity = EngineIdentity::from_engine(&engine);
    let worker_id_base = (runtime.lane * runtime.workers_per_lane.get()) as u64;
    let mut pool = WorkerPool::new(runtime.workers_per_lane, worker_id_base);
    let mut episodes = Vec::new();
    let mut roots_exhausted = false;
    let mut next_episode_id = (runtime.lane as u64) << 32;

    loop {
        if !roots_exhausted {
            let mut admission = Admission {
                search: runtime.search,
                identity,
                context: runtime.context,
                next_episode_id: &mut next_episode_id,
            };
            roots_exhausted = pool.admit(&mut engine, &mut roots, &mut admission)?.1;
        }

        for completed in pool.drive(&mut engine, "worker blocked", Some(&mut extractor))? {
            release_episode_handles(&mut engine, &completed.episode, &[])?;
            episodes.push(completed);
        }
        send_featurized_parked(runtime.lane, &mut pool, &runtime.intake_tx)?;

        if roots_exhausted && !pool.active() {
            return Ok(LaneEpisodes {
                lane: runtime.lane,
                episodes,
            });
        }

        if pool.has_parked() {
            receive_replies(&mut pool, &runtime.reply_rx)?;
        }
    }
}

fn run_featurized_lane_with_replay<E, R, X, P>(
    mut engine: E,
    mut roots: R,
    mut extractor: X,
    mut provider: P,
    runtime: FeaturizedReplayLaneRuntime<'_>,
) -> EngineResult<ReplayLaneResult<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    R: RootSource<E>,
    X: FeatureExtractor<E>,
    P: ReferenceProvider<E>,
{
    let identity = EngineIdentity::from_engine(&engine);
    let worker_id_base = (runtime.lane * runtime.workers_per_lane.get()) as u64;
    let mut pool = WorkerPool::new(runtime.workers_per_lane, worker_id_base);
    let mut episodes = Vec::new();
    let mut references = HashMap::<EpisodeId, Option<Reference<E::Graph>>>::new();
    let mut roots_exhausted = false;
    let mut next_episode_id = (runtime.lane as u64) << 32;
    let mut episodes_dropped = 0;
    let mut rollout = OpponentRollout::new(runtime.search, identity, runtime.context);

    loop {
        if !roots_exhausted {
            rollout.try_admit(
                &mut pool,
                &mut engine,
                &mut roots,
                &mut provider,
                &mut next_episode_id,
            )?;
            if replay_gate_open(runtime.store, runtime.backpressure) {
                let mut admission = Admission {
                    search: runtime.search,
                    identity,
                    context: runtime.context,
                    next_episode_id: &mut next_episode_id,
                };
                let (admitted, exhausted) = pool.admit(&mut engine, &mut roots, &mut admission)?;
                roots_exhausted = exhausted;

                for (episode_id, root) in admitted {
                    references.insert(episode_id, provider.reference(&mut engine, root)?);
                }
            } else if !pool.active()
                && let Some(backpressure) = runtime.backpressure
            {
                std::thread::sleep(backpressure.gate_poll);
            }
        }

        for mut completed in pool.drive(&mut engine, "worker blocked", Some(&mut extractor))? {
            if rollout.intercept(&mut engine, &mut provider, &completed)? {
                continue;
            }
            let reference = references
                .remove(&completed.episode_id)
                .ok_or_else(|| internal("missing replay reference"))?;

            let feature_rows = feature_rows_for_episode(
                &mut engine,
                &mut extractor,
                runtime.search,
                &completed.episode,
            )?;

            if let Some((record, rows)) = project_episode(
                &completed.episode,
                reference.as_ref(),
                Some(&feature_rows.rows),
            ) {
                let reward = record.outcome.learner_reward;
                let append = append_replay_job(&runtime.replay_tx, record, rows);
                release_episode_handles(&mut engine, &completed.episode, &feature_rows.candidates)?;
                append?;
                provider.observe(reward);
            } else {
                episodes_dropped += 1;
                release_episode_handles(&mut engine, &completed.episode, &feature_rows.candidates)?;
            }

            clear_replayed_episode_trace(&mut completed.episode);
            episodes.push(completed);
        }

        send_featurized_parked(runtime.lane, &mut pool, &runtime.intake_tx)?;

        if roots_exhausted && !pool.active() {
            return Ok(ReplayLaneResult {
                lane: LaneEpisodes {
                    lane: runtime.lane,
                    episodes,
                },
                episodes_dropped,
            });
        }

        if pool.has_parked()
            && let Some(version) = receive_replies(&mut pool, &runtime.reply_rx)?
        {
            rollout.observe_version(version);
        }
    }
}

fn send_featurized_parked<G, C>(
    lane: usize,
    pool: &mut WorkerPool<G, C>,
    intake_tx: &SyncSender<FeaturizedEvalJob>,
) -> EngineResult<()>
where
    G: Copy + Eq,
    C: Copy,
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

fn feature_rows_for_episode<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    episode: &GumbelEpisode<E::Graph, E::Candidate>,
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
        let (budget_fraction, budget_step) = search.root_budget(index);
        let root_step = u32::try_from(index).map_err(|_| internal("root step overflow"))?;
        let row = extractor
            .extract(
                engine,
                step.before,
                &candidates,
                PositionFeatures {
                    root_step,
                    leaf_depth: 0,
                    budget_fraction,
                    budget_step,
                },
            )
            .map_err(|_| internal("feature extraction failed"))?;
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

/// Drives opponent rollout episodes for rollout-based reference providers
/// (the policy opponent). Tracks the newest model version seen on eval
/// replies; when the provider reports a rollout due, admits one greedy
/// (single-simulation, no-noise) episode from the fixed root and feeds
/// its measured terminal reward back to the provider. Rollout episodes
/// never reach the replay store or the run summary.
struct OpponentRollout {
    search: GumbelMcts,
    identity: EngineIdentity,
    context: GumbelEpisodeContext,
    latest_version: Option<ModelVersion>,
    in_flight: Option<EpisodeId>,
}

impl OpponentRollout {
    fn new(search: &GumbelMcts, identity: EngineIdentity, context: GumbelEpisodeContext) -> Self {
        Self {
            search: search.policy_rollout(),
            identity,
            context,
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
                ..self.context
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
fn receive_replies<G, C>(
    pool: &mut WorkerPool<G, C>,
    reply_rx: &Receiver<EvalReply>,
) -> EngineResult<Option<ModelVersion>>
where
    G: Copy + Eq,
    C: Copy,
{
    let reply = reply_rx
        .recv()
        .map_err(|_| internal("eval backend unavailable"))?;
    let mut version = reply.output.model_version;
    pool.resume(reply.slot, reply.token, reply.output)?;

    loop {
        match reply_rx.try_recv() {
            Ok(reply) => {
                version = reply.output.model_version;
                pool.resume(reply.slot, reply.token, reply.output)?;
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
    let mut batch_sizes = Vec::new();
    let mut batch = Vec::with_capacity(config.max_batch.get());
    let mut routing = Vec::with_capacity(config.max_batch.get());
    let mut rows = Vec::with_capacity(config.max_batch.get());
    let mut action_counts = Vec::with_capacity(config.max_batch.get());
    let mut bytes = Vec::new();

    loop {
        let first = match intake_rx.recv() {
            Ok(job) => job,
            Err(_) => return Ok(batch_sizes),
        };
        batch.clear();
        batch.push(first);
        let deadline = Instant::now() + config.flush_after;

        while batch.len() < config.max_batch.get() {
            let now = Instant::now();
            let remaining = deadline.saturating_duration_since(now);
            match intake_rx.recv_timeout(remaining) {
                Ok(job) => batch.push(job),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
            }
        }

        routing.clear();
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
        let outputs = backend
            .eval(&bytes, &action_counts)
            .map_err(|_| internal("feature eval backend failed"))?;
        validate_backend_outputs(&outputs, &action_counts)?;
        batch_sizes.push(routing.len());

        for ((lane, slot, token, _), row) in routing.drain(..).zip(outputs.rows) {
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

fn map_replay_error(_error: ReplayError) -> gz_engine::EngineError {
    internal("replay sink failed")
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

fn validate_collator_capacity(
    collator: &FeatureCollator,
    config: ThreadedOrchestratorConfig,
) -> EngineResult<()> {
    if collator.batch_capacity() != config.max_batch {
        return Err(internal("feature batch capacity mismatch"));
    }
    Ok(())
}
