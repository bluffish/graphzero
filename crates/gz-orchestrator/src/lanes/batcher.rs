use super::{
    EvalReply, FeaturizedBatcherContext, FeaturizedEvalJob, LaneReply, ThreadedOrchestratorConfig,
};
use crate::admission::{EVAL_PIPELINE_DEPTH, EvalPressure};
use crate::internal;
use crate::leases::ModelLeaseRegistry;
use gz_engine::EngineResult;
use gz_eval::EvalOutput;
use gz_eval_service::{BackendOutputs, FeatureEvalBackend, ModelGeneration};
use gz_features::FeatureCollator;
use gz_search::WorkToken;
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::time::{Duration, Instant};

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
pub(super) fn run_featurized_batcher<B, G>(
    mut backend: B,
    mut collator: FeatureCollator,
    intake_rx: Receiver<FeaturizedEvalJob>,
    reply_txs: Vec<SyncSender<LaneReply<G>>>,
    config: ThreadedOrchestratorConfig,
    context: FeaturizedBatcherContext,
) -> EngineResult<Vec<usize>>
where
    B: FeatureEvalBackend,
    G: Send,
{
    type Routing = BatcherRouting;
    let max_batch = collator.batch_capacity().get();

    // Up to PIPELINE_DEPTH submitted batches ride the backend at once
    // (the evaluator moves outputs off its static buffers at launch, so
    // its GPU queue holds a batch while the previous one drains); replies
    // are FIFO. Depth 3: one computing, one staged behind it, one in the
    // socket buffer, so the server never starves between client drains.
    // Machine-parsed by the trainer driver (eval fill metrics); field
    // changes must update its parser. Counters are cumulative: the
    // driver computes rates and window means from deltas.
    const STATS_INTERVAL: Duration = Duration::from_secs(30);

    let mut batch_sizes = Vec::new();
    let mut batch = Vec::with_capacity(max_batch);
    let mut rows = Vec::with_capacity(max_batch);
    let mut action_counts = Vec::with_capacity(max_batch);
    let mut bytes = Vec::new();
    let mut deferred: VecDeque<FeaturizedEvalJob> = VecDeque::with_capacity(max_batch);
    let mut in_flight: VecDeque<(Routing, gz_eval_service::PendingBatch, ModelGeneration)> =
        VecDeque::with_capacity(EVAL_PIPELINE_DEPTH);
    let mut capacity_accounted_at = None;
    let mut intake_open = true;
    let mut stats_batches: usize = 0;
    let mut last_stats = Instant::now();

    while intake_open || !in_flight.is_empty() || !deferred.is_empty() {
        release_releasable_models(&mut backend, &context.model_registry)?;
        batch.clear();
        let mut batch_model = None;
        if in_flight.len() < EVAL_PIPELINE_DEPTH && (intake_open || !deferred.is_empty()) {
            if let Some(first) = deferred.pop_front() {
                batch_model = Some(first.model);
                batch.push(first);
                let queued = deferred.len();
                for _ in 0..queued {
                    let job = deferred
                        .pop_front()
                        .expect("deferred eval queue length changed");
                    if batch.len() < max_batch && Some(job.model) == batch_model {
                        batch.push(job);
                    } else {
                        deferred.push_back(job);
                    }
                }
            }
            // Fill toward a FULL batch. The evaluator's buffers (and its
            // CUDA-graph forward) are capacity-shaped, so a half batch
            // costs the same GPU time as a full one: padding rows are
            // pure waste. While the backend holds work, a partial batch
            // therefore waits -- each flush-window timeout drains the
            // oldest reply instead, and the workers that unblocks come
            // straight back with new evals to finish the fill. Only a
            // backend about to go idle flushes a partial batch.
            loop {
                if batch.len() >= max_batch {
                    break;
                }
                if batch.is_empty() && in_flight.is_empty() {
                    // Nothing anywhere: block for work.
                    match intake_rx.recv() {
                        Ok(job) => {
                            batch_model = Some(job.model);
                            batch.push(job);
                        }
                        Err(_) => {
                            intake_open = false;
                            break;
                        }
                    }
                    continue;
                }
                if !intake_open {
                    break;
                }
                match intake_rx.recv_timeout(config.flush_after) {
                    Ok(job) => {
                        if batch_model.is_none() {
                            batch_model = Some(job.model);
                            batch.push(job);
                        } else if Some(job.model) == batch_model {
                            batch.push(job);
                        } else {
                            deferred.push_back(job);
                            if deferred.len() >= max_batch {
                                break;
                            }
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if in_flight.is_empty() {
                            // Backend idle: ship what we have now.
                            break;
                        }
                        drain_oldest(
                            &mut backend,
                            &mut in_flight,
                            &reply_txs,
                            &mut batch_sizes,
                            &context,
                            max_batch,
                            EvalCapacityAccounting {
                                pressure: context.eval_pressure.as_deref(),
                                accounted_at: &mut capacity_accounted_at,
                            },
                        )?;
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        intake_open = false;
                        break;
                    }
                }
            }
        }

        let submitted = if batch.is_empty() {
            false
        } else {
            let model = batch_model.ok_or_else(|| internal("missing eval batch model"))?;
            let mut routing: Routing = Vec::with_capacity(batch.len());
            rows.clear();
            action_counts.clear();
            for job in batch.drain(..) {
                if job.model != model {
                    return Err(internal("mixed model generations in eval batch"));
                }
                routing.push((job.lane, job.slot, job.token, job.action_count));
                action_counts.push(job.action_count);
                rows.push(job.row);
            }
            collator
                .collate_into(&rows, &mut bytes)
                .map_err(|_| internal("feature collation failed"))?;
            if in_flight.is_empty() {
                capacity_accounted_at = Some(Instant::now());
            }
            release_releasable_models(&mut backend, &context.model_registry)?;
            let pending = backend
                .submit_for_model(model, &bytes, &action_counts)
                .map_err(|_| internal("feature eval backend failed"))?;
            in_flight.push_back((routing, pending, model));
            true
        };

        // Drain the oldest reply when the pipeline is full, when this
        // round collected nothing (idle lanes are waiting on replies),
        // or when intake closed and only the tail remains.
        let must_drain =
            in_flight.len() >= EVAL_PIPELINE_DEPTH || (!submitted && !in_flight.is_empty());
        if must_drain {
            drain_oldest(
                &mut backend,
                &mut in_flight,
                &reply_txs,
                &mut batch_sizes,
                &context,
                max_batch,
                EvalCapacityAccounting {
                    pressure: context.eval_pressure.as_deref(),
                    accounted_at: &mut capacity_accounted_at,
                },
            )?;
        }
        if last_stats.elapsed() >= STATS_INTERVAL && batch_sizes.len() > stats_batches {
            stats_batches = batch_sizes.len();
            let stats_rows: u64 = batch_sizes.iter().map(|&size| size as u64).sum();
            last_stats = Instant::now();
            eprintln!("event=eval_stats role=current batches={stats_batches} rows={stats_rows}");
        }
    }

    release_releasable_models(&mut backend, &context.model_registry)?;
    Ok(batch_sizes)
}

type BatcherRouting = Vec<(usize, usize, WorkToken, u32)>;

struct EvalCapacityAccounting<'a> {
    pressure: Option<&'a EvalPressure>,
    accounted_at: &'a mut Option<Instant>,
}

fn drain_oldest<B, G>(
    backend: &mut B,
    in_flight: &mut VecDeque<(
        BatcherRouting,
        gz_eval_service::PendingBatch,
        ModelGeneration,
    )>,
    reply_txs: &[SyncSender<LaneReply<G>>],
    batch_sizes: &mut Vec<usize>,
    context: &FeaturizedBatcherContext,
    max_batch: usize,
    capacity: EvalCapacityAccounting<'_>,
) -> EngineResult<()>
where
    B: FeatureEvalBackend,
    G: Send,
{
    let Some((routing, pending, model)) = in_flight.pop_front() else {
        return Ok(());
    };
    let capacity_work = backend.capacity_work(routing.len(), max_batch);
    let outputs = backend
        .receive(pending)
        .map_err(|_| internal("feature eval backend failed"))?;
    let completed_at = Instant::now();
    if outputs.model_version != model.version {
        return Err(internal("evaluator served the wrong model version"));
    }
    let counts = routing
        .iter()
        .map(|&(_, _, _, action_count)| action_count)
        .collect::<Vec<_>>();
    validate_backend_outputs(&outputs, &counts)?;
    context.model_registry.publish(outputs.active_generation)?;
    let completed = routing.len();
    batch_sizes.push(completed);

    for ((lane, slot, token, _), row) in routing.into_iter().zip(outputs.rows) {
        let _ = reply_txs[lane].send(LaneReply::Eval(EvalReply {
            slot,
            token,
            output: EvalOutput {
                model_version: outputs.model_version,
                policy_logits: row.policy_logits,
                value: row.value,
            },
        }));
    }
    if let Some(eval_pressure) = capacity.pressure {
        let capacity_started = capacity
            .accounted_at
            .take()
            .ok_or_else(|| internal("missing evaluator capacity clock"))?;
        let capacity_busy = completed_at.saturating_duration_since(capacity_started);
        eval_pressure.complete_current_batch(completed, capacity_work, capacity_busy);
        if !in_flight.is_empty() {
            *capacity.accounted_at = Some(completed_at);
        }
    }
    release_releasable_models(backend, &context.model_registry)?;
    Ok(())
}

fn release_releasable_models<B>(
    backend: &mut B,
    model_registry: &ModelLeaseRegistry,
) -> EngineResult<()>
where
    B: FeatureEvalBackend,
{
    for model in model_registry.take_releasable() {
        backend
            .release_model_generation(model)
            .map_err(|_| internal("feature eval backend failed"))?;
    }
    Ok(())
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
