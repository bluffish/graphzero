use crate::lanes::{LaneReply, MeasureReply};
use gz_engine::{EngineError, EngineResult, ErrorCode, ErrorMessage, GraphEngine, MeasureOptions};
use gz_measure_service::{CoordinatorHandle, MeasureSubmission};
use gz_search::WorkToken;
use std::sync::Arc;
use std::sync::mpsc::SyncSender;

pub trait MeasureSubmissionEncoder<E: GraphEngine>: Send + Sync {
    fn encode(
        &self,
        engine: &E,
        graph: E::Graph,
        options: MeasureOptions,
    ) -> EngineResult<MeasureSubmission>;
}

impl<E, F> MeasureSubmissionEncoder<E> for F
where
    E: GraphEngine,
    F: Fn(&E, E::Graph, MeasureOptions) -> EngineResult<MeasureSubmission> + Send + Sync,
{
    fn encode(
        &self,
        engine: &E,
        graph: E::Graph,
        options: MeasureOptions,
    ) -> EngineResult<MeasureSubmission> {
        self(engine, graph, options)
    }
}

pub struct RemoteMeasurementRuntime<E: GraphEngine> {
    pub coordinator: CoordinatorHandle,
    pub encoder: Arc<dyn MeasureSubmissionEncoder<E>>,
}

impl<E: GraphEngine> RemoteMeasurementRuntime<E> {
    #[must_use]
    pub fn new(
        coordinator: CoordinatorHandle,
        encoder: Arc<dyn MeasureSubmissionEncoder<E>>,
    ) -> Self {
        Self {
            coordinator,
            encoder,
        }
    }
}

pub(crate) struct RemoteMeasureJob<G> {
    pub lane: usize,
    pub slot: usize,
    pub token: WorkToken,
    pub graph: G,
    pub options: MeasureOptions,
    pub submission: MeasureSubmission,
}

pub(crate) fn run_measure_gateway<G>(
    coordinator: CoordinatorHandle,
    mut intake: tokio::sync::mpsc::Receiver<RemoteMeasureJob<G>>,
    reply_txs: Vec<SyncSender<LaneReply<G>>>,
) -> EngineResult<()>
where
    G: Copy + Send + 'static,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| remote_error(error.to_string()))?;
    runtime.block_on(async move {
        let mut pending = tokio::task::JoinSet::new();
        let mut intake_open = true;
        while intake_open || !pending.is_empty() {
            tokio::select! {
                biased;
                result = pending.join_next(), if !pending.is_empty() => {
                    result
                        .expect("non-empty measurement task set")
                        .map_err(|error| remote_error(error.to_string()))?;
                }
                job = intake.recv(), if intake_open => {
                    let Some(job) = job else {
                        intake_open = false;
                        continue;
                    };
                    let coordinator = coordinator.clone();
                    let reply_tx = reply_txs[job.lane].clone();
                    pending.spawn(async move {
                        let result = coordinator
                            .measure(job.submission)
                            .await
                            .map_err(|error| remote_error(error.to_string()))
                            .and_then(|committed| {
                                committed.into_measure_result(job.graph, job.options)
                            });
                        let _ = reply_tx.send(LaneReply::Measure(MeasureReply {
                            slot: job.slot,
                            token: job.token,
                            result,
                        }));
                    });
                }
            }
        }
        Ok(())
    })
}

fn remote_error(message: String) -> EngineError {
    let mut bounded = String::new();
    for character in message.chars() {
        if bounded.len() + character.len_utf8() > ErrorMessage::MAX_LEN {
            break;
        }
        bounded.push(character);
    }
    EngineError::Internal {
        code: ErrorCode::new(2),
        message: ErrorMessage::new(bounded).expect("bounded remote measurement error"),
    }
}
