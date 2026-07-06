use crate::pool::{Admission, WorkerPool};
use crate::root::RootSource;
use crate::serial::OrchestratedEpisode;
use gz_engine::{EngineResult, GraphEngine};
use gz_eval::{EvalOutput, EvalRequest, Evaluator, eval_error_to_engine_error, validate_outputs};
use gz_search::{EngineIdentity, GumbelEpisodeContext, GumbelMcts};
use std::num::NonZeroUsize;

pub struct BatchedGumbelOrchestrator<E, V> {
    engine: E,
    evaluator: V,
    search: GumbelMcts,
    workers: NonZeroUsize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchedRun<G, C> {
    pub episodes: Vec<OrchestratedEpisode<G, C>>,
    pub batch_sizes: Vec<usize>,
}

impl<E, V> BatchedGumbelOrchestrator<E, V>
where
    E: GraphEngine,
    V: Evaluator,
{
    pub const fn new(engine: E, evaluator: V, search: GumbelMcts, workers: NonZeroUsize) -> Self {
        Self {
            engine,
            evaluator,
            search,
            workers,
        }
    }

    pub fn run<R>(
        &mut self,
        roots: &mut R,
        context: GumbelEpisodeContext,
    ) -> EngineResult<BatchedRun<E::Graph, E::Candidate>>
    where
        R: RootSource<E>,
    {
        let identity = EngineIdentity::from_engine(&self.engine);
        let mut pool = WorkerPool::new(self.workers, 0);
        let mut episodes = Vec::new();
        let mut batch_sizes = Vec::new();
        let mut next_episode_id = 0;
        let mut roots_exhausted = false;

        loop {
            if !roots_exhausted {
                let mut admission = Admission {
                    search: &self.search,
                    identity,
                    context,
                    next_episode_id: &mut next_episode_id,
                };
                roots_exhausted = pool.admit(
                    &mut self.engine,
                    roots,
                    &mut admission,
                    |_, _, _, context| Ok(context),
                )?;
            }

            episodes.extend(pool.drive(&mut self.engine, "batched driver blocked", None)?);

            if roots_exhausted && !pool.active() {
                return Ok(BatchedRun {
                    episodes,
                    batch_sizes,
                });
            }

            let parked = pool.parked();
            if parked.is_empty() {
                continue;
            }

            let requests = parked
                .iter()
                .map(|parked| parked.request.clone())
                .collect::<Vec<_>>();
            let outputs = self.evaluate_batch(&requests)?;
            batch_sizes.push(requests.len());

            for (parked, output) in parked.into_iter().zip(outputs) {
                pool.resume(&mut self.engine, parked.slot, parked.token, output)?;
            }
        }
    }

    fn evaluate_batch(&mut self, requests: &[EvalRequest]) -> EngineResult<Vec<EvalOutput>> {
        let mut outputs = Vec::with_capacity(requests.len());
        self.evaluator
            .evaluate_batch(requests, &mut outputs)
            .map_err(eval_error_to_engine_error)?;
        validate_outputs(requests, &outputs).map_err(eval_error_to_engine_error)?;
        Ok(outputs)
    }
}
