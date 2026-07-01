use crate::service::{internal, service_work};
use crate::{EpisodeId, WorkerId};
use gz_engine::{EngineResult, GraphEngine};
use gz_eval::EngineEvaluator;
use gz_search::{
    EngineIdentity, GumbelEpisode, GumbelEpisodeContext, GumbelEpisodeTask, GumbelMcts, SearchPoll,
};

pub struct SerialGumbelOrchestrator<E, V> {
    worker_id: WorkerId,
    next_episode_id: u64,
    engine: E,
    evaluator: V,
    search: GumbelMcts,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrchestratedEpisode<G, C> {
    pub worker_id: WorkerId,
    pub episode_id: EpisodeId,
    pub episode: GumbelEpisode<G, C>,
}

pub type SerialEpisode<G, C> = OrchestratedEpisode<G, C>;

impl<E, V> SerialGumbelOrchestrator<E, V>
where
    E: GraphEngine,
    V: EngineEvaluator<E>,
{
    pub fn new(worker_id: WorkerId, engine: E, evaluator: V, search: GumbelMcts) -> Self {
        Self {
            worker_id,
            next_episode_id: 0,
            engine,
            evaluator,
            search,
        }
    }

    pub fn run_from_root(
        &mut self,
        context: GumbelEpisodeContext,
    ) -> EngineResult<SerialEpisode<E::Graph, E::Candidate>> {
        let root = self.engine.root();
        self.run(root, context)
    }

    pub fn run(
        &mut self,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<SerialEpisode<E::Graph, E::Candidate>> {
        let identity = EngineIdentity::from_engine(&self.engine);
        let mut task = GumbelEpisodeTask::new(&self.search, identity, root, context);

        loop {
            match task.poll()? {
                SearchPoll::Work(work) => {
                    let token = work.token();
                    let result = service_work(&mut self.engine, &mut self.evaluator, work)?;
                    task.resume(token, result)?;
                }
                SearchPoll::Blocked => return Err(internal("serial driver blocked")),
                SearchPoll::Done(episode) => {
                    let episode_id = EpisodeId::new(self.next_episode_id);
                    self.next_episode_id += 1;
                    return Ok(OrchestratedEpisode {
                        worker_id: self.worker_id,
                        episode_id,
                        episode,
                    });
                }
            }
        }
    }
}
