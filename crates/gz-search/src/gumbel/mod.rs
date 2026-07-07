mod schedule;
mod task;
mod tree;
mod types;

pub use schedule::considered_visit_sequence;
pub use task::{GumbelEpisodeTask, GumbelRootTask};
pub use types::{
    GumbelEpisode, GumbelEpisodeContext, GumbelHandleBatch, GumbelMctsConfig,
    GumbelOpponentContext, GumbelRootResult, GumbelRootStats, GumbelSearchContext, GumbelStep,
    GumbelStopReason,
};

use crate::gumbel::schedule::budget_fraction;
use crate::gumbel_search_config_hash;
use crate::support::{candidate_info, internal};
use crate::work::{
    EngineIdentity, ExpandResult, ExpandWork, ExpandedCandidate, SearchPoll, SearchWork,
    SearchWorkResult,
};
use gz_engine::{EngineResult, GraphEngine, SearchConfigHash};
use gz_eval::{EngineEvalRequest, EngineEvaluator};
use std::num::NonZeroUsize;

pub struct GumbelMcts {
    config: GumbelMctsConfig,
    search_config_hash: SearchConfigHash,
}

impl GumbelMcts {
    #[must_use]
    pub fn new(config: GumbelMctsConfig) -> Self {
        assert!(config.gumbel_scale.is_finite() && config.gumbel_scale >= 0.0);
        assert!(config.c_visit.is_finite() && config.c_visit >= 0.0);
        assert!(config.c_scale.is_finite() && config.c_scale >= 0.0);

        let search_config_hash = gumbel_search_config_hash(
            config.max_steps,
            config.simulations.get(),
            config.max_considered_actions.get(),
            config.seed,
            config.gumbel_scale,
            config.c_visit,
            config.c_scale,
            config.temperature_moves,
            config.tree_reuse,
            config.mask_stop,
            config.no_backtrack,
            config.candidate_options,
            config.measure_options,
        );

        Self {
            config,
            search_config_hash,
        }
    }

    #[must_use]
    pub const fn config(&self) -> GumbelMctsConfig {
        self.config
    }

    /// The opponent-rollout search derived from this one: a single
    /// simulation over a single considered action with no noise -- a
    /// greedy argmax-policy rollout at temperature 0, with STOP masked
    /// so the reference must play rewrites to its step budget. Step
    /// budget and engine options carry over unchanged.
    #[must_use]
    pub fn policy_rollout(&self) -> Self {
        Self::new(GumbelMctsConfig {
            simulations: NonZeroUsize::MIN,
            max_considered_actions: NonZeroUsize::MIN,
            gumbel_scale: 0.0,
            temperature_moves: 0,
            tree_reuse: false,
            mask_stop: true,
            // The reference is a plain greedy rollout (whittlezero's
            // policy_rollout has no revisit masking either).
            no_backtrack: false,
            ..self.config
        })
    }

    #[must_use]
    pub const fn search_config_hash(&self) -> SearchConfigHash {
        self.search_config_hash
    }

    #[must_use]
    pub fn root_budget(&self, step: usize) -> (f32, f32) {
        let budget_step = if self.config.max_steps == 0 {
            0.0
        } else {
            1.0 / self.config.max_steps as f32
        };

        (budget_fraction(self.config.max_steps, step), budget_step)
    }

    pub fn run_from_root<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
    ) -> EngineResult<GumbelEpisode<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let root = engine.root();
        self.run(engine, evaluator, root, GumbelEpisodeContext::default())
    }

    pub fn run<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<GumbelEpisode<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let identity = EngineIdentity::from_engine(engine);
        let mut task = GumbelEpisodeTask::new(self, identity, root, context);

        loop {
            let poll = match task.poll() {
                Ok(poll) => poll,
                Err(error) => {
                    release_task_all(engine, &mut task)?;
                    return Err(error);
                }
            };
            match poll {
                SearchPoll::Work(work) => {
                    release_task_releasable(engine, &mut task)?;
                    let token = work.token();
                    let result = match service_search_work(engine, evaluator, work) {
                        Ok(result) => result,
                        Err(error) => {
                            release_task_all(engine, &mut task)?;
                            return Err(error);
                        }
                    };
                    if let Err(error) = task.resume(token, result) {
                        release_task_all(engine, &mut task)?;
                        return Err(error);
                    }
                    release_task_releasable(engine, &mut task)?;
                }
                SearchPoll::Blocked => {
                    release_task_all(engine, &mut task)?;
                    return Err(internal("serial driver blocked"));
                }
                SearchPoll::Done(result) => return Ok(result),
            }
        }
    }

    pub fn search_root<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: GumbelSearchContext,
    ) -> EngineResult<GumbelRootResult<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let identity = EngineIdentity::from_engine(engine);
        let mut task = GumbelRootTask::new(self, identity, root, context);

        loop {
            match task.poll()? {
                SearchPoll::Work(work) => {
                    let token = work.token();
                    let result = service_search_work(engine, evaluator, work)?;
                    task.resume(token, result)?;
                }
                SearchPoll::Blocked => return Err(internal("serial driver blocked")),
                SearchPoll::Done(result) => return Ok(result),
            }
        }
    }
}

fn release_task_releasable<E>(
    engine: &mut E,
    task: &mut GumbelEpisodeTask<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    release_handles(engine, task.take_releasable())
}

fn release_task_all<E>(
    engine: &mut E,
    task: &mut GumbelEpisodeTask<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    release_handles(engine, task.take_all_handles())
}

fn release_handles<E>(
    engine: &mut E,
    handles: GumbelHandleBatch<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    if handles.is_empty() {
        return Ok(());
    }
    engine.release(&handles.graphs, &handles.candidates)
}

fn service_search_work<E, V>(
    engine: &mut E,
    evaluator: &mut V,
    work: SearchWork<E::Graph, E::Candidate>,
) -> EngineResult<SearchWorkResult<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    V: EngineEvaluator<E>,
{
    match work {
        SearchWork::Expand(work) => service_expand_work(engine, work).map(SearchWorkResult::Expand),
        SearchWork::Apply(work) => engine
            .apply(work.graph, work.candidate)
            .map(SearchWorkResult::Apply),
        SearchWork::Measure(work) => engine
            .measure(work.graph, work.options)
            .map(SearchWorkResult::Measure),
        SearchWork::Eval(work) => {
            let output = evaluator.evaluate(
                engine,
                EngineEvalRequest {
                    graph: work.graph,
                    candidates: &work.candidates,
                    request: &work.request,
                    measure_options: work.measure_options,
                },
            )?;
            Ok(SearchWorkResult::Eval(output))
        }
    }
}

pub(crate) fn service_expand_work<E>(
    engine: &mut E,
    work: ExpandWork<E::Graph>,
) -> EngineResult<ExpandResult<E::Candidate>>
where
    E: GraphEngine,
{
    let mut candidates = Vec::new();
    engine.candidates(work.graph, work.options, &mut candidates)?;
    let graph_hash = engine.hash(work.graph)?;
    let candidates = candidates
        .into_iter()
        .map(|candidate| {
            candidate_info(engine, work.graph, candidate).map(|info| ExpandedCandidate {
                candidate,
                candidate_hash: info.candidate_hash,
                kind: info.kind,
                tags: info.tags,
                static_prior: info.static_prior,
            })
        })
        .collect::<EngineResult<Vec<_>>>()?;

    Ok(ExpandResult {
        graph_hash,
        candidates,
    })
}
