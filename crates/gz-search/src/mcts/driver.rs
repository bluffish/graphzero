use super::strategy::MctsStrategy;
use super::task::{MctsEpisodeTask, MctsRootTask};
use super::types::{MctsEpisode, MctsHandleBatch, MctsRootResult};
use crate::support::{candidate_info, internal};
use crate::work::{
    ExpandResult, ExpandWork, ExpandedCandidate, SearchPoll, SearchWork, SearchWorkResult,
};
use gz_engine::{EngineResult, GraphEngine};
use gz_eval::{EngineEvalRequest, EngineEvaluator};
use std::hash::Hash;

pub(crate) fn run_root<E, V, S>(
    engine: &mut E,
    evaluator: &mut V,
    mut task: MctsRootTask<E::Graph, E::Candidate, S>,
) -> EngineResult<MctsRootResult<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    V: EngineEvaluator<E>,
    S: MctsStrategy<E::Graph, E::Candidate>,
{
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

pub(crate) fn run_episode<E, V, S>(
    engine: &mut E,
    evaluator: &mut V,
    mut task: MctsEpisodeTask<E::Graph, E::Candidate, S>,
) -> EngineResult<MctsEpisode<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    E::Graph: Eq + Hash,
    E::Candidate: Eq + Hash,
    V: EngineEvaluator<E>,
    S: MctsStrategy<E::Graph, E::Candidate>,
{
    loop {
        let poll = match task.poll() {
            Ok(poll) => poll,
            Err(error) => {
                release_handles(engine, task.take_all_handles())?;
                return Err(error);
            }
        };
        match poll {
            SearchPoll::Work(work) => {
                release_handles(engine, task.take_releasable())?;
                let token = work.token();
                let result = match service_search_work(engine, evaluator, work) {
                    Ok(result) => result,
                    Err(error) => {
                        release_handles(engine, task.take_all_handles())?;
                        return Err(error);
                    }
                };
                if let Err(error) = task.resume(token, result) {
                    release_handles(engine, task.take_all_handles())?;
                    return Err(error);
                }
                release_handles(engine, task.take_releasable())?;
            }
            SearchPoll::Blocked => {
                release_handles(engine, task.take_all_handles())?;
                return Err(internal("serial driver blocked"));
            }
            SearchPoll::Done(result) => return Ok(result),
        }
    }
}

fn release_handles<E>(
    engine: &mut E,
    handles: MctsHandleBatch<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    if handles.is_empty() {
        return Ok(());
    }
    engine.release(&handles.graphs, &handles.candidates)
}

pub(crate) fn service_search_work<E, V>(
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
        SearchWork::MeasurePair(work) => {
            let first = engine.measure(work.graphs[0], work.options)?;
            let second = engine.measure(work.graphs[1], work.options)?;
            Ok(SearchWorkResult::MeasurePair([first, second]))
        }
        SearchWork::Eval(work) => evaluator
            .evaluate(
                engine,
                EngineEvalRequest {
                    graph: work.graph,
                    candidates: &work.candidates,
                    request: &work.request,
                    measure_options: work.measure_options,
                },
            )
            .map(SearchWorkResult::Eval),
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
