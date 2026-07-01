use gz_engine::{EngineError, EngineResult, ErrorCode, ErrorMessage, GraphEngine};
use gz_eval::{EngineEvalRequest, EngineEvaluator};
use gz_search::{ExpandResult, ExpandedCandidate, SearchWork, SearchWorkResult};

pub(crate) fn service_engine_work<E>(
    engine: &mut E,
    work: &SearchWork<E::Graph, E::Candidate>,
) -> EngineResult<Option<SearchWorkResult<E::Graph, E::Candidate>>>
where
    E: GraphEngine,
{
    match work {
        SearchWork::Expand(work) => {
            service_expand_work(engine, *work).map(|result| Some(SearchWorkResult::Expand(result)))
        }
        SearchWork::Apply(work) => engine
            .apply(work.graph, work.candidate)
            .map(|result| Some(SearchWorkResult::Apply(result))),
        SearchWork::Measure(work) => engine
            .measure(work.graph, work.options)
            .map(|result| Some(SearchWorkResult::Measure(result))),
        SearchWork::Eval(_) => Ok(None),
        _ => Err(internal("unsupported search work")),
    }
}

pub(crate) fn service_work<E, V>(
    engine: &mut E,
    evaluator: &mut V,
    work: SearchWork<E::Graph, E::Candidate>,
) -> EngineResult<SearchWorkResult<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    V: EngineEvaluator<E>,
{
    if let Some(result) = service_engine_work(engine, &work)? {
        return Ok(result);
    }

    match work {
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
        _ => Err(internal("unsupported search work")),
    }
}

fn service_expand_work<E>(
    engine: &mut E,
    work: gz_search::ExpandWork<E::Graph>,
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
            engine
                .candidate_info(work.graph, candidate)?
                .validate()
                .map_err(|_| internal("invalid candidate info"))
                .map(|info| ExpandedCandidate {
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

pub(crate) fn internal(message: &'static str) -> EngineError {
    EngineError::Internal {
        code: ErrorCode::new(1),
        message: ErrorMessage::new(message).expect("internal orchestrator message is short"),
    }
}
