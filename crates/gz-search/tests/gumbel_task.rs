#[allow(dead_code)]
mod common;

use common::{TestEngine, measure_options};
use gz_engine::{CandidateOptions, GraphEngine, ModelVersion};
use gz_eval::EvalOutput;
use gz_search::{
    EngineIdentity, EvalWork, ExpandResult, ExpandedCandidate, GumbelEpisodeContext,
    GumbelEpisodeTask, GumbelMcts, GumbelMctsConfig, GumbelOpponentContext, GumbelRootTask,
    GumbelSearchContext, SearchPoll, SearchWork, SearchWorkResult, WorkToken,
};
use std::num::NonZeroUsize;

fn config(max_steps: usize) -> GumbelMctsConfig {
    GumbelMctsConfig {
        max_steps,
        simulations: NonZeroUsize::new(1).unwrap(),
        max_considered_actions: NonZeroUsize::new(8).unwrap(),
        seed: 0,
        gumbel_scale: 0.0,
        gumbel_noise_overlap: -1.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        mask_stop: false,
        no_backtrack: false,
        candidate_options: CandidateOptions::default(),
        measure_options: measure_options(),
    }
}

fn expand_result(
    engine: &mut TestEngine,
    graph: u8,
    options: CandidateOptions,
) -> ExpandResult<u8> {
    let mut candidates = Vec::new();
    engine.candidates(graph, options, &mut candidates).unwrap();
    let graph_hash = engine.hash(graph).unwrap();
    let candidates = candidates
        .into_iter()
        .map(|candidate| {
            let info = engine.candidate_info(graph, candidate).unwrap();
            ExpandedCandidate {
                candidate,
                candidate_hash: info.candidate_hash,
                kind: info.kind,
                tags: info.tags,
                static_prior: info.static_prior,
            }
        })
        .collect();

    ExpandResult {
        graph_hash,
        candidates,
    }
}

fn output(len: usize, value: f32) -> EvalOutput {
    EvalOutput {
        model_version: ModelVersion::from_bytes([7; 16]),
        policy_logits: vec![0.0; len],
        value,
    }
}

fn output_with_logits(policy_logits: Vec<f32>, value: f32) -> EvalOutput {
    EvalOutput {
        model_version: ModelVersion::from_bytes([7; 16]),
        policy_logits,
        value,
    }
}

fn first_expand(task: &mut GumbelRootTask<u8, u8>) -> (WorkToken, u8, CandidateOptions) {
    match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Expand(work)) => (work.token, work.graph, work.options),
        other => panic!("expected expand, got {other:?}"),
    }
}

fn first_eval(
    task: &mut GumbelRootTask<u8, u8>,
    engine: &mut TestEngine,
    token: WorkToken,
    graph: u8,
    options: CandidateOptions,
) -> EvalWork<u8, u8> {
    task.resume(
        token,
        SearchWorkResult::Expand(expand_result(engine, graph, options)),
    )
    .unwrap();

    match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Eval(work)) => work,
        other => panic!("expected eval, got {other:?}"),
    }
}

#[test]
fn root_task_first_emits_expand_then_eval() {
    let mut engine = TestEngine::new().candidates(0, [1, 2]);
    let search = GumbelMcts::new(config(1));
    let mut task = GumbelRootTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelSearchContext::default(),
    );

    let (token, graph, options) = first_expand(&mut task);
    assert_eq!(graph, 0);

    let eval = first_eval(&mut task, &mut engine, token, graph, options);
    assert_eq!(eval.graph, 0);
    assert_eq!(eval.candidates, vec![1, 2]);
    assert_eq!(eval.request.action_count(), 3);
}

#[test]
fn root_task_rejects_unknown_token() {
    let mut engine = TestEngine::new();
    let search = GumbelMcts::new(config(1));
    let mut task = GumbelRootTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelSearchContext::default(),
    );
    let (token, graph, options) = first_expand(&mut task);

    let error = task
        .resume(
            WorkToken::new(token.value() + 1),
            SearchWorkResult::Expand(expand_result(&mut engine, graph, options)),
        )
        .unwrap_err();

    assert!(error.to_string().contains("unknown work token"));
}

#[test]
fn root_task_rejects_mismatched_result_variant() {
    let engine = TestEngine::new();
    let search = GumbelMcts::new(config(1));
    let mut task = GumbelRootTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelSearchContext::default(),
    );
    let (token, _, _) = first_expand(&mut task);

    let error = task
        .resume(token, SearchWorkResult::Eval(output(1, 0.0)))
        .unwrap_err();

    assert!(error.to_string().contains("mismatched work result"));
}

#[test]
fn root_task_rejects_wrong_eval_length() {
    let mut engine = TestEngine::new().candidates(0, [1]);
    let search = GumbelMcts::new(config(1));
    let mut task = GumbelRootTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelSearchContext::default(),
    );
    let (token, graph, options) = first_expand(&mut task);
    let eval = first_eval(&mut task, &mut engine, token, graph, options);

    let error = task
        .resume(eval.token, SearchWorkResult::Eval(output(1, 0.0)))
        .unwrap_err();

    assert!(error.to_string().contains("eval failed"));
}

#[test]
fn root_task_blocks_while_pending_and_rejects_double_resume() {
    let mut engine = TestEngine::new();
    let search = GumbelMcts::new(config(1));
    let mut task = GumbelRootTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelSearchContext::default(),
    );
    let (token, graph, options) = first_expand(&mut task);

    assert!(matches!(task.poll().unwrap(), SearchPoll::Blocked));
    task.resume(
        token,
        SearchWorkResult::Expand(expand_result(&mut engine, graph, options)),
    )
    .unwrap();
    let error = task
        .resume(
            token,
            SearchWorkResult::Expand(expand_result(&mut engine, graph, options)),
        )
        .unwrap_err();

    assert!(error.to_string().contains("resume without pending work"));
}

#[test]
fn dropping_episode_task_with_outstanding_token_is_safe() {
    let engine = TestEngine::new();
    let search = GumbelMcts::new(config(1));
    let mut task: GumbelEpisodeTask<u8, u8> = GumbelEpisodeTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelEpisodeContext::default(),
    );

    assert!(matches!(task.poll().unwrap(), SearchPoll::Work(_)));
}

#[test]
fn rejected_apply_masks_action_and_next_poll_emits_work() {
    let mut engine = TestEngine::new()
        .candidates(0, [1, 2])
        .rejected(0, 1)
        .apply(0, 2, 3);
    let search = GumbelMcts::new(config(1));
    let mut task = GumbelRootTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelSearchContext::default(),
    );
    let (token, graph, options) = first_expand(&mut task);
    let eval = first_eval(&mut task, &mut engine, token, graph, options);
    task.resume(
        eval.token,
        SearchWorkResult::Eval(output_with_logits(vec![10.0, 9.0, 0.0], 0.0)),
    )
    .unwrap();

    let first_apply = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Apply(work)) => work,
        other => panic!("expected first apply, got {other:?}"),
    };
    assert_eq!(first_apply.candidate, 1);

    let rejected =
        GraphEngine::apply(&mut engine, first_apply.graph, first_apply.candidate).unwrap();
    task.resume(first_apply.token, SearchWorkResult::Apply(rejected))
        .unwrap();

    let second_apply = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Apply(work)) => work,
        other => panic!("expected second apply, got {other:?}"),
    };
    assert_eq!(second_apply.graph, 0);
    assert_eq!(second_apply.candidate, 2);
}

#[test]
fn root_task_poll_after_done_is_rejected() {
    let mut engine = TestEngine::new().candidates(0, []);
    let search = GumbelMcts::new(config(1));
    let mut task = GumbelRootTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelSearchContext::default(),
    );
    let (token, graph, options) = first_expand(&mut task);
    let eval = first_eval(&mut task, &mut engine, token, graph, options);
    task.resume(eval.token, SearchWorkResult::Eval(output(1, 0.0)))
        .unwrap();
    assert!(matches!(task.poll().unwrap(), SearchPoll::Done(_)));

    let error = task.poll().unwrap_err();
    assert!(error.to_string().contains("poll after done"));
}

#[test]
fn opponent_stop_alignment_emits_second_eval() {
    let mut engine = TestEngine::new().candidates(0, []);
    let search = GumbelMcts::new(config(1));
    let mut task = GumbelRootTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelSearchContext {
            root_step: 1,
            opponent: Some(GumbelOpponentContext {
                trajectory_id: 9,
                row_count: 4,
                final_reward: -2.0,
            }),
            ..GumbelSearchContext::default()
        },
    );
    let (token, graph, options) = first_expand(&mut task);
    let eval = first_eval(&mut task, &mut engine, token, graph, options);
    task.resume(eval.token, SearchWorkResult::Eval(output(1, 0.0)))
        .unwrap();

    let stop_eval = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Eval(work)) => work,
        other => panic!("expected stop eval, got {other:?}"),
    };

    assert_eq!(stop_eval.request.position.leaf_depth, 2);
    assert_eq!(stop_eval.request.position.opponent_row(), Some(3));
}

#[test]
fn episode_task_emits_final_measure() {
    let mut engine = TestEngine::new().candidates(0, []);
    let search = GumbelMcts::new(config(1));
    let mut task = GumbelEpisodeTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelEpisodeContext::default(),
    );

    let expand = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Expand(work)) => work,
        other => panic!("expected expand, got {other:?}"),
    };
    task.resume(
        expand.token,
        SearchWorkResult::Expand(expand_result(&mut engine, expand.graph, expand.options)),
    )
    .unwrap();
    let eval = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Eval(work)) => work,
        other => panic!("expected eval, got {other:?}"),
    };
    task.resume(eval.token, SearchWorkResult::Eval(output(1, 0.0)))
        .unwrap();

    let measure = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Measure(work)) => work,
        other => panic!("expected measure, got {other:?}"),
    };

    assert_eq!(measure.graph, 0);
}

#[test]
fn episode_task_exposes_releasable_handles_before_measure() {
    let mut engine = TestEngine::new()
        .candidates(0, [1, 2])
        .candidates(20, [])
        .apply(0, 2, 20)
        .reward(20, 20.0);
    let search = GumbelMcts::new(config(1));
    let mut task = GumbelEpisodeTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelEpisodeContext::default(),
    );

    let expand = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Expand(work)) => work,
        other => panic!("expected root expand, got {other:?}"),
    };
    task.resume(
        expand.token,
        SearchWorkResult::Expand(expand_result(&mut engine, expand.graph, expand.options)),
    )
    .unwrap();

    let eval = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Eval(work)) => work,
        other => panic!("expected root eval, got {other:?}"),
    };
    task.resume(
        eval.token,
        SearchWorkResult::Eval(output_with_logits(vec![0.0, 10.0, -10.0], 0.0)),
    )
    .unwrap();

    let apply = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Apply(work)) => work,
        other => panic!("expected apply, got {other:?}"),
    };
    assert_eq!(apply.candidate, 2);
    let applied = GraphEngine::apply(&mut engine, apply.graph, apply.candidate).unwrap();
    task.resume(apply.token, SearchWorkResult::Apply(applied))
        .unwrap();

    let child_expand = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Expand(work)) => work,
        other => panic!("expected child expand, got {other:?}"),
    };
    task.resume(
        child_expand.token,
        SearchWorkResult::Expand(expand_result(
            &mut engine,
            child_expand.graph,
            child_expand.options,
        )),
    )
    .unwrap();

    let child_eval = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Eval(work)) => work,
        other => panic!("expected child eval, got {other:?}"),
    };
    task.resume(child_eval.token, SearchWorkResult::Eval(output(1, 0.0)))
        .unwrap();

    let measure = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Measure(work)) => work,
        other => panic!("expected measure, got {other:?}"),
    };
    let releasable = task.take_releasable();
    assert_eq!(releasable.graphs, Vec::<u8>::new());
    assert_eq!(releasable.candidates, vec![1, 2]);

    let measured = GraphEngine::measure(&mut engine, measure.graph, measure.options).unwrap();
    task.resume(measure.token, SearchWorkResult::Measure(measured))
        .unwrap();

    let episode = match task.poll().unwrap() {
        SearchPoll::Done(episode) => episode,
        other => panic!("expected done, got {other:?}"),
    };
    assert_eq!(episode.created_graphs, vec![20]);
    assert_eq!(episode.created_candidates, Vec::<u8>::new());
}
