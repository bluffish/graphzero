use gz_engine::GraphEngine;
use gz_engine_whittle::{WhittleCandidateId, WhittleEngine, WhittleEngineConfig, WhittleGraphId};
use gz_eval::{
    EvalError, EvalOutput, EvalRequest, EvalResult, Evaluator, RandomValueEvaluator,
    RandomValueEvaluatorConfig,
};
use gz_orchestrator::{
    BatchedGumbelOrchestrator, CountedRoots, OrchestratedEpisode, SerialGumbelOrchestrator,
    WorkerId,
};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig};
use std::num::NonZeroUsize;

fn search(engine: &WhittleEngine, tree_reuse: bool) -> GumbelMcts {
    GumbelMcts::new(GumbelMctsConfig {
        max_steps: 2,
        simulations: NonZeroUsize::new(2).unwrap(),
        max_considered_actions: NonZeroUsize::new(4).unwrap(),
        seed: 5,
        gumbel_scale: 0.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse,
        mask_stop: false,
        no_backtrack: false,
        candidate_options: gz_engine::CandidateOptions::default(),
        export_position: true,
        measure_options: engine.measure_options(),
    })
}

fn evaluator() -> RandomValueEvaluator {
    RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: 13,
        ..RandomValueEvaluatorConfig::default()
    })
    .unwrap()
}

fn repeated_roots(
    count: u64,
) -> CountedRoots<impl FnMut(&mut WhittleEngine) -> gz_engine::EngineResult<WhittleGraphId>> {
    CountedRoots::new(count, |engine: &mut WhittleEngine| Ok(engine.root()))
}

fn run_batched(
    workers: usize,
    roots: u64,
    tree_reuse: bool,
) -> Vec<OrchestratedEpisode<WhittleGraphId, WhittleCandidateId>> {
    let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
    let search = search(&engine, tree_reuse);
    let mut orchestrator = BatchedGumbelOrchestrator::new(
        engine,
        evaluator(),
        search,
        NonZeroUsize::new(workers).unwrap(),
    );
    let mut roots = repeated_roots(roots);

    orchestrator
        .run(&mut roots, GumbelEpisodeContext::default())
        .unwrap()
        .episodes
}

fn run_serial(
    roots: u64,
    tree_reuse: bool,
) -> Vec<OrchestratedEpisode<WhittleGraphId, WhittleCandidateId>> {
    let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
    let search = search(&engine, tree_reuse);
    let mut orchestrator =
        SerialGumbelOrchestrator::new(WorkerId::new(0), engine, evaluator(), search);

    (0..roots)
        .map(|_| {
            orchestrator
                .run_from_root(GumbelEpisodeContext::default())
                .unwrap()
        })
        .collect()
}

#[test]
fn one_worker_matches_serial() {
    for tree_reuse in [false, true] {
        let batched = run_batched(1, 3, tree_reuse);
        let serial = run_serial(3, tree_reuse);

        assert_eq!(batched.len(), serial.len());
        for (batched, serial) in batched.iter().zip(&serial) {
            assert_eq!(batched.episode, serial.episode);
        }
    }
}

#[test]
fn multi_worker_matches_serial_and_assigns_admission_ids() {
    for tree_reuse in [false, true] {
        let batched = run_batched(4, 7, tree_reuse);
        let serial = run_serial(7, tree_reuse);

        assert_eq!(batched.len(), serial.len());
        for (index, (batched, serial)) in batched.iter().zip(&serial).enumerate() {
            assert_eq!(batched.episode_id.value(), index as u64);
            assert_eq!(batched.episode, serial.episode);
        }
    }
}

#[test]
fn first_batch_fills_available_workers() {
    let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
    let search = search(&engine, false);
    let mut orchestrator =
        BatchedGumbelOrchestrator::new(engine, evaluator(), search, NonZeroUsize::new(4).unwrap());
    let mut roots = repeated_roots(6);

    let run = orchestrator
        .run(&mut roots, GumbelEpisodeContext::default())
        .unwrap();

    assert_eq!(run.batch_sizes[0], 4);
    assert_eq!(run.episodes.len(), 6);
}

#[test]
fn empty_root_source_returns_empty_run() {
    let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
    let search = search(&engine, false);
    let mut orchestrator =
        BatchedGumbelOrchestrator::new(engine, evaluator(), search, NonZeroUsize::new(4).unwrap());
    let mut roots = repeated_roots(0);

    let run = orchestrator
        .run(&mut roots, GumbelEpisodeContext::default())
        .unwrap();

    assert!(run.episodes.is_empty());
    assert!(run.batch_sizes.is_empty());
}

#[test]
fn evaluator_failure_aborts_run() {
    let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
    let search = search(&engine, false);
    let mut orchestrator = BatchedGumbelOrchestrator::new(
        engine,
        FailingEvaluator,
        search,
        NonZeroUsize::new(2).unwrap(),
    );
    let mut roots = repeated_roots(2);

    let error = orchestrator
        .run(&mut roots, GumbelEpisodeContext::default())
        .unwrap_err();

    assert!(error.to_string().contains("eval failed"));
}

#[test]
fn batched_run_is_deterministic() {
    let first = {
        let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
        let search = search(&engine, false);
        let mut orchestrator = BatchedGumbelOrchestrator::new(
            engine,
            evaluator(),
            search,
            NonZeroUsize::new(4).unwrap(),
        );
        let mut roots = repeated_roots(7);
        orchestrator
            .run(&mut roots, GumbelEpisodeContext::default())
            .unwrap()
    };
    let second = {
        let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
        let search = search(&engine, false);
        let mut orchestrator = BatchedGumbelOrchestrator::new(
            engine,
            evaluator(),
            search,
            NonZeroUsize::new(4).unwrap(),
        );
        let mut roots = repeated_roots(7);
        orchestrator
            .run(&mut roots, GumbelEpisodeContext::default())
            .unwrap()
    };

    assert_eq!(first.batch_sizes, second.batch_sizes);
    assert_eq!(first.episodes, second.episodes);
}

struct FailingEvaluator;

impl Evaluator for FailingEvaluator {
    fn evaluate_batch(
        &mut self,
        _requests: &[EvalRequest],
        _out: &mut Vec<EvalOutput>,
    ) -> EvalResult<()> {
        Err(EvalError::NonFiniteValue { value: f32::NAN })
    }
}
