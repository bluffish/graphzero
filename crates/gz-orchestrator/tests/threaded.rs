use gz_engine::{EngineResult, GraphEngine};
use gz_engine_whittle::{WhittleCandidateId, WhittleEngine, WhittleEngineConfig, WhittleGraphId};
use gz_eval::{
    EvalError, EvalOutput, EvalRequest, EvalResult, Evaluator, RandomValueEvaluator,
    RandomValueEvaluatorConfig,
};
use gz_orchestrator::{
    CountedRoots, LaneEpisodes, OrchestratedEpisode, SerialGumbelOrchestrator,
    ThreadedGumbelOrchestrator, ThreadedOrchestratorConfig, WorkerId,
};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig};
use std::num::NonZeroUsize;
use std::time::Duration;

type Roots = CountedRoots<fn(&mut WhittleEngine) -> EngineResult<WhittleGraphId>>;

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

fn config(workers_per_lane: usize, max_batch: usize) -> ThreadedOrchestratorConfig {
    ThreadedOrchestratorConfig {
        workers_per_lane: NonZeroUsize::new(workers_per_lane).unwrap(),
        max_batch: NonZeroUsize::new(max_batch).unwrap(),
        flush_after: Duration::from_millis(50),
    }
}

fn evaluator() -> RandomValueEvaluator {
    RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: 13,
        ..RandomValueEvaluatorConfig::default()
    })
    .unwrap()
}

fn root_factory(engine: &mut WhittleEngine) -> EngineResult<WhittleGraphId> {
    Ok(engine.root())
}

fn repeated_roots(count: u64) -> Roots {
    CountedRoots::new(count, root_factory)
}

fn engines(count: usize) -> Vec<WhittleEngine> {
    (0..count)
        .map(|_| WhittleEngine::new(WhittleEngineConfig::default()).unwrap())
        .collect()
}

fn run_threaded(
    lanes: usize,
    workers_per_lane: usize,
    roots_per_lane: &[u64],
    tree_reuse: bool,
) -> gz_orchestrator::ThreadedRun<WhittleGraphId, WhittleCandidateId> {
    let engines = engines(lanes);
    let search = search(&engines[0], tree_reuse);
    let orchestrator =
        ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(workers_per_lane, 8));
    let roots = roots_per_lane
        .iter()
        .copied()
        .map(repeated_roots)
        .collect::<Vec<_>>();

    orchestrator
        .run(roots, GumbelEpisodeContext::default())
        .unwrap()
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
fn one_lane_one_worker_matches_serial() {
    for tree_reuse in [false, true] {
        let run = run_threaded(1, 1, &[3], tree_reuse);
        let serial = run_serial(3, tree_reuse);

        assert_lane_matches_serial(&run.lanes[0], &serial);
    }
}

#[test]
fn two_lanes_match_serial_per_lane_and_assign_lane_ids() {
    for tree_reuse in [false, true] {
        let run = run_threaded(2, 2, &[3, 5], tree_reuse);

        assert_eq!(run.lanes.len(), 2);
        assert_lane_matches_serial(&run.lanes[0], &run_serial(3, tree_reuse));
        assert_lane_matches_serial(&run.lanes[1], &run_serial(5, tree_reuse));

        for lane in &run.lanes {
            for (index, episode) in lane.episodes.iter().enumerate() {
                assert_eq!(
                    episode.episode_id.value(),
                    ((lane.lane as u64) << 32) + index as u64
                );
            }
        }
    }
}

#[test]
fn threaded_run_conserves_all_episodes_when_counts_do_not_align() {
    let run = run_threaded(2, 3, &[5, 4], false);
    let total = run
        .lanes
        .iter()
        .map(|lane| lane.episodes.len())
        .sum::<usize>();

    assert_eq!(total, 9);
}

#[test]
fn slow_evaluator_batches_active_workers() {
    let engines = engines(1);
    let search = search(&engines[0], false);
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        SlowEvaluator {
            inner: evaluator(),
            delay: Duration::from_millis(20),
        },
        search,
        ThreadedOrchestratorConfig {
            workers_per_lane: NonZeroUsize::new(8).unwrap(),
            max_batch: NonZeroUsize::new(8).unwrap(),
            flush_after: Duration::from_millis(250),
        },
    );
    let run = orchestrator
        .run(vec![repeated_roots(16)], GumbelEpisodeContext::default())
        .unwrap();
    let evals = run.batch_sizes.iter().sum::<usize>();
    let average = evals as f64 / run.batch_sizes.len() as f64;

    assert_eq!(run.lanes[0].episodes.len(), 16);
    assert_eq!(run.batch_sizes[0], 8);
    assert!(average >= 4.0);
}

#[test]
fn eval_failure_returns_without_hanging() {
    let engines = engines(1);
    let search = search(&engines[0], false);
    let orchestrator =
        ThreadedGumbelOrchestrator::new(engines, FailingEvaluator, search, config(2, 4));

    let error = orchestrator
        .run(vec![repeated_roots(2)], GumbelEpisodeContext::default())
        .unwrap_err();

    assert!(error.to_string().contains("eval failed"));
}

#[test]
fn lane_count_mismatch_is_rejected() {
    let engines = engines(2);
    let search = search(&engines[0], false);
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(2, 4));

    let error = orchestrator
        .run(vec![repeated_roots(1)], GumbelEpisodeContext::default())
        .unwrap_err();

    assert!(error.to_string().contains("lane count mismatch"));
}

fn assert_lane_matches_serial(
    lane: &LaneEpisodes<WhittleGraphId, WhittleCandidateId>,
    serial: &[OrchestratedEpisode<WhittleGraphId, WhittleCandidateId>],
) {
    assert_eq!(lane.episodes.len(), serial.len());
    for (actual, expected) in lane.episodes.iter().zip(serial) {
        assert_eq!(actual.episode, expected.episode);
    }
}

struct SlowEvaluator<V> {
    inner: V,
    delay: Duration,
}

impl<V> Evaluator for SlowEvaluator<V>
where
    V: Evaluator,
{
    fn evaluate_batch(
        &mut self,
        requests: &[EvalRequest],
        out: &mut Vec<EvalOutput>,
    ) -> EvalResult<()> {
        std::thread::sleep(self.delay);
        self.inner.evaluate_batch(requests, out)
    }
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
