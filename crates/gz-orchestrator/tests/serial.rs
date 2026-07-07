use gz_engine_whittle::{WhittleEngine, WhittleEngineConfig};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_orchestrator::{SerialGumbelOrchestrator, WorkerId};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig};
use std::num::NonZeroUsize;

fn search(engine: &WhittleEngine, tree_reuse: bool) -> GumbelMcts {
    GumbelMcts::new(GumbelMctsConfig {
        max_steps: 2,
        simulations: NonZeroUsize::new(2).unwrap(),
        max_considered_actions: NonZeroUsize::new(4).unwrap(),
        seed: 5,
        gumbel_scale: 0.0,
        gumbel_noise_overlap: -1.0,
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

#[test]
fn serial_orchestrator_matches_direct_gumbel_run() {
    for tree_reuse in [false, true] {
        let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
        let mut orchestrator =
            SerialGumbelOrchestrator::new(WorkerId::new(3), engine, evaluator(), {
                let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
                search(&engine, tree_reuse)
            });
        let serial = orchestrator
            .run_from_root(GumbelEpisodeContext::default())
            .unwrap();

        let mut direct_engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
        let mut direct_eval = evaluator();
        let direct_search = search(&direct_engine, tree_reuse);
        let direct = direct_search
            .run_from_root(&mut direct_engine, &mut direct_eval)
            .unwrap();

        assert_eq!(serial.worker_id, WorkerId::new(3));
        assert_eq!(serial.episode_id.value(), 0);
        assert_eq!(serial.episode, direct);
    }
}

#[test]
fn serial_orchestrator_episode_ids_increment() {
    let engine = WhittleEngine::new(WhittleEngineConfig::default()).unwrap();
    let search = search(&engine, false);
    let mut orchestrator =
        SerialGumbelOrchestrator::new(WorkerId::new(1), engine, evaluator(), search);

    let first = orchestrator
        .run_from_root(GumbelEpisodeContext::default())
        .unwrap();
    let second = orchestrator
        .run_from_root(GumbelEpisodeContext::default())
        .unwrap();

    assert_eq!(first.episode_id.value(), 0);
    assert_eq!(second.episode_id.value(), 1);
}
