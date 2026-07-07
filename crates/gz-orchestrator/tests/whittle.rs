use gz_engine_whittle::{WhittleEngine, WhittleEngineConfig, WhittleRoot};
use gz_eval_whittle::WhittleMeasureEvaluator;
use gz_orchestrator::{SerialGumbelOrchestrator, WorkerId};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig};
use std::num::NonZeroUsize;

const NO_NODE: u32 = u32::MAX;

#[test]
fn serial_orchestrator_drives_whittle_measure_evaluator() {
    let engine = and_engine();
    let gumbel = search(&engine);
    let mut orchestrator = SerialGumbelOrchestrator::new(
        WorkerId::new(8),
        engine,
        WhittleMeasureEvaluator::new(),
        gumbel,
    );
    let serial = orchestrator
        .run_from_root(GumbelEpisodeContext::default())
        .unwrap();

    let mut direct_engine = and_engine();
    let direct_search = search(&direct_engine);
    let mut direct_eval = WhittleMeasureEvaluator::new();
    let direct = direct_search
        .run_from_root(&mut direct_engine, &mut direct_eval)
        .unwrap();

    assert!(serial.episode.final_measure.measured);
    assert!(serial.episode.final_measure.valid);
    assert_eq!(serial.episode, direct);
}

fn search(engine: &WhittleEngine) -> GumbelMcts {
    GumbelMcts::new(GumbelMctsConfig {
        max_steps: 2,
        simulations: NonZeroUsize::new(4).unwrap(),
        max_considered_actions: NonZeroUsize::new(4).unwrap(),
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
        candidate_options: gz_engine::CandidateOptions::default(),
        measure_options: engine.measure_options(),
    })
}

fn and_engine() -> WhittleEngine {
    WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Artifact(and_idempotent_artifact()),
        ..WhittleEngineConfig::default()
    })
    .unwrap()
}

fn and_idempotent_artifact() -> Vec<u8> {
    wav1(1, 16, 2, &[(0, 0, NO_NODE), (2, 0, 0), (5, 1, NO_NODE)])
}

fn wav1(arity: u16, capacity: u16, output_node: u32, nodes: &[(i8, u32, u32)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"WAV1");
    bytes.extend_from_slice(&arity.to_le_bytes());
    bytes.extend_from_slice(&capacity.to_le_bytes());
    bytes.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&output_node.to_le_bytes());

    for (op, arg0, arg1) in nodes {
        bytes.push(*op as u8);
        bytes.extend_from_slice(&arg0.to_le_bytes());
        bytes.extend_from_slice(&arg1.to_le_bytes());
    }

    bytes
}
