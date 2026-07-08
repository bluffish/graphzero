//! Exports the production fixed root (generator seed from argv) as a
//! whittlezero fixed-graph-set jsonl line: the engines share the WAV1
//! serialization, so the artifact bytes hex-encode directly into their
//! "graph" field.
use gz_engine::GraphEngine;
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleGraphGenerator, WhittleGraphGeneratorConfig,
    WhittleRoot,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seed = std::env::args()
        .nth(1)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(42);
    let generator_config = WhittleGraphGeneratorConfig::default();
    let mut engine = WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Input {
            arity: generator_config.arity,
            capacity: generator_config.capacity,
            input_index: 0,
        },
        ..WhittleEngineConfig::default()
    })?;
    let mut generator = WhittleGraphGenerator::from_seed(generator_config, seed);
    let root = generator.sample_into(&mut engine)?.graph;
    let measured = engine.measure(root, engine.measure_options())?;
    let cost = -measured.scalar_reward.ok_or("unmeasured root")?;
    let artifact = engine.export_graph(root)?;
    let hex: String = artifact
        .bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    println!(
        "{{\"cost\": {cost}, \"generation\": {{\"source\": \"graphzero-seed{seed}\"}}, \"graph\": \"{hex}\"}}"
    );
    Ok(())
}
