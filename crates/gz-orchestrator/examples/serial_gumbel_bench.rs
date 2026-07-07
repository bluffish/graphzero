use gz_engine::{CandidateOptions, GraphEngine};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleGraphGenerator, WhittleGraphGeneratorConfig,
};
use gz_eval_whittle::WhittleMeasureEvaluator;
use gz_orchestrator::{
    SelfplayBenchConfig, SelfplayEpisodeStats, SerialGumbelOrchestrator, WorkerId,
    run_serial_selfplay_benchmark,
};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig};
use std::num::NonZeroUsize;

fn main() -> gz_engine::EngineResult<()> {
    let config = Args::parse();
    let mut engine = WhittleEngine::new(WhittleEngineConfig::default())?;
    let mut generator =
        WhittleGraphGenerator::from_seed(WhittleGraphGeneratorConfig::default(), config.graph_seed);
    let generated = generator.sample_into(&mut engine)?;
    let root_hash = engine.hash(generated.graph)?;
    let search = GumbelMcts::new(GumbelMctsConfig {
        max_steps: config.max_steps,
        simulations: config.simulations,
        max_considered_actions: config.max_considered_actions,
        seed: config.search_seed,
        gumbel_scale: 1.0,
        gumbel_noise_overlap: -1.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        mask_stop: false,
        no_backtrack: false,
        candidate_options: CandidateOptions::default(),
        measure_options: engine.measure_options(),
    });
    let mut orchestrator = SerialGumbelOrchestrator::new(
        WorkerId::new(0),
        engine,
        WhittleMeasureEvaluator::new(),
        search,
    );
    let report = run_serial_selfplay_benchmark(SelfplayBenchConfig::new(config.episodes), || {
        let episode = orchestrator.run(generated.graph, GumbelEpisodeContext::default())?;
        Ok(SelfplayEpisodeStats::new(episode.episode.steps.len() as u64))
    })?;

    println!("runner=serial_gumbel");
    println!("root_mode=fixed");
    println!("episodes={}", report.episodes);
    println!("elapsed_ms={:.3}", report.elapsed.as_secs_f64() * 1000.0);
    println!("episodes_per_sec={:.3}", report.episodes_per_second());
    println!("steps={}", report.steps);
    println!("steps_per_sec={:.3}", report.steps_per_second());
    println!("graph_seed={}", config.graph_seed);
    println!("search_seed={}", config.search_seed);
    println!("root_hash={root_hash}");
    println!("max_steps={}", config.max_steps);
    println!("simulations={}", config.simulations);
    println!("max_considered_actions={}", config.max_considered_actions);

    Ok(())
}

struct Args {
    episodes: u64,
    max_steps: usize,
    simulations: NonZeroUsize,
    max_considered_actions: NonZeroUsize,
    graph_seed: u64,
    search_seed: u64,
}

impl Args {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        Self {
            episodes: parse_u64(args.next(), 100),
            max_steps: parse_usize(args.next(), 64),
            simulations: parse_nonzero_usize(args.next(), 128),
            max_considered_actions: parse_nonzero_usize(args.next(), 16),
            graph_seed: parse_u64(args.next(), 42),
            search_seed: parse_u64(args.next(), 0),
        }
    }
}

fn parse_u64(value: Option<String>, default: u64) -> u64 {
    value
        .as_deref()
        .map_or(default, |value| value.parse().unwrap_or(default))
}

fn parse_usize(value: Option<String>, default: usize) -> usize {
    value
        .as_deref()
        .map_or(default, |value| value.parse().unwrap_or(default))
}

fn parse_nonzero_usize(value: Option<String>, default: usize) -> NonZeroUsize {
    NonZeroUsize::new(parse_usize(value, default).max(1)).expect("value is clamped to non-zero")
}
