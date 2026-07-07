use gz_engine::{CandidateOptions, GraphEngine, PortableSearchActionRef};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleGraphGenerator, WhittleGraphGeneratorConfig,
    rule_name,
};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig, SearchAction};
use std::num::NonZeroUsize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph_seed = u64_arg(1)?.unwrap_or(42);
    let search_seed = u64_arg(2)?.unwrap_or(0);
    let eval_seed = u64_arg(3)?.unwrap_or(0);
    let max_steps = usize_arg(4)?.unwrap_or(8);
    let simulations = nonzero_arg(5)?.unwrap_or(NonZeroUsize::new(32).unwrap());
    let max_considered_actions = nonzero_arg(6)?.unwrap_or(NonZeroUsize::new(16).unwrap());

    let mut engine = WhittleEngine::new(WhittleEngineConfig::default())?;
    let generator_config = WhittleGraphGeneratorConfig {
        arity: 6,
        ..WhittleGraphGeneratorConfig::default()
    };
    let mut generator = WhittleGraphGenerator::from_seed(generator_config, graph_seed);
    let generated = generator.sample_into(&mut engine)?;

    let search = GumbelMcts::new(GumbelMctsConfig {
        max_steps,
        simulations,
        max_considered_actions,
        seed: search_seed,
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
    let mut evaluator = RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: eval_seed,
        ..RandomValueEvaluatorConfig::default()
    })?;
    let episode = search.run(
        &mut engine,
        &mut evaluator,
        generated.graph,
        GumbelEpisodeContext::default(),
    )?;

    let start_measure = engine.measure(generated.graph, engine.measure_options())?;

    println!("graph_seed={graph_seed}");
    println!("search_seed={search_seed}");
    println!("eval_seed={eval_seed}");
    println!("max_steps={max_steps}");
    println!("simulations={}", simulations.get());
    println!("max_considered_actions={}", max_considered_actions.get());
    println!("generated_graph={}", generated.graph.raw());
    println!("seed_graph={}", generated.seed_graph.raw());
    println!(
        "prewalk_requested={} prewalk_applied={}",
        generated.prewalk_steps_requested, generated.prewalk_steps_applied
    );
    println!(
        "generated_start_cost={} generated_final_cost={}",
        generated.start_cost, generated.final_cost
    );
    println!("root_hash={}", episode.root_context.graph.graph_hash);
    println!("root_reward={:?}", start_measure.scalar_reward);
    println!("steps={}", episode.steps.len());
    println!("stop_reason={:?}", episode.stop_reason);
    println!("final_graph={}", episode.final_graph.raw());
    println!("final_hash={}", episode.final_context.graph.graph_hash);
    println!("final_reward={:?}", episode.final_measure.scalar_reward);
    println!("search_config_hash={}", episode.search_config_hash);

    for (index, step) in episode.steps.iter().enumerate() {
        let after_measure = engine.measure(step.after, engine.measure_options())?;
        let selected_policy = step
            .policy_target
            .get(step.selected_rank)
            .copied()
            .unwrap_or(0.0);

        match step.action {
            SearchAction::Candidate(candidate) => {
                let rule = step
                    .selected_candidate
                    .map(|summary| rule_name(summary.kind.get() as u16))
                    .unwrap_or("Unknown");

                println!(
                    "step={index} action=candidate rule={rule} candidate={} rank={} candidates={} actions={} considered={:?} policy={selected_policy:.6} root_value={:.6} search_value={:.6} q_max={:.6} after_reward={:?} after={}",
                    candidate.raw(),
                    step.selected_rank,
                    step.engine_candidate_count,
                    step.action_count,
                    step.considered_action_indices,
                    step.root_value,
                    step.root_search_value,
                    step.root_q_max,
                    after_measure.scalar_reward,
                    step.after.raw()
                );
            }
            SearchAction::Stop => {
                println!(
                    "step={index} action=STOP rank={} candidates={} actions={} considered={:?} policy={selected_policy:.6} root_value={:.6} search_value={:.6} q_max={:.6} after_reward={:?}",
                    step.selected_rank,
                    step.engine_candidate_count,
                    step.action_count,
                    step.considered_action_indices,
                    step.root_value,
                    step.root_search_value,
                    step.root_q_max,
                    after_measure.scalar_reward
                );
            }
        }

        if let PortableSearchActionRef::Candidate(candidate) = step.selected_action {
            println!("step={index} candidate_hash={}", candidate.candidate_hash);
        }
    }

    Ok(())
}

fn u64_arg(index: usize) -> Result<Option<u64>, Box<dyn std::error::Error>> {
    std::env::args()
        .nth(index)
        .map(|arg| arg.parse::<u64>().map_err(Into::into))
        .transpose()
}

fn usize_arg(index: usize) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    std::env::args()
        .nth(index)
        .map(|arg| arg.parse::<usize>().map_err(Into::into))
        .transpose()
}

fn nonzero_arg(index: usize) -> Result<Option<NonZeroUsize>, Box<dyn std::error::Error>> {
    let Some(value) = usize_arg(index)? else {
        return Ok(None);
    };
    Ok(Some(
        NonZeroUsize::new(value).ok_or("argument must be nonzero")?,
    ))
}
