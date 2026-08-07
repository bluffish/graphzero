use gz_engine::{CandidateOptions, GraphEngine};
use gz_engine_compiler::{
    CompilerEngine, CompilerFeatureExtractor, CompilerFeatureExtractorConfig,
    CompilerFeatureLayout, CompilerStateId,
};
use gz_features::{FeatureExtractor, PositionFeatures};
use gz_measure_agent::CompilerDeployment;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct CompilerInventoryConfig {
    pub deployment: PathBuf,
    pub feature_layout: CompilerFeatureLayout,
    pub validate_root_actions: bool,
    pub rollouts: usize,
    pub max_steps: usize,
    pub seed: u64,
}

impl CompilerInventoryConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.rollouts == 0 {
            return Err("--rollouts must be greater than zero".to_owned());
        }
        if self.max_steps == 0 {
            return Err("--max-steps must be greater than zero".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompilerInventorySummary {
    pub states: u64,
    pub root_actions: u32,
    pub max_nodes: u32,
    pub max_edges: u32,
    pub max_actions: u32,
    pub max_subjects: u32,
    pub recommended_max_nodes: u32,
    pub recommended_max_edges: u32,
    pub recommended_max_actions: u32,
    pub recommended_max_subjects: u32,
}

pub fn inventory(config: CompilerInventoryConfig) -> Result<CompilerInventorySummary, String> {
    config.validate()?;
    let deployment = CompilerDeployment::load(&config.deployment)?;
    let mut engine = deployment.build_search_engine()?;
    let mut extractor = CompilerFeatureExtractor::new(
        &engine,
        CompilerFeatureExtractorConfig {
            layout: config.feature_layout,
            max_nodes: u32::from(u16::MAX),
            max_edges: u32::MAX,
            max_actions: u32::MAX,
            max_subjects: u32::from(u8::MAX),
        },
    )
    .map_err(|error| error.to_string())?;
    let options = CandidateOptions {
        max_candidates: None,
        deterministic_order: true,
    };
    let mut summary = CompilerInventorySummary::default();

    let root = engine.root();
    let mut root_candidates = Vec::new();
    engine
        .candidates(root, options, &mut root_candidates)
        .map_err(|error| error.to_string())?;
    observe(
        &engine,
        &mut extractor,
        root,
        &root_candidates,
        &mut summary,
    )?;
    summary.root_actions = u32::try_from(root_candidates.len())
        .map_err(|_| "compiler root action count exceeds u32".to_owned())?;

    let mut apply_failures = Vec::new();
    if config.validate_root_actions {
        for candidate in root_candidates.iter().copied() {
            let info = engine
                .candidate_info(root, candidate)
                .map_err(|error| error.to_string())?;
            let child = match engine.apply(root, candidate) {
                Ok(child) => child,
                Err(error) => {
                    apply_failures.push(format!("{}: {error}", info.display_name));
                    continue;
                }
            };
            let mut child_candidates = Vec::new();
            engine
                .candidates(child.after, options, &mut child_candidates)
                .map_err(|error| error.to_string())?;
            observe(
                &engine,
                &mut extractor,
                child.after,
                &child_candidates,
                &mut summary,
            )?;
            engine
                .release(&[child.after], &child_candidates)
                .map_err(|error| error.to_string())?;
        }
    }
    engine
        .release(&[], &root_candidates)
        .map_err(|error| error.to_string())?;
    if !apply_failures.is_empty() {
        return Err(format!(
            "{} root candidates failed to apply:\n{}",
            apply_failures.len(),
            apply_failures.join("\n")
        ));
    }

    let mut random = SplitMix64(config.seed);
    let mut rollout_failures = Vec::new();
    for rollout in 0..config.rollouts {
        let mut graph = root;
        let mut graph_is_owned = false;
        for step in 0..config.max_steps {
            let mut candidates = Vec::new();
            engine
                .candidates(graph, options, &mut candidates)
                .map_err(|error| error.to_string())?;
            observe(&engine, &mut extractor, graph, &candidates, &mut summary)?;
            if candidates.is_empty() {
                engine
                    .release(graph_is_owned.then_some(graph).as_slice(), &candidates)
                    .map_err(|error| error.to_string())?;
                graph_is_owned = false;
                break;
            }
            let selected = candidates[random.index(candidates.len())];
            let info = engine
                .candidate_info(graph, selected)
                .map_err(|error| error.to_string())?;
            let child = match engine.apply(graph, selected) {
                Ok(child) => child,
                Err(error) => {
                    rollout_failures.push(format!(
                        "rollout {rollout} step {step} candidate {} failed to apply: {error}",
                        info.display_name
                    ));
                    engine
                        .release(graph_is_owned.then_some(graph).as_slice(), &candidates)
                        .map_err(|error| error.to_string())?;
                    graph_is_owned = false;
                    break;
                }
            };
            engine
                .release(graph_is_owned.then_some(graph).as_slice(), &candidates)
                .map_err(|error| error.to_string())?;
            graph = child.after;
            graph_is_owned = true;
        }
        if graph_is_owned {
            engine
                .release(&[graph], &[])
                .map_err(|error| error.to_string())?;
        }
    }
    if !rollout_failures.is_empty() {
        return Err(format!(
            "{} rollout candidates failed to apply:\n{}",
            rollout_failures.len(),
            rollout_failures.join("\n")
        ));
    }

    summary.recommended_max_nodes = headroom(summary.max_nodes, 256)?;
    summary.recommended_max_edges = headroom(summary.max_edges, 1_024)?;
    summary.recommended_max_actions = round_up(summary.max_actions, 256)?;
    summary.recommended_max_subjects = round_up(summary.max_subjects.max(1), 8)?.min(255);
    Ok(summary)
}

fn observe(
    engine: &CompilerEngine,
    extractor: &mut CompilerFeatureExtractor,
    graph: CompilerStateId,
    candidates: &[<CompilerEngine as GraphEngine>::Candidate],
    summary: &mut CompilerInventorySummary,
) -> Result<(), String> {
    let row = extractor
        .extract(
            engine,
            graph,
            candidates,
            PositionFeatures {
                root_step: 0,
                leaf_depth: 0,
                budget_fraction: 1.0,
                budget_step: 0.0,
                opponent_reward: 0.0,
                opponent_present: false,
            },
        )
        .map_err(|error| error.to_string())?;
    summary.states += 1;
    summary.max_nodes = summary.max_nodes.max(row.node_count);
    summary.max_edges = summary
        .max_edges
        .max(u32::try_from(row.edges.len()).unwrap_or(u32::MAX));
    summary.max_actions = summary
        .max_actions
        .max(u32::try_from(row.actions.len()).unwrap_or(u32::MAX));
    summary.max_subjects = summary.max_subjects.max(
        row.actions
            .iter()
            .map(|action| u32::try_from(action.subjects.len()).unwrap_or(u32::MAX))
            .max()
            .unwrap_or(0),
    );
    Ok(())
}

fn headroom(value: u32, multiple: u32) -> Result<u32, String> {
    let expanded = u64::from(value)
        .checked_mul(5)
        .and_then(|value| value.checked_add(3))
        .map(|value| value / 4)
        .ok_or_else(|| "compiler inventory headroom overflow".to_owned())?;
    let expanded = u32::try_from(expanded)
        .map_err(|_| "compiler inventory headroom exceeds u32".to_owned())?;
    round_up(expanded, multiple)
}

fn round_up(value: u32, multiple: u32) -> Result<u32, String> {
    value
        .checked_add(multiple - 1)
        .map(|value| value / multiple * multiple)
        .ok_or_else(|| "compiler inventory bound overflow".to_owned())
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, len: usize) -> usize {
        (self.next() as usize) % len
    }
}
