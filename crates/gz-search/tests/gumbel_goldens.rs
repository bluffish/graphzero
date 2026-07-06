mod common;

use common::{TestEngine, measure_options};
use gz_engine::{
    MeasureResult, ModelVersion, PortableCandidateRef, PortableGraphId, PortableSearchActionRef,
    ReplayGraphContext, SearchConfigHash, SearchStepRef,
};
use gz_eval::{EvalOutput, EvalRequest, EvalResult, Evaluator};
use gz_search::{
    GumbelEpisode, GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig, GumbelOpponentContext,
    GumbelRootResult, GumbelRootStats, GumbelSearchContext, GumbelStopReason,
};
use std::collections::BTreeMap;
use std::num::NonZeroUsize;

#[derive(Clone)]
struct EvalRow {
    logits: Vec<f32>,
    value: f32,
}

#[derive(Default)]
struct RecordedEvaluator {
    rows: BTreeMap<u8, EvalRow>,
}

impl RecordedEvaluator {
    fn row(mut self, graph: u8, logits: impl Into<Vec<f32>>, value: f32) -> Self {
        self.rows.insert(
            graph,
            EvalRow {
                logits: logits.into(),
                value,
            },
        );
        self
    }
}

impl Evaluator for RecordedEvaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvalRequest],
        out: &mut Vec<EvalOutput>,
    ) -> EvalResult<()> {
        out.clear();

        for request in requests {
            request.validate_ref()?;
            let graph = request.context.graph.graph_hash.as_bytes()[0];
            let row = self.rows.get(&graph).cloned().unwrap_or(EvalRow {
                logits: vec![0.0; request.action_count()],
                value: 0.0,
            });

            out.push(EvalOutput {
                model_version: ModelVersion::from_bytes([7; 16]),
                policy_logits: row.logits,
                value: row.value,
            });
        }

        Ok(())
    }
}

fn config(max_steps: usize) -> GumbelMctsConfig {
    GumbelMctsConfig {
        max_steps,
        simulations: NonZeroUsize::new(1).unwrap(),
        max_considered_actions: NonZeroUsize::new(8).unwrap(),
        seed: 0,
        gumbel_scale: 0.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        candidate_options: gz_engine::CandidateOptions::default(),
        measure_options: measure_options(),
    }
}

#[test]
fn g1_multi_step_episode_golden() {
    let mut engine = TestEngine::new()
        .candidates(0, [1, 2])
        .candidates(10, [3, 4])
        .candidates(20, [])
        .candidates(30, [])
        .candidates(40, [])
        .apply(0, 1, 10)
        .apply(0, 2, 20)
        .apply(10, 3, 30)
        .apply(10, 4, 40)
        .reward(30, 30.0)
        .reward(40, 40.0);
    let mut evaluator = RecordedEvaluator::default()
        .row(0, [0.5, 0.1, -0.3], 0.0)
        .row(10, [1.0, -0.5, -1.0], 0.3)
        .row(20, [0.0], 0.2)
        .row(30, [0.0], 0.4)
        .row(40, [0.0], 0.5);
    let mut cfg = config(2);
    cfg.seed = 17;
    cfg.gumbel_scale = 0.7;
    cfg.simulations = NonZeroUsize::new(4).unwrap();
    cfg.max_considered_actions = NonZeroUsize::new(3).unwrap();
    let search = GumbelMcts::new(cfg);

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_fingerprint(
        "g1",
        &episode_fingerprint(&episode),
        "b4ec672040979841bc6f39594092b454bff9069738d158900d1e080a63553dd7",
    );
}

#[test]
fn g1_reuse_on_multi_step_episode_golden() {
    let mut engine = TestEngine::new()
        .candidates(0, [1, 2])
        .candidates(10, [3, 4])
        .candidates(20, [])
        .candidates(30, [])
        .candidates(40, [])
        .apply(0, 1, 10)
        .apply(0, 2, 20)
        .apply(10, 3, 30)
        .apply(10, 4, 40)
        .reward(30, 30.0)
        .reward(40, 40.0);
    let mut evaluator = RecordedEvaluator::default()
        .row(0, [0.5, 0.1, -0.3], 0.0)
        .row(10, [1.0, -0.5, -1.0], 0.3)
        .row(20, [0.0], 0.2)
        .row(30, [0.0], 0.4)
        .row(40, [0.0], 0.5);
    let mut cfg = config(2);
    cfg.tree_reuse = true;
    cfg.seed = 17;
    cfg.gumbel_scale = 0.7;
    cfg.simulations = NonZeroUsize::new(4).unwrap();
    cfg.max_considered_actions = NonZeroUsize::new(3).unwrap();
    let search = GumbelMcts::new(cfg);

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_fingerprint(
        "g1-reuse",
        &episode_fingerprint(&episode),
        "95544107fc12ef962236e7048fd3a580a985e122b8a7901327438354f1a05b69",
    );
}

#[test]
fn g2_temperature_episode_golden() {
    let mut engine = TestEngine::new()
        .candidates(0, [1, 2])
        .candidates(10, [])
        .candidates(20, [])
        .apply(0, 1, 10)
        .apply(0, 2, 20);
    let mut evaluator = RecordedEvaluator::default()
        .row(0, [0.0, 0.0, -5.0], 0.0)
        .row(10, [0.0], 1.0)
        .row(20, [0.0], 0.8);
    let mut cfg = config(2);
    cfg.seed = 9;
    cfg.simulations = NonZeroUsize::new(3).unwrap();
    cfg.max_considered_actions = NonZeroUsize::new(2).unwrap();
    cfg.temperature_moves = 2;
    let search = GumbelMcts::new(cfg);

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_fingerprint(
        "g2",
        &episode_fingerprint(&episode),
        "9cc0edb881cd82596c93635838badf44aaad01fcb22264b3d490b2b7e3227826",
    );
}

#[test]
fn g3_opponent_stop_reeval_episode_golden() {
    let mut engine = TestEngine::new().candidates(0, []).reward(0, 0.0);
    let mut evaluator = RecordedEvaluator::default().row(0, [0.0], 0.25);
    let search = GumbelMcts::new(config(1));

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext {
                opponent: Some(GumbelOpponentContext {
                    trajectory_id: 11,
                    row_count: 4,
                    final_reward: 0.0,
                }),
                noise_seed: 0,
            },
        )
        .unwrap();

    assert_fingerprint(
        "g3",
        &episode_fingerprint(&episode),
        "f05a0f70407c073bb16917dc902e07199314a99c4982b94e9b71df6730894c6d",
    );
}

#[test]
fn g4_rejected_candidate_episode_golden() {
    let mut engine = TestEngine::new()
        .candidates(0, [1])
        .rejected(0, 1)
        .reward(0, 0.0);
    let mut evaluator = RecordedEvaluator::default().row(0, [10.0, 0.0], 0.0);
    let mut cfg = config(1);
    cfg.max_considered_actions = NonZeroUsize::new(1).unwrap();
    let search = GumbelMcts::new(cfg);

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_fingerprint(
        "g4",
        &episode_fingerprint(&episode),
        "86d44a4c0c538929c46594d810ada69ef1ae3858fa0d42449a9b0ced5060aa17",
    );
}

#[test]
fn g5_zero_step_episode_golden() {
    let mut engine = TestEngine::new().reward(0, 0.0);
    let mut evaluator = RecordedEvaluator::default();
    let search = GumbelMcts::new(config(0));

    let episode = search
        .run(
            &mut engine,
            &mut evaluator,
            0,
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_fingerprint(
        "g5",
        &episode_fingerprint(&episode),
        "2d2ae48e0d8126fec4d0080f9b807aa24ea1885a05a54925af6fd011e1bad737",
    );
}

#[test]
fn g6_root_result_golden() {
    let mut engine = TestEngine::new()
        .candidates(0, [1, 2])
        .candidates(10, [])
        .candidates(20, [])
        .apply(0, 1, 10)
        .apply(0, 2, 20);
    let mut evaluator = RecordedEvaluator::default()
        .row(0, [0.2, 0.4, -0.1], 0.0)
        .row(10, [0.0], 0.3)
        .row(20, [0.0], 0.9);
    let mut cfg = config(1);
    cfg.seed = 33;
    cfg.gumbel_scale = 0.4;
    cfg.simulations = NonZeroUsize::new(4).unwrap();
    cfg.max_considered_actions = NonZeroUsize::new(2).unwrap();
    let search = GumbelMcts::new(cfg);

    let result = search
        .search_root(
            &mut engine,
            &mut evaluator,
            0,
            GumbelSearchContext::default(),
        )
        .unwrap();

    assert_fingerprint(
        "g6",
        &root_fingerprint(&result),
        "3f34cd6f373dc2898d72aaa5854664f87aea6ad4f9b53fcb6604ab44bc9ccea6",
    );
}

fn assert_fingerprint(name: &str, actual: &str, expected: &str) {
    assert_eq!(actual, expected, "{name} fingerprint: {actual}");
}

fn episode_fingerprint<G, C>(episode: &GumbelEpisode<G, C>) -> String {
    let mut out = Fingerprint::new();
    out.search_config_hash(episode.search_config_hash);
    out.stop_reason(episode.stop_reason);
    out.context(episode.root_context);
    out.context(episode.final_context);
    out.measure(&episode.final_measure);
    out.len(episode.steps.len());
    for step in &episode.steps {
        out.step_ref(step.step_ref);
        out.usize(step.selected_rank);
        out.usize(step.engine_candidate_count);
        out.usize(step.action_count);
        out.f32_slice(&step.policy_target);
        out.usize_slice(&step.considered_action_indices);
        out.f32(step.root_value);
        out.f32(step.root_search_value);
        out.f32(step.root_q_max);
        out.model_version(step.model_version);
    }
    out.len(episode.root_stats.len());
    for stats in &episode.root_stats {
        out.root_stats(*stats);
    }
    out.finish()
}

fn root_fingerprint<G, C>(result: &GumbelRootResult<G, C>) -> String {
    let mut out = Fingerprint::new();
    out.action_ref(result.selected_action_ref);
    out.usize(result.selected_action_index);
    out.usize(result.engine_candidate_count);
    out.usize(result.action_count);
    out.usize_slice(&result.considered_action_indices);
    out.f32_slice(&result.policy_target);
    out.f32(result.root_value);
    out.f32(result.root_search_value);
    out.f32(result.root_q_max);
    out.model_version(result.model_version);
    out.root_stats(result.stats);
    out.finish()
}

struct Fingerprint {
    hasher: blake3::Hasher,
}

impl Fingerprint {
    fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }

    fn finish(self) -> String {
        self.hasher.finalize().to_hex().to_string()
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.len(bytes.len());
        self.hasher.update(bytes);
    }

    fn u8(&mut self, value: u8) {
        self.hasher.update(&[value]);
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u32(&mut self, value: u32) {
        self.hasher.update(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.hasher.update(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn len(&mut self, value: usize) {
        self.usize(value);
    }

    fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    fn f32_option(&mut self, value: Option<f32>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.f32(value);
            }
            None => self.u8(0),
        }
    }

    fn f32_slice(&mut self, values: &[f32]) {
        self.len(values.len());
        for value in values {
            self.f32(*value);
        }
    }

    fn usize_slice(&mut self, values: &[usize]) {
        self.len(values.len());
        for value in values {
            self.usize(*value);
        }
    }

    fn search_config_hash(&mut self, hash: SearchConfigHash) {
        self.bytes(hash.as_bytes());
    }

    fn model_version(&mut self, version: ModelVersion) {
        self.bytes(version.as_bytes());
    }

    fn stop_reason(&mut self, reason: GumbelStopReason) {
        self.u8(match reason {
            GumbelStopReason::MaxSteps => 0,
            GumbelStopReason::SelectedStop => 1,
        });
    }

    fn context(&mut self, context: ReplayGraphContext) {
        self.graph_id(context.graph);
        self.bytes(context.action_set_hash.as_bytes());
    }

    fn graph_id(&mut self, graph: PortableGraphId) {
        self.bytes(graph.graph_hash.as_bytes());
        self.bytes(graph.engine_id.as_bytes());
        self.bytes(graph.engine_version.as_bytes());
    }

    fn candidate_ref(&mut self, candidate: PortableCandidateRef) {
        self.context(candidate.context);
        self.bytes(candidate.candidate_hash.as_bytes());
    }

    fn action_ref(&mut self, action: PortableSearchActionRef) {
        match action {
            PortableSearchActionRef::Candidate(candidate) => {
                self.u8(0);
                self.candidate_ref(candidate);
            }
            PortableSearchActionRef::Stop { context } => {
                self.u8(1);
                self.context(context);
            }
        }
    }

    fn step_ref(&mut self, step: SearchStepRef) {
        self.context(step.before);
        self.action_ref(step.action);
        self.context(step.after);
    }

    fn measure<G>(&mut self, measure: &MeasureResult<G>) {
        self.bytes(measure.graph_hash.as_bytes());
        self.bool(measure.measured);
        self.bool(measure.valid);
        self.f32_option(measure.scalar_reward);
    }

    fn root_stats(&mut self, stats: GumbelRootStats) {
        self.usize(stats.simulations);
        self.usize(stats.expanded_nodes);
        self.usize(stats.eval_count);
        self.usize(stats.carried_nodes);
        self.u32(stats.carried_root_visits);
    }
}
