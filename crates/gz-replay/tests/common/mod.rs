#![allow(dead_code)]

use gz_engine::{
    ActionSetHash, CandidateHash, EngineId, EngineVersion, GraphHash, MeasureConfigHash,
    MeasureSummary, PortableCandidateRef, PortableGraphId, PortableSearchActionRef,
    ReplayGraphContext, SearchConfigHash, SearchStepRef,
};
use gz_features::{
    ActionFeature, FeatureEdge, FeatureRow, FeatureSchema, FeatureSchemaConfig, PositionFeatures,
    encode_feature_row,
};
use gz_replay::{
    ReplayEpisodeRecord, ReplayOutcome, ReplayReference, ReplayReferenceKind, ReplayRow,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

pub struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("gz-replay-test-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();

        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

pub fn temp_dir() -> TestDir {
    TestDir::new()
}

pub fn graph(byte: u8) -> PortableGraphId {
    PortableGraphId::new(
        GraphHash::from_bytes([byte; 32]),
        EngineId::from_bytes([1; 16]),
        EngineVersion::from_bytes([2; 16]),
    )
}

pub fn context(byte: u8) -> ReplayGraphContext {
    ReplayGraphContext::new(graph(byte), ActionSetHash::from_bytes([3; 32]))
}

pub fn candidate_action(context: ReplayGraphContext, byte: u8) -> PortableSearchActionRef {
    PortableSearchActionRef::candidate(PortableCandidateRef::new(
        context,
        CandidateHash::from_bytes([byte; 32]),
    ))
}

pub fn stop_action(context: ReplayGraphContext) -> PortableSearchActionRef {
    PortableSearchActionRef::stop(context)
}

pub fn search_hash() -> SearchConfigHash {
    SearchConfigHash::from_bytes([9; 32])
}

pub fn measure(reward: Option<f32>, measured: bool, valid: bool) -> MeasureSummary {
    MeasureSummary {
        graph_hash: GraphHash::from_bytes([8; 32]),
        config_hash: MeasureConfigHash::from_bytes([7; 32]),
        measured,
        valid,
        latency: None,
        scalar_reward: reward,
        failure_code: None,
    }
}

pub fn episode_with_rows(row_count: usize) -> (ReplayEpisodeRecord, Vec<ReplayRow>) {
    let root = context(0);
    let final_graph = context(row_count as u8);
    let final_measure = measure(Some(5.0), true, true);
    let mut steps = Vec::new();
    let mut rows = Vec::new();
    let mut history = Vec::new();

    for index in 0..row_count {
        let before = context(index as u8);
        let after = context(index as u8 + 1);
        let action = candidate_action(before, 40 + index as u8);
        steps.push(SearchStepRef::new(before, action, after).unwrap());
        rows.push(ReplayRow {
            step_index: index as u32,
            root,
            state: before,
            action_history: history.clone(),
            legal_actions: vec![action, stop_action(before)],
            policy_target: vec![1.0, 0.0],
            selected_action: action,
            value_target: Some(1.0),
            reward_target: Some(5.0),
            final_measure: final_measure.clone(),
            model_version: None,
            search_config_hash: search_hash(),
            feature_row: None,
        });
        history.push(action);
    }

    let record = ReplayEpisodeRecord {
        root,
        final_graph,
        steps,
        final_measure,
        outcome: ReplayOutcome {
            value_target: Some(1.0),
            learner_reward: 5.0,
            stopped: false,
            reference: Some(ReplayReference {
                kind: ReplayReferenceKind::RootBaseline,
                reward: 4.0,
                final_graph: None,
                trajectory_id: None,
                search_config_hash: None,
                model_version: None,
            }),
        },
        search_config_hash: search_hash(),
        row_count: row_count as u32,
    };

    (record, rows)
}

pub fn feature_schema_config() -> FeatureSchemaConfig {
    FeatureSchemaConfig {
        name: "replay-test-v1".to_string(),
        node_vocab_size: 8,
        node_attr_dim: 1,
        edge_type_count: 2,
        action_kind_vocab_size: 16,
        max_nodes: 4,
        max_edges: 4,
        max_actions: 4,
        max_subjects: 3,
        opponent_reward_scale: 256.0,
        expander_degree: 0,
        expander_seed: 0,
    }
}

pub fn feature_row_bytes(step_index: u32, action_count: usize) -> Vec<u8> {
    let schema = FeatureSchema::new(feature_schema_config()).unwrap();
    let mut row = FeatureRow {
        node_count: 2,
        node_tokens: vec![1, 3],
        node_attrs: vec![0.5, 1.5],
        edges: vec![FeatureEdge {
            src: 0,
            dst: 1,
            edge_type: 0,
        }],
        actions: Vec::new(),
        position: PositionFeatures {
            root_step: step_index,
            leaf_depth: 0,
            budget_fraction: 1.0,
            budget_step: 0.5,
            opponent_reward: 0.0,
            opponent_present: false,
        },
    };
    for index in 0..action_count.saturating_sub(1) {
        row.actions.push(ActionFeature {
            kind_token: 2 + index as u32,
            static_prior: 0.0,
            subjects: vec![0],
        });
    }
    row.actions.push(ActionFeature {
        kind_token: 1,
        static_prior: 0.0,
        subjects: Vec::new(),
    });

    let mut bytes = Vec::new();
    encode_feature_row(&row, &schema, &mut bytes).unwrap();
    bytes
}

pub fn episode_with_feature_rows(row_count: usize) -> (ReplayEpisodeRecord, Vec<ReplayRow>) {
    let (record, mut rows) = episode_with_rows(row_count);
    for row in &mut rows {
        row.feature_row = Some(feature_row_bytes(row.step_index, row.legal_actions.len()));
    }
    (record, rows)
}
