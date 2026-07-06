use gz_features::{
    ActionFeature, FeatureCollator, FeatureEdge, FeatureRow, FeatureSchema, FeatureSchemaConfig,
    PositionFeatures, RowTargets, encode_training_targets,
};
use std::num::NonZeroUsize;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python/tests/fixtures"));
    std::fs::create_dir_all(&out_dir)?;

    write_attr1(out_dir.join("batch_attr1.gzfb"))?;
    write_attr0(out_dir.join("batch_attr0.gzfb"))?;
    write_expander(out_dir.join("batch_expander.gzfb"))?;
    write_targets(out_dir.join("targets.gzft"))?;
    Ok(())
}

fn write_targets(path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let targets = [
        RowTargets {
            policy: vec![0.75, 0.25, 0.0],
            value: Some(1.0),
            reward: 2.5,
        },
        RowTargets {
            policy: vec![1.0, 0.0, 0.0],
            value: Some(-1.0),
            reward: -3.0,
        },
    ];
    let mut bytes = Vec::new();
    encode_training_targets(&targets, 2, 3, &mut bytes)?;
    std::fs::write(path, &bytes)?;
    Ok(())
}

fn write_attr1(path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let schema = FeatureSchema::new(FeatureSchemaConfig {
        name: "gz-fixture-v1".to_string(),
        node_vocab_size: 7,
        node_attr_dim: 1,
        edge_type_count: 2,
        action_kind_vocab_size: 12,
        max_nodes: 8,
        max_edges: 4,
        max_actions: 6,
        max_subjects: 2,
        opponent_reward_scale: 256.0,
        expander_degree: 0,
        expander_seed: 0,
    })?;
    let rows = vec![
        FeatureRow {
            node_count: 3,
            node_tokens: vec![1, 2, 3],
            node_attrs: vec![0.5, -1.0, 2.0],
            edges: vec![
                FeatureEdge {
                    src: 0,
                    dst: 2,
                    edge_type: 0,
                },
                FeatureEdge {
                    src: 1,
                    dst: 2,
                    edge_type: 1,
                },
            ],
            actions: vec![
                ActionFeature {
                    kind_token: 4,
                    static_prior: 0.25,
                    subjects: vec![2],
                },
                stop(),
            ],
            position: PositionFeatures {
                root_step: 0,
                leaf_depth: 0,
                budget_fraction: 1.0,
                budget_step: 0.125,
                opponent_reward: 0.0,
                opponent_present: false,
            },
        },
        FeatureRow {
            node_count: 1,
            node_tokens: vec![6],
            node_attrs: vec![1.5],
            edges: Vec::new(),
            actions: vec![stop()],
            position: PositionFeatures {
                root_step: 1,
                leaf_depth: 2,
                budget_fraction: 0.75,
                budget_step: 0.125,
                opponent_reward: 0.0,
                opponent_present: false,
            },
        },
        FeatureRow {
            node_count: 5,
            node_tokens: vec![1, 1, 4, 5, 2],
            node_attrs: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            edges: vec![
                FeatureEdge {
                    src: 0,
                    dst: 2,
                    edge_type: 0,
                },
                FeatureEdge {
                    src: 1,
                    dst: 2,
                    edge_type: 1,
                },
                FeatureEdge {
                    src: 2,
                    dst: 4,
                    edge_type: 0,
                },
                FeatureEdge {
                    src: 3,
                    dst: 4,
                    edge_type: 1,
                },
            ],
            actions: vec![
                ActionFeature {
                    kind_token: 2,
                    static_prior: -0.5,
                    subjects: vec![0, 1],
                },
                ActionFeature {
                    kind_token: 3,
                    static_prior: 0.0,
                    subjects: Vec::new(),
                },
                ActionFeature {
                    kind_token: 4,
                    static_prior: 1.0,
                    subjects: vec![4],
                },
                ActionFeature {
                    kind_token: 5,
                    static_prior: 0.125,
                    subjects: vec![2, 3],
                },
                ActionFeature {
                    kind_token: 6,
                    static_prior: -1.0,
                    subjects: vec![0],
                },
                stop(),
            ],
            position: PositionFeatures {
                root_step: 3,
                leaf_depth: 1,
                budget_fraction: 0.5,
                budget_step: 0.25,
                opponent_reward: 0.0,
                opponent_present: false,
            },
        },
    ];
    write_batch(schema, 4, &rows, path)
}

fn write_attr0(path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let schema = FeatureSchema::new(FeatureSchemaConfig {
        name: "gz-fixture-attr0-v1".to_string(),
        node_vocab_size: 7,
        node_attr_dim: 0,
        edge_type_count: 2,
        action_kind_vocab_size: 12,
        max_nodes: 4,
        max_edges: 2,
        max_actions: 3,
        max_subjects: 2,
        opponent_reward_scale: 256.0,
        expander_degree: 0,
        expander_seed: 0,
    })?;
    let rows = vec![FeatureRow {
        node_count: 2,
        node_tokens: vec![1, 2],
        node_attrs: Vec::new(),
        edges: vec![FeatureEdge {
            src: 0,
            dst: 1,
            edge_type: 1,
        }],
        actions: vec![
            ActionFeature {
                kind_token: 4,
                static_prior: 0.5,
                subjects: vec![1],
            },
            stop(),
        ],
        position: PositionFeatures {
            root_step: 0,
            leaf_depth: 1,
            budget_fraction: 0.25,
            budget_step: 0.25,
            opponent_reward: 0.0,
            opponent_present: false,
        },
    }];
    write_batch(schema, 2, &rows, path)
}

fn write_expander(path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let schema = FeatureSchema::new(FeatureSchemaConfig {
        name: "gz-fixture-expander-v1".to_string(),
        node_vocab_size: 7,
        node_attr_dim: 0,
        edge_type_count: 3,
        action_kind_vocab_size: 12,
        max_nodes: 4,
        max_edges: 10,
        max_actions: 4,
        max_subjects: 2,
        opponent_reward_scale: 256.0,
        expander_degree: 2,
        expander_seed: 99,
    })?;
    let rows = vec![FeatureRow {
        node_count: 3,
        node_tokens: vec![1, 2, 3],
        node_attrs: Vec::new(),
        edges: vec![
            FeatureEdge {
                src: 0,
                dst: 2,
                edge_type: 0,
            },
            FeatureEdge {
                src: 0,
                dst: 1,
                edge_type: 2,
            },
            FeatureEdge {
                src: 1,
                dst: 2,
                edge_type: 2,
            },
            FeatureEdge {
                src: 2,
                dst: 0,
                edge_type: 2,
            },
        ],
        actions: vec![
            ActionFeature {
                kind_token: 4,
                static_prior: 0.5,
                subjects: vec![1, 2],
            },
            stop(),
        ],
        position: PositionFeatures {
            root_step: 2,
            leaf_depth: 0,
            budget_fraction: 0.75,
            budget_step: 0.125,
            opponent_reward: 0.0,
            opponent_present: false,
        },
    }];
    write_batch(schema, 2, &rows, path)
}

fn write_batch(
    schema: FeatureSchema,
    capacity: usize,
    rows: &[FeatureRow],
    path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut collator = FeatureCollator::new(schema, NonZeroUsize::new(capacity).unwrap());
    let mut bytes = Vec::new();
    collator.collate_into(rows, &mut bytes)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn stop() -> ActionFeature {
    ActionFeature {
        kind_token: 1,
        static_prior: 0.0,
        subjects: Vec::new(),
    }
}
